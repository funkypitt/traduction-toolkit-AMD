#!/usr/bin/env python3
"""
Traduction daemon — local HTTP bridge for the Traduction Chrome extension.

Listens on http://127.0.0.1:47318 and exposes:

  GET  /ping                -> {"ok": true}
  GET  /scripts             -> {"scripts": [{"name": ..., "path": ...}, ...]}
  POST /process             -> {"job_id": "..."}
     body: {"url": str, "script": str, "args": str}
     Downloads the URL with yt-dlp into the repo dir (TRADUCTION_DIR), then runs
     `python <script> <downloaded_file> <args...>` in that directory.
  GET  /status/<job_id>     -> {"state": ..., "file": ..., "log": "..."}
  POST /cancel/<job_id>     -> {"ok": true}

Everything is purely local. No credentials, no external calls beyond yt-dlp
fetching the requested video. Run via systemd user service or by hand:

    python3 ~/bin/traduction-daemon.py
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

# ─── configuration ──────────────────────────────────────────────────────────

HOST = os.environ.get("TRADUCTION_HOST", "127.0.0.1")
PORT = int(os.environ.get("TRADUCTION_PORT", "47318"))

# Défaut portable : la racine du dépôt (ce daemon vit dans chrome-extension/daemon/,
# les scripts du toolkit sont deux niveaux au-dessus). Surchargeable par env.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRADUCTION_DIR = Path(
    os.environ.get("TRADUCTION_DIR", str(_REPO_ROOT))
).expanduser()

SCRIPTS_DIR = Path(
    os.environ.get("TRADUCTION_SCRIPTS_DIR", str(TRADUCTION_DIR))
).expanduser()

# ─── import modules depuis le répertoire des scripts ─────────────────────────
# apollohealth.py est dans le même répertoire que les scripts de traduction.
sys.path.insert(0, str(SCRIPTS_DIR))
try:
    import apollohealth
except ImportError:
    apollohealth = None

# yt-dlp: TRADUCTION_YTDLP pins a copy; otherwise the toolkit's own, which ytdlp_auto.py
# (in the scripts directory) fetches once and keeps up to date, at most once a day before a
# download. Without that module: PATH, then the miniconda copy.
try:
    from ytdlp_auto import ytdlp as _ytdlp_auto
except ImportError:
    _ytdlp_auto = None


def ytdlp_bin() -> str:
    pinned = os.environ.get("TRADUCTION_YTDLP")
    if pinned:
        return pinned
    if _ytdlp_auto is not None:
        try:
            return _ytdlp_auto()
        except Exception as exc:
            print(f"[traduction-daemon] ytdlp_auto: {exc}", file=sys.stderr)
    return shutil.which("yt-dlp") or str(Path.home() / "miniconda3" / "bin" / "yt-dlp")

PYTHON_BIN = os.environ.get("TRADUCTION_PYTHON") or sys.executable

# Scripts excluded from the dropdown (internal helpers or unsupported shapes).
SCRIPT_EXCLUDE_PATTERNS = (
    re.compile(r"_bridge\.py$"),
    re.compile(r"^mix_"),
    re.compile(r"^__"),
    # doubler-mp3-batch is batch-mode (scans CWD), doesn't take a positional file.
    re.compile(r"^doubler-mp3-batch\.py$"),
)

# Max log bytes kept in memory per job.
MAX_LOG_BYTES = 256 * 1024

# Marqueurs émis par les scripts pour les questions interactives.
# Voir interactive_voice_map() dans doubler.py.
VOICEMAP_REQUEST_MARKER = "@@VOICEMAP_REQUEST@@"
VOICEMAP_DONE_MARKER = "@@VOICEMAP_DONE@@"

# Concurrent GPU-bound script executions (1 = strict serialization).
# Downloads still run in parallel; only the script phase is gated.
GPU_CONCURRENCY = max(1, int(os.environ.get("TRADUCTION_GPU_CONCURRENCY", "1")))
GPU_SEM = threading.Semaphore(GPU_CONCURRENCY)


# ─── job state ──────────────────────────────────────────────────────────────

class Job:
    def __init__(self, job_id: str, url: str, script: str, args: str):
        self.id = job_id
        self.url = url
        self.script = script
        self.args = args
        self.state = "pending"  # pending|downloading|queued|running|done|error|cancelled
        self.file: Path | None = None
        self.log = bytearray()
        self.log_lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.created_at = time.time()
        self.finished_at: float | None = None
        self.error: str | None = None
        # Question interactive en attente (appariement voix↔locuteur).
        # Renseignée quand le script émet @@VOICEMAP_REQUEST@@ ; effacée quand
        # l'extension répond via POST /respond ou quand le script confirme.
        self.prompt: dict | None = None
        self.prompt_response_file: str | None = None

    def append_log(self, chunk: bytes) -> None:
        with self.log_lock:
            self.log.extend(chunk)
            if len(self.log) > MAX_LOG_BYTES:
                del self.log[: len(self.log) - MAX_LOG_BYTES]

    def snapshot(self) -> dict:
        with self.log_lock:
            log_text = bytes(self.log).decode("utf-8", errors="replace")
        return {
            "id": self.id,
            "state": self.state,
            "url": self.url,
            "script": self.script,
            "args": self.args,
            "file": str(self.file) if self.file else None,
            "log": log_text,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "prompt": self.prompt,
        }


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


# ─── helpers ────────────────────────────────────────────────────────────────

def list_scripts() -> list[dict]:
    if not SCRIPTS_DIR.is_dir():
        return []
    out = []
    for p in sorted(SCRIPTS_DIR.glob("*.py")):
        name = p.name
        if any(pat.search(name) for pat in SCRIPT_EXCLUDE_PATTERNS):
            continue
        out.append({"name": p.stem, "filename": name, "path": str(p)})
    return out


def resolve_script(script_name: str) -> Path:
    """Accept either 'traduire' or 'traduire.py', but only files in SCRIPTS_DIR."""
    candidate = script_name.strip()
    if not candidate:
        raise ValueError("script name is empty")
    # Block traversal: only allow plain filenames.
    if "/" in candidate or ".." in candidate:
        raise ValueError(f"invalid script name: {candidate!r}")
    if not candidate.endswith(".py"):
        candidate += ".py"
    path = (SCRIPTS_DIR / candidate).resolve()
    # Make sure the resolved path is still inside SCRIPTS_DIR.
    if SCRIPTS_DIR.resolve() not in path.parents and path.parent != SCRIPTS_DIR.resolve():
        raise ValueError(f"script outside of scripts dir: {candidate!r}")
    if not path.is_file():
        raise FileNotFoundError(f"script not found: {candidate}")
    return path


def split_args(args: str) -> list[str]:
    args = (args or "").strip()
    if not args:
        return []
    return shlex.split(args)


def is_supported_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


# ─── interactive prompts (voice mapping) ─────────────────────────────────────

# Dossiers dont on autorise la lecture audio par l'extension (échantillons de
# locuteurs et voix de référence). On reste strictement local et en lecture.
AUDIO_ALLOWED_ROOTS = [
    TRADUCTION_DIR.resolve(),
    SCRIPTS_DIR.resolve(),
]


def _is_allowed_audio(path: Path) -> bool:
    try:
        rp = path.resolve()
    except Exception:
        return False
    if rp.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
        return False
    if not rp.is_file():
        return False
    for root in AUDIO_ALLOWED_ROOTS:
        if rp == root or root in rp.parents:
            return True
    return False


def _maybe_handle_marker(job: Job, raw: bytes) -> None:
    """Détecte les marqueurs interactifs dans une ligne de stdout du script."""
    try:
        line = raw.decode("utf-8", errors="replace").strip()
    except Exception:
        return
    if line.startswith(VOICEMAP_REQUEST_MARKER):
        req_path = line[len(VOICEMAP_REQUEST_MARKER):].strip()
        try:
            with open(req_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            job.prompt = payload
            job.prompt_response_file = payload.get("response_file")
        except Exception as e:
            job.append_log(f"[daemon] échec lecture voicemap_request: {e}\n".encode())
    elif line.startswith(VOICEMAP_DONE_MARKER):
        # Le script a reçu la réponse : on efface la question côté UI.
        job.prompt = None
        job.prompt_response_file = None


def respond_prompt(job_id: str, answer: dict) -> bool:
    """Écrit la réponse d'appariement attendue par le script en cours."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job or not job.prompt or not job.prompt_response_file:
        return False
    tmp = job.prompt_response_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(answer, f, ensure_ascii=False)
    os.replace(tmp, job.prompt_response_file)
    # On efface localement ; le script confirmera via @@VOICEMAP_DONE@@.
    job.prompt = None
    return True


# ─── job pipeline ───────────────────────────────────────────────────────────

def run_job(job: Job) -> None:
    try:
        TRADUCTION_DIR.mkdir(parents=True, exist_ok=True)

        # ── 1. download ─────────────────────────────────────────────────────
        job.state = "downloading"
        job.append_log(f"[daemon] destination: {TRADUCTION_DIR}\n".encode())
        job.append_log(f"[daemon] downloading {job.url}\n".encode())

        final_path: str | None = None

        # ── 1a. Apollo Health : téléchargement via apollohealth.py ──────────
        if apollohealth and apollohealth.is_apollo_url(job.url):
            job.append_log(b"[daemon] source: Apollo Health (Vimeo)\n")
            cookies = apollohealth.load_cookies()
            apollo_page = apollohealth.fetch_apollo_page(job.url, cookies)
            job.append_log(f"[daemon] titre: {apollo_page.title}\n".encode())
            if apollo_page.transcript:
                job.append_log(f"[daemon] transcription: {len(apollo_page.transcript)} spans\n".encode())
            final_path = apollohealth.download_apollo_video(
                apollo_page, output_dir=str(TRADUCTION_DIR))
            apollohealth.save_apollo_meta(apollo_page, final_path)

        # ── 1b. Autres URLs : yt-dlp ───────────────────────────────────────
        else:
            ytdlp_exe = ytdlp_bin()
            if not Path(ytdlp_exe).exists() and not shutil.which(ytdlp_exe):
                raise RuntimeError(f"yt-dlp binary not found: {ytdlp_exe}")

            ytdlp_cmd = [
                ytdlp_exe,
                "--no-playlist",
                "--newline",
                "--restrict-filenames",
                "--merge-output-format", "mp4",
                "-f", "bv*+ba/b",
                "-P", str(TRADUCTION_DIR),
                "-o", "%(title).150B [%(id)s].%(ext)s",
                "--print", "after_move:filepath",
                job.url,
            ]
            job.append_log(("[daemon] $ " + " ".join(shlex.quote(a) for a in ytdlp_cmd) + "\n").encode())

            job.proc = subprocess.Popen(
                ytdlp_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(TRADUCTION_DIR),
            )
            assert job.proc.stdout is not None
            for raw in job.proc.stdout:
                job.append_log(raw)
                line = raw.decode("utf-8", errors="replace").rstrip()
                # yt-dlp --print after_move:filepath emits the path on its own line.
                if line and (line.endswith(".mp4") or line.endswith(".mkv") or line.endswith(".webm") or line.endswith(".m4a") or line.endswith(".mp3")):
                    p = Path(line)
                    if p.is_absolute() and p.exists():
                        final_path = line
            rc = job.proc.wait()
            if job.state == "cancelled":
                return
            if rc != 0:
                raise RuntimeError(f"yt-dlp exited with code {rc}")
            if not final_path:
                # Fallback: pick newest file created in TRADUCTION_DIR since job start.
                candidates = [
                    p for p in TRADUCTION_DIR.iterdir()
                    if p.is_file() and p.stat().st_mtime >= job.created_at - 1
                    and p.suffix.lower() in {".mp4", ".mkv", ".webm", ".m4a", ".mp3"}
                ]
                if candidates:
                    final_path = str(max(candidates, key=lambda p: p.stat().st_mtime))

        if not final_path or not Path(final_path).exists():
            raise RuntimeError("could not determine downloaded filepath")

        job.file = Path(final_path)
        job.append_log(f"[daemon] downloaded → {job.file}\n".encode())

        # ── 2. wait for a GPU slot ─────────────────────────────────────────
        job.state = "queued"
        job.append_log(f"[daemon] queued (GPU concurrency: {GPU_CONCURRENCY})\n".encode())
        wait_start = time.time()
        while True:
            if job.state == "cancelled":
                return
            if GPU_SEM.acquire(timeout=1.0):
                break
        waited = time.time() - wait_start
        if waited >= 1.0:
            job.append_log(f"[daemon] GPU slot acquired after {waited:.0f}s\n".encode())

        try:
            # ── 3. run the chosen script ──────────────────────────────────
            job.state = "running"
            script_path = resolve_script(job.script)
            extra_args = split_args(job.args)
            # {file} placeholder substitution: if present in args, we do NOT add the
            # file as a first positional. Otherwise, file comes first (the common
            # shape for traduire, traduire-pro, doubler, clipper).
            if any("{file}" in a for a in extra_args):
                substituted = [a.replace("{file}", str(job.file)) for a in extra_args]
                run_cmd = [PYTHON_BIN, str(script_path), *substituted]
            else:
                run_cmd = [PYTHON_BIN, str(script_path), str(job.file), *extra_args]
            job.append_log(("[daemon] $ " + " ".join(shlex.quote(a) for a in run_cmd) + "\n").encode())

            job.proc = subprocess.Popen(
                run_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(TRADUCTION_DIR),
            )
            assert job.proc.stdout is not None
            for raw in job.proc.stdout:
                job.append_log(raw)
                _maybe_handle_marker(job, raw)
            rc = job.proc.wait()
            if job.state == "cancelled":
                return
            if rc != 0:
                raise RuntimeError(f"{script_path.name} exited with code {rc}")

            job.state = "done"
            job.finished_at = time.time()
            job.append_log(b"[daemon] done\n")
        finally:
            GPU_SEM.release()

    except Exception as e:
        job.error = str(e)
        job.state = "error"
        job.finished_at = time.time()
        job.append_log(f"[daemon] ERROR: {e}\n".encode())
        job.append_log(traceback.format_exc().encode())


def start_job(url: str, script: str, args: str) -> Job:
    if not is_supported_url(url):
        raise ValueError("invalid URL")
    # Validate script early so the extension gets a clear error.
    resolve_script(script)
    job_id = uuid.uuid4().hex[:12]
    job = Job(job_id, url, script, args)
    with JOBS_LOCK:
        JOBS[job_id] = job
    t = threading.Thread(target=run_job, args=(job,), daemon=True)
    t.start()
    return job


def cancel_job(job_id: str) -> bool:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return False
    if job.state in ("done", "error", "cancelled"):
        return True
    job.state = "cancelled"
    if job.proc and job.proc.poll() is None:
        try:
            job.proc.send_signal(signal.SIGTERM)
        except Exception:
            pass
    return True


# ─── HTTP server ────────────────────────────────────────────────────────────

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age": "600",
}


class Handler(BaseHTTPRequestHandler):
    server_version = "TraductionDaemon/0.1"

    def log_message(self, fmt, *args):  # quieter
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as e:
            raise ValueError(f"invalid JSON body: {e}")

    def _serve_audio(self) -> None:
        """Sert un fichier audio local (échantillon locuteur / voix de réf).

        Strictement local et restreint aux dossiers autorisés (AUDIO_ALLOWED_ROOTS).
        Utilisé par l'extension pour l'écoute pendant l'appariement des voix.
        """
        from urllib.parse import parse_qs
        qs = parse_qs(urlparse(self.path).query)
        raw_path = (qs.get("path") or [""])[0]
        if not raw_path:
            return self._send_json(400, {"error": "missing path"})
        target = Path(raw_path)
        if not _is_allowed_audio(target):
            return self._send_json(403, {"error": "path not allowed"})
        data = target.resolve().read_bytes()
        ctype = {
            ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
            ".ogg": "audio/ogg", ".m4a": "audio/mp4",
        }.get(target.suffix.lower(), "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Accept-Ranges", "none")
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path == "/ping":
                return self._send_json(200, {
                    "ok": True,
                    "traduction_dir": str(TRADUCTION_DIR),
                    "scripts_dir": str(SCRIPTS_DIR),
                    "ytdlp": os.environ.get("TRADUCTION_YTDLP") or "ytdlp_auto (own copy, kept up to date)",
                    "python": PYTHON_BIN,
                })
            if path == "/scripts":
                return self._send_json(200, {
                    "scripts": list_scripts(),
                    "defaults": {
                        "traduire": "-s en -t fr",
                        "doubler": "-s en -t fr",
                        "traduire-pro": "-s en -t fr",
                        "resumer": "-s en -t fr",
                    },
                })
            if path.startswith("/status/"):
                job_id = path[len("/status/"):]
                with JOBS_LOCK:
                    job = JOBS.get(job_id)
                if not job:
                    return self._send_json(404, {"error": "unknown job"})
                return self._send_json(200, job.snapshot())
            if path == "/audio":
                return self._serve_audio()
            return self._send_json(404, {"error": "not found"})
        except Exception as e:
            return self._send_json(500, {"error": str(e)})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path == "/process":
                body = self._read_json()
                url = (body.get("url") or "").strip()
                script = (body.get("script") or "").strip()
                args = body.get("args") or ""
                job = start_job(url, script, args)
                return self._send_json(200, {"job_id": job.id})
            if path.startswith("/cancel/"):
                job_id = path[len("/cancel/"):]
                ok = cancel_job(job_id)
                return self._send_json(200 if ok else 404, {"ok": ok})
            if path.startswith("/respond/"):
                job_id = path[len("/respond/"):]
                body = self._read_json()
                # body attendu : {"map": {"SPEAKER_00": "/abs/voix/femme1-ok.wav", ...}}
                answer = {"map": body.get("map") or {}}
                ok = respond_prompt(job_id, answer)
                return self._send_json(200 if ok else 404,
                                       {"ok": ok} if ok else {"error": "no pending prompt"})
            return self._send_json(404, {"error": "not found"})
        except ValueError as e:
            return self._send_json(400, {"error": str(e)})
        except FileNotFoundError as e:
            return self._send_json(400, {"error": str(e)})
        except Exception as e:
            return self._send_json(500, {"error": str(e)})


def main() -> int:
    TRADUCTION_DIR.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[traduction-daemon] listening on http://{HOST}:{PORT}")
    print(f"[traduction-daemon] traduction dir: {TRADUCTION_DIR}")
    print(f"[traduction-daemon] scripts dir:    {SCRIPTS_DIR}")
    print(f"[traduction-daemon] yt-dlp:         {os.environ.get('TRADUCTION_YTDLP') or 'own copy, kept up to date (ytdlp_auto.py)'}")
    print(f"[traduction-daemon] python:         {PYTHON_BIN}")
    print(f"[traduction-daemon] gpu slots:      {GPU_CONCURRENCY}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[traduction-daemon] shutting down")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
