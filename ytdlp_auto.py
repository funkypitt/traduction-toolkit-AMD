"""yt-dlp, tenu à jour tout seul.

    from ytdlp_auto import ytdlp
    subprocess.run([ytdlp(), "--no-playlist", url])

YouTube, X et les autres changent toutes les quelques semaines et yt-dlp suit en quelques
jours : une copie installée une fois (apt, pip, conda) finit par ne plus rien télécharger, avec
des erreurs qui ressemblent à autre chose (« No video formats found », « Sign in to confirm… »).
Les scripts ne se servent donc pas du yt-dlp du système. Ils partagent leur propre copie, le
binaire autonome des publications GitHub de yt-dlp, récupérée la première fois dans

    ~/.local/share/yt-dlp-auto/yt-dlp        (macOS : ~/Library/Application Support/…)

et `yt-dlp -U` y est lancé au plus une fois par jour, juste avant un usage. Sans réseau, la
copie présente sert telle quelle ; si elle n'a jamais pu être récupérée, reste le yt-dlp du système.

    YTDLP=/chemin/yt-dlp     impose une autre copie, à laquelle on ne touche pas
    python3 ytdlp_auto.py    affiche le chemin et la version (et met à jour si un jour a passé)
    python3 ytdlp_auto.py -U met à jour tout de suite

Le même fichier vit dans plusieurs dépôts (traduction, newspapers, causeries-podcast, …) :
tous partagent cette copie. Bibliothèque standard seulement.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

RELEASES = "https://github.com/yt-dlp/yt-dlp/releases/latest/download/"
CHECK_EVERY = 24 * 3600

if sys.platform == "darwin":
    HOME = Path.home() / "Library" / "Application Support" / "yt-dlp-auto"
elif sys.platform == "win32":
    HOME = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "yt-dlp-auto"
else:
    HOME = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "yt-dlp-auto"
OWN = HOME / ("yt-dlp.exe" if sys.platform == "win32" else "yt-dlp")
STAMP = HOME / "checked"

_ready: str | None = None      # one look per process: a script calls yt-dlp many times


def _asset() -> str:
    if sys.platform == "win32":
        return "yt-dlp.exe"
    if sys.platform == "darwin":
        return "yt-dlp_macos"
    return "yt-dlp_linux_aarch64" if platform.machine().lower() in ("aarch64", "arm64") else "yt-dlp_linux"


def _say(text: str) -> None:
    print(f"[yt-dlp] {text}", file=sys.stderr, flush=True)


class _Lock:
    """Two scripts at once (cron): one fetch, one update. No lock where fcntl does not exist."""

    def __enter__(self):
        HOME.mkdir(parents=True, exist_ok=True)
        self.fh = open(HOME / "lock", "w")
        try:
            import fcntl
            fcntl.flock(self.fh, fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        return self

    def __exit__(self, *exc):
        self.fh.close()


def _system() -> str:
    """A yt-dlp installed some other way: what is left when ours cannot be fetched."""
    for cand in (shutil.which("yt-dlp"), str(Path.home() / ".local" / "bin" / "yt-dlp"),
                 str(Path.home() / "miniconda3" / "bin" / "yt-dlp")):
        if cand and os.path.exists(cand):
            return cand
    return ""


def _fetch() -> None:
    part = OWN.with_name(OWN.name + ".part")
    request = urllib.request.Request(RELEASES + _asset(), headers={"User-Agent": "yt-dlp-auto"})
    with urllib.request.urlopen(request, timeout=60) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f)
    part.chmod(0o755)
    part.replace(OWN)
    STAMP.write_text(str(time.time()))


def _checked() -> float:
    try:
        return float(STAMP.read_text())
    except (OSError, ValueError):
        return 0.0


def _mark() -> tuple:
    st = OWN.stat()
    return st.st_size, st.st_mtime_ns


def update(force: bool = False) -> bool:
    """`yt-dlp -U` on our copy, at most once a day (or now, with `force`). Whether a newer one
    came. No network is not an error; the date is written only when `-U` answered."""
    if os.environ.get("YTDLP") or not OWN.exists():
        return False
    with _Lock():
        if not force and time.time() - _checked() < CHECK_EVERY:
            return False
        before = _mark()
        try:
            done = subprocess.run([str(OWN), "-U"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, errors="replace", timeout=600)
        except (OSError, subprocess.TimeoutExpired) as exc:
            _say(f"mise à jour impossible ({exc})")
            return False
        if done.returncode == 0:
            STAMP.write_text(str(time.time()))
        changed = _mark() != before
        if changed:
            lines = (done.stdout or "").strip().splitlines()
            _say(lines[-1] if lines else "mis à jour")
        return changed


def ytdlp() -> str:
    """The yt-dlp to run, fetched the first time and brought up to date when a day has passed."""
    global _ready
    if _ready:
        return _ready
    pinned = os.environ.get("YTDLP")
    if pinned:
        _ready = pinned
        return pinned
    if not OWN.exists():
        with _Lock():
            if not OWN.exists():      # another script may have fetched it while we waited
                _say("téléchargement de la copie des scripts (une fois, 35 Mo)…")
                try:
                    _fetch()
                except Exception as exc:      # pas de réseau, GitHub injoignable
                    system = _system()
                    if not system:
                        raise RuntimeError(f"yt-dlp n'a pas pu être téléchargé : {exc}") from exc
                    _say(f"téléchargement impossible ({exc}) : copie du système {system}")
                    return system      # not remembered: the next call tries again
    else:
        update()
    _ready = str(OWN)
    return _ready


if __name__ == "__main__":
    exe = ytdlp()
    if "-U" in sys.argv[1:]:
        update(force=True)
    version = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout.strip()
    print(exe, version)
