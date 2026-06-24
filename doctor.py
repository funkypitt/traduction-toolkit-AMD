#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
doctor.py — Diagnostic & installation des dépendances de la boîte à outils.

Conçu pour être « failproof » : n'utilise QUE la bibliothèque standard, donc il
tourne même sur une machine vierge où rien n'est installé. Il vérifie tout ce
dont les scripts ont besoin (ffmpeg, env conda « interview » + paquets Python,
Ollama + modèles, clés API, GPU) et propose des correctifs.

    python3 doctor.py              # diagnostic seul, ne modifie rien
    python3 doctor.py --install    # tente d'installer/corriger ce qui manque
    python3 doctor.py --json       # sortie machine (pour la GUI)

Sur une machine déjà configurée (la tienne), il affiche tout en vert et te dit
« rien à faire ». Sur une machine neuve, il guide pas à pas.
"""

import json
import os
import shutil
import subprocess
import sys
import urllib.request

# ── couleurs ANSI (désactivées si pas un terminal) ────────────────────────────
_TTY = sys.stdout.isatty()
def _c(code, s): return f"\033[{code}m{s}\033[0m" if _TTY else s
def green(s):  return _c("32", s)
def red(s):    return _c("31", s)
def yellow(s): return _c("33", s)
def cyan(s):   return _c("36", s)
def bold(s):   return _c("1", s)

OK, WARN, FAIL = "ok", "warn", "fail"
MARK = {OK: green("✓"), WARN: yellow("!"), FAIL: red("✗")}

# ── configuration attendue ────────────────────────────────────────────────────
def _resolve_python():
    """Interpréteur du toolkit : TRADUCTION_PYTHON, sinon env conda
    « interview »/« traduction », sinon l'interpréteur courant."""
    cands = [os.environ.get("TRADUCTION_PYTHON")]
    for env in ("interview", "traduction"):
        cands.append(os.path.expanduser(f"~/miniconda3/envs/{env}/bin/python"))
    for c in cands:
        if c and os.path.exists(c):
            return c
    return sys.executable


INTERVIEW_PY = _resolve_python()

# Paquets pip critiques de l'env principal (import_name, pip_name)
CORE_PKGS = [
    ("anthropic", "anthropic"), ("whisperx", "whisperx"), ("torch", "torch"),
    ("torchaudio", "torchaudio"), ("demucs", "demucs"), ("pydub", "pydub"),
    ("soundfile", "soundfile"), ("numpy", "numpy"), ("flask", "flask"),
]
OPTIONAL_PKGS = [
    ("TTS", "TTS"), ("parselmouth", "praat-parselmouth"), ("pyworld", "pyworld"),
    ("docx", "python-docx"),
]
# Envs conda secondaires (bridges TTS) — facultatifs
TTS_ENVS = ["qwen3tts"]
# Modèles Ollama recommandés (les deux retenus + le défaut résumé)
OLLAMA_RECOMMENDED = ["qwen3.6:27b", "gemma4:31b", "mistral-small:latest"]
OLLAMA_URL = "http://localhost:11434"


class Report:
    def __init__(self):
        self.items = []
    def add(self, status, name, detail="", fix=""):
        self.items.append({"status": status, "name": name, "detail": detail, "fix": fix})
        if not ARGS_JSON:
            line = f"  {MARK[status]} {name}"
            if detail:
                line += "  " + (cyan(detail) if status == OK else detail)
            print(line)
            if fix and status != OK:
                print(f"      → {yellow(fix)}")
    def counts(self):
        c = {OK: 0, WARN: 0, FAIL: 0}
        for it in self.items:
            c[it["status"]] += 1
        return c


def section(title):
    if not ARGS_JSON:
        print("\n" + bold(cyan(title)))


def run(cmd, timeout=20):
    """Exécute une commande, renvoie (rc, stdout). Ne lève jamais."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as e:
        return 1, str(e)


def py_has(py, import_name):
    rc, _ = run([py, "-c", f"import {import_name}"])
    return rc == 0


# ── checks ────────────────────────────────────────────────────────────────────

def check_system(rep):
    section("Système")
    v = sys.version_info
    rep.add(OK if v >= (3, 9) else WARN, "Python (lanceur)",
            f"{v.major}.{v.minor}.{v.micro}")
    if shutil.which("ffmpeg"):
        rc, out = run(["ffmpeg", "-version"])
        ver = out.splitlines()[0] if out else "présent"
        rep.add(OK, "ffmpeg", ver.replace("ffmpeg version ", "")[:40])
    else:
        rep.add(FAIL, "ffmpeg", "introuvable",
                "Installe-le : sudo apt install ffmpeg  (ou: brew install ffmpeg)")
    # GPU — multi-fournisseur via hw.py (NVIDIA/CUDA, AMD/ROCm, ou CPU)
    try:
        import hw
        v = hw.vendor()
        if v == "nvidia" and shutil.which("nvidia-smi"):
            rc, out = run(["nvidia-smi", "--query-gpu=name,memory.total",
                           "--format=csv,noheader"])
            rep.add(OK if rc == 0 else WARN, "GPU NVIDIA",
                    out.strip().splitlines()[0] if rc == 0 else "nvidia-smi présent")
        elif v == "amd":
            free = hw.gpu_memory_free_mib()
            rep.add(OK if hw.has_gpu() else WARN, "GPU AMD/ROCm",
                    f"gfx (Strix Halo) — device={hw.device()}, "
                    f"mémoire unifiée={hw.is_unified_memory()}, "
                    f"VRAM libre≈{free} Mio" if free is not None
                    else f"device={hw.device()} (mémoire non lisible — voir README-AMD §8)")
            if not hw.has_gpu():
                rep.add(WARN, "torch ROCm",
                        "torch ne voit pas le GPU — build ROCm ? HSA_OVERRIDE_GFX_VERSION ?",
                        "cf. README-AMD.md §2 et install-amd.sh")
        else:
            rep.add(WARN, "GPU", "absent — CPU only (WhisperX/TTS très lents)")
        # détail complet
        for line in hw.describe().splitlines():
            print(line)
    except Exception as e:
        rep.add(WARN, "GPU (hw.py)", f"détection impossible : {e}")


def find_interview_py(rep):
    section("Environnement Python « interview »")
    if os.path.exists(INTERVIEW_PY):
        rc, out = run([INTERVIEW_PY, "--version"])
        rep.add(OK, "env conda interview", out.strip() or INTERVIEW_PY)
        return INTERVIEW_PY
    # fallback : conda dispo ?
    if shutil.which("conda"):
        rep.add(FAIL, "env conda interview", "absent",
                "Crée-le : conda create -n interview python=3.11")
    else:
        rep.add(WARN, "env conda interview",
                "absent (et conda introuvable) — j'utiliserai le Python courant",
                "Installe Miniconda, ou ignore si tes paquets sont déjà globaux")
    return sys.executable


def check_packages(rep, py):
    section(f"Paquets Python  ({os.path.basename(os.path.dirname(os.path.dirname(py)))})")
    missing = []
    for imp, pip in CORE_PKGS:
        if py_has(py, imp):
            rep.add(OK, imp)
        else:
            rep.add(FAIL, imp, "manquant", f"pip install {pip}")
            missing.append(pip)
    for imp, pip in OPTIONAL_PKGS:
        if py_has(py, imp):
            rep.add(OK, imp + " (option)")
        else:
            rep.add(WARN, imp + " (option)", "manquant", f"pip install {pip}")
    # CUDA visible depuis torch ?
    if py_has(py, "torch"):
        rc, out = run([py, "-c", "import torch;print(torch.cuda.is_available())"])
        avail = "True" in out
        rep.add(OK if avail else WARN, "torch CUDA",
                "disponible" if avail else "indisponible (CPU)")
    return missing


def check_tts_envs(rep):
    section("Bridges TTS (env conda secondaires, facultatifs)")
    base = os.path.expanduser("~/miniconda3/envs")
    for env in TTS_ENVS:
        if os.path.isdir(os.path.join(base, env)):
            rep.add(OK, f"env {env}")
        else:
            rep.add(WARN, f"env {env}", "absent",
                    f"conda create -n {env} python=3.11  (requis seulement pour ce backend)")


def ollama_models():
    try:
        req = urllib.request.Request(OLLAMA_URL + "/api/tags")
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return None


def check_ollama(rep):
    section("LLM local (Ollama)")
    if not shutil.which("ollama"):
        rep.add(WARN, "ollama (binaire)", "absent",
                "Optionnel — requis seulement pour --llm local : https://ollama.com/download")
        return
    rep.add(OK, "ollama (binaire)")
    models = ollama_models()
    if models is None:
        rep.add(WARN, "serveur ollama", "injoignable",
                "Démarre-le : ollama serve  (ou lance l'app Ollama)")
        return
    rep.add(OK, "serveur ollama", f"{len(models)} modèle(s)")
    for m in OLLAMA_RECOMMENDED:
        if m in models:
            rep.add(OK, f"modèle {m}")
        else:
            rep.add(WARN, f"modèle {m}", "non installé", f"ollama pull {m}")


def check_api_keys(rep):
    section("Clés API")
    if os.environ.get("ANTHROPIC_API_KEY"):
        rep.add(OK, "ANTHROPIC_API_KEY", "définie")
    else:
        rep.add(WARN, "ANTHROPIC_API_KEY", "absente",
                "Requise pour --llm claude. Inutile en 100% local. export ANTHROPIC_API_KEY=sk-…")
    if os.environ.get("HF_TOKEN"):
        rep.add(OK, "HF_TOKEN", "définie")
    else:
        rep.add(WARN, "HF_TOKEN", "absente",
                "Requise pour la diarisation (doublage). export HF_TOKEN=hf_…")


# ── mode --install ────────────────────────────────────────────────────────────

def do_install(rep, py, missing_pkgs):
    section("Installation des correctifs")
    fails = [it for it in rep.items if it["status"] == FAIL]
    warns_pull = [it for it in rep.items if it["status"] == WARN and it["fix"].startswith("ollama pull")]

    if not fails and not missing_pkgs and not warns_pull:
        print(green("  Rien à installer — tout est déjà en place. 🎉"))
        return

    # 1) paquets pip manquants (env interview)
    if missing_pkgs:
        print(f"  Installation de {len(missing_pkgs)} paquet(s) dans {py} …")
        rc, out = run([py, "-m", "pip", "install", "--break-system-packages", *missing_pkgs],
                      timeout=1800)
        print(green("  ✓ pip OK") if rc == 0 else red("  ✗ pip a échoué :\n" + out[-600:]))

    # 2) modèles Ollama recommandés manquants
    for it in warns_pull:
        model = it["fix"].split()[-1]
        if _confirm(f"Télécharger le modèle Ollama {model} ?"):
            print(f"  Pull {model} … (peut être long)")
            rc, out = run(["ollama", "pull", model], timeout=7200)
            print(green(f"  ✓ {model}") if rc == 0 else red(f"  ✗ {model} : {out[-300:]}"))

    # 3) clés API manquantes → proposer de les écrire dans le shell rc
    _offer_api_keys(rep)


def _confirm(msg):
    try:
        return input(f"  {msg} [o/N] ").strip().lower() in ("o", "oui", "y", "yes")
    except EOFError:
        return False


def _offer_api_keys(rep):
    need = [it["name"] for it in rep.items
            if it["status"] == WARN and it["name"] in ("ANTHROPIC_API_KEY", "HF_TOKEN")]
    if not need:
        return
    rc_file = os.path.expanduser("~/.bashrc")
    for key in need:
        try:
            val = input(f"  Saisir une valeur pour {key} (vide = ignorer) : ").strip()
        except EOFError:
            val = ""
        if not val:
            continue
        if _confirm(f"Ajouter 'export {key}=…' à {rc_file} ?"):
            try:
                with open(rc_file, "a") as f:
                    f.write(f'\nexport {key}="{val}"\n')
                print(green(f"  ✓ {key} ajoutée à {rc_file} (relance ton shell)"))
            except Exception as e:
                print(red(f"  ✗ écriture impossible : {e}"))


# ── main ──────────────────────────────────────────────────────────────────────

ARGS_JSON = "--json" in sys.argv


def main():
    install = "--install" in sys.argv
    rep = Report()

    if not ARGS_JSON:
        print(bold(cyan("═" * 56)))
        print(bold(cyan("  🩺  Diagnostic — boîte à outils Traduction")))
        print(bold(cyan("═" * 56)))

    check_system(rep)
    py = find_interview_py(rep)
    missing = check_packages(rep, py)
    check_tts_envs(rep)
    check_ollama(rep)
    check_api_keys(rep)

    if install and not ARGS_JSON:
        do_install(rep, py, missing)

    c = rep.counts()
    if ARGS_JSON:
        print(json.dumps({"items": rep.items, "counts": c}, ensure_ascii=False))
        return 0 if c[FAIL] == 0 else 1

    print("\n" + bold("Résumé : ")
          + green(f"{c[OK]} OK") + "  "
          + yellow(f"{c[WARN]} avertissement(s)") + "  "
          + red(f"{c[FAIL]} bloquant(s)"))
    if c[FAIL] == 0 and c[WARN] == 0:
        print(green(bold("\n  ✅ Tout est prêt — rien à faire sur cette machine.\n")))
    elif c[FAIL] == 0:
        print(green("\n  ✅ Prêt à l'emploi. Les avertissements concernent des "
                    "fonctions optionnelles (backends TTS, LLM local, doublage).\n"))
        if not install:
            print(yellow("  Pour installer ces options : python3 doctor.py --install\n"))
    else:
        print(red(bold("\n  ⚠️  Des éléments bloquants manquent.")))
        if not install:
            print(yellow("  Lance : python3 doctor.py --install\n"))
    return 0 if c[FAIL] == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrompu.")
        sys.exit(130)
