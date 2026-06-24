"""
hw.py — Abstraction matérielle pour traduction-toolkit-AMD.

Fork AMD (ROCm) du toolkit, à l'origine 100 % NVIDIA/CUDA. Ce module centralise
TOUTE la logique dépendante du matériel pour que les scripts (doubler, resumer,
traduire, clipper, …) n'aient plus de « cuda » / « nvidia-smi » codés en dur.

Cibles :
  - AMD Ryzen AI Max+ 395 « Strix Halo » (Radeon 8060S, gfx1151) via ROCm/HIP,
    à mémoire UNIFIÉE (jusqu'à ~96-120 Go partagés CPU/GPU).
  - NVIDIA CUDA — compat. ascendante : le module fonctionne aussi sur la machine
    d'origine, ce qui permet de VALIDER toute la logique non-AMD sur du vrai
    matériel.
  - CPU pur (repli).

⚠️  SUPPOSITION PRUDENTE : les branches AMD (ROCm) n'ont PAS été testées sur le
    matériel réel. Tout ce qui est marqué « ⚠️ AMD » est à vérifier sur la
    machine cible. Voir README-AMD.md → « Checklist de validation matérielle ».

Détection du fournisseur :
  - torch compilé ROCm  → torch.version.hip est défini   → 'amd'
  - torch compilé CUDA  → torch.version.cuda défini       → 'nvidia'
  - sinon, on déduit des binaires présents (rocm-smi/amd-smi vs nvidia-smi).
  IMPORTANT : sous ROCm, PyTorch expose le GPU comme un device « cuda » (HIP
  masque CUDA), donc device() renvoie « cuda » dans les DEUX cas GPU.
"""
from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
import time

# ── Variables d'environnement ROCm (Strix Halo / gfx1151) ───────────────────
# gfx1151 n'est pas (encore) dans la matrice officielle ROCm → overrides requis.
# Réfs publiques (2026) :
#   HSA_OVERRIDE_GFX_VERSION=11.5.1  → fait passer gfx1151 pour une cible connue
#   HSA_ENABLE_SDMA=0                → corrige des segfaults ROCm (kernels récents)
#   ROCBLAS_USE_HIPBLASLT=1          → meilleur chemin long-contexte
# Ajustables via l'environnement réel (on ne SURCHARGE jamais une valeur déjà
# posée par l'utilisateur : os.environ.setdefault).
ROCM_ENV_DEFAULTS = {
    "HSA_OVERRIDE_GFX_VERSION": "11.5.1",
    "HSA_ENABLE_SDMA": "0",
    "ROCBLAS_USE_HIPBLASLT": "1",
}


@functools.lru_cache(maxsize=1)
def _torch():
    try:
        import torch
        return torch
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def vendor() -> str:
    """Renvoie 'amd' (ROCm/HIP), 'nvidia' (CUDA) ou 'cpu'."""
    t = _torch()
    if t is not None:
        if getattr(getattr(t, "version", None), "hip", None):
            return "amd"
        if getattr(getattr(t, "version", None), "cuda", None):
            return "nvidia"
    # Repli sans torch (ou torch CPU) : déduire des outils présents.
    if shutil.which("rocm-smi") or shutil.which("amd-smi"):
        return "amd"
    if shutil.which("nvidia-smi"):
        return "nvidia"
    return "cpu"


def _is_amd_lightweight() -> bool:
    """Détection AMD SANS importer torch (pour pouvoir poser l'env ROCm AVANT
    l'init de torch/HIP, où ces variables doivent déjà être présentes)."""
    return bool(shutil.which("rocm-smi") or shutil.which("amd-smi")
                or os.path.isdir("/opt/rocm"))


def setup_rocm_env(verbose: bool = False) -> None:
    """Pose les variables ROCm requises pour gfx1151 SI absentes. À appeler tôt,
    au chargement du module appelant (AVANT le premier import de torch).
    Détection légère (sans torch) ; sans effet hors AMD."""
    if not _is_amd_lightweight():
        return
    for k, v in ROCM_ENV_DEFAULTS.items():
        os.environ.setdefault(k, v)
    if verbose:
        shown = {k: os.environ.get(k) for k in ROCM_ENV_DEFAULTS}
        print(f"   [hw] env ROCm : {shown}")


@functools.lru_cache(maxsize=1)
def has_gpu() -> bool:
    """True si un GPU exploitable par torch est présent (CUDA ou ROCm/HIP)."""
    t = _torch()
    try:
        return bool(t and t.cuda.is_available())  # HIP s'expose comme 'cuda'
    except Exception:
        return False


def device() -> str:
    """Device torch. « cuda » couvre CUDA **et** ROCm/HIP ; sinon « cpu »."""
    return "cuda" if has_gpu() else "cpu"


def whisper_compute_type() -> str:
    """compute_type pour faster-whisper / WhisperX.
    fp16 sur GPU (CUDA, ou la fork CTranslate2-ROCm sur Strix Halo), int8 sur CPU.
    ⚠️ AMD : nécessite une build CTranslate2-ROCm (voir install-amd.sh) ; sans
    elle, faster-whisper retombe sur CPU et int8 serait préférable — d'où la
    possibilité de forcer via TRADUCTION_WHISPER_COMPUTE."""
    forced = os.environ.get("TRADUCTION_WHISPER_COMPUTE")
    if forced:
        return forced
    return "float16" if has_gpu() else "int8"


@functools.lru_cache(maxsize=1)
def is_unified_memory() -> bool:
    """True sur APU à mémoire unifiée (Strix Halo) : la « VRAM » est de la RAM
    système partagée (96-120 Go), donc l'OOM GPU est improbable et les mesures
    de mémoire libre sont peu fiables (carve-out UMA vs pool GTT).
    Surchargeable via TRADUCTION_UNIFIED_MEM=0/1."""
    env = os.environ.get("TRADUCTION_UNIFIED_MEM")
    if env is not None:
        return env not in ("0", "false", "no", "")
    return vendor() == "amd"  # le toolkit cible une APU Strix Halo


# ── Interrogation mémoire GPU (multi-fournisseur) ───────────────────────────
def gpu_memory_free_mib() -> "int | None":
    """Mémoire GPU libre en Mio, ou None si indéterminable.
    ⚠️ AMD/APU : la valeur peut refléter seulement le carve-out UMA et non le
    pool GTT — l'appelant DOIT dégrader gracieusement (cf. wait_for_vram_release)."""
    v = vendor()
    if v == "nvidia":
        return _query_int(["nvidia-smi", "--query-gpu=memory.free",
                           "--format=csv,noheader,nounits"])
    if v == "amd":
        val = _amd_smi_free_mib()
        return val if val is not None else _rocm_smi_free_mib()
    return None


def _query_int(cmd) -> "int | None":
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def _amd_smi_free_mib() -> "int | None":
    """amd-smi (ROCm ≥ 6.2). Format JSON variable selon version → parsing
    défensif : on cherche un champ « free » de mémoire VRAM, en octets ou Mio."""
    if not shutil.which("amd-smi"):
        return None
    for cmd in (["amd-smi", "metric", "-m", "--json"],
                ["amd-smi", "metric", "--mem-usage", "--json"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            data = json.loads(out.stdout)
            mib = _find_free_mib(data)
            if mib is not None:
                return mib
        except Exception:
            continue
    return None


def _rocm_smi_free_mib() -> "int | None":
    """rocm-smi --showmeminfo vram --json → total - used (octets)."""
    if not shutil.which("rocm-smi"):
        return None
    try:
        out = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"],
                             capture_output=True, text=True, timeout=10)
        data = json.loads(out.stdout)
        for card, fields in data.items():
            total = used = None
            for k, val in fields.items():
                kl = k.lower()
                if "vram" in kl and "total" in kl and "used" not in kl:
                    total = _to_int(val)
                elif "vram" in kl and "used" in kl:
                    used = _to_int(val)
            if total is not None and used is not None:
                return max(0, (total - used) // (1024 * 1024))
    except Exception:
        return None
    return None


def _find_free_mib(obj) -> "int | None":
    """Cherche récursivement une paire (free, unit) plausible dans un JSON
    amd-smi hétérogène. Heuristique tolérante aux variations de schéma."""
    def walk(o):
        if isinstance(o, dict):
            # cas : {"free": {"value": X, "unit": "MB"}} ou {"free": 1234}
            for key in ("free", "free_vram", "vram_free"):
                if key in o:
                    return _scalar_mib(o[key])
            for val in o.values():
                r = walk(val)
                if r is not None:
                    return r
        elif isinstance(o, list):
            for val in o:
                r = walk(val)
                if r is not None:
                    return r
        return None
    return walk(obj)


def _scalar_mib(val) -> "int | None":
    if isinstance(val, dict):
        num = _to_int(val.get("value"))
        unit = str(val.get("unit", "MB")).upper()
        if num is None:
            return None
        if unit in ("B", "BYTES"):
            return num // (1024 * 1024)
        if unit in ("KB", "KIB"):
            return num // 1024
        if unit in ("GB", "GIB"):
            return num * 1024
        return num  # MB/MiB
    return _to_int(val)


def _to_int(val) -> "int | None":
    try:
        return int(str(val).strip().split()[0])
    except Exception:
        return None


# ── Libération mémoire avant une grosse charge GPU (WhisperX, TTS) ──────────
def stop_ollama_models() -> None:
    """Décharge tout modèle Ollama résident (y compris ceux lancés hors toolkit,
    ex. open-webui). Identique sur tous les fournisseurs."""
    try:
        ps = subprocess.run(["ollama", "ps"], capture_output=True,
                            text=True, timeout=10)
        for line in ps.stdout.splitlines()[1:]:
            parts = line.split()
            if parts:
                subprocess.run(["ollama", "stop", parts[0]],
                               capture_output=True, timeout=30)
    except Exception:
        pass


def wait_for_vram_release(min_free_mib: int = 6000, timeout: float = 60.0,
                          poll: float = 1.5) -> bool:
    """Attend que la VRAM soit RÉELLEMENT libérée avant de charger un gros
    modèle GPU.

    NVIDIA (VRAM discrète) : sonde nvidia-smi jusqu'à min_free_mib libres — le
    `ollama stop` n'unmappe pas instantanément ~20 Go (cf. test 2026-06-23).

    ⚠️ AMD / mémoire UNIFIÉE : la mesure « VRAM libre » est peu fiable (UMA vs
    GTT) ET l'OOM est improbable (96-120 Go partagés). On NE bloque donc PAS sur
    un chiffre potentiellement faux : court délai puis on continue. On journalise
    quand même la valeur lue, pour diagnostic."""
    if is_unified_memory():
        free = gpu_memory_free_mib()
        msg = f"{free} Mio" if free is not None else "inconnue"
        print(f"   ✅ Mémoire unifiée (APU) — pas de blocage VRAM (libre lue : {msg})")
        time.sleep(2)
        return True

    deadline = time.time() + timeout
    last_free = None
    while time.time() < deadline:
        last_free = gpu_memory_free_mib()
        if last_free is None:
            return True  # indéterminable (CPU / pas d'outil) → ne pas attendre
        if last_free >= min_free_mib:
            print(f"   ✅ VRAM libérée : {last_free} Mo disponibles")
            return True
        time.sleep(poll)
    print(f"   ⚠️  VRAM encore basse après {timeout:.0f}s "
          f"({last_free} Mo libres) — chargement tenté malgré tout")
    return False


def free_gpu_for_task(min_free_mib: int = 6000, timeout: float = 60.0) -> None:
    """Décharge un éventuel modèle Ollama résident PUIS vérifie/attend la VRAM
    avant de charger WhisperX ou le TTS. Le verrou GPU du toolkit sérialise les
    tâches internes, mais Ollama (keep_alive) vit EN DEHORS du verrou.

    Sur mémoire unifiée, l'étape `ollama stop` reste utile (réduit la pression
    mémoire / bande passante partagée) même si l'attente VRAM est assouplie."""
    stop_ollama_models()
    wait_for_vram_release(min_free_mib=min_free_mib, timeout=timeout)


# ── Diagnostic (doctor.py) ──────────────────────────────────────────────────
def describe() -> str:
    """Résumé lisible de l'état matériel détecté."""
    v = vendor()
    lines = [f"fournisseur : {v}"]
    t = _torch()
    if t is not None:
        ver = getattr(t, "__version__", "?")
        hip = getattr(getattr(t, "version", None), "hip", None)
        cu = getattr(getattr(t, "version", None), "cuda", None)
        lines.append(f"torch {ver} (hip={hip}, cuda={cu}), gpu={has_gpu()}")
    else:
        lines.append("torch absent")
    if v == "amd":
        lines.append(f"mémoire unifiée : {is_unified_memory()}")
        lines.append("env ROCm : " + ", ".join(
            f"{k}={os.environ.get(k, '∅')}" for k in ROCM_ENV_DEFAULTS))
    free = gpu_memory_free_mib()
    if free is not None:
        lines.append(f"VRAM libre (approx.) : {free} Mio")
    lines.append(f"device torch : {device()} | whisper compute_type : {whisper_compute_type()}")
    return "\n".join("   " + ln for ln in lines)


if __name__ == "__main__":
    setup_rocm_env(verbose=True)
    print(describe())
