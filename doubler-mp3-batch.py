#!/usr/bin/env python3
"""
Pipeline de traduction vocale IA — batch MP3/MP4 (Qwen3-TTS / XTTS)
==================================================================================
Traite tous les fichiers .mp3 et .mp4 du dossier courant et produit
pour chacun un MP3 traduit dans la langue cible. La piste de sortie
contient UNIQUEMENT la voix synthétisée (pas de bande originale, pas
de mixage). Pour les MP4, seul l'audio est utilisé (la vidéo est ignorée).

Philosophie : rendu NATUREL, respectueux de la ponctuation et du sens.
Aucune contrainte de durée, aucune isochronie, aucun calage temporel.
Si l'output dure plus longtemps que l'original, c'est parfait — on ne
fait pas du doublage mais de la traduction orale pure.

Architecture SEGMENT-PAR-SEGMENT en 9 passes :
  1. WhisperX          → transcription + timestamps (segments naturels)
  2. Pyannote           → diarisation (identification des locuteurs)
  3. Demucs             → séparation voix (pour échantillons de clonage)
  3b. Échantillons      → extraction échantillons vocaux par locuteur
  4a. Claude            → analyse du contenu (glossaire, ton, locuteurs)
  4b. Claude            → traduction par segments avec CONTEXTE fenêtré
                          (chaque segment est traduit individuellement,
                           mais Claude voit les segments voisins pour le contexte)
  4c. Claude            → relecture hiérarchisée (fidélité, naturel, glossaire,
                          politesse, fluidité, chants/prières)
  4d. Claude            → cohérence globale (terminologie, registre, ton)
  4e. Claude            → vérification glossaire (violations → corrections)
  5. TTS               → synthèse vocale SEGMENT PAR SEGMENT
                          (chaque segment WhisperX = 1 clip TTS, pas de split)
  5b. Normalisation     → cohérence volume RMS par locuteur
  6. Assemblage         → concaténation séquentielle des segments → MP3

POURQUOI SEGMENT-PAR-SEGMENT (et pas par blocs fusionnés) :
  Les segments WhisperX sont courts (~50-150 caractères), ce qui est idéal
  pour XTTS v2 dont le décodeur autorégressif boucle au-delà de ~250 car.
  En synthétisant chaque segment individuellement :
  - Pas besoin de re-découper en phrases (split_into_sentences)
  - Pas de crossfade destructif entre phrases
  - Pas de _trim_silence qui mange les consonnes douces
  - Chaque segment tient dans les limites XTTS sans manipulation
  
  Le contexte de traduction est préservé grâce au fenêtrage : Claude voit
  les segments précédents/suivants mais ne traduit que la fenêtre courante.

Usage :
  python doubler-mp3-batch.py                                    # EN → FR, tous les MP3+MP4
  python doubler-mp3-batch.py -s en -t es                        # EN → ES
  python doubler-mp3-batch.py --xtts-speaker "Craig Gutsy"       # voix preset XTTS
  python doubler-mp3-batch.py --ref-voice ref_fr.wav             # voix de référence
  python doubler-mp3-batch.py --speakers 1                       # monologue (pas de diarisation)
  python doubler-mp3-batch.py --context "podcast tech, registre familier"
  python doubler-mp3-batch.py --file specific.mp3                # un seul fichier
  python doubler-mp3-batch.py --file interview.mp4               # un seul MP4 (audio extrait)
  python doubler-mp3-batch.py --pause 800                        # 800ms entre segments

Prérequis :
  pip install whisperx anthropic torch torchaudio demucs pydub soundfile \\
              numpy praat-parselmouth TTS --break-system-packages
  # + ffmpeg installé
  # + ANTHROPIC_API_KEY
  # + HF_TOKEN (pour pyannote — diarisation)
"""

import argparse
import gc
import glob
import json
import math
import fcntl
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Répertoires structurés ────────────────────────────────────────────────
SCRIPT_DIR   = Path(os.path.abspath(__file__)).parent
INPUT_DIR    = Path(os.environ.get("TRADUCTION_INPUT_DIR",  str(SCRIPT_DIR / "input")))
WORK_DIR     = Path(os.environ.get("TRADUCTION_WORK_DIR",   str(SCRIPT_DIR / "work-files")))
OUTPUT_DIR   = Path(os.environ.get("TRADUCTION_OUTPUT_DIR", str(SCRIPT_DIR / "output")))

def resolve_source(path_str: str) -> Path:
    """Résout le fichier source : chemin absolu → tel quel, nom simple → INPUT_DIR."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    candidate = INPUT_DIR / p.name
    if candidate.exists():
        return candidate
    return p  # laisse l'appelant vérifier l'existence

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

WHISPER_MODEL = "large-v3"
WHISPER_BATCH_SIZE = 16
WHISPER_COMPUTE_TYPE = "float16"

CLAUDE_MODEL = "claude-opus-4-5"  # cf. A/B 2026-05-25 (traduire.py) : sonnet-4-6
# produit doublons synonymiques et étoffements malgré prompt explicite.
CLAUDE_MAX_TOKENS = 8192
CLAUDE_RETRY_MAX = 5
CLAUDE_RETRY_DELAY = 10.0

SAMPLE_RATE = 44100
MIN_SPEAKER_SAMPLE_SEC = 5
MAX_SPEAKER_SAMPLE_SEC = 30

# Assemblage
DEFAULT_PAUSE_MS = 600          # pause entre segments du même locuteur
DEFAULT_SPEAKER_PAUSE_MS = 900  # pause lors d'un changement de locuteur

# Noms de langues
LANGUAGE_NAMES = {
    "en": "anglais", "fr": "français", "es": "espagnol", "de": "allemand",
    "it": "italien", "pt": "portugais", "nl": "néerlandais", "ru": "russe",
    "ja": "japonais", "zh": "chinois", "ko": "coréen", "ar": "arabe",
    "hi": "hindi", "tr": "turc", "pl": "polonais", "sv": "suédois",
    "da": "danois", "no": "norvégien", "fi": "finnois", "cs": "tchèque",
    "ro": "roumain", "hu": "hongrois", "el": "grec", "he": "hébreu",
    "th": "thaï", "vi": "vietnamien", "uk": "ukrainien", "id": "indonésien",
    "ms": "malais", "ca": "catalan", "eu": "basque", "gl": "galicien",
}

LANGUAGE_NAMES_EN = {
    "en": "English", "fr": "French", "es": "Spanish", "de": "German",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
    "ja": "Japanese", "zh": "Chinese", "ko": "Korean", "ar": "Arabic",
    "hi": "Hindi", "tr": "Turkish", "pl": "Polish", "sv": "Swedish",
    "da": "Danish", "no": "Norwegian", "fi": "Finnish", "cs": "Czech",
    "ro": "Romanian", "hu": "Hungarian", "el": "Greek", "he": "Hebrew",
    "th": "Thai", "vi": "Vietnamese", "uk": "Ukrainian", "id": "Indonesian",
    "ms": "Malay", "ca": "Catalan", "eu": "Basque", "gl": "Galician",
}

def lang_name(code, in_english=False):
    d = LANGUAGE_NAMES_EN if in_english else LANGUAGE_NAMES
    return d.get(code, code.upper())


# Fragments de prompt connus — si le texte de sortie en contient un,
# c'est une fuite de prompt, pas une vraie traduction/adaptation.
_PROMPT_LEAK_FRAGMENTS = [
    "fluide et naturel",
    "prononcé à voix haute",
    "prononcé à haute voix",
    "style télégraphique",
    "nombre de caractères",
    "aucune correction",
    "voici le texte",
    "texte adapté",
    "texte corrigé",
    "mots de liaison",
    "sujet et un verbe",
    "sujet + verbe",
    "reformule le texte",
    "caractères cible",
    "cible ≈",           # instruction d'adaptation isochrone
    "déborde de",        # instruction de raccourcissement
    "car.)",             # fin de métadonnée d'adaptation
]

def _est_fuite_prompt(texte_nouveau: str, texte_original: str = "") -> bool:
    """Détecte si texte_nouveau est une fuite de prompt plutôt qu'une adaptation."""
    t = texte_nouveau.lower()
    # Vérification par fragments de prompt connus
    for f in _PROMPT_LEAK_FRAGMENTS:
        if f in t:
            return True
    # Vérification par chevauchement de mots : si le nouveau texte ne partage
    # quasi aucun mot avec l'original, c'est probablement du méta-commentaire
    if texte_original:
        mots_orig = set(texte_original.lower().split())
        mots_new = set(t.split())
        # Seulement pour les textes assez longs (≥5 mots chacun)
        if len(mots_orig) >= 5 and len(mots_new) >= 5:
            overlap = len(mots_orig & mots_new)
            if overlap < 2:
                return True
    return False


def _strip_claude_artifacts(txt: str) -> str:
    """Nettoie les artefacts courants de la sortie Claude."""
    # Notation fléchée du reviewer : "ancien" → "nouveau"
    if '→' in txt:
        txt = txt.split('→', 1)[-1].strip().strip('""«»\u201c\u201d').strip()
    # Préfixe de langue (FR:, EN:, etc.)
    txt = re.sub(r'^[A-Z]{2}:\s*', '', txt).strip()
    # Métadonnées entre parenthèses en début de ligne
    txt = re.sub(r'^\([^)]*\)\s*', '', txt).strip()
    return txt


def _parse_translations(text, segments, s, e):
    """Parse les traductions Claude et les affecte aux segments, avec garde anti-fuite."""
    for line in text.strip().split("\n"):
        m = re.match(r'\[S(\d+)\]\s*(.*)', line.strip())
        if m:
            sid, txt = int(m.group(1)), m.group(2).strip()
            txt = _strip_claude_artifacts(txt)
            if txt and not _est_fuite_prompt(txt):
                for seg in segments[s:e]:
                    if seg.index == sid:
                        seg.text_tgt = txt; break


def _retry_missing(segments, s, e, system, client, source_lang="en", target_lang="fr"):
    """Retry hiérarchique à 2 niveaux pour les segments manquants."""
    # Retry 1 : contexte minimal
    missing = [seg for seg in segments[s:e] if not seg.text_tgt]
    if not missing: return
    parts = ["Segments manquants à traduire :\n"]
    for seg in missing:
        ctx = [x for x in segments[max(0, seg.index-4):seg.index-1] if x.text_tgt]
        if ctx: parts.append(f"  (contexte: [S{ctx[-1].index}] {ctx[-1].text_tgt})")
        parts.append(f"[S{seg.index}] {seg.text}")
    resp = _claude_create(client, model=CLAUDE_MODEL, max_tokens=CLAUDE_MAX_TOKENS,
                          system=system,
                          messages=[{"role": "user", "content": "\n".join(parts)}])
    _parse_translations(resp.content[0].text, segments, s, e)

    # Retry 2 : contexte bilingue étendu (3 segments avant/après chaque manquant)
    missing2 = [seg for seg in segments[s:e] if not seg.text_tgt]
    if not missing2: return
    print(f"   ⚠️  Retry 2 avec contexte bilingue étendu ({len(missing2)} manquants)...")
    parts2 = ["Segments toujours manquants — contexte bilingue étendu :\n"]
    for seg in missing2:
        # 3 segments traduits avant
        before = [x for x in segments[max(0, seg.index-4):seg.index] if x.text_tgt][-3:]
        for bseg in before:
            parts2.append(f"  [S{bseg.index}] {source_lang.upper()}: {bseg.text}")
            parts2.append(f"  [S{bseg.index}] {target_lang.upper()}: {bseg.text_tgt}")
        parts2.append(f"[S{seg.index}] {seg.text}")
        # 3 segments traduits après
        after = [x for x in segments[seg.index+1:min(len(segments), seg.index+4)] if x.text_tgt][:3]
        for aseg in after:
            parts2.append(f"  [S{aseg.index}] {source_lang.upper()}: {aseg.text}")
            parts2.append(f"  [S{aseg.index}] {target_lang.upper()}: {aseg.text_tgt}")
        parts2.append("")
    resp2 = _claude_create(client, model=CLAUDE_MODEL, max_tokens=CLAUDE_MAX_TOKENS,
                           system=system,
                           messages=[{"role": "user", "content": "\n".join(parts2)}])
    _parse_translations(resp2.content[0].text, segments, s, e)


# Ollama (LLM local — alternative gratuite à l'API Claude)
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma4:31b"        # cf. bench 2026-06-22 : meilleur FR oral (traduction)
OLLAMA_NUM_PREDICT = 16384         # marge large (tokens de réflexion inclus)


class _OllamaClient:
    """Client Ollama mimant l'interface Anthropic (client.messages.create)."""

    def __init__(self, base_url, model):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.messages = self            # client.messages.create → self.create

    def create(self, **kwargs):
        import urllib.request
        msgs = []
        system = kwargs.get("system", "")
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(kwargs.get("messages", []))
        payload = json.dumps({
            "model": self.model,
            "messages": msgs,
            "stream": False,
            "think": False,            # désactive la réflexion (raisonneurs type qwen3.6/qwen3)
            "options": {"num_predict": OLLAMA_NUM_PREDICT},
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read())
        text = data["message"]["content"]
        # Nettoyer les balises <think>…</think> (mode réflexion des raisonneurs)
        text = re.sub(r'<think>[\s\S]*?</think>\s*', '', text)
        return type("Resp", (), {"content": [type("B", (), {"text": text})()]})()


def _claude_create(client, **kwargs):
    """Appel LLM avec retry automatique (API Claude ou Ollama local)."""
    is_local = isinstance(client, _OllamaClient)
    for attempt in range(1, CLAUDE_RETRY_MAX + 1):
        try:
            return client.messages.create(**kwargs)
        except Exception as exc:
            retryable = False
            if is_local:
                retryable = True        # Ollama : retry sur toute erreur réseau/timeout
            else:
                import anthropic
                if isinstance(exc, anthropic.APIStatusError):
                    status = getattr(exc, 'status_code', 0)
                    retryable = status in (429, 529) or status >= 500
                elif isinstance(exc, anthropic.APIConnectionError):
                    retryable = True
            if retryable and attempt < CLAUDE_RETRY_MAX:
                delay = CLAUDE_RETRY_DELAY * (2 ** (attempt - 1))
                label = "Ollama" if is_local else "API Claude"
                print(f"   ⏳ {label} erreur ({type(exc).__name__}), retry {attempt}/{CLAUDE_RETRY_MAX} dans {delay:.0f}s...")
                time.sleep(delay)
            else:
                raise


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURES DE DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Segment:
    """Segment de transcription enrichi pour la traduction vocale."""
    index: int
    start: float
    end: float
    text: str                               # texte source
    text_tgt: str = ""                      # traduction
    speaker: str = "SPEAKER_00"
    tts_path: str = ""
    words: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class SpeakerProfile:
    """Profil d'un locuteur détecté."""
    speaker_id: str
    sample_path: str = ""
    sample_text: str = ""                   # transcription de l'échantillon
    total_duration: float = 0.0
    segment_count: int = 0
    ref_clips: list = field(default_factory=list)  # [(path, text), ...]
    gender: str = "unknown"
    f0_median: float = 0.0


# Limites de caractères XTTS par langue (au-delà, l'audio est tronqué)
# Source : Coqui TTS tokenizer limits. On prend une marge de sécurité.
XTTS_CHAR_LIMITS = {
    "fr": 273, "en": 250, "es": 253, "de": 253, "it": 213, "pt": 253,
    "pl": 253, "tr": 226, "ru": 182, "nl": 253, "cs": 253, "ar": 166,
    "zh": 82, "ja": 100, "ko": 100, "hu": 253, "hi": 150,
}
XTTS_CHAR_LIMIT_DEFAULT = 230  # fallback conservateur

# ═══════════════════════════════════════════════════════════════════════════════
# VÉRIFICATIONS
# ═══════════════════════════════════════════════════════════════════════════════


# Constantes Qwen3-TTS (Alibaba Qwen — local, GPU, 10 langues dont FR, Apache 2.0)
QWEN3TTS_MAX_CHARS = 300
QWEN3TTS_LANG_MAP = {
    "zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
    "de": "German", "fr": "French", "ru": "Russian", "pt": "Portuguese",
    "es": "Spanish", "it": "Italian",
}

def _get_qwen3tts_python() -> str:
    """Retourne le chemin du python de l'env conda 'qwen3tts', ou '' si introuvable."""
    for base in [os.path.expanduser("~/miniconda3"), os.path.expanduser("~/anaconda3"),
                 "/opt/conda", os.path.expanduser("~/miniforge3")]:
        py = os.path.join(base, "envs", "qwen3tts", "bin", "python")
        if os.path.isfile(py):
            return py
    try:
        r = subprocess.run(["conda", "info", "--envs"], capture_output=True, text=True, timeout=10)
        for line in r.stdout.splitlines():
            if "qwen3tts" in line and "/envs/" in line:
                env_path = line.split()[-1].strip()
                py = os.path.join(env_path, "bin", "python")
                if os.path.isfile(py):
                    return py
    except Exception:
        pass
    return ""

QWEN3TTS_BRIDGE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "qwen3tts_bridge.py")


def check_dependencies(tts_backend="qwen3tts", local=False):
    """Vérifie toutes les dépendances du pipeline."""
    print("🔍 Vérification des dépendances...")
    ok = True

    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        v = r.stdout.split('\n')[0].split(' ')[2] if r.returncode == 0 else "?"
        print(f"   ffmpeg      : ✅ ({v})")
    except FileNotFoundError:
        print("   ffmpeg      : ❌  → sudo apt install ffmpeg"); ok = False

    for pkg, name in [("whisperx", "whisperx"), ("anthropic", "anthropic"),
                      ("demucs", "demucs"), ("pydub", "pydub"),
                      ("soundfile", "soundfile"), ("numpy", "numpy"),
                      ("parselmouth", "praat-parselmouth")]:
        try:
            __import__(pkg); print(f"   {name:12s} : ✅")
        except ImportError:
            print(f"   {name:12s} : ❌  → pip install {name} --break-system-packages"); ok = False

    # TTS backend
    if tts_backend == "qwen3tts":
        qwen3_py = _get_qwen3tts_python()
        if qwen3_py:
            print("   qwen3tts    : ✅  (env isolé)")
        else:
            print("   qwen3tts    : ❌  → conda create -n qwen3tts python=3.12 && "
                  "conda run -n qwen3tts pip install -U qwen-tts soundfile"); ok = False
    else:
        try:
            __import__("TTS"); print("   xtts-v2     : ✅")
        except ImportError:
            print("   xtts-v2     : ❌  → pip install TTS --break-system-packages"); ok = False

    if not local and not os.environ.get("ANTHROPIC_API_KEY"):
        print("   ⚠️  ANTHROPIC_API_KEY non définie"); ok = False

    if not ok:
        print("\n❌ Dépendances manquantes. Installez-les et relancez.")
        sys.exit(1)
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 1 : EXTRACTION AUDIO + TRANSCRIPTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_audio_16k(input_path: str, output_path: str) -> str:
    """Extrait/convertit l'audio en WAV mono 16kHz pour WhisperX (MP3 ou MP4)."""
    ext = Path(input_path).suffix.lower()
    label = "vidéo" if ext == ".mp4" else "audio"
    print(f"\n🎵 Passe 1a — Extraction audio depuis {label} (16kHz mono)...")
    cmd = ["ffmpeg", "-y", "-i", input_path,
           "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"❌ ffmpeg : {r.stderr}"); sys.exit(1)
    print(f"   ✅ {output_path}")
    return output_path


def extract_audio_hq(input_path: str, output_path: str) -> str:
    """Extrait/convertit l'audio en WAV stéréo 44.1kHz pour Demucs (MP3 ou MP4)."""
    ext = Path(input_path).suffix.lower()
    label = "vidéo" if ext == ".mp4" else "audio"
    print(f"🎵 Passe 1b — Extraction audio HQ depuis {label} (44.1kHz stéréo)...")
    cmd = ["ffmpeg", "-y", "-i", input_path,
           "-vn", "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "2", output_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"❌ ffmpeg : {r.stderr}"); sys.exit(1)
    mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"   ✅ {output_path} ({mb:.1f} Mo)")
    return output_path


# ── Verrou GPU global du toolkit ──────────────────────────────────────────────
_GPU_LOCK_PATH = os.path.expanduser("~/.cache/traduction_gpu.lock")
_gpu_lock_fh = None


def acquire_gpu_lock():
    """Sérialise toutes les tâches GPU du toolkit (WhisperX, TTS, LLM local) via
    un verrou fichier partagé : empêche deux tâches GPU de tourner en même temps
    (→ plus d'OOM / crash GPU). Le verrou est tenu jusqu'à la fin du processus.
    """
    global _gpu_lock_fh
    if _gpu_lock_fh is not None:
        return
    os.makedirs(os.path.dirname(_GPU_LOCK_PATH), exist_ok=True)
    _gpu_lock_fh = open(_GPU_LOCK_PATH, "w")
    try:
        fcntl.flock(_gpu_lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("   ⏳ GPU occupé par une autre tâche — attente du verrou…", flush=True)
        fcntl.flock(_gpu_lock_fh, fcntl.LOCK_EX)
    print("   🔒 Verrou GPU acquis")


def transcribe_whisperx(audio_path: str, source_lang: str,
                        hf_token: Optional[str] = None) -> list[Segment]:
    """Transcrit avec WhisperX + alignement mot par mot."""
    acquire_gpu_lock()
    import whisperx, torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n📝 Passe 1c — Transcription WhisperX ({WHISPER_MODEL}) [{device}]...")

    t0 = time.time()
    model = whisperx.load_model(WHISPER_MODEL, device,
                                compute_type=WHISPER_COMPUTE_TYPE, language=source_lang)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=WHISPER_BATCH_SIZE, language=source_lang)
    print(f"   Transcription : {time.time()-t0:.1f}s")

    print("   🔧 Alignement mot par mot...")
    t1 = time.time()
    model_a, metadata = whisperx.load_align_model(language_code=source_lang, device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device,
                            return_char_alignments=False)
    print(f"   Alignement : {time.time()-t1:.1f}s")

    del model, model_a; gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    segments = [
        Segment(index=i+1, start=s["start"], end=s["end"],
                text=s["text"].strip(), words=s.get("words", []))
        for i, s in enumerate(result["segments"])
    ]
    dur = segments[-1].end if segments else 0
    print(f"   ✅ {len(segments)} segments ({dur/60:.1f} min)")
    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 2 : DIARISATION
# ═══════════════════════════════════════════════════════════════════════════════

def diarize_speakers(audio_path: str, segments: list[Segment],
                     hf_token: str, num_speakers: Optional[int] = None) -> list[Segment]:
    """Identifie qui parle quand avec pyannote via WhisperX."""
    import whisperx, torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n👥 Passe 2 — Diarisation des locuteurs...")

    if not hf_token:
        print("   ⚠️  HF_TOKEN manquant — tous les segments assignés à SPEAKER_00")
        return segments

    t0 = time.time()
    from whisperx.diarize import DiarizationPipeline
    diarize_model = DiarizationPipeline(token=hf_token, device=device)

    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers

    audio = whisperx.load_audio(audio_path)
    diarize_result = diarize_model(audio, **kwargs)

    whisperx_segments = [
        {"start": s.start, "end": s.end, "text": s.text, "words": s.words}
        for s in segments
    ]
    result = whisperx.assign_word_speakers(diarize_result, {"segments": whisperx_segments})

    for i, seg_data in enumerate(result["segments"]):
        if i < len(segments):
            segments[i].speaker = seg_data.get("speaker") or "SPEAKER_00"

    # Consolider si trop de locuteurs
    if num_speakers:
        unique_speakers = {}
        for s in segments:
            unique_speakers.setdefault(s.speaker, 0.0)
            unique_speakers[s.speaker] += s.duration

        if len(unique_speakers) > num_speakers:
            ranked = sorted(unique_speakers.items(), key=lambda x: x[1], reverse=True)
            keep = {spk for spk, _ in ranked[:num_speakers]}
            extra = {spk for spk in unique_speakers if spk not in keep}

            print(f"   🔧 Consolidation : {len(unique_speakers)} → {num_speakers}")

            keep_segments = {}
            for s in segments:
                if s.speaker in keep:
                    keep_segments.setdefault(s.speaker, []).append(
                        (s.start + s.end) / 2)

            for s in segments:
                if s.speaker in extra:
                    mid = (s.start + s.end) / 2
                    best_spk, best_dist = None, float('inf')
                    for spk, mids in keep_segments.items():
                        dist = min(abs(mid - m) for m in mids)
                        if dist < best_dist:
                            best_dist, best_spk = dist, spk
                    s.speaker = best_spk or ranked[0][0]

    speakers = {}
    for s in segments:
        speakers.setdefault(s.speaker, {"count": 0, "dur": 0.0})
        speakers[s.speaker]["count"] += 1
        speakers[s.speaker]["dur"] += s.duration

    del diarize_model; gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"   ✅ {len(speakers)} locuteur(s) ({time.time()-t0:.1f}s)")
    for spk, info in sorted(speakers.items()):
        print(f"      {spk} : {info['count']} segments, {info['dur']:.1f}s")

    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 3 : SÉPARATION DE SOURCES (DEMUCS) — pour échantillons vocaux
# ═══════════════════════════════════════════════════════════════════════════════

def _get_audio_duration(audio_path: str) -> float:
    """Retourne la durée en secondes via ffprobe."""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "csv=p=0", audio_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        return float(r.stdout.strip())
    return 0.0


def separate_sources(audio_hq_path: str, work_dir: str) -> str:
    """
    Sépare voix et fond sonore avec Demucs.
    On ne retourne que le stem vocal (pour les échantillons de clonage).
    """
    print(f"\n🎛️  Passe 3 — Séparation de sources (Demucs)...")
    t0 = time.time()

    out_dir = os.path.join(work_dir, "demucs_out")
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems=vocals",
        "--segment", "6",
        "--overlap", "0.25",
        "-o", out_dir,
        "--filename", "{stem}.{ext}",
        audio_hq_path
    ]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"   ❌ Demucs erreur :\n{r.stderr[-600:]}")
        return ""

    model_dir = os.path.join(out_dir, "htdemucs")
    vocals = os.path.join(model_dir, "vocals.wav")

    if not os.path.exists(vocals):
        for root, dirs, files in os.walk(out_dir):
            for f in files:
                if "vocal" in f.lower() and f.endswith(".wav"):
                    if "no_vocal" not in f.lower() and "no-vocal" not in f.lower():
                        vocals = os.path.join(root, f)

    if not os.path.exists(vocals):
        print(f"   ❌ vocals.wav introuvable dans {out_dir}")
        return ""

    mv = os.path.getsize(vocals) / (1024*1024)
    print(f"   ✅ vocals.wav ({mv:.1f} Mo) [{time.time()-t0:.0f}s]")
    return vocals


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 3b : EXTRACTION DES ÉCHANTILLONS PAR LOCUTEUR
# ═══════════════════════════════════════════════════════════════════════════════

def _estimate_gender(audio_mono, sr,
                     threshold_female=165.0, threshold_male=155.0):
    """Estime le genre par analyse F0 (autocorrélation)."""
    import numpy as np

    max_samples = int(10 * sr)
    audio = audio_mono[:max_samples] if len(audio_mono) > max_samples else audio_mono

    frame_size = int(0.04 * sr)
    hop = int(0.02 * sr)
    f0_min, f0_max = 70, 400
    lag_min = sr // f0_max
    lag_max = sr // f0_min

    f0_values = []
    for start in range(0, len(audio) - frame_size, hop):
        frame = audio[start:start + frame_size]
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < 0.005:
            continue

        frame = frame - np.mean(frame)
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr) // 2:]
        if corr[0] == 0:
            continue
        corr = corr / corr[0]

        search = corr[lag_min:min(lag_max, len(corr))]
        if len(search) < 2:
            continue
        peak_idx = np.argmax(search)
        if search[peak_idx] > 0.3:
            f0_values.append(sr / (lag_min + peak_idx))

    if not f0_values:
        return "unknown", 0.0

    f0_median = float(np.median(f0_values))
    if f0_median > threshold_female:
        return "female", f0_median
    elif f0_median < threshold_male:
        return "male", f0_median
    return "unknown", f0_median


def extract_speaker_samples(segments: list[Segment], vocals_path: str,
                            work_dir: str) -> dict[str, SpeakerProfile]:
    """Extrait des échantillons audio par locuteur depuis le stem vocal."""
    import soundfile as sf
    import numpy as np

    print(f"\n🎤 Passe 3b — Extraction des échantillons vocaux...")

    samples_dir = os.path.join(work_dir, "speaker_samples")
    os.makedirs(samples_dir, exist_ok=True)

    vocals, sr = sf.read(vocals_path)
    if vocals.ndim == 2:
        vocals = vocals.mean(axis=1)

    speaker_segs: dict[str, list[Segment]] = {}
    for seg in segments:
        speaker_segs.setdefault(seg.speaker, []).append(seg)

    profiles: dict[str, SpeakerProfile] = {}

    for spk_id, segs in speaker_segs.items():
        segs_sorted = sorted(segs, key=lambda s: s.duration, reverse=True)
        total_dur = sum(s.duration for s in segs)
        profile = SpeakerProfile(
            speaker_id=spk_id,
            total_duration=total_dur,
            segment_count=len(segs)
        )

        # Concaténer les meilleurs segments comme échantillon
        sample_chunks = []
        sample_texts = []
        accumulated = 0.0
        for seg in segs_sorted:
            if accumulated >= MAX_SPEAKER_SAMPLE_SEC:
                break
            if seg.duration < 0.5:
                continue
            start_s = int(seg.start * sr)
            end_s = min(int(seg.end * sr), len(vocals))
            if start_s < end_s:
                sample_chunks.append(vocals[start_s:end_s])
                sample_texts.append(seg.text.strip())
                accumulated += seg.duration

        if sample_chunks:
            silence = np.zeros(int(0.1 * sr))
            full_sample = np.concatenate(
                [x for chunk in sample_chunks for x in [chunk, silence]][:-1]
            )
            sample_path = os.path.join(samples_dir, f"{spk_id}.wav")
            sf.write(sample_path, full_sample, sr)
            profile.sample_path = sample_path
            profile.sample_text = " ".join(sample_texts)

        # Clips individuels pour clonage
        ref_clips = []
        for ci, seg in enumerate(segs_sorted):
            if len(ref_clips) >= 10:
                break
            if seg.duration < 3.0 or seg.duration > 15.0:
                continue
            start_s = int(seg.start * sr)
            end_s = min(int(seg.end * sr), len(vocals))
            if start_s >= end_s:
                continue
            clip_path = os.path.join(samples_dir, f"{spk_id}_ref{ci:02d}.wav")
            sf.write(clip_path, vocals[start_s:end_s], sr)
            ref_clips.append((clip_path, seg.text.strip()))

        profile.ref_clips = ref_clips

        # Genre
        if sample_chunks:
            gender, f0_median = _estimate_gender(full_sample, sr)
            profile.gender = gender
            profile.f0_median = f0_median
            icon = "♀️" if gender == "female" else "♂️" if gender == "male" else "❓"
            print(f"   🎤 {spk_id} {icon} : {accumulated:.1f}s échantillon, "
                  f"{len(ref_clips)} clips ref "
                  f"({len(segs)} seg, {total_dur:.0f}s) [F0={f0_median:.0f}Hz]")

        profiles[spk_id] = profile

    return profiles


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 4 : ANALYSE + TRADUCTION PAR SEGMENTS (CLAUDE)
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_content(segments, client, src_lang, tgt_lang, context=""):
    """Analyse le contenu pour guider la traduction."""
    print(f"\n🔍 Passe 4a — Analyse du contenu...")

    full = "\n".join(f"[{s.index}] ({s.speaker}) {s.text}" for s in segments)
    if len(full) > 80000:
        q = len(segments) // 4
        mid = len(segments) // 2
        full = ("\n".join(f"[{s.index}] ({s.speaker}) {s.text}" for s in segments[:q])
                + "\n[...]\n"
                + "\n".join(f"[{s.index}] ({s.speaker}) {s.text}" for s in segments[mid-q//2:mid+q//2])
                + "\n[...]\n"
                + "\n".join(f"[{s.index}] ({s.speaker}) {s.text}" for s in segments[-q:]))

    ctx = f"\nCONTEXTE UTILISATEUR :\n{context}\n" if context else ""
    src_n, tgt_n = lang_name(src_lang), lang_name(tgt_lang)

    prompt = f"""Tu es un traducteur professionnel {src_n} → {tgt_n}.
{ctx}
Analyse cette transcription et fournis en JSON strict :
- "summary": résumé 3-5 phrases (en {tgt_n})
- "glossary": {{"terme_{src_lang}": "traduction_{tgt_lang}"}} pour termes techniques/noms/expressions
- "speakers_description": description de chaque locuteur identifié (voix, rôle, registre)
- "tone": registre global
- "domain": domaine principal

TRANSCRIPTION :
{full}

Réponds UNIQUEMENT en JSON, sans markdown."""

    resp = _claude_create(client, model=CLAUDE_MODEL, max_tokens=4096,
                          messages=[{"role": "user", "content": prompt}])
    txt = resp.content[0].text

    jm = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', txt, re.DOTALL)
    if jm: txt = jm.group(1)
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        bs, be = txt.find('{'), txt.rfind('}') + 1
        try: data = json.loads(txt[bs:be]) if bs >= 0 else {}
        except: data = {}

    print(f"   📋 {data.get('summary', '')[:120]}...")
    print(f"   📖 {len(data.get('glossary', {}))} termes | 🎭 {data.get('tone', '?')}")
    return data


def translate_segments(segments, analysis, client, src_lang, tgt_lang, context=""):
    """
    Traduction SEGMENT PAR SEGMENT avec contexte fenêtré.

    Chaque segment WhisperX est traduit individuellement, mais Claude voit
    les segments précédents et suivants pour le contexte. Cela donne :
    - Des traductions qui tiennent compte du contexte (comme les blocs)
    - Des segments courts qui passent directement dans XTTS (pas de split)
    - Aucune manipulation destructive (pas de merge/split/crossfade)
    """
    print(f"\n🌍 Passe 4b — Traduction par segments {src_lang}→{tgt_lang}...")

    src_n, tgt_n = lang_name(src_lang), lang_name(tgt_lang)
    user_ctx = f"\nINSTRUCTIONS : {context}\n" if context else ""

    # Règles de registre de politesse
    lang_rules = {
        "fr": "Tutoiement/vouvoiement cohérent selon le contexte et le registre.",
        "de": "Du/Sie cohérent selon le contexte et le registre.",
        "es": "Tú/Usted cohérent selon le contexte et le registre.",
        "ja": "Niveau de politesse (敬語) cohérent selon le contexte.",
        "ko": "Niveau de politesse (존댓말/반말) cohérent selon le contexte.",
        "pt": "Tu/Você cohérent selon le contexte et le registre.",
    }
    politesse = lang_rules.get(tgt_lang, "")
    politesse_line = f"\n- {politesse}" if politesse else ""

    system = f"""Tu es un traducteur professionnel {src_n} → {tgt_n} spécialisé dans la
traduction orale de haute qualité. Le texte traduit sera PRONONCÉ À HAUTE VOIX
par un système de synthèse vocale.

CONTEXTE : {analysis.get('summary', '')}
Domaine : {analysis.get('domain', '')} | Ton : {analysis.get('tone', '')}
Locuteurs : {analysis.get('speakers_description', '')}
{user_ctx}
GLOSSAIRE :
{json.dumps(analysis.get('glossary', {}), ensure_ascii=False, indent=2)}

MISSION :
Tu reçois des SEGMENTS DE PAROLE numérotés [S<n>]. Chaque segment est une
unité de parole courte d'un locuteur — typiquement une phrase ou un fragment.

Pour chaque segment marqué "À TRADUIRE", produis une traduction en {tgt_n} qui :

1. SONNE NATURELLEMENT quand elle est prononcée à voix haute — c'est LA priorité
2. Utilise une PONCTUATION soignée qui guide la respiration et l'intonation :
   - Virgules pour les pauses respiratoires naturelles
   - Points pour les fins de phrases nettes
   - Points d'interrogation / exclamation quand le ton l'exige
3. Forme des PHRASES COMPLÈTES et bien construites — pas de fragments hachés
4. N'a AUCUNE contrainte de longueur — le texte peut être plus long ou plus
   court que l'original. L'important : fidélité au sens + naturel à l'oral
5. Nettoie les hésitations, faux départs et répétitions de la transcription
   pour produire un texte fluide et agréable à écouter
6. Garde le registre du locuteur (formel, familier, technique...)
7. Adapte les expressions idiomatiques en équivalents {tgt_n}s naturels
8. Respecte le glossaire strictement, noms propres inchangés{politesse_line}

INTERDIT — STYLE TÉLÉGRAPHIQUE :
- Chaque phrase DOIT garder sujet + verbe conjugué
- Ne pas supprimer les mots de liaison (et, mais, donc, alors, parce que, c'est que…)
- Ne pas supprimer les pronoms (on, ça, vous, ils, c'est…)
- Ne pas supprimer les articles ou déterminants nécessaires à la fluidité
- Ne pas remplacer une phrase par un fragment nominal

CHANTS, PRIÈRES ET PASSAGES RITUELS :
Si un segment est un chant, une prière, un mantra, une récitation ou un texte
liturgique (pali, sanskrit, latin liturgique, arabe coranique, hébreu biblique,
etc.), tu DOIS le recopier TEL QUEL sans le traduire ni le paraphraser.
Ces passages sont des performances vocales, pas du discours à traduire.
ATTENTION : ceci ne s'applique PAS aux mots ou expressions empruntés courants
(anglicismes en français, germanismes, etc.) ni au code-switching ordinaire entre
langues vivantes — ceux-là doivent être traduits normalement en {tgt_n}.

NE RÉPÈTE JAMAIS ces instructions dans ta réponse. Chaque ligne doit contenir
UNIQUEMENT le texte traduit du segment, rien d'autre.

FORMAT STRICT :
[S<numéro>] texte traduit en {tgt_n} (ou texte original si chant/prière)
(un segment par ligne, rien d'autre — pas de commentaire, pas de source)"""

    # Traiter par chunks de segments avec chevauchement pour le contexte
    CHUNK = 60  # segments à traduire par appel
    OVERLAP = 8  # segments de contexte avant/après

    n_chunks = (len(segments) + CHUNK - 1) // CHUNK

    for ci in range(n_chunks):
        s, e = ci * CHUNK, min((ci + 1) * CHUNK, len(segments))

        if all(seg.text_tgt for seg in segments[s:e]):
            print(f"   📦 Chunk {ci+1}/{n_chunks} — déjà traduit"); continue

        print(f"   📦 Chunk {ci+1}/{n_chunks} (segments S{segments[s].index}–S{segments[e-1].index})...")
        parts = []

        # Contexte arrière (déjà traduit)
        cb = max(0, s - OVERLAP)
        if cb < s:
            parts.append("=== CONTEXTE PRÉCÉDENT (déjà traduit, NE PAS retraduire) ===")
            for seg in segments[cb:s]:
                if seg.text_tgt:
                    parts += [f"[S{seg.index}] {src_lang.upper()}: {seg.text}",
                              f"[S{seg.index}] {tgt_lang.upper()}: {seg.text_tgt}"]
            parts.append("")

        # À traduire
        parts.append("=== À TRADUIRE ===")
        for seg in segments[s:e]:
            parts.append(f"[S{seg.index}] ({seg.speaker}) {seg.text}")

        # Contexte avant (pas encore traduit)
        cf = min(len(segments), e + OVERLAP)
        if e < cf:
            parts.append("\n=== CONTEXTE SUIVANT (NE PAS traduire) ===")
            for seg in segments[e:cf]:
                parts.append(f"[S{seg.index}] ({seg.speaker}) {seg.text}")

        resp = _claude_create(client, model=CLAUDE_MODEL, max_tokens=CLAUDE_MAX_TOKENS,
                              system=system,
                              messages=[{"role": "user", "content": "\n".join(parts)}])
        _parse_translations(resp.content[0].text, segments, s, e)

        done = sum(1 for seg in segments[s:e] if seg.text_tgt)
        if done < e - s:
            print(f"   ⚠️  {done}/{e-s} traduits — relance des manquants...")
            _retry_missing(segments, s, e, system, client, src_lang, tgt_lang)

        done = sum(1 for seg in segments[s:e] if seg.text_tgt)
        print(f"   ✅ {done}/{e-s}")

    total = sum(1 for seg in segments if seg.text_tgt)
    print(f"\n   🌍 Traduit : {total}/{len(segments)} segments")
    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 4c : RELECTURE (NATUREL, PONCTUATION, EXPRESSIVITÉ)
# ═══════════════════════════════════════════════════════════════════════════════

def review_segments(segments, analysis, client, src_lang, tgt_lang, context=""):
    """
    Relecture des segments traduits — critères hiérarchisés.
    Vérifie : fidélité, naturel oral, glossaire, politesse, fluidité, chants.
    """
    print(f"\n📖 Passe 4c — Relecture (naturel oral + ponctuation)...")

    src_n = lang_name(src_lang)
    tgt_n = lang_name(tgt_lang)

    # Règle de politesse spécifique à la langue cible
    if tgt_lang == "fr":
        lang_politeness = "5. POLITESSE : tutoiement/vouvoiement cohérent d'un segment à l'autre"
    elif tgt_lang == "de":
        lang_politeness = "5. POLITESSE : Du/Sie cohérent d'un segment à l'autre"
    elif tgt_lang == "es":
        lang_politeness = "5. POLITESSE : tú/usted cohérent d'un segment à l'autre"
    elif tgt_lang == "ja":
        lang_politeness = "5. POLITESSE : niveau de 敬語 cohérent d'un segment à l'autre"
    elif tgt_lang == "ko":
        lang_politeness = "5. POLITESSE : 존댓말/반말 cohérent d'un segment à l'autre"
    else:
        lang_politeness = "5. REGISTRE : niveau de formalité cohérent d'un segment à l'autre"

    WIN, OVL = 80, 15
    n_win = max(1, (len(segments) + WIN - OVL - 1) // (WIN - OVL))
    fixes = 0

    ctx_note = f"\nINSTRUCTIONS UTILISATEUR : {context}\n" if context else ""
    glossary = analysis.get("glossary", {})

    for wi in range(n_win):
        s = wi * (WIN - OVL)
        e = min(s + WIN, len(segments))
        print(f"   🔎 Fenêtre {wi+1}/{n_win} (S{segments[s].index}–S{segments[e-1].index})...")

        pairs = []
        for seg in segments[s:e]:
            pairs += [
                f"[S{seg.index}] ({seg.speaker}) {src_lang.upper()}: {seg.text}",
                f"[S{seg.index}] {tgt_lang.upper()}: {seg.text_tgt}",
                ""
            ]

        prompt = f"""Réviseur professionnel de traduction orale en {tgt_n}.

Le texte traduit sera PRONONCÉ à haute voix (synthèse vocale TTS), pas lu en sous-titres.
Il n'y a AUCUNE contrainte de longueur ou de durée.

GLOSSAIRE : {json.dumps(glossary, ensure_ascii=False, indent=2)}
{ctx_note}
Vérifie ces critères par ordre de priorité :
1. FIDÉLITÉ : pas de contresens, chiffres et noms propres corrects
2. NATUREL ORAL : prononciation fluide à voix haute, pas de tournures écrites (relatives longues, passifs, inversions)
3. GLOSSAIRE : termes du glossaire strictement respectés
{lang_politeness}
6. FLUIDITÉ DE PRONONCIATION : pas de mots difficiles enchaînés, pas de suites de consonnes imprononçables
7. CHANTS/PRIÈRES : si un segment est un chant, une prière, un mantra ou un
   texte liturgique (pali, sanskrit, latin liturgique, arabe coranique, etc.),
   il doit être CONSERVÉ TEL QUEL — ne jamais le modifier ni le traduire.
   (Ceci ne concerne PAS les emprunts courants entre langues vivantes.)

Si correction : [S<numéro>] texte corrigé
Si correct : ne pas inclure
Si tout OK : AUCUNE CORRECTION

{chr(10).join(pairs)}"""

        resp = _claude_create(client, model=CLAUDE_MODEL, max_tokens=CLAUDE_MAX_TOKENS,
                              messages=[{"role": "user", "content": prompt}])
        r = resp.content[0].text.strip()
        if "AUCUNE CORRECTION" in r.upper():
            continue

        for line in r.split("\n"):
            m = re.match(r'\[S(\d+)\]\s*(.*)', line.strip())
            if m:
                sid, new = int(m.group(1)), m.group(2).strip()
                new = _strip_claude_artifacts(new)
                if new:
                    for seg in segments:
                        if seg.index == sid and seg.text_tgt != new:
                            if _est_fuite_prompt(new, seg.text_tgt):
                                print(f"   ⚠️  Fuite de prompt détectée seg [S{sid}], ignoré : {new[:60]}…")
                                break
                            seg.text_tgt = new
                            fixes += 1
                            break

    print(f"   📖 {fixes} segments corrigés")
    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 4d : COHÉRENCE GLOBALE (CLAUDE)
# ═══════════════════════════════════════════════════════════════════════════════

def check_consistency(segments, analysis, client,
                      source_lang="en", target_lang="fr"):
    """Passe de cohérence globale : terminologie, registre de politesse, ton."""
    print("\n🔗 Passe 4d — Cohérence globale...")
    tgt_name = lang_name(target_lang)

    glossary = analysis.get("glossary", {})
    tone = analysis.get("tone", "")
    domain = analysis.get("domain", "")

    # Échantillon : max 500 segments pour fichiers longs
    sample = segments if len(segments) <= 500 else (
        segments[:170] + segments[len(segments)//2 - 80:len(segments)//2 + 80] + segments[-170:]
    )

    # Fenêtrage à 200 segments par appel
    WIN = 200
    fixes = 0
    for wi in range(0, len(sample), WIN):
        batch = sample[wi:wi + WIN]
        lines = [f"[S{seg.index}] {seg.text_tgt}" for seg in batch if seg.text_tgt]
        if not lines:
            continue

        prompt = f"""Vérificateur de cohérence pour traduction orale en {tgt_name}.

GLOSSAIRE : {json.dumps(glossary, ensure_ascii=False, indent=2)}
Ton attendu : {tone} | Domaine : {domain}

Vérifie UNIQUEMENT ces 3 points sur l'ensemble des segments ci-dessous :
1. TERMINOLOGIE : un même concept source est-il toujours traduit de la même façon ?
2. REGISTRE DE POLITESSE : le niveau de formalité est-il constant ?
3. TON : le ton ({tone}) est-il maintenu uniformément ?

Ne corrige PAS le style, la grammaire ou la concision — seulement les incohérences ci-dessus.

Si correction : [S<numéro>] texte corrigé
Si tout OK : AUCUNE CORRECTION

{chr(10).join(lines)}"""

        resp = _claude_create(client, model=CLAUDE_MODEL, max_tokens=CLAUDE_MAX_TOKENS,
                              messages=[{"role": "user", "content": prompt}])
        r = resp.content[0].text.strip()
        if "AUCUNE CORRECTION" in r.upper():
            continue

        for line in r.split("\n"):
            m = re.match(r'\[S(\d+)\]\s*(.*)', line.strip())
            if m:
                idx, new = int(m.group(1)), m.group(2).strip()
                new = _strip_claude_artifacts(new)
                if new and not _est_fuite_prompt(new):
                    for seg in segments:
                        if seg.index == idx and seg.text_tgt != new:
                            seg.text_tgt = new; fixes += 1; break

    print(f"   🔗 {fixes} corrections de cohérence")
    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 4e : VÉRIFICATION GLOSSAIRE (CLAUDE)
# ═══════════════════════════════════════════════════════════════════════════════

def verify_glossary(segments, analysis, client,
                    source_lang="en", target_lang="fr"):
    """Vérifie que les termes du glossaire sont appliqués et corrige les violations."""
    glossary = analysis.get("glossary", {})
    if not glossary:
        print("\n📖 Vérification glossaire — glossaire vide, passage ignoré")
        return segments

    print("\n📖 Passe 4e — Vérification glossaire...")
    tgt_name = lang_name(target_lang)

    # Scan case-insensitive des violations
    violations = []
    for seg in segments:
        if not seg.text_tgt:
            continue
        tgt_lower = seg.text_tgt.lower()
        src_lower = seg.text.lower()
        for src_term, tgt_term in glossary.items():
            if src_term.lower() in src_lower:
                if tgt_term.lower() not in tgt_lower:
                    violations.append((seg, src_term, tgt_term))
                    break  # une violation suffit par segment

    if not violations:
        print("   ✅ Glossaire respecté partout")
        return segments

    print(f"   ⚠️  {len(violations)} violations détectées")

    # Envoi par lots de 30
    BATCH = 30
    fixes = 0
    for bi in range(0, len(violations), BATCH):
        batch = violations[bi:bi + BATCH]
        lines = []
        for seg, src_term, tgt_term in batch:
            lines.append(f"[S{seg.index}] {source_lang.upper()}: {seg.text}")
            lines.append(f"[S{seg.index}] {target_lang.upper()}: {seg.text_tgt}")
            lines.append(f"  → Le terme « {src_term} » devrait être traduit « {tgt_term} »")
            lines.append("")

        prompt = f"""Correcteur de glossaire pour traduction orale en {tgt_name}.

Pour chaque segment ci-dessous, le glossaire n'a pas été respecté.
Corrige NATURELLEMENT la traduction pour intégrer le terme correct du glossaire,
sans rendre la phrase artificielle. Le texte sera prononcé à haute voix.

GLOSSAIRE COMPLET : {json.dumps(glossary, ensure_ascii=False, indent=2)}

Format : [S<numéro>] texte corrigé (un par ligne)

{chr(10).join(lines)}"""

        resp = _claude_create(client, model=CLAUDE_MODEL, max_tokens=CLAUDE_MAX_TOKENS,
                              messages=[{"role": "user", "content": prompt}])
        for line in resp.content[0].text.strip().split("\n"):
            m = re.match(r'\[S(\d+)\]\s*(.*)', line.strip())
            if m:
                idx, new = int(m.group(1)), m.group(2).strip()
                new = _strip_claude_artifacts(new)
                if new and not _est_fuite_prompt(new):
                    for seg in segments:
                        if seg.index == idx and seg.text_tgt != new:
                            seg.text_tgt = new; fixes += 1; break

    print(f"   📖 {fixes} corrections glossaire")
    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 5 : SYNTHÈSE VOCALE PAR SEGMENTS (XTTS v2)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Nettoyage du texte pour éviter les artefacts TTS ────────────────────────

def sanitize_for_tts(text: str, lang: str = "fr", backend: str = "xtts") -> str:
    """
    Nettoyage phonétique pré-TTS : supprime les caractères que les moteurs TTS
    prononcent littéralement ("point", "guillemet", etc.).
    Principe : le texte doit contenir UNIQUEMENT ce qu'on veut entendre.

    Tous les backends : les points "." sont remplacés par ";" (pause naturelle
    sans vocalisation). Aucun moteur TTS ne prononce le point-virgule.
    """
    if not text:
        return text

    # 1. Ellipses → virgule (pause naturelle)
    text = re.sub(r'\.{3,}', ',', text)
    text = text.replace("…", ",")

    # 2. Points multiples résiduels
    text = re.sub(r'\.{2}', '.', text)

    # 3. Abréviations courantes → forme longue (évite "point" prononcé)
    abbrevs_fr = {
        r'\bM\.\s': 'Monsieur ',
        r'\bMme\.\s': 'Madame ',
        r'\bMlle\.\s': 'Mademoiselle ',
        r'\bDr\.\s': 'Docteur ',
        r'\bPr\.\s': 'Professeur ',
        r'\bSt\.\s': 'Saint ',
        r'\bvs\.\s': 'versus ',
        r'\betc\.\s': 'et cetera, ',
        r'\betc\.(\s|$)': r'et cetera\1',
    }
    abbrevs_en = {
        r'\bMr\.\s': 'Mister ',
        r'\bMrs\.\s': 'Misses ',
        r'\bMs\.\s': 'Miss ',
        r'\bDr\.\s': 'Doctor ',
        r'\bvs\.\s': 'versus ',
        r'\betc\.\s': 'et cetera, ',
        r'\betc\.(\s|$)': r'et cetera\1',
    }
    abbrevs = abbrevs_fr if lang == "fr" else abbrevs_en
    for pat, repl in abbrevs.items():
        text = re.sub(pat, repl, text)

    # 3b. Acronymes / initiales avec points (P.D.G., J.F.K., U.S.A.)
    # → supprimer les points internes pour éviter "point" vocalisé
    text = re.sub(r'\b([A-ZÀ-Ü])\.\s?([A-ZÀ-Ü])\.\s?([A-ZÀ-Ü])\.?', r'\1\2\3', text)
    text = re.sub(r'\b([A-ZÀ-Ü])\.\s?([A-ZÀ-Ü])\.?', r'\1\2', text)
    # Point après un seul chiffre en début de phrase (numérotation "1. Bonjour")
    text = re.sub(r'^(\d+)\.\s', r'\1, ', text)

    # 4. Nombres avec point décimal → "virgule" (français) ou laisser tel quel
    if lang == "fr":
        text = re.sub(r'(\d)\.(\d)', r'\1 virgule \2', text)

    # 5. Guillemets (XTTS les lit parfois comme "guillemet")
    text = re.sub(r'[«»""„]', '', text)
    text = text.replace('"', '')

    # 6. Tirets cadratins/demi-cadratins → virgule (pause)
    text = re.sub(r'[—–]', ',', text)
    # Tirets entourés d'espaces → espace simple (préserve mots composés)
    text = re.sub(r'\s+-\s+', ' ', text)

    # 7. Caractères spéciaux que XTTS pourrait vocaliser
    text = re.sub(r'[#*_~`|\\]', '', text)
    # Contenu entre parenthèses/crochets → supprimé (souvent des didascalies)
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[[^\]]*\]', '', text)

    # 8. Esperluette → mot selon la langue
    ampersand_map = {"fr": "et", "en": "and", "es": "y", "de": "und",
                     "it": "e", "pt": "e", "nl": "en", "ru": "и",
                     "ja": "と", "zh": "和", "ko": "와"}
    text = text.replace("&", f" {ampersand_map.get(lang, 'et')} ")

    # 9. NUCLEAR FIX : remplacer TOUS les points restants par des points-virgules.
    # Tous les backends TTS (XTTS, Qwen3-TTS) vocalisent "."
    # comme "point" en français. Le point-virgule produit une pause naturelle de
    # frontière de phrase sans jamais être prononcé par aucun moteur TTS.
    # Les abréviations (M., etc.), acronymes (P.D.G.) et décimales (3.14) ont déjà
    # été traités plus haut — il ne reste que des points de fin de phrase.
    text = text.replace('.', ';')

    # 10. Ponctuation doublée résiduelle
    text = re.sub(r'([,;:!?])\s*\1+', r'\1', text)

    # 11. Espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()

    # 12. S'assurer que le texte finit par une ponctuation (évite coupure abrupte)
    if text and text[-1] not in '!?,;:':
        text += ';'

    return text


def _trim_tts_silence(audio_path, threshold_db=-40.0,
                      min_silence_ms=100, keep_ms=50):
    """
    Supprime le silence parasite en début et fin d'un clip TTS (in-place).
    XTTS ajoute souvent 200-500ms de silence avant/après la voix, qui peut
    contenir des clics ou pops du décodeur.

    - threshold_db : seuil en dB sous lequel on considère comme silence
    - min_silence_ms : durée min de silence pour être considéré comme "trimmable"
    - keep_ms : marge conservée pour protéger les consonnes douces (fricatives,
      plosives) qui peuvent être en dessous du seuil dB

    Retourne la nouvelle durée en secondes.
    """
    import soundfile as sf
    import numpy as np

    data, sr = sf.read(audio_path, dtype="float32")
    if len(data) == 0:
        return 0.0

    mono = data if data.ndim == 1 else data.mean(axis=1)
    window = int(sr * 0.01)  # 10ms
    if len(mono) < window:
        return len(mono) / sr

    # RMS par fenêtre
    n_frames = len(mono) // window
    frames = mono[:n_frames * window].reshape(n_frames, window)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    rms_db = 20 * np.log10(rms + 1e-10)

    keep_frames = max(0, keep_ms // 10)

    above = np.where(rms_db > threshold_db)[0]
    if len(above) == 0:
        return len(data) / sr  # tout est silence → ne pas toucher

    first_voice = above[0]
    last_voice = above[-1]

    trim_start = max(0, first_voice - keep_frames) * window
    trim_end = min(len(data), (last_voice + 1 + keep_frames) * window)

    start_silence = first_voice * window / sr * 1000
    end_silence = (n_frames - last_voice - 1) * window / sr * 1000

    trimmed = False
    if start_silence > min_silence_ms:
        trimmed = True
    else:
        trim_start = 0

    if end_silence > min_silence_ms:
        trimmed = True
    else:
        trim_end = len(data)

    if trimmed and trim_start < trim_end:
        result = data[trim_start:trim_end]
        # Micro-fondus aux frontières pour éviter les clics
        fade_samples = min(int(sr * 0.010), len(result) // 4)  # 10ms
        if fade_samples > 0:
            if trim_start > 0:
                result[:fade_samples] *= np.linspace(0, 1, fade_samples)
            if trim_end < len(data):
                result[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        sf.write(audio_path, result, sr, subtype="PCM_16")
        return len(result) / sr

    return len(data) / sr


def pad_tts_audio(audio_path: str, tail_pad_ms: int = 200) -> str:
    """
    Ajoute du silence en fin de clip TTS pour éviter les mots coupés.

    XTTS génère parfois un audio qui se termine pile sur le dernier
    phonème sans silence résiduel. Ce padding garantit une marge.

    Détecte aussi les fins abruptes (énergie encore haute en fin de clip)
    et ajoute un padding supplémentaire + fade-out correctif.
    """
    import soundfile as sf
    import numpy as np

    audio, sr = sf.read(audio_path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if len(audio) < sr * 0.1:  # < 100ms → trop court
        return audio_path

    # Vérifier si la fin est abrupte (énergie élevée dans les dernières 50ms)
    tail_samples = int(sr * 0.050)
    if tail_samples > 0 and len(audio) > tail_samples * 2:
        tail_rms = np.sqrt(np.mean(audio[-tail_samples:] ** 2))
        body_rms = np.sqrt(np.mean(audio[:-tail_samples] ** 2))

        if body_rms > 0 and tail_rms / body_rms > 0.25:
            # Fin abrupte détectée → fade-out correctif de 80ms
            fade_samples = min(int(sr * 0.080), len(audio) // 4)
            fade = np.linspace(1.0, 0.0, fade_samples)
            audio[-fade_samples:] *= fade
            tail_pad_ms = max(tail_pad_ms, 250)  # padding plus généreux

    # Ajouter le silence de fin
    pad_samples = int(sr * tail_pad_ms / 1000)
    padded = np.concatenate([audio, np.zeros(pad_samples)])

    sf.write(audio_path, padded, sr)
    return audio_path


# ── Découpage texte pour TTS ─────────────────────────────────────────────────

def split_text_for_tts(text, max_chars=XTTS_CHAR_LIMIT_DEFAULT):
    """
    Découpe un texte en morceaux de max_chars aux frontières naturelles.
    Priorité : phrase (. ! ?) > clause (, ; :) > espace > coupure brute.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    remaining = text.strip()

    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        cut = -1

        # Priorité 1 : fin de phrase
        for sep in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
            idx = window.rfind(sep)
            if idx > max_chars // 3:
                cut = idx + 1  # inclure le séparateur
                break

        # Priorité 2 : clause (virgule, point-virgule, deux-points)
        if cut < 0:
            for sep in [", ", "; ", ": ", " – ", " — "]:
                idx = window.rfind(sep)
                if idx > max_chars // 3:
                    cut = idx + len(sep)
                    break

        # Priorité 3 : n'importe quel espace
        if cut < 0:
            idx = window.rfind(" ")
            if idx > max_chars // 4:
                cut = idx + 1

        # Fallback : couper brutalement
        if cut < 0:
            cut = max_chars

        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()

    if remaining:
        chunks.append(remaining)

    return chunks


# ── Voix preset XTTS v2 ──────────────────────────────────────────────────────

XTTS_VOICES_FEMALE = [
    "Ana Florence", "Brenda Stern", "Claribel Dervla", "Gracie Wise",
    "Henriette Usha", "Sofia Hellen", "Tanja Adelina", "Alma María",
    "Daisy Studious", "Gitta Nikolina", "Tamaru Naoko", "Lidiya Szekeres",
]
XTTS_VOICES_MALE = [
    "Craig Gutsy", "Damien Black", "Viktor Menelaos", "Baldur Sanjin",
    "Dionisio Schuyler", "Royston Min", "Abrahan Mack", "Gilberto Mathias",
    "Kazuhiko Atallah", "Torcull Diarmuid", "Zacharie Aimilios", "Viktor Eka",
]
XTTS_DEFAULT_VOICES = [
    "Craig Gutsy", "Ana Florence", "Damien Black", "Brenda Stern",
    "Viktor Menelaos", "Claribel Dervla", "Baldur Sanjin", "Gracie Wise",
]

# Mapping langue → code XTTS
XTTS_LANG_MAP = {
    "fr": "fr", "en": "en", "es": "es", "de": "de", "it": "it",
    "pt": "pt", "pl": "pl", "tr": "tr", "ru": "ru", "nl": "nl",
    "cs": "cs", "ar": "ar", "zh": "zh", "ja": "ja", "ko": "ko",
    "hu": "hu", "hi": "hi",
}


class XTTSBackend:
    """Backend XTTS v2 pour synthèse vocale."""

    def __init__(self, target_lang="fr", ref_voice=None, xtts_speaker=None):
        from TTS.api import TTS as CoquiTTS
        print("   🔊 Chargement du modèle XTTS v2...")
        self.model = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
        self.target_lang = XTTS_LANG_MAP.get(target_lang, target_lang)
        self.profiles = {}
        self.ref_voice = ref_voice
        self.xtts_speaker = xtts_speaker
        self.voice_map = {}
        # Warm-up : les 1-2 premiers appels XTTS produisent souvent un audio
        # dégradé (décodeur autorégressif pas encore stabilisé). On force
        # 2 appels avec les mêmes paramètres que la synthèse réelle.
        try:
            import tempfile
            warmup_path = os.path.join(tempfile.gettempdir(), "_xtts_warmup.wav")
            warmup_kwargs = dict(
                repetition_penalty=5.0, temperature=0.65,
                length_penalty=1.0, top_k=50, top_p=0.85,
                enable_text_splitting=False,
            )
            for warmup_text in [
                "Bonjour, bienvenue dans cette présentation,",
                "Nous allons maintenant commencer notre discussion,",
            ]:
                self.model.tts_to_file(
                    text=warmup_text,
                    speaker="Craig Gutsy",
                    language=self.target_lang,
                    file_path=warmup_path,
                    **warmup_kwargs,
                )
            if os.path.exists(warmup_path):
                os.remove(warmup_path)
        except Exception:
            pass  # ne pas bloquer si le warm-up échoue
        print(f"   ✅ XTTS v2 prêt (langue cible: {self.target_lang})")

    def setup_voices(self, profiles):
        self.profiles = profiles
        speakers = sorted(profiles.keys())

        if self.ref_voice:
            print(f"      🎯 Voix de référence externe : {self.ref_voice}")

        if self.xtts_speaker:
            for spk in speakers:
                self.voice_map[spk] = self.xtts_speaker
            print(f"      🎯 Voix preset '{self.xtts_speaker}' pour {len(speakers)} locuteur(s)")
        else:
            female_idx = male_idx = unknown_idx = 0
            for spk in speakers:
                p = profiles[spk]
                has_ref = bool(p.ref_clips) or (p.sample_path and os.path.exists(p.sample_path))

                if has_ref:
                    ref_count = len(p.ref_clips) if p.ref_clips else 1
                    print(f"      🎙️ {spk} : clonage vocal ({ref_count} refs, "
                          f"{p.total_duration:.0f}s)")
                else:
                    gender = p.gender
                    if gender == "female":
                        voice = XTTS_VOICES_FEMALE[female_idx % len(XTTS_VOICES_FEMALE)]
                        female_idx += 1
                    elif gender == "male":
                        voice = XTTS_VOICES_MALE[male_idx % len(XTTS_VOICES_MALE)]
                        male_idx += 1
                    else:
                        voice = XTTS_DEFAULT_VOICES[unknown_idx % len(XTTS_DEFAULT_VOICES)]
                        unknown_idx += 1
                    self.voice_map[spk] = voice
                    icon = "♀️" if gender == "female" else "♂️" if gender == "male" else "❓"
                    print(f"      🎤 {spk} {icon} → preset \"{voice}\"")

    def _get_best_ref(self, speaker_id):
        if self.ref_voice:
            return self.ref_voice
        profile = self.profiles.get(speaker_id)
        if not profile:
            profile = next((p for p in self.profiles.values()
                            if p.ref_clips or p.sample_path), None)
        if not profile:
            return ""
        if profile.ref_clips:
            return profile.ref_clips[0][0]
        if profile.sample_path and os.path.exists(profile.sample_path):
            return profile.sample_path
        return ""

    def synthesize(self, text, speaker_id, output_path):
        """
        Synthèse d'un segment avec découpage intelligent des textes longs.

        Flux :
        1. sanitize_for_tts() → nettoyage ponctuation
        2. Si texte > char_limit → split_text_for_tts() + _synthesize_and_concat()
           Sinon → _tts_one() directement
        3. Anti-bégaiement :
           - seuil_1 → retry agressif (_tts_one ou _synthesize_and_concat)
           - seuil_2 → re-split en morceaux courts + _synthesize_and_concat
           - dernier recours → troncature audio
        4. --verify-tts → vérification ASR, re-split si échec
        5. pad_tts_audio() → padding de fin
        """
        import soundfile as sf
        import numpy as np

        clean = sanitize_for_tts(text, self.target_lang)
        if not clean:
            return ""

        char_limit = XTTS_CHAR_LIMITS.get(self.target_lang, XTTS_CHAR_LIMIT_DEFAULT)
        preset_voice = self.voice_map.get(speaker_id)

        ref_path = None
        if not preset_voice:
            ref_path = self._get_best_ref(speaker_id)
            if not ref_path:
                print(f"      ❌ Aucun échantillon pour {speaker_id}")
                return ""

        try:
            # ── Étape 1 : synthèse initiale (avec split si nécessaire) ───
            chunks = split_text_for_tts(clean, char_limit)
            if len(chunks) > 1:
                print(f"      📝 Texte long ({len(clean)} car.) découpé en {len(chunks)} morceaux")
                ok = self._synthesize_and_concat(
                    chunks, preset_voice, ref_path, output_path)
                if not ok:
                    print(f"      ❌ Échec synthèse multi-morceaux [{speaker_id}]")
                    return ""
            else:
                self._tts_one(clean, preset_voice, ref_path, output_path)

            # ── Étape 2 : anti-bégaiement ────────────────────────────────
            audio, sr = sf.read(output_path)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            duration = len(audio) / sr
            chars = len(clean)

            SLOW_LANGS = {"fr", "es", "it", "pt", "de", "pl", "nl", "cs", "ro", "hu"}
            if self.target_lang in SLOW_LANGS:
                stutter_threshold_1 = 0.090   # 90ms/char → retry (relevé : 60 trop agressif
                                              # pour texte mixte pali/sanskrit + langue cible)
                stutter_threshold_2 = 0.130   # 130ms/char → re-split (vrai bégaiement seulement)
            else:
                stutter_threshold_1 = 0.075   # 75ms/char → retry
                stutter_threshold_2 = 0.110   # 110ms/char → re-split

            if chars > 10 and duration / chars > stutter_threshold_1:
                ms_per_char = 1000 * duration / chars
                print(f"      ⚠️  Bégaiement probable ({duration:.1f}s pour {chars} car., "
                      f"{ms_per_char:.0f}ms/car) → retry agressif")

                # Retry agressif (même découpage)
                if len(chunks) > 1:
                    self._synthesize_and_concat(
                        chunks, preset_voice, ref_path, output_path,
                        aggressive=True)
                else:
                    self._tts_one(clean, preset_voice, ref_path, output_path,
                                  aggressive=True)

                audio, sr = sf.read(output_path)
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                duration = len(audio) / sr

                # Si toujours aberrant → re-split en morceaux courts
                if chars > 10 and duration / chars > stutter_threshold_2:
                    short_limit = max(char_limit // 2, 40)
                    short_chunks = split_text_for_tts(clean, short_limit)

                    if len(short_chunks) > 1:
                        print(f"      🔄 Re-synthèse: {len(short_chunks)} morceaux "
                              f"(limite {short_limit} car.)")
                        ok = self._synthesize_and_concat(
                            short_chunks, preset_voice, ref_path, output_path,
                            aggressive=True)
                        if ok:
                            audio, sr = sf.read(output_path)
                            if audio.ndim == 2:
                                audio = audio.mean(axis=1)
                            duration = len(audio) / sr

                    # Info si toujours au-dessus du seuil (mais PAS de troncature —
                    # la troncature détruisait le contenu, surtout sur texte mixte
                    # pali/sanskrit + langue cible où le débit est naturellement lent)
                    if duration / chars > stutter_threshold_2:
                        ms_retry = 1000 * duration / chars
                        print(f"      ℹ️  Débit toujours lent ({ms_retry:.0f}ms/car) — "
                              f"accepté tel quel (pas de troncature)")

            # ── Étape 3 : vérification ASR optionnelle ───────────────────
            if not self._verify_tts_output(clean, output_path):
                short_limit = max(char_limit // 2, 40)
                short_chunks = split_text_for_tts(clean, short_limit)
                if len(short_chunks) > 1:
                    print(f"      🔄 Vérification ASR échouée → re-synthèse en "
                          f"{len(short_chunks)} morceaux")
                    self._synthesize_and_concat(
                        short_chunks, preset_voice, ref_path, output_path,
                        aggressive=True)

            # Padding de fin (protège les derniers mots)
            pad_tts_audio(output_path)
            return output_path

        except Exception as e:
            print(f"      ❌ XTTS échoué [{speaker_id}] : {e}")
            return ""

    def _tts_one(self, text, preset_voice, ref_path, output_path,
                 aggressive=False):
        """Synthèse d'un seul segment de texte avec paramètres anti-bégaiement.

        Paramètres XTTS passés via kwargs :
        - repetition_penalty : pénalise le décodeur autorégressif quand il boucle
          (valeur élevée = moins de répétitions/bégaiements)
        - temperature : contrôle la variabilité de la génération
          (plus bas = plus conservateur, moins de risque de boucle)
        - length_penalty : contrôle la longueur de sortie
        - top_k / top_p : paramètres d'échantillonnage

        En mode aggressive (retry ou segment court) : repetition_penalty
        et temperature plus stricts pour forcer le décodeur hors des boucles.
        """
        is_short = len(text) < 30
        if aggressive or is_short:
            xtts_kwargs = dict(
                repetition_penalty=5.5,   # Pénalité forte mais pas extrême (9.0 causait
                                          # des arrêts prématurés sur texte mixte pali/fr)
                temperature=0.55,         # Conservateur mais laisse assez de marge pour
                                          # que le décodeur ne choisisse pas "stop" trop tôt
                length_penalty=1.0,
                top_k=40,                 # Vocabulaire un peu plus ouvert
                top_p=0.80,               # Noyau moins étroit
                enable_text_splitting=False,  # Ne pas re-découper un texte déjà court
            )
        else:
            xtts_kwargs = dict(
                repetition_penalty=5.0,   # Fortement pénaliser les répétitions
                temperature=0.65,         # Conservateur pour éviter les boucles
                length_penalty=1.0,
                top_k=50,
                top_p=0.85,
                enable_text_splitting=False,
            )
        if preset_voice:
            self.model.tts_to_file(
                text=text, speaker=preset_voice,
                language=self.target_lang, file_path=output_path,
                **xtts_kwargs,
            )
        else:
            self.model.tts_to_file(
                text=text, speaker_wav=ref_path,
                language=self.target_lang, file_path=output_path,
                **xtts_kwargs,
            )

    def _synthesize_and_concat(self, chunks, preset_voice, ref_path,
                               output_path, aggressive=False, gap_ms=60):
        """Synthétise chaque morceau dans un WAV temporaire puis concatène.

        Les morceaux sont séparés par gap_ms millisecondes de silence pour
        maintenir un rythme naturel entre les fragments.
        """
        import soundfile as sf
        import numpy as np

        tmp_dir = os.path.dirname(output_path) or "."
        wav_parts = []
        sr = None

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            tmp_path = os.path.join(tmp_dir, f"_chunk_{os.getpid()}_{i}.wav")
            try:
                self._tts_one(chunk, preset_voice, ref_path, tmp_path,
                              aggressive=aggressive)
                audio, file_sr = sf.read(tmp_path)
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                if sr is None:
                    sr = file_sr
                wav_parts.append(audio)
            except Exception as e:
                print(f"      ⚠️  Échec morceau {i+1}/{len(chunks)} : {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        if not wav_parts or sr is None:
            return False

        # Concaténer avec micro-crossfade de 128 samples aux jonctions
        crossfade_samples = min(128, min(len(p) for p in wav_parts) // 2) if wav_parts else 0
        gap_samples = int(sr * gap_ms / 1000)
        combined = []
        for i, part in enumerate(wav_parts):
            if i > 0:
                # Crossfade : fade-out les derniers samples du chunk précédent,
                # gap de silence, puis fade-in les premiers samples du chunk suivant
                if crossfade_samples > 0 and len(combined) > 0:
                    # Fade-out fin du chunk précédent
                    prev = combined[-1]
                    fade_out = np.linspace(1.0, 0.0, crossfade_samples).astype(prev.dtype)
                    prev[-crossfade_samples:] *= fade_out
                    combined[-1] = prev
                    # Gap de silence
                    combined.append(np.zeros(gap_samples, dtype=part.dtype))
                    # Fade-in début du chunk suivant
                    fade_in = np.linspace(0.0, 1.0, crossfade_samples).astype(part.dtype)
                    part = part.copy()
                    part[:crossfade_samples] *= fade_in
                else:
                    combined.append(np.zeros(gap_samples, dtype=part.dtype))
            combined.append(part)

        sf.write(output_path, np.concatenate(combined), sr)
        return True

    def _verify_tts_output(self, text, audio_path):
        """Vérifie via ASR que le TTS a bien vocalisé tout le texte.

        Charge WhisperX 'base' de manière paresseuse (1er appel uniquement).
        Compare le nombre de mots ASR vs texte original.
        Retourne True si >= 70% des mots sont détectés.
        """
        if not hasattr(self, 'verify_tts') or not self.verify_tts:
            return True

        try:
            import whisperx
            import torch

            # Chargement paresseux du modèle ASR léger
            if not hasattr(self, '_whisperx_model'):
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute = "float16" if device == "cuda" else "int8"
                print("      🔍 Chargement WhisperX base pour vérification TTS...")
                self._whisperx_model = whisperx.load_model(
                    "base", device, compute_type=compute)
                self._whisperx_device = device

            audio = whisperx.load_audio(audio_path)
            result = self._whisperx_model.transcribe(
                audio, batch_size=4, language=self.target_lang)

            asr_text = " ".join(seg["text"] for seg in result.get("segments", []))
            asr_words = len(asr_text.split())
            expected_words = len(text.split())

            if expected_words == 0:
                return True

            ratio = asr_words / expected_words
            print(f"      🔍 ASR: {asr_words}/{expected_words} mots détectés ({ratio:.0%})")

            return ratio >= 0.70

        except Exception as e:
            print(f"      ⚠️  Vérification ASR échouée : {e}")
            return True  # ne pas bloquer en cas d'erreur

    def cleanup(self):
        if hasattr(self, '_whisperx_model'):
            del self._whisperx_model
        del self.model; gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class Qwen3TTSBackend:
    """Backend Qwen3-TTS via subprocess bridge (env conda isolé).

    Modèle Base 1.7B pour le clonage vocal zero-shot. 10 langues dont FR.
    Qwen3-TTS exige le transcript de l'audio de référence (ref_text) pour
    un clonage optimal ; sinon bascule en mode x_vector_only (timbre seul).
    Adapté au contexte batch : pas de two-pass, pas de speed adjustment.
    """

    def __init__(self, target_lang="fr", source_lang="en"):
        qwen3_py = _get_qwen3tts_python()
        if not qwen3_py:
            raise RuntimeError(
                "Env conda 'qwen3tts' introuvable. Créer avec :\n"
                "  conda create -n qwen3tts python=3.12 && "
                "conda run -n qwen3tts pip install -U qwen-tts soundfile")

        print("   🔊 Chargement du modèle Qwen3-TTS 1.7B Base (env isolé)...")
        self._proc = subprocess.Popen(
            [qwen3_py, QWEN3TTS_BRIDGE_SCRIPT],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )

        # Drainer stderr en continu pour éviter un deadlock si le buffer pipe se remplit
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

        resp = self._bridge_call({"cmd": "init"})
        if not resp.get("ok"):
            raise RuntimeError(f"Qwen3-TTS init échoué : {resp.get('error', '?')}")

        self.sample_rate = resp.get("sample_rate", 24000)
        self.target_lang = target_lang
        self.source_lang = source_lang
        # En doublage inter-langues, le mode ICL transfère l'accent source
        # On force x_vector_only pour ne garder que le timbre sans l'accent étranger.
        self.cross_lang = (source_lang != target_lang)
        self.qwen_lang = QWEN3TTS_LANG_MAP.get(target_lang, "French")
        self.profiles = {}
        self.voice_map = {}
        self.verify_tts = False

        device = resp.get("device", "?")
        mode_info = "x_vector_only (timbre seul, accent natif)" if self.cross_lang else "ICL (timbre + prosodie)"
        print(f"   ✅ Qwen3-TTS prêt (langue: {self.qwen_lang}, mode: {mode_info}, device={device})")

    def _drain_stderr(self):
        """Lit stderr en continu pour éviter le deadlock par buffer plein."""
        try:
            for line in self._proc.stderr:
                pass
        except (ValueError, OSError):
            pass

    def _bridge_call(self, cmd):
        """Envoie une commande JSON au bridge et lit la réponse."""
        try:
            self._proc.stdin.write(json.dumps(cmd) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
            if not line:
                return {"ok": False, "error": "Bridge subprocess fermé prématurément"}
            return json.loads(line)
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def setup_voices(self, profiles):
        self.profiles = profiles
        speakers = sorted(profiles.keys())

        for spk in speakers:
            p = profiles[spk]
            has_ref = bool(p.ref_clips) or bool(p.sample_path and os.path.exists(p.sample_path))

            if has_ref:
                self.voice_map[spk] = f"clone:{spk}"
                ref_count = len(p.ref_clips) if p.ref_clips else 1
                print(f"      🎙️  {spk} : clonage vocal ({ref_count} refs, "
                      f"{p.total_duration:.0f}s)")
            else:
                self.voice_map[spk] = "default"
                print(f"      ⚠️  {spk} : pas de référence audio, voix Qwen3-TTS par défaut")

    def _get_best_ref(self, speaker_id):
        """Retourne (chemin_audio, transcript) du meilleur clip de référence."""
        profile = self.profiles.get(speaker_id)
        if not profile:
            profile = next((p for p in self.profiles.values()
                            if p.ref_clips or p.sample_path), None)
        if not profile:
            return ("", "")
        if profile.ref_clips:
            path, ref_text = profile.ref_clips[0]
            return (path, ref_text)
        if profile.sample_path and os.path.exists(profile.sample_path):
            return (profile.sample_path, getattr(profile, 'sample_text', ''))
        return ("", "")

    def _synthesize_single(self, text, speaker_id, output_path):
        """Synthèse d'un seul morceau via le bridge Qwen3-TTS."""
        ref_path, ref_text = self._get_best_ref(speaker_id)

        try:
            cmd = {
                "cmd": "generate",
                "text": text,
                "language": self.qwen_lang,
                "output_path": output_path,
            }
            if ref_path:
                cmd["ref_audio_path"] = ref_path
                # En doublage inter-langues, ne pas passer ref_text pour
                # forcer x_vector_only : timbre préservé, accent source éliminé.
                if ref_text and not self.cross_lang:
                    cmd["ref_text"] = ref_text

            resp = self._bridge_call(cmd)
            if not resp.get("ok"):
                print(f"      ❌ Qwen3-TTS échoué [{speaker_id}] : {resp.get('error', '?')}")
                return ""

            # Resample 24000 → 44100 pour cohérence pipeline
            tmp_path = output_path + ".resample.wav"
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", output_path,
                 "-ar", str(SAMPLE_RATE), "-ac", "1", "-acodec", "pcm_s16le",
                 tmp_path],
                capture_output=True, text=True
            )
            if r.returncode == 0 and os.path.exists(tmp_path):
                shutil.move(tmp_path, output_path)
            else:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

            return output_path

        except Exception as e:
            print(f"      ❌ Qwen3-TTS échoué [{speaker_id}] : {e}")
            return ""

    def synthesize(self, text, speaker_id, output_path):
        """Synthèse vocale Qwen3-TTS (batch — pas de speed adjustment)."""
        clean = sanitize_for_tts(text, self.target_lang, backend="qwen3tts")
        if not clean:
            return ""

        try:
            chunks = split_text_for_tts(clean, QWEN3TTS_MAX_CHARS)
            if len(chunks) > 1:
                print(f"      📝 Texte long ({len(clean)} car.) découpé en "
                      f"{len(chunks)} morceaux (Qwen3-TTS)")
                ok = self._synthesize_and_concat(chunks, speaker_id, output_path)
                if not ok:
                    print(f"      ❌ Échec synthèse multi-morceaux [{speaker_id}]")
                    return ""
            else:
                result = self._synthesize_single(clean, speaker_id, output_path)
                if not result:
                    return ""

            pad_tts_audio(output_path)
            return output_path

        except Exception as e:
            print(f"      ❌ Qwen3-TTS échoué [{speaker_id}] : {e}")
            return ""

    def _synthesize_and_concat(self, chunks, speaker_id, output_path, gap_ms=60):
        """Synthétise chaque morceau via le bridge puis concatène."""
        import soundfile as sf
        import numpy as np

        chunk_paths = []
        out_dir = os.path.dirname(output_path) or "."
        base = os.path.splitext(os.path.basename(output_path))[0]

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            chunk_path = os.path.join(out_dir, f"{base}_part{i:02d}.wav")
            result = self._synthesize_single(chunk, speaker_id, chunk_path)
            if result:
                chunk_paths.append(chunk_path)
            else:
                print(f"      ⚠️  Échec morceau {i+1}/{len(chunks)}")

        if not chunk_paths:
            return False

        if len(chunk_paths) == 1:
            shutil.move(chunk_paths[0], output_path)
            return True

        arrays = []
        sr = None
        for cp in chunk_paths:
            data, file_sr = sf.read(cp, dtype="float32")
            if sr is None:
                sr = file_sr
            arrays.append(data)

        crossfade_samples = min(128, min(len(a) for a in arrays) // 2) if arrays else 0
        gap_samples = int(sr * gap_ms / 1000)
        combined = []
        for i, arr in enumerate(arrays):
            if i > 0:
                if crossfade_samples > 0 and len(combined) > 0:
                    prev = combined[-1]
                    fade_out = np.linspace(1.0, 0.0, crossfade_samples).astype(prev.dtype)
                    prev[-crossfade_samples:] *= fade_out
                    combined[-1] = prev
                    combined.append(np.zeros(gap_samples, dtype=arr.dtype))
                    fade_in = np.linspace(0.0, 1.0, crossfade_samples).astype(arr.dtype)
                    arr = arr.copy()
                    arr[:crossfade_samples] *= fade_in
                else:
                    combined.append(np.zeros(gap_samples, dtype=arr.dtype))
            combined.append(arr)

        sf.write(output_path, np.concatenate(combined), sr)

        for cp in chunk_paths:
            if os.path.exists(cp):
                os.remove(cp)

        return True

    def cleanup(self):
        try:
            self._bridge_call({"cmd": "quit"})
            self._proc.wait(timeout=10)
        except Exception:
            self._proc.kill()


def synthesize_segments(segments, profiles, tts, work_dir):
    """
    Synthétise UN clip TTS par segment de parole.

    Chaque segment WhisperX est court → XTTS produit un audio propre
    sans besoin de découpage ou de crossfade.
    """
    print(f"\n🗣️  Passe 5 — Synthèse vocale par segments (XTTS v2)...")

    tts.setup_voices(profiles)

    tts_dir = os.path.join(work_dir, "tts_segments")
    os.makedirs(tts_dir, exist_ok=True)

    # Invalidation cache
    voice_map_path = os.path.join(tts_dir, "_voice_map.json")
    current_map = {
        "voices": getattr(tts, 'voice_map', {}),
        "speakers": sorted({s.speaker for s in segments}),
    }
    current_sig = json.dumps(current_map, sort_keys=True, default=str)

    cache_valid = False
    if os.path.exists(voice_map_path):
        try:
            cache_valid = (open(voice_map_path).read() == current_sig)
        except Exception:
            pass

    if not cache_valid:
        old_clips = [f for f in os.listdir(tts_dir) if f.endswith(".wav")]
        if old_clips:
            print(f"   🧹 Cache invalidé → suppression de {len(old_clips)} clips")
            for f in old_clips:
                os.remove(os.path.join(tts_dir, f))

    with open(voice_map_path, "w") as f:
        f.write(current_sig)

    t0 = time.time()
    success = 0

    for i, seg in enumerate(segments):
        if not seg.text_tgt:
            continue

        out_path = os.path.join(tts_dir, f"seg_{seg.index:04d}.wav")

        if os.path.exists(out_path):
            seg.tts_path = out_path
            success += 1
            continue

        result = tts.synthesize(seg.text_tgt, seg.speaker, out_path)
        if result:
            seg.tts_path = result
            success += 1

        if (i + 1) % 10 == 0 or i == len(segments) - 1:
            print(f"   🗣️  {success}/{i+1} segments synthétisés...")

    elapsed = time.time() - t0
    print(f"   ✅ {success}/{len(segments)} segments TTS ({elapsed:.0f}s)")

    # Trimmer le silence parasite début/fin de chaque clip
    import soundfile as sf
    trimmed = 0
    for seg in segments:
        if seg.tts_path and os.path.exists(seg.tts_path):
            dur_before = sf.info(seg.tts_path).duration
            dur_after = _trim_tts_silence(seg.tts_path)
            if dur_after < dur_before - 0.05:
                trimmed += 1
    if trimmed:
        print(f"   ✂️  {trimmed} clips trimmés (silence début/fin supprimé)")

    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 5b : NORMALISATION RMS (COHÉRENCE DE VOLUME)
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_volumes(segments, work_dir):
    """Normalise le volume RMS de tous les clips TTS par locuteur pour cohérence."""
    import soundfile as sf
    import numpy as np

    print(f"\n🎛️  Passe 5b — Normalisation de volume (RMS)...")
    t0 = time.time()

    norm_dir = os.path.join(work_dir, "tts_normalized")
    os.makedirs(norm_dir, exist_ok=True)

    # Collecter les RMS par locuteur
    speaker_clips = {}
    for seg in segments:
        if not seg.tts_path or not os.path.exists(seg.tts_path):
            continue
        audio, sr = sf.read(seg.tts_path)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if len(audio) / sr < 0.3:
            continue

        rms = np.sqrt(np.mean(audio ** 2))
        rms_db = 20 * np.log10(max(rms, 1e-10))

        speaker_clips.setdefault(seg.speaker, []).append({
            "seg": seg, "audio": audio, "sr": sr, "rms_db": rms_db
        })

    if not speaker_clips:
        return segments

    adjusted = 0
    for spk_id, clips in speaker_clips.items():
        rms_values = [c["rms_db"] for c in clips if c["rms_db"] > -80]
        if not rms_values:
            continue
        target_rms_db = float(np.median(rms_values))

        for c in clips:
            seg, audio, sr = c["seg"], c["audio"], c["sr"]
            out_path = os.path.join(norm_dir, f"seg_{seg.index:04d}.wav")

            rms_diff = target_rms_db - c["rms_db"]
            if abs(rms_diff) > 2.0:
                gain = 10 ** (rms_diff / 20)
                gain = max(0.25, min(4.0, gain))
                audio = audio * gain
                peak = np.max(np.abs(audio))
                if peak > 0.98:
                    audio = audio * (0.98 / peak)
                adjusted += 1

            sf.write(out_path, audio, sr)
            seg.tts_path = out_path

    total = sum(len(clips) for clips in speaker_clips.values())
    print(f"   ✅ {total} clips : {adjusted} ajustés ({time.time()-t0:.1f}s)")
    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING : RÉPARATION DES CLICS AUDIO
# ═══════════════════════════════════════════════════════════════════════════════

def _repair_clicks(audio_path):
    """
    Détecte et corrige les micro-artefacts sonores (clics/blips) dans un fichier audio.

    Algorithme :
      1. Charger en float32
      2. Calculer l'enveloppe d'énergie par fenêtres de 1 ms
      3. Calculer la médiane locale glissante sur 50 ms
      4. Détecter les pics d'énergie > 8× la médiane locale ET durée < 3 ms
      5. Vérifier l'absence de structure harmonique (autocorrélation faible)
      6. Interpoler par spline cubique sur la micro-zone touchée
      7. Réécrire le fichier

    Un "oui" rapide (~150 ms = 150 fenêtres) n'est jamais détecté comme faux positif.
    """
    import soundfile as sf
    import numpy as np
    from scipy.interpolate import CubicSpline

    data, sr = sf.read(audio_path, dtype="float32")
    if len(data) == 0:
        return

    mono = data if data.ndim == 1 else data.mean(axis=1)

    # Fenêtre de 1 ms
    win = max(1, int(sr * 0.001))
    n_frames = len(mono) // win
    if n_frames < 50:
        return

    # Enveloppe d'énergie par fenêtre de 1 ms
    frames = mono[:n_frames * win].reshape(n_frames, win)
    energy = np.sqrt(np.mean(frames ** 2, axis=1))

    # Médiane locale glissante sur 50 ms (50 fenêtres)
    med_half = 25
    local_median = np.empty_like(energy)
    for i in range(n_frames):
        lo = max(0, i - med_half)
        hi = min(n_frames, i + med_half + 1)
        local_median[i] = np.median(energy[lo:hi])

    # Seuil : pic > 8× la médiane locale
    threshold_ratio = 8.0
    max_click_frames = 3  # durée max d'un clic = 3 ms

    # Trouver les pics candidats
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(local_median > 0, energy / local_median, 0)
    candidates = np.where(ratio > threshold_ratio)[0]

    if len(candidates) == 0:
        return

    # Regrouper les candidats contigus en clusters
    clicks = []
    cluster_start = candidates[0]
    cluster_end = candidates[0]
    for idx in candidates[1:]:
        if idx <= cluster_end + 1:
            cluster_end = idx
        else:
            if (cluster_end - cluster_start + 1) <= max_click_frames:
                clicks.append((cluster_start, cluster_end))
            cluster_start = idx
            cluster_end = idx
    if (cluster_end - cluster_start + 1) <= max_click_frames:
        clicks.append((cluster_start, cluster_end))

    if not clicks:
        return

    # Vérifier chaque clic et interpoler
    repaired = 0
    for frame_start, frame_end in clicks:
        sample_start = frame_start * win
        sample_end = min((frame_end + 1) * win, len(mono))

        # Vérifier l'absence de structure harmonique (autocorrélation faible)
        segment = mono[sample_start:sample_end]
        if len(segment) < 4:
            continue
        # Autocorrélation normalisée — un vrai son harmonique a une autocorr > 0.3
        ac = np.correlate(segment - segment.mean(), segment - segment.mean(), mode='full')
        ac = ac[len(ac)//2:]
        if ac[0] > 0:
            ac = ac / ac[0]
            # Chercher un pic secondaire significatif (harmonique)
            if len(ac) > 2 and np.max(ac[1:]) > 0.3:
                continue  # structure harmonique = ce n'est pas un clic

        # Zone d'interpolation : étendre de 2 ms de chaque côté pour la spline
        margin = int(sr * 0.002)
        interp_start = max(0, sample_start - margin)
        interp_end = min(len(mono), sample_end + margin)

        # Points d'ancrage = les bords sains
        left_end = sample_start
        right_start = sample_end
        n_left = left_end - interp_start
        n_right = interp_end - right_start

        if n_left < 2 or n_right < 2:
            continue

        # Construire la spline sur les points sains
        x_left = np.arange(interp_start, left_end)
        x_right = np.arange(right_start, interp_end)
        x_anchor = np.concatenate([x_left, x_right])
        y_anchor = np.concatenate([mono[interp_start:left_end],
                                   mono[right_start:interp_end]])

        if len(x_anchor) < 4:
            continue

        cs = CubicSpline(x_anchor, y_anchor)
        x_repair = np.arange(left_end, right_start)
        repaired_samples = cs(x_repair)

        # Appliquer la réparation
        if data.ndim == 1:
            data[left_end:right_start] = repaired_samples.astype(np.float32)
        else:
            # Appliquer le même ratio de correction à tous les canaux
            for ch in range(data.shape[1]):
                ch_segment = data[left_end:right_start, ch]
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio_ch = np.where(
                        np.abs(mono[left_end:right_start]) > 1e-10,
                        repaired_samples / mono[left_end:right_start],
                        0)
                data[left_end:right_start, ch] = (ch_segment * ratio_ch).astype(np.float32)
            # Mettre à jour mono aussi
            mono[left_end:right_start] = repaired_samples

        repaired += 1

    if repaired > 0:
        sf.write(audio_path, data, sr)
        print(f"      🔧 {repaired} clic(s) réparé(s) dans {os.path.basename(audio_path)}")


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 6 : ASSEMBLAGE SÉQUENTIEL → MP3
# ═══════════════════════════════════════════════════════════════════════════════

def assemble_sequential_mp3(segments, output_path,
                            pause_ms=DEFAULT_PAUSE_MS,
                            speaker_pause_ms=DEFAULT_SPEAKER_PAUSE_MS):
    """
    Concatène les clips TTS des segments dans l'ordre séquentiel avec des
    pauses naturelles. Pas de calage temporel sur l'original.

    - Pause courte entre segments du même locuteur (respiration)
    - Pause longue lors d'un changement de locuteur (tour de parole)
    - Micro-fondus anti-clic en entrée et sortie de chaque clip
    """
    from pydub import AudioSegment

    ANTI_CLICK_MS = 15  # micro-fondu anti-clic (préserve les attaques)

    print(f"\n🎚️  Passe 6 — Assemblage séquentiel → MP3...")
    t0 = time.time()

    combined = AudioSegment.empty()
    overlaid = 0
    skipped = 0
    prev_speaker = None

    for seg in segments:
        if not seg.tts_path or not os.path.exists(seg.tts_path):
            continue

        try:
            clip = AudioSegment.from_wav(seg.tts_path)

            # Clips très courts → probablement un artefact, pas de la parole
            if len(clip) < 150:
                skipped += 1
                continue

            # Micro-fondus anti-clic en entrée ET sortie
            fade = min(ANTI_CLICK_MS, len(clip) // 4)
            if fade > 0:
                clip = clip.fade_in(fade).fade_out(fade)

            # Pause avant le clip
            if overlaid > 0:
                if seg.speaker != prev_speaker:
                    combined += AudioSegment.silent(duration=speaker_pause_ms)
                else:
                    combined += AudioSegment.silent(duration=pause_ms)

            combined += clip
            overlaid += 1
            prev_speaker = seg.speaker

        except Exception as e:
            print(f"      ⚠️  segment S{seg.index} échoué : {e}")

    if len(combined) < 500:
        print(f"   ❌ Audio trop court ({len(combined)}ms)")
        return ""

    # Normalisation finale
    if combined.dBFS != float('-inf'):
        change = -18 - combined.dBFS
        combined = combined.apply_gain(change)

    # Réparation des clics : export WAV temporaire → _repair_clicks → MP3 final
    tmp_wav = output_path.rsplit(".", 1)[0] + "_pre_repair.wav"
    combined.export(tmp_wav, format="wav")
    _repair_clicks(tmp_wav)
    from pydub import AudioSegment as _AS
    combined = _AS.from_wav(tmp_wav)
    os.remove(tmp_wav)

    combined.export(output_path, format="mp3", bitrate="192k")
    mb = os.path.getsize(output_path) / (1024 * 1024)
    dur = len(combined) / 1000

    skip_msg = f", {skipped} artefacts ignorés" if skipped else ""
    print(f"   ✅ {output_path} ({dur:.1f}s, {mb:.1f} Mo) — {overlaid} segments{skip_msg} [{time.time()-t0:.0f}s]")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════════

def save_segments(segs, path):
    data = [{"index": s.index, "start": s.start, "end": s.end,
             "text": s.text, "text_tgt": s.text_tgt,
             "speaker": s.speaker}
            for s in segs]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"   💾 → {path}")


def load_segments(path):
    with open(path) as f:
        data = json.load(f)
    segments = [
        Segment(
            index=d["index"], start=d["start"], end=d["end"],
            text=d["text"],
            text_tgt=d.get("text_tgt", d.get("text_fr", "")),
            speaker=d.get("speaker", "SPEAKER_00"),
        )
        for d in data
    ]
    # Scan de décontamination : vider les text_tgt qui sont des fuites de prompt
    contaminated = 0
    for seg in segments:
        if seg.text_tgt and _est_fuite_prompt(seg.text_tgt):
            print(f"   ⚠️  Segment [S{seg.index}] contaminé, re-traduction forcée : {seg.text_tgt[:50]}…")
            seg.text_tgt = ""
            contaminated += 1
    if contaminated:
        print(f"   🧹 {contaminated} segment(s) décontaminé(s) — re-traduction nécessaire")
    return segments


def generate_report(segments, profiles, output_path, src_lang, tgt_lang):
    """Génère un rapport texte de la traduction."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"RAPPORT DE TRADUCTION VOCALE ({src_lang.upper()} → {tgt_lang.upper()})\n")
        f.write("=" * 70 + "\n\n")

        f.write("LOCUTEURS :\n")
        for spk_id, p in profiles.items():
            f.write(f"  {spk_id} : {p.segment_count} segments, {p.total_duration:.1f}s\n")
        f.write("\n")

        f.write("SEGMENTS TRADUITS :\n\n")
        for seg in segments:
            f.write(f"[S{seg.index}] ({seg.speaker}) {seg.start:.1f}–{seg.end:.1f}s\n")
            f.write(f"  {src_lang.upper()} : {seg.text}\n")
            f.write(f"  {tgt_lang.upper()} : {seg.text_tgt}\n")
            f.write(f"  TTS  : {'✅' if seg.tts_path else '❌'}\n\n")


# ═══════════════════════════════════════════════════════════════════════════════
# TRAITEMENT D'UN FICHIER (MP3/MP4)
# ═══════════════════════════════════════════════════════════════════════════════

def process_one_file(input_path: str, args, claude_client, tts_backend=None,
                     analysis_client=None):
    """
    Pipeline complet pour un fichier MP3 ou MP4.
    Retourne le backend TTS (réutilisé entre fichiers).
    """
    analysis_client = analysis_client or claude_client
    src_lang = args.source_lang
    tgt_lang = args.target_lang

    src = Path(input_path)
    base = src.stem

    work_dir = str(WORK_DIR / f"{base}_translation_work")
    os.makedirs(work_dir, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = str(OUTPUT_DIR / f"{base}_{tgt_lang}.mp3")
    audio_16k = os.path.join(work_dir, "audio_16k.wav")
    audio_hq = os.path.join(work_dir, "audio_hq.wav")
    seg_json = os.path.join(work_dir, "segments.json")
    ana_json = os.path.join(work_dir, "analysis.json")
    report_txt = str(OUTPUT_DIR / f"{base}_translation_report.txt")

    src_n, tgt_n = lang_name(src_lang), lang_name(tgt_lang)

    print("\n" + "=" * 60)
    print(f"🎙️  Traduction vocale : {src.name}")
    print("=" * 60)
    print(f"   Langues  : {src_lang.upper()} → {tgt_lang.upper()} ({src_n} → {tgt_n})")
    print(f"   Sortie   : {output}")
    if args.context:
        print(f"   Contexte : {args.context[:80]}{'...' if len(args.context) > 80 else ''}")
    print("=" * 60)

    t_global = time.time()

    # ── PASSE 1 : Transcription ──────────────────────────────────────────
    if args.segments:
        print(f"\n🔄 Chargement des segments depuis {args.segments}")
        segments = load_segments(args.segments)
        print(f"   {len(segments)} segments chargés")
    elif os.path.exists(seg_json):
        print(f"\n🔄 Reprise depuis {seg_json}")
        segments = load_segments(seg_json)
        print(f"   {len(segments)} segments")
    else:
        extract_audio_16k(input_path, audio_16k)
        segments = transcribe_whisperx(audio_16k, src_lang, args.hf_token)
        save_segments(segments, seg_json)

    # ── PASSE 1b + 3 : Audio HQ + Demucs ────────────────────────────────
    if not os.path.exists(audio_hq):
        extract_audio_hq(input_path, audio_hq)
    vocals_path = separate_sources(audio_hq, work_dir)

    # ── PASSE 2 : Diarisation ────────────────────────────────────────────
    if args.speakers == 1:
        print(f"\n👤 Passe 2 — Diarisation sautée (--speakers 1)")
        for s in segments:
            s.speaker = "SPEAKER_00"
        save_segments(segments, seg_json)
    elif all(s.speaker == "SPEAKER_00" for s in segments):
        if not os.path.exists(audio_16k):
            extract_audio_16k(input_path, audio_16k)
        segments = diarize_speakers(audio_16k, segments, args.hf_token, args.speakers)
        save_segments(segments, seg_json)

    # ── PASSE 3b : Échantillons vocaux ───────────────────────────────────
    profiles = {}
    if vocals_path:
        profiles = extract_speaker_samples(segments, vocals_path, work_dir)

    # Genre forcé
    if args.gender != "auto":
        for p in profiles.values():
            p.gender = args.gender
        print(f"   🔧 Genre forcé → {args.gender}")

    # ── PASSE 4 : Analyse + Traduction (sur les SEGMENTS) ────────────────
    if not any(seg.text_tgt for seg in segments):
        analysis = analyze_content(segments, analysis_client, src_lang, tgt_lang, args.context)
        with open(ana_json, "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)

        segments = translate_segments(segments, analysis, claude_client,
                                      src_lang, tgt_lang, args.context)
        save_segments(segments, seg_json)

        # Relecture
        if not args.skip_review:
            segments = review_segments(segments, analysis, claude_client,
                                       src_lang, tgt_lang, args.context)
            save_segments(segments, seg_json)

            # Passe 4d : cohérence globale
            segments = check_consistency(segments, analysis, claude_client,
                                         src_lang, tgt_lang)
            save_segments(segments, seg_json)

            # Passe 4e : vérification glossaire
            segments = verify_glossary(segments, analysis, claude_client,
                                       src_lang, tgt_lang)
            save_segments(segments, seg_json)
    else:
        print(f"\n⏩ Traduction déjà présente — passe 4 sautée")
        if os.path.exists(ana_json):
            with open(ana_json) as f:
                analysis = json.load(f)
        else:
            analysis = {}

    # ── PASSE 5 : Synthèse TTS (sur les SEGMENTS) ───────────────────────
    if tts_backend is None:
        model_choice = getattr(args, 'model', 'qwen3tts')
        if model_choice == "qwen3tts":
            tts_backend = Qwen3TTSBackend(
                target_lang=tgt_lang,
                source_lang=src_lang,
            )
        else:
            tts_backend = XTTSBackend(
                target_lang=tgt_lang,
                ref_voice=getattr(args, 'ref_voice', None),
                xtts_speaker=getattr(args, 'xtts_speaker', None),
            )
        tts_backend.verify_tts = getattr(args, 'verify_tts', False)

    try:
        segments = synthesize_segments(segments, profiles, tts_backend, work_dir)
        save_segments(segments, seg_json)
    except Exception as e:
        print(f"   ❌ Erreur TTS : {e}")

    # ── PASSE 5b : Normalisation volume ──────────────────────────────────
    segments = normalize_volumes(segments, work_dir)

    # ── PASSE 6 : Assemblage MP3 ────────────────────────────────────────
    assemble_sequential_mp3(segments, output,
                            pause_ms=args.pause,
                            speaker_pause_ms=args.speaker_pause)

    # ── Rapport ──────────────────────────────────────────────────────────
    generate_report(segments, profiles, report_txt, src_lang, tgt_lang)

    # ── Nettoyage ────────────────────────────────────────────────────────
    if os.path.exists(audio_16k):
        os.remove(audio_16k)

    # ── Résumé ───────────────────────────────────────────────────────────
    elapsed = time.time() - t_global
    tts_ok = sum(1 for s in segments if s.tts_path)

    print(f"\n{'='*60}")
    print(f"🎉 Traduction terminée en {elapsed/60:.1f} min !")
    print(f"{'='*60}")
    print(f"   👥 {len(profiles)} locuteur(s)")
    print(f"   🗣️  {tts_ok}/{len(segments)} segments synthétisés")
    print(f"   🎧 Audio traduit : {output}")
    print(f"   📝 Rapport       : {report_txt}")
    print(f"   📁 Dossier       : {work_dir}")
    print(f"{'='*60}\n")

    return tts_backend


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global WHISPER_MODEL, CLAUDE_MODEL

    p = argparse.ArgumentParser(
        description="Traduction vocale IA — batch MP3/MP4 → MP3 traduit (Qwen3-TTS / XTTS v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Exemples :
              python doubler-mp3-batch.py                                    # tous MP3+MP4, EN → FR
              python doubler-mp3-batch.py -s en -t es                        # EN → ES
              python doubler-mp3-batch.py --file podcast.mp3                 # un seul MP3
              python doubler-mp3-batch.py --file interview.mp4               # un seul MP4
              python doubler-mp3-batch.py --xtts-speaker "Craig Gutsy"       # voix preset XTTS
              python doubler-mp3-batch.py --model xtts --file test.mp3       # backend XTTS v2
              python doubler-mp3-batch.py --ref-voice ma_voix.wav            # voix de référence
              python doubler-mp3-batch.py --speakers 1                       # monologue
              python doubler-mp3-batch.py --pause 1000 --speaker-pause 1500  # pauses longues
              python doubler-mp3-batch.py --context "podcast tech décontracté"
              python doubler-mp3-batch.py --list-xtts-speakers               # lister voix XTTS
        """))

    p.add_argument("--file", metavar="FILE",
                   help="Traiter un seul fichier MP3 ou MP4 (défaut: tous les *.mp3 + *.mp4 du dossier courant)")
    p.add_argument("-s", "--source-lang", default="en",
                   help="Langue source (code ISO 639-1, défaut: en)")
    p.add_argument("-t", "--target-lang", default="fr",
                   help="Langue cible (code ISO 639-1, défaut: fr)")
    p.add_argument("--ref-voice", metavar="WAV",
                   help="Audio de référence pour le clonage vocal (10-15s)")
    p.add_argument("--xtts-speaker", metavar="NAME",
                   help="Voix preset XTTS v2 (ex: 'Craig Gutsy'). "
                        "Liste : --list-xtts-speakers")
    p.add_argument("--list-xtts-speakers", action="store_true",
                   help="Lister les voix preset XTTS v2 et quitter")
    p.add_argument("--model", choices=["qwen3tts", "xtts"], default="qwen3tts",
                   help="Backend TTS à utiliser (défaut: qwen3tts)")
    p.add_argument("--segments", metavar="JSON",
                   help="Reprendre depuis un fichier segments existant")
    p.add_argument("--speakers", type=int, default=None,
                   help="Forcer le nombre de locuteurs (1 = monologue, défaut: auto)")
    p.add_argument("--gender", choices=["male", "female", "auto"], default="auto",
                   help="Forcer le genre pour le choix de voix TTS (défaut: auto)")
    p.add_argument("--pause", type=int, default=DEFAULT_PAUSE_MS,
                   help=f"Pause entre segments même locuteur, en ms (défaut: {DEFAULT_PAUSE_MS})")
    p.add_argument("--speaker-pause", type=int, default=DEFAULT_SPEAKER_PAUSE_MS,
                   help=f"Pause lors d'un changement de locuteur, en ms (défaut: {DEFAULT_SPEAKER_PAUSE_MS})")
    p.add_argument("--skip-review", action="store_true",
                   help="Passer la relecture de traduction (passe 4c)")
    p.add_argument("--skip-checks", action="store_true",
                   help="Passer la vérification des dépendances")
    p.add_argument("--verify-tts", action="store_true",
                   help="Vérifier chaque sortie TTS par ASR (WhisperX base) — "
                        "re-synthétise si <70%% des mots détectés")
    p.add_argument("--context", type=str, default="",
                   help="Contexte pour guider la traduction (noms, sujet, registre...)")
    p.add_argument("--whisper-model", default=WHISPER_MODEL)
    p.add_argument("--claude-model", default=CLAUDE_MODEL)
    p.add_argument("--llm", choices=["claude", "local"], default="local",
                   help="Backend LLM : local (Ollama, défaut) ou claude (API Anthropic)")
    p.add_argument("--analysis-llm", choices=["auto", "claude", "local"], default="auto",
                   help="LLM de la passe d'analyse/contexte : auto = Claude si "
                        "ANTHROPIC_API_KEY dispo, sinon local")
    p.add_argument("--ollama-model", default=OLLAMA_MODEL,
                   help=f"Modèle Ollama (défaut: {OLLAMA_MODEL})")
    p.add_argument("--ollama-url", default=OLLAMA_URL,
                   help=f"URL du serveur Ollama (défaut: {OLLAMA_URL})")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = p.parse_args()

    # ── Liste des voix ──────────────────────────────────────────────────
    if args.list_xtts_speakers:
        print("\n🎤 Voix preset XTTS v2")
        print("=" * 55)
        print("\n   ♀️  Voix féminines :")
        for v in XTTS_VOICES_FEMALE:
            print(f"      • {v}")
        print("\n   ♂️  Voix masculines :")
        for v in XTTS_VOICES_MALE:
            print(f"      • {v}")
        print(f"\n   Total : {len(XTTS_VOICES_FEMALE) + len(XTTS_VOICES_MALE)} voix")
        print(f"\n   Usage : --xtts-speaker \"Craig Gutsy\"\n")
        sys.exit(0)

    src_lang = args.source_lang.lower()
    tgt_lang = args.target_lang.lower()

    if src_lang == tgt_lang:
        print(f"❌ Langue source et cible identiques ({src_lang})"); sys.exit(1)

    # ── Validation backend / langue ──────────────────────────────────
    if args.model == "qwen3tts" and tgt_lang not in QWEN3TTS_LANG_MAP:
        langs = ", ".join(sorted(QWEN3TTS_LANG_MAP.keys()))
        print(f"❌ Qwen3-TTS ne supporte pas la langue cible '{tgt_lang}'.")
        print(f"   Langues supportées : {langs}")
        sys.exit(1)

    if args.xtts_speaker and args.model != "xtts":
        print(f"⚠️  --xtts-speaker ignoré (backend actif : {args.model})")

    WHISPER_MODEL = args.whisper_model
    CLAUDE_MODEL = args.claude_model

    # ── Déterminer les fichiers à traiter ────────────────────────────────
    if args.file:
        src_file = resolve_source(args.file)
        if not src_file.exists():
            print(f"❌ Introuvable : {args.file}"); sys.exit(1)
        media_files = [str(src_file)]
    else:
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        mp3_files = sorted(str(p) for p in INPUT_DIR.glob("*.mp3"))
        mp4_files = sorted(str(p) for p in INPUT_DIR.glob("*.mp4"))
        mp3_files = [f for f in mp3_files if not f.endswith(f"_{tgt_lang}.mp3")]
        mp4_files = [f for f in mp4_files if not f.endswith(f"_{tgt_lang}.mp4")]
        media_files = mp3_files + mp4_files

    if not media_files:
        print(f"❌ Aucun fichier MP3 ou MP4 trouvé dans {INPUT_DIR}")
        print(f"   (les fichiers *_{tgt_lang}.mp3/mp4 sont exclus)")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"📁 Batch : {len(media_files)} fichier(s) à traduire")
    print(f"{'='*60}")
    for i, f in enumerate(media_files, 1):
        print(f"   {i}. {f}")
    print(f"{'='*60}\n")

    # ── Vérifications ────────────────────────────────────────────────────
    if not args.skip_checks:
        check_dependencies(tts_backend=args.model, local=(args.llm == "local"))

    # ── Backend LLM (API Claude ou Ollama local) ─────────────────────────
    if args.llm == "local":
        claude = _OllamaClient(args.ollama_url, args.ollama_model)
        print(f"   🧠 LLM local : Ollama {args.ollama_model}")
    else:
        import anthropic
        claude = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Analyse : Claude si dispo (meilleure connaissance du monde) → meilleur
    # contexte ET meilleure traduction. Un seul appel, peu coûteux.
    analysis_client = claude
    if args.analysis_llm != "local":
        _want_claude = args.analysis_llm == "claude" or (
            args.analysis_llm == "auto" and os.environ.get("ANTHROPIC_API_KEY"))
        if _want_claude:
            try:
                import anthropic
                analysis_client = anthropic.Anthropic()
                print("   🧠 Analyse du contexte via Claude (traduction ensuite en local)")
            except Exception as _e:
                print(f"   ⚠️  Claude indispo pour l'analyse ({_e}) — analyse en local")

    # ── Traitement batch ─────────────────────────────────────────────────
    t_batch = time.time()
    tts_backend = None
    results = []

    for fi, filepath in enumerate(media_files, 1):
        print(f"\n{'━'*60}")
        print(f"📀 Fichier {fi}/{len(media_files)} : {filepath}")
        print(f"{'━'*60}")

        try:
            tts_backend = process_one_file(filepath, args, claude, tts_backend,
                                           analysis_client=analysis_client)
            results.append((filepath, "✅"))
        except Exception as e:
            print(f"\n❌ Erreur sur {filepath} : {e}")
            results.append((filepath, f"❌ {e}"))

    # Cleanup TTS
    if tts_backend:
        tts_backend.cleanup()

    # ── Bilan batch ──────────────────────────────────────────────────────
    elapsed = time.time() - t_batch
    ok = sum(1 for _, s in results if "✅" in s)

    print(f"\n{'='*60}")
    print(f"🏁 Batch terminé en {elapsed/60:.1f} min")
    print(f"{'='*60}")
    for mp3, status in results:
        print(f"   {status} {mp3}")
    print(f"\n   {ok}/{len(results)} fichiers traduits avec succès")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
