#!/usr/bin/env python3
"""
Pipeline de doublage IA — XTTS v2 / Qwen3-TTS / ElevenLabs
=========================================================================
Produit une vidéo doublée dans une langue cible avec clonage vocal
par locuteur, séparation de sources, et mixage professionnel.
Backends TTS : XTTS v2 (défaut, local, GPU),
Qwen3-TTS (local, GPU, 10 langues) ou ElevenLabs (API cloud).

Architecture en 14 passes :
  1. WhisperX          → transcription + timestamps mot par mot
  2. Pyannote           → diarisation (identification des locuteurs)
  3. Demucs (Meta)      → séparation voix / fond sonore
  4a. Claude            → analyse du contenu (glossaire, ton, locuteurs)
  4b. Claude            → traduction contextuelle pour doublage
  4c. Claude            → relecture (naturel oral, glossaire, contresens)
  4d. Claude            → cohérence globale (terminologie, registre, ton)
  4e. Claude            → vérification glossaire (violations → corrections)
  5. Claude             → adaptation isochronique (durée ≈ original)
  5b. Claude            → relecture fluidité post-adaptation (connecteurs, ponctuation)
  6. XTTS v2 two-pass   → synthèse vocale + ajustement speed natif
  6c. Boucle qualité    → speed-first, puis réécriture Claude si insuffisant
  6b. Normalisation     → RMS (volume) par défaut ; + F0 WORLD avec --fix-pitch
  7. pydub              → mixage voice-over (voix doublées + originales + fond)
  8. ffmpeg             → assemblage vidéo finale

La passe 6 utilise le two-pass TTS : synthèse à speed=1.0, mesure de la durée,
re-synthèse avec speed ajusté si nécessaire. Le paramètre speed natif d'XTTS
modifie le débit au niveau du décodeur, sans artefacts de time-stretching.
Ça élimine la plupart des problèmes de chevauchement/silence dès la synthèse,
réduisant drastiquement le besoin de réécriture Claude en passe 6c.

Par défaut, le mixage se fait en mode « voice-over » style Arte :
la voix originale reste audible en arrière-plan (lead-in, ducking,
lead-out) comme dans les reportages et documentaires. Désactiver
avec --no-voiceover pour un doublage pur.

Usage :
  python doubler.py video.mp4                                      # EN → FR, clonage vocal
  python doubler.py "https://www.youtube.com/watch?v=XXXXX"        # depuis YouTube
  python doubler.py video.mp4 --xtts-speaker "Craig Gutsy"         # voix preset XTTS
  python doubler.py video.mp4 -s en -t es                          # EN → ES
  python doubler.py video.mp4 --segments segments.json             # reprendre traduction
  python doubler.py video.mp4 --keep-original 0.05                 # garder 5 % voix originale
  python doubler.py video.mp4 --vo-style jt                        # voice-over style JT
  python doubler.py video.mp4 --no-voiceover                       # doublage pur
  python doubler.py video.mp4 --remove-music                        # supprimer musique de fond
  python doubler.py video.mp4 --speakers 2                         # forcer 2 locuteurs
  python doubler.py video.mp4 --ref-voice ref_fr.wav               # voix de référence externe
  python doubler.py --list-xtts-speakers                           # lister les voix preset

Prérequis :
  pip install whisperx anthropic torch torchaudio demucs pydub soundfile \\
              numpy praat-parselmouth pyworld TTS --break-system-packages
  # + ffmpeg installé
  # + ANTHROPIC_API_KEY
  # + HF_TOKEN (pour pyannote — diarisation)
"""

import argparse
import gc
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
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import hw
hw.setup_rocm_env()  # AMD/ROCm (gfx1151) : pose HSA_OVERRIDE_* avant tout import torch

import epochtimes
import apollohealth

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

# Ollama (LLM local — alternative gratuite à l'API Claude)
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma4:31b"        # cf. bench 2026-06-22 : meilleur FR oral (traduction)
OLLAMA_NUM_PREDICT = 16384             # marge large (tokens réflexion Qwen3 inclus)

# TTS
TTS_BACKEND = "xtts"
XTTS_MAX_CHARS = 250                # marge de sécurité sous la limite XTTS (273 pour le français)

# Doublage audio
SAMPLE_RATE = 44100
MIN_SPEAKER_SAMPLE_SEC = 5          # durée min d'échantillon par locuteur
MAX_SPEAKER_SAMPLE_SEC = 30         # durée max (au-delà = diminishing returns)
DUCK_DB = 10                        # atténuation du fond pendant les voix (dB)
DUCK_FADE_MS = 60                   # fondu entrée/sortie du ducking (ms)
CROSSFADE_MS = 30                   # fondu entre segments TTS consécutifs (concaténation multi-chunks)
GAP_BETWEEN_CLIPS_MS = 80            # silence minimum entre clips TTS consécutifs (anti-mitraillette)
TTS_ANTI_CLICK_MS = 15               # micro-fondu anti-clic sur chaque clip TTS (préserve les attaques)

# Assemblage séquentiel (mode --audio-only : pas de calage temporel)
AUDIO_ONLY_PAUSE_MS = 600            # pause entre segments d'un même locuteur
AUDIO_ONLY_SPEAKER_PAUSE_MS = 900    # pause lors d'un changement de locuteur

# Two-pass TTS : ajustement de vitesse natif XTTS (speed parameter)
# Élimine la plupart des chevauchements sans réécriture Claude ni time-stretching
XTTS_SPEED_MIN = 0.82               # en dessous, la qualité XTTS se dégrade
XTTS_SPEED_MAX = 1.30               # au-dessus, la qualité XTTS se dégrade
XTTS_SPEED_TOLERANCE = 0.10         # ±10% → pas de re-synthèse (assez proche)
XTTS_SPEED_RESCUE_MAX = 1.40        # vitesse max en "sauvetage" (passe 6c)
SPEED_COMFORT_MAX = 1.20             # au-delà, accélération perceptible → préférer réécriture

# ElevenLabs (API cloud — alternative à XTTS)
ELEVENLABS_MAX_CHARS = 5000
ELEVENLABS_MODEL_DEFAULT = "eleven_multilingual_v2"
ELEVENLABS_OUTPUT_FORMAT = "mp3_44100_128"
ELEVENLABS_SPEED_MIN = 0.70
ELEVENLABS_SPEED_MAX = 1.50
ELEVENLABS_STABILITY = 0.50
ELEVENLABS_SIMILARITY_BOOST = 0.75
ELEVENLABS_STYLE = 0.0
ELEVENLABS_RETRY_MAX = 3
ELEVENLABS_RETRY_DELAY = 2.0

# Retry API Claude (erreurs transitoires : 529 Overloaded, 429 Rate Limit, 500+)
CLAUDE_RETRY_MAX = 5
CLAUDE_RETRY_DELAY = 10.0              # délai initial en secondes (backoff exponentiel)

# Mode voice-over (style Arte / documentaire / reportage)
# Réf : EBU R 128, pratiques Arte/France 2/BBC pour interviews traduits
# Par défaut activé — désactiver avec --no-voiceover pour un doublage pur
VO_LEAD_IN_MS = 2000                # durée d'écoute de la voix originale AVANT le doublage
VO_LEAD_OUT_MS = 1000               # durée d'écoute de la voix originale APRÈS le doublage
VO_ORIG_DUCK_DB = 15                # atténuation de la voix originale PENDANT le doublage
VO_BG_DUCK_DB = 5                   # atténuation du fond sonore pendant les voix
VO_FADE_MS = 250                    # fondus entrée/sortie (lents = plus doux, style Arte)
VO_TTS_TARGET_DBFS = -16            # niveau cible de la voix doublée (premier plan)
VO_ORIG_BETWEEN_DB = -2             # légère atténuation de la voix originale entre segments
VO_MAX_BRIDGE_GAP_MS = 800          # gap original max pour considérer une phrase continue
VO_MAX_DRIFT_MS = 1500              # dérive max tolérée pour le push-later cascade (anti-troncature)
VO_GLUE_CONTINUATIONS = True        # coller les continuations de phrase (même locuteur, segment
                                    # précédent sans ponctuation finale) juste après la 1re moitié,
                                    # même si le silence original > VO_MAX_BRIDGE_GAP_MS et en
                                    # remontant avant seg.start. Évite « début de phrase … long
                                    # silence … fin de phrase ». Léger désync VF/voix originale
                                    # toléré en voice-over (voix originale duckée). False = ancien
                                    # comportement strictement isochrone.

# Traduction (repris de traduire.py)
CHUNK_SIZE = 60
CHUNK_OVERLAP = 8

# Caractères/seconde moyens par langue (parole fluide, voix normale).
# Sert à calibrer la longueur cible de chaque traduction pour éviter
# les sur-résumés (Claude tend à compresser les phrases longues).
CPS_TARGETS = {
    "fr": 14.5, "es": 15.0, "it": 14.8, "pt": 14.5, "de": 13.0,
    "en": 13.5, "nl": 13.5, "ru": 13.0, "ja": 7.5, "zh": 5.0,
    "ko": 12.0, "ar": 12.5, "hi": 13.0, "tr": 14.0, "pl": 14.0,
    "sv": 13.0,
}
# Plage [min, max] de longueur cible par segment (% du target).
# < 75% → traduction trop courte (résumé excessif)
# > 118% → traduction trop longue (débordera la fenêtre TTS)
TRANSLATION_LEN_MIN_RATIO = 0.75
TRANSLATION_LEN_MAX_RATIO = 1.18

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
        # Nettoyer les balises <think>…</think> (mode réflexion Qwen3)
        text = re.sub(r'<think>[\s\S]*?</think>\s*', '', text)
        return type("Resp", (), {"content": [type("B", (), {"text": text})()]})()


def _claude_create(client, **kwargs):
    """Appel LLM avec retry automatique (Claude API ou Ollama local)."""
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


# ── Sortie structurée { id → text } pour les passes de traduction/relecture ──
# Force Claude à appeler un outil dont le schéma valide qu'on a bien {id, text}.
# Élimine les fuites de méta-commentaire qui passaient par le parsing texte libre.
SUBMIT_TEXTS_TOOL = {
    "name": "submit_texts",
    "description": (
        "Soumet la liste finale des textes produits pour chaque segment traité. "
        "Le champ text ne contient QUE le texte du segment — jamais de commentaire, "
        "jamais de préfixe « AVANT: » ou « APRÈS: », jamais de note éditoriale."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "description": (
                    "Une entrée par segment à renvoyer. N'inclure que les segments "
                    "effectivement traités/modifiés ; omettre les segments inchangés "
                    "lorsque la passe est itérative."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "integer",
                            "description": "Numéro du segment, sans les crochets."
                        },
                        "text": {
                            "type": "string",
                            "description": (
                                "Le texte du segment et UNIQUEMENT le texte. "
                                "Pas de préfixe, pas de note, pas de méta-commentaire."
                            )
                        }
                    },
                    "required": ["id", "text"]
                }
            }
        },
        "required": ["items"]
    }
}


def _claude_submit_texts(client, user_prompt: str,
                         system: str = "",
                         max_tokens: int = CLAUDE_MAX_TOKENS) -> dict[int, str]:
    """
    Appelle le LLM en exigeant une sortie structurée { id : texte }.

    - Avec l'API Claude : tool_use forcé sur `submit_texts`. Claude ne peut
      plus répondre en prose ; le seul moyen de produire une sortie est de
      remplir le schéma. Les fuites de méta-commentaire deviennent quasi
      impossibles (il faudrait que Claude colle le commentaire DANS le champ
      `text`, ce qui demande un effort contraire à l'instruction).

    - Avec Ollama (pas de tool use) : on retombe sur le parsing texte libre,
      ligne par ligne, comme l'ancien format `[N] texte`.
    """
    is_local = isinstance(client, _OllamaClient)
    if is_local:
        # Sans tool_use, on impose explicitement le format ligne-par-ligne
        # attendu par le parser, sinon les modèles répondent en prose libre.
        user_prompt += (
            "\n\n=== FORMAT DE RÉPONSE OBLIGATOIRE ===\n"
            "Traduis UNIQUEMENT les segments de la section « À TRADUIRE ».\n"
            "Réponds avec EXACTEMENT une ligne par segment, au format :\n"
            "[N] traduction\n"
            "où N est le numéro entre crochets du segment source. N'écris rien "
            "d'autre : ni préambule, ni commentaire, ni ligne vide, ni le texte source."
        )

    kwargs = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if system:
        kwargs["system"] = system

    if not is_local:
        kwargs["tools"] = [SUBMIT_TEXTS_TOOL]
        kwargs["tool_choice"] = {"type": "tool", "name": "submit_texts"}

    resp = _claude_create(client, **kwargs)

    if is_local:
        # Fallback Ollama : parsing texte libre
        result: dict[int, str] = {}
        text = resp.content[0].text if resp.content else ""
        for line in text.strip().split("\n"):
            m = re.match(r'\[(\d+)\]\s*(.*)', line.strip())
            if m:
                idx, txt = int(m.group(1)), m.group(2).strip()
                txt = _strip_claude_artifacts(txt)
                if txt:
                    result[idx] = txt
        return result

    # API Claude : extraire le bloc tool_use
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", "") == "submit_texts":
            data = block.input if isinstance(block.input, dict) else {}
            items = data.get("items", []) if isinstance(data, dict) else []
            out: dict[int, str] = {}
            for it in items:
                if not isinstance(it, dict):
                    continue
                if "id" not in it or "text" not in it:
                    continue
                try:
                    idx = int(it["id"])
                except (ValueError, TypeError):
                    continue
                txt = str(it["text"]).strip()
                txt = _strip_claude_artifacts(txt)
                if txt:
                    out[idx] = txt
            return out
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURES DE DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DubSegment:
    """Segment enrichi pour le doublage."""
    index: int
    start: float
    end: float
    text: str                               # texte source
    text_tgt: str = ""                      # traduction brute
    text_adapted: str = ""                  # traduction adaptée (isochronie)
    speaker: str = "SPEAKER_00"             # ID locuteur (pyannote)
    tts_path: str = ""                      # chemin audio TTS généré
    is_sentence_start: bool = True           # début de phrase (lead-in appliqué) ou mid-phrase (lead-in=0)
    words: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def speech_text(self) -> str:
        """Texte final à synthétiser (adapté > traduit > source)."""
        return self.text_adapted or self.text_tgt or self.text


def detect_sentence_boundaries(segments):
    """Marque les segments qui commencent une nouvelle phrase.

    Un segment est mid-sentence (is_sentence_start=False) si le segment
    précédent du même locuteur ne se termine pas par une ponctuation de
    fin de phrase (. ? ! … :).
    """
    import re
    for i, seg in enumerate(segments):
        if i == 0:
            seg.is_sentence_start = True
            continue
        prev = segments[i - 1]
        prev_text = (prev.text_tgt or prev.text or "").strip()
        # Changement de locuteur → toujours début de phrase
        if seg.speaker != prev.speaker:
            seg.is_sentence_start = True
        # Le segment précédent se termine par une ponctuation finale → nouvelle phrase
        elif re.search(r'[.!?…:;][\s»"\')\]]*$', prev_text):
            seg.is_sentence_start = True
        else:
            seg.is_sentence_start = False
    return segments


@dataclass
class SpeakerProfile:
    """Profil d'un locuteur détecté."""
    speaker_id: str
    sample_path: str = ""                   # audio d'échantillon concaténé
    sample_text: str = ""                   # transcription de l'échantillon
    total_duration: float = 0.0             # durée totale de parole
    voice_id: str = ""                      # ID voix interne
    segment_count: int = 0
    ref_clips: list = field(default_factory=list)  # [(path, text), ...] clips individuels pour clonage
    gender: str = "unknown"                 # "male" | "female" | "unknown" (estimé par F0)
    f0_median: float = 0.0                  # F0 médian source (Hz) — cible pour la normalisation


# ═══════════════════════════════════════════════════════════════════════════════
# VÉRIFICATIONS PRÉALABLES
# ═══════════════════════════════════════════════════════════════════════════════

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

# Qwen3-TTS (Alibaba Qwen — local, GPU, 10 langues dont FR, Apache 2.0)
QWEN3TTS_MAX_CHARS = 300
QWEN3TTS_SPEED_MIN = 0.70
QWEN3TTS_SPEED_MAX = 1.50
QWEN3TTS_SPEED_TOLERANCE = 0.10
QWEN3TTS_SPEED_RESCUE_MAX = 1.40    # vitesse max en "sauvetage" (passe 6c) — alignée sur XTTS
QWEN3TTS_LANG_MAP = {
    "zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
    "de": "German", "fr": "French", "ru": "Russian", "pt": "Portuguese",
    "es": "Spanish", "it": "Italian",
}


def check_dependencies(tts_backend: str):
    """Vérifie toutes les dépendances du pipeline."""
    print("🔍 Vérification des dépendances...")
    ok = True

    # ffmpeg
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        v = r.stdout.split('\n')[0].split(' ')[2] if r.returncode == 0 else "?"
        print(f"   ffmpeg      : ✅ ({v})")
    except FileNotFoundError:
        print("   ffmpeg      : ❌  → sudo apt install ffmpeg"); ok = False

    # Python packages
    for pkg, name in [("whisperx", "whisperx"), ("anthropic", "anthropic"),
                      ("demucs", "demucs"), ("pydub", "pydub"),
                      ("soundfile", "soundfile"),
                      ("numpy", "numpy"), ("parselmouth", "praat-parselmouth"),
                      ("pyworld", "pyworld")]:
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
    elif tts_backend == "elevenlabs":
        try:
            __import__("elevenlabs"); print("   elevenlabs  : ✅")
        except ImportError:
            print("   elevenlabs  : ❌  → pip install elevenlabs --break-system-packages"); ok = False
        if not os.environ.get("ELEVENLABS_API_KEY"):
            print("   ⚠️  ELEVENLABS_API_KEY non définie"); ok = False
    else:
        try:
            __import__("TTS"); print("   xtts-v2     : ✅")
        except ImportError:
            print("   xtts-v2     : ❌  → pip install TTS --break-system-packages"); ok = False

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("   ⚠️  ANTHROPIC_API_KEY non définie"); ok = False

    if not ok:
        print("\n❌ Dépendances manquantes. Installez-les et relancez.")
        sys.exit(1)
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITAIRE : PARSING --skip MM:SS
# ═══════════════════════════════════════════════════════════════════════════════

def parse_skip(skip_str: Optional[str]) -> float:
    """Parse une chaîne MM:SS ou SS en secondes. Retourne 0.0 si None."""
    if not skip_str:
        return 0.0
    parts = skip_str.strip().split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 1:
            return int(parts[0])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except ValueError:
        pass
    print(f"❌ Format --skip invalide : {skip_str} (attendu MM:SS ou HH:MM:SS)")
    sys.exit(1)


def format_skip(seconds: float) -> str:
    """Formate un nombre de secondes en MM:SS lisible."""
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITAIRE : TÉLÉCHARGEMENT YOUTUBE (yt-dlp)
# ═══════════════════════════════════════════════════════════════════════════════

def is_youtube_url(s: str) -> bool:
    """Détecte si la chaîne est un lien YouTube."""
    return bool(re.match(
        r'https?://(www\.)?(youtube\.com/(watch|shorts|live)|youtu\.be/)', s))


def download_youtube(url: str, output_dir: str = ".") -> str:
    """Télécharge une vidéo YouTube via yt-dlp et retourne le chemin du fichier MP4."""
    # Vérifier que yt-dlp est installé
    if not shutil.which("yt-dlp"):
        print("❌ yt-dlp n'est pas installé.")
        print("   → pip install yt-dlp --break-system-packages")
        print("   ou : sudo apt install yt-dlp")
        sys.exit(1)

    print(f"\n📥 Téléchargement YouTube...")
    print(f"   URL : {url}")

    # Récupérer le titre pour nommer le fichier
    try:
        r = subprocess.run(
            ["yt-dlp", "--print", "title", "--no-warnings", url],
            capture_output=True, text=True, timeout=30)
        title = r.stdout.strip() if r.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        title = ""

    # Nettoyer le titre pour en faire un nom de fichier
    if title:
        safe_title = re.sub(r'[^\w\s\-]', '', title).strip()
        safe_title = re.sub(r'\s+', '_', safe_title)[:80]
    else:
        safe_title = "youtube_video"

    output_template = os.path.join(output_dir, f"{safe_title}.%(ext)s")
    output_mp4 = os.path.join(output_dir, f"{safe_title}.mp4")

    # Si déjà téléchargé, ne pas re-télécharger
    if os.path.exists(output_mp4) and os.path.getsize(output_mp4) > 0:
        print(f"   ⏩ Déjà téléchargé : {output_mp4}")
        return output_mp4

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--no-playlist",
        "--no-warnings",
        url,
    ]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print("❌ Téléchargement YouTube expiré (>10 min)")
        sys.exit(1)

    if r.returncode != 0:
        print(f"❌ Échec du téléchargement YouTube :")
        print(f"   {r.stderr.strip()}")
        sys.exit(1)

    # yt-dlp peut ajouter un suffixe inattendu — chercher le fichier produit
    if not os.path.exists(output_mp4):
        # Chercher le fichier le plus récent correspondant au pattern
        from glob import glob
        candidates = sorted(glob(os.path.join(output_dir, f"{safe_title}.*")),
                            key=os.path.getmtime, reverse=True)
        candidates = [c for c in candidates if c.endswith(('.mp4', '.mkv', '.webm'))]
        if candidates:
            output_mp4 = candidates[0]
        else:
            print(f"❌ Fichier téléchargé introuvable après yt-dlp")
            sys.exit(1)

    size_mb = os.path.getsize(output_mp4) / (1024 * 1024)
    print(f"   ✅ Téléchargé : {output_mp4} ({size_mb:.1f} Mo)")
    return output_mp4


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 1 : EXTRACTION AUDIO + TRANSCRIPTION WHISPERX
# ═══════════════════════════════════════════════════════════════════════════════

def extract_audio(video_path: str, output_path: str, skip_seconds: float = 0.0) -> str:
    """Extrait l'audio en WAV mono 16kHz pour WhisperX."""
    skip_msg = f" (début à {format_skip(skip_seconds)})" if skip_seconds > 0 else ""
    print(f"\n🎵 Passe 1a — Extraction audio{skip_msg}...")
    cmd = ["ffmpeg", "-y"]
    if skip_seconds > 0:
        cmd += ["-ss", str(skip_seconds)]
    cmd += ["-i", video_path,
           "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"❌ ffmpeg : {r.stderr}"); sys.exit(1)
    print(f"   ✅ {output_path}")
    return output_path


def extract_audio_hq(video_path: str, output_path: str, skip_seconds: float = 0.0) -> str:
    """Extrait l'audio en WAV stéréo 44.1kHz pour Demucs + mixage final."""
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"🎵 Passe 1b — Extraction audio HQ (44.1kHz stéréo)...")
        print(f"   ⏩ Déjà extrait : {output_path} ({mb:.1f} Mo)")
        return output_path
    skip_msg = f" (début à {format_skip(skip_seconds)})" if skip_seconds > 0 else ""
    print(f"🎵 Passe 1b — Extraction audio HQ (44.1kHz stéréo){skip_msg}...")
    cmd = ["ffmpeg", "-y"]
    if skip_seconds > 0:
        cmd += ["-ss", str(skip_seconds)]
    cmd += ["-i", video_path,
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


def wait_for_vram_release(min_free_mib: int = 6000, timeout: float = 60.0,
                          poll: float = 1.5) -> bool:
    """Délègue à hw.py (multi-fournisseur : nvidia-smi / amd-smi / rocm-smi, et
    logique mémoire unifiée assouplie pour les APU Strix Halo)."""
    return hw.wait_for_vram_release(min_free_mib=min_free_mib, timeout=timeout, poll=poll)


def free_gpu_for_task(min_free_mib: int = 6000, timeout: float = 60.0):
    """Délègue à hw.py — décharge tout modèle Ollama résident puis vérifie/attend
    la VRAM avant un gros chargement GPU (WhisperX, TTS)."""
    return hw.free_gpu_for_task(min_free_mib=min_free_mib, timeout=timeout)


def transcribe_whisperx(audio_path: str, source_lang: str,
                        hf_token: Optional[str] = None) -> list[DubSegment]:
    """Transcrit avec WhisperX + alignement mot par mot."""
    acquire_gpu_lock()
    free_gpu_for_task(min_free_mib=6000, timeout=60)  # purge un Ollama résident avant WhisperX
    import whisperx, torch

    device = hw.device()  # « cuda » couvre CUDA et ROCm/HIP
    print(f"\n📝 Passe 1c — Transcription WhisperX ({WHISPER_MODEL}) [{device}]...")

    t0 = time.time()
    model = whisperx.load_model(WHISPER_MODEL, device,
                                compute_type=hw.whisper_compute_type(), language=source_lang)
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
        import torch; torch.cuda.empty_cache()

    segments = [
        DubSegment(index=i+1, start=s["start"], end=s["end"],
                   text=s["text"].strip(), words=s.get("words", []))
        for i, s in enumerate(result["segments"])
    ]
    dur = segments[-1].end if segments else 0
    print(f"   ✅ {len(segments)} segments ({dur/60:.1f} min)")
    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 2 : DIARISATION (IDENTIFICATION DES LOCUTEURS)
# ═══════════════════════════════════════════════════════════════════════════════

def diarize_speakers(audio_path: str, segments: list[DubSegment],
                     hf_token: str, num_speakers: Optional[int] = None) -> list[DubSegment]:
    """Identifie qui parle quand avec pyannote via WhisperX."""
    import whisperx, torch

    device = hw.device()  # « cuda » couvre CUDA et ROCm/HIP
    print(f"\n👥 Passe 2 — Diarisation des locuteurs...")

    if not hf_token:
        print("   ⚠️  HF_TOKEN manquant — pyannote requiert un token HuggingFace")
        print("   → https://huggingface.co/settings/tokens")
        print("   → export HF_TOKEN=hf_...")
        print("   ℹ️  Tous les segments assignés à SPEAKER_00")
        return segments

    t0 = time.time()
    from whisperx.diarize import DiarizationPipeline
    diarize_model = DiarizationPipeline(
        token=hf_token, device=device
    )

    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers

    audio = whisperx.load_audio(audio_path)
    diarize_result = diarize_model(audio, **kwargs)

    # Associer les locuteurs aux segments WhisperX
    # On reconstruit le format attendu par assign_word_speakers
    whisperx_segments = [
        {"start": s.start, "end": s.end, "text": s.text, "words": s.words}
        for s in segments
    ]
    result = whisperx.assign_word_speakers(diarize_result, {"segments": whisperx_segments})

    for i, seg_data in enumerate(result["segments"]):
        if i < len(segments):
            segments[i].speaker = seg_data.get("speaker") or "SPEAKER_00"

    # ── Consolider les locuteurs si on en a trop ────────────────────────────
    # assign_word_speakers peut créer plus de labels que demandé.
    # On garde les N locuteurs les plus présents (par durée totale),
    # et on réattribue chaque segment excédentaire au locuteur principal
    # le plus proche temporellement.
    if num_speakers:
        unique_speakers = {}
        for s in segments:
            unique_speakers.setdefault(s.speaker, 0.0)
            unique_speakers[s.speaker] += s.duration

        if len(unique_speakers) > num_speakers:
            # Garder les top-N par durée
            ranked = sorted(unique_speakers.items(), key=lambda x: x[1], reverse=True)
            keep = {spk for spk, _ in ranked[:num_speakers]}
            extra = {spk for spk in unique_speakers if spk not in keep}

            print(f"   🔧 Consolidation : {len(unique_speakers)} labels → {num_speakers} "
                  f"(réattribution de {', '.join(sorted(extra))})")

            # Pour chaque segment excédentaire, trouver le locuteur "keep"
            # le plus proche (par distance temporelle au segment le plus proche)
            keep_segments = {}  # spk → [(mid_time, ...)]
            for s in segments:
                if s.speaker in keep:
                    keep_segments.setdefault(s.speaker, []).append(
                        (s.start + s.end) / 2)

            for s in segments:
                if s.speaker in extra:
                    mid = (s.start + s.end) / 2
                    best_spk = None
                    best_dist = float('inf')
                    for spk, mids in keep_segments.items():
                        dist = min(abs(mid - m) for m in mids)
                        if dist < best_dist:
                            best_dist = dist
                            best_spk = spk
                    s.speaker = best_spk or ranked[0][0]

    # Stats
    speakers = {}
    for s in segments:
        speakers.setdefault(s.speaker, {"count": 0, "dur": 0.0})
        speakers[s.speaker]["count"] += 1
        speakers[s.speaker]["dur"] += s.duration

    del diarize_model; gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"   ✅ {len(speakers)} locuteur(s) détecté(s) ({time.time()-t0:.1f}s)")
    for spk, info in sorted(speakers.items()):
        print(f"      {spk} : {info['count']} segments, {info['dur']:.1f}s")

    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 3 : SÉPARATION DE SOURCES (DEMUCS)
# ═══════════════════════════════════════════════════════════════════════════════

# Seuil au-delà duquel on découpe en chunks (en secondes).
# 20 min = ~540 Mo de WAV stéréo 44.1kHz → Demucs utilise ~8 Go RAM par chunk.
DEMUCS_CHUNK_DURATION = 20 * 60   # 20 minutes
DEMUCS_CHUNK_OVERLAP  = 10        # 10 secondes de chevauchement pour crossfade


def _get_audio_duration(audio_path: str) -> float:
    """Retourne la durée en secondes via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        audio_path
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        return float(r.stdout.strip())
    return 0.0


def _split_audio_chunk(audio_path: str, start: float, duration: float,
                       output_path: str) -> str:
    """Extrait un morceau d'audio avec ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", audio_path,
        "-t", str(duration),
        "-c:a", "pcm_s16le",
        output_path
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"   ❌ ffmpeg split erreur : {r.stderr[-300:]}")
        return ""
    return output_path


def _run_demucs_single(audio_path: str, out_dir: str) -> tuple[str, str]:
    """Lance Demucs sur un fichier audio et retourne (vocals, no_vocals)."""
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems=vocals",
        "--segment", "6",
        "--overlap", "0.25",
        "-o", out_dir,
        "--filename", "{stem}.{ext}",
        audio_path
    ]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"   ❌ Demucs erreur :\n{r.stderr[-600:]}")
        return "", ""

    model_dir = os.path.join(out_dir, "htdemucs")
    vocals = os.path.join(model_dir, "vocals.wav")
    no_vocals = os.path.join(model_dir, "no_vocals.wav")

    if not os.path.exists(vocals):
        for root, dirs, files in os.walk(out_dir):
            for f in files:
                if "vocal" in f.lower() and f.endswith(".wav"):
                    if "no_vocal" not in f.lower() and "no-vocal" not in f.lower():
                        vocals = os.path.join(root, f)
                    else:
                        no_vocals = os.path.join(root, f)

    return vocals, no_vocals


def _crossfade_concat(chunk_paths: list[str], overlap_sec: float,
                      output_path: str, sample_rate: int) -> str:
    """
    Concatène des fichiers WAV avec crossfade linéaire dans les zones
    de chevauchement. Économise la RAM en traitant chunk par chunk.
    """
    import soundfile as sf
    import numpy as np

    overlap_samples = int(overlap_sec * sample_rate)

    if len(chunk_paths) == 1:
        shutil.copy2(chunk_paths[0], output_path)
        return output_path

    # Lire le premier chunk
    current, sr = sf.read(chunk_paths[0], dtype="float32")
    if sr != sample_rate:
        print(f"   ⚠️  Sample rate inattendu : {sr} vs {sample_rate}")

    for i in range(1, len(chunk_paths)):
        next_chunk, _ = sf.read(chunk_paths[i], dtype="float32")

        if overlap_samples > 0 and overlap_samples < len(current) and overlap_samples < len(next_chunk):
            # Zone de crossfade
            fade_out = np.linspace(1.0, 0.0, overlap_samples, dtype=np.float32)
            fade_in  = np.linspace(0.0, 1.0, overlap_samples, dtype=np.float32)

            # Adapter pour stéréo (2D) ou mono (1D)
            if current.ndim == 2:
                fade_out = fade_out[:, np.newaxis]
                fade_in  = fade_in[:, np.newaxis]

            # Blend de la zone d'overlap
            blended = (current[-overlap_samples:] * fade_out +
                       next_chunk[:overlap_samples] * fade_in)

            # Assemblage : current sans overlap + blend + next sans overlap
            current = np.concatenate([
                current[:-overlap_samples],
                blended,
                next_chunk[overlap_samples:]
            ], axis=0)
        else:
            # Pas d'overlap possible → simple concaténation
            current = np.concatenate([current, next_chunk], axis=0)

        del next_chunk
        gc.collect()

    sf.write(output_path, current, sample_rate, subtype="PCM_16")
    del current
    gc.collect()
    return output_path


def separate_sources(audio_hq_path: str, work_dir: str) -> tuple[str, str]:
    """
    Sépare voix et fond sonore avec Demucs (Meta).
    Pour les audios > 20 min, découpe en chunks avec crossfade
    pour éviter les dépassements de RAM.
    Retourne (vocals_path, background_path).
    """
    print(f"\n🎛️  Passe 3 — Séparation de sources (Demucs)...")
    t0 = time.time()

    duration = _get_audio_duration(audio_hq_path)
    duration_min = duration / 60

    out_dir = os.path.join(work_dir, "demucs_out")
    os.makedirs(out_dir, exist_ok=True)

    # Résultats finaux
    final_vocals = os.path.join(out_dir, "vocals.wav")
    final_no_vocals = os.path.join(out_dir, "no_vocals.wav")

    # Sauter si déjà traité
    if (os.path.exists(final_vocals) and os.path.getsize(final_vocals) > 1000
            and os.path.exists(final_no_vocals) and os.path.getsize(final_no_vocals) > 1000):
        mv = os.path.getsize(final_vocals) / (1024*1024)
        mb = os.path.getsize(final_no_vocals) / (1024*1024)
        print(f"   ⏩ Déjà séparé : vocals.wav ({mv:.1f} Mo) + no_vocals.wav ({mb:.1f} Mo)")
        return final_vocals, final_no_vocals

    # ── Mode direct : audio court ───────────────────────────────────────────
    if duration <= DEMUCS_CHUNK_DURATION + 60:  # marge de 1 min
        print(f"   📏 Durée : {duration_min:.1f} min → traitement direct")

        vocals, no_vocals = _run_demucs_single(audio_hq_path, out_dir)
        if not vocals or not os.path.exists(vocals):
            print(f"   ❌ Fichiers Demucs introuvables dans {out_dir}")
            sys.exit(1)

        mv = os.path.getsize(vocals) / (1024*1024)
        mb = os.path.getsize(no_vocals) / (1024*1024)
        print(f"   ✅ vocals.wav ({mv:.1f} Mo) + no_vocals.wav ({mb:.1f} Mo)  [{time.time()-t0:.0f}s]")
        return vocals, no_vocals

    # ── Mode chunked : audio long ───────────────────────────────────────────
    step = DEMUCS_CHUNK_DURATION - DEMUCS_CHUNK_OVERLAP
    n_chunks = math.ceil((duration - DEMUCS_CHUNK_OVERLAP) / step)
    print(f"   📏 Durée : {duration_min:.1f} min → découpage en {n_chunks} chunks "
          f"de ~{DEMUCS_CHUNK_DURATION//60} min (overlap {DEMUCS_CHUNK_OVERLAP}s)")

    chunk_dir = os.path.join(work_dir, "demucs_chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    vocal_chunks = []
    bg_chunks = []

    for ci in range(n_chunks):
        chunk_start = ci * step
        chunk_dur = min(DEMUCS_CHUNK_DURATION, duration - chunk_start)

        chunk_audio = os.path.join(chunk_dir, f"chunk_{ci:03d}.wav")
        chunk_out = os.path.join(chunk_dir, f"chunk_{ci:03d}_demucs")
        os.makedirs(chunk_out, exist_ok=True)

        start_m, start_s = divmod(int(chunk_start), 60)
        end_m, end_s = divmod(int(chunk_start + chunk_dur), 60)
        print(f"\n   🔹 Chunk {ci+1}/{n_chunks} : {start_m}:{start_s:02d} → "
              f"{end_m}:{end_s:02d} ({chunk_dur/60:.1f} min)")

        # Découper
        _split_audio_chunk(audio_hq_path, chunk_start, chunk_dur, chunk_audio)

        # Demucs sur le chunk
        v, bg = _run_demucs_single(chunk_audio, chunk_out)
        if not v or not os.path.exists(v):
            print(f"   ❌ Demucs a échoué sur le chunk {ci+1}")
            sys.exit(1)

        vocal_chunks.append(v)
        bg_chunks.append(bg)

        # Nettoyer le chunk source pour économiser du disque
        if os.path.exists(chunk_audio):
            os.remove(chunk_audio)

        # Forcer le GC entre les chunks
        gc.collect()

    # ── Crossfade et concaténation ──────────────────────────────────────────
    print(f"\n   🔀 Assemblage des {n_chunks} chunks avec crossfade de {DEMUCS_CHUNK_OVERLAP}s...")

    _crossfade_concat(vocal_chunks, DEMUCS_CHUNK_OVERLAP, final_vocals, SAMPLE_RATE)
    _crossfade_concat(bg_chunks, DEMUCS_CHUNK_OVERLAP, final_no_vocals, SAMPLE_RATE)

    # Nettoyage des chunks Demucs
    shutil.rmtree(chunk_dir, ignore_errors=True)

    mv = os.path.getsize(final_vocals) / (1024*1024)
    mb = os.path.getsize(final_no_vocals) / (1024*1024)
    print(f"   ✅ vocals.wav ({mv:.1f} Mo) + no_vocals.wav ({mb:.1f} Mo)  [{time.time()-t0:.0f}s]")
    return final_vocals, final_no_vocals


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 3b : EXTRACTION DES ÉCHANTILLONS PAR LOCUTEUR
# ═══════════════════════════════════════════════════════════════════════════════

def _estimate_gender(audio_mono: 'np.ndarray', sr: int,
                     threshold_female: float = 165.0,
                     threshold_male: float = 155.0) -> tuple[str, float]:
    """
    Estime le genre du locuteur par analyse de la fréquence fondamentale (F0).

    Utilise l'autocorrélation pour estimer le pitch médian :
      - F0 médian > threshold_female Hz → "female"
      - F0 médian < threshold_male Hz   → "male"
      - entre les deux                  → "unknown"

    Retourne (genre, f0_median_hz).
    """
    import numpy as np

    # Travailler sur un extrait (max 10s) pour la vitesse
    max_samples = int(10 * sr)
    audio = audio_mono[:max_samples] if len(audio_mono) > max_samples else audio_mono

    # Paramètres
    frame_size = int(0.04 * sr)      # fenêtre 40ms
    hop = int(0.02 * sr)             # saut 20ms
    f0_min, f0_max = 70, 400         # plage de recherche (Hz)
    lag_min = sr // f0_max           # ~110 samples @ 44.1kHz
    lag_max = sr // f0_min           # ~630 samples @ 44.1kHz

    f0_values = []

    for start in range(0, len(audio) - frame_size, hop):
        frame = audio[start:start + frame_size]

        # Vérifier que la frame contient du signal (pas du silence)
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < 0.005:
            continue

        # Autocorrélation normalisée
        frame = frame - np.mean(frame)
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr) // 2:]  # partie positive

        if corr[0] == 0:
            continue
        corr = corr / corr[0]  # normaliser

        # Chercher le premier pic dans la plage [lag_min, lag_max]
        search = corr[lag_min:min(lag_max, len(corr))]
        if len(search) < 2:
            continue

        # Trouver le pic maximal
        peak_idx = np.argmax(search)
        peak_val = search[peak_idx]

        # Seuil de confiance : le pic doit être suffisamment prononcé
        if peak_val > 0.3:
            lag = lag_min + peak_idx
            f0 = sr / lag
            f0_values.append(f0)

    if not f0_values:
        return "unknown", 0.0

    f0_median = float(np.median(f0_values))

    if f0_median > threshold_female:
        return "female", f0_median
    elif f0_median < threshold_male:
        return "male", f0_median
    else:
        return "unknown", f0_median


def _read_vocal_segment(vocals_path: str, start_sec: float, end_sec: float,
                        sr: int, n_frames: int) -> 'np.ndarray':
    """Lit un segment du fichier vocal sans charger tout le fichier en RAM."""
    import soundfile as sf
    import numpy as np

    start_sample = int(start_sec * sr)
    end_sample = min(int(end_sec * sr), n_frames)
    if start_sample >= end_sample:
        return np.array([], dtype=np.float64)

    data, _ = sf.read(vocals_path, start=start_sample, stop=end_sample, always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data


def extract_speaker_samples(
    segments: list[DubSegment], vocals_path: str, work_dir: str
) -> dict[str, SpeakerProfile]:
    """
    Extrait des échantillons audio propres par locuteur depuis le stem vocal.
    Sélectionne les segments les plus longs pour chaque locuteur.
    Lecture par segment (sf.read start/stop) pour éviter de charger tout le fichier.
    """
    import soundfile as sf
    import numpy as np

    print(f"\n🎤 Passe 3b — Extraction des échantillons vocaux...")

    samples_dir = os.path.join(work_dir, "speaker_samples")
    os.makedirs(samples_dir, exist_ok=True)

    # Obtenir les métadonnées sans charger le fichier
    info = sf.info(vocals_path)
    sr = info.samplerate
    n_frames = info.frames

    # Grouper les segments par locuteur
    speaker_segs: dict[str, list[DubSegment]] = {}
    for seg in segments:
        speaker_segs.setdefault(seg.speaker, []).append(seg)

    profiles: dict[str, SpeakerProfile] = {}

    for spk_id, segs in speaker_segs.items():
        # Trier par durée décroissante (meilleurs échantillons en premier)
        segs_sorted = sorted(segs, key=lambda s: s.duration, reverse=True)

        total_dur = sum(s.duration for s in segs)
        profile = SpeakerProfile(
            speaker_id=spk_id,
            total_duration=total_dur,
            segment_count=len(segs)
        )

        # Concaténer les segments les plus longs jusqu'à MAX_SPEAKER_SAMPLE_SEC
        sample_chunks = []
        sample_texts = []
        accumulated = 0.0

        for seg in segs_sorted:
            if accumulated >= MAX_SPEAKER_SAMPLE_SEC:
                break
            if seg.duration < 0.5:  # ignorer les segments trop courts
                continue

            chunk = _read_vocal_segment(vocals_path, seg.start, seg.end, sr, n_frames)
            if len(chunk) > 0:
                sample_chunks.append(chunk)
                sample_texts.append(seg.text)
                accumulated += seg.duration

        if not sample_chunks or accumulated < MIN_SPEAKER_SAMPLE_SEC:
            print(f"   ⚠️  {spk_id} : seulement {accumulated:.1f}s — qualité de clonage réduite")

        # Sauvegarder l'échantillon concaténé
        if sample_chunks:
            # Ajouter de petits silences entre les chunks (100ms)
            silence = np.zeros(int(0.1 * sr))
            full_sample = np.concatenate(
                [x for chunk in sample_chunks for x in [chunk, silence]][:-1]
            )

            sample_path = os.path.join(samples_dir, f"{spk_id}.wav")
            sf.write(sample_path, full_sample, sr)

            profile.sample_path = sample_path
            profile.sample_text = " ".join(sample_texts)

        # Sauvegarder aussi des clips INDIVIDUELS (pour clonage vocal)
        # Chaque clip a son propre fichier + texte exact → pas de fuite ref_text
        ref_clips = []
        for ci, seg in enumerate(segs_sorted):
            if len(ref_clips) >= 10:  # max 10 clips de référence
                break
            if seg.duration < 3.0 or seg.duration > 15.0:
                continue  # ni trop court ni trop long

            clip = _read_vocal_segment(vocals_path, seg.start, seg.end, sr, n_frames)
            if len(clip) == 0:
                continue

            clip_path = os.path.join(samples_dir, f"{spk_id}_ref{ci:02d}.wav")
            sf.write(clip_path, clip, sr)
            ref_clips.append((clip_path, seg.text.strip()))

        profile.ref_clips = ref_clips

        # ── Estimation du genre par analyse F0 (fréquence fondamentale) ──
        if sample_chunks:
            gender, f0_median = _estimate_gender(full_sample, sr)
            profile.gender = gender
            profile.f0_median = f0_median
            gender_icon = "♀️" if gender == "female" else "♂️" if gender == "male" else "❓"
            print(f"   🎤 {spk_id} {gender_icon} : {accumulated:.1f}s échantillon, "
                  f"{len(ref_clips)} clips ref "
                  f"({len(segs)} segments, {total_dur:.0f}s total)"
                  f" [F0={f0_median:.0f}Hz]")

        profiles[spk_id] = profile

    return profiles


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 4 : ANALYSE + TRADUCTION (CLAUDE)
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_content(segments, client, src_lang, tgt_lang, context=""):
    """Analyse le contenu pour guider la traduction (identique à traduire.py)."""
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

    prompt = f"""Tu es un directeur de doublage professionnel {src_n} → {tgt_n}.
{ctx}
Analyse cette transcription et fournis en JSON strict :
- "summary": résumé 3-5 phrases (en {tgt_n})
- "glossary": {{"terme_{src_lang}": "traduction_{tgt_lang}"}} pour termes techniques/noms/expressions
- "speakers_description": description de chaque locuteur identifié (voix, rôle, registre)
- "tone": registre global
- "domain": domaine principal
- "speaking_rate": rythme de parole estimé (lent/normal/rapide)

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


def translate_for_dubbing(segments, analysis, client, src_lang, tgt_lang, context=""):
    """
    Traduction optimisée pour le doublage (style oral, concision).
    Contrairement au sous-titrage, on privilégie le naturel parlé.
    """
    print(f"\n🌍 Passe 4b — Traduction pour doublage {src_lang}→{tgt_lang}...")

    src_n, tgt_n = lang_name(src_lang), lang_name(tgt_lang)
    user_ctx = f"\nINSTRUCTIONS : {context}\n" if context else ""
    tgt_cps = CPS_TARGETS.get(tgt_lang, 14.0)

    # Règles spécifiques selon la langue cible (registre de politesse)
    lang_specific_rule = ""
    if tgt_lang == "fr":
        lang_specific_rule = "\n4. Tutoiement/vouvoiement cohérent selon le contexte et le registre"
    elif tgt_lang == "de":
        lang_specific_rule = "\n4. Du/Sie cohérent selon le contexte et le registre"
    elif tgt_lang == "es":
        lang_specific_rule = "\n4. Tú/Usted cohérent selon le contexte et le registre"
    elif tgt_lang == "ja":
        lang_specific_rule = "\n4. Niveau de politesse (敬語) cohérent selon le contexte"
    elif tgt_lang == "ko":
        lang_specific_rule = "\n4. Niveau de politesse (존댓말/반말) cohérent selon le contexte"
    elif tgt_lang == "pt":
        lang_specific_rule = "\n4. Tu/Você cohérent selon le contexte et le registre"

    system = f"""Tu es un directeur de doublage professionnel {src_n} → {tgt_n}.

CONTEXTE : {analysis.get('summary', '')}
Domaine : {analysis.get('domain', '')} | Ton : {analysis.get('tone', '')}
Locuteurs : {analysis.get('speakers_description', '')}
{user_ctx}
GLOSSAIRE :
{json.dumps(analysis.get('glossary', {}), ensure_ascii=False, indent=2)}

RÈGLES SPÉCIFIQUES AU DOUBLAGE :
1. Traduction ORALE et NATURELLE — c'est du texte qui sera PRONONCÉ à haute voix
2. LONGUEUR : chaque segment indique une cible chiffrée et un minimum.
   Tu DOIS rester entre min et cible+18%. JAMAIS en dessous du min.
   Si le texte source est dense, tu gardes TOUTES les informations.
   Ne jamais résumer, condenser ou supprimer un détail (énumération,
   exemple, qualificatif, subordonnée) pour faire plus court — c'est
   destructeur de sens, et la fenêtre temporelle est calculée pour
   accueillir le contenu complet à débit naturel.
3. Évite les constructions littéraires ou écrites (relatives longues, passifs, inversions){lang_specific_rule}
5. Adapte les expressions idiomatiques en équivalents naturels en {tgt_n}
6. Noms propres inchangés sauf conventions établies
7. Respecte le glossaire strictement
8. Correspondance 1:1 des numéros — ne JAMAIS fusionner ni diviser
9. Hésitations et faux départs : ne supprime que les véritables disfluences
   (« euh », « hmm », « je veux dire » répété). Ne supprime PAS les
   reformulations qui portent du sens, ni les répétitions stylistiques.
10. Garde le REGISTRE de chaque locuteur (formel, familier, technique...)
11. Préfère les mots courts mais ne sacrifie JAMAIS un détail informatif
    (chiffre, exemple, qualificatif) pour gagner des caractères.

CHANTS, PRIÈRES ET PASSAGES RITUELS :
Si un segment est un chant, une prière, un mantra, une récitation ou un texte
liturgique (pali, sanskrit, latin liturgique, arabe coranique, hébreu biblique,
etc.), tu DOIS le recopier TEL QUEL sans le traduire ni le paraphraser.
Ces passages sont des performances vocales, pas du discours à traduire.
ATTENTION : ceci ne s'applique PAS aux mots ou expressions empruntés courants
(anglicismes en français, germanismes, etc.) ni au code-switching ordinaire entre
langues vivantes — ceux-là doivent être traduits normalement en {tgt_n}.

Appelle l'outil submit_texts avec un item PAR SEGMENT à traduire.
Le champ text contient UNIQUEMENT la traduction en {tgt_n} (ou le texte
original si chant/prière). Pas de commentaire, pas de préfixe."""

    n_chunks = (len(segments) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for ci in range(n_chunks):
        s, e = ci * CHUNK_SIZE, min((ci + 1) * CHUNK_SIZE, len(segments))

        if all(seg.text_tgt for seg in segments[s:e]):
            print(f"   📦 Chunk {ci+1}/{n_chunks} — déjà traduit"); continue

        print(f"   📦 Chunk {ci+1}/{n_chunks} (seg {s+1}–{e})...")
        parts = []

        # Contexte arrière
        cb = max(0, s - CHUNK_OVERLAP)
        if cb < s:
            parts.append("=== CONTEXTE PRÉCÉDENT (NE PAS retraduire) ===")
            for seg in segments[cb:s]:
                if seg.text_tgt:
                    parts += [f"[{seg.index}] {src_lang.upper()}: {seg.text}",
                              f"[{seg.index}] {tgt_lang.upper()}: {seg.text_tgt}"]
            parts.append("")

        # À traduire — cible et min de longueur calculés depuis la durée réelle
        parts.append("=== À TRADUIRE ===")
        for seg in segments[s:e]:
            target_chars = max(int(seg.duration * tgt_cps), 8)
            min_chars = max(int(target_chars * TRANSLATION_LEN_MIN_RATIO), 5)
            parts.append(
                f"[{seg.index}] ({seg.speaker}, durée {seg.duration:.1f}s, "
                f"cible ≈{target_chars} car., min {min_chars}) {seg.text}"
            )

        # Contexte avant
        cf = min(len(segments), e + CHUNK_OVERLAP)
        if e < cf:
            parts.append("\n=== CONTEXTE SUIVANT (NE PAS traduire) ===")
            parts += [f"[{seg.index}] {seg.text}" for seg in segments[e:cf]]

        translations = _claude_submit_texts(client, "\n".join(parts), system=system)
        _apply_translations(translations, segments, s, e)

        done = sum(1 for seg in segments[s:e] if seg.text_tgt)
        if done < e - s:
            print(f"   ⚠️  {done}/{e-s} traduits — relance...")
            _retry_missing(segments, s, e, system, client, src_lang, tgt_lang)

        print(f"   ✅ {sum(1 for seg in segments[s:e] if seg.text_tgt)}/{e-s}")

    total = sum(1 for s in segments if s.text_tgt)
    print(f"\n   🌍 Traduit : {total}/{len(segments)}")
    return segments


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

_STOPWORDS_FUITE = {
    "le", "la", "les", "un", "une", "des", "de", "du", "d'", "l'",
    "et", "ou", "à", "au", "aux", "en", "est", "c'est", "ça",
    "the", "a", "an", "of", "and", "or", "to", "is", "it",
}

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
        # Cas long : ≥5 mots de chaque côté avec moins de 2 mots en commun
        if len(mots_orig) >= 5 and len(mots_new) >= 5:
            if len(mots_orig & mots_new) < 2:
                return True
        # Cas court : le nouveau texte fait ≤3 mots et ne partage AUCUN mot
        # significatif avec l'original — typiquement « Fluide », « OK »,
        # « Pas de correction » renvoyé par le réviseur à la place d'un
        # texte corrigé (passe 5b notamment).
        if len(mots_orig) >= 3 and 0 < len(mots_new) <= 3:
            sig_orig = mots_orig - _STOPWORDS_FUITE
            sig_new = mots_new - _STOPWORDS_FUITE
            if sig_orig and sig_new and not (sig_orig & sig_new):
                return True
    return False


def _strip_claude_artifacts(txt: str) -> str:
    """Nettoie les artefacts courants de la sortie Claude."""
    # Notation fléchée du reviewer : "ancien" → "nouveau"
    if '→' in txt:
        txt = txt.split('→', 1)[-1].strip().strip('""«»\u201c\u201d').strip()
    # Préfixes méta-narratifs parrotés depuis le format d'entrée des prompts
    # de relecture (AVANT:/APRÈS:/AFTER:/BEFORE:). Jamais un vrai texte cible.
    txt = re.sub(r'^(?:APR[ÈE]S|AVANT|AFTER|BEFORE)\s*:\s*', '', txt,
                 flags=re.IGNORECASE).strip()
    # Préfixe de langue (FR:, EN:, etc.)
    txt = re.sub(r'^[A-Z]{2}:\s*', '', txt).strip()
    # Métadonnées entre parenthèses en début de ligne
    txt = re.sub(r'^\([^)]*\)\s*', '', txt).strip()
    return txt


def _apply_translations(translations: dict[int, str], segments, s, e):
    """Affecte les traductions { id → text } reçues du LLM aux segments du chunk."""
    for idx, txt in translations.items():
        for seg in segments[s:e]:
            if seg.index == idx:
                seg.text_tgt = txt
                break


def _retry_missing(segments, s, e, system, client, source_lang="en", target_lang="fr"):
    # Retry 1 : contexte minimal
    missing = [seg for seg in segments[s:e] if not seg.text_tgt]
    if not missing: return
    parts = ["Segments manquants à traduire :\n"]
    for seg in missing:
        ctx = [x for x in segments[max(0, seg.index-4):seg.index-1] if x.text_tgt]
        if ctx: parts.append(f"  (contexte: [{ctx[-1].index}] {ctx[-1].text_tgt})")
        parts.append(f"[{seg.index}] {seg.text}")
    translations = _claude_submit_texts(client, "\n".join(parts), system=system)
    _apply_translations(translations, segments, s, e)

    # Retry 2 : contexte bilingue étendu (3 segments avant/après chaque manquant)
    missing2 = [seg for seg in segments[s:e] if not seg.text_tgt]
    if not missing2: return
    print(f"   ⚠️  Retry 2 avec contexte bilingue étendu ({len(missing2)} manquants)...")
    parts2 = ["Segments toujours manquants — contexte bilingue étendu :\n"]
    for seg in missing2:
        # 3 segments traduits avant
        before = [x for x in segments[max(0, seg.index-4):seg.index] if x.text_tgt][-3:]
        for bseg in before:
            parts2.append(f"  [{bseg.index}] {source_lang.upper()}: {bseg.text}")
            parts2.append(f"  [{bseg.index}] {target_lang.upper()}: {bseg.text_tgt}")
        parts2.append(f"[{seg.index}] {seg.text}")
        # 3 segments traduits après
        after = [x for x in segments[seg.index+1:min(len(segments), seg.index+4)] if x.text_tgt][:3]
        for aseg in after:
            parts2.append(f"  [{aseg.index}] {source_lang.upper()}: {aseg.text}")
            parts2.append(f"  [{aseg.index}] {target_lang.upper()}: {aseg.text_tgt}")
        parts2.append("")
    translations2 = _claude_submit_texts(client, "\n".join(parts2), system=system)
    _apply_translations(translations2, segments, s, e)


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 4c : RELECTURE DE LA TRADUCTION (CLAUDE)
# ═══════════════════════════════════════════════════════════════════════════════

def review_dubbing_translation(segments, analysis, client,
                               src_lang, tgt_lang, context=""):
    """
    Relecture de la traduction avant isochronie.
    Vérifie : fidélité, naturel oral, glossaire, durée, politesse, fluidité.
    Prompt structuré avec 6 critères hiérarchisés et estimation CPS.
    """
    print(f"\n📖 Passe 4c — Relecture de la traduction...")

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

    # Seuil CPS pour le doublage (plus souple qu'en sous-titrage)
    CPS_WARNING = 16
    # Seuil bas : en dessous, la traduction est probablement un sur-résumé
    # (du contenu source a été perdu). On vise CPS naturel pour la langue cible.
    tgt_cps = CPS_TARGETS.get(tgt_lang, 14.0)
    CPS_TOO_SHORT = tgt_cps * 0.55

    for wi in range(n_win):
        s = wi * (WIN - OVL)
        e = min(s + WIN, len(segments))
        print(f"   🔎 Fenêtre {wi+1}/{n_win} (seg {s+1}–{e})...")

        pairs = []
        for seg in segments[s:e]:
            seg_dur = seg.duration
            tgt_len = len(seg.text_tgt) if seg.text_tgt else 0
            cps = tgt_len / seg_dur if seg_dur > 0.1 and tgt_len > 0 else 0
            if cps > CPS_WARNING:
                cps_marker = " ⚠️ TROP LONG"
            elif seg_dur >= 1.5 and 0 < cps < CPS_TOO_SHORT:
                # Seulement pour des segments substantiels (>1.5s) — les segments
                # très courts ont naturellement un CPS bas (interjections).
                cps_marker = " ⛔ TROP COURT (sur-résumé probable)"
            else:
                cps_marker = ""
            pairs += [
                f"[{seg.index}] {src_lang.upper()}: {seg.text} ({seg_dur:.1f}s)",
                f"[{seg.index}] {tgt_lang.upper()}: {seg.text_tgt} (~{cps:.0f} CPS){cps_marker}",
                ""
            ]

        prompt = f"""Réviseur professionnel de doublage en {tgt_n}.

Le texte traduit sera PRONONCÉ à haute voix (doublage audio), pas lu en sous-titres.

GLOSSAIRE : {json.dumps(glossary, ensure_ascii=False, indent=2)}
{ctx_note}
Vérifie ces critères par ordre de priorité :
1. FIDÉLITÉ : pas de contresens, chiffres et noms propres corrects, contenu source intégralement préservé
2. NATUREL ORAL : prononciation fluide à voix haute, pas de tournures écrites (relatives longues, passifs, inversions)
3. GLOSSAIRE : termes du glossaire strictement respectés
4. DURÉE TROP LONGUE (⚠️) : segment > {CPS_WARNING} car/s → reformuler plus court SANS perdre d'information
5. DURÉE TROP COURTE (⛔) : segment < {CPS_TOO_SHORT:.1f} car/s sur fenêtre ≥ 1.5s → traduction très probablement sur-résumée.
   Récupère depuis le source les détails manquants (énumérations, exemples, qualificatifs,
   subordonnées) pour étoffer la traduction. Vise un CPS proche de {tgt_cps:.1f}. Aucune
   information du source ne doit être absente sans raison.
{lang_politeness}
7. FLUIDITÉ DE PRONONCIATION : pas de mots difficiles enchaînés, pas de suites de consonnes imprononçables
8. CHANTS/PRIÈRES : si un segment est un chant, une prière, un mantra ou un
   texte liturgique (pali, sanskrit, latin liturgique, arabe coranique, etc.),
   il doit être CONSERVÉ TEL QUEL — ne jamais le modifier ni le traduire.
   (Ceci ne concerne PAS les emprunts courants entre langues vivantes.)

Appelle submit_texts avec un item UNIQUEMENT pour les segments à corriger.
Le champ text contient la traduction corrigée — pas de commentaire, pas de
préfixe. Si aucun segment ne nécessite de correction, items=[].

{chr(10).join(pairs)}"""

        corrected = _claude_submit_texts(client, prompt)

        for idx, new in corrected.items():
            for seg in segments:
                if seg.index == idx and seg.text_tgt != new:
                    if _est_fuite_prompt(new, seg.text_tgt):
                        print(f"   ⚠️  Fuite de prompt détectée seg [{idx}], ignoré : {new[:60]}…")
                        break
                    seg.text_tgt = new
                    fixes += 1
                    break
        if fixes:
            print(f"   ✏️  corrections en cours...")

    print(f"   📖 {fixes} corrections totales")
    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 4d : COHÉRENCE GLOBALE (CLAUDE)
# ═══════════════════════════════════════════════════════════════════════════════

def check_dubbing_consistency(segments, analysis, client,
                              source_lang="en", target_lang="fr"):
    """Passe de cohérence globale : terminologie, registre de politesse, ton."""
    print("\n🔗 Passe 4d — Cohérence globale...")
    tgt_name = lang_name(target_lang)

    glossary = analysis.get("glossary", {})
    tone = analysis.get("tone", "")
    domain = analysis.get("domain", "")

    # Échantillon : max 500 segments pour vidéos longues
    sample = segments if len(segments) <= 500 else (
        segments[:170] + segments[len(segments)//2 - 80:len(segments)//2 + 80] + segments[-170:]
    )

    # Fenêtrage à 200 segments par appel
    WIN = 200
    fixes = 0
    for wi in range(0, len(sample), WIN):
        batch = sample[wi:wi + WIN]
        lines = [f"[{seg.index}] {seg.text_tgt}" for seg in batch if seg.text_tgt]
        if not lines:
            continue

        prompt = f"""Vérificateur de cohérence pour doublage en {tgt_name}.

GLOSSAIRE : {json.dumps(glossary, ensure_ascii=False, indent=2)}
Ton attendu : {tone} | Domaine : {domain}

Vérifie UNIQUEMENT ces 3 points sur l'ensemble des segments ci-dessous :
1. TERMINOLOGIE : un même concept source est-il toujours traduit de la même façon ?
2. REGISTRE DE POLITESSE : le niveau de formalité est-il constant ?
3. TON : le ton ({tone}) est-il maintenu uniformément ?

Ne corrige PAS le style, la grammaire ou la concision — seulement les incohérences ci-dessus.

Appelle submit_texts avec un item UNIQUEMENT pour les segments à corriger.
items=[] si tout est déjà cohérent.

{chr(10).join(lines)}"""

        corrected = _claude_submit_texts(client, prompt)

        for idx, new in corrected.items():
            for seg in segments:
                if seg.index == idx and seg.text_tgt != new:
                    if _est_fuite_prompt(new, seg.text_tgt):
                        print(f"   ⚠️  Fuite de prompt détectée seg [{idx}], ignoré : {new[:60]}…")
                        break
                    seg.text_tgt = new
                    fixes += 1
                    break

    print(f"   🔗 {fixes} corrections de cohérence")
    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 4e : VÉRIFICATION GLOSSAIRE (CLAUDE)
# ═══════════════════════════════════════════════════════════════════════════════

def verify_dubbing_glossary(segments, analysis, client,
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
            lines.append(f"[{seg.index}] {source_lang.upper()}: {seg.text}")
            lines.append(f"[{seg.index}] {target_lang.upper()}: {seg.text_tgt}")
            lines.append(f"  → Le terme « {src_term} » devrait être traduit « {tgt_term} »")
            lines.append("")

        prompt = f"""Correcteur de glossaire pour doublage en {tgt_name}.

Pour chaque segment ci-dessous, le glossaire n'a pas été respecté.
Corrige NATURELLEMENT la traduction pour intégrer le terme correct du glossaire,
sans rendre la phrase artificielle. Le texte sera prononcé à haute voix.

GLOSSAIRE COMPLET : {json.dumps(glossary, ensure_ascii=False, indent=2)}

Appelle submit_texts avec un item par segment corrigé.
Le champ text contient uniquement la traduction corrigée — pas de commentaire.

{chr(10).join(lines)}"""

        corrected = _claude_submit_texts(client, prompt)

        for idx, new in corrected.items():
            for seg in segments:
                if seg.index == idx and seg.text_tgt != new:
                    if _est_fuite_prompt(new, seg.text_tgt):
                        print(f"   ⚠️  Fuite de prompt détectée seg [{idx}], ignoré : {new[:60]}…")
                        break
                    seg.text_tgt = new
                    fixes += 1
                    break

    print(f"   📖 {fixes} corrections glossaire")
    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 5 : ADAPTATION ISOCHRONIQUE (CLAUDE)
# ═══════════════════════════════════════════════════════════════════════════════

def adapt_isochrony(segments: list[DubSegment], client,
                    src_lang: str, tgt_lang: str,
                    lead_in_sec: float = 0.0) -> list[DubSegment]:
    """
    Adapte la longueur de chaque traduction pour qu'elle soit prononçable
    dans ≈ la même durée que l'original. Crucial pour la synchronisation.

    Heuristique : ~14 caractères/seconde en français parlé, ~13 en anglais, etc.
    On estime le nombre de caractères cible et on demande à Claude d'ajuster.
    
    lead_in_sec: temps de lead-in voice-over à soustraire de la durée disponible.
    """
    print(f"\n⏱️  Passe 5 — Adaptation isochronique...")

    tgt_cps = CPS_TARGETS.get(tgt_lang, 14.0)

    if lead_in_sec > 0:
        print(f"   ℹ️  Lead-in voice-over : -{lead_in_sec:.1f}s par segment")

    # Identifier les segments problématiques (>15% de déviation)
    to_adapt = []
    for seg in segments:
        if not seg.text_tgt or seg.duration < 0.5:
            seg.text_adapted = seg.text_tgt
            continue

        # Durée effective = durée originale - lead-in (la voix doublée commence après)
        seg_lead_in = lead_in_sec if seg.is_sentence_start else 0.0
        effective_duration = max(seg.duration - seg_lead_in, 0.5)
        target_chars = int(effective_duration * tgt_cps)
        actual_chars = len(seg.text_tgt)
        ratio = actual_chars / target_chars if target_chars > 0 else 1.0

        if ratio > TRANSLATION_LEN_MAX_RATIO or ratio < TRANSLATION_LEN_MIN_RATIO:
            to_adapt.append((seg, target_chars, ratio))
        else:
            seg.text_adapted = seg.text_tgt  # OK, pas besoin d'adapter

    if not to_adapt:
        print(f"   ✅ Toutes les traductions sont dans la plage temporelle")
        return segments

    n_short = sum(1 for _, _, r in to_adapt if r < TRANSLATION_LEN_MIN_RATIO)
    n_long = sum(1 for _, _, r in to_adapt if r > TRANSLATION_LEN_MAX_RATIO)
    print(f"   📏 {len(to_adapt)}/{len(segments)} segments à adapter "
          f"({n_short} à allonger, {n_long} à raccourcir)")

    src_n = lang_name(src_lang)
    tgt_n = lang_name(tgt_lang)

    # Traiter par batchs de 40
    BATCH = 40
    adapted = 0
    for bi in range(0, len(to_adapt), BATCH):
        batch = to_adapt[bi:bi+BATCH]

        def _qualitative_marker(r: float) -> str:
            # Marqueurs QUALITATIFS — pas de cible chiffrée : éviter que Claude
            # rabote des mots-outils ("des", "soi-disant", "potentiellement") juste
            # pour atteindre un nombre. Mieux vaut un léger débordement absorbé
            # par le TTS speed qu'une phrase mutilée.
            if r >= 2.0:   return "RACCOURCIR (TRÈS LONG)"
            if r >= 1.4:   return "RACCOURCIR (LONG)"
            if r > 1.0:    return "RACCOURCIR (UN PEU LONG)"
            if r >= 0.6:   return "ALLONGER (COURT)"
            return "ALLONGER (TRÈS COURT)"

        lines = []
        for seg, _target_chars, ratio in batch:
            direction = _qualitative_marker(ratio)
            # Pour ALLONGER, on inclut le source : sans lui Claude ne peut pas
            # récupérer le contenu manquant (cause #1 des "phrases manquantes"
            # dans le doublage).
            if ratio < 1.0:
                lines.append(
                    f"[{seg.index}] ({direction})\n"
                    f"  ORIGINAL_{src_n.upper()}: {seg.text}\n"
                    f"  TRADUCTION_ACTUELLE: {seg.text_tgt}"
                )
            else:
                lines.append(
                    f"[{seg.index}] ({direction}) {seg.text_tgt}"
                )

        prompt = f"""Tu es un adaptateur de doublage en {tgt_n}.
Pour chaque segment, reformule le texte selon le marqueur indiqué (TRÈS COURT,
COURT, UN PEU LONG, LONG, TRÈS LONG), SANS changer le sens et SANS casser
la grammaire.

Le texte sera PRONONCÉ à voix haute — il doit rester fluide et naturel à l'oral.

POSITION SUR LA GRAMMAIRE — ABSOLUE :
La grammaire et le sens passent AVANT le respect strict de la durée. Un
débordement de 10-20% est acceptable (le TTS peut accélérer) ; une phrase
mutilée ne l'est pas. Si tu ne peux pas raccourcir sans casser la grammaire
ou perdre une nuance, laisse la phrase un peu trop longue.

Pour les segments ALLONGER : la traduction est trop courte. Compare-la à
l'ORIGINAL et **récupère les détails manquants** (énumérations, exemples,
qualificatifs, subordonnées, reformulations porteuses de sens). Tu N'INVENTES
rien — tu rends seulement ce qui était dans l'original mais perdu pendant
la traduction.

Pour les segments RACCOURCIR : reformule plus concisément sans perdre
d'information. Préfère couper des mots redondants ou simplifier la syntaxe
plutôt que de toucher aux mots ci-dessous.

DOIVENT ÊTRE PRÉSERVÉS (ne JAMAIS supprimer) :
- Articles et déterminants : « le, la, les, un, une, des, du, de la, ce, ces, mon, son, leur… »
  Après une préposition (avec, chez, par, dans, pour, sous, entre, parmi…), l'article reste.
- Mots de liaison : « et, mais, donc, alors, car, parce que, c'est que, après tout… »
- Pronoms : « on, ça, vous, ils, c'est… »
- Adverbes-qualifieurs éditoriaux : « soi-disant, prétendument, potentiellement,
  vraiment, probablement, particulièrement, principalement, éventuellement,
  approximativement ». Ces mots portent le cadrage de l'auteur — les enlever
  change la portée de la phrase.
- Sujet + verbe conjugué : chaque phrase doit en avoir. Pas de fragment nominal.

EXEMPLES À NE PAS REPRODUIRE :
❌ « les gens avec troubles mentaux »          → ✅ « les gens avec des troubles mentaux »
❌ « chez adultes sains »                      → ✅ « chez des adultes sains »
❌ « par patients, organisations »             → ✅ « par les patients, les organisations »
❌ « entre génotypes sérotoninergiques »       → ✅ « entre les génotypes sérotoninergiques »
❌ « ils risquent plus les tueries »           → ✅ « ils ont plus de risques de tuer »
❌ « C'est une idée fausse mortelle »          → ✅ « C'est une idée fausse potentiellement mortelle »
❌ « Faux. » (au lieu d'une phrase complète)   → ✅ « Mais ils ne l'ont pas fait. »
❌ « Un malade. » (fragment nominal)           → ✅ « Seul un fou ferait ça. »
❌ « Tout à fait. » (à la place d'un oui)      → ✅ « Oui, c'est tout à fait correct. »
❌ « médicaments de transition »               → ✅ « soi-disant médicaments de transition »

INTERDIT (résumé) :
- Changer le sens, le ton, ou la portée éditoriale (soi-disant, potentiellement…)
- Inventer du contenu absent de l'original
- Couper des phrases en deux ou fusionner
- Style TÉLÉGRAPHIQUE / fragment nominal
- Supprimer un mot de la liste « DOIVENT ÊTRE PRÉSERVÉS »

PONCTUATION : conserve une ponctuation soignée (virgules de respiration,
points de suspension pour les hésitations, deux-points si présents dans l'original).

Appelle l'outil submit_texts avec un item par segment.
Le champ text contient UNIQUEMENT le texte adapté du segment — pas de
commentaire, pas de préfixe, pas de méta-discussion.

{chr(10).join(lines)}"""

        adapted_map = _claude_submit_texts(client, prompt)

        for idx, txt in adapted_map.items():
            for seg, _, _ in batch:
                if seg.index == idx:
                    if _est_fuite_prompt(txt, seg.text_tgt):
                        print(f"   ⚠️  Fuite de prompt détectée seg [{idx}], ignoré : {txt[:60]}…")
                        break
                    seg.text_adapted = txt
                    adapted += 1
                    break

    # Fallback : les non-adaptés gardent la traduction brute
    for seg in segments:
        if not seg.text_adapted:
            seg.text_adapted = seg.text_tgt

    print(f"   ✅ {adapted} segments adaptés")
    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 5b : RELECTURE FLUIDITÉ POST-ADAPTATION (CLAUDE)
# ═══════════════════════════════════════════════════════════════════════════════

def review_adapted_fluency(segments: list, client,
                           tgt_lang: str) -> list:
    """
    Relecture légère des segments modifiés par l'adaptation isochronique.
    Ne touche QUE les segments où text_adapted ≠ text_tgt (ceux raccourcis/allongés).
    Vérifie : fluidité orale, connecteurs, structure grammaticale, ponctuation.
    Ne modifie PAS la longueur — corrige uniquement les maladresses introduites
    par l'adaptation.
    """
    modified = [seg for seg in segments
                if seg.text_adapted and seg.text_tgt
                and seg.text_adapted != seg.text_tgt]

    if not modified:
        print(f"\n⏩ Passe 5b — Aucun segment modifié par l'adaptation, relecture sautée")
        return segments

    print(f"\n🗣️  Passe 5b — Relecture fluidité post-adaptation ({len(modified)} segments)...")

    tgt_n = lang_name(tgt_lang)

    BATCH = 60
    fixes = 0

    for bi in range(0, len(modified), BATCH):
        batch = modified[bi:bi + BATCH]
        lines = []
        for seg in batch:
            lines += [
                f"[{seg.index}] AVANT: {seg.text_tgt}",
                f"[{seg.index}] APRÈS: {seg.text_adapted}",
                ""
            ]

        prompt = f"""Tu es un réviseur de doublage en {tgt_n}. Les textes ci-dessous ont été
raccourcis ou allongés pour tenir dans la durée de l'original. Certains ont pu perdre
en fluidité orale pendant ce processus.

Pour chaque segment, vérifie UNIQUEMENT ces critères :
1. FLUIDITÉ ORALE : le texte sonne-t-il naturel quand on le prononce à voix haute ?
2. CONNECTEURS : les mots de liaison nécessaires sont-ils présents (et, mais, donc,
   alors, c'est que, parce que…) ? Un segment qui commence au milieu d'une idée
   commencée dans le segment précédent doit garder son connecteur d'ouverture.
3. STRUCTURE : chaque phrase a-t-elle un sujet et un verbe ? Pas de style télégraphique.
4. PONCTUATION : virgules de respiration, points de suspension, deux-points — la
   ponctuation doit guider la prononciation naturelle du texte.

IMPORTANT :
- Ne change PAS la longueur du texte de façon significative (±5 caractères max).
- Le but est de POLIR, pas de réécrire. Corrections minimales uniquement.
- Si le texte APRÈS est déjà fluide → ne l'inclure PAS dans ta réponse.

Appelle l'outil submit_texts avec un item PAR SEGMENT MODIFIÉ uniquement.
Le champ text contient le texte corrigé final, sans aucun commentaire ni
préfixe. Si aucun segment ne nécessite de correction, appelle l'outil avec
items=[] (liste vide).

{chr(10).join(lines)}"""

        corrected = _claude_submit_texts(client, prompt)

        for idx, new in corrected.items():
            for seg in segments:
                if seg.index == idx and seg.text_adapted != new:
                    if _est_fuite_prompt(new, seg.text_adapted):
                        print(f"   ⚠️  Fuite de prompt détectée seg [{idx}], ignoré : {new[:60]}…")
                        break
                    seg.text_adapted = new
                    fixes += 1
                    break

    print(f"   🗣️  {fixes} corrections de fluidité")
    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 5c : RESTAURATION DES QUALIFIEURS PROTÉGÉS (CLAUDE)
# ═══════════════════════════════════════════════════════════════════════════════

# Adverbes-qualifieurs éditoriaux dont la suppression change la portée d'une
# phrase ("soi-disant médicaments" ≠ "médicaments"). Liste minimaliste : seuls
# les mots dont la perte est presque toujours une faute sont inclus.
PROTECTED_QUALIFIERS_FR = [
    "soi-disant", "prétendument", "potentiellement", "vraiment",
    "probablement", "particulièrement", "principalement", "éventuellement",
    "approximativement", "apparemment",
]


def restore_protected_qualifiers(segments: list, client, tgt_lang: str) -> list:
    """
    Détecte les qualifieurs présents dans text_tgt mais absents de text_adapted,
    et demande à Claude de les réinsérer SANS toucher au reste de la phrase.
    Sans cible chiffrée, sans toucher la longueur — uniquement réinsertion ciblée.
    """
    if tgt_lang != "fr":
        # La liste actuelle est française ; d'autres langues à étendre au besoin.
        return segments

    to_fix = []
    for seg in segments:
        if not seg.text_adapted or not seg.text_tgt:
            continue
        if seg.text_adapted == seg.text_tgt:
            continue
        tgt_lower = seg.text_tgt.lower()
        adp_lower = seg.text_adapted.lower()
        missing = [q for q in PROTECTED_QUALIFIERS_FR
                   if q in tgt_lower and q not in adp_lower]
        if missing:
            to_fix.append((seg, missing))

    if not to_fix:
        print(f"\n⏩ Passe 5c — Aucun qualifieur à restaurer")
        return segments

    print(f"\n🔧 Passe 5c — Restauration de qualifieurs ({len(to_fix)} segments)...")

    tgt_n = lang_name(tgt_lang)
    BATCH = 30
    restored = 0

    for bi in range(0, len(to_fix), BATCH):
        batch = to_fix[bi:bi + BATCH]
        blocks = []
        for seg, missing in batch:
            blocks += [
                f"[{seg.index}] MANQUE: {', '.join(missing)}",
                f"[{seg.index}] ACTUELLE: {seg.text_adapted}",
                f"[{seg.index}] RÉFÉRENCE: {seg.text_tgt}",
                "",
            ]

        prompt = f"""Réinsère des adverbes-qualifieurs manquants dans des phrases en {tgt_n}.

Pour chaque segment :
- MANQUE     : la liste des adverbes à remettre
- ACTUELLE   : la phrase actuelle, à laquelle il manque ces adverbes
- RÉFÉRENCE  : la phrase d'origine (qui montre où chaque adverbe se plaçait)

TA SEULE TÂCHE : réinsérer chaque adverbe manquant dans ACTUELLE, à l'emplacement
où il apparaissait dans RÉFÉRENCE. Tu ne touches RIEN d'autre : ni la structure,
ni les autres mots, ni la ponctuation, ni la longueur globale (à part l'ajout
des adverbes eux-mêmes).

CAS PARTICULIER : si l'adverbe ne peut plus se loger dans ACTUELLE parce que le
mot/groupe qu'il qualifiait a disparu lors du raccourcissement, soit tu le
remets quand même avec un ajustement minimal (préférable), soit tu OMETS
simplement ce segment de la liste items (pas besoin de marqueur spécial).

Appelle submit_texts avec un item par segment effectivement modifié.
Le champ text contient uniquement la phrase corrigée — pas de commentaire.

{chr(10).join(blocks)}"""

        corrected = _claude_submit_texts(client, prompt)

        for idx, new in corrected.items():
            for seg, _missing in batch:
                if seg.index == idx and seg.text_adapted != new:
                    if _est_fuite_prompt(new, seg.text_adapted):
                        print(f"   ⚠️  Fuite de prompt détectée seg [{idx}], ignoré : {new[:60]}…")
                        break
                    new_lower = new.lower()
                    still_missing = [q for q in _missing if q not in new_lower]
                    if still_missing == _missing:
                        # Aucune restauration effective — on garde l'ancien.
                        break
                    seg.text_adapted = new
                    restored += 1
                    break

    print(f"   🔧 {restored} qualifieurs restaurés")
    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 6 : SYNTHÈSE VOCALE (XTTS v2)
# ═══════════════════════════════════════════════════════════════════════════════

class TTSBackend:
    """Interface abstraite pour les backends TTS."""

    # Surcharge explicite locuteur → fichier voix (rempli par --map-voices).
    # Prioritaire sur l'affectation par genre / l'estimation F0.
    _voice_overrides: dict = {}

    def set_voice_overrides(self, overrides: dict):
        """Force un fichier de voix précis par locuteur (SPEAKER_xx → chemin WAV).

        Appelé avant la synthèse quand l'utilisateur a apparié les voix à la
        main (--map-voices). Les backends à clonage consultent _speaker_ref_voice
        en priorité dans _get_best_ref, donc on y injecte directement la map.
        """
        self._voice_overrides = dict(overrides or {})

    def _apply_voice_overrides(self):
        """À appeler en fin de setup_voices : applique la map explicite.

        Écrase l'affectation genrée pour chaque locuteur apparié manuellement.
        """
        if not getattr(self, "_voice_overrides", None):
            return
        if not hasattr(self, "_speaker_ref_voice"):
            self._speaker_ref_voice = {}
        if not hasattr(self, "voice_map"):
            self.voice_map = {}
        for spk, path in self._voice_overrides.items():
            if not path:
                continue
            self._speaker_ref_voice[spk] = path
            self.voice_map[spk] = f"clone:{spk}"
            print(f"      🎯 {spk} → voix appariée : {os.path.basename(path)}")

    def setup_voices(self, profiles: dict[str, SpeakerProfile]):
        raise NotImplementedError

    def synthesize(self, text: str, speaker_id: str, output_path: str,
                   target_duration: float = 0.0) -> str:
        """target_duration > 0 active le two-pass (ajustement speed natif)."""
        raise NotImplementedError

    def synthesize_with_speed(self, text: str, speaker_id: str,
                              output_path: str, speed: float = 1.0) -> str:
        """Synthèse directe à une vitesse donnée (pour sauvetage 6c)."""
        raise NotImplementedError

    def cleanup(self):
        pass



# ── Voix preset XTTS v2 ──────────────────────────────────────────────────────
# Stockées dans speakers_xtts.pth du modèle. Accès via tts.speakers ou --list_speaker_idxs.
# Classées par genre, priorité broadcast (voix posées en premier).
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


class XTTSBackend(TTSBackend):
    """Backend XTTS v2 (Coqui TTS — local, GPU, multilingue, zero-shot cloning)."""

    # Mapping des codes langue doubler.py → codes XTTS v2
    XTTS_LANG_MAP = {
        "fr": "fr", "en": "en", "es": "es", "de": "de", "it": "it",
        "pt": "pt", "pl": "pl", "tr": "tr", "ru": "ru", "nl": "nl",
        "cs": "cs", "ar": "ar", "zh": "zh", "ja": "ja", "ko": "ko",
        "hu": "hu", "hi": "hi",
    }

    def __init__(self, target_lang: str = "fr", ref_voice: Optional[str] = None,
                 xtts_speaker: Optional[str] = None):
        from TTS.api import TTS as CoquiTTS
        print("   🔊 Chargement du modèle XTTS v2...")
        self.model = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
        self.target_lang = self.XTTS_LANG_MAP.get(target_lang, target_lang)
        self.profiles: dict[str, SpeakerProfile] = {}
        self.ref_voice = ref_voice  # override optionnel
        self.xtts_speaker = xtts_speaker  # voix preset (ex: "Craig Gutsy")
        self.voice_map: dict[str, str] = {}  # speaker_id → preset voice name
        # Warm-up : le premier appel XTTS produit souvent un audio dégradé
        # (décodeur autorégressif pas encore stabilisé). On force un appel
        # silencieux avec un preset pour "chauffer" le modèle.
        try:
            import tempfile
            warmup_path = os.path.join(tempfile.gettempdir(), "_xtts_warmup.wav")
            self.model.tts_to_file(
                text="Bonjour, bienvenue.",
                speaker="Craig Gutsy",
                language=self.target_lang,
                file_path=warmup_path,
            )
            if os.path.exists(warmup_path):
                os.remove(warmup_path)
        except Exception:
            pass  # ne pas bloquer si le warm-up échoue
        print(f"   ✅ XTTS v2 prêt (langue cible: {self.target_lang})")

    def setup_voices(self, profiles: dict[str, SpeakerProfile]):
        self.profiles = profiles
        speakers = sorted(profiles.keys())

        if self.ref_voice:
            print(f"      🎯 Voix de référence externe : {self.ref_voice}")

        # Assigner des voix preset pour les locuteurs sans ref_clips
        if self.xtts_speaker:
            # Override unique
            for spk in speakers:
                self.voice_map[spk] = self.xtts_speaker
            print(f"      🎯 Voix preset '{self.xtts_speaker}' pour {len(speakers)} locuteur(s)")
        else:
            female_idx = male_idx = unknown_idx = 0
            for spk in speakers:
                p = profiles[spk]
                has_ref = bool(p.ref_clips) or (p.sample_path and os.path.exists(p.sample_path))

                if has_ref and not self.xtts_speaker:
                    # On utilisera le clonage vocal → pas de preset
                    icon = "🎙️"
                    ref_count = len(p.ref_clips) if p.ref_clips else 1
                    print(f"      {icon} {spk} : clonage vocal ({ref_count} refs, "
                          f"{p.total_duration:.0f}s)")
                else:
                    # Assigner une voix preset par genre
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
                    gender_icon = "♀️" if gender == "female" else "♂️" if gender == "male" else "❓"
                    print(f"      🎤 {spk} {gender_icon} → preset \"{voice}\"")

        # Voix appariées manuellement (--map-voices) : on force le clonage depuis
        # le fichier choisi. Chez XTTS, un voice_map non vide = preset ; on le
        # vide donc pour que _synthesize retombe sur _get_best_ref (→ fichier).
        if getattr(self, "_voice_overrides", None):
            if not hasattr(self, "_speaker_ref_voice"):
                self._speaker_ref_voice = {}
            for spk, path in self._voice_overrides.items():
                if not path:
                    continue
                self._speaker_ref_voice[spk] = path
                self.voice_map.pop(spk, None)
                print(f"      🎯 {spk} → voix appariée : {os.path.basename(path)}")

    def _get_best_ref(self, speaker_id: str) -> str:
        """Retourne le chemin du meilleur audio de référence pour le cloning."""
        if self.ref_voice:
            return self.ref_voice

        # Voix appariée manuellement (--map-voices) : priorité absolue
        if speaker_id in getattr(self, "_speaker_ref_voice", {}):
            return self._speaker_ref_voice[speaker_id]

        profile = self.profiles.get(speaker_id)
        if not profile:
            profile = next((p for p in self.profiles.values()
                            if p.ref_clips or p.sample_path), None)
        if not profile:
            return ""

        # Préférer le clip le plus long (premier de ref_clips, trié par durée)
        if profile.ref_clips:
            return profile.ref_clips[0][0]  # (path, text) → path

        if profile.sample_path and os.path.exists(profile.sample_path):
            return profile.sample_path

        return ""

    @staticmethod
    def _split_text_for_tts(text: str, max_chars: int = XTTS_MAX_CHARS) -> list[str]:
        """
        Découpe un texte en morceaux de max_chars aux frontières naturelles.
        Priorité : phrase (. ! ?) > clause (, ; :) > espace.
        """
        if len(text) <= max_chars:
            return [text]

        chunks = []
        remaining = text.strip()

        while len(remaining) > max_chars:
            # Chercher le meilleur point de coupure dans la fenêtre [0, max_chars]
            window = remaining[:max_chars]
            cut = -1

            # Priorité 1 : fin de phrase
            for sep in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
                idx = window.rfind(sep)
                if idx > max_chars // 3:  # pas trop tôt dans le texte
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

    def synthesize(self, text: str, speaker_id: str, output_path: str,
                   target_duration: float = 0.0) -> str:
        """
        Synthèse vocale avec ajustement de vitesse two-pass.
        
        Si target_duration > 0 :
          Passe 1 → synthèse à speed=1.0, trim silence, mesure de la durée réelle
          Calcul du ratio needed_speed = durée_trimmée / target_duration
          Passe 2 → re-synthèse avec speed ajusté, trim, vérification
          Si encore trop long → passe 2b avec speed corrigé (max 1 retry)
        
        Le paramètre speed natif d'XTTS modifie le débit au niveau du décodeur,
        sans artefacts de time-stretching.
        """
        chunks = self._split_text_for_tts(text) if len(text) > XTTS_MAX_CHARS else [text]

        # ── Passe 1 : synthèse à speed=1.0 ──────────────────────────────────
        result = self._synthesize_chunks(chunks, speaker_id, output_path, speed=1.0)
        if not result:
            return ""

        # ── Sans cible temporelle → on garde la passe 1 ─────────────────────
        if target_duration <= 0:
            return result

        # ── Trimmer AVANT de mesurer (supprime le silence parasite XTTS) ─────
        trimmed_dur = _trim_tts_silence(result)
        if trimmed_dur < 0.1:
            return result

        needed_speed = trimmed_dur / target_duration

        # ── Dans la tolérance ? → garder tel quel ───────────────────────────
        if abs(needed_speed - 1.0) <= XTTS_SPEED_TOLERANCE:
            return result

        # ── Clamper à la plage de qualité XTTS ──────────────────────────────
        clamped_speed = max(XTTS_SPEED_MIN, min(XTTS_SPEED_MAX, needed_speed))

        # ── Passe 2 : re-synthèse avec vitesse ajustée ─────────────────────
        result2 = self._synthesize_chunks(chunks, speaker_id, output_path,
                                          speed=clamped_speed)
        if not result2:
            return result  # fallback sur passe 1

        # Trim + mesure post-passe-2
        pass2_dur = _trim_tts_silence(result2)

        # ── Vérification : si encore trop long, retry avec speed corrigé ────
        # Le speed XTTS n'est pas linéaire, donc on corrige empiriquement.
        if pass2_dur > target_duration * 1.08 and clamped_speed < XTTS_SPEED_MAX:
            # Correction : ratio réel observé entre passe 1 et passe 2
            # speed_effective = trimmed_dur / pass2_dur (combien le speed a réellement accéléré)
            # On ajuste pour atteindre la cible
            correction_speed = clamped_speed * (pass2_dur / target_duration)
            correction_speed = max(XTTS_SPEED_MIN, min(XTTS_SPEED_MAX, correction_speed))

            if abs(correction_speed - clamped_speed) > 0.03:  # correction significative
                result3 = self._synthesize_chunks(chunks, speaker_id, output_path,
                                                  speed=correction_speed)
                if result3:
                    pass3_dur = _trim_tts_silence(result3)
                    # Garder le meilleur (le plus proche de la cible sans la dépasser)
                    if pass3_dur <= target_duration * 1.08 or pass3_dur < pass2_dur:
                        pass2_dur = pass3_dur
                        clamped_speed = correction_speed

        icon = "🏃" if clamped_speed > 1.0 else "🐢"
        clamp_note = ""
        if abs(clamped_speed - needed_speed) > 0.01:
            clamp_note = f" [clampé, idéal={needed_speed:.2f}]"
        overshoot = ""
        if pass2_dur > target_duration * 1.05:
            overshoot = f" ⚠️+{(pass2_dur - target_duration)*1000:.0f}ms"
        print(f"      {icon} Two-pass: {trimmed_dur:.2f}s→{pass2_dur:.2f}s "
              f"(cible {target_duration:.2f}s, speed={clamped_speed:.2f}{clamp_note}{overshoot})")

        return output_path

    def _synthesize_chunks(self, chunks: list[str], speaker_id: str,
                           output_path: str, speed: float = 1.0) -> str:
        """
        Synthétise une liste de chunks de texte et les concatène.
        Factorise la logique mono-chunk et multi-chunk.
        """
        if len(chunks) == 1:
            return self._synthesize_single(chunks[0], speaker_id, output_path,
                                           speed=speed)

        import soundfile as sf
        import numpy as np

        chunk_paths = []
        out_dir = os.path.dirname(output_path)
        base = os.path.splitext(os.path.basename(output_path))[0]

        for ci, chunk_text in enumerate(chunks):
            chunk_path = os.path.join(out_dir, f"{base}_part{ci:02d}.wav")
            result = self._synthesize_single(chunk_text, speaker_id, chunk_path,
                                             speed=speed)
            if result:
                chunk_paths.append(result)
            else:
                print(f"      ⚠️  Chunk {ci+1}/{len(chunks)} échoué ({len(chunk_text)} chars)")

        if not chunk_paths:
            return ""

        if len(chunk_paths) == 1:
            shutil.move(chunk_paths[0], output_path)
            return output_path

        # Concaténer avec micro-crossfade de 128 samples aux jonctions
        arrays = []
        sr = None
        for cp in chunk_paths:
            data, file_sr = sf.read(cp, dtype="float32")
            if sr is None:
                sr = file_sr
            arrays.append(data)

        crossfade_samples = min(128, min(len(a) for a in arrays) // 2) if arrays else 0
        gap_samples = int(sr * 0.08)
        combined = []
        for i, arr in enumerate(arrays):
            if i > 0:
                if crossfade_samples > 0 and len(combined) > 0:
                    prev = combined[-1]
                    fade_out = np.linspace(1.0, 0.0, crossfade_samples, dtype=np.float32)
                    prev[-crossfade_samples:] *= fade_out
                    combined[-1] = prev
                    combined.append(np.zeros(gap_samples, dtype=np.float32))
                    fade_in = np.linspace(0.0, 1.0, crossfade_samples, dtype=np.float32)
                    arr = arr.copy()
                    arr[:crossfade_samples] *= fade_in
                else:
                    combined.append(np.zeros(gap_samples, dtype=np.float32))
            combined.append(arr)

        result = np.concatenate(combined, axis=0)
        sf.write(output_path, result, sr, subtype="PCM_16")

        # Nettoyage des fichiers temporaires
        for cp in chunk_paths:
            if os.path.exists(cp):
                os.remove(cp)

        return output_path

    def synthesize_with_speed(self, text: str, speaker_id: str,
                              output_path: str, speed: float = 1.0) -> str:
        """
        Synthèse directe à une vitesse donnée (sans logique two-pass).
        Utilisé par verify_and_fix_timing pour le sauvetage par speed.
        Trimme automatiquement le silence XTTS.
        """
        chunks = self._split_text_for_tts(text) if len(text) > XTTS_MAX_CHARS else [text]
        result = self._synthesize_chunks(chunks, speaker_id, output_path, speed=speed)
        if result:
            _trim_tts_silence(result)
        return result

    def _synthesize_single(self, text: str, speaker_id: str, output_path: str,
                           speed: float = 1.0) -> str:
        """
        Synthèse d'un seul morceau de texte (doit être ≤ XTTS_MAX_CHARS).
        Utilise model.tts() (pas tts_to_file) pour exposer le paramètre speed natif XTTS.
        speed=1.0 → débit normal ; speed=1.2 → 20% plus rapide au niveau du modèle.

        Paramètres anti-bégaiement à 3 paliers selon la longueur du texte :
        - < 20 chars : très agressif (le décodeur boucle facilement sur les textes très courts)
        - 20–50 chars : intermédiaire
        - > 50 chars : relâché (le contexte suffit à stabiliser le décodeur)
        """
        import soundfile as sf
        import numpy as np

        # Nettoyage phonétique — garantit que XTTS ne prononce pas la ponctuation
        text = _clean_for_tts(text, lang=self.target_lang)
        if not text:
            return ""

        n_chars = len(text)
        if n_chars < 20:
            # Très court : paramètres fermes mais pas extrêmes
            # (12.0/0.30 causait des arrêts prématurés sur mots étrangers)
            xtts_kwargs = dict(
                repetition_penalty=7.0,
                temperature=0.45,
                top_k=30,
                top_p=0.70,
                enable_text_splitting=False,
            )
        elif n_chars <= 50:
            # Intermédiaire
            xtts_kwargs = dict(
                repetition_penalty=5.5,
                temperature=0.55,
                top_k=40,
                top_p=0.80,
                enable_text_splitting=False,
            )
        else:
            # Normal : relâché
            xtts_kwargs = dict(
                repetition_penalty=5.0,
                temperature=0.65,
                top_k=50,
                top_p=0.85,
                enable_text_splitting=False,
            )

        preset_voice = self.voice_map.get(speaker_id)

        try:
            if preset_voice:
                wav = self.model.tts(
                    text=text,
                    speaker=preset_voice,
                    language=self.target_lang,
                    speed=speed,
                    **xtts_kwargs,
                )
            else:
                ref_path = self._get_best_ref(speaker_id)
                if not ref_path:
                    print(f"      ❌ Aucun échantillon disponible pour {speaker_id}")
                    return ""
                wav = self.model.tts(
                    text=text,
                    speaker_wav=ref_path,
                    language=self.target_lang,
                    speed=speed,
                    **xtts_kwargs,
                )

            # model.tts() retourne une liste de floats — sauvegarder manuellement
            wav_np = np.array(wav, dtype=np.float32)
            # Sample rate XTTS v2 = 24000 Hz (via model.synthesizer.output_sample_rate)
            out_sr = getattr(self.model.synthesizer, 'output_sample_rate', 24000)
            sf.write(output_path, wav_np, out_sr, subtype="PCM_16")
            return output_path
        except Exception as e:
            print(f"      ❌ XTTS échoué [{speaker_id}] speed={speed:.2f} : {e}")
            return ""

    def cleanup(self):
        del self.model; gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ElevenLabsBackend(TTSBackend):
    """Backend ElevenLabs (API cloud — clonage vocal instantané, haute qualité)."""

    def __init__(self, target_lang: str = "fr", ref_voice: Optional[str] = None,
                 elevenlabs_voice: Optional[str] = None,
                 elevenlabs_model: Optional[str] = None):
        from elevenlabs import ElevenLabs
        self.client = ElevenLabs()
        self.target_lang = target_lang
        self.ref_voice = ref_voice
        self.elevenlabs_voice = elevenlabs_voice  # voice_id pré-existant
        self.model_id = elevenlabs_model or ELEVENLABS_MODEL_DEFAULT
        self.profiles: dict[str, SpeakerProfile] = {}
        self.voice_map: dict[str, str] = {}  # speaker_id → voice_id ElevenLabs
        self._cloned_voice_ids: list[str] = []  # voix IVC à supprimer dans cleanup()
        self._total_chars = 0
        self.speed_rescue_max = SPEED_COMFORT_MAX
        print(f"   ✅ ElevenLabs prêt (modèle: {self.model_id}, langue: {self.target_lang})")

    def setup_voices(self, profiles: dict[str, SpeakerProfile]):
        self.profiles = profiles
        speakers = sorted(profiles.keys())

        if self.elevenlabs_voice:
            # Voix pré-existante → tous les speakers l'utilisent
            for spk in speakers:
                self.voice_map[spk] = self.elevenlabs_voice
            print(f"      🎯 Voix ElevenLabs pré-existante '{self.elevenlabs_voice}' "
                  f"pour {len(speakers)} locuteur(s)")
            return

        # Clonage IVC par speaker
        for spk in speakers:
            p = profiles[spk]
            ref_files = []

            # Collecter les fichiers audio de référence
            if self.ref_voice:
                ref_files = [self.ref_voice]
            elif p.ref_clips:
                ref_files = [clip[0] for clip in p.ref_clips]  # (path, text) → path
            elif p.sample_path and os.path.exists(p.sample_path):
                ref_files = [p.sample_path]

            if not ref_files:
                print(f"      ⚠️  {spk} : aucun audio de référence, voix par défaut")
                # Pas de voice_id → utilisera le modèle sans clonage
                continue

            try:
                # Ouvrir les fichiers audio pour l'upload
                file_handles = []
                for rf in ref_files:
                    file_handles.append(open(rf, "rb"))

                voice = self.client.voices.ivc.create_preview(
                    voice_name=f"dubbing_{spk}",
                    voice_description=f"Cloned voice for {spk}",
                    files=file_handles,
                )
                # Créer la voix permanente à partir du preview
                created_voice = self.client.voices.ivc.create_voice_from_preview(
                    voice_name=f"dubbing_{spk}",
                    voice_description=f"Cloned voice for {spk}",
                    generated_voice_id=voice.generated_voice_id,
                )
                voice_id = created_voice.voice_id

                # Fermer les handles
                for fh in file_handles:
                    fh.close()

                self.voice_map[spk] = voice_id
                self._cloned_voice_ids.append(voice_id)
                ref_info = f"{len(ref_files)} refs" if len(ref_files) > 1 else "1 ref"
                print(f"      🎙️  {spk} : clonage IVC OK (voice_id={voice_id[:12]}..., {ref_info})")
            except Exception as e:
                # Fermer les handles en cas d'erreur
                for fh in file_handles:
                    try:
                        fh.close()
                    except Exception:
                        pass
                print(f"      ⚠️  {spk} : clonage IVC échoué ({e}), voix par défaut")

    def synthesize(self, text: str, speaker_id: str, output_path: str,
                   target_duration: float = 0.0) -> str:
        """
        Synthèse vocale ElevenLabs avec ajustement temporel via ffmpeg atempo.

        Si target_duration > 0 :
          Synthèse à vitesse normale → mesure durée → atempo pour ajuster.
          Plage : 0.70–1.50 (traitement signal pur, pas de dégradation modèle).
        """
        text = _clean_for_tts(text, lang=self.target_lang)
        if not text:
            return ""

        chunks = self._split_text_for_tts(text) if len(text) > ELEVENLABS_MAX_CHARS else [text]

        # Synthèse
        result = self._synthesize_chunks(chunks, speaker_id, output_path)
        if not result:
            return ""

        # Sans cible temporelle → on garde tel quel
        if target_duration <= 0:
            return result

        # Mesurer la durée réelle
        import soundfile as sf
        info = sf.info(result)
        real_dur = info.duration
        if real_dur < 0.1:
            return result

        needed_speed = real_dur / target_duration

        # Dans la tolérance ? → garder tel quel
        if abs(needed_speed - 1.0) <= XTTS_SPEED_TOLERANCE:
            return result

        if needed_speed < 1.0:
            # TTS trop court → silence padding, jamais de ralentissement
            _pad_silence(result, target_duration)
            info2 = sf.info(result)
            post_dur = info2.duration
            print(f"      🔇 Two-pass: {real_dur:.2f}s→{post_dur:.2f}s "
                  f"(cible {target_duration:.2f}s, silence +{(target_duration - real_dur)*1000:.0f}ms)")
        else:
            # TTS trop long → atempo modéré
            clamped_speed = min(SPEED_COMFORT_MAX, needed_speed)
            self._apply_atempo(result, clamped_speed)

            info2 = sf.info(result)
            post_dur = info2.duration

            clamp_note = ""
            if abs(clamped_speed - needed_speed) > 0.01:
                clamp_note = f" [clampé, idéal={needed_speed:.2f}]"
            overshoot = ""
            if post_dur > target_duration * 1.05:
                overshoot = f" ⚠️+{(post_dur - target_duration)*1000:.0f}ms"
            print(f"      🏃 Two-pass: {real_dur:.2f}s→{post_dur:.2f}s "
                  f"(cible {target_duration:.2f}s, atempo={clamped_speed:.2f}{clamp_note}{overshoot})")

        return output_path

    def synthesize_with_speed(self, text: str, speaker_id: str,
                              output_path: str, speed: float = 1.0) -> str:
        """Synthèse directe puis atempo (accélération uniquement) ou silence padding."""
        text = _clean_for_tts(text, lang=self.target_lang)
        if not text:
            return ""

        chunks = self._split_text_for_tts(text) if len(text) > ELEVENLABS_MAX_CHARS else [text]
        result = self._synthesize_chunks(chunks, speaker_id, output_path)
        if result and abs(speed - 1.0) > 0.01:
            if speed > 1.0:
                self._apply_atempo(result, speed)
            # speed < 1.0 → ne pas ralentir (le silence sera ajouté par synthesize)
        return result

    def _synthesize_chunks(self, chunks: list[str], speaker_id: str,
                           output_path: str) -> str:
        """Synthétise une liste de chunks et les concatène."""
        if len(chunks) == 1:
            return self._synthesize_single(chunks[0], speaker_id, output_path)

        import soundfile as sf
        import numpy as np

        chunk_paths = []
        out_dir = os.path.dirname(output_path)
        base = os.path.splitext(os.path.basename(output_path))[0]

        for ci, chunk_text in enumerate(chunks):
            chunk_path = os.path.join(out_dir, f"{base}_part{ci:02d}.wav")
            result = self._synthesize_single(chunk_text, speaker_id, chunk_path)
            if result:
                chunk_paths.append(result)
            else:
                print(f"      ⚠️  Chunk {ci+1}/{len(chunks)} échoué ({len(chunk_text)} chars)")

        if not chunk_paths:
            return ""

        if len(chunk_paths) == 1:
            shutil.move(chunk_paths[0], output_path)
            return output_path

        # Concaténer avec micro-crossfade de 128 samples aux jonctions
        arrays = []
        sr = None
        for cp in chunk_paths:
            data, file_sr = sf.read(cp, dtype="float32")
            if sr is None:
                sr = file_sr
            arrays.append(data)

        crossfade_samples = min(128, min(len(a) for a in arrays) // 2) if arrays else 0
        gap_samples = int(sr * 0.08)
        combined = []
        for i, arr in enumerate(arrays):
            if i > 0:
                if crossfade_samples > 0 and len(combined) > 0:
                    prev = combined[-1]
                    fade_out = np.linspace(1.0, 0.0, crossfade_samples, dtype=np.float32)
                    prev[-crossfade_samples:] *= fade_out
                    combined[-1] = prev
                    combined.append(np.zeros(gap_samples, dtype=np.float32))
                    fade_in = np.linspace(0.0, 1.0, crossfade_samples, dtype=np.float32)
                    arr = arr.copy()
                    arr[:crossfade_samples] *= fade_in
                else:
                    combined.append(np.zeros(gap_samples, dtype=np.float32))
            combined.append(arr)

        result = np.concatenate(combined, axis=0)
        sf.write(output_path, result, sr, subtype="PCM_16")

        # Nettoyage des fichiers temporaires
        for cp in chunk_paths:
            if os.path.exists(cp):
                os.remove(cp)

        return output_path

    def _synthesize_single(self, text: str, speaker_id: str,
                           output_path: str) -> str:
        """
        Synthèse d'un seul morceau de texte via l'API ElevenLabs.
        Retourne le chemin WAV ou "" en cas d'échec.
        """
        voice_id = self.voice_map.get(speaker_id)
        if not voice_id:
            # Pas de voix clonée → utiliser le premier voice_id disponible ou échouer
            if self.voice_map:
                voice_id = next(iter(self.voice_map.values()))
            else:
                print(f"      ❌ Aucune voix ElevenLabs pour {speaker_id}")
                return ""

        mp3_path = output_path.replace(".wav", ".mp3")

        for attempt in range(ELEVENLABS_RETRY_MAX):
            try:
                audio_gen = self.client.text_to_speech.convert(
                    voice_id=voice_id,
                    text=text,
                    model_id=self.model_id,
                    output_format=ELEVENLABS_OUTPUT_FORMAT,
                    voice_settings={
                        "stability": ELEVENLABS_STABILITY,
                        "similarity_boost": ELEVENLABS_SIMILARITY_BOOST,
                        "style": ELEVENLABS_STYLE,
                    },
                )
                # Écrire le MP3
                with open(mp3_path, "wb") as f:
                    for chunk in audio_gen:
                        f.write(chunk)

                self._total_chars += len(text)

                # Convertir MP3 → WAV 44100Hz mono PCM_16
                r = subprocess.run(
                    ["ffmpeg", "-y", "-i", mp3_path,
                     "-ar", "44100", "-ac", "1", "-acodec", "pcm_s16le",
                     output_path],
                    capture_output=True, text=True
                )
                if os.path.exists(mp3_path):
                    os.remove(mp3_path)

                if r.returncode != 0:
                    print(f"      ❌ ffmpeg conversion échouée : {r.stderr[:200]}")
                    return ""

                return output_path

            except Exception as e:
                err_str = str(e)
                # Retry sur rate limit (429)
                if "429" in err_str or "rate" in err_str.lower():
                    delay = ELEVENLABS_RETRY_DELAY * (2 ** attempt)
                    print(f"      ⏳ Rate limit ElevenLabs, retry dans {delay:.0f}s...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"      ❌ ElevenLabs échoué [{speaker_id}] : {e}")
                    return ""

        print(f"      ❌ ElevenLabs échoué après {ELEVENLABS_RETRY_MAX} tentatives [{speaker_id}]")
        return ""

    @staticmethod
    def _split_text_for_tts(text: str, max_chars: int = ELEVENLABS_MAX_CHARS) -> list[str]:
        """
        Découpe un texte en morceaux de max_chars aux frontières naturelles.
        Même logique que XTTSBackend (sentence > clause > space > hard).
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
                    cut = idx + 1
                    break

            # Priorité 2 : clause
            if cut < 0:
                for sep in [", ", "; ", ": ", " – ", " — "]:
                    idx = window.rfind(sep)
                    if idx > max_chars // 3:
                        cut = idx + len(sep)
                        break

            # Priorité 3 : espace
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

    @staticmethod
    def _apply_atempo(wav_path: str, speed: float):
        """Applique ffmpeg atempo in-place sur un fichier WAV."""
        if abs(speed - 1.0) < 0.01:
            return

        tmp_path = wav_path + ".atempo.wav"

        # ffmpeg atempo accepte 0.5–100.0, mais on chaîne si hors [0.5, 2.0]
        # Notre plage 0.70–1.50 est toujours dans [0.5, 2.0] → un seul filtre
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path,
             "-filter:a", f"atempo={speed:.4f}",
             "-ar", "44100", "-ac", "1", "-acodec", "pcm_s16le",
             tmp_path],
            capture_output=True, text=True
        )
        if r.returncode == 0 and os.path.exists(tmp_path):
            shutil.move(tmp_path, wav_path)
        else:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def cleanup(self):
        # Supprimer les voix IVC clonées
        for voice_id in self._cloned_voice_ids:
            try:
                self.client.voices.delete(voice_id=voice_id)
                print(f"      🗑️  Voix clonée supprimée : {voice_id[:12]}...")
            except Exception as e:
                print(f"      ⚠️  Impossible de supprimer la voix {voice_id[:12]}... : {e}")

        if self._total_chars > 0:
            print(f"   📊 ElevenLabs : {self._total_chars:,} caractères utilisés au total")


def _scan_ref_voices_dir(directory: str) -> dict[str, list[str]]:
    """Scanne un dossier pour des voix de référence genrées.

    Convention : homme*.wav → male, femme*.wav → female.
    Retourne {"male": [chemins], "female": [chemins]}.
    """
    import glob as _glob
    result = {"male": [], "female": []}
    for path in sorted(_glob.glob(os.path.join(directory, "homme*.wav"))):
        result["male"].append(path)
    for path in sorted(_glob.glob(os.path.join(directory, "femme*.wav"))):
        result["female"].append(path)
    total = len(result["male"]) + len(result["female"])
    print(f"      📂 Voix de référence : {len(result['male'])} homme(s), "
          f"{len(result['female'])} femme(s) dans {directory}")
    if total == 0:
        print(f"      ⚠️  Aucun fichier homme*.wav / femme*.wav trouvé dans {directory}")
    return result


def _list_ref_voice_files(directory: str) -> list[dict]:
    """Liste toutes les voix d'un dossier avec leur genre déduit du nom.

    Convention : homme*.wav → male, femme*.wav → female, sinon unknown.
    Retourne [{"name", "path", "gender"}, ...] trié femmes puis hommes.
    """
    import glob as _glob
    out = []
    for path in sorted(_glob.glob(os.path.join(directory, "*.wav"))):
        base = os.path.basename(path).lower()
        if base.startswith("femme"):
            gender = "female"
        elif base.startswith("homme"):
            gender = "male"
        else:
            gender = "unknown"
        out.append({"name": os.path.splitext(os.path.basename(path))[0],
                    "path": os.path.abspath(path), "gender": gender})
    # Femmes d'abord puis hommes puis le reste (lisibilité de la liste)
    order = {"female": 0, "male": 1, "unknown": 2}
    out.sort(key=lambda v: (order[v["gender"]], v["name"]))
    return out


def _try_play_audio(path: str):
    """Joue un WAV via ffplay/aplay/afplay si dispo (best-effort, non bloquant fatal)."""
    import shutil as _sh
    import subprocess as _sp
    for player, pargs in (("ffplay", ["-autoexit", "-nodisp", "-loglevel", "quiet"]),
                          ("aplay", []), ("afplay", []), ("paplay", [])):
        if _sh.which(player):
            try:
                _sp.run([player, *pargs, path], check=False)
            except Exception:
                pass
            return True
    return False


# Marqueurs lus par le daemon (traduction-daemon.py) pour relayer la question
# vers l'extension Chrome. Une ligne stdout = un événement.
VOICEMAP_REQUEST_MARKER = "@@VOICEMAP_REQUEST@@"
VOICEMAP_DONE_MARKER = "@@VOICEMAP_DONE@@"


def interactive_voice_map(profiles: dict, ref_voices_dir: str,
                          work_dir: str) -> dict:
    """Appariement interactif voix↔locuteur (--map-voices).

    Pour chaque locuteur diarisé, propose d'écouter un échantillon et de choisir
    une voix du dossier ref_voices_dir. Deux modes selon l'environnement :

      • Terminal (stdout est un TTY) : saisie clavier, lecture audio via ffplay.
      • Daemon / extension (stdout redirigé) : écrit voicemap_request.json,
        émet une ligne "@@VOICEMAP_REQUEST@@ <chemin>" puis attend
        voicemap_response.json (écrit par le daemon depuis l'UI).

    Retourne {SPEAKER_xx: chemin_voix_absolu}. Locuteurs non appariés → absents
    (le pipeline retombe alors sur l'affectation genrée habituelle).
    """
    voices = _list_ref_voice_files(ref_voices_dir)
    if not voices:
        print(f"   ⚠️  --map-voices : aucune voix dans {ref_voices_dir} — appariement ignoré")
        return {}

    # Locuteurs triés par durée décroissante (les plus présents d'abord)
    speakers = sorted(
        [p for p in profiles.values() if p.sample_path and os.path.exists(p.sample_path)],
        key=lambda p: p.total_duration, reverse=True,
    )
    if not speakers:
        print("   ⚠️  --map-voices : aucun échantillon de locuteur — appariement ignoré")
        return {}

    def _suggest(gender: str) -> int:
        """Index de voix suggéré par défaut selon le genre estimé."""
        for i, v in enumerate(voices):
            if v["gender"] == gender:
                return i
        return 0

    speaker_payload = []
    for p in speakers:
        snippet = (p.sample_text or "").strip().replace("\n", " ")
        speaker_payload.append({
            "id": p.speaker_id,
            "sample": os.path.abspath(p.sample_path),
            "f0": round(p.f0_median, 1),
            "gender_guess": p.gender,
            "duration": round(p.total_duration, 1),
            "text": snippet[:200],
            "suggested": _suggest(p.gender),
        })

    # ── Mode terminal interactif ───────────────────────────────────────────
    if sys.stdout.isatty() and sys.stdin.isatty():
        print("\n🎚️  Appariement interactif des voix (--map-voices)")
        print(f"   {len(voices)} voix disponibles dans {ref_voices_dir} :")
        for i, v in enumerate(voices):
            icon = "♀️" if v["gender"] == "female" else "♂️" if v["gender"] == "male" else "❓"
            print(f"     [{i}] {icon} {v['name']}")
        mapping = {}
        for sp in speaker_payload:
            icon = "♀️" if sp["gender_guess"] == "female" else "♂️" if sp["gender_guess"] == "male" else "❓"
            print(f"\n── {sp['id']}  ({icon} F0≈{sp['f0']:.0f}Hz, {sp['duration']:.0f}s)")
            if sp["text"]:
                print(f"   « {sp['text'][:120]} »")
            print(f"   Échantillon : {sp['sample']}")
            while True:
                ans = input(f"   Voix [0-{len(voices)-1}] "
                            f"(Entrée={sp['suggested']} {voices[sp['suggested']]['name']}, "
                            f"'p N'=écouter, 's'=garder auto) : ").strip()
                if ans == "":
                    idx = sp["suggested"]
                elif ans.lower() == "s":
                    idx = None
                elif ans.lower().startswith("p"):
                    parts = ans.split()
                    target = sp["sample"] if len(parts) < 2 else (
                        voices[int(parts[1])]["path"] if parts[1].isdigit()
                        and int(parts[1]) < len(voices) else sp["sample"])
                    if not _try_play_audio(target):
                        print("   (aucun lecteur audio trouvé — ouvrez le fichier manuellement)")
                    continue
                else:
                    try:
                        idx = int(ans)
                        if not (0 <= idx < len(voices)):
                            print("   ↳ index hors plage")
                            continue
                    except ValueError:
                        print("   ↳ entrée invalide")
                        continue
                break
            if idx is not None:
                mapping[sp["id"]] = voices[idx]["path"]
                print(f"   ✅ {sp['id']} → {voices[idx]['name']}")
            else:
                print(f"   ⏭️  {sp['id']} → auto (genre estimé)")
        return mapping

    # ── Mode daemon / extension (handshake par fichiers) ───────────────────
    request = {
        "type": "voicemap_request",
        "speakers": speaker_payload,
        "voices": voices,
        "response_file": os.path.abspath(os.path.join(work_dir, "voicemap_response.json")),
    }
    req_path = os.path.abspath(os.path.join(work_dir, "voicemap_request.json"))
    resp_path = request["response_file"]
    try:
        if os.path.exists(resp_path):
            os.remove(resp_path)
    except OSError:
        pass
    with open(req_path, "w", encoding="utf-8") as f:
        json.dump(request, f, ensure_ascii=False)

    # Annonce sur stdout : le daemon lit cette ligne et la relaie à l'extension.
    print(f"{VOICEMAP_REQUEST_MARKER} {req_path}", flush=True)
    print("   ⏳ En attente de l'appariement des voix depuis l'extension…", flush=True)

    timeout_s = float(os.environ.get("TRADUCTION_VOICEMAP_TIMEOUT", "1800"))
    waited = 0.0
    while waited < timeout_s:
        if os.path.exists(resp_path):
            try:
                with open(resp_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                time.sleep(0.5); waited += 0.5; continue
            raw = data.get("map", {}) or {}
            valid = {spk: path for spk, path in raw.items()
                     if path and os.path.exists(path)}
            print(f"{VOICEMAP_DONE_MARKER} {len(valid)} voix appariée(s)", flush=True)
            for spk, path in valid.items():
                print(f"   ✅ {spk} → {os.path.basename(path)}")
            return valid
        time.sleep(1.0); waited += 1.0

    print("   ⚠️  --map-voices : délai dépassé, appariement automatique conservé", flush=True)
    return {}


class Qwen3TTSBackend(TTSBackend):
    """Backend Qwen3-TTS via subprocess bridge (env conda isolé).

    Modèle Base 1.7B pour le clonage vocal zero-shot. 10 langues dont FR.
    Qwen3-TTS exige le transcript de l'audio de référence (ref_text) pour
    un clonage optimal ; sinon bascule en mode x_vector_only (timbre seul).
    """

    def __init__(self, target_lang: str = "fr", ref_voice: Optional[str] = None,
                 source_lang: str = "en", ref_voices_dir: Optional[str] = None):
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
        # (ex: accent allemand sur du français). On force x_vector_only
        # pour ne garder que le timbre sans l'accent étranger.
        self.cross_lang = (source_lang != target_lang)
        self.qwen_lang = QWEN3TTS_LANG_MAP.get(target_lang, "French")
        self.ref_voice = ref_voice
        self.ref_voices: Optional[dict[str, list[str]]] = None
        self._ref_voices_counters: dict[str, int] = {"male": 0, "female": 0}
        if ref_voices_dir and not ref_voice:
            self.ref_voices = _scan_ref_voices_dir(ref_voices_dir)
            if not self.ref_voices["male"] and not self.ref_voices["female"]:
                self.ref_voices = None
        self.profiles: dict[str, SpeakerProfile] = {}
        self.voice_map: dict[str, str] = {}
        self.speed_rescue_max = QWEN3TTS_SPEED_RESCUE_MAX

        device = resp.get("device", "?")
        mode_info = "x_vector_only (timbre seul, accent natif)" if self.cross_lang else "ICL (timbre + prosodie)"
        print(f"   ✅ Qwen3-TTS prêt (langue: {self.qwen_lang}, mode: {mode_info}, device={device})")

    def _drain_stderr(self):
        """Lit stderr en continu pour éviter le deadlock par buffer plein."""
        try:
            for line in self._proc.stderr:
                pass  # Ignorer (les erreurs critiques sont dans le JSON)
        except (ValueError, OSError):
            pass

    def _bridge_call(self, cmd: dict) -> dict:
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

    def setup_voices(self, profiles: dict[str, SpeakerProfile]):
        self.profiles = profiles
        speakers = sorted(profiles.keys())

        if self.ref_voice:
            print(f"      🎯 Voix de référence externe : {self.ref_voice}")

        # Affectation genrée round-robin si --ref-voices
        self._speaker_ref_voice: dict[str, str] = {}
        if self.ref_voices:
            for spk in speakers:
                p = profiles[spk]
                gender = p.gender  # "male" | "female" | "unknown"
                voice_path = self._pick_ref_voice(gender)
                if voice_path:
                    self._speaker_ref_voice[spk] = voice_path
                    icon = "♂️" if gender == "male" else ("♀️" if gender == "female" else "❓")
                    print(f"      🎙️  {spk} ({icon} {gender}) : {os.path.basename(voice_path)}")

        for spk in speakers:
            p = profiles[spk]
            has_ref = bool(p.ref_clips) or bool(p.sample_path and os.path.exists(p.sample_path))

            if self.ref_voice:
                self.voice_map[spk] = f"clone:{spk}"
                print(f"      🎙️  {spk} : ref externe")
            elif spk in self._speaker_ref_voice:
                self.voice_map[spk] = f"clone:{spk}"
            elif has_ref:
                self.voice_map[spk] = f"clone:{spk}"
                ref_count = len(p.ref_clips) if p.ref_clips else 1
                print(f"      🎙️  {spk} : clonage vocal ({ref_count} refs, "
                      f"{p.total_duration:.0f}s)")
            else:
                self.voice_map[spk] = "default"
                print(f"      ⚠️  {spk} : pas de référence audio, voix Qwen3-TTS par défaut")

        self._apply_voice_overrides()

    def _pick_ref_voice(self, gender: str) -> Optional[str]:
        """Sélectionne une voix de référence par genre (round-robin)."""
        if not self.ref_voices:
            return None
        # Essayer le genre exact, sinon fallback sur l'autre genre
        for g in [gender, "male", "female"]:
            voices = self.ref_voices.get(g, [])
            if voices:
                idx = self._ref_voices_counters[g] % len(voices)
                self._ref_voices_counters[g] += 1
                return voices[idx]
        return None

    def _get_best_ref(self, speaker_id: str) -> tuple[str, str]:
        """Retourne (chemin_audio, transcript) du meilleur clip de référence.

        Priorité : ref_voice (--ref-voice) → ref_voices (--ref-voices, par genre)
        → ref_clips extraits → sample_path → vide.

        Qwen3-TTS a besoin du transcript (ref_text) pour le mode ICL complet.
        Sans ref_text, le bridge bascule en x_vector_only (timbre seul,
        similarité ~0.75 vs ~0.89 avec ICL). On fait donc l'effort de
        toujours fournir le transcript quand il est disponible.
        """
        if self.ref_voice:
            return (self.ref_voice, "")
        # Voix genrée pré-assignée par setup_voices
        if hasattr(self, '_speaker_ref_voice') and speaker_id in self._speaker_ref_voice:
            return (self._speaker_ref_voice[speaker_id], "")
        profile = self.profiles.get(speaker_id)
        if not profile:
            profile = next((p for p in self.profiles.values()
                            if p.ref_clips or p.sample_path), None)
        if not profile:
            return ("", "")
        if profile.ref_clips:
            # ref_clips = [(path, text), ...] — le text est le transcript source
            path, ref_text = profile.ref_clips[0]
            return (path, ref_text)
        if profile.sample_path and os.path.exists(profile.sample_path):
            return (profile.sample_path, "")
        return ("", "")

    def _synthesize_single(self, text: str, speaker_id: str,
                           output_path: str) -> str:
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
                # En doublage inter-langues (ex: DE→FR), on ne passe PAS ref_text
                # pour forcer x_vector_only dans le bridge : timbre préservé,
                # accent source éliminé. En même langue, ICL complet.
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
                 "-ar", "44100", "-ac", "1", "-acodec", "pcm_s16le",
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

    @staticmethod
    def _split_text_for_tts(text: str, max_chars: int = QWEN3TTS_MAX_CHARS) -> list[str]:
        """Découpe un texte en morceaux aux frontières naturelles."""
        if len(text) <= max_chars:
            return [text]

        chunks = []
        remaining = text.strip()

        while len(remaining) > max_chars:
            window = remaining[:max_chars]
            cut = -1

            for sep in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
                idx = window.rfind(sep)
                if idx > max_chars // 3:
                    cut = idx + 1
                    break

            if cut < 0:
                for sep in [", ", "; ", ": ", " – ", " — "]:
                    idx = window.rfind(sep)
                    if idx > max_chars // 3:
                        cut = idx + len(sep)
                        break

            if cut < 0:
                idx = window.rfind(" ")
                if idx > max_chars // 4:
                    cut = idx + 1

            if cut < 0:
                cut = max_chars

            chunks.append(remaining[:cut].strip())
            remaining = remaining[cut:].strip()

        if remaining:
            chunks.append(remaining)

        return chunks

    def _synthesize_chunks(self, chunks: list[str], speaker_id: str,
                           output_path: str) -> str:
        """Synthétise une liste de chunks et les concatène."""
        if len(chunks) == 1:
            return self._synthesize_single(chunks[0], speaker_id, output_path)

        import soundfile as sf
        import numpy as np

        chunk_paths = []
        out_dir = os.path.dirname(output_path)
        base = os.path.splitext(os.path.basename(output_path))[0]

        for ci, chunk_text in enumerate(chunks):
            chunk_path = os.path.join(out_dir, f"{base}_part{ci:02d}.wav")
            result = self._synthesize_single(chunk_text, speaker_id, chunk_path)
            if result:
                chunk_paths.append(result)
            else:
                print(f"      ⚠️  Chunk {ci+1}/{len(chunks)} échoué ({len(chunk_text)} chars)")

        if not chunk_paths:
            return ""

        if len(chunk_paths) == 1:
            shutil.move(chunk_paths[0], output_path)
            return output_path

        arrays = []
        sr = None
        for cp in chunk_paths:
            data, file_sr = sf.read(cp, dtype="float32")
            if sr is None:
                sr = file_sr
            arrays.append(data)

        crossfade_samples = min(128, min(len(a) for a in arrays) // 2) if arrays else 0
        gap_samples = int(sr * GAP_BETWEEN_CLIPS_MS / 1000)
        combined = []
        for i, arr in enumerate(arrays):
            if i > 0:
                if crossfade_samples > 0 and len(combined) > 0:
                    prev = combined[-1]
                    fade_out = np.linspace(1.0, 0.0, crossfade_samples, dtype=np.float32)
                    prev[-crossfade_samples:] *= fade_out
                    combined[-1] = prev
                    combined.append(np.zeros(gap_samples, dtype=np.float32))
                    fade_in = np.linspace(0.0, 1.0, crossfade_samples, dtype=np.float32)
                    arr = arr.copy()
                    arr[:crossfade_samples] *= fade_in
                else:
                    combined.append(np.zeros(gap_samples, dtype=np.float32))
            combined.append(arr)

        result = np.concatenate(combined, axis=0)
        sf.write(output_path, result, sr, subtype="PCM_16")

        for cp in chunk_paths:
            if os.path.exists(cp):
                os.remove(cp)

        return output_path

    def synthesize(self, text: str, speaker_id: str, output_path: str,
                   target_duration: float = 0.0) -> str:
        """
        Synthèse vocale Qwen3-TTS avec ajustement temporel.

        Qwen3-TTS n'expose pas de speed natif via l'API Python.
        Si target_duration > 0 : synthèse normale → mesure → atempo pour ajuster.
        """
        text = _clean_for_tts(text, lang=self.target_lang, backend="qwen3tts")
        if not text:
            return ""

        chunks = self._split_text_for_tts(text) if len(text) > QWEN3TTS_MAX_CHARS else [text]

        result = self._synthesize_chunks(chunks, speaker_id, output_path)
        if not result:
            return ""

        if target_duration <= 0:
            return result

        import soundfile as sf
        info = sf.info(result)
        real_dur = info.duration
        if real_dur < 0.1:
            return result

        needed_speed = real_dur / target_duration

        if abs(needed_speed - 1.0) <= QWEN3TTS_SPEED_TOLERANCE:
            return result

        if needed_speed < 1.0:
            # TTS trop court → silence padding, jamais de ralentissement
            _pad_silence(result, target_duration)
            info2 = sf.info(result)
            post_dur = info2.duration
            print(f"      🔇 Two-pass: {real_dur:.2f}s→{post_dur:.2f}s "
                  f"(cible {target_duration:.2f}s, silence +{(target_duration - real_dur)*1000:.0f}ms)")
        else:
            # TTS trop long → atempo modéré
            clamped_speed = min(SPEED_COMFORT_MAX, needed_speed)
            self._apply_atempo(result, clamped_speed)

            info2 = sf.info(result)
            post_dur = info2.duration

            clamp_note = ""
            if abs(clamped_speed - needed_speed) > 0.01:
                clamp_note = f" [clampé, idéal={needed_speed:.2f}]"
            overshoot = ""
            if post_dur > target_duration * 1.05:
                overshoot = f" ⚠️+{(post_dur - target_duration)*1000:.0f}ms"
            print(f"      🏃 Two-pass: {real_dur:.2f}s→{post_dur:.2f}s "
                  f"(cible {target_duration:.2f}s, atempo={clamped_speed:.2f}{clamp_note}{overshoot})")

        return output_path

    def synthesize_with_speed(self, text: str, speaker_id: str,
                              output_path: str, speed: float = 1.0) -> str:
        """Synthèse directe puis atempo (accélération uniquement)."""
        text = _clean_for_tts(text, lang=self.target_lang, backend="qwen3tts")
        if not text:
            return ""

        # Qwen3-TTS n'a pas de speed natif → synthèse à 1.0 + atempo si accélération
        chunks = self._split_text_for_tts(text) if len(text) > QWEN3TTS_MAX_CHARS else [text]
        result = self._synthesize_chunks(chunks, speaker_id, output_path)

        if result and speed > 1.0 and abs(speed - 1.0) > 0.01:
            clamped = min(QWEN3TTS_SPEED_MAX, speed)
            self._apply_atempo(result, clamped)
        # speed < 1.0 → ne pas ralentir (le silence sera ajouté par synthesize)

        return result

    @staticmethod
    def _apply_atempo(wav_path: str, speed: float):
        """Applique ffmpeg atempo in-place sur un fichier WAV."""
        if abs(speed - 1.0) < 0.01:
            return

        tmp_path = wav_path + ".atempo.wav"
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path,
             "-filter:a", f"atempo={speed:.4f}",
             "-ar", "44100", "-ac", "1", "-acodec", "pcm_s16le",
             tmp_path],
            capture_output=True, text=True
        )
        if r.returncode == 0 and os.path.exists(tmp_path):
            shutil.move(tmp_path, wav_path)
        else:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def cleanup(self):
        try:
            self._bridge_call({"cmd": "quit"})
            self._proc.wait(timeout=10)
        except Exception:
            self._proc.kill()


def create_tts_backend(ref_voice: Optional[str] = None,
                       target_lang: str = "fr",
                       source_lang: str = "en",
                       xtts_speaker: Optional[str] = None,
                       backend: str = "xtts",
                       elevenlabs_voice: Optional[str] = None,
                       elevenlabs_model: Optional[str] = None,
                       ref_voices_dir: Optional[str] = None) -> TTSBackend:
    """Crée le backend TTS selon le choix utilisateur."""
    if backend == "qwen3tts":
        return Qwen3TTSBackend(
            target_lang=target_lang,
            ref_voice=ref_voice,
            source_lang=source_lang,
            ref_voices_dir=ref_voices_dir,
        )
    elif backend == "elevenlabs":
        return ElevenLabsBackend(
            target_lang=target_lang,
            ref_voice=ref_voice,
            elevenlabs_voice=elevenlabs_voice,
            elevenlabs_model=elevenlabs_model,
        )
    return XTTSBackend(target_lang=target_lang, ref_voice=ref_voice,
                       xtts_speaker=xtts_speaker)


def _trim_tts_silence(audio_path: str, threshold_db: float = -40.0,
                      min_silence_ms: int = 100, keep_ms: int = 50) -> float:
    """
    Supprime le silence en début et fin d'un clip TTS (in-place).
    XTTS ajoute souvent 200-500ms de silence parasite.

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

    # Calculer l'enveloppe RMS par fenêtre de 10ms
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

    # Trouver le premier frame au-dessus du seuil
    above = np.where(rms_db > threshold_db)[0]
    if len(above) == 0:
        return len(data) / sr  # tout est silence, ne pas toucher

    first_voice = above[0]
    last_voice = above[-1]

    # Vérifier qu'il y a assez de silence à trimmer
    trim_start = max(0, first_voice - keep_frames) * window
    trim_end = min(len(data), (last_voice + 1 + keep_frames) * window)

    # Ne trimmer que si on gagne au moins min_silence_ms
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


def _clean_for_tts(text, lang="fr", backend="xtts"):
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
    # Tirets entourés d'espaces → espace simple
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
    # Tous les backends TTS (XTTS, Qwen3-TTS, ElevenLabs) vocalisent "."
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


def synthesize_all(segments: list[DubSegment], profiles: dict[str, SpeakerProfile],
                   tts: TTSBackend, work_dir: str,
                   lead_in_sec: float = 0.0,
                   audio_only: bool = False) -> list[DubSegment]:
    """
    Synthétise l'audio TTS pour chaque segment avec ajustement two-pass.

    Le two-pass utilise le paramètre speed natif d'XTTS pour que chaque clip
    tienne dans sa fenêtre temporelle. Ça remplace la majeure partie de la
    boucle de réécriture Claude (passe 6c) et élimine le time-stretching.

    lead_in_sec : durée du lead-in voice-over (réduit la fenêtre disponible).
    audio_only  : si True, synthèse single-pass (target_duration=0) — pas de
                  contrainte temporelle, utilisé par le mode --audio-only.
    """
    label = "single-pass" if audio_only else "two-pass"
    print(f"\n🗣️  Passe 6 — Synthèse vocale {label} ({tts.__class__.__name__})...")
    if lead_in_sec > 0 and not audio_only:
        print(f"   ℹ️  Lead-in voice-over : -{lead_in_sec:.1f}s par segment")

    tts.setup_voices(profiles)

    tts_dir = os.path.join(work_dir, "tts_clips")
    os.makedirs(tts_dir, exist_ok=True)

    # ── Invalidation du cache si le mapping voix a changé ────────────────
    voice_map_path = os.path.join(tts_dir, "_voice_map.json")
    current_map = {
        "backend": tts.__class__.__name__,
        "voices": getattr(tts, 'voice_map', {}),
        "instruct": getattr(tts, 'instruct', ''),
        "speakers": {seg.speaker for seg in segments},
        "two_pass": not audio_only,  # invalide le cache des anciennes synthèses sans speed
        "audio_only": audio_only,
    }
    current_map["speakers"] = sorted(current_map["speakers"])
    current_sig = json.dumps(current_map, sort_keys=True, default=str)

    cache_valid = False
    if os.path.exists(voice_map_path):
        try:
            old_sig = open(voice_map_path).read()
            cache_valid = (old_sig == current_sig)
        except Exception:
            pass

    if not cache_valid:
        old_clips = [f for f in os.listdir(tts_dir) if f.endswith(".wav")]
        if old_clips:
            print(f"   🧹 Paramètres TTS modifiés → suppression de {len(old_clips)} clips en cache")
            for f in old_clips:
                os.remove(os.path.join(tts_dir, f))

    with open(voice_map_path, "w") as f:
        f.write(current_sig)

    # ── Pré-calculer la durée cible par segment ─────────────────────────
    # La contrainte réelle : le clip TTS de seg[i] (placé à seg[i].start + lead_in)
    # ne doit pas déborder sur seg[i+1].start. On utilise donc la fenêtre
    # jusqu'au prochain segment, pas juste seg.duration.
    # En mode audio_only, on laisse target_durations vide → synthèse single-pass.
    target_durations: dict[int, float] = {}
    if not audio_only:
        sorted_active = sorted(
            [(i, seg) for i, seg in enumerate(segments) if seg.speech_text and seg.duration >= 0.3],
            key=lambda x: x[1].start
        )

        gap_sec = GAP_BETWEEN_CLIPS_MS / 1000
        for pos, (idx, seg) in enumerate(sorted_active):
            # Fenêtre = temps depuis le début du speech TTS jusqu'au début du segment suivant
            # moins le gap de sécurité inter-clips
            seg_lead_in = lead_in_sec if seg.is_sentence_start else 0.0
            if pos < len(sorted_active) - 1:
                next_start = sorted_active[pos + 1][1].start
                available = next_start - seg.start - seg_lead_in - gap_sec
            else:
                available = seg.duration - seg_lead_in
            target_durations[seg.index] = max(available, 0.5)

    t0 = time.time()
    success = 0
    two_pass_count = 0

    for i, seg in enumerate(segments):
        if not seg.speech_text or seg.duration < 0.3:
            continue

        out_path = os.path.join(tts_dir, f"seg_{seg.index:04d}.wav")

        if os.path.exists(out_path):
            seg.tts_path = out_path
            success += 1
            continue

        target_dur = target_durations.get(seg.index, 0.0)
        result = tts.synthesize(seg.speech_text, seg.speaker, out_path,
                                target_duration=target_dur)
        if result:
            seg.tts_path = result
            success += 1
            # Compter les two-pass effectifs (le print "Two-pass:" vient de synthesize())
            actual = _get_clip_duration(result)
            if target_dur > 0 and abs(actual / target_dur - 1.0) > XTTS_SPEED_TOLERANCE:
                two_pass_count += 1
            max_chars = ELEVENLABS_MAX_CHARS if isinstance(tts, ElevenLabsBackend) else XTTS_MAX_CHARS
            if len(seg.speech_text) > max_chars:
                n_parts = len(tts._split_text_for_tts(seg.speech_text))
                print(f"      ✂️  seg {seg.index} : {len(seg.speech_text)} chars → {n_parts} parties")

        # Progress
        if (i + 1) % 20 == 0 or i == len(segments) - 1:
            print(f"   🗣️  {success}/{i+1} segments synthétisés...")

    elapsed = time.time() - t0
    print(f"   ✅ {success}/{len(segments)} clips TTS ({elapsed:.0f}s)")
    if two_pass_count:
        print(f"   🏃 {two_pass_count} segments ajustés par two-pass speed")

    # Trimmer tous les clips — supprime le silence parasite XTTS début/fin
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
# PASSE 6c : VÉRIFICATION TEMPORELLE POST-SYNTHÈSE
# ═══════════════════════════════════════════════════════════════════════════════

def _pad_silence(wav_path: str, target_duration: float):
    """Ajoute du silence en fin de WAV pour atteindre target_duration."""
    import soundfile as sf
    info = sf.info(wav_path)
    pad_sec = target_duration - info.duration
    if pad_sec <= 0:
        return
    tmp_path = wav_path + ".padded.wav"
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path,
         "-af", f"apad=pad_dur={pad_sec:.4f}",
         "-ar", "44100", "-ac", "1", "-acodec", "pcm_s16le",
         tmp_path],
        capture_output=True, text=True
    )
    if r.returncode == 0 and os.path.exists(tmp_path):
        shutil.move(tmp_path, wav_path)
    else:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _get_clip_duration(path: str) -> float:
    """Retourne la durée en secondes d'un fichier audio."""
    import soundfile as sf
    try:
        info = sf.info(path)
        return info.duration
    except Exception:
        return 0.0


def _shorten_texts_claude(items: list[tuple], client, tgt_lang: str) -> dict[int, str]:
    """
    Demande à Claude de raccourcir les textes des segments qui chevauchent
    le segment suivant.
    items: [(seg_index, current_text, target_chars, actual_chars, overflow_ms)]
    Retourne {seg_index: shortened_text}
    """
    tgt_n = lang_name(tgt_lang)
    lines = []
    for idx, text, target_chars, actual_chars, overflow_ms in items:
        lines.append(f"[{idx}] (actuellement {actual_chars} car., cible ≈{target_chars} car., "
                     f"déborde de {overflow_ms:.0f}ms) {text}")

    prompt = f"""Tu es un adaptateur de doublage en {tgt_n} pour un reportage télé.
Ces segments sont trop longs : ils chevauchent le segment suivant.
Raccourcis chaque texte vers la cible indiquée SANS casser la grammaire ni
gommer la portée éditoriale.

Le texte sera PRONONCÉ à voix haute — il doit rester fluide et naturel à l'oral.

PRIORITÉ : grammaire et sens d'abord, durée ensuite. Si tu ne peux pas atteindre
la cible chiffrée sans mutiler la phrase, vise une cible un peu plus haute —
le TTS speed (jusqu'à ×1.40) absorbera le reste.

COMMENT RACCOURCIR (par ordre de préférence) :
1. Couper un détail SECONDAIRE entier (subordonnée, exemple, parenthèse, redite)
2. Reformuler un groupe verbal en plus direct (« est en train de » → « fait »)
3. Synonyme plus court à mot-à-mot équivalent
4. Supprimer une partie redondante d'une énumération
NE RABOTE PAS les mots-outils pour gagner 1-2 caractères.

DOIVENT ÊTRE PRÉSERVÉS (ne JAMAIS supprimer pour gagner de la place) :
- Articles et déterminants : « le, la, les, un, une, des, du, de la, ce, ces… »
  Après une préposition (avec, chez, par, dans, pour, sous, entre, parmi…), l'article reste.
- Mots de liaison : « et, mais, donc, alors, car, parce que, c'est que… »
- Pronoms : « on, ça, vous, ils, c'est… »
- Adverbes-qualifieurs éditoriaux : « soi-disant, prétendument, potentiellement,
  vraiment, probablement, particulièrement, principalement, éventuellement,
  approximativement, apparemment ». Ces mots portent le cadrage de l'auteur.
- Sujet + verbe conjugué : chaque phrase doit en avoir. Pas de fragment nominal.

INTERDIT :
- Changer le sens fondamental ou la portée éditoriale
- Couper une phrase en deux ou fusionner
- Style TÉLÉGRAPHIQUE / fragment nominal
- Supprimer un mot de la liste « DOIVENT ÊTRE PRÉSERVÉS »

EXEMPLES À NE PAS REPRODUIRE :
❌ « les gens avec troubles mentaux »   → ✅ « les gens avec des troubles mentaux »
❌ « chez adultes sains »               → ✅ « chez des adultes sains »
❌ « idée fausse mortelle »             → ✅ « idée fausse potentiellement mortelle »
❌ « médicaments de transition »        → ✅ « soi-disant médicaments de transition »
❌ « Faux. » / « Un malade. »           → ✅ phrase complète avec sujet+verbe

PONCTUATION : conserve une ponctuation soignée (virgules de respiration,
points de suspension, deux-points si présents dans l'original).

Appelle l'outil submit_texts avec un item par segment à raccourcir.
Chaque text ne contient que le texte raccourci — pas de commentaire, pas
de méta-discussion, pas de préfixe.

{chr(10).join(lines)}"""

    raw = _claude_submit_texts(client, prompt)

    results = {}
    orig_texts = {idx: text for idx, text, _, _, _ in items}
    for idx, txt in raw.items():
        if _est_fuite_prompt(txt, orig_texts.get(idx, "")):
            print(f"   ⚠️  Fuite de prompt détectée seg [{idx}], ignoré : {txt[:60]}…")
            continue
        results[idx] = txt
    return results


def verify_and_fix_timing(segments: list, tts, client,
                          tgt_lang: str, work_dir: str,
                          max_rounds: int = 5,
                          overlap_tolerance_ms: float = 150,
                          lead_in_sec: float = 0.0) -> list:
    """
    Passe 6c — Vérifie les chevauchements réels entre clips TTS consécutifs.

    Stratégie en 4 étapes :
      1. Round 1 : ajuster la vitesse XTTS (jusqu'à SPEED_RESCUE_MAX=1.40)
         → gratuit (pas d'appel API), qualité préservée par le modèle
      2. Round 2+ : réécriture Claude (cible ~95% longueur) + re-synthèse two-pass
      3. Round final agressif : réécriture drastique (cible 70% longueur)
         + re-synthèse à speed rescue max
      4. Troncature de secours : les clips encore trop longs sont tronqués
         pour garantir zéro chevauchement dans le mix final.

    lead_in_sec : offset voice-over (les clips sont placés à seg.start + lead_in
    dans le mix final, donc la fenêtre réelle est réduite d'autant).
    """
    print(f"\n🔍 Passe 6c — Vérification des chevauchements (speed-first)...")

    # Segments actifs triés par timestamp
    active = sorted(
        [(i, seg) for i, seg in enumerate(segments)
         if seg.tts_path and os.path.exists(seg.tts_path)],
        key=lambda x: x[1].start
    )

    if len(active) < 2:
        print("   ✅ Moins de 2 clips TTS — rien à vérifier")
        return segments

    def _detect_overlaps():
        """Détecte les chevauchements entre clips consécutifs.
        Tolérance adaptative : 10% de la durée du segment, plafonnée à overlap_tolerance_ms."""
        overlaps = []
        for pos in range(len(active) - 1):
            _, seg = active[pos]
            tts_dur = _get_clip_duration(seg.tts_path)
            if tts_dur < 0.1:
                continue
            seg_lead_in = lead_in_sec if seg.is_sentence_start else 0.0
            tts_end = seg.start + seg_lead_in + tts_dur
            next_start = active[pos + 1][1].start
            overflow_ms = max(0.0, (tts_end - next_start) * 1000)
            # Tolérance proportionnelle à la durée du segment (segments courts = moins de tolérance)
            seg_dur_ms = (next_start - seg.start) * 1000
            adaptive_tolerance = min(overlap_tolerance_ms, seg_dur_ms * 0.10)
            if overflow_ms > adaptive_tolerance:
                overlaps.append((seg, tts_dur, overflow_ms, next_start))
        return overlaps

    # ── Bilan initial ────────────────────────────────────────────────────
    overlaps = _detect_overlaps()
    if not overlaps:
        print(f"   ✅ Aucun chevauchement détecté")
        return segments

    print(f"   ⚠️  {len(overlaps)} chevauchements détectés (>{overlap_tolerance_ms:.0f}ms)")

    # ══════════════════════════════════════════════════════════════════════
    # ROUND 1 : AJUSTEMENT PAR SPEED NATIF XTTS (pas d'appel Claude)
    # ══════════════════════════════════════════════════════════════════════
    speed_fixed = 0
    speed_failed = []

    for seg, tts_dur, overflow_ms, next_start in overlaps:
        text = seg.text_adapted or seg.text_tgt or ""
        if not text:
            continue

        # Fenêtre disponible (avec marge pour éviter le chevauchement + gap inter-clips)
        seg_lead_in = lead_in_sec if seg.is_sentence_start else 0.0
        available = next_start - seg.start - seg_lead_in - (overlap_tolerance_ms / 2000) - GAP_BETWEEN_CLIPS_MS / 1000
        available = max(available, 0.3)

        # Vitesse nécessaire pour tenir dans la fenêtre
        needed_speed = tts_dur / available

        rescue_max = getattr(tts, 'speed_rescue_max', XTTS_SPEED_RESCUE_MAX)
        if needed_speed <= rescue_max:
            # ── Résolvable par speed seul ────────────────────────────
            clamped = min(needed_speed, rescue_max)
            result = tts.synthesize_with_speed(text, seg.speaker,
                                               seg.tts_path, speed=clamped)
            if result:
                new_dur = _get_clip_duration(result)
                print(f"      🏃 seg {seg.index}: speed={clamped:.2f} → "
                      f"{tts_dur:.2f}s→{new_dur:.2f}s (fenêtre {available:.2f}s)")
                speed_fixed += 1
            else:
                speed_failed.append((seg, tts_dur, overflow_ms, next_start))
        else:
            # Speed insuffisant → on collecte pour le round Claude
            speed_failed.append((seg, tts_dur, overflow_ms, next_start))

    print(f"   🏃 Round 1 (speed): {speed_fixed} corrigés, "
          f"{len(speed_failed)} nécessitent réécriture")

    if not speed_failed:
        # Vérification finale
        remaining = _detect_overlaps()
        if remaining:
            print(f"   ⚠️  {len(remaining)} chevauchements résiduels (tolérés)")
        else:
            print(f"   ✅ Tous les chevauchements résolus par speed")
        return segments

    # ══════════════════════════════════════════════════════════════════════
    # ROUND 2+ : RÉÉCRITURE CLAUDE + RE-SYNTHÈSE TWO-PASS
    # (seulement pour les cas où speed > RESCUE_MAX)
    # ══════════════════════════════════════════════════════════════════════
    for round_num in range(2, max_rounds + 1):
        still_bad = _detect_overlaps()

        if not still_bad:
            print(f"   ✅ Round {round_num}: tous les chevauchements résolus")
            break

        print(f"   🔄 Round {round_num}: {len(still_bad)} chevauchements → réécriture Claude")

        # CPS réel du TTS pour calibrer les cibles
        CPS_TTS: dict[str, list[float]] = {}
        for _, seg in active:
            tts_dur = _get_clip_duration(seg.tts_path)
            text = seg.text_adapted or seg.text_tgt or ""
            if text and tts_dur > 0.1:
                CPS_TTS.setdefault(seg.speaker, []).append(len(text) / tts_dur)

        cps_avg = {spk: sum(v)/len(v) for spk, v in CPS_TTS.items()}
        all_cps = [v for vals in CPS_TTS.values() for v in vals]
        global_cps = (sum(all_cps) / len(all_cps)) if all_cps else 14.0

        # Préparer les demandes de raccourcissement
        rewrite_items = []
        for seg, tts_dur, overflow_ms, next_start in still_bad:
            text = seg.text_adapted or seg.text_tgt
            if not text:
                continue
            seg_lead_in = lead_in_sec if seg.is_sentence_start else 0.0
            available_dur = next_start - seg.start - seg_lead_in - (overlap_tolerance_ms / 2000) - GAP_BETWEEN_CLIPS_MS / 1000
            available_dur = max(available_dur, 0.5)
            spk_cps = cps_avg.get(seg.speaker, global_cps)
            # Viser un speed de ~1.15 après réécriture (confortable pour XTTS)
            target_chars = int(available_dur * spk_cps * 1.15 * 0.95)
            target_chars = max(target_chars, 5)
            rewrite_items.append((seg.index, text, target_chars, len(text), overflow_ms))

        if not rewrite_items:
            break

        # Appel Claude par batch + re-synthèse with two-pass
        BATCH = 40
        rewritten = 0
        for bi in range(0, len(rewrite_items), BATCH):
            batch = rewrite_items[bi:bi+BATCH]
            shortened = _shorten_texts_claude(batch, client, tgt_lang)

            for seg, _, overflow_ms, next_start in still_bad:
                if seg.index in shortened:
                    new_text = shortened[seg.index]
                    seg.text_adapted = new_text

                    # Re-synthèse avec two-pass (speed auto-ajusté, trim inclus)
                    seg_lead_in = lead_in_sec if seg.is_sentence_start else 0.0
                    available = next_start - seg.start - seg_lead_in - (overlap_tolerance_ms / 2000) - GAP_BETWEEN_CLIPS_MS / 1000
                    result = tts.synthesize(new_text, seg.speaker, seg.tts_path,
                                           target_duration=max(available, 0.5))
                    if result:
                        rewritten += 1

        print(f"      ✏️  {rewritten} segments réécrits + re-synthétisés (two-pass)")

    # ══════════════════════════════════════════════════════════════════════
    # ROUND FINAL AGRESSIF : réécriture drastique pour les cas irrésolus
    # Vise 70% de la longueur actuelle + speed rescue max
    # ══════════════════════════════════════════════════════════════════════
    for rescue_round in range(2):
        final_overlaps = _detect_overlaps()
        if not final_overlaps:
            break

        print(f"\n   🔥 Round agressif {rescue_round + 1}/2 : {len(final_overlaps)} chevauchements résiduels "
              f"→ réécriture drastique")

        # Recalculer CPS
        CPS_TTS_final: dict[str, list[float]] = {}
        for _, seg in active:
            dur = _get_clip_duration(seg.tts_path)
            text = seg.text_adapted or seg.text_tgt or ""
            if text and dur > 0.1:
                CPS_TTS_final.setdefault(seg.speaker, []).append(len(text) / dur)
        cps_avg_final = {spk: sum(v)/len(v) for spk, v in CPS_TTS_final.items()}
        all_cps_final = [v for vals in CPS_TTS_final.values() for v in vals]
        global_cps_final = (sum(all_cps_final) / len(all_cps_final)) if all_cps_final else 14.0

        rescue_items = []
        for seg, tts_dur, overflow_ms, next_start in final_overlaps:
            text = seg.text_adapted or seg.text_tgt
            if not text:
                continue
            seg_lead_in = lead_in_sec if seg.is_sentence_start else 0.0
            available_dur = next_start - seg.start - seg_lead_in - GAP_BETWEEN_CLIPS_MS / 1000
            available_dur = max(available_dur, 0.3)
            spk_cps = cps_avg_final.get(seg.speaker, global_cps_final)
            # Cible agressive : viser speed=1.0 avec 70% de marge
            target_chars = int(available_dur * spk_cps * 0.70)
            target_chars = max(target_chars, 3)
            rescue_items.append((seg.index, text, target_chars, len(text), overflow_ms))

        if not rescue_items:
            break

        # Réécriture par batch
        BATCH = 40
        rescued = 0
        for bi in range(0, len(rescue_items), BATCH):
            batch = rescue_items[bi:bi + BATCH]
            shortened = _shorten_texts_claude(batch, client, tgt_lang)

            for seg, tts_dur, overflow_ms, next_start in final_overlaps:
                if seg.index in shortened:
                    new_text = shortened[seg.index]
                    seg.text_adapted = new_text

                    # Re-synthèse à speed rescue max pour garantir la durée
                    seg_lead_in = lead_in_sec if seg.is_sentence_start else 0.0
                    available = next_start - seg.start - seg_lead_in - GAP_BETWEEN_CLIPS_MS / 1000
                    result = tts.synthesize(new_text, seg.speaker, seg.tts_path,
                                           target_duration=max(available, 0.3))
                    if result:
                        rescued += 1

        print(f"      🔥 {rescued} segments raccourcis drastiquement + re-synthétisés")

    # ══════════════════════════════════════════════════════════════════════
    # DERNIER RECOURS : on N'EFFECTUE PLUS de troncature destructive du WAV.
    # Les clips qui débordent encore sont signalés ici puis gérés par le
    # mixage voice-over (push-later cascade dans mix_audio_voiceover), qui
    # décale les clips suivants au lieu de couper le mot en cours.
    # ══════════════════════════════════════════════════════════════════════
    final_overlaps = _detect_overlaps()
    if final_overlaps:
        for seg, tts_dur, overflow_ms, _next_start in final_overlaps[:5]:
            print(f"      ⚠️  seg {seg.index}: déborde de {overflow_ms:.0f}ms "
                  f"(TTS {tts_dur:.2f}s) — sera décalé au mixage")
        if len(final_overlaps) > 5:
            print(f"      ⚠️  ... et {len(final_overlaps) - 5} autres")
        print(f"   ➡️  {len(final_overlaps)} clips débordants — résolution déléguée au mixage")

    # ── Bilan final ──────────────────────────────────────────────────────
    remaining = _detect_overlaps()
    if remaining:
        print(f"   ⚠️  {len(remaining)} chevauchements résiduels après tous les rounds")
    else:
        print(f"   ✅ Tous les chevauchements résolus (zéro superposition)")

    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 6b : NORMALISATION PROSODIQUE (COHÉRENCE VOCALE)
# ═══════════════════════════════════════════════════════════════════════════════

# Seuils de normalisation prosodique (WORLD)
PROSODY_F0_TOLERANCE = 0.5     # ±0.5 demi-ton → pas de correction F0 médian
PROSODY_RANGE_TOLERANCE = 0.15 # ±15% de la plage F0 source → pas de rescale
PROSODY_RMS_TOLERANCE = 2.0    # ±2 dB du RMS médian → pas de correction
PROSODY_MIN_CLIP_SEC = 0.5     # ignorer les clips < 0.5s
PROSODY_F0_MAX_SHIFT_ST = 3.0  # correction max ±3 demi-tons
PROSODY_RANGE_MAX_SCALE = 1.8  # compression/expansion max de la plage F0


def _extract_f0_parselmouth(audio_mono: 'np.ndarray', sr: int,
                             f0_floor: float = 70, f0_ceil: float = 500
                             ) -> tuple['np.ndarray', float, float, float]:
    """
    Extrait la courbe F0 avec Praat (parselmouth) — gold standard en phonétique.

    Retourne :
        f0_array  : courbe F0 brute (0 = unvoiced)
        f0_median : F0 médian des zones voisées (Hz)
        f0_std    : écart-type F0 des zones voisées (Hz)
        voiced_fraction : fraction du signal qui est voisée
    """
    import parselmouth
    import numpy as np

    snd = parselmouth.Sound(audio_mono, sampling_frequency=sr)
    pitch = snd.to_pitch_ac(
        time_step=0.01,          # 10ms hop
        pitch_floor=f0_floor,
        pitch_ceiling=f0_ceil,
        very_accurate=True,
    )

    f0_array = pitch.selected_array['frequency']  # 0 = unvoiced
    voiced = f0_array[f0_array > 0]

    if len(voiced) < 3:
        return f0_array, 0.0, 0.0, 0.0

    f0_median = float(np.median(voiced))
    f0_std = float(np.std(voiced))
    voiced_fraction = len(voiced) / max(len(f0_array), 1)

    return f0_array, f0_median, f0_std, voiced_fraction


def _world_analyze(audio_mono: 'np.ndarray', sr: int,
                   f0_floor: float = 70, f0_ceil: float = 500
                   ) -> tuple['np.ndarray', 'np.ndarray', 'np.ndarray', float]:
    """
    Décompose le signal avec WORLD vocoder en 3 composantes indépendantes :
        - f0         : courbe de fréquence fondamentale (Hz, 0 = unvoiced)
        - spectral   : enveloppe spectrale (timbre/formants)
        - aperiodic  : apériodicité (souffle/bruit)
        - frame_period : période entre frames (ms)
    """
    import pyworld as pw
    import numpy as np

    audio_f64 = audio_mono.astype(np.float64)

    # Harvest : extraction F0 robuste
    f0, timeaxis = pw.harvest(audio_f64, sr,
                              f0_floor=f0_floor, f0_ceil=f0_ceil,
                              frame_period=5.0)  # 5ms pour haute résolution

    # Stonemask : raffinement du F0
    f0 = pw.stonemask(audio_f64, f0, timeaxis, sr)

    # CheapTrick : enveloppe spectrale
    spectral = pw.cheaptrick(audio_f64, f0, timeaxis, sr)

    # D4C : apériodicité
    aperiodic = pw.d4c(audio_f64, f0, timeaxis, sr)

    return f0, spectral, aperiodic, 5.0


def _world_synthesize(f0: 'np.ndarray', spectral: 'np.ndarray',
                      aperiodic: 'np.ndarray', sr: int,
                      frame_period: float = 5.0) -> 'np.ndarray':
    """Resynthétise le signal à partir des composantes WORLD."""
    import pyworld as pw
    import numpy as np

    audio = pw.synthesize(f0, spectral, aperiodic, sr, frame_period=frame_period)
    return audio.astype(np.float32)


def _shift_f0_world(f0: 'np.ndarray', semitones: float) -> 'np.ndarray':
    """
    Décale le F0 de N demi-tons sans toucher au timbre/formants.
    Ne modifie que les frames voisées (f0 > 0).
    """
    import numpy as np
    f0_shifted = f0.copy()
    voiced = f0_shifted > 0
    f0_shifted[voiced] *= 2.0 ** (semitones / 12.0)
    return f0_shifted


def _rescale_f0_range(f0: 'np.ndarray', target_std: float,
                      current_median: float) -> 'np.ndarray':
    """
    Compresse ou étend la plage dynamique du F0 autour du médian.

    Transforme en espace log (perceptuellement linéaire), scale la variance,
    reconvertit. Préserve le contour prosodique tout en ajustant l'expressivité.
    """
    import numpy as np

    f0_out = f0.copy()
    voiced = f0_out > 0

    if np.sum(voiced) < 3 or current_median <= 0 or target_std <= 0:
        return f0_out

    log_f0 = np.log2(f0_out[voiced])
    log_median = np.log2(current_median)
    current_log_std = np.std(log_f0)

    if current_log_std < 1e-6:
        return f0_out

    # Ratio de compression/expansion
    scale = target_std / current_log_std
    scale = max(1.0 / PROSODY_RANGE_MAX_SCALE, min(PROSODY_RANGE_MAX_SCALE, scale))

    # Recentrer, scaler, décentrer
    log_f0_scaled = log_median + (log_f0 - log_median) * scale
    f0_out[voiced] = 2.0 ** log_f0_scaled

    return f0_out


def _compute_rms_db(audio_mono: 'np.ndarray') -> float:
    """Calcule le RMS en dB d'un signal audio mono."""
    import numpy as np
    rms = np.sqrt(np.mean(audio_mono ** 2))
    if rms < 1e-10:
        return -100.0
    return 20 * np.log10(rms)


def normalize_prosody(segments: list[DubSegment], work_dir: str,
                      profiles: Optional[dict[str, SpeakerProfile]] = None,
                      fix_pitch: bool = False
                      ) -> list[DubSegment]:
    """
    Passe 6b — Normalisation prosodique.

    Par défaut : normalisation RMS seule (cohérence de volume inter-segments).
    Simple multiplication d'amplitude — zéro re-synthèse, zéro dégradation.

    Avec fix_pitch=True : correction F0 via WORLD vocoder + parselmouth.
    Décompose chaque clip TTS en composantes WORLD (F0, spectre, apériodicité),
    modifie le F0 (shift médian + rescale plage), puis resynthétise.
    Meilleure cohérence de pitch entre segments, mais re-synthèse WORLD
    qui peut dégrader la qualité sur des voix TTS.
    """
    import soundfile as sf
    import numpy as np

    mode_label = "WORLD F0 + RMS" if fix_pitch else "RMS seul"
    print(f"\n🎛️  Passe 6b — Normalisation prosodique ({mode_label})...")
    t0 = time.time()

    norm_dir = os.path.join(work_dir, "tts_normalized")
    os.makedirs(norm_dir, exist_ok=True)

    # ── 1. Analyser tous les clips TTS par locuteur ──────────────────────
    speaker_clips: dict[str, list[dict]] = {}

    for seg in segments:
        if not seg.tts_path or not os.path.exists(seg.tts_path):
            continue

        audio, sr = sf.read(seg.tts_path)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        dur = len(audio) / sr
        if dur < PROSODY_MIN_CLIP_SEC:
            continue

        clip_info = {
            "seg": seg,
            "audio": audio,
            "sr": sr,
            "rms_db": _compute_rms_db(audio),
            "dur": dur,
        }

        # Analyse F0 uniquement si --fix-pitch
        if fix_pitch:
            f0_curve, f0_med, f0_std, voiced_frac = _extract_f0_parselmouth(audio, sr)
            clip_info["f0_median"] = f0_med
            clip_info["f0_std"] = f0_std
            clip_info["f0_std_log"] = float(np.std(np.log2(f0_curve[f0_curve > 0]))) if np.any(f0_curve > 0) else 0.0
            clip_info["voiced_frac"] = voiced_frac

        speaker_clips.setdefault(seg.speaker, []).append(clip_info)

    if not speaker_clips:
        print("   ⚠️  Aucun clip à normaliser")
        return segments

    # ── 2. Traitement par locuteur ───────────────────────────────────────
    pitch_shifts = 0
    range_adjusts = 0
    gain_adjusts = 0
    skipped = 0

    for spk_id, clips in speaker_clips.items():
        rms_values = [c["rms_db"] for c in clips if c["rms_db"] > -80]

        if not rms_values:
            print(f"   ⚠️  {spk_id} : pas assez de données → copie brute")
            for c in clips:
                out_path = os.path.join(norm_dir, f"seg_{c['seg'].index:04d}.wav")
                sf.write(out_path, c["audio"], c["sr"])
                c["seg"].tts_path = out_path
            continue

        target_rms_db = float(np.median(rms_values))
        rms_range = max(rms_values) - min(rms_values) if len(rms_values) > 1 else 0

        # ── Cibles F0 (uniquement si --fix-pitch) ────────────────────
        target_f0 = 0.0
        target_f0_std_log = None
        target_label = ""

        if fix_pitch:
            tts_f0_values = [c["f0_median"] for c in clips if c["f0_median"] > 0]

            if not tts_f0_values:
                fix_pitch_spk = False  # pas de F0 → RMS seul pour ce speaker
            else:
                fix_pitch_spk = True

                # Cible F0 : profil source si disponible, sinon médian TTS
                source_f0 = 0.0
                source_f0_std_log = 0.0
                if profiles and spk_id in profiles:
                    source_f0 = profiles[spk_id].f0_median

                if source_f0 > 0:
                    target_f0 = source_f0
                    target_label = f"source={source_f0:.0f}Hz"
                else:
                    target_f0 = float(np.median(tts_f0_values))
                    target_label = f"médian TTS={target_f0:.0f}Hz"

                # Cible de plage F0 depuis le sample source
                if source_f0 > 0 and profiles and spk_id in profiles:
                    p = profiles[spk_id]
                    if p.sample_path and os.path.exists(p.sample_path):
                        src_audio, src_sr = sf.read(p.sample_path)
                        if src_audio.ndim == 2:
                            src_audio = src_audio.mean(axis=1)
                        src_f0_curve = _extract_f0_parselmouth(src_audio, src_sr)[0]
                        src_voiced = src_f0_curve[src_f0_curve > 0]
                        if len(src_voiced) > 3:
                            source_f0_std_log = float(np.std(np.log2(src_voiced)))

                target_f0_std_log = source_f0_std_log if source_f0_std_log > 0 else None

                tts_f0_med = float(np.median(tts_f0_values))
                tts_f0_range = max(tts_f0_values) - min(tts_f0_values) if len(tts_f0_values) > 1 else 0

                print(f"   📊 {spk_id} ({len(clips)} clips) : "
                      f"cible F0={target_label}, "
                      f"TTS médian={tts_f0_med:.0f}Hz (Δ={tts_f0_range:.0f}Hz), "
                      f"RMS={target_rms_db:.1f}dB (Δ={rms_range:.1f}dB)"
                      + (f", plage log σ_src={source_f0_std_log:.3f}" if source_f0_std_log > 0 else ""))
        else:
            fix_pitch_spk = False
            print(f"   📊 {spk_id} ({len(clips)} clips) : "
                  f"RMS={target_rms_db:.1f}dB (Δ={rms_range:.1f}dB)")

        # ── 3. Correction clip par clip ──────────────────────────────────
        for c in clips:
            seg = c["seg"]
            audio = c["audio"]
            sr = c["sr"]
            clip_rms = c["rms_db"]
            modified = False

            out_path = os.path.join(norm_dir, f"seg_{seg.index:04d}.wav")

            # ── A. Correction F0 via WORLD (seulement si --fix-pitch) ──
            if fix_pitch_spk and target_f0 > 0:
                clip_f0_med = c["f0_median"]
                clip_f0_std_log = c["f0_std_log"]

                if clip_f0_med > 0:
                    try:
                        f0_w, sp, ap, fp = _world_analyze(audio, sr)

                        # A1. Shift F0 médian vers la cible
                        semitones = 12.0 * math.log2(target_f0 / clip_f0_med)
                        semitones = max(-PROSODY_F0_MAX_SHIFT_ST,
                                       min(PROSODY_F0_MAX_SHIFT_ST, semitones))

                        if abs(semitones) > PROSODY_F0_TOLERANCE:
                            f0_w = _shift_f0_world(f0_w, semitones)
                            pitch_shifts += 1
                            modified = True

                        # A2. Rescale plage F0 (expressivité)
                        if (target_f0_std_log is not None
                                and clip_f0_std_log > 0
                                and target_f0_std_log > 0):
                            ratio = target_f0_std_log / clip_f0_std_log
                            if abs(ratio - 1.0) > PROSODY_RANGE_TOLERANCE:
                                voiced_w = f0_w[f0_w > 0]
                                new_median = float(np.median(voiced_w)) if len(voiced_w) > 0 else target_f0
                                f0_w = _rescale_f0_range(f0_w, target_f0_std_log, new_median)
                                range_adjusts += 1
                                modified = True

                        # A3. Resynthèse WORLD (F0 modifié, timbre intact)
                        if modified:
                            audio = _world_synthesize(f0_w, sp, ap, sr, fp)
                            orig_len = len(c["audio"])
                            if len(audio) > orig_len:
                                # Micro-fondu avant troncature pour éviter un clic
                                fade_n = min(int(sr * 0.005), len(audio) - orig_len)
                                if fade_n > 0:
                                    audio[orig_len - fade_n:orig_len] *= np.linspace(1, 0, fade_n)
                                audio = audio[:orig_len]
                            elif len(audio) < orig_len:
                                audio = np.pad(audio, (0, orig_len - len(audio)))

                    except Exception as e:
                        print(f"      ⚠️  seg {seg.index} WORLD échoué : {e}")

            # ── B. Gain RMS (toujours) ──
            if clip_rms > -80 and target_rms_db > -80:
                rms_diff = target_rms_db - clip_rms
                if abs(rms_diff) > PROSODY_RMS_TOLERANCE:
                    gain_linear = 10 ** (rms_diff / 20)
                    gain_linear = max(0.25, min(4.0, gain_linear))
                    audio = audio * gain_linear

                    peak = np.max(np.abs(audio))
                    if peak > 0.98:
                        audio = audio * (0.98 / peak)

                    gain_adjusts += 1
                    modified = True

            if not modified:
                skipped += 1

            sf.write(out_path, audio, sr)
            seg.tts_path = out_path

    elapsed = time.time() - t0
    total = sum(len(clips) for clips in speaker_clips.values())
    if fix_pitch:
        print(f"   ✅ {total} clips : "
              f"{pitch_shifts} F0-shiftés (WORLD), "
              f"{range_adjusts} plage-ajustés, "
              f"{gain_adjusts} gain-ajustés, "
              f"{skipped} inchangés ({elapsed:.1f}s)")
    else:
        print(f"   ✅ {total} clips : "
              f"{gain_adjusts} gain-ajustés, "
              f"{skipped} inchangés ({elapsed:.1f}s)")

    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 7 : MIXAGE AUDIO
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_overlaps_push_later(segments: list) -> int:
    """
    Cascade « push-later » anti-troncature.

    Pour chaque clip TTS qui déborderait sur le suivant, on décale ce dernier
    plus tard (au lieu de tronquer le WAV). La cascade s'éponge à la première
    pause naturelle ; plafonnée à VO_MAX_DRIFT_MS pour éviter une dérive lourde.

    Écrit le résultat dans seg._tts_place_ms (en millisecondes).
    Retourne le nombre de clips décalés.
    """
    active = []
    for seg in segments:
        if seg.tts_path and os.path.exists(seg.tts_path):
            dur_ms = _get_clip_duration(seg.tts_path) * 1000
            if dur_ms >= 50:
                start_ms = getattr(seg, '_tts_place_ms', int(seg.start * 1000))
                active.append([seg, int(start_ms), dur_ms])
    active.sort(key=lambda x: x[1])

    pushed = 0
    max_drift = 0
    skipped = 0
    for i in range(1, len(active)):
        seg, tts_start, tts_dur = active[i]
        _, prev_start, prev_dur = active[i - 1]
        prev_tts_end = prev_start + prev_dur
        required_start = int(prev_tts_end + GAP_BETWEEN_CLIPS_MS)
        if tts_start >= required_start:
            continue
        natural_start = int(seg.start * 1000)
        drift = required_start - natural_start
        if drift > VO_MAX_DRIFT_MS:
            skipped += 1
            continue
        seg._tts_place_ms = required_start
        active[i] = [seg, required_start, tts_dur]
        max_drift = max(max_drift, drift)
        pushed += 1

    if pushed:
        print(f"   ➡️  {pushed} clips décalés en aval "
              f"(anti-troncature, dérive max {max_drift}ms)")
    if skipped:
        print(f"   ⚠️  {skipped} clips toujours en chevauchement "
              f"(dérive > {VO_MAX_DRIFT_MS}ms — overlap toléré)")
    return pushed


def mix_audio(segments: list[DubSegment], background_path: str,
              vocals_path: str, output_path: str,
              keep_original: float = 0.0) -> str:
    """
    Mixe les voix synthétisées sur le fond sonore :
      1. Charge le fond (no_vocals de Demucs)
      2. Optionnel : blend léger de la voix originale (keep_original)
      3. Applique un ducking (atténuation) pendant les passages doublés
      4. Overlay les clips TTS aux bons timestamps
      5. Exporte en WAV haute qualité
    """
    from pydub import AudioSegment
    import numpy as np

    print(f"\n🎚️  Passe 7 — Mixage audio...")
    t0 = time.time()

    # Résoudre les éventuels chevauchements TTS par décalage en aval
    # (remplace l'ancienne troncature destructive).
    _resolve_overlaps_push_later(segments)

    # Charger le fond sonore
    bg = AudioSegment.from_wav(background_path)
    print(f"   📀 Fond sonore : {len(bg)/1000:.1f}s, {bg.frame_rate}Hz")

    # Optionnel : mélanger un peu de la voix originale (pour ambiance)
    if keep_original > 0 and os.path.exists(vocals_path):
        orig_vocals = AudioSegment.from_wav(vocals_path)
        # Réduire le volume des voix originales
        orig_db_reduction = -20 * math.log10(max(keep_original, 0.01))
        orig_vocals = orig_vocals - orig_db_reduction
        bg = bg.overlay(orig_vocals)
        print(f"   🔈 Voix originale conservée à {keep_original*100:.0f}%")

    # Pré-calculer les zones de parole pour le ducking
    # On utilise _tts_place_ms (positions ajustées par le push-later) pour
    # que le ducking couvre la zone réellement parlée par la voix doublée.
    speech_zones = []
    for seg in segments:
        if seg.tts_path and os.path.exists(seg.tts_path):
            tts_start_ms = getattr(seg, '_tts_place_ms', int(seg.start * 1000))
            tts_dur_ms = int(_get_clip_duration(seg.tts_path) * 1000)
            zone_end = max(int(seg.end * 1000), tts_start_ms + tts_dur_ms)
            speech_zones.append((min(int(seg.start * 1000), tts_start_ms), zone_end))

    # Appliquer le ducking
    if speech_zones:
        print(f"   🔇 Ducking : -{DUCK_DB}dB sur {len(speech_zones)} zones...")
        bg = _apply_ducking(bg, speech_zones, DUCK_DB, DUCK_FADE_MS)

    # Overlay les clips TTS
    print(f"   🗣️  Superposition des voix doublées...")
    overlaid = 0
    for seg in segments:
        if not seg.tts_path or not os.path.exists(seg.tts_path):
            continue

        try:
            tts_clip = AudioSegment.from_wav(seg.tts_path)

            # Resampling HQ si nécessaire (TTS souvent en 22050/24000Hz, mix en 44100Hz)
            if tts_clip.frame_rate != bg.frame_rate:
                tts_clip = _hq_resample_segment(tts_clip, bg.frame_rate)

            # Normaliser le volume du clip TTS
            tts_clip = _normalize_audio(tts_clip, target_dBFS=-18)

            # Positionner au bon timestamp (ajusté par le push-later)
            position_ms = getattr(seg, '_tts_place_ms', int(seg.start * 1000))

            # Micro-fondu anti-clic (court pour préserver les attaques consonantiques)
            if len(tts_clip) > TTS_ANTI_CLICK_MS * 2:
                tts_clip = tts_clip.fade_in(TTS_ANTI_CLICK_MS).fade_out(TTS_ANTI_CLICK_MS)

            bg = bg.overlay(tts_clip, position=position_ms)
            overlaid += 1
        except Exception as e:
            print(f"      ⚠️  seg {seg.index} overlay échoué : {e}")

    # Normalisation finale
    bg = _normalize_audio(bg, target_dBFS=-16)

    # Export + réparation des clics
    bg.export(output_path, format="wav")
    _repair_clicks(output_path)
    mb = os.path.getsize(output_path) / (1024*1024)
    print(f"   ✅ {output_path} ({mb:.1f} Mo) — {overlaid} clips mixés [{time.time()-t0:.0f}s]")
    return output_path


def _apply_ducking(audio, zones, duck_db, fade_ms):
    """Applique un ducking avec fondus sur les zones de parole."""
    from pydub import AudioSegment

    # Convertir en samples pour manipulation fine
    samples = audio.get_array_of_samples()
    import numpy as np
    arr = np.array(samples, dtype=np.float64)

    if audio.channels == 2:
        arr = arr.reshape((-1, 2))

    sr = audio.frame_rate
    fade_samples = int(fade_ms * sr / 1000)

    # Créer un masque de volume (1.0 = plein volume, réduit = ducking)
    duck_factor = 10 ** (-duck_db / 20)
    gain = np.ones(len(arr) if arr.ndim == 1 else arr.shape[0], dtype=np.float64)

    for start_ms, end_ms in zones:
        start_s = int(start_ms * sr / 1000)
        end_s = int(end_ms * sr / 1000)
        start_s = max(0, start_s)
        end_s = min(len(gain), end_s)

        # Zone de ducking
        gain[start_s:end_s] = duck_factor

        # Fade in (avant la zone)
        fade_start = max(0, start_s - fade_samples)
        if fade_start < start_s:
            fade_len = start_s - fade_start
            fade_curve = np.linspace(1.0, duck_factor, fade_len)
            gain[fade_start:start_s] = np.minimum(gain[fade_start:start_s], fade_curve)

        # Fade out (après la zone)
        fade_end = min(len(gain), end_s + fade_samples)
        if end_s < fade_end:
            fade_len = fade_end - end_s
            fade_curve = np.linspace(duck_factor, 1.0, fade_len)
            gain[end_s:fade_end] = np.minimum(gain[end_s:fade_end], fade_curve)

    # Appliquer le gain
    if arr.ndim == 2:
        arr = arr * gain[:, np.newaxis]
    else:
        arr = arr * gain

    # Reconvertir
    arr = np.clip(arr, -32768, 32767).astype(np.int16)
    return audio._spawn(arr.tobytes())


def _normalize_audio(audio, target_dBFS=-18):
    """Normalise le volume d'un segment audio."""
    if audio.dBFS == float('-inf'):
        return audio
    change = target_dBFS - audio.dBFS
    return audio.apply_gain(change)


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
        ac = np.correlate(segment - segment.mean(), segment - segment.mean(), mode='full')
        ac = ac[len(ac)//2:]
        if ac[0] > 0:
            ac = ac / ac[0]
            if len(ac) > 2 and np.max(ac[1:]) > 0.3:
                continue  # structure harmonique = pas un clic

        # Zone d'interpolation : étendre de 2 ms de chaque côté
        margin = int(sr * 0.002)
        interp_start = max(0, sample_start - margin)
        interp_end = min(len(mono), sample_end + margin)

        left_end = sample_start
        right_start = sample_end
        n_left = left_end - interp_start
        n_right = interp_end - right_start

        if n_left < 2 or n_right < 2:
            continue

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

        if data.ndim == 1:
            data[left_end:right_start] = repaired_samples.astype(np.float32)
        else:
            for ch in range(data.shape[1]):
                ch_segment = data[left_end:right_start, ch]
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio_ch = np.where(
                        np.abs(mono[left_end:right_start]) > 1e-10,
                        repaired_samples / mono[left_end:right_start],
                        0)
                data[left_end:right_start, ch] = (ch_segment * ratio_ch).astype(np.float32)
            mono[left_end:right_start] = repaired_samples

        repaired += 1

    if repaired > 0:
        sf.write(audio_path, data, sr)
        print(f"      🔧 {repaired} clic(s) réparé(s) dans {os.path.basename(audio_path)}")


def _hq_resample_segment(segment, target_rate):
    """
    Resample un AudioSegment pydub vers target_rate en utilisant
    scipy.signal.resample_poly (filtrage anti-aliasing de qualité).

    Remplace pydub.set_frame_rate() qui utilise une interpolation linéaire
    basique générant des artefacts audibles.
    """
    from pydub import AudioSegment
    import numpy as np
    from scipy.signal import resample_poly

    src_rate = segment.frame_rate
    if src_rate == target_rate:
        return segment

    # Ratio de resampling via GCD
    g = math.gcd(target_rate, src_rate)
    up = target_rate // g
    down = src_rate // g

    # Extraire les samples
    samples = np.array(segment.get_array_of_samples(), dtype=np.float64)

    if segment.channels == 2:
        samples = samples.reshape((-1, 2))
        resampled = np.column_stack([
            resample_poly(samples[:, ch], up, down)
            for ch in range(2)
        ])
        resampled = resampled.flatten()
    else:
        resampled = resample_poly(samples, up, down)

    # Clipper et convertir en int16
    resampled = np.clip(resampled, -32768, 32767).astype(np.int16)

    return segment._spawn(
        resampled.tobytes(),
        overrides={
            "frame_rate": target_rate,
            "frame_count": 0,  # recalculé par pydub
            "frame_width": segment.sample_width,
        }
    )


def mix_audio_voiceover(segments: list[DubSegment], background_path: str,
                        vocals_path: str, output_path: str) -> str:
    """
    Mixage « style reportage » / « UN-style voice-over ».

    Reproduit la technique utilisée dans les JT (France 2, BBC, CNN, Arte)
    et les documentaires pour les interviews en langue étrangère :

      1. On entend d'abord la voix originale seule pendant ~1,5 s (lead-in)
      2. La voix originale descend à environ -18 dB (perceptible mais en arrière-plan)
      3. La voix doublée entre au premier plan (normalisée à -16 dBFS)
      4. À la fin du segment doublé, la voix doublée s'arrête
      5. La voix originale remonte brièvement (~0,8 s) avant le segment suivant
      6. Entre les segments doublés, la voix originale est audible à volume quasi-normal
      7. Le fond sonore (musique/ambiance) est légèrement atténué pendant les voix

    Niveaux de référence (conformes EBU R 128 / pratiques broadcast) :
      - Voix doublée (premier plan)  : -16 dBFS  (ancre du mix)
      - Voix originale (pendant TTS) : -34 dBFS  (≈ -18 dB sous la doublée)
      - Voix originale (entre segs)  : -19 dBFS  (quasi-naturel)
      - Fond sonore (pendant voix)   : -22 dBFS  (légèrement ducké)
      - Mix intégré visé             : ~-23 LUFS (EBU R 128)
    """
    from pydub import AudioSegment
    import numpy as np

    print(f"\n🎚️  Passe 7 — Mixage voice-over (style reportage)...")
    t0 = time.time()

    # ── Charger les métadonnées (sans charger les samples en mémoire) ────────
    import soundfile as sf
    bg_info = sf.info(background_path)
    vocals_info = sf.info(vocals_path)
    sr = bg_info.samplerate
    total_ms = int(bg_info.frames / sr * 1000)

    print(f"   📀 Fond sonore      : {total_ms/1000:.1f}s, {sr}Hz")
    print(f"   🗣️  Voix originales  : {vocals_info.duration:.1f}s")

    # ── Calculer les zones de parole doublée (basées sur les durées TTS réelles) ─
    # Chaque zone = (lead_in_start_ms, tts_start_ms, tts_end_ms, lead_out_end_ms)
    speech_zones = []
    for seg in segments:
        if seg.tts_path and os.path.exists(seg.tts_path):
            tts_dur_ms = _get_clip_duration(seg.tts_path) * 1000
            if tts_dur_ms < 50:
                continue

            seg_start_ms = int(seg.start * 1000)
            seg_end_ms = int(seg.end * 1000)
            seg_dur_ms = seg_end_ms - seg_start_ms

            # Lead-in adaptatif : réduire si le segment est court
            # pour ne pas comprimer la voix doublée
            lead_in = VO_LEAD_IN_MS if seg.is_sentence_start else 0
            lead_out = VO_LEAD_OUT_MS
            available_for_tts = seg_dur_ms - lead_in - lead_out
            if available_for_tts < tts_dur_ms:
                # Pas assez de place → réduire progressivement lead-in et lead-out
                excess = tts_dur_ms - available_for_tts
                # D'abord réduire le lead-out (moins important)
                lead_out = max(0, lead_out - excess // 2)
                excess = tts_dur_ms - (seg_dur_ms - lead_in - lead_out)
                # Puis réduire le lead-in
                if excess > 0:
                    lead_in = max(0, lead_in - excess)

            tts_start_ms = seg_start_ms + lead_in
            tts_end_ms = tts_start_ms + tts_dur_ms
            # Empêcher le débordement dans le segment suivant
            tts_end_ms = min(tts_end_ms, seg_end_ms + lead_out)

            speech_zones.append((seg_start_ms, tts_start_ms,
                                 int(tts_end_ms), seg_end_ms, lead_in, lead_out))

            # Stocker la position effective pour l'overlay plus tard
            seg._tts_place_ms = tts_start_ms

    if not speech_zones:
        print("   ⚠️  Aucun segment TTS — export de l'audio original")
        bg = AudioSegment.from_wav(background_path)
        orig_vocals = AudioSegment.from_wav(vocals_path)
        bg.overlay(orig_vocals).export(output_path, format="wav")
        return output_path

    # ── Rapprocher les clips TTS consécutifs du même locuteur ────────────────
    # Quand la segmentation coupe une phrase en deux, le lead-in du 2ème
    # segment crée un silence artificiel. On glisse le clip vers l'avant
    # pour préserver le rythme naturel.
    active_tts = []
    for seg in segments:
        if seg.tts_path and os.path.exists(seg.tts_path) and hasattr(seg, '_tts_place_ms'):
            dur = _get_clip_duration(seg.tts_path) * 1000
            if dur >= 50:
                active_tts.append([seg, seg._tts_place_ms, dur])  # mutable
    active_tts.sort(key=lambda x: x[1])

    bridged = 0
    glued = 0
    for i in range(1, len(active_tts)):
        seg, tts_start, tts_dur = active_tts[i]
        prev_seg, prev_start, prev_dur = active_tts[i - 1]

        if seg.speaker != prev_seg.speaker:
            continue
        original_gap_ms = (seg.start - prev_seg.end) * 1000
        if original_gap_ms < 0:
            continue

        prev_tts_end = prev_start + prev_dur

        # ── Option A : « coller » une continuation de phrase ─────────────────
        # Si le segment précédent (même locuteur) ne se termine pas par une
        # ponctuation finale, ce segment est la suite de la même phrase
        # (is_sentence_start=False). On le ramène juste après la 1re moitié,
        # SANS le plafond VO_MAX_BRIDGE_GAP_MS et SANS le clamp ≥ seg.start :
        # une vraie pause mid-phrase de la source (> 800 ms) serait sinon
        # reproduite telle quelle (« début de phrase … long silence … fin »).
        # On tire le clip plus tôt → il finit plus tôt aussi, donc aucun
        # risque de chevauchement avec le clip suivant.
        if VO_GLUE_CONTINUATIONS and not getattr(seg, "is_sentence_start", True):
            new_start = int(prev_tts_end + GAP_BETWEEN_CLIPS_MS)
            if new_start < tts_start - 50:        # ne bouger que si ça rapproche vraiment
                seg._tts_place_ms = new_start
                active_tts[i] = [seg, new_start, tts_dur]
                glued += 1
            continue

        # ── Cas général : pontage isochrone (petits gaps seulement) ──────────
        if original_gap_ms >= VO_MAX_BRIDGE_GAP_MS:
            continue
        current_gap = tts_start - prev_tts_end
        target_gap = max(original_gap_ms, GAP_BETWEEN_CLIPS_MS)

        if current_gap <= target_gap + 50:
            continue

        new_start = int(prev_tts_end + target_gap)
        new_start = max(new_start, int(seg.start * 1000))
        if new_start < tts_start:
            seg._tts_place_ms = new_start
            active_tts[i] = [seg, new_start, tts_dur]
            bridged += 1

    if glued or bridged:
        parts = []
        if glued:
            parts.append(f"{glued} continuation(s) collée(s)")
        if bridged:
            parts.append(f"{bridged} segment(s) rapproché(s)")
        print(f"   🔗 {', '.join(parts)} (continuité intra-phrase)")

    # Anti-troncature : décaler les clips qui débordent au lieu de tronquer.
    _resolve_overlaps_push_later(segments)

    # Reconstruire speech_zones avec les positions ajustées
    speech_zones = []
    for seg in segments:
        if seg.tts_path and os.path.exists(seg.tts_path) and hasattr(seg, '_tts_place_ms'):
            tts_dur_ms = _get_clip_duration(seg.tts_path) * 1000
            if tts_dur_ms < 50:
                continue
            tts_start_ms = seg._tts_place_ms
            tts_end_ms = tts_start_ms + tts_dur_ms
            seg_start_ms = int(seg.start * 1000)
            seg_end_ms = int(seg.end * 1000)
            lead_in = max(0, tts_start_ms - seg_start_ms)
            lead_out = max(0, seg_end_ms - int(tts_end_ms))
            speech_zones.append((seg_start_ms, tts_start_ms,
                                 int(tts_end_ms), seg_end_ms, lead_in, lead_out))

    print(f"   📍 {len(speech_zones)} zones de doublage détectées")

    # ── Construire le profil de gain pour la voix originale ──────────────────
    #
    # Le principe : la voix originale a un profil de volume dynamique :
    #   - Volume quasi-normal entre les segments doublés
    #   - Lead-in  : plein volume → duck (fondu de VO_FADE_MS)
    #   - Pendant la voix doublée : duckée à -VO_ORIG_DUCK_DB
    #   - Lead-out : duck → plein volume (fondu de VO_FADE_MS)
    #
    total_samples = int(total_ms * sr / 1000)
    fade_samples = int(VO_FADE_MS * sr / 1000)

    # Gain de base : légère atténuation entre les segments (naturel)
    orig_gain = np.ones(total_samples, dtype=np.float32)
    between_factor = 10 ** (VO_ORIG_BETWEEN_DB / 20)  # ~0.71 (-3 dB)
    orig_gain[:] = between_factor

    # Facteur de duck pendant le doublage
    duck_factor = 10 ** (-VO_ORIG_DUCK_DB / 20)  # ~0.126 (-18 dB)

    for seg_start_ms, tts_start_ms, tts_end_ms, seg_end_ms, lead_in, lead_out in speech_zones:
        # La zone duckée = là où la voix doublée joue effectivement
        duck_start_ms = tts_start_ms
        duck_end_ms = tts_end_ms

        duck_start_s = int(duck_start_ms * sr / 1000)
        duck_end_s = int(duck_end_ms * sr / 1000)
        duck_start_s = max(0, min(duck_start_s, total_samples))
        duck_end_s = max(0, min(duck_end_s, total_samples))

        # Zone duckée centrale
        if duck_start_s < duck_end_s:
            orig_gain[duck_start_s:duck_end_s] = duck_factor

        # Fade-in vers le duck (plein volume → duck)
        fi_start = max(0, duck_start_s - fade_samples)
        if fi_start < duck_start_s:
            fade_len = duck_start_s - fi_start
            fade_curve = np.linspace(between_factor, duck_factor, fade_len)
            orig_gain[fi_start:duck_start_s] = np.minimum(
                orig_gain[fi_start:duck_start_s], fade_curve)

        # Fade-out du duck (duck → plein volume)
        fo_end = min(total_samples, duck_end_s + fade_samples)
        if duck_end_s < fo_end:
            fade_len = fo_end - duck_end_s
            fade_curve = np.linspace(duck_factor, between_factor, fade_len)
            orig_gain[duck_end_s:fo_end] = np.minimum(
                orig_gain[duck_end_s:fo_end], fade_curve)

    # Appliquer le gain à la voix originale (float32 suffit pour audio 16-bit)
    # Charger directement via soundfile (pas de copie pydub intermédiaire)
    orig_arr, orig_sr = sf.read(vocals_path, dtype='float32')
    # Convertir en échelle int16 in-place (évite promotion float64)
    orig_arr *= np.float32(32768.0)

    n_orig = orig_arr.shape[0]
    n_gain = len(orig_gain)
    n = min(n_orig, n_gain)

    if orig_arr.ndim == 2:
        orig_arr[:n] = orig_arr[:n] * orig_gain[:n, np.newaxis]
    else:
        orig_arr[:n] = orig_arr[:n] * orig_gain[:n]
    del orig_gain

    # ── Construire le profil de gain pour le fond sonore ─────────────────────
    # Le fond est légèrement ducké pendant TOUTES les voix (originale ou doublée)
    bg_duck_factor = 10 ** (-VO_BG_DUCK_DB / 20)  # ~0.5 (-6 dB)
    bg_gain = np.ones(total_samples, dtype=np.float32)

    # Étendre les zones pour couvrir lead-in et lead-out
    for seg_start_ms, tts_start_ms, tts_end_ms, seg_end_ms, lead_in, lead_out in speech_zones:
        # Le fond est ducké sur toute la zone voix (original + doublée)
        bg_start = max(0, int(seg_start_ms * sr / 1000))
        bg_end = min(total_samples, int(max(tts_end_ms, seg_end_ms) * sr / 1000))

        bg_gain[bg_start:bg_end] = bg_duck_factor

        # Fondus
        fi_start = max(0, bg_start - fade_samples)
        if fi_start < bg_start:
            fl = bg_start - fi_start
            bg_gain[fi_start:bg_start] = np.minimum(
                bg_gain[fi_start:bg_start],
                np.linspace(1.0, bg_duck_factor, fl))
        fo_end = min(total_samples, bg_end + fade_samples)
        if bg_end < fo_end:
            fl = fo_end - bg_end
            bg_gain[bg_end:fo_end] = np.minimum(
                bg_gain[bg_end:fo_end],
                np.linspace(bg_duck_factor, 1.0, fl))

    # Appliquer le gain au fond (charger via soundfile, float32)
    bg_arr, bg_sr = sf.read(background_path, dtype='float32')
    bg_arr *= np.float32(32768.0)

    n_bg = bg_arr.shape[0]
    n = min(n_bg, len(bg_gain))

    if bg_arr.ndim == 2:
        bg_arr[:n] = bg_arr[:n] * bg_gain[:n, np.newaxis]
    else:
        bg_arr[:n] = bg_arr[:n] * bg_gain[:n]
    del bg_gain

    # ── Mixer en float64 : sommation précise ─────────────────────────────────
    # Aligner les longueurs (prendre le min pour éviter les débordements)
    mix_len = min(orig_arr.shape[0] if orig_arr.ndim == 2 else len(orig_arr),
                  bg_arr.shape[0] if bg_arr.ndim == 2 else len(bg_arr))
    # Réutiliser orig_arr comme buffer mix (économise ~4GB pour vidéos longues)
    orig_arr[:mix_len] += bg_arr[:mix_len]
    mix_arr = orig_arr
    del bg_arr  # libérer ~4GB

    # ── Superposer les clips TTS directement en numpy (voix doublée au 1er plan)
    # Évite 2000+ appels pydub .overlay() qui copient tout le mix à chaque fois
    print(f"   🗣️  Superposition des voix doublées (premier plan)...")
    overlaid = 0
    n_channels = 2 if mix_arr.ndim == 2 else 1
    for seg in segments:
        if not seg.tts_path or not os.path.exists(seg.tts_path):
            continue
        try:
            tts_clip = AudioSegment.from_wav(seg.tts_path)
            # Resampling HQ si nécessaire (TTS souvent en 22050/24000Hz, mix en 44100Hz)
            if tts_clip.frame_rate != sr:
                tts_clip = _hq_resample_segment(tts_clip, sr)
            tts_clip = _normalize_audio(tts_clip, target_dBFS=VO_TTS_TARGET_DBFS)

            # Micro-fondu anti-clic (court pour préserver les attaques consonantiques)
            if len(tts_clip) > TTS_ANTI_CLICK_MS * 2:
                tts_clip = tts_clip.fade_in(TTS_ANTI_CLICK_MS).fade_out(TTS_ANTI_CLICK_MS)

            # Position : utiliser le placement adaptatif calculé plus haut
            tts_start_ms = getattr(seg, '_tts_place_ms', int(seg.start * 1000))
            start_sample = int(tts_start_ms * sr / 1000)

            # Extraire les samples du clip TTS en float32
            tts_samples = np.array(tts_clip.get_array_of_samples(), dtype=np.float32)
            if tts_clip.channels == 2 and n_channels == 2:
                tts_samples = tts_samples.reshape((-1, 2))
            elif tts_clip.channels == 1 and n_channels == 2:
                tts_samples = np.column_stack([tts_samples, tts_samples])
            elif tts_clip.channels == 2 and n_channels == 1:
                tts_samples = tts_samples.reshape((-1, 2)).mean(axis=1)

            # Overlay par addition directe dans le buffer
            end_sample = min(start_sample + len(tts_samples), mix_len)
            clip_len = end_sample - start_sample
            if clip_len > 0 and start_sample >= 0:
                if mix_arr.ndim == 2:
                    mix_arr[start_sample:end_sample] += tts_samples[:clip_len]
                else:
                    mix_arr[start_sample:end_sample] += tts_samples[:clip_len]
            overlaid += 1
            del tts_samples
        except Exception as e:
            print(f"      ⚠️  seg {seg.index} overlay échoué : {e}")

    # ── Normalisation finale (cibler ~-23 LUFS ≈ -18 dBFS en crête) ─────────
    # Calcul du RMS par chunks pour éviter une copie float64 de tout le mix
    chunk_size = 1_000_000
    sum_sq = np.float64(0.0)
    total_count = 0
    for ci in range(0, mix_len, chunk_size):
        chunk = mix_arr[ci:min(ci + chunk_size, mix_len)]
        flat = chunk.ravel() if chunk.ndim == 2 else chunk
        sum_sq += np.sum(flat.astype(np.float64) ** 2)
        total_count += len(flat)
    rms = float(np.sqrt(sum_sq / total_count)) if total_count > 0 else 0.0
    if rms > 0:
        current_dbfs = 20 * math.log10(rms / 32768)
        gain_db = -18 - current_dbfs
        gain_factor = 10 ** (gain_db / 20)
        mix_arr[:mix_len] *= np.float32(gain_factor)

    np.clip(mix_arr, -32768, 32767, out=mix_arr)
    mix_arr = mix_arr.astype(np.int16)

    # Export direct via soundfile (évite de créer un gros AudioSegment pydub)
    if mix_arr.ndim == 2:
        sf.write(output_path, mix_arr[:mix_len], sr, subtype='PCM_16')
    else:
        sf.write(output_path, mix_arr[:mix_len], sr, subtype='PCM_16')
    del mix_arr  # libérer
    _repair_clicks(output_path)
    mb = os.path.getsize(output_path) / (1024*1024)

    print(f"   ✅ {output_path} ({mb:.1f} Mo) — {overlaid} clips doublés")
    print(f"      🎙️  Lead-in : {VO_LEAD_IN_MS}ms | Lead-out : {VO_LEAD_OUT_MS}ms")
    print(f"      🔉 Voix orig pendant doublage : -{VO_ORIG_DUCK_DB}dB")
    print(f"      🔊 Voix doublée : {VO_TTS_TARGET_DBFS} dBFS")
    print(f"      ⏱️  {time.time()-t0:.0f}s")
    return output_path


def mix_audio_onlydub(segments: list, background_path: str,
                      output_path: str) -> str:
    """
    Mode « only dub » : produit un MP3 contenant UNIQUEMENT les voix
    doublées positionnées à leur timestamp, sans voix originale ni fond sonore.
    Utile pour le diagnostic qualité et le benchmark.
    """
    from pydub import AudioSegment

    print(f"\n🎚️  Passe 7 — Audio doublage pur (--onlydub)...")
    t0 = time.time()

    # Créer une piste silencieuse de la même durée que le fond
    bg = AudioSegment.from_wav(background_path)
    sr = bg.frame_rate
    silence = AudioSegment.silent(duration=len(bg), frame_rate=sr)

    overlaid = 0
    for seg in segments:
        if not seg.tts_path or not os.path.exists(seg.tts_path):
            continue
        try:
            tts_clip = AudioSegment.from_wav(seg.tts_path)

            # Resampling HQ si nécessaire
            if tts_clip.frame_rate != sr:
                tts_clip = _hq_resample_segment(tts_clip, sr)

            tts_clip = _normalize_audio(tts_clip, target_dBFS=-16)

            position_ms = int(seg.start * 1000)

            # Fondus anti-clic (aligné sur TTS_ANTI_CLICK_MS)
            if len(tts_clip) > TTS_ANTI_CLICK_MS * 2:
                tts_clip = tts_clip.fade_in(TTS_ANTI_CLICK_MS).fade_out(TTS_ANTI_CLICK_MS)

            silence = silence.overlay(tts_clip, position=position_ms)
            overlaid += 1
        except Exception as e:
            print(f"      ⚠️  seg {seg.index} overlay échoué : {e}")

    # Réparation des clics : export WAV temporaire → _repair_clicks → MP3 final
    tmp_wav = output_path.rsplit(".", 1)[0] + "_pre_repair.wav"
    silence.export(tmp_wav, format="wav")
    _repair_clicks(tmp_wav)
    from pydub import AudioSegment as _AS
    silence = _AS.from_wav(tmp_wav)
    os.remove(tmp_wav)

    silence.export(output_path, format="mp3", bitrate="192k")
    mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"   ✅ {output_path} ({mb:.1f} Mo) — {overlaid} clips, doublage pur [{time.time()-t0:.0f}s]")
    return output_path


def assemble_sequential_mp3(segments: list, output_path: str,
                            pause_ms: int = AUDIO_ONLY_PAUSE_MS,
                            speaker_pause_ms: int = AUDIO_ONLY_SPEAKER_PAUSE_MS) -> str:
    """
    Mode « audio-only » : concatène les clips TTS dans l'ordre séquentiel avec
    des pauses naturelles. Pas de calage temporel sur l'original — la durée du
    MP3 final n'a pas à correspondre à celle de la vidéo source.

    - Pause courte entre segments d'un même locuteur (respiration)
    - Pause longue lors d'un changement de locuteur (tour de parole)
    - Micro-fondus anti-clic en entrée et sortie de chaque clip
    """
    from pydub import AudioSegment

    print(f"\n🎚️  Passe 7 — Assemblage séquentiel → MP3 (--audio-only)...")
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
            fade = min(TTS_ANTI_CLICK_MS, len(clip) // 4)
            if fade > 0:
                clip = clip.fade_in(fade).fade_out(fade)

            # Pause avant le clip (sauf en tête)
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
        print(f"   ❌ Audio trop court ({len(combined)} ms)")
        return ""

    # Normalisation finale RMS → -18 dBFS
    combined = _normalize_audio(combined, target_dBFS=-18)

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
    print(f"   ✅ {output_path} ({dur:.1f}s, {mb:.1f} Mo) — "
          f"{overlaid} segments{skip_msg} [{time.time()-t0:.0f}s]")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 8 : ASSEMBLAGE VIDÉO FINALE
# ═══════════════════════════════════════════════════════════════════════════════

def _get_video_resolution(video_path: str) -> tuple[int, int]:
    """Retourne (largeur, hauteur) de la vidéo via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        video_path
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0 and "x" in r.stdout.strip():
        w, h = r.stdout.strip().split("x")
        return int(w), int(h)
    return 1920, 1080  # fallback


def _video_needs_reencode(video_path: str) -> bool:
    """True si le codec vidéo source n'est pas H.264 (incompatible X/Twitter)."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "csv=s=x:p=0",
        video_path
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    codec = r.stdout.strip().lower() if r.returncode == 0 else ""
    if codec and codec != "h264":
        print(f"   ⚠️  Codec source : {codec} — ré-encodage H.264 pour compatibilité X")
        return True
    return False


def _build_watermark_filter(video_path: str) -> str:
    """
    Construit le filtre ffmpeg drawtext pour le watermark.
    Texte blanc semi-transparent avec ombre noire, coin supérieur droit,
    taille proportionnelle à la résolution.
    """
    w, h = _get_video_resolution(video_path)

    # Taille de police : ~1.4% de la hauteur (sobre mais lisible)
    # 1080p → 15px, 720p → 10px, 2160p → 30px
    fontsize = max(10, int(h * 0.014))
    margin = max(8, int(h * 0.012))

    # Polices disponibles sur Linux, par ordre de préférence
    # Open Sans > DejaVu Sans > Liberation Sans > sans-serif générique
    font_candidates = [
        "/usr/share/fonts/truetype/open-sans/OpenSans-Regular.ttf",
        "/usr/share/fonts/opentype/open-sans/OpenSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    fontfile = ""
    for f in font_candidates:
        if os.path.exists(f):
            fontfile = f
            break

    # Construire le filtre drawtext
    text = "doublage IA - @resilientstv"
    # Échapper les caractères spéciaux ffmpeg
    text_escaped = text.replace(":", "\\:").replace("'", "\\'")

    parts = [
        f"drawtext=text='{text_escaped}'",
        f"fontsize={fontsize}",
        f"fontcolor=white@0.85",
        f"shadowcolor=black@0.5",
        f"shadowx={max(1, fontsize // 12)}",
        f"shadowy={max(1, fontsize // 12)}",
        f"x=w-tw-{margin}",
        f"y={margin}",
    ]
    if fontfile:
        parts.insert(1, f"fontfile='{fontfile}'")

    return ":".join(parts)


def assemble_video(video_path: str, mixed_audio_path: str,
                   output_path: str, watermark: bool = True,
                   skip_seconds: float = 0.0) -> str:
    """Combine la vidéo originale avec le nouvel audio doublé."""
    skip_msg = f" (début à {format_skip(skip_seconds)})" if skip_seconds > 0 else ""
    print(f"\n🎬 Passe 8 — Assemblage vidéo finale{skip_msg}...")
    t0 = time.time()

    # Quand on skip, il faut -ss AVANT -i pour un seek rapide sur la vidéo,
    # et on ré-encode pour garantir la précision frame-exact.
    ss_args = ["-ss", str(skip_seconds)] if skip_seconds > 0 else []
    needs_reencode = watermark or skip_seconds > 0 or _video_needs_reencode(video_path)

    if needs_reencode:
        vf_parts = []
        if watermark:
            vf_parts.append(_build_watermark_filter(video_path))
        vf_args = ["-vf", ",".join(vf_parts)] if vf_parts else []
        cmd = [
            "ffmpeg", "-y",
            *ss_args,
            "-i", video_path,
            "-i", mixed_audio_path,
            *vf_args,
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:a", "aac", "-b:a", "192k", "-ac", "2",
            "-shortest",
            "-movflags", "+faststart",
            output_path
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", mixed_audio_path,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k", "-ac", "2",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            "-movflags", "+faststart",
            output_path
        ]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"   ❌ ffmpeg erreur : {r.stderr[-400:]}")
        return ""

    mb = os.path.getsize(output_path) / (1024*1024)
    wm = " + watermark" if watermark else ""
    print(f"   ✅ {output_path} ({mb:.1f} Mo){wm} [{time.time()-t0:.0f}s]")
    return output_path


def assemble_dual_audio(video_path: str, mixed_audio_path: str,
                        output_path: str, watermark: bool = True,
                        skip_seconds: float = 0.0) -> str:
    """Produit un MP4 avec 2 pistes : doublage (piste 1) + original (piste 2)."""
    print(f"   🎬 Variante : vidéo bi-piste audio...")

    ss_args = ["-ss", str(skip_seconds)] if skip_seconds > 0 else []
    needs_reencode = watermark or skip_seconds > 0 or _video_needs_reencode(video_path)

    if needs_reencode:
        vf_parts = []
        if watermark:
            vf_parts.append(_build_watermark_filter(video_path))
        vf_args = ["-vf", ",".join(vf_parts)] if vf_parts else []
        cmd = [
            "ffmpeg", "-y",
            *ss_args,
            "-i", video_path,
            "-i", mixed_audio_path,
            *vf_args,
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-map", "0:a:0",
            "-c:a", "aac", "-b:a", "192k", "-ac", "2",
            "-metadata:s:a:0", "language=fra",
            "-metadata:s:a:0", "title=Doublage",
            "-metadata:s:a:1", "language=eng",
            "-metadata:s:a:1", "title=Original",
            "-shortest",
            "-movflags", "+faststart",
            output_path
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", mixed_audio_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-map", "0:a:0",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k", "-ac", "2",
            "-metadata:s:a:0", "language=fra",
            "-metadata:s:a:0", "title=Doublage",
            "-metadata:s:a:1", "language=eng",
            "-metadata:s:a:1", "title=Original",
            "-shortest",
            "-movflags", "+faststart",
            output_path
        ]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"   ❌ ffmpeg bi-piste : {r.stderr[-300:]}")
        return ""

    mb = os.path.getsize(output_path) / (1024*1024)
    print(f"   ✅ {output_path} ({mb:.1f} Mo) — 2 pistes audio")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════════

def save_segments(segs: list[DubSegment], path: str):
    data = []
    for s in segs:
        d = {"index": s.index, "start": s.start, "end": s.end,
             "text": s.text, "text_tgt": s.text_tgt,
             "text_adapted": s.text_adapted, "speaker": s.speaker}
        # Préserver les timings mot-à-mot WhisperX pour réutilisation (clipper karaoke).
        if s.words:
            d["words"] = s.words
        data.append(d)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"   💾 → {path}")


def load_segments(path: str) -> list[DubSegment]:
    with open(path) as f:
        data = json.load(f)
    segments = [
        DubSegment(
            index=d["index"], start=d["start"], end=d["end"],
            text=d["text"],
            text_tgt=d.get("text_tgt", d.get("text_fr", "")),
            text_adapted=d.get("text_adapted", ""),
            speaker=d.get("speaker", "SPEAKER_00"),
            words=d.get("words", []),
        )
        for d in data
    ]
    # Scan de décontamination : vider les text_tgt qui sont des fuites de prompt
    contaminated = 0
    for seg in segments:
        if seg.text_tgt and _est_fuite_prompt(seg.text_tgt):
            print(f"   ⚠️  Segment [{seg.index}] contaminé, re-traduction forcée : {seg.text_tgt[:50]}…")
            seg.text_tgt = ""
            contaminated += 1
    if contaminated:
        print(f"   🧹 {contaminated} segment(s) décontaminé(s) — re-traduction nécessaire")
    return segments


def load_srt_translations(srt_path: str) -> list[tuple]:
    """Parse un fichier SRT → liste de (start_sec, end_sec, text)."""
    import re
    entries = []
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()
    # Découper en blocs séparés par des lignes vides
    blocks = re.split(r"\n\s*\n", content.strip())
    ts_re = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        # Chercher la ligne de timecode
        ts_match = None
        ts_line_idx = -1
        for i, line in enumerate(lines):
            ts_match = ts_re.search(line)
            if ts_match:
                ts_line_idx = i
                break
        if not ts_match:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = [int(x) for x in ts_match.groups()]
        start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0
        end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0
        # Le texte = toutes les lignes après le timecode
        text = " ".join(lines[ts_line_idx + 1:]).strip()
        if text:
            entries.append((start, end, text))
    print(f"   📄 SRT chargé : {len(entries)} sous-titres depuis {os.path.basename(srt_path)}")
    return entries


def align_srt_to_segments(srt_entries, segments):
    """Aligne les sous-titres SRT sur les segments WhisperX par chevauchement temporel.

    Pour chaque segment WhisperX, on collecte tous les SRT qui chevauchent
    et on concatène leurs textes (dédupliqués) dans seg.text_tgt.
    """
    aligned = 0
    no_match = 0
    for seg in segments:
        # Collecter tous les SRT qui chevauchent ce segment
        overlapping_texts = []
        for srt_start, srt_end, srt_text in srt_entries:
            overlap_start = max(seg.start, srt_start)
            overlap_end = min(seg.end, srt_end)
            overlap = overlap_end - overlap_start
            if overlap > 0:
                overlapping_texts.append(srt_text)
        if overlapping_texts:
            # Concaténer les textes dédupliqués (préserver l'ordre)
            seen = set()
            unique = []
            for t in overlapping_texts:
                if t not in seen:
                    seen.add(t)
                    unique.append(t)
            seg.text_tgt = " ".join(unique)
            aligned += 1
        else:
            no_match += 1
    print(f"   ✅ {aligned} segments alignés depuis SRT professionnel")
    if no_match:
        print(f"   ⚠️  {no_match} segments sans correspondance SRT (resteront non traduits)")
    return segments


def load_traduire_segments(path: str) -> list[DubSegment]:
    """Charge les segments depuis traduire.py (format compatible)."""
    with open(path) as f:
        data = json.load(f)
    return [
        DubSegment(
            index=d["index"], start=d["start"], end=d["end"],
            text=d["text"],
            text_tgt=d.get("text_tgt", d.get("text_fr", "")),
            speaker="SPEAKER_00",  # pas de diarisation dans traduire.py
            words=d.get("words", []),
        )
        for d in data
    ]


def generate_report(segments: list[DubSegment], profiles: dict[str, SpeakerProfile],
                    output_path: str, src_lang: str, tgt_lang: str):
    """Génère un rapport détaillé du doublage."""
    import soundfile as sf

    # ── Métrique de couverture : audio TTS produit vs parole originale ──
    total_speech_orig = sum(max(s.end - s.start, 0) for s in segments)
    total_tts = 0.0
    sparse_segments = []  # segments où dur_tts < 70% de dur_orig (potentiel contenu manquant)
    for s in segments:
        dur_orig = max(s.end - s.start, 0)
        dur_tts = 0.0
        if s.tts_path and os.path.exists(s.tts_path):
            try:
                dur_tts = sf.info(s.tts_path).duration
            except Exception:
                pass
        total_tts += dur_tts
        if dur_orig >= 2.0 and dur_tts < 0.70 * dur_orig:
            sparse_segments.append((s, dur_orig, dur_tts))

    coverage = (total_tts / total_speech_orig) if total_speech_orig > 0 else 0.0

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"RAPPORT DE DOUBLAGE ({src_lang.upper()} → {tgt_lang.upper()})\n")
        f.write("=" * 70 + "\n\n")

        # ── Section couverture ─────────────────────────────────────────
        f.write("COUVERTURE :\n")
        f.write(f"  Parole originale (somme)  : {total_speech_orig:.1f}s\n")
        f.write(f"  Audio TTS produit         : {total_tts:.1f}s\n")
        f.write(f"  Ratio                     : {coverage*100:.1f}%\n")
        if coverage < 0.85:
            f.write(f"  ⚠️  ALERTE : couverture < 85% — du contenu source est probablement perdu.\n")
            f.write(f"     Inspecte la passe de traduction (4b) pour les segments suivants :\n")
        elif sparse_segments:
            f.write(f"  ℹ️  {len(sparse_segments)} segment(s) avec dur_tts < 70% de dur_orig.\n")
        if sparse_segments:
            for s, dur_orig, dur_tts in sparse_segments:
                gap = dur_orig - dur_tts
                f.write(f"     [{s.index}] {s.start:.1f}s  orig={dur_orig:.1f}s  tts={dur_tts:.1f}s  manque≈{gap:.1f}s\n")
        f.write("\n")

        f.write("LOCUTEURS :\n")
        for spk_id, p in profiles.items():
            f.write(f"  {spk_id} : {p.segment_count} segments, {p.total_duration:.1f}s\n")
        f.write("\n")

        f.write("SEGMENTS :\n\n")
        for s in segments:
            f.write(f"[{s.index}] {s.start:.2f}–{s.end:.2f}s ({s.speaker})\n")
            f.write(f"  {src_lang.upper()} : {s.text}\n")
            f.write(f"  {tgt_lang.upper()} : {s.text_tgt}\n")
            if s.text_adapted and s.text_adapted != s.text_tgt:
                f.write(f"  ADAPT: {s.text_adapted}\n")
            f.write(f"  TTS  : {'✅' if s.tts_path else '❌'}\n\n")

    # Print à la console aussi pour que ce soit visible immédiatement
    if coverage < 0.85:
        print(f"\n⚠️  COUVERTURE FAIBLE : audio TTS = {coverage*100:.1f}% de la parole originale")
        print(f"    {len(sparse_segments)} segments suspects (voir rapport)")
    elif total_speech_orig > 0:
        print(f"\n✅ Couverture audio : {coverage*100:.1f}%")


def generate_social_txt(segments, analysis, client, output_path: str,
                        source_lang: str, target_lang: str):
    """Génère un fichier de partage social (citations verbatim + mise en contexte)."""
    print(f"\n📱 Génération du résumé social...")

    # Construire la transcription — utiliser la traduction FR si disponible
    use_tgt = target_lang.startswith("fr")
    lines = []
    for s in segments:
        txt = ""
        if use_tgt:
            txt = getattr(s, "text_adapted", "") or getattr(s, "text_tgt", "") or s.text
        else:
            txt = s.text
        if txt:
            lines.append(txt)
    transcript = "\n".join(lines)

    if len(transcript) > 80000:
        transcript = transcript[:80000] + "\n[... tronqué ...]"

    # Extraire le contexte de l'analyse
    if isinstance(analysis, dict):
        summary = analysis.get("summary", "")
        speakers = analysis.get("speakers_description", "")
    else:
        summary = getattr(analysis, "summary", "")
        speakers = getattr(analysis, "speakers_description", "")

    ctx = ""
    if summary:
        ctx += f"\nRÉSUMÉ DE L'ANALYSE : {summary}\n"
    if speakers:
        ctx += f"LOCUTEURS : {speakers}\n"

    prompt = f"""Tu rédiges un court texte de partage pour les réseaux sociaux, EN FRANÇAIS.
À partir de cette transcription vidéo, produis :
1. Un titre court résumant le sujet (1 ligne)
2. Les noms et rôles des intervenants (1 ligne, format: "Intervenants : Nom1 (rôle), Nom2 (rôle)")
3. Une mise en contexte (2-3 phrases max, section "Contexte :")
4. Exactement 2 ou 3 citations VERBATIM marquantes entre guillemets « »
   (les plus percutantes, surprenantes ou éclairantes)

Les citations doivent apparaître MOT POUR MOT dans la transcription ci-dessous.
Pas d'emojis, pas de hashtags. Style sobre et informatif.
Rédige TOUT en français, même si la transcription est dans une autre langue.

Format attendu :
[Titre]

Intervenants : ...

Contexte : ...

Citations marquantes :

« Citation 1 »

« Citation 2 »

« Citation 3 (optionnelle) »
{ctx}
TRANSCRIPTION :
{transcript}"""

    try:
        resp = _claude_create(client, model=CLAUDE_MODEL, max_tokens=1024,
                              messages=[{"role": "user", "content": prompt}])
        result = resp.content[0].text.strip()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result + "\n")
        print(f"   ✅ Résumé social : {output_path}")
    except Exception as e:
        print(f"   ⚠️  Résumé social échoué : {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK TTS
# ═══════════════════════════════════════════════════════════════════════════════

def run_tts_benchmark(segments: list, profiles: dict, work_dir: str,
                      base_name: str, target_lang: str,
                      source_lang: str = "en",
                      ref_voice: Optional[str] = None,
                      xtts_speaker: Optional[str] = None,
                      backend: str = "xtts",
                      elevenlabs_voice: Optional[str] = None,
                      elevenlabs_model: Optional[str] = None,
                      ref_voices_dir: Optional[str] = None):
    """
    Synthétise des segments test pour évaluer la qualité TTS.
    Produit un fichier MP3 : {base_name}-{backend}.mp3
    """
    from pydub import AudioSegment

    # Collecter assez de segments pour ~60s de parole
    BENCH_TARGET_SEC = 60
    chars_per_sec = 14
    test_segments = []
    total_chars = 0

    for seg in segments:
        text = seg.text_adapted or seg.text_tgt
        if text and len(text) > 10:
            test_segments.append(seg)
            total_chars += len(text)
            if total_chars >= BENCH_TARGET_SEC * chars_per_sec:
                break

    if not test_segments:
        print("   ❌ Aucun segment traduit disponible pour le benchmark")
        return

    est_duration = total_chars / chars_per_sec
    test_spk = test_segments[0].speaker or next(iter(profiles.keys()), "SPEAKER_00")

    backend_label = "ElevenLabs" if backend == "elevenlabs" else "XTTS v2"
    first_text = test_segments[0].text_adapted or test_segments[0].text_tgt
    print(f"\n{'='*60}")
    print(f"🏁 BENCHMARK TTS — {backend_label}")
    print(f"{'='*60}")
    print(f"   Segments: {len(test_segments)} (~{est_duration:.0f}s estimées)")
    print(f"   Premier : \"{first_text[:80]}{'...' if len(first_text) > 80 else ''}\"")
    print(f"   Langue  : {target_lang}")
    print(f"   Speaker : {test_spk}")

    bench_dir = os.path.join(work_dir, "benchmark")
    os.makedirs(bench_dir, exist_ok=True)

    print(f"\n   🔊 {backend} — synthèse en cours...")
    t0 = time.time()

    try:
        tts = create_tts_backend(ref_voice=ref_voice,
                                 target_lang=target_lang,
                                 source_lang=source_lang,
                                 xtts_speaker=xtts_speaker,
                                 backend=backend,
                                 elevenlabs_voice=elevenlabs_voice,
                                 elevenlabs_model=elevenlabs_model,
                                 ref_voices_dir=ref_voices_dir)
        tts.setup_voices(profiles)

        combined = AudioSegment.empty()
        seg_ok = 0
        for i, seg in enumerate(test_segments):
            text = seg.text_adapted or seg.text_tgt
            spk = seg.speaker or test_spk
            wav_path = os.path.join(bench_dir, f"bench_{backend}_{i:03d}.wav")

            result_path = tts.synthesize(text, spk, wav_path)
            if result_path and os.path.exists(result_path):
                try:
                    chunk = AudioSegment.from_wav(result_path)
                    if len(chunk) > 100:
                        combined += chunk
                        seg_ok += 1
                except Exception:
                    pass

        tts.cleanup()

        if seg_ok > 0 and len(combined) > 500:
            elapsed = time.time() - t0
            dur = len(combined) / 1000
            mp3_name = f"{base_name}-{backend}.mp3"
            mp3_path = os.path.join(os.path.dirname(work_dir), mp3_name)
            combined.export(mp3_path, format="mp3", bitrate="192k")
            print(f"      ✅ {mp3_name} ({dur:.1f}s, {seg_ok} segments, généré en {elapsed:.1f}s)")
        else:
            print(f"      ❌ Aucun audio produit")
    except Exception as e:
        print(f"      ❌ Erreur : {e}")

    print()


AUTO_CLIP_THRESHOLD_SEC = 45 * 60   # vidéos > 45 min → extraction automatique de clips
AUTO_CLIP_COUNT = 4

def _probe_duration(path: str) -> float:
    """Durée du média via ffprobe (secondes, 0.0 si échec)."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, check=True)
        return float(r.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return 0.0


def auto_clip_if_long(dubbed_video: str, seg_json: str, tgt_lang: str):
    """Si la vidéo doublée dépasse AUTO_CLIP_THRESHOLD_SEC, invoque clipper.py
    sur la vidéo doublée (pour conserver l'audio doublé dans les clips)
    en réutilisant les segments existants (skip WhisperX/Claude)."""
    duration = _probe_duration(dubbed_video)
    if duration <= AUTO_CLIP_THRESHOLD_SEC:
        return
    if not os.path.exists(seg_json):
        print(f"\n⚠️  Auto-clip ignoré : segments introuvables ({seg_json})")
        return
    print(f"\n{'='*60}")
    print(f"✂️  Vidéo longue ({duration/60:.1f} min > {AUTO_CLIP_THRESHOLD_SEC/60:.0f} min) "
          f"— extraction de {AUTO_CLIP_COUNT} clips...")
    print(f"{'='*60}")
    clipper_path = str(Path(__file__).resolve().parent / "clipper.py")
    cmd = [sys.executable, clipper_path, dubbed_video,
           "--pre-segments", seg_json,
           "-n", str(AUTO_CLIP_COUNT),
           "--target-lang", tgt_lang]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"   ⚠️  Clipper a échoué (code {e.returncode}) — vidéo doublée préservée")
    except FileNotFoundError:
        print(f"   ⚠️  clipper.py introuvable à {clipper_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global WHISPER_MODEL, CLAUDE_MODEL

    # Réserver ~15 % de la VRAM au pilote d'affichage / au navigateur pour éviter
    # de geler le bureau : sans plafond, le pipeline peut saturer les 24 Go et
    # provoquer une faute MMU (Xid 31) qui bloque tout l'écran. Avec ce plafond,
    # une saturation devient une simple erreur CUDA récupérable dans le script.
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.85)
    except ImportError:
        pass

    p = argparse.ArgumentParser(
        description="Pipeline de doublage IA — XTTS v2 (clonage vocal multilingue)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Exemples :
              python doubler.py interview.mp4                           # EN → FR, clonage vocal
              python doubler.py "https://youtube.com/watch?v=XXXXX"   # depuis YouTube
              python doubler.py video.mp4 --xtts-speaker "Craig Gutsy"  # voix preset
              python doubler.py video.mp4 -s en -t es                   # EN → ES
              python doubler.py video.mp4 --segments segs.json          # reprendre traduction
              python doubler.py video.mp4 --keep-original 0.05          # 5% voix originale
              python doubler.py video.mp4 --speakers 2                  # forcer 2 locuteurs
              python doubler.py video.mp4 --dual-audio                  # 2 pistes (doublage + original)
              python doubler.py video.mp4 --vo-style jt                 # voice-over style JT France 2
              python doubler.py video.mp4 --vo-style jt-flat            # JT, voix orig constante et basse
              python doubler.py video.mp4 --vo-style bbc               # voice-over style BBC
              python doubler.py video.mp4 --vo-lead-in 2500             # lead-in personnalisé (2,5s)
              python doubler.py video.mp4 --vo-duck-db 12               # voix orig très présente
              python doubler.py video.mp4 --no-voiceover                # doublage pur (pas de VO)
              python doubler.py video.mp4 --ref-voice ref_fr.wav        # voix de référence externe
              python doubler.py video.mp4 --benchmark                   # benchmark XTTS
              python doubler.py video.mp4 --onlydub                     # piste doublée MP3 seule
              python doubler.py video.mp4 --audio-only                  # MP3 séquentiel, sans calage temporel
              python doubler.py --list-xtts-speakers                    # lister les voix preset
        """))

    p.add_argument("video", nargs="?", help="Fichier vidéo source ou lien YouTube")
    p.add_argument("-s", "--source-lang", default="en",
                   help="Langue source (code ISO 639-1, défaut: en)")
    p.add_argument("-t", "--target-lang", default="fr",
                   help="Langue cible (code ISO 639-1, défaut: fr)")
    p.add_argument("-o", "--output", help="MP4 sortie (défaut: source_dubbed_{target}.mp4)")
    p.add_argument("--ref-voice", metavar="WAV",
                   help="Audio de référence pour le clonage vocal (10-15s). "
                        "Utilisé à la place des échantillons extraits.")
    p.add_argument("--ref-voices", metavar="DIR", default="voix",
                   help="Dossier de voix de référence genrées (homme*.wav / femme*.wav). "
                        "Affectation automatique selon le genre détecté. Défaut: voix")
    p.add_argument("--clone-original", action="store_true",
                   help="Cloner la voix originale du locuteur (ref_clips extraits) "
                        "au lieu des voix de référence du dossier voix/")
    p.add_argument("--map-voices", action="store_true",
                   help="Appariement interactif voix↔locuteur : après la diarisation, "
                        "écouter un échantillon par locuteur détecté et choisir "
                        "manuellement une voix du dossier --ref-voices. Fonctionne en "
                        "terminal (saisie clavier) et dans l'extension Chrome (UI). "
                        "Contourne l'estimation F0 / l'ordre d'apparition.")
    p.add_argument("--use-srt", metavar="SRT",
                   help="Utiliser un fichier SRT traduit professionnellement comme base. "
                        "Les sous-titres sont alignés par timecode sur les segments WhisperX, "
                        "les passes de revue Claude (4c–4e) sont sautées.")
    p.add_argument("--xtts-speaker", metavar="NAME",
                   help="Voix preset XTTS v2 (ex: 'Craig Gutsy', 'Ana Florence'). "
                        "Défaut: clonage vocal depuis les ref_clips extraits, "
                        "ou preset par genre si aucune ref disponible. "
                        "Liste complète : --list-xtts-speakers")
    p.add_argument("--list-xtts-speakers", action="store_true",
                   help="Afficher la liste des voix preset XTTS v2 et quitter")
    p.add_argument("--tts", choices=["qwen3tts", "xtts", "elevenlabs"],
                   default="qwen3tts",
                   help="Backend TTS (défaut: qwen3tts)")
    p.add_argument("--elevenlabs-voice", metavar="VOICE_ID",
                   help="Voice ID ElevenLabs pré-existant (évite le clonage IVC)")
    p.add_argument("--elevenlabs-model", metavar="MODEL",
                   default=ELEVENLABS_MODEL_DEFAULT,
                   help=f"Modèle ElevenLabs (défaut: {ELEVENLABS_MODEL_DEFAULT})")
    p.add_argument("--list-elevenlabs-voices", action="store_true",
                   help="Lister les voix ElevenLabs disponibles et quitter")
    p.add_argument("--segments", metavar="JSON",
                   help="Reprendre depuis un fichier segments (de doubler.py ou traduire.py)")
    p.add_argument("--speakers", type=int, default=None,
                   help="Forcer le nombre de locuteurs (défaut: auto)")
    p.add_argument("--gender", default="auto",
                   help="Forcer le genre des locuteurs pour le choix de voix TTS. "
                        "Valeurs : auto (estimation F0), male ou female (tous), "
                        "male,female (par ordre d'apparition), "
                        "ou SPEAKER_00=male,SPEAKER_01=female (explicite)")
    p.add_argument("--keep-original", type=float, default=0.0,
                   help="Proportion de voix originale à conserver 0.0–1.0 (défaut: 0.0)")
    p.add_argument("--dual-audio", action="store_true",
                   help="Produire un MP4 avec 2 pistes : doublage + original")
    p.add_argument("--voiceover", action="store_true", default=True,
                   help="Mode voice-over « reportage » : la voix originale reste "
                        "audible en arrière-plan (ACTIVÉ PAR DÉFAUT)")
    p.add_argument("--no-voiceover", action="store_true",
                   help="Désactiver le voice-over et produire un doublage pur "
                        "(remplace entièrement la voix originale)")
    p.add_argument("--remove-music", action="store_true",
                   help="Supprimer la musique/ambiance d'arrière-plan : "
                        "le mixage final ne contient que les voix originales "
                        "et le doublage, sur du silence "
                        "(supprime aussi les SFX d'ambiance)")
    p.add_argument("--vo-style", choices=["arte", "jt", "jt-flat", "bbc"],
                   default="jt-flat",
                   help="Preset de mixage voice-over : "
                        "jt (France 2, couvrant, DÉFAUT), "
                        "jt-flat (voix orig constante et basse), "
                        "arte (doux), bbc (intermédiaire)")
    p.add_argument("--vo-lead-in", type=int, default=None, metavar="MS",
                   help=f"Durée d'écoute de la voix originale avant le doublage "
                        f"(défaut dépend du style)")
    p.add_argument("--vo-lead-out", type=int, default=None, metavar="MS",
                   help=f"Durée d'écoute de la voix originale après le doublage "
                        f"(défaut dépend du style)")
    p.add_argument("--vo-duck-db", type=int, default=None, metavar="DB",
                   help=f"Atténuation de la voix originale pendant le doublage "
                        f"(défaut dépend du style)")
    p.add_argument("--skip-review", action="store_true",
                   help="Passer la relecture de traduction (passe 4c)")
    p.add_argument("--skip-checks", action="store_true",
                   help="Passer la vérification des dépendances")
    p.add_argument("--skip-isochrony", action="store_true",
                   help="Passer l'adaptation isochronique (passe 5)")
    p.add_argument("--skip-normalize", action="store_true",
                   help="Passer la normalisation prosodique (passe 6b)")
    p.add_argument("--skip", metavar="MM:SS", default=None,
                   help="Ignorer les N premières minutes:secondes "
                        "(ex: 2:30 pour sauter une bande-annonce)")
    p.add_argument("--fix-pitch", action="store_true",
                   help="Activer la correction F0 via WORLD vocoder en passe 6b "
                        "(par défaut : normalisation RMS seule, sans re-synthèse)")
    p.add_argument("--skip-video", action="store_true",
                   help="Produire uniquement l'audio mixé (pas de vidéo)")
    p.add_argument("--watermark", action="store_true",
                   help="Incruster le watermark 'doublage IA - @resilientstv' "
                        "dans la vidéo (désactivé par défaut, nécessite un ré-encodage)")
    p.add_argument("--benchmark", action="store_true",
                   help="Mode benchmark : produit un MP3 par backend TTS disponible, "
                        "puis s'arrête. Utilise le 1er segment traduit + voix extraite.")
    p.add_argument("--onlydub", action="store_true",
                   help="Produire uniquement un MP3 avec la piste doublée (sans voix "
                        "originale, sans fond sonore, sans vidéo). Diagnostic qualité.")
    p.add_argument("--audio-only", action="store_true",
                   help="Mode audio-only : produit un MP3 par concaténation séquentielle "
                        "des clips TTS avec pauses naturelles entre les locuteurs. "
                        "Pas de calage temporel sur l'original (durée libre), "
                        "synthèse single-pass (pas d'ajustement de vitesse), "
                        "saute isochronie, vérification temporelle, mixage et vidéo. "
                        "--speakers et --gender restent actifs (clonage par locuteur).")
    p.add_argument("--audio-only-pause", type=int, default=AUDIO_ONLY_PAUSE_MS,
                   metavar="MS",
                   help=f"Pause entre segments d'un même locuteur en mode --audio-only "
                        f"(défaut: {AUDIO_ONLY_PAUSE_MS} ms)")
    p.add_argument("--audio-only-speaker-pause", type=int,
                   default=AUDIO_ONLY_SPEAKER_PAUSE_MS, metavar="MS",
                   help=f"Pause lors d'un changement de locuteur en mode --audio-only "
                        f"(défaut: {AUDIO_ONLY_SPEAKER_PAUSE_MS} ms)")
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
    p.add_argument("--cookies", default=None,
                   help="Chemin vers le fichier cookies JSON (Epoch Times / Apollo Health)")
    args = p.parse_args()

    # --clone-original désactive le dossier voix/ pour forcer l'utilisation
    # des ref_clips extraits de la vidéo source (clonage de la voix originale)
    if getattr(args, 'clone_original', False):
        args.ref_voices = None

    # ── --list-xtts-speakers : afficher la liste et quitter ───────────────
    if getattr(args, 'list_xtts_speakers', False):
        print("\n🎤 Voix preset XTTS v2 (intégrées au modèle)")
        print("=" * 55)
        print("\n   ♀️  Voix féminines :")
        for v in XTTS_VOICES_FEMALE:
            print(f"      • {v}")
        print("\n   ♂️  Voix masculines :")
        for v in XTTS_VOICES_MALE:
            print(f"      • {v}")
        print(f"\n   Total : {len(XTTS_VOICES_FEMALE) + len(XTTS_VOICES_MALE)} voix")
        print(f"\n   Usage : --tts xtts --xtts-speaker \"Craig Gutsy\"")
        print(f"   Note  : sans --xtts-speaker, XTTS clone la voix")
        print(f"           depuis les ref_clips extraits de la vidéo.\n")
        sys.exit(0)

    # ── --list-elevenlabs-voices : afficher la liste et quitter ────────────
    if getattr(args, 'list_elevenlabs_voices', False):
        try:
            from elevenlabs import ElevenLabs
            client = ElevenLabs()
            response = client.voices.get_all()
            voices = response.voices
            print(f"\n🎤 Voix ElevenLabs disponibles ({len(voices)})")
            print("=" * 70)
            for v in voices:
                labels = ", ".join(f"{k}={val}" for k, val in (v.labels or {}).items()) if v.labels else ""
                print(f"   {v.name:30s}  {v.voice_id}  {labels}")
            print(f"\n   Usage : --tts elevenlabs --elevenlabs-voice <VOICE_ID>")
            print(f"   Note  : sans --elevenlabs-voice, le pipeline clone la voix")
            print(f"           depuis les ref_clips extraits de la vidéo (IVC).\n")
        except Exception as e:
            print(f"\n❌ Impossible de lister les voix ElevenLabs : {e}")
            print("   Vérifiez ELEVENLABS_API_KEY et le SDK (pip install elevenlabs)")
        sys.exit(0)

    if not args.video:
        p.error("l'argument video est requis (sauf avec --list-xtts-speakers / --list-elevenlabs-voices)")

    # ── Téléchargement YouTube / Epoch Times / Apollo Health si lien ─────
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    epoch_page = None
    apollo_page = None
    if is_youtube_url(args.video):
        args.video = download_youtube(args.video, output_dir=str(INPUT_DIR))
    elif epochtimes.is_epochtimes_url(args.video):
        print(f"\n📰 Source : Epoch Times")
        cookies = epochtimes.load_cookies(args.cookies)
        epoch_page = epochtimes.fetch_epoch_page(args.video, cookies)
        print(f"   Titre     : {epoch_page.title}")
        if epoch_page.speakers:
            print(f"   Locuteurs : {', '.join(epoch_page.speakers)}")
        if epoch_page.transcript:
            print(f"   Transcription : {len(epoch_page.transcript)} paragraphes")
        args.video = epochtimes.download_epoch_video(epoch_page, output_dir=str(INPUT_DIR))
    elif apollohealth.is_apollo_url(args.video):
        print(f"\n🏥 Source : Apollo Health")
        cookies = apollohealth.load_cookies(args.cookies)
        apollo_page = apollohealth.fetch_apollo_page(args.video, cookies)
        print(f"   Titre     : {apollo_page.title}")
        if apollo_page.transcript:
            print(f"   Transcription : {len(apollo_page.transcript)} spans")
        args.video = apollohealth.download_apollo_video(apollo_page, output_dir=str(INPUT_DIR))
        apollohealth.save_apollo_meta(apollo_page, args.video)

    src_lang = args.source_lang.lower()
    tgt_lang = args.target_lang.lower()

    if src_lang == tgt_lang:
        print(f"❌ Langue source et cible identiques ({src_lang})"); sys.exit(1)

    WHISPER_MODEL = args.whisper_model
    CLAUDE_MODEL = args.claude_model

    # Parsing --skip
    skip_seconds = parse_skip(args.skip)

    # Charger les sidecars si présents (créés par le daemon)
    if not epoch_page:
        epoch_page = epochtimes.load_epoch_meta(args.video)
    if not apollo_page:
        apollo_page = apollohealth.load_apollo_meta(args.video)

    # Chemins
    src = resolve_source(args.video)
    if not src.exists():
        print(f"❌ Introuvable : {args.video}"); sys.exit(1)
    args.video = str(src)
    base = src.stem
    work_dir = str(WORK_DIR / f"{base}_dubbing_work")
    os.makedirs(work_dir, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output      = args.output or str(OUTPUT_DIR / f"{base}_dubbed_{tgt_lang}.mp4")
    audio_16k   = os.path.join(work_dir, "audio_16k.wav")
    audio_hq    = os.path.join(work_dir, "audio_hq.wav")
    seg_json    = os.path.join(work_dir, "segments.json")
    ana_json    = os.path.join(work_dir, "analysis.json")
    mixed_audio = os.path.join(work_dir, "mixed_audio.wav")
    report_txt  = str(OUTPUT_DIR / f"{base}_dubbing_report.txt")

    src_n, tgt_n = lang_name(src_lang), lang_name(tgt_lang)

    print("=" * 60)
    print(f"🎙️  Pipeline de doublage IA {src_n} → {tgt_n}")
    print("=" * 60)
    print(f"   Source   : {args.video}")
    print(f"   Langues  : {src_lang.upper()} → {tgt_lang.upper()} ({src_n} → {tgt_n})")
    if args.tts == "elevenlabs":
        tts_label = "ElevenLabs"
        if getattr(args, 'elevenlabs_voice', None):
            tts_label += f" (voice: {args.elevenlabs_voice})"
        else:
            tts_label += " (clonage IVC)"
        tts_label += f" — modèle: {args.elevenlabs_model}"
    else:
        tts_label = "XTTS v2"
        if getattr(args, 'xtts_speaker', None):
            tts_label += f" (preset: {args.xtts_speaker})"
        else:
            tts_label += " (clonage vocal)"
    print(f"   TTS      : {tts_label}")
    print(f"   Sortie   : {output}")
    llm_label = f"Ollama {args.ollama_model}" if args.llm == "local" else f"Claude {CLAUDE_MODEL}"
    print(f"   Whisper  : {WHISPER_MODEL} | LLM : {llm_label}")
    if args.speakers:
        print(f"   Locuteurs: {args.speakers} (forcé)")
    if args.keep_original > 0 and args.no_voiceover:
        print(f"   Voix orig: {args.keep_original*100:.0f}%")
    if not args.no_voiceover:
        print(f"   Voice-over: 🎙️  style {args.vo_style}"
              f"{' (lead-in ' + str(args.vo_lead_in) + 'ms)' if args.vo_lead_in is not None else ''}"
              f"{' (duck -' + str(args.vo_duck_db) + 'dB)' if args.vo_duck_db is not None else ''}")
    else:
        print(f"   Voice-over: ❌ désactivé (doublage pur)")
    if args.context:
        print(f"   Contexte : {args.context[:80]}{'...' if len(args.context) > 80 else ''}")
    if skip_seconds > 0:
        print(f"   ✂️  Skip    : {format_skip(skip_seconds)} (bande-annonce ignorée)")
    if args.benchmark:
        print(f"   🏁 MODE BENCHMARK — comparaison de tous les backends TTS")
    if args.onlydub:
        print(f"   🎤 MODE ONLYDUB — piste doublée MP3 uniquement (pas de vidéo)")
    if args.audio_only:
        print(f"   🎧 MODE AUDIO-ONLY — MP3 séquentiel, sans calage temporel")
        print(f"      Pauses : {args.audio_only_pause}ms (même locuteur), "
              f"{args.audio_only_speaker_pause}ms (changement)")
    print("=" * 60)

    if args.skip_checks:
        print("   ⏩ Vérification des dépendances sautée (--skip-checks)")
    else:
        check_dependencies(args.tts)

    t_global = time.time()

    # ── API clients ──────────────────────────────────────────────────────────
    if args.llm == "local":
        claude = _OllamaClient(args.ollama_url, args.ollama_model)
    else:
        import anthropic
        claude = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Analyse : Claude apporte une meilleure connaissance du monde (noms propres,
    # domaine, glossaire) → meilleur contexte ET meilleure traduction. Un seul
    # appel, peu coûteux. "auto" = Claude si une clé est dispo.
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

    # ── PASSE 1 : Transcription ─────────────────────────────────────────────
    if args.segments:
        seg_path = args.segments
        if not os.path.exists(seg_path):
            candidate = os.path.join(work_dir, Path(seg_path).name)
            if os.path.exists(candidate):
                seg_path = candidate
        print(f"\n🔄 Chargement des segments depuis {seg_path}")
        segments = load_segments(seg_path) if "text_adapted" in open(seg_path).read() \
                   else load_traduire_segments(seg_path)
        has_translations = any(s.text_tgt for s in segments)
        print(f"   {len(segments)} segments chargés" +
              (f" ({sum(1 for s in segments if s.text_tgt)} traduits)" if has_translations else ""))
    else:
        extract_audio(args.video, audio_16k, skip_seconds)
        segments = transcribe_whisperx(audio_16k, src_lang, args.hf_token)
        save_segments(segments, seg_json)

    # ── Relecture Epoch Times (correction noms propres) ─────────────────────
    epoch_name_map = {}
    if epoch_page and epoch_page.transcript and not args.segments:
        segments, epoch_name_map = epochtimes.align_transcript_to_segments(
            epoch_page.transcript, segments, epoch_page.speakers)
        save_segments(segments, seg_json)

    # ── Relecture Apollo Health (correction noms propres, termes médicaux) ─
    if apollo_page and apollo_page.transcript and not args.segments:
        segments = apollohealth.align_transcript_to_segments(
            apollo_page.transcript, segments)
        save_segments(segments, seg_json)

    # ── Injection SRT professionnel ──────────────────────────────────────────
    skip_review = False
    if args.use_srt:
        srt_entries = load_srt_translations(args.use_srt)
        segments = align_srt_to_segments(srt_entries, segments)
        save_segments(segments, seg_json)
        skip_review = True

    # ── PASSE 1b + 3 : Audio HQ + Demucs ────────────────────────────────────
    extract_audio_hq(args.video, audio_hq, skip_seconds)
    vocals_path, bg_path = separate_sources(audio_hq, work_dir)

    if args.remove_music:
        import soundfile as sf
        info = sf.info(vocals_path)
        layout = "stereo" if info.channels == 2 else ("mono" if info.channels == 1 else f"{info.channels}c")
        silent_bg = os.path.join(work_dir, "demucs_out", "no_vocals_silent.wav")
        print(f"\n🔇 --remove-music — fond sonore remplacé par du silence "
              f"({info.duration:.1f}s, {info.samplerate} Hz, {layout})")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi",
            "-i", f"anullsrc=channel_layout={layout}:sample_rate={info.samplerate}",
            "-t", f"{info.duration:.3f}",
            "-c:a", "pcm_s16le",
            silent_bg,
        ], check=True)
        bg_path = silent_bg

    # ── PASSE 2 : Diarisation ───────────────────────────────────────────────
    if args.speakers == 1:
        # Monologue : pas besoin de diarisation, tout est SPEAKER_00
        print(f"\n👤 Passe 2 — Diarisation sautée (--speakers 1, monologue)")
        for s in segments:
            s.speaker = "SPEAKER_00"
        save_segments(segments, seg_json)
    elif epoch_name_map:
        # Attribution locuteurs depuis transcription Epoch Times — pas besoin de Pyannote
        print(f"\n👤 Passe 2 — Diarisation via transcription Epoch Times (Pyannote sautée)")
        for spk_id, name in sorted(epoch_name_map.items()):
            dur = sum(s.duration for s in segments if s.speaker == spk_id)
            print(f"   {spk_id} ({name}) : {dur:.1f}s")
        save_segments(segments, seg_json)
    elif not args.segments or all(s.speaker == "SPEAKER_00" for s in segments):
        # Diarisation nécessaire
        if not os.path.exists(audio_16k):
            extract_audio(args.video, audio_16k, skip_seconds)
        segments = diarize_speakers(audio_16k, segments, args.hf_token, args.speakers)
        save_segments(segments, seg_json)

    # ── Nettoyage mémoire avant chargement vocals ──────────────────────────
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    # ── PASSE 3b : Échantillons vocaux ──────────────────────────────────────
    profiles = extract_speaker_samples(segments, vocals_path, work_dir)

    # Forcer le genre si demandé (contourne l'estimation F0 qui peut se tromper)
    if args.gender != "auto":
        if "=" in args.gender:
            # Mapping explicite : SPEAKER_00=male,SPEAKER_01=female
            gender_map = {}
            for pair in args.gender.split(","):
                spk, g = pair.strip().split("=", 1)
                g = g.strip().lower()
                if g not in ("male", "female"):
                    print(f"   ❌ Genre invalide pour {spk} : {g} (attendu: male/female)")
                    sys.exit(1)
                gender_map[spk.strip()] = g
            for spk_id, prof in profiles.items():
                if spk_id in gender_map:
                    prof.gender = gender_map[spk_id]
                    icon = "♂️" if prof.gender == "male" else "♀️"
                    print(f"   🔧 Genre forcé → {spk_id} {icon} {prof.gender}")
                else:
                    icon = "♂️" if prof.gender == "male" else ("♀️" if prof.gender == "female" else "❓")
                    print(f"   ℹ️  {spk_id} {icon} {prof.gender} (auto F0)")
        elif "," in args.gender:
            # Liste par ordre d'apparition : male,female
            genders = [g.strip().lower() for g in args.gender.split(",")]
            for g in genders:
                if g not in ("male", "female"):
                    print(f"   ❌ Genre invalide : {g} (attendu: male/female)")
                    sys.exit(1)
            # Ordre d'apparition = premier timestamp par locuteur
            first_ts = {}
            for s in segments:
                if s.speaker and s.speaker not in first_ts:
                    first_ts[s.speaker] = s.start
            speakers_ordered = sorted(first_ts, key=lambda sp: first_ts[sp])
            if len(genders) < len(speakers_ordered):
                print(f"   ⚠️  {len(genders)} genre(s) fourni(s) pour "
                      f"{len(speakers_ordered)} locuteur(s) — les restants gardent l'auto F0")
            for i, spk_id in enumerate(speakers_ordered):
                if i < len(genders) and spk_id in profiles:
                    profiles[spk_id].gender = genders[i]
                    icon = "♂️" if genders[i] == "male" else "♀️"
                    print(f"   🔧 Genre forcé → {spk_id} {icon} {genders[i]} (apparition #{i+1})")
                elif spk_id in profiles:
                    p = profiles[spk_id]
                    icon = "♂️" if p.gender == "male" else ("♀️" if p.gender == "female" else "❓")
                    print(f"   ℹ️  {spk_id} {icon} {p.gender} (auto F0)")
        else:
            # Genre unique pour tous : male ou female
            if args.gender not in ("male", "female"):
                print(f"   ❌ --gender invalide : {args.gender} "
                      f"(attendu: auto, male, female, male,female, ou SPEAKER_XX=male,...)")
                sys.exit(1)
            for p in profiles.values():
                p.gender = args.gender
            print(f"   🔧 Genre forcé → {args.gender} pour {len(profiles)} locuteur(s)")

    # ── Appariement interactif voix↔locuteur (--map-voices) ───────────────
    # Contourne l'estimation F0 / l'ordre d'apparition : l'utilisateur écoute
    # un échantillon par locuteur et choisit la voix. Stocké pour être injecté
    # dans le backend TTS juste avant la synthèse.
    voice_overrides = {}
    if getattr(args, "map_voices", False):
        voice_overrides = interactive_voice_map(
            profiles, getattr(args, "ref_voices", "voix") or "voix", work_dir)

    # ── Enrichir le contexte Claude avec les métadonnées source ───────────
    if epoch_page:
        epoch_ctx = epochtimes.build_epoch_context(epoch_page)
        if epoch_ctx:
            args.context = (epoch_ctx + "\n\n" + args.context).strip() if args.context else epoch_ctx
    if apollo_page:
        apollo_ctx = apollohealth.build_apollo_context(apollo_page)
        if apollo_ctx:
            args.context = (apollo_ctx + "\n\n" + args.context).strip() if args.context else apollo_ctx

    # ── PASSE 4 : Analyse + Traduction ──────────────────────────────────────
    if not any(s.text_tgt for s in segments):
        analysis = analyze_content(segments, analysis_client, src_lang, tgt_lang, args.context)
        with open(ana_json, "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)

        segments = translate_for_dubbing(segments, analysis, claude,
                                         src_lang, tgt_lang, args.context)
        save_segments(segments, seg_json)

        # ── PASSE 4c : Relecture ────────────────────────────────────────────
        if not args.skip_review:
            segments = review_dubbing_translation(segments, analysis, claude,
                                                  src_lang, tgt_lang, args.context)
            save_segments(segments, seg_json)
        else:
            print(f"\n⏩ Relecture sautée")
    else:
        print(f"\n⏩ Traduction déjà présente — passe 4 sautée")
        if os.path.exists(ana_json):
            with open(ana_json) as f: analysis = json.load(f)
        else:
            analysis = analyze_content(segments, analysis_client, src_lang, tgt_lang, args.context)

    # ── PASSE 4d : Cohérence globale ──────────────────────────────────────
    if not skip_review:
        segments = check_dubbing_consistency(segments, analysis, claude,
                                             src_lang, tgt_lang)
        save_segments(segments, seg_json)
    else:
        print(f"\n⏩ Passe 4d — Cohérence sautée (SRT professionnel)")

    # ── PASSE 4e : Vérification glossaire ─────────────────────────────────
    if not skip_review:
        segments = verify_dubbing_glossary(segments, analysis, claude,
                                           src_lang, tgt_lang)
        save_segments(segments, seg_json)
    else:
        print(f"\n⏩ Passe 4e — Glossaire sauté (SRT professionnel)")

    # ── Résolution du preset voice-over (nécessaire avant l'isochronie) ──────
    # En mode --audio-only : pas de mixage avec la piste originale, donc pas de
    # voice-over et pas d'isochronie (la durée du MP3 final est libre).
    use_voiceover = args.voiceover and not args.no_voiceover and not args.audio_only
    vo_lead_in_sec = 0.0

    if use_voiceover:
        global VO_LEAD_IN_MS, VO_LEAD_OUT_MS, VO_ORIG_DUCK_DB, VO_BG_DUCK_DB, VO_FADE_MS, VO_ORIG_BETWEEN_DB

        VO_PRESETS = {
            #                lead_in  lead_out  duck_db  bg_duck  fade_ms  between_db
            "arte":         (2000,    1000,     15,      5,       250,     -2),
            "jt":           (1200,    600,      20,      8,       150,     -3),
            "jt-flat":      (0,       0,        20,      8,       150,     -20),
            "bbc":          (1500,    800,      17,      6,       200,     -2),
        }
        style = args.vo_style
        preset = VO_PRESETS[style]
        VO_LEAD_IN_MS      = args.vo_lead_in  if args.vo_lead_in  is not None else preset[0]
        VO_LEAD_OUT_MS     = args.vo_lead_out if args.vo_lead_out is not None else preset[1]
        VO_ORIG_DUCK_DB    = args.vo_duck_db  if args.vo_duck_db  is not None else preset[2]
        VO_BG_DUCK_DB      = preset[3]
        VO_FADE_MS         = preset[4]
        VO_ORIG_BETWEEN_DB = preset[5]

        vo_lead_in_sec = VO_LEAD_IN_MS / 1000.0

    # ── Détection des frontières de phrase (pour lead-in conditionnel) ──────
    segments = detect_sentence_boundaries(segments)
    n_mid = sum(1 for s in segments if not s.is_sentence_start)
    if n_mid:
        print(f"   📝 {n_mid} segments mid-phrase détectés (lead-in supprimé)")

    # ── PASSE 5 : Isochronie ───────────────────────────────────────────────
    # En mode --audio-only, l'isochronie n'a pas de sens (durée libre).
    if args.audio_only:
        print(f"\n⏩ Adaptation isochronique sautée (--audio-only)")
        for seg in segments:
            seg.text_adapted = seg.text_tgt
    elif not args.skip_isochrony:
        segments = adapt_isochrony(segments, claude, src_lang, tgt_lang,
                                   lead_in_sec=vo_lead_in_sec)
        save_segments(segments, seg_json)

        # ── PASSE 5b : Relecture fluidité post-adaptation ────────────────
        segments = review_adapted_fluency(segments, claude, tgt_lang)
        save_segments(segments, seg_json)

        # ── PASSE 5c : Restauration des qualifieurs protégés ─────────────
        segments = restore_protected_qualifiers(segments, claude, tgt_lang)
        save_segments(segments, seg_json)
    else:
        print(f"\n⏩ Adaptation isochronique sautée")
        for seg in segments:
            seg.text_adapted = seg.text_tgt

    # ── BENCHMARK MODE ──────────────────────────────────────────────────────
    if args.benchmark:
        run_tts_benchmark(
            segments, profiles, work_dir, base, tgt_lang,
            source_lang=src_lang,
            ref_voice=getattr(args, 'ref_voice', None),
            xtts_speaker=getattr(args, 'xtts_speaker', None),
            backend=args.tts,
            elevenlabs_voice=getattr(args, 'elevenlabs_voice', None),
            elevenlabs_model=getattr(args, 'elevenlabs_model', None),
            ref_voices_dir=getattr(args, 'ref_voices', None))
        print("🏁 Benchmark terminé — pipeline arrêté.")
        sys.exit(0)

    # ── PASSE 6 : Synthèse TTS (two-pass avec speed natif) ─────────────────
    acquire_gpu_lock()   # sérialise même en reprise (--segments saute la transcription)
    # Libérer la VRAM du LLM local avant de charger le modèle TTS : sinon le
    # modèle Ollama (ex. gemma4:31b ~20 Go) reste résident (keep_alive) et le
    # TTS provoque un CUDA OOM sur 24 Go (révélé par le test du 2026-06-23).
    if args.llm == "local":
        # Ollama ne libère pas la VRAM assez vite : on décharge tout modèle
        # résident ET on VÉRIFIE qu'elle est libre avant de charger le TTS.
        free_gpu_for_task(min_free_mib=6000, timeout=60)
    tts = create_tts_backend(ref_voice=getattr(args, 'ref_voice', None),
                             target_lang=tgt_lang,
                             source_lang=src_lang,
                             xtts_speaker=getattr(args, 'xtts_speaker', None),
                             backend=args.tts,
                             elevenlabs_voice=getattr(args, 'elevenlabs_voice', None),
                             elevenlabs_model=getattr(args, 'elevenlabs_model', None),
                             ref_voices_dir=getattr(args, 'ref_voices', None))
    # Appariement manuel (--map-voices) : prioritaire sur l'affectation genrée.
    if voice_overrides:
        tts.set_voice_overrides(voice_overrides)
    try:
        segments = synthesize_all(segments, profiles, tts, work_dir,
                                  lead_in_sec=vo_lead_in_sec,
                                  audio_only=args.audio_only)
        save_segments(segments, seg_json)

        # ── PASSE 6c : Vérification temporelle + corrections ────────────
        if args.audio_only:
            print(f"\n⏩ Vérification temporelle sautée (--audio-only)")
        elif not args.skip_isochrony:
            segments = verify_and_fix_timing(segments, tts, claude,
                                             tgt_lang, work_dir,
                                             lead_in_sec=vo_lead_in_sec)
            save_segments(segments, seg_json)
        else:
            print(f"\n⏩ Vérification temporelle sautée (isochronie désactivée)")
    finally:
        tts.cleanup()
        # Libérer la mémoire GPU/CPU du modèle TTS avant le mixage
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ── PASSE 6b : Normalisation prosodique ─────────────────────────────────
    # Sautée en --audio-only : l'assemblage séquentiel fait sa propre
    # normalisation RMS finale, et la normalisation prosodique par locuteur
    # n'a pas d'intérêt sans calage temporel.
    if args.audio_only:
        print(f"\n⏩ Normalisation prosodique sautée (--audio-only)")
    elif not args.skip_normalize:
        segments = normalize_prosody(segments, work_dir, profiles=profiles,
                                     fix_pitch=args.fix_pitch)
    else:
        print(f"\n⏩ Normalisation prosodique sautée")

    # ── PASSE 7 : Mixage / Assemblage audio ────────────────────────────────
    audio_only_mp3 = None
    if args.audio_only:
        audio_only_mp3 = str(OUTPUT_DIR / f"{base}_audio_only_{tgt_lang}.mp3")
        assemble_sequential_mp3(segments, audio_only_mp3,
                                pause_ms=args.audio_only_pause,
                                speaker_pause_ms=args.audio_only_speaker_pause)
    elif args.onlydub:
        onlydub_mp3 = str(OUTPUT_DIR / f"{base}_onlydub_{tgt_lang}.mp3")
        mix_audio_onlydub(segments, bg_path, onlydub_mp3)
    else:
        if use_voiceover:
            mix_audio_voiceover(segments, bg_path, vocals_path, mixed_audio)
        else:
            mix_audio(segments, bg_path, vocals_path, mixed_audio, args.keep_original)

    # ── PASSE 8 : Assemblage vidéo ─────────────────────────────────────────
    if not args.onlydub and not args.audio_only and not args.skip_video:
        if args.dual_audio:
            assemble_dual_audio(args.video, mixed_audio, output,
                                watermark=args.watermark,
                                skip_seconds=skip_seconds)
        else:
            assemble_video(args.video, mixed_audio, output,
                           watermark=args.watermark,
                           skip_seconds=skip_seconds)

        # ── Extraction auto de clips (vidéo doublée > 45 min) ──────────────
        # On part de la vidéo doublée pour que les clips portent l'audio doublé.
        # doubler n'incruste aucun sous-titre, donc seules les captions karaoke
        # de clipper.py apparaîtront à l'écran.
        if os.path.exists(output):
            auto_clip_if_long(output, seg_json, tgt_lang)

    # ── Rapport ─────────────────────────────────────────────────────────────
    generate_report(segments, profiles, report_txt, src_lang, tgt_lang)

    # ── Résumé social ──────────────────────────────────────────────────────
    social_txt = str(OUTPUT_DIR / f"{base}_social.txt")
    generate_social_txt(segments, analysis, claude, social_txt, src_lang, tgt_lang)

    # ── Nettoyage partiel ───────────────────────────────────────────────────
    if os.path.exists(audio_16k):
        os.remove(audio_16k)

    # ── Résumé final ────────────────────────────────────────────────────────
    elapsed = time.time() - t_global
    tts_ok = sum(1 for s in segments if s.tts_path)
    spk_count = len(profiles)

    print(f"\n{'='*60}")
    print(f"🎉 Doublage terminé en {elapsed/60:.1f} min !")
    print(f"{'='*60}")
    print(f"   👥 {spk_count} locuteur(s)")
    print(f"   🗣️  {tts_ok}/{len(segments)} segments doublés")
    if args.audio_only:
        print(f"   🎧 MP3 audio-only   : {audio_only_mp3}")
    elif args.onlydub:
        print(f"   🎤 Piste doublée    : {onlydub_mp3}")
    else:
        if not args.skip_video:
            print(f"   🎬 Vidéo doublée    : {output}")
        print(f"   🎚️  Audio mixé       : {mixed_audio}")
    print(f"   📝 Rapport          : {report_txt}")
    if os.path.exists(social_txt):
        print(f"   📱 Résumé social    : {social_txt}")
    print(f"   💾 Segments (reprise): {seg_json}")
    print(f"   📁 Dossier de travail: {work_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
