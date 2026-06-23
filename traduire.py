#!/usr/bin/env python3
"""
Pipeline de traduction et sous-titrage vidéo (multilingue)
============================================================
Traduit une vidéo d'une langue source vers une langue cible
avec sous-titres incrustés.

Architecture en 8 passes :
  1. WhisperX          → transcription + timestamps mot par mot
  2. Claude (analyse)   → résumé, glossaire, terminologie
  3. Claude (traduc.)   → traduction contextuelle par chunks
  4. Claude (relect.)   → relecture cohérence + naturel
  4b. Claude (cohérence) → terminologie, registre, ton uniformes
  4c. Claude (glossaire) → vérification et correction des termes du glossaire
  5. Re-segmentation    → adaptation aux contraintes sous-titres + audit CPS
  6. ffmpeg             → incrustation SRT dans MP4

Usage :
  python traduire.py video.mp4                          # EN → FR (défaut)
  python traduire.py video.mp4 --source en --target es  # EN → ES
  python traduire.py video.mp4 -s ja -t en              # JA → EN
  python traduire.py video.mp4 -o video_fr.mp4
  python traduire.py video.mp4 --style netflix
  python traduire.py video.mp4 --skip-burn
  python traduire.py video.mp4 --resume seg.json
  python traduire.py video.mp4 --srt-only sub.srt
  python traduire.py video.mp4 --delogo 1060:685:210:30   # supprimer watermark

Prérequis :
  pip install whisperx anthropic torch torchaudio --break-system-packages
  # + clé API Anthropic (ANTHROPIC_API_KEY)
  # + ffmpeg installé
"""

import argparse
import json
import math
import fcntl
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

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

CLAUDE_MODEL = "claude-opus-4-5"  # 2026-06-16 : claude-sonnet-4-20250514 retiré le
# 15/06/2026 (404 not_found). On évite sonnet-4-6 (cf. A/B 2026-05-25 : doublons
# synonymiques « fléaux et épidémies » pour « Seuchen », étoffements et paraphrases
# malgré les règles explicites du prompt). Opus 4.5 : meilleur suivi d'instructions,
# traduction concise plus fidèle sur les sous-titres.
CLAUDE_MAX_TOKENS = 8192
CLAUDE_RETRY_MAX = 5
CLAUDE_RETRY_DELAY = 10.0              # délai initial en secondes (backoff exponentiel)

# Ollama (LLM local — alternative gratuite à l'API Claude)
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma4:31b"        # cf. bench 2026-06-22 : meilleur FR oral (traduction)
OLLAMA_NUM_PREDICT = 16384             # marge large (tokens réflexion Qwen3 inclus)

# Contraintes de sous-titrage (normes professionnelles)
MAX_CHARS_PER_LINE = 42
MAX_LINES_PER_SUB = 2
MAX_CPS = 17                  # Caractères/seconde (lecture confortable)
MIN_DURATION_SEC = 1.0
MAX_DURATION_SEC = 7.0
GAP_BETWEEN_SUBS_MS = 80
PAUSE_SPLIT_THRESHOLD = 1.5    # silence interne (s) déclenchant le découpage d'un segment
PAUSE_SPLIT_PADDING = 0.35     # padding lecture (s) appliqué à la fin de chaque sous-segment

# Détection des word timings aberrants (échecs d'alignement wav2vec2)
ABERRANT_WORD_MAX_DUR_SEC = 3.0    # un mot ne devrait jamais durer + de 3 s
ABERRANT_WORD_RATIO = 4.0          # facteur sur la médiane du segment
CHARS_PER_SECOND_ESTIMATE = 14.0   # vitesse de parole typique (≈ 0.07 s/char)
# Orphelins temporels : mot dont le voisin intra-segment est ≥ ce seuil plus loin.
# Whisper en passe globale ancre parfois un mot trop tôt/tard sur un signal acoustique
# non-pertinent (collage sonore, jingle, voix-off), même quand Demucs a laissé du résidu
# voix. Une re-transcription locale du même clip ne retrouve PAS le mot à cet endroit.
ORPHAN_GAP_THRESHOLD_SEC = 5.0
# Le snap n'est déclenché QUE si le gap est majoritairement rempli de signal acoustique
# (collage, voix-off, musique-comme-parole). Un vrai silence théâtral est laissé intact.
ORPHAN_AUDIO_THRESHOLD_DB = -50.0     # seuil RMS au-dessus duquel le gap est "actif"
ORPHAN_ACTIVE_FRACTION = 0.30         # fraction minimale du gap "actif" pour qualifier d'hallucination

# VAD strict sur vocals.wav (sortie Demucs) pour démasquer les hallucinations Whisper.
# Un mot dont [start,end] ne chevauche aucune zone speech est considéré halluciné
# et snappé sur la zone speech la plus proche temporellement.
VAD_NOISE_DB = -30.0               # seuil en dB sous lequel vocals.wav est jugé silencieux
VAD_MIN_SILENCE_SEC = 0.3          # durée minimale d'une zone de silence
VAD_WORD_MIN_OVERLAP_SEC = 0.05    # chevauchement minimal mot/zone-speech pour être valide

# Silero VAD (utilisé quand Demucs est désactivé — détection de parole sur l'audio original)
SILERO_VAD_THRESHOLD = 0.5
SILERO_MIN_SPEECH_MS = 250
SILERO_MIN_SILENCE_MS = 300

# Comblement des lacunes Demucs (parole classée musique → trou dans la transcription)
GAP_FILL_MIN_SEC = 5.0             # trou inter-segment minimal pour déclencher le comblement
GAP_FILL_SILENCE_DB = -40.0        # seuil volumedetect : en-dessous = silence pur → pas de re-transcription
GAP_FILL_ACTIVE_FRACTION = 0.25    # fraction minimale du gap « active » pour tenter WhisperX

# Chunks de traduction
CHUNK_SIZE = 60               # ~3-5 min de parole
CHUNK_OVERLAP = 8

# Styles ffmpeg
# MarginV : marge depuis le bord INFÉRIEUR (Alignment=2 par défaut).
# Valeurs basses = sous-titres plus proches du bord bas.
SUBTITLE_STYLES = {
    "default": (
        "FontName=Arial,FontSize=24,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1,MarginV=16"
    ),
    "netflix": (
        "FontName=Arial,FontSize=22,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H40000000,BorderStyle=3,Outline=0,Shadow=0,MarginV=14"
    ),
    "youtube": (
        "FontName=Roboto,FontSize=22,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H40000000,BorderStyle=3,Outline=0,Shadow=0,MarginV=12"
    ),
    "minimal": (
        "FontName=Arial,FontSize=20,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BorderStyle=1,Outline=1,Shadow=0,MarginV=14"
    ),
    "box": (
        "FontName=Arial,FontSize=22,PrimaryColour=&H00FFFFFF,"
        "BackColour=&H00000000,BorderStyle=4,Outline=0,Shadow=0,MarginV=16"
    ),
}

# Noms de langues (pour les prompts Claude — en français)
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

# Noms de langues EN ANGLAIS (pour les prompts quand nécessaire)
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

def lang_name(code: str, in_english: bool = False) -> str:
    """Retourne le nom d'une langue depuis son code ISO 639-1."""
    d = LANGUAGE_NAMES_EN if in_english else LANGUAGE_NAMES
    return d.get(code, code.upper())

def source_lang_description(source_lang: str) -> str:
    """'en' → 'anglais', 'en,he' → 'anglais et hébreu (multilingue)'"""
    langs = [x.strip() for x in source_lang.split(",")]
    names = [lang_name(l) for l in langs]
    if len(names) == 1:
        return names[0]
    return " et ".join([", ".join(names[:-1]), names[-1]]) + " (multilingue)"

def source_lang_label(source_lang: str) -> str:
    """'en' → 'EN', 'en,he' → 'EN+HE'"""
    return "+".join(l.strip().upper() for l in source_lang.split(","))

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


# Sanitizer : supprime les annotations CPS qu'Opus peut accidentellement
# laisser dans les textes corrigés (ex. « Bonjour (~21 CPS) ⚠️ »).
# Le pattern couvre aussi les variantes avec/sans espace, warning, tilde.
_CPS_ANNOT_RE = re.compile(r'\s*\(\s*~?\s*\d+(?:[.,]\d+)?\s*(?:CPS|cps|C\.P\.S\.)\s*\)\s*⚠?\uFE0F?\s*')

def _strip_cps_annot(text: str) -> str:
    """Retire toute annotation « (~N CPS) ⚠️ » d'une chaîne corrigée."""
    if not text:
        return text
    return _CPS_ANNOT_RE.sub(' ', text).strip()


# ── Sortie structurée { id → text } pour les passes Claude (traduction/relecture/cohérence) ──
# Force Claude à appeler un outil dont le schéma valide qu'on a bien {id, text}.
# Élimine les fuites de méta-commentaire (raisonnement sur tu/vous, glossaire, etc.)
# qui passaient autrefois par le parsing texte libre « [N] texte ».
SUBMIT_TEXTS_TOOL = {
    "name": "submit_texts",
    "description": (
        "Soumet la liste finale des textes produits pour chaque segment traité. "
        "Le champ text ne contient QUE le texte du segment — jamais de commentaire, "
        "jamais de raisonnement sur la politesse/tutoiement/vouvoiement, "
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


# Fragments de prompt connus — si le texte de sortie en contient un,
# c'est une fuite de prompt, pas une vraie traduction.
_PROMPT_LEAK_FRAGMENTS = [
    "tutoiement/vouvoiement",
    "tutoiement / vouvoiement",
    "passage du tu au vous",
    "passage du vous au tu",
    "passer du tu au vous",
    "passer du vous au tu",
    "cohérent selon le contexte",
    "cohérent d'un segment",
    "aucune correction",
    "voici le texte",
    "texte corrigé",
    "texte adapté",
    "nombre de caractères",
    "registre de politesse",
    "niveau de formalité",
    "du/sie cohérent",
    "tú/usted cohérent",
]

_STOPWORDS_FUITE = {
    "le", "la", "les", "un", "une", "des", "de", "du", "d'", "l'",
    "et", "ou", "à", "au", "aux", "en", "est", "c'est", "ça",
    "the", "a", "an", "of", "and", "or", "to", "is", "it",
}


def _est_fuite_prompt(texte_nouveau: str, ref_meme_langue: str = "") -> bool:
    """Détecte si texte_nouveau est une fuite de prompt plutôt qu'une traduction.

    Le second argument `ref_meme_langue` est une référence DANS LA MÊME LANGUE
    que texte_nouveau (typiquement : l'ancienne traduction qu'on compare à la
    nouvelle dans une passe de relecture/cohérence). Le chevauchement de mots
    n'a de sens qu'en intra-langue ; pour la traduction initiale source→cible,
    on n'a pas de référence dans la langue cible, donc ne PAS passer la source
    sous peine de rejeter en masse les vraies traductions (mots quasi disjoints
    entre langues éloignées comme DE↔FR).
    """
    if not texte_nouveau:
        return False
    t = texte_nouveau.lower()
    for f in _PROMPT_LEAK_FRAGMENTS:
        if f in t:
            return True
    if ref_meme_langue:
        mots_orig = set(ref_meme_langue.lower().split())
        mots_new = set(t.split())
        if len(mots_orig) >= 5 and len(mots_new) >= 5:
            if len(mots_orig & mots_new) < 2:
                return True
        if len(mots_orig) >= 3 and 0 < len(mots_new) <= 3:
            sig_orig = mots_orig - _STOPWORDS_FUITE
            sig_new = mots_new - _STOPWORDS_FUITE
            if sig_orig and sig_new and not (sig_orig & sig_new):
                return True
    return False


def _strip_claude_artifacts(txt: str) -> str:
    """Nettoie les artefacts courants de la sortie Claude (préfixes méta, flèches…)."""
    if not txt:
        return txt
    if '→' in txt:
        txt = txt.split('→', 1)[-1].strip().strip('""«»“”').strip()
    txt = re.sub(r'^(?:APR[ÈE]S|AVANT|AFTER|BEFORE)\s*:\s*', '', txt,
                 flags=re.IGNORECASE).strip()
    txt = re.sub(r'^[A-Z]{2}:\s*', '', txt).strip()
    txt = re.sub(r'^\([^)]*\)\s*', '', txt).strip()
    txt = _strip_cps_annot(txt)
    return txt


def _claude_submit_texts(client, user_prompt: str,
                         system: str = "",
                         max_tokens: int = None) -> dict:
    """
    Appelle le LLM en exigeant une sortie structurée { id : texte }.

    - Avec l'API Claude : tool_use forcé sur `submit_texts`. Claude ne peut
      plus répondre en prose ; il doit remplir le schéma. Les fuites de
      méta-commentaire (raisonnement sur tu/vous, etc.) deviennent quasi
      impossibles — il faudrait coller le commentaire DANS le champ `text`.

    - Avec Ollama (pas de tool use) : on retombe sur le parsing texte libre,
      ligne par ligne, comme l'ancien format `[N] texte`.
    """
    if max_tokens is None:
        max_tokens = CLAUDE_MAX_TOKENS
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
        if is_local:
            kwargs["system"] = system
        else:
            kwargs["system"] = [{"type": "text", "text": system,
                                 "cache_control": {"type": "ephemeral"}}]

    if not is_local:
        kwargs["tools"] = [SUBMIT_TEXTS_TOOL]
        kwargs["tool_choice"] = {"type": "tool", "name": "submit_texts"}

    resp = _claude_create(client, **kwargs)

    if is_local:
        # Fallback Ollama : parsing texte libre
        result = {}
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
            out = {}
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


# Patterns de coupure de ligne par famille de langues (pour _findsplit)
SPLIT_PATTERNS = {
    "fr": {
        "punctuation": r'[,;:!?\.…]\s',
        "conjunctions": r'\s(?:et|mais|ou|car|donc|puis|alors|parce|puisque|quand|si|que|qui)\s',
        "prepositions": r'\s(?:de|du|des|à|au|aux|en|dans|sur|pour|par|avec|sans|chez)\s',
    },
    "en": {
        "punctuation": r'[,;:!?\.…]\s',
        "conjunctions": r'\s(?:and|but|or|so|then|because|when|if|that|which|who)\s',
        "prepositions": r'\s(?:of|the|in|on|at|to|for|with|from|by|about|into)\s',
    },
    "es": {
        "punctuation": r'[,;:!?\.…¿¡]\s',
        "conjunctions": r'\s(?:y|pero|o|porque|cuando|si|que|quien|aunque|donde)\s',
        "prepositions": r'\s(?:de|del|en|a|al|por|para|con|sin|desde|sobre|entre)\s',
    },
    "de": {
        "punctuation": r'[,;:!?\.…]\s',
        "conjunctions": r'\s(?:und|aber|oder|denn|weil|wenn|dass|als|ob|sondern)\s',
        "prepositions": r'\s(?:in|an|auf|mit|von|zu|für|über|nach|aus|bei|durch)\s',
    },
    "it": {
        "punctuation": r'[,;:!?\.…]\s',
        "conjunctions": r'\s(?:e|ma|o|perché|quando|se|che|cui|anche|però)\s',
        "prepositions": r'\s(?:di|del|in|a|da|per|con|su|tra|fra|senza)\s',
    },
    "pt": {
        "punctuation": r'[,;:!?\.…]\s',
        "conjunctions": r'\s(?:e|mas|ou|porque|quando|se|que|quem|embora|porém)\s',
        "prepositions": r'\s(?:de|do|da|em|a|por|para|com|sem|sobre|entre)\s',
    },
}

# Séparateurs de conjonctions/locutions pour _splittext(), par langue cible
CONJUNCTION_SEPARATORS = {
    "fr": [' et ', ' mais ', ' ou ', ' car ', ' donc ', ' puis ', ' alors '],
    "en": [' and ', ' but ', ' or ', ' so ', ' then ', ' because '],
    "es": [' y ', ' pero ', ' o ', ' porque ', ' entonces ', ' aunque '],
    "de": [' und ', ' aber ', ' oder ', ' denn ', ' weil ', ' dann '],
    "it": [' e ', ' ma ', ' o ', ' perché ', ' quindi ', ' anche '],
    "pt": [' e ', ' mas ', ' ou ', ' porque ', ' então ', ' embora '],
}

# Mots qui ne doivent JAMAIS rester seuls en fin de sous-titre (articles, déterminants, prépositions courtes)
ORPHAN_WORDS = {
    "fr": {"le", "la", "les", "l", "un", "une", "des", "du", "de", "d", "au", "aux",
           "ce", "cet", "cette", "ces", "mon", "ma", "mes", "ton", "ta", "tes",
           "son", "sa", "ses", "notre", "nos", "votre", "vos", "leur", "leurs",
           "à", "en", "et", "ou", "ne", "se", "je", "tu", "il", "on", "nous", "vous", "ils", "elles"},
    "en": {"the", "a", "an", "my", "your", "his", "her", "its", "our", "their",
           "this", "that", "these", "those", "to", "of", "in", "on", "and", "or",
           "is", "it", "he", "she", "we", "they", "i"},
    "es": {"el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del",
           "al", "en", "y", "o", "su", "sus", "mi", "tu", "se", "me", "te", "le", "nos"},
    "de": {"der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem", "eines",
           "und", "oder", "in", "an", "auf", "zu", "von", "mit", "ich", "er", "sie", "es", "wir"},
    "it": {"il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "di", "del", "della",
           "dei", "delle", "in", "e", "o", "si", "mi", "ti", "ci", "vi", "ne"},
    "pt": {"o", "a", "os", "as", "um", "uma", "uns", "umas", "de", "do", "da",
           "dos", "das", "em", "e", "ou", "se", "me", "te", "nos"},
}

# Longueur minimale d'un sous-titre (en caractères) pour éviter les sous-titres trop courts
MIN_CHARS_PER_SUB = 10

def get_split_patterns(lang_code: str) -> list[str]:
    """Retourne les patterns de coupure pour une langue, avec fallback générique."""
    pats = SPLIT_PATTERNS.get(lang_code)
    if pats:
        return [pats["punctuation"], pats["conjunctions"], pats["prepositions"]]
    # Fallback : ponctuation uniquement (fonctionne pour toutes les langues)
    return [r'[,;:!?\.…]\s']


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURES DE DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Segment:
    index: int
    start: float
    end: float
    text: str
    text_tgt: str = ""
    words: list = field(default_factory=list)
    lang: str = ""  # langue détectée (ISO 639-1), vide = langue source unique

@dataclass
class Subtitle:
    index: int
    start: float
    end: float
    text: str

    def to_srt(self) -> str:
        return f"{self.index}\n{_fmt(self.start)} --> {_fmt(self.end)}\n{self.text}\n"

@dataclass
class ContentAnalysis:
    summary: str = ""
    glossary: dict = field(default_factory=dict)
    speakers_description: str = ""
    tone: str = ""
    domain: str = ""


def _fmt(sec: float) -> str:
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    ms = int((sec % 1) * 1000)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"


# ═══════════════════════════════════════════════════════════════════════════════
# VÉRIFICATIONS PRÉALABLES
# ═══════════════════════════════════════════════════════════════════════════════

def check_ffmpeg():
    """Vérifie que ffmpeg est disponible et que libass est compilé."""
    import shutil as _shutil
    # 1. ffmpeg existe ?
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if r.returncode != 0:
            print("❌ ffmpeg introuvable. Installez-le :"); 
            print("   Ubuntu : sudo apt install ffmpeg")
            print("   macOS  : brew install ffmpeg")
            print("   Windows: https://www.gyan.dev/ffmpeg/builds/ (version 'full')")
            sys.exit(1)
        # Extraire le chemin et la version
        version_line = r.stdout.split('\n')[0]
        ffmpeg_path = _shutil.which("ffmpeg") or "ffmpeg"
        print(f"   ffmpeg : {ffmpeg_path} ({version_line.split(' ')[2] if len(version_line.split(' ')) > 2 else '?'})")
    except FileNotFoundError:
        print("❌ ffmpeg introuvable dans le PATH.")
        sys.exit(1)

    # 2. filtre subtitles (libass) disponible ?
    r = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True)
    if "subtitles" not in r.stdout:
        ffmpeg_path = _shutil.which("ffmpeg") or "ffmpeg"
        print(f"⚠️  ATTENTION : votre ffmpeg ({ffmpeg_path}) n'a PAS le filtre 'subtitles' (libass manquant).")
        print(f"   L'incrustation des sous-titres dans la vidéo ne fonctionnera PAS.")
        print(f"   La traduction et le SRT seront générés normalement.")
        print()
        if "linuxbrew" in ffmpeg_path or "homebrew" in ffmpeg_path.lower():
            print(f"   💡 Cause probable : vous utilisez le ffmpeg Homebrew/Linuxbrew,")
            print(f"      qui est souvent compilé SANS libass.")
            print(f"      Solution : forcer le ffmpeg système :")
            print(f"        export PATH=\"/usr/bin:$PATH\"")
            print(f"        echo 'export PATH=\"/usr/bin:$PATH\"' >> ~/.bashrc")
        else:
            print(f"   💡 Solutions :")
            print(f"      Ubuntu : sudo apt install ffmpeg libass-dev")
            print(f"      macOS  : brew install ffmpeg")
            print(f"      Windows: télécharger ffmpeg 'full' (pas 'essentials') sur gyan.dev")
        print()
        return False
    else:
        print(f"   libass : ✅ (filtre subtitles disponible)")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITAIRE : TÉLÉCHARGEMENT YOUTUBE (yt-dlp)
# ═══════════════════════════════════════════════════════════════════════════════

def is_youtube_url(s: str) -> bool:
    """Détecte si la chaîne est un lien YouTube."""
    return bool(re.match(
        r'https?://(www\.)?(youtube\.com/(watch|shorts|live)|youtu\.be/)', s))


def download_youtube(url: str, output_dir: str = ".") -> str:
    """Télécharge une vidéo YouTube via yt-dlp et retourne le chemin du fichier MP4."""
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
            ["yt-dlp", "--print", "title", "--no-warnings",
             "--cookies-from-browser", "firefox",
             "--remote-components", "ejs:github", url],
            capture_output=True, text=True, timeout=30)
        title = r.stdout.strip() if r.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        title = ""

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
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--no-playlist",
        "--no-warnings",
        "--cookies-from-browser", "firefox",
        "--remote-components", "ejs:github",
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

def extract_audio(video_path: str, output_path: str) -> str:
    print("\n🎵 Extraction de l'audio...")
    cmd = ["ffmpeg", "-y", "-i", video_path,
           "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"❌ ffmpeg : {r.stderr}"); sys.exit(1)
    print(f"   ✅ {output_path}")
    return output_path


DEMUCS_CHUNK_MINUTES = 20.0  # borne RAM Demucs : > 1h en bloc → OOM-killer (~17 Go RSS)


def separate_vocals(audio_path: str, work_dir: str,
                    chunk_minutes: float = DEMUCS_CHUNK_MINUTES) -> str:
    """Isole la voix avec Demucs (utile pour les vidéos avec musique de fond,
    voix-off mixées, collages sonores). Sans ça, le VAD/wav2vec2 de WhisperX
    confond la musique vocale-like avec de la vraie parole et place les mots
    n'importe où dans les zones musicales.

    Découpe l'audio en tranches de `chunk_minutes` pour borner la RAM : un
    Demucs htdemucs sur 2h en bloc consomme > 17 Go RSS et fait tomber le
    process par OOM-killer. Réutilise un vocals.wav (final ou par tranche)
    déjà extrait pour reprendre après crash.

    Renvoie le chemin du fichier vocals.wav, ou `audio_path` en cas d'échec."""
    print(f"\n🎛️  Isolation de la voix (Demucs)...")
    t0 = time.time()
    out_dir = os.path.join(work_dir, "demucs_out")
    os.makedirs(out_dir, exist_ok=True)
    vocals_final_dir = os.path.join(out_dir, "htdemucs")
    vocals_final = os.path.join(vocals_final_dir, "vocals.wav")

    if os.path.exists(vocals_final) and os.path.getsize(vocals_final) > 0:
        mv = os.path.getsize(vocals_final) / (1024 * 1024)
        print(f"   ↩️  vocals.wav existant réutilisé ({mv:.1f} Mo)")
        return vocals_final

    duration = _probe_duration(audio_path)
    chunk_sec = chunk_minutes * 60.0
    n_chunks = max(1, math.ceil(duration / chunk_sec)) if duration > 0 else 1
    if n_chunks > 1:
        print(f"   🔪 Découpe en {n_chunks} tranche(s) de ~{chunk_minutes:.0f} min "
              f"(durée totale {duration/60:.1f} min)")

    chunks_dir = os.path.join(out_dir, "_chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    vocals_parts: list[str] = []

    for i in range(n_chunks):
        chunk_in = os.path.join(chunks_dir, f"chunk_{i:03d}.wav")
        chunk_out_dir = os.path.join(chunks_dir, f"out_{i:03d}")
        chunk_vocals = os.path.join(chunk_out_dir, "htdemucs", "vocals.wav")

        if os.path.exists(chunk_vocals) and os.path.getsize(chunk_vocals) > 0:
            print(f"   ↩️  tranche {i+1}/{n_chunks} déjà extraite, skip")
            vocals_parts.append(chunk_vocals)
            continue

        if n_chunks > 1:
            start = i * chunk_sec
            cut_cmd = ["ffmpeg", "-y", "-loglevel", "error",
                       "-ss", f"{start}", "-t", f"{chunk_sec}",
                       "-i", audio_path,
                       "-c:a", "pcm_s16le", chunk_in]
            cr = subprocess.run(cut_cmd, capture_output=True, text=True)
            if cr.returncode != 0:
                print(f"   ⚠️  Découpe ffmpeg tranche {i+1} échouée : {cr.stderr[-300:]}")
                return audio_path
            demucs_input = chunk_in
        else:
            demucs_input = audio_path

        print(f"   ▶️  tranche {i+1}/{n_chunks}...")
        # Pas de capture_output : stream direct, évite l'accumulation des
        # TQDM dans le buffer Python (qui empire la pression mémoire).
        cmd = [
            sys.executable, "-m", "demucs",
            "--two-stems=vocals",
            "--segment", "6",
            "--overlap", "0.25",
            "-o", chunk_out_dir,
            "--filename", "{stem}.{ext}",
            demucs_input,
        ]
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"   ⚠️  Demucs a échoué sur tranche {i+1} (rc={r.returncode}) — "
                  f"transcription sur l'audio original")
            return audio_path

        if not os.path.exists(chunk_vocals):
            for root, _dirs, files in os.walk(chunk_out_dir):
                for f in files:
                    if f.lower() == "vocals.wav":
                        chunk_vocals = os.path.join(root, f); break
        if not os.path.exists(chunk_vocals):
            print(f"   ⚠️  vocals.wav introuvable pour tranche {i+1}")
            return audio_path

        vocals_parts.append(chunk_vocals)
        if n_chunks > 1 and os.path.exists(chunk_in):
            try: os.remove(chunk_in)
            except OSError: pass

    os.makedirs(vocals_final_dir, exist_ok=True)
    if len(vocals_parts) == 1:
        if vocals_parts[0] != vocals_final:
            shutil.copy2(vocals_parts[0], vocals_final)
    else:
        list_file = os.path.join(chunks_dir, "concat.txt")
        with open(list_file, "w") as f:
            for p in vocals_parts:
                f.write(f"file '{os.path.abspath(p)}'\n")
        concat_cmd = ["ffmpeg", "-y", "-loglevel", "error",
                      "-f", "concat", "-safe", "0", "-i", list_file,
                      "-c", "copy", vocals_final]
        cr = subprocess.run(concat_cmd, capture_output=True, text=True)
        if cr.returncode != 0:
            concat_cmd = ["ffmpeg", "-y", "-loglevel", "error",
                          "-f", "concat", "-safe", "0", "-i", list_file,
                          "-c:a", "pcm_s16le", vocals_final]
            cr = subprocess.run(concat_cmd, capture_output=True, text=True)
            if cr.returncode != 0:
                print(f"   ⚠️  Concaténation vocals échouée : {cr.stderr[-300:]}")
                return audio_path

    try:
        shutil.rmtree(chunks_dir)
    except OSError:
        pass

    mv = os.path.getsize(vocals_final) / (1024 * 1024)
    print(f"   ✅ vocals.wav ({mv:.1f} Mo) [{time.time()-t0:.0f}s]")
    return vocals_final


def _detect_speech_intervals(vocals_path: str,
                              noise_db: float = VAD_NOISE_DB,
                              min_silence: float = VAD_MIN_SILENCE_SEC) -> list[tuple[float, float]]:
    """Détecte les zones de parole dans `vocals_path` (sortie Demucs) via
    `ffmpeg silencedetect`. Renvoie une liste d'intervalles (start, end) où la
    voix est effectivement présente. Demucs ayant déjà filtré la musique, ce qui
    reste au-dessus de VAD_NOISE_DB est de la parole avec très peu de faux
    positifs."""
    cmd = ["ffmpeg", "-i", vocals_path,
           "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
           "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    # Parse les paires silence_start / silence_end depuis stderr
    silences: list[tuple[float, float]] = []
    cur_start: Optional[float] = None
    for line in r.stderr.splitlines():
        line = line.strip()
        if "silence_start:" in line:
            try:
                cur_start = float(line.split("silence_start:")[1].split()[0])
            except (ValueError, IndexError):
                cur_start = None
        elif "silence_end:" in line and cur_start is not None:
            try:
                end_tok = line.split("silence_end:")[1].split("|")[0].strip()
                end = float(end_tok.split()[0])
                silences.append((cur_start, end))
            except (ValueError, IndexError):
                pass
            cur_start = None
    # Durée totale du fichier pour fermer le dernier intervalle de parole
    pr = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=noprint_wrappers=1:nokey=1", vocals_path],
                         capture_output=True, text=True)
    try:
        total_dur = float(pr.stdout.strip())
    except ValueError:
        total_dur = silences[-1][1] if silences else 0.0
    # Complémentaire : zones speech = [0, dur] - silences
    speech: list[tuple[float, float]] = []
    cursor = 0.0
    for s, e in silences:
        if s > cursor + 0.01:
            speech.append((cursor, s))
        cursor = max(cursor, e)
    if total_dur > cursor + 0.01:
        speech.append((cursor, total_dur))
    total_speech = sum(b - a for a, b in speech)
    print(f"   🔊 VAD vocals.wav : {len(speech)} zones de parole, {total_speech:.0f}s / {total_dur:.0f}s")
    return speech


def _detect_speech_intervals_silero(audio_path: str,
                                    threshold: float = SILERO_VAD_THRESHOLD,
                                    min_speech_ms: int = SILERO_MIN_SPEECH_MS,
                                    min_silence_ms: int = SILERO_MIN_SILENCE_MS
                                    ) -> list[tuple[float, float]]:
    """Détecte les zones de parole via Silero VAD (réseau neuronal).
    Contrairement à silencedetect, fonctionne sur l'audio original avec
    fond musical — distingue parole vs musique/bruit."""
    import torch
    t0 = time.time()
    print("   🔊 Silero VAD sur l'audio original...")
    model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad',
                                   force_reload=False, trust_repo=True)
    get_speech_ts = utils[0]  # get_speech_timestamps
    import whisperx
    audio = whisperx.load_audio(audio_path)
    wav = torch.from_numpy(audio).float()
    stamps = get_speech_ts(wav, model, threshold=threshold,
                           sampling_rate=16000,
                           min_speech_duration_ms=min_speech_ms,
                           min_silence_duration_ms=min_silence_ms,
                           return_seconds=True)
    speech = [(s["start"], s["end"]) for s in stamps]
    total_speech = sum(e - a for a, e in speech)
    total_dur = len(audio) / 16000
    print(f"   🔊 Silero VAD : {len(speech)} zones de parole, "
          f"{total_speech:.0f}s / {total_dur:.0f}s [{time.time()-t0:.1f}s]")
    return speech


def _word_speech_zone(w: dict, speech: list[tuple[float, float]],
                       min_overlap: float = VAD_WORD_MIN_OVERLAP_SEC) -> Optional[tuple[float, float]]:
    """Renvoie la zone speech que le mot chevauche d'au moins `min_overlap`, ou None."""
    ws, we = w["start"], w["end"]
    for s, e in speech:
        if min(we, e) - max(ws, s) >= min_overlap:
            return (s, e)
    return None


def _nearest_speech_zone(t: float, speech: list[tuple[float, float]]) -> Optional[tuple[float, float]]:
    """Zone speech qui contient t, ou la plus proche temporellement."""
    if not speech:
        return None
    best = None; best_d = float("inf")
    for s, e in speech:
        if s <= t <= e:
            return (s, e)
        d = min(abs(s - t), abs(e - t))
        if d < best_d:
            best_d = d; best = (s, e)
    return best


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


def transcribe_whisperx(audio_path: str, source_lang: str = "en",
                        hf_token: Optional[str] = None,
                        vad_onset: Optional[float] = None,
                        vad_offset: Optional[float] = None) -> list[Segment]:
    acquire_gpu_lock()
    import whisperx, torch, gc

    source_langs = [x.strip() for x in source_lang.split(",")]
    is_multilingual = len(source_langs) > 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lang_label = source_lang_label(source_lang)
    print(f"\n📝 Transcription WhisperX ({WHISPER_MODEL}) sur {device} [{lang_label}]...")

    t0 = time.time()
    # Multilingue : language=None pour auto-détection ; mono : langue fixée
    wx_lang = None if is_multilingual else source_langs[0]
    vad_options = {}
    if vad_onset is not None: vad_options["vad_onset"] = vad_onset
    if vad_offset is not None: vad_options["vad_offset"] = vad_offset
    if vad_options:
        print(f"   ⚙️  VAD overrides: {vad_options}")
    model = whisperx.load_model(WHISPER_MODEL, device,
                                compute_type=WHISPER_COMPUTE_TYPE,
                                language=wx_lang,
                                vad_options=vad_options or None)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=WHISPER_BATCH_SIZE, language=wx_lang)
    print(f"   Transcription : {time.time()-t0:.1f}s")

    # Alignement mot par mot
    print("   🔧 Alignement mot par mot...")
    t1 = time.time()
    if is_multilingual:
        # Essayer l'alignement avec chaque langue ; utiliser la première qui fonctionne
        aligned = False
        for try_lang in source_langs:
            try:
                model_a, metadata = whisperx.load_align_model(language_code=try_lang, device=device)
                result = whisperx.align(result["segments"], model_a, metadata, audio, device,
                                        return_char_alignments=False)
                aligned = True
                print(f"   Alignement réussi avec modèle [{try_lang}]")
                del model_a
                break
            except Exception as e:
                print(f"   ⚠️  Alignement [{try_lang}] échoué ({e}), essai suivant...")
                continue
        if not aligned:
            print("   ⚠️  Alignement impossible pour toutes les langues — timestamps segment-level uniquement")
    else:
        model_a, metadata = whisperx.load_align_model(language_code=source_langs[0], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device,
                                return_char_alignments=False)
        del model_a
    print(f"   Alignement : {time.time()-t1:.1f}s")

    del model; gc.collect()
    if device == "cuda": torch.cuda.empty_cache()

    segments = [
        Segment(index=i+1, start=s["start"], end=s["end"],
                text=s["text"].strip(), words=s.get("words", []),
                lang=s.get("language", ""))
        for i, s in enumerate(result["segments"])
    ]

    # WhisperX align() vide parfois le champ text tout en gardant les mots —
    # reconstruire le texte à partir des mots dans ce cas
    rebuilt = 0
    for seg in segments:
        if not seg.text and seg.words:
            seg.text = " ".join(w["word"] for w in seg.words if "word" in w).strip()
            if seg.text:
                rebuilt += 1
    if rebuilt:
        print(f"   🔧 {rebuilt} segment(s) reconstruits depuis l'alignement mot-à-mot")

    # Réparer les timings de mots aberrants (alignement wav2vec2 raté sur passages
    # musicaux/silencieux : un mot étalé sur 5-15 s au lieu de 0.5 s).
    segments = _repair_aberrant_word_timings(segments)
    # Le snap orphelin (gap intra-segment ≥ 5 s) est appelé dans main() avec
    # garantie que l'audio passé est vocals.wav (Demucs activé). Sinon le garde-fou
    # acoustique serait pollué par la musique de fond et générerait des faux positifs.

    dur = segments[-1].end if segments else 0
    print(f"   ✅ {len(segments)} segments ({dur/60:.1f} min)")
    if is_multilingual:
        lang_counts = {}
        for s in segments:
            l = s.lang or "?"
            lang_counts[l] = lang_counts.get(l, 0) + 1
        print(f"   🌐 Langues détectées : {', '.join(f'{k}={v}' for k, v in sorted(lang_counts.items()))}")
    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 1 bis : COMBLEMENT DES LACUNES DEMUCS
# ═══════════════════════════════════════════════════════════════════════════════

def _fill_transcription_gaps(segments, original_audio, source_lang,
                              hf_token=None, min_gap=GAP_FILL_MIN_SEC):
    """Détecte les trous dans la transcription causés par Demucs (parole classée
    en musique) et les comble en re-transcrivant l'audio original.

    Parcourt les segments consécutifs ; quand seg[i+1].start - seg[i].end > min_gap,
    extrait la région correspondante de l'audio ORIGINAL (pré-Demucs), vérifie qu'il
    y a du signal audible (pas un vrai silence), puis lance WhisperX dessus. Les
    nouveaux segments sont insérés à la bonne position avec des timestamps absolus.
    Vérifie aussi le début (avant le premier segment) et la fin de l'audio."""
    import whisperx, torch, gc

    if not segments:
        return segments

    audio_duration = _probe_duration(original_audio)
    if audio_duration <= 0:
        return segments

    # Construire la liste des trous (start, end) à investiguer
    gaps = []

    # Trou au début de l'audio
    if segments[0].start > min_gap:
        gaps.append((0.0, segments[0].start))

    # Trous inter-segments
    for i in range(len(segments) - 1):
        gap_start = segments[i].end
        gap_end = segments[i + 1].start
        if gap_end - gap_start > min_gap:
            gaps.append((gap_start, gap_end))

    # Trou à la fin de l'audio
    if audio_duration - segments[-1].end > min_gap:
        gaps.append((segments[-1].end, audio_duration))

    if not gaps:
        return segments

    print(f"\n🔍 Comblement des lacunes Demucs : {len(gaps)} trou(s) détecté(s) (seuil {min_gap:.0f}s)")
    for g_start, g_end in gaps:
        print(f"   📍 {g_start:.1f}s → {g_end:.1f}s ({g_end - g_start:.1f}s)")

    # Filtrer les trous silencieux AVANT de charger WhisperX
    active_gaps = []
    for g_start, g_end in gaps:
        frac = _gap_active_fraction(original_audio, g_start, g_end,
                                     threshold_db=GAP_FILL_SILENCE_DB)
        if frac < GAP_FILL_ACTIVE_FRACTION:
            print(f"   🤫 Trou {g_start:.1f}→{g_end:.1f}s : silence ({frac*100:.0f}% actif), ignoré")
        else:
            print(f"   🎤 Trou {g_start:.1f}→{g_end:.1f}s : signal détecté ({frac*100:.0f}% actif)")
            active_gaps.append((g_start, g_end))

    if not active_gaps:
        print(f"   ℹ️  Tous les trous sont du silence — rien à combler")
        return segments

    print(f"   🔄 {len(active_gaps)} trou(s) à re-transcrire, chargement WhisperX...")

    source_langs = [x.strip() for x in source_lang.split(",")]
    wx_lang = None if len(source_langs) > 1 else source_langs[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = whisperx.load_model(WHISPER_MODEL, device,
                                compute_type=WHISPER_COMPUTE_TYPE,
                                language=wx_lang)
    filled_segments = []
    try:
        for g_start, g_end in active_gaps:
            tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_wav.close()
            try:
                pad = 0.5
                extract_start = max(0, g_start - pad)
                extract_end = min(audio_duration, g_end + pad)
                cmd = ["ffmpeg", "-y", "-loglevel", "error",
                       "-ss", f"{extract_start:.3f}",
                       "-i", original_audio,
                       "-t", f"{extract_end - extract_start:.3f}",
                       "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
                       tmp_wav.name]
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    print(f"   ⚠️  Extraction ffmpeg échouée : {r.stderr[-200:]}")
                    continue

                clip_audio = whisperx.load_audio(tmp_wav.name)
                try:
                    result = model.transcribe(clip_audio, batch_size=WHISPER_BATCH_SIZE,
                                               language=wx_lang)
                except (IndexError, RuntimeError) as e:
                    print(f"   ⚠️  WhisperX n'a rien trouvé dans le trou "
                          f"{g_start:.1f}→{g_end:.1f}s ({e.__class__.__name__})")
                    continue

                if result.get("segments"):
                    try:
                        model_a, metadata = whisperx.load_align_model(
                            language_code=source_langs[0], device=device)
                        result = whisperx.align(result["segments"], model_a, metadata,
                                                 clip_audio, device,
                                                 return_char_alignments=False)
                        del model_a
                    except Exception as e:
                        print(f"   ⚠️  Alignement échoué sur le clip : {e}")

                clip_segs = result.get("segments", [])
                if not clip_segs:
                    print(f"   ⚠️  Aucun segment trouvé dans le trou {g_start:.1f}→{g_end:.1f}s")
                    continue

                n_new = 0
                for cs in clip_segs:
                    abs_start = cs["start"] + extract_start
                    abs_end = cs["end"] + extract_start
                    words = []
                    for w in cs.get("words", []):
                        wc = dict(w)
                        if isinstance(wc.get("start"), (int, float)):
                            wc["start"] += extract_start
                        if isinstance(wc.get("end"), (int, float)):
                            wc["end"] += extract_start
                        words.append(wc)
                    text = cs.get("text", "").strip()
                    if not text and words:
                        text = " ".join(w.get("word", "") for w in words
                                        if "word" in w).strip()
                    if not text:
                        continue
                    filled_segments.append(Segment(
                        index=0,
                        start=abs_start,
                        end=abs_end,
                        text=text,
                        words=words,
                        lang=cs.get("language", "")
                    ))
                    n_new += 1

                print(f"   ✅ {n_new} segment(s) récupéré(s) dans le trou "
                      f"{g_start:.1f}→{g_end:.1f}s")

            finally:
                try:
                    os.unlink(tmp_wav.name)
                except OSError:
                    pass
    finally:
        del model; gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    if not filled_segments:
        print(f"   ℹ️  Aucun segment récupéré dans les trous")
        return segments

    # Fusionner les segments récupérés avec les segments existants, trier par start
    all_segs = segments + filled_segments
    all_segs.sort(key=lambda s: s.start)

    # Renuméroter
    for i, seg in enumerate(all_segs):
        seg.index = i + 1

    print(f"   ✅ Comblement terminé : {len(filled_segments)} segment(s) ajouté(s) "
          f"({len(segments)} → {len(all_segs)} segments)")
    return all_segs


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 1b : OCR DES SOUS-TITRES INCRUSTÉS
# ═══════════════════════════════════════════════════════════════════════════════

def ocr_supplement_segments(segments, video_path, ocr_langs=None):
    """Extrait les sous-titres incrustés par OCR et complète les segments WhisperX faibles.

    Trois filtres empêchent l'OCR de capturer du texte non-pertinent :
    1. Confiance EasyOCR ≥ 0.4 (élimine le farsi/arabe mal lu comme du latin)
    2. Position : zone centrale uniquement (élimine les watermarks dans les coins)
    3. Qualité texte : longueur raisonnable, pas trop de chiffres/symboles
       (élimine les graphiques, documents, tickers)

    L'OCR ne remplace WhisperX que si celui-ci est faible (CPS < 3),
    ce qui correspond aux passages en langue étrangère ou aux silences.
    """
    import easyocr

    if ocr_langs is None:
        ocr_langs = ["en"]
    print(f"\n🔎 OCR des sous-titres incrustés ({', '.join(ocr_langs)})...")
    reader = easyocr.Reader(ocr_langs, gpu=True)

    # Résolution vidéo pour le crop
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path],
        capture_output=True, text=True
    )
    info = json.loads(probe.stdout)
    vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
    vw, vh = int(vstream["width"]), int(vstream["height"])
    # Zone sous-titres : bas 25%
    crop_y = int(vh * 0.75)
    crop_h = vh - crop_y
    crop_filter = f"crop={vw}:{crop_h}:0:{crop_y}"

    replaced = 0
    with tempfile.TemporaryDirectory(prefix="ocr_") as tmpdir:
        for seg in segments:
            duration = seg.end - seg.start
            if duration < 0.5:
                continue

            # Échantillonner des frames toutes les ~2s dans le segment
            sample_interval = 2.0
            n_samples = max(1, int(duration / sample_interval))
            timestamps = [seg.start + (i + 0.5) * duration / n_samples for i in range(n_samples)]

            subtitle_texts = []
            for ts in timestamps:
                frame_path = os.path.join(tmpdir, f"frame_{seg.index}_{ts:.2f}.png")
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", f"{ts:.3f}", "-i", video_path,
                     "-vf", crop_filter, "-frames:v", "1", "-q:v", "2", frame_path],
                    capture_output=True
                )
                if not os.path.exists(frame_path):
                    continue
                # detail=1 → (bbox, texte, confiance) pour chaque détection
                results = reader.readtext(frame_path, detail=1, paragraph=False)
                for (bbox, txt, conf) in results:
                    txt = txt.strip()
                    if not txt or conf < 0.4:
                        continue
                    # Position horizontale du centre du texte (0=gauche, 1=droite)
                    x_center = (bbox[0][0] + bbox[2][0]) / 2 / vw
                    # Filtrer les coins (watermarks) : garder zone centrale 15%–85%
                    if x_center < 0.15 or x_center > 0.85:
                        continue
                    # Filtrer les textes trop courts (lettres isolées)
                    if len(txt) < 3:
                        continue
                    if txt not in subtitle_texts:
                        subtitle_texts.append(txt)

            if not subtitle_texts:
                continue

            # Assembler et filtrer le texte final
            ocr_combined = " ".join(subtitle_texts)

            # Filtre qualité : le texte doit ressembler à des sous-titres,
            # pas à un document/graphique/tableau
            if not _is_subtitle_quality(ocr_combined):
                continue

            whisper_len = len(seg.text.strip())
            ocr_len = len(ocr_combined)
            whisper_cps = whisper_len / duration if duration > 0 else 999

            # Remplacer si WhisperX est faible (langue étrangère, silence, hallucination)
            # CPS < 3 = quasi rien de cohérent détecté pour cette durée
            is_weak = duration > 1.5 and whisper_cps < 3

            if is_weak and ocr_len > 5:
                print(f"   📝 Seg {seg.index} ({seg.start:.1f}–{seg.end:.1f}s) : "
                      f"WhisperX «{seg.text[:50]}» → OCR «{ocr_combined[:60]}»")
                seg.text = ocr_combined
                replaced += 1

    print(f"   ✅ OCR terminé : {replaced} segment(s) remplacé(s) sur {len(segments)}")
    return segments


def _is_subtitle_quality(text):
    """Le texte OCR ressemble-t-il à des sous-titres (vs document/graphique) ?

    Sous-titres : 1-2 phrases courtes, texte propre, peu de chiffres.
    Documents : très long, beaucoup de chiffres, caractères spéciaux, fragmenté.
    """
    if not text or len(text) < 5:
        return False
    # Trop long pour des sous-titres (2 lignes × ~50 chars max)
    if len(text) > 150:
        return False
    # Ratio de chiffres trop élevé = graphique/tableau
    digits = sum(c.isdigit() for c in text)
    if digits / len(text) > 0.20:
        return False
    # Trop de caractères OCR-garbage (symboles non standard)
    garbage = sum(c in '{}[]|\\@#$%^&*~`<>€£¥°§' for c in text)
    if garbage > 2:
        return False
    # Mots trop courts en moyenne = OCR fragmenté sur du texte non-latin
    words = text.split()
    if words and sum(len(w) for w in words) / len(words) < 2.5:
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 2 : ANALYSE DU CONTENU PAR CLAUDE
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_content(segments: list[Segment], client,
                    source_lang: str = "en", target_lang: str = "fr",
                    context: str = "") -> ContentAnalysis:
    print("\n🔍 Analyse du contenu...")

    src_name = source_lang_description(source_lang)
    tgt_name = lang_name(target_lang)

    full = "\n".join(f"[{s.index}] {s.text}" for s in segments)
    if len(full) > 80000:
        q = len(segments) // 4; mid = len(segments) // 2
        full = ("\n".join(f"[{s.index}] {s.text}" for s in segments[:q])
                + "\n[...]\n"
                + "\n".join(f"[{s.index}] {s.text}" for s in segments[mid-q//2:mid+q//2])
                + "\n[...]\n"
                + "\n".join(f"[{s.index}] {s.text}" for s in segments[-q:]))

    ctx_block = ""
    if context:
        ctx_block = f"""
INFORMATIONS FOURNIES PAR L'UTILISATEUR (à intégrer dans l'analyse,
surtout pour l'orthographe des noms propres et le glossaire) :
{context}
"""

    prompt = f"""Tu es un traducteur professionnel {src_name} → {tgt_name} spécialisé en sous-titrage.
{ctx_block}
Analyse cette transcription en {src_name} et fournis en JSON strict :
- "summary": résumé 3-5 phrases (en {tgt_name})
- "glossary": {{"terme_{source_lang}": "traduction_{target_lang}"}} pour termes techniques, noms, expressions
- "speakers_description": description brève des locuteurs (en {tgt_name})
- "tone": registre (formel/conversationnel/technique/etc.)
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

    a = ContentAnalysis(
        summary=data.get("summary", ""),
        glossary=data.get("glossary", {}),
        speakers_description=data.get("speakers_description", ""),
        tone=data.get("tone", "conversationnel"),
        domain=data.get("domain", "général"),
    )
    print(f"   📋 {a.summary[:120]}...")
    print(f"   📖 {len(a.glossary)} termes | 🎭 {a.tone} | {a.domain}")
    return a


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 3 : TRADUCTION CONTEXTUELLE PAR CHUNKS
# ═══════════════════════════════════════════════════════════════════════════════

def build_system_translate(source_lang: str, target_lang: str, analysis: ContentAnalysis,
                           context: str = "") -> str:
    """Construit le prompt système de traduction pour n'importe quelle paire de langues."""
    src_name = source_lang_description(source_lang)
    tgt_name = lang_name(target_lang)

    # Règles spécifiques selon la langue cible
    lang_specific_rules = ""
    if target_lang == "fr":
        lang_specific_rules = "3. Tutoiement/vouvoiement cohérent selon le contexte\n"
    elif target_lang == "de":
        lang_specific_rules = "3. Du/Sie cohérent selon le contexte\n"
    elif target_lang == "es":
        lang_specific_rules = "3. Tú/Usted cohérent selon le contexte\n"
    elif target_lang == "ja":
        lang_specific_rules = "3. Niveau de politesse (敬語) cohérent selon le contexte\n"
    elif target_lang == "ko":
        lang_specific_rules = "3. Niveau de politesse (존댓말/반말) cohérent selon le contexte\n"

    user_context = f"\nINSTRUCTIONS UTILISATEUR :\n{context}\n" if context else ""

    return f"""Tu es un traducteur professionnel {src_name} → {tgt_name} spécialisé en sous-titrage audiovisuel.

CONTEXTE : {analysis.summary}
Domaine : {analysis.domain} | Ton : {analysis.tone} | Locuteurs : {analysis.speakers_description}
{user_context}
GLOSSAIRE (à respecter strictement) :
{json.dumps(analysis.glossary, ensure_ascii=False, indent=2)}

RÈGLES :
1. Traduction NATURELLE et IDIOMATIQUE en {tgt_name}, jamais littérale
2. Adapte les expressions idiomatiques en équivalents naturels en {tgt_name}
{lang_specific_rules}4. Noms propres inchangés sauf conventions établies
5. Respecte le glossaire pour toute terminologie
6. SOUS-TITRAGE : vise ≤84 caractères par réplique et un débit ≤17 car/s.
   Ce sont des guides de concision, pas des limites dures ; privilégie toujours
   la clarté et le naturel, mais évite les formulations inutilement longues
7. Correspondance 1:1 stricte des numéros — ne JAMAIS fusionner
8. Ne traduis QUE les segments "À TRADUIRE", pas le contexte
9. Style oral naturel, pas littéraire
10. Nettoie hésitations et faux départs
11. UN SEUL choix de mot : si la source utilise UN mot, traduis par UN mot.
    JAMAIS d'alternatives séparées par « / » (interdit : « épidémies / fléaux »).
    JAMAIS de doublons synonymiques liés par « et »/« ou » quand la source n'a
    qu'UN mot (interdit : « les Seuchen » → « les fléaux et épidémies » ;
    « ein Treffen » → « une réunion ou une rencontre »).
    JAMAIS de gloses explicatives entre parenthèses
    (interdit : « mesures (sanitaires) claires », « le BAG (Office) »).
    Choisis le mot le plus naturel et écris-le SEUL. Les acronymes établis
    entre parenthèses sont autorisés uniquement à la première occurrence
    (ex : « l'Office fédéral de la santé publique (OFSP) »).
12. ACCORD strict en genre et nombre dans la langue cible. En français,
    « les grandes épidémies » (féminin pluriel), pas « les grands épidémies ».
13. PAS D'AJOUT contextuel : ne pas étoffer la phrase au-delà de ce que
    dit la source (interdit : ajouter « de l'histoire », « bien sûr »,
    « comme on le sait » si ce n'est pas dans le source). La traduction
    doit avoir la même densité informationnelle que l'original.

CHANTS, PRIÈRES ET PASSAGES RITUELS :
Si un segment est un chant, une prière, un mantra, une récitation ou un texte
liturgique (pali, sanskrit, latin liturgique, arabe coranique, hébreu biblique,
etc.), tu DOIS le recopier TEL QUEL sans le traduire ni le paraphraser.
Ces passages sont des performances vocales, pas du discours à traduire.
ATTENTION : ceci ne s'applique PAS aux mots ou expressions empruntés courants
(anglicismes en français, germanismes, etc.) ni au code-switching ordinaire entre
langues vivantes — ceux-là doivent être traduits normalement en {tgt_name}.

SEGMENTS DÉJÀ EN {tgt_name.upper()} :
Si un segment est DÉJÀ intégralement en {tgt_name} (pas un simple emprunt
ou code-switching isolé, mais un passage entier prononcé en {tgt_name}),
renvoie exactement le texte « [SKIP] » pour ce segment.
Le public comprend le {tgt_name}, ces passages ne seront pas sous-titrés.

SORTIE : appelle l'outil submit_texts avec un item PAR SEGMENT à traduire
({{id: numéro_segment, text: traduction_en_{tgt_name}}} — ou texte original
si chant/prière, ou « [SKIP] » si déjà en {tgt_name}).
Le champ text ne contient QUE le texte de la réplique :
aucun préfixe, aucune note, aucun raisonnement sur la politesse/tutoiement."""


def translate_chunks(segments: list[Segment], analysis: ContentAnalysis, client,
                     source_lang: str = "en", target_lang: str = "fr",
                     context: str = "") -> list[Segment]:
    print(f"\n🌍 Traduction {source_lang_label(source_lang)}→{target_lang.upper()} ({CHUNK_SIZE} seg/chunk, {CHUNK_OVERLAP} overlap)...")

    system = build_system_translate(source_lang, target_lang, analysis, context)

    n_chunks = (len(segments) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for ci in range(n_chunks):
        s, e = ci * CHUNK_SIZE, min((ci + 1) * CHUNK_SIZE, len(segments))

        if all(seg.text_tgt or not seg.text.strip() or seg.lang == target_lang
               for seg in segments[s:e]):
            print(f"   📦 Chunk {ci+1}/{n_chunks} — déjà traduit")
            continue

        print(f"   📦 Chunk {ci+1}/{n_chunks} (seg {s+1}–{e})...")
        parts = []

        # Contexte arrière (déjà traduit)
        cb = max(0, s - CHUNK_OVERLAP)
        if cb < s:
            parts.append("=== CONTEXTE PRÉCÉDENT (NE PAS retraduire) ===")
            for seg in segments[cb:s]:
                if seg.text_tgt:
                    parts += [f"[{seg.index}] {source_lang_label(source_lang)}: {seg.text}", f"[{seg.index}] {target_lang.upper()}: {seg.text_tgt}"]
            parts.append("")

        # À traduire (exclure les segments vides et ceux déjà en langue cible)
        parts.append("=== À TRADUIRE ===")
        parts += [f"[{seg.index}] {seg.text}" for seg in segments[s:e]
                  if seg.text.strip() and seg.lang != target_lang]
        parts.append("")

        # Contexte avant
        cf = min(len(segments), e + CHUNK_OVERLAP)
        if e < cf:
            parts.append("=== CONTEXTE SUIVANT (NE PAS traduire) ===")
            parts += [f"[{seg.index}] {seg.text}" for seg in segments[e:cf]]

        translations = _claude_submit_texts(client, "\n".join(parts), system=system)
        _apply_tool_translations(translations, segments, s, e, target_lang=target_lang)

        done = sum(1 for seg in segments[s:e]
                   if seg.text_tgt or not seg.text.strip() or seg.lang == target_lang)
        expected = e - s
        if done < expected:
            print(f"   ⚠️  {done}/{expected} — relance manquants...")
            _retry(segments, s, e, system, client, source_lang, target_lang)
            done = sum(1 for seg in segments[s:e]
                       if seg.text_tgt or not seg.text.strip() or seg.lang == target_lang)

        print(f"   ✅ {done}/{expected}")

    skipped = sum(1 for s in segments if s.lang == target_lang)
    translatable = sum(1 for s in segments if s.text.strip() and s.lang != target_lang)
    total = sum(1 for s in segments if s.text_tgt)
    if skipped:
        print(f"\n   🌍 Traduit : {total}/{translatable} (+ {skipped} déjà en {target_lang}, non sous-titrés)")
    else:
        print(f"\n   🌍 Traduit : {total}/{translatable}")
    return segments


_MARKER_RE = re.compile(r'^[àa]\s*traduire\b', re.IGNORECASE)

# Artefacts typiques de Claude : alternatives via « X / Y » et gloses entre parenthèses.
# Filet de sécurité appliqué après traduction (le prompt l'interdit déjà mais des
# fuites passent occasionnellement).
_ARTIFACT_SLASH_RE = re.compile(
    r"(\b[^\W\d_][\w'’\-]*(?:\s+[^\W\d_][\w'’\-]*){0,2})\s+/\s+([^\W\d_][\w'’\-]*(?:\s+[^\W\d_][\w'’\-]*){0,2}\b)",
    re.UNICODE)
# Glose : parenthèses contenant 3+ lettres minuscules sans majuscule ni chiffre
# (les acronymes type OFSP/BAG/SARS-CoV-2 contiennent au moins une majuscule
# et sont donc préservés).
_ARTIFACT_PAREN_RE = re.compile(r'\s*\(([^\W\d_]{2,}(?:[ \-][^\W\d_]+)*)\)\s*', re.UNICODE)


def _clean_translation_artifacts(segments: list[Segment]) -> list[Segment]:
    """Nettoie les artefacts de traduction Claude : alternatives séparées par
    « / » (« épidémies / fléaux » → « épidémies ») et gloses explicatives entre
    parenthèses (« mesures (sanitaires) claires » → « mesures sanitaires
    claires », ou « mesures claires » selon la glose).

    Stratégie :
    - « X / Y » avec X et Y mots de 3+ lettres entourés d'espaces → garde X
      (le mot avant le / est typiquement le choix « premier » de Claude).
      Préserve « km/h », « et/ou », acronymes/dates qui n'ont pas d'espaces
      autour du /.
    - « (mot…) » sans majuscule ni chiffre → glose explicative → supprime
      la parenthèse en gardant son contenu si la glose paraît être une
      précision plutôt qu'un terme à effacer. Heuristique : on conserve
      le contenu de la parenthèse en supprimant juste « ( » et « ) »,
      sauf si le contenu est un quasi-synonyme du mot précédent (cas rare,
      laissé à la relecture Claude). Preserve acronymes (majuscules) et
      chiffres."""
    cleaned = 0
    for seg in segments:
        if not seg.text_tgt:
            continue
        original = seg.text_tgt
        new = _ARTIFACT_SLASH_RE.sub(lambda m: m.group(1), original)
        # Pour les parenthèses : on enlève les parenthèses mais on garde le
        # contenu (ainsi « (sanitaires) » devient « sanitaires », évitant la
        # perte d'info). Si la glose est un doublon contigu (« mot (mot) »),
        # on simplifie.
        def paren_repl(m: re.Match) -> str:
            content = m.group(1)
            if any(c.isupper() for c in content):
                return m.group(0)  # acronyme → préserver
            # On garde le contenu sans parenthèses, entouré d'espaces appropriés
            return f" {content} "
        new = _ARTIFACT_PAREN_RE.sub(paren_repl, new)
        # Normalisation espaces : on ne touche QUE les espaces multiples et
        # l'espace parasite avant virgule/point. On préserve l'espace insécable
        # devant ?!:; (typographie française).
        new = re.sub(r'\s{2,}', ' ', new).strip()
        new = re.sub(r'\s+([,.])', r'\1', new)
        if new != original:
            print(f"   🧹 Artefact nettoyé : seg #{seg.index} «{original[:80]}» → «{new[:80]}»")
            seg.text_tgt = new
            cleaned += 1
    if cleaned:
        print(f"   ✅ {cleaned} segment(s) débarrassé(s) d'alternatives / gloses")
    return segments


_STOPWORDS = {
    "fr": {"le","la","les","de","du","des","un","une","et","en","est","sont",
            "que","qui","il","elle","nous","vous","ils","elles","ce","cette",
            "pas","plus","dans","pour","sur","avec","au","aux","ont","fait",
            "très","mais","aussi","comme","tout","tous","par","se","ne"},
    "en": {"the","a","an","is","are","was","were","have","has","had","will",
            "would","this","that","these","those","they","them","their","we",
            "you","he","she","it","not","but","and","or","for","with","from",
            "been","being","which","who","what","there"},
    "de": {"der","die","das","ein","eine","und","ist","sind","hat","haben",
            "wir","sie","ich","nicht","auch","auf","mit","für","den","dem",
            "des","von","zu","dass","sich","als","nach","bei","wie","wenn",
            "man","noch","aber","nur","über"},
    "es": {"el","la","los","las","un","una","de","del","en","es","que","y",
            "no","por","con","para","como","más","pero","se","su","al"},
    "it": {"il","la","le","lo","di","del","un","una","che","è","e","non",
            "per","con","come","più","ma","si","suo","al","nel"},
    "ja": {"の","は","が","を","に","で","と","も","か","な","て","だ"},
}

def _detect_lang_stopwords(text: str, candidates: set[str]) -> str:
    """Détecte la langue d'un texte parmi les candidats via les stopwords."""
    words = set(text.lower().split())
    if not words:
        return ""
    best_lang, best_score = "", 0.0
    for lang in candidates:
        sw = _STOPWORDS.get(lang, set())
        if not sw:
            continue
        score = len(words & sw) / len(words)
        if score > best_score:
            best_score = score
            best_lang = lang
    return best_lang if best_score >= 0.10 else ""


def _clear_same_lang_translations(segments: list[Segment],
                                  source_lang: str, target_lang: str) -> list[Segment]:
    """Filet de sécurité : détecte les segments dont la « traduction » est quasi
    identique à la source (signe que le segment est déjà en langue cible) et
    efface text_tgt pour éviter un sous-titre inutile.
    Ne touche pas aux segments dont la source est dans une langue source
    (ceux-là sont des échecs de traduction, pas des passages en langue cible)."""
    source_langs = {x.strip() for x in source_lang.split(",")}
    if target_lang in source_langs:
        return segments
    candidates = source_langs | {target_lang}
    cleared = 0
    for seg in segments:
        if not seg.text_tgt or not seg.text.strip():
            continue
        if seg.lang == target_lang:
            continue
        src_words = set(seg.text.lower().split())
        tgt_words = set(seg.text_tgt.lower().split())
        if not src_words or not tgt_words:
            continue
        overlap = len(src_words & tgt_words) / max(len(src_words), len(tgt_words))
        if overlap < 0.80:
            continue
        detected = _detect_lang_stopwords(seg.text, candidates)
        if detected != target_lang:
            continue
        seg.text_tgt = ""
        seg.lang = target_lang
        cleared += 1
    if cleared:
        print(f"   🧹 {cleared} segment(s) déjà en {target_lang} — sous-titres retirés")
    return segments


_SKIP_RE = re.compile(r'^\[SKIP\]$', re.IGNORECASE)

def _apply_tool_translations(translations: dict, segments: list[Segment], s: int, e: int,
                             target_lang: str = ""):
    """Affecte les traductions { id → text } issues de submit_texts aux segments du chunk.
    Filtre les fuites de prompt par fragments connus, et les marqueurs « À TRADUIRE »
    régurgités. NE PAS comparer txt à seg.text — c'est cross-langue, le heuristique
    de chevauchement de mots rejetterait à tort la quasi-totalité des traductions
    DE→FR ou EN→FR longues."""
    for idx, txt in translations.items():
        if not txt or _MARKER_RE.match(txt):
            continue
        for seg in segments[s:e]:
            if seg.index == idx:
                if _SKIP_RE.match(txt.strip()):
                    seg.lang = target_lang
                    seg.text_tgt = ""
                    print(f"      ⏭️  [{idx}] déjà en {target_lang} — pas de sous-titre")
                    break
                if _est_fuite_prompt(txt):
                    print(f"      ⚠️  fuite détectée sur [{idx}] — texte rejeté")
                    break
                seg.text_tgt = txt
                break


def _retry(segments: list[Segment], s: int, e: int, system: str, client,
           source_lang: str = "en", target_lang: str = "fr"):
    # Retry 1 : contexte minimal (identique à l'ancien comportement)
    # Exclure les segments sans texte source et ceux déjà en langue cible
    missing = [seg for seg in segments[s:e]
               if not seg.text_tgt and seg.text.strip() and seg.lang != target_lang]
    if not missing: return
    parts = ["Segments manquants :\n"]
    for seg in missing:
        ctx = [x for x in segments[max(0, seg.index-4):seg.index-1] if x.text_tgt]
        if ctx: parts.append(f"  (ctx: [{ctx[-1].index}] {ctx[-1].text_tgt})")
        parts.append(f"[{seg.index}] {seg.text}")
    translations = _claude_submit_texts(client, "\n".join(parts), system=system)
    _apply_tool_translations(translations, segments, s, e, target_lang=target_lang)

    # Retry 2 : contexte enrichi (3 segments traduits avant et après chaque manquant)
    missing2 = [seg for seg in segments[s:e]
                if not seg.text_tgt and seg.text.strip() and seg.lang != target_lang]
    if not missing2: return
    print(f"   ⚠️  Retry 2 ({len(missing2)} encore manquants)...")
    parts2 = ["Segments encore manquants — voici le contexte bilingue autour de chacun :\n"]
    for seg in missing2:
        # 3 segments traduits AVANT
        before = [x for x in segments[max(0, seg.index-6):seg.index-1] if x.text_tgt][-3:]
        for b in before:
            parts2.append(f"  [{b.index}] {source_lang_label(source_lang)}: {b.text}")
            parts2.append(f"  [{b.index}] {target_lang.upper()}: {b.text_tgt}")
        parts2.append(f"[{seg.index}] {seg.text}")
        # 3 segments traduits APRÈS
        after = [x for x in segments[seg.index:min(len(segments), seg.index+5)] if x.text_tgt][:3]
        for a in after:
            parts2.append(f"  [{a.index}] {source_lang_label(source_lang)}: {a.text}")
            parts2.append(f"  [{a.index}] {target_lang.upper()}: {a.text_tgt}")
        parts2.append("")
    translations2 = _claude_submit_texts(client, "\n".join(parts2), system=system)
    _apply_tool_translations(translations2, segments, s, e, target_lang=target_lang)


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 4 : RELECTURE
# ═══════════════════════════════════════════════════════════════════════════════

def review_translation(segments: list[Segment], analysis: ContentAnalysis, client,
                       source_lang: str = "en", target_lang: str = "fr",
                       context: str = "") -> list[Segment]:
    print("\n📖 Relecture...")
    src_name = source_lang_description(source_lang)
    tgt_name = lang_name(target_lang)
    WIN, OVL = 80, 15
    n_win = max(1, (len(segments) + WIN - OVL - 1) // (WIN - OVL))
    fixes = 0

    ctx_note = f"\nINSTRUCTIONS UTILISATEUR : {context}\n" if context else ""

    # Règle de politesse spécifique à la langue cible
    lang_politeness = ""
    if target_lang == "fr":
        lang_politeness = "5. POLITESSE : tutoiement/vouvoiement cohérent d'un segment à l'autre"
    elif target_lang == "de":
        lang_politeness = "5. POLITESSE : Du/Sie cohérent d'un segment à l'autre"
    elif target_lang == "es":
        lang_politeness = "5. POLITESSE : tú/usted cohérent d'un segment à l'autre"
    elif target_lang == "ja":
        lang_politeness = "5. POLITESSE : niveau de 敬語 cohérent d'un segment à l'autre"
    elif target_lang == "ko":
        lang_politeness = "5. POLITESSE : 존댓말/반말 cohérent d'un segment à l'autre"
    else:
        lang_politeness = "5. REGISTRE : niveau de formalité cohérent d'un segment à l'autre"

    for wi in range(n_win):
        s = wi * (WIN - OVL)
        e = min(s + WIN, len(segments))
        print(f"   🔎 Fenêtre {wi+1}/{n_win} (seg {s+1}–{e})...")

        pairs = []
        for seg in segments[s:e]:
            if not seg.text_tgt:
                continue
            dur = seg.end - seg.start
            cps = len(seg.text_tgt) / dur if dur > 0 else 0
            cps_warn = " ⚠️" if cps > MAX_CPS else ""
            pairs.append(f"[{seg.index}] {source_lang_label(source_lang)}: {seg.text} ({dur:.1f}s)")
            pairs.append(f"[{seg.index}] {target_lang.upper()}: {seg.text_tgt} (~{cps:.0f} CPS){cps_warn}")
            pairs.append("")

        prompt = f"""Réviseur professionnel de sous-titres en {tgt_name}.

GLOSSAIRE : {json.dumps(analysis.glossary, ensure_ascii=False, indent=2)}
{ctx_note}
Critères de relecture (par ordre de priorité) :
1. FIDÉLITÉ : pas de contresens, chiffres/noms/données factuelles exacts
2. NATUREL : formulations idiomatiques en {tgt_name}, style oral, pas littéraire
3. GLOSSAIRE : terminologie conforme au glossaire ci-dessus
4. CONCISION/CPS : les segments marqués ⚠️ dépassent {MAX_CPS} car/s — raccourcir si possible sans perdre le sens
{lang_politeness}
6. LISIBILITÉ : phrases compréhensibles en une lecture, pas de structures alambiquées
7. CHANTS/PRIÈRES : si un segment est un chant, une prière, un mantra ou un
   texte liturgique (pali, sanskrit, latin liturgique, arabe coranique, etc.),
   il doit être CONSERVÉ TEL QUEL — ne jamais le modifier ni le traduire.
   (Ceci ne concerne PAS les emprunts courants entre langues vivantes.)
8. UN SEUL choix de mot : remplacer toute alternative séparée par « / »
   (« épidémies / fléaux » → UN mot), tout doublon synonymique lié par
   « et »/« ou » quand la source n'a qu'UN mot (« fléaux et épidémies »
   pour « Seuchen » → un seul mot), et toute glose explicative entre
   parenthèses (« mesures (sanitaires) claires » → « mesures sanitaires
   claires »). Seuls les acronymes établis sont autorisés entre parenthèses
   à la première occurrence.
9. ACCORD strict en genre et nombre. Corriger toute faute d'accord
   (« les grands épidémies » → « les grandes épidémies »).
10. PAS D'ÉTOFFEMENT : retirer tout ajout contextuel absent de la source
    (« de l'histoire », « comme on sait », « bien sûr » ajoutés ad libitum).

SORTIE — appelle l'outil submit_texts avec un item UNIQUEMENT pour les
segments à corriger ({{id: numéro, text: nouvelle_traduction}}).
- Omets les segments qui restent corrects (n'envoie pas de tableau "items"
  pour eux ; un appel avec items vide signifie « aucune correction »).
- Le champ text contient UNIQUEMENT la nouvelle traduction. JAMAIS l'annotation
  « (~N CPS) », JAMAIS d'icône ⚠️, JAMAIS de raisonnement sur la politesse
  ou le tutoiement/vouvoiement, JAMAIS de préfixe « AVANT/APRÈS ».

{chr(10).join(pairs)}"""

        orig_map = {seg.index: seg.text_tgt for seg in segments[s:e]}
        corrections = _claude_submit_texts(client, prompt)
        for idx, new in corrections.items():
            if not new:
                continue
            for seg in segments:
                if seg.index == idx and seg.text_tgt != new:
                    if _est_fuite_prompt(new, orig_map.get(idx, seg.text_tgt)):
                        print(f"      ⚠️  fuite détectée sur [{idx}] — correction ignorée")
                        break
                    seg.text_tgt = new
                    fixes += 1
                    break
        if fixes: print(f"   ✏️  corrections en cours...")

    print(f"   📖 {fixes} corrections totales")
    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 4b : COHÉRENCE GLOBALE
# ═══════════════════════════════════════════════════════════════════════════════

def check_consistency(segments: list[Segment], analysis: ContentAnalysis, client,
                      source_lang: str = "en", target_lang: str = "fr") -> list[Segment]:
    """Passe de cohérence globale : terminologie, registre de politesse, ton."""
    print("\n🔗 Cohérence globale...")
    tgt_name = lang_name(target_lang)

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

        prompt = f"""Vérificateur de cohérence pour sous-titres en {tgt_name}.

GLOSSAIRE : {json.dumps(analysis.glossary, ensure_ascii=False, indent=2)}
Ton attendu : {analysis.tone} | Domaine : {analysis.domain}

Vérifie UNIQUEMENT ces 3 points sur l'ensemble des sous-titres ci-dessous :
1. TERMINOLOGIE : un même concept source est-il toujours traduit de la même façon ?
2. REGISTRE DE POLITESSE : le niveau de formalité est-il constant ?
3. TON : le ton ({analysis.tone}) est-il maintenu uniformément ?

Ne corrige PAS le style, la grammaire ou la concision — seulement les incohérences ci-dessus.

SORTIE — appelle l'outil submit_texts avec un item UNIQUEMENT pour chaque
segment incohérent à corriger ({{id: numéro, text: nouvelle_traduction}}).
- Si tout est cohérent, renvoie items vide.
- Le champ text contient UNIQUEMENT la nouvelle traduction du sous-titre.
- JAMAIS de raisonnement sur le tutoiement/vouvoiement ou le registre dans text.

{chr(10).join(lines)}"""

        orig_map = {seg.index: seg.text_tgt for seg in batch}
        corrections = _claude_submit_texts(client, prompt)
        for idx, new in corrections.items():
            if not new:
                continue
            for seg in segments:
                if seg.index == idx and seg.text_tgt != new:
                    if _est_fuite_prompt(new, orig_map.get(idx, seg.text_tgt)):
                        print(f"      ⚠️  fuite détectée sur [{idx}] — correction ignorée")
                        break
                    seg.text_tgt = new
                    fixes += 1
                    break

    print(f"   🔗 {fixes} corrections de cohérence")
    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 4c : VÉRIFICATION GLOSSAIRE
# ═══════════════════════════════════════════════════════════════════════════════

def verify_glossary(segments: list[Segment], analysis: ContentAnalysis, client,
                    source_lang: str = "en", target_lang: str = "fr") -> list[Segment]:
    """Vérifie que les termes du glossaire sont appliqués et corrige les violations."""
    if not analysis.glossary:
        print("\n📖 Vérification glossaire — glossaire vide, passage ignoré")
        return segments

    print("\n📖 Vérification glossaire...")
    tgt_name = lang_name(target_lang)

    # Scan case-insensitive des violations
    violations = []
    for seg in segments:
        if not seg.text_tgt:
            continue
        tgt_lower = seg.text_tgt.lower()
        src_lower = seg.text.lower()
        for src_term, tgt_term in analysis.glossary.items():
            # Le terme source apparaît dans le texte source
            if src_term.lower() in src_lower:
                # Mais la traduction attendue n'apparaît pas dans le texte cible
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
            lines.append(f"[{seg.index}] {source_lang_label(source_lang)}: {seg.text}")
            lines.append(f"[{seg.index}] {target_lang.upper()}: {seg.text_tgt}")
            lines.append(f"  → Le terme « {src_term} » devrait être traduit « {tgt_term} »")
            lines.append("")

        prompt = f"""Correcteur de glossaire pour sous-titres en {tgt_name}.

Pour chaque segment ci-dessous, le glossaire n'a pas été respecté.
Corrige NATURELLEMENT la traduction pour intégrer le terme correct du glossaire,
sans rendre la phrase artificielle.

GLOSSAIRE COMPLET : {json.dumps(analysis.glossary, ensure_ascii=False, indent=2)}

SORTIE — appelle l'outil submit_texts avec un item par segment corrigé
({{id: numéro, text: traduction_corrigée_intégrant_le_terme_du_glossaire}}).
Le champ text contient UNIQUEMENT la nouvelle traduction, sans préfixe ni note.

{chr(10).join(lines)}"""

        orig_map = {seg.index: seg.text_tgt for seg, _, _ in batch}
        corrections = _claude_submit_texts(client, prompt)
        for idx, new in corrections.items():
            if not new:
                continue
            for seg in segments:
                if seg.index == idx and seg.text_tgt != new:
                    if _est_fuite_prompt(new, orig_map.get(idx, seg.text_tgt)):
                        print(f"      ⚠️  fuite détectée sur [{idx}] — correction ignorée")
                        break
                    seg.text_tgt = new
                    fixes += 1
                    break

    print(f"   📖 {fixes} corrections glossaire")
    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 5 : RE-SEGMENTATION SOUS-TITRES
# ═══════════════════════════════════════════════════════════════════════════════

def _gap_active_fraction(audio_path: str, t_start: float, t_end: float,
                          threshold_db: float = ORPHAN_AUDIO_THRESHOLD_DB,
                          win_sec: float = 0.25, hop_sec: float = 0.05) -> float:
    """Renvoie la fraction du gap [t_start, t_end] où le RMS lissé est au-dessus
    de `threshold_db`. Sert à distinguer un trou rempli de collage/voix-off
    (fraction haute → hallucination probable du mot orphelin) d'un vrai silence
    théâtral (fraction basse → garder le mot intact)."""
    if t_end - t_start < 0.5:
        return 0.0
    try:
        import soundfile as sf, numpy as np
    except ImportError:
        return 1.0  # pas de soundfile → comportement aggressif par défaut
    try:
        info = sf.info(audio_path)
        sr = info.samplerate
        # Lire uniquement la fenêtre [t_start, t_end] pour éviter de charger tout l'audio
        frames_start = max(0, int(t_start * sr))
        frames_end = min(info.frames, int(t_end * sr))
        if frames_end - frames_start < int(win_sec * sr):
            return 0.0
        y, _ = sf.read(audio_path, start=frames_start, stop=frames_end, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
    except Exception:
        return 1.0
    win = int(win_sec * sr); hop = int(hop_sec * sr)
    if len(y) < win:
        return 0.0
    n = (len(y) - win) // hop + 1
    active = 0
    for i in range(n):
        seg = y[i*hop : i*hop + win]
        rms = float((seg * seg).mean()) ** 0.5
        if rms < 1e-12:
            continue
        db = 20.0 * (math.log10(rms) if rms > 0 else -10)
        if db > threshold_db:
            active += 1
    return active / max(1, n)


def _snap_orphan_words(segments: list[Segment], audio_path: Optional[str] = None) -> list[Segment]:
    """Détecte et corrige les mots orphelins : isolés au sein d'un segment par
    un trou intra-segment >= ORPHAN_GAP_THRESHOLD_SEC, ET dont le trou contient
    en majorité du signal acoustique non-silencieux (collage, voix-off, musique).

    Sans le contrôle acoustique, un vrai silence théâtral (orateur qui se tait
    15 s entre deux mots pour l'effet) serait à tort « corrigé » → bug majeur.
    Avec le contrôle, on ne snap que les cas où Whisper a clairement placé le
    mot sur un signal acoustique non-pertinent (typiquement un collage sonore
    multilingue qui partage le segment d'un mot allemand)."""
    if not audio_path or not os.path.exists(audio_path):
        return segments  # impossible de qualifier acoustiquement → ne rien faire

    snapped = 0
    skipped_silence = 0
    for seg in segments:
        ws = [w for w in (seg.words or []) if isinstance(w, dict)
              and isinstance(w.get('start'), (int, float))
              and isinstance(w.get('end'), (int, float))]
        if len(ws) < 2:
            continue
        # Orphelins de début
        i = 0
        while i < len(ws) - 1:
            gap = ws[i+1]['start'] - ws[i]['end']
            if gap < ORPHAN_GAP_THRESHOLD_SEC:
                break
            frac_active = _gap_active_fraction(audio_path, ws[i]['end'], ws[i+1]['start'])
            if frac_active < ORPHAN_ACTIVE_FRACTION:
                skipped_silence += 1
                print(f"   🤫 Silence théâtral préservé : seg #{seg.index} avant «{ws[i+1].get('word','')}» "
                      f"({ws[i]['end']:.2f}→{ws[i+1]['start']:.2f}, gap {gap:.1f}s, {frac_active*100:.0f}% actif)")
                break
            est_dur = max(0.25, len(str(ws[i].get('word','')).strip()) / CHARS_PER_SECOND_ESTIMATE)
            old_s, old_e = ws[i]['start'], ws[i]['end']
            new_e = ws[i+1]['start'] - 0.05
            new_s = new_e - est_dur
            if i > 0:
                new_s = max(new_s, ws[i-1]['end'] + 0.05)
            ws[i]['start'], ws[i]['end'] = new_s, new_e
            print(f"   🚚 Mot orphelin (début) recalé : seg #{seg.index} «{ws[i].get('word','')}» "
                  f"({old_s:.2f}→{old_e:.2f}) → ({new_s:.2f}→{new_e:.2f}, gap {frac_active*100:.0f}% actif)")
            snapped += 1
            i += 1
        # Orphelins de fin
        j = len(ws) - 1
        while j > 0:
            gap = ws[j]['start'] - ws[j-1]['end']
            if gap < ORPHAN_GAP_THRESHOLD_SEC:
                break
            frac_active = _gap_active_fraction(audio_path, ws[j-1]['end'], ws[j]['start'])
            if frac_active < ORPHAN_ACTIVE_FRACTION:
                skipped_silence += 1
                print(f"   🤫 Silence théâtral préservé : seg #{seg.index} après «{ws[j-1].get('word','')}» "
                      f"({ws[j-1]['end']:.2f}→{ws[j]['start']:.2f}, gap {gap:.1f}s, {frac_active*100:.0f}% actif)")
                break
            est_dur = max(0.25, len(str(ws[j].get('word','')).strip()) / CHARS_PER_SECOND_ESTIMATE)
            old_s, old_e = ws[j]['start'], ws[j]['end']
            new_s = ws[j-1]['end'] + 0.05
            new_e = new_s + est_dur
            if j + 1 < len(ws):
                new_e = min(new_e, ws[j+1]['start'] - 0.05)
            ws[j]['start'], ws[j]['end'] = new_s, new_e
            print(f"   🚚 Mot orphelin (fin) recalé : seg #{seg.index} «{ws[j].get('word','')}» "
                  f"({old_s:.2f}→{old_e:.2f}) → ({new_s:.2f}→{new_e:.2f}, gap {frac_active*100:.0f}% actif)")
            snapped += 1
            j -= 1
        if ws:
            seg.start = ws[0]['start']
            seg.end = ws[-1]['end']
    if snapped or skipped_silence:
        print(f"   ✅ Orphelins : {snapped} recalé(s), {skipped_silence} silence(s) théâtral(aux) préservé(s)")
    return segments


def _snap_hallucinated_words(segments: list[Segment],
                              speech_intervals: list[tuple[float, float]]) -> list[Segment]:
    """Pour chaque mot dont [start,end] ne chevauche aucune zone de parole de
    vocals.wav, le mot est une hallucination de Whisper (transcrit dans une zone
    silencieuse après isolation Demucs). On le snap sur la zone speech la plus
    proche, en le collant au voisin fiable (mot transcrit qui EST dans une zone
    speech) le plus proche temporellement, inter-segments compris.

    Garantit qu'aucun mot ne peut survivre dans une zone que Demucs juge
    silencieuse — élimine les sous-titres apparaissant en avance sur du silence.
    """
    if not speech_intervals:
        return segments

    # Liste à plat de tous les mots avec leur position (seg_idx, word_idx)
    flat: list[tuple[int, int, dict]] = []
    for si, seg in enumerate(segments):
        for wi, w in enumerate(seg.words or []):
            if isinstance(w, dict) and isinstance(w.get("start"), (int, float)) \
                                     and isinstance(w.get("end"), (int, float)):
                flat.append((si, wi, w))
    if not flat:
        return segments

    # Pour chaque mot, savoir s'il est dans une zone speech (cache)
    in_speech = [_word_speech_zone(w, speech_intervals) is not None for _, _, w in flat]

    # Garde-fou : si la VAD est manifestement défaillante, ne pas snapper
    n_in = sum(in_speech)
    n_out = len(in_speech) - n_in
    if n_out > 0 and len(in_speech) > 10:
        ratio_out = n_out / len(in_speech)
        if ratio_out > 0.50:
            total_speech = sum(b - a for a, b in speech_intervals)
            audio_span = flat[-1][2]["end"] - flat[0][2]["start"]
            coverage = total_speech / audio_span if audio_span > 0 else 1.0
            print(f"   ⚠️  VAD suspecte : {n_out}/{len(in_speech)} mots ({ratio_out:.0%}) hors zones de parole, "
                  f"couverture VAD {coverage:.0%} — snapping désactivé pour éviter la compression catastrophique")
            return segments

    snapped = 0
    for fi, (si, _wi, w) in enumerate(flat):
        if in_speech[fi]:
            continue
        # Mot halluciné : chercher le voisin fiable le plus proche temporellement
        next_w = None; next_d = float("inf")
        for fj in range(fi + 1, len(flat)):
            if in_speech[fj]:
                next_w = flat[fj][2]
                next_d = next_w["start"] - w["end"]
                break
        prev_w = None; prev_d = float("inf")
        for fj in range(fi - 1, -1, -1):
            if in_speech[fj]:
                prev_w = flat[fj][2]
                prev_d = w["start"] - prev_w["end"]
                break
        if next_w is None and prev_w is None:
            continue  # aucun voisin fiable, on laisse tel quel

        est_dur = max(0.25, len(str(w.get("word", "")).strip()) / CHARS_PER_SECOND_ESTIMATE)
        old_s, old_e = w["start"], w["end"]
        snapped_to = None

        # Choix : voisin le plus proche temporellement (en distance absolue)
        prefer_next = (next_w is not None) and (next_d <= prev_d or prev_w is None)
        if prefer_next:
            zone = _nearest_speech_zone(next_w["start"], speech_intervals)
            if zone is not None:
                new_e = max(zone[0] + est_dur, min(zone[1], next_w["start"] - 0.05))
                new_s = max(zone[0], new_e - est_dur)
                # Garde-fou : éviter de chevaucher prev_w
                if prev_w is not None and new_s < prev_w["end"] + 0.05:
                    new_s = prev_w["end"] + 0.05
                    new_e = max(new_s + 0.20, new_e)
                if new_s < new_e:
                    w["start"], w["end"] = new_s, new_e
                    snapped_to = "next"
        if snapped_to is None and prev_w is not None:
            zone = _nearest_speech_zone(prev_w["end"], speech_intervals)
            if zone is not None:
                new_s = min(zone[1] - est_dur, max(zone[0], prev_w["end"] + 0.05))
                new_e = min(zone[1], new_s + est_dur)
                # Garde-fou : éviter de chevaucher next_w
                if next_w is not None and new_e > next_w["start"] - 0.05:
                    new_e = next_w["start"] - 0.05
                    new_s = min(new_e - 0.20, new_s)
                if new_s < new_e:
                    w["start"], w["end"] = new_s, new_e
                    snapped_to = "prev"
        if snapped_to is None:
            continue
        snapped += 1
        print(f"   🎯 Mot halluciné snappé : seg #{segments[si].index} «{w.get('word','')}» "
              f"({old_s:.2f}→{old_e:.2f}) → ({w['start']:.2f}→{w['end']:.2f}) [voisin={snapped_to}]")

    if snapped:
        # Re-synchroniser les bornes de segment sur leurs mots (après déplacement)
        for seg in segments:
            ws = [x for x in (seg.words or []) if isinstance(x, dict)
                  and isinstance(x.get("start"), (int, float))
                  and isinstance(x.get("end"), (int, float))]
            if ws:
                seg.start = ws[0]["start"]
                seg.end = ws[-1]["end"]
        print(f"   ✅ {snapped} mot(s) halluciné(s) recalé(s) sur la VAD vocals.wav")
    return segments


def _repair_aberrant_word_timings(segments: list[Segment]) -> list[Segment]:
    """Détecte les mots dont la durée est aberrante (échec d'alignement wav2vec2)
    et reconstruit le bord manquant à partir d'une durée estimée.

    Cause typique : sur des passages musicaux ou des collages sonores, le forced
    alignment échoue et étale un mot sur toute la fenêtre de silence disponible
    (ex : « Klare » qui dure 7 s au lieu de 0.5 s, parce que le speaker fait une
    longue pause théâtrale et que wav2vec2 « remplit » jusqu'au mot suivant).

    Critère d'aberration : durée > max(ABERRANT_WORD_MAX_DUR_SEC,
    ABERRANT_WORD_RATIO × médiane des durées du segment).

    Réparation : on garde le bord proche d'un voisin fiable (gap < 2 s) et on
    reconstruit l'autre bord via len(mot) / CHARS_PER_SECOND_ESTIMATE. Pour un
    mot isolé (les deux gaps > 2 s), on garde `end` (souvent contraint par le
    mot suivant) et on remonte `start`."""
    repaired = 0
    for seg in segments:
        ws = [w for w in (seg.words or []) if isinstance(w, dict) and 'start' in w and 'end' in w]
        if len(ws) < 2:
            continue
        durs = sorted(w['end'] - w['start'] for w in ws)
        median_dur = durs[len(durs) // 2]
        threshold = max(ABERRANT_WORD_MAX_DUR_SEC, ABERRANT_WORD_RATIO * median_dur)

        for i, w in enumerate(ws):
            dur = w['end'] - w['start']
            if dur <= threshold:
                continue
            est_dur = max(0.25, len(w.get('word', '')) / CHARS_PER_SECOND_ESTIMATE)
            prev_gap = w['start'] - ws[i-1]['end'] if i > 0 else 999.0
            next_gap = ws[i+1]['start'] - w['end'] if i + 1 < len(ws) else 999.0
            old_s, old_e = w['start'], w['end']
            # Stratégie conservative : on suppose qu'un seul des deux bords est faux
            # (celui éloigné du voisin) et on raccourcit la fenêtre vers le bord
            # « proche d'un voisin ». Pas de devinette sur l'autre bord.
            if next_gap < 2.0:
                # Bord droit fiable → raccourcir start vers end
                w['start'] = max(w['end'] - est_dur, ws[i-1]['end'] if i > 0 else 0.0)
            elif prev_gap < 2.0:
                # Bord gauche fiable → raccourcir end vers start
                w['end'] = min(w['start'] + est_dur, ws[i+1]['start'] if i+1 < len(ws) else w['end'])
            else:
                # Mot isolé : on ne sait pas où la parole est réellement. On garde
                # `start` (souvent contraint par la borne de segment) et on raccourcit
                # `end` à une durée plausible — accepte que le timing soit imparfait.
                w['end'] = w['start'] + est_dur
            print(f"   🔧 Word aberrant réparé : seg #{seg.index} «{w.get('word','')}» "
                  f"({old_s:.2f}→{old_e:.2f}, {dur:.1f}s) → "
                  f"({w['start']:.2f}→{w['end']:.2f}, {w['end']-w['start']:.2f}s)")
            repaired += 1

        # Si le 1er ou dernier mot a été décalé, resynchroniser la borne de segment
        # correspondante (sinon le sous-titre garde l'ancienne fin/début aberrant).
        if ws[0]['start'] > seg.start + 0.5:
            seg.start = ws[0]['start']
        if ws[-1]['end'] < seg.end - 0.5:
            seg.end = ws[-1]['end']
    if repaired:
        print(f"   ✅ {repaired} timing(s) de mot aberrant(s) réparé(s)")
    return segments


def _split_tgt_on_punct(tgt: str, fractions: list[float]) -> list[str]:
    """Découpe `tgt` en N portions selon les fractions cumulées données.
    Préfère couper sur ponctuation forte (. ! ? ; : ,) proche de la position cible
    ; fallback : coupe sur espace. Renvoie une liste de N chaînes (potentiellement
    vides en cas d'échec — l'appelant doit vérifier)."""
    n = len(fractions)
    if n <= 1:
        return [tgt.strip()]
    total = len(tgt)
    parts = []
    pos = 0
    acc = 0.0
    for f in fractions[:-1]:
        acc += f
        target = int(acc * total)
        window = max(10, int(0.25 * total))
        lo, hi = max(pos + 1, target - window), min(total - 1, target + window)
        cut = -1
        if lo <= hi:
            best_score = -1e9
            for j in range(lo, hi + 1):
                ch = tgt[j-1]
                if ch in '.!?': base = 100
                elif ch in ';:': base = 80
                elif ch == ',': base = 60
                elif ch == ' ': base = 20
                else: continue
                score = base - abs(j - target)
                if score > best_score:
                    best_score = score; cut = j
        if cut < 0:
            cut = max(pos + 1, min(total - 1, target))
        parts.append(tgt[pos:cut].strip())
        pos = cut
    parts.append(tgt[pos:].strip())
    return parts


def _split_on_pauses(segments: list[Segment]) -> list[Segment]:
    """Quand WhisperX produit un long segment de parole traversé par un silence
    interne (>= PAUSE_SPLIT_THRESHOLD secondes), redécoupe le segment en
    sous-segments calés sur les clusters de parole réels. Le texte cible est
    réparti aux frontières de ponctuation proportionnellement à la longueur
    source de chaque cluster."""
    result = []
    split_count = 0
    for seg in segments:
        words = seg.words or []
        wts = [w for w in words if isinstance(w, dict) and 'start' in w and 'end' in w]
        if len(wts) < 2 or not seg.text_tgt:
            result.append(seg); continue

        # Clusterisation par silence
        clusters = [[wts[0]]]
        for w in wts[1:]:
            if w['start'] - clusters[-1][-1]['end'] >= PAUSE_SPLIT_THRESHOLD:
                clusters.append([w])
            else:
                clusters[-1].append(w)

        if len(clusters) == 1:
            result.append(seg); continue

        # Répartition du texte cible selon la longueur source de chaque cluster
        cluster_chars = [sum(len(w['word']) for w in c) for c in clusters]
        total_src = sum(cluster_chars) or 1
        fractions = [c / total_src for c in cluster_chars]
        sub_texts = _split_tgt_on_punct(seg.text_tgt.strip(), fractions)

        # Garde-fou : si une portion est vide, on retombe sur le segment original
        if any(not s for s in sub_texts):
            result.append(seg); continue

        # Création des sous-segments
        for ci, cluster in enumerate(clusters):
            st = cluster[0]['start']
            raw_en = cluster[-1]['end']
            if ci + 1 < len(clusters):
                next_st = clusters[ci+1][0]['start']
                en = min(raw_en + PAUSE_SPLIT_PADDING, next_st - 0.10)
            else:
                en = min(raw_en + PAUSE_SPLIT_PADDING, seg.end)
            en = max(en, st + MIN_DURATION_SEC)
            result.append(Segment(
                index=seg.index, start=st, end=en,
                text=' '.join(w['word'] for w in cluster),
                text_tgt=sub_texts[ci],
                words=cluster,
            ))
        split_count += 1

    if split_count:
        print(f"   ✂️  {split_count} segment(s) redécoupé(s) sur silences internes (≥ {PAUSE_SPLIT_THRESHOLD}s)")
    return result


def resegment(segments: list[Segment], target_lang: str = "fr") -> list[Subtitle]:
    print("\n📐 Re-segmentation...")

    # ── Phase 0a : réparer les word timings aberrants (alignement wav2vec2 raté) ──
    segments = _repair_aberrant_word_timings(segments)

    # ── Phase 0b : redécouper les segments traversés par un long silence ──
    # (sinon les sous-titres apparaissent en avance pendant les pauses du locuteur)
    segments = _split_on_pauses(segments)

    # ── Phase 1 : fusionner les segments trop courts ──
    merged = _merge_short(segments, target_lang)

    # ── Phase 2 : découpage classique ──
    subs = []; idx = 1
    for seg in merged:
        if not seg.text_tgt or seg.end - seg.start <= 0: continue
        txt, dur = seg.text_tgt.strip(), seg.end - seg.start
        cps = len(txt) / dur

        if cps <= MAX_CPS and len(txt) <= MAX_CHARS_PER_LINE * MAX_LINES_PER_SUB:
            subs.append(Subtitle(idx, seg.start, seg.end, _fmtlines(txt, target_lang)))
            idx += 1
        else:
            parts = _splittext(txt, dur, target_lang)
            total_chars = sum(len(p) for p in parts)
            total_gap = (len(parts) - 1) * GAP_BETWEEN_SUBS_MS / 1000
            usable = max(0, dur - total_gap)
            t = seg.start
            for i, p in enumerate(parts):
                frac = len(p) / total_chars if total_chars > 0 else 1.0 / len(parts)
                d = max(MIN_DURATION_SEC, usable * frac)
                te = min(t + d, seg.end)
                if i == len(parts) - 1:
                    te = seg.end
                if t >= seg.end:
                    # Plus de place : fusionner le texte restant dans le dernier sous-titre
                    if subs:
                        remaining = ' '.join(parts[i:])
                        prev = subs[-1]
                        merged_txt = prev.text.replace('\n', ' ') + ' ' + remaining
                        subs[-1] = Subtitle(prev.index, prev.start, seg.end,
                                            _fmtlines(merged_txt, target_lang))
                    break
                subs.append(Subtitle(idx, t, te, _fmtlines(p, target_lang)))
                idx += 1
                t = te + GAP_BETWEEN_SUBS_MS / 1000

    # ── Phase 3 : anti-orphelin ──
    subs = _deorphan(subs, target_lang)

    subs = _fixtiming(subs)
    subs = _audit_cps(subs)
    print(f"   ✅ {len(subs)} sous-titres")
    return subs


def _merge_short(segments: list[Segment], target_lang: str) -> list[Segment]:
    """Fusionne les segments consécutifs trop courts pour former des sous-titres lisibles.
    On ne fusionne que si le résultat reste dans les limites (durée et longueur)."""
    max_chars = MAX_CHARS_PER_LINE * MAX_LINES_PER_SUB
    merged = []
    for seg in segments:
        if not seg.text_tgt or seg.end - seg.start <= 0:
            continue
        txt = seg.text_tgt.strip()

        # Essayer de fusionner avec le précédent si le segment actuel est court
        if (merged
            and len(txt) < MIN_CHARS_PER_SUB
            and len(txt.split()) <= 3):
            prev = merged[-1]
            combined_txt = prev.text_tgt + " " + txt
            combined_dur = seg.end - prev.start
            combined_cps = len(combined_txt) / combined_dur if combined_dur > 0 else 999

            # Fusionner seulement si ça reste dans les limites
            if (len(combined_txt) <= max_chars
                and combined_dur <= MAX_DURATION_SEC
                and combined_cps <= MAX_CPS):
                merged[-1] = Segment(
                    index=prev.index, start=prev.start, end=seg.end,
                    text=prev.text + " " + seg.text,
                    text_tgt=combined_txt
                )
                continue

        merged.append(Segment(
            index=seg.index, start=seg.start, end=seg.end,
            text=seg.text, text_tgt=txt
        ))

    return merged


def _fmtlines(txt: str, target_lang: str = "fr") -> str:
    # Normaliser : supprimer les retours à la ligne internes (Claude en injecte parfois)
    txt = ' '.join(txt.split()).strip()
    if len(txt) <= MAX_CHARS_PER_LINE: return txt
    orphans = ORPHAN_WORDS.get(target_lang, set())

    def _apply_antiorphan(l1: str, l2: str) -> tuple[str, str]:
        """Si l1 finit par un orphelin, déplace-le vers l2 (si ça tient)."""
        w1 = l1.split()
        if w1 and w1[-1].lower().rstrip("'") in orphans and len(w1) > 1:
            new_l1 = ' '.join(w1[:-1])
            new_l2 = w1[-1] + ' ' + l2
            if len(new_l1) <= MAX_CHARS_PER_LINE and len(new_l2) <= MAX_CHARS_PER_LINE:
                return new_l1, new_l2
        return l1, l2

    sp = _findsplit(txt, target_lang)
    if sp > 0:
        l1, l2 = txt[:sp].strip(), txt[sp:].strip()
        if len(l1) <= MAX_CHARS_PER_LINE and len(l2) <= MAX_CHARS_PER_LINE:
            l1, l2 = _apply_antiorphan(l1, l2)
            return f"{l1}\n{l2}"

    # Fallback : chercher le meilleur espace tel que les DEUX moitiés ≤ MAX_CHARS_PER_LINE.
    # Pour un texte de len L, il faut L-MAX ≤ pos ≤ MAX (condition de faisabilité : L ≤ 2*MAX).
    L = len(txt)
    min_pos = max(1, L - MAX_CHARS_PER_LINE)
    max_pos = min(L - 1, MAX_CHARS_PER_LINE)
    if min_pos <= max_pos:
        mid = L // 2
        best_p = -1
        best_d = L
        for i, ch in enumerate(txt):
            if ch == ' ' and min_pos <= i <= max_pos:
                d = abs(i - mid)
                if d < best_d:
                    best_d = d
                    best_p = i
        if best_p > 0:
            l1, l2 = txt[:best_p].strip(), txt[best_p:].strip()
            if len(l1) <= MAX_CHARS_PER_LINE and len(l2) <= MAX_CHARS_PER_LINE:
                l1, l2 = _apply_antiorphan(l1, l2)
                return f"{l1}\n{l2}"

    # Dernier recours : le plus proche du milieu, même si ça dépasse un peu.
    # (Cas pathologiques — ne devrait arriver que si un mot seul dépasse MAX_CHARS_PER_LINE.)
    mid = L // 2
    p = txt.rfind(' ', 0, mid + 10)
    if p < 0:
        p = txt.find(' ', mid)
    if p > 0:
        l1, l2 = txt[:p].strip(), txt[p:].strip()
        l1, l2 = _apply_antiorphan(l1, l2)
        return f"{l1}\n{l2}"
    return txt


def _findsplit(txt: str, target_lang: str = "fr") -> int:
    pats = get_split_patterns(target_lang)
    tgt = len(txt) // 2; best, bd = -1, len(txt)
    for pat in pats:
        for m in re.finditer(pat, txt):
            pos = m.start() + len(m.group()) - 1
            if len(txt[:pos].strip()) <= MAX_CHARS_PER_LINE and len(txt[pos:].strip()) <= MAX_CHARS_PER_LINE:
                d = abs(pos - tgt)
                if d < bd: bd = d; best = pos
        if best >= 0: break
    return best


def _splittext(txt: str, dur: float, target_lang: str = "fr") -> list[str]:
    mx = MAX_CHARS_PER_LINE * MAX_LINES_PER_SUB
    n_chars = -(-len(txt) // min(mx, max(20, int(dur * MAX_CPS))))
    # Assurer assez de parties pour que chacune tienne dans MAX_DURATION_SEC
    n_time = math.ceil(dur / MAX_DURATION_SEC) if dur > MAX_DURATION_SEC else 1
    n = max(1, n_chars, n_time)
    if n == 1: return [txt]

    orphans = ORPHAN_WORDS.get(target_lang, set())

    # Séparateurs par priorité (du plus fort au plus faible)
    conj = CONJUNCTION_SEPARATORS.get(target_lang, [])
    separators = ['. ', '? ', '! ', '; ', ', ', ' — ', ' – '] + conj + [' ']

    parts, rem, tgt = [], txt, len(txt) // n
    for _ in range(n - 1):
        if len(rem) <= tgt + 10: break
        pos = -1
        for sep in separators:
            p = rem.rfind(sep, tgt - 15, tgt + 15)
            if p > 0:
                pos = p + len(sep)
                break
        if pos < 0:
            pos = rem.rfind(' ', 0, tgt + 5)
            if pos < 0: pos = tgt

        # Anti-orphelin : vérifier que la partie ne finit pas par un article/déterminant
        candidate = rem[:pos].strip()
        last_word = candidate.split()[-1].lower().rstrip("'") if candidate.split() else ""
        if last_word in orphans:
            # Reculer au mot d'avant
            words = candidate.split()
            if len(words) > 1:
                new_end = candidate.rindex(' ')  # position du dernier espace
                new_pos = new_end  # couper AVANT le dernier mot
                if new_pos > 5:  # garder au moins un minimum
                    pos = len(rem[:pos]) - (len(candidate) - new_pos)

        part = rem[:pos].strip()
        if part:
            parts.append(part)
        rem = rem[pos:].strip()
    if rem: parts.append(rem)
    return parts


def _deorphan(subs: list[Subtitle], target_lang: str) -> list[Subtitle]:
    """Post-traitement anti-orphelin : si un sous-titre ne contient qu'un ou deux mots courts,
    ou si un sous-titre se termine par un article/déterminant, on fusionne avec l'adjacent."""
    if len(subs) <= 1:
        return subs
    orphans = ORPHAN_WORDS.get(target_lang, set())
    max_chars = MAX_CHARS_PER_LINE * MAX_LINES_PER_SUB

    changed = True
    while changed:
        changed = False
        new_subs = []
        i = 0
        while i < len(subs):
            sub = subs[i]
            raw = sub.text.replace('\n', ' ').strip()
            words = raw.split()

            # Cas 1 : sous-titre trop court (1-2 mots, < MIN_CHARS)
            is_too_short = len(words) <= 2 and len(raw) < MIN_CHARS_PER_SUB

            # Cas 2 : sous-titre finit par un mot orphelin
            last_word = words[-1].lower().rstrip("'«\"(") if words else ""
            ends_with_orphan = last_word in orphans and len(words) > 1

            if is_too_short:
                # Essayer de fusionner avec le précédent d'abord
                if new_subs:
                    prev = new_subs[-1]
                    prev_raw = prev.text.replace('\n', ' ').strip()
                    combined = prev_raw + " " + raw
                    if len(combined) <= max_chars:
                        new_subs[-1] = Subtitle(prev.index, prev.start, sub.end,
                                                _fmtlines(combined, target_lang))
                        changed = True
                        i += 1
                        continue
                # Sinon fusionner avec le suivant
                if i + 1 < len(subs):
                    nxt = subs[i + 1]
                    nxt_raw = nxt.text.replace('\n', ' ').strip()
                    combined = raw + " " + nxt_raw
                    if len(combined) <= max_chars:
                        new_subs.append(Subtitle(sub.index, sub.start, nxt.end,
                                                 _fmtlines(combined, target_lang)))
                        changed = True
                        i += 2
                        continue

            elif ends_with_orphan:
                # Déplacer le mot orphelin au sous-titre suivant
                if i + 1 < len(subs):
                    nxt = subs[i + 1]
                    nxt_raw = nxt.text.replace('\n', ' ').strip()
                    orphan_word = words[-1]
                    new_txt = ' '.join(words[:-1])
                    new_nxt = orphan_word + " " + nxt_raw

                    if (len(new_txt) >= 5
                        and len(new_nxt) <= max_chars
                        and len(new_txt) <= max_chars):
                        new_subs.append(Subtitle(sub.index, sub.start, sub.end,
                                                 _fmtlines(new_txt, target_lang)))
                        # Modifier le suivant pour inclure le mot orphelin
                        subs[i + 1] = Subtitle(nxt.index, nxt.start, nxt.end,
                                               _fmtlines(new_nxt, target_lang))
                        changed = True
                        i += 1
                        continue

            new_subs.append(sub)
            i += 1

        subs = new_subs

    # Renuméroter
    for i, s in enumerate(subs):
        s.index = i + 1
    return subs


def _fixtiming(subs: list[Subtitle]) -> list[Subtitle]:
    if not subs:
        return subs
    gap = GAP_BETWEEN_SUBS_MS / 1000

    # Traiter sub[0] : durée minimale
    if subs[0].end - subs[0].start < MIN_DURATION_SEC:
        desired_end = subs[0].start + MIN_DURATION_SEC
        if len(subs) > 1:
            subs[0].end = min(desired_end, subs[1].start - gap)
        else:
            subs[0].end = desired_end

    for i in range(1, len(subs)):
        if subs[i] is None:
            continue
        # Trouver le précédent non-None
        prev = i - 1
        while prev >= 0 and subs[prev] is None:
            prev -= 1
        if prev < 0:
            continue

        # Corriger chevauchement : raccourcir la fin du précédent
        if subs[prev].end > subs[i].start - gap:
            subs[prev].end = subs[i].start - gap
            # Vérifier que le précédent garde une durée minimale
            if subs[prev].end - subs[prev].start < MIN_DURATION_SEC:
                prev2 = prev - 1
                while prev2 >= 0 and subs[prev2] is None:
                    prev2 -= 1
                earliest = subs[prev2].end + gap if prev2 >= 0 else 0.0
                subs[prev].start = max(earliest, subs[prev].end - MIN_DURATION_SEC)
            # Vérifier que le raccourcissement ne crée pas un CPS absurde
            prev_dur = subs[prev].end - subs[prev].start
            prev_txt = subs[prev].text.replace('\n', ' ')
            if prev_dur > 0 and len(prev_txt) / prev_dur > MAX_CPS * 2:
                # CPS > 2x la norme → fusionner avec le suivant si possible
                combined = prev_txt + " " + subs[i].text.replace('\n', ' ')
                if len(combined) <= MAX_CHARS_PER_LINE * MAX_LINES_PER_SUB:
                    subs[i] = Subtitle(subs[prev].index, subs[prev].start, subs[i].end,
                                       _fmtlines(combined))
                    subs[prev] = None  # marquer pour suppression

        # Durée minimale du sous-titre courant
        if subs[i] is not None and subs[i].end - subs[i].start < MIN_DURATION_SEC:
            desired_end = subs[i].start + MIN_DURATION_SEC
            # Borner à l'espace disponible avant le suivant
            nxt = i + 1
            while nxt < len(subs) and subs[nxt] is None:
                nxt += 1
            if nxt < len(subs):
                subs[i].end = min(desired_end, subs[nxt].start - gap)
            else:
                subs[i].end = desired_end

    # Supprimer les entrées fusionnées ou à durée négative, et re-numéroter
    subs = [s for s in subs if s is not None and s.end > s.start]
    for i, s in enumerate(subs): s.index = i + 1
    return subs


def _audit_cps(subs: list[Subtitle]) -> list[Subtitle]:
    """Ajuste les timings des sous-titres dont le CPS dépasse MAX_CPS."""
    if not subs:
        return subs
    gap = GAP_BETWEEN_SUBS_MS / 1000

    for i, sub in enumerate(subs):
        txt = sub.text.replace('\n', ' ')
        dur = sub.end - sub.start
        if dur <= 0:
            continue
        cps = len(txt) / dur
        if cps <= MAX_CPS:
            continue

        needed_dur = len(txt) / MAX_CPS

        # Tenter d'étendre end (espace disponible avant le suivant)
        if i + 1 < len(subs):
            max_end = subs[i+1].start - gap
        else:
            max_end = sub.end + 10.0  # pas de borne supérieure stricte pour le dernier
        new_end = min(sub.start + needed_dur, max_end)
        sub.end = max(sub.end, new_end)

        # Si toujours insuffisant, tenter de reculer start
        dur = sub.end - sub.start
        if dur > 0 and len(txt) / dur > MAX_CPS:
            if i > 0:
                min_start = subs[i-1].end + gap
            else:
                min_start = 0.0
            new_start = max(sub.end - needed_dur, min_start)
            sub.start = min(sub.start, new_start)

        # Logger si toujours au-dessus
        dur = sub.end - sub.start
        if dur > 0 and len(txt) / dur > MAX_CPS:
            print(f"   ⚠️  CPS irréductible : sous-titre {sub.index} ({len(txt)/dur:.0f} CPS)")

    return subs


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 6 : SRT + INCRUSTATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_srt(subs: list[Subtitle], path: str):
    print(f"\n💾 SRT : {path}")
    with open(path, "w", encoding="utf-8") as f:
        for s in subs: f.write(s.to_srt() + "\n")
    print(f"   ✅ {len(subs)} sous-titres")


def _parse_delogo(delogo: str) -> str:
    """Parse X:Y:W:H string into ffmpeg delogo filter, or None."""
    if not delogo:
        return None
    parts = delogo.split(":")
    if len(parts) == 4:
        return f"delogo=x={parts[0]}:y={parts[1]}:w={parts[2]}:h={parts[3]}"
    print(f"   ⚠️  --delogo ignoré (format attendu: X:Y:W:H, reçu: {delogo})")
    return None


def _get_video_dimensions(video: str) -> tuple[int, int]:
    """Récupère (largeur, hauteur) de la vidéo via ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", video],
            capture_output=True, text=True)
        w, h = r.stdout.strip().split("x")
        return int(w), int(h)
    except Exception:
        return 1920, 1080  # fallback paysage standard


def _get_video_height(video: str) -> int:
    """Compatibilité : retourne la hauteur uniquement."""
    return _get_video_dimensions(video)[1]


def _scale_style_for_video(style_str: str, video_width: int, video_height: int) -> str:
    """Ajuste FontSize et MarginV pour les formats NON-PAYSAGE uniquement.

    libass scale déjà FontSize/MarginV proportionnellement à la résolution vidéo
    via son PlayRes interne (≈384×288). Un FontSize=24 rend la même taille
    relative sur 720p, 1080p ou 4K — PAS besoin de scaling résolution.

    Par contre, pour les formats plus étroits que 16:9 (carré, vertical), les
    lignes de 42 caractères débordent. On réduit alors FontSize proportionnellement
    à la largeur manquante par rapport au 16:9 attendu.
    """
    import re
    if video_width <= 0 or video_height <= 0:
        return style_str

    aspect = video_width / video_height
    if aspect >= 16 / 9 - 0.05:  # paysage standard ou plus large : libass gère
        return style_str

    # Format plus étroit que 16:9 : réduire pour que les lignes tiennent en largeur
    expected_width = video_height * 16 / 9
    width_ratio = video_width / expected_width

    def scale_field(match):
        name = match.group(1)
        val = int(match.group(2))
        new = round(val * width_ratio)
        if name == "FontSize":
            new = max(14, new)  # plancher lisibilité
        else:  # MarginV
            new = max(8, new)
        return f"{name}={new}"
    return re.sub(r"(FontSize|MarginV)=(\d+)", scale_field, style_str)


def _scale_style_for_height(style_str: str, video_height: int) -> str:
    """Compatibilité : ancienne API. Suppose 16:9."""
    return _scale_style_for_video(style_str, int(video_height * 16 / 9), video_height)


def burn_subtitles(video: str, srt: str, output: str, style: str = "default",
                   delogo: str = None):
    print(f"\n🎬 Incrustation (style: {style})...")
    import shutil, tempfile
    fs = SUBTITLE_STYLES.get(style, SUBTITLE_STYLES["default"])
    video_width, video_height = _get_video_dimensions(video)
    # Ajuster seulement pour formats non-paysage (carré/vertical) ;
    # pour le paysage, libass scale déjà proportionnellement à la résolution.
    fs_before = fs
    fs = _scale_style_for_video(fs, video_width, video_height)
    if fs != fs_before:
        print(f"   📐 Format non-standard {video_width}x{video_height} → style adapté")
    else:
        print(f"   📐 Dimensions vidéo: {video_width}x{video_height}")
    # ffmpeg filter parser is picky about paths — use a temp copy with a simple name
    tmp_dir = tempfile.mkdtemp()
    tmp_srt = os.path.join(tmp_dir, "subs.srt")
    shutil.copy2(srt, tmp_srt)
    # Build video filter chain: optional delogo + subtitles
    delogo_f = _parse_delogo(delogo)
    sub_f = f"subtitles={tmp_srt}:force_style='{fs}'"
    vf = ",".join(filter(None, [delogo_f, sub_f]))
    try:
        cmd = ["ffmpeg", "-y", "-i", video,
               "-vf", vf,
               "-c:v", "libx264", "-crf", "18", "-preset", "slow",
               "-c:a", "aac", "-b:a", "192k", "-ac", "2",
               "-movflags", "+faststart", output]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print("   ⚠️  Retry sans style...")
            vf2 = ",".join(filter(None, [delogo_f, f"subtitles={tmp_srt}"]))
            cmd2 = ["ffmpeg", "-y", "-i", video, "-vf", vf2,
                    "-c:v", "libx264", "-crf", "18",
                    "-c:a", "aac", "-b:a", "192k", "-ac", "2",
                    "-movflags", "+faststart", output]
            r = subprocess.run(cmd2, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"   ❌ Incrustation échouée.")
                if "Error parsing" in r.stderr or "filter" in r.stderr.lower():
                    print(f"   💡 Cause probable : ffmpeg n'a pas le filtre 'subtitles' (libass manquant).")
                    print(f"      Vérifiez : ffmpeg -filters 2>/dev/null | grep subtitles")
                    print(f"      Le fichier SRT a été généré, vous pouvez l'incruster plus tard :")
                    print(f"      python traduire.py {video} --srt-only {srt}")
                else:
                    print(f"   {r.stderr[-400:]}")
                return
        mb = os.path.getsize(output) / (1024*1024)
        print(f"   ✅ {output} ({mb:.1f} Mo, {time.time()-t0:.0f}s)")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════════

def save_seg(segs, path):
    def _seg_dict(s):
        d = {"index":s.index,"start":s.start,"end":s.end,
             "text":s.text,"text_tgt":s.text_tgt}
        if s.lang:
            d["lang"] = s.lang
        # Préserver l'alignement mot-à-mot (utile pour détecter les passages
        # musicaux et ajuster les fins de sous-titres).
        if s.words:
            d["words"] = s.words
        return d
    with open(path, "w", encoding="utf-8") as f:
        json.dump([_seg_dict(s) for s in segs], f, ensure_ascii=False, indent=2)
    print(f"   💾 → {path}")

def load_seg(path):
    with open(path) as f:
        segments = [Segment(d["index"],d["start"],d["end"],d["text"],
                        d.get("text_tgt", d.get("text_fr", "")),
                        words=d.get("words", []),
                        lang=d.get("lang", ""))
                for d in json.load(f)]
    # Reconstruire le texte source depuis les mots si vide (bug WhisperX align)
    rebuilt = 0
    for seg in segments:
        if not seg.text.strip() and seg.words:
            seg.text = " ".join(w["word"] for w in seg.words if "word" in w).strip()
            if seg.text:
                rebuilt += 1
    # Purger les « traductions » qui sont en fait des marqueurs régurgités
    purged = 0
    for seg in segments:
        if seg.text_tgt and _MARKER_RE.match(seg.text_tgt):
            seg.text_tgt = ""
            purged += 1
    if rebuilt or purged:
        print(f"   🔧 Chargement : {rebuilt} texte(s) reconstruit(s), {purged} marqueur(s) purgé(s)")
    return segments

def save_bilingual(segs, path, source_lang: str = "en", target_lang: str = "fr"):
    src_label = source_lang_label(source_lang)
    tgt_label = target_lang.upper()
    with open(path, "w", encoding="utf-8") as f:
        f.write("="*70 + f"\nTRADUCTION BILINGUE ({src_label} → {tgt_label})\n" + "="*70 + "\n\n")
        for s in segs:
            f.write(f"[{s.index}] {_fmt(s.start)} → {_fmt(s.end)}\n  {src_label}: {s.text}\n  {tgt_label}: {s.text_tgt}\n\n")

def save_src_srt(segs, path):
    with open(path, "w", encoding="utf-8") as f:
        for s in segs:
            f.write(f"{s.index}\n{_fmt(s.start)} --> {_fmt(s.end)}\n{s.text}\n\n")


# ═══════════════════════════════════════════════════════════════════════════════
# VARIANTE DOUBLAGE : double sous-titre (actuel + prochain)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_dubbing_ass(subs: list[Subtitle], path: str, video_width: int = 1920, video_height: int = 1080):
    """
    Génère un fichier ASS pour le doublage avec 2 sous-titres simultanés
    EMPILÉS verticalement pour éviter tout chevauchement :
      - ACTUEL : en bas-centre, jaune, taille normale, collé au bord
      - PROCHAIN : juste au-dessus, cyan, taille réduite

    Le doubleur lit le sous-titre actuel (jaune) tout en voyant d'un coup
    d'œil ce qui vient ensuite (cyan). Les deux sont centrés et empilés
    sur deux « étages » distincts, donc pas de superposition possible
    même avec du texte long.
    """
    print(f"\n🎙️  Génération ASS doublage : {path}")

    # ── Mise à l'échelle selon la résolution de la vidéo ─────────────
    # Les tailles de base sont calibrées pour du 1080p ; on les met à
    # l'échelle proportionnellement à la hauteur réelle de la vidéo pour
    # rester lisibles en 4K comme en 720p.
    scale = video_height / 1080
    # Polices AGRANDIES : le doublage n'est pas un produit consommateur final,
    # donc on privilégie le confort de lecture du doubleur sur la visibilité
    # des visages.
    fs_current = max(32, round(72 * scale))   # sous-titre actuel (jaune)
    fs_next    = max(26, round(52 * scale))   # prochain sous-titre (cyan)
    fs_counter = max(16, round(28 * scale))   # compteur coin haut-droite
    ol_current = max(2, round(3 * scale))
    ol_next    = max(1, round(2 * scale))
    ol_counter = max(1, round(1 * scale))

    # ── Calcul des marges verticales (empilage) ──────────────────────
    # Current est collé au bord bas (MarginV faible). Next est posé
    # au-dessus de Current avec assez de clearance pour 2 lignes
    # maximum du courant (le resegment garantit ≤2 lignes).
    # Hauteur d'une ligne ≈ fontsize * 1.25 (line-height ASS).
    margin_v_current = round(12 * scale)   # tout en bas
    current_block_h  = round(fs_current * 1.25 * 2) + round(10 * scale)
    margin_v_next    = margin_v_current + current_block_h
    margin_h         = round(30 * scale)   # marge gauche/droite identique
    margin_cnt       = round(18 * scale)

    with open(path, "w", encoding="utf-8") as f:
        # ── En-tête ASS ──────────────────────────────────────────────
        f.write(f"""[Script Info]
Title: Doublage - sous-titres courant + suivant
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Current,Arial,{fs_current},&H0000DBFF,&H000000FF,&H00000000,&HB0000000,1,0,0,0,100,100,0,0,3,{ol_current},0,2,{margin_h},{margin_h},{margin_v_current},1
Style: Next,Arial,{fs_next},&H00FFFF00,&H000000FF,&H00000000,&HB0000000,0,0,0,0,100,100,0,0,3,{ol_next},0,2,{margin_h},{margin_h},{margin_v_next},1
Style: Counter,Arial,{fs_counter},&H0080FFFF,&H000000FF,&H00000000,&HA0000000,0,0,0,0,100,100,0,0,1,{ol_counter},0,9,{margin_cnt},{margin_cnt},{margin_cnt},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""")
        print(f"   📏 Résolution {video_width}×{video_height} → tailles {fs_current}/{fs_next}/{fs_counter} (scale ×{scale:.2f})")
        print(f"   📐 Empilage : courant à MarginV={margin_v_current}, suivant à MarginV={margin_v_next}")
        # Notes sur les styles ASS :
        # - PrimaryColour format : &HAABBGGRR (AA=alpha inversé, 00=opaque)
        # - Current: jaune (&H0000DBFF = FFdb00 en RGB), gras, bas-centre (\an2)
        # - Next:    cyan   (&H00FFFF00 = 00FFFF en RGB), bas-centre (\an2) MAIS avec
        #            MarginV plus grand → affiché au-dessus de Current (empilage).
        # - Counter: petit compteur en haut à droite (\an9)
        # - Alignment: 1=bas-gauche, 2=bas-centre, 3=bas-droite, 9=haut-droite
        
        # ── Événements ───────────────────────────────────────────────
        for i, sub in enumerate(subs):
            start_tc = _fmt_ass(sub.start)
            end_tc = _fmt_ass(sub.end)
            
            # Nettoyer le texte (remplacer \n du SRT par \N pour ASS)
            current_text = sub.text.replace("\n", "\\N")
            
            # Sous-titre ACTUEL (jaune, bas-gauche)
            f.write(f"Dialogue: 0,{start_tc},{end_tc},Current,,0,0,0,,{current_text}\n")
            
            # Sous-titre PROCHAIN (cyan, bas-droite) — affiché pendant le sous-titre actuel
            if i + 1 < len(subs):
                next_text = subs[i + 1].text.replace("\n", "\\N")
                # Préfixer avec ► pour signaler visuellement que c'est "le prochain"
                f.write(f"Dialogue: 1,{start_tc},{end_tc},Next,,0,0,0,,► {next_text}\n")
            else:
                # Dernier sous-titre : afficher "[FIN]"
                f.write(f"Dialogue: 1,{start_tc},{end_tc},Next,,0,0,0,,► [FIN]\n")
            
            # Compteur de progression (haut-droite, discret)
            f.write(f"Dialogue: 2,{start_tc},{end_tc},Counter,,0,0,0,,{i+1}/{len(subs)}\n")
    
    print(f"   ✅ {len(subs)} paires (actuel + prochain) → {path}")
    return path


def _fmt_ass(sec: float) -> str:
    """Format ASS : H:MM:SS.cc (centièmes, pas millisecondes)."""
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    cs = int((sec % 1) * 100)
    return f"{int(h)}:{int(m):02d}:{int(s):02d}.{cs:02d}"


def get_video_resolution(video_path: str) -> tuple[int, int]:
    """Récupère la résolution de la vidéo via ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json", video_path
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(r.stdout)
        w = info["streams"][0]["width"]
        h = info["streams"][0]["height"]
        return w, h
    except Exception:
        return 1920, 1080  # fallback


def burn_dubbing_video(video: str, ass: str, output: str):
    """Incruste le fichier ASS de doublage dans la vidéo."""
    print(f"\n🎙️  Incrustation doublage...")
    import shutil, tempfile
    tmp_dir = tempfile.mkdtemp()
    tmp_ass = os.path.join(tmp_dir, "subs.ass")
    shutil.copy2(ass, tmp_ass)
    try:
        cmd = ["ffmpeg", "-y", "-i", video,
               "-vf", f"ass={tmp_ass}",
               "-c:v", "libx264", "-crf", "18", "-preset", "slow",
               "-c:a", "aac", "-b:a", "192k", "-ac", "2",
               "-movflags", "+faststart", output]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"   ❌ Erreur ffmpeg : {r.stderr[-400:]}")
            print(f"   💡 Le fichier .ass a été généré, utilisable dans VLC (Ctrl+Shift+S)")
            return ""
        mb = os.path.getsize(output) / (1024*1024)
        print(f"   ✅ {output} ({mb:.1f} Mo, {time.time()-t0:.0f}s)")
        return output
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def generate_social_txt(segments, analysis, client, output_path: str,
                        source_lang: str, target_lang: str):
    """Génère un fichier de partage social (citations verbatim + mise en contexte)."""
    print(f"\n📱 Génération du résumé social...")

    # Construire la transcription — utiliser la traduction FR si disponible
    use_tgt = target_lang.startswith("fr")
    lines = []
    for s in segments:
        txt = ""
        if use_tgt and s.text_tgt:
            txt = s.text_tgt
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


def auto_clip_if_long(source_video: str, seg_json: str, tgt_lang: str):
    """Si la vidéo dépasse AUTO_CLIP_THRESHOLD_SEC, invoque clipper.py
    en réutilisant les segments déjà transcrits/traduits (skip WhisperX)."""
    duration = _probe_duration(source_video)
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
    cmd = [sys.executable, clipper_path, source_video,
           "--pre-segments", seg_json,
           "-n", str(AUTO_CLIP_COUNT),
           "--target-lang", tgt_lang]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"   ⚠️  Clipper a échoué (code {e.returncode}) — vidéo principale préservée")
    except FileNotFoundError:
        print(f"   ⚠️  clipper.py introuvable à {clipper_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global WHISPER_MODEL, CLAUDE_MODEL
    p = argparse.ArgumentParser(
        description="Pipeline traduction & sous-titrage vidéo multilingue",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Exemples :
              python traduire.py interview.mp4                       # EN → FR (défaut)
              python traduire.py talk.mp4 -s en -t es                # EN → ES
              python traduire.py video.mp4 --source ja --target en   # JA → EN
              python traduire.py talk.mp4 -o talk_fr.mp4 --style netflix
              python traduire.py podcast.mp4 --skip-burn
              python traduire.py video.mp4 --resume segments.json
              python traduire.py video.mp4 --srt-only existing.srt
        """))

    p.add_argument("source", help="Fichier MP4 source ou URL YouTube")
    p.add_argument("-s", "--source-lang", default="en",
                   help="Langue(s) source (code ISO 639-1, défaut: en). "
                        "Multilingue : séparées par virgule, ex: en,he")
    p.add_argument("-t", "--target-lang", default="fr",
                   help="Langue cible (code ISO 639-1, défaut: fr)")
    p.add_argument("-o", "--output", help="MP4 sortie (défaut: source_{target}.mp4)")
    p.add_argument("--style", choices=list(SUBTITLE_STYLES.keys()), default="default")
    p.add_argument("--skip-burn", action="store_true", help="SRT uniquement")
    p.add_argument("--skip-review", action="store_true", help="Passer la relecture")
    p.add_argument("--dubbing", action="store_true",
                   help="Générer un 2e MP4 pour le doublage (sous-titre actuel + prochain)")
    p.add_argument("--context", type=str, default="",
                   help="Contexte pour guider la traduction : noms, sujet, registre, etc. "
                        "Ex: --context \"Interview de Mary-Anne DeMasi, journaliste d'investigation\"")
    p.add_argument("--resume", metavar="JSON", help="Reprendre depuis sauvegarde")
    p.add_argument("--srt-only", metavar="SRT", help="Incruster un SRT existant")
    p.add_argument("--whisper-model", default=WHISPER_MODEL)
    p.add_argument("--claude-model", default=CLAUDE_MODEL)
    p.add_argument("--llm", choices=["claude", "local"], default="local",
                   help="Backend LLM : local (Ollama, défaut) ou claude (API Anthropic)")
    p.add_argument("--analysis-llm", choices=["auto", "claude", "local"], default="auto",
                   help="LLM de la passe d'analyse/contexte (glossaire, noms propres, "
                        "domaine) : auto = Claude si ANTHROPIC_API_KEY dispo, sinon local")
    p.add_argument("--ollama-model", default=OLLAMA_MODEL,
                   help=f"Modèle Ollama (défaut: {OLLAMA_MODEL})")
    p.add_argument("--ollama-url", default=OLLAMA_URL,
                   help=f"URL du serveur Ollama (défaut: {OLLAMA_URL})")
    p.add_argument("--delogo", metavar="X:Y:W:H",
                   help="Supprimer un watermark (ex: NotebookLM) via ffmpeg delogo. "
                        "Format: X:Y:W:H en pixels (ex: 1060:685:210:30)")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--vad-onset", type=float, default=None,
                   help="Seuil VAD onset (défaut WhisperX: 0.500). Augmenter (ex: 0.6-0.7) "
                        "pour des frontières de segment plus strictes sur audio bruité.")
    p.add_argument("--vad-offset", type=float, default=None,
                   help="Seuil VAD offset (défaut WhisperX: 0.363).")
    p.add_argument("--no-demucs", dest="demucs", action="store_false", default=True,
                   help="Désactiver l'isolation Demucs (active par défaut). "
                        "Sans Demucs, Silero VAD sur l'audio original sert de garde-fou, "
                        "mais WhisperX peut mal détecter la langue des segments.")
    p.add_argument("--ocr", action="store_true",
                   help="Extraire les sous-titres incrustés par OCR pour compléter WhisperX")
    p.add_argument("--cookies", default=None,
                   help="Chemin vers le fichier cookies JSON (Epoch Times / Apollo Health)")
    args = p.parse_args()

    src_lang = args.source_lang.lower().strip()
    tgt_lang = args.target_lang.lower().strip()

    # Multilingue : "en,he" → ["en", "he"]
    source_langs = [x.strip() for x in src_lang.split(",")]
    if tgt_lang in source_langs:
        print(f"❌ Langue cible ({tgt_lang}) présente dans les langues sources ({src_lang})"); sys.exit(1)

    # Label fichiers : "en" ou "en+he"
    src_file_label = "+".join(source_langs)

    # ── Téléchargement YouTube / Epoch Times / Apollo Health si lien ─────
    epoch_page = None
    apollo_page = None
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    if is_youtube_url(args.source):
        args.source = download_youtube(args.source, output_dir=str(INPUT_DIR))
    elif epochtimes.is_epochtimes_url(args.source):
        print(f"\n📰 Source : Epoch Times")
        cookies = epochtimes.load_cookies(args.cookies)
        epoch_page = epochtimes.fetch_epoch_page(args.source, cookies)
        print(f"   Titre     : {epoch_page.title}")
        if epoch_page.transcript:
            print(f"   Transcription : {len(epoch_page.transcript)} paragraphes")
        args.source = epochtimes.download_epoch_video(epoch_page, output_dir=str(INPUT_DIR))
    elif apollohealth.is_apollo_url(args.source):
        print(f"\n🏥 Source : Apollo Health")
        cookies = apollohealth.load_cookies(args.cookies)
        apollo_page = apollohealth.fetch_apollo_page(args.source, cookies)
        print(f"   Titre     : {apollo_page.title}")
        if apollo_page.transcript:
            print(f"   Transcription : {len(apollo_page.transcript)} spans")
        args.source = apollohealth.download_apollo_video(apollo_page, output_dir=str(INPUT_DIR))
        apollohealth.save_apollo_meta(apollo_page, args.source)

    # Charger les sidecars si présents (créés par le daemon)
    if not epoch_page:
        epoch_page = epochtimes.load_epoch_meta(args.source)
    if not apollo_page:
        apollo_page = apollohealth.load_apollo_meta(args.source)

    src = resolve_source(args.source)
    if not src.exists():
        print(f"❌ Introuvable : {args.source}"); sys.exit(1)
    args.source = str(src)
    base = src.stem
    work_base = WORK_DIR / base
    work_base.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output   = args.output or str(OUTPUT_DIR / f"{base}_{tgt_lang}.mp4")
    srt_tgt  = str(OUTPUT_DIR / f"{base}_{tgt_lang}.srt")
    srt_src  = str(OUTPUT_DIR / f"{base}_{src_file_label}.srt")
    seg_json = str(work_base / f"{base}_segments.json")
    bil_txt  = str(OUTPUT_DIR / f"{base}_{src_file_label}_{tgt_lang}_bilingue.txt")
    ana_json = str(work_base / f"{base}_analyse.json")
    audio    = str(work_base / f"{base}_audio.wav")
    ass_dub  = str(work_base / f"{base}_doublage.ass")
    out_dub  = str(work_base / f"{base}_doublage.mp4")

    if args.srt_only:
        print("🎬 Incrustation d'un SRT existant")
        check_ffmpeg()
        burn_subtitles(args.source, args.srt_only, output, args.style,
                       delogo=args.delogo); return

    WHISPER_MODEL = args.whisper_model; CLAUDE_MODEL = args.claude_model

    if args.llm == "local":
        client = _OllamaClient(args.ollama_url, args.ollama_model)
        llm_label = f"Ollama {args.ollama_model}"
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key: print("❌ ANTHROPIC_API_KEY manquante"); sys.exit(1)
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        llm_label = f"Claude {CLAUDE_MODEL}"

    # Analyse : Claude apporte une bien meilleure connaissance du monde (noms
    # propres, domaine, glossaire) → meilleur contexte ET meilleure traduction.
    # C'est UN seul appel, peu coûteux. "auto" = Claude si une clé est dispo.
    analysis_client = client
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

    src_name = source_lang_description(src_lang)
    tgt_name = lang_name(tgt_lang)
    src_label = source_lang_label(src_lang)

    print("="*60)
    print(f"🎬 Pipeline traduction vidéo {src_name} → {tgt_name}")
    print("="*60)
    print(f"   Source  : {args.source}")
    print(f"   Langues : {src_label} → {tgt_lang.upper()} ({src_name} → {tgt_name})")
    print(f"   Sortie  : {output}")
    print(f"   Style   : {args.style}")
    print(f"   Whisper : {WHISPER_MODEL} | LLM : {llm_label}")
    if args.context:
        print(f"   Contexte: {args.context[:80]}{'...' if len(args.context) > 80 else ''}")

    has_libass = check_ffmpeg()
    if not has_libass and not args.skip_burn:
        print("   ⚠️  --skip-burn activé automatiquement (libass manquant)")
        args.skip_burn = True

    print("="*60)

    t0 = time.time()

    if args.resume:
        resume_path = args.resume
        if not os.path.exists(resume_path):
            candidate = str(work_base / Path(resume_path).name)
            if os.path.exists(candidate):
                resume_path = candidate
        print(f"\n🔄 Reprise depuis {resume_path}")
        segments = load_seg(resume_path)
        done = sum(1 for s in segments if s.text_tgt)
        print(f"   {done}/{len(segments)} déjà traduits")

        if os.path.exists(ana_json):
            with open(ana_json) as f: analysis = ContentAnalysis(**json.load(f))
        else:
            analysis = analyze_content(segments, analysis_client, src_lang, tgt_lang, args.context)
            with open(ana_json, "w", encoding="utf-8") as f:
                json.dump(asdict(analysis), f, ensure_ascii=False, indent=2)

        if done < len(segments):
            segments = translate_chunks(segments, analysis, client, src_lang, tgt_lang, args.context)
            save_seg(segments, seg_json)
        if not args.skip_review:
            segments = review_translation(segments, analysis, client, src_lang, tgt_lang, args.context)
            save_seg(segments, seg_json)

        # Passe 4b — cohérence globale (toujours, même avec --skip-review)
        segments = check_consistency(segments, analysis, client, src_lang, tgt_lang)
        save_seg(segments, seg_json)

        # Passe 4c — vérification glossaire (toujours)
        segments = verify_glossary(segments, analysis, client, src_lang, tgt_lang)
        save_seg(segments, seg_json)

        # Passe 4d — nettoyage des artefacts (alternatives / gloses)
        segments = _clean_translation_artifacts(segments)

        # Passe 4e — filet de sécurité : retirer les sous-titres sur passages en langue cible
        segments = _clear_same_lang_translations(segments, src_lang, tgt_lang)
        save_seg(segments, seg_json)
    else:
        # Passe 1
        extract_audio(args.source, audio)
        transcribe_audio = audio
        vocals_path: Optional[str] = None
        if args.demucs:
            transcribe_audio = separate_vocals(audio, str(work_base))
            if transcribe_audio != audio:
                vocals_path = transcribe_audio
        segments = transcribe_whisperx(transcribe_audio, src_lang, args.hf_token,
                                       vad_onset=args.vad_onset, vad_offset=args.vad_offset)
        if vocals_path:
            # Demucs activé : gardes acoustiques sur vocals.wav + comblement des trous
            segments = _snap_orphan_words(segments, vocals_path)
            speech_intervals = _detect_speech_intervals(vocals_path)
            segments = _snap_hallucinated_words(segments, speech_intervals)
            segments = _fill_transcription_gaps(segments, audio, src_lang, args.hf_token)
        else:
            # Sans Demucs : Silero VAD sur l'audio original comme garde-fou
            segments = _snap_orphan_words(segments, audio)
            speech_intervals = _detect_speech_intervals_silero(audio)
            segments = _snap_hallucinated_words(segments, speech_intervals)
        if args.ocr:
            segments = ocr_supplement_segments(segments, args.source)

        # Relecture Epoch Times (correction noms propres)
        if epoch_page and epoch_page.transcript:
            segments, _ = epochtimes.align_transcript_to_segments(
                epoch_page.transcript, segments, epoch_page.speakers)

        # Relecture Apollo Health (correction noms propres, termes médicaux)
        if apollo_page and apollo_page.transcript:
            segments = apollohealth.align_transcript_to_segments(
                apollo_page.transcript, segments)

        save_seg(segments, seg_json)
        save_src_srt(segments, srt_src)

        # Enrichir le contexte Claude avec les métadonnées source
        if epoch_page:
            epoch_ctx = epochtimes.build_epoch_context(epoch_page)
            if epoch_ctx:
                args.context = (epoch_ctx + "\n\n" + args.context).strip() if args.context else epoch_ctx
        if apollo_page:
            apollo_ctx = apollohealth.build_apollo_context(apollo_page)
            if apollo_ctx:
                args.context = (apollo_ctx + "\n\n" + args.context).strip() if args.context else apollo_ctx

        # Passe 2
        analysis = analyze_content(segments, analysis_client, src_lang, tgt_lang, args.context)
        with open(ana_json, "w", encoding="utf-8") as f:
            json.dump(asdict(analysis), f, ensure_ascii=False, indent=2)

        # Passe 3
        segments = translate_chunks(segments, analysis, client, src_lang, tgt_lang, args.context)
        save_seg(segments, seg_json)

        # Passe 4
        if not args.skip_review:
            segments = review_translation(segments, analysis, client, src_lang, tgt_lang, args.context)
            save_seg(segments, seg_json)

        # Passe 4b — cohérence globale (toujours, même avec --skip-review)
        segments = check_consistency(segments, analysis, client, src_lang, tgt_lang)
        save_seg(segments, seg_json)

        # Passe 4c — vérification glossaire (toujours)
        segments = verify_glossary(segments, analysis, client, src_lang, tgt_lang)
        save_seg(segments, seg_json)

        # Passe 4d — nettoyage des artefacts (alternatives / gloses)
        segments = _clean_translation_artifacts(segments)

        # Passe 4e — filet de sécurité : retirer les sous-titres sur passages en langue cible
        segments = _clear_same_lang_translations(segments, src_lang, tgt_lang)
        save_seg(segments, seg_json)

    save_bilingual(segments, bil_txt, src_lang, tgt_lang)

    # Passe 5
    subtitles = resegment(segments, tgt_lang)

    # Passe 6
    generate_srt(subtitles, srt_tgt)
    if not args.skip_burn:
        burn_subtitles(args.source, srt_tgt, output, args.style,
                           delogo=args.delogo)
    
    # Passe 7 (optionnel) : version doublage
    if args.dubbing:
        vw, vh = get_video_resolution(args.source)
        generate_dubbing_ass(subtitles, ass_dub, vw, vh)
        if not args.skip_burn:
            burn_dubbing_video(args.source, ass_dub, out_dub)

    if os.path.exists(audio): os.remove(audio)

    # ── Résumé social ──────────────────────────────────────────────────────
    social_txt = str(OUTPUT_DIR / f"{base}_social.txt")
    generate_social_txt(segments, analysis, client, social_txt, src_lang, tgt_lang)

    # ── Extraction auto de clips (vidéo > 45 min) ──────────────────────────
    # On part de la source originale, jamais du fichier sous-titré incrusté :
    # les clips ne doivent porter que les sous-titres karaoke de clipper.py.
    auto_clip_if_long(args.source, seg_json, tgt_lang)

    el = time.time() - t0
    print(f"\n{'='*60}")
    print(f"🎉 Terminé en {el/60:.1f} min !")
    print(f"{'='*60}")
    print(f"   📄 SRT {tgt_lang.upper()}            : {srt_tgt}")
    print(f"   📄 SRT {src_label:<15s} : {srt_src}")
    if not args.skip_burn:
        print(f"   🎬 Vidéo sous-titrée  : {output}")
    if args.dubbing:
        print(f"   🎙️  ASS doublage       : {ass_dub}")
        if not args.skip_burn:
            print(f"   🎙️  Vidéo doublage     : {out_dub}")
    print(f"   📝 Bilingue           : {bil_txt}")
    if os.path.exists(social_txt):
        print(f"   📱 Résumé social      : {social_txt}")
    print(f"   💾 Segments (reprise) : {seg_json}")
    print(f"   🔍 Analyse            : {ana_json}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
