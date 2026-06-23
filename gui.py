#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gui.py — Panneau de contrôle web pour la boîte à outils de traduction/doublage.

Lance un petit serveur local (Flask) qui présente, pour chaque script du repo,
un formulaire élégant : champs, listes déroulantes, interrupteurs, sélecteur de
fichiers, aperçu de la commande en direct et console de sortie en streaming.

    ~/miniconda3/envs/interview/bin/python gui.py
    → ouvre http://127.0.0.1:5005

Aucune dépendance nouvelle : Flask est déjà présent dans l'env « interview ».
Tous les scripts sont lancés avec le même interpréteur (cf. memory python_env).
Identité visuelle alignée sur l'extension Chrome de traduction (sombre + orange).
"""

import os
import sys
import json
import shlex
import shutil
import signal
import subprocess
import threading
import time
import uuid


def _resolve_python():
    """Interpréteur Python du toolkit, portable d'une machine à l'autre :
    surcharge TRADUCTION_PYTHON, sinon env conda « interview » ou « traduction »,
    sinon l'interpréteur courant."""
    cands = [os.environ.get("TRADUCTION_PYTHON")]
    for env in ("interview", "traduction"):
        cands.append(os.path.expanduser(f"~/miniconda3/envs/{env}/bin/python"))
    for c in cands:
        if c and os.path.exists(c):
            return c
    return sys.executable


try:
    from flask import Flask, request, jsonify, Response
except ImportError:
    print("❌ Flask n'est pas installé dans cet interpréteur.")
    print(f"   Lance plutôt :  {_resolve_python()} gui.py")
    print("   Ou diagnostique l'installation :  python3 doctor.py --install")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_BIN = _resolve_python()
HOST = "127.0.0.1"
PORT = 5005

# Modèles Ollama recommandés (servent de suggestions dans les listes déroulantes,
# en plus des modèles réellement installés détectés via `ollama list`).
OLLAMA_SUGGESTIONS = [
    "qwen3.6:27b", "gemma4:31b", "mistral-small:latest",
    "aya-expanse:32b", "qwen3:32b", "command-r:35b",
]

# ═══════════════════════════════════════════════════════════════════════════════
# MANIFESTE DES SCRIPTS
# ═══════════════════════════════════════════════════════════════════════════════
# Chaque champ :
#   name      identifiant interne (clé des valeurs)
#   flag      drapeau CLI (None → argument positionnel)
#   label     libellé affiché
#   type      source | file | dir | text | int | float | select | toggle | textarea | ollama
#   default   valeur par défaut
#   choices   pour select
#   off_flag  pour un toggle « activé par défaut » : drapeau émis quand on DÉSACTIVE
#   adv       True → rangé dans la section « Avancé »
#   depends   {champ: valeur} → affiché seulement si la condition est vraie
#   help      info-bulle
#   placeholder

LANG = dict(type="text", placeholder="ex: en, fr, ja…")

SCRIPTS = [
    {
        "id": "traduire", "file": "traduire.py", "label": "Sous-titres", "icon": "💬",
        "desc": "Traduction + incrustation de sous-titres (6 passes)",
        "fields": [
            {"name": "source", "flag": None, "label": "Source (fichier vidéo ou URL YouTube)", "type": "source", "required": True},
            {"name": "source_lang", "flag": "-s", "label": "Langue source", "default": "en", **LANG},
            {"name": "target_lang", "flag": "-t", "label": "Langue cible", "default": "fr", **LANG},
            {"name": "llm", "flag": "--llm", "label": "Moteur LLM", "type": "select", "choices": ["local", "claude"], "default": "local"},
            {"name": "ollama_model", "flag": "--ollama-model", "label": "Modèle local (Ollama)", "type": "ollama", "default": "gemma4:31b", "depends": {"llm": "local"}},
            {"name": "style", "flag": "--style", "label": "Style sous-titres", "type": "select", "choices": ["default", "box", "minimal", "netflix", "youtube"], "default": "default"},
            {"name": "context", "flag": "--context", "label": "Contexte (noms propres, sujet…)", "type": "textarea"},
            {"name": "dubbing", "flag": "--dubbing", "label": "Doublage audio en plus", "type": "toggle"},
            {"name": "skip_burn", "flag": "--skip-burn", "label": "SRT uniquement (pas d'incrustation)", "type": "toggle"},
            {"name": "skip_review", "flag": "--skip-review", "label": "Sauter la relecture", "type": "toggle"},
            {"name": "output", "flag": "-o", "label": "Sortie MP4", "type": "file", "adv": True},
            {"name": "ocr", "flag": "--ocr", "label": "OCR (sous-titres incrustés)", "type": "toggle", "adv": True},
            {"name": "demucs", "flag": None, "off_flag": "--no-demucs", "label": "Séparation voix (Demucs)", "type": "toggle", "default": True, "adv": True},
            {"name": "resume", "flag": "--resume", "label": "Reprendre (segments JSON)", "type": "file", "adv": True},
            {"name": "srt_only", "flag": "--srt-only", "label": "Incruster un SRT existant", "type": "file", "adv": True},
            {"name": "delogo", "flag": "--delogo", "label": "Masquer logo", "type": "text", "placeholder": "X:Y:W:H", "adv": True},
            {"name": "vad_onset", "flag": "--vad-onset", "label": "VAD onset", "type": "float", "adv": True},
            {"name": "vad_offset", "flag": "--vad-offset", "label": "VAD offset", "type": "float", "adv": True},
            {"name": "whisper_model", "flag": "--whisper-model", "label": "Modèle Whisper", "type": "text", "default": "large-v3", "adv": True},
            {"name": "claude_model", "flag": "--claude-model", "label": "Modèle Claude", "type": "text", "default": "claude-opus-4-5", "adv": True, "depends": {"llm": "claude"}},
            {"name": "ollama_url", "flag": "--ollama-url", "label": "URL Ollama", "type": "text", "default": "http://localhost:11434", "adv": True, "depends": {"llm": "local"}},
            {"name": "cookies", "flag": "--cookies", "label": "Cookies JSON", "type": "file", "adv": True},
        ],
    },
    {
        "id": "resumer", "file": "resumer.py", "label": "Résumé", "icon": "📄",
        "desc": "Résumé structuré en PDF + EPUB",
        "fields": [
            {"name": "input", "flag": None, "label": "Source (vidéo, URL YouTube ou Apollo)", "type": "source", "required": True},
            {"name": "source", "flag": "-s", "label": "Langue source", "default": "", **LANG},
            {"name": "target", "flag": "-t", "label": "Langue du résumé", "default": "fr", **LANG},
            {"name": "llm", "flag": "--llm", "label": "Moteur LLM", "type": "select", "choices": ["local", "claude"], "default": "local"},
            {"name": "ollama_model", "flag": "--ollama-model", "label": "Modèle local (Ollama)", "type": "ollama", "default": "qwen3.6:27b", "depends": {"llm": "local"}},
            {"name": "pages", "flag": "--pages", "label": "Nombre de pages cible", "type": "int"},
            {"name": "context", "flag": "--context", "label": "Contexte", "type": "textarea"},
            {"name": "html", "flag": "--html", "label": "Générer aussi un HTML", "type": "toggle"},
            {"name": "output_dir", "flag": "--output-dir", "label": "Dossier de copie finale", "type": "dir", "adv": True},
            {"name": "resume", "flag": "--resume", "label": "Reprendre (segments JSON)", "type": "file", "adv": True},
            {"name": "claude_model", "flag": "--claude-model", "label": "Modèle Claude", "type": "text", "adv": True, "depends": {"llm": "claude"}},
            {"name": "ollama_url", "flag": "--ollama-url", "label": "URL Ollama", "type": "text", "default": "http://localhost:11434", "adv": True, "depends": {"llm": "local"}},
            {"name": "cookies", "flag": "--cookies", "label": "Cookies JSON (Apollo)", "type": "file", "adv": True},
        ],
    },
    {
        "id": "clipper", "file": "clipper.py", "label": "Clips viraux", "icon": "✂️",
        "desc": "Extraction de passages + sous-titres karaoké",
        "fields": [
            {"name": "source", "flag": None, "label": "Source (vidéo ou URL)", "type": "source", "required": True},
            {"name": "criteria", "flag": "--criteria", "label": "Critère de sélection", "type": "text", "placeholder": "ex: passage le plus marquant"},
            {"name": "max_clips", "flag": "-n", "label": "Nombre de clips", "type": "int", "default": 3},
            {"name": "duration", "flag": "--duration", "label": "Durée (sec)", "type": "text", "default": "139-900", "placeholder": "min-max"},
            {"name": "source_lang", "flag": "-s", "label": "Langue source", **LANG},
            {"name": "target_lang", "flag": "-t", "label": "Langue cible (traduction)", **LANG},
            {"name": "llm", "flag": "--llm", "label": "Moteur LLM", "type": "select", "choices": ["local", "claude"], "default": "local"},
            {"name": "ollama_model", "flag": "--ollama-model", "label": "Modèle local (Ollama)", "type": "ollama", "default": "qwen3.6:27b", "depends": {"llm": "local"}},
            {"name": "context", "flag": "--context", "label": "Contexte", "type": "textarea"},
            {"name": "post", "flag": "--post", "label": "Mode publication", "type": "toggle", "adv": True},
            {"name": "skip_burn", "flag": "--skip-burn", "label": "Pas d'incrustation", "type": "toggle", "adv": True},
            {"name": "words_per_group", "flag": "--words-per-group", "label": "Mots / groupe karaoké", "type": "int", "adv": True},
            {"name": "speaker", "flag": "--speaker", "label": "Locuteur", "type": "text", "adv": True},
            {"name": "date", "flag": "--date", "label": "Date", "type": "text", "adv": True},
            {"name": "url", "flag": "--url", "label": "URL source (métadonnée)", "type": "text", "adv": True},
            {"name": "pre_segments", "flag": "--pre-segments", "label": "Segments pré-calculés", "type": "file", "adv": True},
            {"name": "resume", "flag": "--resume", "label": "Reprendre (clips JSON)", "type": "file", "adv": True},
            {"name": "whisper_model", "flag": "--whisper-model", "label": "Modèle Whisper", "type": "text", "default": "large-v3", "adv": True},
            {"name": "cookies", "flag": "--cookies", "label": "Cookies JSON", "type": "file", "adv": True},
        ],
    },
    {
        "id": "doubler_mp3", "file": "doubler-mp3-batch.py", "label": "Doublage MP3", "icon": "🎙️",
        "desc": "Doublage audio par lot (MP3/MP4 du dossier)",
        "fields": [
            {"name": "file", "flag": "--file", "label": "Fichier précis (sinon tout le dossier)", "type": "source"},
            {"name": "source_lang", "flag": "-s", "label": "Langue source", "default": "en", **LANG},
            {"name": "target_lang", "flag": "-t", "label": "Langue cible", "default": "fr", **LANG},
            {"name": "model", "flag": "--model", "label": "Moteur TTS", "type": "select", "choices": ["qwen3tts", "xtts"], "default": "qwen3tts"},
            {"name": "llm", "flag": "--llm", "label": "Moteur LLM (traduction)", "type": "select", "choices": ["local", "claude"], "default": "local"},
            {"name": "ollama_model", "flag": "--ollama-model", "label": "Modèle local (Ollama)", "type": "ollama", "default": "gemma4:31b", "depends": {"llm": "local"}},
            {"name": "ref_voice", "flag": "--ref-voice", "label": "Voix de référence (WAV)", "type": "file"},
            {"name": "speakers", "flag": "--speakers", "label": "Nombre de locuteurs", "type": "int"},
            {"name": "gender", "flag": "--gender", "label": "Genre voix", "type": "select", "choices": ["auto", "male", "female"], "default": "auto"},
            {"name": "context", "flag": "--context", "label": "Contexte", "type": "textarea"},
            {"name": "xtts_speaker", "flag": "--xtts-speaker", "label": "Speaker XTTS", "type": "text", "adv": True},
            {"name": "pause", "flag": "--pause", "label": "Pause (ms)", "type": "int", "adv": True},
            {"name": "speaker_pause", "flag": "--speaker-pause", "label": "Pause inter-locuteurs (ms)", "type": "int", "adv": True},
            {"name": "segments", "flag": "--segments", "label": "Segments JSON", "type": "file", "adv": True},
            {"name": "skip_review", "flag": "--skip-review", "label": "Sauter la relecture", "type": "toggle", "adv": True},
            {"name": "skip_checks", "flag": "--skip-checks", "label": "Sauter les vérifications", "type": "toggle", "adv": True},
            {"name": "verify_tts", "flag": "--verify-tts", "label": "Vérifier le TTS", "type": "toggle", "adv": True},
            {"name": "whisper_model", "flag": "--whisper-model", "label": "Modèle Whisper", "type": "text", "default": "large-v3", "adv": True},
        ],
    },
    {
        "id": "doubler_xtts", "file": "doubler.py", "label": "Doublage vidéo", "icon": "🎬",
        "desc": "Doublage vidéo avec voix-off (11 passes)",
        "fields": [
            {"name": "video", "flag": None, "label": "Source (vidéo ou URL YouTube)", "type": "source", "required": True},
            {"name": "source_lang", "flag": "-s", "label": "Langue source", "default": "en", **LANG},
            {"name": "target_lang", "flag": "-t", "label": "Langue cible", "default": "fr", **LANG},
            {"name": "tts", "flag": "--tts", "label": "Moteur TTS", "type": "select", "choices": ["qwen3tts", "xtts", "elevenlabs"], "default": "qwen3tts"},
            {"name": "llm", "flag": "--llm", "label": "Moteur LLM", "type": "select", "choices": ["local", "claude"], "default": "local"},
            {"name": "ollama_model", "flag": "--ollama-model", "label": "Modèle local (Ollama)", "type": "ollama", "default": "gemma4:31b", "depends": {"llm": "local"}},
            {"name": "voiceover", "flag": None, "off_flag": "--no-voiceover", "label": "Voix-off (ducking)", "type": "toggle", "default": True},
            {"name": "vo_style", "flag": "--vo-style", "label": "Style voix-off", "type": "select", "choices": ["", "arte", "jt", "jt-flat", "bbc"], "default": ""},
            {"name": "ref_voice", "flag": "--ref-voice", "label": "Voix de référence (WAV)", "type": "file"},
            {"name": "ref_voices", "flag": "--ref-voices", "label": "Dossier de voix", "type": "dir", "default": "voix"},
            {"name": "map_voices", "flag": "--map-voices", "label": "Appariement voix↔locuteur (interactif)", "type": "toggle"},
            {"name": "clone_original", "flag": "--clone-original", "label": "Cloner la voix d'origine", "type": "toggle"},
            {"name": "gender", "flag": "--gender", "label": "Genre voix", "type": "select", "choices": ["auto", "male", "female"], "default": "auto"},
            {"name": "speakers", "flag": "--speakers", "label": "Nombre de locuteurs", "type": "int"},
            {"name": "context", "flag": "--context", "label": "Contexte", "type": "textarea"},
            {"name": "output", "flag": "-o", "label": "Sortie MP4", "type": "file", "adv": True},
            {"name": "remove_music", "flag": "--remove-music", "label": "Retirer la musique", "type": "toggle", "adv": True},
            {"name": "keep_original", "flag": "--keep-original", "label": "Garder l'original (volume)", "type": "float", "adv": True},
            {"name": "dual_audio", "flag": "--dual-audio", "label": "Piste audio double", "type": "toggle", "adv": True},
            {"name": "use_srt", "flag": "--use-srt", "label": "Utiliser un SRT", "type": "file", "adv": True},
            {"name": "vo_lead_in", "flag": "--vo-lead-in", "label": "Voix-off lead-in (ms)", "type": "int", "adv": True},
            {"name": "vo_lead_out", "flag": "--vo-lead-out", "label": "Voix-off lead-out (ms)", "type": "int", "adv": True},
            {"name": "vo_duck_db", "flag": "--vo-duck-db", "label": "Ducking (dB)", "type": "int", "adv": True},
            {"name": "elevenlabs_voice", "flag": "--elevenlabs-voice", "label": "ElevenLabs voice id", "type": "text", "adv": True, "depends": {"tts": "elevenlabs"}},
            {"name": "elevenlabs_model", "flag": "--elevenlabs-model", "label": "ElevenLabs model", "type": "text", "adv": True, "depends": {"tts": "elevenlabs"}},
            {"name": "skip", "flag": "--skip", "label": "Sauter jusqu'à (MM:SS)", "type": "text", "adv": True},
            {"name": "skip_review", "flag": "--skip-review", "label": "Sauter relecture", "type": "toggle", "adv": True},
            {"name": "skip_checks", "flag": "--skip-checks", "label": "Sauter vérifications", "type": "toggle", "adv": True},
            {"name": "skip_isochrony", "flag": "--skip-isochrony", "label": "Sauter isochronie", "type": "toggle", "adv": True},
            {"name": "skip_normalize", "flag": "--skip-normalize", "label": "Sauter normalisation", "type": "toggle", "adv": True},
            {"name": "fix_pitch", "flag": "--fix-pitch", "label": "Correction de hauteur", "type": "toggle", "adv": True},
            {"name": "audio_only", "flag": "--audio-only", "label": "Audio seul", "type": "toggle", "adv": True},
            {"name": "onlydub", "flag": "--onlydub", "label": "Doublage seul", "type": "toggle", "adv": True},
            {"name": "watermark", "flag": "--watermark", "label": "Filigrane", "type": "toggle", "adv": True},
            {"name": "segments", "flag": "--segments", "label": "Segments JSON", "type": "file", "adv": True},
            {"name": "whisper_model", "flag": "--whisper-model", "label": "Modèle Whisper", "type": "text", "default": "large-v3", "adv": True},
            {"name": "claude_model", "flag": "--claude-model", "label": "Modèle Claude", "type": "text", "adv": True, "depends": {"llm": "claude"}},
            {"name": "ollama_url", "flag": "--ollama-url", "label": "URL Ollama", "type": "text", "default": "http://localhost:11434", "adv": True, "depends": {"llm": "local"}},
            {"name": "cookies", "flag": "--cookies", "label": "Cookies JSON", "type": "file", "adv": True},
        ],
    },
    {
        "id": "transcrire", "file": "transcrire.py", "label": "Transcription", "icon": "📝",
        "desc": "Transcription nettoyée → DOCX",
        "fields": [
            {"name": "input", "flag": None, "label": "Source (vidéo, URL ou playlist)", "type": "source", "required": True},
            {"name": "invites", "flag": "--invites", "label": "Invité·es", "type": "text", "placeholder": "Sophie, Marc…"},
            {"name": "interviewers", "flag": "--interviewers", "label": "Intervieweur·euses", "type": "text"},
            {"name": "output", "flag": "-o", "label": "Sortie DOCX", "type": "file"},
            {"name": "playlist", "flag": "--playlist", "label": "Playlist entière", "type": "toggle"},
            {"name": "model", "flag": "--model", "label": "Modèle Claude", "type": "text", "default": "claude-opus-4-5", "adv": True},
            {"name": "whisper_model", "flag": "--whisper-model", "label": "Modèle Whisper", "type": "text", "default": "large-v3", "adv": True},
            {"name": "raw_only", "flag": "--raw-only", "label": "Transcription brute seule", "type": "toggle", "adv": True},
            {"name": "skip_heuristics", "flag": "--skip-heuristics", "label": "Sauter heuristiques", "type": "toggle", "adv": True},
            {"name": "skip_claude_fix", "flag": "--skip-claude-fix", "label": "Sauter correction Claude", "type": "toggle", "adv": True},
            {"name": "rewrite_only", "flag": "--rewrite-only", "label": "Réécriture seule", "type": "toggle", "adv": True},
            {"name": "ignore", "flag": "--ignore", "label": "Ignorer", "type": "text", "adv": True},
            {"name": "skip", "flag": "--skip", "label": "Sauter", "type": "text", "adv": True},
            {"name": "resume", "flag": "--resume", "label": "Reprendre", "type": "text", "adv": True},
        ],
    },
    {
        "id": "traduire_pro", "file": "traduire-pro.py", "label": "Sous-titres Pro", "icon": "🎞️",
        "desc": "Pipeline sous-titres avancé (audit, résumé, doublage)",
        "fields": [
            {"name": "source", "flag": None, "label": "Source (vidéo ou URL)", "type": "source", "required": True},
            {"name": "source_lang", "flag": "-s", "label": "Langue source", "default": "en", **LANG},
            {"name": "target_lang", "flag": "-t", "label": "Langue cible", "default": "fr", **LANG},
            {"name": "style", "flag": "--style", "label": "Style sous-titres", "type": "select", "choices": ["default", "box", "minimal", "netflix", "youtube"], "default": "default"},
            {"name": "context", "flag": "--context", "label": "Contexte", "type": "textarea"},
            {"name": "llm", "flag": "--llm", "label": "Moteur LLM", "type": "select", "choices": ["local", "claude"], "default": "local"},
            {"name": "ollama_model", "flag": "--ollama-model", "label": "Modèle local (Ollama)", "type": "ollama", "default": "gemma4:31b", "depends": {"llm": "local"}},
            {"name": "no_dubbing", "flag": "--no-dubbing", "label": "Sans doublage", "type": "toggle"},
            {"name": "audit_cuts", "flag": "--audit-cuts", "label": "Auditer les coupes", "type": "toggle"},
            {"name": "num_speakers", "flag": "--num-speakers", "label": "Nombre de locuteurs", "type": "int"},
            {"name": "skip_summary", "flag": "--skip-summary", "label": "Sauter résumé", "type": "toggle", "adv": True},
            {"name": "skip_review", "flag": "--skip-review", "label": "Sauter relecture", "type": "toggle", "adv": True},
            {"name": "skip_burn", "flag": "--skip-burn", "label": "Pas d'incrustation", "type": "toggle", "adv": True},
            {"name": "no_trim_music", "flag": "--no-trim-music", "label": "Ne pas couper la musique", "type": "toggle", "adv": True},
            {"name": "max_cps", "flag": "--max-cps", "label": "CPS max", "type": "int", "adv": True},
            {"name": "delogo", "flag": "--delogo", "label": "Masquer logo", "type": "text", "placeholder": "X:Y:W:H", "adv": True},
            {"name": "ocr", "flag": "--ocr", "label": "OCR", "type": "toggle", "adv": True},
            {"name": "resume", "flag": "--resume", "label": "Reprendre (JSON)", "type": "file", "adv": True},
            {"name": "claude_model", "flag": "--claude-model", "label": "Modèle Claude", "type": "text", "default": "claude-opus-4-5", "adv": True},
            {"name": "whisper_model", "flag": "--whisper-model", "label": "Modèle Whisper", "type": "text", "default": "large-v3", "adv": True},
        ],
    },
    {
        "id": "sous_titrer_docx", "file": "sous_titrer_docx.py", "label": "Sous-titres depuis DOCX", "icon": "🗂️",
        "desc": "Génère un SRT/incrustation à partir d'un DOCX traduit",
        "fields": [
            {"name": "video", "flag": None, "label": "Vidéo source", "type": "source", "required": True},
            {"name": "docx", "flag": None, "label": "DOCX traduit", "type": "file", "required": True},
            {"name": "style", "flag": "--style", "label": "Style sous-titres", "type": "select", "choices": ["default", "box", "minimal", "netflix", "youtube"], "default": "default"},
            {"name": "srt_only", "flag": "--srt-only", "label": "SRT seul (pas d'incrustation)", "type": "toggle"},
            {"name": "llm", "flag": "--llm", "label": "Moteur LLM (alignement)", "type": "select", "choices": ["local", "claude"], "default": "local"},
            {"name": "ollama_model", "flag": "--ollama-model", "label": "Modèle local (Ollama)", "type": "ollama", "default": "qwen3.6:27b", "depends": {"llm": "local"}},
            {"name": "resume", "flag": "--resume", "label": "Reprendre (segments alignés JSON)", "type": "file", "adv": True},
            {"name": "whisperx_json", "flag": "--whisperx-json", "label": "JSON WhisperX pré-calculé", "type": "file", "adv": True},
        ],
    },
]

# Ordre d'affichage dans la GUI : les usages principaux d'abord.
_SCRIPT_ORDER = ["traduire", "doubler_xtts", "clipper", "resumer",
                 "doubler_mp3", "traduire_pro", "transcrire", "sous_titrer_docx"]
SCRIPTS.sort(key=lambda s: _SCRIPT_ORDER.index(s["id"])
             if s["id"] in _SCRIPT_ORDER else len(_SCRIPT_ORDER))

SCRIPTS_BY_ID = {s["id"]: s for s in SCRIPTS}

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTRUCTION DE LA COMMANDE
# ═══════════════════════════════════════════════════════════════════════════════

def build_command(script_id, values):
    """Reconstruit la liste d'arguments depuis le manifeste + les valeurs saisies."""
    spec = SCRIPTS_BY_ID[script_id]
    positionals, options = [], []

    def visible(field):
        dep = field.get("depends")
        if not dep:
            return True
        return all(str(values.get(k, "")) == str(v) for k, v in dep.items())

    for f in spec["fields"]:
        if not visible(f):
            continue
        name, flag, ftype = f["name"], f.get("flag"), f.get("type", "text")
        val = values.get(name, None)

        if ftype == "toggle":
            on = bool(val)
            if f.get("off_flag"):           # activé par défaut → drapeau quand on coupe
                if not on:
                    options.append(f["off_flag"])
            elif on and flag:
                options.append(flag)
            continue

        if val is None or str(val).strip() == "":
            continue
        val = str(val).strip()

        if flag is None:                    # argument positionnel
            positionals.append(val)
        else:
            options.append(flag)
            options.append(val)

    return positionals, options


def full_command(script_id, values):
    spec = SCRIPTS_BY_ID[script_id]
    pos, opt = build_command(script_id, values)
    return [PYTHON_BIN, spec["file"]] + pos + opt


# ═══════════════════════════════════════════════════════════════════════════════
# GESTION DES PROCESSUS
# ═══════════════════════════════════════════════════════════════════════════════

RUNS = {}            # id → {proc, lines:[], done:bool, code, cmd}
RUNS_LOCK = threading.Lock()


def _reader(run_id, proc):
    for line in iter(proc.stdout.readline, ""):
        with RUNS_LOCK:
            RUNS[run_id]["lines"].append(line.rstrip("\n"))
    proc.stdout.close()
    code = proc.wait()
    with RUNS_LOCK:
        RUNS[run_id]["done"] = True
        RUNS[run_id]["code"] = code


def start_run(script_id, values):
    cmd = full_command(script_id, values)
    run_id = uuid.uuid4().hex[:12]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd, cwd=SCRIPT_DIR, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, start_new_session=True,
    )
    with RUNS_LOCK:
        RUNS[run_id] = {"proc": proc, "lines": [], "done": False, "code": None,
                        "cmd": " ".join(shlex.quote(c) for c in cmd)}
    threading.Thread(target=_reader, args=(run_id, proc), daemon=True).start()
    return run_id, RUNS[run_id]["cmd"]


def stop_run(run_id):
    with RUNS_LOCK:
        run = RUNS.get(run_id)
    if not run or run["done"]:
        return False
    try:
        os.killpg(os.getpgid(run["proc"].pid), signal.SIGTERM)
        time.sleep(0.5)
        if run["proc"].poll() is None:
            os.killpg(os.getpgid(run["proc"].pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    return True


def list_ollama_models():
    models = []
    try:
        out = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=8)
        for line in out.stdout.splitlines()[1:]:
            name = line.split()[0] if line.split() else ""
            if name:
                models.append(name)
    except Exception:
        pass
    # fusion avec les suggestions, sans doublon, modèles installés en tête
    for s in OLLAMA_SUGGESTIONS:
        if s not in models:
            models.append(s)
    return models


def browse(path):
    path = os.path.abspath(os.path.expanduser(path or SCRIPT_DIR))
    if not os.path.isdir(path):
        path = SCRIPT_DIR
    dirs, files = [], []
    try:
        for name in sorted(os.listdir(path), key=str.lower):
            if name.startswith("."):
                continue
            full = os.path.join(path, name)
            if os.path.isdir(full):
                dirs.append(name)
            else:
                files.append(name)
    except PermissionError:
        pass
    return {"path": path, "parent": os.path.dirname(path), "dirs": dirs, "files": files}


# ═══════════════════════════════════════════════════════════════════════════════
# SERVEUR FLASK
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)


@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.route("/api/scripts")
def api_scripts():
    return jsonify(SCRIPTS)


@app.route("/api/ollama")
def api_ollama():
    return jsonify(list_ollama_models())


@app.route("/api/browse")
def api_browse():
    return jsonify(browse(request.args.get("path", "")))


@app.route("/api/preview", methods=["POST"])
def api_preview():
    data = request.get_json(force=True)
    cmd = full_command(data["script"], data.get("values", {}))
    return jsonify({"cmd": " ".join(shlex.quote(c) for c in cmd)})


@app.route("/api/run", methods=["POST"])
def api_run():
    data = request.get_json(force=True)
    run_id, cmd = start_run(data["script"], data.get("values", {}))
    return jsonify({"id": run_id, "cmd": cmd})


@app.route("/api/stop/<run_id>", methods=["POST"])
def api_stop(run_id):
    return jsonify({"stopped": stop_run(run_id)})


@app.route("/api/stream/<run_id>")
def api_stream(run_id):
    def gen():
        sent = 0
        yield "retry: 2000\n\n"
        while True:
            with RUNS_LOCK:
                run = RUNS.get(run_id)
                if not run:
                    yield "event: error\ndata: run introuvable\n\n"
                    return
                lines = run["lines"][sent:]
                sent += len(lines)
                done, code = run["done"], run["code"]
            for ln in lines:
                yield "data: " + json.dumps(ln) + "\n\n"
            if done and sent >= len(RUNS[run_id]["lines"]):
                yield "event: done\ndata: " + json.dumps({"code": code}) + "\n\n"
                return
            time.sleep(0.25)
    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ═══════════════════════════════════════════════════════════════════════════════
# FRONTEND (HTML + CSS + JS embarqués) — palette de l'extension de traduction
# ═══════════════════════════════════════════════════════════════════════════════

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Traduction — Panneau de contrôle</title>
<style>
:root{
  --bg:#14161a; --panel:#1c1f26; --panel2:#0c0e12; --border:#2a2e38;
  --text:#e6e8ec; --muted:#8a90a0; --accent:#ff8a3d; --accent-hover:#ffa362;
  --ok:#4ad27e; --warn:#e5b04b; --danger:#e04b4b;
}
*{box-sizing:border-box}
html,body{margin:0;height:100%}
body{background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;display:flex;height:100vh;overflow:hidden}
a{color:var(--accent)}

/* Sidebar */
#sidebar{width:248px;flex:none;background:var(--panel2);border-right:1px solid var(--border);display:flex;flex-direction:column}
#sidebar .brand{padding:18px 18px 12px;font-weight:700;font-size:16px;letter-spacing:.2px}
#sidebar .brand small{display:block;color:var(--muted);font-weight:400;font-size:11px;margin-top:2px}
#scriptlist{overflow-y:auto;padding:6px}
.scriptitem{display:flex;gap:10px;align-items:center;padding:10px 12px;border-radius:10px;cursor:pointer;color:var(--text);transition:background .12s}
.scriptitem:hover{background:#181b22}
.scriptitem.active{background:#23262f;outline:1px solid var(--border)}
.scriptitem .ic{font-size:18px;width:22px;text-align:center}
.scriptitem .t{font-weight:600}
.scriptitem .d{font-size:11px;color:var(--muted);line-height:1.25}

/* Main */
#main{flex:1;display:flex;flex-direction:column;min-width:0}
#header{padding:16px 22px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:12px}
#header h1{font-size:17px;margin:0}
#header .sub{color:var(--muted);font-size:12px}
#body{flex:1;display:flex;min-height:0}

/* Form column */
#formwrap{flex:1;overflow-y:auto;padding:20px 22px;min-width:0}
.section-title{font-size:11px;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);margin:6px 0 12px;font-weight:700}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px}
.field{display:flex;flex-direction:column;gap:5px;min-width:0}
.field.full{grid-column:1/-1}
.field label{font-size:12px;color:var(--muted);font-weight:600}
.field .req{color:var(--accent)}
input[type=text],input[type=number],select,textarea{
  background:var(--panel);border:1px solid var(--border);color:var(--text);
  border-radius:9px;padding:9px 11px;font-size:13px;width:100%;outline:none;transition:border .12s}
input:focus,select:focus,textarea:focus{border-color:var(--accent)}
textarea{resize:vertical;min-height:62px;font-family:inherit}
select{appearance:none;background-image:linear-gradient(45deg,transparent 50%,var(--muted) 50%),linear-gradient(135deg,var(--muted) 50%,transparent 50%);background-position:calc(100% - 16px) 17px,calc(100% - 11px) 17px;background-size:5px 5px;background-repeat:no-repeat;padding-right:30px}
.inputrow{display:flex;gap:8px}
.inputrow input{flex:1}
.browsebtn{background:var(--panel);border:1px solid var(--border);color:var(--muted);border-radius:9px;padding:0 12px;cursor:pointer;font-size:12px;white-space:nowrap}
.browsebtn:hover{border-color:var(--accent);color:var(--text)}

/* Toggle */
.toggle{display:flex;align-items:center;gap:10px;background:var(--panel);border:1px solid var(--border);border-radius:9px;padding:9px 11px;cursor:pointer}
.toggle:hover{border-color:#3a3f4c}
.toggle .sw{width:36px;height:20px;border-radius:20px;background:#33384a;position:relative;flex:none;transition:background .15s}
.toggle .sw::after{content:"";position:absolute;width:16px;height:16px;border-radius:50%;background:#cfd3dc;top:2px;left:2px;transition:left .15s}
.toggle.on .sw{background:var(--accent)}
.toggle.on .sw::after{left:18px;background:#1b1205}
.toggle .lab{font-size:12px;color:var(--text)}

.advtoggle{margin:22px 0 10px;color:var(--muted);cursor:pointer;font-size:12px;font-weight:600;user-select:none;display:inline-flex;align-items:center;gap:6px}
.advtoggle:hover{color:var(--text)}
#advanced{display:none}
#advanced.open{display:block}

/* Console */
#console-col{width:42%;min-width:340px;max-width:680px;border-left:1px solid var(--border);display:flex;flex-direction:column;background:var(--panel2)}
#cmdbar{padding:12px 16px;border-bottom:1px solid var(--border)}
#cmdbar .lbl{font-size:10px;text-transform:uppercase;letter-spacing:.6px;color:var(--muted);margin-bottom:5px}
#cmdpreview{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;color:#c6cad2;background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:9px 11px;white-space:pre-wrap;word-break:break-all;max-height:96px;overflow-y:auto}
#runbar{padding:10px 16px;display:flex;gap:10px;align-items:center;border-bottom:1px solid var(--border)}
.btn{border:none;border-radius:9px;padding:9px 18px;font-size:13px;font-weight:700;cursor:pointer}
.btn-run{background:var(--accent);color:#1b1205}
.btn-run:hover{background:var(--accent-hover)}
.btn-stop{background:var(--danger);color:#fff}
.btn:disabled{opacity:.4;cursor:not-allowed}
#status{font-size:12px;color:var(--muted);margin-left:auto}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;vertical-align:middle}
.dot.idle{background:#555}.dot.run{background:var(--accent);animation:pulse 1s infinite}.dot.ok{background:var(--ok)}.dot.err{background:var(--danger)}
@keyframes pulse{50%{opacity:.3}}
#console{flex:1;overflow-y:auto;padding:12px 16px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;line-height:1.55;white-space:pre-wrap;word-break:break-word}
#console .l-ok{color:var(--ok)}#console .l-err{color:#ff8d8d}#console .l-warn{color:var(--warn)}#console .l-step{color:var(--accent-hover);font-weight:600}#console .l-dim{color:var(--muted)}
#console .placeholder{color:var(--muted)}

/* Modal file browser */
#modal{position:fixed;inset:0;background:rgba(0,0,0,.55);display:none;align-items:center;justify-content:center;z-index:50}
#modal.open{display:flex}
.modalbox{width:620px;max-width:92vw;max-height:80vh;background:var(--panel);border:1px solid var(--border);border-radius:14px;display:flex;flex-direction:column;overflow:hidden}
.modalhead{padding:14px 16px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:10px}
.modalhead .path{font-family:ui-monospace,monospace;font-size:12px;color:var(--muted);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;direction:rtl;text-align:left}
.modalhead button{background:var(--panel2);border:1px solid var(--border);color:var(--text);border-radius:8px;padding:6px 12px;cursor:pointer;font-size:12px}
.modalhead button:hover{border-color:var(--accent)}
.modallist{overflow-y:auto;padding:8px}
.entry{display:flex;align-items:center;gap:10px;padding:8px 11px;border-radius:8px;cursor:pointer}
.entry:hover{background:#23262f}
.entry .ic{width:18px;text-align:center}
.entry.dir .ic{color:var(--accent)}
.entry.media{color:var(--text)}
.entry.file{color:var(--muted)}
.modalfoot{padding:10px 16px;border-top:1px solid var(--border);display:flex;justify-content:flex-end;gap:8px}
.empty{padding:40px;text-align:center;color:var(--muted)}
::-webkit-scrollbar{width:10px;height:10px}::-webkit-scrollbar-thumb{background:#2c313c;border-radius:6px}::-webkit-scrollbar-track{background:transparent}
</style>
</head>
<body>
<div id="sidebar">
  <div class="brand">🎛️ Traduction <small>panneau de contrôle local</small></div>
  <div id="scriptlist"></div>
</div>

<div id="main">
  <div id="header">
    <h1 id="h-title">—</h1>
    <span class="sub" id="h-desc"></span>
  </div>
  <div id="body">
    <div id="formwrap">
      <div class="section-title">Paramètres principaux</div>
      <div class="grid" id="main-fields"></div>
      <div class="advtoggle" id="adv-toggle">▸ Options avancées</div>
      <div id="advanced">
        <div class="grid" id="adv-fields"></div>
      </div>
    </div>
    <div id="console-col">
      <div id="cmdbar">
        <div class="lbl">Commande</div>
        <div id="cmdpreview">—</div>
      </div>
      <div id="runbar">
        <button class="btn btn-run" id="btn-run">▶ Lancer</button>
        <button class="btn btn-stop" id="btn-stop" disabled>■ Stop</button>
        <span id="status"><span class="dot idle"></span>prêt</span>
      </div>
      <div id="console"><span class="placeholder">La sortie du script s'affichera ici…</span></div>
    </div>
  </div>
</div>

<div id="modal">
  <div class="modalbox">
    <div class="modalhead">
      <button id="m-up">⬆ Parent</button>
      <span class="path" id="m-path"></span>
      <button id="m-close">✕</button>
    </div>
    <div class="modallist" id="m-list"></div>
    <div class="modalfoot">
      <button class="browsebtn" id="m-pickdir" style="display:none">📁 Choisir ce dossier</button>
    </div>
  </div>
</div>

<script>
const MEDIA = ['.mp4','.mkv','.mov','.avi','.webm','.mp3','.wav','.m4a','.flac','.json','.srt','.docx','.ass'];
let SCRIPTS=[], OLLAMA=[], current=null, values={}, evtSource=null, modalTarget=null, modalDirMode=false, modalPath='';

const $=s=>document.querySelector(s);
const el=(t,c,h)=>{const e=document.createElement(t);if(c)e.className=c;if(h!=null)e.innerHTML=h;return e;};

async function boot(){
  SCRIPTS = await (await fetch('/api/scripts')).json();
  OLLAMA  = await (await fetch('/api/ollama')).json();
  const list=$('#scriptlist');
  SCRIPTS.forEach(s=>{
    const it=el('div','scriptitem');
    it.innerHTML=`<div class="ic">${s.icon}</div><div><div class="t">${s.label}</div><div class="d">${s.desc}</div></div>`;
    it.onclick=()=>select(s.id);
    it.dataset.id=s.id; list.appendChild(it);
  });
  select(SCRIPTS[0].id);
}

function select(id){
  current=SCRIPTS.find(s=>s.id===id);
  values={};
  current.fields.forEach(f=>{ values[f.name]= 'default' in f ? f.default : (f.type==='toggle'?false:''); });
  document.querySelectorAll('.scriptitem').forEach(e=>e.classList.toggle('active',e.dataset.id===id));
  $('#h-title').textContent=current.icon+'  '+current.label;
  $('#h-desc').textContent=current.desc;
  $('#advanced').classList.remove('open');
  $('#adv-toggle').innerHTML='▸ Options avancées';
  render();
}

function visible(f){
  if(!f.depends) return true;
  return Object.entries(f.depends).every(([k,v])=>String(values[k])===String(v));
}

function widget(f){
  const wrap=el('div','field'+(['textarea'].includes(f.type)?' full':''));
  const req=f.required?' <span class="req">*</span>':'';
  if(f.type!=='toggle') wrap.appendChild(el('label',null,f.label+req));
  const set=v=>{values[f.name]=v;updatePreview();};

  if(f.type==='toggle'){
    const t=el('div','toggle'+(values[f.name]?' on':''));
    t.innerHTML=`<div class="sw"></div><div class="lab">${f.label}</div>`;
    t.onclick=()=>{values[f.name]=!values[f.name];t.classList.toggle('on');updatePreview();};
    wrap.appendChild(t); return wrap;
  }
  if(f.type==='select'){
    const s=el('select');
    f.choices.forEach(c=>{const o=el('option',null,c===''?'(défaut)':c);o.value=c;if(String(values[f.name])===String(c))o.selected=true;s.appendChild(o);});
    s.onchange=()=>{set(s.value); if(['llm','tts'].includes(f.name)) render();};
    wrap.appendChild(s); return wrap;
  }
  if(f.type==='textarea'){
    const t=el('textarea'); t.value=values[f.name]||''; t.oninput=()=>set(t.value);
    wrap.appendChild(t); return wrap;
  }
  if(f.type==='ollama'){
    const s=el('select');
    OLLAMA.forEach(m=>{const o=el('option',null,m);o.value=m;if(values[f.name]===m)o.selected=true;s.appendChild(o);});
    if(!OLLAMA.includes(values[f.name])&&values[f.name]){const o=el('option',null,values[f.name]+' (non installé)');o.value=values[f.name];o.selected=true;s.insertBefore(o,s.firstChild);}
    s.onchange=()=>set(s.value);
    wrap.appendChild(s); return wrap;
  }
  // text / int / float / file / dir / source
  const needsBrowse=['file','dir','source'].includes(f.type);
  const inp=el('input'); inp.type=(f.type==='int'||f.type==='float')?'number':'text';
  if(f.type==='float')inp.step='0.01';
  inp.placeholder=f.placeholder||(f.type==='source'?'chemin du fichier ou URL…':'');
  inp.value=values[f.name]??''; inp.oninput=()=>set(inp.value);
  if(needsBrowse){
    const row=el('div','inputrow'); row.appendChild(inp);
    const b=el('button','browsebtn',f.type==='dir'?'📁':'📂'); b.type='button';
    b.onclick=()=>openModal(f.name,f.type==='dir');
    row.appendChild(b); wrap.appendChild(row);
  } else wrap.appendChild(inp);
  return wrap;
}

function render(){
  const mf=$('#main-fields'), af=$('#adv-fields');
  mf.innerHTML=''; af.innerHTML='';
  let advCount=0;
  current.fields.forEach(f=>{
    if(!visible(f)) return;
    const w=widget(f);
    if(f.adv){af.appendChild(w);advCount++;} else mf.appendChild(w);
  });
  $('#adv-toggle').style.display=advCount?'inline-flex':'none';
  updatePreview();
}

let previewTimer=null;
function updatePreview(){
  clearTimeout(previewTimer);
  previewTimer=setTimeout(async()=>{
    const r=await fetch('/api/preview',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({script:current.id,values})});
    $('#cmdpreview').textContent=(await r.json()).cmd;
  },120);
}

/* ---- file browser modal ---- */
async function openModal(target,dirMode){
  modalTarget=target; modalDirMode=dirMode;
  $('#m-pickdir').style.display=dirMode?'inline-block':'none';
  const start=values[target] && values[target].startsWith('/') ? values[target] : '';
  await loadDir(start);
  $('#modal').classList.add('open');
}
async function loadDir(path){
  const d=await (await fetch('/api/browse?path='+encodeURIComponent(path||''))).json();
  modalPath=d.path; $('#m-path').textContent=d.path;
  const list=$('#m-list'); list.innerHTML='';
  d.dirs.forEach(name=>{
    const e=el('div','entry dir'); e.innerHTML=`<span class="ic">📁</span><span>${name}</span>`;
    e.onclick=()=>loadDir(d.path+'/'+name); list.appendChild(e);
  });
  d.files.forEach(name=>{
    const isMedia=MEDIA.some(x=>name.toLowerCase().endsWith(x));
    const e=el('div','entry '+(isMedia?'media':'file')); e.innerHTML=`<span class="ic">${isMedia?'🎬':'📄'}</span><span>${name}</span>`;
    e.onclick=()=>{ values[modalTarget]=d.path+'/'+name; closeModal(); render(); };
    list.appendChild(e);
  });
  if(!d.dirs.length&&!d.files.length) list.appendChild(el('div','empty',null,'(dossier vide)'));
}
function closeModal(){$('#modal').classList.remove('open');}
$('#m-up').onclick=()=>loadDir(modalPath.replace(/\/[^/]+$/,'')||'/');
$('#m-close').onclick=closeModal;
$('#m-pickdir').onclick=()=>{values[modalTarget]=modalPath;closeModal();render();};
$('#modal').onclick=e=>{if(e.target.id==='modal')closeModal();};

/* ---- run / stream ---- */
function classify(line){
  if(/❌|Error|Traceback|Exception|❗/.test(line))return'l-err';
  if(/✅|terminé|✔/.test(line))return'l-ok';
  if(/⏳|⚠️|attention/i.test(line))return'l-warn';
  if(/^(={3,}|PASSE|🎬|🧠|📝|🎙️|🔊|═)/.test(line)||/^\s*[➤▶]/.test(line))return'l-step';
  if(/^\s+/.test(line))return'l-dim';
  return'';
}
function appendLine(line){
  const c=$('#console');
  if(c.querySelector('.placeholder'))c.innerHTML='';
  const div=el('div',classify(line)); div.textContent=line||' ';
  c.appendChild(div); c.scrollTop=c.scrollHeight;
}
function setStatus(cls,txt){$('#status').innerHTML=`<span class="dot ${cls}"></span>${txt}`;}

let currentRun=null;
$('#btn-run').onclick=async()=>{
  $('#console').innerHTML='';
  const r=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({script:current.id,values})});
  const {id,cmd}=await r.json(); currentRun=id;
  appendLine('$ '+cmd); appendLine('');
  $('#btn-run').disabled=true; $('#btn-stop').disabled=false; setStatus('run','en cours…');
  evtSource=new EventSource('/api/stream/'+id);
  evtSource.onmessage=e=>appendLine(JSON.parse(e.data));
  evtSource.addEventListener('done',e=>{
    const code=JSON.parse(e.data).code;
    appendLine(''); appendLine(code===0?'✅ Terminé (code 0)':'❌ Terminé (code '+code+')');
    setStatus(code===0?'ok':'err',code===0?'terminé':'erreur ('+code+')');
    cleanup();
  });
  evtSource.addEventListener('error',()=>{setStatus('err','flux interrompu');cleanup();});
};
$('#btn-stop').onclick=async()=>{ if(currentRun) await fetch('/api/stop/'+currentRun,{method:'POST'}); setStatus('err','arrêté'); };
function cleanup(){ if(evtSource)evtSource.close(); evtSource=null; currentRun=null; $('#btn-run').disabled=false; $('#btn-stop').disabled=true; }

$('#adv-toggle').onclick=()=>{
  const a=$('#advanced'); a.classList.toggle('open');
  $('#adv-toggle').innerHTML=(a.classList.contains('open')?'▾':'▸')+' Options avancées';
};

boot();
</script>
</body>
</html>
"""


def main():
    print("=" * 60)
    print("  🎛️  Panneau de contrôle Traduction")
    print("=" * 60)
    print(f"  Interpréteur : {PYTHON_BIN}")
    print(f"  Dossier      : {SCRIPT_DIR}")
    print(f"  → http://{HOST}:{PORT}")
    # Bilan santé express — failproof, ne bloque jamais le démarrage
    hints = []
    if not shutil.which("ffmpeg"):
        hints.append("ffmpeg introuvable (incrustation/audio HS)")
    if not os.path.exists(PYTHON_BIN):
        hints.append(f"interpréteur interview absent ({PYTHON_BIN})")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        hints.append("ANTHROPIC_API_KEY absente (mets --llm local, ou configure la clé)")
    if hints:
        print("-" * 60)
        for h in hints:
            print("  ⚠️  " + h)
        print("  → diagnostic complet :  python3 doctor.py")
    print("=" * 60)
    app.run(host=HOST, port=PORT, threaded=True, debug=False)


if __name__ == "__main__":
    main()
