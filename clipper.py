#!/usr/bin/env python3
"""
Extraction intelligente de clips viraux (type Opus Clip)
=========================================================
Extrait jusqu'à N clips d'une longue vidéo podcast/interview,
sélectionnés par Claude selon un critère utilisateur, avec
sous-titres karaoke Instagram (ligne par ligne, gros texte sur fond).

Architecture en 6 passes :
  1. WhisperX          → transcription + timestamps mot par mot
  2. Claude (sélection) → choix des meilleurs passages selon --criteria
  3. Claude (traduction) → traduction optionnelle si --target-lang
  4. ffmpeg (extraction) → découpage des clips
  5. ASS karaoke        → sous-titres mot-par-mot groupés
  6. ffmpeg (burn)       → incrustation sous-titres

Usage :
  python clipper.py video.mp4 --criteria "passage le plus marquant"
  python clipper.py video.mp4 --criteria "moment drôle" --duration 180-600 -n 2
  python clipper.py video.mp4 --criteria "key insights" --target-lang fr
  python clipper.py "https://www.youtube.com/watch?v=XXXXX" --criteria "best moment"
  python clipper.py video.mp4 --resume video_clips.json --criteria "test"
"""

import argparse
import json
import fcntl
import os
import re
import shutil
import subprocess
import sys
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
CLAUDE_RETRY_MAX = 5
CLAUDE_RETRY_DELAY = 10.0

# Karaoke par défaut
DEFAULT_WORDS_PER_GROUP = 3

# Noms de langues (pour les prompts Claude)
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


def lang_name(code: str, in_english: bool = False) -> str:
    d = LANGUAGE_NAMES_EN if in_english else LANGUAGE_NAMES
    return d.get(code, code.upper())


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
    speaker: str = ""        # nom réel du locuteur (via transcription Epoch), vide sinon


@dataclass
class ClipPart:
    """Un morceau d'un clip montage (sous-plage de segments contigus)."""
    seg_start: int
    seg_end: int
    start: float
    end: float


@dataclass
class ClipSelection:
    clip_index: int
    seg_start: int          # indice du premier segment (ou premier du premier part)
    seg_end: int            # indice du dernier segment (ou dernier du dernier part)
    start: float            # timecode début (secondes)
    end: float              # timecode fin (secondes)
    titre: str = ""
    justification: str = ""
    segments: list = field(default_factory=list)  # liste de Segment
    parts: list = field(default_factory=list)     # liste de ClipPart (vide = clip continu)

    @property
    def is_montage(self) -> bool:
        return len(self.parts) >= 2


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt(sec: float) -> str:
    """Format SRT : HH:MM:SS,mmm"""
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    ms = int((sec % 1) * 1000)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"


def _fmt_ass(sec: float) -> str:
    """Format ASS : H:MM:SS.cc (centièmes)."""
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    cs = int((sec % 1) * 100)
    return f"{int(h)}:{int(m):02d}:{int(s):02d}.{cs:02d}"


# Ollama (LLM local — alternative gratuite à l'API Claude)
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3.6:27b"
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


def check_ffmpeg():
    """Vérifie que ffmpeg est disponible et que libass est compilé."""
    import shutil as _shutil
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if r.returncode != 0:
            print("❌ ffmpeg introuvable.")
            sys.exit(1)
        version_line = r.stdout.split('\n')[0]
        ffmpeg_path = _shutil.which("ffmpeg") or "ffmpeg"
        print(f"   ffmpeg : {ffmpeg_path} ({version_line.split(' ')[2] if len(version_line.split(' ')) > 2 else '?'})")
    except FileNotFoundError:
        print("❌ ffmpeg introuvable dans le PATH.")
        sys.exit(1)

    r = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True)
    if "subtitles" not in r.stdout:
        print(f"⚠️  ffmpeg sans filtre 'subtitles' (libass manquant). L'incrustation ne fonctionnera pas.")
        return False
    else:
        print(f"   libass : ✅")
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


def transcribe_whisperx(audio_path: str, source_lang: Optional[str] = None,
                        hf_token: Optional[str] = None) -> tuple[list, str]:
    """Transcrit l'audio. Retourne (segments, langue_détectée)."""
    acquire_gpu_lock()
    import whisperx, torch, gc

    device = hw.device()  # « cuda » couvre CUDA et ROCm/HIP
    lang_display = source_lang.upper() if source_lang else "auto"
    print(f"\n📝 Transcription WhisperX ({WHISPER_MODEL}) sur {device} [{lang_display}]...")

    t0 = time.time()
    model = whisperx.load_model(WHISPER_MODEL, device,
                                compute_type=hw.whisper_compute_type(),
                                language=source_lang)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=WHISPER_BATCH_SIZE,
                              language=source_lang)

    # Récupérer la langue détectée (utile si source_lang=None)
    detected_lang = result.get("language", source_lang or "en")
    if source_lang is None:
        print(f"   🌐 Langue détectée : {detected_lang}")
    print(f"   Transcription : {time.time()-t0:.1f}s")

    print("   🔧 Alignement mot par mot...")
    t1 = time.time()
    model_a, metadata = whisperx.load_align_model(language_code=detected_lang, device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device,
                            return_char_alignments=False)
    print(f"   Alignement : {time.time()-t1:.1f}s")

    del model, model_a; gc.collect()
    if device == "cuda": torch.cuda.empty_cache()

    segments = [
        Segment(index=i+1, start=s["start"], end=s["end"],
                text=s["text"].strip(), words=s.get("words", []))
        for i, s in enumerate(result["segments"])
    ]
    dur = segments[-1].end if segments else 0
    print(f"   ✅ {len(segments)} segments ({dur/60:.1f} min)")
    return segments, detected_lang


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT : SAUVEGARDE / CHARGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(segments: list[Segment], clips: list[ClipSelection], path: str):
    """Sauvegarde complète : segments + clips sélectionnés."""
    data = {
        "segments": [
            {"index": s.index, "start": s.start, "end": s.end,
             "text": s.text, "text_tgt": s.text_tgt,
             "words": s.words, "speaker": s.speaker}
            for s in segments
        ],
        "clips": [
            {"clip_index": c.clip_index, "seg_start": c.seg_start, "seg_end": c.seg_end,
             "start": c.start, "end": c.end, "titre": c.titre,
             "justification": c.justification,
             "parts": [{"seg_start": p.seg_start, "seg_end": p.seg_end,
                         "start": p.start, "end": p.end} for p in c.parts]}
            for c in clips
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"   💾 → {path}")


def load_checkpoint(path: str) -> tuple[list[Segment], list[ClipSelection]]:
    """Charge segments + clips depuis le checkpoint."""
    with open(path) as f:
        data = json.load(f)

    segments = [
        Segment(d["index"], d["start"], d["end"], d["text"],
                d.get("text_tgt", ""), d.get("words", []),
                d.get("speaker", ""))
        for d in data["segments"]
    ]

    clips = []
    for c in data.get("clips", []):
        parts = [ClipPart(p["seg_start"], p["seg_end"], p["start"], p["end"])
                 for p in c.get("parts", [])]
        clips.append(ClipSelection(
            clip_index=c["clip_index"], seg_start=c["seg_start"], seg_end=c["seg_end"],
            start=c["start"], end=c["end"], titre=c.get("titre", ""),
            justification=c.get("justification", ""), parts=parts))

    return segments, clips


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 2 : SÉLECTION DES CLIPS PAR CLAUDE
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_speaker_names(segments: list[Segment], name_map: dict) -> int:
    """Remplace les ID locuteurs (SPEAKER_xx) par les noms réels via name_map.

    name_map : {"SPEAKER_00": "Jan Jekielek", ...} renvoyé par align_transcript.
    Les segments non attribués (ou inconnus de name_map) reçoivent "".
    Retourne le nombre de segments nommés.
    """
    named = 0
    for s in segments:
        sid = getattr(s, "speaker", "") or ""
        if sid.startswith("SPEAKER_"):
            s.speaker = name_map.get(sid, "")
        if s.speaker:
            named += 1
    return named


def build_transcript(segments: list[Segment]) -> str:
    """Transcription indexée + timecodes, préfixée du nom du locuteur si connu."""
    lines = []
    for s in segments:
        tc = f"{_fmt(s.start)} → {_fmt(s.end)}"
        spk = f"{s.speaker} : " if getattr(s, "speaker", "") else ""
        lines.append(f"[{s.index}] ({tc}) {spk}{s.text}")
    return "\n".join(lines)


def select_clips(segments: list[Segment], client, criteria: str,
                 max_clips: int = 3, duration_min: float = 30, duration_max: float = 90,
                 source_lang: str = "en", context: str = "") -> list[ClipSelection]:
    """Claude sélectionne les meilleurs passages selon le critère utilisateur."""
    print(f"\n🎯 Sélection des clips (critère : {criteria[:80]}{'...' if len(criteria) > 80 else ''})...")

    src_name = lang_name(source_lang)

    # Construire la transcription avec indices, timecodes et locuteurs
    transcript = build_transcript(segments)

    # Tronquer si trop long (garder sous ~100k caractères pour le contexte Claude)
    if len(transcript) > 120000:
        transcript = transcript[:120000] + "\n[... transcription tronquée ...]"

    speaker_hint = ""
    if any(getattr(s, "speaker", "") for s in segments):
        speaker_hint = ("\n- Chaque ligne est préfixée du nom du locuteur « Nom : » — "
                        "sers-t'en pour repérer qui dit quoi, ne pas couper une réponse "
                        "de sa question, et nommer l'intervenant dans le titre/la "
                        "justification quand c'est pertinent")

    ctx_block = ""
    if context:
        ctx_block = f"\nCONTEXTE FOURNI PAR L'UTILISATEUR :\n{context}\n"

    montage_block = ""
    if max_clips >= 2:
        montage_block = f"""

CLIP MONTAGE (OBLIGATOIRE) :
- Exactement UN des {max_clips} clips doit être un « montage » : un assemblage de 2 à 5 morceaux
  NON-CONTIGUS de la vidéo, collés ensemble pour former un clip cohérent et percutant.
- Le montage doit raconter une histoire fluide ou construire un argument convaincant
  en sélectionnant les meilleurs moments éparpillés dans la vidéo et en coupant les creux.
- Les morceaux doivent s'enchaîner de façon propre : chaque coupure doit tomber sur une fin
  de phrase, jamais au milieu d'un mot ou d'une idée.
- La durée TOTALE du montage (somme des morceaux) doit rester entre {duration_min:.0f}s et {duration_max:.0f}s.
- Pour ce clip montage, utilise le champ "parts" (tableau de morceaux) au lieu de seg_start/seg_end simples.
- Les autres clips restent des clips continus classiques (seg_start/seg_end, sans "parts")."""

    prompt = f"""Tu es un expert en montage vidéo virale et réseaux sociaux.

TRANSCRIPTION COMPLÈTE ({src_name}, {len(segments)} segments) :
{transcript}
{ctx_block}
CRITÈRE DE SÉLECTION : {criteria}

CONSIGNES :
- Sélectionne les {max_clips} meilleur(s) passage(s) correspondant au critère
- Chaque clip doit durer entre {duration_min:.0f}s et {duration_max:.0f}s
- Indique les indices de segments (seg_start et seg_end inclus)
- Privilégie :
  • Accroche forte dès les premières secondes (hook)
  • Contenu dense et percutant, pas de creux
  • Fin propre (phrase complète, pas coupée au milieu)
  • Autonomie : le clip doit être compréhensible hors contexte
- Évite les passages avec trop de « euh », hésitations, ou digressions
- Chaque clip doit avoir un titre court et accrocheur (pour Instagram/TikTok){speaker_hint}
{montage_block}

Réponds UNIQUEMENT en JSON strict (sans markdown) :
{{
  "clips": [
    {{
      "seg_start": <indice premier segment>,
      "seg_end": <indice dernier segment>,
      "titre": "<titre court et accrocheur>",
      "justification": "<pourquoi ce passage est pertinent>"
    }},
    {{
      "parts": [
        {{"seg_start": <indice>, "seg_end": <indice>}},
        {{"seg_start": <indice>, "seg_end": <indice>}}
      ],
      "titre": "<titre du montage>",
      "justification": "<pourquoi ce montage fonctionne>"
    }}
  ]
}}"""

    resp = _claude_create(client, model=CLAUDE_MODEL, max_tokens=CLAUDE_MAX_TOKENS,
                          messages=[{"role": "user", "content": prompt}])
    txt = resp.content[0].text

    # Parser le JSON
    jm = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', txt, re.DOTALL)
    if jm:
        txt = jm.group(1)
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        bs, be = txt.find('{'), txt.rfind('}') + 1
        try:
            data = json.loads(txt[bs:be]) if bs >= 0 else {"clips": []}
        except json.JSONDecodeError:
            print(f"   ❌ Impossible de parser la réponse Claude")
            print(f"   Réponse brute : {txt[:500]}")
            return []

    # Construire les ClipSelection
    seg_by_idx = {s.index: s for s in segments}
    clips = []
    for i, c in enumerate(data.get("clips", [])[:max_clips]):
        raw_parts = c.get("parts", [])

        if raw_parts and len(raw_parts) >= 2:
            # ── Clip montage (multi-parts) ──
            parts = []
            clip_segs = []
            valid = True
            for p in raw_parts:
                ps, pe = p.get("seg_start", 0), p.get("seg_end", 0)
                pfirst, plast = seg_by_idx.get(ps), seg_by_idx.get(pe)
                if not pfirst or not plast:
                    print(f"   ⚠️  Clip {i+1} montage : part {ps}-{pe} introuvable, ignorée")
                    valid = False
                    break
                parts.append(ClipPart(seg_start=ps, seg_end=pe,
                                      start=pfirst.start, end=plast.end))
                clip_segs.extend(s for s in segments if ps <= s.index <= pe)
            if not valid or not parts:
                continue

            clip = ClipSelection(
                clip_index=i + 1,
                seg_start=parts[0].seg_start,
                seg_end=parts[-1].seg_end,
                start=parts[0].start,
                end=parts[-1].end,
                titre=c.get("titre", f"Clip {i+1}"),
                justification=c.get("justification", ""),
                segments=clip_segs,
                parts=parts,
            )

            total_dur = sum(p.end - p.start for p in parts)
            parts_desc = " + ".join(f"{_fmt(p.start)}→{_fmt(p.end)}" for p in parts)
            print(f"   🎬 Clip {clip.clip_index} [MONTAGE {len(parts)} parts] ({total_dur:.1f}s)")
            print(f"      Parts : {parts_desc}")
            print(f"      Titre : {clip.titre}")
            print(f"      Motif : {clip.justification[:100]}")
        else:
            # ── Clip continu classique ──
            seg_s = c.get("seg_start", 0)
            seg_e = c.get("seg_end", 0)

            first = seg_by_idx.get(seg_s)
            last = seg_by_idx.get(seg_e)
            if not first or not last:
                print(f"   ⚠️  Clip {i+1} : segments {seg_s}-{seg_e} introuvables, ignoré")
                continue

            clip_segs = [s for s in segments if seg_s <= s.index <= seg_e]

            clip = ClipSelection(
                clip_index=i + 1,
                seg_start=seg_s,
                seg_end=seg_e,
                start=first.start,
                end=last.end,
                titre=c.get("titre", f"Clip {i+1}"),
                justification=c.get("justification", ""),
                segments=clip_segs,
            )

            duration = clip.end - clip.start
            print(f"   🎬 Clip {clip.clip_index} : [{_fmt(clip.start)} → {_fmt(clip.end)}] ({duration:.1f}s)")
            print(f"      Titre : {clip.titre}")
            print(f"      Motif : {clip.justification[:100]}")

        clips.append(clip)

    # ── Filtrer les clips hors fourchette de durée ──
    filtered = []
    for clip in clips:
        if clip.parts:
            dur = sum(p.end - p.start for p in clip.parts)
        else:
            dur = clip.end - clip.start
        if dur < duration_min:
            print(f"   ⚠️  Clip {clip.clip_index} rejeté : {dur:.0f}s < minimum {duration_min:.0f}s")
        elif dur > duration_max:
            print(f"   ⚠️  Clip {clip.clip_index} rejeté : {dur:.0f}s > maximum {duration_max:.0f}s")
        else:
            filtered.append(clip)
    clips = filtered

    # Renuméroter les clips restants
    for i, clip in enumerate(clips):
        clip.clip_index = i + 1

    if not clips:
        print("   ❌ Aucun clip sélectionné (tous hors fourchette de durée) !")
    else:
        print(f"\n   ✅ {len(clips)} clip(s) sélectionné(s)")

    return clips


def select_educational_clip(segments: list[Segment], client,
                            existing_clips: list[ClipSelection],
                            duration_min: float = 30, duration_max: float = 90,
                            source_lang: str = "en", context: str = "") -> ClipSelection | None:
    """2ème appel Claude : sélectionne LE passage à densité informationnelle maximale (peut chevaucher)."""
    print(f"\n📚 Sélection du clip éducatif bonus (densité informationnelle maximale)...")

    src_name = lang_name(source_lang)

    # Construire la transcription avec indices, timecodes et locuteurs
    transcript = build_transcript(segments)

    if len(transcript) > 120000:
        transcript = transcript[:120000] + "\n[... transcription tronquée ...]"

    # Résumé des clips déjà sélectionnés
    existing_desc = "\n".join(
        f"  - Clip {c.clip_index} « {c.titre} » : {_fmt(c.start)} → {_fmt(c.end)}"
        for c in existing_clips
    )

    ctx_block = ""
    if context:
        ctx_block = f"\nCONTEXTE FOURNI PAR L'UTILISATEUR :\n{context}\n"

    prompt = f"""Tu es un expert en contenu éducatif et vulgarisation.

TRANSCRIPTION COMPLÈTE ({src_name}, {len(segments)} segments) :
{transcript}
{ctx_block}
CLIPS DÉJÀ SÉLECTIONNÉS :
{existing_desc}

MISSION : Sélectionne LE passage où le spectateur apprend le PLUS sur le sujet en un MINIMUM de temps.
Densité informationnelle maximale : chaque seconde doit apporter une information nouvelle, un fait,
une explication ou un insight. Pas de bavardage, pas d'hésitation, pas de répétition.

CONTRAINTES :
- Le clip doit durer entre {duration_min:.0f}s et {duration_max:.0f}s
- Ce clip PEUT chevaucher les clips déjà sélectionnés ci-dessus (c'est autorisé)
- Le passage doit être compréhensible hors contexte
- Fin propre (phrase complète)
- Privilégie les passages qui contiennent des faits, chiffres, mécanismes ou explications concrètes

Réponds UNIQUEMENT en JSON strict (sans markdown) :
{{
  "seg_start": <indice premier segment>,
  "seg_end": <indice dernier segment>,
  "titre": "<titre court décrivant ce qu'on apprend>",
  "justification": "<pourquoi ce passage a la plus haute densité informationnelle>"
}}"""

    resp = _claude_create(client, model=CLAUDE_MODEL, max_tokens=2048,
                          messages=[{"role": "user", "content": prompt}])
    txt = resp.content[0].text

    # Parser le JSON
    jm = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', txt, re.DOTALL)
    if jm:
        txt = jm.group(1)
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        bs, be = txt.find('{'), txt.rfind('}') + 1
        try:
            data = json.loads(txt[bs:be]) if bs >= 0 else {}
        except json.JSONDecodeError:
            print(f"   ❌ Impossible de parser la réponse Claude pour le clip éducatif")
            return None

    seg_s = data.get("seg_start", 0)
    seg_e = data.get("seg_end", 0)
    seg_by_idx = {s.index: s for s in segments}
    first = seg_by_idx.get(seg_s)
    last = seg_by_idx.get(seg_e)
    if not first or not last:
        print(f"   ⚠️  Clip éducatif : segments {seg_s}-{seg_e} introuvables, ignoré")
        return None

    clip_segs = [s for s in segments if seg_s <= s.index <= seg_e]
    clip = ClipSelection(
        clip_index=len(existing_clips) + 1,
        seg_start=seg_s,
        seg_end=seg_e,
        start=first.start,
        end=last.end,
        titre="📚 " + data.get("titre", "Clip éducatif"),
        justification=data.get("justification", ""),
        segments=clip_segs,
    )

    duration = clip.end - clip.start
    if duration < duration_min or duration > duration_max:
        print(f"   ⚠️  Clip éducatif rejeté : {duration:.0f}s hors fourchette [{duration_min:.0f}s-{duration_max:.0f}s]")
        return None

    print(f"   📚 Clip {clip.clip_index} [ÉDUCATIF] : [{_fmt(clip.start)} → {_fmt(clip.end)}] ({duration:.1f}s)")
    print(f"      Titre : {clip.titre}")
    print(f"      Motif : {clip.justification[:100]}")

    return clip


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 3 : TRADUCTION OPTIONNELLE
# ═══════════════════════════════════════════════════════════════════════════════

def translate_clips(clips: list[ClipSelection], client,
                    source_lang: str, target_lang: str,
                    context: str = "") -> list[ClipSelection]:
    """Traduit les segments de chaque clip (simple, pas de chunks — clips courts)."""
    src_name = lang_name(source_lang)
    tgt_name = lang_name(target_lang)
    print(f"\n🌍 Traduction {source_lang}→{target_lang} des clips...")

    for clip in clips:
        # Vérifier si déjà traduit (segments sans texte source comptent comme OK)
        if all(s.text_tgt or not s.text.strip() for s in clip.segments):
            print(f"   📦 Clip {clip.clip_index} — déjà traduit")
            continue

        # Ne traduire que les segments qui en ont besoin
        to_translate = [s for s in clip.segments if not s.text_tgt and s.text.strip()]
        print(f"   📦 Clip {clip.clip_index} ({len(to_translate)} segments à traduire"
              f" / {len(clip.segments)} total)...")

        lines = "\n".join(f"[{s.index}] {s.text}" for s in to_translate)

        # Fournir les traductions existantes comme contexte
        ctx_parts = []
        if context:
            ctx_parts.append(f"CONTEXTE : {context}")
        already = [s for s in clip.segments if s.text_tgt]
        if already:
            ctx_parts.append("TRADUCTIONS EXISTANTES (ne pas retraduire) :")
            ctx_parts.extend(f"[{s.index}] {tgt_name}: {s.text_tgt}" for s in already)
        ctx_block = "\n".join(ctx_parts) + "\n" if ctx_parts else ""

        prompt = f"""Tu es un traducteur professionnel {src_name} → {tgt_name} pour sous-titres de clips viraux.
{ctx_block}
Traduis chaque segment en {tgt_name}. Garde le même numéro d'index.
La traduction doit être naturelle, percutante et adaptée aux réseaux sociaux.

SEGMENTS À TRADUIRE :
{lines}

Réponds UNIQUEMENT avec les traductions, une par ligne, format :
[numéro] traduction"""

        resp = _claude_create(client, model=CLAUDE_MODEL, max_tokens=CLAUDE_MAX_TOKENS,
                              messages=[{"role": "user", "content": prompt}])

        # Parser les traductions — ne pas écraser les traductions existantes
        seg_by_idx = {s.index: s for s in clip.segments}
        for line in resp.content[0].text.strip().split("\n"):
            m = re.match(r'\[(\d+)\]\s*(.*)', line.strip())
            if m:
                idx, txt = int(m.group(1)), m.group(2).strip()
                txt = re.sub(r'^[A-Z]{2}:\s*', '', txt).strip()
                if txt and idx in seg_by_idx and not seg_by_idx[idx].text_tgt:
                    seg_by_idx[idx].text_tgt = txt

        done = sum(1 for s in clip.segments if s.text_tgt or not s.text.strip())
        print(f"   ✅ Clip {clip.clip_index} : {done}/{len(clip.segments)} traduits")

    return clips


# ═══════════════════════════════════════════════════════════════════════════════
# GÉNÉRATION DE POST SOCIAL
# ═══════════════════════════════════════════════════════════════════════════════

def generate_social_post(clip: ClipSelection, client,
                         speaker: str = None, url: str = None,
                         target_lang: str = None, date: str = None) -> str:
    """Génère un post viral prêt à copier-coller pour les réseaux sociaux."""
    # Construire le texte du clip (traduction si dispo, sinon original)
    lines = []
    for s in clip.segments:
        text = s.text_tgt if (target_lang and s.text_tgt) else s.text
        lines.append(text)
    transcript = "\n".join(lines)

    speaker_block = f"\nLOCUTEUR : {speaker}" if speaker else ""
    url_block = f"\nLIEN SOURCE : {url}" if url else ""
    date_block = f"\nDATE DE LA DÉCLARATION : {date}" if date else ""

    prompt = f"""Tu es un expert en rédaction de posts pour les réseaux sociaux.

TITRE DU CLIP : {clip.titre}
{speaker_block}{date_block}{url_block}

TRANSCRIPTION DU CLIP :
{transcript}

Écris un post pour les réseaux sociaux en français, composé UNIQUEMENT de citations.

RÈGLES STRICTES :
- Le post ne contient QUE des citations littérales entre guillemets français « »
- Chaque citation DOIT apparaître MOT POUR MOT dans la transcription ci-dessus (pas de reformulation, pas d'invention, pas de résumé)
- Tu peux sélectionner 1 à 3 citations percutantes parmi la transcription
- Tu peux couper une phrase longue avec [...] mais JAMAIS changer les mots
- Après les citations, ajoute l'attribution sur une nouvelle ligne :
  • Si un locuteur ET une date sont fournis : — Prénom Nom, date
  • Si seulement un locuteur : — Prénom Nom
  • Si seulement une date : — date
  • Si ni l'un ni l'autre : pas de ligne d'attribution
- Si un lien source est fourni, place-le seul sur la dernière ligne
- AUCUN commentaire, aucune phrase de ton cru, aucune reformulation, aucune introduction
- Pas de hashtags, pas d'emojis
- Prêt à copier-coller, rien d'autre

Réponds UNIQUEMENT avec le texte du post."""

    try:
        resp = _claude_create(client, model=CLAUDE_MODEL, max_tokens=1024,
                              messages=[{"role": "user", "content": prompt}])
        return resp.content[0].text.strip()
    except Exception as e:
        print(f"   ⚠️  Erreur génération post : {e}")
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 4 : EXTRACTION DES CLIPS (ffmpeg)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_clip(video_path: str, output_path: str, start: float, end: float) -> str:
    """Extrait un clip de la vidéo avec re-encode."""
    duration = end - start
    print(f"   ✂️  Extraction {_fmt(start)} → {_fmt(end)} ({duration:.1f}s)...")

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-crf", "18", "-preset", "slow",
        "-c:a", "aac", "-b:a", "192k", "-ac", "2",
        "-movflags", "+faststart",
        output_path
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"   ❌ ffmpeg extraction : {r.stderr[-400:]}")
        return ""

    mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"   ✅ {output_path} ({mb:.1f} Mo)")
    return output_path


def extract_montage_clip(video_path: str, output_path: str, parts: list) -> str:
    """Extrait et concatène plusieurs morceaux non-contigus en un seul clip."""
    import tempfile
    tmp_dir = tempfile.mkdtemp()
    FADE_DUR = 0.5  # durée du fondu entre morceaux (secondes)

    try:
        part_files = []
        for j, part in enumerate(parts):
            part_path = os.path.join(tmp_dir, f"part_{j:03d}.mp4")
            dur = part.end - part.start
            print(f"   ✂️  Morceau {j+1}/{len(parts)} : {_fmt(part.start)} → {_fmt(part.end)} ({dur:.1f}s)")

            # Fondus entre morceaux (pas au tout début ni à la toute fin du montage)
            vfilters, afilters = [], []
            if j > 0:
                vfilters.append(f"fade=t=in:st=0:d={FADE_DUR}")
                afilters.append(f"afade=t=in:st=0:d={FADE_DUR}")
            if j < len(parts) - 1:
                fade_start = max(0, dur - FADE_DUR)
                vfilters.append(f"fade=t=out:st={fade_start:.3f}:d={FADE_DUR}")
                afilters.append(f"afade=t=out:st={fade_start:.3f}:d={FADE_DUR}")

            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{part.start:.3f}",
                "-i", video_path,
                "-t", f"{dur:.3f}",
            ]
            if vfilters:
                cmd += ["-vf", ",".join(vfilters)]
            if afilters:
                cmd += ["-af", ",".join(afilters)]
            cmd += [
                "-c:v", "libx264", "-crf", "18", "-preset", "slow",
                "-c:a", "aac", "-b:a", "192k",
                part_path
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"   ❌ ffmpeg extraction morceau {j+1} : {r.stderr[-400:]}")
                return ""
            part_files.append(part_path)

        # Concaténation via ffmpeg concat demuxer
        concat_list = os.path.join(tmp_dir, "concat.txt")
        with open(concat_list, "w") as f:
            for pf in part_files:
                f.write(f"file '{pf}'\n")

        print(f"   🔗 Assemblage de {len(parts)} morceaux...")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c:v", "libx264", "-crf", "18", "-preset", "slow",
            "-c:a", "aac", "-b:a", "192k", "-ac", "2",
            "-movflags", "+faststart",
            output_path
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"   ❌ ffmpeg concat : {r.stderr[-400:]}")
            return ""

        mb = os.path.getsize(output_path) / (1024 * 1024)
        total_dur = sum(p.end - p.start for p in parts)
        print(f"   ✅ {output_path} ({mb:.1f} Mo, {total_dur:.1f}s)")
        return output_path
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 5 : SOUS-TITRES ASS KARAOKE
# ═══════════════════════════════════════════════════════════════════════════════

def get_video_resolution(video_path: str) -> tuple:
    """Récupère la résolution via ffprobe."""
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
        return 1920, 1080


def is_16_9(w: int, h: int) -> bool:
    """Vérifie si la résolution est au format 16:9 (tolérance 3%)."""
    if h == 0:
        return False
    ratio = w / h
    return abs(ratio - 16 / 9) < 0.05


def interpolate_word_times(words: list, seg_start: float, seg_end: float) -> list:
    """Interpole les timestamps manquants pour les mots sans timing WhisperX."""
    if not words:
        return []

    result = []
    for w in words:
        result.append({
            "word": w.get("word", ""),
            "start": w.get("start"),
            "end": w.get("end"),
        })

    # Passe 1 : remplir les bornes manquantes par interpolation linéaire
    # D'abord, assigner la borne du segment aux extrêmes
    if result[0]["start"] is None:
        result[0]["start"] = seg_start
    if result[-1]["end"] is None:
        result[-1]["end"] = seg_end

    # Interpoler les manquants : trouver les runs de None et répartir
    n = len(result)
    for i in range(n):
        if result[i]["start"] is None:
            # Trouver la borne précédente connue
            prev_end = seg_start
            for j in range(i - 1, -1, -1):
                if result[j]["end"] is not None:
                    prev_end = result[j]["end"]
                    break
            # Trouver la prochaine borne connue
            next_start = seg_end
            for j in range(i, n):
                if result[j]["start"] is not None:
                    next_start = result[j]["start"]
                    break

            # Compter combien de mots à interpoler dans ce run
            run_start = i
            run_end = i
            while run_end < n and result[run_end]["start"] is None:
                run_end += 1

            span = next_start - prev_end
            count = run_end - run_start
            for k in range(count):
                word_start = prev_end + span * k / count
                word_end = prev_end + span * (k + 1) / count
                result[run_start + k]["start"] = word_start
                result[run_start + k]["end"] = word_end

    # Passe 2 : combler les end manquants
    for i in range(n):
        if result[i]["end"] is None:
            if i + 1 < n and result[i + 1]["start"] is not None:
                result[i]["end"] = result[i + 1]["start"]
            else:
                result[i]["end"] = result[i]["start"] + 0.1

    return result


def distribute_words_uniformly(text: str, seg_start: float, seg_end: float) -> list:
    """Distribue uniformément les mots d'un texte traduit sur la durée du segment."""
    words_list = text.split()
    if not words_list:
        return []
    duration = seg_end - seg_start
    word_dur = duration / len(words_list)
    result = []
    for i, w in enumerate(words_list):
        result.append({
            "word": w,
            "start": seg_start + i * word_dur,
            "end": seg_start + (i + 1) * word_dur,
        })
    return result


def generate_karaoke_ass(clip: ClipSelection, ass_path: str,
                         clip_start_offset: float,
                         words_per_group: int = 3,
                         use_translation: bool = False,
                         video_width: int = 1920, video_height: int = 1080):
    """
    Génère un fichier ASS avec sous-titres karaoke Instagram.
    Chaque groupe de ~N mots s'affiche l'un après l'autre (remplace le précédent).
    """
    print(f"   📝 Génération ASS karaoke : {ass_path}")

    # Collecter tous les mots avec timing, ajustés au début du clip
    all_words = []

    if clip.is_montage:
        # Montage : chaque part est collée bout à bout dans la vidéo concaténée,
        # donc on rebase les timecodes de chaque part sur le temps cumulé.
        cumulative_offset = 0.0
        for part in clip.parts:
            part_segs = [s for s in clip.segments
                         if part.seg_start <= s.index <= part.seg_end]
            for seg in part_segs:
                text_to_use = seg.text_tgt if (use_translation and seg.text_tgt) else seg.text
                if use_translation and seg.text_tgt:
                    words = distribute_words_uniformly(text_to_use, seg.start, seg.end)
                elif seg.words:
                    words = interpolate_word_times(seg.words, seg.start, seg.end)
                else:
                    words = distribute_words_uniformly(text_to_use, seg.start, seg.end)
                for w in words:
                    all_words.append({
                        "word": w["word"],
                        "start": w["start"] - part.start + cumulative_offset,
                        "end": w["end"] - part.start + cumulative_offset,
                    })
            cumulative_offset += part.end - part.start
    else:
        for seg in clip.segments:
            text_to_use = seg.text_tgt if (use_translation and seg.text_tgt) else seg.text

            if use_translation and seg.text_tgt:
                words = distribute_words_uniformly(text_to_use, seg.start, seg.end)
            elif seg.words:
                words = interpolate_word_times(seg.words, seg.start, seg.end)
            else:
                words = distribute_words_uniformly(text_to_use, seg.start, seg.end)

            for w in words:
                all_words.append({
                    "word": w["word"],
                    "start": w["start"] - clip_start_offset,
                    "end": w["end"] - clip_start_offset,
                })

    if not all_words:
        print(f"   ⚠️  Aucun mot trouvé pour le clip {clip.clip_index}")
        return

    # Grouper par paquets de N mots
    groups = []
    for i in range(0, len(all_words), words_per_group):
        group = all_words[i:i + words_per_group]
        text = " ".join(w["word"] for w in group)
        start = group[0]["start"]
        end = group[-1]["end"]
        # Garder un petit overlap pour éviter les flashs
        groups.append({"text": text, "start": max(0, start), "end": max(0, end)})

    # Taille de police proportionnelle à la résolution (style karaoke social media)
    # 10% de la plus petite dimension = gros texte lisible sur mobile
    font_size = max(64, int(min(video_width, video_height) * 0.10))
    margin_v = max(50, int(video_height * 0.07))
    margin_lr = max(30, int(video_width * 0.05))

    # Formats plus étroits que 16:9 (carré, vertical) : réduction légère uniquement.
    # Les groupes karaoke sont courts (~3 mots), pas besoin de réduire autant
    # que pour des sous-titres de 42 caractères.
    if video_width > 0 and video_height > 0:
        aspect = video_width / video_height
        if aspect < 16 / 9 - 0.05:
            width_ratio = video_width / (video_height * 16 / 9)
            # Atténuer : racine carrée du ratio au lieu du ratio brut
            # 9:16 → ratio 0.32 → sqrt = 0.56 (au lieu de 0.32)
            soft_ratio = width_ratio ** 0.5
            font_size = max(56, round(font_size * soft_ratio))
            margin_v = max(30, round(margin_v * soft_ratio))
            margin_lr = max(15, round(margin_lr * width_ratio))

    # Écrire le fichier ASS
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("[Script Info]\n")
        f.write("Title: Karaoke Clip\n")
        f.write("ScriptType: v4.00+\n")
        f.write(f"PlayResX: {video_width}\n")
        f.write(f"PlayResY: {video_height}\n")
        f.write("WrapStyle: 0\n")
        f.write("ScaledBorderAndShadow: yes\n")
        f.write("\n")

        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
                "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
                "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                "Alignment, MarginL, MarginR, MarginV, Encoding\n")
        # Style média social : texte blanc gras, contour noir épais, ombre portée
        # BorderStyle=1 = outline (pas de boîte opaque)
        # Outline=3.5, Shadow=1.5 pour lisibilité sur tout fond
        # BackColour semi-transparent pour l'ombre portée
        # Alignment=2 = centré en bas
        outline = max(2, font_size * 0.05)
        shadow = max(1, font_size * 0.02)
        f.write(f"Style: Karaoke,Arial Black,{font_size},&H00FFFFFF,&H0000FFFF,"
                f"&H00000000,&HA0000000,-1,0,0,0,"
                f"100,100,0,0,1,{outline:.1f},{shadow:.1f},"
                f"2,{margin_lr},{margin_lr},{margin_v},1\n")
        f.write("\n")

        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

        for g in groups:
            start_tc = _fmt_ass(g["start"])
            end_tc = _fmt_ass(g["end"])
            # Échapper les caractères spéciaux ASS
            text = g["text"].replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
            f.write(f"Dialogue: 0,{start_tc},{end_tc},Karaoke,,0,0,0,,{text}\n")

    print(f"   ✅ {len(groups)} groupes karaoke → {ass_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 6 : INCRUSTATION (ffmpeg burn)
# ═══════════════════════════════════════════════════════════════════════════════

def burn_subtitles(video_path: str, ass_path: str, output_path: str,
                   crop_filter: str = None) -> str:
    """Incruste le fichier ASS dans la vidéo.
    Si crop_filter est fourni (ex: 'crop=1080:1080:420:0'), applique le recadrage avant l'ASS."""
    label = "1:1" if crop_filter else "16:9" if "16x9" in output_path else ""
    print(f"   🔥 Incrustation sous-titres{f' ({label})' if label else ''}...")
    import shutil, tempfile

    # Copier l'ASS dans un tmp (évite les problèmes de chemin avec ffmpeg)
    tmp_dir = tempfile.mkdtemp()
    tmp_ass = os.path.join(tmp_dir, "subs.ass")
    shutil.copy2(ass_path, tmp_ass)

    try:
        vf = f"{crop_filter},ass={tmp_ass}" if crop_filter else f"ass={tmp_ass}"
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", vf,
            "-c:v", "libx264", "-crf", "18", "-preset", "slow",
            "-c:a", "aac", "-b:a", "192k", "-ac", "2",
            "-movflags", "+faststart",
            output_path
        ]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"   ❌ ffmpeg burn : {r.stderr[-400:]}")
            print(f"   💡 Le fichier .ass reste disponible pour VLC")
            return ""

        mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   ✅ {output_path} ({mb:.1f} Mo, {time.time()-t0:.0f}s)")
        return output_path
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE PUBLICATION INTERACTIVE (--post)
# ═══════════════════════════════════════════════════════════════════════════════

def find_postable_clips(directory: str = ".") -> list[tuple[str, str]]:
    """Trouve les paires MP4+TXT publiables dans le répertoire.
    Exclut les fichiers *_raw.mp4 (intermédiaires).
    Retourne [(mp4_path, txt_path), ...] triés par nom."""
    pairs = []
    for f in sorted(os.listdir(directory)):
        if not f.endswith(".mp4") or f.endswith("_raw.mp4"):
            continue
        stem = f[:-4]
        txt = os.path.join(directory, stem + ".txt")
        if os.path.exists(txt):
            pairs.append((os.path.join(directory, f), txt))
    return pairs


def post_to_telegram(video_path: str, caption: str, token: str, chat_id: str) -> bool:
    """Envoie une vidéo sur Telegram via l'API Bot (sendVideo)."""
    import requests

    size_mb = os.path.getsize(video_path) / (1024 * 1024)
    if size_mb > 50:
        print(f"   ⚠️  Fichier trop gros pour Telegram ({size_mb:.1f} Mo > 50 Mo)")
        return False

    url = f"https://api.telegram.org/bot{token}/sendVideo"
    # Tronquer la caption à 1024 caractères (limite Telegram)
    caption_tg = caption[:1024]
    try:
        with open(video_path, "rb") as vf:
            resp = requests.post(url, data={"chat_id": chat_id, "caption": caption_tg,
                                            "parse_mode": "HTML"},
                                 files={"video": vf}, timeout=120)
        if resp.status_code == 200 and resp.json().get("ok"):
            print(f"   ✅ Telegram : envoyé")
            return True
        else:
            err = resp.json().get("description", resp.text[:200])
            print(f"   ❌ Telegram : {err}")
            return False
    except Exception as e:
        print(f"   ❌ Telegram : {e}")
        return False


def post_to_twitter(video_path: str, tweet_text: str,
                    api_key: str, api_secret: str,
                    access_token: str, access_secret: str) -> bool:
    """Publie une vidéo sur X/Twitter (upload chunked v1.1 + tweet v2)."""
    try:
        import tweepy
        from requests_oauthlib import OAuth1
    except ImportError:
        print("   ❌ X/Twitter : dépendances manquantes")
        print("      → pip install tweepy requests-oauthlib --break-system-packages")
        return False

    import requests, math

    size = os.path.getsize(video_path)
    size_mb = size / (1024 * 1024)

    # Authentification OAuth1 pour l'upload media (API v1.1)
    auth = OAuth1(api_key, api_secret, access_token, access_secret)
    upload_url = "https://upload.twitter.com/1.1/media/upload.json"

    try:
        # INIT
        init_data = {
            "command": "INIT",
            "total_bytes": size,
            "media_type": "video/mp4",
            "media_category": "tweet_video",
        }
        resp = requests.post(upload_url, data=init_data, auth=auth, timeout=30)
        if resp.status_code != 202 and resp.status_code != 200:
            print(f"   ❌ X upload INIT : {resp.status_code} {resp.text[:200]}")
            return False
        media_id = resp.json()["media_id_string"]

        # APPEND (chunks de 1 Mo)
        chunk_size = 1024 * 1024
        with open(video_path, "rb") as vf:
            segment = 0
            while True:
                chunk = vf.read(chunk_size)
                if not chunk:
                    break
                append_data = {"command": "APPEND", "media_id": media_id,
                               "segment_index": segment}
                resp = requests.post(upload_url, data=append_data,
                                     files={"media_data": chunk}, auth=auth, timeout=60)
                if resp.status_code not in (200, 204):
                    print(f"   ❌ X upload APPEND seg {segment} : {resp.status_code}")
                    return False
                segment += 1

        # FINALIZE
        resp = requests.post(upload_url,
                             data={"command": "FINALIZE", "media_id": media_id},
                             auth=auth, timeout=30)
        if resp.status_code not in (200, 201):
            print(f"   ❌ X upload FINALIZE : {resp.status_code} {resp.text[:200]}")
            return False

        result = resp.json()

        # STATUS — attendre le traitement si nécessaire
        if "processing_info" in result:
            state = result["processing_info"].get("state", "")
            while state in ("pending", "in_progress"):
                wait = result["processing_info"].get("check_after_secs", 5)
                print(f"   ⏳ X traitement vidéo... ({wait}s)")
                time.sleep(wait)
                resp = requests.get(upload_url,
                                    params={"command": "STATUS", "media_id": media_id},
                                    auth=auth, timeout=30)
                result = resp.json()
                state = result.get("processing_info", {}).get("state", "succeeded")
            if state == "failed":
                err = result.get("processing_info", {}).get("error", {})
                print(f"   ❌ X traitement échoué : {err}")
                return False

        # Publier le tweet avec tweepy v2
        client = tweepy.Client(
            consumer_key=api_key, consumer_secret=api_secret,
            access_token=access_token, access_token_secret=access_secret)
        # Tronquer à 280 caractères
        tweet = tweet_text[:280]
        resp = client.create_tweet(text=tweet, media_ids=[media_id])
        tweet_id = resp.data.get("id", "?")
        print(f"   ✅ X/Twitter : publié (tweet {tweet_id})")
        return True

    except Exception as e:
        print(f"   ❌ X/Twitter : {e}")
        return False


def post_mode():
    """Mode publication interactif : sélectionner et poster un clip existant."""
    print("=" * 60)
    print("📡 Clipper — mode publication")
    print("=" * 60)

    # Détecter les plateformes configurées
    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    tg_chat = os.environ.get("TELEGRAM_CHAT_ID", "")
    tw_key = os.environ.get("TWITTER_API_KEY", "")
    tw_secret = os.environ.get("TWITTER_API_SECRET", "")
    tw_access = os.environ.get("TWITTER_ACCESS_TOKEN", "")
    tw_access_secret = os.environ.get("TWITTER_ACCESS_SECRET", "")

    has_telegram = bool(tg_token and tg_chat)
    has_twitter = bool(tw_key and tw_secret and tw_access and tw_access_secret)

    print("\n   Plateformes configurées :")
    print(f"   {'✅' if has_telegram else '❌'} Telegram" + (f" (chat: {tg_chat})" if has_telegram else " (TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID manquants)"))
    print(f"   {'✅' if has_twitter else '❌'} X/Twitter" + ("" if has_twitter else " (TWITTER_API_KEY/SECRET/ACCESS manquants)"))

    if not has_telegram and not has_twitter:
        print("\n   ❌ Aucune plateforme configurée.")
        print("   → Voir clipper-post-setup.txt pour la configuration")
        return

    # Boucle principale
    while True:
        clips = find_postable_clips(".")
        if not clips:
            print("\n   ❌ Aucun clip publiable trouvé (paires MP4+TXT) dans le répertoire courant.")
            return

        print(f"\n   📋 Clips disponibles :\n")
        for i, (mp4, txt) in enumerate(clips, 1):
            size_mb = os.path.getsize(mp4) / (1024 * 1024)
            name = os.path.basename(mp4)
            print(f"   {i:3d}. {name} ({size_mb:.1f} Mo)")

        print(f"\n   (q pour quitter)")

        try:
            choix = input("\n   → Numéro du clip : ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if choix.lower() in ("q", "quit", ""):
            return

        try:
            idx = int(choix) - 1
            if idx < 0 or idx >= len(clips):
                print(f"   ⚠️  Choix invalide (1-{len(clips)})")
                continue
        except ValueError:
            print(f"   ⚠️  Entrez un numéro ou 'q'")
            continue

        mp4_path, txt_path = clips[idx]
        print(f"\n   📎 {os.path.basename(mp4_path)}")
        print(f"   {'─' * 50}")

        # Lire et afficher le post
        with open(txt_path, "r", encoding="utf-8") as f:
            post_text = f.read().strip()
        print(f"\n{post_text}\n")
        print(f"   {'─' * 50}")

        # Menu action
        try:
            action = input("   [E]nvoyer / [M]odifier / [A]nnuler : ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if action in ("a", "annuler"):
            continue
        elif action in ("m", "modifier"):
            # Ouvrir dans $EDITOR
            editor = os.environ.get("EDITOR", "nano")
            try:
                subprocess.run([editor, txt_path])
            except FileNotFoundError:
                print(f"   ⚠️  Éditeur '{editor}' introuvable, essai avec nano...")
                try:
                    subprocess.run(["nano", txt_path])
                except FileNotFoundError:
                    print(f"   ❌ Aucun éditeur trouvé. Modifiez {txt_path} manuellement.")
                    continue

            # Relire après édition
            with open(txt_path, "r", encoding="utf-8") as f:
                post_text = f.read().strip()
            print(f"\n   Texte mis à jour :")
            print(f"\n{post_text}\n")

            try:
                confirm = input("   Envoyer ? [O]ui / [A]nnuler : ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if confirm not in ("o", "oui", "y", "yes"):
                continue
        elif action not in ("e", "envoyer"):
            continue

        # Publication
        print(f"\n   📤 Publication en cours...")

        if has_telegram:
            post_to_telegram(mp4_path, post_text, tg_token, tg_chat)

        if has_twitter:
            post_to_twitter(mp4_path, post_text,
                            tw_key, tw_secret, tw_access, tw_access_secret)

        print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global WHISPER_MODEL, CLAUDE_MODEL

    p = argparse.ArgumentParser(
        description="Extraction intelligente de clips viraux avec sous-titres karaoke",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Exemples :
              python clipper.py video.mp4 --criteria "passage le plus marquant"
              python clipper.py video.mp4 --criteria "moment drôle" --duration 180-600 -n 2
              python clipper.py video.mp4 --criteria "key insights" --target-lang fr
              python clipper.py "https://youtube.com/watch?v=XXXXX" --criteria "best moment"
              python clipper.py video.mp4 --resume video_clips.json --criteria "test"
        """))

    p.add_argument("source", nargs="?", default=None,
                   help="Fichier vidéo source (MP4) ou lien YouTube")
    p.add_argument("--criteria", required=False,
                   help="Critère de sélection des clips (en langage naturel)")
    p.add_argument("--post", action="store_true",
                   help="Mode publication : poster un clip existant sur Telegram et X")
    p.add_argument("-n", "--max-clips", type=int, default=3,
                   help="Nombre maximum de clips (défaut: 3)")
    p.add_argument("--duration", default="139-900",
                   help="Fourchette de durée min-max en secondes (défaut: 139-900)")
    p.add_argument("-s", "--source-lang", default=None,
                   help="Langue source (code ISO 639-1, défaut: auto-détection)")
    p.add_argument("-t", "--target-lang", default=None,
                   help="Langue cible pour traduction (optionnel)")
    p.add_argument("--resume", metavar="JSON",
                   help="Reprendre depuis un checkpoint JSON")
    p.add_argument("--pre-segments", metavar="JSON",
                   help="Charger des segments pré-calculés (traduire.py / doubler) "
                        "pour sauter la transcription WhisperX")
    p.add_argument("--skip-burn", action="store_true",
                   help="Générer les ASS sans les incruster")
    p.add_argument("--words-per-group", type=int, default=DEFAULT_WORDS_PER_GROUP,
                   help=f"Mots par groupe karaoke (défaut: {DEFAULT_WORDS_PER_GROUP})")
    p.add_argument("--context", type=str, default="",
                   help="Contexte pour guider la sélection : sujet, intervenants, etc.")
    p.add_argument("--speaker", type=str, default=None,
                   help="Nom de la personne qui parle (pour le post social)")
    p.add_argument("--date", type=str, default=None,
                   help="Date de la déclaration, ex: '12 mars 2025' (pour le post social)")
    p.add_argument("--url", type=str, default=None,
                   help="URL source (YouTube, etc.) à inclure dans le post social")
    p.add_argument("--whisper-model", default=WHISPER_MODEL)
    p.add_argument("--claude-model", default=CLAUDE_MODEL)
    p.add_argument("--llm", choices=["claude", "local"], default="local",
                   help="Backend LLM : local (Ollama, défaut) ou claude (API Anthropic)")
    p.add_argument("--ollama-model", default=OLLAMA_MODEL,
                   help=f"Modèle Ollama (défaut: {OLLAMA_MODEL})")
    p.add_argument("--ollama-url", default=OLLAMA_URL,
                   help=f"URL du serveur Ollama (défaut: {OLLAMA_URL})")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--cookies", default=None,
                   help="Chemin vers le fichier cookies JSON (Epoch Times / Apollo Health)")
    args = p.parse_args()

    # ── Mode publication interactif ──
    if args.post:
        return post_mode()

    # ── Mode extraction normal — valider les arguments requis ──
    if not args.source:
        p.error("source est requis (sauf en mode --post)")
    if not args.criteria:
        args.criteria = ("Sélectionne le ou les passages les plus viraux. "
                         "La viralité repose sur : "
                         "1) la SURPRISE (apprendre un fait inattendu, contre-intuitif ou méconnu), "
                         "2) le CHOC (information choquante, révélation, indignation), "
                         "3) l'IMPACT ÉMOTIONNEL (passage inspirant, enthousiasmant ou qui suscite la compassion). "
                         "Privilégie les moments qui provoquent une réaction forte chez le spectateur.")

    src_lang = args.source_lang.lower() if args.source_lang else None
    tgt_lang = args.target_lang.lower() if args.target_lang else None

    # Parser la durée min-max
    try:
        parts = args.duration.split("-")
        dur_min = float(parts[0])
        dur_max = float(parts[1]) if len(parts) > 1 else dur_min * 3
    except (ValueError, IndexError):
        print(f"❌ Format durée invalide : {args.duration} (attendu: min-max, ex: 30-90)")
        sys.exit(1)

    # ── Capturer l'URL YouTube avant téléchargement ──
    youtube_url = args.source if is_youtube_url(args.source) else args.url
    speaker = args.speaker

    # ── Téléchargement YouTube / Epoch Times / Apollo Health si lien ─────
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    epoch_page = None
    apollo_page = None
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
    clip_json = str(work_base / f"{base}_clips.json")
    audio = str(work_base / f"{base}_clip_audio.wav")

    WHISPER_MODEL = args.whisper_model; CLAUDE_MODEL = args.claude_model

    if args.llm == "local":
        client = _OllamaClient(args.ollama_url, args.ollama_model)
        print(f"   🧠 LLM local : Ollama {args.ollama_model}")
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("❌ ANTHROPIC_API_KEY manquante"); sys.exit(1)
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

    print("=" * 60)
    print(f"✂️  Clipper — extraction de clips viraux")
    print("=" * 60)
    print(f"   Source   : {args.source}")
    print(f"   Critère  : {args.criteria}")
    print(f"   Clips    : max {args.max_clips}, durée {dur_min:.0f}-{dur_max:.0f}s")
    print(f"   Langue   : {src_lang.upper() if src_lang else 'AUTO'}" + (f" → {tgt_lang.upper()}" if tgt_lang else ""))
    print(f"   Karaoke  : {args.words_per_group} mots/groupe")
    print(f"   Whisper  : {WHISPER_MODEL} | Claude : {CLAUDE_MODEL}")
    if args.context:
        print(f"   Contexte : {args.context[:80]}{'...' if len(args.context) > 80 else ''}")

    has_libass = check_ffmpeg()
    if not has_libass and not args.skip_burn:
        print("   ⚠️  --skip-burn activé automatiquement (libass manquant)")
        args.skip_burn = True

    print("=" * 60)

    t0 = time.time()

    # ── Chargement ou transcription ──

    segments = []
    clips = []

    if args.resume:
        resume_path = args.resume
        if not os.path.exists(resume_path):
            candidate = str(work_base / Path(resume_path).name)
            if os.path.exists(candidate):
                resume_path = candidate
        print(f"\n🔄 Reprise depuis {resume_path}")
        segments, clips = load_checkpoint(resume_path)
        print(f"   {len(segments)} segments, {len(clips)} clips chargés")
        if src_lang is None:
            src_lang = "en"
            print(f"   ⚠️  Langue source non spécifiée, défaut : EN (utilisez -s pour préciser)")
    elif args.pre_segments:
        pre_path = args.pre_segments
        if not os.path.exists(pre_path):
            candidate = str(work_base / Path(pre_path).name)
            if os.path.exists(candidate):
                pre_path = candidate
        if not os.path.exists(pre_path):
            print(f"❌ --pre-segments introuvable : {args.pre_segments}"); sys.exit(1)
        print(f"\n📥 Chargement segments pré-calculés : {pre_path}")
        with open(pre_path) as f:
            data = json.load(f)
        # text_adapted (doubler) prioritaire sur text_tgt (traduire) si présent
        segments = [
            Segment(
                index=d["index"], start=d["start"], end=d["end"],
                text=d.get("text", ""),
                text_tgt=(d.get("text_adapted") or d.get("text_tgt")
                          or d.get("text_fr") or ""),
                words=d.get("words", []),
                speaker=d.get("speaker", ""),
            )
            for d in data
        ]
        # Les segments doubler portent des ID (SPEAKER_xx) : les résoudre en noms
        # réels via la transcription Epoch si elle est disponible.
        if epoch_page and epoch_page.transcript and any(
                s.speaker.startswith("SPEAKER_") for s in segments):
            _, epoch_name_map = epochtimes.align_transcript_to_segments(
                epoch_page.transcript, segments, epoch_page.speakers)
            resolve_speaker_names(segments, epoch_name_map)
        missing_words = sum(1 for s in segments if not s.words)
        print(f"   {len(segments)} segments chargés"
              + (f" ({missing_words} sans timing mot-à-mot)" if missing_words else ""))
        if src_lang is None:
            src_lang = (data[0].get("lang") if data else None) or "en"
            print(f"   ℹ️  Langue source : {src_lang.upper()} (depuis JSON ou défaut)")
    else:
        # Passe 1 : transcription
        extract_audio(args.source, audio)
        segments, detected_lang = transcribe_whisperx(audio, src_lang, args.hf_token)
        src_lang = detected_lang

        # Relecture Epoch Times (correction noms propres + attribution locuteurs)
        if epoch_page and epoch_page.transcript:
            segments, epoch_name_map = epochtimes.align_transcript_to_segments(
                epoch_page.transcript, segments, epoch_page.speakers)
            named = resolve_speaker_names(segments, epoch_name_map)
            if named:
                print(f"   🗣️  {named}/{len(segments)} segments nommés "
                      f"({', '.join(sorted(set(epoch_name_map.values())))})")

        # Relecture Apollo Health (correction noms propres, termes médicaux)
        if apollo_page and apollo_page.transcript:
            segments = apollohealth.align_transcript_to_segments(
                apollo_page.transcript, segments)

        # Nettoyer le fichier audio temporaire
        try:
            os.remove(audio)
        except OSError:
            pass

    # ── Enrichir le contexte Claude avec les métadonnées source ───────────
    if epoch_page:
        epoch_ctx = epochtimes.build_epoch_context(epoch_page)
        if epoch_ctx:
            args.context = (epoch_ctx + "\n\n" + args.context).strip() if args.context else epoch_ctx
    if apollo_page:
        apollo_ctx = apollohealth.build_apollo_context(apollo_page)
        if apollo_ctx:
            args.context = (apollo_ctx + "\n\n" + args.context).strip() if args.context else apollo_ctx

    # ── Auto-traduction vers le français si la vidéo n'est pas en français ──
    if tgt_lang is None and src_lang and src_lang != "fr":
        tgt_lang = "fr"
        print(f"\n🌍 Langue détectée : {src_lang.upper()} — traduction automatique vers FR")
    elif tgt_lang is None and src_lang == "fr":
        print(f"\n🇫🇷 Vidéo en français — pas de traduction nécessaire")

    # ── Passe 2 : sélection des clips ──

    if not clips:
        clips = select_clips(segments, client, args.criteria,
                             max_clips=args.max_clips,
                             duration_min=dur_min, duration_max=dur_max,
                             source_lang=src_lang, context=args.context)
        if not clips:
            print("\n❌ Aucun clip trouvé. Essayez un critère différent.")
            sys.exit(1)

        # Clip bonus éducatif (densité informationnelle maximale)
        if args.max_clips >= 3:
            edu_clip = select_educational_clip(
                segments, client, existing_clips=clips,
                duration_min=dur_min, duration_max=dur_max,
                source_lang=src_lang, context=args.context)
            if edu_clip:
                clips.append(edu_clip)

        save_checkpoint(segments, clips, clip_json)

    # Remplir les segments de chaque clip (y compris montages, nécessaire après resume)
    for clip in clips:
        if not clip.segments:
            if clip.is_montage:
                # Montage : collecter les segments de chaque part
                clip.segments = [s for s in segments
                                 if any(p.seg_start <= s.index <= p.seg_end
                                        for p in clip.parts)]
            else:
                clip.segments = [s for s in segments
                                 if clip.seg_start <= s.index <= clip.seg_end]

    # ── Passe 3 : traduction optionnelle ──

    if tgt_lang:
        clips = translate_clips(clips, client, src_lang, tgt_lang, args.context)
        # Mettre à jour les segments principaux aussi
        clip_seg_idx = {}
        for clip in clips:
            for s in clip.segments:
                clip_seg_idx[s.index] = s.text_tgt
        for s in segments:
            if s.index in clip_seg_idx:
                s.text_tgt = clip_seg_idx[s.index]
        save_checkpoint(segments, clips, clip_json)

    # ── Passes 4-5-6 : extraction, ASS, burn pour chaque clip ──

    use_translation = bool(tgt_lang)
    lang_suffix = f"_{tgt_lang}" if tgt_lang else ""

    print(f"\n🎬 Génération de {len(clips)} clip(s)...")

    for clip in clips:
        ci = clip.clip_index
        clip_raw = str(work_base / f"{base}_clip{ci}_raw.mp4")

        print(f"\n{'─'*40}")
        if clip.is_montage:
            total_dur = sum(p.end - p.start for p in clip.parts)
            print(f"   Clip {ci}/{len(clips)} [MONTAGE {len(clip.parts)} parts] : {clip.titre}")
            print(f"   Durée totale : {total_dur:.1f}s")
        else:
            print(f"   Clip {ci}/{len(clips)} : {clip.titre}")
            print(f"   {_fmt(clip.start)} → {_fmt(clip.end)} ({clip.end - clip.start:.1f}s)")

        # Passe 4 : extraction
        if not os.path.exists(clip_raw):
            if clip.is_montage:
                result = extract_montage_clip(args.source, clip_raw, clip.parts)
            else:
                result = extract_clip(args.source, clip_raw, clip.start, clip.end)
            if not result:
                print(f"   ⚠️  Extraction échouée, clip {ci} ignoré")
                continue

        # Résolution du clip
        w, h = get_video_resolution(clip_raw)
        wide = is_16_9(w, h)

        # Post social
        post_path = str(OUTPUT_DIR / f"{base}_clip{ci}{lang_suffix}.txt")
        if not os.path.exists(post_path):
            post_text = generate_social_post(clip, client, speaker=speaker,
                                             url=youtube_url, target_lang=tgt_lang,
                                             date=args.date)
            if post_text:
                with open(post_path, "w", encoding="utf-8") as f:
                    f.write(post_text)
                print(f"   📝 Post : {post_path}")

        if wide:
            # ── Double export 16:9 + 1:1 (recadrage centre) ──
            clip_ass_16 = str(work_base / f"{base}_clip{ci}{lang_suffix}_16x9.ass")
            clip_final_16 = str(OUTPUT_DIR / f"{base}_clip{ci}{lang_suffix}_16x9.mp4")
            sq = h  # côté du carré = hauteur
            clip_ass_sq = str(work_base / f"{base}_clip{ci}{lang_suffix}_1x1.ass")
            clip_final_sq = str(OUTPUT_DIR / f"{base}_clip{ci}{lang_suffix}_1x1.mp4")

            # Passe 5a : ASS karaoke 16:9
            generate_karaoke_ass(clip, clip_ass_16, clip.start,
                                 words_per_group=args.words_per_group,
                                 use_translation=use_translation,
                                 video_width=w, video_height=h)

            # Passe 5b : ASS karaoke 1:1 (résolution carrée)
            generate_karaoke_ass(clip, clip_ass_sq, clip.start,
                                 words_per_group=args.words_per_group,
                                 use_translation=use_translation,
                                 video_width=sq, video_height=sq)

            # Passe 6 : burn
            if args.skip_burn:
                print(f"   ⏭️  Incrustation ignorée (--skip-burn)")
                if not os.path.exists(clip_final_16):
                    import shutil
                    shutil.copy2(clip_raw, clip_final_16)
            else:
                burn_subtitles(clip_raw, clip_ass_16, clip_final_16)
                crop = f"crop={sq}:{sq}:({w}-{sq})/2:0"
                burn_subtitles(clip_raw, clip_ass_sq, clip_final_sq,
                               crop_filter=crop)
                try:
                    os.remove(clip_raw)
                except OSError:
                    pass
        else:
            # ── Export simple (format non-16:9) ──
            clip_ass = str(work_base / f"{base}_clip{ci}{lang_suffix}.ass")
            clip_final = str(OUTPUT_DIR / f"{base}_clip{ci}{lang_suffix}.mp4")

            # Passe 5 : ASS karaoke
            generate_karaoke_ass(clip, clip_ass, clip.start,
                                 words_per_group=args.words_per_group,
                                 use_translation=use_translation,
                                 video_width=w, video_height=h)

            # Passe 6 : burn
            if args.skip_burn:
                print(f"   ⏭️  Incrustation ignorée (--skip-burn)")
                if not os.path.exists(clip_final):
                    os.rename(clip_raw, clip_final)
            else:
                result = burn_subtitles(clip_raw, clip_ass, clip_final)
                if result:
                    try:
                        os.remove(clip_raw)
                    except OSError:
                        pass

    # ── Résumé final ──

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"✅ Terminé en {elapsed:.0f}s")
    print(f"   Checkpoint : {clip_json}")

    for clip in clips:
        ci = clip.clip_index
        # Chercher les deux formats (16:9 + 1:1) ou le format unique
        variants = []
        for suffix in ["_16x9", "_1x1", ""]:
            mp4 = str(OUTPUT_DIR / f"{base}_clip{ci}{lang_suffix}{suffix}.mp4")
            ass = str(work_base / f"{base}_clip{ci}{lang_suffix}{suffix}.ass")
            if os.path.exists(mp4):
                mb = os.path.getsize(mp4) / (1024 * 1024)
                print(f"   🎬 {mp4} ({mb:.1f} Mo) — {clip.titre}")
                variants.append(mp4)
            elif os.path.exists(ass):
                print(f"   📝 {ass} — {clip.titre}")
        clip_post = str(OUTPUT_DIR / f"{base}_clip{ci}{lang_suffix}.txt")
        if os.path.exists(clip_post):
            print(f"   📄 {clip_post}")
    print("=" * 60)


if __name__ == "__main__":
    main()
