#!/usr/bin/env python3
"""
Résumé structuré de vidéo en PDF + EPUB
=========================================
Produit un résumé thématique de haute qualité en français à partir
d'une vidéo YouTube ou locale, via transcription WhisperX + synthèse Claude.

Architecture en 6 passes :
  1. Téléchargement + extraction audio (yt-dlp ou fichier local → WAV 16kHz)
  2. WhisperX          → transcription + timestamps + détection langue
  3. Claude (analyse)   → résumé, glossaire, ton, domaine, locuteurs
  4. Claude (traduction) → traduction FR si source ≠ français (chunks 60/8)
  5. Claude (synthèse)  → résumé structuré markdown (~350 mots/page)
  6. Génération         → PDF A4 (WeasyPrint) + EPUB (ebooklib)

Usage :
  python resumer.py video.mp4
  python resumer.py "https://www.youtube.com/watch?v=XXXXX"
  python resumer.py video.mp4 -s en --pages 8
  python resumer.py video.mp4 --resume video_segments.json
  python resumer.py video.mp4 --context "Interview avec Dr. X"
  python resumer.py video.mp4 --claude-model claude-opus-4

Prérequis :
  pip install whisperx anthropic torch torchaudio --break-system-packages
  # + clé API Anthropic (ANTHROPIC_API_KEY)
  # + ffmpeg installé
  # Pour PDF/EPUB (optionnel) :
  pip install weasyprint ebooklib --break-system-packages
"""

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Optional

import hw
hw.setup_rocm_env()  # AMD/ROCm (gfx1151) : pose HSA_OVERRIDE_* avant tout import torch

import apollohealth

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

WHISPER_MODEL = "large-v3"
WHISPER_BATCH_SIZE = 16
WHISPER_COMPUTE_TYPE = "float16"

CLAUDE_MODEL = "claude-opus-4-5"  # cf. A/B 2026-05-25 (traduire.py) : sonnet-4-6
# produit doublons synonymiques et étoffements malgré prompt explicite.
CLAUDE_MAX_TOKENS = 16384          # élevé pour la passe synthèse
CLAUDE_RETRY_MAX = 5
CLAUDE_RETRY_DELAY = 10.0

# Ollama (LLM local — alternative gratuite à l'API Claude)
# Défaut : mistral-small (14 Go) — tient ENTIÈREMENT en VRAM 24 Go (100% GPU,
# ~53 tok/s) → 5,3× plus rapide de bout en bout que qwen3.6:27b (qui, à 25 Go
# en exécution, déborde toujours ~12% sur le CPU → ~24 tok/s) ; qualité FR
# équivalente (prose structurée, natif français). Bench/validation 2026-06-24.
# qwen3.6:27b reste accessible via --ollama-model qwen3.6:27b (raisonneur dense).
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral-small:latest"
OLLAMA_NUM_PREDICT = 6144          # sortie ~4000 mots max (réflexion désactivée) ; tient dans num_ctx
OLLAMA_NUM_CTX_MAX = 24576        # plafond du contexte dynamique (couvre l'analyse ~20k tokens)
# Synthèse locale : chunks plus petits que pour Claude → moins de prefill sur
# CPU (qwen3.6:27b 25 Go ne tient pas en VRAM 24 Go, déborde toujours un peu) et
# chaque appel reste dans un petit num_ctx → nettement plus rapide (cf. 2026-06-24).
OLLAMA_SUMMARY_CHUNK_CHARS = 20000  # ~5000 tokens (vs 60000 pour Claude)
OLLAMA_SINGLE_CALL_MAX_CHARS = 20000  # au-delà → 2 passes (vs 100000 pour Claude)

# Chunks de traduction
CHUNK_SIZE = 60
CHUNK_OVERLAP = 8

# Synthèse
WORDS_PER_PAGE = 350               # mots/page cible
PAGES_PER_MINUTE = 0.3             # ~5 pages pour 17 min

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


def lang_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code.upper())


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


@dataclass
class ContentAnalysis:
    summary: str = ""
    glossary: dict = field(default_factory=dict)
    speakers_description: str = ""
    tone: str = ""
    domain: str = ""


@dataclass
class ResumeMetadata:
    title: str = ""
    channel: str = ""
    url: str = ""
    duration: float = 0.0
    source_lang: str = ""
    date: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# VÉRIFICATIONS PRÉALABLES
# ═══════════════════════════════════════════════════════════════════════════════

def check_ffmpeg():
    """Vérifie que ffmpeg est disponible."""
    if not shutil.which("ffmpeg"):
        print("❌ ffmpeg n'est pas installé.")
        sys.exit(1)


def check_dependencies(local=False):
    """Vérifie les dépendances Python essentielles.

    En mode local (Ollama), ni le module anthropic ni la clé API ne sont requis.
    """
    if local:
        return
    try:
        import anthropic
    except ImportError:
        print("❌ anthropic non installé → pip install anthropic --break-system-packages")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY non défini")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# APPEL CLAUDE AVEC RETRY
# ═══════════════════════════════════════════════════════════════════════════════

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
        # num_ctx DYNAMIQUE : on dimensionne le contexte à la taille réelle de
        # l'entrée (~3 caractères/token en français, conservateur) + marge de
        # sortie. Petits appels (chunks de synthèse) → petit contexte → qwen3.6:27b
        # déborde MOINS sur le CPU (~27 tok/s à 12k vs ~18 à 32k) ; gros appel
        # (analyse) → contexte élevé pour NE PAS tronquer. Borné à OLLAMA_NUM_CTX_MAX.
        in_chars = sum(len(m.get("content", "")) for m in msgs)
        need = in_chars // 3 + OLLAMA_NUM_PREDICT
        num_ctx = max(4096, min(OLLAMA_NUM_CTX_MAX, ((need + 2047) // 2048) * 2048))
        payload = json.dumps({
            "model": self.model,
            "messages": msgs,
            "stream": False,
            "think": False,            # désactive la réflexion (raisonneurs type qwen3.6/qwen3)
            "options": {"num_predict": OLLAMA_NUM_PREDICT, "num_ctx": num_ctx},
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
# PASSE 1 : TÉLÉCHARGEMENT + EXTRACTION AUDIO
# ═══════════════════════════════════════════════════════════════════════════════

def is_youtube_url(s: str) -> bool:
    """Détecte si la chaîne est un lien YouTube."""
    return bool(re.match(
        r'https?://(www\.)?(youtube\.com/(watch|shorts|live)|youtu\.be/)', s))


def get_youtube_metadata(url: str) -> ResumeMetadata:
    """Extrait titre, chaîne, durée via yt-dlp --print."""
    meta = ResumeMetadata(url=url)
    try:
        r = subprocess.run(
            ["yt-dlp", "--print", "title", "--print", "channel",
             "--print", "duration", "--no-warnings",
             "--cookies-from-browser", "firefox", url],
            capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            lines = r.stdout.strip().split("\n")
            if len(lines) >= 1:
                meta.title = lines[0].strip()
            if len(lines) >= 2:
                meta.channel = lines[1].strip()
            if len(lines) >= 3:
                try:
                    meta.duration = float(lines[2].strip())
                except ValueError:
                    pass
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    meta.date = date.today().isoformat()
    return meta


def download_youtube(url: str, output_dir: str = ".") -> str:
    """Télécharge une vidéo YouTube via yt-dlp et retourne le chemin du fichier MP4."""
    if not shutil.which("yt-dlp"):
        print("❌ yt-dlp n'est pas installé.")
        print("   → pip install yt-dlp --break-system-packages")
        sys.exit(1)

    print(f"\n📥 Téléchargement YouTube...")
    print(f"   URL : {url}")

    try:
        r = subprocess.run(
            ["yt-dlp", "--print", "title", "--no-warnings",
             "--cookies-from-browser", "firefox", url],
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
        "--cookies-from-browser", "firefox",
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


def extract_audio(video_path: str, output_path: str) -> str:
    """Extrait l'audio en WAV 16kHz mono."""
    print("\n🎵 Extraction de l'audio...")
    cmd = ["ffmpeg", "-y", "-i", video_path,
           "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"❌ ffmpeg : {r.stderr}"); sys.exit(1)
    print(f"   ✅ {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 2 : TRANSCRIPTION WHISPERX
# ═══════════════════════════════════════════════════════════════════════════════

# ── Verrou GPU global du toolkit ──────────────────────────────────────────────
_GPU_LOCK_PATH = os.path.expanduser("~/.cache/traduction_gpu.lock")
_gpu_lock_fh = None


def acquire_gpu_lock():
    """Sérialise toutes les tâches GPU du toolkit (WhisperX + LLM local) via un
    verrou fichier partagé : empêche deux tâches GPU de tourner en même temps
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
    la VRAM avant de charger WhisperX (assoupli sur mémoire unifiée AMD)."""
    return hw.free_gpu_for_task(min_free_mib=min_free_mib, timeout=timeout)


def transcribe_whisperx(audio_path: str, source_lang: Optional[str] = None,
                        hf_token: Optional[str] = None) -> tuple[list, str]:
    """Transcrit l'audio. Retourne (segments, langue_détectée)."""
    acquire_gpu_lock()
    free_gpu_for_task(min_free_mib=6000, timeout=60)  # purge un Ollama résident avant WhisperX
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
# CHECKPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

def save_segments(segments: list[Segment], path: str):
    """Sauvegarde les segments en JSON."""
    data = [
        {"index": s.index, "start": s.start, "end": s.end,
         "text": s.text, "text_tgt": s.text_tgt,
         "words": s.words}
        for s in segments
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"   💾 → {path}")


def load_segments(path: str) -> list[Segment]:
    """Charge les segments depuis un checkpoint JSON."""
    with open(path) as f:
        data = json.load(f)
    # Gestion format clipper (dict avec clé "segments") et format plat (liste)
    if isinstance(data, dict) and "segments" in data:
        data = data["segments"]
    return [
        Segment(d["index"], d["start"], d["end"], d["text"],
                d.get("text_tgt", ""), d.get("words", []))
        for d in data
    ]


def save_analysis(analysis: ContentAnalysis, path: str):
    """Sauvegarde l'analyse en JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(analysis), f, ensure_ascii=False, indent=2)
    print(f"   💾 → {path}")


def load_analysis(path: str) -> ContentAnalysis:
    """Charge l'analyse depuis un checkpoint JSON."""
    with open(path) as f:
        data = json.load(f)
    return ContentAnalysis(**data)


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 3 : ANALYSE DU CONTENU PAR CLAUDE
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_content(segments: list[Segment], client, claude_model: str,
                    source_lang: str = "en", target_lang: str = "fr",
                    context: str = "") -> ContentAnalysis:
    print("\n🔍 Analyse du contenu...")

    src_name = lang_name(source_lang)
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

    resp = _claude_create(client, model=claude_model, max_tokens=4096,
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
# PASSE 4 : TRADUCTION PAR CHUNKS (SI SOURCE ≠ FR)
# ═══════════════════════════════════════════════════════════════════════════════

def build_system_translate(source_lang: str, target_lang: str,
                           analysis: ContentAnalysis, context: str = "") -> str:
    """Construit le prompt système de traduction."""
    src_name = lang_name(source_lang)
    tgt_name = lang_name(target_lang)

    lang_specific_rules = ""
    if target_lang == "fr":
        lang_specific_rules = "3. Tutoiement/vouvoiement cohérent selon le contexte\n"
    elif target_lang == "de":
        lang_specific_rules = "3. Du/Sie cohérent selon le contexte\n"
    elif target_lang == "es":
        lang_specific_rules = "3. Tú/Usted cohérent selon le contexte\n"

    user_context = f"\nINSTRUCTIONS UTILISATEUR :\n{context}\n" if context else ""

    return f"""Tu es un traducteur professionnel {src_name} → {tgt_name}.

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
6. Correspondance 1:1 stricte des numéros — ne JAMAIS fusionner
7. Ne traduis QUE les segments "À TRADUIRE", pas le contexte
8. Style oral naturel, pas littéraire
9. Nettoie hésitations et faux départs

FORMAT : [numéro] texte en {tgt_name} (un par ligne, rien d'autre)"""


def _parse_translation(text: str, segments: list[Segment], s: int, e: int):
    """Parse la réponse de traduction Claude."""
    for line in text.strip().split("\n"):
        m = re.match(r'\[(\d+)\]\s*(.*)', line.strip())
        if m:
            idx, txt = int(m.group(1)), m.group(2).strip()
            txt = re.sub(r'^[A-Z]{2}:\s*', '', txt).strip()
            if txt:
                for seg in segments[s:e]:
                    if seg.index == idx:
                        seg.text_tgt = txt; break


def _retry_translation(segments: list[Segment], s: int, e: int,
                       system: str, client, claude_model: str,
                       source_lang: str, target_lang: str):
    """Retry pour segments manquants."""
    missing = [seg for seg in segments[s:e] if not seg.text_tgt]
    if not missing: return
    parts = ["Segments manquants :\n"]
    for seg in missing:
        ctx = [x for x in segments[max(0, seg.index-4):seg.index-1] if x.text_tgt]
        if ctx: parts.append(f"  (ctx: [{ctx[-1].index}] {ctx[-1].text_tgt})")
        parts.append(f"[{seg.index}] {seg.text}")
    resp = _claude_create(client, model=claude_model, max_tokens=CLAUDE_MAX_TOKENS,
                          system=system,
                          messages=[{"role": "user", "content": "\n".join(parts)}])
    _parse_translation(resp.content[0].text, segments, s, e)


def translate_chunks(segments: list[Segment], analysis: ContentAnalysis, client,
                     claude_model: str, source_lang: str = "en",
                     target_lang: str = "fr", context: str = "") -> list[Segment]:
    """Traduction par chunks avec contexte glissant."""
    print(f"\n🌍 Traduction {source_lang}→{target_lang} ({CHUNK_SIZE} seg/chunk, {CHUNK_OVERLAP} overlap)...")

    system = build_system_translate(source_lang, target_lang, analysis, context)
    n_chunks = (len(segments) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for ci in range(n_chunks):
        s, e = ci * CHUNK_SIZE, min((ci + 1) * CHUNK_SIZE, len(segments))

        if all(seg.text_tgt for seg in segments[s:e]):
            print(f"   📦 Chunk {ci+1}/{n_chunks} — déjà traduit")
            continue

        print(f"   📦 Chunk {ci+1}/{n_chunks} (seg {s+1}–{e})...")
        parts = []

        cb = max(0, s - CHUNK_OVERLAP)
        if cb < s:
            parts.append("=== CONTEXTE PRÉCÉDENT (NE PAS retraduire) ===")
            for seg in segments[cb:s]:
                if seg.text_tgt:
                    parts += [f"[{seg.index}] {source_lang.upper()}: {seg.text}",
                              f"[{seg.index}] {target_lang.upper()}: {seg.text_tgt}"]
            parts.append("")

        parts.append("=== À TRADUIRE ===")
        parts += [f"[{seg.index}] {seg.text}" for seg in segments[s:e]]
        parts.append("")

        cf = min(len(segments), e + CHUNK_OVERLAP)
        if e < cf:
            parts.append("=== CONTEXTE SUIVANT (NE PAS traduire) ===")
            parts += [f"[{seg.index}] {seg.text}" for seg in segments[e:cf]]

        resp = _claude_create(client, model=claude_model, max_tokens=CLAUDE_MAX_TOKENS,
                              system=system,
                              messages=[{"role": "user", "content": "\n".join(parts)}])
        _parse_translation(resp.content[0].text, segments, s, e)

        done = sum(1 for seg in segments[s:e] if seg.text_tgt)
        if done < e - s:
            print(f"   ⚠️  {done}/{e-s} — relance manquants...")
            _retry_translation(segments, s, e, system, client, claude_model,
                               source_lang, target_lang)
            done = sum(1 for seg in segments[s:e] if seg.text_tgt)

        print(f"   ✅ {done}/{e-s}")

    total = sum(1 for s in segments if s.text_tgt)
    print(f"\n   🌍 Traduit : {total}/{len(segments)}")
    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 5 : SYNTHÈSE STRUCTURÉE CLAUDE
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_target_words(duration_sec: float, override_pages: Optional[int] = None) -> int:
    """Calcule le nombre de mots cible pour le résumé."""
    if override_pages:
        return override_pages * WORDS_PER_PAGE
    pages = max(2, duration_sec / 60.0 * PAGES_PER_MINUTE)
    return int(pages * WORDS_PER_PAGE)


def generate_summary(segments: list[Segment], analysis: ContentAnalysis,
                     metadata: ResumeMetadata, client, claude_model: str,
                     target_words: int, context: str = "") -> str:
    """Génère le résumé structuré en markdown via Claude."""
    is_local = isinstance(client, _OllamaClient)
    if is_local:
        acquire_gpu_lock()   # chemin --resume (pas de transcription) avec LLM local
    # Texte source : utiliser text_tgt si traduit, sinon text
    full_text = "\n".join(
        f"[{s.start:.0f}s] {s.text_tgt or s.text}" for s in segments
    )
    total_chars = len(full_text)

    ctx_block = ""
    if context:
        ctx_block = f"\nCONTEXTE UTILISATEUR : {context}\n"

    meta_block = ""
    if metadata.title:
        meta_block += f"Titre : {metadata.title}\n"
    if metadata.channel:
        meta_block += f"Chaîne/Auteur : {metadata.channel}\n"
    if metadata.duration > 0:
        mins = metadata.duration / 60
        meta_block += f"Durée : {mins:.0f} min\n"

    system_prompt = f"""Tu es un rédacteur professionnel français spécialisé dans la synthèse de contenus vidéo.

Tu produis des résumés structurés de haute qualité : prose élégante, claire, fidèle au contenu original.

RÈGLES STRICTES :
1. STRUCTURE THÉMATIQUE — organise par thèmes/idées, PAS chronologiquement
2. CITATIONS VERBATIM — inclus 2-4 citations mot-pour-mot entre guillemets français (« »), choisies pour leur force ou leur clarté
3. FIDÉLITÉ — ne jamais inventer, extrapoler ou ajouter des informations absentes de la transcription
4. PROSE — phrases variées, vocabulaire précis, registre soutenu mais accessible
5. FORMAT — markdown avec titres ## pour les sections thématiques, **gras** pour les concepts clés
6. LONGUEUR — vise ~{target_words} mots ({target_words // WORDS_PER_PAGE} pages)
7. INTRODUCTION — commence par un paragraphe d'accroche qui résume l'essentiel en 2-3 phrases
8. PAS de table des matières, pas de « Introduction » ou « Conclusion » comme titres de section
9. Rédige entièrement en français"""

    # Pour les contenus courts : un seul appel (seuil réduit en local pour
    # garder l'appel dans un petit contexte GPU)
    single_call_max = OLLAMA_SINGLE_CALL_MAX_CHARS if is_local else 100000
    if total_chars < single_call_max:
        user_prompt = f"""{meta_block}{ctx_block}
ANALYSE PRÉALABLE :
Résumé : {analysis.summary}
Domaine : {analysis.domain} | Ton : {analysis.tone}
Locuteurs : {analysis.speakers_description}
Glossaire : {json.dumps(analysis.glossary, ensure_ascii=False)}

TRANSCRIPTION COMPLÈTE :
{full_text}

Rédige le résumé structuré (~{target_words} mots)."""

        print(f"\n✍️  Synthèse Claude ({target_words} mots cible, appel unique)...")
        resp = _claude_create(client, model=claude_model, max_tokens=CLAUDE_MAX_TOKENS,
                              system=system_prompt,
                              messages=[{"role": "user", "content": user_prompt}])
        return resp.content[0].text.strip()

    # Pour les contenus longs : 2 passes (extraction + synthèse)
    print(f"\n✍️  Synthèse Claude ({target_words} mots cible, 2 passes — contenu long)...")

    # Passe 1 : extraction des points clés par chunks
    chunk_size_chars = OLLAMA_SUMMARY_CHUNK_CHARS if is_local else 60000
    key_points = []
    chunks = []
    pos = 0
    lines = full_text.split("\n")
    current_chunk = []
    current_len = 0
    for line in lines:
        if current_len + len(line) > chunk_size_chars and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_len = 0
        current_chunk.append(line)
        current_len += len(line) + 1
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    for i, chunk in enumerate(chunks):
        print(f"   📦 Extraction points clés {i+1}/{len(chunks)}...")
        resp = _claude_create(client, model=claude_model, max_tokens=8192,
                              system="Tu extrais les points clés, arguments, exemples et citations remarquables d'une transcription. Réponds en français, en bullet points structurés.",
                              messages=[{"role": "user", "content": f"Extrais les points clés de ce segment :\n\n{chunk}"}])
        key_points.append(resp.content[0].text.strip())

    # Passe 2 : synthèse à partir des points clés
    all_points = "\n\n---\n\n".join(key_points)
    user_prompt = f"""{meta_block}{ctx_block}
ANALYSE PRÉALABLE :
Résumé : {analysis.summary}
Domaine : {analysis.domain} | Ton : {analysis.tone}
Locuteurs : {analysis.speakers_description}

POINTS CLÉS EXTRAITS :
{all_points}

Rédige le résumé structuré (~{target_words} mots) à partir de ces points clés."""

    print(f"   ✍️  Rédaction finale...")
    resp = _claude_create(client, model=claude_model, max_tokens=CLAUDE_MAX_TOKENS,
                          system=system_prompt,
                          messages=[{"role": "user", "content": user_prompt}])
    return resp.content[0].text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 6 : GÉNÉRATION PDF + EPUB
# ═══════════════════════════════════════════════════════════════════════════════

def markdown_to_html(md: str) -> str:
    """Convertisseur léger markdown → HTML (##, **, *, >, paragraphes)."""
    lines = md.split("\n")
    html_parts = []
    in_blockquote = False
    paragraph_lines = []

    def flush_paragraph():
        if paragraph_lines:
            text = " ".join(paragraph_lines)
            text = _inline_format(text)
            html_parts.append(f"<p>{text}</p>")
            paragraph_lines.clear()

    def _inline_format(text):
        # Gras
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        # Italique
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        # Guillemets français
        text = re.sub(r'«\s*', '«\u202f', text)
        text = re.sub(r'\s*»', '\u202f»', text)
        return text

    for line in lines:
        stripped = line.strip()

        # Ligne vide : flush
        if not stripped:
            if in_blockquote:
                html_parts.append("</blockquote>")
                in_blockquote = False
            flush_paragraph()
            continue

        # Titres
        if stripped.startswith("## "):
            flush_paragraph()
            title = _inline_format(stripped[3:].strip())
            html_parts.append(f"<h2>{title}</h2>")
            continue
        if stripped.startswith("# "):
            flush_paragraph()
            title = _inline_format(stripped[2:].strip())
            html_parts.append(f"<h1>{title}</h1>")
            continue

        # Blockquote
        if stripped.startswith("> "):
            flush_paragraph()
            if not in_blockquote:
                html_parts.append("<blockquote>")
                in_blockquote = True
            text = _inline_format(stripped[2:].strip())
            html_parts.append(f"<p>{text}</p>")
            continue

        # Si on était dans un blockquote et la ligne n'est pas une quote
        if in_blockquote:
            html_parts.append("</blockquote>")
            in_blockquote = False

        # Ligne de texte normal → accumule dans le paragraphe
        paragraph_lines.append(stripped)

    # Flush final
    if in_blockquote:
        html_parts.append("</blockquote>")
    flush_paragraph()

    return "\n".join(html_parts)


# -- CSS PDF A4 --

PDF_CSS = """
@page {
    size: A4;
    margin: 28mm 25mm 30mm 30mm;
    @bottom-center {
        content: counter(page);
        font-family: "Noto Sans", sans-serif;
        font-size: 8pt;
        color: #999;
    }
    @top-center {
        content: string(book-title);
        font-family: "Noto Serif", serif;
        font-size: 7.5pt;
        font-style: italic;
        color: #aaa;
        letter-spacing: 0.03em;
    }
}
@page :first {
    margin: 0;
    @bottom-center { content: none; }
    @top-center { content: none; }
}
@page chapter-first {
    @top-center { content: none; }
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: "Noto Serif", Georgia, serif;
    font-size: 10.5pt;
    line-height: 1.65;
    color: #1a1a1a;
    text-align: justify;
    hyphens: auto;
    -webkit-hyphens: auto;
    orphans: 3;
    widows: 3;
}

/* -- Couverture ------------------------------------ */
.cover {
    page-break-after: always;
    width: 210mm; height: 297mm;
    display: flex; flex-direction: column;
    justify-content: center; align-items: center;
    text-align: center;
    background: #faf9f7;
    padding: 40mm 30mm;
    position: relative;
}
.cover::before {
    content: ""; position: absolute;
    top: 25mm; left: 30mm; right: 30mm;
    height: 0.6pt; background: #B8860B;
}
.cover::after {
    content: ""; position: absolute;
    bottom: 25mm; left: 30mm; right: 30mm;
    height: 0.6pt; background: #B8860B;
}
.cover-label {
    font-family: "Noto Sans", sans-serif;
    font-size: 9pt;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: #B8860B;
    margin-bottom: 8mm;
}
.cover-title {
    font-family: "Noto Serif", serif;
    font-size: 24pt;
    font-weight: 700;
    line-height: 1.2;
    color: #1a1a1a;
    margin-bottom: 6mm;
    letter-spacing: -0.01em;
    hyphens: none;
}
.cover-subtitle {
    font-family: "Noto Serif", serif;
    font-size: 11pt;
    font-style: italic;
    color: #666;
    max-width: 120mm;
    line-height: 1.6;
    margin-bottom: 15mm;
}
.cover-author {
    font-family: "Noto Sans", sans-serif;
    font-size: 11pt;
    font-weight: 600;
    color: #333;
    letter-spacing: 0.05em;
    margin-bottom: 4mm;
}
.cover-meta {
    font-family: "Noto Sans", sans-serif;
    font-size: 8.5pt;
    color: #999;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* -- Titre courant --------------------------------- */
h1.book-title-string {
    string-set: book-title content();
    font-size: 0; height: 0; margin: 0; padding: 0;
    visibility: hidden;
}

/* -- Corps ----------------------------------------- */
.resume-body {
    page: chapter-first;
}
.resume-body h2 {
    font-family: "Noto Serif", serif;
    font-size: 14pt;
    font-weight: 700;
    margin-top: 8mm;
    margin-bottom: 4mm;
    color: #1a1a1a;
    letter-spacing: -0.01em;
    page-break-after: avoid;
}
.resume-body h1 {
    font-family: "Noto Serif", serif;
    font-size: 18pt;
    font-weight: 700;
    margin-top: 10mm;
    margin-bottom: 5mm;
    color: #1a1a1a;
}
.resume-body p {
    margin-bottom: 3.5mm;
    text-indent: 0;
}
.resume-body p + p {
    text-indent: 5mm;
}
.resume-body strong {
    font-weight: 700;
}
.resume-body em {
    font-style: italic;
}
.resume-body blockquote {
    margin: 4mm 0 4mm 4mm;
    padding: 2mm 0 2mm 4mm;
    border-left: 2pt solid #B8860B;
    font-style: italic;
    color: #444;
}
.resume-body blockquote p {
    text-indent: 0;
    margin-bottom: 2mm;
}

/* -- Colophon -------------------------------------- */
.colophon {
    page-break-before: always;
    padding-top: 60mm;
    text-align: center;
}
.colophon p {
    font-family: "Noto Sans", sans-serif;
    font-size: 8pt;
    color: #999;
    line-height: 1.8;
}
.colophon .colophon-title {
    font-family: "Noto Serif", serif;
    font-size: 12pt;
    color: #333;
    margin-bottom: 3mm;
    font-weight: 600;
}
.colophon .colophon-rule {
    width: 30mm; height: 0.4pt;
    background: #B8860B;
    margin: 8mm auto;
}
"""

# -- CSS EPUB --

EPUB_CSS = """
body {
    font-family: Georgia, "Times New Roman", serif;
    font-size: 1em;
    line-height: 1.7;
    color: #1a1a1a;
    margin: 0;
    padding: 0;
}
h1 {
    font-size: 1.6em;
    font-weight: 700;
    line-height: 1.25;
    margin: 0 0 0.3em;
    color: #1a1a1a;
}
h2 {
    font-size: 1.3em;
    font-weight: 700;
    line-height: 1.3;
    margin: 1.2em 0 0.4em;
    color: #1a1a1a;
}
.resume-meta {
    font-size: 0.8em;
    color: #999;
    margin-bottom: 1em;
}
p {
    margin-bottom: 0.8em;
    text-align: justify;
}
strong { font-weight: 700; }
em { font-style: italic; }
blockquote {
    border-left: 3px solid #B8860B;
    padding-left: 1em;
    margin: 1em 0;
    font-style: italic;
    color: #444;
}
blockquote p { margin-bottom: 0.5em; }
.colophon {
    text-align: center;
    margin-top: 3em;
    color: #999;
    font-size: 0.85em;
    line-height: 1.8;
}
"""


def _esc(s: str) -> str:
    """Échappe les caractères HTML."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _build_full_html(markdown: str, metadata: ResumeMetadata) -> str:
    """Construit le HTML complet (couverture + corps + colophon)."""
    body_html = markdown_to_html(markdown)

    title_esc = _esc(metadata.title or "Résumé vidéo")
    channel_esc = _esc(metadata.channel) if metadata.channel else ""
    dur_str = ""
    if metadata.duration > 0:
        dur_str = f"{metadata.duration / 60:.0f} min"

    subtitle_parts = []
    if metadata.channel:
        subtitle_parts.append(channel_esc)
    if dur_str:
        subtitle_parts.append(dur_str)
    subtitle = " · ".join(subtitle_parts)

    cover_html = f"""
    <div class="cover">
        <div class="cover-label">Résumé</div>
        <div class="cover-title">{title_esc}</div>
        <div class="cover-subtitle">{subtitle}</div>
        {f'<div class="cover-author">{channel_esc}</div>' if metadata.channel else ''}
        <div class="cover-meta">{_esc(metadata.date)}</div>
    </div>
    <h1 class="book-title-string">{title_esc}</h1>
    """

    colophon_html = f"""
    <div class="colophon">
        <div class="colophon-title">Résumé</div>
        <p>{title_esc}</p>
        <div class="colophon-rule"></div>
        <p>Transcription WhisperX · Synthèse Claude</p>
        <p style="margin-top: 5mm;">
            Résumé produit automatiquement par IA à partir<br>
            de la transcription de la vidéo originale.
        </p>
    </div>
    """

    return f"""<!DOCTYPE html>
<html lang="fr">
<head><meta charset="UTF-8"><style>{PDF_CSS}</style></head>
<body>
{cover_html}
<div class="resume-body">
{body_html}
</div>
{colophon_html}
</body></html>"""


def build_pdf(markdown: str, metadata: ResumeMetadata, output_path: str):
    """Génère un PDF A4 avec WeasyPrint."""
    try:
        from weasyprint import HTML
    except ImportError:
        print("   ⚠️  weasyprint non disponible — PDF non généré")
        print("   → pip install weasyprint --break-system-packages")
        return False

    full_html = _build_full_html(markdown, metadata)
    HTML(string=full_html).write_pdf(str(output_path))
    print(f"   📄 PDF → {output_path}")
    return True


def build_html(markdown: str, metadata: ResumeMetadata, output_path: str):
    """Génère un fichier HTML standalone."""
    full_html = _build_full_html(markdown, metadata)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"   🌐 HTML → {output_path}")
    return True


def build_epub(markdown: str, metadata: ResumeMetadata, output_path: str):
    """Génère un EPUB."""
    try:
        from ebooklib import epub
    except ImportError:
        print("   ⚠️  ebooklib non disponible — EPUB non généré")
        print("   → pip install ebooklib --break-system-packages")
        return False

    body_html = markdown_to_html(markdown)
    title = metadata.title or "Résumé vidéo"

    book = epub.EpubBook()
    book.set_identifier(f"resume-{hash(title) & 0xFFFFFFFF:08x}")
    book.set_title(title)
    book.set_language("fr")
    if metadata.channel:
        book.add_author(metadata.channel)
    book.add_metadata("DC", "description",
                      f"Résumé de : {title}")

    # CSS
    style = epub.EpubItem(uid="style", file_name="style/default.css",
                          media_type="text/css", content=EPUB_CSS.encode())
    book.add_item(style)

    # Chapitre unique : le résumé
    dur_str = f"{metadata.duration / 60:.0f} min" if metadata.duration > 0 else ""
    meta_parts = []
    if metadata.channel:
        meta_parts.append(_esc(metadata.channel))
    if dur_str:
        meta_parts.append(dur_str)
    meta_str = " · ".join(meta_parts)

    ch = epub.EpubHtml(title=title, file_name="resume.xhtml", lang="fr")
    ch.content = f"""<html><head></head><body>
<h1>{_esc(title)}</h1>
<div class="resume-meta">{meta_str}</div>
{body_html}
</body></html>"""
    ch.add_item(style)
    book.add_item(ch)

    # Colophon
    colophon = epub.EpubHtml(title="À propos", file_name="colophon.xhtml", lang="fr")
    colophon.content = f"""<html><head></head><body>
<div class="colophon">
<p><strong>Résumé</strong></p>
<p>{_esc(title)}</p>
<p>Transcription WhisperX · Synthèse Claude</p>
<p>Résumé produit automatiquement par IA à partir
de la transcription de la vidéo originale.</p>
</div></body></html>"""
    colophon.add_item(style)
    book.add_item(colophon)

    book.toc = [ch]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", ch, colophon]

    epub.write_epub(str(output_path), book, {})
    print(f"   📚 EPUB → {output_path}")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Résumé structuré de vidéo en PDF + EPUB")
    parser.add_argument("input", help="Vidéo locale ou URL YouTube")
    parser.add_argument("-s", "--source", default=None,
                        help="Langue source (code ISO, ex: en). Auto-détection si omis.")
    parser.add_argument("-t", "--target", default="fr",
                        help="Langue cible pour le résumé (défaut: fr)")
    parser.add_argument("--pages", type=int, default=None,
                        help="Nombre de pages cible (sinon calculé depuis la durée)")
    parser.add_argument("--resume", default=None,
                        help="Reprendre depuis un checkpoint segments JSON")
    parser.add_argument("--context", default="",
                        help="Contexte additionnel (noms propres, sujet, etc.)")
    parser.add_argument("--claude-model", default=None,
                        help=f"Modèle Claude (défaut: {CLAUDE_MODEL})")
    parser.add_argument("--llm", choices=["claude", "local"], default="local",
                        help="Backend LLM : local (Ollama, défaut) ou claude (API Anthropic)")
    parser.add_argument("--analysis-llm", choices=["auto", "claude", "local"], default="auto",
                        help="LLM de la passe d'analyse/contexte : auto = Claude si "
                             "ANTHROPIC_API_KEY dispo, sinon local")
    parser.add_argument("--ollama-model", default=OLLAMA_MODEL,
                        help=f"Modèle Ollama (défaut: {OLLAMA_MODEL})")
    parser.add_argument("--ollama-url", default=OLLAMA_URL,
                        help=f"URL du serveur Ollama (défaut: {OLLAMA_URL})")
    parser.add_argument("--output-dir", default=None,
                        help="Répertoire de copie des documents finaux (créé si nécessaire)")
    parser.add_argument("--html", action="store_true",
                        help="Générer un fichier HTML standalone")
    parser.add_argument("--cookies", default=None,
                        help="Chemin vers le fichier cookies JSON (Apollo Health)")
    args = parser.parse_args()

    # Modèle LLM (Claude API ou Ollama local)
    is_local = args.llm == "local"
    claude_model = args.ollama_model if is_local else (args.claude_model or CLAUDE_MODEL)

    # Vérifications
    check_dependencies(local=is_local)
    check_ffmpeg()

    if is_local:
        client = _OllamaClient(args.ollama_url, args.ollama_model)
        print(f"   🧠 LLM local : Ollama {args.ollama_model}")
    else:
        import anthropic
        client = anthropic.Anthropic()

    # Analyse : Claude apporte une meilleure connaissance du monde (noms propres,
    # domaine, glossaire) → meilleur résumé. Un seul appel, peu coûteux.
    analysis_client = client
    analysis_model = claude_model
    if args.analysis_llm != "local":
        _want_claude = args.analysis_llm == "claude" or (
            args.analysis_llm == "auto" and os.environ.get("ANTHROPIC_API_KEY"))
        if _want_claude:
            try:
                import anthropic
                analysis_client = anthropic.Anthropic()
                analysis_model = CLAUDE_MODEL
                print("   🧠 Analyse du contexte via Claude (reste en local)")
            except Exception as _e:
                print(f"   ⚠️  Claude indispo pour l'analyse ({_e}) — analyse en local")

    # Déterminer le fichier vidéo et les métadonnées
    is_yt = is_youtube_url(args.input)
    is_apollo = apollohealth.is_apollo_url(args.input)
    metadata = ResumeMetadata(date=date.today().isoformat())
    apollo_page = None

    if is_yt:
        print("=" * 70)
        print("  RÉSUMÉ VIDÉO — YouTube")
        print("=" * 70)
        metadata = get_youtube_metadata(args.input)
        video_path = download_youtube(args.input)
    elif is_apollo:
        print("=" * 70)
        print("  RÉSUMÉ VIDÉO — Apollo Health")
        print("=" * 70)
        cookies = apollohealth.load_cookies(args.cookies)
        apollo_page = apollohealth.fetch_apollo_page(args.input, cookies)
        print(f"   Titre     : {apollo_page.title}")
        if apollo_page.transcript:
            print(f"   Transcription : {len(apollo_page.transcript)} spans")
        video_path = apollohealth.download_apollo_video(apollo_page, output_dir=".")
        apollohealth.save_apollo_meta(apollo_page, video_path)
        metadata.title = apollo_page.title
        metadata.channel = apollo_page.author
        # Enrichir le contexte Claude
        apollo_ctx = apollohealth.build_apollo_context(apollo_page)
        if apollo_ctx:
            args.context = (apollo_ctx + "\n\n" + args.context).strip() if args.context else apollo_ctx
    else:
        video_path = args.input
        if not os.path.exists(video_path):
            print(f"❌ Fichier introuvable : {video_path}")
            sys.exit(1)
        print("=" * 70)
        print(f"  RÉSUMÉ VIDÉO — {os.path.basename(video_path)}")
        print("=" * 70)
        metadata.title = Path(video_path).stem.replace("_", " ")

    # Charger le sidecar Apollo Health si présent
    if not apollo_page:
        apollo_page = apollohealth.load_apollo_meta(video_path if not is_yt else args.input)
        if apollo_page:
            apollo_ctx = apollohealth.build_apollo_context(apollo_page)
            if apollo_ctx:
                args.context = (apollo_ctx + "\n\n" + args.context).strip() if args.context else apollo_ctx

    base = Path(video_path).stem
    segments_path = f"{base}_segments.json"
    analysis_path = f"{base}_analyse.json"
    md_path = f"{base}_resume.md"
    target_lang = args.target

    # ── PASSE 1+2 : Transcription ──
    segments = None
    detected_lang = args.source

    if args.resume:
        print(f"\n⏩ Reprise depuis {args.resume}")
        segments = load_segments(args.resume)
        # Essayer de détecter la langue depuis le fichier ou l'argument
        detected_lang = args.source or "en"
        print(f"   {len(segments)} segments chargés")

        # Charger l'analyse si elle existe
        if os.path.exists(analysis_path):
            print(f"   📋 Analyse chargée depuis {analysis_path}")
    else:
        audio_path = base + "_audio.wav"
        if not os.path.exists(audio_path):
            extract_audio(video_path, audio_path)
        else:
            print(f"\n⏩ Audio déjà extrait : {audio_path}")

        segments, detected_lang = transcribe_whisperx(audio_path, args.source)

        # Relecture Apollo Health (correction noms propres, termes médicaux)
        if apollo_page and apollo_page.transcript:
            segments = apollohealth.align_transcript_to_segments(
                apollo_page.transcript, segments)

        save_segments(segments, segments_path)

        # Nettoyer le WAV temporaire
        try:
            os.remove(audio_path)
        except OSError:
            pass

    if not segments:
        print("❌ Aucun segment transcrit")
        sys.exit(1)

    # Mettre à jour la durée si pas déjà connue
    if metadata.duration == 0 and segments:
        metadata.duration = segments[-1].end
    metadata.source_lang = detected_lang

    # ── PASSE 3 : Analyse ──
    analysis = None
    if os.path.exists(analysis_path):
        analysis = load_analysis(analysis_path)
        print(f"\n⏩ Analyse chargée depuis {analysis_path}")
    else:
        analysis = analyze_content(segments, analysis_client, analysis_model,
                                   detected_lang, target_lang, args.context)
        save_analysis(analysis, analysis_path)

    # ── PASSE 4 : Traduction (si source ≠ cible) ──
    if detected_lang != target_lang:
        already_translated = sum(1 for s in segments if s.text_tgt)
        if already_translated < len(segments):
            segments = translate_chunks(segments, analysis, client, claude_model,
                                        detected_lang, target_lang, args.context)
            save_segments(segments, segments_path)
        else:
            print(f"\n⏩ Déjà traduit ({already_translated}/{len(segments)})")
    else:
        print(f"\n⏩ Pas de traduction nécessaire (source = cible = {target_lang})")
        # Copier text vers text_tgt pour uniformiser
        for s in segments:
            if not s.text_tgt:
                s.text_tgt = s.text

    # ── PASSE 5 : Synthèse ──
    target_words = calculate_target_words(metadata.duration, args.pages)
    target_pages = target_words // WORDS_PER_PAGE

    if os.path.exists(md_path):
        print(f"\n⏩ Résumé markdown déjà généré : {md_path}")
        with open(md_path, encoding="utf-8") as f:
            summary_md = f.read()
    else:
        summary_md = generate_summary(segments, analysis, metadata, client,
                                      claude_model, target_words, args.context)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(summary_md)
        print(f"   💾 → {md_path}")

    word_count = len(summary_md.split())
    print(f"   📝 {word_count} mots (~{word_count / WORDS_PER_PAGE:.1f} pages)")

    # ── PASSE 6 : Génération PDF + EPUB + HTML ──
    print(f"\n📦 Génération des documents...")
    pdf_path = f"{base}_{target_lang}_resume.pdf"
    epub_path = f"{base}_{target_lang}_resume.epub"
    html_path = f"{base}_{target_lang}_resume.html"

    build_pdf(summary_md, metadata, pdf_path)
    build_epub(summary_md, metadata, epub_path)

    generate_html = args.html or args.output_dir
    if generate_html:
        build_html(summary_md, metadata, html_path)

    # ── Copie vers output-dir ──
    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in [pdf_path, epub_path, html_path]:
            if os.path.exists(p):
                shutil.copy2(p, out_dir / os.path.basename(p))
                print(f"   📋 Copié → {out_dir / os.path.basename(p)}")

    # ── Résumé final ──
    print("\n" + "=" * 70)
    print("  ✅ TERMINÉ")
    print("=" * 70)
    print(f"   📝 Markdown : {md_path}")
    if os.path.exists(pdf_path):
        size = os.path.getsize(pdf_path) / 1024
        print(f"   📄 PDF      : {pdf_path} ({size:.0f} Ko)")
    if os.path.exists(epub_path):
        size = os.path.getsize(epub_path) / 1024
        print(f"   📚 EPUB     : {epub_path} ({size:.0f} Ko)")
    if os.path.exists(html_path):
        size = os.path.getsize(html_path) / 1024
        print(f"   🌐 HTML     : {html_path} ({size:.0f} Ko)")
    print(f"   📊 {word_count} mots · ~{word_count / WORDS_PER_PAGE:.1f} pages · {metadata.duration / 60:.0f} min source")


if __name__ == "__main__":
    main()
