#!/usr/bin/env python3
"""
Module partagé pour le support des vidéos Apollo Health (Town Halls).

Extrait la transcription horodatée et la vidéo Vimeo depuis les pages
app.apollohealthco.com/town-halls/*, télécharge via yt-dlp, et aligne
la transcription aux segments WhisperX.

Utilisé par : doubler.py, traduire.py, clipper.py, resumer.py
"""

import json
import os
import re
import subprocess
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError


# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ApolloHealthPage:
    """Métadonnées et transcription d'une page Apollo Health Town Hall."""
    title: str = ""
    subtitle: str = ""
    author: str = ""
    published: str = ""
    description: str = ""
    tags: list = field(default_factory=list)
    vimeo_url: str = ""        # URL complète de l'iframe Vimeo
    vimeo_id: str = ""         # ID Vimeo (ex: 828160022)
    vimeo_hash: str = ""       # Hash privé (ex: fb58f3e08b)
    transcript: list = field(default_factory=list)   # [{"start": float, "end": float, "text": "..."}, ...]
    paragraphs: list = field(default_factory=list)    # [{"text": "...", "spans": [...]}, ...]
    url: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# DÉTECTION URL
# ═══════════════════════════════════════════════════════════════════════════════

def is_apollo_url(s: str) -> bool:
    """Détecte si la chaîne est un lien Apollo Health Town Hall."""
    if not s:
        return False
    return bool(re.match(
        r'https?://(app\.)?apollohealthco\.com/town-halls', s))


# ═══════════════════════════════════════════════════════════════════════════════
# COOKIES
# ═══════════════════════════════════════════════════════════════════════════════

def load_cookies(path: Optional[str] = None) -> dict:
    """Charge les cookies depuis un fichier Cookie-Editor JSON.

    Cherche dans l'ordre :
    1. Le chemin explicite (si fourni)
    2. apollo-cookies.json dans le répertoire courant
    3. apollo-cookies.json dans le répertoire du script

    Retourne un dict {nom: valeur} filtré sur le domaine apollohealthco.com.
    """
    candidates = []
    if path:
        candidates.append(path)
    candidates.append("apollo-cookies.json")
    candidates.append(str(Path(__file__).parent / "apollo-cookies.json"))

    cookie_file = None
    for c in candidates:
        if os.path.exists(c):
            cookie_file = c
            break

    if not cookie_file:
        raise FileNotFoundError(
            "apollo-cookies.json introuvable.\n"
            "   → Installez l'extension Cookie-Editor dans votre navigateur\n"
            "   → Connectez-vous à app.apollohealthco.com\n"
            "   → Exportez les cookies (format JSON) dans apollo-cookies.json")

    with open(cookie_file, encoding="utf-8") as f:
        raw = json.load(f)

    # Format Cookie-Editor : liste de dicts avec name, value, domain
    cookies = {}
    for c in raw:
        domain = c.get("domain", "")
        if "apollohealthco.com" in domain:
            cookies[c["name"]] = c["value"]

    if not cookies:
        print(f"   ⚠️  Aucun cookie apollohealthco.com trouvé dans {cookie_file}")

    return cookies


# ═══════════════════════════════════════════════════════════════════════════════
# FETCH PAGE
# ═══════════════════════════════════════════════════════════════════════════════

def _strip_html(text: str) -> str:
    """Supprime toutes les balises HTML et entités d'un texte."""
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace('&nbsp;', ' ').replace('&amp;', '&')
    text = text.replace('&lt;', '<').replace('&gt;', '>')
    text = re.sub(r'&#?\w+;', '', text)  # entités restantes
    return text


def fetch_apollo_page(url: str, cookies: dict) -> ApolloHealthPage:
    """Récupère et parse une page Apollo Health Town Hall.

    Site Rails + Hotwire — structure HTML standard avec :
    - iframe Vimeo pour la vidéo
    - Transcription avec <span data-start="..." data-finish="...">
    """
    cookie_str = "; ".join(f"{k}={v}" for k, v in cookies.items())

    req = Request(url, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
        "Cookie": cookie_str,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9,fr;q=0.8",
    })

    try:
        with urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except HTTPError as e:
        if e.code in (401, 403):
            raise RuntimeError(
                f"HTTP {e.code} — Cookies expirés ou invalides.\n"
                "   → Re-exportez vos cookies depuis votre navigateur (Cookie-Editor)")
        raise

    # Vérifier qu'on n'est pas redirigé vers la page de login
    if 'name="user[email]"' in html or '<form' in html[:3000] and 'sign_in' in html[:3000]:
        raise RuntimeError(
            "Redirigé vers la page de connexion — cookies expirés.\n"
            "   → Re-exportez vos cookies depuis votre navigateur (Cookie-Editor)")

    return _parse_apollo_html(html, url)


def _parse_apollo_html(html: str, url: str) -> ApolloHealthPage:
    """Parse le HTML d'une page Town Hall Apollo Health."""
    page = ApolloHealthPage(url=url)

    # Titre : <h3 class="long-title">...</h3>
    m = re.search(r'<h3[^>]*class="[^"]*long-title[^"]*"[^>]*>(.*?)</h3>', html, re.DOTALL)
    if m:
        page.title = _strip_html(m.group(1)).strip()

    # Sous-titre : <h4> qui suit le h3
    m = re.search(r'<h3[^>]*class="[^"]*long-title[^"]*"[^>]*>.*?</h3>\s*<h4[^>]*>(.*?)</h4>', html, re.DOTALL)
    if m:
        page.subtitle = _strip_html(m.group(1)).strip()

    # Métadonnées : auteur, date publiée
    m = re.search(r'<p[^>]*class="[^"]*article-info[^"]*"[^>]*>(.*?)</p>', html, re.DOTALL)
    if m:
        info = m.group(1)
        am = re.search(r'By:\s*(.*?)(?:<br|$)', info)
        if am:
            page.author = _strip_html(am.group(1)).strip()
        pm = re.search(r'Published on:\s*([\d/]+)', info)
        if pm:
            page.published = pm.group(1).strip()

    # Description : <div class="article-body"> — extraire les <p> uniquement
    m = re.search(r'<div[^>]*class="[^"]*article-body[^"]*"[^>]*>(.*?)</div>', html, re.DOTALL)
    if m:
        body = m.group(1)
        # Extraire le texte des paragraphes seulement (ignorer boutons, liens d'UI)
        desc_parts = re.findall(r'<p[^>]*>(.*?)</p>', body, re.DOTALL)
        page.description = " ".join(_strip_html(p).strip() for p in desc_parts if _strip_html(p).strip())

    # Tags : <span class="badge ..."> dans la section article-tags
    tags_section = re.search(r'<div[^>]*class="[^"]*article-tags[^"]*"[^>]*>(.*?)</div>', html, re.DOTALL)
    if tags_section:
        page.tags = re.findall(r'<span[^>]*class="[^"]*badge[^"]*"[^>]*>(.*?)</span>', tags_section.group(1))
        page.tags = [_strip_html(t).strip() for t in page.tags
                     if _strip_html(t).strip() and len(_strip_html(t).strip()) > 1]

    # Vimeo : <iframe ... src="https://player.vimeo.com/video/ID?h=HASH&...">
    m = re.search(
        r'<iframe[^>]*src="(https://player\.vimeo\.com/video/(\d+)\?h=([a-f0-9]+)[^"]*)"',
        html)
    if m:
        page.vimeo_url = m.group(1)
        page.vimeo_id = m.group(2)
        page.vimeo_hash = m.group(3)
    else:
        # Fallback sans hash privé
        m = re.search(
            r'<iframe[^>]*src="(https://player\.vimeo\.com/video/(\d+)[^"]*)"',
            html)
        if m:
            page.vimeo_url = m.group(1)
            page.vimeo_id = m.group(2)

    # Transcription : <span data-start="12" data-finish="17">texte</span>
    page.transcript, page.paragraphs = _parse_transcript(html)

    return page


def _parse_transcript(html: str) -> tuple:
    """Parse la transcription horodatée d'une page Apollo Health.

    Structure : div.transcript-inner > p > span[data-start][data-finish]

    Retourne (spans, paragraphs) :
    - spans : [{"start": 12.0, "end": 17.0, "text": "..."}, ...]  (liste plate)
    - paragraphs : [{"text": "...", "spans": [indices...]}, ...]    (regroupements par <p>)
    """
    # Extraire le bloc transcription
    m = re.search(
        r'<div[^>]*class="[^"]*transcript-inner[^"]*"[^>]*>(.*?)</div>',
        html, re.DOTALL)
    if not m:
        return [], []

    transcript_html = m.group(1)

    # Extraire tous les spans avec timing
    spans = []
    span_pattern = re.compile(
        r'<span[^>]*data-start="(\d+)"[^>]*data-finish="(\d+)"[^>]*>(.*?)</span>',
        re.DOTALL)

    for sm in span_pattern.finditer(transcript_html):
        start = float(sm.group(1))
        end = float(sm.group(2))
        text = _strip_html(sm.group(3)).strip()
        if text:
            spans.append({"start": start, "end": end, "text": text})

    # Regrouper par paragraphe <p>
    paragraphs = []
    p_pattern = re.compile(r'<p[^>]*>(.*?)</p>', re.DOTALL)
    global_idx = 0

    for pm in p_pattern.finditer(transcript_html):
        p_html = pm.group(1)
        p_spans = list(span_pattern.finditer(p_html))
        if p_spans:
            p_text = " ".join(_strip_html(s.group(3)).strip() for s in p_spans if _strip_html(s.group(3)).strip())
            span_indices = list(range(global_idx, global_idx + len(p_spans)))
            paragraphs.append({"text": p_text, "spans": span_indices})
            global_idx += len(p_spans)

    return spans, paragraphs


# ═══════════════════════════════════════════════════════════════════════════════
# TÉLÉCHARGEMENT VIDÉO
# ═══════════════════════════════════════════════════════════════════════════════

def download_apollo_video(page: ApolloHealthPage, output_dir: str = ".") -> str:
    """Télécharge la vidéo Vimeo via yt-dlp. Retourne le chemin du MP4.

    Utilise le titre de la page pour nommer le fichier.
    Si le fichier existe déjà, skip le téléchargement.
    """
    if not page.vimeo_url:
        raise ValueError(
            "Pas de vidéo Vimeo sur cette page Apollo Health.\n"
            "   → Vérifiez que la page contient bien une vidéo.")

    if not shutil.which("yt-dlp"):
        raise RuntimeError("yt-dlp n'est pas installé (nécessaire pour le téléchargement Vimeo).")

    # Nettoyer le titre pour en faire un nom de fichier
    safe_title = re.sub(r'[^\w\s\-]', '', page.title)
    safe_title = re.sub(r'\s+', '_', safe_title).strip('_')
    if not safe_title:
        safe_title = "apollo_video"
    safe_title = safe_title[:150]

    output_path = os.path.join(output_dir, f"{safe_title}.mp4")

    # Skip si déjà téléchargé
    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        if size > 1_000_000:  # > 1 Mo = probablement complet
            print(f"   ✅ Vidéo déjà téléchargée : {output_path} ({size/1e6:.1f} Mo)")
            return output_path

    print(f"\n📥 Téléchargement vidéo Apollo Health (Vimeo)...")
    print(f"   Vimeo ID : {page.vimeo_id}")
    print(f"   Dest     : {output_path}")

    # Construire l'URL de téléchargement
    # Referer Apollo Health nécessaire pour que Vimeo accepte le flux
    referer = "https://app.apollohealthco.com/"
    vimeo_dl_url = f"https://player.vimeo.com/video/{page.vimeo_id}"
    if page.vimeo_hash:
        vimeo_dl_url += f"?h={page.vimeo_hash}"

    cmd = [
        "yt-dlp",
        "--referer", referer,
        "-f", "bestvideo+bestaudio/best",
        "--merge-output-format", "mp4",
        "-o", output_path,
        vimeo_dl_url
    ]

    for attempt in range(2):
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0 and os.path.exists(output_path):
                size = os.path.getsize(output_path)
                print(f"   ✅ Téléchargé : {size/1e6:.1f} Mo")
                return output_path
            if attempt == 0:
                print(f"   ⚠️  Tentative 1 échouée, retry...")
                print(f"   stderr: {result.stderr[-300:]}")
                continue
            raise RuntimeError(
                f"yt-dlp a échoué (code {result.returncode}).\n"
                f"   stderr: {result.stderr[-500:]}")
        except subprocess.TimeoutExpired:
            if attempt == 0:
                print(f"   ⚠️  Timeout, retry...")
                continue
            raise RuntimeError("Téléchargement Vimeo timeout après 2 tentatives.")

    return output_path  # ne devrait pas arriver


# ═══════════════════════════════════════════════════════════════════════════════
# SIDECAR META
# ═══════════════════════════════════════════════════════════════════════════════

def save_apollo_meta(page: ApolloHealthPage, video_path: str) -> str:
    """Sauvegarde les métadonnées à côté de la vidéo (sidecar JSON)."""
    base = Path(video_path).stem
    meta_path = str(Path(video_path).parent / f"{base}_apollo_meta.json")

    meta = {
        "title": page.title,
        "subtitle": page.subtitle,
        "author": page.author,
        "published": page.published,
        "description": page.description,
        "tags": page.tags,
        "vimeo_url": page.vimeo_url,
        "vimeo_id": page.vimeo_id,
        "vimeo_hash": page.vimeo_hash,
        "transcript": page.transcript,
        "url": page.url,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"   💾 Métadonnées sauvegardées : {meta_path}")
    return meta_path


def load_apollo_meta(video_path: str) -> Optional[ApolloHealthPage]:
    """Charge le sidecar _apollo_meta.json si présent."""
    base = Path(video_path).stem
    meta_path = str(Path(video_path).parent / f"{base}_apollo_meta.json")

    if not os.path.exists(meta_path):
        return None

    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        page = ApolloHealthPage(
            title=meta.get("title", ""),
            subtitle=meta.get("subtitle", ""),
            author=meta.get("author", ""),
            published=meta.get("published", ""),
            description=meta.get("description", ""),
            tags=meta.get("tags", []),
            vimeo_url=meta.get("vimeo_url", ""),
            vimeo_id=meta.get("vimeo_id", ""),
            vimeo_hash=meta.get("vimeo_hash", ""),
            transcript=meta.get("transcript", []),
            url=meta.get("url", ""),
        )
        print(f"   🏥 Métadonnées Apollo Health chargées depuis {meta_path}")
        return page
    except (json.JSONDecodeError, KeyError) as e:
        print(f"   ⚠️  Erreur lecture {meta_path} : {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# ALIGNEMENT TRANSCRIPTION → SEGMENTS WHISPERX
# ═══════════════════════════════════════════════════════════════════════════════

def align_transcript_to_segments(transcript: list, segments: list) -> list:
    """Aligne la transcription horodatée Apollo Health aux segments WhisperX.

    La transcription Apollo a des timestamps en secondes entières — on les
    utilise pour :
    1. Corriger le texte WhisperX (noms propres, termes médicaux)
    2. Améliorer la segmentation via le texte officiel

    Stratégie : pour chaque segment WhisperX, trouver les spans de la
    transcription qui chevauchent temporellement, puis remplacer le texte
    si l'alignement est bon.

    Retourne les segments modifiés.
    """
    if not transcript:
        return segments

    print(f"\n🏥 Alignement transcription Apollo Health → segments WhisperX...")
    print(f"   Spans transcription : {len(transcript)}")
    print(f"   Segments WhisperX   : {len(segments)}")

    # Construire un index temporel des spans
    # Chaque span : {"start": float, "end": float, "text": str}
    corrections = 0

    for seg in segments:
        seg_start = seg.start if hasattr(seg, 'start') else 0
        seg_end = seg.end if hasattr(seg, 'end') else 0
        seg_text = seg.text if hasattr(seg, 'text') else ""

        if not seg_text or seg_end <= seg_start:
            continue

        # Trouver les spans qui chevauchent ce segment
        overlapping = []
        for sp in transcript:
            sp_start = sp["start"]
            sp_end = sp["end"]
            # Chevauchement : max(starts) < min(ends)
            overlap_start = max(seg_start, sp_start)
            overlap_end = min(seg_end, sp_end)
            if overlap_start < overlap_end:
                overlap_dur = overlap_end - overlap_start
                overlapping.append((overlap_dur, sp))

        if not overlapping:
            continue

        # Trier par durée de chevauchement décroissante
        overlapping.sort(key=lambda x: x[0], reverse=True)

        # Construire le texte de la transcription pour ce segment
        # Prendre les spans dont le chevauchement couvre > 50% de leur durée
        trans_parts = []
        for overlap_dur, sp in overlapping:
            sp_dur = sp["end"] - sp["start"]
            if sp_dur > 0 and overlap_dur / sp_dur > 0.5:
                trans_parts.append(sp["text"])

        if not trans_parts:
            continue

        trans_text = " ".join(trans_parts)

        # Comparer les textes (normalisés) pour décider de la correction
        seg_norm = _normalize_for_compare(seg_text)
        trans_norm = _normalize_for_compare(trans_text)

        if not seg_norm or not trans_norm:
            continue

        # Calculer la similarité par mots
        seg_words = seg_norm.split()
        trans_words = trans_norm.split()

        if not seg_words or not trans_words:
            continue

        # Compter les mots communs (ordre non important)
        common = set(seg_words) & set(trans_words)
        similarity = len(common) / max(len(seg_words), len(trans_words))

        # Si > 50% de mots en commun, on prend le texte de la transcription
        # (meilleur pour les noms propres et termes médicaux)
        if similarity > 0.50:
            seg.text = trans_text
            corrections += 1

    print(f"   Corrections texte   : {corrections}/{len(segments)} segments")

    return segments


def _normalize_for_compare(text: str) -> str:
    """Normalise un texte pour comparaison (minuscule, sans ponctuation)."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXTE CLAUDE
# ═══════════════════════════════════════════════════════════════════════════════

def build_apollo_context(page: ApolloHealthPage) -> str:
    """Construit un bloc de contexte à prépendre au contexte utilisateur pour Claude."""
    lines = []
    if page.title:
        lines.append(f"Titre : {page.title}")
    if page.subtitle:
        lines.append(f"Sous-titre : {page.subtitle}")
    if page.author:
        lines.append(f"Auteur : {page.author}")
    if page.description:
        lines.append(f"Description : {page.description}")
    if page.tags:
        lines.append(f"Thèmes : {', '.join(page.tags)}")

    if not lines:
        return ""

    return "[Source : Apollo Health]\n" + "\n".join(lines)
