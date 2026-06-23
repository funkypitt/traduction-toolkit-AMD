#!/usr/bin/env python3
"""
Module partagé pour le support des vidéos Epoch Times (EpochTV).

Extrait la transcription avec attribution de locuteurs depuis la page web,
télécharge la vidéo HLS, et aligne la transcription aux segments WhisperX
pour corriger les noms propres et attribuer les locuteurs sans Pyannote.

Utilisé par : doubler.py, traduire.py, clipper.py
"""

import bisect
import difflib
import json
import os
import re
import subprocess
import sys
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
class EpochTimesPage:
    """Métadonnées et transcription d'une page Epoch Times."""
    title: str = ""
    excerpt: str = ""
    authors: list = field(default_factory=list)
    video_url: str = ""          # HLS m3u8
    audio_url: str = ""          # MP3 direct (Captivate.fm)
    duration: float = 0.0
    transcript: list = field(default_factory=list)   # [{"speaker": "...", "text": "..."}, ...]
    speakers: list = field(default_factory=list)      # noms uniques, ordre d'apparition
    raw_post: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# DÉTECTION URL
# ═══════════════════════════════════════════════════════════════════════════════

def is_epochtimes_url(s: str) -> bool:
    """Détecte si la chaîne est un lien Epoch Times."""
    if not s:
        return False
    return bool(re.match(
        r'https?://(www\.)?(theepochtimes\.com|epochtimes\.com)/', s))


# ═══════════════════════════════════════════════════════════════════════════════
# COOKIES
# ═══════════════════════════════════════════════════════════════════════════════

def load_cookies(path: Optional[str] = None) -> dict:
    """Charge les cookies depuis un fichier Cookie-Editor JSON.

    Cherche dans l'ordre :
    1. Le chemin explicite (si fourni)
    2. epoch-cookies.json dans le répertoire courant
    3. epoch-cookies.json dans le répertoire du script

    Retourne un dict {nom: valeur} filtré sur le domaine .theepochtimes.com.
    """
    candidates = []
    if path:
        candidates.append(path)
    candidates.append("epoch-cookies.json")
    candidates.append(str(Path(__file__).parent / "epoch-cookies.json"))

    cookie_file = None
    for c in candidates:
        if os.path.exists(c):
            cookie_file = c
            break

    if not cookie_file:
        raise FileNotFoundError(
            "epoch-cookies.json introuvable.\n"
            "   → Installez l'extension Cookie-Editor dans votre navigateur\n"
            "   → Connectez-vous à theepochtimes.com\n"
            "   → Exportez les cookies (format JSON) dans epoch-cookies.json")

    with open(cookie_file, encoding="utf-8") as f:
        raw = json.load(f)

    # Format Cookie-Editor : liste de dicts avec name, value, domain
    cookies = {}
    for c in raw:
        domain = c.get("domain", "")
        if "theepochtimes.com" in domain:
            cookies[c["name"]] = c["value"]

    if not cookies:
        print(f"   ⚠️  Aucun cookie theepochtimes.com trouvé dans {cookie_file}")

    return cookies


# ═══════════════════════════════════════════════════════════════════════════════
# FETCH PAGE
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_epoch_page(url: str, cookies: dict) -> EpochTimesPage:
    """Récupère et parse une page Epoch Times.

    Supporte deux formats :
    - Legacy : <script id="__NEXT_DATA__"> → JSON → props.pageProps.post
    - App Router (RSC) : self.__next_f.push() → post JSON embarqué
    """
    # Construire le header Cookie
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

    post = None

    # Méthode 1 : __NEXT_DATA__ (Pages Router)
    m = re.search(
        r'<script\s+id="__NEXT_DATA__"\s+type="application/json">(.*?)</script>',
        html, re.DOTALL)
    if m:
        try:
            next_data = json.loads(m.group(1))
            post = (next_data.get("props", {})
                             .get("pageProps", {})
                             .get("post", {}))
        except json.JSONDecodeError:
            pass

    # Méthode 2 : RSC payload (App Router) — self.__next_f.push()
    if not post:
        post = _extract_post_from_rsc(html)

    if not post:
        raise ValueError(
            "Page Epoch Times sans données structurées.\n"
            "   → Vérifiez l'URL et vos cookies.")

    return _build_page_from_post(post)


def _extract_post_from_rsc(html: str) -> Optional[dict]:
    """Extrait l'objet post depuis le payload RSC (React Server Components).

    Le post est embarqué dans un appel self.__next_f.push([1,"..."]) sous forme
    de JSON doublement échappé (\\" au lieu de "), contenant \\"post\\":{\\"id\\":...}.
    """
    # Dans le RSC, les guillemets sont doublement échappés : \\"
    # Chercher le marqueur du post (plus spécifique d'abord)
    for marker in ['\\"post\\":{\\"id\\":', '"post":{"id":']:
        idx = html.find(marker)
        if idx >= 0:
            break
    else:
        return None

    # Trouver le script contenant ce marqueur
    script_start = html.rfind('<script>', 0, idx)
    script_end = html.find('</script>', idx)
    if script_start < 0 or script_end < 0:
        return None

    script_content = html[script_start + len('<script>'):script_end]

    # Extraire la chaîne du push : self.__next_f.push([1,"PAYLOAD"])
    m = re.search(r'self\.__next_f\.push\(\[1,"(.*)"\]\)', script_content, re.DOTALL)
    if not m:
        return None

    # Le payload est une chaîne JS échappée — la dé-sérialiser
    js_string = m.group(1)
    try:
        # json.loads d'une chaîne entre guillemets pour dé-échapper
        payload = json.loads('"' + js_string + '"')
    except json.JSONDecodeError:
        # Fallback : dé-échappement manuel
        payload = js_string.replace('\\"', '"').replace('\\/', '/').replace('\\\\', '\\')

    # Maintenant chercher "post":{"id":... dans le payload dé-échappé
    post_marker = '"post":{"id":'
    pidx = payload.find(post_marker)
    if pidx < 0:
        return None

    # Extraire le JSON du post en comptant les accolades
    json_start = pidx + len('"post":')
    depth = 0
    i = json_start
    while i < len(payload) and i < json_start + 500_000:
        c = payload[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                break
        elif c == '"':
            # Sauter la chaîne entière (contenu entre guillemets)
            i += 1
            while i < len(payload):
                if payload[i] == '\\':
                    i += 1  # sauter le caractère échappé
                elif payload[i] == '"':
                    break
                i += 1
        i += 1

    if depth != 0:
        return None

    raw_json = payload[json_start:i + 1]

    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        return None


def _build_page_from_post(post: dict) -> EpochTimesPage:
    """Construit un EpochTimesPage à partir de l'objet post extrait."""
    page = EpochTimesPage(raw_post=post)

    # Titre et excerpt
    page.title = post.get("title", "").strip()
    page.excerpt = _strip_html(post.get("excerpt", "")).strip()

    # Auteurs
    authors_raw = post.get("authors", [])
    if isinstance(authors_raw, list):
        for a in authors_raw:
            name = a.get("name", "") if isinstance(a, dict) else str(a)
            if name:
                page.authors.append(name)

    # Vidéo HLS
    video = post.get("video", {})
    if isinstance(video, dict):
        page.video_url = video.get("url", "")
        page.duration = float(video.get("duration", 0) or 0)

    # Audio MP3
    audio = post.get("audio", {})
    if isinstance(audio, dict):
        page.audio_url = audio.get("url", "")

    # Transcription
    content = post.get("content", [])
    if isinstance(content, list):
        page.transcript, page.speakers = _parse_transcript(content)

    return page


# ═══════════════════════════════════════════════════════════════════════════════
# PARSING TRANSCRIPTION
# ═══════════════════════════════════════════════════════════════════════════════

def _strip_html(text: str) -> str:
    """Supprime toutes les balises HTML d'un texte."""
    return re.sub(r'<[^>]+>', '', text)


def _parse_transcript(content: list) -> tuple:
    """Parse le contenu HTML de la transcription Epoch Times.

    Structure typique :
    - Paragraphe avec <b>Nom Locuteur:</b> → nouveau locuteur
    - <span>texte du locuteur</span> dans le même paragraphe ou les suivants

    Retourne (paragraphs, speakers) :
    - paragraphs : [{"speaker": "Jan Jekielek", "text": "..."}, ...]
    - speakers   : ["Jan Jekielek", "Dr. Robert Malone", ...]  (ordre d'apparition)
    """
    paragraphs = []
    speakers = []
    current_speaker = ""

    for item in content:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type", "")
        text_raw = item.get("text", "")

        if not text_raw or item_type not in ("p", "paragraph", "div", ""):
            # Ignorer les éléments non-textuels (images, embeds, etc.)
            if text_raw and not item_type:
                # Certaines pages ont des items sans type
                pass
            else:
                continue

        # Détecter un changement de locuteur : <b>Nom:</b> ou **Nom:**
        speaker_match = re.match(r'<b>(.+?):</b>', text_raw)
        if not speaker_match:
            # Essayer aussi le format sans balise : "Nom Locuteur:" en début
            speaker_match = re.match(r'<strong>(.+?):</strong>', text_raw)

        if speaker_match:
            candidate = _strip_html(speaker_match.group(1)).strip()
            # Vérifier que c'est bien un nom (pas trop long, pas du texte)
            if candidate and len(candidate) < 60:
                current_speaker = candidate
                if current_speaker not in speakers:
                    speakers.append(current_speaker)
                # Le texte qui suit le nom du locuteur dans le même paragraphe
                text_after = text_raw[speaker_match.end():]
                text_clean = _strip_html(text_after).strip()
                if text_clean:
                    paragraphs.append({
                        "speaker": current_speaker,
                        "text": text_clean
                    })
                continue

        # Paragraphe normal — attribuer au locuteur courant
        text_clean = _strip_html(text_raw).strip()
        if text_clean:
            paragraphs.append({
                "speaker": current_speaker,
                "text": text_clean
            })

    return paragraphs, speakers


# ═══════════════════════════════════════════════════════════════════════════════
# TÉLÉCHARGEMENT VIDÉO
# ═══════════════════════════════════════════════════════════════════════════════

def download_epoch_video(page: EpochTimesPage, output_dir: str = ".") -> str:
    """Télécharge la vidéo HLS (m3u8) via ffmpeg. Retourne le chemin du MP4.

    Utilise le titre de la page pour nommer le fichier.
    Si le fichier existe déjà, skip le téléchargement.
    """
    if not page.video_url:
        raise ValueError(
            "Pas de vidéo sur cette page Epoch Times.\n"
            "   → Vérifiez que l'article contient bien une vidéo EpochTV.")

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg n'est pas installé (nécessaire pour le téléchargement HLS).")

    # Nettoyer le titre pour en faire un nom de fichier
    safe_title = re.sub(r'[^\w\s\-]', '', page.title)
    safe_title = re.sub(r'\s+', '_', safe_title).strip('_')
    if not safe_title:
        safe_title = "epoch_video"
    safe_title = safe_title[:150]  # limiter la longueur

    output_path = os.path.join(output_dir, f"{safe_title}.mp4")

    # Skip si déjà téléchargé
    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        if size > 1_000_000:  # > 1 Mo = probablement complet
            print(f"   ✅ Vidéo déjà téléchargée : {output_path} ({size/1e6:.1f} Mo)")
            return output_path

    print(f"\n📥 Téléchargement vidéo Epoch Times (HLS)...")
    print(f"   URL  : {page.video_url[:80]}...")
    print(f"   Dest : {output_path}")

    # Brightchat HLS nécessite un Referer pour servir le flux
    referer = "https://vod.brightchat.com/"
    cmd = [
        "ffmpeg", "-y",
        "-headers", f"Referer: {referer}\r\n",
        "-i", page.video_url,
        "-c", "copy",
        "-bsf:a", "aac_adtstoasc",
        output_path
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
                continue
            raise RuntimeError(
                f"ffmpeg a échoué (code {result.returncode}).\n"
                f"   stderr: {result.stderr[-500:]}")
        except subprocess.TimeoutExpired:
            if attempt == 0:
                print(f"   ⚠️  Timeout, retry...")
                continue
            raise RuntimeError("Téléchargement HLS timeout après 2 tentatives.")

    return output_path  # ne devrait pas arriver


# ═══════════════════════════════════════════════════════════════════════════════
# SIDECAR META (pour le daemon)
# ═══════════════════════════════════════════════════════════════════════════════

def save_epoch_meta(page: EpochTimesPage, video_path: str) -> str:
    """Sauvegarde les métadonnées à côté de la vidéo (sidecar JSON).

    Permet au script invoqué de retrouver la transcription sans re-fetch.
    """
    base = Path(video_path).stem
    meta_path = str(Path(video_path).parent / f"{base}_epoch_meta.json")

    meta = {
        "title": page.title,
        "excerpt": page.excerpt,
        "authors": page.authors,
        "video_url": page.video_url,
        "audio_url": page.audio_url,
        "duration": page.duration,
        "transcript": page.transcript,
        "speakers": page.speakers,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"   💾 Métadonnées sauvegardées : {meta_path}")
    return meta_path


def load_epoch_meta(video_path: str) -> Optional[EpochTimesPage]:
    """Charge le sidecar _epoch_meta.json si présent.

    Retourne un EpochTimesPage ou None.
    """
    base = Path(video_path).stem
    meta_path = str(Path(video_path).parent / f"{base}_epoch_meta.json")

    if not os.path.exists(meta_path):
        return None

    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        page = EpochTimesPage(
            title=meta.get("title", ""),
            excerpt=meta.get("excerpt", ""),
            authors=meta.get("authors", []),
            video_url=meta.get("video_url", ""),
            audio_url=meta.get("audio_url", ""),
            duration=meta.get("duration", 0.0),
            transcript=meta.get("transcript", []),
            speakers=meta.get("speakers", []),
        )
        print(f"   📰 Métadonnées Epoch Times chargées depuis {meta_path}")
        return page
    except (json.JSONDecodeError, KeyError) as e:
        print(f"   ⚠️  Erreur lecture {meta_path} : {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# ALIGNEMENT TRANSCRIPTION → SEGMENTS WHISPERX
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_word(w: str) -> str:
    """Normalise un mot pour la comparaison (minuscule, sans ponctuation)."""
    return re.sub(r'[^\w]', '', w.lower())


# Titres de civilité à ignorer lors du rapprochement des noms de locuteurs.
_HONORIFICS = {
    "mr", "mrs", "ms", "miss", "mister", "dr", "drs", "prof", "professor",
    "sir", "madam", "madame", "mme", "mlle", "rev", "hon",
}


def _name_tokens(name: str) -> set:
    """Tokens significatifs d'un nom (minuscule, sans civilité ni ponctuation).

    Ex : "Ms. Moncrieff" → {"moncrieff"} ; "Joanna Moncrieff" → {"joanna", "moncrieff"}.
    """
    toks = set()
    for w in name.split():
        wn = re.sub(r'[^\w]', '', w.lower())
        if wn and wn not in _HONORIFICS:
            toks.add(wn)
    return toks


def _canonicalize_speakers(speakers: list, transcript: list) -> tuple:
    """Fusionne les variantes d'un même locuteur (civilité, prénom omis...).

    Epoch Times tague parfois la même personne sous plusieurs libellés
    ("Joanna Moncrieff" puis "Ms. Moncrieff"). On les rapproche par sous-ensemble
    de tokens (intersection non vide) pour éviter de créer plusieurs voix TTS.

    Retourne (transcript_réécrit, speakers_canoniques, alias_map).
    alias_map : {libellé_original: représentant_canonique}.
    """
    names = list(speakers)
    for p in transcript:
        s = p.get("speaker", "")
        if s and s not in names:
            names.append(s)

    order = {n: i for i, n in enumerate(names)}
    toks = {n: _name_tokens(n) for n in names}

    # Représentants établis en premier sur les noms les plus complets.
    ranked = sorted(names, key=lambda n: (-len(toks[n]), order[n]))
    reps = []
    alias = {}
    for n in ranked:
        tn = toks[n]
        matched = None
        for r in reps:
            tr = toks[r]
            if not tn or not tr:
                continue
            if tn <= tr or tr <= tn:   # même personne (l'un sous-ensemble de l'autre)
                matched = r
                break
        if matched:
            alias[n] = matched
        else:
            reps.append(n)
            alias[n] = n

    canon_speakers = sorted(reps, key=lambda n: order[n])
    new_transcript = [{**p, "speaker": alias.get(p.get("speaker", ""), p.get("speaker", ""))}
                      for p in transcript]

    merged = {n: r for n, r in alias.items() if n != r}
    if merged:
        print(f"   🔗 Locuteurs fusionnés : "
              + ", ".join(f"{n} → {r}" for n, r in merged.items()))

    return new_transcript, canon_speakers, alias


def align_transcript_to_segments(transcript: list, segments: list,
                                  speakers: list = None) -> tuple:
    """Aligne la transcription Epoch Times aux segments WhisperX.

    1. Aplatit la transcription en mots tagués (mot, para_idx, speaker)
    2. Aplatit les segments en mots (mot, seg_idx)
    3. SequenceMatcher sur les séquences de mots normalisés
    4. Vote majoritaire par segment → attribution du locuteur
    5. Relecture : remplacement des mots WhisperX par ceux de la transcription

    Retourne (segments_modifiés, speaker_name_map).
    speaker_name_map : {"SPEAKER_00": "Jan Jekielek", ...}
    """
    if not transcript:
        return segments, {}

    print(f"\n📰 Alignement transcription Epoch Times → segments WhisperX...")

    # 0. Fusionner les variantes d'un même locuteur (civilité, prénom omis...)
    transcript, speakers, _alias = _canonicalize_speakers(speakers or [], transcript)

    # 1. Aplatir la transcription en mots tagués
    trans_words = []  # [(mot_normalisé, para_idx, speaker, mot_original)]
    for para_idx, para in enumerate(transcript):
        speaker = para.get("speaker", "")
        text = para.get("text", "")
        for word in text.split():
            norm = _normalize_word(word)
            if norm:
                trans_words.append((norm, para_idx, speaker, word))

    # 2. Aplatir les segments WhisperX en mots
    seg_words = []  # [(mot_normalisé, seg_idx, word_idx_in_text)]
    seg_word_positions = []  # positions dans le texte original pour remplacement
    for seg_idx, seg in enumerate(segments):
        text = seg.text if hasattr(seg, 'text') else ""
        words = text.split()
        for word_idx, word in enumerate(words):
            norm = _normalize_word(word)
            if norm:
                seg_words.append((norm, seg_idx, word_idx))
                seg_word_positions.append((seg_idx, word_idx, word))

    if not trans_words or not seg_words:
        print(f"   ⚠️  Transcription ou segments vides — alignement impossible")
        return segments, {}

    # 3. SequenceMatcher sur les mots normalisés
    trans_norms = [w[0] for w in trans_words]
    seg_norms = [w[0] for w in seg_words]

    matcher = difflib.SequenceMatcher(None, seg_norms, trans_norms, autojunk=False)
    matching_blocks = matcher.get_matching_blocks()

    total_matched = sum(b.size for b in matching_blocks)
    match_ratio = total_matched / len(seg_norms) if seg_norms else 0

    print(f"   Mots transcription : {len(trans_words)}")
    print(f"   Mots WhisperX      : {len(seg_words)}")
    print(f"   Mots alignés       : {total_matched} ({match_ratio:.0%})")

    if match_ratio < 0.30:
        print(f"   ⚠️  Alignement très faible ({match_ratio:.0%}) — "
              f"transcription probablement incompatible, abandon")
        return segments, {}

    weak_alignment = match_ratio < 0.50
    if weak_alignment:
        print(f"   ⚠️  Alignement faible ({match_ratio:.0%}) — relecture partielle seulement")

    # 4. Construire le mapping seg_word_idx → trans_word_idx pour chaque bloc
    seg_to_trans = {}  # seg_word_idx → trans_word_idx
    for block in matching_blocks:
        for i in range(block.size):
            seg_to_trans[block.a + i] = block.b + i

    # 5. Vote majoritaire par segment → locuteur
    from collections import Counter
    seg_speaker_votes = {}  # seg_idx → Counter des locuteurs

    for seg_word_idx, trans_word_idx in seg_to_trans.items():
        seg_idx = seg_words[seg_word_idx][1]
        speaker = trans_words[trans_word_idx][2]
        if speaker:
            if seg_idx not in seg_speaker_votes:
                seg_speaker_votes[seg_idx] = Counter()
            seg_speaker_votes[seg_idx][speaker] += 1

    # Mapper les noms de locuteurs → SPEAKER_00, SPEAKER_01...
    speaker_order = speakers if speakers else []
    # Compléter avec les locuteurs trouvés dans les votes
    for seg_idx in sorted(seg_speaker_votes.keys()):
        top_speaker = seg_speaker_votes[seg_idx].most_common(1)
        if top_speaker:
            name = top_speaker[0][0]
            if name and name not in speaker_order:
                speaker_order.append(name)

    name_to_id = {}
    for i, name in enumerate(speaker_order):
        name_to_id[name] = f"SPEAKER_{i:02d}"

    speaker_name_map = {v: k for k, v in name_to_id.items()}

    # Attribuer les locuteurs aux segments
    speakers_assigned = 0
    for seg_idx, votes in seg_speaker_votes.items():
        if seg_idx < len(segments):
            top = votes.most_common(1)
            if top:
                name = top[0][0]
                spk_id = name_to_id.get(name, "SPEAKER_00")
                segments[seg_idx].speaker = spk_id
                speakers_assigned += 1

    # ── Combler les segments sans vote ──────────────────────────────────────
    # L'intro/outro est un montage : extraits (« teasers ») rejoués hors ordre +
    # narration de l'hôte. L'alignement global (monotone) ne sait pas les placer.
    # Trois recours, dans l'ordre :
    #   1. Recherche locale du texte dans les paragraphes ATTRIBUÉS (un teaser
    #      rejoué retrouve le locuteur qui le prononce dans le corps).
    #   2. Intro/outro restante → narration de l'hôte (entretien à 2 locuteurs :
    #      l'invité répond à l'accueil en premier, donc le premier locuteur tagué
    #      est l'invité et l'hôte est l'autre).
    #   3. Trou interne (glitch WhisperX) → plus proche voisin déjà attribué.
    voted = sorted(seg_speaker_votes.keys())

    # Paragraphes attribués → listes de tokens, pour la recherche locale.
    # On exige une correspondance VERBATIM (long bloc contigu) et non un simple
    # recouvrement de vocabulaire : la narration de l'hôte partage les mots du
    # domaine (« dépression », « sérotonine ») avec les propos de l'invité sans
    # être un extrait rejoué. Seul un teaser est un bloc identique consécutif.
    attr_paras = [
        ([_normalize_word(w) for w in p["text"].split() if _normalize_word(w)],
         p["speaker"])
        for p in transcript if p.get("speaker")
    ]

    def _lookup_speaker(text):
        stoks = [_normalize_word(w) for w in text.split() if _normalize_word(w)]
        if len(stoks) < 4:        # trop court → trop de faux positifs
            return None
        best, best_run = None, 0
        for ptoks, spk in attr_paras:
            if not ptoks:
                continue
            m = difflib.SequenceMatcher(None, stoks, ptoks, autojunk=False) \
                .find_longest_match(0, len(stoks), 0, len(ptoks))
            if m.size > best_run:
                best_run, best = m.size, spk
        # Teaser = bloc verbatim. Deux signatures sûres :
        #   - bloc ≥4 mots couvrant ≥70 % du segment (court clip intégral) ;
        #   - bloc ≥12 mots consécutifs (la narration ne reproduit jamais 12 mots
        #     identiques d'affilée — recouvrement de vocabulaire ≠ citation).
        if (best_run >= 4 and best_run >= 0.7 * len(stoks)) or best_run >= 12:
            return best
        return None

    # Détection de l'hôte (entretien à 2 locuteurs).
    first_tagged = next((p["speaker"] for p in transcript if p.get("speaker")), None)
    canon_present = [n for n in (speakers or []) if n]
    host_name = None
    if first_tagged and len(set(canon_present)) == 2:
        others = [n for n in canon_present if n != first_tagged]
        if others:
            host_name = others[0]

    first_voted = voted[0] if voted else 0
    last_voted = voted[-1] if voted else len(segments) - 1

    filled_match = filled_host = filled_neighbor = 0
    for i in range(len(segments)):
        if i in seg_speaker_votes:
            continue
        spk_name = _lookup_speaker(segments[i].text)
        if spk_name:
            filled_match += 1
        elif host_name and (i < first_voted or i > last_voted):
            spk_name = host_name
            filled_host += 1
        elif voted:
            # Trou interne (glitch WhisperX). On ne devine jamais en travers d'une
            # frontière : on ne comble que si les segments votés de part et d'autre
            # désignent le même locuteur ; sinon plus proche voisin par défaut.
            pos = bisect.bisect_left(voted, i)
            prev_v = voted[pos - 1] if pos > 0 else None
            next_v = voted[pos] if pos < len(voted) else None
            prev_spk = seg_speaker_votes[prev_v].most_common(1)[0][0] if prev_v is not None else None
            next_spk = seg_speaker_votes[next_v].most_common(1)[0][0] if next_v is not None else None
            if prev_spk and prev_spk == next_spk:
                spk_name = prev_spk
            else:
                nearest = min(voted, key=lambda j: abs(j - i))
                spk_name = seg_speaker_votes[nearest].most_common(1)[0][0]
            if spk_name:
                filled_neighbor += 1
        if spk_name:
            segments[i].speaker = name_to_id.get(spk_name, segments[i].speaker)

    filled = filled_match + filled_host + filled_neighbor
    detail = ""
    if filled:
        parts = []
        if filled_match:
            parts.append(f"{filled_match} par texte")
        if filled_host:
            parts.append(f"{filled_host} narration hôte")
        if filled_neighbor:
            parts.append(f"{filled_neighbor} voisinage")
        detail = f" (+{filled} comblés : {', '.join(parts)})"

    print(f"   Locuteurs assignés : {speakers_assigned}/{len(segments)} segments{detail}")
    for spk_id, name in sorted(speaker_name_map.items()):
        print(f"     {spk_id} → {name}")

    # 6. Relecture : remplacer les mots WhisperX par la transcription
    #    Seulement là où l'alignement est fiable (contexte d'au moins 2 mots adjacents)
    corrections = 0

    # Identifier les blocs d'au moins 2 mots consécutifs
    for block in matching_blocks:
        if block.size < 2:
            continue  # pas assez fiable pour corriger

        for i in range(block.size):
            seg_word_idx = block.a + i
            trans_word_idx = block.b + i

            seg_idx = seg_words[seg_word_idx][1]
            word_idx = seg_words[seg_word_idx][2]

            orig_word = seg_word_positions[seg_word_idx][2]
            trans_word = trans_words[trans_word_idx][3]

            # Ne corriger que si les mots diffèrent (casse ou contenu)
            if orig_word != trans_word and _normalize_word(orig_word) == _normalize_word(trans_word):
                # Même mot, casse différente → prendre la version transcription
                # (ex: "malone" → "Malone")
                if seg_idx < len(segments):
                    old_text = segments[seg_idx].text
                    words_list = old_text.split()
                    if word_idx < len(words_list):
                        words_list[word_idx] = trans_word
                        segments[seg_idx].text = " ".join(words_list)
                        corrections += 1

    # Relecture plus agressive si alignement fort : corriger aussi les mots
    # dont la forme normalisée diffère (erreurs WhisperX)
    if not weak_alignment:
        for block in matching_blocks:
            if block.size < 3:
                continue  # besoin de plus de contexte pour corriger des vrais mots différents

            for i in range(block.size):
                seg_word_idx = block.a + i
                trans_word_idx = block.b + i

                seg_idx = seg_words[seg_word_idx][1]
                word_idx = seg_words[seg_word_idx][2]

                orig_word = seg_word_positions[seg_word_idx][2]
                trans_word = trans_words[trans_word_idx][3]

                # Mots identiques normalisés → déjà traité au-dessus
                if _normalize_word(orig_word) == _normalize_word(trans_word):
                    continue

                # Mots différents → correction WhisperX (noms propres, termes techniques)
                if seg_idx < len(segments):
                    old_text = segments[seg_idx].text
                    words_list = old_text.split()
                    if word_idx < len(words_list):
                        words_list[word_idx] = trans_word
                        segments[seg_idx].text = " ".join(words_list)
                        corrections += 1

    print(f"   Corrections texte  : {corrections} mots")

    return segments, speaker_name_map


def build_epoch_context(page: EpochTimesPage) -> str:
    """Construit un bloc de contexte à prépendre au contexte utilisateur pour Claude."""
    lines = []
    if page.title:
        lines.append(f"Titre : {page.title}")
    if page.excerpt:
        lines.append(f"Résumé : {page.excerpt}")
    if page.authors:
        lines.append(f"Auteurs : {', '.join(page.authors)}")
    if page.speakers:
        lines.append(f"Locuteurs : {', '.join(page.speakers)}")

    if not lines:
        return ""

    return "[Source : Epoch Times]\n" + "\n".join(lines)
