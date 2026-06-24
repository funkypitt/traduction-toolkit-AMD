#!/usr/bin/env python3
"""
Sous-titrage d'une vidéo à partir d'une traduction existante dans un fichier DOCX.

Usage :
    python sous_titrer_docx.py video.mp4 traduction.docx [--style default] [--resume segments.json]

Pipeline :
    1. Transcription WhisperX (EN) → segments avec word-level timestamps
    2. Parsing du DOCX → blocs FR nettoyés (filtre notes, slides, commentaires)
    3. Alignement Claude : mapping séquentiel FR → segments EN (fenêtre glissante)
    4. Re-segmentation (logique traduire.py) → sous-titres formatés
    5. Génération SRT + gravure ffmpeg
"""

import argparse, json, math, os, re, subprocess, sys, time
from dataclasses import dataclass, field
from pathlib import Path

import hw
hw.setup_rocm_env()  # AMD/ROCm (gfx1151) : pose HSA_OVERRIDE_* avant tout import torch

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES (alignées sur traduire.py)
# ═══════════════════════════════════════════════════════════════════════════════

CLAUDE_MODEL = "claude-opus-4-5"
WHISPER_MODEL = "large-v3"

# Ollama (LLM local — alternative gratuite à l'API Claude)
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3.6:27b"
OLLAMA_NUM_PREDICT = 16384
# Backend LLM choisi en CLI (renseigné par main()) : ("claude") ou ("local", model, url)
LLM_BACKEND = ("claude",)


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
        text = re.sub(r'<think>[\s\S]*?</think>\s*', '', text)
        return type("Resp", (), {"content": [type("B", (), {"text": text})()]})()


def _make_llm_client():
    """Construit le client LLM selon le backend choisi en CLI."""
    if LLM_BACKEND[0] == "local":
        _, model, url = LLM_BACKEND
        print(f"   🧠 LLM local : Ollama {model}")
        return _OllamaClient(url, model)
    from anthropic import Anthropic
    return Anthropic()

MAX_CHARS_PER_LINE = 42
MAX_LINES_PER_SUB = 2
MAX_CPS = 17
MIN_DURATION_SEC = 1.0
MAX_DURATION_SEC = 7.0
GAP_BETWEEN_SUBS_MS = 80
MIN_CHARS_PER_SUB = 10
PAUSE_SPLIT_THRESHOLD = 1.5
PAUSE_SPLIT_PADDING = 0.3

ALIGN_WINDOW = 50            # segments EN par fenêtre d'alignement
ALIGN_FR_MARGIN = 0.35       # marge de texte FR en plus (35%)

ORPHAN_WORDS = {
    "fr": {"le", "la", "les", "l", "un", "une", "des", "du", "de", "d", "au", "aux",
           "ce", "cet", "cette", "ces", "mon", "ma", "mes", "ton", "ta", "tes",
           "son", "sa", "ses", "notre", "nos", "votre", "vos", "leur", "leurs",
           "à", "en", "et", "ou", "ne", "se", "je", "tu", "il", "on", "nous", "vous", "ils", "elles"},
}

SPLIT_PATTERNS = {
    "fr": {
        "punctuation": r'[,;:!?\.…]\s',
        "conjunctions": r'\s(?:et|mais|ou|car|donc|puis|alors|parce|puisque|quand|si|que|qui)\s',
        "prepositions": r'\s(?:de|du|des|à|au|aux|en|dans|sur|pour|par|avec|sans|chez)\s',
    },
}

CONJUNCTION_SEPARATORS = {
    "fr": [' et ', ' mais ', ' ou ', ' car ', ' donc ', ' puis ', ' alors '],
}

SUBTITLE_STYLES = {
    "default": (
        "FontName=Arial,FontSize=24,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1,MarginV=16"
    ),
    "netflix": (
        "FontName=Arial,FontSize=22,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H40000000,BorderStyle=3,Outline=0,Shadow=0,MarginV=14"
    ),
    "minimal": (
        "FontName=Roboto,FontSize=22,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H40000000,BorderStyle=3,Outline=0,Shadow=0,MarginV=12"
    ),
    "box": (
        "FontName=Arial,FontSize=22,PrimaryColour=&H00FFFFFF,"
        "BackColour=&H00000000,BorderStyle=4,Outline=0,Shadow=0,MarginV=16"
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURES
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
class Subtitle:
    index: int
    start: float
    end: float
    text: str

    def to_srt(self) -> str:
        return f"{self.index}\n{_fmt(self.start)} --> {_fmt(self.end)}\n{self.text}\n"

@dataclass
class FrBlock:
    index: int
    speaker: str
    text: str


def _fmt(sec: float) -> str:
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    ms = int((sec % 1) * 1000)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — PARSING DU DOCX
# ═══════════════════════════════════════════════════════════════════════════════

SPEAKER_PATTERNS = [
    re.compile(r'^(Kevin McCairn|K McC|K\s*McC)\s*:\s*', re.IGNORECASE),
    re.compile(r'^(Jeanee-?Rose Andrewartha|JR A|JR\s*A)\s*:\s*', re.IGNORECASE),
]

SPEAKER_NORMALIZE = {
    "kevin mccairn": "KMcC",
    "k mcc": "KMcC",
    "kmcc": "KMcC",
    "jeanee-rose andrewartha": "JRA",
    "jeaneerose andrewartha": "JRA",
    "jr a": "JRA",
    "jra": "JRA",
}

SKIP_PATTERNS = [
    re.compile(r'^https?://'),
    re.compile(r'^_+$'),
    re.compile(r'^Diapositive', re.IGNORECASE),
    re.compile(r'^\[Diapositive'),
    re.compile(r'^\[Coupe\s'),
    re.compile(r'^\[Cartographie\s'),
    re.compile(r'^\[The\s+thrombo'),
    re.compile(r'^\[Expériences\s'),
]

EDITORIAL_NOTE = re.compile(
    r'(?:'
    r'\((?:extrait d\'une interview précédente|rires|il corrige[^)]*|Je peux\.)\)'
    r'|'
    r'\[(?:il (?:choisit|corrige|montre|partage)[^\]]*|Diapositive[^\]]*|Coupe[^\]]*|diapositive[^\]]*)\]'
    r')'
)


def parse_docx(docx_path: str) -> list[FrBlock]:
    from docx import Document
    doc = Document(docx_path)

    blocks = []
    current_speaker = ""
    idx = 0

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        if any(p.match(text) for p in SKIP_PATTERNS):
            continue

        if text.startswith('«') or text.startswith('»'):
            continue

        speaker_found = None
        clean_text = text
        for sp in SPEAKER_PATTERNS:
            m = sp.match(text)
            if m:
                raw_speaker = m.group(1).lower().replace("-", "").replace("  ", " ").strip()
                speaker_found = SPEAKER_NORMALIZE.get(raw_speaker, raw_speaker)
                clean_text = text[m.end():].strip()
                break

        if speaker_found:
            current_speaker = speaker_found

        clean_text = EDITORIAL_NOTE.sub('', clean_text).strip()

        if not clean_text:
            continue
        if len(clean_text) < 3:
            continue

        blocks.append(FrBlock(index=idx, speaker=current_speaker, text=clean_text))
        idx += 1

    return blocks


def filter_preamble_and_disclaimer(blocks: list[FrBlock]) -> list[FrBlock]:
    """Le document contient un préambule (extraits/teaser) puis un disclaimer
    avant l'interview proprement dite. On détecte le début de l'interview
    (premier « Bonjour ») et on sépare ce qui précède."""
    interview_start = None
    for i, b in enumerate(blocks):
        if 'bonjour' in b.text.lower()[:30] or 'bienvenue' in b.text.lower()[:30]:
            interview_start = i
            break

    if interview_start is None:
        return blocks

    preamble = blocks[:interview_start]
    interview = blocks[interview_start:]

    filtered_preamble = []
    for b in preamble:
        if any(kw in b.text.lower() for kw in ['avertissement', 'informations fournies',
               'garantie', 'risques', 'éducatives', 'divertissement']):
            continue
        filtered_preamble.append(b)

    result = filtered_preamble + interview
    for i, b in enumerate(result):
        b.index = i
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — TRANSCRIPTION WHISPERX
# ═══════════════════════════════════════════════════════════════════════════════

def transcribe_whisperx(video_path: str, cache_json: str = None) -> list[dict]:
    if cache_json and os.path.exists(cache_json):
        print(f"📂 Chargement transcription WhisperX depuis {cache_json}")
        with open(cache_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        segments = data.get('segments', data) if isinstance(data, dict) else data
        print(f"   ✅ {len(segments)} segments chargés")
        return segments

    import whisperx, torch, gc

    device = hw.device()  # « cuda » couvre CUDA et ROCm/HIP
    compute_type = hw.whisper_compute_type()
    print(f"\n🎤 Transcription WhisperX ({device})...")

    model = whisperx.load_model(WHISPER_MODEL, device, compute_type=compute_type, language='en')
    audio = whisperx.load_audio(video_path)
    result = model.transcribe(audio, batch_size=16, language='en')
    print(f"   Transcription : {len(result['segments'])} segments")

    del model; gc.collect()
    if device == 'cuda':
        import torch; torch.cuda.empty_cache()

    model_a, metadata = whisperx.load_align_model(language_code='en', device=device)
    result = whisperx.align(result['segments'], model_a, metadata, audio, device,
                            return_char_alignments=False)
    print(f"   Alignement : {len(result['segments'])} segments")

    del model_a; gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    out_path = cache_json or video_path.rsplit('.', 1)[0] + '_whisperx.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"   💾 Sauvé : {out_path}")

    return result.get('segments', result) if isinstance(result, dict) else result


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — ALIGNEMENT CLAUDE (phrase par phrase)
# ═══════════════════════════════════════════════════════════════════════════════

def align_with_claude(en_segments: list[dict], fr_blocks: list[FrBlock],
                      cache_path: str = None) -> list[Segment]:
    """Alignement phrase-par-phrase : pour chaque segment EN (= phrase avec timing),
    Claude attribue le texte FR correspondant. Résultat : chaque segment EN reçoit
    sa traduction FR, et le timing est hérité directement."""

    if cache_path and os.path.exists(cache_path):
        print(f"📂 Chargement alignement depuis {cache_path}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        segments = []
        for d in data:
            segments.append(Segment(
                index=d['index'], start=d['start'], end=d['end'],
                text=d['text'], text_tgt=d['text_tgt'],
                words=d.get('words', [])
            ))
        print(f"   ✅ {len(segments)} segments alignés chargés")
        return segments

    client = _make_llm_client()

    # Concaténer tout le texte FR en un seul flux
    fr_full = "\n".join(b.text for b in fr_blocks)
    total_en_chars = sum(len(s.get('text', '')) for s in en_segments)
    total_fr_chars = len(fr_full)
    video_duration = en_segments[-1].get('end', 0)

    print(f"\n🔗 Alignement phrase-par-phrase ({len(en_segments)} segments EN)")
    print(f"   Texte EN total: {total_en_chars} chars")
    print(f"   Texte FR total: {total_fr_chars} chars")
    print(f"   Durée vidéo: {video_duration:.0f}s ({video_duration/60:.0f}min)")

    all_results = []  # liste de (en_idx, fr_text)
    en_cursor = 0

    window_num = 0
    while en_cursor < len(en_segments):
        window_num += 1
        en_end = min(en_cursor + ALIGN_WINDOW, len(en_segments))
        en_window = en_segments[en_cursor:en_end]

        # Estimation proportionnelle basée sur le TEMPS (pas le curseur)
        win_time_start = en_window[0].get('start', 0)
        win_time_end = en_window[-1].get('end', 0)
        frac_start = max(0, win_time_start / video_duration - 0.02)
        frac_end = min(1, win_time_end / video_duration + 0.05)
        fr_slice_start = max(0, int(frac_start * total_fr_chars))
        fr_slice_end = min(total_fr_chars, int(frac_end * total_fr_chars))

        # Aligner sur des frontières de mots/phrases
        while fr_slice_start > 0 and fr_full[fr_slice_start] not in ' \n':
            fr_slice_start -= 1
        while fr_slice_end < total_fr_chars and fr_full[fr_slice_end] not in ' \n':
            fr_slice_end += 1

        fr_slice = fr_full[fr_slice_start:fr_slice_end]

        if not fr_slice.strip():
            en_cursor = en_end
            continue

        # Construire le prompt
        en_text_block = "\n".join(
            f"[{en_cursor + i}] {s.get('text', '').strip()}"
            for i, s in enumerate(en_window)
        )

        prompt = f"""Tu dois faire correspondre chaque phrase anglaise à sa traduction française.

PHRASES ANGLAISES (numérotées) :
{en_text_block}

TEXTE FRANÇAIS (traduction séquentielle) :
{fr_slice}

TÂCHE : Pour chaque phrase anglaise ci-dessus, extrais le passage français qui en est la traduction. Le texte français suit le MÊME ORDRE que les phrases anglaises.

RÈGLES :
- Copie le texte français EXACTEMENT tel qu'il apparaît (pas de modification)
- Chaque morceau de texte français ne peut être attribué qu'à UNE seule phrase anglaise
- Si une phrase anglaise est un mot de remplissage ("Right.", "Mm-hmm.", "Yeah.") et n'a pas de traduction distincte, mets "" (vide)
- Si une phrase anglaise est traduite par la même portion de texte qu'une phrase voisine (le traducteur a fusionné), mets le texte sur la PREMIÈRE et "" sur les suivantes
- TOUT le texte français doit être distribué (rien ne doit rester non attribué)

Réponds UNIQUEMENT en JSON — un array avec un objet par phrase anglaise :
[
  {{"en_idx": {en_cursor}, "fr": "texte français correspondant"}},
  {{"en_idx": {en_cursor + 1}, "fr": "..."}},
  ...
]"""

        for attempt in range(3):
            try:
                resp = client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=16384,
                    messages=[{"role": "user", "content": prompt}]
                )
                raw = resp.content[0].text
                json_match = re.search(r'\[.*\]', raw, re.DOTALL)
                if json_match:
                    window_results = json.loads(json_match.group())
                else:
                    raise ValueError("Pas de JSON trouvé")
                break
            except Exception as e:
                if attempt < 2:
                    print(f"   ⚠️  Fenêtre {window_num} tentative {attempt+1} : {e}")
                    time.sleep(2)
                else:
                    print(f"   ❌ Fenêtre {window_num} échouée")
                    window_results = []

        # Collecter les résultats
        n_ok = 0
        for wr in window_results:
            en_idx = wr.get('en_idx', -1)
            fr_text = wr.get('fr', '').strip()
            if en_idx >= 0:
                all_results.append((en_idx, fr_text))
                if fr_text:
                    n_ok += 1

        print(f"   Fenêtre {window_num}: EN {en_cursor}–{en_end-1} "
              f"({win_time_start:.0f}–{win_time_end:.0f}s) → {n_ok}/{len(en_window)} "
              f"(FR slice: {fr_slice_start}–{fr_slice_end})")

        en_cursor = en_end

    # Construire les Segments
    n_with_text = sum(1 for _, fr in all_results if fr)
    print(f"\n   ✅ {n_with_text}/{len(en_segments)} segments avec traduction FR")

    segments = _build_aligned_segments(all_results, en_segments)

    if cache_path:
        data = [{'index': s.index, 'start': s.start, 'end': s.end,
                 'text': s.text, 'text_tgt': s.text_tgt, 'words': s.words}
                for s in segments]
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"   💾 Sauvé : {cache_path}")

    return segments


def _build_aligned_segments(results: list[tuple], en_segments: list[dict]) -> list[Segment]:
    """Construit les Segments à partir des résultats d'alignement phrase-par-phrase.
    Chaque segment EN avec une traduction FR devient un Segment."""
    # Index lookup
    fr_by_en = {}
    for en_idx, fr_text in results:
        if en_idx not in fr_by_en and fr_text:
            fr_by_en[en_idx] = fr_text

    segments = []
    idx = 0
    for i, en_seg in enumerate(en_segments):
        fr_text = fr_by_en.get(i, '')
        if not fr_text:
            continue

        segments.append(Segment(
            index=idx,
            start=en_seg.get('start', 0),
            end=en_seg.get('end', 0),
            text=en_seg.get('text', '').strip(),
            text_tgt=fr_text,
            words=en_seg.get('words', []),
        ))
        idx += 1

    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — RE-SEGMENTATION (adapté de traduire.py)
# ═══════════════════════════════════════════════════════════════════════════════

def get_split_patterns(lang_code: str) -> list[str]:
    pats = SPLIT_PATTERNS.get(lang_code)
    if pats:
        return [pats["punctuation"], pats["conjunctions"], pats["prepositions"]]
    return [r'[,;:!?\.…]\s']


def _fmtlines(txt: str, target_lang: str = "fr") -> str:
    txt = ' '.join(txt.split()).strip()
    if len(txt) <= MAX_CHARS_PER_LINE:
        return txt
    orphans = ORPHAN_WORDS.get(target_lang, set())

    def _apply_antiorphan(l1, l2):
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

    L = len(txt)
    min_pos = max(1, L - MAX_CHARS_PER_LINE)
    max_pos = min(L - 1, MAX_CHARS_PER_LINE)
    if min_pos <= max_pos:
        mid = L // 2
        best_p, best_d = -1, L
        for i, ch in enumerate(txt):
            if ch == ' ' and min_pos <= i <= max_pos:
                d = abs(i - mid)
                if d < best_d:
                    best_d = d; best_p = i
        if best_p > 0:
            l1, l2 = txt[:best_p].strip(), txt[best_p:].strip()
            if len(l1) <= MAX_CHARS_PER_LINE and len(l2) <= MAX_CHARS_PER_LINE:
                l1, l2 = _apply_antiorphan(l1, l2)
                return f"{l1}\n{l2}"

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
                if d < bd:
                    bd = d; best = pos
        if best >= 0:
            break
    return best


def _splittext(txt: str, dur: float, target_lang: str = "fr") -> list[str]:
    mx = MAX_CHARS_PER_LINE * MAX_LINES_PER_SUB
    n_chars = -(-len(txt) // min(mx, max(20, int(dur * MAX_CPS))))
    n_time = math.ceil(dur / MAX_DURATION_SEC) if dur > MAX_DURATION_SEC else 1
    n = max(1, n_chars, n_time)
    if n == 1:
        return [txt]

    orphans = ORPHAN_WORDS.get(target_lang, set())
    conj = CONJUNCTION_SEPARATORS.get(target_lang, [])
    separators = ['. ', '? ', '! ', '; ', ', ', ' — ', ' – '] + conj + [' ']

    parts, rem, tgt = [], txt, len(txt) // n
    for _ in range(n - 1):
        if len(rem) <= tgt + 10:
            break
        pos = -1
        for sep in separators:
            p = rem.rfind(sep, tgt - 15, tgt + 15)
            if p > 0:
                pos = p + len(sep)
                break
        if pos < 0:
            pos = rem.rfind(' ', 0, tgt + 5)
            if pos < 0:
                pos = tgt

        candidate = rem[:pos].strip()
        last_word = candidate.split()[-1].lower().rstrip("'") if candidate.split() else ""
        if last_word in orphans:
            words = candidate.split()
            if len(words) > 1:
                new_end = candidate.rindex(' ')
                pos = new_end

        parts.append(rem[:pos].strip())
        rem = rem[pos:].strip()

    if rem.strip():
        parts.append(rem.strip())
    return parts


def _split_tgt_on_punct(text: str, fractions: list[float]) -> list[str]:
    sents = re.split(r'(?<=[.!?;])\s+', text)
    if len(sents) < len(fractions):
        sents = re.split(r'(?<=[,.])\s+', text)
    if len(sents) < len(fractions):
        return [text] + [''] * (len(fractions) - 1)

    total_chars = sum(len(s) for s in sents)
    result = []
    sent_idx = 0
    for fi, frac in enumerate(fractions):
        target_chars = frac * total_chars
        collected = []
        chars_so_far = 0
        while sent_idx < len(sents):
            s = sents[sent_idx]
            if collected and chars_so_far + len(s) > target_chars * 1.3 and fi < len(fractions) - 1:
                break
            collected.append(s)
            chars_so_far += len(s)
            sent_idx += 1
            if chars_so_far >= target_chars * 0.8 and fi < len(fractions) - 1:
                break
        result.append(' '.join(collected))

    while sent_idx < len(sents):
        if result:
            result[-1] += ' ' + sents[sent_idx]
        sent_idx += 1

    return result


def _split_on_pauses(segments: list[Segment]) -> list[Segment]:
    result = []
    split_count = 0
    for seg in segments:
        words = seg.words or []
        wts = [w for w in words if isinstance(w, dict) and 'start' in w and 'end' in w]
        if len(wts) < 2 or not seg.text_tgt:
            result.append(seg)
            continue

        clusters = [[wts[0]]]
        for w in wts[1:]:
            if w['start'] - clusters[-1][-1]['end'] >= PAUSE_SPLIT_THRESHOLD:
                clusters.append([w])
            else:
                clusters[-1].append(w)

        if len(clusters) == 1:
            result.append(seg)
            continue

        cluster_chars = [sum(len(w.get('word', '')) for w in c) for c in clusters]
        total_src = sum(cluster_chars) or 1
        fractions = [c / total_src for c in cluster_chars]
        sub_texts = _split_tgt_on_punct(seg.text_tgt.strip(), fractions)

        if any(not s for s in sub_texts):
            result.append(seg)
            continue

        for ci, cluster in enumerate(clusters):
            st = cluster[0]['start']
            raw_en = cluster[-1]['end']
            if ci + 1 < len(clusters):
                next_st = clusters[ci + 1][0]['start']
                en = min(raw_en + PAUSE_SPLIT_PADDING, next_st - 0.10)
            else:
                en = min(raw_en + PAUSE_SPLIT_PADDING, seg.end)
            en = max(en, st + MIN_DURATION_SEC)
            result.append(Segment(
                index=seg.index, start=st, end=en,
                text=' '.join(w.get('word', '') for w in cluster),
                text_tgt=sub_texts[ci],
                words=cluster,
            ))
        split_count += 1

    if split_count:
        print(f"   ✂️  {split_count} segment(s) redécoupé(s) sur silences internes")
    return result


def _merge_short(segments: list[Segment], target_lang: str) -> list[Segment]:
    max_chars = MAX_CHARS_PER_LINE * MAX_LINES_PER_SUB
    merged = []
    for seg in segments:
        if not seg.text_tgt or seg.end - seg.start <= 0:
            continue
        txt = seg.text_tgt.strip()

        if (merged
                and len(txt) < MIN_CHARS_PER_SUB
                and len(txt.split()) <= 3):
            prev = merged[-1]
            combined_txt = prev.text_tgt + " " + txt
            combined_dur = seg.end - prev.start
            combined_cps = len(combined_txt) / combined_dur if combined_dur > 0 else 999

            if (len(combined_txt) <= max_chars
                    and combined_dur <= MAX_DURATION_SEC
                    and combined_cps <= MAX_CPS):
                merged[-1] = Segment(
                    index=prev.index, start=prev.start, end=seg.end,
                    text=prev.text + " " + seg.text,
                    text_tgt=combined_txt,
                    words=(prev.words or []) + (seg.words or []),
                )
                continue

        merged.append(Segment(
            index=seg.index, start=seg.start, end=seg.end,
            text=seg.text, text_tgt=txt, words=seg.words,
        ))

    return merged


def _deorphan(subs: list[Subtitle], target_lang: str) -> list[Subtitle]:
    orphans = ORPHAN_WORDS.get(target_lang, set())
    max_chars = MAX_CHARS_PER_LINE * MAX_LINES_PER_SUB
    result = []
    for sub in subs:
        txt = sub.text.replace('\n', ' ').strip()
        words = txt.split()
        if words and words[-1].lower().rstrip("'") in orphans and len(words) > 1:
            new_txt = ' '.join(words[:-1])
            orphan = words[-1]
            if result:
                prev = result[-1]
                prev_txt = prev.text.replace('\n', ' ').strip()
                combined = prev_txt + ' ' + orphan
                if len(combined) <= max_chars:
                    result[-1] = Subtitle(prev.index, prev.start, prev.end,
                                          _fmtlines(combined, target_lang))
                    if new_txt:
                        result.append(Subtitle(sub.index, sub.start, sub.end,
                                               _fmtlines(new_txt, target_lang)))
                    continue
        result.append(sub)
    return result


def _split_overlong(subs: list[Subtitle], target_lang: str) -> list[Subtitle]:
    """Redécoupe les sous-titres dont le texte dépasse 2 lignes de 42 chars.
    Limite le nombre de parties à ce que le temps disponible permet."""
    max_chars = MAX_CHARS_PER_LINE * MAX_LINES_PER_SUB
    result = []
    idx = 1
    split_count = 0
    for si, sub in enumerate(subs):
        flat = sub.text.replace('\n', ' ').strip()
        if len(flat) <= max_chars:
            result.append(Subtitle(idx, sub.start, sub.end, _fmtlines(flat, target_lang)))
            idx += 1
            continue

        dur = sub.end - sub.start
        gap = GAP_BETWEEN_SUBS_MS / 1000

        # Calcul du nombre max de parts que le temps permet (min 1.0s chacune)
        max_parts_by_time = max(1, int((dur + gap) / (MIN_DURATION_SEC + gap)))

        # Nombre de parts nécessaires pour le texte
        parts_needed = max(1, -(-len(flat) // max_chars))

        # Prendre le minimum : on préfère des sous-titres lisibles (max_chars)
        # mais pas de durées < MIN_DURATION_SEC
        n_parts = min(parts_needed, max_parts_by_time)

        if n_parts <= 1:
            result.append(Subtitle(idx, sub.start, sub.end, _fmtlines(flat, target_lang)))
            idx += 1
            continue

        # Découper le texte en n_parts parts
        parts = _splittext_n(flat, n_parts, target_lang)

        # Chercher du temps supplémentaire après ce sous-titre si CPS trop élevé
        total_needed = len(flat) / MAX_CPS
        extra_time = 0
        if total_needed > dur and si + 1 < len(subs):
            gap_to_next = subs[si + 1].start - sub.end
            extra_time = min(total_needed - dur, gap_to_next * 0.8)

        effective_dur = dur + extra_time
        effective_end = sub.end + extra_time
        usable = max(0, effective_dur - (len(parts) - 1) * gap)
        total_chars = sum(len(p) for p in parts)
        t = sub.start
        for i, p in enumerate(parts):
            frac = len(p) / total_chars if total_chars > 0 else 1.0 / len(parts)
            d = max(MIN_DURATION_SEC, usable * frac)
            te = min(t + d, effective_end) if i < len(parts) - 1 else effective_end
            if te <= t:
                # Plus de temps : fusionner le reste dans le dernier sous-titre
                if result:
                    remaining = ' '.join(parts[i:])
                    prev = result[-1]
                    merged = prev.text.replace('\n', ' ') + ' ' + remaining
                    result[-1] = Subtitle(prev.index, prev.start, effective_end,
                                          _fmtlines(merged, target_lang))
                break
            result.append(Subtitle(idx, t, te, _fmtlines(p, target_lang)))
            idx += 1
            t = te + gap
        split_count += 1

    if split_count:
        print(f"   ✂️  {split_count} sous-titre(s) trop longs redécoupés")
    return result


def _splittext_n(txt: str, n: int, target_lang: str = "fr") -> list[str]:
    """Découpe le texte en exactement n parts aux meilleurs points de coupure."""
    if n <= 1:
        return [txt]

    orphans = ORPHAN_WORDS.get(target_lang, set())
    conj = CONJUNCTION_SEPARATORS.get(target_lang, [])
    separators = ['. ', '? ', '! ', '; ', ', ', ' — ', ' – '] + conj + [' ']

    parts = []
    rem = txt
    tgt = len(txt) // n

    for _ in range(n - 1):
        if not rem.strip():
            break
        if len(rem) <= tgt + 10:
            break
        pos = -1
        for sep in separators:
            p = rem.rfind(sep, max(0, tgt - 20), tgt + 20)
            if p > 0:
                pos = p + len(sep)
                break
        if pos < 0:
            pos = rem.rfind(' ', 0, tgt + 10)
            if pos < 0:
                pos = tgt

        candidate = rem[:pos].strip()
        last_word = candidate.split()[-1].lower().rstrip("'") if candidate.split() else ""
        if last_word in orphans and len(candidate.split()) > 1:
            new_end = candidate.rindex(' ')
            pos = new_end

        parts.append(rem[:pos].strip())
        rem = rem[pos:].strip()

    if rem.strip():
        parts.append(rem.strip())
    return parts


def _fixtiming(subs: list[Subtitle]) -> list[Subtitle]:
    if not subs:
        return subs
    gap = GAP_BETWEEN_SUBS_MS / 1000
    if subs[0].end - subs[0].start < MIN_DURATION_SEC:
        desired_end = subs[0].start + MIN_DURATION_SEC
        limit = subs[1].start - gap if len(subs) > 1 else desired_end
        subs[0] = Subtitle(subs[0].index, subs[0].start,
                            min(desired_end, limit), subs[0].text)

    for i in range(1, len(subs)):
        prev = i - 1
        if subs[i].start < subs[prev].end + gap:
            subs[prev] = Subtitle(subs[prev].index, subs[prev].start,
                                   subs[i].start - gap, subs[prev].text)
        if subs[i].end - subs[i].start < MIN_DURATION_SEC:
            desired = subs[i].start + MIN_DURATION_SEC
            limit = subs[i + 1].start - gap if i + 1 < len(subs) else desired
            subs[i] = Subtitle(subs[i].index, subs[i].start,
                                min(desired, limit), subs[i].text)
    return subs


def _audit_cps(subs: list[Subtitle]) -> list[Subtitle]:
    result = []
    for i, sub in enumerate(subs):
        txt = sub.text.replace('\n', ' ').strip()
        dur = sub.end - sub.start
        if dur <= 0:
            continue
        cps = len(txt) / dur
        if cps > MAX_CPS and dur < MAX_DURATION_SEC:
            needed = len(txt) / MAX_CPS
            new_end = sub.start + needed
            # Ne pas dépasser le début du sous-titre suivant
            if i + 1 < len(subs):
                new_end = min(new_end, subs[i + 1].start - GAP_BETWEEN_SUBS_MS / 1000)
            new_end = min(new_end, sub.end + 1.5)
            sub = Subtitle(sub.index, sub.start, max(sub.end, new_end), sub.text)
        result.append(sub)
    return result


def resegment(segments: list[Segment], target_lang: str = "fr") -> list[Subtitle]:
    print("\n📐 Re-segmentation...")

    segments = _split_on_pauses(segments)
    merged = _merge_short(segments, target_lang)

    subs = []
    idx = 1
    for seg in merged:
        if not seg.text_tgt or seg.end - seg.start <= 0:
            continue
        txt = seg.text_tgt.strip()
        dur = seg.end - seg.start
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

    subs = _deorphan(subs, target_lang)
    subs = _split_overlong(subs, target_lang)
    subs = _fixtiming(subs)
    subs = _audit_cps(subs)
    subs = _fixtiming(subs)
    print(f"   ✅ {len(subs)} sous-titres")
    return subs


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 5 — GÉNÉRATION SRT + GRAVURE
# ═══════════════════════════════════════════════════════════════════════════════

def generate_srt(subs: list[Subtitle], path: str):
    print(f"\n💾 SRT : {path}")
    with open(path, "w", encoding="utf-8") as f:
        for s in subs:
            f.write(s.to_srt() + "\n")
    print(f"   ✅ {len(subs)} sous-titres écrits")


def _get_video_dimensions(video: str) -> tuple:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", video],
            capture_output=True, text=True)
        w, h = r.stdout.strip().split("x")
        return int(w), int(h)
    except Exception:
        return 1920, 1080


def _scale_style_for_video(style_str: str, video_width: int, video_height: int) -> str:
    if video_width <= 0 or video_height <= 0:
        return style_str
    aspect = video_width / video_height
    if aspect >= 16 / 9 - 0.05:
        return style_str
    expected_width = video_height * 16 / 9
    width_ratio = video_width / expected_width

    def scale_field(match):
        name = match.group(1)
        val = int(match.group(2))
        new = round(val * width_ratio)
        if name == "FontSize":
            new = max(14, new)
        else:
            new = max(8, new)
        return f"{name}={new}"
    return re.sub(r"(FontSize|MarginV)=(\d+)", scale_field, style_str)


def burn_subtitles(video: str, srt: str, output: str, style: str = "default"):
    import shutil, tempfile
    print(f"\n🎬 Incrustation (style: {style})...")
    fs = SUBTITLE_STYLES.get(style, SUBTITLE_STYLES["default"])
    vw, vh = _get_video_dimensions(video)
    fs = _scale_style_for_video(fs, vw, vh)
    print(f"   📐 Dimensions vidéo: {vw}x{vh}")

    tmp_dir = tempfile.mkdtemp()
    tmp_srt = os.path.join(tmp_dir, "subs.srt")
    shutil.copy2(srt, tmp_srt)

    sub_f = f"subtitles={tmp_srt}:force_style='{fs}'"
    cmd = ["ffmpeg", "-y", "-i", video,
           "-vf", sub_f,
           "-c:v", "libx264", "-crf", "18", "-preset", "slow",
           "-c:a", "aac", "-b:a", "192k", "-ac", "2",
           "-movflags", "+faststart", output]

    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if r.returncode != 0:
        print(f"   ⚠️  Retry sans style...")
        cmd2 = ["ffmpeg", "-y", "-i", video,
                "-vf", f"subtitles={tmp_srt}",
                "-c:v", "libx264", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k", "-ac", "2",
                "-movflags", "+faststart", output]
        r = subprocess.run(cmd2, capture_output=True, text=True)
        elapsed = time.time() - t0
        if r.returncode != 0:
            print(f"   ❌ Incrustation échouée.")
            print(f"   Le SRT est disponible : {srt}")
            return

    shutil.rmtree(tmp_dir, ignore_errors=True)
    size_mb = os.path.getsize(output) / (1024 * 1024)
    print(f"   ✅ {output} ({size_mb:.0f} Mo, {elapsed:.0f}s)")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Sous-titrer une vidéo à partir d'un DOCX traduit")
    parser.add_argument("video", help="Fichier vidéo source")
    parser.add_argument("docx", help="Fichier DOCX contenant la traduction")
    parser.add_argument("--style", default="default", choices=list(SUBTITLE_STYLES.keys()))
    parser.add_argument("--resume", help="Reprendre depuis un JSON de segments alignés")
    parser.add_argument("--whisperx-json", help="JSON WhisperX pré-calculé")
    parser.add_argument("--srt-only", action="store_true", help="Générer le SRT sans graver")
    parser.add_argument("--llm", choices=["claude", "local"], default="local",
                        help="Backend LLM : local (Ollama, défaut) ou claude (API Anthropic)")
    parser.add_argument("--ollama-model", default=OLLAMA_MODEL,
                        help=f"Modèle Ollama (défaut: {OLLAMA_MODEL})")
    parser.add_argument("--ollama-url", default=OLLAMA_URL,
                        help=f"URL du serveur Ollama (défaut: {OLLAMA_URL})")
    args = parser.parse_args()

    global LLM_BACKEND
    if args.llm == "local":
        LLM_BACKEND = ("local", args.ollama_model, args.ollama_url)

    video = args.video
    base = os.path.splitext(video)[0]

    # 1. Parser le DOCX
    print("📄 Parsing du DOCX...")
    fr_blocks = parse_docx(args.docx)
    fr_blocks = filter_preamble_and_disclaimer(fr_blocks)
    print(f"   ✅ {len(fr_blocks)} blocs FR extraits")
    for b in fr_blocks[:5]:
        print(f"      [{b.index}] ({b.speaker}) {b.text[:80]}...")

    # 2. Transcription WhisperX
    whisperx_json = args.whisperx_json or f"{base}_whisperx.json"
    en_segments = transcribe_whisperx(video, cache_json=whisperx_json)

    # 3. Alignement
    alignment_json = args.resume or f"{base}_alignment.json"
    segments = align_with_claude(en_segments, fr_blocks, cache_path=alignment_json)
    print(f"\n📊 {len(segments)} segments alignés (durée couverte: "
          f"{segments[0].start:.1f}s → {segments[-1].end:.1f}s)")

    # 4. Re-segmentation
    subs = resegment(segments, target_lang="fr")

    # 5. Génération SRT
    srt_path = f"{base}_fr.srt"
    generate_srt(subs, srt_path)

    # 6. Gravure
    if not args.srt_only:
        output = f"{base}_fr.mp4"
        burn_subtitles(video, srt_path, output, style=args.style)
    else:
        print(f"\n✅ SRT généré : {srt_path}")


if __name__ == "__main__":
    main()
