#!/usr/bin/env python3
"""
traduire-pro.py — Kit vidéo complet deluxe (Claude Opus)
=========================================================
Produit en une seule passe un kit vidéo soigné et prêt à diffuser :
sous-titres SRT + vidéo incrustée + variante « pour doubleur » + résumé
structuré + titre viral, description & chapitres YouTube/X + JSON complet
pour un futur pipeline de doublage ElevenLabs.

Pipeline (13 passes) :
  1. WhisperX large-v3 + alignement mot à mot                (transcription)
  2. Pyannote diarisation + profils locuteurs + genre/F0      (speakers)
  3. Analyse enrichie Claude Opus + glossaire + noms propres  (analyse)
  4. Vérification des noms propres segment par segment        (CRITICAL)
  5. Traduction contextuelle Claude Opus (chunks + retry)     (traduction)
  6. Revue qualité Claude Opus (3 passes : relecture /
     cohérence / glossaire)                                   (review)
  7. Re-segmentation pro : CPS lisible, coupures sémantiques,
     anti-orphelin, formatage 2 lignes équilibrées            (resegment)
  7b. Audit sémantique Opus des coupures problématiques       (audit-cuts)
  7c. Génération SRT source + SRT cible + bilingue            (SRT)
  8. Incrustation vidéo + ASS « doubleur » jaune/cyan (défaut)(burn)
  9. Résumé détaillé structuré (portage de resumer.py)        (summary)
 10. Titre viral + description + chapitrage YouTube & X       (promo kit)
 11. Export `_doublage.json` complet pour doubler-deluxe.py   (doublage kit)
 12. Rapport final de session                                 (report)

Usage :
  python traduire-pro.py video.mp4
  python traduire-pro.py video.mp4 --context "Interview de Trita Parsi (Quincy Institute)"
  python traduire-pro.py video.mp4 -s en -t fr
  python traduire-pro.py video.mp4 --no-dubbing --no-audit-cuts
  python traduire-pro.py video.mp4 --resume video_pro_work/segments_finaux.json
  python traduire-pro.py video.mp4 --claude-model claude-opus-4-1-20250805

Points clés :
  - Modèle par défaut : Claude Opus 4.5 (qualité maximale, ~5x coût Sonnet)
  - MAX_CPS abaissé à 15 (lecture encore plus confortable qu'en 17)
  - ASS « pour doubleur » généré par DÉFAUT (désactivable via --no-dubbing)
  - Le `--context` est injecté PARTOUT : analyse, vérif noms, traduction,
    revue, résumé, promo kit. Obligatoire pour toute vidéo avec des noms
    propres difficiles.
  - Tous les checkpoints intermédiaires vont dans `{base}_pro_work/` pour
    permettre des reprises chirurgicales.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Fork-par-import : tout le pipeline de base de traduire.py est réutilisé
# tel quel. On ne duplique pas 1800 lignes — on override simplement les
# constantes dont dépendent les fonctions internes (module globals).
import traduire
from traduire import (
    Segment, Subtitle, ContentAnalysis,
    _claude_create, _OllamaClient, lang_name, source_lang_description, source_lang_label,
    check_ffmpeg, is_youtube_url, download_youtube,
    extract_audio, transcribe_whisperx, ocr_supplement_segments,
    analyze_content, translate_chunks,
    review_translation, check_consistency, verify_glossary,
    resegment, generate_srt, burn_subtitles,
    save_seg, load_seg, save_bilingual, save_src_srt,
    generate_dubbing_ass, get_video_resolution, burn_dubbing_video,
    SUBTITLE_STYLES,
    CLAUDE_MAX_TOKENS,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES PRO (override des valeurs de traduire.py)
# ═══════════════════════════════════════════════════════════════════════════════

# Modèle Opus par défaut (user explicite, override du default skill)
CLAUDE_MODEL_PRO = "claude-opus-4-5"

# CPS plafond : 24 = cible principale pour EN→FR (expansion linguistique ~20 %).
# Règle pragmatique : on vise ≤24, mais si après plusieurs réécritures un sous-titre
# reste irréductible, on tolère jusqu'à MAX_CPS_PRO + CPS_TOLERANCE_PRO (= 26)
# plutôt que de tronquer du contenu. Mieux vaut un sous-titre un peu rapide qu'un
# sous-titre amputé.
# Plus strict que l'ancien standard DVD (25-30) mais plus tolérant que Netflix (17).
MAX_CPS_PRO = 24
CPS_TOLERANCE_PRO = 2  # irréducible toléré : MAX_CPS + 2 (= 26 par défaut)

# Paramètres de profils locuteurs (portés de doubler.py)
MIN_SPEAKER_SAMPLE_SEC = 5
MAX_SPEAKER_SAMPLE_SEC = 30

# Mots par page / pages par minute pour le résumé (portés de resumer.py)
WORDS_PER_PAGE = 350
PAGES_PER_MINUTE = 0.3

# Estimation coût Opus 4.5 (indicatif, pour affichage début passe 3)
OPUS_COST_INPUT_PER_1M = 15.0   # $/1M input tokens (Opus)
OPUS_COST_OUTPUT_PER_1M = 75.0  # $/1M output tokens (Opus)


# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASSES SUPPLÉMENTAIRES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SpeakerProfile:
    """Profil d'un locuteur issu de la diarisation + analyse audio."""
    speaker_id: str
    gender: str = "unknown"          # "male" | "female" | "unknown"
    f0_median: float = 0.0            # Hz (médiane de la fréquence fondamentale)
    total_duration: float = 0.0       # durée totale de parole
    segment_count: int = 0
    sample_path: str = ""             # fichier WAV concaténé (plus longs segments)
    sample_text: str = ""             # texte des extraits concaténés
    ref_clips: list = field(default_factory=list)  # [(wav_path, text), ...]
    name: str = ""                    # nom humain (peut être rempli manuellement)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_workdir(base: str, parent: Path) -> Path:
    """Crée le dossier de travail `{base}_pro_work/` si nécessaire."""
    wd = parent / f"{base}_pro_work"
    wd.mkdir(parents=True, exist_ok=True)
    return wd


def _save_json(obj, path: str) -> None:
    """Sauvegarde JSON utf-8 indentée."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _fmt_timecode_youtube(sec: float) -> str:
    """Formate un timecode pour chapitrage YouTube : m:ss ou h:mm:ss."""
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 2 : DIARISATION + PROFILS LOCUTEURS (portage depuis doubler)
# ═══════════════════════════════════════════════════════════════════════════════

def diarize_speakers_pro(audio_path: str, segments: list[Segment],
                          hf_token: Optional[str] = None,
                          num_speakers: Optional[int] = None) -> dict[int, str]:
    """
    Identifie qui parle quand via Pyannote. Attache un `speaker` à chaque
    Segment (attribut dynamique, puisque le Segment de traduire.py n'a pas
    ce champ) et retourne aussi une map index → speaker_id pour sérialisation.
    """
    print("\n👥 Passe 2 — Diarisation des locuteurs...")

    if not hf_token:
        print("   ⚠️  HF_TOKEN manquant — pyannote requiert un token HuggingFace")
        print("   → export HF_TOKEN=hf_...")
        print("   ℹ️  Tous les segments assignés à SPEAKER_00")
        for s in segments:
            s.speaker = "SPEAKER_00"
        return {s.index: "SPEAKER_00" for s in segments}

    import whisperx, torch, gc
    from whisperx.diarize import DiarizationPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    diarize_model = DiarizationPipeline(token=hf_token, device=device)

    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers

    audio = whisperx.load_audio(audio_path)
    diarize_result = diarize_model(audio, **kwargs)

    # Reformatage au format attendu par assign_word_speakers
    whisperx_segments = [
        {"start": s.start, "end": s.end, "text": s.text, "words": s.words}
        for s in segments
    ]
    result = whisperx.assign_word_speakers(
        diarize_result, {"segments": whisperx_segments}
    )

    for i, seg_data in enumerate(result["segments"]):
        if i < len(segments):
            segments[i].speaker = seg_data.get("speaker") or "SPEAKER_00"

    # Consolidation si trop de labels (cf. doubler.py:648-681)
    if num_speakers:
        dur_by_speaker: dict[str, float] = {}
        for s in segments:
            dur_by_speaker.setdefault(s.speaker, 0.0)
            dur_by_speaker[s.speaker] += (s.end - s.start)

        if len(dur_by_speaker) > num_speakers:
            ranked = sorted(dur_by_speaker.items(), key=lambda x: x[1], reverse=True)
            keep = {spk for spk, _ in ranked[:num_speakers]}
            extra = {spk for spk in dur_by_speaker if spk not in keep}

            print(f"   🔧 Consolidation : {len(dur_by_speaker)} → {num_speakers} "
                  f"(réattribution de {', '.join(sorted(extra))})")

            keep_mids: dict[str, list[float]] = {}
            for s in segments:
                if s.speaker in keep:
                    keep_mids.setdefault(s.speaker, []).append((s.start + s.end) / 2)

            for s in segments:
                if s.speaker in extra:
                    mid = (s.start + s.end) / 2
                    best_spk, best_dist = ranked[0][0], float("inf")
                    for spk, mids in keep_mids.items():
                        dist = min(abs(mid - m) for m in mids)
                        if dist < best_dist:
                            best_dist, best_spk = dist, spk
                    s.speaker = best_spk

    # Stats
    stats: dict[str, dict] = {}
    for s in segments:
        stats.setdefault(s.speaker, {"count": 0, "dur": 0.0})
        stats[s.speaker]["count"] += 1
        stats[s.speaker]["dur"] += (s.end - s.start)

    del diarize_model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"   ✅ {len(stats)} locuteur(s) détecté(s) ({time.time()-t0:.1f}s)")
    for spk, info in sorted(stats.items()):
        print(f"      {spk} : {info['count']} segments, {info['dur']:.1f}s")

    return {s.index: s.speaker for s in segments}


def _estimate_gender(audio_mono, sr: int,
                     threshold_female: float = 165.0,
                     threshold_male: float = 155.0) -> tuple[str, float]:
    """
    Estime le genre du locuteur via F0 médian (autocorrélation par frames).
    Porté verbatim de doubler.py:937.
    """
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
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2:]
        if corr[0] == 0:
            continue
        corr = corr / corr[0]
        search = corr[lag_min:min(lag_max, len(corr))]
        if len(search) < 2:
            continue
        peak_idx = int(np.argmax(search))
        peak_val = search[peak_idx]
        if peak_val > 0.3:
            lag = lag_min + peak_idx
            f0_values.append(sr / lag)

    if not f0_values:
        return "unknown", 0.0

    f0_median = float(np.median(f0_values))
    if f0_median > threshold_female:
        return "female", f0_median
    if f0_median < threshold_male:
        return "male", f0_median
    return "unknown", f0_median


def _read_audio_slice(audio_path: str, start_sec: float, end_sec: float,
                     sr: int, n_frames: int):
    """Lit un segment audio sans charger tout le fichier en RAM."""
    import soundfile as sf
    import numpy as np

    start_sample = int(start_sec * sr)
    end_sample = min(int(end_sec * sr), n_frames)
    if start_sample >= end_sample:
        return np.array([], dtype=np.float64)

    data, _ = sf.read(audio_path, start=start_sample, stop=end_sample, always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data


def extract_speaker_profiles(segments: list[Segment], audio_path: str,
                              work_dir: str) -> dict[str, SpeakerProfile]:
    """
    Extrait un profil (sample concaténé + ref_clips + genre + F0) par locuteur.
    Alimente (a) la vérification de noms propres, (b) le _doublage.json.

    NB : on travaille directement sur l'audio extrait par ffmpeg (pas de
    séparation Demucs — le bruit de fond n'impacte pas la détection F0).
    """
    import soundfile as sf
    import numpy as np

    print("\n🎤 Passe 2b — Extraction des profils locuteurs...")

    samples_dir = os.path.join(work_dir, "speaker_samples")
    os.makedirs(samples_dir, exist_ok=True)

    info = sf.info(audio_path)
    sr = info.samplerate
    n_frames = info.frames

    # Groupement par locuteur (s.speaker doit avoir été posé par diarize_speakers_pro)
    speaker_segs: dict[str, list[Segment]] = {}
    for seg in segments:
        spk = getattr(seg, "speaker", "SPEAKER_00")
        speaker_segs.setdefault(spk, []).append(seg)

    profiles: dict[str, SpeakerProfile] = {}

    for spk_id, segs in speaker_segs.items():
        segs_sorted = sorted(segs, key=lambda s: (s.end - s.start), reverse=True)
        total_dur = sum(s.end - s.start for s in segs)

        profile = SpeakerProfile(
            speaker_id=spk_id,
            total_duration=total_dur,
            segment_count=len(segs),
        )

        sample_chunks = []
        sample_texts = []
        accumulated = 0.0

        for seg in segs_sorted:
            dur = seg.end - seg.start
            if accumulated >= MAX_SPEAKER_SAMPLE_SEC:
                break
            if dur < 0.5:
                continue
            chunk = _read_audio_slice(audio_path, seg.start, seg.end, sr, n_frames)
            if len(chunk) > 0:
                sample_chunks.append(chunk)
                sample_texts.append(seg.text)
                accumulated += dur

        full_sample = None
        if sample_chunks:
            silence = np.zeros(int(0.1 * sr))
            parts = []
            for i, chunk in enumerate(sample_chunks):
                parts.append(chunk)
                if i < len(sample_chunks) - 1:
                    parts.append(silence)
            full_sample = np.concatenate(parts)

            sample_path = os.path.join(samples_dir, f"{spk_id}.wav")
            sf.write(sample_path, full_sample, sr)
            profile.sample_path = sample_path
            profile.sample_text = " ".join(sample_texts)

        if accumulated < MIN_SPEAKER_SAMPLE_SEC:
            print(f"   ⚠️  {spk_id} : seulement {accumulated:.1f}s — qualité de clonage réduite")

        # Ref clips individuels (3-15s, max 10)
        ref_clips = []
        for ci, seg in enumerate(segs_sorted):
            if len(ref_clips) >= 10:
                break
            dur = seg.end - seg.start
            if dur < 3.0 or dur > 15.0:
                continue
            clip = _read_audio_slice(audio_path, seg.start, seg.end, sr, n_frames)
            if len(clip) == 0:
                continue
            clip_path = os.path.join(samples_dir, f"{spk_id}_ref{ci:02d}.wav")
            sf.write(clip_path, clip, sr)
            ref_clips.append([clip_path, seg.text.strip(), float(seg.start), float(seg.end)])
        profile.ref_clips = ref_clips

        if full_sample is not None:
            gender, f0_median = _estimate_gender(full_sample, sr)
            profile.gender = gender
            profile.f0_median = f0_median
            icon = "♀️" if gender == "female" else "♂️" if gender == "male" else "❓"
            print(f"   🎤 {spk_id} {icon} : {accumulated:.1f}s sample, "
                  f"{len(ref_clips)} clips ref, {len(segs)} seg, "
                  f"{total_dur:.0f}s total [F0={f0_median:.0f}Hz]")

        profiles[spk_id] = profile

    return profiles


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 4 : VÉRIFICATION DES NOMS PROPRES (CRITIQUE)
# ═══════════════════════════════════════════════════════════════════════════════

def verify_proper_nouns(segments: list[Segment], analysis: ContentAnalysis,
                         client, src_lang: str, context: str = "",
                         claude_model: str = CLAUDE_MODEL_PRO) -> list[Segment]:
    """
    Passe critique : Whisper écrit souvent les noms propres phonétiquement
    (« Treeta Parci » au lieu de « Trita Parsi »). On demande à Opus de
    scanner chaque fenêtre de 80 segments en comparant avec le context
    utilisateur et le glossaire produit par analyze_content, et de
    corriger le champ `text` source lorsqu'une incohérence est détectée.

    Retourne la liste de segments (mêmes objets, `text` possiblement mis à jour).
    """
    print("\n🔍 Passe 4 — Vérification des noms propres (correction orthographique source)...")

    if not context and not analysis.glossary:
        print("   ⏩ Pas de contexte ni de glossaire — passe ignorée")
        return segments

    window_size = 80
    n_windows = (len(segments) + window_size - 1) // window_size

    glossary_terms = []
    for src_term, tgt_term in (analysis.glossary or {}).items():
        glossary_terms.append(f"{src_term} → {tgt_term}")

    ctx_block = f"\nCONTEXTE UTILISATEUR : {context}\n" if context else ""
    gloss_block = ""
    if glossary_terms:
        gloss_block = "\nGLOSSAIRE (orthographe de référence) :\n" + \
                      "\n".join(f"  - {t}" for t in glossary_terms) + "\n"

    system_prompt = f"""Tu es un correcteur spécialisé dans la transcription automatique (Whisper).
Whisper transcrit souvent les NOMS PROPRES phonétiquement (mauvaise orthographe).
Ta mission : identifier et corriger UNIQUEMENT les noms propres mal orthographiés
dans le texte source, en t'appuyant sur le contexte fourni et le glossaire.
{ctx_block}{gloss_block}
RÈGLES STRICTES :
1. NE CORRIGE QUE les noms propres (personnes, organisations, lieux, œuvres, marques).
2. NE MODIFIE PAS les mots communs, la syntaxe, ou la structure des phrases.
3. NE TRADUIS RIEN — garde la langue source.
4. Si un nom propre est déjà correct, ne le touche pas.
5. Conserve intégralement la casse et la ponctuation des mots non modifiés.

FORMAT DE RÉPONSE — JSON strict :
{{
  "corrections": [
    {{"index": <int>, "corrected_text": "<texte corrigé du segment>"}},
    ...
  ]
}}
Ne retourne QUE les segments qui contiennent une correction. Liste vide si rien."""

    total_corrections = 0
    for wi in range(n_windows):
        start_idx = wi * window_size
        end_idx = min(start_idx + window_size, len(segments))
        window = segments[start_idx:end_idx]

        payload = "\n".join(f"[{s.index}] {s.text}" for s in window)
        user_prompt = f"SEGMENTS À VÉRIFIER (fenêtre {wi+1}/{n_windows}) :\n\n{payload}\n"

        try:
            resp = _claude_create(
                client, model=claude_model, max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as e:
            print(f"   ⚠️  Fenêtre {wi+1} erreur : {e}")
            continue

        text = resp.content[0].text.strip()
        # Extraire le JSON même si Opus ajoute du texte autour
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            continue
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue

        corrections = data.get("corrections", [])
        for corr in corrections:
            idx = corr.get("index")
            new_text = corr.get("corrected_text", "").strip()
            if not isinstance(idx, int) or not new_text:
                continue
            # Retrouver le segment (les index sont 1-based dans save_seg)
            for s in window:
                if s.index == idx and s.text != new_text:
                    s.text = new_text
                    total_corrections += 1
                    break

        print(f"   ✅ Fenêtre {wi+1}/{n_windows} — {len(corrections)} correction(s) proposée(s)")

    print(f"   🎯 Total : {total_corrections} correction(s) de noms propres appliquée(s)")
    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 7b : AUDIT SÉMANTIQUE OPUS DES COUPURES PROBLÉMATIQUES
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt_hms(sec: float) -> str:
    """Formate un timecode h:mm:ss (hh toujours affiché)."""
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 1b — Détection + récupération des trous de transcription WhisperX
# ═══════════════════════════════════════════════════════════════════════════════
#
# WhisperX (via sa VAD Pyannote + son chunking interne) peut parfois sauter
# un bout de parole — typiquement quand deux locuteurs se chevauchent, ou
# quand la segmentation tombe pile au milieu d'une phrase. Le symptôme est
# un trou de plusieurs secondes entre deux segments consécutifs, alors que
# l'audio contient clairement de la voix.
#
# Cette passe :
#   1. Détecte tous les trous ≥ min_gap_sec entre segments consécutifs
#   2. Mesure le niveau audio de chaque trou (ffmpeg volumedetect)
#   3. Pour chaque trou NON silencieux, ré-extrait la tranche et la passe
#      à un WhisperX isolé — sans VAD préalable, donc sans le risque de
#      chunking hostile qui avait sauté le passage.
#   4. Insère les segments récupérés dans la liste, re-numérotée.
#
# Coût : +1 chargement WhisperX (même GPU), mais uniquement si des trous
# suspects existent. Quelques secondes par trou.

def fill_transcription_gaps(segments: list[Segment], audio_path: str,
                             source_lang: str,
                             min_gap_sec: float = 2.5,
                             silence_db_threshold: float = -40.0,
                             pad_sec: float = 0.3) -> list[Segment]:
    """Détecte les trous suspects entre segments WhisperX (≥ min_gap_sec),
    vérifie qu'ils contiennent de la voix (niveau > silence_db_threshold),
    et re-transcrit chacun pour récupérer le texte manqué.

    Retourne la liste combinée, triée et re-numérotée.
    """
    if len(segments) < 2:
        return segments

    # ── Étape 1 : trous candidats ────────────────────────────────────────
    candidates = []
    for i in range(len(segments) - 1):
        a, b = segments[i], segments[i+1]
        gap = b.start - a.end
        if gap >= min_gap_sec:
            candidates.append((i, a.end, b.start, gap))

    if not candidates:
        print(f"   ✅ Aucun trou ≥ {min_gap_sec:.1f}s dans la transcription")
        return segments

    print(f"   🔍 {len(candidates)} trou(s) ≥ {min_gap_sec:.1f}s à vérifier")

    # ── Étape 2 : filtrage par volume (ignorer les vrais silences) ───────
    voice_gaps = []
    for (idx, gstart, gend, gdur) in candidates:
        ss = max(0.0, gstart - pad_sec)
        to = gend + pad_sec
        cmd = [
            "ffmpeg", "-hide_banner", "-nostats",
            "-ss", f"{ss:.3f}", "-to", f"{to:.3f}",
            "-i", audio_path,
            "-vn", "-af", "volumedetect", "-f", "null", "-"
        ]
        mean_db = -90.0
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            out = (r.stderr or "") + (r.stdout or "")
            for line in out.splitlines():
                if "mean_volume:" in line:
                    try:
                        mean_db = float(line.split("mean_volume:")[-1].split("dB")[0].strip())
                    except ValueError:
                        pass
                    break
        except Exception as e:
            print(f"      ⚠️  volumedetect échec [{_fmt_hms(gstart)}] : {e}")

        if mean_db >= silence_db_threshold:
            voice_gaps.append((idx, gstart, gend, gdur, mean_db))
            print(f"      🎙️  [{_fmt_hms(gstart)}→{_fmt_hms(gend)}] "
                  f"{gdur:.1f}s  vol={mean_db:.1f}dB  → voix probable")
        else:
            print(f"      🤫 [{_fmt_hms(gstart)}→{_fmt_hms(gend)}] "
                  f"{gdur:.1f}s  vol={mean_db:.1f}dB  → silence")

    if not voice_gaps:
        print("   ✅ Tous les trous sont de vrais silences, rien à récupérer")
        return segments

    # ── Étape 3 : re-transcription WhisperX par tranche ──────────────────
    import whisperx, torch, gc, tempfile

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = traduire.WHISPER_COMPUTE_TYPE if device == "cuda" else "int8"

    source_langs = [x.strip() for x in source_lang.split(",")]
    is_multi = len(source_langs) > 1
    wx_lang = None if is_multi else source_langs[0]

    print(f"   📥 Chargement WhisperX {traduire.WHISPER_MODEL} ({device}) pour récupération…")
    model = whisperx.load_model(traduire.WHISPER_MODEL, device,
                                 compute_type=compute_type, language=wx_lang)

    align_cache: dict = {}   # lang_code → (align_model, metadata) | None
    new_segments: list[Segment] = []

    for (idx, gstart, gend, gdur, vol) in voice_gaps:
        ss = max(0.0, gstart - pad_sec)
        to = gend + pad_sec
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{ss:.3f}", "-to", f"{to:.3f}",
                "-i", audio_path, "-vn",
                "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                tmp_path,
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            audio_slice = whisperx.load_audio(tmp_path)
            # batch_size=1 : tranche courte, pas besoin de parallélisme
            res = model.transcribe(audio_slice, batch_size=1, language=wx_lang)
            raw_segs = res.get("segments", []) or []
            lang_code = res.get("language") or wx_lang or "en"

            # Alignement optionnel pour récupérer les mots
            if raw_segs and lang_code not in align_cache:
                try:
                    ma, md = whisperx.load_align_model(
                        language_code=lang_code, device=device)
                    align_cache[lang_code] = (ma, md)
                except Exception as e:
                    print(f"      ⚠️  alignement [{lang_code}] indisponible : {e}")
                    align_cache[lang_code] = None
            am = align_cache.get(lang_code)
            if am is not None and raw_segs:
                try:
                    res_a = whisperx.align(
                        raw_segs, am[0], am[1], audio_slice, device,
                        return_char_alignments=False)
                    raw_segs = res_a.get("segments", raw_segs) or raw_segs
                except Exception as e:
                    print(f"      ⚠️  align() échec : {e}")

            # Reconstruire les Segment avec timings absolus, clampés au trou
            added_here = 0
            for rs in raw_segs:
                abs_start = ss + float(rs.get("start", 0))
                abs_end   = ss + float(rs.get("end", 0))
                text = (rs.get("text") or "").strip()
                if not text:
                    continue
                abs_start = max(abs_start, gstart)
                abs_end   = min(abs_end,   gend)
                if abs_end - abs_start < 0.2:
                    continue

                words = []
                for w in rs.get("words", []) or []:
                    try:
                        ws = ss + float(w.get("start", 0))
                        we = ss + float(w.get("end", 0))
                    except (TypeError, ValueError):
                        continue
                    if ws >= gstart and we <= gend:
                        words.append({
                            "word": w.get("word", ""),
                            "start": ws, "end": we,
                            "score": w.get("score", 0),
                        })

                new_segments.append(Segment(
                    index=-1,  # re-numéroté plus bas
                    start=abs_start, end=abs_end,
                    text=text, words=words,
                    lang=lang_code if is_multi else "",
                ))
                added_here += 1
                print(f"      ➕ [{_fmt_hms(abs_start)}] {text[:100]}")

            if added_here == 0:
                print(f"      ⚠️  [{_fmt_hms(gstart)}] aucun segment récupéré "
                      f"(voix mais transcription vide)")

        except subprocess.CalledProcessError as e:
            err = (e.stderr or b"").decode("utf-8", errors="replace").strip()[:200]
            print(f"      ❌ ffmpeg échec [{_fmt_hms(gstart)}] : {err}")
        except Exception as e:
            print(f"      ❌ récupération [{_fmt_hms(gstart)}] : {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except OSError: pass

    # Libérer la VRAM
    del model
    for am in align_cache.values():
        if am is not None:
            del am
    align_cache.clear()
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    if not new_segments:
        print("   ✅ Aucun segment récupéré (trous sans parole exploitable)")
        return segments

    # ── Étape 4 : fusionner et re-numéroter ──────────────────────────────
    all_segs = list(segments) + new_segments
    all_segs.sort(key=lambda s: s.start)
    for i, s in enumerate(all_segs):
        s.index = i + 1

    print(f"   ✅ {len(new_segments)} segment(s) récupéré(s) "
          f"→ {len(all_segs)} total (avant: {len(segments)})")
    return all_segs


def audit_cuts_with_opus(subtitles: list[Subtitle], client,
                          tgt_lang: str, context: str = "",
                          claude_model: str = CLAUDE_MODEL_PRO,
                          cps_threshold: float = 14.0,
                          log_path: str | None = None) -> list[Subtitle]:
    """
    Passe les sous-titres avec CPS proche de la limite (ou coupures suspectes)
    à Opus en batch pour reformulation plus courte/plus naturelle sans
    dépasser MAX_CHARS_PER_LINE.

    Ne modifie QUE le texte des sous-titres problématiques. Les timings
    sont conservés. Après application, le code appelant doit ré-exécuter
    l'audit CPS et le deorphan via `resegment` (ou les helpers exposés).

    Si `log_path` est fourni, un rapport listant chaque reformulation
    appliquée (avec timecode h:mm:ss, CPS source, texte avant/après) est
    écrit à cet emplacement — permet à l'utilisateur de relire manuellement
    les endroits où une perte de sens a pu se produire.
    """
    print("\n🧠 Passe 7b — Audit sémantique Opus des coupures problématiques...")

    if not subtitles:
        return subtitles

    # Identifier les sous-titres à auditer
    flagged = []
    for i, sub in enumerate(subtitles):
        dur = sub.end - sub.start
        if dur <= 0:
            continue
        txt_len = len(sub.text.replace("\n", " "))
        cps = txt_len / dur
        if cps > cps_threshold:
            flagged.append(i)

    if not flagged:
        print("   ⏩ Aucune coupure problématique détectée")
        return subtitles

    print(f"   🎯 {len(flagged)} sous-titre(s) flaggé(s) (CPS > {cps_threshold})")

    ctx_block = f"\nCONTEXTE : {context}\n" if context else ""
    tgt_name = lang_name(tgt_lang)

    system_prompt = f"""Tu es un sous-titreur professionnel {tgt_name}.
On te présente des sous-titres trop rapides à lire (CPS > {cps_threshold}).
Ta mission : proposer une reformulation PLUS CONCISE UNIQUEMENT SI une
formulation équivalente existe SANS perdre aucune idée.
{ctx_block}
RÈGLES ABSOLUES — FIDÉLITÉ > BRIÈVETÉ :
1. INTERDICTION de supprimer un élément de sens, même subtil
   (politesse, nuance, précision, modalisateur, clause subordonnée).
2. Tu peux UNIQUEMENT :
   - supprimer de vraies redondances strictes (répétitions)
   - remplacer un mot par un synonyme plus court équivalent
   - contracter des tournures verbeuses évidentes
     (ex: "est en train de" → "est", "à ce moment-là" → "alors")
3. Maximum {traduire.MAX_CHARS_PER_LINE} caractères par ligne, maximum 2 lignes.
4. Ne coupe JAMAIS un article/préposition en fin de ligne.
5. Conserve la ponctuation terminale (. ? !) si présente.
6. **Si tu ne peux pas raccourcir significativement SANS perdre de sens,
   OMETS CE SOUS-TITRE de ta réponse** (ne le reformule pas du tout).
   RÈGLE D'OR : un sous-titre à 26 CPS qui conserve TOUT le sens est
   BIEN PRÉFÉRABLE à un sous-titre « dans les clous » mais amputé.
   N'inclus ta reformulation que si tu es certain de ne RIEN perdre.
7. NE JAMAIS inclure d'annotation « (~N CPS) » ou « ⚠️ » dans ta réponse :
   ce sont des métadonnées d'entrée, pas du contenu à restituer.

FORMAT DE RÉPONSE — JSON strict (omets les sous-titres que tu ne peux pas
raccourcir fidèlement) :
{{
  "reformulations": [
    {{"index": <int>, "new_text": "<texte reformulé avec \\n si 2 lignes>"}},
    ...
  ]
}}"""

    batch_size = 20
    total_applied = 0
    # Collecte des reformulations appliquées — chaque entrée est un sous-titre
    # où Opus a accepté de remplacer le texte original : c'est un point où
    # une perte de sens est possible et doit être relue manuellement.
    applied_rewrites: list[dict] = []

    for batch_start in range(0, len(flagged), batch_size):
        batch = flagged[batch_start:batch_start + batch_size]

        batch_items = []
        for idx in batch:
            sub = subtitles[idx]
            dur = sub.end - sub.start
            cps = len(sub.text.replace("\n", " ")) / dur if dur > 0 else 0
            ctx_before = subtitles[idx - 1].text.replace("\n", " / ") if idx > 0 else ""
            ctx_after = subtitles[idx + 1].text.replace("\n", " / ") if idx < len(subtitles) - 1 else ""
            batch_items.append(
                f"[{sub.index}] ({dur:.1f}s, {cps:.0f} cps)\n"
                f"  avant : {ctx_before}\n"
                f"  ACTUEL : {sub.text}\n"
                f"  après : {ctx_after}\n"
            )

        user_prompt = "SOUS-TITRES À REFORMULER :\n\n" + "\n".join(batch_items)

        try:
            resp = _claude_create(
                client, model=claude_model, max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as e:
            print(f"   ⚠️  Batch {batch_start // batch_size + 1} erreur : {e}")
            continue

        text = resp.content[0].text.strip()
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            continue
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue

        for reform in data.get("reformulations", []):
            idx = reform.get("index")
            new_text = reform.get("new_text", "").strip()
            if not isinstance(idx, int) or not new_text:
                continue
            # Défense en profondeur : strip toute annotation CPS que
            # Opus aurait laissée malgré la consigne.
            new_text = traduire._strip_cps_annot(new_text)
            if not new_text:
                continue
            for i in batch:
                if subtitles[i].index == idx:
                    old_text = subtitles[i].text.replace("\n", " ").strip()
                    new_plain = new_text.replace("\n", " ").strip()
                    # Garde-fou anti-troncature renforcé : refuser si Opus
                    # a perdu plus de 15 % du contenu (risque élevé de
                    # suppression d'une clause subordonnée ou d'une nuance).
                    # Mieux vaut garder l'original à 26 CPS que tronquer.
                    if len(new_plain) < len(old_text) * 0.85:
                        continue
                    sub = subtitles[i]
                    old_dur = max(sub.end - sub.start, 0.001)
                    old_cps = len(old_text) / old_dur
                    new_cps = len(new_plain) / old_dur
                    applied_rewrites.append({
                        "index": sub.index,
                        "start": float(sub.start),
                        "end": float(sub.end),
                        "old_cps": round(old_cps, 1),
                        "new_cps": round(new_cps, 1),
                        "old_text": old_text,
                        "new_text": new_plain,
                        "shrink_pct": round(
                            100 * (1 - len(new_plain) / max(len(old_text), 1)), 1
                        ),
                    })
                    subtitles[i].text = new_text
                    total_applied += 1
                    break

    print(f"   ✅ {total_applied} reformulation(s) appliquée(s)")

    # ──────────────────────────────────────────────────────────────────
    # Rapport des timestamps où une perte de sens est possible
    # ──────────────────────────────────────────────────────────────────
    if applied_rewrites:
        print(f"\n   ⚠️  {len(applied_rewrites)} sous-titre(s) reformulé(s) — "
              f"à relire pour éventuelle perte de sens :")
        for r in applied_rewrites[:20]:
            print(f"      [{_fmt_hms(r['start'])}] #{r['index']}  "
                  f"{r['old_cps']:.0f}→{r['new_cps']:.0f} cps  "
                  f"(−{r['shrink_pct']:.0f} %)")
            print(f"         avant : {r['old_text']}")
            print(f"         après : {r['new_text']}")
        if len(applied_rewrites) > 20:
            print(f"      … et {len(applied_rewrites) - 20} autre(s) "
                  f"(voir le log complet)")

        if log_path:
            try:
                with open(log_path, "w", encoding="utf-8") as fh:
                    fh.write("# Rapport des sous-titres reformulés par Opus\n")
                    fh.write(
                        "# Chaque entrée est un point où le texte original a été\n"
                        "# remplacé par une version plus courte — à relire pour\n"
                        "# détecter une éventuelle perte de sens.\n\n"
                    )
                    fh.write(f"Total : {len(applied_rewrites)} reformulation(s)\n")
                    fh.write("=" * 72 + "\n\n")
                    for r in applied_rewrites:
                        fh.write(
                            f"[{_fmt_hms(r['start'])} → {_fmt_hms(r['end'])}]  "
                            f"sous-titre #{r['index']}\n"
                        )
                        fh.write(
                            f"  CPS : {r['old_cps']:.1f} → {r['new_cps']:.1f}   "
                            f"contraction : −{r['shrink_pct']:.1f} %\n"
                        )
                        fh.write(f"  AVANT : {r['old_text']}\n")
                        fh.write(f"  APRÈS : {r['new_text']}\n\n")
                print(f"   📝 Log détaillé : {log_path}")
            except OSError as e:
                print(f"   ⚠️  Impossible d'écrire le log : {e}")

    return subtitles


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 7c : TRIM DES SOUS-TITRES DÉBORDANT SUR LES JINGLES / SILENCES
# ═══════════════════════════════════════════════════════════════════════════════

def _segments_words_map(segments: list[Segment]) -> list[tuple[float, float, float]]:
    """
    Construit une liste plate (start, end, last_word_end) des segments qui
    ont un alignement mot-à-mot disponible. Utilisé pour savoir où se
    termine la parole réelle à l'intérieur d'un segment.
    """
    out = []
    for s in segments:
        words = getattr(s, "words", None) or []
        last_end = None
        for w in words:
            we = w.get("end")
            if we is not None:
                last_end = we
        if last_end is not None:
            out.append((s.start, s.end, float(last_end)))
    return out


def trim_music_padding(subtitles: list[Subtitle],
                       segments: list[Segment] | None = None,
                       min_dur: float = 7.0,
                       max_cps_for_trim: float = 7.0,
                       target_reading_cps: float = 13.0,
                       buffer_sec: float = 0.5,
                       min_gap_next: float = 0.3) -> list[Subtitle]:
    """
    Raccourcit la fin des sous-titres qui "lingerent" au-delà de la parole
    réelle — typiquement pendant un jingle musical ou un long silence
    transitoire que Whisper a rattaché au dernier segment parlé.

    Deux stratégies combinées :

    1. **Alignement mot-à-mot** (si disponible) : pour chaque sous-titre,
       trouve le segment source le plus proche qui contient son intervalle ;
       si `last_word.end + buffer < sub.end`, raccourcit `sub.end`.

    2. **Heuristique CPS** (fallback, sans words) : si un sous-titre a
       `duration >= min_dur` et `cps <= max_cps_for_trim`, c'est un signal
       qu'il déborde d'un jingle/silence. On raccourcit à
       `start + (text_len / target_reading_cps + buffer)`.

    Dans les deux cas, on garantit un écart `min_gap_next` avant le
    sous-titre suivant et on garde au moins 2 s de lecture.
    """
    print("\n🎵 Passe 7c — Trim des sous-titres débordant sur les silences/jingles...")

    if not subtitles:
        return subtitles

    # Map segments → last_word_end (si alignement dispo)
    word_map = _segments_words_map(segments or [])
    has_words = bool(word_map)

    def _last_word_end_for(sub: Subtitle) -> float | None:
        """Cherche le segment source qui couvre le sub, retourne son last word end."""
        if not has_words:
            return None
        # Prend le segment dont l'intervalle recouvre le mieux celui du sub
        best = None
        best_overlap = 0.0
        for (seg_s, seg_e, last_end) in word_map:
            overlap = max(0.0, min(seg_e, sub.end) - max(seg_s, sub.start))
            if overlap > best_overlap:
                best_overlap = overlap
                best = last_end
        return best

    trimmed_count = 0
    total_saved = 0.0

    for i, sub in enumerate(subtitles):
        dur = sub.end - sub.start
        if dur <= 0:
            continue
        txt_plain = sub.text.replace("\n", " ").strip()
        txt_len = len(txt_plain)
        if txt_len == 0:
            continue
        cps = txt_len / dur

        new_end = None
        reason = ""

        # Stratégie 1 — alignement mot-à-mot
        lw_end = _last_word_end_for(sub)
        if lw_end is not None:
            candidate = lw_end + buffer_sec
            if candidate < sub.end - 0.3:
                new_end = candidate
                reason = "word_end"

        # Stratégie 2 — heuristique CPS (si pas de word alignment ou pas utile)
        if new_end is None and dur >= min_dur and cps <= max_cps_for_trim:
            reading_time = txt_len / target_reading_cps + buffer_sec
            reading_time = max(2.0, reading_time)
            candidate = sub.start + reading_time
            if candidate < sub.end - 0.3:
                new_end = candidate
                reason = "cps_heuristic"

        if new_end is None:
            continue

        # Contraintes : min 2 s, écart avec sub suivant
        new_end = max(new_end, sub.start + 2.0)
        if i + 1 < len(subtitles):
            nxt = subtitles[i + 1]
            new_end = min(new_end, nxt.start - min_gap_next)
        # Ne pas aggraver (ne jamais étendre)
        if new_end >= sub.end:
            continue
        if new_end <= sub.start + 1.0:
            continue  # garde-fou : ne pas créer un sub trop court

        saved = sub.end - new_end
        sub.end = new_end
        trimmed_count += 1
        total_saved += saved

    mode = "word-alignment" if has_words else "heuristique CPS"
    print(f"   🎯 Mode : {mode}")
    print(f"   ✅ {trimmed_count} sous-titre(s) raccourci(s), "
          f"{total_saved:.1f}s de jingle/silence masqué(s)")
    return subtitles


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 9 : RÉSUMÉ STRUCTURÉ (portage de resumer.py:585)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_target_words(duration_sec: float) -> int:
    """Calcule le nombre de mots cible pour le résumé (cf. resumer.py:577)."""
    pages = max(2, duration_sec / 60.0 * PAGES_PER_MINUTE)
    return int(pages * WORDS_PER_PAGE)


def generate_summary_pro(segments: list[Segment], analysis: ContentAnalysis,
                          client, duration_sec: float,
                          context: str = "",
                          claude_model: str = CLAUDE_MODEL_PRO,
                          title: str = "") -> str:
    """
    Génère un résumé markdown structuré thématiquement, avec 2-4 citations
    verbatim. Porté et adapté de resumer.py:585 — simplifié pour n'avoir
    pas besoin de ResumeMetadata (on passe juste le titre et la durée).
    """
    target_words = calculate_target_words(duration_sec)

    full_text = "\n".join(
        f"[{s.start:.0f}s] {s.text_tgt or s.text}" for s in segments
    )
    total_chars = len(full_text)

    ctx_block = f"\nCONTEXTE UTILISATEUR : {context}\n" if context else ""
    meta_block = ""
    if title:
        meta_block += f"Titre : {title}\n"
    if duration_sec > 0:
        meta_block += f"Durée : {duration_sec/60:.0f} min\n"

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

    print(f"\n✍️  Passe 9 — Synthèse Claude ({target_words} mots cible, {total_chars} car.)...")

    if total_chars < 100000:
        user_prompt = f"""{meta_block}{ctx_block}
ANALYSE PRÉALABLE :
Résumé : {analysis.summary}
Domaine : {analysis.domain} | Ton : {analysis.tone}
Locuteurs : {analysis.speakers_description}
Glossaire : {json.dumps(analysis.glossary, ensure_ascii=False)}

TRANSCRIPTION COMPLÈTE :
{full_text}

Rédige le résumé structuré (~{target_words} mots)."""

        resp = _claude_create(
            client, model=claude_model, max_tokens=16384,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return resp.content[0].text.strip()

    # Contenu long : 2 passes (extraction + synthèse)
    print("   📚 Contenu long — extraction de points clés en 2 passes")
    chunk_size_chars = 60000
    chunks = []
    current_chunk = []
    current_len = 0
    for line in full_text.split("\n"):
        if current_len + len(line) > chunk_size_chars and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_len = 0
        current_chunk.append(line)
        current_len += len(line) + 1
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    key_points = []
    for i, chunk in enumerate(chunks):
        print(f"   📦 Extraction points clés {i+1}/{len(chunks)}...")
        resp = _claude_create(
            client, model=claude_model, max_tokens=8192,
            system=("Tu extrais les points clés, arguments, exemples et citations "
                    "remarquables d'une transcription. Réponds en français, "
                    "en bullet points structurés."),
            messages=[{"role": "user",
                       "content": f"Extrais les points clés de ce segment :\n\n{chunk}"}],
        )
        key_points.append(resp.content[0].text.strip())

    all_points = "\n\n---\n\n".join(key_points)
    user_prompt = f"""{meta_block}{ctx_block}
ANALYSE PRÉALABLE :
Résumé : {analysis.summary}
Domaine : {analysis.domain} | Ton : {analysis.tone}
Locuteurs : {analysis.speakers_description}

POINTS CLÉS EXTRAITS :
{all_points}

Rédige le résumé structuré (~{target_words} mots) à partir de ces points clés."""

    print("   ✍️  Rédaction finale...")
    resp = _claude_create(
        client, model=claude_model, max_tokens=16384,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return resp.content[0].text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 10 : TITRE VIRAL + DESCRIPTION + CHAPITRAGE (YouTube & X)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_promo_kit(segments: list[Segment], analysis: ContentAnalysis,
                        client, duration_sec: float,
                        context: str = "",
                        claude_model: str = CLAUDE_MODEL_PRO) -> dict:
    """
    Demande à Opus un JSON avec titre viral, descriptions YouTube/X, et
    chapitrage. Force en post-traitement le snap des chapitres sur le
    `start` du segment le plus proche (Opus peut halluciner des timecodes).
    """
    print("\n📣 Passe 10 — Titre viral + description + chapitrage...")

    # Timeline compacte (index + start + traduction) pour Opus
    timeline = "\n".join(
        f"[{s.start:.0f}s] {s.text_tgt or s.text}"
        for s in segments
    )
    if len(timeline) > 60000:
        # Échantillonnage : on garde 1 segment sur N pour tenir en 60k car
        step = max(1, len(segments) // 200)
        timeline = "\n".join(
            f"[{segments[i].start:.0f}s] {segments[i].text_tgt or segments[i].text}"
            for i in range(0, len(segments), step)
        )

    ctx_block = f"\nCONTEXTE UTILISATEUR : {context}\n" if context else ""

    system_prompt = f"""Tu es un expert en stratégie de contenu YouTube et X/Twitter francophones.
Tu produis des kits de lancement viraux : titres accrocheurs, descriptions
optimisées pour le SEO et l'engagement, chapitrage YouTube conforme aux normes.
{ctx_block}
ANALYSE DU CONTENU :
Domaine : {analysis.domain}
Ton : {analysis.tone}
Locuteurs : {analysis.speakers_description}
Résumé : {analysis.summary}
Durée totale : {duration_sec/60:.1f} minutes

RÈGLES STRICTES POUR LES CHAPITRES YOUTUBE :
- Le PREMIER chapitre doit commencer exactement à 0 seconde.
- Au moins 3 chapitres au total.
- Chaque chapitre dure au moins 10 secondes.
- Les `start` doivent être pris DANS les timestamps de la timeline fournie
  (ce sont de vrais moments de transition thématique).
- Les titres de chapitres sont courts, descriptifs, en français (2-6 mots).

FORMAT DE RÉPONSE — JSON STRICT uniquement, sans commentaire :
{{
  "titre_viral": "<titre français accrocheur, max 100 car.>",
  "description_youtube": "<description de 3-5 paragraphes avec hashtags en fin>",
  "description_x": "<post X/Twitter de max 270 caractères, avec 2-3 hashtags>",
  "chapitres": [
    {{"start": 0.0, "titre": "Introduction"}},
    {{"start": 182.0, "titre": "Le contexte géopolitique"}}
  ]
}}"""

    user_prompt = f"""TIMELINE DE LA VIDÉO (timestamps en secondes) :

{timeline}

Produis le kit de promotion en JSON strict selon le format demandé."""

    resp = _claude_create(
        client, model=claude_model, max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    text = resp.content[0].text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        print("   ⚠️  Réponse Opus non parseable — kit promo vide")
        return {"titre_viral": "", "description_youtube": "", "description_x": "", "chapitres": []}

    try:
        kit = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        print(f"   ⚠️  JSON invalide : {e}")
        return {"titre_viral": "", "description_youtube": "", "description_x": "", "chapitres": []}

    # Snap des chapitres sur le `start` du segment le plus proche
    if segments and kit.get("chapitres"):
        seg_starts = [s.start for s in segments]
        snapped = []
        for ch in kit["chapitres"]:
            raw_start = float(ch.get("start", 0))
            # Trouver le segment le plus proche
            closest = min(seg_starts, key=lambda t: abs(t - raw_start))
            snapped.append({"start": closest, "titre": ch.get("titre", "").strip()})
        # Forcer le premier chapitre à 0
        if snapped:
            snapped[0]["start"] = 0.0
        # Déduplication + tri
        seen = set()
        dedup = []
        for ch in sorted(snapped, key=lambda c: c["start"]):
            if ch["start"] in seen:
                continue
            seen.add(ch["start"])
            dedup.append(ch)
        # Garantir au moins 10s entre chapitres successifs
        filtered = []
        for ch in dedup:
            if not filtered or (ch["start"] - filtered[-1]["start"]) >= 10.0:
                filtered.append(ch)
        kit["chapitres"] = filtered

    print(f"   ✅ Titre : {kit.get('titre_viral', '')[:60]}")
    print(f"   ✅ Chapitres : {len(kit.get('chapitres', []))}")
    return kit


def write_promo_files(kit: dict, youtube_path: str, x_path: str) -> None:
    """Écrit les fichiers de description YouTube et X à partir du kit."""
    chapitres = kit.get("chapitres", [])
    titre = kit.get("titre_viral", "")
    desc_yt = kit.get("description_youtube", "")
    desc_x = kit.get("description_x", "")

    # YouTube : titre + description + chapitres au format h:mm:ss
    yt_lines = []
    if titre:
        yt_lines.append(titre)
        yt_lines.append("")
    if desc_yt:
        yt_lines.append(desc_yt)
        yt_lines.append("")
    if chapitres:
        yt_lines.append("CHAPITRES")
        for ch in chapitres:
            tc = _fmt_timecode_youtube(ch.get("start", 0))
            yt_lines.append(f"{tc} {ch.get('titre', '')}")
    with open(youtube_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yt_lines).rstrip() + "\n")

    # X/Twitter : version courte (post + chapitres compactés en mention)
    x_lines = []
    if titre:
        x_lines.append(titre)
        x_lines.append("")
    if desc_x:
        x_lines.append(desc_x)
    if chapitres:
        x_lines.append("")
        x_lines.append("Chapitres :")
        for ch in chapitres[:6]:  # limiter pour rester lisible
            tc = _fmt_timecode_youtube(ch.get("start", 0))
            x_lines.append(f"  {tc} — {ch.get('titre', '')}")
    with open(x_path, "w", encoding="utf-8") as f:
        f.write("\n".join(x_lines).rstrip() + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 10b : SUGGESTIONS DE TITRES (du plus viral au plus sobre)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_title_suggestions(segments: list[Segment], analysis: ContentAnalysis,
                                client, context: str = "",
                                claude_model: str = CLAUDE_MODEL_PRO) -> str:
    """
    Génère 8-10 suggestions de titres classées du plus viral au plus sobre.
    Retourne le texte brut prêt à écrire dans un fichier.
    """
    print("\n🏷️  Passe 10b — Suggestions de titres (viral → sobre)...")

    timeline = "\n".join(
        f"[{s.start:.0f}s] {s.text_tgt or s.text}"
        for s in segments
    )
    if len(timeline) > 60000:
        step = max(1, len(segments) // 200)
        timeline = "\n".join(
            f"[{segments[i].start:.0f}s] {segments[i].text_tgt or segments[i].text}"
            for i in range(0, len(segments), step)
        )

    ctx_block = f"\nCONTEXTE UTILISATEUR : {context}\n" if context else ""

    system_prompt = f"""Tu es un expert en stratégie de contenu vidéo francophone.
Tu proposes des titres avec différents niveaux de « viralité » : du plus accrocheur/clickbait
au plus sobre/académique. Chaque titre est étiqueté avec son registre.
{ctx_block}
ANALYSE DU CONTENU :
Domaine : {analysis.domain}
Ton : {analysis.tone}
Locuteurs : {analysis.speakers_description}
Résumé : {analysis.summary}

REGISTRES À COUVRIR (dans cet ordre) :
1. 🔥 VIRAL — clickbait assumé, émotion forte, interpellation directe
2. 🔥 VIRAL — variante différente, angle alternatif
3. 📢 ACCROCHEUR — accrocheur mais honnête, promesse claire
4. 📢 ACCROCHEUR — variante
5. 📰 INFORMATIF — factuel et engageant, style presse en ligne
6. 📰 INFORMATIF — variante
7. 🎓 SOBRE — style documentaire ou académique, neutre
8. 🎓 SOBRE — variante

RÈGLES :
- Chaque titre fait 60-100 caractères max
- En français
- Un titre par ligne, préfixé par son étiquette de registre
- Pas de numérotation
- Pas de guillemets autour du titre
- 8 à 10 titres au total

Réponds UNIQUEMENT avec la liste de titres, rien d'autre."""

    user_prompt = f"""TIMELINE DE LA VIDÉO :

{timeline}

Propose 8 à 10 titres classés du plus viral au plus sobre."""

    resp = _claude_create(
        client, model=claude_model, max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    text = resp.content[0].text.strip()
    # Garder uniquement les lignes qui commencent par un emoji de registre
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    filtered = [l for l in lines if any(l.startswith(tag) for tag in ("🔥", "📢", "📰", "🎓"))]
    result = "\n".join(filtered) if filtered else text

    count = len(filtered) if filtered else len(lines)
    print(f"   ✅ {count} titres générés")
    return result


def generate_youtube_kit_en(segments: list[Segment], analysis: ContentAnalysis,
                             kit_fr: dict, client, duration_sec: float,
                             context: str = "",
                             claude_model: str = CLAUDE_MODEL_PRO) -> str:
    """
    Génère un fichier unique en anglais pour YouTube : titres (viral → sobre),
    description et chapitrage. Utilise le kit FR existant pour les timecodes
    des chapitres (déjà snappés sur les segments).
    """
    print("\n🇬🇧 Passe 10b-en — Kit YouTube anglais (titres + description + chapitres)...")

    timeline = "\n".join(
        f"[{s.start:.0f}s] {s.text_tgt or s.text}"
        for s in segments
    )
    if len(timeline) > 60000:
        step = max(1, len(segments) // 200)
        timeline = "\n".join(
            f"[{segments[i].start:.0f}s] {segments[i].text_tgt or segments[i].text}"
            for i in range(0, len(segments), step)
        )

    # Chapitres FR déjà validés — on les fournit pour que Claude les traduise
    chapitres_fr = kit_fr.get("chapitres", [])
    chap_block = ""
    if chapitres_fr:
        chap_lines = [f"  {_fmt_timecode_youtube(ch['start'])} {ch['titre']}" for ch in chapitres_fr]
        chap_block = f"""
CHAPITRES (version française, à traduire en anglais — NE PAS modifier les timecodes) :
{chr(10).join(chap_lines)}
"""

    ctx_block = f"\nUSER CONTEXT: {context}\n" if context else ""

    system_prompt = f"""You are an expert YouTube content strategist.
You produce English-language YouTube launch kits: catchy titles, SEO-optimized
descriptions, and chapter timestamps.
{ctx_block}
CONTENT ANALYSIS:
Domain: {analysis.domain}
Tone: {analysis.tone}
Speakers: {analysis.speakers_description}
Summary: {analysis.summary}
Total duration: {duration_sec/60:.1f} minutes
{chap_block}
TASK — produce a SINGLE text file with three sections:

1. TITLES (8-10 suggestions, from most viral/clickbait to most sober/academic)
   Each title on its own line, prefixed with its register tag:
   🔥 VIRAL — ...
   📢 CATCHY — ...
   📰 INFORMATIVE — ...
   🎓 FORMAL — ...

2. YOUTUBE DESCRIPTION (3-5 paragraphs, with hashtags at the end)

3. CHAPTERS (use the EXACT same timecodes from the French chapters above,
   just translate the titles into English)
   Format: m:ss or h:mm:ss followed by title

Separate each section with a blank line and a header (TITLES / DESCRIPTION / CHAPTERS).
Output the raw text only, no markdown formatting, no JSON."""

    user_prompt = f"""VIDEO TIMELINE (timestamps in seconds, text is the French translation):

{timeline}

Produce the full English YouTube kit as described."""

    resp = _claude_create(
        client, model=claude_model, max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    text = resp.content[0].text.strip()
    # Compter les sections trouvées
    sections = sum(1 for header in ("TITLES", "DESCRIPTION", "CHAPTERS") if header in text.upper())
    print(f"   ✅ Kit anglais généré ({sections}/3 sections)")
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 10c : EXTRAITS VIRAUX + POSTS SOCIAUX
# ═══════════════════════════════════════════════════════════════════════════════

def select_viral_excerpts(segments: list[Segment], analysis: ContentAnalysis,
                           client, context: str = "",
                           claude_model: str = CLAUDE_MODEL_PRO,
                           max_excerpts: int = 3) -> list[dict]:
    """
    Sélectionne les meilleurs extraits viraux dans la transcription traduite.
    Retourne une liste de dicts {seg_start, seg_end, start, end, titre, justification}.
    """
    print(f"\n🎬 Passe 10c — Sélection de {max_excerpts} extraits viraux...")

    transcript = "\n".join(
        f"[{i}] [{s.start:.1f}s → {s.end:.1f}s] {s.text_tgt or s.text}"
        for i, s in enumerate(segments)
    )
    if len(transcript) > 60000:
        step = max(1, len(segments) // 200)
        transcript = "\n".join(
            f"[{i}] [{segments[i].start:.1f}s → {segments[i].end:.1f}s] {segments[i].text_tgt or segments[i].text}"
            for i in range(0, len(segments), step)
        )

    ctx_block = f"\nCONTEXTE : {context}\n" if context else ""

    system_prompt = f"""Tu es un expert en montage vidéo virale et réseaux sociaux.
{ctx_block}
ANALYSE DU CONTENU :
Domaine : {analysis.domain}
Ton : {analysis.tone}
Locuteurs : {analysis.speakers_description}
Résumé : {analysis.summary}

CONSIGNES :
- Sélectionne les {max_excerpts} meilleur(s) passage(s) pour devenir des extraits viraux
- Chaque extrait doit durer entre 30s et 120s
- Indique les indices de segments (seg_start et seg_end inclus)
- Privilégie :
  • Accroche forte dès les premières secondes (hook)
  • Contenu dense et percutant, pas de creux
  • Fin propre (phrase complète, pas coupée au milieu)
  • Autonomie : l'extrait doit être compréhensible hors contexte
- Évite les passages avec trop de hésitations ou de digressions
- Chaque extrait doit avoir un titre court et accrocheur

Réponds UNIQUEMENT en JSON strict (sans markdown) :
{{
  "excerpts": [
    {{
      "seg_start": <indice premier segment>,
      "seg_end": <indice dernier segment>,
      "titre": "<titre court et accrocheur>",
      "justification": "<pourquoi ce passage est percutant>"
    }}
  ]
}}"""

    user_prompt = f"""TRANSCRIPTION TRADUITE ({len(segments)} segments) :

{transcript}

Sélectionne les {max_excerpts} meilleurs extraits viraux en JSON strict."""

    resp = _claude_create(
        client, model=claude_model, max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    text = resp.content[0].text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        print("   ⚠️  Réponse non parseable — aucun extrait")
        return []

    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        print(f"   ⚠️  JSON invalide : {e}")
        return []

    excerpts = data.get("excerpts", [])
    # Valider et enrichir avec les vrais timecodes
    validated = []
    for ex in excerpts:
        seg_s = int(ex.get("seg_start", 0))
        seg_e = int(ex.get("seg_end", seg_s))
        if seg_s < 0 or seg_e >= len(segments) or seg_s > seg_e:
            continue
        validated.append({
            "seg_start": seg_s,
            "seg_end": seg_e,
            "start": segments[seg_s].start,
            "end": segments[seg_e].end,
            "titre": ex.get("titre", "").strip(),
            "justification": ex.get("justification", "").strip(),
        })

    print(f"   ✅ {len(validated)} extraits sélectionnés")
    for i, ex in enumerate(validated, 1):
        dur = ex["end"] - ex["start"]
        print(f"      {i}. [{ex['start']:.0f}s → {ex['end']:.0f}s] ({dur:.0f}s) — {ex['titre']}")
    return validated


def generate_excerpt_social_post(excerpt: dict, segments: list[Segment],
                                  analysis: ContentAnalysis, client,
                                  claude_model: str = CLAUDE_MODEL_PRO) -> str:
    """
    Génère un post social (citations uniquement) pour un extrait donné.
    Même format que clipper.py : guillemets français, attribution, pas de hashtags.
    """
    seg_s, seg_e = excerpt["seg_start"], excerpt["seg_end"]
    transcript = "\n".join(
        (s.text_tgt or s.text) for s in segments[seg_s:seg_e + 1]
    )

    # Bloc locuteur
    speaker_block = ""
    if analysis.speakers_description:
        speaker_block = f"\nLOCUTEUR(S) : {analysis.speakers_description}\n"

    prompt = f"""Tu es un expert en rédaction de posts pour les réseaux sociaux.

TITRE DU CLIP : {excerpt['titre']}
{speaker_block}
TRANSCRIPTION DU CLIP :
{transcript}

Écris un post pour les réseaux sociaux en français, composé UNIQUEMENT de citations.

RÈGLES STRICTES :
- Le post ne contient QUE des citations littérales entre guillemets français « »
- Chaque citation DOIT apparaître MOT POUR MOT dans la transcription ci-dessus (pas de reformulation, pas d'invention, pas de résumé)
- Tu peux sélectionner 1 à 3 citations percutantes parmi la transcription
- Tu peux couper une phrase longue avec [...] mais JAMAIS changer les mots
- Après les citations, ajoute l'attribution sur une nouvelle ligne :
  • Si un locuteur est fourni : — Locuteur
  • Sinon : — Source vidéo
- AUCUN commentaire, aucune phrase de ton cru, aucune reformulation, aucune introduction
- Pas de hashtags, pas d'emojis
- Prêt à copier-coller, rien d'autre

Réponds UNIQUEMENT avec le texte du post."""

    resp = _claude_create(
        client, model=claude_model, max_tokens=1024,
        system="Tu rédiges des posts sociaux composés exclusivement de citations verbatim.",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def _fmt_timecode_hms(sec: float) -> str:
    """Formate en H:MM:SS pour affichage lisible."""
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def write_excerpt_files(excerpts: list[dict], posts: list[str],
                         base_path: str) -> list[str]:
    """
    Écrit un fichier _excerpt{N}_socialpost.txt par extrait.
    Retourne la liste des chemins créés.
    """
    paths = []
    for i, (ex, post) in enumerate(zip(excerpts, posts), 1):
        path = f"{base_path}_excerpt{i}_socialpost.txt"
        dur = ex["end"] - ex["start"]
        tc_start = _fmt_timecode_hms(ex["start"])
        tc_end = _fmt_timecode_hms(ex["end"])
        lines = [
            post,
            "",
            "---",
            f"Extrait : {tc_start} → {tc_end} (durée {dur:.0f}s)",
            f"Titre : {ex['titre']}",
        ]
        if ex.get("justification"):
            lines.append(f"Justification : {ex['justification']}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines).rstrip() + "\n")
        paths.append(path)
    return paths


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 11 : EXPORT `_doublage.json` (pour un futur doubler-deluxe.py)
# ═══════════════════════════════════════════════════════════════════════════════

def export_doublage_kit(
    output_path: str,
    source_video: str,
    src_lang: str,
    tgt_lang: str,
    segments: list[Segment],
    analysis: ContentAnalysis,
    context: str,
    speaker_profiles: dict[str, SpeakerProfile],
) -> None:
    """Sérialise le kit complet en JSON pour un pipeline de doublage ultérieur."""
    print("\n💾 Passe 11 — Export du kit de doublage (JSON)...")

    speakers_out = []
    for spk_id, prof in speaker_profiles.items():
        speakers_out.append({
            "id": spk_id,
            "name": prof.name,
            "gender": prof.gender,
            "f0_median": prof.f0_median,
            "total_duration": prof.total_duration,
            "segment_count": prof.segment_count,
            "sample_path": prof.sample_path,
            "sample_text": prof.sample_text,
            "ref_clips": prof.ref_clips,
        })

    segments_out = []
    for s in segments:
        segments_out.append({
            "index": s.index,
            "start": s.start,
            "end": s.end,
            "duration": s.end - s.start,
            "text_src": s.text,
            "text_tgt": s.text_tgt,
            "speaker_id": getattr(s, "speaker", "SPEAKER_00"),
            "lang": s.lang,
            "words": s.words,
            "notes": "",
        })

    kit = {
        "schema_version": "1.0",
        "source_video": os.path.basename(source_video),
        "source_lang": src_lang,
        "target_lang": tgt_lang,
        "context": context,
        "analysis": asdict(analysis),
        "proper_nouns": getattr(analysis, "glossary", {}) or {},
        "speakers": speakers_out,
        "segments": segments_out,
        "recommendations": {
            "tts_backend": "elevenlabs",
            "voice_matching": "clone par locuteur via ref_clips",
            "isochrony_target_ratio": 1.0,
            "notes": (
                "Utiliser `ref_clips` pour cloner la voix de chaque locuteur. "
                "Cible isochrone 1.0 = respect strict des timings source."
            ),
        },
    }

    _save_json(kit, output_path)
    print(f"   ✅ {output_path} ({len(segments_out)} seg, {len(speakers_out)} speakers)")


# ═══════════════════════════════════════════════════════════════════════════════
# PASSE 12 : RAPPORT FINAL
# ═══════════════════════════════════════════════════════════════════════════════

def write_final_report(path: str, args, outputs: dict, elapsed: float,
                        claude_model: str, context: str) -> None:
    """Rapport texte récapitulatif de la session."""
    lines = [
        "=" * 70,
        "TRADUIRE-PRO — RAPPORT DE SESSION",
        "=" * 70,
        f"Source         : {args.source}",
        f"Langues        : {args.source_lang} → {args.target_lang}",
        f"Modèle Claude  : {claude_model}",
        f"Durée session  : {elapsed/60:.1f} min",
        f"Contexte user  : {context or '(aucun)'}",
        "",
        "PASSES EXÉCUTÉES",
        f"  - Dubbing ASS         : {'non' if args.no_dubbing else 'oui'}",
        f"  - Audit cuts Opus     : {'oui' if args.audit_cuts else 'non'}",
        f"  - Trim music/silence  : {'non' if args.no_trim_music else 'oui'}",
        f"  - Résumé              : {'non' if args.skip_summary else 'oui'}",
        f"  - Burn vidéo          : {'non' if args.skip_burn else 'oui'}",
        f"  - Max CPS             : {args.max_cps} (tol. +{CPS_TOLERANCE_PRO})",
        "",
        "LIVRABLES",
    ]
    for label, path_val in outputs.items():
        exists = "✅" if path_val and os.path.exists(path_val) else "⏭️"
        lines.append(f"  {exists} {label:<22s} : {path_val or '(non produit)'}")
    lines.append("=" * 70)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def _attach_speakers_from_json(segments: list[Segment], speakers_json: str) -> None:
    """Restore les speakers depuis un fichier {base}_pro_work/speakers.json."""
    if not os.path.exists(speakers_json):
        for s in segments:
            s.speaker = "SPEAKER_00"
        return
    data = _load_json(speakers_json)
    idx_map = {int(k): v for k, v in data.get("index_to_speaker", {}).items()}
    for s in segments:
        s.speaker = idx_map.get(s.index, "SPEAKER_00")


def _get_audio_duration(audio_path: str) -> float:
    """Retourne la durée en secondes via ffprobe."""
    import subprocess
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "csv=p=0", audio_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        try:
            return float(r.stdout.strip())
        except ValueError:
            return 0.0
    return 0.0


def main():
    p = argparse.ArgumentParser(
        description="Kit vidéo complet deluxe : sous-titres pro + résumé + promo + kit doublage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Exemples :
              python traduire-pro.py interview.mp4
              python traduire-pro.py video.mp4 --context "Interview de Trita Parsi (Quincy Institute)"
              python traduire-pro.py talk.mp4 -s en -t fr --no-audit-cuts
              python traduire-pro.py video.mp4 --resume video_pro_work/segments_finaux.json
              python traduire-pro.py video.mp4 --claude-model claude-opus-4-1-20250805
        """))

    p.add_argument("source", help="Fichier MP4 source ou URL YouTube")
    p.add_argument("-s", "--source-lang", default="en",
                   help="Langue(s) source (défaut: en, multilingue ex: en,he)")
    p.add_argument("-t", "--target-lang", default="fr",
                   help="Langue cible (défaut: fr)")
    p.add_argument("--context", type=str, default="",
                   help="CRITIQUE : contexte pour ancrer noms propres, sujet, "
                        "registre. Ex: --context \"Interview de Trita Parsi\"")
    p.add_argument("--style", choices=list(SUBTITLE_STYLES.keys()), default="default")
    p.add_argument("--no-dubbing", action="store_true",
                   help="Désactive la génération de l'ASS + vidéo pour doubleur")
    p.add_argument("--audit-cuts", action="store_true",
                   help="(opt-in) Passe Opus sur les sous-titres à CPS "
                        "irréductible pour tenter une reformulation plus "
                        "concise. ATTENTION : peut supprimer du contenu. "
                        "Désactivé par défaut.")
    p.add_argument("--no-trim-music", action="store_true",
                   help="Ne pas raccourcir les sous-titres qui débordent "
                        "sur un jingle musical / silence (activé par défaut)")
    p.add_argument("--skip-burn", action="store_true",
                   help="Ne pas incruster les sous-titres dans la vidéo")
    p.add_argument("--skip-summary", action="store_true",
                   help="Ne pas générer le résumé ni le kit promo")
    p.add_argument("--skip-review", action="store_true",
                   help="Passer la relecture (déconseillé en mode pro)")
    p.add_argument("--max-cps", type=int, default=MAX_CPS_PRO,
                   help=f"Plafond CPS (défaut PRO : {MAX_CPS_PRO})")
    p.add_argument("--delogo", metavar="X:Y:W:H",
                   help="Supprimer un watermark via ffmpeg delogo (X:Y:W:H)")
    p.add_argument("--whisper-model", default=traduire.WHISPER_MODEL)
    p.add_argument("--claude-model", default=CLAUDE_MODEL_PRO,
                   help=f"Modèle Claude (défaut : {CLAUDE_MODEL_PRO})")
    p.add_argument("--llm", choices=["claude", "local"], default="local",
                   help="Backend LLM : local (Ollama, défaut) ou claude (API Anthropic)")
    p.add_argument("--ollama-model", default=traduire.OLLAMA_MODEL,
                   help=f"Modèle Ollama (défaut : {traduire.OLLAMA_MODEL})")
    p.add_argument("--ollama-url", default=traduire.OLLAMA_URL,
                   help=f"URL du serveur Ollama (défaut : {traduire.OLLAMA_URL})")
    p.add_argument("--num-speakers", type=int, default=None,
                   help="Nombre de locuteurs si connu (améliore la diarisation)")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                   help="Token HuggingFace pour Pyannote")
    p.add_argument("--resume", metavar="JSON",
                   help="Reprendre depuis un checkpoint intermédiaire")
    p.add_argument("--ocr", action="store_true",
                   help="Compléter WhisperX avec OCR des sous-titres incrustés")
    args = p.parse_args()

    # ── Propagation des overrides dans le module traduire ──
    # Ces assignations sont vues par les fonctions de traduire.py via leur
    # lookup dynamique des globals — c'est le cœur du fork-par-import.
    traduire.CLAUDE_MODEL = args.claude_model
    traduire.WHISPER_MODEL = args.whisper_model
    traduire.MAX_CPS = args.max_cps

    src_lang = args.source_lang.lower().strip()
    tgt_lang = args.target_lang.lower().strip()
    source_langs = [x.strip() for x in src_lang.split(",")]
    if tgt_lang in source_langs:
        print(f"❌ Langue cible ({tgt_lang}) présente dans les sources ({src_lang})")
        sys.exit(1)
    src_file_label = "+".join(source_langs)

    # ── YouTube ──
    if is_youtube_url(args.source):
        args.source = download_youtube(args.source, output_dir=".")
    if not os.path.exists(args.source):
        print(f"❌ Introuvable : {args.source}")
        sys.exit(1)

    src = Path(args.source)
    base = src.stem
    wd = src.parent
    work = _ensure_workdir(base, wd)

    # ── Chemins ──
    audio = str(work / "audio.wav")
    segs_raw_json      = str(work / "segments.json")
    segs_gapfilled_json = str(work / "segments_gapfilled.json")
    speakers_json      = str(work / "speakers.json")
    ana_json           = str(work / "analyse.json")
    segs_corr_json     = str(work / "segments_corriges.json")
    segs_tra_json      = str(work / "segments_traduits.json")
    segs_fin_json      = str(work / "segments_finaux.json")
    promo_raw_json     = str(work / "promo_kit.json")
    meaning_loss_log   = str(work / "meaning_loss_risks.txt")

    srt_src  = str(wd / f"{base}_{src_file_label}.srt")
    srt_tgt  = str(wd / f"{base}_{tgt_lang}.srt")
    bil_txt  = str(wd / f"{base}_{src_file_label}_{tgt_lang}_bilingue.txt")
    out_mp4  = str(wd / f"{base}_{tgt_lang}.mp4")
    ass_dub  = str(wd / f"{base}_doublage.ass")
    out_dub  = str(wd / f"{base}_doublage.mp4")
    resume_txt = str(wd / f"{base}_resume.txt")
    yt_txt   = str(wd / f"{base}-description_youtube.txt")
    x_txt    = str(wd / f"{base}-description_x.txt")
    doublage_json = str(wd / f"{base}_doublage.json")
    titles_txt    = str(wd / f"{base}_titles.txt")
    titles_en_txt = str(wd / f"{base}_titles_en.txt")
    excerpts_json = str(work / "excerpts.json")
    report_txt    = str(wd / f"{base}_pro_report.txt")

    if args.llm == "local":
        client = _OllamaClient(args.ollama_url, args.ollama_model)
        print(f"   🧠 LLM local : Ollama {args.ollama_model}")
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("❌ ANTHROPIC_API_KEY manquante")
            sys.exit(1)
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

    src_desc = source_lang_description(src_lang)
    tgt_desc = lang_name(tgt_lang)
    src_label = source_lang_label(src_lang)

    print("=" * 70)
    print(f"🎬 TRADUIRE-PRO — {src_desc} → {tgt_desc}")
    print("=" * 70)
    print(f"   Source      : {args.source}")
    print(f"   Dossier work: {work}")
    print(f"   Claude      : {args.claude_model} (~5× coût Sonnet)")
    print(f"   Whisper     : {args.whisper_model}")
    print(f"   Max CPS     : {args.max_cps} (cible, tolérance +{CPS_TOLERANCE_PRO} si irréductible)")
    if args.context:
        trunc = args.context[:80]
        print(f"   Contexte    : {trunc}{'...' if len(args.context) > 80 else ''}")
    print(f"   ASS doublage: {'NON' if args.no_dubbing else 'OUI (défaut)'}")
    print(f"   Audit cuts  : {'OUI' if args.audit_cuts else 'NON'}")
    print(f"   Trim music  : {'NON' if args.no_trim_music else 'OUI (défaut)'}")
    print("=" * 70)

    has_libass = check_ffmpeg()
    if not has_libass and not args.skip_burn:
        print("   ⚠️  --skip-burn activé automatiquement (libass manquant)")
        args.skip_burn = True

    t_start = time.time()

    # ════════════════════════════════════════════════════════════════════════
    # REPRISE : déterminer le point d'entrée du pipeline
    # ════════════════════════════════════════════════════════════════════════
    resume_stage = None
    if args.resume:
        resume_stage = os.path.abspath(args.resume)
        print(f"\n🔄 Reprise depuis : {resume_stage}")

    segments: list[Segment] = []
    analysis: Optional[ContentAnalysis] = None
    speaker_profiles: dict[str, SpeakerProfile] = {}

    def _load_analysis():
        if os.path.exists(ana_json):
            return ContentAnalysis(**_load_json(ana_json))
        return None

    # ────────────────────────────────────────────────────────────────────────
    # PASSE 1 — Transcription WhisperX
    # ────────────────────────────────────────────────────────────────────────
    if resume_stage and os.path.exists(resume_stage):
        segments = load_seg(resume_stage)
        _attach_speakers_from_json(segments, speakers_json)
        print(f"   📂 {len(segments)} segments chargés depuis {resume_stage}")
    elif os.path.exists(segs_gapfilled_json):
        segments = load_seg(segs_gapfilled_json)
        print(f"   ⏩ Passe 1 — Transcription + anti-trous déjà faits ({len(segments)} seg)")
    elif os.path.exists(segs_raw_json):
        segments = load_seg(segs_raw_json)
        print(f"   ⏩ Passe 1 — Transcription déjà faite ({len(segments)} seg)")
        # Même sur resume, on tente la passe anti-trous si pas déjà faite
        if not os.path.exists(audio):
            extract_audio(args.source, audio)
        print("\n🔎 Passe 1b — Détection + récupération des trous de transcription…")
        segments = fill_transcription_gaps(segments, audio, src_lang)
        save_seg(segments, segs_gapfilled_json)
        save_src_srt(segments, srt_src)
    else:
        print("\n🎵 Passe 1 — Transcription WhisperX large-v3...")
        extract_audio(args.source, audio)
        segments = transcribe_whisperx(audio, src_lang, args.hf_token)
        if args.ocr:
            segments = ocr_supplement_segments(segments, args.source)
        save_seg(segments, segs_raw_json)
        print("\n🔎 Passe 1b — Détection + récupération des trous de transcription…")
        segments = fill_transcription_gaps(segments, audio, src_lang)
        save_seg(segments, segs_gapfilled_json)
        save_src_srt(segments, srt_src)

    duration_sec = segments[-1].end if segments else 0.0

    # ────────────────────────────────────────────────────────────────────────
    # PASSE 2 — Diarisation + profils locuteurs
    # ────────────────────────────────────────────────────────────────────────
    if os.path.exists(speakers_json):
        print("\n⏩ Passe 2 — Speakers déjà en cache")
        _attach_speakers_from_json(segments, speakers_json)
        data = _load_json(speakers_json)
        for spk in data.get("profiles", []):
            prof = SpeakerProfile(
                speaker_id=spk["id"],
                name=spk.get("name", ""),
                gender=spk.get("gender", "unknown"),
                f0_median=spk.get("f0_median", 0.0),
                total_duration=spk.get("total_duration", 0.0),
                segment_count=spk.get("segment_count", 0),
                sample_path=spk.get("sample_path", ""),
                sample_text=spk.get("sample_text", ""),
                ref_clips=spk.get("ref_clips", []),
            )
            speaker_profiles[prof.speaker_id] = prof
    else:
        # On a besoin de l'audio pour la diarisation + profils
        if not os.path.exists(audio):
            extract_audio(args.source, audio)
        diarize_speakers_pro(audio, segments, args.hf_token, args.num_speakers)
        speaker_profiles = extract_speaker_profiles(segments, audio, str(work))

        _save_json({
            "index_to_speaker": {s.index: getattr(s, "speaker", "SPEAKER_00") for s in segments},
            "profiles": [
                {
                    "id": p.speaker_id, "name": p.name,
                    "gender": p.gender, "f0_median": p.f0_median,
                    "total_duration": p.total_duration,
                    "segment_count": p.segment_count,
                    "sample_path": p.sample_path, "sample_text": p.sample_text,
                    "ref_clips": p.ref_clips,
                }
                for p in speaker_profiles.values()
            ],
        }, speakers_json)

    # ────────────────────────────────────────────────────────────────────────
    # PASSE 3 — Analyse Claude Opus (avec context)
    # ────────────────────────────────────────────────────────────────────────
    analysis = _load_analysis()
    if analysis:
        print("\n⏩ Passe 3 — Analyse déjà en cache")
    else:
        analysis = analyze_content(segments, client, src_lang, tgt_lang, args.context)
        _save_json(asdict(analysis), ana_json)

    # ────────────────────────────────────────────────────────────────────────
    # PASSE 4 — Vérification des noms propres (CRITIQUE)
    # ────────────────────────────────────────────────────────────────────────
    if os.path.exists(segs_corr_json):
        print("\n⏩ Passe 4 — Correction noms propres déjà en cache")
        segments = load_seg(segs_corr_json)
        _attach_speakers_from_json(segments, speakers_json)
    else:
        segments = verify_proper_nouns(segments, analysis, client, src_lang,
                                        args.context, args.claude_model)
        save_seg(segments, segs_corr_json)

    # ────────────────────────────────────────────────────────────────────────
    # PASSE 5 — Traduction contextuelle
    # ────────────────────────────────────────────────────────────────────────
    need_translate = not os.path.exists(segs_tra_json) and not any(s.text_tgt for s in segments)
    if os.path.exists(segs_tra_json):
        print("\n⏩ Passe 5 — Traduction déjà en cache")
        segments = load_seg(segs_tra_json)
        _attach_speakers_from_json(segments, speakers_json)
    elif need_translate or not all(s.text_tgt for s in segments):
        segments = translate_chunks(segments, analysis, client, src_lang, tgt_lang, args.context)
        save_seg(segments, segs_tra_json)

    # ────────────────────────────────────────────────────────────────────────
    # PASSE 6 — Revue qualité (3 passes)
    # ────────────────────────────────────────────────────────────────────────
    if os.path.exists(segs_fin_json):
        print("\n⏩ Passe 6 — Revue qualité déjà en cache")
        segments = load_seg(segs_fin_json)
        _attach_speakers_from_json(segments, speakers_json)
    else:
        if not args.skip_review:
            segments = review_translation(segments, analysis, client, src_lang, tgt_lang, args.context)
        segments = check_consistency(segments, analysis, client, src_lang, tgt_lang)
        segments = verify_glossary(segments, analysis, client, src_lang, tgt_lang)
        save_seg(segments, segs_fin_json)

    save_bilingual(segments, bil_txt, src_lang, tgt_lang)

    # ────────────────────────────────────────────────────────────────────────
    # PASSE 7 — Re-segmentation pro (MAX_CPS overridé via module global)
    # ────────────────────────────────────────────────────────────────────────
    print("\n✂️  Passe 7 — Re-segmentation pro (CPS ≤ "
          f"{traduire.MAX_CPS}, coupures sémantiques, anti-orphelin)...")
    subtitles = resegment(segments, tgt_lang)

    # ────────────────────────────────────────────────────────────────────────
    # PASSE 7b — Audit sémantique Opus des coupures
    # ────────────────────────────────────────────────────────────────────────
    if args.audit_cuts:
        # Seuil = MAX_CPS (24 par défaut) : on tente un rewrite sur tous les
        # sous-titres qui dépassent la cible. Opus ne conserve la reformulation
        # que si elle préserve 85 %+ du contenu. Les sous-titres entre MAX_CPS
        # et MAX_CPS+CPS_TOLERANCE (24–26) qui ne peuvent pas être raccourcis
        # fidèlement restent tels quels — c'est la tolérance explicitement
        # souhaitée : mieux vaut 26 CPS fidèle qu'un sous-titre tronqué.
        subtitles = audit_cuts_with_opus(
            subtitles, client, tgt_lang, args.context, args.claude_model,
            cps_threshold=float(traduire.MAX_CPS),
            log_path=meaning_loss_log,
        )

    # ────────────────────────────────────────────────────────────────────────
    # PASSE 7c — Trim des sous-titres débordant sur jingles musicaux / silences
    # ────────────────────────────────────────────────────────────────────────
    if not args.no_trim_music:
        subtitles = trim_music_padding(subtitles, segments=segments)

    # ────────────────────────────────────────────────────────────────────────
    # PASSE 7d — Génération SRT
    # ────────────────────────────────────────────────────────────────────────
    generate_srt(subtitles, srt_tgt)

    # ────────────────────────────────────────────────────────────────────────
    # PASSE 8 — Incrustation vidéo (sous-titres + ASS doubleur)
    # ────────────────────────────────────────────────────────────────────────
    if not args.skip_burn:
        burn_subtitles(args.source, srt_tgt, out_mp4, args.style, delogo=args.delogo)

    if not args.no_dubbing:
        vw, vh = get_video_resolution(args.source)
        generate_dubbing_ass(subtitles, ass_dub, vw, vh)
        if not args.skip_burn:
            burn_dubbing_video(args.source, ass_dub, out_dub)

    # ────────────────────────────────────────────────────────────────────────
    # PASSE 9 — Résumé structuré
    # ────────────────────────────────────────────────────────────────────────
    if not args.skip_summary:
        summary_md = generate_summary_pro(
            segments, analysis, client, duration_sec,
            args.context, args.claude_model, title=base,
        )
        with open(resume_txt, "w", encoding="utf-8") as f:
            f.write(summary_md)
        print(f"   💾 {resume_txt}")

    # ────────────────────────────────────────────────────────────────────────
    # PASSE 10 — Kit promo (titre viral + description + chapitres)
    # ────────────────────────────────────────────────────────────────────────
    if not args.skip_summary:
        kit = generate_promo_kit(
            segments, analysis, client, duration_sec,
            args.context, args.claude_model,
        )
        _save_json(kit, promo_raw_json)
        write_promo_files(kit, yt_txt, x_txt)
        print(f"   💾 {yt_txt}")
        print(f"   💾 {x_txt}")

    # ────────────────────────────────────────────────────────────────────────
    # PASSE 10b — Suggestions de titres (viral → sobre)
    # ────────────────────────────────────────────────────────────────────────
    if not args.skip_summary:
        titles_text = generate_title_suggestions(
            segments, analysis, client, args.context, args.claude_model,
        )
        with open(titles_txt, "w", encoding="utf-8") as f:
            f.write(titles_text.rstrip() + "\n")
        print(f"   💾 {titles_txt}")

        # Version anglaise (titres + description + chapitres YouTube)
        titles_en_text = generate_youtube_kit_en(
            segments, analysis, kit, client, duration_sec,
            args.context, args.claude_model,
        )
        with open(titles_en_txt, "w", encoding="utf-8") as f:
            f.write(titles_en_text.rstrip() + "\n")
        print(f"   💾 {titles_en_txt}")

    # ────────────────────────────────────────────────────────────────────────
    # PASSE 10c — Extraits viraux + posts sociaux
    # ────────────────────────────────────────────────────────────────────────
    excerpt_paths = []
    if not args.skip_summary:
        excerpts = select_viral_excerpts(
            segments, analysis, client, args.context, args.claude_model,
        )
        _save_json(excerpts, excerpts_json)
        posts = []
        for i, ex in enumerate(excerpts, 1):
            print(f"   ✍️  Post social extrait {i}/{len(excerpts)}...")
            post = generate_excerpt_social_post(
                ex, segments, analysis, client, args.claude_model,
            )
            posts.append(post)
        excerpt_paths = write_excerpt_files(excerpts, posts, str(wd / base))
        for p in excerpt_paths:
            print(f"   💾 {p}")

    # ────────────────────────────────────────────────────────────────────────
    # PASSE 11 — Export kit de doublage (JSON complet)
    # ────────────────────────────────────────────────────────────────────────
    export_doublage_kit(
        doublage_json, args.source, src_lang, tgt_lang,
        segments, analysis, args.context, speaker_profiles,
    )

    # ────────────────────────────────────────────────────────────────────────
    # PASSE 12 — Rapport final
    # ────────────────────────────────────────────────────────────────────────
    outputs = {
        "SRT source": srt_src,
        "SRT cible": srt_tgt,
        "Bilingue": bil_txt,
        "Vidéo sous-titrée": out_mp4 if not args.skip_burn else "",
        "ASS doublage": ass_dub if not args.no_dubbing else "",
        "Vidéo doublage": out_dub if (not args.no_dubbing and not args.skip_burn) else "",
        "Résumé .txt": resume_txt if not args.skip_summary else "",
        "Description YouTube": yt_txt if not args.skip_summary else "",
        "Description X": x_txt if not args.skip_summary else "",
        "Titres suggérés": titles_txt if not args.skip_summary else "",
        "Kit YouTube EN": titles_en_txt if not args.skip_summary else "",
        "Extraits viraux": ", ".join(excerpt_paths) if excerpt_paths else "",
        "Kit doublage JSON": doublage_json,
        "Dossier work": str(work),
    }

    el = time.time() - t_start
    write_final_report(report_txt, args, outputs, el, args.claude_model, args.context)

    # Cleanup audio intermédiaire
    if os.path.exists(audio):
        try:
            os.remove(audio)
        except OSError:
            pass

    print(f"\n{'=' * 70}")
    print(f"🎉 TRADUIRE-PRO terminé en {el/60:.1f} min")
    print(f"{'=' * 70}")
    for label, path_val in outputs.items():
        if path_val:
            icon = "✅" if os.path.exists(path_val) else "⏭️"
            print(f"   {icon} {label:<22s} : {path_val}")
    print(f"   📋 Rapport              : {report_txt}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
