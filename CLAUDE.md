# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## AMD / ROCm fork

This is the **AMD Ryzen AI Max+ 395 "Strix Halo" (gfx1151, ROCm)** fork of the toolkit. All hardware coupling lives in **`hw.py`** (vendor detection CUDA/ROCm/CPU; `hw.device()` returns `"cuda"` for both CUDA and ROCm since PyTorch-HIP masquerades as cuda; `hw.whisper_compute_type()`; multi-vendor GPU-memory query; unified-memory-aware VRAM logic; `hw.setup_rocm_env()`). Scripts must NOT hardcode `"cuda"`/`nvidia-smi` — call `hw.*`. Install via `install-amd.sh`; AMD specifics and the on-hardware validation checklist are in `README-AMD.md`. ROCm paths are untested on the target hardware (cautious guesswork) — assumptions flagged "⚠️ AMD".

## Project Overview

Multilingual AI translation toolkit for video/audio content. Four independent Python scripts, each implementing a complete pipeline — no shared library, no build system.

- **`traduire.py`** — Video subtitle translation (6 passes: WhisperX → Claude analysis → Claude translation → Claude review → re-segmentation → ffmpeg burn)
- **`doubler-mp3-batch.py`** — Batch audio dubbing to MP3 (7 passes: WhisperX → Pyannote → Demucs → Claude translation → TTS synthesis → normalization → assembly). No timing constraints, output can be longer than source. TTS backends: Qwen3-TTS (default), XTTS v2.
- **`doubler.py`** — Video dubbing with voice-over mixing (11 passes: adds isochronic adaptation, two-pass TTS with speed adjustment, voice-over mixing with ducking). TTS backends: Qwen3-TTS (default), XTTS v2, ElevenLabs.
- **`clipper.py`** — Viral clip extraction (6 passes: WhisperX → Claude clip selection → optional Claude translation → ffmpeg cut → ASS karaoke subtitles → ffmpeg burn). Selects best passages via `--criteria`, outputs Instagram-style karaoke subtitles (word-by-word groups on black background).
- **`nettoyer.py`** — Light-touch audio restoration for dhamma talks (7 passes: analysis/hum detection → 48 kHz conditioning + 60 Hz high-pass → denoise → optional gentle leveling → linear BS.1770 loudness normalization (no limiter unless bell peaks force it) → MP3 LAME V0 → DNSMOS before/after QC + JSON report). Denoise engines via `--moteur`: `auto` (default — DeepFilterNet3 first, DNSMOS-arbitrated fallback to MossFormer2 when a voice reacts badly to DFN), `dfn`, `mossformer2` (ClearerVoice in dedicated `clearvoice` conda env via `mossformer_bridge.py`; caveat: compresses dynamics ~8 dB), `afftdn` (gentlest, modest cleaning). Hybrid iZotope RX workflow via `--exporter-rx` / `--importer-rx`. Best engine is per-recording — trust ears over metrics.

## Running the Scripts

```bash
# Subtitles
python traduire.py video.mp4                        # EN → FR (default)
python traduire.py video.mp4 -s ja -t en             # JA → EN
python traduire.py video.mp4 --style netflix
python traduire.py video.mp4 --resume segments.json  # resume from checkpoint

# Audio dubbing (batch — processes all MP3+MP4 in current directory)
python doubler-mp3-batch.py
python doubler-mp3-batch.py --file specific.mp3

# Video dubbing with voice-over
python doubler.py video.mp4
python doubler.py video.mp4 --no-voiceover  # pure dubbing
python doubler.py video.mp4 --tts qwen3tts  # Qwen3-TTS (FR excellent)
python doubler.py video.mp4 --tts xtts     # XTTS v2 fallback
python doubler-mp3-batch.py --model xtts                 # XTTS v2 fallback

# Audio restoration (dhamma talks — batch or single file)
python nettoyer.py causerie.mp3                  # denoise + normalize → causerie_nettoye.mp3
python nettoyer.py dossier/ -o propre/           # batch
python nettoyer.py causerie.mp3 --reduction 10   # lighter denoise (more room tone kept)
python nettoyer.py causerie.mp3 --exporter-rx    # hybrid iZotope RX workflow

# Viral clip extraction
python clipper.py video.mp4 --criteria "passage le plus marquant"
python clipper.py video.mp4 --criteria "moment drôle" --duration 180-600 -n 2
python clipper.py video.mp4 --criteria "key insights" --target-lang fr
python clipper.py video.mp4 --resume video_clips.json --criteria "test"
```

## Environment Requirements

- **`ANTHROPIC_API_KEY`** (optional — scripts default to a local LLM via Ollama; only needed with `--llm claude`). With `--analysis-llm auto` (the default), the **analysis pass** alone (glossary/proper-nouns/domain) uses Claude when this key is set — one cheap call that also improves the local translation; falls back to local otherwise.
- **`HF_TOKEN`** (required for dubbing scripts — Pyannote speaker diarization)
- **`ffmpeg`** system binary
- GPU with CUDA recommended (WhisperX, XTTS v2, Qwen3-TTS)

```bash
# Main env (interview) — all deps
pip install whisperx anthropic torch torchaudio demucs pydub soundfile \
            numpy praat-parselmouth pyworld TTS flask --break-system-packages

# Qwen3-TTS (default dubbing backend) runs in its own conda env
conda create -n qwen3tts python=3.12
conda run -n qwen3tts pip install -U qwen-tts soundfile
conda run -n qwen3tts pip install -U flash-attn --no-build-isolation  # recommended
```

### TTS Bridge Isolation

Each TTS backend with incompatible dependencies runs in a dedicated conda env, communicating via a **bridge subprocess** (JSON-lines over stdin/stdout). Bridges protect stdout from library spam by redirecting it to stderr.

| Backend | Conda env | Bridge | Notes |
|---------|-----------|--------|-------|
| Qwen3-TTS | `qwen3tts` | `qwen3tts_bridge.py` | 1.7B Base, 10 langs, voice cloning with ref_text |

(XTTS v2 runs in-process in the main env; ElevenLabs is API-based — neither needs a bridge.)

Bridges are spawned automatically when `--tts <backend>` is used; no manual activation needed.

## Architecture Notes

### Claude's Three Roles in Every Pipeline
1. **Analyst** — content summary, glossary, tone/domain detection (`ContentAnalysis`)
2. **Translator** — contextual translation using overlapping chunk windows (60 segments, 8 overlap)
3. **Reviewer** — quality check for naturalness, coherence, contresens

### Key Constants (top of each script)
- `CLAUDE_MODEL = "claude-sonnet-4-20250514"` — the model used for all Claude calls
- `WHISPER_MODEL = "large-v3"` — transcription model
- Subtitle constraints in `traduire.py`: `MAX_CHARS_PER_LINE = 42`, `MAX_CPS = 17`
- TTS speed range in dubbing: `XTTS_SPEED_MIN = 0.82`, `XTTS_SPEED_MAX = 1.30` (XTTS), `QWEN3TTS_SPEED_MIN = 0.70`, `QWEN3TTS_SPEED_MAX = 1.50` (Qwen3-TTS via atempo)

### Core Data Structures (dataclasses)
- **`Segment`** — atomic speech unit with timing, source text, translated text, speaker label, word-level alignment
- **`ContentAnalysis`** — summary, glossary, speakers_description, tone, domain
- **`SpeakerProfile`** (dubbing scripts) — gender, F0 median, reference clips for voice cloning
- **`ClipSelection`** (clipper.py) — clip_index, seg_start/end, start/end times, titre, justification, segments list

### TTS Bridges (`*_bridge.py`)
- Run under `~/miniconda3/envs/<backend>/bin/python`
- Protect stdout (JSON channel) by redirecting all library prints to stderr
- Commands: `init` (load model), `generate` (text → WAV at 24kHz), `quit`
- Model stays loaded for the entire dubbing session (one subprocess per run)
- Text chunking, concatenation, resampling (44.1kHz), and atempo speed adjustment remain in the main scripts
- **Qwen3-TTS specifics**: caches voice clone prompts per ref_audio; uses `x_vector_only_mode` when `ref_text` (transcript of reference) is not provided

### Resumption
All scripts save intermediate JSON files (segments, analysis) and can resume from checkpoints. Outputs are placed next to the input file with language suffix (e.g., `video_fr.mp4`, `video_fr.srt`).

### Language-Aware Processing
- Language-specific line-breaking rules (orphan word prevention for articles/prepositions)
- XTTS character limits vary by language
- Politeness conventions (tu/vous, du/Sie) handled in translation prompts

## Code Style

- Written entirely in French (comments, docstrings, variable names, log messages)
- No type hints beyond dataclass fields and `Optional`
- No tests, no linter config — scripts validated through built-in `check_dependencies()` / `check_ffmpeg()`
- 4-space indentation, standard Python conventions
