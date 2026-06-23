#!/usr/bin/env python3
"""
Bridge subprocess pour Qwen3-TTS (voice cloning, qualité maximale).

Conçu pour tourner dans un conda env dédié (qwen3tts) qui a ses propres
versions de torch/transformers incompatibles avec l'env principal (interview).

Protocole : JSON-lines sur stdin/stdout.
  → {"cmd": "init"}
  ← {"ok": true, "sample_rate": 24000, "device": "cuda"}

  → {"cmd": "generate", "text": "...", "language": "French",
     "ref_audio_path": "/tmp/ref.wav", "ref_text": "transcript of ref",
     "output_path": "/tmp/out.wav"}
  ← {"ok": true, "duration": 1.23}

  → {"cmd": "quit"}
  ← {"ok": true}

Le modèle Base (1.7B) est utilisé pour le clonage vocal zero-shot.
Qwen3-TTS exige le transcript de l'audio de référence (ref_text) pour
le mode ICL (In-Context Learning), qui capture timbre ET prosodie.
Sans ref_text, bascule en mode x_vector_only (timbre seul, ~0.75 vs ~0.89
de similarité vocale).

Paramètres de génération optimisés pour la qualité maximale :
  - temperature=0.9, top_k=50, repetition_penalty=1.05 (défauts officiels)
  - subtalker_temperature=0.9, subtalker_top_k=50 (codec audio)
  - max_new_tokens=4096 (marge confortable pour textes longs)
  - do_sample=True (obligatoire pour une parole naturelle)
  - bfloat16 sur Ampere+ (RTX 30xx/40xx) — précision officielle des benchmarks
  - FlashAttention 2 si disponible (~30-40% plus rapide, ~20-25% VRAM en moins)

Référence audio idéale : 10-15 secondes de parole claire, mono, ≥ 24 kHz.
Au-delà de 30s, risque de boucle infinie (ref_audio_max_seconds=30).

VRAM : ~7-8 Go pour le 1.7B en bfloat16. RTX 3090 (24 Go) = très à l'aise.

Installation :
  conda create -n qwen3tts python=3.12
  conda activate qwen3tts
  pip install -U qwen-tts soundfile
  pip install -U flash-attn --no-build-isolation  # recommandé (~8 min)
"""

import json
import sys
import os
import gc

# ── Protéger stdout : rediriger tout print() vers stderr ──
# Les bibliothèques (huggingface_hub, tqdm, torch) peuvent écrire sur stdout
# lors du chargement du modèle, ce qui corromprait le canal JSON.
_json_out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
sys.stdout = sys.stderr


def _respond(obj):
    """Écrit une réponse JSON sur le canal dédié (vrai stdout) et flush."""
    _json_out.write(json.dumps(obj) + "\n")
    _json_out.flush()


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMÈTRES DE GÉNÉRATION — qualité maximale
# ═══════════════════════════════════════════════════════════════════════════════
# Source : generation_config.json officiel + qwen3-TTS-studio "quality" preset
# Ces valeurs sont celles des benchmarks Qwen (WER 1.835%, similarité 0.789).
# On ne cherche PAS la vitesse, on cherche le meilleur rendu vocal possible.

GENERATE_KWARGS = dict(
    max_new_tokens=4096,            # 2048 suffit souvent, 4096 = marge confortable
    do_sample=True,                 # obligatoire — greedy tue la prosodie
    temperature=0.9,                # défaut officiel, naturel et expressif
    top_k=50,                       # défaut officiel
    top_p=1.0,                      # 1.0 = désactivé (laisse top_k filtrer)
    repetition_penalty=1.05,        # défaut officiel, évite les boucles
    subtalker_dosample=True,        # sampling aussi pour le décodeur codec
    subtalker_temperature=0.9,      # défaut officiel
    subtalker_top_k=50,             # défaut officiel
    subtalker_top_p=1.0,            # défaut officiel
)


def main():
    import torch
    import soundfile as sf

    model = None
    sample_rate = 24000
    # Cache du prompt de clonage vocal : {cache_key: prompt_items}
    # Clé = (ref_audio_path, ref_text) pour invalider si le texte change
    _voice_cache = {}

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            cmd = json.loads(line)
        except json.JSONDecodeError as e:
            _respond({"ok": False, "error": f"JSON invalide : {e}"})
            continue

        action = cmd.get("cmd")

        if action == "init":
            try:
                from qwen_tts import Qwen3TTSModel

                device = "cuda" if torch.cuda.is_available() else "cpu"
                # bfloat16 = précision officielle des benchmarks Qwen.
                # Même plage d'exposant que float32 → pas d'overflow.
                # float32 possible mais 2× la VRAM sans gain audible.
                dtype = torch.bfloat16 if device == "cuda" else torch.float32

                # FlashAttention 2 : ~30-40% plus rapide, ~20-25% VRAM en moins.
                # Nécessite Ampere+ (RTX 30xx/40xx). Sinon SDPA (PyTorch natif).
                attn_impl = "sdpa"
                try:
                    import flash_attn  # noqa: F401
                    attn_impl = "flash_attention_2"
                except ImportError:
                    pass

                print(f"Chargement Qwen3-TTS-12Hz-1.7B-Base "
                      f"(attn={attn_impl}, dtype={dtype})...", file=sys.stderr)

                model = Qwen3TTSModel.from_pretrained(
                    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                    device_map=f"{device}:0" if device == "cuda" else device,
                    dtype=dtype,
                    attn_implementation=attn_impl,
                )

                _respond({"ok": True, "sample_rate": sample_rate, "device": device,
                          "attn": attn_impl, "dtype": str(dtype)})
            except Exception as e:
                _respond({"ok": False, "error": str(e)})

        elif action == "generate":
            if model is None:
                _respond({"ok": False, "error": "Modèle non initialisé (appeler init d'abord)"})
                continue

            try:
                text = cmd["text"]
                language = cmd.get("language", "French")
                ref_audio = cmd.get("ref_audio_path")
                ref_text = cmd.get("ref_text", "")
                output_path = cmd["output_path"]

                if ref_audio and os.path.exists(ref_audio):
                    # ── Clonage vocal ──
                    # Clé de cache = (chemin, texte) pour invalider si on
                    # change de ref_text pour le même fichier audio
                    cache_key = (ref_audio, ref_text)

                    if cache_key not in _voice_cache:
                        # Créer le prompt de clonage et le cacher.
                        # Le prompt encode l'empreinte vocale une seule fois,
                        # réutilisé pour tous les segments du même locuteur.
                        clone_kwargs = dict(ref_audio=ref_audio)

                        if ref_text:
                            # Mode ICL complet : capture timbre + prosodie
                            # Similarité vocale ~0.89 (vs ~0.75 en x_vector_only)
                            clone_kwargs["ref_text"] = ref_text
                            clone_kwargs["x_vector_only_mode"] = False
                            mode = "ICL"
                        else:
                            # Fallback : timbre seul (pas de prosodie)
                            clone_kwargs["x_vector_only_mode"] = True
                            mode = "x_vector_only"

                        print(f"  Création prompt clonage [{mode}] : "
                              f"{os.path.basename(ref_audio)}", file=sys.stderr)

                        prompt_items = model.create_voice_clone_prompt(**clone_kwargs)
                        _voice_cache[cache_key] = prompt_items

                    prompt_items = _voice_cache[cache_key]

                    wavs, sr = model.generate_voice_clone(
                        text=text,
                        language=language,
                        voice_clone_prompt=prompt_items,
                        **GENERATE_KWARGS,
                    )
                else:
                    # ── Sans référence : voix par défaut du modèle ──
                    wavs, sr = model.generate_voice_clone(
                        text=text,
                        language=language,
                        **GENERATE_KWARGS,
                    )

                # wavs est une liste de numpy arrays, sr = 24000
                wav_data = wavs[0]
                sf.write(output_path, wav_data, sr)

                duration = len(wav_data) / sr
                _respond({"ok": True, "duration": round(duration, 3),
                          "sample_rate": sr})

            except Exception as e:
                _respond({"ok": False, "error": str(e)})

        elif action == "quit":
            if model is not None:
                del model
                _voice_cache.clear()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            _respond({"ok": True})
            break

        else:
            _respond({"ok": False, "error": f"Commande inconnue : {action}"})


if __name__ == "__main__":
    main()
