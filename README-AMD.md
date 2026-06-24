# traduction-toolkit-AMD

Fork du *traduction-toolkit* (à l'origine 100 % NVIDIA/CUDA) porté sur
**AMD Ryzen AI Max+ 395 « Strix Halo »** (iGPU Radeon 8060S, `gfx1151`, NPU XDNA 2,
mémoire **unifiée** LPDDR5X jusqu'à 128 Go).

> ⚠️ **Statut : portage « à l'aveugle ».** Ce fork a été écrit **sans accès au
> matériel cible**. Toute la logique non-AMD (NVIDIA/CPU) est validée sur la
> machine d'origine ; les chemins AMD/ROCm relèvent de la **supposition prudente**
> et sont à confirmer via la *Checklist de validation matérielle* (fin de doc).
> Chaque hypothèse non vérifiée est marquée « ⚠️ AMD » dans le code et ici.

---

## 0. Démarrage rapide — premier·ère testeur·euse 🧪

Tu es probablement la **première personne à lancer ce fork sur du vrai matériel AMD**
(jusqu'ici tout est validé sur NVIDIA, jamais sur un Strix Halo). Merci ! Le plus
utile en retour : la sortie de `python hw.py` et `python doctor.py`.

**Pré-requis (À FAIRE AVANT) :** ROCm doit déjà être installé (l'installeur le
*vérifie* mais ne l'installe pas). Noyau ≥ 6.16.9 ; gfx1151 marche officieusement
via `HSA_OVERRIDE_GFX_VERSION=11.5.1` (posé automatiquement). Détails : §2.

```bash
# 1) Installer
sudo apt update && sudo apt install -y git && \
git clone https://github.com/funkypitt/traduction-toolkit-AMD.git && \
cd traduction-toolkit-AMD && chmod +x install-amd.sh && ./install-amd.sh

# 2) Vérifier la détection matérielle
conda activate traduction-amd
python hw.py        # attendu : « fournisseur : amd | device : cuda | gpu=True »
python doctor.py    # diagnostic complet (dépendances + GPU)

# 3) Premier essai sur un CLIP COURT
python doubler.py video.mp4 --tts xtts          # doublage (XTTS = le plus sûr sous ROCm)
python resumer.py video.mp4 -s en --llm local   # résumé via LLM local
```

**Si la transcription tombe sur CPU** (la fork `CTranslate2-ROCm` n'est pas en
place — c'est la seule étape manuelle, cf. §3) : pour démarrer tout de suite,
forcer le repli CPU —
```bash
export TRADUCTION_WHISPER_COMPUTE=int8     # lent mais fiable (Zen 5 16 cœurs)
```

**Si quelque chose cloche :** la §8 (*Checklist de validation matérielle*) liste
exactement quoi vérifier, et où ajuster `hw.py` si `amd-smi`/`rocm-smi` renvoient
un format différent sur ta version. Reporte les écarts (idéalement avec la sortie
de `hw.py`/`doctor.py`).

---

## 1. Ce qui change par rapport au toolkit NVIDIA (et pourquoi)

| Sujet | NVIDIA (origine) | AMD Strix Halo (ce fork) | Raison |
|---|---|---|---|
| Compute GPU | CUDA | **ROCm / HIP** | pas de CUDA sur AMD |
| Device torch | `"cuda"` | **`"cuda"` aussi** | PyTorch-ROCm expose le GPU comme `cuda` (HIP masque CUDA) — la majorité du code marche tel quel |
| Transcription | WhisperX → faster-whisper → **CTranslate2 (CUDA only)** | WhisperX → faster-whisper → **CTranslate2-ROCm (fork)** | CTranslate2 amont n'a pas de backend ROCm ; une fork ROCm existe (cf. §3) → WhisperX reste, seul l'install change |
| Repli transcription | — | **whisper.cpp (HIP/Vulkan)** ou CPU int8 | si la fork CTranslate2-ROCm ne build pas |
| TTS | XTTS v2 (torch) + Qwen3-TTS (flash-attn) | **XTTS v2 par défaut** ; Qwen3-TTS = ⚠️ (flash-attn ROCm aléatoire) | XTTS est du pur torch → marche sous ROCm ; flash-attn sur RDNA est instable |
| Mémoire | VRAM **discrète** 24 Go → modèle doit « tenir » | **unifiée** 96-120 Go → 70B Q4 tiennent | l'APU partage la RAM ; la contrainte 24 Go disparaît |
| Modèle LLM local par défaut | `mistral-small` (forcé par les 24 Go) | **au choix** — `qwen3.6:27b`/`gemma4:31b` tiennent | la raison du swap mistral-small n'existe plus ici |
| Surveillance VRAM | `nvidia-smi` | **`amd-smi` / `rocm-smi`** (via `hw.py`) | pas de nvidia-smi |
| Purge VRAM avant WhisperX/TTS | bloque jusqu'à N Go libres | **assouplie** (mémoire unifiée → OOM improbable, mesure peu fiable) | cf. `hw.wait_for_vram_release` |
| NPU XDNA 2 | — | **hors périmètre** (note §7) | pile Ryzen AI/ONNX, surtout Windows ; Linux immature |

Le cœur du portage est le module **`hw.py`** : toute la dépendance matérielle y est
centralisée. Les scripts appellent `hw.device()`, `hw.whisper_compute_type()`,
`hw.free_gpu_for_task()`, etc. au lieu de coder « cuda »/« nvidia-smi » en dur.

## 2. Prérequis système (Strix Halo, Linux)

- **Noyau ≥ 6.16.9** (sinon ROCm ne voit qu'une fraction de la mémoire allouée).
  Kernel 6.19.x OK d'après les retours 2026.
- **ROCm** : la version compte énormément. Retours publics 2026 :
  - `ROCm 6.4.4` + `ROCBLAS_USE_HIPBLASLT=1` → meilleur débit long-contexte ;
  - `ROCm 7.2.x` → meilleure compat. silicium récent (recette CTranslate2-ROCm).
  ⚠️ à arbitrer sur la machine (voir checklist).
- **Mémoire GPU** : garder le carve-out UMA BIOS petit (4-8 Go) et laisser le pool
  **GTT** faire le gros (`amdgpu.gttsize` en paramètre noyau, ou laisser dynamique).
- Variables ROCm posées automatiquement par `hw.setup_rocm_env()` si absentes :
  `HSA_OVERRIDE_GFX_VERSION=11.5.1`, `HSA_ENABLE_SDMA=0`, `ROCBLAS_USE_HIPBLASLT=1`.

## 3. Installation

```bash
./install-amd.sh            # torch-ROCm, whisperx, CTranslate2-ROCm, XTTS, demucs…
python doctor.py            # vérifie la détection matérielle (section AMD)
python hw.py                # imprime fournisseur/device/compute_type/mémoire
```

Transcription — trois chemins, par ordre de préférence (cf. `install-amd.sh`) :
1. **CTranslate2-ROCm** (fork) + faster-whisper + WhisperX → le plus rapide,
   code inchangé. Recette « no-build » connue pour Strix Halo (ROCm 7.2.2).
2. **whisper.cpp** (HIP ou Vulkan) → robuste, ~4–5× temps réel sur large-v3 ;
   backend optionnel (`TRADUCTION_WHISPER_BACKEND=whispercpp`, à implémenter si
   la voie 1 échoue).
3. **CPU int8** (CTranslate2 amont) → repli universel, lent mais fiable
   (`TRADUCTION_WHISPER_COMPUTE=int8`, le Zen 5 16-cœurs encaisse).

## 4. Modèle LLM local (Ollama)

La mémoire unifiée (96-120 Go) **supprime** la contrainte des 24 Go qui avait
imposé `mistral-small` côté NVIDIA. Sur Strix Halo, `qwen3.6:27b` ou `gemma4:31b`
(voire 70B Q4) tiennent. Choisir via `--ollama-model` ; défaut laissé à
`mistral-small` (rapide, sûr) mais **vous pouvez remonter en qualité**.

Backend Ollama : **ROCm (HIP)** ou **Vulkan** (`OLLAMA_VULKAN=1`, réputé le plus
fiable sous Linux sur cette APU). À tester (checklist).

## 5. Variables d'environnement utiles

| Variable | Effet |
|---|---|
| `HSA_OVERRIDE_GFX_VERSION=11.5.1` | fait reconnaître gfx1151 par ROCm |
| `HSA_ENABLE_SDMA=0` | corrige des segfauts ROCm (kernels récents) |
| `ROCBLAS_USE_HIPBLASLT=1` | meilleur chemin long-contexte |
| `TRADUCTION_WHISPER_COMPUTE=int8|float16` | force le compute_type WhisperX |
| `TRADUCTION_UNIFIED_MEM=0|1` | force/désactive la logique mémoire unifiée |
| `OLLAMA_VULKAN=1` | backend Vulkan pour Ollama |

## 6. État du portage par script

- **Module `hw.py`** : ✅ écrit, branche NVIDIA/CPU **validée sur matériel réel** ;
  branches AMD (`amd-smi`/`rocm-smi`, env ROCm) ⚠️ à valider.
- **doubler.py / doubler-mp3-batch.py / resumer.py** : helpers VRAM `nvidia-smi`
  remplacés par `hw.*` ; device/compute_type WhisperX via `hw.*`.
- **traduire.py / traduire-pro.py / clipper.py / transcrire.py /
  sous_titrer_docx.py** : device/compute_type WhisperX via `hw.*`.
- **doctor.py** : section diagnostic AMD (via `hw.describe()`).
- **Tout script additionnel** suivant le même motif `device="cuda"` /
  `compute_type` se porte en une ligne : `import hw` puis `hw.device()` /
  `hw.whisper_compute_type()`.

## 7. NPU XDNA 2 — hors périmètre (pour l'instant)

Le NPU (~50 TOPS) n'est pas exploité : son accès passe par Ryzen AI SW + ONNX
Runtime (surtout Windows ; Linux immature en 2026) et aucun des frameworks du
toolkit (torch/WhisperX/Coqui) ne le cible nativement. L'accélération pratique
ici = **iGPU via ROCm**. Une voie « Whisper sur NPU » existe (ONNX) mais
nécessiterait un backend de transcription dédié — piste future.

## 8. ✅ Checklist de validation matérielle (à faire sur la machine cible)

1. `python hw.py` → `fournisseur : amd`, `gpu=True`, `device : cuda`. Sinon, la
   build torch n'est pas ROCm → réinstaller (`install-amd.sh`).
2. `python -c "import torch;print(torch.version.hip, torch.cuda.get_device_name(0))"`
   → doit nommer le Radeon 8060S.
3. **Mémoire** : `rocm-smi --showmeminfo vram` / `amd-smi metric -m`. Vérifier que
   `hw.gpu_memory_free_mib()` renvoie une valeur cohérente (sinon ajuster le
   parsing dans `hw.py` au format réel de votre version d'outil).
4. **Transcription** : lancer `transcrire.py` sur un clip court. Confirmer GPU
   (et non CPU) ; sinon CTranslate2-ROCm absent → voie 2/3 du §3.
5. **TTS** : `doubler.py --tts xtts` sur un clip court → vérifier le chargement
   XTTS sur GPU sans OOM.
6. **LLM** : `resumer.py … --llm local --ollama-model qwen3.6:27b` → vérifier
   qu'un gros modèle tient (mémoire unifiée) et le débit (ROCm vs `OLLAMA_VULKAN=1`).
7. **Pipeline complet** : un `doubler.py` de bout en bout sur une vidéo courte.

> Reporter les écarts (formats `amd-smi`/`rocm-smi`, env ROCm, version optimale)
> directement dans `hw.py` (`ROCM_ENV_DEFAULTS`, parsing mémoire) et ce README.
