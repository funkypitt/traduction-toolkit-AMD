#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# install-amd.sh — Installation du toolkit sur AMD Ryzen AI Max+ 395 (Strix Halo)
# ═══════════════════════════════════════════════════════════════════════════════
# Cible : iGPU Radeon 8060S (gfx1151) via ROCm/HIP, mémoire UNIFIÉE.
# Diffère de install.sh (NVIDIA) par : détection AMD, torch-ROCm, CTranslate2-ROCm
# (pour WhisperX), variables d'environnement gfx1151.
#
# ⚠️ PORTAGE À L'AVEUGLE — voir README-AMD.md. Les versions ROCm / wheels ci-dessous
#    sont des valeurs PAR DÉFAUT « supposition prudente » : ajustez ROCM_WHL et la
#    recette CTranslate2-ROCm à votre stack réel (ROCm 6.4.4 vs 7.2.x).
#
# Usage :  chmod +x install-amd.sh && ./install-amd.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

ENV_NAME="traduction-amd"
PYTHON_VERSION="3.11"
# ⚠️ AJUSTER à votre ROCm : wheels torch ROCm. ex. rocm6.2 / rocm6.3 ; rocm7.x via
#    l'index AMD (https://repo.radeon.com) si non publié sur download.pytorch.org.
ROCM_WHL="${ROCM_WHL:-https://download.pytorch.org/whl/rocm6.2}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Installation — Toolkit traduction/doublage IA — AMD ROCm (Strix Halo)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── 1. Paquets système ─────────────────────────────────────────────────────
if command -v ffmpeg &>/dev/null; then
    ok "ffmpeg présent ($(ffmpeg -version 2>&1 | head -1 | cut -d' ' -f3))"
else
    info "Installation de ffmpeg..."; sudo apt update -qq && sudo apt install -y ffmpeg libavcodec-extra; ok "ffmpeg installé"
fi

# ── 2. GPU AMD / ROCm ───────────────────────────────────────────────────────
info "Vérification du GPU AMD / ROCm..."
if command -v rocm-smi &>/dev/null || command -v amd-smi &>/dev/null; then
    (rocm-smi --showproductname 2>/dev/null || amd-smi static 2>/dev/null | head -20) | sed 's/^/      /' || true
    ok "Outils ROCm détectés."
else
    warn "Ni rocm-smi ni amd-smi trouvés — ROCm n'est probablement PAS installé."
    warn "Installez ROCm AVANT (cf. README-AMD.md §2). Sur gfx1151 : kernel ≥ 6.16.9,"
    warn "ROCm 6.4.4 ou 7.2.x, et l'override HSA_OVERRIDE_GFX_VERSION=11.5.1."
    read -p "   Continuer (torch tombera en CPU) ? (o/N) " -n 1 -r; echo ""
    [[ $REPLY =~ ^[OoYy]$ ]] || exit 0
fi

# Noyau (mémoire unifiée correctement vue ≥ 6.16.9)
KREL="$(uname -r)"; info "Noyau : $KREL (≥ 6.16.9 recommandé pour la mémoire GTT)"

# ── 3. Conda ────────────────────────────────────────────────────────────────
if command -v conda &>/dev/null; then ok "Conda présent ($(conda --version))"; else
    info "Installation de Miniconda..."; curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh
    bash /tmp/mc.sh -b -p "$HOME/miniconda3"; rm /tmp/mc.sh; eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"; "$HOME/miniconda3/bin/conda" init bash
fi
eval "$(conda shell.bash hook 2>/dev/null)" || true

# ── 4. Environnement Python ──────────────────────────────────────────────────
if conda env list 2>/dev/null | grep -q "^${ENV_NAME} "; then
    read -p "   Env '${ENV_NAME}' existe — recréer ? (o/N) " -n 1 -r; echo ""
    [[ $REPLY =~ ^[OoYy]$ ]] && { conda deactivate 2>/dev/null || true; conda env remove -n "$ENV_NAME" -y; }
fi
conda env list 2>/dev/null | grep -q "^${ENV_NAME} " || conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
conda activate "$ENV_NAME"; ok "Env activé : $(python --version)"

# ── 5. PyTorch ROCm ───────────────────────────────────────────────────────────
info "Installation de PyTorch ROCm depuis : $ROCM_WHL"
if command -v rocm-smi &>/dev/null || command -v amd-smi &>/dev/null; then
    pip install -q torch torchaudio --index-url "$ROCM_WHL" \
        || warn "Échec wheels ROCm ($ROCM_WHL) — vérifiez la version ROCm / l'index AMD (repo.radeon.com)."
else
    info "Pas de ROCm → torch CPU."; pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cpu
fi
# Vérif : sur une build ROCm, torch.version.hip est défini et cuda.is_available() True.
python -c "import torch;print(f'   torch {torch.__version__} | hip={torch.version.hip} | gpu={torch.cuda.is_available()}')" || true

# ── 6. Dépendances Python ────────────────────────────────────────────────────
info "Dépendances Python..."
pip install -q anthropic demucs pydub soundfile numpy praat-parselmouth pyworld flask yt-dlp
# WhisperX (transcription) — l'alignement (wav2vec2) tourne sous torch-ROCm.
pip install -q whisperx
ok "Dépendances de base installées"

# ── 6a. CTranslate2-ROCm pour faster-whisper (le point dur) ───────────────────
# CTranslate2 amont = CUDA/CPU uniquement. Sur Strix Halo, utiliser une FORK ROCm.
# ⚠️ AJUSTER : il existe des recettes « no-build » (ex. ROCm 7.2.2) et des wheels
#    CTranslate2-rocm communautaires (gfx900–gfx1151). Renseignez la vôtre :
echo ""
warn "Transcription GPU : CTranslate2 amont n'a PAS de ROCm."
echo "   Options (cf. README-AMD.md §3), à faire MANUELLEMENT selon votre stack :"
echo "     1) CTranslate2-ROCm (fork) — pip install la wheel ROCm ciblant gfx1151,"
echo "        puis 'pip install faster-whisper' → WhisperX accélère sur l'iGPU."
echo "     2) whisper.cpp (HIP/Vulkan) — backend alternatif (à câbler)."
echo "     3) Repli CPU : export TRADUCTION_WHISPER_COMPUTE=int8  (fonctionne tout de suite)."
read -p "   Tenter 'pip install faster-whisper' (utile si une CTranslate2-ROCm est déjà là) ? (o/N) " -n 1 -r; echo ""
[[ $REPLY =~ ^[OoYy]$ ]] && pip install -q faster-whisper || true

# ── 6b. TTS — XTTS v2 (pur torch → marche sous ROCm) ──────────────────────────
info "TTS Coqui XTTS v2 (backend par défaut conseillé sur AMD)..."
pip install -q TTS || warn "Échec TTS — réessayez ; XTTS est le backend AMD recommandé."
warn "Qwen3-TTS : nécessite flash-attn, instable sur ROCm/RDNA → préférez --tts xtts."

# ── 6c. Ollama (LLM local) — mémoire unifiée : gros modèles OK ────────────────
echo ""
read -p "   Installer Ollama (LLM local) ? (O/n) " -n 1 -r; echo ""
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    command -v ollama &>/dev/null || curl -fsSL https://ollama.com/install.sh | sh
    info "Mémoire unifiée → les gros modèles tiennent. Backends : ROCm (HIP) ou Vulkan."
    info "Si ROCm instable sur gfx1151 :  export OLLAMA_VULKAN=1  (souvent plus fiable)."
    info "Modèles conseillés (qualité, ils tiennent ici) : qwen3.6:27b, gemma4:31b."
    ollama pull qwen3.6:27b || warn "pull différé (à la demande)."
fi

# ── 7. Variables d'environnement ROCm (gfx1151) → ~/.bashrc ───────────────────
echo ""
info "Variables ROCm pour gfx1151 (ajout à ~/.bashrc si absentes)..."
add_env() { grep -q "$1=" ~/.bashrc 2>/dev/null || { echo "export $1=$2" >> ~/.bashrc; ok "  + $1=$2"; }; }
add_env HSA_OVERRIDE_GFX_VERSION 11.5.1
add_env HSA_ENABLE_SDMA 0
add_env ROCBLAS_USE_HIPBLASLT 1
warn "Rechargez le shell : source ~/.bashrc  (ou relancez la session)"

# ── 8. Vérification finale (via hw.py) ────────────────────────────────────────
echo ""; echo "═══════════════════════════════════════════════════════════════"
echo "  Vérification matérielle (hw.py)"; echo "═══════════════════════════════════════════════════════════════"
export HSA_OVERRIDE_GFX_VERSION=11.5.1 HSA_ENABLE_SDMA=0 ROCBLAS_USE_HIPBLASLT=1
python doctor.py 2>/dev/null || true
echo ""; python hw.py || true
echo ""
ok "Installation terminée. SUIVEZ la « Checklist de validation matérielle » du README-AMD.md."
echo "   Démarrage rapide :  conda activate ${ENV_NAME} && python doubler.py video.mp4 --tts xtts"
echo ""
