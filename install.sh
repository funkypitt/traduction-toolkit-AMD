#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# install.sh — Installation automatique du toolkit de traduction/doublage IA
# ═══════════════════════════════════════════════════════════════════════════════
# Testé sur Ubuntu 22.04 / 24.04 avec GPU NVIDIA.
# Installe : Miniconda (si absent), environnement Python 3.11, toutes les
# dépendances (torch+CUDA, whisperx, XTTS v2, demucs, etc.), ffmpeg, yt-dlp.
#
# Usage :
#   chmod +x install.sh
#   ./install.sh
#
# Après l'installation :
#   conda activate traduction
#   export ANTHROPIC_API_KEY="sk-ant-api03-..."
#   export HF_TOKEN="hf_..."
#   python traduire.py video.mp4
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

ENV_NAME="traduction"
PYTHON_VERSION="3.11"

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Installation — Toolkit de traduction et doublage IA"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── 1. Paquets système (ffmpeg) ───────────────────────────────────────────────

info "Vérification des paquets système..."

if command -v ffmpeg &>/dev/null; then
    ok "ffmpeg déjà installé ($(ffmpeg -version 2>&1 | head -1 | cut -d' ' -f3))"
else
    info "Installation de ffmpeg..."
    sudo apt update -qq
    sudo apt install -y ffmpeg libavcodec-extra
    ok "ffmpeg installé"
fi

# ── 2. GPU NVIDIA ─────────────────────────────────────────────────────────────

info "Vérification du GPU NVIDIA..."

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    ok "GPU détecté : ${GPU_NAME} (driver ${DRIVER_VER})"
else
    warn "nvidia-smi introuvable — pas de GPU NVIDIA détecté."
    warn "Les scripts fonctionneront sur CPU mais seront BEAUCOUP plus lents."
    warn "Pour installer les drivers NVIDIA : https://developer.nvidia.com/cuda-downloads"
    echo ""
    read -p "   Continuer sans GPU ? (o/N) " -n 1 -r
    echo ""
    [[ $REPLY =~ ^[OoYy]$ ]] || exit 0
fi

# ── 3. Conda ──────────────────────────────────────────────────────────────────

info "Vérification de Conda..."

if command -v conda &>/dev/null; then
    ok "Conda déjà installé ($(conda --version 2>&1))"
else
    info "Conda non trouvé — installation de Miniconda..."
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    MINICONDA_SH="/tmp/miniconda_install.sh"
    curl -fsSL "$MINICONDA_URL" -o "$MINICONDA_SH"
    bash "$MINICONDA_SH" -b -p "$HOME/miniconda3"
    rm "$MINICONDA_SH"

    # Initialiser conda dans le shell courant
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    "$HOME/miniconda3/bin/conda" init bash

    ok "Miniconda installé dans $HOME/miniconda3"
    warn "Rechargez votre shell après l'installation : source ~/.bashrc"
fi

# S'assurer que conda est disponible dans ce script
eval "$(conda shell.bash hook 2>/dev/null)" || true

# ── 4. Environnement Python ──────────────────────────────────────────────────

info "Configuration de l'environnement Python..."

if conda env list 2>/dev/null | grep -q "^${ENV_NAME} "; then
    warn "L'environnement '${ENV_NAME}' existe déjà."
    read -p "   Le recréer de zéro ? (o/N) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[OoYy]$ ]]; then
        conda deactivate 2>/dev/null || true
        conda env remove -n "$ENV_NAME" -y
        info "Environnement supprimé, recréation..."
    else
        info "Conservation de l'environnement existant, mise à jour des paquets..."
    fi
fi

if ! conda env list 2>/dev/null | grep -q "^${ENV_NAME} "; then
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    ok "Environnement '${ENV_NAME}' créé (Python ${PYTHON_VERSION})"
fi

conda activate "$ENV_NAME"
ok "Environnement activé : $(python --version)"

# ── 5. PyTorch + CUDA ─────────────────────────────────────────────────────────

info "Installation de PyTorch..."

# Détecter la version CUDA disponible
if command -v nvidia-smi &>/dev/null; then
    CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    info "Installation de torch avec support CUDA (cu121)..."
    pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    info "Installation de torch (CPU uniquement)..."
    pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Vérifier
python -c "import torch; print(f'   torch {torch.__version__}, CUDA={torch.cuda.is_available()}')"
ok "PyTorch installé"

# ── 6. Dépendances Python ────────────────────────────────────────────────────

info "Installation des dépendances Python..."

# whisperx (transcription)
info "  whisperx..."
pip install -q whisperx

# anthropic (Claude API)
info "  anthropic..."
pip install -q anthropic

# Doublage : séparation + audio + TTS
info "  demucs, pydub, soundfile, numpy..."
pip install -q demucs pydub soundfile numpy

# Analyse prosodique
info "  praat-parselmouth, pyworld..."
pip install -q praat-parselmouth pyworld

# XTTS v2 (synthèse vocale — backend de doublage dans l'env principal)
info "  TTS (Coqui XTTS v2) — peut prendre quelques minutes..."
pip install -q TTS

# GUI (panneau de contrôle web)
info "  flask (GUI)..."
pip install -q flask

# YouTube
info "  yt-dlp..."
pip install -q yt-dlp

ok "Toutes les dépendances Python installées"

# ── 6b. Backend TTS du doublage (env conda isolé) ────────────────────────────
# Backend de doublage par défaut : Qwen3-TTS, dans son PROPRE env conda (bridge).
# Alternative : XTTS v2, déjà inclus dans l'env principal (--tts xtts).
echo ""
read -p "   Installer le backend TTS par défaut Qwen3-TTS (env conda dédié) ? (O/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    if conda env list 2>/dev/null | grep -q "^qwen3tts "; then
        ok "env 'qwen3tts' déjà présent"
    else
        info "Création de l'env 'qwen3tts'..."
        conda create -n qwen3tts python=3.12 -y -q
        conda run -n qwen3tts pip install -q -U qwen-tts soundfile
        ok "Backend Qwen3-TTS installé (env 'qwen3tts')"
    fi
else
    warn "Qwen3-TTS ignoré — le doublage utilisera XTTS v2 (--tts xtts), inclus dans l'env principal."
fi

# ── 6c. LLM local (Ollama) — alternative gratuite à l'API Claude ──────────────
echo ""
read -p "   Installer Ollama + modèles locaux (résumé/traduction 100% local) ? (O/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    if command -v ollama &>/dev/null; then
        ok "Ollama déjà installé ($(ollama --version 2>&1 | head -1))"
    else
        info "Installation d'Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
        ok "Ollama installé"
    fi
    info "Téléchargement des modèles recommandés (gemma4:31b traduction, qwen3.6:27b résumé)..."
    info "  (~36 Go — peut être long ; Ctrl-C pour passer, les modèles se pullent à la demande)"
    ollama pull gemma4:31b   || warn "pull gemma4:31b interrompu (réessayez plus tard)"
    ollama pull qwen3.6:27b  || warn "pull qwen3.6:27b interrompu (réessayez plus tard)"
else
    warn "Ollama ignoré — les scripts utiliseront l'API Claude (--llm claude, défaut)."
fi

# ── 6d. Application GUI (.deb) — panneau de contrôle dans le menu ─────────────
echo ""
read -p "   Installer l'application GUI (.deb — panneau de contrôle dans le menu) ? (O/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [ -f "$REPO_DIR/packaging/build-deb.sh" ]; then
        info "Construction du paquet .deb (lanceur pointant vers $REPO_DIR)..."
        TRADUCTION_DIR="$REPO_DIR" bash "$REPO_DIR/packaging/build-deb.sh" >/dev/null 2>&1 || true
        DEB="$(ls -t "$REPO_DIR"/dist/traduction-gui_*.deb 2>/dev/null | head -1)"
        if [ -n "$DEB" ]; then
            info "Installation du paquet (sudo requis)..."
            sudo apt install -y "$DEB" 2>/dev/null || sudo dpkg -i "$DEB" || true
            ok "GUI installée — cherchez « Traduction » dans le menu (ou : traduction-gui)"
        else
            warn "Build du .deb échoué — à refaire plus tard : ./packaging/build-deb.sh"
        fi
    else
        warn "packaging/build-deb.sh introuvable — étape ignorée."
    fi
else
    info "GUI utilisable sans .deb :  python gui.py  → http://127.0.0.1:5005"
fi

# ── 6e. Clés API (optionnel — configuration interactive) ──────────────────────
echo ""
echo "   ── Clés API ──────────────────────────────────────────────────────────"
echo -e "   • ${BLUE}ANTHROPIC_API_KEY${NC} (Claude) — ${YELLOW}OPTIONNELLE${NC} : tout marche en local sans elle."
echo "       Si configurée, la passe d'analyse (glossaire/contexte) passe par Claude"
echo "       → meilleur contexte ET meilleure traduction (1 appel, quelques centimes)."
echo "       Obtenir une clé :  https://console.anthropic.com/  → API Keys → Create Key"
echo -e "   • ${BLUE}HF_TOKEN${NC} (HuggingFace) — ${YELLOW}REQUISE pour la diarisation${NC} (doublage)."
echo "       https://huggingface.co/settings/tokens  (accès read), puis ACCEPTER :"
echo "       huggingface.co/pyannote/speaker-diarization-3.1  et  .../segmentation-3.0"
echo ""
read -p "   Configurer ces clés maintenant (écrites dans ~/.bashrc) ? (o/N) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[OoYy]$ ]]; then
    read -r -p "      ANTHROPIC_API_KEY (laisser vide pour ignorer) : " ANT_KEY || true
    if [ -n "${ANT_KEY:-}" ]; then
        echo "export ANTHROPIC_API_KEY=\"$ANT_KEY\"" >> ~/.bashrc
        export ANTHROPIC_API_KEY="$ANT_KEY"
        ok "ANTHROPIC_API_KEY ajoutée à ~/.bashrc"
    fi
    read -r -p "      HF_TOKEN (laisser vide pour ignorer) : " HF_KEY || true
    if [ -n "${HF_KEY:-}" ]; then
        echo "export HF_TOKEN=\"$HF_KEY\"" >> ~/.bashrc
        export HF_TOKEN="$HF_KEY"
        ok "HF_TOKEN ajouté à ~/.bashrc"
    fi
    info "Pensez à recharger votre shell :  source ~/.bashrc"
else
    info "Vous pourrez les définir plus tard (voir le rappel ci-dessous)."
fi

# ── 7. Vérification finale ───────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Vérification finale"
echo "═══════════════════════════════════════════════════════════════"
echo ""

ALL_OK=true

check_pkg() {
    if python -c "import $1" 2>/dev/null; then
        ok "$2"
    else
        warn "$2 — échec de l'import"
        ALL_OK=false
    fi
}

check_pkg "whisperx"     "whisperx      (transcription)"
check_pkg "anthropic"    "anthropic     (Claude API)"
check_pkg "torch"        "torch         (PyTorch)"
check_pkg "torchaudio"   "torchaudio    (audio PyTorch)"
check_pkg "demucs"       "demucs        (séparation de sources)"
check_pkg "pydub"        "pydub         (mixage audio)"
check_pkg "soundfile"    "soundfile     (I/O audio)"
check_pkg "numpy"        "numpy         (calcul numérique)"
check_pkg "parselmouth"  "parselmouth   (analyse acoustique)"
check_pkg "pyworld"      "pyworld       (vocoder WORLD)"
check_pkg "TTS"          "TTS           (XTTS v2)"

if command -v ffmpeg &>/dev/null; then
    ok "ffmpeg        (traitement vidéo)"
else
    warn "ffmpeg manquant"
    ALL_OK=false
fi

if command -v yt-dlp &>/dev/null; then
    ok "yt-dlp        (téléchargement YouTube)"
else
    warn "yt-dlp manquant"
    ALL_OK=false
fi

# GPU
echo ""
python -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f'   GPU : {name} ({mem:.0f} Go VRAM)')
else:
    print('   GPU : aucun (mode CPU)')
"

echo ""
echo "═══════════════════════════════════════════════════════════════"

if $ALL_OK; then
    ok "Installation terminée avec succès !"
else
    warn "Installation terminée avec des avertissements (voir ci-dessus)"
fi

echo ""
echo "   Prochaines étapes :"
echo ""
echo "   1. Configurez vos clés API (toutes deux OPTIONNELLES) :"
echo "      export ANTHROPIC_API_KEY=\"sk-ant-api03-...\"   # facultatif (voir ci-dessous)"
echo "      export HF_TOKEN=\"hf_...\"                       # requis pour la diarisation (doublage)"
echo ""
echo "      → Sans clé Anthropic, tout tourne en LOCAL (Ollama). Si la clé est"
echo "        présente, la passe d'ANALYSE (glossaire/contexte) passe par Claude"
echo "        (1 appel, qqs centimes) → meilleur contexte ET meilleure traduction,"
echo "        la traduction restant locale. Forçable : --analysis-llm claude|local."
echo ""
echo "   2. Activez l'environnement :"
echo "      conda activate ${ENV_NAME}"
echo ""
echo "   3. Lancez un script :"
echo "      python traduire.py video.mp4"
echo "      python doubler.py video.mp4"
echo ""
echo -e "   ${YELLOW}4. IMPORTANT pour la QUALITÉ du doublage français${NC} — ajoutez des"
echo "      voix de référence dans le dossier  ./voix/  :"
echo "        • nommage par genre : homme1.wav, homme2.wav, … / femme1.wav, femme2.wav, …"
echo "        • 15-30 s de parole française CLAIRE, un seul locuteur, sans bruit ni musique"
echo "        • utilisées AUTOMATIQUEMENT par genre (aucun argument requis)"
echo "        • sans elles, la voix d'origine (accent étranger) est clonée → moins naturel"
echo ""
echo "   Voir README.md pour plus de détails."
echo "═══════════════════════════════════════════════════════════════"
echo ""
