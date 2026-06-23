# Toolkit de traduction et doublage IA pour vidéos

Scripts Python indépendants pour **traduire, sous-titrer, doubler, résumer** et extraire des **clips viraux** à partir de vidéos dans n'importe quelle langue, avec WhisperX, Qwen3-TTS / XTTS v2, et un **LLM au choix** : 100 % **local** (Ollama, gratuit, par défaut) ou **Claude** (API Anthropic, via `--llm claude`).

> 🆕 **LLM local par défaut.** Depuis 2026-06, les scripts utilisent un modèle local (Ollama) par défaut — aucune clé API requise. Ajoutez `--llm claude` pour utiliser l'API Anthropic. Voir [LLM : local ou Claude](#llm--local-ollama-ou-claude).
>
> 🖥️ Une **interface graphique** (`gui.py`), un **paquet `.deb`** et une **extension de navigateur** (Chrome/Brave/Edge) permettent de tout piloter sans ligne de commande. Voir [Interface graphique](#interface-graphique-gui) et [Extension Chrome](#extension-chrome-panneau-navigateur).

## Installation rapide (Ubuntu / Pop!_OS, GPU NVIDIA)

```bash
sudo apt update && sudo apt install -y git && \
git clone https://github.com/funkypitt/traduction-toolkit.git && \
cd traduction-toolkit && chmod +x install.sh && ./install.sh
```

L'installeur est **interactif** : il met en place ffmpeg, Miniconda, l'environnement Python + PyTorch/CUDA et toutes les dépendances, puis **propose** (o/n) Qwen3-TTS, Ollama + modèles locaux, l'application GUI (`.deb`) et la configuration des clés API. Comptez ~35 Go de téléchargement si vous prenez les modèles locaux. Détails ci-dessous.

## Les scripts

| Script | Fonction | Sortie |
|--------|----------|--------|
| `traduire.py` | Sous-titrage traduit incrusté dans la vidéo | `video_fr.mp4` + `video_fr.srt` |
| `doubler.py` | Doublage vocal avec clonage de voix et mixage voice-over | `video_dubbed_fr.mp4` |
| `doubler-mp3-batch.py` | Doublage audio en lot (MP3/MP4 du dossier courant) | `fichier_fr.mp3` |
| `resumer.py` | Résumé structuré d'une vidéo en PDF + EPUB | `video.pdf` + `video.epub` |
| `clipper.py` | Extraction de clips viraux avec sous-titres karaoké | `clip.mp4` + `clip.txt` |
| `gui.py` | Interface graphique web pour piloter tous les scripts | — |
| `doctor.py` | Diagnostic et installation des dépendances | — |

## LLM : local (Ollama) ou Claude

Toutes les étapes d'IA textuelle (analyse, traduction, relecture, résumé) tournent **par défaut sur un LLM local** via [Ollama](https://ollama.com) — **gratuit, hors-ligne, aucune clé API**. Pour utiliser l'API Claude à la place, ajoutez `--llm claude`.

```bash
# Installer Ollama puis les modèles recommandés (cf. bench interne)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma4:31b    # traduction / doublage (meilleur français oral)
ollama pull qwen3.6:27b   # résumé / sélection (raisonneur structuré)

# Utilisation (local = défaut)
python traduire.py video.mp4                                    # LLM local
python traduire.py video.mp4 --llm claude                       # API Claude
python traduire.py video.mp4 --ollama-model mistral-small:latest  # autre modèle local
```

| Tâche | Modèle local par défaut |
|-------|-------------------------|
| Traduction (sous-titres, doublage) | `gemma4:31b` |
| Résumé, sélection de clips, alignement | `qwen3.6:27b` |

> GPU NVIDIA recommandé (les modèles ~27–31B tiennent sur 24 Go de VRAM). Les tâches GPU sont **sérialisées** automatiquement (verrou global `~/.cache/traduction_gpu.lock`) pour éviter toute saturation.

### Analyse hybride : Claude pour le contexte (recommandé si vous avez une clé)

Le LLM joue trois rôles : **analyse** (résumé, glossaire, noms propres, domaine), **traduction**, **relecture**. C'est l'analyse qui profite le plus de la connaissance du monde de Claude — et comme la traduction s'appuie sur ce contexte, **un meilleur glossaire améliore aussi la traduction**.

L'option `--analysis-llm` permet d'utiliser **Claude uniquement pour l'analyse**, tout en gardant la **traduction (massive) en local** :

```bash
python traduire.py video.mp4                          # défaut : analyse via Claude SI clé dispo, sinon local
python traduire.py video.mp4 --analysis-llm claude    # forcer Claude pour l'analyse
python traduire.py video.mp4 --analysis-llm local     # tout en local, zéro appel API
```

| Valeur | Comportement |
|--------|--------------|
| `auto` (défaut) | Claude si `ANTHROPIC_API_KEY` est défini, sinon **local** (aucune config requise) |
| `claude` | force Claude pour l'analyse |
| `local` | tout en local, aucun appel API |

> 💰 **Coût** : l'analyse est **un seul appel** Claude (quelques centimes, même sur une vidéo d'une heure) ; la traduction reste **locale et gratuite**. Libre à vous de l'activer (`auto`/`claude`) ou de rester 100 % local (`local`).

## Clés API (optionnelles)

### 1. ANTHROPIC_API_KEY (seulement avec `--llm claude`)

Clé API Claude (Anthropic), pour l'analyse, la traduction et la relecture quand vous préférez Claude au LLM local.

- Créer un compte sur https://console.anthropic.com/
- Aller dans **API Keys** → **Create Key**
- Tarif : environ 3$/million de tokens d'entrée avec Claude Sonnet (quelques centimes par vidéo courte, 1-2$ pour un long reportage)

```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."
```

### 2. HF_TOKEN (obligatoire pour le doublage)

Token HuggingFace, utilisé par Pyannote pour la diarisation des locuteurs (identifier qui parle quand). Requis par `doubler.py` et `doubler-mp3-batch.py`. Non requis par `traduire.py`.

- Créer un compte sur https://huggingface.co/
- Aller dans https://huggingface.co/settings/tokens → **New token** (accès `read`)
- **Accepter les conditions d'utilisation** des modèles Pyannote :
  - https://huggingface.co/pyannote/speaker-diarization-3.1 → Accept
  - https://huggingface.co/pyannote/segmentation-3.0 → Accept

```bash
export HF_TOKEN="hf_..."
```

### 3. ELEVENLABS_API_KEY (optionnel)

Uniquement si vous utilisez le backend ElevenLabs (`--tts elevenlabs`) au lieu d'XTTS v2 dans les scripts de doublage. Non requis par défaut.

```bash
export ELEVENLABS_API_KEY="..."
```

### Rendre les clés permanentes

Ajoutez les exports dans votre `~/.bashrc` :

```bash
echo 'export ANTHROPIC_API_KEY="sk-ant-api03-..."' >> ~/.bashrc
echo 'export HF_TOKEN="hf_..."' >> ~/.bashrc
source ~/.bashrc
```

## Installation

### Automatique (recommandé)

```bash
chmod +x install.sh
./install.sh
```

Le script `install.sh` est **interactif** et installe tout : Miniconda (si absent), l'environnement Python, PyTorch+CUDA, les paquets, ffmpeg, yt-dlp, la GUI (Flask), puis **propose** (interactif) le backend de doublage Qwen3-TTS (env conda dédié), **Ollama + les modèles locaux** (LLM gratuit), et l'**application GUI** (`.deb`, panneau de contrôle dans le menu). Lancez ensuite `python doctor.py` à tout moment pour un diagnostic complet (`--install` pour réparer).

### Mise à jour

Pour une mise à jour de routine, inutile de tout réinstaller. `update.sh` **affiche d'abord ce qui a changé sur GitHub** (commits + fichiers), puis applique et redémarre le daemon de l'extension si besoin :

```bash
./update.sh            # voir les nouveautés PUIS les appliquer
./update.sh --check    # seulement voir ce qui a changé, sans rien appliquer
```

Relancez `install.sh` seulement pour la première installation, pour **ajouter** un composant (Qwen3-TTS, Ollama, app `.deb`, clés API), ou si de nouvelles dépendances Python sont requises (rare).

### Manuelle

#### Prérequis système

```bash
sudo apt update
sudo apt install -y ffmpeg git curl
```

**GPU NVIDIA** (fortement recommandé) : il faut les drivers NVIDIA + CUDA. Vérifier avec :
```bash
nvidia-smi          # doit afficher le GPU
nvcc --version      # doit afficher CUDA 11.8 ou 12.x
```

Si CUDA n'est pas installé, suivre https://developer.nvidia.com/cuda-downloads pour Ubuntu.

#### Environnement Python (Conda)

```bash
conda create -n traduction python=3.11 -y
conda activate traduction

# PyTorch avec CUDA (adapter la version CUDA si nécessaire)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Dépendances communes
pip install whisperx anthropic

# Dépendances supplémentaires pour le doublage
pip install demucs pydub soundfile numpy praat-parselmouth pyworld TTS

# YouTube (optionnel, pour doubler.py)
pip install yt-dlp
```

## Interface graphique (GUI)

Plutôt que la ligne de commande, vous pouvez tout piloter depuis un **panneau de contrôle web** :

```bash
python gui.py        # puis ouvrez http://127.0.0.1:5005
```

Il liste tous les scripts, propose des formulaires (champs, listes déroulantes, interrupteurs), un sélecteur de fichiers, le choix Claude/local + le modèle Ollama, un aperçu de la commande en direct et une console de sortie en streaming.

**Application de bureau (`.deb`)** — pour une vraie icône d'application (menu + dock) :

```bash
./packaging/build-deb.sh
sudo apt install ./dist/traduction-gui_*.deb
# puis cherchez « Traduction » dans le menu, ou lancez : traduction-gui
```

## Extension Chrome (panneau navigateur)

Une **extension de navigateur** (Chrome / Brave / Edge / Chromium) lance le toolkit **directement depuis une vidéo YouTube/X**, sans ligne de commande : elle ouvre un **panneau latéral** où l'on choisit un script et suit la progression.

Tout reste **local** — l'extension parle uniquement à un petit **daemon HTTP local** (`127.0.0.1:47318`) qui télécharge la vidéo et lance le script. Aucun appel externe hormis `yt-dlp`.

### Installation

```bash
cd chrome-extension
./install.sh        # installe le daemon en service systemd utilisateur (sans sudo)
```

Puis dans le navigateur :
1. ouvrir `chrome://extensions` → activer le **Mode développeur** (en haut à droite) ;
2. **Charger l'extension non empaquetée** → sélectionner le dossier `~/traduction-extension` (créé par `install.sh`) ;
3. sur une vidéo **YouTube/X**, cliquer l'icône **Traduction** pour ouvrir le panneau, choisir un script (`traduire`, `doubler`, `clipper`, `resumer`…), ajuster les options, puis **Lancer**. La progression s'affiche dans le panneau, le fichier produit est enregistré localement.

> **Clés API (optionnel)** : les services systemd ne lisent pas `~/.bashrc`. Placez vos clés dans `~/.config/traduction-daemon.env` (`ANTHROPIC_API_KEY=…`, `HF_TOKEN=…`, mode 600), puis `systemctl --user restart traduction-daemon`. Sans clé Claude, tout tourne en **local** ; `HF_TOKEN` reste requis pour le doublage. Détails et dépannage : [`chrome-extension/README.md`](chrome-extension/README.md).

## Usage

### Sous-titrage (traduire.py)

```bash
# Anglais → Français (défaut)
python traduire.py video.mp4

# Japonais → Anglais
python traduire.py video.mp4 -s ja -t en

# Avec style Netflix (ligne courtes, CPS contrôlé)
python traduire.py video.mp4 --style netflix

# Produire seulement le SRT (pas de vidéo)
python traduire.py video.mp4 --skip-burn

# Reprendre un traitement interrompu
python traduire.py video.mp4 --resume segments.json
```

### Doublage vidéo (doubler.py)

```bash
# Doublage EN → FR avec clonage vocal
python doubler.py video.mp4

# Depuis une URL YouTube
python doubler.py "https://www.youtube.com/watch?v=XXXXX"

# Anglais → Espagnol
python doubler.py video.mp4 -s en -t es

# Doublage pur (sans voix originale en fond)
python doubler.py video.mp4 --no-voiceover

# Forcer 2 locuteurs
python doubler.py video.mp4 --speakers 2

# Voice-over style Arte (doux)
python doubler.py video.mp4 --vo-style arte

# Utiliser une voix de référence externe
python doubler.py video.mp4 --ref-voice ma_voix.wav

# Reprendre un traitement interrompu
python doubler.py video.mp4 --segments segments.json
```

### 🎙️ Voix de référence — crucial pour un français naturel

Par défaut, le doublage **clone la voix d'origine** de chaque locuteur (extraite de la vidéo). Mais cette voix porte l'accent de la langue source → le français sonne moins naturel.

**Pour un bien meilleur résultat**, déposez des échantillons de voix françaises dans le dossier **`voix/`** (à côté des scripts). Ils sont utilisés **automatiquement, par genre, sans aucun argument** :

| | |
|---|---|
| **Dossier** | `voix/` (défaut ; surcharge via `--ref-voices <dossier>`) |
| **Nommage** | `homme1.wav`, `homme2.wav`, … (voix masculines) · `femme1.wav`, `femme2.wav`, … (voix féminines) |
| **Contenu** | 15-30 s de parole française **claire**, **un seul locuteur**, sans musique ni bruit de fond |
| **Format** | WAV mono, 24 kHz ou plus |
| **Assignation** | automatique par genre (détecté, ou imposé via `--gender male,female,…`) |

- Voix présentes dans `voix/` → chaque locuteur reçoit une voix du dossier selon son genre.
- Dossier vide ou absent → **fallback : clonage** de la voix d'origine.
- `--map-voices` apparie **manuellement** (interactif) une voix précise à chaque locuteur.

> 💡 La qualité de l'échantillon fait la qualité du doublage : une voix claire et expressive de ~20 s donne un bien meilleur rendu qu'un clip bruité. (Le dossier `voix/` n'est pas versionné — ce sont vos propres voix.)

### Doublage audio en lot (doubler-mp3-batch.py)

```bash
# Traiter tous les MP3/MP4 du dossier courant
python doubler-mp3-batch.py

# Traiter un fichier spécifique
python doubler-mp3-batch.py --file interview.mp3
```

### Extraction de clips viraux (clipper.py)

```bash
# Extraire le passage le plus marquant
python clipper.py video.mp4 --criteria "passage le plus marquant"

# Extraire 2 clips de 3-10 minutes
python clipper.py video.mp4 --criteria "moments clés" --duration 180-600 -n 2

# Extraire et traduire en français
python clipper.py video.mp4 --criteria "key insights" --target-lang fr

# Reprendre un traitement interrompu
python clipper.py video.mp4 --resume video_clips.json --criteria "test"

# Publier automatiquement sur Telegram/X
python clipper.py video.mp4 --criteria "moment drôle" --post
```

## Langues supportées

Les scripts supportent toutes les langues prises en charge par WhisperX et XTTS v2, notamment :
anglais, français, espagnol, allemand, italien, portugais, néerlandais, russe, japonais, chinois, coréen, arabe, hindi, turc, polonais, suédois, danois, norvégien, finnois, tchèque, roumain, hongrois, grec, hébreu, thaï, vietnamien, ukrainien, indonésien, malais, catalan, basque, galicien.

## Compatibilité (Linux / Windows / macOS)

Ce projet est conçu **pour Linux** (testé sur Ubuntu / Pop!_OS) avec un **GPU NVIDIA / CUDA**. C'est l'environnement de référence : tout y fonctionne sans bricolage (`install.sh`, paquet `.deb`, env conda, bridges TTS). Les autres OS sont possibles « pour les courageux », moyennant les obstacles ci-dessous.

| OS | État | Obstacles à lever |
|---|---|---|
| **Linux** (Ubuntu/Pop!_OS, GPU NVIDIA) | ✅ **Supporté nativement** | — |
| **Windows** | ⚠️ Via **WSL2** (recommandé) | Sous **WSL2 + Ubuntu + CUDA**, c'est identique à Linux. En **natif**, deux bloquants : (1) `install.sh` est du bash + `apt` → à refaire à la main (Miniconda Windows, ffmpeg, yt-dlp) ; (2) le **verrou GPU** s'appuie sur `fcntl` (module **Unix**, absent en Windows natif) → les scripts échouent à l'import tant qu'on ne remplace pas ce verrou (par ex. `msvcrt.locking` ou la lib `filelock`). Le `.deb` ne s'applique pas. |
| **macOS** | ⚠️ Tourne, mais **lent** | Le code s'exécute (Unix, `fcntl` OK) et **Ollama fonctionne** sur Apple Silicon (Metal) ; mais **pas de CUDA** → WhisperX, Demucs, Pyannote et le TTS retombent sur **CPU (très lent)**. `install.sh` (`apt`) est à refaire avec **Homebrew**. Praticable pour de courts extraits / tests, pas pour de longues vidéos. |

**En résumé** : Linux + NVIDIA = expérience cible ; sous Windows, passez par **WSL2**. Sur Windows natif ou macOS, c'est adaptable en levant les points ci-dessus (verrou `fcntl`, installeur `apt`→`brew`, accélération GPU).

## Configuration matérielle (GPU)

La VRAM nécessaire dépend surtout du **moteur LLM** choisi (la traduction est l'étape la plus gourmande). Les tâches GPU étant **sérialisées** (un seul modèle en VRAM à la fois), c'est le **pic** qui compte, pas la somme.

| Scénario | VRAM minimum | Cartes typiques |
|---|---|---|
| **LLM = Claude** (`--llm claude`) — sous-titres ou doublage | **~12 Go** | RTX 3060 12 Go / 4070 |
| **LLM local léger** (`--ollama-model mistral-small:latest`) | **~16 Go** | RTX 4060 Ti 16 Go / 4080 |
| **100 % local par défaut** (gemma4:31b traduction, qwen3.6:27b résumé) | **24 Go** | RTX 3090 / 4090 |

En mode **Claude**, le pic GPU vient de **WhisperX large-v3** (~10 Go) ; le doublage ajoute Demucs, la diarisation et le TTS (chacun plus modeste, mais plus de temps de calcul). En mode **local**, le pic vient du **modèle Ollama** (~20 Go mesurés pour un 27-31B).

- **RAM** : 16 Go recommandé (8 Go minimum, serré pour le doublage d'une longue vidéo).
- **Disque** : ~10 Go (WhisperX large-v3 + XTTS v2, téléchargés au 1er lancement) **+ ~36 Go** si vous installez les modèles Ollama locaux par défaut (gemma4 + qwen3.6).
- **Sans GPU** : techniquement possible mais **très lent** — déconseillé au-delà de quelques minutes de vidéo.

> 💡 Carte ≤ 16 Go : gardez le local avec un modèle plus léger (`--ollama-model mistral-small:latest`), ou passez le LLM sur **Claude** (`--llm claude`). En cas d'OOM sur WhisperX, réduisez `WHISPER_BATCH_SIZE` (voir [Dépannage](#dépannage)).

## Fichiers produits

Chaque exécution crée des fichiers à côté de la vidéo source :

```
video.mp4                          # source
video_fr.mp4                       # sous-titres incrustés (traduire.py)
video_fr.srt                       # sous-titres SRT
video_dubbed_fr.mp4                # vidéo doublée (doubler.py)
video_dubbing_report.txt           # rapport de doublage
video_dubbing_work/                # dossier de travail (segments, audio intermédiaire)
  ├── segments.json                # segments (checkpoint, réutilisable avec --segments)
  ├── analysis.json                # analyse Claude
  ├── audio_hq.wav                 # audio extrait
  ├── vocals.wav / no_vocals.wav   # séparation Demucs
  ├── mixed_audio.wav              # audio mixé final
  └── tts_*.wav                    # clips TTS individuels
```

## Dépannage

| Problème | Solution |
|----------|----------|
| `CUDA out of memory` | Réduire `WHISPER_BATCH_SIZE` en haut du script (8 au lieu de 16) |
| Diarisation échoue | Vérifier que `HF_TOKEN` est défini et que les modèles Pyannote sont acceptés sur HuggingFace |
| `ffmpeg: subtitle filter not found` | Réinstaller ffmpeg avec libass : `sudo apt install ffmpeg libavcodec-extra` |
| `yt-dlp` ne fonctionne pas | Mettre à jour : `pip install -U yt-dlp` |
| Premier lancement très long | Normal — les modèles WhisperX (~5 Go) et XTTS v2 (~2 Go) se téléchargent |
| Traitement interrompu | Utiliser `--resume segments.json` (traduire.py) ou `--segments segments.json` (doublage) |
