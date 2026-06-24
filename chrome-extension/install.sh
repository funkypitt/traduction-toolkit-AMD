#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# install.sh — Installe le daemon de l'extension Chrome « Traduction »
# ═══════════════════════════════════════════════════════════════════════════════
# L'extension Chrome (panneau latéral) parle à un petit daemon HTTP LOCAL
# (127.0.0.1:47318) qui lance les scripts du toolkit sur la vidéo de l'onglet
# courant. Ce script installe ce daemon en service systemd utilisateur (pas de
# sudo), avec des chemins portables (il détecte la racine du dépôt et le Python).
#
#   ./install.sh           # depuis chrome-extension/
#
# Puis on charge l'extension dans Chrome (« Load unpacked ») — instructions à la fin.
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # …/chrome-extension
REPO="$(cd "$HERE/.." && pwd)"                          # racine du dépôt (scripts)
SYSTEMD_DIR="$HOME/.config/systemd/user"
EXT_LINK="$HOME/traduction-extension"                   # cible du « Load unpacked »
SERVICE="$SYSTEMD_DIR/traduction-daemon.service"

# Interpréteur du toolkit (interview/traduction/override), comme gui.py/doctor.py
PY="${TRADUCTION_PYTHON:-}"
if [ ! -x "$PY" ]; then
  for e in traduction-amd interview traduction; do   # AMD fork : env ROCm en premier
    [ -x "$HOME/miniconda3/envs/$e/bin/python" ] && PY="$HOME/miniconda3/envs/$e/bin/python" && break
  done
fi
[ -x "$PY" ] || PY="$(command -v python3)"

mkdir -p "$SYSTEMD_DIR"
chmod +x "$HERE/daemon/traduction-daemon.py"

echo "→ génération du service systemd (dépôt: $REPO)"
cat > "$SERVICE" <<EOF
[Unit]
Description=Traduction local HTTP daemon (pont pour l'extension Chrome)
After=network.target

[Service]
Type=simple
ExecStart=$PY $HERE/daemon/traduction-daemon.py
Restart=on-failure
RestartSec=2
# Clés API (ANTHROPIC_API_KEY, HF_TOKEN, ELEVENLABS_API_KEY) — un KEY=value par
# ligne, mode 600. Les services systemd ne lisent PAS ~/.bashrc.
EnvironmentFile=-%h/.config/traduction-daemon.env
Environment=TRADUCTION_PYTHON=$PY
Environment=TRADUCTION_SCRIPTS_DIR=$REPO

[Install]
WantedBy=default.target
EOF

echo "→ lien du dossier extension (pour Chrome « Load unpacked »)"
if [ -L "$EXT_LINK" ]; then rm "$EXT_LINK"
elif [ -e "$EXT_LINK" ]; then mv "$EXT_LINK" "$EXT_LINK.backup.$(date +%s)"; fi
ln -s "$HERE/extension" "$EXT_LINK"

echo "→ activation + démarrage du service"
systemctl --user daemon-reload
systemctl --user enable --now traduction-daemon.service
sleep 1
if curl -sf "http://127.0.0.1:47318/ping" >/dev/null 2>&1; then
  echo "  ✓ daemon en écoute sur http://127.0.0.1:47318"
else
  echo "  ⚠️  daemon pas encore joignable — voir : journalctl --user -u traduction-daemon -e"
fi

cat <<EOF

✅ Daemon installé.

Étapes côté navigateur (Chrome / Brave / Edge / Chromium) :
  1. Ouvrir  chrome://extensions
  2. Activer « Mode développeur » (en haut à droite)
  3. « Charger l'extension non empaquetée » → choisir :  $EXT_LINK
  4. Sur une page YouTube/X, ouvrir le panneau latéral « Traduction »,
     choisir un script (traduire, doubler, clipper…) et lancer.

Clés API (optionnel) : créez ~/.config/traduction-daemon.env (mode 600), ex. :
    ANTHROPIC_API_KEY=sk-ant-...
    HF_TOKEN=hf_...
puis :  systemctl --user restart traduction-daemon
(Sans clé Claude, les scripts tournent en local ; HF_TOKEN requis pour le doublage.)
EOF
