#!/usr/bin/env bash
# Construit le paquet .deb « traduction-gui » (lanceur d'application léger).
# Le .deb n'embarque PAS la stack ML : il lance gui.py depuis le dépôt
# (~/code/traduction par défaut, sinon $TRADUCTION_DIR) avec l'env interview.
#
#   ./packaging/build-deb.sh            → dist/traduction-gui_<ver>_all.deb
#   sudo apt install ./dist/traduction-gui_*_all.deb
set -euo pipefail

VERSION="${VERSION:-1.0.0}"
PKG="traduction-gui"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
BUILD="$ROOT/build/deb"
DIST="$ROOT/dist"

rm -rf "$BUILD"
mkdir -p "$BUILD/DEBIAN" \
         "$BUILD/usr/bin" \
         "$BUILD/usr/share/applications" \
         "$BUILD/usr/share/pixmaps" \
         "$DIST"

# 1) Lanceur — si TRADUCTION_DIR est défini au build, on fige ce chemin comme
#    défaut (utile quand le dépôt n'est pas dans ~/code/traduction). L'override
#    par variable d'environnement TRADUCTION_DIR au runtime reste possible.
install -m 0755 "$HERE/$PKG" "$BUILD/usr/bin/$PKG"
if [ -n "${TRADUCTION_DIR:-}" ]; then
  sed -i "s|\$HOME/code/traduction|${TRADUCTION_DIR}|g" "$BUILD/usr/bin/$PKG"
fi

# 2) Entrée de menu / dock
install -m 0644 "$HERE/$PKG.desktop" "$BUILD/usr/share/applications/$PKG.desktop"

# 3) Icône — bundlée dans le dépôt (portable), avec repli sur l'extension
if [ -f "$HERE/icon.png" ]; then
  install -m 0644 "$HERE/icon.png" "$BUILD/usr/share/pixmaps/$PKG.png"
elif [ -f "$HOME/code/traduction-extension-repo/extension/icon.png" ]; then
  install -m 0644 "$HOME/code/traduction-extension-repo/extension/icon.png" "$BUILD/usr/share/pixmaps/$PKG.png"
fi

# 4) Métadonnées du paquet
cat > "$BUILD/DEBIAN/control" <<EOF
Package: $PKG
Version: $VERSION
Section: utils
Priority: optional
Architecture: all
Depends: python3, curl, xdg-utils
Recommends: chromium | brave-browser | google-chrome-stable
Maintainer: ${DEB_MAINTAINER:-Toolkit Traduction <noreply@example.com>}
Description: Panneau de contrôle Traduction (interface graphique)
 Interface locale pour la boite a outils de traduction/doublage : sous-titres,
 doublage video/audio, resume PDF/EPUB, clips viraux. Lance les scripts Python
 avec choix du moteur LLM (API Claude ou modeles locaux via Ollama) et une
 console de sortie en direct.
EOF

# 5) Rafraîchir le cache des menus/icônes après installation
cat > "$BUILD/DEBIAN/postinst" <<'EOF'
#!/bin/sh
set -e
if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database -q /usr/share/applications || true
fi
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
  gtk-update-icon-cache -q /usr/share/icons/hicolor 2>/dev/null || true
fi
exit 0
EOF
chmod 0755 "$BUILD/DEBIAN/postinst"

# 6) Construction
OUT="$DIST/${PKG}_${VERSION}_all.deb"
dpkg-deb --build --root-owner-group "$BUILD" "$OUT"

echo
echo "✅ Paquet construit : $OUT"
echo "   Installer :  sudo apt install $OUT"
echo "   (ou)         sudo dpkg -i $OUT"
