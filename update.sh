#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# update.sh — Met à jour la boîte à outils (rapide, non interactif)
# ═══════════════════════════════════════════════════════════════════════════════
# Montre d'abord CE QUI A CHANGÉ sur GitHub, puis applique et redémarre le daemon
# de l'extension si besoin.
#
#   ./update.sh            # voir les nouveautés PUIS les appliquer
#   ./update.sh --check    # voir les nouveautés SANS rien appliquer
#
# Relancez install.sh à la place pour : la première installation, AJOUTER un
# composant (env Qwen3-TTS, Ollama, app .deb, clés API), ou si de NOUVELLES
# dépendances Python sont requises (rare).
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; NC='\033[0m'
info() { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()   { echo -e "${GREEN}[OK]${NC}    $*"; }
note() { echo -e "${YELLOW}[NOTE]${NC}  $*"; }

CHECK_ONLY=0; [ "${1:-}" = "--check" ] && CHECK_ONLY=1
CHANGED=0

# ── 1) Voir ce qui a changé sur GitHub ───────────────────────────────────────
if [ ! -d .git ]; then
    note "Pas un dépôt git — téléchargez la dernière version manuellement."
else
    info "Vérification des nouveautés sur GitHub (git fetch)…"
    git fetch --quiet
    BEFORE="$(git rev-parse HEAD)"
    UPSTREAM="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null \
                || echo "origin/$(git rev-parse --abbrev-ref HEAD)")"
    REMOTE="$(git rev-parse "$UPSTREAM" 2>/dev/null || echo "$BEFORE")"

    if [ "$BEFORE" = "$REMOTE" ]; then
        ok "Déjà à jour ($(git rev-parse --short HEAD)) — rien de nouveau sur GitHub."
    else
        N="$(git rev-list --count "$BEFORE..$REMOTE")"
        echo
        info "$N nouveau(x) commit(s) en attente sur GitHub :"
        git log --oneline --no-decorate "$BEFORE..$REMOTE" | sed 's/^/    /'
        echo
        info "Fichiers concernés :"
        git diff --stat "$BEFORE" "$REMOTE" | sed 's/^/    /'
        echo
        if [ "$CHECK_ONLY" = 1 ]; then
            note "Mode --check : rien n'a été appliqué. Lancez  ./update.sh  pour mettre à jour."
            exit 0
        fi
        info "Application (git pull)…"
        git pull --ff-only --quiet
        AFTER="$(git rev-parse HEAD)"
        CHANGED=1
        ok "Mis à jour : $(git rev-parse --short "$BEFORE") → $(git rev-parse --short "$AFTER")"
    fi
fi

[ "$CHECK_ONLY" = 1 ] && exit 0

# ── 2) Redémarrer le daemon de l'extension (si présent et si le code a changé) ─
if [ "$CHANGED" = 1 ] && systemctl --user list-unit-files 2>/dev/null | grep -q '^traduction-daemon\.service'; then
    info "Redémarrage du daemon de l'extension…"
    systemctl --user restart traduction-daemon.service || true
    ok "Daemon redémarré (http://127.0.0.1:47318)."
fi

# ── 3) .deb : rebuild seulement si le LANCEUR a changé (gui.py est lu à chaud) ─
if [ "$CHANGED" = 1 ] && dpkg -s traduction-gui >/dev/null 2>&1 \
   && git diff --name-only "$BEFORE" "$AFTER" 2>/dev/null | grep -q '^packaging/'; then
    note "Le lanceur .deb a changé — pour le mettre à jour :"
    note "  TRADUCTION_DIR=\"$HERE\" ./packaging/build-deb.sh && sudo apt install ./dist/traduction-gui_*.deb"
fi

echo
ok "Terminé."
if [ "$CHANGED" = 1 ]; then
    note "Nouvelles dépendances Python ? (rare) relancez  ./install.sh  (idempotent)."
    note "Modèles Ollama par défaut changés ? mettez à jour avec  ollama pull <modèle>."
fi
