#!/usr/bin/env bash
# install-hermes.sh — Install vault-memory as a Hermes memory plugin
#
# Usage:
#   bash install-hermes.sh [--db /path/to/vectors.db] [--gemini-key YOUR_KEY]
#
# What it does:
#   1. Symlinks this repo into ~/.hermes/hermes-agent/plugins/memory/vault-memory
#   2. Adds VAULT_MEMORY_DB and GEMINI_API_KEY to ~/.hermes/.env (if not set)
#   3. Creates the data directory
#   4. Prints next steps

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
PLUGINS_DIR="$HERMES_HOME/hermes-agent/plugins/memory"
PLUGIN_LINK="$PLUGINS_DIR/vault-memory"
HERMES_ENV="$HERMES_HOME/.env"
DEFAULT_DB="$HOME/hache/data/vectors.db"

# ── Parse args ────────────────────────────────────────────────────────────────
DB_PATH=""
GEMINI_KEY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --db) DB_PATH="$2"; shift 2 ;;
        --gemini-key) GEMINI_KEY="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

DB_PATH="${DB_PATH:-$DEFAULT_DB}"

# ── 1. Symlink plugin ─────────────────────────────────────────────────────────
if [ ! -d "$PLUGINS_DIR" ]; then
    echo "Error: Hermes plugins directory not found at $PLUGINS_DIR"
    echo "Is Hermes installed? Expected: $HERMES_HOME/hermes-agent/"
    exit 1
fi

if [ -L "$PLUGIN_LINK" ]; then
    echo "ℹ️  Plugin symlink already exists — updating..."
    rm "$PLUGIN_LINK"
fi

ln -s "$SCRIPT_DIR" "$PLUGIN_LINK"
echo "✅ Plugin linked: $PLUGIN_LINK → $SCRIPT_DIR"

# ── 2. Data directory ─────────────────────────────────────────────────────────
mkdir -p "$(dirname "$DB_PATH")"
echo "✅ Data directory ready: $(dirname "$DB_PATH")"

# ── 3. Environment variables ──────────────────────────────────────────────────
touch "$HERMES_ENV"

add_env() {
    local key="$1" val="$2"
    if grep -q "^${key}=" "$HERMES_ENV" 2>/dev/null; then
        echo "ℹ️  $key already in .env — skipping"
    elif grep -q "^#.*${key}" "$HERMES_ENV" 2>/dev/null; then
        echo "ℹ️  $key found commented in .env — skipping (edit manually)"
    else
        echo "${key}=${val}" >> "$HERMES_ENV"
        echo "✅ Added $key to $HERMES_ENV"
    fi
}

add_env "VAULT_MEMORY_DB" "$DB_PATH"
[ -n "$GEMINI_KEY" ] && add_env "GEMINI_API_KEY" "$GEMINI_KEY"

# ── 4. Next steps ─────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " vault-memory installed ✅"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next steps:"
echo ""

if ! grep -q "^GEMINI_API_KEY=" "$HERMES_ENV" 2>/dev/null; then
    echo "  1. Add your Gemini API key to ~/.hermes/.env:"
    echo "     echo 'GEMINI_API_KEY=your_key' >> ~/.hermes/.env"
    echo "     (Get a free key at https://aistudio.google.com/app/apikey)"
    echo ""
fi

echo "  2. Enable in Hermes:"
echo "     hermes memory setup"
echo "     → select: vault-memory"
echo ""
echo "  3. Index your collections:"
echo "     vm index ~/obsidian-vault         --collection obsidian"
echo "     vm index ~/.hermes/memories       --collection hermes-memory"
echo "     vm index ~/.hermes/skills         --collection hermes-memory"
echo ""
echo "  4. Set up nightly reindex (optional):"
echo "     crontab -e"
echo "     # Add: 0 4 * * * /home/\$USER/hache/scripts/vector-reindex.sh >> ~/hache/logs/vector-reindex.log 2>&1"
echo ""
echo "  DB path: $DB_PATH"
