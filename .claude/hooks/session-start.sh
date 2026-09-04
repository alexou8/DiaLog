#!/bin/bash
# SessionStart hook for Claude Code on the web.
#
# The remote container is ephemeral: node_modules, globally installed CLIs and
# ~/.codex are all wiped between sessions. This restores enough of that for the
# Claude <-> Codex workflow in CLAUDE.md to work without manual setup.
#
# Optional environment variable, set in the Claude Code environment settings:
#
#   CODEX_AUTH_JSON_B64  base64 of a working ~/.codex/auth.json, so Codex stays
#                        signed in to your ChatGPT plan across sessions. Produce
#                        it on your own machine after `codex login`:
#                          base64 -w0 ~/.codex/auth.json
#                        Without it the session still works; Codex delegation is
#                        simply unavailable until you run `codex login`.
#
# This script never prints the credential. Do not add `set -x`.

set -euo pipefail

# Local checkouts manage their own toolchain and their own Codex login.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}"

# --- Node dependencies -------------------------------------------------------
# `install` rather than `ci` so the cached container layer is reused. This also
# guarantees the *pinned* prettier/eslint/tsc are on disk: without node_modules,
# `npx` silently fetches the latest release and disagrees with CI. See AGENTS.md.
if [ ! -d node_modules ]; then
  npm install --no-audit --no-fund
fi

# --- Prisma client -----------------------------------------------------------
# postinstall covers a fresh install; this covers a warm cache with a newer schema.
npx --no-install prisma generate >/dev/null 2>&1 || true

# --- Codex CLI ---------------------------------------------------------------
if ! command -v codex >/dev/null 2>&1; then
  npm install -g @openai/codex --no-audit --no-fund
fi

# --- Codex authentication ----------------------------------------------------
if codex login status >/dev/null 2>&1; then
  echo "codex: already authenticated"
elif [ -n "${CODEX_AUTH_JSON_B64:-}" ]; then
  mkdir -p "$HOME/.codex"
  umask 077
  if printf '%s' "$CODEX_AUTH_JSON_B64" | base64 -d > "$HOME/.codex/auth.json.tmp" 2>/dev/null &&
     python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$HOME/.codex/auth.json.tmp" 2>/dev/null; then
    chmod 600 "$HOME/.codex/auth.json.tmp"
    mv "$HOME/.codex/auth.json.tmp" "$HOME/.codex/auth.json"
    if codex login status >/dev/null 2>&1; then
      echo "codex: authentication restored from CODEX_AUTH_JSON_B64"
    else
      # Most likely the refresh token was revoked or has expired.
      echo "codex: CODEX_AUTH_JSON_B64 restored but is no longer valid — run 'codex login --device-auth' and refresh the variable" >&2
    fi
  else
    rm -f "$HOME/.codex/auth.json.tmp"
    echo "codex: CODEX_AUTH_JSON_B64 is not valid base64-encoded JSON — leaving it alone" >&2
  fi
else
  echo "codex: not authenticated and CODEX_AUTH_JSON_B64 is unset — run 'codex login --device-auth' to delegate to Codex this session" >&2
fi
