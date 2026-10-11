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
# `ci`, not `install`: it installs exactly what package-lock.json records and
# never rewrites the lockfile, which is what AGENTS.md and CI do. `install` could
# quietly resolve newer versions and leave the working tree dirty. Skipped when
# node_modules already exists (a warm container layer), which also guarantees the
# *pinned* prettier/eslint/tsc are on disk: without node_modules, `npx` silently
# fetches the latest release and disagrees with CI. See AGENTS.md.
if [ ! -d node_modules ]; then
  npm ci --no-audit --no-fund
fi

# --- Prisma client -----------------------------------------------------------
# postinstall covers a fresh install; this covers a warm cache with a newer
# schema. A failure here is actionable — every typecheck and test that imports
# @prisma/client would then fail for a confusing, unrelated-looking reason — so
# it is not swallowed: the output is shown and the hook fails.
if ! prisma_output="$(npx --no-install prisma generate 2>&1)"; then
  echo "$prisma_output" >&2
  echo "session-start: 'prisma generate' failed. Fix the schema or the install before relying on typecheck or tests." >&2
  exit 1
fi

# --- Codex CLI ---------------------------------------------------------------
# Pinned. An unpinned `npm install -g @openai/codex` runs whatever the registry
# published most recently — third-party code with access to this session's
# environment — on every remote session. Bump this deliberately. (The manual
# `npm install -g` snippet in CLAUDE.md is a human setup step and is not pinned.)
CODEX_VERSION="0.162.1"
if ! command -v codex >/dev/null 2>&1; then
  if ! npm install -g "@openai/codex@${CODEX_VERSION}" --no-audit --no-fund; then
    # Codex is optional for the repository itself, so do not fail the whole
    # session — but do say clearly what is lost, rather than hiding the error.
    echo "codex: installing @openai/codex@${CODEX_VERSION} failed — Codex delegation (CLAUDE.md) is unavailable this session. Everything else works." >&2
    exit 0
  fi
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
