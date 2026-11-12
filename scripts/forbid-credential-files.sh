#!/usr/bin/env bash
# pre-commit hook: refuse to commit a real credentials file.
#
# .gitignore already excludes .env, but `git add -f` overrides it, a rename can slip one past,
# and .gitignore only knows the shapes it lists. Your LiveKit API key and secret are
# admin-equivalent for the whole project, so never commit them: a commit containing them is a
# credential leak for everyone using that project.
#
# Matching is case-insensitive on purpose: macOS filesystems are case-insensitive, so `.ENV`
# opens the same file locally while `.gitignore`'s `.env` would not catch it on a case-sensitive
# filesystem -- a Linux teammate or CI.
#
# Tracked templates (.env.example and friends) hold only variable names and are always allowed.
set -euo pipefail
shopt -s nocasematch

status=0

for path in "$@"; do
  base=$(basename -- "$path")
  case "$base" in
    .env.example | .env.template | .env.sample | *.env.example)
      continue
      ;;
    *.py | *.md | *.mdx | *.rst)
      # Source and prose, whatever it is called. `credentials.*` below is a filename heuristic for
      # files that *are* a credential store, and it false-positives on code that merely works with
      # credentials -- `scripts/check_credentials.py` is exactly that, and was blocked by it.
      # Secrets hardcoded inside a source file are gitleaks' job, not this hook's.
      # Deliberately narrow. Not .json or .ini (`credentials.json` is a real credential store),
      # not .toml (`.streamlit/secrets.toml` is one), and not .sh (an exports file is one).
      # Excluding .sh costs a false positive on a name like `rotate_credentials.sh`; that is the
      # right side to err on, and the message tells you to rename.
      continue
      ;;
    .env | .env.* | *.env | .envrc | .netrc | .pgpass)
      reason="credentials file"
      ;;
    credentials | credentials.* | *-credentials.* | *_credentials.*)
      reason="credentials file"
      ;;
    id_rsa | id_dsa | id_ecdsa | id_ed25519 | *.pem | *.p12 | *.pfx | *.keystore)
      reason="private key or keystore"
      ;;
    *)
      continue
      ;;
  esac

  echo "refusing to commit $path -- this looks like a $reason." >&2
  echo "  Only templates such as .env.example belong in git." >&2
  status=1
done

exit "$status"
