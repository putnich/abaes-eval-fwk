#!/usr/bin/env bash
# Clean-room reader gate: install from a wheel in a fresh venv outside the checkout, run the
# demo offline, then extend the framework from a reader-owned file. Runs unattended.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$here/verify_clean_clone.py" "$@"
