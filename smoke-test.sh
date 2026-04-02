#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/backend"
DEV_MODE=lite python -m pytest tests/test_smoke.py -v --tb=short -x "$@"
