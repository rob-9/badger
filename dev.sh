#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "starting dev servers.."
echo ""
echo "backend: http://localhost:8000"
echo "frontend: http://localhost:3000"
echo ""

cleanup() {
  echo "stopping servers..."
  kill "$BACKEND_PID" 2>/dev/null
  exit 0
}
trap cleanup INT TERM

# Start backend in background
cd "$SCRIPT_DIR/backend" && uvicorn boom.api.server:app --reload --port 8000 --log-level info 2>&1 | sed 's/^/[backend] /' &
BACKEND_PID=$!

# Start frontend in foreground
cd "$SCRIPT_DIR/frontend" && npm run dev

cleanup
