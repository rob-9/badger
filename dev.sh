#!/bin/bash
# Simple dev script to run both backend and frontend

echo "starting dev servers.."
echo ""
echo "backend: http://localhost:8000"
echo "frontend: http://localhost:3000"
echo ""

# Start backend in background
cd backend && uvicorn boom.api.server:app --reload --port 8000 &
BACKEND_PID=$!

# Start frontend in foreground
cd frontend && npm run dev

# Cleanup: Kill backend when frontend stops
kill $BACKEND_PID
