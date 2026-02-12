# Boom

AI-powered reading assistant.

## Structure

```
boom/
├── backend/          # Python FastAPI server
├── frontend/         # Next.js web app
└── mcp/             # MCP server (future)
```

## Setup

**Backend:**
```bash
cd backend
pip install -e .
cp .env.example .env  # Add API keys
```

**Frontend:**
```bash
cd frontend
npm install
```

## Run

```bash
./dev.sh
```

Or manually:
```bash
# Terminal 1
cd backend && uvicorn boom.api.server:app --reload --port 8000

# Terminal 2
cd frontend && npm run dev
```

Open http://localhost:3000
