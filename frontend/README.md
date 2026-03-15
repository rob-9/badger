# Badger Frontend

Next.js frontend for Badger reading assistant.

## Quick Start

```bash
npm install
npm run dev
```

Runs on http://localhost:3000

## Structure

```
frontend/
├── src/
│   ├── app/           # Next.js App Router
│   │   └── page.tsx   # Main app
│   ├── components/    # React components
│   │   ├── EpubReader.tsx
│   │   ├── ChatPanel.tsx
│   │   └── ...
│   └── lib/          # Utilities
│       ├── api.ts         # Backend API client
│       └── bookStorage.ts # IndexedDB storage
└── public/           # Static assets
```

## Features

- **EPUB Reader** - Full-featured EPUB reading experience
- **PDF Viewer** - Read PDF documents
- **Book Library** - Browse your reading history
- **AI Chat** - Ask questions about books (calls Python backend)
- **Text Selection** - Select text to get AI explanations

## Backend Integration

This frontend calls the Python backend API at `http://localhost:8000`:
- `/api/rag/index` - Index books for search
- `/api/rag/query` - Query books with RAG
- `/api/agent` - Get AI assistance

See `src/lib/api.ts` for API client implementation.
