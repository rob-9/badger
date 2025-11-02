# boom

Read Better with AI.

## Setup

**Requirements:**
- Node.js 18+

**Install:**
```bash
npm install
npm run dev
```

Open `http://localhost:3000`

## Usage

1. **Upload** - Drag file or click to browse
2. **Read** - Select text for AI explanations  
3. **Navigate** - Use toolbar controls

**Supported formats:** TXT, PDF, EPUB, DOCX

## AI Integration

Add OpenAI API key to `.env.local`:
```
OPENAI_API_KEY=your_key_here
```

Uncomment integration code in `src/app/api/agent/route.ts`

## Architecture

```
src/
├── app/              # Next.js routes
├── components/       # UI components
├── lib/             # Utilities
├── hooks/           # React hooks
└── types/           # TypeScript types
```

**Key Components:**
- `DocumentViewer` - Main reader interface
- `AgentAssistant` - AI explanation modal
- `FileUploader` - Document upload
- `TextHighlighter` - Text selection/highlighting

## Commands

```bash
npm run dev        # Development server
npm run build      # Production build
npm run lint       # Code linting
npm run typecheck  # Type checking
```

## Stack

- Next.js 14 + TypeScript
- Tailwind CSS  
- Lucide React icons
- OpenAI API (optional)
