export interface Document {
  id: string
  title: string
  content: string
  type: 'pdf' | 'epub' | 'docx' | 'txt'
  createdAt: Date
  lastRead: Date
}

export interface Highlight {
  id: string
  documentId: string
  text: string
  startOffset: number
  endOffset: number
  color: string
  note?: string
  createdAt: Date
}

export interface AgentContext {
  selectedText: string
  surroundingText: string
  documentTitle: string
  currentPage?: number
}

export interface AgentResponse {
  explanation: string
  definitions?: string[]
  relatedConcepts?: string[]
  suggestions?: string[]
}

export interface ReadingSession {
  documentId: string
  startTime: Date
  endTime?: Date
  wordsRead: number
  highlights: Highlight[]
}