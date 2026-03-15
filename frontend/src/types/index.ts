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
