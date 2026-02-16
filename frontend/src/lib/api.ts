/**
 * API client for Python backend.
 *
 * All API calls go to the Python FastAPI server running on localhost:8000
 */

import type { StructuredBook } from '@/lib/parseEpub'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

async function parseErrorResponse(response: Response, fallback: string): Promise<string> {
  try {
    const error = await response.json()
    return error.detail || fallback
  } catch {
    return `${fallback} (${response.status})`
  }
}

async function apiFetch(url: string, options: RequestInit): Promise<Response> {
  try {
    return await fetch(url, options)
  } catch (error) {
    throw new Error(
      `Cannot reach backend at ${API_URL}. Is the server running?`
    )
  }
}

export async function isBookIndexed(bookId: string): Promise<boolean> {
  try {
    const response = await apiFetch(`${API_URL}/api/rag/indexed/${encodeURIComponent(bookId)}`, {})
    if (!response.ok) return false
    const data = await response.json()
    return data.indexed
  } catch {
    return false
  }
}

export async function indexBook(bookId: string, content: StructuredBook | string): Promise<void> {
  const payload = typeof content === 'string'
    ? { book_id: bookId, text: content }
    : { book_id: bookId, structured_content: content }

  const response = await apiFetch(`${API_URL}/api/rag/index`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })

  if (!response.ok) {
    throw new Error(await parseErrorResponse(response, 'Failed to index book'))
  }
}

export async function queryBook(params: {
  bookId?: string
  question: string
  selectedText?: string
  useRag?: boolean
  readerPosition?: number
}): Promise<{ answer: string; sources?: any[] }> {
  const response = await apiFetch(`${API_URL}/api/rag/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      book_id: params.bookId,
      question: params.question,
      selected_text: params.selectedText,
      use_rag: params.useRag ?? !!params.bookId,
      reader_position: params.readerPosition
    })
  })

  if (!response.ok) {
    throw new Error(await parseErrorResponse(response, 'Failed to query book'))
  }

  return response.json()
}

export async function getAgentAssistance(params: {
  selectedText: string
  surroundingText?: string
  documentTitle?: string
}): Promise<{
  explanation: string
  definitions: string[]
  relatedConcepts: string[]
  suggestions: string[]
}> {
  const response = await apiFetch(`${API_URL}/api/agent`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      selected_text: params.selectedText,
      surrounding_text: params.surroundingText,
      document_title: params.documentTitle
    })
  })

  if (!response.ok) {
    throw new Error(await parseErrorResponse(response, 'Failed to get agent assistance'))
  }

  return response.json()
}
