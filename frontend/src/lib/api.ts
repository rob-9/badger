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

export function queryBookStream(
  params: {
    bookId?: string
    question: string
    selectedText?: string
    useRag?: boolean
    readerPosition?: number
  },
  callbacks: {
    onStatus?: (stage: string) => void
    onToken?: (text: string) => void
    onSources?: (sources: any[]) => void
    onDone?: () => void
    onError?: (error: string) => void
  }
): { abort: () => void } {
  const controller = new AbortController()

  const run = async () => {
    try {
      const response = await fetch(`${API_URL}/api/rag/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          book_id: params.bookId,
          question: params.question,
          selected_text: params.selectedText,
          use_rag: params.useRag ?? !!params.bookId,
          reader_position: params.readerPosition,
        }),
        signal: controller.signal,
      })

      if (!response.ok) {
        const msg = await parseErrorResponse(response, 'Failed to query book')
        callbacks.onError?.(msg)
        return
      }

      const reader = response.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let currentEvent = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop()! // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7)
          } else if (line.startsWith('data: ') && currentEvent) {
            const data = line.slice(6)
            try {
              const parsed = JSON.parse(data)
              switch (currentEvent) {
                case 'status':
                  callbacks.onStatus?.(parsed.stage)
                  break
                case 'token':
                  callbacks.onToken?.(parsed.text)
                  break
                case 'sources':
                  callbacks.onSources?.(parsed)
                  break
                case 'done':
                  callbacks.onDone?.()
                  break
                case 'error':
                  callbacks.onError?.(parsed.message)
                  break
              }
            } catch {
              // Skip unparseable data lines
            }
            currentEvent = ''
          } else if (line === '') {
            currentEvent = ''
          }
        }
      }
    } catch (error: any) {
      if (error.name !== 'AbortError') {
        callbacks.onError?.(error.message || 'Stream failed')
      }
    }
  }

  run()

  return { abort: () => controller.abort() }
}

export async function importLocalEpub(path: string): Promise<{ arrayBuffer: ArrayBuffer; filename: string }> {
  const response = await apiFetch(`${API_URL}/api/epub/import-local`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path })
  })

  if (!response.ok) {
    throw new Error(await parseErrorResponse(response, 'Failed to import EPUB'))
  }

  const arrayBuffer = await response.arrayBuffer()
  const filename = response.headers.get('X-Filename') || 'imported.epub'
  return { arrayBuffer, filename }
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
