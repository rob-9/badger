/**
 * Tests for queryBookStream — SSE parsing, fallback, and abort behavior.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { queryBookStream } from './api'

// --- helpers ---

/** Build an SSE text chunk from event/data pairs. */
function sseChunk(events: Array<{ event: string; data: any }>): string {
  return events.map(e =>
    `event: ${e.event}\ndata: ${JSON.stringify(e.data)}\n\n`
  ).join('')
}

/** Create a ReadableStream that yields the given text chunks. */
function makeStream(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder()
  let i = 0
  return new ReadableStream({
    pull(controller) {
      if (i < chunks.length) {
        controller.enqueue(encoder.encode(chunks[i++]))
      } else {
        controller.close()
      }
    },
  })
}

/** Create a mock Response with the given SSE body. */
function sseResponse(chunks: string[]): Response {
  return new Response(makeStream(chunks), {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream' },
  })
}

/** Wait for all pending microtasks / async work to complete. */
function flush(): Promise<void> {
  return new Promise(r => setTimeout(r, 50))
}

const defaultParams = { question: 'What happens?', selectedText: 'test' }

// --- tests ---

beforeEach(() => {
  vi.restoreAllMocks()
})

describe('queryBookStream', () => {
  it('parses SSE tokens and calls onDone', async () => {
    const body = sseChunk([
      { event: 'status', data: { stage: 'generating' } },
      { event: 'token', data: { text: 'Hello' } },
      { event: 'token', data: { text: ' world' } },
      { event: 'done', data: {} },
    ])
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(sseResponse([body]))

    const tokens: string[] = []
    const statuses: string[] = []
    let done = false

    queryBookStream(defaultParams, {
      onStatus: (s) => statuses.push(s),
      onToken: (t) => tokens.push(t),
      onDone: () => { done = true },
    })
    await flush()

    expect(statuses).toEqual(['generating'])
    expect(tokens).toEqual(['Hello', ' world'])
    expect(done).toBe(true)
  })

  it('parses sources event', async () => {
    const body = sseChunk([
      { event: 'sources', data: [{ text: 'chunk', score: 0.9 }] },
      { event: 'done', data: {} },
    ])
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(sseResponse([body]))

    let sources: any[] = []
    queryBookStream(defaultParams, {
      onSources: (s) => { sources = s },
    })
    await flush()

    expect(sources).toEqual([{ text: 'chunk', score: 0.9 }])
  })

  it('handles SSE error events', async () => {
    const body = sseChunk([
      { event: 'error', data: { message: 'something broke' } },
    ])
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(sseResponse([body]))

    let error = ''
    queryBookStream(defaultParams, {
      onError: (e) => { error = e },
    })
    await flush()

    expect(error).toBe('something broke')
  })

  it('handles chunks split across multiple reads', async () => {
    // Split an SSE event across two chunks
    const fullEvent = 'event: token\ndata: {"text":"split"}\n\n'
    const chunk1 = fullEvent.slice(0, 15)
    const chunk2 = fullEvent.slice(15)

    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(sseResponse([chunk1, chunk2]))

    const tokens: string[] = []
    queryBookStream(defaultParams, {
      onToken: (t) => tokens.push(t),
    })
    await flush()

    expect(tokens).toEqual(['split'])
  })

  it('falls back to non-streaming when stream fetch fails', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
    // First call (stream) fails
    fetchSpy.mockRejectedValueOnce(new Error('Connection refused'))
    // Second call (fallback) succeeds
    fetchSpy.mockResolvedValueOnce(
      new Response(JSON.stringify({ answer: 'fallback answer', sources: [{ text: 'src' }] }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    )

    const tokens: string[] = []
    const statuses: string[] = []
    let sources: any[] = []
    let done = false

    queryBookStream(defaultParams, {
      onStatus: (s) => statuses.push(s),
      onToken: (t) => tokens.push(t),
      onSources: (s) => { sources = s },
      onDone: () => { done = true },
    })
    await flush()

    // Should have fallen back
    expect(fetchSpy).toHaveBeenCalledTimes(2)
    expect(fetchSpy.mock.calls[1][0]).toContain('/api/rag/query')
    expect(statuses).toContain('generating')
    expect(tokens).toEqual(['fallback answer'])
    expect(sources).toEqual([{ text: 'src' }])
    expect(done).toBe(true)
  })

  it('falls back when stream returns non-ok status', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
    // Stream returns 500
    fetchSpy.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: 'Internal error' }), { status: 500 })
    )
    // Fallback succeeds
    fetchSpy.mockResolvedValueOnce(
      new Response(JSON.stringify({ answer: 'recovered' }), { status: 200 })
    )

    const tokens: string[] = []
    let done = false

    queryBookStream(defaultParams, {
      onToken: (t) => tokens.push(t),
      onDone: () => { done = true },
    })
    await flush()

    expect(fetchSpy).toHaveBeenCalledTimes(2)
    expect(tokens).toEqual(['recovered'])
    expect(done).toBe(true)
  })

  it('calls onError when both stream and fallback fail', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
    fetchSpy.mockRejectedValueOnce(new Error('stream down'))
    fetchSpy.mockRejectedValueOnce(new Error('fallback down'))

    let error = ''
    queryBookStream(defaultParams, {
      onError: (e) => { error = e },
    })
    await flush()

    expect(error).toBe('fallback down')
  })

  it('does not call onError when aborted', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
    const abortError = new DOMException('The operation was aborted.', 'AbortError')
    fetchSpy.mockRejectedValueOnce(abortError)

    let errorCalled = false
    queryBookStream(defaultParams, {
      onError: () => { errorCalled = true },
    })
    await flush()

    expect(errorCalled).toBe(false)
    // Should not attempt fallback either
    expect(fetchSpy).toHaveBeenCalledTimes(1)
  })

  it('abort() cancels the stream', async () => {
    // Hang the stream so abort is the only way out
    const neverResolve = new Promise<Response>(() => {})
    vi.spyOn(globalThis, 'fetch').mockReturnValueOnce(neverResolve)

    const handle = queryBookStream(defaultParams, {})
    handle.abort()
    // Just verify it doesn't throw
    await flush()
  })
})
