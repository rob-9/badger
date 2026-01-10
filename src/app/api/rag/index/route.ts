/**
 * API Route: Index a book for RAG
 *
 * POST /api/rag/index
 * Body: { bookId: string, text: string }
 *
 * This endpoint:
 * 1. Chunks the book text
 * 2. Embeds each chunk
 * 3. Stores in vector store
 */

import { NextRequest, NextResponse } from 'next/server'
import { indexBook } from '@/lib/rag'

export async function POST(request: NextRequest) {
  try {
    const { bookId, text } = await request.json()

    if (!bookId || !text) {
      return NextResponse.json(
        { error: 'Missing bookId or text' },
        { status: 400 }
      )
    }

    console.log(`[API] Indexing book: ${bookId}, text length: ${text.length}`)

    await indexBook(bookId, text)

    return NextResponse.json({
      success: true,
      message: `Book successfully indexed: ${bookId}`
    })
  } catch (error) {
    console.error('[API] Index error:', error)
    return NextResponse.json(
      { error: 'Failed to index book' },
      { status: 500 }
    )
  }
}
