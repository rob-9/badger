/**
 * API Route: Query a book with RAG
 *
 * POST /api/rag/query
 * Body: { bookId: string, question: string, selectedText?: string }
 *
 * This endpoint:
 * 1. Embeds the question
 * 2. Finds relevant chunks
 * 3. Generates answer with Claude
 */

import { NextRequest, NextResponse } from 'next/server'
import { queryBook, querySimple } from '@/lib/rag'

export async function POST(request: NextRequest) {
  try {
    const { bookId, question, selectedText, useRag = true } = await request.json()

    if (!question) {
      return NextResponse.json(
        { error: 'Missing question' },
        { status: 400 }
      )
    }

    console.log(`[API] Query: "${question}", bookId: ${bookId}, useRag: ${useRag}`)

    // If RAG is disabled or no bookId, use simple query
    if (!useRag || !bookId) {
      if (!selectedText) {
        return NextResponse.json(
          { error: 'Selected text required when RAG is disabled' },
          { status: 400 }
        )
      }

      const answer = await querySimple(question, selectedText)
      return NextResponse.json({ answer, sources: [] })
    }

    // Use full RAG
    const response = await queryBook(bookId, question, selectedText)

    return NextResponse.json(response)
  } catch (error) {
    console.error('[API] Query error:', error)
    return NextResponse.json(
      { error: 'Failed to process query' },
      { status: 500 }
    )
  }
}
