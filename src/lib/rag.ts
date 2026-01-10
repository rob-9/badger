/**
 * RAG (Retrieval Augmented Generation) SERVICE
 *
 * This ties together:
 * 1. Chunking - Split book into pieces
 * 2. Embedding - Convert text to vectors (Voyage AI)
 * 3. Storage - Store vectors for search
 * 4. Retrieval - Find relevant chunks
 * 5. Generation - Answer with context (Claude)
 *
 * Learn more:
 * - https://docs.anthropic.com/en/docs/build-with-claude/retrieval-augmented-generation
 * - https://docs.voyageai.com/
 * - https://github.com/anthropics/claude-cookbooks
 */

import { VoyageAIClient } from 'voyageai'
import Anthropic from '@anthropic-ai/sdk'
import { chunkText, type TextChunk } from './chunker'
import { vectorStore, type VectorEntry } from './vectorStore'

// Initialize clients (server-side only)
// These will be initialized when the API routes are called
let voyage: VoyageAIClient | null = null
let anthropic: Anthropic | null = null

function getVoyage(): VoyageAIClient {
  if (!voyage) {
    voyage = new VoyageAIClient({
      apiKey: process.env.VOYAGE_API_KEY
    })
  }
  return voyage
}

function getAnthropic(): Anthropic {
  if (!anthropic) {
    anthropic = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY
    })
  }
  return anthropic
}

/**
 * STEP 1: Get embeddings from Voyage AI
 *
 * Why Voyage AI for embeddings?
 * - Anthropic's official recommendation for embeddings
 * - voyage-3 model: 1024 dimensions, optimized for RAG
 * - Better retrieval quality than general-purpose embeddings
 * - Supports input_type parameter for query vs document optimization
 *
 * Learn more:
 * - https://docs.voyageai.com/
 * - https://github.com/anthropics/claude-cookbooks
 */
export async function getEmbedding(text: string, inputType: 'query' | 'document' = 'query'): Promise<number[]> {
  console.log('[RAG] STEP: Embedding single text')
  console.log(`[RAG] Text length: ${text.length} characters`)
  console.log(`[RAG] Input type: ${inputType}`)

  const response = await getVoyage().embed({
    input: text,
    model: 'voyage-3',
    inputType
  })

  console.log(`[RAG] Generated embedding with ${response.data[0].embedding.length} dimensions`)
  return response.data[0].embedding
}

/**
 * Get embeddings for multiple texts (batched for efficiency)
 */
export async function getEmbeddings(texts: string[], inputType: 'query' | 'document' = 'document'): Promise<number[][]> {
  console.log('[RAG] STEP: Embedding multiple texts in batch')
  console.log(`[RAG] Batch size: ${texts.length} chunks`)
  console.log(`[RAG] Input type: ${inputType}`)

  // Voyage AI supports batching
  const response = await getVoyage().embed({
    input: texts,
    model: 'voyage-3',
    inputType
  })

  console.log(`[RAG] Generated ${response.data.length} embeddings`)
  return response.data.map(d => d.embedding)
}

/**
 * STEP 2: Index a book
 *
 * Process:
 * 1. Chunk the text into ~500 token pieces
 * 2. Embed each chunk
 * 3. Store in vector store
 */
export async function indexBook(bookId: string, text: string): Promise<void> {
  console.log('='.repeat(60))
  console.log('[RAG] Starting book indexing process')
  console.log(`[RAG] Book ID: ${bookId}`)
  console.log(`[RAG] Text length: ${text.length} characters`)

  // Check if already indexed
  if (vectorStore.hasBook(bookId)) {
    console.log('[RAG] Book already indexed, skipping')
    console.log('='.repeat(60))
    return
  }

  // STEP 1: Chunk the text
  console.log('[RAG] STEP 1: Chunking text')
  const chunks = chunkText(text, bookId)
  console.log(`[RAG] Created ${chunks.length} chunks`)
  console.log(`[RAG] Average chunk size: ${Math.round(text.length / chunks.length)} characters`)

  // STEP 2: Get embeddings for all chunks (batched)
  console.log('[RAG] STEP 2: Generating embeddings')
  const texts = chunks.map(c => c.text)
  const embeddings = await getEmbeddings(texts, 'document') // Use 'document' type for book chunks

  // STEP 3: Create vector entries
  console.log('[RAG] STEP 3: Creating vector entries')
  const entries: VectorEntry[] = chunks.map((chunk, i) => ({
    chunk,
    embedding: embeddings[i]
  }))
  console.log(`[RAG] Created ${entries.length} vector entries`)

  // STEP 4: Store in vector store
  console.log('[RAG] STEP 4: Storing in vector store')
  await vectorStore.addBook(bookId, entries)

  console.log('[RAG] Indexing complete!')
  console.log('='.repeat(60))
}

/**
 * STEP 3: Query the book with RAG
 *
 * Process:
 * 1. Embed the question
 * 2. Find similar chunks (retrieval)
 * 3. Build prompt with context
 * 4. Generate answer with Claude
 */
export interface RAGResponse {
  answer: string
  sources: Array<{
    text: string
    fullText: string  // Complete chunk text for debugging
    score: number
    chunkIndex: number
  }>
}

export async function queryBook(
  bookId: string,
  question: string,
  selectedText?: string
): Promise<RAGResponse> {
  console.log('='.repeat(60))
  console.log('[RAG] Starting query process')
  console.log(`[RAG] Book ID: ${bookId}`)
  console.log(`[RAG] Question: "${question}"`)
  if (selectedText) {
    console.log(`[RAG] Selected text: "${selectedText.slice(0, 100)}..."`)
  }

  // STEP 1: Embed the question
  console.log('[RAG] STEP 1: Embedding question')
  const questionEmbedding = await getEmbedding(question, 'query') // Use 'query' type for questions

  // STEP 2: Find relevant chunks
  console.log('[RAG] STEP 2: Searching for relevant chunks')
  const results = await vectorStore.search(bookId, questionEmbedding, 5)

  if (results.length === 0) {
    console.log('[RAG] No relevant chunks found')
    console.log('='.repeat(60))
    return {
      answer: "I couldn't find relevant information in the book to answer this question.",
      sources: []
    }
  }

  console.log(`[RAG] Found ${results.length} relevant chunks:`)
  results.forEach((r, i) => {
    console.log(`[RAG]   ${i + 1}. Chunk ${r.chunk.metadata.chunkIndex} (similarity: ${r.score.toFixed(3)})`)
  })

  // STEP 3: Build context from retrieved chunks
  console.log('[RAG] STEP 3: Building context from chunks')
  const context = results
    .map((r, i) => `[Source ${i + 1}]\n${r.chunk.text}`)
    .join('\n\n---\n\n')
  console.log(`[RAG] Context length: ${context.length} characters`)

  // STEP 4: Build the prompt
  console.log('[RAG] STEP 4: Building prompt')
  const systemPrompt = `You are a helpful reading assistant. Answer questions about the book using ONLY the provided context. If the answer isn't in the context, say so.

Be concise but thorough. Reference specific parts of the text when relevant.`

  const userPrompt = selectedText
    ? `The user selected this text: "${selectedText}"

Context from the book:
${context}

Question: ${question}`
    : `Context from the book:
${context}

Question: ${question}`

  console.log(`[RAG] Prompt tokens (approx): ${Math.ceil(userPrompt.length / 4)}`)

  // STEP 5: Generate answer with Claude
  console.log('[RAG] STEP 5: Generating answer with Claude')
  const response = await getAnthropic().messages.create({
    model: 'claude-sonnet-4-20250514',
    max_tokens: 1024,
    system: systemPrompt,
    messages: [
      { role: 'user', content: userPrompt }
    ]
  })

  const answer = response.content[0].type === 'text'
    ? response.content[0].text
    : 'Unable to generate response'

  console.log(`[RAG] Generated answer: ${answer.slice(0, 100)}...`)
  console.log('[RAG] Query complete!')
  console.log('='.repeat(60))

  return {
    answer,
    sources: results.map(r => ({
      text: r.chunk.text.slice(0, 200) + '...',
      fullText: r.chunk.text,  // Complete chunk for debugging
      score: r.score,
      chunkIndex: r.chunk.metadata.chunkIndex
    }))
  }
}

/**
 * Simple query without RAG (just uses Claude with selected text)
 * Good for quick questions about highlighted text
 */
export async function querySimple(
  question: string,
  selectedText: string
): Promise<string> {
  const response = await getAnthropic().messages.create({
    model: 'claude-sonnet-4-20250514',
    max_tokens: 1024,
    system: 'You are a helpful reading assistant. Answer questions about the provided text concisely.',
    messages: [
      {
        role: 'user',
        content: `Text: "${selectedText}"\n\nQuestion: ${question}`
      }
    ]
  })

  return response.content[0].type === 'text'
    ? response.content[0].text
    : 'Unable to generate response'
}
