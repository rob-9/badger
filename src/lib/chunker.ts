/**
 * TEXT CHUNKING UTILITY
 *
 * Why chunking matters:
 * - Embeddings work best on coherent pieces of text (not too long, not too short)
 * - Typical chunk size: 500-1000 tokens (~2000-4000 characters)
 * - Overlap helps preserve context at chunk boundaries
 *
 * Learn more:
 * - https://www.pinecone.io/learn/chunking-strategies/
 * - https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/
 */

export interface TextChunk {
  id: string
  text: string
  metadata: {
    bookId: string
    chunkIndex: number
    startChar: number
    endChar: number
  }
}

interface ChunkOptions {
  chunkSize?: number      // Target size in characters
  chunkOverlap?: number   // Overlap between chunks to preserve context
}

const DEFAULT_CHUNK_SIZE = 2000      // ~500 tokens
const DEFAULT_OVERLAP = 200          // ~50 tokens overlap

/**
 * Split text into overlapping chunks
 *
 * Example with chunkSize=100, overlap=20:
 *
 * Text: "The quick brown fox jumps over the lazy dog. The dog was sleeping..."
 *
 * Chunk 1: "The quick brown fox jumps over the lazy dog. The dog"  [0-100]
 * Chunk 2: "dog. The dog was sleeping peacefully in the sun..."     [80-180]
 *           ↑ overlap preserves context
 */
export function chunkText(
  text: string,
  bookId: string,
  options: ChunkOptions = {}
): TextChunk[] {
  const {
    chunkSize = DEFAULT_CHUNK_SIZE,
    chunkOverlap = DEFAULT_OVERLAP
  } = options

  const chunks: TextChunk[] = []

  // Clean the text
  const cleanedText = text
    .replace(/\s+/g, ' ')  // Normalize whitespace
    .trim()

  let startIndex = 0
  let chunkIndex = 0

  while (startIndex < cleanedText.length) {
    // Calculate end position
    let endIndex = Math.min(startIndex + chunkSize, cleanedText.length)

    // Try to break at a sentence boundary (. ! ? or newline)
    if (endIndex < cleanedText.length) {
      const searchStart = Math.max(startIndex + chunkSize - 200, startIndex)
      const searchText = cleanedText.slice(searchStart, endIndex)

      // Find last sentence boundary
      const sentenceEnd = Math.max(
        searchText.lastIndexOf('. '),
        searchText.lastIndexOf('! '),
        searchText.lastIndexOf('? '),
        searchText.lastIndexOf('\n')
      )

      if (sentenceEnd > 0) {
        endIndex = searchStart + sentenceEnd + 1
      }
    }

    // Extract chunk text
    const chunkText = cleanedText.slice(startIndex, endIndex).trim()

    if (chunkText.length > 0) {
      chunks.push({
        id: `${bookId}-chunk-${chunkIndex}`,
        text: chunkText,
        metadata: {
          bookId,
          chunkIndex,
          startChar: startIndex,
          endChar: endIndex
        }
      })
      chunkIndex++
    }

    // Move start position (accounting for overlap)
    startIndex = endIndex - chunkOverlap

    // Prevent infinite loop
    if (startIndex >= cleanedText.length - chunkOverlap) {
      break
    }
  }

  return chunks
}

/**
 * Estimate token count (rough approximation)
 * Rule of thumb: 1 token ≈ 4 characters for English
 */
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4)
}
