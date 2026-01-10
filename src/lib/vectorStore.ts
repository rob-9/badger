/**
 * VECTOR STORE WITH DUAL PERSISTENCE
 *
 * What is a vector store?
 * - A database optimized for storing and searching vectors (embeddings)
 * - Uses similarity metrics (cosine, dot product) to find "close" vectors
 *
 * This implementation:
 * - In-memory for fast queries
 * - File system persistence (server-side, survives restarts)
 * - IndexedDB persistence (client-side)
 *
 * Production alternatives: Pinecone, Chroma, Weaviate, pgvector
 *
 * Learn more:
 * - https://www.pinecone.io/learn/vector-database/
 * - https://docs.trychroma.com/
 */

import type { TextChunk } from './chunker'
import * as fs from 'fs'
import * as path from 'path'

export interface VectorEntry {
  chunk: TextChunk
  embedding: number[]
}

export interface SearchResult {
  chunk: TextChunk
  score: number  // Cosine similarity: 1 = identical, 0 = orthogonal, -1 = opposite
}

// IndexedDB config (client-side)
const DB_NAME = 'boom-vectors'
const DB_VERSION = 1
const STORE_NAME = 'embeddings'

// File system config (server-side)
const VECTORS_DIR = '.next/cache/vectors'

/**
 * COSINE SIMILARITY
 *
 * Measures the angle between two vectors (ignores magnitude).
 * Perfect for comparing embeddings because we care about direction, not length.
 *
 * Formula: cos(θ) = (A · B) / (||A|| × ||B||)
 *
 * Where:
 * - A · B = dot product (sum of element-wise multiplication)
 * - ||A|| = magnitude (sqrt of sum of squares)
 *
 * Returns: -1 to 1 (1 = most similar)
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error('Vectors must have same length')
  }

  let dotProduct = 0
  let magnitudeA = 0
  let magnitudeB = 0

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i]
    magnitudeA += a[i] * a[i]
    magnitudeB += b[i] * b[i]
  }

  magnitudeA = Math.sqrt(magnitudeA)
  magnitudeB = Math.sqrt(magnitudeB)

  if (magnitudeA === 0 || magnitudeB === 0) {
    return 0
  }

  return dotProduct / (magnitudeA * magnitudeB)
}

/**
 * Environment detection
 */
function isClient(): boolean {
  return typeof window !== 'undefined' && typeof indexedDB !== 'undefined'
}

function isServer(): boolean {
  return typeof window === 'undefined'
}

/**
 * File system helpers for server-side persistence
 */
function ensureVectorsDir(): void {
  if (!isServer()) return

  try {
    if (!fs.existsSync(VECTORS_DIR)) {
      fs.mkdirSync(VECTORS_DIR, { recursive: true })
      console.log('[VectorStore] Created vectors cache directory')
    }
  } catch (error) {
    console.warn('[VectorStore] Failed to create vectors directory:', error)
  }
}

function getVectorFilePath(bookId: string): string {
  return path.join(VECTORS_DIR, `${bookId}.json`)
}

async function saveToFile(bookId: string, entries: VectorEntry[]): Promise<void> {
  if (!isServer()) return

  try {
    ensureVectorsDir()
    const filePath = getVectorFilePath(bookId)

    // Convert to JSON-serializable format
    const data = {
      bookId,
      timestamp: Date.now(),
      entryCount: entries.length,
      entries: entries
    }

    fs.writeFileSync(filePath, JSON.stringify(data), 'utf-8')
    console.log(`[VectorStore] Saved ${entries.length} entries to disk: ${bookId}`)
  } catch (error) {
    console.warn('[VectorStore] Failed to save to file:', error)
  }
}

async function loadFromFile(bookId: string): Promise<VectorEntry[] | null> {
  if (!isServer()) return null

  try {
    const filePath = getVectorFilePath(bookId)

    if (!fs.existsSync(filePath)) {
      return null
    }

    const fileContent = fs.readFileSync(filePath, 'utf-8')
    const data = JSON.parse(fileContent)

    console.log(`[VectorStore] Loaded ${data.entryCount} entries from disk: ${bookId}`)
    return data.entries
  } catch (error) {
    console.warn('[VectorStore] Failed to load from file:', error)
    return null
  }
}

async function deleteFromFile(bookId: string): Promise<void> {
  if (!isServer()) return

  try {
    const filePath = getVectorFilePath(bookId)
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath)
      console.log(`[VectorStore] Deleted from disk: ${bookId}`)
    }
  } catch (error) {
    console.warn('[VectorStore] Failed to delete from file:', error)
  }
}

/**
 * IndexedDB helpers for client-side persistence
 */

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION)

    request.onerror = () => reject(request.error)
    request.onsuccess = () => resolve(request.result)

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'bookId' })
      }
    }
  })
}

/**
 * Vector store with optional IndexedDB persistence
 *
 * For production, replace with:
 * - Pinecone (hosted, scalable)
 * - Chroma (local, Python-based)
 * - Qdrant (local or hosted)
 */
export class VectorStore {
  private entries: Map<string, VectorEntry[]> = new Map()
  private initialized: boolean = false

  /**
   * Initialize store and load from IndexedDB if available
   */
  async init(): Promise<void> {
    if (this.initialized) return

    if (isClient()) {
      try {
        const db = await openDB()
        const tx = db.transaction(STORE_NAME, 'readonly')
        const store = tx.objectStore(STORE_NAME)

        const allBooks = await new Promise<any[]>((resolve, reject) => {
          const request = store.getAll()
          request.onerror = () => reject(request.error)
          request.onsuccess = () => resolve(request.result)
        })

        for (const book of allBooks) {
          this.entries.set(book.bookId, book.entries)
          console.log(`[VectorStore] Loaded ${book.entries.length} chunks for: ${book.bookId}`)
        }

        console.log(`[VectorStore] Initialized from IndexedDB: ${allBooks.length} books`)
      } catch (error) {
        console.warn('[VectorStore] Failed to load from IndexedDB:', error)
      }
    }

    this.initialized = true
  }

  /**
   * Save a book's embeddings to IndexedDB
   */
  private async persistBook(bookId: string, entries: VectorEntry[]): Promise<void> {
    if (!isClient()) return

    try {
      const db = await openDB()
      const tx = db.transaction(STORE_NAME, 'readwrite')
      const store = tx.objectStore(STORE_NAME)

      await new Promise<void>((resolve, reject) => {
        const request = store.put({ bookId, entries })
        request.onerror = () => reject(request.error)
        request.onsuccess = () => resolve()
      })

      console.log(`[VectorStore] Persisted ${entries.length} chunks for: ${bookId}`)
    } catch (error) {
      console.warn('[VectorStore] Failed to persist to IndexedDB:', error)
    }
  }

  /**
   * Remove a book from IndexedDB
   */
  private async unpersistBook(bookId: string): Promise<void> {
    if (!isClient()) return

    try {
      const db = await openDB()
      const tx = db.transaction(STORE_NAME, 'readwrite')
      const store = tx.objectStore(STORE_NAME)

      await new Promise<void>((resolve, reject) => {
        const request = store.delete(bookId)
        request.onerror = () => reject(request.error)
        request.onsuccess = () => resolve()
      })
    } catch (error) {
      console.warn('[VectorStore] Failed to delete from IndexedDB:', error)
    }
  }

  /**
   * Add chunks with their embeddings for a specific book
   */
  async addBook(bookId: string, entries: VectorEntry[]): Promise<void> {
    this.entries.set(bookId, entries)
    console.log(`[VectorStore] Added ${entries.length} chunks to in-memory store`)
    console.log(`[VectorStore] Book ID: ${bookId}`)

    // Persist to file system (server-side)
    if (isServer()) {
      console.log('[VectorStore] Persisting to file system')
      await saveToFile(bookId, entries)
    }

    // Persist to IndexedDB (client-side)
    if (isClient()) {
      console.log('[VectorStore] Persisting to IndexedDB')
      await this.persistBook(bookId, entries)
    }
  }

  /**
   * Search for similar chunks across a book
   *
   * @param bookId - The book to search in
   * @param queryEmbedding - The embedded question
   * @param topK - Number of results to return
   */
  async search(bookId: string, queryEmbedding: number[], topK: number = 5): Promise<SearchResult[]> {
    console.log(`[VectorStore] Searching book: ${bookId}`)
    console.log(`[VectorStore] Query embedding dimensions: ${queryEmbedding.length}`)
    console.log(`[VectorStore] Returning top ${topK} results`)

    // Try to get from memory first
    let bookEntries = this.entries.get(bookId)

    // If not in memory and on server, try loading from file
    if (!bookEntries && isServer()) {
      console.log('[VectorStore] Not in memory, loading from disk')
      bookEntries = await loadFromFile(bookId)

      if (bookEntries) {
        // Cache in memory for future queries
        this.entries.set(bookId, bookEntries)
        console.log('[VectorStore] Loaded from disk and cached in memory')
      }
    }

    if (!bookEntries || bookEntries.length === 0) {
      console.warn(`[VectorStore] No entries found for book: ${bookId}`)
      return []
    }

    console.log(`[VectorStore] Comparing against ${bookEntries.length} chunks`)

    // Calculate similarity for each chunk
    const results: SearchResult[] = bookEntries.map(entry => ({
      chunk: entry.chunk,
      score: cosineSimilarity(queryEmbedding, entry.embedding)
    }))

    // Sort by similarity (highest first) and take top K
    const topResults = results
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)

    console.log(`[VectorStore] Best match similarity: ${topResults[0]?.score.toFixed(3)}`)
    return topResults
  }

  /**
   * Check if a book has been indexed
   */
  hasBook(bookId: string): boolean {
    return this.entries.has(bookId)
  }

  /**
   * Remove a book from the store
   */
  async removeBook(bookId: string): Promise<void> {
    this.entries.delete(bookId)

    // Delete from file system (server-side)
    if (isServer()) {
      await deleteFromFile(bookId)
    }

    // Delete from IndexedDB (client-side)
    await this.unpersistBook(bookId)
  }

  /**
   * Get stats about the store
   */
  getStats(): { bookCount: number; totalChunks: number } {
    let totalChunks = 0
    for (const entries of this.entries.values()) {
      totalChunks += entries.length
    }
    return {
      bookCount: this.entries.size,
      totalChunks
    }
  }

  /**
   * Get all entries for a book (useful for sending to server)
   */
  getBookEntries(bookId: string): VectorEntry[] | undefined {
    return this.entries.get(bookId)
  }
}

// Singleton instance for the application
export const vectorStore = new VectorStore()
