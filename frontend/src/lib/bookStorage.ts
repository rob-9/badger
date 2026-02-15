// Book history storage using IndexedDB for files and localStorage for metadata

export interface BookMetadata {
  id: string
  fileName: string
  addedAt: number
  lastReadAt: number
  lastPosition?: string // CFI location
  coverUrl?: string // data URL of cover image
}

const DB_NAME = 'boom-books'
const DB_VERSION = 1
const STORE_NAME = 'epubs'
const METADATA_KEY = 'book-history'

// IndexedDB helpers
function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION)

    request.onerror = () => reject(request.error)
    request.onsuccess = () => resolve(request.result)

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'id' })
      }
    }
  })
}

// Generate a simple unique ID
function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
}

// Get all book metadata from localStorage
export function getBookHistory(): BookMetadata[] {
  if (typeof window === 'undefined') return []
  const data = localStorage.getItem(METADATA_KEY)
  if (!data) return []
  try {
    return JSON.parse(data)
  } catch {
    return []
  }
}

// Save book metadata to localStorage
function saveBookHistory(history: BookMetadata[]): void {
  localStorage.setItem(METADATA_KEY, JSON.stringify(history))
}

// Add a new book to history
export async function addBook(fileName: string, data: ArrayBuffer, coverUrl?: string): Promise<string> {
  const id = generateId()

  // Store file in IndexedDB
  const db = await openDB()
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite')
    const store = tx.objectStore(STORE_NAME)
    const request = store.put({ id, data })
    request.onerror = () => reject(request.error)
    request.onsuccess = () => resolve()
  })

  // Store metadata in localStorage
  const history = getBookHistory()
  const metadata: BookMetadata = {
    id,
    fileName,
    addedAt: Date.now(),
    lastReadAt: Date.now(),
    coverUrl,
  }
  history.unshift(metadata) // Add to beginning
  saveBookHistory(history)

  return id
}

// Get book file from IndexedDB
export async function getBookData(id: string): Promise<ArrayBuffer | null> {
  const db = await openDB()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly')
    const store = tx.objectStore(STORE_NAME)
    const request = store.get(id)
    request.onerror = () => reject(request.error)
    request.onsuccess = () => {
      resolve(request.result?.data || null)
    }
  })
}

// Update last read time and position
export function updateReadProgress(id: string, position?: string): void {
  const history = getBookHistory()
  const book = history.find((b) => b.id === id)
  if (book) {
    book.lastReadAt = Date.now()
    if (position) {
      book.lastPosition = position
    }
    saveBookHistory(history)
  }
}

// Remove a book from history
export async function removeBook(id: string): Promise<void> {
  // Remove from IndexedDB
  const db = await openDB()
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite')
    const store = tx.objectStore(STORE_NAME)
    const request = store.delete(id)
    request.onerror = () => reject(request.error)
    request.onsuccess = () => resolve()
  })

  // Remove from localStorage
  const history = getBookHistory().filter((b) => b.id !== id)
  saveBookHistory(history)
}
