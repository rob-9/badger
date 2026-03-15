// Book storage — all data in IndexedDB (metadata + file blobs)

export interface BookMetadata {
  id: string
  fileName: string
  addedAt: number
  lastReadAt: number
  lastPosition?: string // CFI location
  coverUrl?: string // data URL of cover image
}

const DB_NAME = 'badger-books'
const DB_VERSION = 2 // Bumped to add metadata store
const FILES_STORE = 'epubs'
const META_STORE = 'metadata'
const LEGACY_KEY = 'book-history' // Old localStorage key for migration

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION)

    request.onerror = () => reject(request.error)
    request.onsuccess = () => resolve(request.result)

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result

      if (!db.objectStoreNames.contains(FILES_STORE)) {
        db.createObjectStore(FILES_STORE, { keyPath: 'id' })
      }

      if (!db.objectStoreNames.contains(META_STORE)) {
        const metaStore = db.createObjectStore(META_STORE, { keyPath: 'id' })
        metaStore.createIndex('lastReadAt', 'lastReadAt', { unique: false })

        // Migrate from localStorage if data exists
        const legacyData = localStorage.getItem(LEGACY_KEY)
        if (legacyData) {
          try {
            const books: BookMetadata[] = JSON.parse(legacyData)
            const tx = (event.target as IDBOpenDBRequest).transaction!
            const store = tx.objectStore(META_STORE)
            for (const book of books) {
              store.put(book)
            }
            localStorage.removeItem(LEGACY_KEY)
          } catch {
            // Migration failed — stale data will be lost
          }
        }
      }
    }
  })
}

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
}

// Get all book metadata, sorted by lastReadAt descending
export async function getBookHistory(): Promise<BookMetadata[]> {
  if (typeof window === 'undefined') return []
  const db = await openDB()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(META_STORE, 'readonly')
    const store = tx.objectStore(META_STORE)
    const request = store.getAll()
    request.onerror = () => reject(request.error)
    request.onsuccess = () => {
      const books: BookMetadata[] = request.result || []
      books.sort((a, b) => b.lastReadAt - a.lastReadAt)
      resolve(books)
    }
  })
}

// Add a new book
export async function addBook(fileName: string, data: ArrayBuffer, coverUrl?: string): Promise<string> {
  const id = generateId()
  const db = await openDB()

  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction([FILES_STORE, META_STORE], 'readwrite')
    tx.onerror = () => reject(tx.error)
    tx.oncomplete = () => resolve()

    tx.objectStore(FILES_STORE).put({ id, data })
    tx.objectStore(META_STORE).put({
      id,
      fileName,
      addedAt: Date.now(),
      lastReadAt: Date.now(),
      coverUrl,
    } satisfies BookMetadata)
  })

  return id
}

// Get book file data
export async function getBookData(id: string): Promise<ArrayBuffer | null> {
  const db = await openDB()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(FILES_STORE, 'readonly')
    const request = tx.objectStore(FILES_STORE).get(id)
    request.onerror = () => reject(request.error)
    request.onsuccess = () => resolve(request.result?.data || null)
  })
}

// Update last read time and position
export async function updateReadProgress(id: string, position?: string): Promise<void> {
  const db = await openDB()
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(META_STORE, 'readwrite')
    const store = tx.objectStore(META_STORE)
    const request = store.get(id)
    request.onerror = () => reject(request.error)
    request.onsuccess = () => {
      const book = request.result as BookMetadata | undefined
      if (book) {
        book.lastReadAt = Date.now()
        if (position) book.lastPosition = position
        store.put(book)
      }
    }
    tx.oncomplete = () => resolve()
  })
}

// Remove a book completely
export async function removeBook(id: string): Promise<void> {
  const db = await openDB()
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction([FILES_STORE, META_STORE], 'readwrite')
    tx.onerror = () => reject(tx.error)
    tx.oncomplete = () => resolve()

    tx.objectStore(FILES_STORE).delete(id)
    tx.objectStore(META_STORE).delete(id)
  })
}
