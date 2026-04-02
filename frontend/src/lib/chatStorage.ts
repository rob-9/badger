// Chat thread storage — thread metadata + messages in IndexedDB

import { openDB } from './bookStorage'

const THREADS_STORE = 'threads'
const THREAD_MESSAGES_STORE = 'thread-messages'
const MAX_THREADS_PER_BOOK = 50

export interface ThreadMeta {
  id: string
  bookId: string
  title: string
  createdAt: number
  updatedAt: number
  messageCount: number
}

export interface ThreadMessage {
  id: string
  threadId: string
  role: 'user' | 'assistant'
  content: string
  context?: string
  readerPosition?: number
  sources?: Array<{ text: string; full_text: string; score: number; chunk_index: number; source_number: number; label: string; chapter_title?: string }>
  createdAt: number
}

export async function getThreadsForBook(bookId: string): Promise<ThreadMeta[]> {
  if (typeof window === 'undefined') return []
  const db = await openDB()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(THREADS_STORE, 'readonly')
    const store = tx.objectStore(THREADS_STORE)
    const index = store.index('bookId')
    const request = index.getAll(bookId)
    request.onerror = () => reject(request.error)
    request.onsuccess = () => {
      const threads: ThreadMeta[] = request.result || []
      threads.sort((a, b) => b.updatedAt - a.updatedAt)
      resolve(threads)
    }
  })
}

export async function getThreadMessages(threadId: string): Promise<ThreadMessage[]> {
  if (typeof window === 'undefined') return []
  const db = await openDB()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(THREAD_MESSAGES_STORE, 'readonly')
    const store = tx.objectStore(THREAD_MESSAGES_STORE)
    const index = store.index('threadId')
    const request = index.getAll(threadId)
    request.onerror = () => reject(request.error)
    request.onsuccess = () => {
      const messages: ThreadMessage[] = request.result || []
      messages.sort((a, b) => a.createdAt - b.createdAt)
      resolve(messages)
    }
  })
}

export async function createThread(bookId: string, title: string): Promise<string> {
  const db = await openDB()
  const id = `thread-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`
  const now = Date.now()

  const thread: ThreadMeta = {
    id,
    bookId,
    title: title.slice(0, 100),
    createdAt: now,
    updatedAt: now,
    messageCount: 0,
  }

  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(THREADS_STORE, 'readwrite')
    tx.onerror = () => reject(tx.error)
    tx.oncomplete = () => resolve()
    tx.objectStore(THREADS_STORE).put(thread)
  })

  // Enforce thread cap — delete oldest if over limit
  const allThreads = await getThreadsForBook(bookId)
  if (allThreads.length > MAX_THREADS_PER_BOOK) {
    const toDelete = allThreads.slice(MAX_THREADS_PER_BOOK)
    for (const old of toDelete) {
      await deleteThread(old.id)
    }
  }

  return id
}

export async function saveMessage(
  threadId: string,
  message: ThreadMessage,
): Promise<void> {
  const db = await openDB()

  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction([THREADS_STORE, THREAD_MESSAGES_STORE], 'readwrite')
    tx.onerror = () => reject(tx.error)
    tx.oncomplete = () => resolve()

    // Save the message
    tx.objectStore(THREAD_MESSAGES_STORE).put(message)

    // Strip full_text from previous assistant messages' sources to save space
    if (message.role === 'assistant') {
      const msgStore = tx.objectStore(THREAD_MESSAGES_STORE)
      const index = msgStore.index('threadId')
      const req = index.openCursor(threadId)
      req.onsuccess = () => {
        const cursor = req.result
        if (!cursor) return
        const existing = cursor.value as ThreadMessage
        if (existing.id !== message.id && existing.role === 'assistant' && existing.sources) {
          const stripped = {
            ...existing,
            sources: existing.sources.map(s => ({
              ...s,
              full_text: '',
              score: 0,
            })),
          }
          cursor.update(stripped)
        }
        cursor.continue()
      }
    }

    // Update thread metadata
    const threadStore = tx.objectStore(THREADS_STORE)
    const getReq = threadStore.get(threadId)
    getReq.onsuccess = () => {
      const thread = getReq.result as ThreadMeta | undefined
      if (thread) {
        thread.updatedAt = Date.now()
        thread.messageCount += 1
        threadStore.put(thread)
      }
    }
  })
}

export async function deleteThread(threadId: string): Promise<void> {
  const db = await openDB()
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction([THREADS_STORE, THREAD_MESSAGES_STORE], 'readwrite')
    tx.onerror = () => reject(tx.error)
    tx.oncomplete = () => resolve()

    // Delete thread metadata
    tx.objectStore(THREADS_STORE).delete(threadId)

    // Delete all messages for this thread
    const msgStore = tx.objectStore(THREAD_MESSAGES_STORE)
    const index = msgStore.index('threadId')
    const req = index.openCursor(threadId)
    req.onsuccess = () => {
      const cursor = req.result
      if (!cursor) return
      cursor.delete()
      cursor.continue()
    }
  })
}

export async function deleteThreadsForBook(bookId: string): Promise<void> {
  const threads = await getThreadsForBook(bookId)
  for (const thread of threads) {
    await deleteThread(thread.id)
  }
}
