'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import DocumentViewer from '@/components/DocumentViewer'
import EpubReader, { type TextSelection, type EpubReaderHandle } from '@/components/EpubReader'
import Sidebar from '@/components/Sidebar'
import type { ViewType } from '@/components/Sidebar'
import LibraryView from '@/components/LibraryView'
import QuestionPopup from '@/components/QuestionPopup'
import ChatPanel, { type ChatMessage } from '@/components/ChatPanel'
import Toast from '@/components/Toast'
import { addBook, getBookData, getBookHistory, removeBook, type BookMetadata } from '@/lib/bookStorage'
import { indexBook, isBookIndexed, queryBookStream } from '@/lib/api'
import { extractCover, extractStructuredText } from '@/lib/parseEpub'
import { createThread, saveMessage, getThreadsForBook, getThreadMessages, deleteThreadsForBook, type ThreadMeta, type ThreadMessage } from '@/lib/chatStorage'

const STATUS_LABELS: Record<string, string> = {
  thinking: 'Thinking...',
  searching: 'Searching the book...',
  generating: 'Writing answer...',
  // Legacy graph pipeline stages (fallback compatibility)
  classifying: 'Classifying question...',
  retrieving: 'Retrieving context...',
  reranking: 'Reranking results...',
  filtering: 'Filtering results...',
  sanitizing: 'Checking for spoilers...',
}

export default function Home() {
  const [document, setDocument] = useState<string | null>(null)
  const [fileName, setFileName] = useState<string>('')
  const [epubData, setEpubData] = useState<ArrayBuffer | null>(null)
  const [isEpub, setIsEpub] = useState(false)
  const [history, setHistory] = useState<BookMetadata[]>([])
  const [historyLoaded, setHistoryLoaded] = useState(false)
  const [bookId, setBookId] = useState<string | null>(null)
  const [isIndexing, setIsIndexing] = useState(false)
  const [isIndexed, setIsIndexed] = useState(false)
  const [readerPosition, setReaderPosition] = useState<number>(0)

  // View state
  const [currentView, setCurrentView] = useState<ViewType>('recent')
  const uploadInputRef = useRef<HTMLInputElement>(null)

  // Chat state
  const [selection, setSelection] = useState<TextSelection | null>(null)
  const [isClosingPopup, setIsClosingPopup] = useState(false)
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
  const [isChatOpen, setIsChatOpen] = useState(false)
  const [isChatLoading, setIsChatLoading] = useState(false)
  const [loadingStatus, setLoadingStatus] = useState('')
  const streamAbortRef = useRef<(() => void) | null>(null)
  const streamSourcesRef = useRef<ChatMessage['sources']>(undefined)
  const epubReaderRef = useRef<EpubReaderHandle>(null)

  // Thread state
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null)
  const [threads, setThreads] = useState<ThreadMeta[]>([])
  const activeThreadIdRef = useRef<string | null>(null)
  const [savedReadingCfi, setSavedReadingCfi] = useState<string | null>(null)

  // Loading transition state
  const [isLoadingBook, setIsLoadingBook] = useState(false)
  const [loadingBook, setLoadingBook] = useState<BookMetadata | null>(null)

  // Toast state
  const [toast, setToast] = useState<{ message: string; type: 'info' | 'success' | 'error' } | null>(null)

  // Keep thread ID ref in sync
  useEffect(() => {
    activeThreadIdRef.current = activeThreadId
  }, [activeThreadId])

  const buildConversationHistory = useCallback(() => {
    return chatMessages
      .filter(m => m.content)
      .map(m => ({
        role: m.role as 'user' | 'assistant',
        content: m.content.length > 1500 ? m.content.slice(0, 1500) + '...' : m.content,
        selected_text: m.context,
        reader_position: m.readerPosition,
      }))
      .slice(-6) // last 3 exchanges
  }, [chatMessages])

  // Load history on mount + restore last open book
  useEffect(() => {
    getBookHistory().then(async (books) => {
      setHistory(books)
      setHistoryLoaded(true)

      // Restore last open book on reload
      const lastBookId = localStorage.getItem('badger-active-book')
      if (lastBookId) {
        const book = books.find(b => b.id === lastBookId)
        if (book) {
          const data = await getBookData(book.id)
          if (data) {
            setEpubData(data)
            setFileName(book.fileName)
            setIsEpub(true)
            setBookId(book.id)
            setDocument('loaded')

            await indexBookIfNeeded(book.id, data)
            setThreads(await getThreadsForBook(book.id))
          }
        }
      }
    })
  }, [])

  const indexBookIfNeeded = async (bookId: string, data: ArrayBuffer) => {
    const indexed = await isBookIndexed(bookId)
    if (indexed) {
      setIsIndexed(true)
      return
    }
    setIsIndexing(true)
    try {
      const structured = await extractStructuredText(data)
      await indexBook(bookId, structured)
      setIsIndexed(true)
    } catch (error) {
      console.error('[App] Failed to index book:', error)
    } finally {
      setIsIndexing(false)
    }
  }

  const handleFileLoad = async (content: string, name: string, arrayBuffer?: ArrayBuffer) => {
    setDocument(content)
    setFileName(name)
    const lowerName = name.toLowerCase()
    const isEpubFile = lowerName.endsWith('.epub') || lowerName.endsWith('.epub.zip')
    setIsEpub(isEpubFile)

    // For EPUB files, store the ArrayBuffer for the reader and save to history
    if (isEpubFile && arrayBuffer) {
      setEpubData(arrayBuffer)
      const coverUrl = await extractCover(arrayBuffer) ?? undefined
      const id = await addBook(name, arrayBuffer, coverUrl)
      setBookId(id)
      localStorage.setItem('badger-active-book', id)
      setHistory(await getBookHistory()) // Refresh history

      // Index the book for RAG
      console.log('[App] Starting RAG indexing for book')
      try {
        await indexBookIfNeeded(id, arrayBuffer)
        console.log('[App] Book indexed successfully')
      } catch (error) {
        setToast({ message: 'AI features unavailable. You can still read.', type: 'error' })
      }
      setThreads([])
    }
  }

  const handleOpenFromHistory = useCallback(async (book: BookMetadata) => {
    // Reset chat/thread state from any previous book
    setActiveThreadId(null)
    setChatMessages([])
    setIsChatOpen(false)

    // Show loading screen immediately
    setLoadingBook(book)
    setIsLoadingBook(true)

    const data = await getBookData(book.id)
    if (data) {
      setEpubData(data)
      setFileName(book.fileName)
      setIsEpub(true)
      setBookId(book.id)
      localStorage.setItem('badger-active-book', book.id)

      // Keep loading screen up, then mount the reader behind it
      await new Promise(r => setTimeout(r, 300))
      setDocument('loaded')
      // Let the reader mount and initialize behind the overlay
      await new Promise(r => setTimeout(r, 200))
      setIsLoadingBook(false)
      setLoadingBook(null)

      await indexBookIfNeeded(book.id, data)
      setThreads(await getThreadsForBook(book.id))
    } else {
      setIsLoadingBook(false)
      setLoadingBook(null)
    }
  }, [])

  const handleDeleteBook = useCallback(async (targetBookId: string) => {
    await removeBook(targetBookId)
    await deleteThreadsForBook(targetBookId)
    setHistory(await getBookHistory())
  }, [])

  const handleBack = async () => {
    setDocument(null)
    setFileName('')
    setEpubData(null)
    setIsEpub(false)
    localStorage.removeItem('badger-active-book')
    setBookId(null)
    setIsIndexing(false)
    setIsIndexed(false)
    setHistory(await getBookHistory()) // Refresh history
    // Reset chat state
    setSelection(null)
    setChatMessages([])
    setIsChatOpen(false)
    setActiveThreadId(null)
    setThreads([])
  }

  const handleTextSelect = useCallback((sel: TextSelection) => {
    // Only set selection if there's actual text selected
    if (sel.text && sel.text.trim()) {
      setSelection(sel)
      setIsClosingPopup(false)
    } else {
      // Trigger graceful close animation before clearing selection
      if (selection) {
        setIsClosingPopup(true)
        setTimeout(() => {
          setSelection(null)
          setIsClosingPopup(false)
        }, 150) // Match animation duration
      }
    }
  }, [selection])

  const startStreaming = useCallback((
    params: Parameters<typeof queryBookStream>[0],
    assistantId: string,
  ) => {
    setIsChatLoading(true)
    setLoadingStatus('Thinking...')
    streamSourcesRef.current = undefined

    const handle = queryBookStream(params, {
      onStatus: (stage) => {
        setLoadingStatus(STATUS_LABELS[stage] || stage)
      },
      onSources: (sources) => {
        streamSourcesRef.current = sources
        setChatMessages(prev => {
          const last = prev[prev.length - 1]
          if (last?.id === assistantId) {
            return [...prev.slice(0, -1), { ...last, sources }]
          }
          return prev
        })
      },
      onToken: (text) => {
        setLoadingStatus('')  // Clear pipeline status once generation starts
        setChatMessages(prev => {
          const last = prev[prev.length - 1]
          if (last?.id === assistantId) {
            return [...prev.slice(0, -1), { ...last, content: last.content + text }]
          }
          // First token — create message with sources from ref (onSources fires before tokens)
          return [...prev, {
            id: assistantId,
            role: 'assistant' as const,
            content: text,
            sources: streamSourcesRef.current,
          }]
        })
      },
      onDone: () => {
        // Filter sources to only those actually cited in the answer
        const allSources = streamSourcesRef.current
        if (allSources?.length) {
          setChatMessages(prev => {
            const last = prev[prev.length - 1]
            if (last?.id === assistantId) {
              const cited = new Set<number>()
              const citePattern = /\[(?:Source\s+)?\d+(?:,\s*(?:Source\s+)?\d+)*\]/g
              let m: RegExpExecArray | null
              while ((m = citePattern.exec(last.content)) !== null) {
                const nums = m[0].match(/\d+/g)
                if (nums) nums.forEach(n => cited.add(parseInt(n)))
              }
              const sources = cited.size > 0
                ? allSources.filter(s => cited.has(s.source_number))
                : allSources
              return [...prev.slice(0, -1), { ...last, sources }]
            }
            return prev
          })
        }
        setIsChatLoading(false)
        setLoadingStatus('')
        streamAbortRef.current = null

        // Save assistant message to IndexedDB
        const tid = activeThreadIdRef.current
        if (tid) {
          setChatMessages(prev => {
            const lastMsg = prev[prev.length - 1]
            if (lastMsg?.id === assistantId && lastMsg.role === 'assistant') {
              saveMessage(tid, {
                id: lastMsg.id,
                threadId: tid,
                role: 'assistant',
                content: lastMsg.content,
                sources: lastMsg.sources,
                createdAt: Date.now(),
              }).catch(e => console.error('[App] Failed to save assistant message:', e))
            }
            return prev // don't modify
          })
        }
      },
      onError: (error) => {
        console.error('[App] Stream error:', error)
        setChatMessages(prev => {
          const last = prev[prev.length - 1]
          if (last?.id === assistantId && last.content) return prev
          return [...prev, {
            id: assistantId,
            role: 'assistant' as const,
            content: 'Sorry, I encountered an error processing your question. Please try again.',
          }]
        })
        setIsChatLoading(false)
        setLoadingStatus('')
        streamAbortRef.current = null
      },
    })
    streamAbortRef.current = handle.abort
  }, [])

  const handleQuestionSubmit = useCallback(async (question: string, context: string) => {
    setSelection(null)
    setIsChatOpen(true)
    streamAbortRef.current?.()

    // Create thread if needed
    let threadId = activeThreadId
    if (!threadId && bookId) {
      threadId = await createThread(bookId, question)
      setActiveThreadId(threadId)
      setThreads(bookId ? await getThreadsForBook(bookId) : [])
    }
    const currentThreadId = threadId

    const conversationHistory = buildConversationHistory()
    const userMessage: ChatMessage = { id: crypto.randomUUID(), role: 'user', content: question, context, readerPosition }
    const assistantId = crypto.randomUUID()
    setChatMessages(prev => [...prev, userMessage])

    // Save user message to IndexedDB
    if (currentThreadId) {
      saveMessage(currentThreadId, {
        id: userMessage.id,
        threadId: currentThreadId,
        role: 'user',
        content: question,
        context,
        readerPosition,
        createdAt: Date.now(),
      }).catch(e => console.error('[App] Failed to save user message:', e))
    }

    startStreaming({
      bookId: bookId || undefined,
      question,
      selectedText: context,
      useRag: !!bookId,
      readerPosition,
      conversationHistory: conversationHistory.length > 0 ? conversationHistory : undefined,
    }, assistantId)
  }, [bookId, readerPosition, startStreaming, activeThreadId, buildConversationHistory])

  const handleChatMessage = useCallback(async (message: string) => {
    streamAbortRef.current?.()

    // Create thread if needed
    let threadId = activeThreadId
    if (!threadId && bookId) {
      threadId = await createThread(bookId, message)
      setActiveThreadId(threadId)
      setThreads(bookId ? await getThreadsForBook(bookId) : [])
    }
    const currentThreadId = threadId

    const conversationHistory = buildConversationHistory()
    const userMessage: ChatMessage = { id: crypto.randomUUID(), role: 'user', content: message, readerPosition }
    const assistantId = crypto.randomUUID()
    setChatMessages(prev => [...prev, userMessage])

    // Save user message to IndexedDB
    if (currentThreadId) {
      saveMessage(currentThreadId, {
        id: userMessage.id,
        threadId: currentThreadId,
        role: 'user',
        content: message,
        readerPosition,
        createdAt: Date.now(),
      }).catch(e => console.error('[App] Failed to save user message:', e))
    }

    startStreaming({
      bookId: bookId || undefined,
      question: message,
      useRag: !!bookId,
      readerPosition,
      conversationHistory: conversationHistory.length > 0 ? conversationHistory : undefined,
    }, assistantId)
  }, [bookId, readerPosition, startStreaming, activeThreadId, buildConversationHistory])

  const handleNavigateToSource = useCallback(async (source: NonNullable<ChatMessage['sources']>[0]) => {
    const currentCfi = epubReaderRef.current?.getCurrentCfi()
    if (currentCfi) {
      setSavedReadingCfi(currentCfi)
    }
    await epubReaderRef.current?.navigateToText(source.full_text, source.chapter_title)
  }, [])

  const handleBackToReading = useCallback(() => {
    if (savedReadingCfi) {
      epubReaderRef.current?.navigateToCfi(savedReadingCfi)
      setSavedReadingCfi(null)
    }
  }, [savedReadingCfi])

  const handleNewThread = useCallback(() => {
    setActiveThreadId(null)
    setChatMessages([])
  }, [])

  const handleSelectThread = useCallback(async (threadId: string) => {
    const messages = await getThreadMessages(threadId)
    setActiveThreadId(threadId)
    setChatMessages(messages.map(m => ({
      id: m.id,
      role: m.role,
      content: m.content,
      context: m.context,
      readerPosition: m.readerPosition,
      sources: m.sources,
    })))
  }, [])

  if (!historyLoaded && !document) {
    return <div className="h-screen bg-[#14120b]" />
  }

  return (
    <div className="flex h-screen overflow-hidden">
      {!document && (
        <Sidebar
          currentView={currentView}
          onViewChange={setCurrentView}
          onUploadClick={() => uploadInputRef.current?.click()}
          bookCount={history.length}
        />
      )}

      <main className="flex-1 overflow-auto">
        {!document ? (
          <LibraryView
            books={history}
            currentFilter={currentView}
            onBookSelect={handleOpenFromHistory}
            onDeleteBook={handleDeleteBook}
            onUpload={handleFileLoad}
            uploadInputRef={uploadInputRef}
          />
        ) : isEpub && epubData ? (
          <>
            <EpubReader
              ref={epubReaderRef}
              epubData={epubData}
              fileName={fileName}
              isIndexing={isIndexing}
              isIndexed={isIndexed}
              isChatOpen={isChatOpen}
              sourceNavCfi={savedReadingCfi}
              onCloseAction={handleBack}
              onTextSelect={handleTextSelect}
              onLocationChange={setReaderPosition}
              onBackToReading={handleBackToReading}
              onOpenChat={() => setIsChatOpen(true)}
              hasChatHistory={chatMessages.length > 0 || threads.length > 0}
            />
            {selection && (
              <QuestionPopup
                selectedText={selection.text}
                position={selection.position}
                pageRect={selection.pageRect}
                onSubmit={handleQuestionSubmit}
                onClose={() => setSelection(null)}
                externalClosing={isClosingPopup}
              />
            )}
            {isChatOpen && (
              <ChatPanel
                messages={chatMessages}
                isLoading={isChatLoading}
                loadingStatus={loadingStatus}
                onSendMessage={handleChatMessage}
                onClose={() => setIsChatOpen(false)}
                onNavigateToSource={handleNavigateToSource}
                threadTitle={activeThreadId ? chatMessages.find(m => m.role === 'user')?.content?.slice(0, 50) : undefined}
                threads={threads}
                onNewThread={handleNewThread}
                onSelectThread={handleSelectThread}
              />
            )}
          </>
        ) : (
          <DocumentViewer
            content={document}
            fileName={fileName}
            onClose={handleBack}
          />
        )}
      </main>

      {/* Book loading transition */}
      {isLoadingBook && loadingBook && (
        <div className="fixed inset-0 z-50 bg-[#14120b] flex flex-col items-center justify-center animate-fade-in">
          <div className="w-32 h-48 rounded-lg overflow-hidden shadow-2xl mb-6">
            {loadingBook.coverUrl ? (
              <img
                src={loadingBook.coverUrl}
                alt={loadingBook.fileName}
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="w-full h-full bg-gradient-to-br from-[#2a2a2a] to-[#1f1f1f] flex items-center justify-center">
                <span className="text-[#f7f7f4]/20 text-4xl">📖</span>
              </div>
            )}
          </div>
          <h2 className="text-[#f7f7f4] text-lg font-medium mb-3 max-w-xs text-center">
            {loadingBook.fileName.replace(/\.(epub|pdf|txt)$/i, '')}
          </h2>
          <div className="w-8 h-0.5 bg-accent/60 rounded-full animate-pulse" />
        </div>
      )}

      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          onClose={() => setToast(null)}
        />
      )}
    </div>
  )
}
