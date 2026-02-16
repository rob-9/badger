'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import DocumentViewer from '@/components/DocumentViewer'
import EpubReader, { type TextSelection } from '@/components/EpubReader'
import Sidebar from '@/components/Sidebar'
import type { ViewType } from '@/components/Sidebar'
import LibraryView from '@/components/LibraryView'
import QuestionPopup from '@/components/QuestionPopup'
import ChatPanel, { type ChatMessage } from '@/components/ChatPanel'
import Toast from '@/components/Toast'
import { addBook, getBookData, getBookHistory, removeBook, type BookMetadata } from '@/lib/bookStorage'
import { indexBook, isBookIndexed, queryBook } from '@/lib/api'
import { extractCover, extractText, extractStructuredText } from '@/lib/parseEpub'

export default function Home() {
  const [document, setDocument] = useState<string | null>(null)
  const [fileName, setFileName] = useState<string>('')
  const [epubData, setEpubData] = useState<ArrayBuffer | null>(null)
  const [isEpub, setIsEpub] = useState(false)
  const [history, setHistory] = useState<BookMetadata[]>([])
  const [historyLoaded, setHistoryLoaded] = useState(false)
  const [bookId, setBookId] = useState<string | null>(null)
  const [isIndexing, setIsIndexing] = useState(false)
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

  // Loading transition state
  const [isLoadingBook, setIsLoadingBook] = useState(false)
  const [loadingBook, setLoadingBook] = useState<BookMetadata | null>(null)

  // Toast state
  const [toast, setToast] = useState<{ message: string; type: 'info' | 'success' | 'error' } | null>(null)

  // Load history on mount + restore last open book
  useEffect(() => {
    getBookHistory().then(async (books) => {
      setHistory(books)
      setHistoryLoaded(true)

      // Restore last open book on reload
      const lastBookId = localStorage.getItem('boom-active-book')
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

            // Only extract text + index if not already indexed
            const indexed = await isBookIndexed(book.id)
            if (!indexed) {
              setIsIndexing(true)
              try {
                const structured = await extractStructuredText(data)
                await indexBook(book.id, structured)
              } catch (error) {
                console.error('[App] Failed to index book on restore:', error)
              } finally {
                setIsIndexing(false)
              }
            }
          }
        }
      }
    })
  }, [])

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
      localStorage.setItem('boom-active-book', id)
      setHistory(await getBookHistory()) // Refresh history

      // Index the book for RAG
      console.log('[App] Starting RAG indexing for book')
      setIsIndexing(true)
      setToast({ message: 'Preparing book for AI questions...', type: 'info' })
      try {
        const structured = await extractStructuredText(arrayBuffer)
        await indexBook(id, structured)
        console.log('[App] Book indexed successfully')
        setToast({ message: 'Book ready! Highlight text to ask questions', type: 'success' })
      } catch (error) {
        console.error('[App] Failed to index book:', error)
        setToast({ message: 'AI features unavailable. You can still read.', type: 'error' })
      } finally {
        setIsIndexing(false)
      }
    }
  }

  const handleOpenFromHistory = useCallback(async (book: BookMetadata) => {
    // Show loading screen immediately
    setLoadingBook(book)
    setIsLoadingBook(true)

    const data = await getBookData(book.id)
    if (data) {
      setEpubData(data)
      setFileName(book.fileName)
      setIsEpub(true)
      setBookId(book.id)
      localStorage.setItem('boom-active-book', book.id)

      // Keep loading screen up, then mount the reader behind it
      await new Promise(r => setTimeout(r, 300))
      setDocument('loaded')
      // Let the reader mount and initialize behind the overlay
      await new Promise(r => setTimeout(r, 200))
      setIsLoadingBook(false)
      setLoadingBook(null)

      // Only extract text + index if not already indexed
      const indexed = await isBookIndexed(book.id)
      if (!indexed) {
        setIsIndexing(true)
        try {
          const structured = await extractStructuredText(data)
          await indexBook(book.id, structured)
        } catch (error) {
          console.error('[App] Failed to index book on reopen:', error)
        } finally {
          setIsIndexing(false)
        }
      }
    } else {
      setIsLoadingBook(false)
      setLoadingBook(null)
    }
  }, [])

  const handleDeleteBook = useCallback(async (bookId: string) => {
    await removeBook(bookId)
    setHistory(await getBookHistory())
  }, [])

  const handleBack = async () => {
    setDocument(null)
    setFileName('')
    setEpubData(null)
    setIsEpub(false)
    localStorage.removeItem('boom-active-book')
    setBookId(null)
    setIsIndexing(false)
    setHistory(await getBookHistory()) // Refresh history
    // Reset chat state
    setSelection(null)
    setChatMessages([])
    setIsChatOpen(false)
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

  const handleQuestionSubmit = useCallback(async (question: string, context: string) => {
    setSelection(null)
    setIsChatOpen(true)

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: question,
      context,
    }
    setChatMessages(prev => [...prev, userMessage])

    setIsChatLoading(true)
    try {
      console.log('[App] Querying with RAG:', { bookId, question, hasContext: !!context })

      const data = await queryBook({
        bookId: bookId || undefined,
        question,
        selectedText: context,
        useRag: !!bookId,
        readerPosition
      })
      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.answer,
      }
      setChatMessages(prev => [...prev, assistantMessage])

      console.log('[App] Got answer from RAG')
    } catch (error) {
      console.error('[App] Failed to query:', error)
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your question. Please try again.',
      }
      setChatMessages(prev => [...prev, errorMessage])
    } finally {
      setIsChatLoading(false)
    }
  }, [bookId])

  const handleChatMessage = useCallback(async (message: string) => {
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: message,
    }
    setChatMessages(prev => [...prev, userMessage])

    setIsChatLoading(true)
    try {
      console.log('[App] Follow-up query with RAG:', { bookId, message })

      const data = await queryBook({
        bookId: bookId || undefined,
        question: message,
        useRag: !!bookId,
        readerPosition
      })
      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.answer,
      }
      setChatMessages(prev => [...prev, assistantMessage])

      console.log('[App] Got follow-up answer from RAG')
    } catch (error) {
      console.error('[App] Failed to query:', error)
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your question. Please try again.',
      }
      setChatMessages(prev => [...prev, errorMessage])
    } finally {
      setIsChatLoading(false)
    }
  }, [bookId])

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
              epubData={epubData}
              fileName={fileName}
              isIndexing={isIndexing}
              isChatOpen={isChatOpen}
              onCloseAction={handleBack}
              onTextSelect={handleTextSelect}
              onLocationChange={setReaderPosition}
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
                onSendMessage={handleChatMessage}
                onClose={() => setIsChatOpen(false)}
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
