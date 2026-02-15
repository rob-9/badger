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
import { indexBook, queryBook } from '@/lib/api'
import { extractCover } from '@/lib/parseEpub'

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

  // Toast state
  const [toast, setToast] = useState<{ message: string; type: 'info' | 'success' | 'error' } | null>(null)

  // Load history on mount
  useEffect(() => {
    getBookHistory().then((books) => {
      setHistory(books)
      setHistoryLoaded(true)
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
      setHistory(await getBookHistory()) // Refresh history

      // Index the book for RAG
      console.log('[App] Starting RAG indexing for book')
      setIsIndexing(true)
      setToast({ message: 'Preparing book for AI questions...', type: 'info' })
      try {
        await indexBook(id, content)
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
    const data = await getBookData(book.id)
    if (data) {
      setEpubData(data)
      setFileName(book.fileName)
      setIsEpub(true)
      setBookId(book.id)
      setDocument('loaded') // Set truthy value to trigger reader view
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
