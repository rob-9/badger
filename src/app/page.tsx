'use client'

import { useState, useEffect, useCallback } from 'react'
import DocumentViewer from '@/components/DocumentViewer'
import EpubReader, { type TextSelection } from '@/components/EpubReader'
import FileUploader from '@/components/FileUploader'
import BookHistory from '@/components/BookHistory'
import QuestionPopup from '@/components/QuestionPopup'
import ChatPanel, { type ChatMessage } from '@/components/ChatPanel'
import { addBook, getBookData, getBookHistory, type BookMetadata } from '@/lib/bookStorage'

export default function Home() {
  const [document, setDocument] = useState<string | null>(null)
  const [fileName, setFileName] = useState<string>('')
  const [epubData, setEpubData] = useState<ArrayBuffer | null>(null)
  const [isEpub, setIsEpub] = useState(false)
  const [history, setHistory] = useState<BookMetadata[]>([])
  const [bookId, setBookId] = useState<string | null>(null)
  const [isIndexing, setIsIndexing] = useState(false)

  // Chat state
  const [selection, setSelection] = useState<TextSelection | null>(null)
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
  const [isChatOpen, setIsChatOpen] = useState(false)
  const [isChatLoading, setIsChatLoading] = useState(false)

  // Load history on mount
  useEffect(() => {
    setHistory(getBookHistory())
  }, [])

  const handleFileLoad = async (content: string, name: string, arrayBuffer?: ArrayBuffer) => {
    setDocument(content)
    setFileName(name)
    const isEpubFile = name.toLowerCase().endsWith('.epub')
    setIsEpub(isEpubFile)

    // For EPUB files, store the ArrayBuffer for the reader and save to history
    if (isEpubFile && arrayBuffer) {
      setEpubData(arrayBuffer)
      const id = await addBook(name, arrayBuffer)
      setBookId(id)
      setHistory(getBookHistory()) // Refresh history

      // Index the book for RAG
      console.log('[App] Starting RAG indexing for book')
      setIsIndexing(true)
      try {
        const response = await fetch('/api/rag/index', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ bookId: id, text: content })
        })

        if (!response.ok) {
          throw new Error('Failed to index book')
        }

        console.log('[App] Book indexed successfully')
      } catch (error) {
        console.error('[App] Failed to index book:', error)
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

  const handleBack = () => {
    setDocument(null)
    setFileName('')
    setEpubData(null)
    setIsEpub(false)
    setBookId(null)
    setIsIndexing(false)
    setHistory(getBookHistory()) // Refresh history
    // Reset chat state
    setSelection(null)
    setChatMessages([])
    setIsChatOpen(false)
  }

  const handleTextSelect = useCallback((sel: TextSelection) => {
    setSelection(sel)
  }, [])

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

      const response = await fetch('/api/rag/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          bookId,
          question,
          selectedText: context,
          useRag: !!bookId // Use RAG if we have a bookId, otherwise simple query
        })
      })

      if (!response.ok) {
        throw new Error('Failed to query book')
      }

      const data = await response.json()
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

      const response = await fetch('/api/rag/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          bookId,
          question: message,
          useRag: !!bookId
        })
      })

      if (!response.ok) {
        throw new Error('Failed to query book')
      }

      const data = await response.json()
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

  return (
    <main className="min-h-screen">
      {!document ? (
        <div className="min-h-screen bg-paper py-12">
          <FileUploader onFileLoadAction={handleFileLoad} />
          {history.length > 0 && (
            <BookHistory history={history} onOpenBook={handleOpenFromHistory} />
          )}
        </div>
      ) : isEpub && epubData ? (
        <>
          <EpubReader
            epubData={epubData}
            fileName={fileName}
            onCloseAction={handleBack}
            onTextSelect={handleTextSelect}
          />
          {selection && (
            <QuestionPopup
              selectedText={selection.text}
              position={selection.position}
              pageRect={selection.pageRect}
              onSubmit={handleQuestionSubmit}
              onClose={() => setSelection(null)}
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
  )
}
