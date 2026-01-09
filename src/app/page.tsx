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
    setIsEpub(name.toLowerCase().endsWith('.epub'))

    // For EPUB files, store the ArrayBuffer for the reader and save to history
    if (name.toLowerCase().endsWith('.epub') && arrayBuffer) {
      setEpubData(arrayBuffer)
      await addBook(name, arrayBuffer)
      setHistory(getBookHistory()) // Refresh history
    }
  }

  const handleOpenFromHistory = useCallback(async (book: BookMetadata) => {
    const data = await getBookData(book.id)
    if (data) {
      setEpubData(data)
      setFileName(book.fileName)
      setIsEpub(true)
      setDocument('loaded') // Set truthy value to trigger reader view
    }
  }, [])

  const handleBack = () => {
    setDocument(null)
    setFileName('')
    setEpubData(null)
    setIsEpub(false)
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

    // TODO: Replace with actual LLM API call
    setIsChatLoading(true)
    try {
      // Placeholder response - replace with RAG/LLM integration
      await new Promise(resolve => setTimeout(resolve, 1000))
      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `This is a placeholder response. To enable real answers, integrate an LLM API.\n\nYour question: "${question}"\n\nContext: "${context.slice(0, 200)}..."`,
      }
      setChatMessages(prev => [...prev, assistantMessage])
    } finally {
      setIsChatLoading(false)
    }
  }, [])

  const handleChatMessage = useCallback(async (message: string) => {
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: message,
    }
    setChatMessages(prev => [...prev, userMessage])

    // TODO: Replace with actual LLM API call
    setIsChatLoading(true)
    try {
      await new Promise(resolve => setTimeout(resolve, 1000))
      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Placeholder follow-up response to: "${message}"`,
      }
      setChatMessages(prev => [...prev, assistantMessage])
    } finally {
      setIsChatLoading(false)
    }
  }, [])

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
