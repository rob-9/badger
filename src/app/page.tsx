'use client'

import { useState, useEffect, useCallback } from 'react'
import DocumentViewer from '@/components/DocumentViewer'
import EpubReader from '@/components/EpubReader'
import FileUploader from '@/components/FileUploader'
import BookHistory from '@/components/BookHistory'
import { addBook, getBookData, getBookHistory, type BookMetadata } from '@/lib/bookStorage'

export default function Home() {
  const [document, setDocument] = useState<string | null>(null)
  const [fileName, setFileName] = useState<string>('')
  const [epubData, setEpubData] = useState<ArrayBuffer | null>(null)
  const [isEpub, setIsEpub] = useState(false)
  const [history, setHistory] = useState<BookMetadata[]>([])

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
    console.log('handleOpenFromHistory called for:', book.id)
    const data = await getBookData(book.id)
    console.log('Got data:', data ? `${data.byteLength} bytes` : 'null')
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
  }

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
        <EpubReader
          epubData={epubData}
          fileName={fileName}
          onCloseAction={handleBack}
        />
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
