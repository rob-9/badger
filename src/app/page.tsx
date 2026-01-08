'use client'

import { useState } from 'react'
import DocumentViewer from '@/components/DocumentViewer'
import EpubReader from '@/components/EpubReader'
import FileUploader from '@/components/FileUploader'

export default function Home() {
  const [document, setDocument] = useState<string | null>(null)
  const [fileName, setFileName] = useState<string>('')
  const [epubData, setEpubData] = useState<ArrayBuffer | null>(null)
  const [isEpub, setIsEpub] = useState(false)

  const handleFileLoad = (content: string, name: string, arrayBuffer?: ArrayBuffer) => {
    setDocument(content)
    setFileName(name)
    setIsEpub(name.toLowerCase().endsWith('.epub'))

    // For EPUB files, store the ArrayBuffer for the reader
    if (name.toLowerCase().endsWith('.epub') && arrayBuffer) {
      setEpubData(arrayBuffer)
    }
  }

  const handleBack = () => {
    setDocument(null)
    setFileName('')
    setEpubData(null)
    setIsEpub(false)
  }

  return (
    <main className="min-h-screen">
      {!document ? (
        <div className="flex items-center justify-center min-h-screen bg-paper">
          <FileUploader onFileLoadAction={handleFileLoad} />
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
