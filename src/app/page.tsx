'use client'

import { useState } from 'react'
import DocumentViewer from '@/components/DocumentViewer'
import FileUploader from '@/components/FileUploader'

export default function Home() {
  const [document, setDocument] = useState<string | null>(null)
  const [fileName, setFileName] = useState<string>('')

  const handleFileLoad = (content: string, name: string) => {
    setDocument(content)
    setFileName(name)
  }

  return (
    <main className="min-h-screen bg-paper">
      {!document ? (
        <div className="flex items-center justify-center min-h-screen">
          <FileUploader onFileLoad={handleFileLoad} />
        </div>
      ) : (
        <DocumentViewer 
          content={document} 
          fileName={fileName}
          onClose={() => {
            setDocument(null)
            setFileName('')
          }}
        />
      )}
    </main>
  )
}