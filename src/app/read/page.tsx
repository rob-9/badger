'use client'

import { useState } from 'react'
import DocumentViewer from '@/components/DocumentViewer'
import FileUploader from '@/components/FileUploader'
import { useRouter } from 'next/navigation'

export default function ReadPage() {
  const [document, setDocument] = useState<string | null>(null)
  const [fileName, setFileName] = useState<string>('')
  const router = useRouter()

  const handleFileLoad = (content: string, name: string) => {
    setDocument(content)
    setFileName(name)
  }

  const handleBack = () => {
    setDocument(null)
    setFileName('')
    router.push('/')
  }

  return (
    <main className="min-h-screen">
      {!document ? (
        <div className="flex items-center justify-center min-h-screen bg-paper">
          <FileUploader onFileLoadAction={handleFileLoad} />
        </div>
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
