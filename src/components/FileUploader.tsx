'use client'

import { useState, useCallback } from 'react'
import { Upload, FileText, Book, File } from 'lucide-react'
import ePub from 'epubjs'

interface FileUploaderProps {
  onFileLoadAction: (content: string, fileName: string) => void
}

interface EpubSection {
  load: (loader: unknown) => Promise<void>
  unload: () => void
  document?: Document
}

export default function FileUploader({ onFileLoadAction }: FileUploaderProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [isLoading, setIsLoading] = useState(false)

  const handleFile = useCallback(async (file: File) => {
    setIsLoading(true)
    
    try {
      let text: string
      
      if (file.name.toLowerCase().endsWith('.epub')) {
        // Handle EPUB files
        const arrayBuffer = await file.arrayBuffer()
        const book = ePub(arrayBuffer)
        await book.opened

        // Get all spine items (chapters)
        const textContent: string[] = []

        // Get sections as an array
        const sections: EpubSection[] = []
        book.spine.each((section: EpubSection) => {
          sections.push(section)
        })

        // Process each section sequentially
        for (const section of sections) {
          try {
            await section.load(book.load.bind(book))

            // Extract text content
            if (section.document) {
              const body = section.document.querySelector('body')
              const text = body?.textContent || section.document.textContent || ''
              if (text.trim()) {
                textContent.push(text.trim())
              }
            }

            section.unload()
          } catch (err) {
            console.warn('Failed to load section:', err)
          }
        }

        text = textContent.join('\n\n')
      } else {
        // Handle other file types as plain text
        text = await file.text()
      }
      
      onFileLoadAction(text, file.name)
    } catch (error) {
      console.error('Error reading file:', error)
    } finally {
      setIsLoading(false)
    }
  }, [onFileLoadAction])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    
    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      handleFile(files[0])
    }
  }, [handleFile])

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      handleFile(files[0])
    }
  }, [handleFile])

  return (
    <div className="w-full max-w-2xl mx-auto p-8">
      <div className="text-center mb-8">
        <Book className="w-16 h-16 mx-auto mb-4 text-accent" />
        <h1 className="text-3xl font-bold mb-2">boom</h1>
        <p className="text-gray-600">Read Better.</p>
      </div>

      <div
        className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
          isDragging
            ? 'border-accent bg-blue-50'
            : 'border-gray-300 hover:border-gray-400'
        }`}
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        onDragEnter={() => setIsDragging(true)}
        onDragLeave={() => setIsDragging(false)}
      >
        {isLoading ? (
          <div className="flex flex-col items-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-accent mb-4"></div>
            <p>Loading document...</p>
          </div>
        ) : (
          <>
            <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
            <h3 className="text-xl font-semibold mb-2">Upload a document</h3>
            <p className="text-gray-600 mb-6">
              Drag and drop your file here, or click to browse
            </p>
            <input
              type="file"
              accept=".txt,.pdf,.epub"
              onChange={handleFileInput}
              className="hidden"
              id="file-input"
            />
            <label
              htmlFor="file-input"
              className="inline-flex items-center px-6 py-3 bg-accent text-white rounded-lg hover:bg-blue-600 cursor-pointer transition-colors"
            >
              <FileText className="w-5 h-5 mr-2" />
              Choose File
            </label>
          </>
        )}
      </div>

      <div className="mt-8 grid grid-cols-3 gap-4 text-center text-sm text-gray-500">
        <div className="flex flex-col items-center">
          <FileText className="w-6 h-6 mb-2" />
          <span>Text Files</span>
        </div>
        <div className="flex flex-col items-center">
          <File className="w-6 h-6 mb-2" />
          <span>PDF Files</span>
        </div>
        <div className="flex flex-col items-center">
          <Book className="w-6 h-6 mb-2" />
          <span>EPUB Books</span>
        </div>
      </div>
    </div>
  )
}