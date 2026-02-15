'use client'

import { Book, Clock, Upload, FileText, File } from 'lucide-react'
import { useRef, useState } from 'react'
import type { BookMetadata } from '@/lib/bookStorage'
import { formatDate } from '@/lib/formatDate'
import { parseEpub } from '@/lib/parseEpub'
import type { ViewType } from '@/components/Sidebar'

interface LibraryViewProps {
  books: BookMetadata[]
  currentFilter: ViewType
  onBookSelect: (book: BookMetadata) => void
  onUpload: (content: string, fileName: string, arrayBuffer?: ArrayBuffer) => void
  uploadInputRef?: React.RefObject<HTMLInputElement | null>
}

export default function LibraryView({
  books,
  currentFilter,
  onBookSelect,
  onUpload,
  uploadInputRef,
}: LibraryViewProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const localInputRef = useRef<HTMLInputElement>(null)
  const fileInputRef = uploadInputRef || localInputRef

  const handleFile = async (file: File) => {
    setIsLoading(true)

    try {
      let text: string
      let arrayBuffer: ArrayBuffer | undefined

      if (file.name.toLowerCase().endsWith('.epub')) {
        arrayBuffer = await file.arrayBuffer()
        text = await parseEpub(file)
      } else {
        text = await file.text()
      }

      onUpload(text, file.name, arrayBuffer)
    } catch (error) {
      console.error('Error reading file:', error)
      alert('Error loading file: ' + (error as Error).message)
    } finally {
      setIsLoading(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      handleFile(files[0])
    }
  }

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      handleFile(files[0])
    }
  }

  const filteredBooks = currentFilter === 'recent'
    ? books.slice(0, 10)
    : books

  const showHero = books.length === 0

  return (
    <div className="min-h-screen bg-paper dark:bg-[#141414]">
      {/* Hidden file input shared between hero and sidebar upload */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".txt,.pdf,.epub"
        onChange={handleFileInput}
        className="hidden"
      />

      {showHero ? (
        <div className="flex items-center justify-center min-h-screen p-8">
          <div className="w-full max-w-2xl">
            <div className="text-center mb-8">
              <Book className="w-20 h-20 mx-auto mb-6 text-accent" />
              <h1 className="text-4xl font-bold mb-3 dark:text-[#e0e0e0]">boom</h1>
              <p className="text-xl text-gray-600 dark:text-[#888] mb-2">Read Better.</p>
              <p className="text-sm text-gray-500 dark:text-[#666]">
                Upload your books and ask questions powered by AI
              </p>
            </div>

            <div
              className={`border-2 border-dashed rounded-xl p-16 text-center transition-all ${
                isDragging
                  ? 'border-accent bg-blue-50 dark:bg-blue-950/20 scale-105'
                  : 'border-gray-300 dark:border-[#333] hover:border-gray-400 dark:hover:border-[#555]'
              }`}
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              onDragEnter={() => setIsDragging(true)}
              onDragLeave={() => setIsDragging(false)}
            >
              {isLoading ? (
                <div className="flex flex-col items-center space-y-4">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-accent"></div>
                  <p className="text-gray-600 dark:text-[#888] text-lg">Loading document...</p>
                </div>
              ) : (
                <>
                  <Upload className="w-16 h-16 mx-auto mb-6 text-gray-400" />
                  <h3 className="text-2xl font-semibold mb-3 dark:text-[#e0e0e0]">
                    Drop your book here
                  </h3>
                  <p className="text-gray-600 dark:text-[#888] mb-8">
                    or click to browse your files
                  </p>
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="inline-flex items-center px-8 py-4 bg-accent text-white rounded-lg hover:bg-blue-600 cursor-pointer transition-all hover:scale-105 text-lg font-medium"
                  >
                    <FileText className="w-6 h-6 mr-3" />
                    Choose File
                  </button>
                </>
              )}
            </div>

            <div className="mt-12 grid grid-cols-3 gap-6 text-center text-sm text-gray-500 dark:text-[#666]">
              <div className="flex flex-col items-center">
                <div className="w-12 h-12 rounded-lg bg-gray-100 dark:bg-[#2a2a2a] flex items-center justify-center mb-3">
                  <FileText className="w-6 h-6" />
                </div>
                <span className="font-medium">Text Files</span>
              </div>
              <div className="flex flex-col items-center">
                <div className="w-12 h-12 rounded-lg bg-gray-100 dark:bg-[#2a2a2a] flex items-center justify-center mb-3">
                  <File className="w-6 h-6" />
                </div>
                <span className="font-medium">PDF Files</span>
              </div>
              <div className="flex flex-col items-center">
                <div className="w-12 h-12 rounded-lg bg-gray-100 dark:bg-[#2a2a2a] flex items-center justify-center mb-3">
                  <Book className="w-6 h-6" />
                </div>
                <span className="font-medium">EPUB Books</span>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="p-8">
          <div className="max-w-7xl mx-auto">
            <div className="mb-8">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                {currentFilter === 'recent' ? 'Recent Books' : 'All Books'}
              </h2>
              <p className="text-gray-600 dark:text-[#888]">
                {filteredBooks.length} {filteredBooks.length === 1 ? 'book' : 'books'}
              </p>
            </div>

            {filteredBooks.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                {filteredBooks.map((book, index) => (
                  <button
                    key={book.id}
                    onClick={() => onBookSelect(book)}
                    className="library-card group text-left bg-white dark:bg-[#1e1e1e] rounded-xl border border-gray-200 dark:border-[#2a2a2a] hover:border-gray-300 dark:hover:border-[#444] hover:shadow-lg transition-all overflow-hidden animate-fade-up cursor-pointer"
                    style={{ animationDelay: `${index * 50}ms` }}
                  >
                    <div className="aspect-[3/4] bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center relative overflow-hidden">
                      {book.coverUrl ? (
                        <img
                          src={book.coverUrl}
                          alt={book.fileName}
                          className="absolute inset-0 w-full h-full object-cover"
                        />
                      ) : (
                        <Book className="w-16 h-16 text-white/90" />
                      )}
                      <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors" />
                    </div>

                    <div className="p-4">
                      <h3 className="font-semibold text-gray-900 dark:text-[#e0e0e0] mb-2 line-clamp-2">
                        {book.fileName.replace(/\.(epub|pdf|txt)$/i, '')}
                      </h3>
                      <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-[#666]">
                        <Clock className="w-4 h-4" />
                        <span>{formatDate(book.lastReadAt)}</span>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            ) : (
              <div className="text-center py-20">
                <Book className="w-16 h-16 mx-auto mb-4 text-gray-300 dark:text-[#444]" />
                <p className="text-gray-500 dark:text-[#666]">
                  No books in this view
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
