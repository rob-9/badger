'use client'

import { Book, Clock, Upload, FileText, File, Trash2 } from 'lucide-react'
import { useRef, useState } from 'react'
import type { BookMetadata } from '@/lib/bookStorage'
import { formatDate } from '@/lib/formatDate'
import { parseEpub } from '@/lib/parseEpub'
import type { ViewType } from '@/components/Sidebar'

interface LibraryViewProps {
  books: BookMetadata[]
  currentFilter: ViewType
  onBookSelect: (book: BookMetadata) => void
  onDeleteBook: (bookId: string) => void
  onUpload: (content: string, fileName: string, arrayBuffer?: ArrayBuffer) => void
  uploadInputRef?: React.RefObject<HTMLInputElement | null>
}

export default function LibraryView({
  books,
  currentFilter,
  onBookSelect,
  onDeleteBook,
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

      const lowerName = file.name.toLowerCase()
      if (lowerName.endsWith('.epub') || lowerName.endsWith('.epub.zip')) {
        arrayBuffer = await file.arrayBuffer()
        text = await parseEpub(file)
      } else {
        text = await file.text()
      }

      // Normalize .epub.zip → .epub for display
      const cleanName = file.name.replace(/\.epub\.zip$/i, '.epub')
      onUpload(text, cleanName, arrayBuffer)
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
    <div className="min-h-screen bg-[#14120b] text-[#f7f7f4] relative">
      {/* Gradient Mesh Background */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[5%] left-[10%] w-[250px] h-[250px] rounded-full bg-[#d9955f] opacity-[0.06] blur-[65px]"></div>
        <div className="absolute top-[15%] right-[15%] w-[220px] h-[220px] rounded-full bg-[#cd7f47] opacity-[0.07] blur-[58px]"></div>
        <div className="absolute bottom-[30%] left-[5%] w-[190px] h-[190px] rounded-full bg-[#e8a965] opacity-[0.05] blur-[54px]"></div>
        <div className="absolute top-[50%] left-[50%] w-[280px] h-[280px] rounded-full bg-[#bc8555] opacity-[0.06] blur-[70px]"></div>
        <div className="absolute bottom-[10%] right-[20%] w-[240px] h-[240px] rounded-full bg-[#d69658] opacity-[0.06] blur-[62px]"></div>
      </div>

      {/* Hidden file input shared between hero and sidebar upload */}
      <input
        ref={fileInputRef as React.RefObject<HTMLInputElement>}
        type="file"
        accept=".txt,.pdf,.epub,.zip"
        onChange={handleFileInput}
        className="hidden"
      />

      {showHero ? (
        <div className="relative z-10 flex items-center justify-center min-h-screen p-8">
          <div className="w-full max-w-2xl">
            <div className="text-center mb-8">
              <Book className="w-20 h-20 mx-auto mb-6 text-accent" />
              <h1 className="text-4xl font-bold mb-3 text-[#f7f7f4]">badger</h1>
              <p className="text-xl text-[#f7f7f4]/80 mb-2">Read Better.</p>
              <p className="text-sm text-[#f7f7f4]/40">
                Upload your books and ask questions powered by AI
              </p>
            </div>

            <div
              className={`border-2 border-dashed rounded-xl p-16 text-center transition-all ${
                isDragging
                  ? 'border-accent bg-accent/10 scale-105'
                  : 'border-[#f7f7f4]/15 hover:border-[#f7f7f4]/25'
              }`}
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              onDragEnter={() => setIsDragging(true)}
              onDragLeave={() => setIsDragging(false)}
            >
              {isLoading ? (
                <div className="flex flex-col items-center space-y-4">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-accent"></div>
                  <p className="text-[#f7f7f4]/60 text-lg">Loading document...</p>
                </div>
              ) : (
                <>
                  <Upload className="w-16 h-16 mx-auto mb-6 text-[#f7f7f4]/30" />
                  <h3 className="text-2xl font-semibold mb-3 text-[#f7f7f4]">
                    Drop your book here
                  </h3>
                  <p className="text-[#f7f7f4]/60 mb-8">
                    or click to browse your files
                  </p>
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="inline-flex items-center px-8 py-4 bg-accent/90 text-[#14120b] rounded-lg hover:bg-accent cursor-pointer transition-all hover:scale-105 text-lg font-medium"
                  >
                    <FileText className="w-6 h-6 mr-3" />
                    Choose File
                  </button>
                </>
              )}
            </div>

            <div className="mt-12 grid grid-cols-3 gap-6 text-center text-sm text-[#f7f7f4]/40">
              <div className="flex flex-col items-center">
                <div className="w-12 h-12 rounded-lg bg-[#f7f7f4]/5 flex items-center justify-center mb-3">
                  <FileText className="w-6 h-6" />
                </div>
                <span className="font-medium">Text Files</span>
              </div>
              <div className="flex flex-col items-center">
                <div className="w-12 h-12 rounded-lg bg-[#f7f7f4]/5 flex items-center justify-center mb-3">
                  <File className="w-6 h-6" />
                </div>
                <span className="font-medium">PDF Files</span>
              </div>
              <div className="flex flex-col items-center">
                <div className="w-12 h-12 rounded-lg bg-[#f7f7f4]/5 flex items-center justify-center mb-3">
                  <Book className="w-6 h-6" />
                </div>
                <span className="font-medium">EPUB Books</span>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="relative z-10 p-8 pl-24 lg:pl-8">
          <div className="max-w-7xl mx-auto">
            <div className="mb-8">
              <h2 className="text-2xl font-bold text-[#f7f7f4] mb-2">
                {currentFilter === 'recent' ? 'Recent Books' : 'All Books'}
              </h2>
              <p className="text-[#f7f7f4]/60">
                {filteredBooks.length} {filteredBooks.length === 1 ? 'book' : 'books'}
              </p>
            </div>

            {filteredBooks.length > 0 ? (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4">
                {filteredBooks.map((book, index) => (
                  <div
                    key={book.id}
                    className="library-card group relative text-left bg-[#1a1812] rounded-xl border border-[#f7f7f4]/10 hover:border-[#f7f7f4]/20 hover:shadow-lg hover:shadow-black/20 transition-all overflow-hidden animate-fade-up"
                    style={{ animationDelay: `${index * 50}ms` }}
                  >
                    {/* Delete button */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        onDeleteBook(book.id)
                      }}
                      className="absolute top-2 right-2 z-10 p-1.5 rounded-lg bg-black/40 text-[#f7f7f4]/60 opacity-0 group-hover:opacity-100 hover:bg-red-500/80 hover:text-white transition-all cursor-pointer"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>

                    <button
                      onClick={() => onBookSelect(book)}
                      className="w-full text-left cursor-pointer"
                    >
                      <div className={`aspect-[2/3] flex items-center justify-center relative overflow-hidden ${book.coverUrl ? '' : 'bg-gradient-to-br from-[#2a2a2a] to-[#1f1f1f]'}`}>
                        {book.coverUrl ? (
                          <img
                            src={book.coverUrl}
                            alt={book.fileName}
                            className="absolute inset-0 w-full h-full object-cover opacity-50 group-hover:opacity-90 transition-opacity duration-200"
                          />
                        ) : (
                          <Book className="w-10 h-10 text-[#f7f7f4]/15" />
                        )}
                        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors" />
                      </div>

                      <div className="p-2.5">
                        <h3 className="font-medium text-sm text-[#f7f7f4] mb-1 line-clamp-2 leading-tight">
                          {book.fileName.replace(/\.(epub|pdf|txt)$/i, '')}
                        </h3>
                        <div className="flex items-center gap-1.5 text-xs text-[#f7f7f4]/40">
                          <Clock className="w-3 h-3" />
                          <span>{formatDate(book.lastReadAt)}</span>
                        </div>
                      </div>
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-20">
                <Book className="w-16 h-16 mx-auto mb-4 text-[#f7f7f4]/20" />
                <p className="text-[#f7f7f4]/40">
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
