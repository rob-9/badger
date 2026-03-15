'use client'

import { Book, Clock, Upload, FileText, File, Trash2, ArrowRight } from 'lucide-react'
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
    <div className="min-h-screen bg-surface-deep text-[#f7f7f4] relative grain-overlay">

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
            {/* Editorial hero */}
            <div className="text-center mb-12 animate-fade-up">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-accent/20 bg-accent/5 mb-8">
                <div className="w-1.5 h-1.5 rounded-full bg-accent animate-glow-pulse" />
                <span className="text-[0.65rem] uppercase tracking-[0.2em] text-accent/80 font-medium">AI-Powered Reading</span>
              </div>

              <h1 className="font-display text-6xl md:text-7xl font-light mb-4 text-[#f7f7f4] tracking-tight">
                badger
              </h1>
              <div className="w-12 h-px bg-accent/40 mx-auto mb-4" />
              <p className="font-display text-2xl text-[#f7f7f4]/70 font-light italic">
                Read better.
              </p>
              <p className="text-sm text-[#f7f7f4]/35 mt-3 tracking-wide">
                Upload your books. Highlight passages. Ask questions.
              </p>
            </div>

            <div
              className={`relative rounded-2xl transition-all duration-500 animate-fade-up ${
                isDragging
                  ? 'border-2 border-accent bg-accent/10 scale-[1.02]'
                  : 'border border-[#f7f7f4]/10 hover:border-accent/30'
              }`}
              style={{ animationDelay: '100ms' }}
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              onDragEnter={() => setIsDragging(true)}
              onDragLeave={() => setIsDragging(false)}
            >
              {/* Inner glow on hover */}
              <div className="absolute inset-0 rounded-2xl bg-gradient-to-b from-accent/[0.03] to-transparent pointer-events-none" />

              <div className="relative p-16 text-center">
                {isLoading ? (
                  <div className="flex flex-col items-center space-y-4">
                    <div className="w-12 h-12 border-2 border-accent/30 border-t-accent rounded-full animate-spin" />
                    <p className="text-[#f7f7f4]/50 text-sm tracking-wide">Loading document...</p>
                  </div>
                ) : (
                  <>
                    <div className="w-16 h-16 mx-auto mb-8 rounded-2xl bg-accent/10 border border-accent/20 flex items-center justify-center">
                      <Upload className="w-7 h-7 text-accent/70" />
                    </div>
                    <h3 className="font-display text-2xl font-light mb-2 text-[#f7f7f4]/90">
                      Drop your book here
                    </h3>
                    <p className="text-[#f7f7f4]/40 text-sm mb-8">
                      or browse your files
                    </p>
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="group inline-flex items-center gap-3 px-8 py-3.5 bg-accent text-surface-deep rounded-xl hover:bg-accent-hover cursor-pointer transition-all duration-300 hover:shadow-lg hover:shadow-accent/20 text-sm font-medium tracking-wide"
                    >
                      <FileText className="w-4 h-4" />
                      Choose File
                      <ArrowRight className="w-4 h-4 opacity-0 -ml-2 group-hover:opacity-100 group-hover:ml-0 transition-all duration-300" />
                    </button>
                  </>
                )}
              </div>
            </div>

            {/* Supported formats — refined */}
            <div className="mt-16 flex justify-center gap-12 text-center animate-fade-up" style={{ animationDelay: '200ms' }}>
              {[
                { icon: FileText, label: 'Text' },
                { icon: File, label: 'PDF' },
                { icon: Book, label: 'EPUB' },
              ].map(({ icon: Icon, label }) => (
                <div key={label} className="flex flex-col items-center gap-2">
                  <div className="w-10 h-10 rounded-xl bg-[#f7f7f4]/[0.04] border border-[#f7f7f4]/[0.06] flex items-center justify-center">
                    <Icon className="w-4 h-4 text-[#f7f7f4]/30" />
                  </div>
                  <span className="text-[0.65rem] uppercase tracking-[0.15em] text-[#f7f7f4]/30 font-medium">{label}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <div className="relative z-10 p-8 pl-24 lg:pl-8">
          <div className="max-w-7xl mx-auto">
            {/* Section header — editorial style */}
            <div className="mb-10 animate-fade-up">
              <h2 className="font-display text-3xl font-light text-[#f7f7f4] mb-1 tracking-tight">
                {currentFilter === 'recent' ? 'Recent' : 'Library'}
              </h2>
              <div className="flex items-center gap-3 mt-2">
                <div className="w-8 h-px bg-accent/40" />
                <p className="text-xs uppercase tracking-[0.15em] text-[#f7f7f4]/40">
                  {filteredBooks.length} {filteredBooks.length === 1 ? 'book' : 'books'}
                </p>
              </div>
            </div>

            {filteredBooks.length > 0 ? (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-5">
                {filteredBooks.map((book, index) => (
                  <div
                    key={book.id}
                    className="library-card group relative text-left rounded-xl overflow-hidden animate-fade-up"
                    style={{ animationDelay: `${index * 60}ms` }}
                  >
                    {/* Delete button */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        onDeleteBook(book.id)
                      }}
                      className="absolute top-2 right-2 z-10 p-1.5 rounded-lg bg-black/50 backdrop-blur-sm text-[#f7f7f4]/60 opacity-0 group-hover:opacity-100 hover:bg-red-500/80 hover:text-white transition-all cursor-pointer"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>

                    <button
                      onClick={() => onBookSelect(book)}
                      className="w-full text-left cursor-pointer"
                    >
                      {/* Cover with spine accent */}
                      <div className="relative">
                        <div className={`aspect-[2/3] flex items-center justify-center relative overflow-hidden rounded-t-xl ${book.coverUrl ? '' : 'bg-gradient-to-br from-surface-raised to-surface-warm'}`}>
                          {/* Spine edge */}
                          <div className="absolute left-0 top-0 bottom-0 w-[3px] bg-gradient-to-b from-accent/40 via-accent/20 to-accent/40 z-10" />

                          {book.coverUrl ? (
                            <img
                              src={book.coverUrl}
                              alt={book.fileName}
                              className="absolute inset-0 w-full h-full object-cover opacity-60 group-hover:opacity-90 transition-opacity duration-500"
                            />
                          ) : (
                            <Book className="w-8 h-8 text-[#f7f7f4]/10" />
                          )}
                          {/* Hover veil */}
                          <div className="absolute inset-0 bg-gradient-to-t from-surface-deep/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                        </div>
                      </div>

                      <div className="p-3 bg-surface-warm border-x border-b border-[#f7f7f4]/[0.06] rounded-b-xl">
                        <h3 className="font-medium text-sm text-[#f7f7f4]/90 mb-1.5 line-clamp-2 leading-snug">
                          {book.fileName.replace(/\.(epub|pdf|txt)$/i, '')}
                        </h3>
                        <div className="flex items-center gap-1.5 text-[0.65rem] text-[#f7f7f4]/30">
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
                <Book className="w-14 h-14 mx-auto mb-4 text-[#f7f7f4]/15" />
                <p className="text-[#f7f7f4]/35 font-display text-lg italic">
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
