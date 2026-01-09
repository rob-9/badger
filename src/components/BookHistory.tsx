'use client'

import { Book, Clock } from 'lucide-react'
import type { BookMetadata } from '@/lib/bookStorage'

interface BookHistoryProps {
  history: BookMetadata[]
  onOpenBook: (book: BookMetadata) => void
}

function formatDate(timestamp: number): string {
  const date = new Date(timestamp)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))

  if (diffDays === 0) {
    return 'Today'
  } else if (diffDays === 1) {
    return 'Yesterday'
  } else if (diffDays < 7) {
    return `${diffDays} days ago`
  } else {
    return date.toLocaleDateString()
  }
}

export default function BookHistory({ history, onOpenBook }: BookHistoryProps) {
  return (
    <div className="w-full max-w-2xl mx-auto px-8 mt-12 relative z-10">
      <div className="flex items-center gap-2 mb-4">
        <Clock className="w-5 h-5 text-gray-500" />
        <h2 className="text-lg font-semibold text-gray-700">Recent Books</h2>
      </div>

      <div className="space-y-2">
        {history.map((book) => (
          <button
            key={book.id}
            onClick={() => onOpenBook(book)}
            className="w-full flex items-center gap-4 p-4 bg-white rounded-lg border border-gray-200 hover:border-gray-300 hover:shadow-sm transition-all text-left cursor-pointer"
          >
            <div className="flex-shrink-0 w-10 h-12 bg-gray-100 rounded flex items-center justify-center">
              <Book className="w-5 h-5 text-gray-400" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="font-medium text-gray-900 truncate">{book.fileName}</p>
              <p className="text-sm text-gray-500">{formatDate(book.lastReadAt)}</p>
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}
