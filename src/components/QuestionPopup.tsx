'use client'

import { useState, useRef, useEffect } from 'react'
import { MessageCircle, Send } from 'lucide-react'

interface QuestionPopupProps {
  selectedText: string
  position: { x: number; y: number }
  pageRect: { left: number; right: number; top: number; bottom: number }
  onSubmit: (question: string, context: string) => void
  onClose: () => void
}

export default function QuestionPopup({ selectedText, position, pageRect, onSubmit, onClose }: QuestionPopupProps) {
  const [question, setQuestion] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)
  const popupRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  // Close on Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose()
      }
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [onClose])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (question.trim()) {
      onSubmit(question.trim(), selectedText)
      setQuestion('')
    }
  }

  // Position popup in margin to the right of the page
  const POPUP_WIDTH = 280
  const MARGIN_GAP = 24

  // Try to position to the right of the page
  let left = pageRect.right + MARGIN_GAP
  let right = 'auto'

  // If not enough space on right, try left
  if (left + POPUP_WIDTH > window.innerWidth - 16) {
    left = pageRect.left - POPUP_WIDTH - MARGIN_GAP
    // If still not enough space, fall back to right edge
    if (left < 16) {
      left = window.innerWidth - POPUP_WIDTH - 16
    }
  }

  const style: React.CSSProperties = {
    position: 'fixed',
    left,
    top: Math.max(100, Math.min(position.y - 50, window.innerHeight - 200)),
    width: POPUP_WIDTH,
    zIndex: 50,
  }

  return (
    <div ref={popupRef} style={style} className="bg-white rounded-xl shadow-2xl border border-gray-200">
      <form onSubmit={handleSubmit} className="p-4">
        <div className="flex items-center gap-2 mb-3">
          <MessageCircle className="w-4 h-4 text-blue-500" />
          <span className="text-sm font-medium text-gray-700">Ask a question</span>
        </div>

        <div className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="What does this mean?"
            className="flex-1 px-3 py-2 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          <button
            type="submit"
            disabled={!question.trim()}
            className="p-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            title="Ask question"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>

        <div className="text-xs text-gray-400 mt-2">Press Esc to close</div>
      </form>
    </div>
  )
}
