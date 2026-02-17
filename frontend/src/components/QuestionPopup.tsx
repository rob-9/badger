'use client'

import { useState, useRef, useEffect } from 'react'
import { Send, Sparkles } from 'lucide-react'

interface QuestionPopupProps {
  selectedText: string
  position: { x: number; y: number }
  pageRect: { left: number; right: number; top: number; bottom: number }
  onSubmit: (question: string, context: string) => void
  onClose: () => void
  externalClosing?: boolean
}

const QUICK_PROMPTS = [
  "What does this mean?",
  "Why is this significant?",
  "Who or what is this?",
  "How does this connect to earlier?",
]

export default function QuestionPopup({ selectedText, position, pageRect, onSubmit, onClose, externalClosing = false }: QuestionPopupProps) {
  const [question, setQuestion] = useState('')
  const [isClosing, setIsClosing] = useState(false)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const popupRef = useRef<HTMLDivElement>(null)

  const shouldClose = isClosing || externalClosing

  const handleClose = () => {
    setIsClosing(true)
    setTimeout(() => {
      onClose()
    }, 150) // Match animation duration
  }

  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') handleClose()
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [])

  // Close on outside click
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (popupRef.current && !popupRef.current.contains(e.target as Node)) {
        handleClose()
      }
    }
    setTimeout(() => document.addEventListener('mousedown', handleClick), 0)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (question.trim()) {
      setIsClosing(true)
      setTimeout(() => {
        onSubmit(question.trim(), selectedText)
      }, 150)
    }
  }

  const handleQuickPrompt = (prompt: string) => {
    setIsClosing(true)
    setTimeout(() => {
      onSubmit(prompt, selectedText)
    }, 150)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (question.trim()) {
        setIsClosing(true)
        setTimeout(() => {
          onSubmit(question.trim(), selectedText)
        }, 150)
      }
    }
  }

  const POPUP_WIDTH = 320
  const MARGIN_GAP = 20
  const isMobile = typeof window !== 'undefined' && window.innerWidth < 768

  let left = pageRect.right + MARGIN_GAP
  if (left + POPUP_WIDTH > window.innerWidth - 16) {
    left = pageRect.left - POPUP_WIDTH - MARGIN_GAP
    if (left < 16) {
      // On mobile, center the popup
      if (isMobile) {
        left = Math.max(16, (window.innerWidth - POPUP_WIDTH) / 2)
      } else {
        left = window.innerWidth - POPUP_WIDTH - 16
      }
    }
  }

  const top = Math.max(80, Math.min(position.y - 60, window.innerHeight - 380))

  return (
    <div
      ref={popupRef}
      style={{ position: 'fixed', left, top, width: POPUP_WIDTH, zIndex: 50 }}
      className={`bg-white dark:bg-[#1e1e1e] rounded-2xl shadow-2xl border border-gray-100 dark:border-[#2a2a2a] overflow-hidden ${shouldClose ? 'animate-fade-scale-out' : 'animate-fade-scale-in'}`}
    >
      {/* Selected text preview */}
      <div className="px-4 pt-4 pb-3 border-b border-gray-100 dark:border-[#2a2a2a]">
        <p className="text-xs text-gray-400 dark:text-[#666] uppercase tracking-wide mb-1.5">Selected</p>
        <p className="text-sm text-gray-600 dark:text-[#aaa] italic line-clamp-3 leading-relaxed">
          &ldquo;{selectedText.slice(0, 120)}{selectedText.length > 120 ? '…' : ''}&rdquo;
        </p>
      </div>

      {/* Quick prompts */}
      <div className="px-4 py-3 border-b border-gray-100 dark:border-[#2a2a2a]">
        <div className="flex items-center gap-1.5 mb-2">
          <Sparkles className="w-3 h-3 text-gray-400 dark:text-[#666]" />
          <p className="text-xs text-gray-400 dark:text-[#666]">Quick ask</p>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {QUICK_PROMPTS.map((prompt) => (
            <button
              key={prompt}
              onClick={() => handleQuickPrompt(prompt)}
              className="text-xs px-2.5 py-1 bg-gray-100 dark:bg-[#2a2a2a] hover:bg-gray-200 dark:hover:bg-[#333] text-gray-600 dark:text-[#aaa] rounded-full transition-colors"
              aria-label={`Ask: ${prompt}`}
            >
              {prompt}
            </button>
          ))}
        </div>
      </div>

      {/* Custom question input */}
      <form onSubmit={handleSubmit} className="p-4">
        <div className="flex gap-2">
          <textarea
            ref={inputRef}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask anything..."
            rows={2}
            className="flex-1 px-3 py-2 text-sm border border-gray-200 dark:border-[#333] dark:bg-[#141414] dark:text-[#e0e0e0] rounded-xl focus:outline-none focus:ring-2 focus:ring-accent focus:border-transparent resize-none placeholder-gray-400 dark:placeholder-[#555]"
          />
          <button
            type="submit"
            disabled={!question.trim()}
            className="self-end p-2 bg-accent text-[#14120b] rounded-xl hover:bg-accent-hover disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            aria-label="Submit question"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
        <p className="text-xs text-gray-300 mt-2">Enter to send · Esc to close</p>
      </form>
    </div>
  )
}
