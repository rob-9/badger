'use client'

import { useState, useRef, useEffect } from 'react'
import { X, Send, BookOpen, Loader2 } from 'lucide-react'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  context?: string
}

interface ChatPanelProps {
  messages: ChatMessage[]
  isLoading: boolean
  onSendMessage: (message: string) => void
  onClose: () => void
}

export default function ChatPanel({ messages, isLoading, onSendMessage, onClose }: ChatPanelProps) {
  const [input, setInput] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (input.trim() && !isLoading) {
      onSendMessage(input.trim())
      setInput('')
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (input.trim() && !isLoading) {
        onSendMessage(input.trim())
        setInput('')
      }
    }
  }

  return (
    <div className="fixed right-0 top-0 h-full w-[400px] max-[768px]:w-full max-[768px]:max-w-[90vw] bg-white dark:bg-[#1e1e1e] border-l border-gray-100 dark:border-[#2a2a2a] shadow-2xl flex flex-col z-40 animate-slide-in-right">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100 dark:border-[#2a2a2a]">
        <div className="flex items-center gap-2.5">
          <BookOpen className="w-4 h-4 text-gray-400 dark:text-[#666]" />
          <h2 className="font-medium text-gray-800 dark:text-[#e0e0e0] text-sm">Agent</h2>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 hover:bg-gray-100 dark:hover:bg-[#2a2a2a] rounded-lg transition-colors text-gray-400 hover:text-gray-600 dark:text-[#666] dark:hover:text-[#aaa]"
          aria-label="Close chat panel"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-5 py-4 space-y-6" role="region" aria-label="Chat messages">
        {messages.length === 0 && (
          <div className="text-center text-gray-400 mt-16">
            <BookOpen className="w-10 h-10 mx-auto mb-3 text-gray-200" />
            <p className="text-sm">Highlight any text to ask a question</p>
          </div>
        )}

        {messages.map((msg) => (
          <div key={msg.id} className="space-y-1.5">
            {msg.role === 'user' ? (
              <div className="flex flex-col items-end gap-1">
                {msg.context && (
                  <div className="max-w-[88%] px-3 py-2 bg-gray-50 dark:bg-[#252525] rounded-xl border-l-2 border-gray-200 dark:border-[#444]">
                    <p className="text-xs text-gray-400 dark:text-[#666] italic line-clamp-2">
                      &ldquo;{msg.context.slice(0, 100)}{msg.context.length > 100 ? '…' : ''}&rdquo;
                    </p>
                  </div>
                )}
                <div className="max-w-[88%] px-4 py-2.5 bg-blue-500 text-white rounded-2xl rounded-br-md">
                  <p className="text-sm leading-relaxed">{msg.content}</p>
                </div>
              </div>
            ) : (
              <div className="max-w-[92%]">
                <div className="px-4 py-3 bg-gray-50 dark:bg-[#252525] rounded-2xl rounded-bl-md">
                  <p className="text-sm text-gray-700 dark:text-[#d4d4d4] leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                </div>
              </div>
            )}
          </div>
        ))}

        {isLoading && (
          <div className="flex items-center gap-2 text-gray-400">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span className="text-xs">Thinking...</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="px-4 py-4 border-t border-gray-100 dark:border-[#2a2a2a] bg-white dark:bg-[#1e1e1e]">
        <form onSubmit={handleSubmit}>
          <div className="flex items-end gap-2 bg-gray-50 dark:bg-[#252525] rounded-2xl px-4 py-3 border border-gray-200 dark:border-[#333] focus-within:border-blue-300 dark:focus-within:border-blue-700 focus-within:ring-2 focus-within:ring-blue-100 dark:focus-within:ring-blue-900/30 transition-all">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a follow-up..."
              disabled={isLoading}
              rows={1}
              className="flex-1 bg-transparent text-sm text-gray-800 dark:text-[#e0e0e0] placeholder-gray-400 dark:placeholder-[#555] focus:outline-none resize-none disabled:opacity-50 max-h-32 overflow-y-auto"
              style={{ lineHeight: '1.5' }}
            />
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="flex-shrink-0 p-1.5 bg-blue-500 text-white rounded-xl hover:bg-blue-600 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
              aria-label="Send message"
            >
              <Send className="w-3.5 h-3.5" />
            </button>
          </div>
          <p className="text-xs text-gray-300 mt-1.5 text-center">Enter to send · Shift+Enter for new line</p>
        </form>
      </div>
    </div>
  )
}
