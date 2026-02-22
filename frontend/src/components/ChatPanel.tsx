'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { X, Send, BookOpen, Loader2, ChevronDown } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  context?: string
  sources?: Array<{ text: string; full_text: string; score: number; chunk_index: number; source_number: number; label: string }>
}

interface ChatPanelProps {
  messages: ChatMessage[]
  isLoading: boolean
  loadingStatus?: string
  onSendMessage: (message: string) => void
  onClose: () => void
}

const LABEL_COLORS: Record<string, string> = {
  PAST: 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400',
  AHEAD: 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400',
}
const DEFAULT_LABEL_COLOR = 'bg-gray-100 dark:bg-gray-800/30 text-gray-600 dark:text-gray-400'

function truncate(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text
  // Break at last space within maxLen to avoid cutting mid-word
  const truncated = text.slice(0, maxLen)
  const lastSpace = truncated.lastIndexOf(' ')
  return (lastSpace > maxLen * 0.6 ? truncated.slice(0, lastSpace) : truncated) + '...'
}

export default function ChatPanel({ messages, isLoading, loadingStatus, onSendMessage, onClose }: ChatPanelProps) {
  const [input, setInput] = useState('')
  const [openPopover, setOpenPopover] = useState<string | null>(null)
  const [expandedSources, setExpandedSources] = useState<Set<string>>(new Set())
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const popoverRef = useRef<HTMLDivElement>(null)
  const citationBtnClass = 'citation-btn'

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  // Close popover on outside click or Escape
  useEffect(() => {
    if (!openPopover) return
    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as HTMLElement
      // Ignore clicks on citation buttons — their onClick handles toggling
      if (target.closest(`.${citationBtnClass}`)) return
      if (popoverRef.current && !popoverRef.current.contains(target)) {
        setOpenPopover(null)
      }
    }
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpenPopover(null)
    }
    document.addEventListener('mousedown', handleClickOutside)
    document.addEventListener('keydown', handleEscape)
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
      document.removeEventListener('keydown', handleEscape)
    }
  }, [openPopover])

  const toggleSources = useCallback((msgId: string) => {
    setExpandedSources(prev => {
      const next = new Set(prev)
      if (next.has(msgId)) {
        next.delete(msgId)
      } else {
        next.add(msgId)
      }
      return next
    })
  }, [])

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

  const lastMsg = messages[messages.length - 1]
  const isStreaming = isLoading && lastMsg?.role === 'assistant'

  return (
    <div className="fixed right-0 top-0 h-full w-[400px] max-[768px]:w-full max-[768px]:max-w-[90vw] bg-white dark:bg-[#1e1e1e] border-l border-gray-100 dark:border-[#2a2a2a] shadow-2xl flex flex-col z-40 animate-slide-in-right">
      {/* Header — matches main navbar height (px-6 py-4) */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-[#2a2a2a]">
        <div className="flex items-center gap-3">
          <BookOpen className="w-5 h-5 text-gray-400 dark:text-[#666]" />
          <h2 className="text-lg font-semibold text-gray-800 dark:text-[#e0e0e0]">Agent</h2>
        </div>
        <button
          onClick={onClose}
          className="p-2 hover:bg-gray-100 dark:hover:bg-[#2a2a2a] rounded-lg transition-colors text-gray-400 hover:text-gray-600 dark:text-[#666] dark:hover:text-[#aaa]"
          aria-label="Close chat panel"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-5 py-4 space-y-6" role="region" aria-label="Chat messages">
        {messages.length === 0 && (
          <div className="text-center text-gray-400 mt-16">
            <BookOpen className="w-10 h-10 mx-auto mb-3 text-gray-200 dark:text-[#f7f7f4]/15" />
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
                <div className="max-w-[88%] px-4 py-2.5 bg-accent text-[#14120b] rounded-2xl rounded-br-md">
                  <p className="text-sm leading-relaxed">{msg.content}</p>
                </div>
              </div>
            ) : (
              <div className="max-w-[92%]">
                <div className="px-4 py-3 bg-gray-50 dark:bg-[#252525] rounded-2xl rounded-bl-md prose prose-sm dark:prose-invert max-w-none relative">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={msg.sources?.length ? (() => {
                      const sources = msg.sources!

                      const processNode = (node: React.ReactNode): React.ReactNode => {
                        if (typeof node !== 'string') return node
                        const parts = node.split(/(\[Source \d+\])/)
                        if (parts.length === 1) return node
                        return parts.map((part, i) => {
                          const match = part.match(/^\[Source (\d+)\]$/)
                          if (!match) return part
                          const num = parseInt(match[1])
                          const source = sources.find(s => s.source_number === num)
                          const popoverKey = `${msg.id}-${num}`
                          const isOpen = openPopover === popoverKey
                          return (
                            <span key={i} className="relative inline">
                              <button
                                onClick={() => setOpenPopover(isOpen ? null : popoverKey)}
                                aria-label={`View source ${num}`}
                                aria-haspopup="dialog"
                                aria-expanded={isOpen}
                                className={`${citationBtnClass} inline-flex items-center justify-center min-w-[1.1em] h-[1.1em] text-[0.6em] font-semibold bg-accent/20 text-accent-foreground rounded-full px-[0.3em] align-super cursor-pointer hover:bg-accent/40 transition-colors mx-[0.1em] leading-none`}
                              >
                                {num}
                              </button>
                              {isOpen && source && (
                                <div
                                  ref={popoverRef}
                                  role="dialog"
                                  aria-label={`Source ${num} details`}
                                  className="absolute right-0 top-full mt-1 z-50 w-64 max-w-[calc(100vw-3rem)] bg-white dark:bg-[#2a2a2a] border border-gray-200 dark:border-[#444] rounded-lg shadow-lg p-3 text-left"
                                  style={{ fontSize: '0.8rem' }}
                                >
                                  <div className="flex items-center gap-1.5 mb-1.5">
                                    <span className={`text-[0.65rem] font-medium px-1.5 py-0.5 rounded ${LABEL_COLORS[source.label] || DEFAULT_LABEL_COLOR}`}>
                                      {source.label}
                                    </span>
                                    <span className="text-[0.65rem] text-gray-400 dark:text-[#666]">
                                      Chunk {source.chunk_index}
                                    </span>
                                  </div>
                                  <p className="text-xs text-gray-600 dark:text-[#bbb] leading-relaxed not-prose">
                                    {truncate(source.text, 200)}
                                  </p>
                                </div>
                              )}
                            </span>
                          )
                        })
                      }

                      const processChildren = (children: React.ReactNode) => {
                        return Array.isArray(children)
                          ? children.map((child, i) => <span key={i}>{processNode(child)}</span>)
                          : processNode(children)
                      }

                      return {
                        p: ({ children }) => <p>{processChildren(children)}</p>,
                        li: ({ children }) => <li>{processChildren(children)}</li>,
                        td: ({ children }) => <td>{processChildren(children)}</td>,
                        blockquote: ({ children }) => <blockquote>{processChildren(children)}</blockquote>,
                      }
                    })() : undefined}
                  >{msg.content}</ReactMarkdown>
                </div>

                {/* Collapsible sources section */}
                {msg.sources && msg.sources.length > 0 && (
                  <div className="mt-1.5">
                    <button
                      onClick={() => toggleSources(msg.id)}
                      aria-expanded={expandedSources.has(msg.id)}
                      className="flex items-center gap-1 text-[0.7rem] text-gray-400 dark:text-[#666] hover:text-gray-600 dark:hover:text-[#999] transition-colors"
                    >
                      <ChevronDown className={`w-3 h-3 transition-transform ${expandedSources.has(msg.id) ? 'rotate-180' : ''}`} />
                      {msg.sources.length} source{msg.sources.length !== 1 ? 's' : ''}
                    </button>
                    {expandedSources.has(msg.id) && (
                      <div className="mt-1.5 space-y-1.5 border-t border-gray-100 dark:border-[#333] pt-2">
                        {msg.sources.map((source) => (
                          <div key={source.source_number} className="flex gap-2 text-xs">
                            <span className="inline-flex items-center justify-center min-w-[1.2em] h-[1.2em] text-[0.65rem] font-semibold bg-accent/20 text-accent-foreground rounded-full px-1 leading-none flex-shrink-0 mt-0.5">
                              {source.source_number}
                            </span>
                            <div className="min-w-0">
                              <span className={`text-[0.6rem] font-medium px-1 py-0.5 rounded mr-1 ${LABEL_COLORS[source.label] || DEFAULT_LABEL_COLOR}`}>
                                {source.label}
                              </span>
                              <span className="text-gray-500 dark:text-[#888] line-clamp-2">
                                {truncate(source.text, 120)}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}

        {/* Pre-streaming spinner (before first token) */}
        {isLoading && !isStreaming && (
          <div className="flex items-center gap-2 text-gray-400" role="status" aria-live="polite">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span className="text-xs">{loadingStatus || 'Thinking...'}</span>
          </div>
        )}

        {/* Status indicator during streaming */}
        {isStreaming && loadingStatus && (
          <div className="flex items-center gap-1.5 text-gray-400 dark:text-[#666]" role="status" aria-live="polite">
            <Loader2 className="w-3 h-3 animate-spin" />
            <span className="text-[0.65rem]">{loadingStatus}</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="px-4 py-4 border-t border-gray-100 dark:border-[#2a2a2a] bg-white dark:bg-[#1e1e1e]">
        <form onSubmit={handleSubmit}>
          <div className="flex items-end gap-2 bg-gray-50 dark:bg-[#252525] rounded-2xl px-4 py-3 border border-gray-200 dark:border-[#333] focus-within:border-accent/50 dark:focus-within:border-accent/40 focus-within:ring-2 focus-within:ring-accent/20 dark:focus-within:ring-accent/10 transition-all">
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
              className="flex-shrink-0 p-1.5 bg-accent text-[#14120b] rounded-xl hover:bg-accent-hover disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
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
