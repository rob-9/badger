'use client'

import { useState, useRef, useCallback } from 'react'
import { X, HelpCircle, Bookmark, Settings, ChevronLeft, ChevronRight } from 'lucide-react'
import AgentAssistant from './AgentAssistant'
import TextHighlighter from './TextHighlighter'

interface DocumentViewerProps {
  content: string
  fileName: string
  onClose: () => void
}

export default function DocumentViewer({ content, fileName, onClose }: DocumentViewerProps) {
  const [selectedText, setSelectedText] = useState('')
  const [selectionPosition, setSelectionPosition] = useState<{ x: number; y: number } | null>(null)
  const [showAgent, setShowAgent] = useState(false)
  const [fontSize, setFontSize] = useState(18)
  const [lineHeight, setLineHeight] = useState(1.7)
  const contentRef = useRef<HTMLDivElement>(null)

  const handleTextSelection = useCallback(() => {
    const selection = window.getSelection()
    if (selection && selection.toString().trim()) {
      const text = selection.toString().trim()
      const range = selection.getRangeAt(0)
      const rect = range.getBoundingClientRect()
      
      setSelectedText(text)
      setSelectionPosition({
        x: rect.left + rect.width / 2,
        y: rect.top - 10
      })
    } else {
      setSelectedText('')
      setSelectionPosition(null)
    }
  }, [])

  const handleAgentRequest = useCallback(() => {
    if (selectedText) {
      setShowAgent(true)
    }
  }, [selectedText])

  const formatContent = (text: string) => {
    return text.split('\n\n').map((paragraph, index) => (
      <p key={index} className="mb-4 text-justify">
        {paragraph}
      </p>
    ))
  }

  return (
    <div className="min-h-screen bg-paper">
      {/* Header */}
      <header className="sticky top-0 bg-white/95 backdrop-blur border-b border-gray-200 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <ChevronLeft className="w-5 h-5" />
          </button>
          <h1 className="text-lg font-semibold truncate max-w-md">{fileName}</h1>
        </div>
        
        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-1 bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => setFontSize(Math.max(12, fontSize - 2))}
              className="px-2 py-1 text-sm hover:bg-white rounded"
            >
              A-
            </button>
            <span className="px-2 py-1 text-sm">{fontSize}px</span>
            <button
              onClick={() => setFontSize(Math.min(24, fontSize + 2))}
              className="px-2 py-1 text-sm hover:bg-white rounded"
            >
              A+
            </button>
          </div>
          
          <button className="p-2 hover:bg-gray-100 rounded-lg">
            <Bookmark className="w-5 h-5" />
          </button>
          <button className="p-2 hover:bg-gray-100 rounded-lg">
            <Settings className="w-5 h-5" />
          </button>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-lg"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
      </header>

      {/* Content */}
      <div className="reading-container">
        <div
          ref={contentRef}
          className="prose prose-lg max-w-none"
          style={{
            fontSize: `${fontSize}px`,
            lineHeight: lineHeight,
            fontFamily: 'Georgia, Times New Roman, serif'
          }}
          onMouseUp={handleTextSelection}
          onTouchEnd={handleTextSelection}
        >
          <TextHighlighter content={content} />
        </div>
      </div>

      {/* Selection Tooltip */}
      {selectedText && selectionPosition && (
        <div
          className="fixed z-50 bg-white border shadow-lg rounded-lg p-2 flex items-center space-x-2"
          style={{
            left: `${selectionPosition.x}px`,
            top: `${selectionPosition.y}px`,
            transform: 'translateX(-50%)'
          }}
        >
          <button
            onClick={handleAgentRequest}
            className="flex items-center space-x-1 px-3 py-1 bg-accent text-white rounded hover:bg-blue-600 transition-colors"
          >
            <HelpCircle className="w-4 h-4" />
            <span className="text-sm">Ask AI</span>
          </button>
          <button className="px-3 py-1 bg-yellow-100 text-yellow-800 rounded hover:bg-yellow-200 transition-colors">
            <span className="text-sm">Highlight</span>
          </button>
        </div>
      )}

      {/* Agent Assistant */}
      {showAgent && (
        <AgentAssistant
          selectedText={selectedText}
          context={{
            selectedText,
            surroundingText: content.slice(
              Math.max(0, content.indexOf(selectedText) - 500),
              content.indexOf(selectedText) + selectedText.length + 500
            ),
            documentTitle: fileName
          }}
          onCloseAction={() => setShowAgent(false)}
        />
      )}
    </div>
  )
}