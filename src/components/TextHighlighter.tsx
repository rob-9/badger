'use client'

import { useState, useCallback } from 'react'

interface TextHighlighterProps {
  content: string
}

interface Highlight {
  id: string
  start: number
  end: number
  color: string
  text: string
}

export default function TextHighlighter({ content }: TextHighlighterProps) {
  const [highlights, setHighlights] = useState<Highlight[]>([])

  const addHighlight = useCallback((start: number, end: number, color: string = 'yellow') => {
    const id = Date.now().toString()
    const text = content.slice(start, end)
    
    setHighlights(prev => [...prev, { id, start, end, color, text }])
  }, [content])

  const removeHighlight = useCallback((id: string) => {
    setHighlights(prev => prev.filter(h => h.id !== id))
  }, [])

  const renderHighlightedText = () => {
    if (highlights.length === 0) {
      return formatContent(content)
    }

    const sortedHighlights = [...highlights].sort((a, b) => a.start - b.start)
    const parts: (string | JSX.Element)[] = []
    let lastIndex = 0

    for (const highlight of sortedHighlights) {
      if (highlight.start > lastIndex) {
        parts.push(content.slice(lastIndex, highlight.start))
      }
      
      parts.push(
        <span
          key={highlight.id}
          className={`highlight bg-${highlight.color}-200 cursor-pointer`}
          onClick={() => removeHighlight(highlight.id)}
          title="Click to remove highlight"
        >
          {highlight.text}
        </span>
      )
      
      lastIndex = highlight.end
    }

    if (lastIndex < content.length) {
      parts.push(content.slice(lastIndex))
    }

    return parts
  }

  const formatContent = (text: string) => {
    return text.split('\n\n').map((paragraph, index) => (
      <p key={index} className="mb-4 text-justify">
        {paragraph}
      </p>
    ))
  }

  return (
    <div>
      {renderHighlightedText()}
    </div>
  )
}