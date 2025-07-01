import { useState, useCallback, useEffect } from 'react'

interface TextSelection {
  text: string
  range: Range | null
  position: { x: number; y: number } | null
}

export function useTextSelection() {
  const [selection, setSelection] = useState<TextSelection>({
    text: '',
    range: null,
    position: null
  })

  const handleSelection = useCallback(() => {
    const windowSelection = window.getSelection()
    
    if (windowSelection && windowSelection.toString().trim()) {
      const text = windowSelection.toString().trim()
      const range = windowSelection.getRangeAt(0)
      const rect = range.getBoundingClientRect()
      
      setSelection({
        text,
        range,
        position: {
          x: rect.left + rect.width / 2,
          y: rect.top - 10
        }
      })
    } else {
      setSelection({
        text: '',
        range: null,
        position: null
      })
    }
  }, [])

  const clearSelection = useCallback(() => {
    setSelection({
      text: '',
      range: null,
      position: null
    })
    
    if (window.getSelection) {
      window.getSelection()?.removeAllRanges()
    }
  }, [])

  useEffect(() => {
    document.addEventListener('mouseup', handleSelection)
    document.addEventListener('touchend', handleSelection)
    
    return () => {
      document.removeEventListener('mouseup', handleSelection)
      document.removeEventListener('touchend', handleSelection)
    }
  }, [handleSelection])

  return {
    selectedText: selection.text,
    selectionRange: selection.range,
    selectionPosition: selection.position,
    clearSelection
  }
}