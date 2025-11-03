import { useState, useCallback, useEffect, RefObject } from 'react'

interface TextSelection {
  text: string
  range: Range | null
  position: { x: number; y: number } | null
}

export function useTextSelection(targetRef: RefObject<HTMLElement>) {
  const [selection, setSelection] = useState<TextSelection>({
    text: '',
    range: null,
    position: null
  })

  const handleSelection = useCallback(() => {
    const windowSelection = window.getSelection()

    if (windowSelection && windowSelection.toString().trim()) {
      // Check if selection is within the target element
      const range = windowSelection.getRangeAt(0)
      if (targetRef.current && targetRef.current.contains(range.commonAncestorContainer)) {
        const text = windowSelection.toString().trim()
        const rect = range.getBoundingClientRect()

        setSelection({
          text,
          range,
          position: {
            x: rect.left + rect.width / 2,
            y: rect.top - 10
          }
        })
      }
    } else {
      setSelection({
        text: '',
        range: null,
        position: null
      })
    }
  }, [targetRef])

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
    const element = targetRef.current
    if (!element) return

    element.addEventListener('mouseup', handleSelection)
    element.addEventListener('touchend', handleSelection)

    return () => {
      element.removeEventListener('mouseup', handleSelection)
      element.removeEventListener('touchend', handleSelection)
    }
  }, [handleSelection, targetRef])

  return {
    selectedText: selection.text,
    selectionRange: selection.range,
    selectionPosition: selection.position,
    clearSelection
  }
}