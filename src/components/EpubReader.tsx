'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
import ePub, { Book, Rendition } from 'epubjs'
import { ChevronLeft, ChevronRight, Bookmark, Settings, ZoomIn, ZoomOut } from 'lucide-react'

interface EpubReaderProps {
  epubData: ArrayBuffer
  fileName: string
  onCloseAction: () => void
}

export default function EpubReader({ epubData, fileName, onCloseAction }: EpubReaderProps) {
  const [fontSize, setFontSize] = useState(100)
  const [isReady, setIsReady] = useState(false)
  const viewerRef = useRef<HTMLDivElement>(null)
  const bookRef = useRef<Book | null>(null)
  const renditionRef = useRef<Rendition | null>(null)

  useEffect(() => {
    if (!viewerRef.current) return

    let rendition: Rendition | null = null
    let mounted = true

    const initBook = async () => {
      try {
        console.log('Initializing EPUB from ArrayBuffer')

        // Create book directly from ArrayBuffer (epub.js supports this!)
        const book = ePub(epubData)
        bookRef.current = book

        await book.ready
        console.log('Book opened successfully')

        if (!mounted || !viewerRef.current) return

        // Create rendition
        rendition = book.renderTo(viewerRef.current, {
          width: '100%',
          height: '100%',
          flow: 'paginated',
          spread: 'none',
        })

        renditionRef.current = rendition

        // Display the book
        await rendition.display()

        console.log('Book displayed!')
        console.log('ViewerRef children:', viewerRef.current?.children)
        console.log('ViewerRef HTML:', viewerRef.current?.innerHTML)

        // Check iframe after a short delay
        setTimeout(() => {
          const iframe = viewerRef.current?.querySelector('iframe')
          console.log('Iframe element:', iframe)
          if (iframe) {
            console.log('Iframe dimensions:', {
              width: iframe.style.width,
              height: iframe.style.height,
              visibility: iframe.style.visibility,
            })
            console.log('Iframe computed style:', window.getComputedStyle(iframe))
          }
        }, 500)

        setIsReady(true)

        // Save location on page change
        rendition.on('relocated', (location: any) => {
          if (location.start?.cfi) {
            localStorage.setItem(`epub-location-${fileName}`, location.start.cfi)
          }
        })

        return () => {
          rendition?.destroy()
        }
      } catch (error) {
        console.error('Error loading EPUB:', error)
      }
    }

    const cleanup = initBook()

    return () => {
      mounted = false
      cleanup.then((cleanupFn) => cleanupFn?.())
    }
  }, [epubData, fileName])

  const handleNext = useCallback(async () => {
    if (renditionRef.current) {
      await renditionRef.current.next()
    }
  }, [])

  const handlePrev = useCallback(async () => {
    if (renditionRef.current) {
      await renditionRef.current.prev()
    }
  }, [])

  // Handle keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowRight') {
        handleNext()
      } else if (e.key === 'ArrowLeft') {
        handlePrev()
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [handleNext, handlePrev])

  return (
    <div className="min-h-screen bg-paper flex flex-col">
      {/* Header */}
      <header className="bg-white/95 backdrop-blur border-b border-gray-200 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <button
            onClick={onCloseAction}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <ChevronLeft className="w-5 h-5" />
          </button>
          <h1 className="text-lg font-semibold truncate max-w-md">{fileName}</h1>
        </div>

        <div className="flex items-center space-x-2">
          {/* Font Size Controls */}
          <div className="flex items-center space-x-1 bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => setFontSize(Math.max(50, fontSize - 10))}
              className="px-2 py-1 text-sm hover:bg-white rounded flex items-center"
              title="Decrease font size"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
            <span className="px-2 py-1 text-sm">{fontSize}%</span>
            <button
              onClick={() => setFontSize(Math.min(200, fontSize + 10))}
              className="px-2 py-1 text-sm hover:bg-white rounded flex items-center"
              title="Increase font size"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
          </div>

          <button
            className="p-2 hover:bg-gray-100 rounded-lg"
            title="Bookmarks"
          >
            <Bookmark className="w-5 h-5" />
          </button>
          <button
            className="p-2 hover:bg-gray-100 rounded-lg"
            title="Settings"
          >
            <Settings className="w-5 h-5" />
          </button>
        </div>
      </header>

      {/* Reader */}
      <div className="flex-1 relative bg-paper p-8">
        <div className="max-w-5xl mx-auto" style={{ height: 'calc(100vh - 200px)' }}>
          <div
            ref={viewerRef}
            className="w-full relative bg-white"
            style={{
              height: '100%',
              border: '2px solid red', // Debug border
            }}
          />

          {/* Navigation Arrows */}
          {isReady && (
            <>
              <button
                onClick={handlePrev}
                className="fixed left-8 top-1/2 -translate-y-1/2 p-4 bg-white rounded-full shadow-lg hover:bg-gray-50 transition-all z-10"
                aria-label="Previous page"
              >
                <ChevronLeft className="w-6 h-6" />
              </button>
              <button
                onClick={handleNext}
                className="fixed right-8 top-1/2 -translate-y-1/2 p-4 bg-white rounded-full shadow-lg hover:bg-gray-50 transition-all z-10"
                aria-label="Next page"
              >
                <ChevronRight className="w-6 h-6" />
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
