'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
import ePub, { Book, Rendition, NavItem } from 'epubjs'
import { ChevronLeft, ChevronRight, Bookmark, Settings, ZoomIn, ZoomOut, List } from 'lucide-react'

export interface TextSelection {
  text: string
  position: { x: number; y: number }
  pageRect: { left: number; right: number; top: number; bottom: number }
}

interface EpubReaderProps {
  epubData: ArrayBuffer
  fileName: string
  onCloseAction: () => void
  onTextSelect?: (selection: TextSelection) => void
}

// Book page aspect ratio (width:height)
const ASPECT_RATIO = 7 / 9

export default function EpubReader({ epubData, fileName, onCloseAction, onTextSelect }: EpubReaderProps) {
  const [fontSize, setFontSize] = useState(100)
  const [isReady, setIsReady] = useState(false)
  const [toc, setToc] = useState<NavItem[]>([])
  const [showToc, setShowToc] = useState(false)
  const viewerRef = useRef<HTMLDivElement>(null)
  const bookRef = useRef<Book | null>(null)
  const renditionRef = useRef<Rendition | null>(null)

  // Initialize book
  useEffect(() => {
    if (!viewerRef.current) return

    let mounted = true

    const initBook = async () => {
      try {
        const book = ePub(epubData)
        bookRef.current = book

        await book.ready

        if (!mounted || !viewerRef.current) return

        // Wait for spine to be parsed
        await book.spine.load(book.packaging)

        if (!book.spine || book.spine.length === 0) {
          console.error('Book spine is empty')
          return
        }

        const rendition = book.renderTo(viewerRef.current, {
          width: '100%',
          height: '100%',
          flow: 'paginated',
          spread: 'none',
        })

        renditionRef.current = rendition

        // Clear any invalid saved location first
        const savedLocation = localStorage.getItem(`epub-location-${fileName}`)

        try {
          if (savedLocation) {
            await rendition.display(savedLocation)
          } else {
            await rendition.display()
          }
        } catch (error) {
          console.error('Error displaying, clearing saved location and restarting:', error)
          // Clear invalid saved location
          localStorage.removeItem(`epub-location-${fileName}`)
          // Start from beginning
          await rendition.display()
        }

        setIsReady(true)

        // Load table of contents
        const navigation = book.navigation
        if (navigation?.toc) {
          setToc(navigation.toc)
        }

        rendition.on('relocated', (location: any) => {
          if (location.start?.cfi) {
            localStorage.setItem(`epub-location-${fileName}`, location.start.cfi)
          }
        })

        // Custom highlight styling
        rendition.themes.default({
          '::selection': {
            'background': 'rgba(59, 130, 246, 0.15)',
            'color': 'inherit'
          }
        })

        // Handle text selection
        rendition.on('selected', (cfiRange: string, contents: any) => {
          if (!onTextSelect) return

          const selection = contents.window.getSelection()
          if (!selection || selection.isCollapsed) return

          const text = selection.toString().trim()
          if (!text) return

          // Get position of selection
          const range = selection.getRangeAt(0)
          const rect = range.getBoundingClientRect()

          // Get iframe and viewer positions
          const iframe = viewerRef.current?.querySelector('iframe')
          const iframeRect = iframe?.getBoundingClientRect() || { left: 0, top: 0 }
          const viewerRect = viewerRef.current?.getBoundingClientRect() || { left: 0, top: 0, right: 0, bottom: 0 }

          onTextSelect({
            text,
            position: {
              x: iframeRect.left + rect.left + rect.width / 2,
              y: iframeRect.top + rect.top,
            },
            pageRect: {
              left: viewerRect.left,
              right: viewerRect.right,
              top: viewerRect.top,
              bottom: viewerRect.bottom,
            },
          })
        })
      } catch (error) {
        console.error('Error loading EPUB:', error)
      }
    }

    initBook()

    return () => {
      mounted = false
      if (renditionRef.current) {
        renditionRef.current.destroy()
        renditionRef.current = null
      }
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

  const handleNavigate = useCallback(async (href: string) => {
    if (renditionRef.current) {
      try {
        await renditionRef.current.display(href)
        setShowToc(false)
      } catch (error) {
        console.error('Error navigating to section:', error)
        // If navigation fails, try going to start of book
        try {
          await renditionRef.current.display()
          setShowToc(false)
        } catch (fallbackError) {
          console.error('Failed to recover from navigation error:', fallbackError)
        }
      }
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
            onClick={() => setShowToc(!showToc)}
            className={`p-2 hover:bg-gray-100 rounded-lg ${showToc ? 'bg-gray-100' : ''}`}
            title="Table of Contents"
          >
            <List className="w-5 h-5" />
          </button>
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

      {/* Table of Contents Sidebar */}
      {showToc && (
        <div className="fixed left-0 top-[73px] h-[calc(100vh-73px)] w-80 bg-white border-r border-gray-200 shadow-lg z-30 overflow-y-auto">
          <div className="p-4">
            <h2 className="text-lg font-semibold mb-4">Table of Contents</h2>
            <div className="space-y-1">
              {toc.map((item, index) => (
                <TocItem key={index} item={item} onNavigate={handleNavigate} />
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Reader */}
      <div className="flex-1 relative bg-paper p-8 flex items-center justify-center overflow-hidden">
        <div
          ref={viewerRef}
          className="relative bg-white rounded-lg shadow-lg"
          style={{
            aspectRatio: '7 / 9',
            height: 'calc(100vh - 140px)',
            maxWidth: 'calc((100vh - 140px) * 7 / 9)',
            fontFamily: 'ui-serif, Georgia, Cambria, "Times New Roman", Times, serif',
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
  )
}

// TOC Item component
function TocItem({ item, onNavigate, level = 0 }: { item: NavItem; onNavigate: (href: string) => void; level?: number }) {
  return (
    <div>
      <button
        onClick={() => onNavigate(item.href)}
        className="w-full text-left px-3 py-2 hover:bg-gray-100 rounded-lg transition-colors text-sm"
        style={{ paddingLeft: `${12 + level * 16}px` }}
      >
        <span className="text-gray-800">{item.label}</span>
      </button>
      {item.subitems && item.subitems.length > 0 && (
        <div>
          {item.subitems.map((subitem, index) => (
            <TocItem key={index} item={subitem} onNavigate={onNavigate} level={level + 1} />
          ))}
        </div>
      )}
    </div>
  )
}
