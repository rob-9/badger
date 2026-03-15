'use client'

import { useState, useCallback, useEffect, useRef, forwardRef, useImperativeHandle } from 'react'
import ePub, { Book, Rendition, NavItem } from 'epubjs'
import { ChevronLeft, ChevronRight, Settings, ZoomIn, ZoomOut, List, Moon, Sun, RotateCcw, ArrowLeft, Minus, Plus } from 'lucide-react'

export interface TextSelection {
  text: string
  position: { x: number; y: number }
  pageRect: { left: number; right: number; top: number; bottom: number }
}

export interface EpubReaderHandle {
  navigateToText: (text: string) => Promise<boolean>
  getCurrentCfi: () => string | null
  navigateToCfi: (cfi: string) => Promise<void>
}

interface EpubReaderProps {
  epubData: ArrayBuffer
  fileName: string
  isIndexing?: boolean
  isIndexed?: boolean
  isChatOpen?: boolean
  sourceNavCfi?: string | null
  onCloseAction: () => void
  onTextSelect?: (selection: TextSelection) => void
  onLocationChange?: (percentage: number) => void
  onBackToReading?: () => void
}

// Book page aspect ratio (width:height)
const ASPECT_RATIO = 7 / 9

const EpubReader = forwardRef<EpubReaderHandle, EpubReaderProps>(function EpubReader({ epubData, fileName, isIndexing, isIndexed, isChatOpen, sourceNavCfi, onCloseAction, onTextSelect, onLocationChange, onBackToReading }, ref) {
  // Initialize font size from localStorage
  const [fontSize, setFontSize] = useState(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('badger-font-size')
      return saved ? parseInt(saved, 10) : 100
    }
    return 100
  })

  // Initialize zoom from localStorage
  const [zoom, setZoom] = useState(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('boom-zoom-level')
      return saved ? parseFloat(saved) : 1
    }
    return 1
  })

  const [isReady, setIsReady] = useState(false)
  const [toc, setToc] = useState<NavItem[]>([])
  const [showToc, setShowToc] = useState(() => {
    if (typeof window !== 'undefined') {
      return window.innerWidth >= 1280
    }
    return false
  })
  const [isWideScreen, setIsWideScreen] = useState(() => {
    if (typeof window !== 'undefined') {
      return window.innerWidth >= 1280
    }
    return false
  })
  const [showToolbar, setShowToolbar] = useState(() => {
    // Default to hidden on mobile
    if (typeof window !== 'undefined') {
      return window.innerWidth >= 768
    }
    return true
  })
  const [isDark, setIsDark] = useState(false)

  // Progress
  const [percentage, setPercentage] = useState(0)

  // Current chapter tracking
  const [currentHref, setCurrentHref] = useState('')

  // Page transition
  const [isFlipping, setIsFlipping] = useState(false)

  // Pan state
  const [isPanning, setIsPanning] = useState(false)
  const [panStart, setPanStart] = useState({ x: 0, y: 0 })
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 })

  // Inject dark mode CSS directly into the epub iframe
  const applyEpubTheme = useCallback((dark: boolean) => {
    if (!renditionRef.current) return
    const iframes = viewerRef.current?.querySelectorAll('iframe')
    iframes?.forEach((iframe) => {
      const doc = iframe.contentDocument
      if (!doc) return
      let style = doc.getElementById('badger-theme')
      if (!style) {
        style = doc.createElement('style')
        style.id = 'badger-theme'
        doc.head.appendChild(style)
      }
      style.textContent = dark
        ? '*, html, body { background: #1a1a1a !important; color: #d4d4d4 !important; border-color: #333 !important; } a { color: #8ab4f8 !important; }'
        : '*, html, body { background: #ffffff !important; color: #1a1a1a !important; } a { color: inherit !important; }'
    })
  }, [])

  const toggleTheme = useCallback(() => {
    const next = !isDark
    setIsDark(next)
    document.documentElement.classList.toggle('dark', next)
    localStorage.setItem('badger-theme', next ? 'dark' : 'light')
    applyEpubTheme(next)
  }, [isDark, applyEpubTheme])
  const viewerRef = useRef<HTMLDivElement>(null)
  const bookRef = useRef<Book | null>(null)
  const renditionRef = useRef<Rendition | null>(null)
  const onTextSelectRef = useRef(onTextSelect)
  useEffect(() => { onTextSelectRef.current = onTextSelect }, [onTextSelect])

  const highlightRef = useRef<HTMLElement | null>(null)
  const currentCfiRef = useRef<string | null>(null)

  const clearHighlight = useCallback(() => {
    if (highlightRef.current) {
      const mark = highlightRef.current
      const parent = mark.parentNode
      if (parent) {
        while (mark.firstChild) {
          parent.insertBefore(mark.firstChild, mark)
        }
        parent.removeChild(mark)
        parent.normalize()
      }
      highlightRef.current = null
    }
  }, [])

  // Clear highlight when sourceNavCfi is cleared (user clicked back or navigated away)
  useEffect(() => {
    if (!sourceNavCfi) clearHighlight()
  }, [sourceNavCfi, clearHighlight])

  useImperativeHandle(ref, () => ({
    getCurrentCfi: () => currentCfiRef.current,

    navigateToCfi: async (cfi: string) => {
      clearHighlight()
      if (renditionRef.current) {
        setIsFlipping(true)
        await new Promise(resolve => setTimeout(resolve, 150))
        await renditionRef.current.display(cfi)
        setIsFlipping(false)
      }
    },

    navigateToText: async (text: string): Promise<boolean> => {
      const book = bookRef.current
      const rendition = renditionRef.current
      if (!book || !rendition) return false

      clearHighlight()

      // Build candidate search keys: try progressively shorter lengths (80 → 60 → 40)
      // trimmed at word boundaries to improve match resilience across whitespace variants
      const normalized = text.replace(/\s+/g, ' ').trim()
      const buildKey = (maxLen: number): string => {
        let key = normalized.slice(0, maxLen)
        if (normalized.length > maxLen) {
          const lastSpace = key.lastIndexOf(' ')
          if (lastSpace > maxLen * 0.5) key = key.slice(0, lastSpace)
        }
        return key
      }
      const searchKeys = [buildKey(80), buildKey(60), buildKey(40)].filter(
        (k, i, arr) => k.length >= 15 && arr.indexOf(k) === i
      )

      const spineItems = (book.spine as any).spineItems || []

      for (const item of spineItems) {
        try {
          const section = book.spine.get(item.index)
          if (!section) continue

          await section.load(book.load.bind(book))
          const sectionText = (section as any).document?.body?.textContent || ''

          // Try each key length against this section
          const matchedKey = searchKeys.find(k => sectionText.includes(k))
          if (!matchedKey) continue

          // Pure fade transition
          setIsFlipping(true)
          await new Promise(resolve => setTimeout(resolve, 150))

          // Navigate to section first
          await rendition.display(item.href)

          // Try CFI-based navigation for exact page within section
          const sectionDoc = (section as any).document
          if (sectionDoc) {
            const tw = sectionDoc.createTreeWalker(sectionDoc.body, NodeFilter.SHOW_TEXT)
            let tn: Text | null
            while ((tn = tw.nextNode() as Text | null)) {
              const c = tn.textContent || ''
              const ci = c.indexOf(matchedKey)
              if (ci !== -1) {
                try {
                  const r = sectionDoc.createRange()
                  r.setStart(tn, ci)
                  r.setEnd(tn, ci + 1)
                  const cfi = (section as any).cfiFromRange(r)
                  if (cfi) await rendition.display(cfi)
                } catch { /* stay on section start */ }
                break
              }
            }
          }

          await new Promise(resolve => setTimeout(resolve, 200))
          // Highlight while still invisible
          const iframe = viewerRef.current?.querySelector('iframe')
          const doc = iframe?.contentDocument
          if (doc) {
            highlightRef.current = highlightTextInDoc(doc, matchedKey)
          }
          // Fade in with content already in position
          setIsFlipping(false)
          return true
        } catch {
          continue
        }
      }
      return false
    },
  }), [clearHighlight])

  // Initialize theme on mount
  useEffect(() => {
    const savedTheme = localStorage.getItem('badger-theme')
    if (savedTheme === 'dark') {
      setIsDark(true)
      document.documentElement.classList.add('dark')
    } else {
      setIsDark(false)
      document.documentElement.classList.remove('dark')
    }
  }, [])

  // Track wide screen for persistent TOC layout
  useEffect(() => {
    const handleResize = () => {
      setIsWideScreen(window.innerWidth >= 1280)
    }
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

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

        // Wait for spine to be loaded
        await book.loaded.spine

        const rendition = book.renderTo(viewerRef.current, {
          width: '100%',
          height: '100%',
          flow: 'paginated',
          spread: 'none',
          allowScriptedContent: true,
        })

        renditionRef.current = rendition

        // Register font, styles, and dark/light theme BEFORE first display
        rendition.themes.register('badger', {
          '@font-face': {
            'font-family': '"Cooper BT"',
            'src': `url("${window.location.origin}/fonts/cooper-bt-light.otf") format("opentype")`,
            'font-weight': '300',
            'font-style': 'normal',
          },
          'body, p, div, span, li, td, th, blockquote': {
            'font-family': '"Cooper BT", Inter, system-ui, sans-serif !important',
          },
          '::selection': { 'background': 'rgba(217, 149, 95, 0.15)', 'color': 'inherit' },
        })
        rendition.themes.select('badger')

        rendition.hooks.content.register((contents: any) => {
          const dark = document.documentElement.classList.contains('dark')
          const doc = contents.document
          if (doc) {
            let style = doc.getElementById('badger-theme')
            if (!style) {
              style = doc.createElement('style')
              style.id = 'badger-theme'
              doc.head.appendChild(style)
            }
            style.textContent = (dark
              ? '*, html, body { background: #1a1a1a !important; color: #d4d4d4 !important; border-color: #333 !important; }'
              : '*, html, body { background: #ffffff !important; color: #1a1a1a !important; }')
              + '\n.badger-source-highlight { background: rgba(217, 149, 95, 0.3); border-radius: 2px; }'
          }
          // Apply current font size to new pages
          if (renditionRef.current) {
            const saved = localStorage.getItem('badger-font-size')
            if (saved) renditionRef.current.themes.fontSize(`${saved}%`)
          }
        })

        // Now display the first page
        const savedLocation = localStorage.getItem(`epub-location-${fileName}`)

        try {
          if (savedLocation) {
            await rendition.display(savedLocation)
          } else {
            await rendition.display()
          }
        } catch (error) {
          console.error('Error displaying, clearing saved location and restarting:', error)
          localStorage.removeItem(`epub-location-${fileName}`)
          await rendition.display()
        }

        setIsReady(true)

        const spineItems = (book.spine as any).spineItems || []
        const spineLength = spineItems.length

        // Load table of contents
        const navigation = await book.loaded.navigation
        if (navigation?.toc) {
          setToc(navigation.toc)
        }

        rendition.on('relocated', (location: any) => {
          if (location.start?.cfi) {
            currentCfiRef.current = location.start.cfi
            localStorage.setItem(`epub-location-${fileName}`, location.start.cfi)
          }

          if (spineLength > 0 && location.start?.index != null) {
            const displayed = location.start.displayed || {}
            const page = displayed.page || 0
            const sectionTotal = displayed.total || 1
            const pct = (location.start.index + page / sectionTotal) / spineLength
            setPercentage(pct)
            if (onLocationChange) onLocationChange(pct)
          }
          // Track current href for TOC highlighting
          if (location.start?.href) {
            setCurrentHref(location.start.href)
          }
        })

        // Detect clicks inside the book to dismiss popup when text is deselected
        rendition.on('click', () => {
          // Check all iframes for active selection
          setTimeout(() => {
            const iframes = viewerRef.current?.querySelectorAll('iframe')
            let hasSelection = false
            iframes?.forEach((iframe) => {
              try {
                const sel = iframe.contentWindow?.getSelection()
                if (sel && !sel.isCollapsed && sel.toString().trim()) {
                  hasSelection = true
                }
              } catch (e) { /* ignore */ }
            })
            if (!hasSelection && onTextSelectRef.current) {
              onTextSelectRef.current({ text: '', position: { x: 0, y: 0 }, pageRect: { left: 0, right: 0, top: 0, bottom: 0 } })
            }
          }, 50)
        })

        // Handle text selection
        rendition.on('selected', (_cfiRange: string, contents: any) => {
          if (!onTextSelectRef.current) return

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

          onTextSelectRef.current({
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
  }, [epubData, fileName, applyEpubTheme])

  const navigate = useCallback(async (direction: 'next' | 'prev') => {
    if (!renditionRef.current || isFlipping) return
    setIsFlipping(true) // fade out current page
    setTimeout(async () => {
      // navigate while invisible
      if (direction === 'next') await renditionRef.current?.next()
      else await renditionRef.current?.prev()
      // fade in new page
      setIsFlipping(false)
    }, 150)
  }, [isFlipping])

  const handleNext = useCallback(() => navigate('next'), [navigate])
  const handlePrev = useCallback(() => navigate('prev'), [navigate])

  const handleNavigate = useCallback(async (href: string) => {
    if (!renditionRef.current || !bookRef.current) return
    try {
      const book = bookRef.current
      // Try direct display first
      const section = book.spine.get(href)
      if (section) {
        await renditionRef.current.display(href)
      } else {
        // TOC hrefs may not match spine hrefs due to path differences.
        // Fuzzy match: find spine item whose href ends with the TOC href or vice versa.
        const cleanHref = href.split('#')[0]
        const fragment = href.includes('#') ? '#' + href.split('#')[1] : ''
        const match = (book.spine as any).spineItems.find((item: any) =>
          item.href === cleanHref ||
          item.href.endsWith('/' + cleanHref) ||
          cleanHref.endsWith('/' + item.href)
        )
        if (match) {
          await renditionRef.current.display(match.href + fragment)
        } else {
          await renditionRef.current.display(href)
        }
      }
      if (window.innerWidth < 1280) setShowToc(false)
    } catch (error) {
      console.error('Error navigating to section:', error)
    }
  }, [])

  // Font size handler with persistence
  const handleFontSizeChange = useCallback((newSize: number) => {
    setFontSize(newSize)
    localStorage.setItem('badger-font-size', newSize.toString())
  }, [])

  // Apply font size to rendition
  useEffect(() => {
    if (renditionRef.current) {
      renditionRef.current.themes.fontSize(`${fontSize}%`)
    }
  }, [fontSize])

  // Zoom handlers with persistence
  const handleZoomIn = useCallback(() => {
    const newZoom = Math.min(2.0, zoom + 0.1)
    setZoom(newZoom)
    localStorage.setItem('boom-zoom-level', newZoom.toString())
  }, [zoom])

  const handleZoomOut = useCallback(() => {
    const newZoom = Math.max(0.5, zoom - 0.1)
    setZoom(newZoom)
    localStorage.setItem('boom-zoom-level', newZoom.toString())
  }, [zoom])

  const handleZoomReset = useCallback(() => {
    setZoom(1)
    setPanOffset({ x: 0, y: 0 })
    localStorage.setItem('boom-zoom-level', '1')
  }, [])

  // Pan handlers
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (zoom > 1) {
      setIsPanning(true)
      setPanStart({ x: e.clientX - panOffset.x, y: e.clientY - panOffset.y })
    }
  }, [zoom, panOffset])

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (isPanning && zoom > 1) {
      setPanOffset({
        x: e.clientX - panStart.x,
        y: e.clientY - panStart.y,
      })
    }
  }, [isPanning, panStart, zoom])

  const handleMouseUp = useCallback(() => {
    setIsPanning(false)
  }, [])

  // Handle keyboard navigation and zoom shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowRight') {
        e.preventDefault()
        ;(document.activeElement as HTMLElement)?.blur()
        handleNext()
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault()
        ;(document.activeElement as HTMLElement)?.blur()
        handlePrev()
      } else if ((e.metaKey || e.ctrlKey) && e.key === '+') {
        e.preventDefault()
        handleZoomIn()
      } else if ((e.metaKey || e.ctrlKey) && e.key === '-') {
        e.preventDefault()
        handleZoomOut()
      } else if ((e.metaKey || e.ctrlKey) && e.key === '0') {
        e.preventDefault()
        handleZoomReset()
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [handleNext, handlePrev, handleZoomIn, handleZoomOut, handleZoomReset])

  // Handle clicks outside the book (on the reader background) to dismiss popup
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (e.target === viewerRef.current?.parentElement && onTextSelectRef.current) {
        onTextSelectRef.current({ text: '', position: { x: 0, y: 0 }, pageRect: { left: 0, right: 0, top: 0, bottom: 0 } })
      }
    }

    document.addEventListener('click', handleClick)
    return () => document.removeEventListener('click', handleClick)
  }, [])

  return (
    <div className={`h-screen overflow-hidden bg-paper dark:bg-[#141414] flex flex-col transition-[margin] duration-300 ${isChatOpen ? 'min-[769px]:mr-[400px]' : ''}`}>
      {/* Header */}
      <header className="bg-white/95 dark:bg-[#1e1e1e]/95 backdrop-blur border-b border-gray-200 dark:border-[#2a2a2a] px-6 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <button
            onClick={onCloseAction}
            className="p-2 hover:bg-gray-100 dark:hover:bg-[#2a2a2a] rounded-lg transition-colors"
            aria-label="Close book"
          >
            <ChevronLeft className="w-5 h-5" />
          </button>
          <h1 className="text-lg font-semibold truncate max-w-md dark:text-[#e0e0e0]">{fileName}</h1>
          {isIndexing && (
            <span className="flex items-center gap-1.5 text-xs text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/30 px-2.5 py-1 rounded-full">
              <span className="w-1.5 h-1.5 bg-amber-500 rounded-full animate-pulse" />
              Indexing...
            </span>
          )}
          {isIndexed && !isIndexing && <ReadyBadge />}
        </div>

        <div className="flex items-center space-x-2">
          {/* Font Size Controls */}
          <div className={`flex items-center space-x-1 bg-gray-100 dark:bg-[#2a2a2a] rounded-lg p-1 transition-all duration-200 origin-right ${
            showToolbar ? 'opacity-100 scale-100' : 'opacity-0 scale-95 pointer-events-none'
          }`}>
            <button
              onClick={() => handleFontSizeChange(Math.max(60, fontSize - 10))}
              className="px-2 py-1 text-sm hover:bg-white dark:hover:bg-[#3a3a3a] rounded flex items-center dark:text-[#ccc]"
              aria-label="Decrease font size"
              title="Decrease font size"
              disabled={!showToolbar || fontSize <= 60}
            >
              <Minus className="w-4 h-4" />
            </button>
            <span className="px-2 py-1 text-xs dark:text-[#ccc] min-w-[3rem] text-center">
              {fontSize}%
            </span>
            <button
              onClick={() => handleFontSizeChange(Math.min(200, fontSize + 10))}
              className="px-2 py-1 text-sm hover:bg-white dark:hover:bg-[#3a3a3a] rounded flex items-center dark:text-[#ccc]"
              aria-label="Increase font size"
              title="Increase font size"
              disabled={!showToolbar || fontSize >= 200}
            >
              <Plus className="w-4 h-4" />
            </button>
            {fontSize !== 100 && (
              <button
                onClick={() => handleFontSizeChange(100)}
                className="px-2 py-1 text-sm hover:bg-white dark:hover:bg-[#3a3a3a] rounded flex items-center dark:text-[#ccc]"
                aria-label="Reset font size"
                title="Reset font size"
                disabled={!showToolbar}
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            )}
          </div>

          <button
            onClick={() => setShowToc(!showToc)}
            className={`p-2 hover:bg-gray-100 dark:hover:bg-[#2a2a2a] rounded-lg transition-all duration-200 origin-right ${
              showToolbar ? 'opacity-100 scale-100' : 'opacity-0 scale-95 pointer-events-none'
            } ${showToc ? 'bg-gray-100 dark:bg-[#2a2a2a]' : ''}`}
            aria-label="Table of contents"
            title="Table of Contents"
            disabled={!showToolbar}
          >
            <List className="w-5 h-5" />
          </button>

          <button
            onClick={toggleTheme}
            className={`p-2 hover:bg-gray-100 dark:hover:bg-[#2a2a2a] rounded-lg transition-all duration-200 origin-right ${
              showToolbar ? 'opacity-100 scale-100' : 'opacity-0 scale-95 pointer-events-none'
            }`}
            aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
            title={isDark ? 'Light mode' : 'Dark mode'}
            disabled={!showToolbar}
          >
            {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          </button>

          <button
            onClick={() => setShowToolbar(!showToolbar)}
            className="p-2 hover:bg-gray-100 dark:hover:bg-[#2a2a2a] rounded-lg transition-colors"
            aria-label={showToolbar ? 'Hide toolbar' : 'Show toolbar'}
            title={showToolbar ? 'Hide toolbar' : 'Show toolbar'}
          >
            <Settings className="w-5 h-5" />
          </button>
        </div>
      </header>

      {/* Progress line */}
      {isReady && (
        <div className="relative h-0.5 bg-gray-200 dark:bg-[#2a2a2a] z-40">
          <div
            className="absolute inset-y-0 left-0 bg-accent transition-all duration-300"
            style={{ width: `${percentage * 100}%` }}
          />
        </div>
      )}

      {/* Table of Contents Sidebar */}
      <div className={`fixed left-0 top-[69px] h-[calc(100vh-69px)] w-80 bg-white dark:bg-[#1e1e1e] border-r border-gray-200 dark:border-[#2a2a2a] shadow-lg z-30 overflow-y-auto transition-transform duration-300 ${
        showToc ? 'translate-x-0' : '-translate-x-full'
      }`}>
        <div className="p-4">
          <h2 className="text-lg font-semibold mb-4 dark:text-[#e0e0e0]">Table of Contents</h2>
          <div className="space-y-1">
            {toc.map((item, index) => (
              <TocItem key={index} item={item} onNavigate={handleNavigate} activeHref={currentHref} />
            ))}
          </div>
        </div>
      </div>

      {/* Reader */}
      <div
        className={`flex-1 relative bg-paper dark:bg-[#141414] p-8 flex items-center justify-center overflow-hidden transition-[margin] duration-300 ${isWideScreen && showToc ? 'ml-80' : ''}`}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{ cursor: zoom > 1 ? (isPanning ? 'grabbing' : 'grab') : 'default' }}
      >
        <div className="flex flex-col items-center">
          <div className="relative" style={{
            aspectRatio: '7 / 9',
            height: 'calc(100vh - 160px)',
            maxWidth: 'calc((100vh - 160px) * 7 / 9)',
          }}>
            {/* Back to reading — pill button, top-left of the book */}
            {sourceNavCfi && (
              <button
                onClick={onBackToReading}
                className="absolute -left-14 top-0 z-20 flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-white/80 dark:bg-[#252525]/80 backdrop-blur-sm border border-gray-200/60 dark:border-[#3a3a3a]/60 shadow-sm hover:bg-white dark:hover:bg-[#2e2e2e] hover:shadow-md transition-all duration-150 animate-fade-in"
                aria-label="Back to reading position"
                title="Back to reading position"
              >
                <ArrowLeft className="w-3.5 h-3.5 text-gray-500 dark:text-gray-400" />
                <span className="text-xs text-gray-500 dark:text-gray-400 font-medium">Return</span>
              </button>
            )}
            <div
              ref={viewerRef}
              className="w-full h-full bg-white dark:bg-[#1a1a1a] rounded-lg shadow-lg"
              style={{
                fontFamily: 'ui-serif, Georgia, Cambria, "Times New Roman", Times, serif',
                transform: `scale(${zoom}) translate(${panOffset.x / zoom}px, ${panOffset.y / zoom}px)`,
                transformOrigin: 'center center',
                opacity: isFlipping ? 0 : 1,
                transition: isFlipping ? 'opacity 0.15s ease-out' : 'opacity 0.35s ease-in',
              }}
            />
          </div>
          {/* Progress indicator */}
          {isReady && (
            <span className="mt-2 text-[0.65rem] text-gray-400 dark:text-[#555] tabular-nums">
              {Math.round(percentage * 100)}%
            </span>
          )}
        </div>

        {/* Navigation Arrows */}
        {isReady && (
          <>
            <button
              onClick={handlePrev}
              className="fixed top-1/2 -translate-y-1/2 p-2 bg-white/40 dark:bg-[#2a2a2a]/40 backdrop-blur-sm rounded-full opacity-40 hover:opacity-100 hover:bg-white/80 dark:hover:bg-[#2a2a2a]/80 transition-all z-10"
              style={{ left: isWideScreen && showToc ? '22rem' : '2rem' }}
              aria-label="Previous page"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              onClick={handleNext}
              className="fixed top-1/2 -translate-y-1/2 p-2 bg-white/40 dark:bg-[#2a2a2a]/40 backdrop-blur-sm rounded-full opacity-40 hover:opacity-100 hover:bg-white/80 dark:hover:bg-[#2a2a2a]/80 transition-all z-10"
              style={{ right: isChatOpen ? 'calc(400px + 2rem)' : '2rem' }}
              aria-label="Next page"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </>
        )}
      </div>

    </div>
  )
})

export default EpubReader

// Helper: find and highlight text in an iframe document
function highlightTextInDoc(doc: Document, searchKey: string): HTMLElement | null {
  // Try progressively shorter keys to handle text node boundaries
  for (const len of [searchKey.length, 40, 25]) {
    let key = searchKey.slice(0, Math.min(len, searchKey.length))
    if (len < searchKey.length) {
      const lastSpace = key.lastIndexOf(' ')
      if (lastSpace > len * 0.5) key = key.slice(0, lastSpace)
    }
    if (key.length < 10) continue

    const walker = doc.createTreeWalker(doc.body, NodeFilter.SHOW_TEXT)
    let node: Text | null
    while ((node = walker.nextNode() as Text | null)) {
      const content = node.textContent || ''
      const idx = content.indexOf(key)
      if (idx !== -1) {
        try {
          const range = doc.createRange()
          range.setStart(node, idx)
          range.setEnd(node, idx + key.length)
          const mark = doc.createElement('mark')
          mark.className = 'badger-source-highlight'
          range.surroundContents(mark)
          return mark
        } catch {
          continue
        }
      }
    }
  }
  return null
}

// Ready badge — shows briefly after indexing completes
function ReadyBadge() {
  const [visible, setVisible] = useState(true)
  useEffect(() => {
    const timer = setTimeout(() => setVisible(false), 3000)
    return () => clearTimeout(timer)
  }, [])
  if (!visible) return null
  return (
    <span className="flex items-center gap-1.5 text-xs text-emerald-600 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-900/30 px-2.5 py-1 rounded-full transition-opacity duration-500">
      <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full" />
      Ready
    </span>
  )
}

// TOC Item component
function TocItem({ item, onNavigate, activeHref, level = 0 }: { item: NavItem; onNavigate: (href: string) => void; activeHref?: string; level?: number }) {
  const itemBase = item.href.split('#')[0]
  const activeBase = (activeHref || '').split('#')[0]
  const isActive = activeBase && (itemBase === activeBase || activeBase.endsWith('/' + itemBase) || itemBase.endsWith('/' + activeBase))

  return (
    <div>
      <button
        onClick={() => onNavigate(item.href)}
        className={`w-full text-left px-3 py-2 rounded-lg transition-colors text-sm ${
          isActive
            ? 'bg-accent/10 text-accent-foreground font-medium'
            : 'hover:bg-gray-100 dark:hover:bg-[#2a2a2a] text-gray-800 dark:text-[#d4d4d4]'
        }`}
        style={{ paddingLeft: `${12 + level * 16}px` }}
      >
        <span>{item.label}</span>
      </button>
      {item.subitems && item.subitems.length > 0 && (
        <div>
          {item.subitems.map((subitem, index) => (
            <TocItem key={index} item={subitem} onNavigate={onNavigate} activeHref={activeHref} level={level + 1} />
          ))}
        </div>
      )}
    </div>
  )
}
