'use client'

import { useEffect, useRef, useState } from 'react'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import {
  ArrowRight,
  BookOpen,
  CheckCircle,
  Sparkles,
  Github,
  Mail,
  Send,
  ChevronRight,
  ChevronLeft,
  AlertTriangle,
  FileText,
  X,
  Menu,
} from 'lucide-react'
import { GATSBY_CHAPTERS } from '@/data/gatsby'
import { supabase } from '@/lib/supabase'

gsap.registerPlugin(ScrollTrigger)

// ═══════════════════════════════════════════════════════════════════════════════
// NOISE OVERLAY
// ═══════════════════════════════════════════════════════════════════════════════

function NoiseOverlay() {
  return (
    <svg className="noise-overlay" width="100%" height="100%">
      <filter id="noise">
        <feTurbulence type="fractalNoise" baseFrequency="0.65" numOctaves="3" stitchTiles="stitch" />
      </filter>
      <rect width="100%" height="100%" filter="url(#noise)" />
    </svg>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// A. NAVBAR — "The Floating Island"
// ═══════════════════════════════════════════════════════════════════════════════

function Navbar() {
  const [morphed, setMorphed] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)

  useEffect(() => {
    const container = document.querySelector('.snap-container')
    if (!container) return
    const handleScroll = () => setMorphed(container.scrollTop > window.innerHeight * 0.05)
    container.addEventListener('scroll', handleScroll, { passive: true })
    return () => container.removeEventListener('scroll', handleScroll)
  }, [])

  const navItems = ['About', 'Features', 'Protocol']

  return (
    <nav
      className="fixed top-0 left-0 right-0 z-50 transition-all duration-300 ease-out"
      style={{
        backgroundColor: morphed || mobileOpen ? 'rgba(20, 18, 11, 0.8)' : 'transparent',
        backdropFilter: morphed || mobileOpen ? 'blur(16px) saturate(1.6)' : 'none',
        WebkitBackdropFilter: morphed || mobileOpen ? 'blur(16px) saturate(1.6)' : 'none',
        borderBottom: morphed || mobileOpen ? '1px solid rgba(242, 240, 233, 0.06)' : '1px solid transparent',
      }}
    >
      <div className="relative flex items-center justify-between px-6 md:px-16 lg:px-[4.7rem] py-2.5">
        <div className="flex items-center gap-2.5">
          <img src="/logo.png" alt="Badger" className="h-8 w-auto" />
          <span
            className="text-base font-semibold tracking-tight"
            style={{ color: '#edecec' }}
          >
            BADGER
          </span>
        </div>

        <div className="hidden sm:flex items-center gap-8 absolute left-1/2 -translate-x-1/2">
          {navItems.map((item) => (
            <a
              key={item}
              href={`#${item.toLowerCase()}`}
              className="text-sm transition-colors duration-200 hover:text-[#edecec]"
              style={{ color: 'rgba(237,236,236,0.5)' }}
            >
              {item}
            </a>
          ))}
        </div>

        <div className="flex items-center gap-3">
          <a
            href="#waitlist"
            className="text-sm text-copper transition-all duration-200 hover:opacity-80 focus:outline-none focus:ring-2 focus:ring-copper focus:ring-offset-2 focus:ring-offset-[#14120b] rounded-sm"
          >
            Join Waitlist
          </a>
          <button
            onClick={() => setMobileOpen(!mobileOpen)}
            className="sm:hidden p-1.5 rounded-md transition-colors hover:bg-white/10 focus:outline-none focus:ring-2 focus:ring-copper"
            aria-label={mobileOpen ? 'Close menu' : 'Open menu'}
            aria-expanded={mobileOpen}
          >
            {mobileOpen ? <X className="w-5 h-5 text-[#edecec]" /> : <Menu className="w-5 h-5 text-[#edecec]" />}
          </button>
        </div>
      </div>

      {/* Mobile slide-down menu */}
      <div
        className="sm:hidden overflow-hidden transition-all duration-300 ease-out"
        style={{
          maxHeight: mobileOpen ? '200px' : '0',
          opacity: mobileOpen ? 1 : 0,
        }}
      >
        <div className="flex flex-col gap-1 px-6 pb-4 pt-2">
          {navItems.map((item) => (
            <a
              key={item}
              href={`#${item.toLowerCase()}`}
              onClick={() => setMobileOpen(false)}
              className="text-sm py-2 transition-colors duration-200 hover:text-[#edecec]"
              style={{ color: 'rgba(237,236,236,0.5)' }}
            >
              {item}
            </a>
          ))}
        </div>
      </div>
    </nav>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// B. HERO SECTION — "The Opening Shot"
// ═══════════════════════════════════════════════════════════════════════════════

// ── Full text of The Great Gatsby imported from data file ──

const DEMO_CHAPTERS = GATSBY_CHAPTERS as unknown as {
  label: string
  heading: string
  pages: string[][]
  highlight: { pageIdx: number; paraIdx: number; text: string } | null
}[]

// Pre-built conversation per chapter for the agent panel
const AGENT_CONVERSATIONS: Record<number, { context: string; question: string; response: string; sources: { label: string }[] } | null> = {
  0: {
    context: '\u201Ca single green light, minute and far away\u201D',
    question: 'Why is Gatsby trembling? He\u2019s just looking at a light.',
    response: 'Because he\u2019s not looking at a light \u2014 he\u2019s performing devotion, and Nick is catching him mid-ritual.\u00B9 The trembling tells you the green light is already sacred to Gatsby before we know why. Fitzgerald gives us the body before the explanation: outstretched arms, fine tremor, a tiny light across the water. It\u2019s staged like a religious gesture witnessed by accident. That gap \u2014 between what Nick sees and what it means \u2014 is the engine of the entire novel.\u00B2',
    sources: [
      { label: 'Chapter 1 \u2014 green light on Daisy\u2019s dock' },
      { label: 'Chapter 5 \u2014 the light\u2019s meaning revealed' },
    ],
  },
  1: {
    context: '\u201Cthe eyes of Doctor T. J. Eckleburg\u201D',
    question: 'Is Fitzgerald being ironic putting a billboard ad in the middle of this wasteland?',
    response: 'Deeply.\u00B9 The billboard is commerce outliving its own purpose \u2014 the ad has faded, the oculist is gone, but the eyes remain, presiding over a landscape of industrial ruin. "No face" \u2014 only enormous retinas behind yellow spectacles \u2014 means there\u2019s surveillance without conscience, observation without judgment. Later, Wilson will mistake these eyes for God\u2019s. That confusion is Fitzgerald\u2019s sharpest move: in this world, the only omniscience left is advertising.\u00B2',
    sources: [
      { label: 'Chapter 2 \u2014 valley of ashes' },
      { label: 'Chapter 8 \u2014 Wilson\u2019s confession' },
    ],
  },
  2: {
    context: '\u201Cone of those rare smiles with a quality of eternal reassurance\u201D',
    question: 'I don\u2019t trust this. Is Nick falling for a performance here?',
    response: 'You\u2019re reading it exactly right.\u00B9 Nick is falling for it \u2014 and he half-knows it. The key phrase is "understood you just as far as you wanted to be understood." That\u2019s not intimacy, it\u2019s salesmanship. The smile is a mirror, not a window \u2014 it reflects your desires back at you. Gatsby has perfected the art of becoming what others need to see. Nick\u2019s seduction here mirrors the reader\u2019s: we want Gatsby to be real, so we accept the performance too.\u00B2',
    sources: [
      { label: 'Chapter 3 \u2014 Gatsby\u2019s introduction' },
      { label: 'Chapter 6 \u2014 Jay Gatz\u2019s invention' },
    ],
  },
  3: {
    context: '\u201CIt was a rich cream color, bright with nickel\u201D',
    question: 'This feels like Fitzgerald wants me to admire and distrust this car at the same time.',
    response: 'Exactly \u2014 the prose performs the seduction it\u2019s warning you about.\u00B9 Every detail \u2014 cream color, bright nickel, "monstrous length" \u2014 makes the car gorgeous and grotesque simultaneously. Fitzgerald is training your eye to see beauty and violence in the same object. This is the car that will kill Myrtle Wilson. He\u2019s showing you the murder weapon chapters before the crime, and making you admire it.\u00B2',
    sources: [
      { label: 'Chapter 4 \u2014 the drive to New York' },
      { label: 'Chapter 7 \u2014 the accident' },
    ],
  },
  4: {
    context: '\u201CDaisy tumbled short of his dreams\u201D',
    question: 'If Gatsby finally has Daisy in the room, why does Nick say she falls short?',
    response: 'Because Nick sees what Gatsby can\u2019t: the dream was never about Daisy.\u00B9 It was about the longing itself \u2014 five years of obsessive imagination had built something no human being could inhabit. "Tumbled short" is devastating because it\u2019s not Daisy\u2019s failure, it\u2019s the failure of reality to compete with fantasy. The green light worked precisely because it was across the water. The tragedy isn\u2019t that Gatsby can\u2019t have Daisy \u2014 it\u2019s that having her was always going to be a diminishment.\u00B2',
    sources: [
      { label: 'Chapter 5 \u2014 the reunion' },
      { label: 'Chapter 1 \u2014 the green light\u2019s promise' },
    ],
  },
  5: {
    context: '\u201CHe wanted nothing less of Daisy than that she should say: I never loved you\u201D',
    question: 'This demand feels unhinged. Does Gatsby realize what he\u2019s asking?',
    response: 'He does \u2014 and that\u2019s what makes it terrifying.\u00B9 Gatsby\u2019s project isn\u2019t romance, it\u2019s the annihilation of time. If Daisy ever loved Tom, then the five years Gatsby spent remaking himself actually happened \u2014 they\u2019re irreversible. He needs those years erased. "Can\u2019t repeat the past? Why of course you can!" isn\u2019t na\u00EFvet\u00E9 \u2014 it\u2019s the foundational delusion his entire identity depends on. He\u2019s not asking Daisy to love him. He\u2019s asking her to unmake history.\u00B2',
    sources: [
      { label: 'Chapter 6 \u2014 Gatsby\u2019s demand' },
      { label: 'Chapter 6 \u2014 Jay Gatz\u2019s invention' },
    ],
  },
  6: {
    context: '\u201CHer voice is full of money\u201D',
    question: 'Wait \u2014 does Gatsby actually understand what he\u2019s chasing here?',
    response: 'For exactly one sentence, yes.\u00B9 This is the only moment in the novel where Gatsby sees Daisy clearly \u2014 and he does it by accident. Five words collapse the entire romance into a class confession: what enchants him isn\u2019t Daisy, it\u2019s the sound of inherited wealth, the thing he can imitate but never authentically possess. Nick\u2019s reaction is instant: "That was it. I\u2019d never understood before." Then the insight passes, and Gatsby goes right back to the fantasy. He touched the truth and couldn\u2019t hold it.\u00B2',
    sources: [
      { label: 'Chapter 7 \u2014 the Plaza Hotel' },
      { label: 'Chapter 1 \u2014 Daisy\u2019s "low, thrilling voice"' },
    ],
  },
  7: {
    context: '\u201Cthe holocaust was complete\u201D',
    question: 'This word feels too big for what just happened. Is Fitzgerald being dramatic?',
    response: 'In 1925, "holocaust" meant a burnt offering \u2014 total destruction by fire, specifically a sacrifice.\u00B9 Fitzgerald is choosing a ritual word, not a dramatic one. Gatsby\u2019s death completes a pattern of self-immolation that began when James Gatz invented Jay Gatsby. He was consumed by his own dream, burned up by the intensity of wanting. Wilson pulling the trigger is almost beside the point \u2014 the real destruction happened at the Plaza, when Daisy chose Tom and the dream collapsed inward.\u00B2',
    sources: [
      { label: 'Chapter 8 \u2014 Gatsby\u2019s death' },
      { label: 'Chapter 7 \u2014 the Plaza confrontation' },
    ],
  },
  8: {
    context: '\u201CSo we beat on, boats against the current\u201D',
    question: 'Why does this ending feel like it\u2019s about me and not just Gatsby?',
    response: 'Because Fitzgerald engineered exactly that shift.\u00B9 The pronoun moves from "he" to "we" in the final lines \u2014 suddenly this isn\u2019t a story about one deluded man, it\u2019s a diagnosis of how all desire works. The metaphor cuts both ways: propelled forward by want, dragged backward by time. "Borne back ceaselessly" means the current always wins. And the sentence itself proves its own point \u2014 its rhythm pulls you forward through beauty while telling you that forward motion is an illusion. You feel the futility and the longing in the same breath.\u00B2',
    sources: [
      { label: 'Chapter 9 \u2014 final passage' },
      { label: 'Chapter 1 \u2014 the green light' },
    ],
  },
}

// Chapters that have agent conversations (for auto-cycling)
const DEMO_CHAPTER_INDICES = Object.keys(AGENT_CONVERSATIONS)
  .map(Number)
  .filter((k) => AGENT_CONVERSATIONS[k] !== null)
  .sort((a, b) => a - b)

function HeroReaderView() {
  const [demoIdx, setDemoIdx] = useState(0)
  const chapterIdx = DEMO_CHAPTER_INDICES[demoIdx] ?? 0
  const [pageIdx, setPageIdx] = useState(0)
  const [streamedChars, setStreamedChars] = useState(0)
  const [showHighlight, setShowHighlight] = useState(false)
  const [showQuestion, setShowQuestion] = useState(false)
  const [showContext, setShowContext] = useState(false)
  const [showSources, setShowSources] = useState(false)
  const chatScrollRef = useRef<HTMLDivElement>(null)

  const chapter = DEMO_CHAPTERS[chapterIdx]
  const pages = chapter.pages
  const currentPage = pages[pageIdx] || pages[0]
  const conversation = AGENT_CONVERSATIONS[chapterIdx] || null

  // Compute total pages and current absolute page for display
  const totalPages = DEMO_CHAPTERS.reduce((sum, ch) => sum + ch.pages.length, 0)
  const currentAbsPage = DEMO_CHAPTERS.slice(0, chapterIdx).reduce((sum, ch) => sum + ch.pages.length, 0) + pageIdx + 1
  const pct = Math.round((currentAbsPage / totalPages) * 100)

  // Sequential animation state machine
  useEffect(() => {
    let cancelled = false
    const timers: ReturnType<typeof setTimeout>[] = []
    const intervals: ReturnType<typeof setInterval>[] = []

    const schedule = (fn: () => void, ms: number) => {
      const t = setTimeout(() => { if (!cancelled) fn() }, ms)
      timers.push(t)
    }

    // Reset everything
    setShowHighlight(false)
    setShowQuestion(false)
    setShowContext(false)
    setShowSources(false)
    setStreamedChars(0)

    // Navigate to highlight page
    const hl = chapter.highlight
    if (hl) {
      setPageIdx(hl.pageIdx)
    } else {
      setPageIdx(0)
    }

    if (!conversation) {
      schedule(() => {
        if (!cancelled) setDemoIdx((d) => (d + 1) % DEMO_CHAPTER_INDICES.length)
      }, 3000)
      return () => { cancelled = true; timers.forEach(clearTimeout) }
    }

    // Phase 1: Reading pause → highlight fades in
    schedule(() => setShowHighlight(true), 1500)

    // Phase 2: Context quote + user question appear in agent panel
    schedule(() => setShowContext(true), 2800)
    schedule(() => setShowQuestion(true), 3600)

    // Phase 3: Stream the answer character by character
    schedule(() => {
      const total = conversation.response.length
      let chars = 0
      const interval = setInterval(() => {
        if (cancelled) { clearInterval(interval); return }
        chars += 3
        if (chars >= total) {
          setStreamedChars(total)
          clearInterval(interval)
        } else {
          setStreamedChars(chars)
        }
      }, 20)
      intervals.push(interval)
    }, 4400)

    // Phase 4: Show sources
    schedule(() => setShowSources(true), 7500)

    // Phase 5: Auto-advance to next chapter
    schedule(() => {
      if (!cancelled) setDemoIdx((d) => (d + 1) % DEMO_CHAPTER_INDICES.length)
    }, 11000)

    return () => {
      cancelled = true
      timers.forEach(clearTimeout)
      intervals.forEach(clearInterval)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [demoIdx])

  // Scroll chat to bottom when streaming
  useEffect(() => {
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTo({ top: chatScrollRef.current.scrollHeight, behavior: 'smooth' })
    }
  }, [streamedChars, showSources, showQuestion])

  // Manual chapter selection resets the auto-cycle
  const goToChapter = (idx: number) => {
    const pos = DEMO_CHAPTER_INDICES.indexOf(idx)
    if (pos !== -1) {
      setDemoIdx(pos)
    }
  }

  const goPrevPage = () => {
    if (pageIdx > 0) setPageIdx(pageIdx - 1)
  }

  const goNextPage = () => {
    if (pageIdx < pages.length - 1) setPageIdx(pageIdx + 1)
  }

  return (
    <div className="flex-1 flex flex-col bg-[#141414] min-h-0 overflow-hidden">
      {/* ─── Reader Header (mimics ~/boom EpubReader header) ─── */}
      <div className="flex items-center justify-between px-4 py-2.5 bg-[#1e1e1e]/95 border-b border-[#2a2a2a] shrink-0">
        <div className="flex items-center gap-2.5">
          <span className="text-[12px] font-semibold text-[#e0e0e0] truncate max-w-[200px]">The Great Gatsby</span>
          <span className="text-[8px] px-1.5 py-0.5 rounded-full bg-copper/15 text-copper font-medium">Ready</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="flex items-center bg-[#2a2a2a] rounded-md overflow-hidden">
            <span className="text-[8px] text-[#555] px-1.5 py-0.5">&minus;</span>
            <span className="text-[9px] text-[#888] font-medium tabular-nums px-1 py-0.5 border-x border-[#333]">100%</span>
            <span className="text-[8px] text-[#555] px-1.5 py-0.5">+</span>
          </div>
        </div>
      </div>

      {/* ─── Progress Bar ─── */}
      <div className="h-[2px] bg-[#2a2a2a] shrink-0">
        <div className="h-full bg-copper transition-all duration-300" style={{ width: `${pct}%` }} />
      </div>

      {/* ─── Main 3-panel area ─── */}
      <div className="flex-1 flex min-h-0 overflow-hidden">

        {/* ─── Left: Table of Contents ─── */}
        <div className="w-[190px] shrink-0 border-r border-[#2a2a2a] flex flex-col bg-[#1e1e1e] overflow-hidden">
          <div className="px-4 py-3 border-b border-[#2a2a2a]">
            <span className="text-[11px] font-semibold text-[#e0e0e0]">Table of Contents</span>
          </div>
          <div className="flex-1 overflow-auto py-2 px-2.5">
            {DEMO_CHAPTERS.map((ch, i) => (
              <button
                key={i}
                onClick={() => goToChapter(i)}
                className={`w-full text-left px-3 py-[7px] rounded-lg text-[11px] leading-snug transition-colors ${
                  i === chapterIdx
                    ? 'bg-copper/10 text-copper font-medium'
                    : 'text-[#d4d4d4]/50 hover:bg-[#2a2a2a] hover:text-[#d4d4d4]/70'
                }`}
              >
                {ch.label}
              </button>
            ))}
          </div>

          {/* Feature shortcuts */}
          <div className="border-t border-[#2a2a2a] px-2.5 py-2 shrink-0 space-y-0.5">
            <div className="flex items-center gap-2 px-3 py-[6px] rounded-lg text-[11px] text-[#d4d4d4]/50 cursor-default">
              <svg className="w-3.5 h-3.5 text-[#555] shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="8" r="4"/><path d="M20 20c0-4.4-3.6-8-8-8s-8 3.6-8 8"/><circle cx="4" cy="8" r="2"/><circle cx="20" cy="8" r="2"/></svg>
              Character Graph
            </div>
            <div className="flex items-center gap-2 px-3 py-[6px] rounded-lg text-[11px] text-[#d4d4d4]/50 cursor-default">
              <svg className="w-3.5 h-3.5 text-[#555] shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="2" x2="12" y2="22"/><circle cx="12" cy="6" r="2"/><circle cx="12" cy="12" r="2"/><circle cx="12" cy="18" r="2"/><line x1="14" y1="6" x2="20" y2="6"/><line x1="14" y1="12" x2="20" y2="12"/><line x1="14" y1="18" x2="20" y2="18"/></svg>
              Timeline
            </div>
            <div className="flex items-center gap-2 px-3 py-[6px] rounded-lg text-[11px] text-[#d4d4d4]/50 cursor-default">
              <svg className="w-3.5 h-3.5 text-[#555] shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/></svg>
              Bookmarks
            </div>
            <div className="flex items-center gap-2 px-3 py-[6px] rounded-lg text-[11px] text-[#d4d4d4]/50 cursor-default">
              <svg className="w-3.5 h-3.5 text-[#555] shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>
              Notes
            </div>
          </div>
        </div>

        {/* ─── Center: Book Viewer ─── */}
        <div className="flex-1 flex flex-col min-w-0 relative bg-[#141414]">
          <div className="flex-1 overflow-hidden px-6 md:px-10 py-5 flex justify-center relative">
            <div className="max-w-[420px] w-full">
              {/* Chapter heading */}
              <div className="mb-5 text-center">
                <span
                  className="text-[22px] font-light text-[#ffffff]/10 tracking-[0.15em]"
                  style={{ fontFamily: 'ui-serif, Georgia, "Times New Roman", serif' }}
                >
                  {chapter.heading}
                </span>
              </div>

              {/* Book text — overflow clipped with bottom fade */}
              {currentPage.map((para, i) => {
                const hl = chapter.highlight
                if (hl && pageIdx === hl.pageIdx && i === hl.paraIdx) {
                  const idx = para.indexOf(hl.text)
                  if (idx !== -1) {
                    const before = para.slice(0, idx)
                    const target = para.slice(idx, idx + hl.text.length)
                    const after = para.slice(idx + hl.text.length)
                    return (
                      <p
                        key={i}
                        className="text-[12px] leading-[1.9] text-[#d4d4d4]/50 mb-3.5"
                        style={{ fontFamily: 'ui-serif, Georgia, "Times New Roman", serif' }}
                      >
                        {before}
                        <span
                          style={{
                            ...getHighlightStyle(showHighlight),
                            padding: '2px 3px',
                            textUnderlineOffset: '3px',
                            textDecorationThickness: '1.5px',
                          }}
                        >
                          {target}
                        </span>
                        {after}
                      </p>
                    )
                  }
                }
                return (
                  <p
                    key={i}
                    className="text-[12px] leading-[1.9] text-[#d4d4d4]/50 mb-3.5"
                    style={{ fontFamily: 'ui-serif, Georgia, "Times New Roman", serif' }}
                  >
                    {para}
                  </p>
                )
              })}
            </div>
            {/* Bottom fade — gracefully clips any overflow */}
            <div className="absolute inset-x-0 bottom-0 h-16 bg-gradient-to-t from-[#141414] via-[#141414]/80 to-transparent pointer-events-none" />
          </div>

          {/* Page nav arrows + percentage */}
          <div className="flex items-center justify-between px-4 py-1.5 shrink-0">
            <button
              onClick={goPrevPage}
              className="p-1.5 rounded-full bg-[#2a2a2a]/40 hover:bg-[#2a2a2a]/80 transition-all opacity-50 hover:opacity-100"
            >
              <ChevronLeft className="w-3 h-3 text-[#d4d4d4]" />
            </button>
            <span className="text-[9px] text-[#555] tabular-nums">{pct}%</span>
            <button
              onClick={goNextPage}
              className="p-1.5 rounded-full bg-[#2a2a2a]/40 hover:bg-[#2a2a2a]/80 transition-all opacity-50 hover:opacity-100"
            >
              <ChevronRight className="w-3 h-3 text-[#d4d4d4]" />
            </button>
          </div>
        </div>

        {/* ─── Right: Badger Panel ─── */}
        <div className="w-[270px] shrink-0 border-l border-[#2a2a2a] flex flex-col bg-[#1e1e1e] overflow-hidden">
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-[#2a2a2a] shrink-0">
            <div className="flex items-center gap-2">
              <BookOpen className="w-4 h-4 text-[#666]" />
              <span className="text-[12px] font-semibold text-[#e0e0e0]">Badger</span>
            </div>
            <X className="w-3 h-3 text-[#666]" />
          </div>

          {/* Messages */}
          <div ref={chatScrollRef} className="flex-1 overflow-auto px-3 py-3 space-y-2.5">
            {conversation ? (
              <>
                {/* Context quote */}
                <div
                  className="overflow-hidden"
                  style={{
                    maxHeight: showContext ? '60px' : '0',
                    opacity: showContext ? 1 : 0,
                    transform: showContext ? 'translateY(0)' : 'translateY(6px)',
                    transition: 'all 500ms cubic-bezier(0.16, 1, 0.3, 1)',
                  }}
                >
                  <div className="px-2.5 py-2 bg-[#252525] rounded-xl border-l-2 border-[#444]">
                    <p
                      className="text-[10px] italic text-[#666] leading-relaxed line-clamp-2"
                      style={{ fontFamily: 'ui-serif, Georgia, serif' }}
                    >
                      {conversation.context}
                    </p>
                  </div>
                </div>

                {/* User message */}
                <div
                  className="flex justify-end"
                  style={{
                    maxHeight: showQuestion ? '80px' : '0',
                    opacity: showQuestion ? 1 : 0,
                    transform: showQuestion ? 'translateY(0)' : 'translateY(8px)',
                    transition: 'all 600ms cubic-bezier(0.16, 1, 0.3, 1)',
                  }}
                >
                  <div className="max-w-[88%] px-3 py-2 bg-copper text-[#14120b] rounded-2xl rounded-br-md">
                    <p className="text-[11px] font-medium leading-snug">{conversation.question}</p>
                  </div>
                </div>

                {/* AI response */}
                {streamedChars > 0 && (
                  <div className="max-w-[95%]">
                    <div className="px-3 py-2.5 bg-[#252525] rounded-2xl rounded-bl-md">
                      <p className="text-[11px] text-[#d4d4d4]/75 leading-[1.7]">
                        {conversation.response.slice(0, streamedChars)}
                        {streamedChars < conversation.response.length && (
                          <span className="inline-block w-[2px] h-[9px] bg-copper ml-[1px] animate-pulse rounded-full align-text-bottom" />
                        )}
                      </p>
                    </div>

                    {/* Sources */}
                    <div
                      className="mt-1.5 overflow-hidden"
                      style={{
                        maxHeight: showSources ? '100px' : '0',
                        opacity: showSources ? 1 : 0,
                        transform: showSources ? 'translateY(0)' : 'translateY(4px)',
                        transition: 'all 500ms cubic-bezier(0.16, 1, 0.3, 1)',
                      }}
                    >
                      <button className="flex items-center gap-1 text-[9px] text-[#666] hover:text-[#999] transition-colors mb-1.5">
                        <ChevronRight className="w-2.5 h-2.5 rotate-90" />
                        <span className="font-medium">{conversation.sources.length} source{conversation.sources.length > 1 ? 's' : ''}</span>
                      </button>
                      <div className="space-y-1 border-t border-[#333] pt-1.5">
                        {conversation.sources.map((src, i) => (
                          <div key={i} className="flex items-center gap-1.5 px-1 py-0.5 rounded-md hover:bg-[#2a2a2a] cursor-pointer transition-colors">
                            <span className="inline-flex items-center justify-center min-w-[1rem] h-4 text-[7px] font-semibold bg-copper/15 text-copper rounded-full px-1">
                              {i + 1}
                            </span>
                            <span className="text-[9px] text-[#aaa] font-medium truncate">{src.label}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-center px-3">
                <Sparkles className="w-4 h-4 text-[#333] mb-2" />
                <p className="text-[9px] text-[#444]">Highlight text or ask a question about this chapter.</p>
              </div>
            )}
          </div>

          {/* Input */}
          <div className="px-2.5 py-2.5 border-t border-[#2a2a2a] shrink-0">
            <div className="flex items-center gap-1.5 bg-[#252525] rounded-xl px-3 py-2 border border-[#333]">
              <span className="flex-1 text-[9px] text-[#555]">Ask a follow-up...</span>
              <div className="p-[3px] bg-copper rounded-md shrink-0">
                <Send className="w-2.5 h-2.5 text-[#14120b]" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function AppHomeScreen() {
  return (
    <div className="w-full h-full flex flex-col">
      <HeroReaderView />
    </div>
  )
}

function Hero() {
  const sectionRef = useRef<HTMLElement>(null)
  const headlineRef = useRef<HTMLHeadingElement>(null)
  const ctaRef = useRef<HTMLDivElement>(null)
  const demoRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const ctx = gsap.context(() => {
      const tl = gsap.timeline({ defaults: { ease: 'power3.out' } })

      tl.fromTo(headlineRef.current, { y: 20, opacity: 0 }, { y: 0, opacity: 1, duration: 0.9, delay: 0.15 })
        .fromTo(ctaRef.current, { y: 12, opacity: 0 }, { y: 0, opacity: 1, duration: 0.7 }, '-=0.5')
        .fromTo(demoRef.current, { y: 25, opacity: 0 }, { y: 0, opacity: 1, duration: 1.0 }, '-=0.3')
    }, sectionRef)

    return () => ctx.revert()
  }, [])

  return (
    <section
      ref={sectionRef}
      className="relative w-full overflow-hidden bg-[#14120b]"
    >
      {/* Headline + CTA */}
      <div className="px-6 md:px-16 lg:px-[5rem] pt-28 pb-12 md:pt-32 md:pb-16">
        <h1 ref={headlineRef} className="mb-12 opacity-0">
          <span className="block font-heading font-bold text-[#edecec] text-xl md:text-2xl tracking-tight leading-tight -mb-2">
            BADGER IS READING
          </span>
          <span className="block font-drama italic text-[2.5rem] md:text-[4rem] lg:text-[5.5rem] leading-[1.3] -ml-1 -mb-[0.4em] bg-gradient-to-r from-copper via-amber-300 to-copper bg-clip-text text-transparent">
            Redefined.
          </span>
        </h1>

        <div ref={ctaRef} className="opacity-0">
          <a
            href="#waitlist"
            className="inline-flex items-center gap-2.5 px-6 py-3 text-sm font-medium rounded-full transition-all duration-200 hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-copper focus:ring-offset-2 focus:ring-offset-[#14120b] active:scale-[0.98]"
            style={{ backgroundColor: '#edecec', color: '#14120b' }}
          >
            Waitlist
            <ArrowRight className="w-4 h-4" />
          </a>
        </div>
      </div>

      {/* Demo window — rounded rect background behind */}
      <div className="relative px-6 md:px-16 lg:px-28 pt-10 pb-24">
        <div
          ref={demoRef}
          className="relative max-w-6xl mx-auto opacity-0"
        >
          {/* Background rectangle — large, left edge near hero text */}
          <div className="absolute -bottom-12 -top-12 rounded-xl" style={{ backgroundColor: '#24221a', left: 'calc(-50vw + 50% + 5rem)', right: 'calc(-50vw + 50% + 5rem)' }} />
          {/* Window */}
          <div
            className="relative rounded-xl overflow-hidden border border-[#ffffff]/[0.06]"
            style={{
              boxShadow: '0 -4px 60px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.02)',
            }}
          >
          {/* Window chrome */}
          <div className="flex items-center px-4 py-3 bg-[#1e1e1e] border-b border-[#ffffff]/[0.06]">
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full bg-[#ffffff]/[0.08]" />
              <span className="w-3 h-3 rounded-full bg-[#ffffff]/[0.08]" />
              <span className="w-3 h-3 rounded-full bg-[#ffffff]/[0.08]" />
            </div>
            <span className="flex-1 text-center text-xs text-[#ffffff]/25 font-medium -ml-12">
              Badger
            </span>
          </div>
          {/* App homescreen */}
          <div className="h-[500px] md:h-[620px]">
            <AppHomeScreen />
          </div>
          </div>
        </div>
      </div>
    </section>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// C. FEATURES — Mini Reader Demos
// ═══════════════════════════════════════════════════════════════════════════════

// ── Card 1: Spoiler Shield — shows the actual spoiler warning flow ──

function SpoilerShieldDemo({ active }: { active: boolean }) {
  type Phase = 'idle' | 'highlight' | 'question' | 'warning'
  const [phase, setPhase] = useState<Phase>('idle')
  const [cycle, setCycle] = useState(0)

  const bookLines = [
    { text: 'He stretched out his arms toward the dark water, and I could have sworn ', highlight: 'he was trembling', rest: '.' },
    { text: 'A single green light, minute and far away, that might have been the end of a dock.', highlight: '', rest: '' },
    { text: 'When I looked once more for Gatsby he had vanished, and I was alone again in the unquiet darkness.', highlight: '', rest: '' },
  ]

  useEffect(() => {
    if (!active) return
    let cancelled = false
    const run = async () => {
      await wait(800)
      if (cancelled) return
      setPhase('highlight')
      await wait(1000)
      if (cancelled) return
      setPhase('question')
      await wait(1500)
      if (cancelled) return
      setPhase('warning')
      await wait(4000)
      if (cancelled) return
      setPhase('idle')
      setCycle((c) => c + 1)
    }
    run()
    return () => { cancelled = true }
  }, [active, cycle])

  return (
    <div className="bg-[#1e1c16] rounded-4xl border border-cream/[0.06] p-6 md:p-8 h-full flex flex-col overflow-hidden">
      <h3 className="font-heading font-bold text-cream text-lg mb-2">Spoiler Shield</h3>
      <p className="text-cream/50 text-sm mb-5 leading-relaxed">
        Never get spoiled while Googling ever again.
      </p>

      {/* Mini reader */}
      <div className="flex-1 bg-[#1a1a1a] rounded-2xl overflow-hidden flex flex-col">
        {/* Reader header */}
        <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/[0.06]">
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-copper" />
            <span className="text-[9px] text-white/40">Literary Fiction</span>
          </div>
          <span className="text-[9px] text-white/30 truncate ml-2">The Great Gatsby</span>
        </div>

        {/* Book text */}
        <div className="flex-1 px-3 py-2.5 space-y-1.5 overflow-hidden">
          {bookLines.map((line, i) => {
            const isHl = i === 0 && (phase === 'highlight' || phase === 'question' || phase === 'warning')
            return (
              <p
                key={i}
                className="text-[10px] leading-[1.7]"
                style={{
                  fontFamily: 'Georgia, serif',
                  color: 'rgba(212,212,212,0.6)',
                }}
              >
                <span>{line.text}</span>
                {line.highlight && (
                  <span style={getHighlightStyle(isHl)}>
                    {line.highlight}
                  </span>
                )}
                <span>{line.rest}</span>
              </p>
            )
          })}
        </div>

        {/* User question bubble */}
        <div
          className="mx-3 transition-all duration-300 overflow-hidden"
          style={{
            maxHeight: phase === 'question' || phase === 'warning' ? '60px' : '0',
            opacity: phase === 'question' || phase === 'warning' ? 1 : 0,
            marginBottom: phase === 'question' || phase === 'warning' ? '6px' : '0',
          }}
        >
          <div className="flex justify-end">
            <div className="px-2.5 py-1.5 bg-copper rounded-2xl rounded-br-sm">
              <p className="text-[10px] text-[#14120b] font-medium">Why does Nick notice Gatsby&apos;s trembling before anything else?</p>
            </div>
          </div>
        </div>

        {/* AI spoiler warning */}
        <div
          className="mx-3 mb-3 transition-all duration-500 overflow-hidden"
          style={{
            maxHeight: phase === 'warning' ? '120px' : '0',
            opacity: phase === 'warning' ? 1 : 0,
          }}
        >
          <div className="px-2.5 py-2 bg-[#252525] rounded-2xl rounded-bl-sm border-l-2 border-amber-500/60">
            <div className="flex items-start gap-1.5">
              <AlertTriangle className="w-3 h-3 text-amber-400 mt-0.5 shrink-0" />
              <p className="text-[10px] text-[#d4d4d4]/80 leading-[1.6]">
                The trembling is doing important narrative work, but the full answer involves <span className="text-copper font-medium">chapters 4–5</span>. You&apos;re on chapter 1. For now, notice what Nick can&apos;t explain yet — the intensity feels disproportionate to a distant light.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Card 2: Pattern Detection — shows AI finding connected motifs ──

function PatternDetectionDemo({ active }: { active: boolean }) {
  type Phase = 'idle' | 'highlight1' | 'highlight2' | 'highlight3' | 'connect'
  const [phase, setPhase] = useState<Phase>('idle')
  const [cycle, setCycle] = useState(0)

  const bookLines = [
    { text: 'The ', highlight: 'eyes of Doctor T. J. Eckleburg', rest: ' are blue and gigantic \u2014 they look out of no face.' },
    { text: 'In the library, a stout man with ', highlight: 'owl-eyed spectacles', rest: ' was examining the books with wonder.' },
    { text: 'He stretched out his arms toward the dark water, ', highlight: 'watching the green light', rest: ' at the end of a dock.' },
  ]

  const activeHighlights = {
    idle: [] as number[],
    highlight1: [0],
    highlight2: [1],
    highlight3: [2],
    connect: [0, 1, 2],
  }

  useEffect(() => {
    if (!active) return
    let cancelled = false
    const run = async () => {
      await wait(600)
      if (cancelled) return
      setPhase('highlight1')
      await wait(1200)
      if (cancelled) return
      setPhase('highlight2')
      await wait(1200)
      if (cancelled) return
      setPhase('highlight3')
      await wait(1200)
      if (cancelled) return
      setPhase('connect')
      await wait(4500)
      if (cancelled) return
      setPhase('idle')
      setCycle((c) => c + 1)
    }
    run()
    return () => { cancelled = true }
  }, [active, cycle])

  const highlighted = activeHighlights[phase]

  return (
    <div className="bg-[#1e1c16] rounded-4xl border border-cream/[0.06] p-6 md:p-8 h-full flex flex-col overflow-hidden">
      <h3 className="font-heading font-bold text-cream text-lg mb-2">Pattern Detection</h3>
      <p className="text-cream/50 text-sm mb-5 leading-relaxed">
        Track motifs and recurring themes of discussion.
      </p>

      {/* Mini reader */}
      <div className="flex-1 bg-[#1a1a1a] rounded-2xl overflow-hidden flex flex-col">
        <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/[0.06]">
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-copper" />
            <span className="text-[9px] text-white/40">Literary Fiction</span>
          </div>
          <span className="text-[9px] text-white/30 truncate ml-2">The Great Gatsby</span>
        </div>

        {/* Book text with progressive highlights */}
        <div className="flex-1 px-3 py-3 space-y-2 overflow-hidden">
          {bookLines.map((line, i) => {
            const isHl = highlighted.includes(i)
            return (
              <p key={i} className="text-[11px] leading-[1.8]" style={{ fontFamily: 'Georgia, serif' }}>
                <span className="text-[#d4d4d4]/60">{line.text}</span>
                <span style={getHighlightStyle(isHl)}>
                  {line.highlight}
                </span>
                <span className="text-[#d4d4d4]/60">{line.rest}</span>
              </p>
            )
          })}
        </div>

        {/* AI pattern insight */}
        <div
          className="mx-3 mb-3 transition-all duration-500 overflow-hidden"
          style={{
            maxHeight: phase === 'connect' ? '100px' : '0',
            opacity: phase === 'connect' ? 1 : 0,
          }}
        >
          <div className="bg-gradient-to-r from-copper/[0.08] to-transparent border border-copper/20 rounded-xl px-3 py-2">
            <div className="flex items-center gap-1.5 mb-1">
              <Sparkles className="w-2.5 h-2.5 text-copper" />
              <span className="text-[8px] uppercase tracking-widest text-copper font-semibold">Pattern Found</span>
            </div>
            <p className="text-[10px] text-[#d4d4d4]/70 leading-[1.6]">
              Fitzgerald threads <span className="text-copper font-medium">disembodied eyes</span> through these chapters — Eckleburg&apos;s billboard, the owl-eyed man, Gatsby&apos;s vigil. Each represents a different kind of seeing: commercial, intellectual, romantic.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Card 3: Memory Recall — shows AI connecting highlights across pages ──

function MemoryRecallDemo({ active }: { active: boolean }) {
  type Phase = 'idle' | 'reading' | 'question' | 'recall'
  const [phase, setPhase] = useState<Phase>('idle')
  const [cycle, setCycle] = useState(0)
  const [streamedChars, setStreamedChars] = useState(0)

  const aiResponse = 'Yes \u2014 and it\u2019s deliberate. You highlighted Tom in chapter 1: "a sturdy straw-haired man" with "cruel body" and "arrogant eyes." His power is inherited, worn carelessly. Now Gatsby is an "elegant young roughneck" \u2014 strength made graceful by effort. Fitzgerald is encoding the novel\u2019s entire class argument into how these two men occupy physical space: Tom\u2019s body dominates, Gatsby\u2019s performs.'

  useEffect(() => {
    if (!active) return
    let cancelled = false
    const run = async () => {
      setStreamedChars(0)
      await wait(600)
      if (cancelled) return
      setPhase('reading')
      await wait(1200)
      if (cancelled) return
      setPhase('question')
      await wait(1500)
      if (cancelled) return
      setPhase('recall')
      // Stream the response
      const total = aiResponse.length
      let chars = 0
      const interval = setInterval(() => {
        if (cancelled) { clearInterval(interval); return }
        chars += 3
        if (chars >= total) {
          setStreamedChars(total)
          clearInterval(interval)
        } else {
          setStreamedChars(chars)
        }
      }, 20)
      await wait(4500)
      if (cancelled) return
      clearInterval(interval)
      setPhase('idle')
      setCycle((c) => c + 1)
    }
    run()
    return () => { cancelled = true }
  }, [active, cycle])

  return (
    <div className="bg-[#1e1c16] rounded-4xl border border-cream/[0.06] p-6 md:p-8 h-full flex flex-col overflow-hidden">
      <h3 className="font-heading font-bold text-cream text-lg mb-2">Memory Recall</h3>
      <p className="text-cream/50 text-sm mb-5 leading-relaxed">
        Easily recall every highlight and note.
      </p>

      {/* Mini reader with chat */}
      <div className="flex-1 bg-[#1a1a1a] rounded-2xl overflow-hidden flex flex-col">
        <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/[0.06]">
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-copper" />
            <span className="text-[9px] text-white/40">Literary Fiction</span>
          </div>
          <span className="text-[9px] text-white/30 truncate ml-2">The Great Gatsby</span>
        </div>

        {/* Book text */}
        <div className="px-3 py-2.5">
          <p className="text-[11px] leading-[1.8] text-[#d4d4d4]/60" style={{ fontFamily: 'Georgia, serif' }}>
            I was looking at an{' '}
            <span
              className="transition-all duration-500"
              style={{
                color: phase !== 'idle' ? '#d4d4d4' : 'rgba(212,212,212,0.6)',
                backgroundColor: phase !== 'idle' ? 'rgba(217,149,95,0.15)' : 'transparent',
                borderRadius: '3px',
                padding: '1px 3px',
                textDecorationLine: phase !== 'idle' ? 'underline' : 'none',
                textDecorationStyle: 'solid' as const,
                textDecorationColor: 'rgba(217,149,95,0.4)',
                textUnderlineOffset: '2px',
              }}
            >
              elegant young roughneck
            </span>, a year or two over thirty.
          </p>
        </div>

        {/* Chat area */}
        <div className="flex-1 px-3 space-y-2 overflow-hidden">
          {/* User question */}
          <div
            className="flex justify-end transition-all duration-300"
            style={{
              opacity: phase === 'question' || phase === 'recall' ? 1 : 0,
              transform: phase === 'question' || phase === 'recall' ? 'translateY(0)' : 'translateY(4px)',
            }}
          >
            <div className="px-2.5 py-1.5 bg-copper rounded-2xl rounded-br-sm">
              <p className="text-[10px] text-[#14120b] font-medium">Is Fitzgerald drawing a contrast between Tom and Gatsby&apos;s physicality?</p>
            </div>
          </div>

          {/* AI response with streaming + sources */}
          <div
            className="transition-all duration-400"
            style={{
              opacity: phase === 'recall' ? 1 : 0,
              transform: phase === 'recall' ? 'translateY(0)' : 'translateY(4px)',
            }}
          >
            <div className="px-2.5 py-2 bg-[#252525] rounded-2xl rounded-bl-sm">
              <p className="text-[10px] text-[#d4d4d4]/80 leading-[1.65]">
                {aiResponse.slice(0, streamedChars)}
                {phase === 'recall' && streamedChars < aiResponse.length && (
                  <span className="inline-block w-[2px] h-[10px] bg-copper ml-[1px] animate-pulse rounded-full align-text-bottom" />
                )}
              </p>
            </div>

            {/* Source citations */}
            {streamedChars >= aiResponse.length && (
              <div className="mt-1.5 pl-1 space-y-0.5">
                <div className="flex items-center gap-1 mb-1">
                  <ChevronRight className="w-2.5 h-2.5 text-white/30 rotate-90" />
                  <span className="text-[8px] text-white/30 font-medium">3 sources</span>
                </div>
                {['Ch.1 — Tom\'s physical description', 'Ch.3 — "elegant young roughneck"', 'Ch.3 — Gatsby\'s smile'].map((src, i) => (
                  <div key={i} className="flex items-center gap-1.5">
                    <span className="inline-flex items-center justify-center w-3.5 h-3.5 text-[7px] font-bold bg-copper/20 text-copper rounded-full shrink-0">
                      {i + 1}
                    </span>
                    <span className="text-[8px] text-white/40">{src}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Input bar */}
        <div className="px-3 py-2 border-t border-white/[0.04]">
          <div className="flex items-center gap-1.5 bg-[#252525] rounded-lg px-2 py-1.5 border border-white/[0.06]">
            <span className="flex-1 text-[8px] text-white/20">Ask a follow-up...</span>
            <div className="p-[3px] bg-copper rounded-md shrink-0">
              <Send className="w-2 h-2 text-[#14120b]" />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Utility ──

function wait(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function getHighlightStyle(active: boolean): React.CSSProperties {
  return {
    background: 'linear-gradient(to right, rgba(217,149,95,0.15) 100%, transparent 100%)',
    backgroundSize: active ? '100% 100%' : '0% 100%',
    backgroundRepeat: 'no-repeat',
    transition: 'background-size 800ms ease-out, color 400ms ease-out, text-decoration-color 400ms ease-out',
    color: active ? '#d4d4d4' : 'rgba(212,212,212,0.6)',
    borderRadius: '3px',
    padding: '1px 3px',
    textDecorationLine: 'underline',
    textDecorationStyle: 'solid',
    textDecorationColor: active ? 'rgba(217,149,95,0.4)' : 'transparent',
    textUnderlineOffset: '2px',
  }
}

function useInView(ref: React.RefObject<HTMLElement | null>, threshold = 0.3) {
  const [inView, setInView] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const root = document.querySelector('.snap-container')
    const obs = new IntersectionObserver(
      ([entry]) => setInView(entry.isIntersecting),
      { root, threshold },
    )
    obs.observe(el)
    return () => obs.disconnect()
  }, [ref, threshold])
  return inView
}

function Features() {
  const sectionRef = useRef<HTMLElement>(null)
  const visible = useInView(sectionRef, 0.15)

  return (
    <section ref={sectionRef} id="features" className="relative py-20 px-8 md:px-16">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8">
          <h2 className="font-drama italic text-4xl md:text-6xl lg:text-7xl mb-4 text-copper">
            Capabilities
          </h2>
          <p className="text-cream/50 max-w-lg mx-auto text-base leading-relaxed">
            Badger understands your context, respects your progress, and helps you make meaningful connections.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 md:grid-cols-2 lg:auto-rows-[480px]">
          <SpoilerShieldDemo active={visible} />
          <PatternDetectionDemo active={visible} />
          <MemoryRecallDemo active={visible} />
        </div>
      </div>
    </section>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// D. PHILOSOPHY — "The Manifesto"
// ═══════════════════════════════════════════════════════════════════════════════

function Philosophy() {
  return (
    <section
      id="about"
      className="relative min-h-screen flex items-center px-8 md:px-16 bg-charcoal overflow-hidden"
    >
      {/* Parallax texture */}
      <div className="absolute inset-0 opacity-[0.07]">
        <img
          src="/bookshelf-bg.jpg"
          alt=""
          className="w-full h-full object-cover"
        />
      </div>
      {/* Top fade */}
      <div className="absolute inset-x-0 top-0 h-32 bg-gradient-to-b from-charcoal to-transparent z-[1]" />
      {/* Bottom fade */}
      <div className="absolute inset-x-0 bottom-0 h-32 bg-gradient-to-t from-charcoal to-transparent z-[1]" />

      <div className="relative z-10 max-w-5xl mx-auto">
        <p className="text-cream/40 text-lg md:text-2xl leading-relaxed mb-8">
          Most reading apps simply organize your library, track pages read, and
          gamify reading streaks. <em>They&apos;re not much better than physical books.</em>
        </p>
        <p className="text-cream text-3xl md:text-5xl lg:text-6xl leading-[1.1]">
          Badger{' '}
          <span className="font-drama italic text-copper">improves understanding</span>
          {'  '}— an entirely new experience for readers.
        </p>
      </div>
    </section>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// E. PROTOCOL — "Sticky Stacking Archive"
// ═══════════════════════════════════════════════════════════════════════════════

// ── Protocol Visual 1: Upload — file formats dropping in ──

function UploadVisual({ active }: { active: boolean }) {
  type Phase = 'idle' | 'hover' | 'click' | 'selected' | 'processing' | 'ready'
  const [phase, setPhase] = useState<Phase>('idle')
  const [progress, setProgress] = useState(0)
  const [cycle, setCycle] = useState(0)

  const cursorPos = {
    idle:       { x: 75, y: 80 },
    hover:      { x: 52, y: 62 },
    click:      { x: 52, y: 62 },
    selected:   { x: 52, y: 62 },
    processing: { x: 52, y: 62 },
    ready:      { x: 52, y: 62 },
  }

  useEffect(() => {
    if (!active) return
    let cancelled = false
    const run = async () => {
      setProgress(0)
      setPhase('idle')
      await wait(500)
      if (cancelled) return
      setPhase('hover')
      await wait(700)
      if (cancelled) return
      setPhase('click')
      await wait(200)
      if (cancelled) return
      setPhase('selected')
      await wait(800)
      if (cancelled) return
      setPhase('processing')
      let p = 0
      const interval = setInterval(() => {
        if (cancelled) { clearInterval(interval); return }
        p += 2
        if (p >= 100) { setProgress(100); clearInterval(interval) }
        else setProgress(p)
      }, 25)
      await wait(1800)
      if (cancelled) return
      clearInterval(interval)
      setProgress(100)
      setPhase('ready')
      await wait(3000)
      if (cancelled) return
      setCycle((c) => c + 1)
    }
    run()
    return () => { cancelled = true }
  }, [cycle, active])

  const showCursor = phase === 'idle' || phase === 'hover' || phase === 'click'
  const pos = cursorPos[phase]
  const btnActive = phase === 'hover' || phase === 'click'

  return (
    <div className="w-[320px] min-h-[220px] bg-[#1a1a1a] rounded-xl overflow-hidden border border-copper/15 relative flex flex-col">
      <div className="flex items-center gap-1.5 px-3 py-1.5 border-b border-copper/10 bg-[#1e1e1e]">
        <BookOpen className="w-3 h-3 text-copper" />
        <span className="text-[9px] text-copper/70 font-medium">Library</span>
        <span className="ml-auto text-[8px] text-copper/40">3 books</span>
      </div>

      <div className="relative px-4 py-4 overflow-hidden flex-1 flex flex-col justify-center">
        <div className="flex items-end gap-2 mb-3">
          {[0, 1, 2].map((i) => (
            <div key={i} className="w-7 rounded-sm bg-copper/[0.06] border border-copper/10" style={{ height: `${24 + i * 5}px` }} />
          ))}
          <div className="w-7 h-6 rounded-sm border border-dashed border-copper/15 flex items-center justify-center">
            <span className="text-copper/30 text-[9px]">+</span>
          </div>
        </div>

        <div className="flex justify-center">
          <div
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[9px] font-medium transition-all duration-200"
            style={{
              backgroundColor: btnActive ? 'rgba(217,149,95,0.2)' : 'rgba(217,149,95,0.1)',
              color: '#d9955f',
              boxShadow: btnActive ? '0 0 0 1px rgba(217,149,95,0.3)' : 'none',
              transform: phase === 'click' ? 'scale(0.95)' : 'scale(1)',
            }}
          >
            <FileText className="w-3 h-3" />
            Upload Book
          </div>
        </div>

        {(phase === 'selected' || phase === 'processing' || phase === 'ready') && (
          <div className="absolute inset-0 bg-[#1a1a1a]/95 flex flex-col items-center justify-center gap-2.5 px-4 overflow-hidden">
            <div className="w-9 h-11 rounded-lg flex items-center justify-center border border-copper/25 bg-copper/[0.08]">
              <FileText className="w-4 h-4 text-copper" />
            </div>
            <span className="text-[9px] text-copper/60 font-medium">gatsby.epub</span>
            {(phase === 'processing' || phase === 'ready') && (
              <div className="w-full max-w-[130px]">
                <div className="h-1 bg-copper/[0.08] rounded-full overflow-hidden">
                  <div
                    className="h-full bg-copper rounded-full transition-all duration-100"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <span className="block text-center text-[8px] text-copper/40 mt-1">
                  {phase === 'ready' ? 'Indexed & ready' : `Analyzing... ${progress}%`}
                </span>
              </div>
            )}
            {phase === 'ready' && (
              <div className="flex items-center gap-1 px-2 py-0.5 bg-copper/10 border border-copper/20 rounded-full">
                <CheckCircle className="w-2.5 h-2.5 text-copper" />
                <span className="text-[8px] text-copper font-medium">Book ready</span>
              </div>
            )}
          </div>
        )}

        {showCursor && (
          <div
            className="absolute pointer-events-none z-10 transition-all duration-500 ease-out"
            style={{ left: `${pos.x}%`, top: `${pos.y}%` }}
          >
            <svg width="14" height="18" viewBox="0 0 14 18" fill="none" className="drop-shadow-lg">
              <path d="M0 0L12.5 10H5.5L2.5 17.5L0 0Z" fill="white" fillOpacity={phase === 'click' ? 0.95 : 0.8} />
            </svg>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Protocol Visual 2: Reader — highlight + quick ask popup ──

function ReaderVisual({ active }: { active: boolean }) {
  const [showHighlight, setShowHighlight] = useState(false)
  const [showPopup, setShowPopup] = useState(false)
  const [selectedPrompt, setSelectedPrompt] = useState(false)

  useEffect(() => {
    if (!active) return
    let cancelled = false
    const run = async () => {
      await wait(800)
      if (cancelled) return
      setShowHighlight(true)
      await wait(900)
      if (cancelled) return
      setShowPopup(true)
      await wait(1200)
      if (cancelled) return
      setSelectedPrompt(true)
      await wait(3000)
      if (cancelled) return
      setShowHighlight(false)
      setShowPopup(false)
      setSelectedPrompt(false)
      await wait(600)
      if (cancelled) return
      run()
    }
    run()
    return () => { cancelled = true }
  }, [active])

  return (
    <div className="w-[320px] min-h-[220px] bg-[#1a1a1a] rounded-xl overflow-hidden border border-white/[0.06]">
      {/* Mini header */}
      <div className="flex items-center gap-1.5 px-3 py-1.5 border-b border-white/[0.06] bg-[#1e1e1e]">
        <ChevronLeft className="w-3 h-3 text-white/30" />
        <span className="text-[9px] text-white/50 font-medium">The Great Gatsby</span>
        <div className="ml-auto flex items-center gap-1 bg-[#2a2a2a] rounded px-1 py-0.5">
          <span className="text-[8px] text-white/30">100%</span>
        </div>
      </div>

      {/* Reading progress */}
      <div className="h-[2px] bg-[#2a2a2a]"><div className="h-full w-[35%] bg-copper rounded-r-full" /></div>

      {/* Book text */}
      <div className="px-3 py-2.5 relative">
        <p className="text-[10px] leading-[1.85] text-white/50" style={{ fontFamily: 'Georgia, serif' }}>
          He stretched out his arms toward the dark water, and I could have sworn{' '}
          <span style={getHighlightStyle(showHighlight)}>
            he was trembling
          </span>.
          A single green light, minute and far away.
        </p>

        {/* Quick ask popup */}
        <div
          className="mt-2 bg-[#1e1e1e] rounded-lg border border-[#2a2a2a] overflow-hidden transition-all duration-300"
          style={{
            maxHeight: showPopup ? '80px' : '0',
            opacity: showPopup ? 1 : 0,
            boxShadow: showPopup ? '0 6px 20px rgba(0,0,0,0.4)' : 'none',
          }}
        >
          <div className="px-2.5 py-2">
            <div className="flex items-center gap-1 mb-1.5">
              <Sparkles className="w-2 h-2 text-copper" />
              <span className="text-[7px] uppercase tracking-widest text-white/30">Quick Ask</span>
            </div>
            <div className="flex flex-wrap gap-1">
              {['Why does this matter?', 'Feels deliberate — why?', 'Seen this before?'].map((p, i) => (
                <span
                  key={i}
                  className="text-[8px] px-1.5 py-[2px] rounded-full transition-all duration-150"
                  style={{
                    backgroundColor: i === 1 && selectedPrompt ? 'rgba(217,149,95,0.15)' : '#2a2a2a',
                    color: i === 1 && selectedPrompt ? '#d9955f' : 'rgba(255,255,255,0.4)',
                    boxShadow: i === 1 && selectedPrompt ? '0 0 0 1px rgba(217,149,95,0.3)' : 'none',
                  }}
                >
                  {p}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Page nav */}
      <div className="flex items-center justify-between px-3 py-1.5 border-t border-white/[0.04]">
        <ChevronLeft className="w-2.5 h-2.5 text-white/20" />
        <span className="text-[8px] text-white/20 font-mono">42 of 318</span>
        <ChevronRight className="w-2.5 h-2.5 text-white/20" />
      </div>
    </div>
  )
}

// ── Protocol Visual 3: Chat — streaming AI response with sources ──

function ChatVisual({ active }: { active: boolean }) {
  const [streamChars, setStreamChars] = useState(0)
  const [showSources, setShowSources] = useState(false)
  const response = 'The green light is deliberately ambiguous at this point. Nick doesn\'t know what Gatsby reaches for. Its meaning accrues retroactively as the novel unfolds.'

  useEffect(() => {
    if (!active) return
    let cancelled = false
    const run = async () => {
      setStreamChars(0)
      setShowSources(false)
      await wait(500)
      // Stream
      const total = response.length
      let chars = 0
      const interval = setInterval(() => {
        if (cancelled) { clearInterval(interval); return }
        chars += 3
        if (chars >= total) { setStreamChars(total); clearInterval(interval) }
        else setStreamChars(chars)
      }, 20)
      await wait(2500)
      if (cancelled) return
      clearInterval(interval)
      setStreamChars(total)
      setShowSources(true)
      await wait(4000)
      if (cancelled) return
      run()
    }
    run()
    return () => { cancelled = true }
  }, [active])

  return (
    <div className="w-[320px] min-h-[220px] bg-[#1e1e1e] rounded-xl overflow-hidden border border-white/[0.06]">
      {/* Chat header */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/[0.06]">
        <div className="flex items-center gap-1.5">
          <BookOpen className="w-3 h-3 text-copper" />
          <span className="text-[10px] font-semibold text-white/70">Badger</span>
        </div>
        <X className="w-2.5 h-2.5 text-white/30" />
      </div>

      <div className="px-3 py-2.5 space-y-2">
        {/* Context quote */}
        <div className="px-2 py-1.5 bg-[#252525] rounded-lg border-l-2 border-copper/40">
          <p className="text-[8px] italic text-white/40" style={{ fontFamily: 'Georgia, serif' }}>
            &quot;a single green light, minute and far away&quot;
          </p>
        </div>

        {/* User bubble */}
        <div className="flex justify-end">
          <div className="px-2.5 py-1.5 bg-copper rounded-2xl rounded-br-sm">
            <p className="text-[9px] text-[#14120b] font-medium">What does the green light symbolize?</p>
          </div>
        </div>

        {/* AI response streaming */}
        <div className="px-2.5 py-2 bg-[#252525] rounded-2xl rounded-bl-sm">
          <p className="text-[9px] text-white/70 leading-[1.65]">
            {response.slice(0, streamChars)}
            {streamChars < response.length && (
              <>
                <span className="inline-block w-[2px] h-[9px] bg-copper ml-[1px] animate-pulse rounded-full align-text-bottom" />
                <span className="invisible">{response.slice(streamChars)}</span>
              </>
            )}
          </p>
        </div>

        {/* Sources */}
        <div
          className="pl-1 transition-opacity duration-400"
          style={{
            opacity: showSources ? 1 : 0,
          }}
        >
          <div className="flex items-center gap-1 mb-1">
            <ChevronRight className="w-2.5 h-2.5 text-white/20 rotate-90" />
            <span className="text-[7px] text-white/20 font-medium">3 sources</span>
          </div>
          {['Ch.1 — green light on dock', 'Ch.5 — Daisy\'s dock revealed', 'Ch.9 — final image'].map((src, i) => (
            <div key={i} className="flex items-center gap-1.5 mb-0.5">
              <span className="inline-flex items-center justify-center w-3 h-3 text-[6px] font-bold bg-copper/20 text-copper rounded-full shrink-0">
                {i + 1}
              </span>
              <span className="text-[7px] text-white/30">{src}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Input */}
      <div className="px-3 py-1.5 border-t border-white/[0.06]">
        <div className="flex items-center gap-1.5 bg-[#252525] rounded-lg px-2 py-1.5 border border-white/[0.06]">
          <span className="flex-1 text-[8px] text-white/20">Ask a follow-up...</span>
          <div className="p-[3px] bg-copper rounded-md shrink-0">
            <Send className="w-2 h-2 text-[#14120b]" />
          </div>
        </div>
      </div>
    </div>
  )
}

const PROTOCOL_STEPS = [
  {
    number: '01',
    title: 'Upload Your Book',
    description: 'Drop in any EPUB, PDF, TXT, or DOCX. Your library stays private and encrypted. Badger reads your book cover-to-cover, building a deep understanding.',
    visual: UploadVisual,
  },
  {
    number: '02',
    title: 'Read & Highlight',
    description: 'A clean, distraction-free reader. Select any passage to ask a question. Quick Ask prompts appear instantly — no typing required.',
    visual: ReaderVisual,
  },
  {
    number: '03',
    title: 'Get AI Insights',
    description: 'An AI that has read your book answers instantly — with sources, context, and zero spoilers. Build theories with evidence from across the text.',
    visual: ChatVisual,
  },
]

function Protocol() {
  const sectionRef = useRef<HTMLElement>(null)
  const visible = useInView(sectionRef, 0.15)

  return (
    <section
      ref={sectionRef}
      id="protocol"
      className="relative py-24 md:py-32 px-8 md:px-16"
    >
      <div className="max-w-5xl mx-auto w-full">
        <div className="text-center mb-8">
          <h2 className="font-drama italic text-4xl md:text-6xl lg:text-7xl text-copper">
            How It Works
          </h2>
        </div>

        <div className="flex flex-col gap-8">
          {PROTOCOL_STEPS.map((step) => {
            const Visual = step.visual
            return (
              <div
                key={step.number}
                className="w-full bg-[#1e1c16] rounded-3xl border border-cream/[0.06] p-8 md:p-12 flex flex-col md:flex-row items-start gap-8 md:gap-16"
                style={{ boxShadow: '0 8px 40px rgba(0,0,0,0.2)' }}
              >
                <div className="flex-1 self-center">
                  <span className="font-mono text-sm text-copper/60">{step.number}</span>
                  <h3 className="font-heading font-bold text-cream text-2xl md:text-4xl mt-2 mb-4 tracking-tight">
                    {step.title}
                  </h3>
                  <p className="text-cream/50 text-base md:text-lg max-w-md leading-relaxed">
                    {step.description}
                  </p>
                </div>
                <div className="flex-shrink-0 flex justify-center self-center overflow-hidden md:max-h-[260px]">
                  <Visual active={visible} />
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// F. GET STARTED CTA
// ═══════════════════════════════════════════════════════════════════════════════

function GetStarted() {
  const [email, setEmail] = useState('')
  const [status, setStatus] = useState<'idle' | 'submitting' | 'success' | 'error'>('idle')
  const [errorMsg, setErrorMsg] = useState('')

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setErrorMsg('')

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
    if (!emailRegex.test(email)) {
      setErrorMsg('Invalid Email.')
      setStatus('error')
      return
    }

    setStatus('submitting')
    try {
      const { error } = await supabase.from('emails').insert([{ email }])
      if (error) {
        if (error.code === '23505') {
          setErrorMsg('Already on the list.')
        } else {
          setErrorMsg('Something went wrong.')
        }
        setStatus('error')
        return
      }
      setStatus('success')
    } catch {
      setErrorMsg('Something went wrong.')
      setStatus('error')
    }
  }

  return (
    <section id="waitlist" className="relative py-28 md:py-36 px-8 md:px-16">
      <div className="max-w-lg mx-auto text-center">
        <h2 className="font-heading font-bold text-4xl md:text-6xl tracking-tight mb-8 pb-2 bg-gradient-to-r from-copper via-amber-300 to-copper bg-clip-text text-transparent">
          Try Badger.
        </h2>

        {status === 'success' ? (
          <div className="flex items-center justify-center gap-2 text-copper text-sm font-medium animate-fade-in-up">
            <CheckCircle className="w-4 h-4" />
            You&apos;re on the list.
          </div>
        ) : (
          <div className="max-w-sm mx-auto">
            <form onSubmit={handleSubmit} noValidate className="flex items-center gap-2">
              <input
                type="email"
                value={email}
                onChange={(e) => { setEmail(e.target.value); setErrorMsg('') }}
                placeholder="you@example.com"
                autoComplete="off"
                aria-invalid={status === 'error'}
                aria-describedby={errorMsg ? 'waitlist-error' : undefined}
                className="flex-1 px-4 py-2.5 bg-transparent border border-cream/10 rounded-xl text-sm text-cream placeholder:text-cream/25 focus:outline-none focus:border-copper/50 transition-colors [&:-webkit-autofill]:[-webkit-text-fill-color:#f2f0e9] [&:-webkit-autofill]:[font-family:inherit] [&:-webkit-autofill]:[-webkit-box-shadow:0_0_0_1000px_#14120b_inset] [&:-webkit-autofill]:[transition:background-color_9999s_ease-in-out_0s]"
              />
              <button
                type="submit"
                disabled={status === 'submitting'}
                className="px-5 py-2.5 bg-cream text-charcoal text-sm font-semibold rounded-xl hover:bg-cream/90 transition-all shrink-0 disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-copper focus:ring-offset-2 focus:ring-offset-[#14120b] active:scale-[0.98]"
              >
                {status === 'submitting' ? '...' : 'Join'}
              </button>
            </form>
            {errorMsg && (
              <p id="waitlist-error" role="alert" className="text-xs text-red-400 mt-2 text-left pl-1">{errorMsg}</p>
            )}
          </div>
        )}
      </div>
    </section>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// G. FOOTER
// ═══════════════════════════════════════════════════════════════════════════════

function Footer() {
  return (
    <footer className="bg-charcoal rounded-t-2xl px-8 md:px-16 pt-16 pb-10">
      <div className="max-w-6xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-12 mb-16">
          {/* Brand */}
          <div className="md:col-span-2">
            <div className="flex items-center gap-2.5 mb-4">
              <img src="/logo.png" alt="Badger" className="h-11 w-auto" />
              <span className="text-xl font-bold text-cream">Badger</span>
            </div>
            <p className="text-cream/40 text-sm max-w-xs leading-relaxed">
              Your intelligent reading buddy.
            </p>
          </div>

          {/* Navigation */}
          <div>
            <h4 className="font-mono text-[10px] uppercase tracking-[0.25em] text-cream/30 mb-4">Navigate</h4>
            <div className="space-y-2.5">
              {['About', 'Features', 'Protocol'].map((item) => (
                <a
                  key={item}
                  href={`#${item.toLowerCase()}`}
                  className="block text-sm text-cream/50 hover:text-cream/80 transition-colors lift-hover"
                >
                  {item}
                </a>
              ))}
              <a
                href="#waitlist"
                className="block text-sm text-copper hover:text-copper-light transition-colors lift-hover"
              >
                Waitlist
              </a>
            </div>
          </div>

          {/* Connect */}
          <div>
            <h4 className="font-mono text-[10px] uppercase tracking-[0.25em] text-cream/30 mb-4">Connect</h4>
            <div className="space-y-2.5">
              <a
                href="https://github.com/boomread"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 text-sm text-cream/50 hover:text-cream/80 transition-colors lift-hover"
              >
                <Github className="w-4 h-4" />
                GitHub
              </a>
              <a
                href="mailto:robert.ji888@gmail.com"
                className="flex items-center gap-2 text-sm text-cream/50 hover:text-cream/80 transition-colors lift-hover"
              >
                <Mail className="w-4 h-4" />
                Contact
              </a>
            </div>
          </div>
        </div>

        </div>
    </footer>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN EXPORT
// ═══════════════════════════════════════════════════════════════════════════════

export default function HomeScreen() {
  const [mounted, setMounted] = useState(false)
  useEffect(() => { setMounted(true) }, [])

  // Smooth-scroll for all # anchor links (native anchors don't smooth-scroll inside the custom scroll container)
  useEffect(() => {
    const container = document.querySelector('.snap-container')
    if (!container) return
    const handleClick = (e: Event) => {
      const link = (e.target as HTMLElement).closest('a[href^="#"]')
      if (!link) return
      const id = link.getAttribute('href')?.slice(1)
      if (!id) return
      const target = document.getElementById(id)
      if (!target) return
      e.preventDefault()
      container.scrollTo({ top: target.getBoundingClientRect().top + container.scrollTop, behavior: 'smooth' })
    }
    container.addEventListener('click', handleClick)
    return () => container.removeEventListener('click', handleClick)
  }, [])

  return (
    <div className="snap-container bg-charcoal text-cream overflow-x-hidden relative" style={{ height: '100vh', overflowY: 'auto', visibility: mounted ? 'visible' : 'hidden' }}>
      <NoiseOverlay />
      <Navbar />
      <Hero />
      <Philosophy />
      <Features />
      <Protocol />
      <GetStarted />
      <Footer />
      {/* Bottom vignette */}
      <div className="fixed inset-x-0 bottom-0 h-28 bg-gradient-to-t from-black/40 to-transparent pointer-events-none z-40" />
    </div>
  )
}
