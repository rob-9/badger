'use client'

import { useState, useEffect } from 'react'
import { MessageCircle, BookOpen, Brain, History, Lightbulb, Sparkles } from 'lucide-react'

type DemoScenario = {
  id: string
  title: string
  icon: typeof BookOpen
  bookTitle: string
  genre: string
  bookText: string[]
  interactions: {
    highlight: number | null
    question: string
    answer: string
    isProactive?: boolean  // AI-initiated vs user-initiated
    delay?: number
  }[]
}

export default function ReaderDemo() {
  const [activeTab, setActiveTab] = useState(0)
  const [interactionStep, setInteractionStep] = useState(0)
  const [isTransitioning, setIsTransitioning] = useState(false)
  const [showAnswer, setShowAnswer] = useState(false)

  const scenarios: DemoScenario[] = [
    {
      id: 'agentic-pattern',
      title: 'Pattern Detection',
      icon: Brain,
      bookTitle: 'And Then There Were None',
      genre: 'Mystery',
      bookText: [
        'Vera looked around the dinner table. Eight guests, two servants, and one host.',
        'The china figurines on the mantelpiece caught her eye—ten little soldier boys.',
        'She counted them twice. Yes, ten.',
        '"Funny," she thought, "there are only ten of us on the island."'
      ],
      interactions: [
        {
          highlight: 1,
          question: 'Why are the figurines important?',
          answer: 'The ten soldier boy figurines mirror the ten people on the island. In Agatha Christie\'s setup, this is significant symbolism—but I\'ll let you discover why as you read. Want me to bookmark this for you?',
          isProactive: false,
          delay: 1000
        },
        {
          highlight: null,
          question: 'I noticed you\'ve asked about "the number ten" three times now. You also highlighted the poem in chapter 1. Seeing the pattern?',
          answer: 'You\'re tracking something important! The poem, the figurines, and the guest count are all connected. I\'ve bookmarked these moments for you. As you keep reading, I\'ll watch for more instances of this pattern and flag them—without spoiling what happens next.',
          isProactive: true,
          delay: 1100
        }
      ]
    },
    {
      id: 'spoiler-aware',
      title: 'Spoiler Shield',
      icon: MessageCircle,
      bookTitle: 'Project Hail Mary',
      genre: 'Science Fiction',
      bookText: [
        'I don\'t know where I am.',
        'I don\'t know who I am.',
        'But I know this: Someone somewhere is in grave danger.',
        'And somehow, it\'s up to me to save them.'
      ],
      interactions: [
        {
          highlight: 2,
          question: 'Who needs to be saved?',
          answer: '⚠️ I can answer this, but it would spoil a major reveal in chapter 8. You\'re only on chapter 1. Want me to help you analyze what you know SO FAR instead?',
          isProactive: false,
          delay: 1000
        },
        {
          highlight: null,
          question: 'You\'ve highlighted 3 memory-related passages. I\'m tracking this theme for you—amnesia seems central to the mystery.',
          answer: 'Based on your reading patterns, you\'re piecing together the protagonist\'s identity. I\'ve bookmarked the key memory scenes you marked. Want me to flag similar moments as you continue reading?',
          isProactive: true,
          delay: 1200
        }
      ]
    },
    {
      id: 'context-memory',
      title: 'Memory Recall',
      icon: History,
      bookTitle: 'The Name of the Wind',
      genre: 'Fantasy',
      bookText: [
        'She reached into her pocket and pulled out a ring—iron, with a pale blue stone.',
        '"My grandmother gave this to me," she said. "It\'s supposed to protect against fae magic."',
        'Kvothe\'s eyes widened slightly.',
        '"Where did you get that stone?" he asked quietly.'
      ],
      interactions: [
        {
          highlight: 2,
          question: 'Why does Kvothe react to the ring?',
          answer: 'The pale blue stone catches his attention—in this world, such stones are rare and connected to ancient fae lore. But I notice you\'re asking about this reaction...',
          isProactive: false,
          delay: 1000
        },
        {
          highlight: null,
          question: 'Remember on page 47, you asked about "blue flames" in the fae realm? And on page 89, you highlighted the description of "moon-colored stones"?',
          answer: 'You\'re tracking something. These three moments all connect to the same ancient magic system. I\'ve linked these passages in your reading history. This pattern will matter more in chapter 12—I\'ll remind you when you get there.',
          isProactive: true,
          delay: 1300
        }
      ]
    },
    {
      id: 'theory-building',
      title: 'Theory Building',
      icon: Lightbulb,
      bookTitle: 'Recursion',
      genre: 'Science Fiction Thriller',
      bookText: [
        'The memories aren\'t real. But they feel real.',
        'Helena has spent her entire life studying memory—how it forms, how it fades.',
        'Now people are dying from memories of lives they never lived.',
        'False Memory Syndrome, they call it.'
      ],
      interactions: [
        {
          highlight: 2,
          question: 'Could these be alternate timeline memories?',
          answer: 'Interesting theory! You\'re thinking about quantum mechanics and parallel realities. The book hasn\'t confirmed this yet, but you\'re asking the right questions...',
          isProactive: false,
          delay: 1000
        },
        {
          highlight: null,
          question: 'You\'ve been building a theory about time travel for the last 6 chapters. You highlighted Helena\'s research, the chair technology, and now this.',
          answer: 'Your timeline theory is getting stronger. Based on your highlights, you\'ve connected: 1) Helena\'s memory research, 2) The mysterious chair, 3) False Memory Syndrome victims all remembering the same "dead" loved ones. You\'re onto something big. Want me to keep tracking evidence for/against your theory?',
          isProactive: true,
          delay: 1400
        }
      ]
    }
  ]

  const currentScenario = scenarios[activeTab]
  const currentInteraction = currentScenario.interactions[interactionStep]

  useEffect(() => {
    // Reset interaction step when tab changes
    setInteractionStep(0)
    setShowAnswer(false)
    setIsTransitioning(true)
    const timeout = setTimeout(() => setIsTransitioning(false), 300)
    return () => clearTimeout(timeout)
  }, [activeTab])

  useEffect(() => {
    // Auto-advance through interactions
    const interval = setInterval(() => {
      setShowAnswer(false)
      setIsTransitioning(true)

      setTimeout(() => {
        // Check if we need to move to next interaction or next tab
        if (interactionStep + 1 >= currentScenario.interactions.length) {
          // Move to next tab (left to right)
          setActiveTab((prevTab) => (prevTab + 1) % scenarios.length)
          // interactionStep will be reset by the activeTab useEffect
        } else {
          // Move to next interaction in same tab
          setInteractionStep((prev) => prev + 1)
        }
        setIsTransitioning(false)
      }, 400)
    }, 8500) // 8.5 seconds per interaction for better readability

    return () => clearInterval(interval)
  }, [interactionStep, currentScenario.interactions.length, scenarios.length])

  useEffect(() => {
    // Show answer with delay
    if (currentInteraction) {
      const timeout = setTimeout(() => {
        setShowAnswer(true)
      }, currentInteraction.delay || 800)
      return () => clearTimeout(timeout)
    }
  }, [interactionStep, currentInteraction])

  return (
    <div className="w-full h-full bg-[#1a1812] flex flex-col">
      {/* Tabs */}
      <div className="flex gap-1 p-2 bg-[#14120b]/50 border-b border-[#d4a574]/10">
        {scenarios.map((scenario, index) => {
          const Icon = scenario.icon
          return (
            <button
              key={scenario.id}
              onClick={() => setActiveTab(index)}
              className={`flex items-center gap-1.5 px-3 py-2 rounded-t text-xs font-medium transition-all duration-300 ${
                activeTab === index
                  ? 'bg-[#1a1812] text-[#d4a574] border-t-2 border-[#d4a574]'
                  : 'text-[#f7f7f4]/50 hover:text-[#f7f7f4]/80 hover:bg-[#1a1812]/50'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">{scenario.title}</span>
            </button>
          )
        })}
      </div>

      <div className="flex-1 p-6 flex flex-col overflow-hidden">
        {/* Mock Reader Header */}
        <div className="flex items-center justify-between mb-4 pb-3 border-b border-[#f7f7f4]/10">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-[#d4a574]"></div>
            <span className="text-xs text-[#f7f7f4]/60">{currentScenario.genre}</span>
          </div>
          <div className="text-xs text-[#f7f7f4]/40 truncate max-w-[200px]">{currentScenario.bookTitle}</div>
        </div>

        {/* Book Content */}
        <div className={`flex-1 mb-4 space-y-2.5 overflow-auto transition-opacity duration-300 ${
          isTransitioning ? 'opacity-50' : 'opacity-100'
        }`}>
          {currentScenario.bookText.map((line, index) => (
            <p
              key={index}
              className={`text-sm leading-relaxed transition-all duration-700 ease-out ${
                currentInteraction?.highlight === index
                  ? 'bg-gradient-to-r from-[#d4a574]/25 via-[#d4a574]/20 to-transparent px-3 py-1.5 rounded text-[#f7f7f4] shadow-sm scale-[1.01] transform'
                  : 'text-[#f7f7f4]/80 scale-100'
              }`}
            >
              {line}
            </p>
          ))}
        </div>

        {/* AI Question/Insight Box */}
        {currentInteraction && (
          <div className={`rounded-lg p-3 space-y-3 animate-in fade-in slide-in-from-bottom-2 duration-500 shadow-lg ${
            currentInteraction.isProactive
              ? 'bg-gradient-to-r from-[#d4a574]/15 to-[#14120b] border-2 border-[#d4a574]/50'
              : 'bg-[#14120b] border border-[#d4a574]/30'
          }`}>
            <div className="flex items-start gap-2">
              {currentInteraction.isProactive ? (
                <div className="relative">
                  <Sparkles className="w-4 h-4 text-[#d4a574] mt-0.5 flex-shrink-0 animate-pulse" />
                  <div className="absolute inset-0 bg-[#d4a574] blur-md opacity-40 animate-pulse"></div>
                </div>
              ) : (
                <MessageCircle className="w-4 h-4 text-[#d4a574] mt-0.5 flex-shrink-0 animate-in zoom-in duration-300" />
              )}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  {currentInteraction.isProactive && (
                    <span className="text-[9px] uppercase tracking-wider text-[#d4a574] font-semibold px-1.5 py-0.5 bg-[#d4a574]/20 rounded">
                      AI Insight
                    </span>
                  )}
                </div>
                <p className={`text-xs mb-2 animate-in fade-in duration-400 ${
                  currentInteraction.isProactive ? 'text-[#f7f7f4] font-medium' : 'text-[#f7f7f4]/90'
                }`}>
                  {currentInteraction.question}
                </p>
                <div className={`overflow-hidden transition-all duration-700 ease-out ${
                  showAnswer ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'
                }`}>
                  <div className={`rounded p-2.5 shadow-md ${
                    currentInteraction.isProactive
                      ? 'bg-[#14120b]/80 border-l-2 border-[#d4a574]'
                      : 'bg-[#1a1812] border-l-2 border-[#d4a574]/50'
                  }`}>
                    <div className="flex items-start gap-2">
                      <Brain className="w-3 h-3 text-[#d4a574] mt-0.5 flex-shrink-0" />
                      <p className="text-xs text-[#f7f7f4]/70 leading-relaxed">
                        {currentInteraction.answer}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Progress Indicator */}
        <div className="mt-3 flex items-center justify-center gap-1.5">
          {currentScenario.interactions.map((_, index) => (
            <div
              key={index}
              className={`h-1 rounded-full transition-all duration-300 ${
                index === interactionStep
                  ? 'w-6 bg-[#d4a574]'
                  : 'w-1.5 bg-[#d4a574]/30'
              }`}
            />
          ))}
        </div>
      </div>
    </div>
  )
}
