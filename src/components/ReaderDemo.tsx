'use client'

import { useState, useEffect } from 'react'
import { MessageCircle, Highlighter, Sparkles } from 'lucide-react'

export default function ReaderDemo() {
  const [step, setStep] = useState(0)
  const [isTransitioning, setIsTransitioning] = useState(false)
  const [showAnswer, setShowAnswer] = useState(false)

  const demoSteps = [
    {
      title: 'Smart Reading',
      bookText: [
        'The old man walked slowly through the foggy streets of London.',
        'He had been searching for the mysterious artifact for decades.',
        'Tonight, he finally had a lead that might change everything.'
      ],
      highlight: null,
      question: null,
      answer: null
    },
    {
      title: 'Highlight & Question',
      bookText: [
        'The old man walked slowly through the foggy streets of London.',
        'He had been searching for the mysterious artifact for decades.',
        'Tonight, he finally had a lead that might change everything.'
      ],
      highlight: 1,
      question: 'What artifact is he looking for?',
      answer: null
    },
    {
      title: 'AI Understanding',
      bookText: [
        'The old man walked slowly through the foggy streets of London.',
        'He had been searching for the mysterious artifact for decades.',
        'Tonight, he finally had a lead that might change everything.'
      ],
      highlight: 1,
      question: 'What artifact is he looking for?',
      answer: "Based on the context, the book hasn't revealed the specific artifact yet. The author is building suspense by keeping it mysterious at this point in the story."
    }
  ]

  const currentStep = demoSteps[step]

  useEffect(() => {
    const interval = setInterval(() => {
      setIsTransitioning(true)
      setShowAnswer(false)
      setTimeout(() => {
        setStep((prev) => (prev + 1) % demoSteps.length)
        setIsTransitioning(false)
      }, 300)
    }, 3500)
    return () => clearInterval(interval)
  }, [demoSteps.length])

  useEffect(() => {
    // Show answer with delay when step has an answer
    if (currentStep.answer) {
      const timeout = setTimeout(() => {
        setShowAnswer(true)
      }, 800)
      return () => clearTimeout(timeout)
    } else {
      setShowAnswer(false)
    }
  }, [step, currentStep.answer])

  return (
    <div className="w-full h-full bg-[#1a1812] p-6 flex flex-col">
      {/* Mock Reader Header */}
      <div className="flex items-center justify-between mb-6 pb-3 border-b border-[#f7f7f4]/10">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-[#d4a574]"></div>
          <span className="text-xs text-[#f7f7f4]/60">Chapter 3</span>
        </div>
        <div className="text-xs text-[#f7f7f4]/40">Reading: Mystery Novel</div>
      </div>

      {/* Book Content */}
      <div className={`flex-1 mb-4 space-y-3 transition-opacity duration-300 ${isTransitioning ? 'opacity-50' : 'opacity-100'}`}>
        {currentStep.bookText.map((line, index) => (
          <p
            key={index}
            className={`text-sm leading-relaxed transition-all duration-700 ease-out ${
              currentStep.highlight === index
                ? 'bg-gradient-to-r from-[#d4a574]/25 via-[#d4a574]/20 to-transparent px-3 py-1.5 rounded text-[#f7f7f4] shadow-sm scale-[1.02] transform'
                : 'text-[#f7f7f4]/80 scale-100'
            }`}
          >
            {line}
          </p>
        ))}
      </div>

      {/* AI Question Box */}
      {currentStep.question && (
        <div className="bg-[#14120b] rounded-lg p-3 border border-[#d4a574]/30 space-y-3 animate-in fade-in slide-in-from-bottom-2 duration-500 shadow-lg">
          <div className="flex items-start gap-2">
            <MessageCircle className="w-4 h-4 text-[#d4a574] mt-0.5 flex-shrink-0 animate-in zoom-in duration-300" />
            <div className="flex-1">
              <p className="text-xs text-[#f7f7f4]/90 mb-2 animate-in fade-in duration-400">{currentStep.question}</p>
              <div className={`overflow-hidden transition-all duration-700 ease-out ${
                showAnswer ? 'max-h-40 opacity-100' : 'max-h-0 opacity-0'
              }`}>
                {currentStep.answer && (
                  <div className="bg-[#1a1812] rounded p-2.5 border-l-2 border-[#d4a574] shadow-md transform transition-transform duration-700 ease-out">
                    <div className="flex items-start gap-2">
                      <Sparkles className="w-3 h-3 text-[#d4a574] mt-0.5 flex-shrink-0 animate-pulse" />
                      <p className="text-xs text-[#f7f7f4]/70 leading-relaxed">{currentStep.answer}</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Feature Label */}
      <div className="mt-4 text-center">
        <span className="text-[10px] uppercase tracking-wider text-[#d4a574]/60 transition-all duration-500 animate-in fade-in">{currentStep.title}</span>
      </div>
    </div>
  )
}
