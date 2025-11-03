'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { Book, Sparkles, ArrowRight, FileText } from 'lucide-react'

export default function HomeScreen() {
  const [currentWord, setCurrentWord] = useState(0)
  const words = ['understands context and plot lines.', "doesn't spoil books like online forums.", "knows exactly what you're thinking."]

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentWord((prev) => (prev + 1) % words.length)
    }, 3000)
    return () => clearInterval(interval)
  }, [words])

  return (
    <div className="min-h-screen bg-[#14120b] text-[#f7f7f4] overflow-hidden relative">
      {/* Gradient Mesh Background - Scattered Orbs */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-[5%] left-[10%] w-[400px] h-[400px] rounded-full bg-[#d9955f] opacity-[0.08] blur-[100px]"></div>
        <div className="absolute top-[15%] right-[15%] w-[350px] h-[350px] rounded-full bg-[#cd7f47] opacity-[0.09] blur-[90px]"></div>
        <div className="absolute bottom-[30%] left-[5%] w-[300px] h-[300px] rounded-full bg-[#e8a965] opacity-[0.07] blur-[85px]"></div>
        <div className="absolute top-[50%] left-[50%] w-[450px] h-[450px] rounded-full bg-[#bc8555] opacity-[0.08] blur-[110px]"></div>
        <div className="absolute bottom-[10%] right-[20%] w-[380px] h-[380px] rounded-full bg-[#d69658] opacity-[0.08] blur-[95px]"></div>
        <div className="absolute top-[35%] right-[5%] w-[320px] h-[320px] rounded-full bg-[#e8a965] opacity-[0.07] blur-[88px]"></div>
        <div className="absolute bottom-[25%] left-[40%] w-[360px] h-[360px] rounded-full bg-[#cd7f47] opacity-[0.09] blur-[92px]"></div>
        <div className="absolute top-[25%] left-[35%] w-[330px] h-[330px] rounded-full bg-[#d9955f] opacity-[0.07] blur-[87px]"></div>
      </div>

      {/* Hero Section */}
      <main className="relative z-10 px-8 min-h-screen flex items-center justify-center">
        <div className="max-w-7xl mx-auto w-full">
          {/* Main Headline */}
          <div className="flex flex-col items-center text-center">
            <h1 className="text-6xl md:text-7xl font-bold mb-12 tracking-tight">
              Read Smarter.
            </h1>

            <div className="text-3xl md:text-4xl text-[#f7f7f4]/80 mb-16 h-12 flex items-center justify-center w-full px-4 gap-3 pl-32">
              <span className="bg-gradient-to-br from-[#d4a574]/20 to-[#c9986a]/20 px-3 py-1 rounded-lg border border-[#d4a574]/30">boom</span>
              <div className="relative h-12 flex items-center min-w-[700px]">
                {words.map((word, index) => {
                  const isActive = index === currentWord
                  // Items that have already been shown should slide up
                  const isPast = index < currentWord

                  return (
                    <span
                      key={index}
                      className={`absolute left-0 transition-all duration-500 whitespace-nowrap ${
                        isActive
                          ? 'opacity-100 translate-y-0'
                          : isPast
                          ? 'opacity-0 -translate-y-full'
                          : 'opacity-0 translate-y-full'
                      }`}
                    >
                      {word}
                    </span>
                  )
                })}
              </div>
            </div>

            {/* CTA Buttons */}
            <div className="flex items-center gap-4">
              <Link
                href="/read"
                className="group px-8 py-4 rounded-lg bg-[#f7f7f4] text-[#14120b] hover:bg-[#f7f7f4]/90 transition-all duration-200 font-semibold text-lg flex items-center gap-2"
              >
                Try boom
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </Link>
              <button className="px-8 py-4 rounded-lg bg-[#f7f7f4]/5 hover:bg-[#f7f7f4]/10 border border-[#f7f7f4]/10 transition-all duration-200 font-semibold text-lg">
                Learn More
              </button>
            </div>
          </div>

        </div>
      </main>

      {/* Supported Formats & Links - Fixed Bottom */}
      <footer className="fixed bottom-0 left-0 right-0 z-20 pb-12 px-8">
        <div className="max-w-7xl mx-auto text-center">
          <div className="flex items-center justify-center mb-4">
            <div className="h-px w-12 bg-gradient-to-r from-transparent to-[#f7f7f4]/20"></div>
            <p className="text-sm text-[#f7f7f4]/40 uppercase tracking-wider mx-4">Supported Formats</p>
            <div className="h-px w-12 bg-gradient-to-l from-transparent to-[#f7f7f4]/20"></div>
          </div>

          <div className="flex items-center justify-center gap-4 flex-wrap text-sm text-[#f7f7f4]/60">
            <span className="font-medium">TXT</span>
            <span>•</span>
            <span className="font-medium">PDF</span>
            <span>•</span>
            <span className="font-medium">EPUB</span>
            <span>•</span>
            <span className="font-medium">DOCX</span>
          </div>
        </div>
      </footer>
    </div>
  )
}
