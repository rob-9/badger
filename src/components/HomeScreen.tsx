'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { Book, Sparkles, ArrowRight, FileText } from 'lucide-react'

export default function HomeScreen() {
  const [currentWord, setCurrentWord] = useState(0)
  const words = ['understands context and plot lines.', "doesn't spoil like online forums.", "knows exactly what you're thinking."]

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentWord((prev) => (prev + 1) % words.length)
    }, 3000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="min-h-screen bg-[#14120b] text-[#f7f7f4] overflow-hidden relative">
      {/* Gradient Mesh Background */}
      <div className="absolute inset-0 overflow-hidden">
        {/* Multiple overlapping gradient orbs for organic mesh effect */}
        <div className="absolute top-[-20%] left-[-10%] w-[800px] h-[800px] rounded-full bg-[#f7f7f4] opacity-[0.05] blur-[150px]"></div>
        <div className="absolute top-[10%] right-[-15%] w-[900px] h-[900px] rounded-full bg-[#f7f7f4] opacity-[0.06] blur-[160px]"></div>
        <div className="absolute bottom-[-25%] left-[20%] w-[850px] h-[850px] rounded-full bg-[#f7f7f4] opacity-[0.05] blur-[155px]"></div>
        <div className="absolute top-[40%] left-[40%] w-[600px] h-[600px] rounded-full bg-[#f7f7f4] opacity-[0.04] blur-[140px]"></div>
        <div className="absolute bottom-[10%] right-[10%] w-[700px] h-[700px] rounded-full bg-[#f7f7f4] opacity-[0.05] blur-[145px]"></div>
      </div>

      {/* Hero Section */}
      <main className="relative z-10 px-8 pt-32 pb-32 min-h-screen flex items-center justify-center">
        <div className="max-w-7xl mx-auto">
          {/* Main Headline */}
          <div className="flex flex-col items-center text-center mb-32">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[#f7f7f4]/10 border border-[#f7f7f4]/20 mb-12">
              <Sparkles className="w-4 h-4 text-[#f7f7f4]" />
              <span className="text-sm text-[#f7f7f4]/80"></span>
            </div>

            <h1 className="text-6xl md:text-7xl font-bold mb-12 tracking-tight">
              Read Smarter.
            </h1>

            <div className="text-3xl md:text-4xl text-[#f7f7f4]/80 mb-16 h-12 flex items-center justify-center w-full">
              <div className="flex items-center">
                <span className="bg-[#f7f7f4]/10 px-2 rounded mr-3">boom</span>
                <div className="relative overflow-hidden h-12 flex items-center justify-center w-[700px]">
                  {words.map((word, index) => (
                    <span
                      key={index}
                      className={`absolute left-1/2 -translate-x-1/2 transition-all duration-500 whitespace-nowrap ${
                        index === currentWord
                          ? 'opacity-100 translate-y-0'
                          : index < currentWord
                          ? 'opacity-0 -translate-y-full'
                          : 'opacity-0 translate-y-full'
                      }`}
                    >
                      {word}
                    </span>
                  ))}
                </div>
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

          {/* Supported Formats & Links */}
          <div className="mt-32 text-center">
            <p className="text-sm text-[#f7f7f4]/40 uppercase tracking-wider mb-8">Supported Formats</p>
            <div className="flex items-center justify-center gap-8 flex-wrap mb-12">
              <div className="flex items-center gap-2 text-[#f7f7f4]/60">
                <FileText className="w-5 h-5" />
                <span className="font-medium">TXT</span>
              </div>
              <div className="flex items-center gap-2 text-[#f7f7f4]/60">
                <FileText className="w-5 h-5" />
                <span className="font-medium">PDF</span>
              </div>
              <div className="flex items-center gap-2 text-[#f7f7f4]/60">
                <Book className="w-5 h-5" />
                <span className="font-medium">EPUB</span>
              </div>
              <div className="flex items-center gap-2 text-[#f7f7f4]/60">
                <FileText className="w-5 h-5" />
                <span className="font-medium">DOCX</span>
              </div>
            </div>

            <div className="flex items-center justify-center gap-6 text-sm text-[#f7f7f4]/60">
              <a
                href="https://github.com/rob-9/boom"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-[#f7f7f4] transition-colors duration-200"
              >
                GitHub
              </a>
              <span>•</span>
              <Link href="/read" className="hover:text-[#f7f7f4] transition-colors duration-200">
                Get Started
              </Link>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
