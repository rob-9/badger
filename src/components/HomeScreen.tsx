'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { Book, Sparkles, ArrowRight, FileText, Github, Mail, FileCode } from 'lucide-react'

export default function HomeScreen() {
  const words = ['understands contexts and plot lines.', "doesn't spoil books like forums do.", "knows exactly what you're thinking."]
  const [currentWord, setCurrentWord] = useState(0)
  const [previousWord, setPreviousWord] = useState(words.length - 1)

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentWord((prev) => {
        setPreviousWord(prev)
        return (prev + 1) % words.length
      })
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

      {/* Top Icons */}
      <div className="fixed top-12 left-1/2 -translate-x-1/2 z-50 flex items-center gap-4">
        <a
          href="https://github.com/rob-9/boom"
          target="_blank"
          rel="noopener noreferrer"
          className="p-2 rounded-lg hover:bg-[#f7f7f4]/10 transition-all duration-200"
          aria-label="GitHub"
        >
          <Github className="w-5 h-5 text-[#f7f7f4]/60 hover:text-[#f7f7f4]" />
        </a>
        <a
          href="mailto:contact@boom.ai"
          className="p-2 rounded-lg hover:bg-[#f7f7f4]/10 transition-all duration-200"
          aria-label="Contact"
        >
          <Mail className="w-5 h-5 text-[#f7f7f4]/60 hover:text-[#f7f7f4]" />
        </a>
        <Link
          href="/read"
          className="p-2 rounded-lg hover:bg-[#f7f7f4]/10 transition-all duration-200"
          aria-label="Application"
        >
          <FileCode className="w-5 h-5 text-[#f7f7f4]/60 hover:text-[#f7f7f4]" />
        </Link>
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
              <span className="relative px-4 py-2">
                <span className="absolute inset-0 bg-gradient-to-r from-[#d4a574]/30 to-[#c9986a]/30 blur-sm rounded-lg"></span>
                <span className="relative">boom</span>
              </span>
              <div className="relative h-12 flex items-center min-w-[700px]">
                {words.map((word, index) => {
                  const isActive = index === currentWord
                  const isPrevious = index === previousWord

                  return (
                    <span
                      key={index}
                      className={`absolute left-0 transition-all duration-500 whitespace-nowrap ${
                        isActive
                          ? 'opacity-100 translate-y-0'
                          : isPrevious
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
                Get Started
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
