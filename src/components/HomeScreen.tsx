'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { Book, Sparkles, ArrowRight, FileText, Github, Mail, FileCode } from 'lucide-react'
import ReaderDemo from './ReaderDemo'

export default function HomeScreen() {
  const words = ["doesn't spoil books like online forums do.", "knows exactly what you're thinking.", 'understands specific contexts and plot lines.', 'is your confidant for thoughts & theories.']
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
        <div className="absolute top-[5%] left-[10%] w-[250px] h-[250px] rounded-full bg-[#d9955f] opacity-[0.08] blur-[65px]"></div>
        <div className="absolute top-[15%] right-[15%] w-[220px] h-[220px] rounded-full bg-[#cd7f47] opacity-[0.09] blur-[58px]"></div>
        <div className="absolute bottom-[30%] left-[5%] w-[190px] h-[190px] rounded-full bg-[#e8a965] opacity-[0.07] blur-[54px]"></div>
        <div className="absolute top-[50%] left-[50%] w-[280px] h-[280px] rounded-full bg-[#bc8555] opacity-[0.08] blur-[70px]"></div>
        <div className="absolute bottom-[10%] right-[20%] w-[240px] h-[240px] rounded-full bg-[#d69658] opacity-[0.08] blur-[62px]"></div>
        <div className="absolute top-[35%] right-[5%] w-[210px] h-[210px] rounded-full bg-[#e8a965] opacity-[0.07] blur-[56px]"></div>
        <div className="absolute bottom-[25%] left-[40%] w-[230px] h-[230px] rounded-full bg-[#cd7f47] opacity-[0.09] blur-[60px]"></div>
        <div className="absolute top-[25%] left-[35%] w-[215px] h-[215px] rounded-full bg-[#d9955f] opacity-[0.07] blur-[56px]"></div>
      </div>

      {/* Top Icons */}
      <div className="fixed top-8 left-1/2 -translate-x-1/2 z-50 flex items-center gap-3">
        <a
          href="https://github.com/rob-9/boom"
          target="_blank"
          rel="noopener noreferrer"
          className="p-2 rounded-lg hover:bg-[#f7f7f4]/10 transition-all duration-200"
          aria-label="GitHub"
        >
          <Github className="w-4 h-4 text-[#f7f7f4]/60 hover:text-[#f7f7f4]" />
        </a>
        <a
          href="mailto:contact@boom.ai"
          className="p-2 rounded-lg hover:bg-[#f7f7f4]/10 transition-all duration-200"
          aria-label="Contact"
        >
          <Mail className="w-4 h-4 text-[#f7f7f4]/60 hover:text-[#f7f7f4]" />
        </a>
        <button
          className="p-2 rounded-lg hover:bg-[#f7f7f4]/10 transition-all duration-200 cursor-not-allowed opacity-50"
          aria-label="Application (Work in Progress)"
          disabled
        >
          <FileCode className="w-4 h-4 text-[#f7f7f4]/60" />
        </button>
      </div>

      {/* Hero Section */}
      <main className="relative z-10 px-8 min-h-screen flex items-center">
        <div className="max-w-7xl mx-auto w-full grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          {/* Left: Hero Content */}
          <div className="flex flex-col items-start text-left">
            <h1 className="text-4xl md:text-5xl font-bold mb-8 tracking-tight">
              Read Smarter.
            </h1>

            <div className="text-xl md:text-2xl text-[#f7f7f4]/80 mb-10 h-8 flex items-center gap-2">
              <span className="relative px-3 py-1.5">
                <span className="absolute inset-0 bg-gradient-to-r from-[#d4a574]/30 to-[#c9986a]/30 blur-sm rounded-lg"></span>
                <span className="relative">boom</span>
              </span>
              <div className="relative h-8 flex items-center min-w-[500px]">
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
            <div className="flex items-center gap-2.5">
              <Link
                href="/read"
                className="group px-5 py-2.5 rounded-lg bg-[#f7f7f4] text-[#14120b] hover:bg-[#f7f7f4]/90 transition-all duration-200 font-semibold text-sm flex items-center gap-1.5"
              >
                Get Started
                <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
              </Link>
              <button className="px-5 py-2.5 rounded-lg bg-[#f7f7f4]/5 hover:bg-[#f7f7f4]/10 border border-[#f7f7f4]/10 transition-all duration-200 font-semibold text-sm">
                Learn More
              </button>
            </div>
          </div>

          {/* Right: Interactive Demo */}
          <div className="relative">
            <div className="aspect-[4/3] bg-[#1a1812] rounded-xl border border-[#f7f7f4]/10 overflow-hidden shadow-2xl">
              <ReaderDemo />
            </div>
          </div>

        </div>
      </main>

      {/* Supported Formats & Links - Fixed Bottom */}
      <footer className="fixed bottom-0 left-0 right-0 z-20 pb-6 px-8">
        <div className="max-w-7xl mx-auto text-center">
          <div className="flex items-center justify-center mb-2.5">
            <div className="h-px w-8 bg-gradient-to-r from-transparent to-[#f7f7f4]/20"></div>
            <p className="text-[10px] text-[#f7f7f4]/40 uppercase tracking-wider mx-2.5">Supported Formats</p>
            <div className="h-px w-8 bg-gradient-to-l from-transparent to-[#f7f7f4]/20"></div>
          </div>

          <div className="flex items-center justify-center gap-2.5 flex-wrap text-[10px] text-[#f7f7f4]/60">
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
