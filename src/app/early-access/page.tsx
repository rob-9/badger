'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { ArrowLeft, Mail, CheckCircle } from 'lucide-react'
import { useRouter } from 'next/navigation'
import { supabase } from '@/lib/supabase'

export default function EarlyAccessPage() {
  const router = useRouter()
  const [email, setEmail] = useState('')
  const [submitted, setSubmitted] = useState(false)
  const [isVisible, setIsVisible] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    // Trigger animation on mount
    setIsVisible(true)
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    // Validate email
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
    if (!emailRegex.test(email)) {
      setError('Please enter a valid email address')
      return
    }

    try {
      // Insert email into Supabase
      const { error: supabaseError } = await supabase
        .from('emails')
        .insert([{ email }])

      if (supabaseError) {
        // Check if email already exists
        if (supabaseError.code === '23505') {
          setError('This email is already on the waitlist')
        } else {
          setError('Something went wrong. Please try again.')
          console.error('Supabase error:', supabaseError)
        }
        return
      }

      setSubmitted(true)

      // Fade out then redirect to home
      setTimeout(() => {
        setIsVisible(false)
        setTimeout(() => {
          router.push('/')
        }, 500)
      }, 1200)
    } catch (err) {
      setError('Something went wrong. Please try again.')
      console.error('Error:', err)
    }
  }

  return (
    <div className="min-h-screen bg-[#14120b] text-[#f7f7f4] flex items-center justify-center px-8 overflow-hidden">
      {/* Background Gradient Orbs */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-[20%] left-[15%] w-[300px] h-[300px] rounded-full bg-[#d9955f] opacity-[0.08] blur-[80px] transition-all duration-1000"></div>
        <div className="absolute bottom-[20%] right-[15%] w-[350px] h-[350px] rounded-full bg-[#cd7f47] opacity-[0.09] blur-[90px] transition-all duration-1000"></div>
      </div>

      <div className={`relative z-10 max-w-md w-full transition-all duration-700 ease-out ${
        isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
      }`}>
        {/* Back Button */}
        <Link
          href="/"
          className="inline-flex items-center gap-2 text-sm text-[#f7f7f4]/60 hover:text-[#f7f7f4] transition-colors mb-8"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Home
        </Link>

        <div className="bg-[#1a1812] rounded-xl border border-[#f7f7f4]/10 p-8 shadow-2xl animate-in fade-in slide-in-from-bottom-4 duration-500">
          <h1 className="text-3xl font-bold mb-3 animate-in fade-in duration-700 delay-100">Get Early Access</h1>
          <p className="text-[#f7f7f4]/70 mb-8 animate-in fade-in duration-700 delay-200">
             Be the first to experience BOOM.
          </p>

          <form onSubmit={handleSubmit} className="space-y-4" autoComplete="off" noValidate>
            <div>
              <label htmlFor="email" className="block text-sm font-medium mb-2">
                Email Address
              </label>
              <input
                type="email"
                id="email"
                name="email"
                value={email}
                onChange={(e) => {
                  setEmail(e.target.value)
                  setError('')
                }}
                placeholder="you@example.com"
                required
                autoComplete="off"
                className={`w-full px-4 py-3 bg-[#14120b] rounded-lg text-[#f7f7f4] placeholder:text-[#f7f7f4]/40 focus:outline-none transition-colors [&:-webkit-autofill]:[-webkit-text-fill-color:#f7f7f4] [&:-webkit-autofill]:[-webkit-box-shadow:0_0_0_1000px_#14120b_inset] [&:-webkit-autofill:hover]:[-webkit-box-shadow:0_0_0_1000px_#14120b_inset] [&:-webkit-autofill:focus]:[-webkit-box-shadow:0_0_0_1000px_#14120b_inset] ${
                  error
                    ? 'border-2 border-red-500/50 focus:border-red-500'
                    : 'border border-[#f7f7f4]/20 focus:border-[#d4a574]'
                }`}
              />
              {error && (
                <p className="mt-2 text-sm text-red-400 animate-in fade-in slide-in-from-top-1 duration-300">
                  {error}
                </p>
              )}
            </div>

            <button
              type="submit"
              disabled={submitted}
              className={`w-full px-6 py-3 rounded-lg transition-all duration-200 font-semibold flex items-center justify-center gap-2 ${
                submitted
                  ? 'bg-[#d4a574] text-[#14120b] cursor-default'
                  : 'bg-[#f7f7f4] text-[#14120b] hover:bg-[#f7f7f4]/90'
              }`}
            >
              {submitted ? (
                <>
                  <CheckCircle className="w-4 h-4" />
                  Received
                </>
              ) : (
                <>
                  <Mail className="w-4 h-4" />
                  Submit
                </>
              )}
            </button>
          </form>

          <p className="text-xs text-[#f7f7f4]/40 mt-6 text-center">
            We&apos;ll notify you on launch.
          </p>
        </div>
      </div>
    </div>
  )
}
