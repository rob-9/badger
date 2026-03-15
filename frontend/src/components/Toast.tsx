'use client'

import { useEffect } from 'react'
import { X, CheckCircle, Info, AlertCircle } from 'lucide-react'

interface ToastProps {
  message: string
  type: 'info' | 'success' | 'error'
  onClose: () => void
}

export default function Toast({ message, type, onClose }: ToastProps) {
  // Auto-dismiss after timeout
  useEffect(() => {
    const timeout = type === 'error' ? 8000 : 5000
    const timer = setTimeout(onClose, timeout)
    return () => clearTimeout(timer)
  }, [type, onClose])

  const config = {
    success: {
      icon: <CheckCircle className="w-4 h-4 text-emerald-400" />,
      bg: 'bg-surface-warm border-emerald-500/20',
      bar: 'bg-emerald-500',
    },
    error: {
      icon: <AlertCircle className="w-4 h-4 text-red-400" />,
      bg: 'bg-surface-warm border-red-500/20',
      bar: 'bg-red-500',
    },
    info: {
      icon: <Info className="w-4 h-4 text-accent" />,
      bg: 'bg-surface-warm border-accent/20',
      bar: 'bg-accent',
    },
  }[type]

  return (
    <div
      className={`fixed top-5 right-5 z-50 flex items-start gap-3 max-w-sm pl-4 pr-3 py-3 rounded-xl border shadow-xl shadow-black/20 animate-toast-enter ${config.bg}`}
      role="alert"
      aria-live="polite"
    >
      {/* Accent bar */}
      <div className={`absolute left-0 top-3 bottom-3 w-[2px] rounded-r ${config.bar}`} />

      <div className="flex-shrink-0 mt-0.5">{config.icon}</div>
      <p className="flex-1 text-sm text-[#e0e0e0] leading-relaxed">
        {message}
      </p>
      <button
        onClick={onClose}
        className="flex-shrink-0 p-1 hover:bg-[#f7f7f4]/5 rounded-lg transition-colors"
        aria-label="Close notification"
      >
        <X className="w-3.5 h-3.5 text-[#f7f7f4]/40 hover:text-[#f7f7f4]/70" />
      </button>
    </div>
  )
}
