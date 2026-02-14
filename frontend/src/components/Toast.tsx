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

  const getIcon = () => {
    switch (type) {
      case 'success':
        return <CheckCircle className="w-5 h-5 text-green-500" />
      case 'error':
        return <AlertCircle className="w-5 h-5 text-red-500" />
      default:
        return <Info className="w-5 h-5 text-blue-500" />
    }
  }

  const getBgColor = () => {
    switch (type) {
      case 'success':
        return 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'
      case 'error':
        return 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
      default:
        return 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800'
    }
  }

  return (
    <div
      className={`fixed top-4 right-4 z-50 flex items-start gap-3 max-w-md px-4 py-3 rounded-lg border shadow-lg animate-fade-up ${getBgColor()}`}
      role="alert"
      aria-live="polite"
    >
      <div className="flex-shrink-0 mt-0.5">{getIcon()}</div>
      <p className="flex-1 text-sm text-gray-800 dark:text-[#e0e0e0] leading-relaxed">
        {message}
      </p>
      <button
        onClick={onClose}
        className="flex-shrink-0 p-1 hover:bg-black/5 dark:hover:bg-white/5 rounded transition-colors"
        aria-label="Close notification"
      >
        <X className="w-4 h-4 text-gray-500 dark:text-[#aaa]" />
      </button>
    </div>
  )
}
