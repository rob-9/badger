'use client'

import { Book, Clock, Star, Upload, Search, MessageSquare, Settings, Menu, X } from 'lucide-react'
import { useState } from 'react'

export type ViewType = 'all' | 'recent' | 'starred' | 'chats'

interface SidebarProps {
  currentView: ViewType
  onViewChange: (view: ViewType) => void
  onUploadClick: () => void
  bookCount: number
}

function NavItem({
  icon: Icon,
  label,
  isActive,
  badge,
  disabled = false,
  onClick,
}: {
  icon: typeof Book
  label: string
  isActive?: boolean
  badge?: number
  disabled?: boolean
  onClick?: () => void
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`
        relative w-full flex items-center gap-3 px-4 py-2.5 rounded-lg transition-all duration-200
        ${isActive
          ? 'bg-accent/10 text-accent'
          : disabled
          ? 'text-[#f7f7f4]/20 cursor-not-allowed'
          : 'text-[#f7f7f4]/50 hover:bg-[#f7f7f4]/[0.04] hover:text-[#f7f7f4]/75'
        }
      `}
    >
      {/* Active indicator bar */}
      {isActive && (
        <div className="absolute left-0 top-1/2 -translate-y-1/2 w-[2px] h-5 bg-accent rounded-r" />
      )}
      <Icon className={`w-[18px] h-[18px] flex-shrink-0 ${isActive ? 'text-accent' : ''}`} />
      <span className="flex-1 text-left text-sm">{label}</span>
      {badge !== undefined && (
        <span className={`text-[0.65rem] tabular-nums px-2 py-0.5 rounded-full ${
          isActive ? 'bg-accent/15 text-accent' : 'bg-[#f7f7f4]/[0.06] text-[#f7f7f4]/40'
        }`}>
          {badge}
        </span>
      )}
    </button>
  )
}

export default function Sidebar({
  currentView,
  onViewChange,
  onUploadClick,
  bookCount,
}: SidebarProps) {
  const [isOpen, setIsOpen] = useState(false)

  const handleViewChange = (view: ViewType) => {
    onViewChange(view)
    setIsOpen(false)
  }

  return (
    <>
      {/* Mobile Menu Button */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          className="lg:hidden fixed top-8 left-8 z-50 p-2.5 bg-surface-warm/90 backdrop-blur-sm border border-[#f7f7f4]/10 rounded-xl shadow-lg hover:bg-[#f7f7f4]/5 transition-all duration-200"
        >
          <Menu className="w-5 h-5 text-[#f7f7f4]/70" />
        </button>
      )}

      {/* Mobile Overlay */}
      {isOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black/60 backdrop-blur-sm z-40 animate-fade-in"
          onClick={() => setIsOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`
          fixed lg:static inset-y-0 left-0 z-40
          w-[260px] bg-surface-warm
          border-r border-[#f7f7f4]/[0.06]
          flex flex-col
          transition-transform duration-300 ease-in-out
          ${isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        `}
      >
        {/* Header */}
        <div className="px-6 py-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <h1 className="font-display text-2xl font-light text-[#f7f7f4] tracking-tight">badger</h1>
            </div>
            <button
              onClick={() => setIsOpen(false)}
              className="lg:hidden p-2 hover:bg-[#f7f7f4]/5 rounded-lg transition-colors"
            >
              <X className="w-4 h-4 text-[#f7f7f4]/50" />
            </button>
          </div>
        </div>

        {/* Divider */}
        <div className="mx-6 h-px bg-gradient-to-r from-accent/20 via-[#f7f7f4]/[0.06] to-transparent" />

        {/* Navigation */}
        <div className="flex-1 overflow-y-auto px-3 py-5 space-y-6">
          <div>
            <h2 className="text-[0.6rem] font-medium text-[#f7f7f4]/25 uppercase tracking-[0.2em] px-4 mb-2">
              Library
            </h2>
            <div className="space-y-0.5">
              <NavItem icon={Book} label="All Books" isActive={currentView === 'all'} badge={bookCount} onClick={() => handleViewChange('all')} />
              <NavItem icon={Clock} label="Recent" isActive={currentView === 'recent'} onClick={() => handleViewChange('recent')} />
              <NavItem icon={Star} label="Starred" disabled />
            </div>
          </div>

          <div>
            <h2 className="text-[0.6rem] font-medium text-[#f7f7f4]/25 uppercase tracking-[0.2em] px-4 mb-2">
              Actions
            </h2>
            <div className="space-y-0.5">
              <button
                onClick={() => {
                  onUploadClick()
                  setIsOpen(false)
                }}
                className="w-full flex items-center gap-3 px-4 py-2.5 rounded-lg transition-all duration-200 bg-accent/90 text-surface-deep font-medium hover:bg-accent group"
              >
                <Upload className="w-[18px] h-[18px] flex-shrink-0 transition-transform group-hover:-translate-y-0.5 duration-200" />
                <span className="flex-1 text-left text-sm">Upload Book</span>
              </button>
            </div>
          </div>

          <div>
            <h2 className="text-[0.6rem] font-medium text-[#f7f7f4]/25 uppercase tracking-[0.2em] px-4 mb-2">
              Coming Soon
            </h2>
            <div className="space-y-0.5">
              <NavItem icon={MessageSquare} label="Chats" disabled />
              <NavItem icon={Search} label="Search" disabled />
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mx-6 h-px bg-[#f7f7f4]/[0.04]" />
        <div className="p-3">
          <button
            disabled
            className="w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-[#f7f7f4]/20 cursor-not-allowed"
          >
            <Settings className="w-[18px] h-[18px]" />
            <span className="text-sm">Settings</span>
          </button>
        </div>
      </aside>
    </>
  )
}
