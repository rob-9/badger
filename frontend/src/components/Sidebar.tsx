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
        w-full flex items-center gap-3 px-4 py-2.5 rounded-lg transition-all
        ${isActive
          ? 'bg-[#f7f7f4]/10 text-[#f7f7f4]'
          : disabled
          ? 'text-[#f7f7f4]/25 cursor-not-allowed'
          : 'text-[#f7f7f4]/60 hover:bg-[#f7f7f4]/5 hover:text-[#f7f7f4]/80'
        }
      `}
    >
      <Icon className="w-5 h-5 flex-shrink-0" />
      <span className="flex-1 text-left text-sm font-medium">{label}</span>
      {badge !== undefined && (
        <span className="text-xs px-2 py-0.5 rounded-full bg-[#f7f7f4]/10 text-[#f7f7f4]/60">
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
      {/* Mobile Menu Button — hidden once sidebar is open */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          className="lg:hidden fixed top-8 left-8 z-50 p-2 bg-[#1a1812] border border-[#f7f7f4]/10 rounded-lg shadow-lg hover:bg-[#f7f7f4]/5 transition-colors"
        >
          <Menu className="w-6 h-6 text-[#f7f7f4]/80" />
        </button>
      )}

      {/* Mobile Overlay */}
      {isOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black/50 z-40 animate-fade-in"
          onClick={() => setIsOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`
          fixed lg:static inset-y-0 left-0 z-40
          w-[280px] bg-[#1a1812]
          border-r border-[#f7f7f4]/10
          flex flex-col
          transition-transform duration-300 ease-in-out
          ${isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        `}
      >
        {/* Header */}
        <div className="p-6 border-b border-[#f7f7f4]/10">
          <div className="flex items-center justify-between">
            <h1 className="text-xl font-bold text-[#f7f7f4]">badger</h1>
            <button
              onClick={() => setIsOpen(false)}
              className="lg:hidden p-2 hover:bg-[#f7f7f4]/5 rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-[#f7f7f4]/60" />
            </button>
          </div>
        </div>

        {/* Navigation */}
        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          <div>
            <h2 className="text-xs font-semibold text-[#f7f7f4]/40 uppercase tracking-wide px-4 mb-2">
              Library
            </h2>
            <div className="space-y-1">
              <NavItem icon={Book} label="All Books" isActive={currentView === 'all'} badge={bookCount} onClick={() => handleViewChange('all')} />
              <NavItem icon={Clock} label="Recent" isActive={currentView === 'recent'} onClick={() => handleViewChange('recent')} />
              <NavItem icon={Star} label="Starred" disabled />
            </div>
          </div>

          <div>
            <h2 className="text-xs font-semibold text-[#f7f7f4]/40 uppercase tracking-wide px-4 mb-2">
              Actions
            </h2>
            <div className="space-y-1">
              <button
                onClick={() => {
                  onUploadClick()
                  setIsOpen(false)
                }}
                className="w-full flex items-center gap-3 px-4 py-2.5 rounded-lg transition-all bg-accent/90 text-[#14120b] font-medium hover:bg-accent"
              >
                <Upload className="w-5 h-5 flex-shrink-0" />
                <span className="flex-1 text-left text-sm font-medium">Upload New Book</span>
              </button>
            </div>
          </div>

          <div>
            <h2 className="text-xs font-semibold text-[#f7f7f4]/40 uppercase tracking-wide px-4 mb-2">
              TBA
            </h2>
            <div className="space-y-1">
              <NavItem icon={MessageSquare} label="Chats" disabled />
              <NavItem icon={Search} label="Search" disabled />
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-[#f7f7f4]/10">
          <button
            disabled
            className="w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-[#f7f7f4]/25 cursor-not-allowed"
          >
            <Settings className="w-5 h-5" />
            <span className="text-sm">Settings</span>
          </button>
        </div>
      </aside>
    </>
  )
}
