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
          ? 'bg-gray-100 dark:bg-[#2a2a2a] text-gray-900 dark:text-white'
          : disabled
          ? 'text-gray-400 dark:text-[#555] cursor-not-allowed'
          : 'text-gray-600 dark:text-[#999] hover:bg-gray-50 dark:hover:bg-[#252525]'
        }
      `}
    >
      <Icon className="w-5 h-5 flex-shrink-0" />
      <span className="flex-1 text-left text-sm font-medium">{label}</span>
      {badge !== undefined && (
        <span className="text-xs px-2 py-0.5 rounded-full bg-gray-200 dark:bg-[#333] text-gray-700 dark:text-[#aaa]">
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
      <button
        onClick={() => setIsOpen(true)}
        className="lg:hidden fixed top-4 left-4 z-50 p-2 bg-white dark:bg-[#1e1e1e] border border-gray-200 dark:border-[#2a2a2a] rounded-lg shadow-lg hover:bg-gray-50 dark:hover:bg-[#252525] transition-colors"
      >
        <Menu className="w-6 h-6 text-gray-700 dark:text-[#ccc]" />
      </button>

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
          w-[280px] bg-white dark:bg-[#1e1e1e]
          border-r border-gray-200 dark:border-[#2a2a2a]
          flex flex-col
          transition-transform duration-300 ease-in-out
          ${isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        `}
      >
        {/* Header */}
        <div className="p-6 border-b border-gray-200 dark:border-[#2a2a2a]">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Book className="w-6 h-6 text-accent" />
              <h1 className="text-xl font-bold text-gray-900 dark:text-white">boom</h1>
            </div>
            <button
              onClick={() => setIsOpen(false)}
              className="lg:hidden p-2 hover:bg-gray-100 dark:hover:bg-[#2a2a2a] rounded-lg transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Navigation */}
        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          <div>
            <h2 className="text-xs font-semibold text-gray-500 dark:text-[#666] uppercase tracking-wide px-4 mb-2">
              Library
            </h2>
            <div className="space-y-1">
              <NavItem icon={Book} label="All Books" isActive={currentView === 'all'} badge={bookCount} onClick={() => handleViewChange('all')} />
              <NavItem icon={Clock} label="Recent" isActive={currentView === 'recent'} onClick={() => handleViewChange('recent')} />
              <NavItem icon={Star} label="Starred" disabled />
            </div>
          </div>

          <div>
            <h2 className="text-xs font-semibold text-gray-500 dark:text-[#666] uppercase tracking-wide px-4 mb-2">
              Actions
            </h2>
            <div className="space-y-1">
              <button
                onClick={() => {
                  onUploadClick()
                  setIsOpen(false)
                }}
                className="w-full flex items-center gap-3 px-4 py-2.5 rounded-lg transition-all bg-accent text-white hover:bg-blue-600"
              >
                <Upload className="w-5 h-5 flex-shrink-0" />
                <span className="flex-1 text-left text-sm font-medium">Upload New Book</span>
              </button>
            </div>
          </div>

          <div>
            <h2 className="text-xs font-semibold text-gray-500 dark:text-[#666] uppercase tracking-wide px-4 mb-2">
              Coming Soon
            </h2>
            <div className="space-y-1">
              <NavItem icon={MessageSquare} label="Chats" disabled />
              <NavItem icon={Search} label="Search" disabled />
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-200 dark:border-[#2a2a2a]">
          <button
            disabled
            className="w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-gray-400 dark:text-[#555] cursor-not-allowed"
          >
            <Settings className="w-5 h-5" />
            <span className="text-sm">Settings</span>
          </button>
        </div>
      </aside>
    </>
  )
}
