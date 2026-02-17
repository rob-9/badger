'use client'

import { useState, useCallback } from 'react'
import { Upload, FileText, Book, File } from 'lucide-react'
import { parseEpub, extractText } from '@/lib/parseEpub'
import { importLocalEpub } from '@/lib/api'

interface FileUploaderProps {
  onFileLoadAction: (content: string, fileName: string, arrayBuffer?: ArrayBuffer) => void
}

export default function FileUploader({ onFileLoadAction }: FileUploaderProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [localPath, setLocalPath] = useState('')

  const handleFile = useCallback(async (file: File) => {
    setIsLoading(true)

    try {
      let text: string
      let arrayBuffer: ArrayBuffer | undefined

      const lowerName = file.name.toLowerCase()
      if (lowerName.endsWith('.epub') || lowerName.endsWith('.epub.zip')) {
        arrayBuffer = await file.arrayBuffer()
        text = await parseEpub(file)
      } else {
        text = await file.text()
      }

      const cleanName = file.name.replace(/\.epub\.zip$/i, '.epub')
      onFileLoadAction(text, cleanName, arrayBuffer)
    } catch (error) {
      console.error('Error reading file:', error)
      alert('Error loading file: ' + (error as Error).message)
    } finally {
      setIsLoading(false)
    }
  }, [onFileLoadAction])

  const handleLocalImport = useCallback(async () => {
    const path = localPath.trim()
    if (!path) return

    setIsLoading(true)
    try {
      const { arrayBuffer, filename } = await importLocalEpub(path)
      const text = await extractText(arrayBuffer)
      onFileLoadAction(text, filename, arrayBuffer)
    } catch (error) {
      console.error('Error importing local EPUB:', error)
      alert('Error importing EPUB: ' + (error as Error).message)
    } finally {
      setIsLoading(false)
    }
  }, [localPath, onFileLoadAction])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      handleFile(files[0])
    }
  }, [handleFile])

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      handleFile(files[0])
    }
  }, [handleFile])

  return (
    <div className="w-full max-w-2xl mx-auto p-8">
      <div className="text-center mb-8">
        <Book className="w-16 h-16 mx-auto mb-4 text-accent" />
        <h1 className="text-3xl font-bold mb-2 dark:text-[#e0e0e0]">boom</h1>
        <p className="text-gray-600 dark:text-[#888]">Read Better.</p>
      </div>

      <div
        className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
          isDragging
            ? 'border-accent bg-blue-50 dark:bg-blue-950/20'
            : 'border-gray-300 dark:border-[#333] hover:border-gray-400 dark:hover:border-[#555]'
        }`}
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        onDragEnter={() => setIsDragging(true)}
        onDragLeave={() => setIsDragging(false)}
      >
        {isLoading ? (
          <div className="flex flex-col items-center space-y-4">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-accent"></div>
            <div className="space-y-3 w-full max-w-xs">
              <div className="h-4 bg-gray-200 dark:bg-[#2a2a2a] rounded animate-pulse"></div>
              <div className="h-4 bg-gray-200 dark:bg-[#2a2a2a] rounded animate-pulse w-5/6 mx-auto"></div>
              <div className="h-4 bg-gray-200 dark:bg-[#2a2a2a] rounded animate-pulse w-4/6 mx-auto"></div>
            </div>
            <p className="text-gray-600 dark:text-[#888]">Loading document...</p>
          </div>
        ) : (
          <>
            <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
            <h3 className="text-xl font-semibold mb-2 dark:text-[#e0e0e0]">Upload Document</h3>
            <input
              type="file"
              accept=".txt,.pdf,.epub,.zip"
              onChange={handleFileInput}
              className="hidden"
              id="file-input"
            />
            <label
              htmlFor="file-input"
              className="inline-flex items-center px-6 py-3 bg-accent text-white rounded-lg hover:bg-blue-600 cursor-pointer transition-colors"
            >
              <FileText className="w-5 h-5 mr-2" />
              Choose File
            </label>
          </>
        )}
      </div>

      <div className="mt-6 flex items-center gap-2">
        <input
          type="text"
          value={localPath}
          onChange={(e) => setLocalPath(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleLocalImport()}
          placeholder="Or paste a local EPUB path..."
          className="flex-1 px-3 py-2 text-sm border rounded-lg bg-white dark:bg-[#1a1a1a] border-gray-300 dark:border-[#333] text-gray-900 dark:text-[#e0e0e0] placeholder-gray-400 dark:placeholder-[#555] focus:outline-none focus:border-accent"
        />
        <button
          onClick={handleLocalImport}
          disabled={!localPath.trim() || isLoading}
          className="px-4 py-2 text-sm bg-accent text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          Import
        </button>
      </div>

      <div className="mt-6 grid grid-cols-3 gap-4 text-center text-sm text-gray-500 dark:text-[#666]">
        <div className="flex flex-col items-center">
          <FileText className="w-6 h-6 mb-2" />
          <span>Text Files</span>
        </div>
        <div className="flex flex-col items-center">
          <File className="w-6 h-6 mb-2" />
          <span>PDF Files</span>
        </div>
        <div className="flex flex-col items-center">
          <Book className="w-6 h-6 mb-2" />
          <span>EPUB Books</span>
        </div>
      </div>
    </div>
  )
}
