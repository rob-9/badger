'use client'

import { useState, useCallback } from 'react'
import { Upload, FileText, Book, File } from 'lucide-react'
import JSZip from 'jszip'

interface FileUploaderProps {
  onFileLoadAction: (content: string, fileName: string, arrayBuffer?: ArrayBuffer) => void
}

export default function FileUploader({ onFileLoadAction }: FileUploaderProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [isLoading, setIsLoading] = useState(false)

  const parseEpub = async (file: File): Promise<string> => {
    const arrayBuffer = await file.arrayBuffer()
    const zip = await JSZip.loadAsync(arrayBuffer)

    // Find the OPF file (contains manifest and spine)
    const containerFile = await zip.file('META-INF/container.xml')?.async('text')
    if (!containerFile) {
      throw new Error('Invalid EPUB: container.xml not found')
    }

    // Parse container.xml to get OPF file path
    const parser = new DOMParser()
    const containerDoc = parser.parseFromString(containerFile, 'text/xml')
    const opfPath = containerDoc.querySelector('rootfile')?.getAttribute('full-path')

    if (!opfPath) {
      throw new Error('Invalid EPUB: OPF file path not found')
    }

    // Get the OPF file
    const opfFile = await zip.file(opfPath)?.async('text')
    if (!opfFile) {
      throw new Error('Invalid EPUB: OPF file not found')
    }

    const opfDoc = parser.parseFromString(opfFile, 'text/xml')
    const opfDir = opfPath.substring(0, opfPath.lastIndexOf('/') + 1)

    // Get spine items (reading order)
    const spineItems = Array.from(opfDoc.querySelectorAll('spine itemref'))
    const manifest = opfDoc.querySelector('manifest')

    const textContent: string[] = []

    // Process each spine item
    for (const spineItem of spineItems) {
      const idref = spineItem.getAttribute('idref')
      if (!idref) continue

      // Find corresponding manifest item
      const manifestItem = manifest?.querySelector(`item[id="${idref}"]`)
      const href = manifestItem?.getAttribute('href')

      if (!href) continue

      // Get the content file
      const contentPath = opfDir + href
      const contentFile = await zip.file(contentPath)?.async('text')

      if (!contentFile) continue

      // Parse HTML and extract text
      try {
        const contentDoc = parser.parseFromString(contentFile, 'text/html')
        const body = contentDoc.querySelector('body')
        const text = body?.textContent || contentDoc.documentElement.textContent || ''

        if (text.trim()) {
          textContent.push(text.trim())
        }
      } catch (err) {
        console.warn('Failed to parse content file:', err)
      }
    }

    return textContent.join('\n\n')
  }

  const handleFile = useCallback(async (file: File) => {
    setIsLoading(true)

    try {
      let text: string
      let arrayBuffer: ArrayBuffer | undefined

      if (file.name.toLowerCase().endsWith('.epub')) {
        // Get ArrayBuffer for EPUB file (needed by react-reader)
        arrayBuffer = await file.arrayBuffer()
        // Still parse for fallback/preview
        text = await parseEpub(file)
      } else {
        // Handle other file types as plain text
        text = await file.text()
      }

      onFileLoadAction(text, file.name, arrayBuffer)
    } catch (error) {
      console.error('Error reading file:', error)
    } finally {
      setIsLoading(false)
    }
  }, [onFileLoadAction])

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
        <h1 className="text-3xl font-bold mb-2">boom</h1>
        <p className="text-gray-600">Read Better.</p>
      </div>

      <div
        className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
          isDragging
            ? 'border-accent bg-blue-50'
            : 'border-gray-300 hover:border-gray-400'
        }`}
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        onDragEnter={() => setIsDragging(true)}
        onDragLeave={() => setIsDragging(false)}
      >
        {isLoading ? (
          <div className="flex flex-col items-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-accent mb-4"></div>
            <p>Loading document...</p>
          </div>
        ) : (
          <>
            <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
            <h3 className="text-xl font-semibold mb-2">Upload Document</h3>
            <p className="text-gray-600 mb-6">
              
            </p>
            <input
              type="file"
              accept=".txt,.pdf,.epub"
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

      <div className="mt-8 grid grid-cols-3 gap-4 text-center text-sm text-gray-500">
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