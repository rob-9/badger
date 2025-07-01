import * as pdfjsLib from 'pdfjs-dist'

// Configure PDF.js worker
if (typeof window !== 'undefined') {
  pdfjsLib.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.js`
}

export interface ParsedDocument {
  content: string
  metadata: {
    title?: string
    author?: string
    pageCount?: number
    wordCount?: number
  }
}

export async function parseTextFile(file: File): Promise<ParsedDocument> {
  const content = await file.text()
  
  return {
    content,
    metadata: {
      title: file.name,
      wordCount: content.split(/\s+/).length
    }
  }
}

export async function parsePDFFile(file: File): Promise<ParsedDocument> {
  const arrayBuffer = await file.arrayBuffer()
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise
  
  let content = ''
  const pageCount = pdf.numPages
  
  for (let i = 1; i <= pageCount; i++) {
    const page = await pdf.getPage(i)
    const textContent = await page.getTextContent()
    const pageText = textContent.items
      .map((item: any) => item.str)
      .join(' ')
    content += pageText + '\n\n'
  }
  
  return {
    content: content.trim(),
    metadata: {
      title: file.name,
      pageCount,
      wordCount: content.split(/\s+/).length
    }
  }
}

export async function parseDocument(file: File): Promise<ParsedDocument> {
  const extension = file.name.split('.').pop()?.toLowerCase()
  
  switch (extension) {
    case 'txt':
    case 'md':
      return parseTextFile(file)
    case 'pdf':
      return parsePDFFile(file)
    default:
      // Fallback to text parsing
      return parseTextFile(file)
  }
}