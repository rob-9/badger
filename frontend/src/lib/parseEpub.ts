import JSZip from 'jszip'

// --- Structured content types for RAG indexing ---

export interface StructuredSection {
  heading?: string
  paragraphs: string[]
}

export interface StructuredChapter {
  index: number
  title: string
  sections: StructuredSection[]
}

export interface StructuredBook {
  chapters: StructuredChapter[]
}

/** Find a manifest item by attribute value (avoids selector injection from EPUB metadata). */
function findByAttr(parent: Element, tag: string, attr: string, value: string): Element | null {
  for (const el of Array.from(parent.querySelectorAll(tag))) {
    if (el.getAttribute(attr) === value) return el
  }
  return null
}

/** Infer image MIME type from a file path. */
function inferImageType(path: string): string {
  const ext = path.split('.').pop()?.toLowerCase()
  switch (ext) {
    case 'png': return 'image/png'
    case 'gif': return 'image/gif'
    case 'webp': return 'image/webp'
    case 'svg': return 'image/svg+xml'
    default: return 'image/jpeg'
  }
}

interface EpubParts {
  zip: JSZip
  opfDoc: Document
  opfDir: string
  manifest: Element
  spineItems: Element[]
  parser: DOMParser
}

/** Load and parse shared EPUB structures (zip, OPF, manifest, spine). */
async function loadEpub(arrayBuffer: ArrayBuffer): Promise<EpubParts> {
  const zip = await JSZip.loadAsync(arrayBuffer)
  const containerFile = await zip.file('META-INF/container.xml')?.async('text')
  if (!containerFile) throw new Error('Invalid EPUB: container.xml not found')

  const parser = new DOMParser()
  const containerDoc = parser.parseFromString(containerFile, 'text/xml')
  const opfPath = containerDoc.querySelector('rootfile')?.getAttribute('full-path')
  if (!opfPath) throw new Error('Invalid EPUB: OPF file path not found')

  const opfFile = await zip.file(opfPath)?.async('text')
  if (!opfFile) throw new Error('Invalid EPUB: OPF file not found')

  const opfDoc = parser.parseFromString(opfFile, 'text/xml')
  const opfDir = opfPath.substring(0, opfPath.lastIndexOf('/') + 1)

  const manifest = opfDoc.querySelector('manifest')
  if (!manifest) throw new Error('Invalid EPUB: manifest not found')

  const spineItems = Array.from(opfDoc.querySelectorAll('spine itemref'))

  return { zip, opfDoc, opfDir, manifest, spineItems, parser }
}

/**
 * Extract the cover image from an EPUB as a data URL.
 * Tries: <meta name="cover"> → manifest item with properties="cover-image" → common filenames.
 * Returns null if no cover is found.
 */
export async function extractCover(arrayBuffer: ArrayBuffer): Promise<string | null> {
  try {
    const { zip, opfDoc, opfDir, manifest } = await loadEpub(arrayBuffer)

    let coverHref: string | null = null
    let mediaType: string | null = null

    // Strategy 1: <meta name="cover" content="item-id"/>
    const coverMeta = opfDoc.querySelector('metadata meta[name="cover"]')
    if (coverMeta) {
      const coverId = coverMeta.getAttribute('content')
      if (coverId) {
        const item = findByAttr(manifest, 'item', 'id', coverId)
        coverHref = item?.getAttribute('href') ?? null
        mediaType = item?.getAttribute('media-type') ?? null
      }
    }

    // Strategy 2: <item properties="cover-image"/> (EPUB 3)
    if (!coverHref) {
      const items = Array.from(manifest.querySelectorAll('item'))
      const coverItem = items.find(item =>
        item.getAttribute('properties')?.includes('cover-image')
      )
      if (coverItem) {
        coverHref = coverItem.getAttribute('href')
        mediaType = coverItem.getAttribute('media-type')
      }
    }

    // Strategy 3: common cover filenames
    if (!coverHref) {
      const candidates = ['cover.jpg', 'cover.jpeg', 'cover.png', 'images/cover.jpg', 'images/cover.jpeg', 'images/cover.png', 'Images/cover.jpg', 'Images/cover.jpeg', 'Images/cover.png']
      for (const candidate of candidates) {
        if (zip.file(opfDir + candidate)) {
          coverHref = candidate
          mediaType = inferImageType(candidate)
          break
        }
      }
    }

    if (!coverHref) return null

    const coverPath = opfDir + coverHref
    const coverData = await zip.file(coverPath)?.async('base64')
    if (!coverData) return null

    const type = mediaType || inferImageType(coverHref)
    return `data:${type};base64,${coverData}`
  } catch (err) {
    console.warn('Failed to extract EPUB cover:', err)
    return null
  }
}

export async function extractText(arrayBuffer: ArrayBuffer): Promise<string> {
  const { zip, opfDir, manifest, spineItems, parser } = await loadEpub(arrayBuffer)

  const textContent: string[] = []

  for (const spineItem of spineItems) {
    const idref = spineItem.getAttribute('idref')
    if (!idref) continue

    const href = findByAttr(manifest, 'item', 'id', idref)?.getAttribute('href')
    if (!href) continue

    const contentFile = await zip.file(opfDir + href)?.async('text')
    if (!contentFile) continue

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

/**
 * Extract structured content from an EPUB, preserving chapter/section/paragraph hierarchy.
 * Used for structure-aware RAG indexing.
 */
export async function extractStructuredText(arrayBuffer: ArrayBuffer): Promise<StructuredBook> {
  const { zip, opfDir, manifest, spineItems, parser } = await loadEpub(arrayBuffer)

  const chapters: StructuredChapter[] = []
  const blockSelector = 'h1, h2, h3, h4, p, blockquote, li, div'

  for (const spineItem of spineItems) {
    const idref = spineItem.getAttribute('idref')
    if (!idref) continue

    const href = findByAttr(manifest, 'item', 'id', idref)?.getAttribute('href')
    if (!href) continue

    const contentFile = await zip.file(opfDir + href)?.async('text')
    if (!contentFile) continue

    try {
      const contentDoc = parser.parseFromString(contentFile, 'text/html')
      const body = contentDoc.querySelector('body')
      if (!body) continue

      // Extract chapter title from first h1/h2 in this spine item
      const titleEl = body.querySelector('h1, h2')
      const chapterTitle = titleEl?.textContent?.trim()
        || href.replace(/\.[^.]+$/, '').replace(/[_-]/g, ' ')

      // Walk block-level elements to build sections
      const sections: StructuredSection[] = []
      let currentSection: StructuredSection = { paragraphs: [] }

      // Collect leaf-level block elements only — skip any element that contains
      // another matched block element to avoid duplicate text from nesting
      // (e.g. <blockquote><p>text</p></blockquote> keeps only the <p>)
      const allBlockEls = Array.from(body.querySelectorAll(blockSelector))
      const blockEls = allBlockEls.filter(el => !el.querySelector(blockSelector))

      for (const el of blockEls) {
        const tag = el.tagName.toLowerCase()
        const text = el.textContent?.trim()
        if (!text) continue

        // Skip the title element we already captured
        if (el === titleEl) continue

        if (tag === 'h1' || tag === 'h2' || tag === 'h3' || tag === 'h4') {
          // Start a new section if current has content
          if (currentSection.heading || currentSection.paragraphs.length > 0) {
            sections.push(currentSection)
          }
          currentSection = { heading: text, paragraphs: [] }
        } else {
          currentSection.paragraphs.push(text)
        }
      }

      // Push final section
      if (currentSection.heading || currentSection.paragraphs.length > 0) {
        sections.push(currentSection)
      }

      // Only add chapter if it has content
      if (sections.length > 0) {
        chapters.push({
          index: chapters.length,
          title: chapterTitle,
          sections,
        })
      }
    } catch (err) {
      console.warn('Failed to parse content file:', href, err)
    }
  }

  return { chapters }
}

export async function parseEpub(file: File): Promise<string> {
  return extractText(await file.arrayBuffer())
}
