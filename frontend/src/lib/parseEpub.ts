import JSZip from 'jszip'

/** Resolve the OPF document and its directory from a loaded EPUB zip. */
async function getOpf(zip: JSZip) {
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

  return { opfDoc, opfDir }
}

/**
 * Extract the cover image from an EPUB as a data URL.
 * Tries: <meta name="cover"> → manifest item with properties="cover-image" → common filenames.
 * Returns null if no cover is found.
 */
export async function extractCover(arrayBuffer: ArrayBuffer): Promise<string | null> {
  try {
    const zip = await JSZip.loadAsync(arrayBuffer)
    const { opfDoc, opfDir } = await getOpf(zip)
    const manifest = opfDoc.querySelector('manifest')
    if (!manifest) return null

    let coverHref: string | null = null
    let mediaType: string | null = null

    // Strategy 1: <meta name="cover" content="item-id"/>
    const coverMeta = opfDoc.querySelector('metadata meta[name="cover"]')
    if (coverMeta) {
      const coverId = coverMeta.getAttribute('content')
      if (coverId) {
        const item = manifest.querySelector(`item[id="${coverId}"]`)
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
          mediaType = candidate.endsWith('.png') ? 'image/png' : 'image/jpeg'
          break
        }
      }
    }

    if (!coverHref) return null

    const coverPath = opfDir + coverHref
    const coverData = await zip.file(coverPath)?.async('base64')
    if (!coverData) return null

    const type = mediaType || (coverHref.endsWith('.png') ? 'image/png' : 'image/jpeg')
    return `data:${type};base64,${coverData}`
  } catch (err) {
    console.warn('Failed to extract EPUB cover:', err)
    return null
  }
}

export async function extractText(arrayBuffer: ArrayBuffer): Promise<string> {
  const zip = await JSZip.loadAsync(arrayBuffer)
  const { opfDoc, opfDir } = await getOpf(zip)

  const spineItems = Array.from(opfDoc.querySelectorAll('spine itemref'))
  const manifest = opfDoc.querySelector('manifest')
  const parser = new DOMParser()

  const textContent: string[] = []

  for (const spineItem of spineItems) {
    const idref = spineItem.getAttribute('idref')
    if (!idref) continue

    const manifestItem = manifest?.querySelector(`item[id="${idref}"]`)
    const href = manifestItem?.getAttribute('href')

    if (!href) continue

    const contentPath = opfDir + href
    const contentFile = await zip.file(contentPath)?.async('text')

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

export async function parseEpub(file: File): Promise<string> {
  return extractText(await file.arrayBuffer())
}
