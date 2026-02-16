"""
Text chunking utility.

Why chunking matters:
- Embeddings work best on coherent pieces of text (not too long, not too short)
- Typical chunk size: 500-1000 tokens (~2000-4000 characters)
- Overlap helps preserve context at chunk boundaries

Learn more:
- https://www.pinecone.io/learn/chunking-strategies/
- https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/
"""

from dataclasses import dataclass
import re


@dataclass
class TextChunk:
    """A chunk of text from a document."""
    id: str
    text: str
    metadata: dict

    def __post_init__(self):
        """Ensure metadata has required fields."""
        for field in ('book_id', 'chunk_index'):
            if field not in self.metadata:
                raise ValueError(f"Missing required metadata field: {field}")


DEFAULT_CHUNK_SIZE = 2000  # ~500 tokens
DEFAULT_OVERLAP = 200      # ~50 tokens overlap


def chunk_text(
    text: str,
    book_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_OVERLAP
) -> list[TextChunk]:
    """
    Split text into overlapping chunks.

    Example with chunk_size=100, overlap=20:
    Text: "The quick brown fox jumps over the lazy dog. The dog was sleeping..."

    Chunk 1: "The quick brown fox jumps over the lazy dog. The dog"  [0-100]
    Chunk 2: "dog. The dog was sleeping peacefully in the sun..."     [80-180]
              ↑ overlap preserves context

    Args:
        text: Text to chunk
        book_id: Unique identifier for the book
        chunk_size: Target size in characters
        chunk_overlap: Overlap between chunks to preserve context

    Returns:
        List of TextChunk objects
    """
    chunks: list[TextChunk] = []

    # Clean the text - normalize whitespace
    cleaned_text = re.sub(r'\s+', ' ', text).strip()

    start_index = 0
    chunk_index = 0

    while start_index < len(cleaned_text):
        # Calculate end position
        end_index = min(start_index + chunk_size, len(cleaned_text))

        # Try to break at a sentence boundary (. ! ? or newline)
        if end_index < len(cleaned_text):
            search_start = max(start_index + chunk_size - 200, start_index)
            search_text = cleaned_text[search_start:end_index]

            # Find last sentence boundary
            sentence_end = max(
                search_text.rfind('. '),
                search_text.rfind('! '),
                search_text.rfind('? '),
                search_text.rfind('\n')
            )

            if sentence_end > 0:
                end_index = search_start + sentence_end + 1

        # Extract chunk text
        chunk_text = cleaned_text[start_index:end_index].strip()

        if len(chunk_text) > 0:
            chunks.append(TextChunk(
                id=f"{book_id}-chunk-{chunk_index}",
                text=chunk_text,
                metadata={
                    'book_id': book_id,
                    'chunk_index': chunk_index,
                    'start_char': start_index,
                    'end_char': end_index
                }
            ))
            chunk_index += 1

        # Move start position (accounting for overlap)
        start_index = end_index - chunk_overlap

        # Prevent infinite loop
        if start_index >= len(cleaned_text) - chunk_overlap:
            break

    return chunks


def _split_paragraph_at_sentences(text: str, chunk_size: int) -> list[str]:
    """Split a large paragraph at sentence boundaries."""
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            search_start = max(start + chunk_size - 200, start)
            search_text = text[search_start:end]
            sentence_end = max(
                search_text.rfind('. '),
                search_text.rfind('! '),
                search_text.rfind('? '),
            )
            if sentence_end > 0:
                end = search_start + sentence_end + 1
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        start = end
    return pieces


def chunk_structured(
    structured_content: dict,
    book_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[TextChunk]:
    """
    Chunk structured EPUB content respecting chapter/section boundaries.

    - Never crosses chapter boundaries
    - Accumulates paragraphs within a chapter until exceeding chunk_size
    - Splits oversized paragraphs at sentence boundaries
    - No overlap (voyage-context-3 captures sibling context)
    """
    chunks: list[TextChunk] = []
    chunk_index = 0

    for chapter in structured_content.get("chapters", []):
        chapter_title = chapter.get("title", "")
        chapter_idx = chapter.get("index", 0)

        current_paragraphs: list[str] = []
        current_len = 0
        current_section_heading: str | None = None

        def emit_chunk():
            nonlocal chunk_index, current_paragraphs, current_len, current_section_heading
            if not current_paragraphs:
                return
            text = "\n\n".join(current_paragraphs)
            chunks.append(TextChunk(
                id=f"{book_id}-chunk-{chunk_index}",
                text=text,
                metadata={
                    "book_id": book_id,
                    "chunk_index": chunk_index,
                    "chapter_title": chapter_title,
                    "chapter_index": chapter_idx,
                    "section_heading": current_section_heading,
                },
            ))
            chunk_index += 1
            current_paragraphs = []
            current_len = 0

        for section in chapter.get("sections", []):
            heading = section.get("heading")
            if heading:
                current_section_heading = heading

            for para in section.get("paragraphs", []):
                para = para.strip()
                if not para:
                    continue

                if len(para) > chunk_size:
                    emit_chunk()
                    for piece in _split_paragraph_at_sentences(para, chunk_size):
                        current_paragraphs = [piece]
                        current_len = len(piece)
                        emit_chunk()
                    continue

                if current_len + len(para) + 2 > chunk_size and current_paragraphs:
                    emit_chunk()

                current_paragraphs.append(para)
                current_len += len(para) + 2

        emit_chunk()

    return chunks


def estimate_tokens(text: str) -> int:
    """
    Estimate token count (rough approximation).
    Rule of thumb: 1 token ≈ 4 characters for English
    """
    return (len(text) + 3) // 4  # Ceiling division
