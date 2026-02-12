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
        required_fields = ['book_id', 'chunk_index', 'start_char', 'end_char']
        for field in required_fields:
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


def estimate_tokens(text: str) -> int:
    """
    Estimate token count (rough approximation).
    Rule of thumb: 1 token ≈ 4 characters for English
    """
    return (len(text) + 3) // 4  # Ceiling division
