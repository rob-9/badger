"""
RAG (Retrieval Augmented Generation) Service.

This ties together:
1. Chunking - Split book into pieces
2. Embedding - Convert text to vectors (Voyage AI)
3. Storage - Store vectors for search
4. Retrieval - Find relevant chunks
5. Generation - Answer with context (Claude)

Learn more:
- https://docs.anthropic.com/en/docs/build-with-claude/retrieval-augmented-generation
- https://docs.voyageai.com/
- https://github.com/anthropics/claude-cookbooks
"""

import os
from dataclasses import dataclass
from typing import Optional
import voyageai
from anthropic import Anthropic

from .chunker import chunk_text
from .vector_store import VectorStore, VectorEntry


@dataclass
class RAGResponse:
    """Response from RAG query."""
    answer: str
    sources: list[dict]


class RAGService:
    """Simple RAG for personal reading assistant."""

    def __init__(self, storage_dir: str = ".data/vectors"):
        """
        Initialize RAG service.

        Args:
            storage_dir: Directory to store vector embeddings
        """
        # Initialize clients
        self.voyage = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        # Initialize vector store
        self.vector_store = VectorStore(storage_dir)

        print("[RAG] Service initialized")

    async def get_embedding(
        self,
        text: str,
        input_type: str = "query"
    ) -> list[float]:
        """
        Get embedding from Voyage AI.

        Why Voyage AI for embeddings?
        - Anthropic's official recommendation for embeddings
        - voyage-3 model: 1024 dimensions, optimized for RAG
        - Better retrieval quality than general-purpose embeddings
        - Supports input_type parameter for query vs document optimization

        Args:
            text: Text to embed
            input_type: Either 'query' or 'document'

        Returns:
            Embedding vector
        """
        print(f"[RAG] Embedding text ({input_type})")
        print(f"[RAG] Text length: {len(text)} characters")

        response = self.voyage.embed(
            texts=[text],
            model="voyage-3",
            input_type=input_type
        )

        if not response.embeddings or len(response.embeddings) == 0:
            raise ValueError("No embedding returned from Voyage AI")

        embedding = response.embeddings[0]
        print(f"[RAG] Generated embedding with {len(embedding)} dimensions")
        return embedding

    async def get_embeddings(
        self,
        texts: list[str],
        input_type: str = "document"
    ) -> list[list[float]]:
        """
        Get embeddings for multiple texts (batched for efficiency).

        Args:
            texts: List of texts to embed
            input_type: Either 'query' or 'document'

        Returns:
            List of embedding vectors
        """
        print(f"[RAG] Embedding {len(texts)} texts in batch ({input_type})")

        response = self.voyage.embed(
            texts=texts,
            model="voyage-3",
            input_type=input_type
        )

        if not response.embeddings:
            raise ValueError("No embeddings returned from Voyage AI")

        print(f"[RAG] Generated {len(response.embeddings)} embeddings")
        return response.embeddings

    async def index_book(self, book_id: str, text: str) -> None:
        """
        Index a book for RAG.

        Process:
        1. Chunk the text into ~500 token pieces
        2. Embed each chunk
        3. Store in vector store

        Args:
            book_id: Unique identifier for the book
            text: Full text of the book
        """
        print("=" * 60)
        print("[RAG] Starting book indexing process")
        print(f"[RAG] Book ID: {book_id}")
        print(f"[RAG] Text length: {len(text)} characters")

        # Check if already indexed
        if self.vector_store.has_book(book_id):
            print("[RAG] Book already indexed, skipping")
            print("=" * 60)
            return

        # STEP 1: Chunk the text
        print("[RAG] STEP 1: Chunking text")
        chunks = chunk_text(text, book_id)
        print(f"[RAG] Created {len(chunks)} chunks")
        if chunks:
            print(f"[RAG] Average chunk size: {len(text) // len(chunks)} characters")

        # STEP 2: Get embeddings for all chunks (batched)
        print("[RAG] STEP 2: Generating embeddings")
        texts = [chunk.text for chunk in chunks]
        embeddings = await self.get_embeddings(texts, input_type="document")

        # STEP 3: Create vector entries
        print("[RAG] STEP 3: Creating vector entries")
        entries = [
            VectorEntry(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunks, embeddings)
        ]
        print(f"[RAG] Created {len(entries)} vector entries")

        # STEP 4: Store in vector store
        print("[RAG] STEP 4: Storing in vector store")
        await self.vector_store.add_book(book_id, entries)

        print("[RAG] Indexing complete!")
        print("=" * 60)

    async def query_book(
        self,
        book_id: str,
        question: str,
        selected_text: Optional[str] = None,
        reader_position: Optional[float] = None
    ) -> RAGResponse:
        """
        Query a book with RAG.

        Process:
        1. Embed the question
        2. Find similar chunks (retrieval)
        3. Build prompt with context
        4. Generate answer with Claude

        Args:
            book_id: Book to query
            question: User's question
            selected_text: Optional selected text for context

        Returns:
            RAGResponse with answer and sources
        """
        print("=" * 60)
        print("[RAG] Starting query process")
        print(f"[RAG] Book ID: {book_id}")
        print(f"[RAG] Question: \"{question}\"")
        if selected_text:
            print(f"[RAG] Selected text: \"{selected_text[:100]}...\"")

        # STEP 1: Embed the question
        print("[RAG] STEP 1: Embedding question")
        question_embedding = await self.get_embedding(question, input_type="query")

        # STEP 2: Find relevant chunks
        print("[RAG] STEP 2: Searching for relevant chunks")
        results = await self.vector_store.search(book_id, question_embedding, top_k=5)

        if not results:
            print("[RAG] No relevant chunks found")
            print("=" * 60)
            return RAGResponse(
                answer="I couldn't find relevant information in the book to answer this question.",
                sources=[]
            )

        print(f"[RAG] Found {len(results)} relevant chunks:")
        for i, result in enumerate(results):
            chunk_idx = result.chunk.metadata['chunk_index']
            print(f"[RAG]   {i + 1}. Chunk {chunk_idx} (similarity: {result.score:.3f})")

        # STEP 3: Tag chunks as PAST or AHEAD based on reader position
        print("[RAG] STEP 3: Tagging chunks by reader position")
        total_chunks = await self.vector_store.get_total_chunks(book_id)
        reader_chunk_index = int((reader_position or 0) * total_chunks)
        print(f"[RAG] Reader at chunk ~{reader_chunk_index} of {total_chunks}")

        past_chunks = []
        ahead_chunks = []
        for result in results:
            if result.chunk.metadata['chunk_index'] <= reader_chunk_index:
                past_chunks.append(result)
            else:
                ahead_chunks.append(result)
        print(f"[RAG] Past chunks: {len(past_chunks)}, Ahead chunks: {len(ahead_chunks)}")

        # STEP 4: Build context with labeled sections
        print("[RAG] STEP 4: Building context from chunks")
        context_parts = []
        if past_chunks:
            context_parts.append(
                "[ALREADY READ]\n" + "\n\n---\n\n".join(
                    f"[Source {i + 1}]\n{r.chunk.text}"
                    for i, r in enumerate(past_chunks)
                )
            )
        if ahead_chunks:
            context_parts.append(
                "[COMING UP - guide only, do not spoil]\n" + "\n\n---\n\n".join(
                    f"[Source {i + 1}]\n{r.chunk.text}"
                    for i, r in enumerate(ahead_chunks)
                )
            )
        context = "\n\n===\n\n".join(context_parts)
        print(f"[RAG] Context length: {len(context)} characters")

        # STEP 5: Build the prompt
        print("[RAG] STEP 5: Building prompt")
        system_prompt = """You are a thoughtful reading companion helping a reader through a book.

You have two types of context:
- [ALREADY READ]: Content the reader has already encountered. You can reference this freely and directly.
- [COMING UP]: Content the reader hasn't reached yet. Use this ONLY to subtly guide their thinking, intuition, or attention — never quote it, reveal what happens, or spoil events.

Your goal: give meaningful, helpful responses that enrich the reading experience. If a question touches on future content, guide the reader toward noticing the right things without revealing outcomes."""

        if selected_text:
            user_prompt = f"""The user selected this text: "{selected_text}"

Context from the book:
{context}

Question: {question}"""
        else:
            user_prompt = f"""Context from the book:
{context}

Question: {question}"""

        print(f"[RAG] Prompt tokens (approx): {len(user_prompt) // 4}")

        # STEP 6: Generate answer with Claude
        print("[RAG] STEP 6: Generating answer with Claude")
        response = self.anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        answer = response.content[0].text if response.content else "Unable to generate response"

        print(f"[RAG] Generated answer: {answer[:100]}...")
        print("[RAG] Query complete!")
        print("=" * 60)

        return RAGResponse(
            answer=answer,
            sources=[
                {
                    'text': result.chunk.text[:200] + '...',
                    'full_text': result.chunk.text,
                    'score': result.score,
                    'chunk_index': result.chunk.metadata['chunk_index']
                }
                for result in results
            ]
        )

    async def query_simple(self, question: str, selected_text: str) -> str:
        """
        Simple query without RAG (just uses Claude with selected text).
        Good for quick questions about highlighted text.

        Args:
            question: User's question
            selected_text: Selected text

        Returns:
            Claude's answer
        """
        response = self.anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system="You are a helpful reading assistant. Answer questions about the provided text concisely.",
            messages=[
                {
                    "role": "user",
                    "content": f'Text: "{selected_text}"\n\nQuestion: {question}'
                }
            ]
        )

        return response.content[0].text if response.content else "Unable to generate response"
