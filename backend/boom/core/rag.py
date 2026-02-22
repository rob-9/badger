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

import asyncio
import json
import logging
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import voyageai
from anthropic import Anthropic

from boom import config
from .chunker import chunk_text, chunk_structured, TextChunk
from .vector_store import VectorStore, VectorEntry

logger = logging.getLogger(__name__)

LOG_DIR = Path(".data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class RAGResponse:
    """Response from RAG query."""
    answer: str
    sources: list[dict]


class RAGService:
    """Simple RAG for personal reading assistant."""

    def __init__(self, storage_dir: str = config.VECTOR_STORAGE_DIR):
        """
        Initialize RAG service.

        Args:
            storage_dir: Directory to store vector embeddings
        """
        self.voyage = voyageai.Client(api_key=config.VOYAGE_API_KEY)
        self.anthropic = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.vector_store = VectorStore(storage_dir)
        logger.info("Service initialized")

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
        logger.debug("Embedding text (%s), %d chars", input_type, len(text))

        response = self.voyage.embed(
            texts=[text],
            model=config.VOYAGE_CONTEXT_MODEL,
            input_type=input_type
        )

        if not response.embeddings or len(response.embeddings) == 0:
            raise ValueError("No embedding returned from Voyage AI")

        embedding = response.embeddings[0]
        logger.debug("Generated embedding with %d dimensions", len(embedding))
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
        logger.info("Embedding %d texts in batch (%s)", len(texts), input_type)

        # Voyage API limits batches to 320k tokens; split into smaller batches
        # by estimating ~4 chars per token and staying well under the limit.
        MAX_CHARS_PER_BATCH = 200_000  # ~50k tokens, safe margin
        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_chars = 0

        for text in texts:
            text_len = len(text)
            if current_batch and current_chars + text_len > MAX_CHARS_PER_BATCH:
                batches.append(current_batch)
                current_batch = []
                current_chars = 0
            current_batch.append(text)
            current_chars += text_len

        if current_batch:
            batches.append(current_batch)

        logger.info("Split into %d batches", len(batches))

        all_embeddings: list[list[float]] = []
        for i, batch in enumerate(batches):
            response = self.voyage.embed(
                texts=batch,
                model=config.VOYAGE_CONTEXT_MODEL,
                input_type=input_type
            )
            if not response.embeddings:
                raise ValueError(f"No embeddings returned from Voyage AI (batch {i+1})")
            all_embeddings.extend(response.embeddings)
            logger.info("Batch %d/%d: %d embeddings", i + 1, len(batches), len(response.embeddings))

        logger.info("Generated %d total embeddings", len(all_embeddings))
        return all_embeddings

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
        logger.info("Indexing book %s (%d chars)", book_id, len(text))

        # Check if already indexed
        if self.vector_store.has_book(book_id):
            logger.info("Book %s already indexed, skipping", book_id)
            return

        # Chunk the text
        chunks = chunk_text(text, book_id)
        logger.info("Created %d chunks (avg %d chars)", len(chunks), len(text) // len(chunks) if chunks else 0)

        # Get embeddings for all chunks (batched)
        texts = [chunk.text for chunk in chunks]
        embeddings = await self.get_embeddings(texts, input_type="document")

        # Create vector entries and store
        entries = [
            VectorEntry(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunks, embeddings)
        ]
        await self.vector_store.add_book(book_id, entries)

        logger.info("Indexing complete for %s", book_id)

    async def get_contextualized_embeddings(
        self,
        chunks: list[TextChunk],
    ) -> list[list[float]]:
        """
        Get contextualized embeddings using voyage-context-3.

        Groups chunks by chapter so each chapter is a separate "document" in the
        API call, preserving within-chapter context. Multiple chapters are packed
        per request up to API limits (32K tokens/document, 120K tokens/request).
        """
        logger.info("Contextualized embedding for %d chunks", len(chunks))

        # voyage-context-3 limits
        MAX_CHARS_PER_DOC = 80_000     # ~20-27K tokens depending on content, safe under 32K limit
        MAX_CHARS_PER_REQUEST = 460_000  # ~115K tokens, safe margin under 120K limit

        # Group chunks by chapter, preserving order
        chapter_groups: list[list[str]] = []
        current_chapter_idx = None
        for chunk in chunks:
            ch_idx = chunk.metadata.get("chapter_index")
            if ch_idx != current_chapter_idx:
                chapter_groups.append([])
                current_chapter_idx = ch_idx
            chapter_groups[- 1].append(chunk.text)

        # Warn about oversized chapters
        for i, group in enumerate(chapter_groups):
            group_chars = sum(len(t) for t in group)
            if group_chars > MAX_CHARS_PER_DOC:
                logger.warning(
                    "Chapter %d has %d chars (~%d tokens) — exceeds per-document limit, "
                    "context at split boundaries will be reduced",
                    i, group_chars, group_chars // 4,
                )

        # Split oversized chapters into sub-groups that fit the per-doc limit
        documents: list[list[str]] = []
        for group in chapter_groups:
            group_chars = sum(len(t) for t in group)
            if group_chars <= MAX_CHARS_PER_DOC:
                documents.append(group)
            else:
                sub: list[str] = []
                sub_chars = 0
                for text in group:
                    if sub and sub_chars + len(text) > MAX_CHARS_PER_DOC:
                        documents.append(sub)
                        sub = []
                        sub_chars = 0
                    sub.append(text)
                    sub_chars += len(text)
                if sub:
                    documents.append(sub)

        # Pack documents into requests respecting the per-request limit
        requests: list[list[list[str]]] = []
        current_request: list[list[str]] = []
        current_request_chars = 0

        for doc in documents:
            doc_chars = sum(len(t) for t in doc)
            if current_request and current_request_chars + doc_chars > MAX_CHARS_PER_REQUEST:
                requests.append(current_request)
                current_request = []
                current_request_chars = 0
            current_request.append(doc)
            current_request_chars += doc_chars

        if current_request:
            requests.append(current_request)

        logger.info(
            "Split %d chapters into %d documents across %d API requests",
            len(chapter_groups), len(documents), len(requests),
        )

        all_embeddings: list[list[float]] = []
        for i, req_docs in enumerate(requests):
            req_chunks = sum(len(d) for d in req_docs)
            req_chars = sum(sum(len(t) for t in d) for d in req_docs)
            logger.info(
                "Embedding request %d/%d: %d docs, %d chunks, ~%dK tokens",
                i + 1, len(requests), len(req_docs), req_chunks, req_chars // 4000,
            )
            result = await asyncio.to_thread(
                self.voyage.contextualized_embed,
                inputs=req_docs,
                model=config.VOYAGE_CONTEXT_MODEL,
                input_type="document",
            )
            for doc_result in result.results:
                all_embeddings.extend(doc_result.embeddings)
            logger.info("Embedding request %d/%d complete", i + 1, len(requests))

        logger.info("Generated %d contextualized embeddings", len(all_embeddings))
        return all_embeddings

    async def index_book_structured(self, book_id: str, structured_content: dict) -> None:
        """
        Index a book using structure-aware chunking + voyage-context-3.
        """
        logger.info("Indexing structured book %s", book_id)

        if self.vector_store.has_book(book_id):
            logger.info("Book %s already indexed, skipping", book_id)
            return

        chunks = chunk_structured(structured_content, book_id)
        if not chunks:
            logger.warning("No chunks produced for %s — structured content may be empty", book_id)
            return

        logger.info("Created %d structured chunks", len(chunks))

        embeddings = await self.get_contextualized_embeddings(chunks)

        entries = [
            VectorEntry(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunks, embeddings)
        ]
        await self.vector_store.add_book(book_id, entries)

        # Generate chapter-level index entries using Haiku (fast + cheap).
        # These aren't narrative summaries — they're retrieval targets optimized
        # for matching broad thematic questions to the right chapter.
        CHAPTER_INDEX_PROMPT = (
            "For this chapter, extract:\n"
            "1. The main themes and arguments\n"
            "2. Key names, terms, and concepts introduced or discussed\n"
            "3. What is established or changes\n\n"
            "Keep it to 3-4 sentences. Optimize for searchability — "
            "someone will be searching for these topics later.\n\n"
        )

        chapters = structured_content.get("chapters", [])

        # Build (index, title, text) for chapters with enough content
        chapter_inputs: list[tuple[int, str, str]] = []
        for i, chapter in enumerate(chapters):
            chapter_title = chapter.get("title", f"Chapter {i + 1}")
            chapter_text = "\n\n".join(
                para
                for section in chapter.get("sections", [])
                for para in section.get("paragraphs", [])
            )[:8000]
            if len(chapter_text) >= 50:
                chapter_inputs.append((i, chapter_title, chapter_text))

        # Limit concurrent Haiku calls to stay under rate limits
        # (50K input tokens/min — each chapter sends ~2K tokens)
        sem = asyncio.Semaphore(2)

        async def _summarize(idx: int, title: str, text: str) -> TextChunk | None:
            async with sem:
                try:
                    response = await asyncio.to_thread(
                        self.anthropic.messages.create,
                        model=config.CLAUDE_HAIKU_MODEL,
                        max_tokens=200,
                        messages=[{
                            "role": "user",
                            "content": f"{CHAPTER_INDEX_PROMPT}Chapter: {title}\n\n{text}",
                        }],
                    )
                except Exception as e:
                    logger.warning("Failed to summarize chapter %d (%s): %s", idx, title, e)
                    return None
                await asyncio.sleep(1)
            summary_text = response.content[0].text if response.content else ""
            if not summary_text:
                return None
            return TextChunk(
                id=f"{book_id}-summary-{idx}",
                text=summary_text,
                metadata={
                    "book_id": book_id,
                    "chunk_index": idx,
                    "chapter_title": title,
                    "chapter_index": idx,
                    "tier": "chapter_summary",
                },
            )

        results = await asyncio.gather(
            *(_summarize(i, t, txt) for i, t, txt in chapter_inputs)
        )
        summary_chunks: list[TextChunk] = [c for c in results if c is not None]

        if summary_chunks:
            summary_embeddings = await self.get_contextualized_embeddings(summary_chunks)
            summary_entries = [
                VectorEntry(chunk=chunk, embedding=emb)
                for chunk, emb in zip(summary_chunks, summary_embeddings)
            ]
            await self.vector_store.save_summaries(book_id, summary_entries)
            logger.info("Generated %d chapter summaries for %s", len(summary_chunks), book_id)

        logger.info("Structured indexing complete for %s", book_id)

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
        logger.info("Query book=%s question=%r", book_id, question[:80])

        # Embed the question (with selected text for context if provided)
        if selected_text:
            query_text = f"{question}\n\nReferring to: {selected_text}"
        else:
            query_text = question

        question_embedding = await self.get_embedding(query_text, input_type="query")

        # Find relevant chunks
        results = await self.vector_store.search(book_id, question_embedding, top_k=5)

        if not results:
            logger.info("No relevant chunks found")
            return RAGResponse(
                answer="I couldn't find relevant information in the book to answer this question.",
                sources=[]
            )

        for i, result in enumerate(results):
            chunk_idx = result.chunk.metadata['chunk_index']
            logger.debug("  %d. Chunk %d (similarity: %.3f)", i + 1, chunk_idx, result.score)

        # Tag chunks as PAST or AHEAD based on reader position
        total_chunks = await self.vector_store.get_total_chunks(book_id)
        reader_chunk_index = int((reader_position or 0) * total_chunks)
        logger.debug("Reader at chunk ~%d of %d", reader_chunk_index, total_chunks)

        past_chunks = []
        ahead_chunks = []
        for result in results:
            if result.chunk.metadata['chunk_index'] <= reader_chunk_index:
                past_chunks.append(result)
            else:
                ahead_chunks.append(result)
        logger.debug("Past chunks: %d, Ahead chunks: %d", len(past_chunks), len(ahead_chunks))

        # Build context with labeled sections
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

        # Build the prompt
        system_prompt = """You are a reading companion.

GROUNDING RULE: Base your answer ONLY on the provided context and selected text. Do not use outside knowledge of this book. If the context doesn't contain enough information to answer, say so honestly rather than guessing.

Be direct and concise. Answer the question, then stop. No filler, no plot recaps, no dramatic narration.
Short paragraphs. Prefer 2-4 sentences over a wall of text.
Address the reader as "you." Never say "the user" or "the reader."

Context types:
- [ALREADY READ]: Content you've both covered. Reference freely.
- [COMING UP]: Content not yet reached. Use ONLY to subtly guide attention — never reveal, quote, or spoil."""

        if selected_text:
            user_prompt = f"""Selected text: "{selected_text}"

Context from the book:
{context}

Question: {question}"""
        else:
            user_prompt = f"""Context from the book:
{context}

Question: {question}"""

        # Generate answer with Claude
        response = self.anthropic.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=400,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        answer = response.content[0].text if response.content else "Unable to generate response"

        # Log full exchange to file
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "book_id": book_id,
            "reader_position": reader_position,
            "reader_chunk_index": reader_chunk_index,
            "total_chunks": total_chunks,
            "question": question,
            "selected_text": selected_text,
            "retrieved_chunks": [
                {
                    "chunk_index": r.chunk.metadata["chunk_index"],
                    "score": round(r.score, 4),
                    "label": "PAST" if r.chunk.metadata["chunk_index"] <= reader_chunk_index else "AHEAD",
                    "text": r.chunk.text,
                }
                for r in results
            ],
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response": {
                "answer": answer,
                "model": response.model,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "stop_reason": response.stop_reason,
            },
        }
        # Machine-readable log
        log_file = LOG_DIR / "queries.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Human-readable log
        readable_log = LOG_DIR / "queries.log"
        with open(readable_log, "a") as f:
            f.write("\n" + "═" * 60 + "\n")
            f.write(f"QUERY @ {log_entry['timestamp']}\n")
            f.write("═" * 60 + "\n\n")

            f.write(f"Book:     {book_id}\n")
            f.write(f"Position: {(reader_position or 0):.1%} (chunk {reader_chunk_index}/{total_chunks})\n")
            selected_display = (selected_text or "")[:80]
            f.write(f"Selected: \"{selected_display}{'…' if selected_text and len(selected_text) > 80 else ''}\"\n")
            f.write(f"Question: {question}\n\n")

            f.write(f"── Retrieved Chunks ({len(results)}) " + "─" * 35 + "\n")
            for i, r in enumerate(results):
                idx = r.chunk.metadata["chunk_index"]
                label = "PAST" if idx <= reader_chunk_index else "AHEAD"
                f.write(f"  [{i+1}] score={r.score:.4f} | chunk {idx} | {label}\n")
                f.write(f"      {r.chunk.text[:120]}…\n\n")

            f.write("── LLM Input " + "─" * 46 + "\n")
            f.write(f"  System: {system_prompt[:200]}…\n\n")
            f.write(f"  User:   {user_prompt[:200]}…\n\n")

            f.write("── LLM Output " + "─" * 45 + "\n")
            f.write(f"  Model:  {response.model}\n")
            f.write(f"  Tokens: {response.usage.input_tokens} in / {response.usage.output_tokens} out\n")
            f.write(f"  Stop:   {response.stop_reason}\n\n")
            f.write("  Answer:\n")
            for line in answer.split("\n"):
                f.write(f"    {line}\n")
            f.write("\n" + "═" * 60 + "\n")

        logger.info("Query complete: %d in / %d out tokens", response.usage.input_tokens, response.usage.output_tokens)

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
            model=config.CLAUDE_MODEL,
            max_tokens=1024,
            system="You are a reading companion. Be direct and concise — answer the question, then stop. Address the reader as \"you.\"",
            messages=[
                {
                    "role": "user",
                    "content": f'Text: "{selected_text}"\n\nQuestion: {question}'
                }
            ]
        )

        return response.content[0].text if response.content else "Unable to generate response"
