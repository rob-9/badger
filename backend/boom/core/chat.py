"""AI chat service for conversing about books."""

from typing import List, Optional
from dataclasses import dataclass


@dataclass
class Message:
    """A chat message."""
    role: str  # 'user' or 'assistant'
    content: str


class ChatService:
    """Chat with AI about your reading."""

    def __init__(self):
        """Initialize chat with Claude client."""
        self.history: List[Message] = []

    async def chat(
        self,
        query: str,
        context: Optional[str] = None,
        stream: bool = False
    ) -> str:
        """
        Chat with AI about your reading.

        Args:
            query: User question
            context: Relevant text from book (from RAG)
            stream: Whether to stream response

        Returns:
            AI response
        """
        # TODO: Build prompt with context
        # TODO: Call Claude API
        # TODO: Return response
        pass

    def clear_history(self):
        """Clear conversation history."""
        self.history.clear()
