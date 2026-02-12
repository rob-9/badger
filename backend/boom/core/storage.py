"""Simple file-based storage for documents and reading state."""

from pathlib import Path
from typing import Optional
import json


class BookLibrary:
    """Manage your personal book library."""

    def __init__(self, library_path: Path):
        """Initialize library with storage path."""
        self.library_path = library_path
        self.library_path.mkdir(parents=True, exist_ok=True)

    def add_book(self, file_path: Path) -> str:
        """
        Add a book to your library.

        Returns:
            book_id
        """
        # TODO: Copy file to library
        # TODO: Extract metadata
        # TODO: Save to index
        pass

    def get_book(self, book_id: str) -> Optional[dict]:
        """Get book metadata and location."""
        # TODO: Load from index
        pass

    def list_books(self) -> list:
        """List all books in library."""
        # TODO: Return all books
        pass

    def save_reading_position(self, book_id: str, position: dict):
        """Save current reading position."""
        # TODO: Save position (chapter, page, etc.)
        pass

    def get_reading_position(self, book_id: str) -> Optional[dict]:
        """Get saved reading position."""
        # TODO: Load position
        pass
