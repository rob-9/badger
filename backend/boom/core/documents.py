"""Document processing for EPUB and PDF files."""

from pathlib import Path
from typing import Optional


class DocumentReader:
    """Read and parse EPUB/PDF documents."""

    def read_epub(self, file_path: Path) -> dict:
        """
        Read EPUB file and extract content.

        Returns:
            dict with title, author, chapters, and full text
        """
        # TODO: Implement EPUB parsing
        pass

    def read_pdf(self, file_path: Path) -> dict:
        """
        Read PDF file and extract content.

        Returns:
            dict with title, pages, and full text
        """
        # TODO: Implement PDF parsing
        pass
