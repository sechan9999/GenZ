"""
PDF Parser Tool with Korean Text Support
한국어 텍스트를 지원하는 PDF 파서 도구
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import unicodedata
import re

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

from crewai_tools import BaseTool

logger = logging.getLogger(__name__)


class KoreanPDFParser:
    """
    Parser for Korean PDF documents with specialized table extraction.
    한국어 PDF 문서 파서 (테이블 추출 특화)
    """

    @staticmethod
    def normalize_korean_text(text: str) -> str:
        """
        Normalize Korean text to NFC form and clean whitespace.

        Args:
            text: Input Korean text

        Returns:
            Normalized text
        """
        # Normalize to NFC (most common form for Korean)
        text = unicodedata.normalize('NFC', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.strip()

    @staticmethod
    def is_korean_name(text: str) -> bool:
        """
        Check if text appears to be a Korean name.

        Args:
            text: Text to check

        Returns:
            True if text looks like a Korean name
        """
        # Korean names are typically 2-4 characters, all Hangul
        return bool(re.match(r'^[가-힣]{2,4}$', text.strip()))

    @staticmethod
    def extract_with_pdfplumber(pdf_path: str) -> Dict[str, Any]:
        """
        Extract text and tables using pdfplumber (recommended for tables).

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with extracted data
        """
        if not PDFPLUMBER_AVAILABLE:
            raise ImportError("pdfplumber is required. Install with: pip install pdfplumber")

        result = {
            "text": "",
            "tables": [],
            "metadata": {},
            "page_count": 0
        }

        with pdfplumber.open(pdf_path) as pdf:
            result["page_count"] = len(pdf.pages)
            result["metadata"] = pdf.metadata or {}

            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text
                page_text = page.extract_text()
                if page_text:
                    result["text"] += f"\n--- Page {page_num} ---\n{page_text}"

                # Extract tables
                tables = page.extract_tables()
                for table_num, table in enumerate(tables, 1):
                    if table:
                        result["tables"].append({
                            "page": page_num,
                            "table_num": table_num,
                            "data": table,
                            "rows": len(table),
                            "cols": len(table[0]) if table else 0
                        })

        # Normalize Korean text
        result["text"] = KoreanPDFParser.normalize_korean_text(result["text"])

        logger.info(f"Extracted {len(result['tables'])} tables from {result['page_count']} pages")
        return result

    @staticmethod
    def extract_with_pypdf2(pdf_path: str) -> Dict[str, Any]:
        """
        Extract text using PyPDF2 (fallback method).

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with extracted data
        """
        if not PYPDF2_AVAILABLE:
            raise ImportError("PyPDF2 is required. Install with: pip install PyPDF2")

        result = {
            "text": "",
            "metadata": {},
            "page_count": 0
        }

        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            result["page_count"] = len(reader.pages)
            result["metadata"] = reader.metadata or {}

            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text:
                    result["text"] += f"\n--- Page {page_num} ---\n{text}"

        # Normalize Korean text
        result["text"] = KoreanPDFParser.normalize_korean_text(result["text"])

        logger.info(f"Extracted text from {result['page_count']} pages")
        return result

    @staticmethod
    def extract_election_data(tables: List[Dict]) -> Dict[str, Any]:
        """
        Extract election-specific data from tables.
        선거 개표상황표에서 데이터 추출

        Args:
            tables: List of extracted tables

        Returns:
            Structured election data
        """
        election_data = {
            "candidates": [],
            "vote_counts": {},
            "voting_locations": [],
            "metadata": {}
        }

        for table in tables:
            data = table.get("data", [])
            if not data:
                continue

            # Look for candidate names (Korean names in cells)
            for row in data:
                for cell in row:
                    if cell and KoreanPDFParser.is_korean_name(cell):
                        normalized_name = KoreanPDFParser.normalize_korean_text(cell)
                        if normalized_name not in election_data["candidates"]:
                            election_data["candidates"].append(normalized_name)

            # Store table for further processing
            election_data["vote_counts"][f"table_{table['page']}_{table['table_num']}"] = data

        logger.info(f"Found {len(election_data['candidates'])} candidates")
        return election_data

    @staticmethod
    def parse_pdf(pdf_path: str, extract_tables: bool = True) -> Dict[str, Any]:
        """
        Main parsing method that tries pdfplumber first, falls back to PyPDF2.

        Args:
            pdf_path: Path to PDF file
            extract_tables: Whether to extract tables (requires pdfplumber)

        Returns:
            Dictionary with all extracted data
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Parsing PDF: {pdf_path}")

        try:
            if extract_tables and PDFPLUMBER_AVAILABLE:
                result = KoreanPDFParser.extract_with_pdfplumber(pdf_path)
                # Extract election-specific data
                if result["tables"]:
                    result["election_data"] = KoreanPDFParser.extract_election_data(result["tables"])
                return result
            elif PYPDF2_AVAILABLE:
                logger.warning("pdfplumber not available, using PyPDF2 (no table extraction)")
                return KoreanPDFParser.extract_with_pypdf2(pdf_path)
            else:
                raise ImportError(
                    "No PDF parsing library available. "
                    "Install pdfplumber or PyPDF2: pip install pdfplumber PyPDF2"
                )
        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")
            raise


class PDFParserTool(BaseTool):
    """
    CrewAI tool wrapper for PDF parsing functionality.
    """
    name: str = "PDF Parser"
    description: str = (
        "Parse PDF files with Korean text support. "
        "Extracts text, tables, and election data from PDF documents. "
        "Use this tool when you need to read and extract data from Korean election PDFs."
    )

    def _run(self, pdf_path: str, extract_tables: bool = True) -> str:
        """
        Run the PDF parser tool.

        Args:
            pdf_path: Path to PDF file
            extract_tables: Whether to extract tables

        Returns:
            JSON string with extracted data
        """
        try:
            result = KoreanPDFParser.parse_pdf(pdf_path, extract_tables)

            # Format result as readable text
            output = f"PDF Analysis Results for: {pdf_path}\n\n"
            output += f"Pages: {result.get('page_count', 0)}\n"
            output += f"Tables found: {len(result.get('tables', []))}\n\n"

            if result.get("election_data"):
                ed = result["election_data"]
                output += "Election Data:\n"
                output += f"  Candidates: {', '.join(ed['candidates'])}\n\n"

            output += "Full Text:\n"
            output += result.get("text", "")[:2000]  # Truncate for readability

            return output

        except Exception as e:
            logger.error(f"PDF parsing failed: {e}")
            return f"Error parsing PDF: {str(e)}"
