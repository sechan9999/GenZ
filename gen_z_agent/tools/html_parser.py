"""
HTML Parser Tool for Korean Election Data
한국어 선거 데이터를 위한 HTML 파서
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import unicodedata
import re

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

from crewai_tools import BaseTool

logger = logging.getLogger(__name__)


class HTMLParser:
    """
    Parser for HTML documents with Korean text support.
    """

    @staticmethod
    def normalize_korean_text(text: str) -> str:
        """Normalize Korean text to NFC form and clean whitespace."""
        text = unicodedata.normalize('NFC', text)
        text = ' '.join(text.split())
        return text.strip()

    @staticmethod
    def parse_html(html_path: str) -> Dict[str, Any]:
        """
        Parse HTML file and extract structured data.

        Args:
            html_path: Path to HTML file

        Returns:
            Dictionary with extracted data
        """
        if not BS4_AVAILABLE:
            raise ImportError("BeautifulSoup4 is required. Install with: pip install beautifulsoup4")

        path = Path(html_path)
        if not path.exists():
            raise FileNotFoundError(f"HTML file not found: {html_path}")

        logger.info(f"Parsing HTML: {html_path}")

        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'lxml')

        result = {
            "title": "",
            "text": "",
            "tables": [],
            "links": [],
            "metadata": {}
        }

        # Extract title
        if soup.title:
            result["title"] = HTMLParser.normalize_korean_text(soup.title.string or "")

        # Extract all text
        result["text"] = HTMLParser.normalize_korean_text(soup.get_text())

        # Extract tables
        for table_num, table in enumerate(soup.find_all('table'), 1):
            rows = []
            for tr in table.find_all('tr'):
                cells = []
                for td in tr.find_all(['td', 'th']):
                    cell_text = HTMLParser.normalize_korean_text(td.get_text())
                    cells.append(cell_text)
                if cells:
                    rows.append(cells)

            if rows:
                result["tables"].append({
                    "table_num": table_num,
                    "data": rows,
                    "rows": len(rows),
                    "cols": len(rows[0]) if rows else 0
                })

        # Extract links
        for a in soup.find_all('a', href=True):
            result["links"].append({
                "text": HTMLParser.normalize_korean_text(a.get_text()),
                "href": a['href']
            })

        logger.info(f"Extracted {len(result['tables'])} tables and {len(result['links'])} links")
        return result

    @staticmethod
    def extract_election_data(tables: List[Dict]) -> Dict[str, Any]:
        """
        Extract election-specific data from HTML tables.

        Args:
            tables: List of extracted tables

        Returns:
            Structured election data
        """
        election_data = {
            "candidates": [],
            "vote_counts": {},
            "voting_locations": [],
        }

        for table in tables:
            data = table.get("data", [])
            if not data:
                continue

            # First row often contains headers
            headers = data[0] if data else []

            # Look for candidate names and vote counts
            for row_num, row in enumerate(data[1:], 1):
                for col_num, cell in enumerate(row):
                    # Check if cell contains a Korean name
                    if re.match(r'[가-힣]{2,4}', cell):
                        normalized = HTMLParser.normalize_korean_text(cell)
                        if normalized not in election_data["candidates"]:
                            election_data["candidates"].append(normalized)

            # Store table data
            election_data["vote_counts"][f"table_{table['table_num']}"] = data

        return election_data


class HTMLParserTool(BaseTool):
    """
    CrewAI tool wrapper for HTML parsing functionality.
    """
    name: str = "HTML Parser"
    description: str = (
        "Parse HTML files with Korean text support. "
        "Extracts text, tables, and election data from HTML documents. "
        "Use this tool when you need to read and extract data from Korean election HTML files."
    )

    def _run(self, html_path: str) -> str:
        """
        Run the HTML parser tool.

        Args:
            html_path: Path to HTML file

        Returns:
            Formatted string with extracted data
        """
        try:
            result = HTMLParser.parse_html(html_path)

            # Extract election data if tables found
            if result["tables"]:
                result["election_data"] = HTMLParser.extract_election_data(result["tables"])

            # Format result as readable text
            output = f"HTML Analysis Results for: {html_path}\n\n"
            output += f"Title: {result.get('title', 'N/A')}\n"
            output += f"Tables found: {len(result.get('tables', []))}\n"
            output += f"Links found: {len(result.get('links', []))}\n\n"

            if result.get("election_data"):
                ed = result["election_data"]
                output += "Election Data:\n"
                output += f"  Candidates: {', '.join(ed['candidates'])}\n\n"

            output += "Full Text:\n"
            output += result.get("text", "")[:2000]  # Truncate for readability

            return output

        except Exception as e:
            logger.error(f"HTML parsing failed: {e}")
            return f"Error parsing HTML: {str(e)}"
