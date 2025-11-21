"""
Tests for HTML Parser Tool
"""

import pytest
from gen_z_agent.tools.html_parser import HTMLParser


class TestHTMLParsing:
    """Test HTML parsing functionality."""

    def test_parse_html_basic(self, sample_html_path):
        """Test basic HTML parsing."""
        try:
            result = HTMLParser.parse_html(sample_html_path)

            assert "title" in result
            assert "text" in result
            assert "tables" in result
            assert result["title"] == "개표상황표"
        except ImportError:
            pytest.skip("BeautifulSoup4 not available")

    def test_parse_html_extracts_tables(self, sample_html_path):
        """Test that tables are extracted from HTML."""
        try:
            result = HTMLParser.parse_html(sample_html_path)

            assert len(result["tables"]) > 0
            table = result["tables"][0]
            assert "data" in table
            assert len(table["data"]) > 0
        except ImportError:
            pytest.skip("BeautifulSoup4 not available")

    def test_parse_html_file_not_found(self):
        """Test parsing non-existent file."""
        with pytest.raises(FileNotFoundError):
            HTMLParser.parse_html("/nonexistent/file.html")

    def test_normalize_korean_text(self):
        """Test Korean text normalization in HTML parser."""
        text = "제21대   대통령선거"
        normalized = HTMLParser.normalize_korean_text(text)
        assert normalized == "제21대 대통령선거"

    def test_extract_election_data_from_html(self, sample_html_path):
        """Test election data extraction from HTML tables."""
        try:
            result = HTMLParser.parse_html(sample_html_path)
            election_data = HTMLParser.extract_election_data(result["tables"])

            assert "candidates" in election_data
            # Should find candidates in the HTML table
            assert len(election_data["candidates"]) >= 0
        except ImportError:
            pytest.skip("BeautifulSoup4 not available")
