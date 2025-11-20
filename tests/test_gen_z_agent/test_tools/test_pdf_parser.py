"""
Tests for PDF Parser Tool
"""

import pytest
from gen_z_agent.tools.pdf_parser import KoreanPDFParser


class TestKoreanTextProcessing:
    """Test Korean text processing utilities."""

    def test_normalize_korean_text(self):
        """Test Korean text normalization."""
        text = "이재명   득표수"
        normalized = KoreanPDFParser.normalize_korean_text(text)
        assert normalized == "이재명 득표수"

    def test_is_korean_name_valid(self):
        """Test Korean name detection - valid names."""
        assert KoreanPDFParser.is_korean_name("이재명") is True
        assert KoreanPDFParser.is_korean_name("김문수") is True
        assert KoreanPDFParser.is_korean_name("홍길동") is True

    def test_is_korean_name_invalid(self):
        """Test Korean name detection - invalid names."""
        assert KoreanPDFParser.is_korean_name("John") is False
        assert KoreanPDFParser.is_korean_name("이") is False  # Too short
        assert KoreanPDFParser.is_korean_name("이재명김문수홍") is False  # Too long
        assert KoreanPDFParser.is_korean_name("123") is False

    def test_is_korean_name_with_spaces(self):
        """Test Korean name detection with whitespace."""
        assert KoreanPDFParser.is_korean_name("  이재명  ") is True

    def test_normalize_preserves_korean_characters(self):
        """Test that normalization preserves Korean characters."""
        text = "후보자: 이재명 (더불어민주당)"
        normalized = KoreanPDFParser.normalize_korean_text(text)
        assert "이재명" in normalized
        assert "더불어민주당" in normalized


class TestElectionDataExtraction:
    """Test election-specific data extraction."""

    def test_extract_election_data_finds_candidates(self):
        """Test extraction of candidate names from tables."""
        tables = [
            {
                "page": 1,
                "table_num": 1,
                "data": [
                    ["후보자", "득표수"],
                    ["이재명", "15234"],
                    ["김문수", "13891"]
                ]
            }
        ]

        result = KoreanPDFParser.extract_election_data(tables)

        assert "이재명" in result["candidates"]
        assert "김문수" in result["candidates"]
        assert len(result["candidates"]) == 2

    def test_extract_election_data_empty_tables(self):
        """Test extraction with empty tables."""
        result = KoreanPDFParser.extract_election_data([])

        assert result["candidates"] == []
        assert result["vote_counts"] == {}

    def test_extract_election_data_no_duplicates(self):
        """Test that candidate names are not duplicated."""
        tables = [
            {
                "page": 1,
                "table_num": 1,
                "data": [
                    ["이재명", "100"],
                    ["이재명", "200"],
                    ["김문수", "150"]
                ]
            }
        ]

        result = KoreanPDFParser.extract_election_data(tables)

        # Should only have unique candidates
        assert result["candidates"].count("이재명") == 1
        assert result["candidates"].count("김문수") == 1
