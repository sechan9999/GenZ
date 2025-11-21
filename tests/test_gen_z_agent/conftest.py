"""
Pytest configuration and fixtures for Gen Z Agent tests
"""

import pytest
from pathlib import Path
import tempfile
import os


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_pdf_path(temp_dir):
    """Create a sample PDF file path (mock)."""
    pdf_path = temp_dir / "sample_election.pdf"
    # Create empty file for testing
    pdf_path.touch()
    return str(pdf_path)


@pytest.fixture
def sample_html_path(temp_dir):
    """Create a sample HTML file with Korean election data."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head><title>개표상황표</title></head>
    <body>
        <h1>제21대 대통령선거 개표상황</h1>
        <table>
            <tr>
                <th>후보자</th>
                <th>득표수</th>
            </tr>
            <tr>
                <td>이재명</td>
                <td>15234</td>
            </tr>
            <tr>
                <td>김문수</td>
                <td>13891</td>
            </tr>
        </table>
    </body>
    </html>
    """
    html_path = temp_dir / "sample_election.html"
    html_path.write_text(html_content, encoding='utf-8')
    return str(html_path)


@pytest.fixture
def sample_election_data():
    """Sample election data structure."""
    return {
        "document_type": "election_count_sheet",
        "extraction_confidence": 95,
        "voting_location": {
            "district": "서울특별시 강남구",
            "location": "삼성동 투표소"
        },
        "candidates": [
            {"name": "이재명", "machine_count": 15234, "human_count": 15234},
            {"name": "김문수", "machine_count": 13891, "human_count": 13891},
            {"name": "이준석", "machine_count": 5432, "human_count": 5432},
        ],
        "total_votes": 34557,
        "discrepancies": []
    }


@pytest.fixture
def sample_analysis_data():
    """Sample analysis results."""
    return {
        "analysis_id": "TEST001",
        "timestamp": "2024-01-01 12:00:00",
        "document_name": "test_election.pdf",
        "key_metrics": {
            "총 투표수": 34557,
            "후보자 수": 3,
            "투표율": "75.2%",
            "이상치": 0
        },
        "candidates": [
            {
                "name": "이재명",
                "votes": 15234,
                "vote_percentage": 44.1,
                "rank": 1,
                "party": "더불어민주당"
            },
            {
                "name": "김문수",
                "votes": 13891,
                "vote_percentage": 40.2,
                "rank": 2,
                "party": "국민의힘"
            },
            {
                "name": "이준석",
                "votes": 5432,
                "vote_percentage": 15.7,
                "rank": 3,
                "party": "개혁신당"
            }
        ],
        "anomalies": []
    }


@pytest.fixture
def mock_llm():
    """Mock LLM for testing (without actual API calls)."""
    class MockLLM:
        def __init__(self):
            self.model = "claude-sonnet-4-5-20250929"
            self.temperature = 0

        def __call__(self, *args, **kwargs):
            return "Mock LLM response"

    return MockLLM()


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test_api_key_12345")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
