"""
Tests for Configuration Management
"""

import pytest
from gen_z_agent.config import Config


def test_config_has_required_attributes():
    """Test that Config has all required attributes."""
    assert hasattr(Config, 'ANTHROPIC_API_KEY')
    assert hasattr(Config, 'CLAUDE_MODEL')
    assert hasattr(Config, 'OUTPUT_DIR')
    assert hasattr(Config, 'CANDIDATES')


def test_config_candidates():
    """Test candidate configuration."""
    assert len(Config.CANDIDATES) > 0
    assert 1 in Config.CANDIDATES
    assert "name" in Config.CANDIDATES[1]
    assert "party" in Config.CANDIDATES[1]


def test_config_vote_types():
    """Test vote type configuration."""
    assert len(Config.VOTE_TYPES) > 0
    assert "관외사전투표" in Config.VOTE_TYPES
    assert Config.VOTE_TYPES["관외사전투표"] == "out_of_area_early_voting"


def test_get_candidate_info_valid():
    """Test getting candidate info for valid candidate number."""
    info = Config.get_candidate_info(1)
    assert "name" in info
    assert "party" in info
    assert info["name"] == "이재명"


def test_get_candidate_info_invalid():
    """Test getting candidate info for invalid candidate number."""
    info = Config.get_candidate_info(999)
    assert "name" in info
    assert "알 수 없음" in info["party"]


def test_get_vote_type_english():
    """Test converting Korean vote type to English."""
    english = Config.get_vote_type_english("관외사전투표")
    assert english == "out_of_area_early_voting"


def test_config_info():
    """Test getting configuration summary."""
    info = Config.info()
    assert "model" in info
    assert "temperature" in info
    assert info["model"] == "claude-sonnet-4-5-20250929"


def test_config_output_path():
    """Test getting output file path."""
    path = Config.get_output_path("test.xlsx")
    assert "test.xlsx" in str(path)
    assert str(Config.OUTPUT_DIR) in str(path)
