"""
Tests for Invoice Data Extractor Agent
"""

import pytest
from gen_z_agent.agents.extractor import create_extractor_agent


def test_create_extractor_agent(mock_llm):
    """Test creating the extractor agent."""
    agent = create_extractor_agent(mock_llm)

    assert agent is not None
    assert "Invoice Data Extractor" in agent.role
    assert agent.verbose is True
    assert agent.allow_delegation is False


def test_extractor_agent_has_tools(mock_llm):
    """Test that extractor agent has required tools."""
    agent = create_extractor_agent(mock_llm)

    assert agent.tools is not None
    assert len(agent.tools) > 0


def test_extractor_agent_korean_backstory(mock_llm):
    """Test that agent has Korean language backstory."""
    agent = create_extractor_agent(mock_llm)

    assert "한국어" in agent.backstory or "선거" in agent.backstory
    assert "개표상황표" in agent.backstory
