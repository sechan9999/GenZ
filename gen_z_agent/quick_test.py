#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test script for Gen Z Agent with embedded API configuration
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Get API key from environment (will be set by the execution context)
api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    print("❌ No API key found in environment")
    print("This script needs to be run from within the Claude environment")
    sys.exit(1)

# Now import and run the demo
try:
    from crewai import Agent, Task, Crew, Process
    from langchain_anthropic import ChatAnthropic
    from dotenv import load_dotenv

    # Override with explicit API key
    os.environ["ANTHROPIC_API_KEY"] = api_key

    print("✅ API key configured")
    print(f"✅ API key length: {len(api_key)} characters")
    print("\nStarting election data analysis...")
    print("=" * 70)

    # Initialize LLM
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        anthropic_api_key=api_key,
        temperature=0
    )

    # Create a simple test agent
    test_agent = Agent(
        role="Election Data Analyst",
        goal="Analyze Korean election data",
        backstory="You are an expert in analyzing Korean election results.",
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    # Sample election data
    election_data = """
    제21대 국회의원선거 개표상황표
    Election Count Sheet - 21st National Assembly

    선거구: 서울 강남구 갑
    Region: Seoul Gangnam-gu Gap
    선거일: 2024년 4월 10일
    Date: April 10, 2024

    후보자 득표 현황:
    Candidate Vote Counts:

    1. 김철수 (Kim Chulsoo) - 45,678 표 (42.3%)
    2. 이영희 (Lee Younghee) - 38,234 표 (35.4%)
    3. 박민수 (Park Minsu) - 24,089 표 (22.3%)

    총 득표수 / Total Votes: 108,001
    투표율 / Turnout: 68.5%
    무효표 / Invalid Votes: 1,234
    """

    # Create a simple analysis task
    analysis_task = Task(
        description=f"""
        Analyze this Korean election data and provide:
        1. Winner identification
        2. Vote share analysis
        3. Margin of victory
        4. Key insights in both Korean and English

        Data:
        {election_data}
        """,
        expected_output="Bilingual analysis with winner and statistics",
        agent=test_agent
    )

    # Create and run crew
    crew = Crew(
        agents=[test_agent],
        tasks=[analysis_task],
        process=Process.sequential,
        verbose=2
    )

    print("\n🚀 Running single-agent election analysis...")
    print("=" * 70)

    result = crew.kickoff()

    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 70)
    print("\n📊 RESULT:")
    print("=" * 70)
    print(result)
    print("=" * 70)

except ImportError as e:
    print(f"❌ Missing package: {e}")
    print("Please install: pip install crewai langchain-anthropic")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
