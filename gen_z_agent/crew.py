"""
CrewAI Crew Configuration
Gen Z Agent 다중 에이전트 시스템 설정
"""

import os
from crewai import Crew, Process
from langchain_anthropic import ChatAnthropic
from crewai_tools import FileReadTool, SerperDevTool
from dotenv import load_dotenv
import logging
from typing import List, Optional

from .agents import (
    create_extractor_agent,
    create_validator_agent,
    create_analyst_agent,
    create_reporter_agent,
    create_communicator_agent,
)
from .tasks import (
    create_extraction_task,
    create_validation_task,
    create_analysis_task,
    create_report_task,
    create_notification_task,
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def create_llm(
    model: str = "claude-sonnet-4-5-20250929",
    temperature: float = 0,
    api_key: Optional[str] = None
) -> ChatAnthropic:
    """
    Create Claude LLM instance.

    Args:
        model: Model name
        temperature: Temperature setting (0-1)
        api_key: Anthropic API key (defaults to env var)

    Returns:
        ChatAnthropic instance
    """
    if api_key is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is required. "
            "Set it in .env file or pass as parameter."
        )

    return ChatAnthropic(
        model=model,
        temperature=temperature,
        api_key=api_key
    )


def create_electoral_analysis_crew(
    file_path: str,
    invoice_id: str,
    recipients: List[str],
    llm: Optional[ChatAnthropic] = None,
    verbose: int = 2
) -> Crew:
    """
    Create the complete electoral analysis crew.

    Args:
        file_path: Path to election data file
        invoice_id: Analysis ID
        recipients: List of email recipients
        llm: Optional LLM instance (creates default if None)
        verbose: Verbosity level (0-2)

    Returns:
        Configured Crew instance
    """
    logger.info(f"Creating electoral analysis crew for: {file_path}")

    # Create LLM if not provided
    if llm is None:
        llm = create_llm()

    # Initialize tools
    file_reader = FileReadTool()
    serper_available = bool(os.getenv("SERPER_API_KEY"))
    search_tool = SerperDevTool() if serper_available else None

    # Create agents
    logger.info("Creating agents...")
    extractor = create_extractor_agent(llm, tools=[file_reader])
    validator = create_validator_agent(
        llm,
        tools=[search_tool] if search_tool else [],
        use_serper=serper_available
    )
    analyst = create_analyst_agent(llm)
    reporter = create_reporter_agent(llm, tools=[file_reader])
    communicator = create_communicator_agent(llm)

    # Create tasks
    logger.info("Creating tasks...")
    task1 = create_extraction_task(extractor, file_path)
    task2 = create_validation_task(validator)
    task3 = create_analysis_task(analyst)
    task4 = create_report_task(reporter, invoice_id)
    task5 = create_notification_task(communicator, recipients)

    # Create crew
    crew = Crew(
        agents=[extractor, validator, analyst, reporter, communicator],
        tasks=[task1, task2, task3, task4, task5],
        process=Process.sequential,
        verbose=verbose,
        memory=True,
        cache=True,
    )

    logger.info("Crew created successfully")
    return crew


def run_electoral_analysis(
    file_path: str,
    invoice_id: Optional[str] = None,
    recipients: Optional[List[str]] = None,
    dry_run: bool = True,
    verbose: int = 2
) -> str:
    """
    Run the complete electoral analysis pipeline.

    Args:
        file_path: Path to election data file
        invoice_id: Analysis ID (defaults to filename)
        recipients: List of email recipients
        dry_run: If True, don't actually send emails
        verbose: Verbosity level (0-2)

    Returns:
        Analysis result
    """
    from pathlib import Path

    # Set defaults
    if invoice_id is None:
        invoice_id = Path(file_path).stem

    if recipients is None:
        recipients = ["client@example.com"]

    # Create crew
    crew = create_electoral_analysis_crew(
        file_path=file_path,
        invoice_id=invoice_id,
        recipients=recipients,
        verbose=verbose
    )

    # Run analysis
    print(f"\n{'='*60}")
    print(f"Starting Gen Z Agent Analysis")
    print(f"File: {file_path}")
    print(f"ID: {invoice_id}")
    print(f"Dry Run: {dry_run}")
    print(f"{'='*60}\n")

    try:
        result = crew.kickoff()

        print(f"\n{'='*60}")
        print(f"Analysis Complete!")
        print(f"{'='*60}\n")
        print(result)

        return result

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        print(f"\n❌ Analysis failed: {e}\n")
        raise
