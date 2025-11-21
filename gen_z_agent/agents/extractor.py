"""
Invoice Data Extractor Agent
청구서 데이터 추출 전문가
"""

from crewai import Agent
from crewai_tools import FileReadTool
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def create_extractor_agent(llm, tools: Optional[list] = None) -> Agent:
    """
    Create the Invoice Data Extractor agent.

    This agent specializes in extracting structured data from Korean election
    count sheets (개표상황표) and invoices in PDF/HTML formats.

    Args:
        llm: Language model instance (Claude)
        tools: Optional list of tools (defaults to FileReadTool)

    Returns:
        Configured Agent instance
    """
    if tools is None:
        tools = [FileReadTool()]

    logger.info("Creating Invoice Data Extractor agent")

    return Agent(
        role="Invoice Data Extractor (청구서 데이터 추출 전문가)",
        goal="한국어 선거 개표상황표 및 청구서에서 모든 구조화된 데이터를 완벽하게 추출",
        backstory="""당신은 OCR 및 스캔 문서 읽기 전문가입니다.
        특히 한국어 선거 개표상황표의 복잡한 테이블 구조를 이해하고,
        후보자별 득표수, 투표소 정보, 검증 데이터를 정확히 추출할 수 있습니다.

        당신의 강점:
        - 복잡한 PDF 레이아웃에서 정확한 데이터 추출
        - 한글 텍스트 인식 및 처리
        - 테이블 구조 이해 및 파싱
        - 불완전한 데이터에서 패턴 인식
        """,
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=15,
        memory=True,
    )
