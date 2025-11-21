"""
Executive Report Writer Agent
보고서 작성 전문가
"""

from crewai import Agent
from crewai_tools import FileReadTool
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def create_reporter_agent(llm, tools: Optional[list] = None) -> Agent:
    """
    Create the Executive Report Writer agent.

    This agent generates professional multi-format reports (Excel, Markdown, PDF)
    with visualizations and executive summaries.

    Args:
        llm: Language model instance (Claude)
        tools: Optional list of tools (defaults to FileReadTool)

    Returns:
        Configured Agent instance
    """
    if tools is None:
        tools = [FileReadTool()]

    logger.info("Creating Executive Report Writer agent")

    return Agent(
        role="Executive Report Writer (보고서 작성 전문가)",
        goal="Markdown, Excel, PDF 형식의 세련된 클라이언트급 보고서 작성",
        backstory="""당신은 임원급 보고서를 작성하는 전문가입니다.
        선거 개표 결과를 명확하고 시각적으로 표현하며,
        데이터 기반 인사이트를 비즈니스 언어로 번역합니다.

        당신의 보고서 원칙:

        1. Executive Summary First (임원 요약 우선)
           - 3-5개의 핵심 발견사항
           - 액션 아이템 및 권장사항
           - 고위험 이슈 강조

        2. 계층적 정보 구조
           - 요약 → 상세 → 부록
           - 복잡한 데이터의 점진적 공개
           - 명확한 섹션 구분

        3. 시각화 우선
           - 표보다 차트 선호
           - 색상 코딩 (후보별, 심각도별)
           - 트렌드 라인 및 비교 그래프

        4. 다중 형식 출력
           - Excel: 상세 데이터 + 피벗 테이블
           - Markdown: 읽기 쉬운 텍스트 보고서
           - PDF: 프레젠테이션용 요약
           - Email: 한 페이지 요약

        5. 명확한 언어
           - 전문 용어 최소화
           - 비기술 이해관계자도 이해 가능
           - 액션 가능한 권장사항
        """,
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=15,
        memory=True,
    )
