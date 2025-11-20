"""
Data Validator & Enricher Agent
데이터 검증 및 보강 전문가
"""

from crewai import Agent
from crewai_tools import SerperDevTool
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def create_validator_agent(llm, tools: Optional[list] = None, use_serper: bool = False) -> Agent:
    """
    Create the Data Validator & Enricher agent.

    This agent validates extracted data for completeness and correctness,
    performs statistical checks, and enriches data with external sources.

    Args:
        llm: Language model instance (Claude)
        tools: Optional list of tools
        use_serper: Whether to include SerperDevTool for web search

    Returns:
        Configured Agent instance
    """
    if tools is None:
        tools = []
        if use_serper:
            tools.append(SerperDevTool())

    logger.info("Creating Data Validator & Enricher agent")

    return Agent(
        role="Data Validator & Enricher (데이터 검증 및 보강 전문가)",
        goal="추출된 데이터를 검증하고 외부 컨텍스트로 보강 (후보자 정당, 투표율, 이상치 등)",
        backstory="""당신은 법의학 회계사이자 선거 데이터 분석 전문가입니다.
        잘못된 데이터를 절대 통과시키지 않으며, 득표수 합계 검증, 투표율 계산,
        통계적 이상치 탐지를 수행합니다.

        당신의 검증 프로토콜:
        1. 합계 검증 (Sum validation)
           - 기계분류 vs 인간확인 득표수 비교
           - 총 득표수 = 각 후보 득표수 합계

        2. 데이터 무결성 검사
           - 누락된 필드 탐지
           - 날짜/시간 형식 검증
           - 숫자 범위 검증 (음수, 이상치)

        3. 통계적 이상치 탐지
           - Z-score 계산 (2 표준편차 초과)
           - 이상 패턴 플래그

        4. 데이터 보강
           - 후보자 정당 정보 추가
           - 역사적 데이터와 비교
           - 신뢰도 점수 계산
        """,
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=15,
        memory=True,
    )
