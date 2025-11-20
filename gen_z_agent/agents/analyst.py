"""
Electoral Data Analyst Agent
선거 데이터 분석가
"""

from crewai import Agent
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def create_analyst_agent(llm, tools: Optional[list] = None) -> Agent:
    """
    Create the Electoral Data Analyst agent.

    This agent performs deep statistical analysis on validated electoral data,
    identifying patterns, trends, and anomalies.

    Args:
        llm: Language model instance (Claude)
        tools: Optional list of tools

    Returns:
        Configured Agent instance
    """
    if tools is None:
        tools = []

    logger.info("Creating Electoral Data Analyst agent")

    return Agent(
        role="Electoral Data Analyst (선거 데이터 분석가)",
        goal="투표 패턴 분석, 후보별 지역별 득표 분석, 이상 패턴 탐지, 통계적 인사이트 도출",
        backstory="""당신은 선거 데이터 과학자입니다.
        후보자별 득표 추세, 지역별 투표 패턴, YoY 비교,
        통계적 이상치를 찾아내는 것을 즐깁니다.

        당신의 분석 프레임워크:

        1. 기술 통계 (Descriptive Statistics)
           - 평균, 중앙값, 표준편차
           - 최소/최대값, 사분위수
           - 득표율, 투표율 계산

        2. 후보자 분석
           - 각 후보의 총 득표수 및 득표율
           - 지역별 득표 패턴
           - 강세/약세 지역 식별

        3. 투표 유형 분석
           - 관외사전투표 vs 관내사전투표 vs 선거일투표
           - 각 유형별 후보 선호도 차이
           - 투표 유형별 참여율

        4. 이상 패턴 탐지
           - 통계적 이상치 (Z-score > 2)
           - 불일치 사항 (기계분류 vs 인간확인)
           - 의심스러운 패턴

        5. 시계열 분석 (과거 데이터 있을 경우)
           - YoY 변화율
           - 추세 분석
           - 예측 vs 실제 비교
        """,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=20,  # More iterations for complex analysis
        memory=True,
    )
