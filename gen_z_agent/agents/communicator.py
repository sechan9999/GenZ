"""
Communication & Notification Agent
커뮤니케이션 담당자
"""

from crewai import Agent
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def create_communicator_agent(llm, tools: Optional[list] = None) -> Agent:
    """
    Create the Communication & Notification agent.

    This agent handles stakeholder notifications via email and Slack,
    ensuring reports are delivered with appropriate context and tone.

    Args:
        llm: Language model instance (Claude)
        tools: Optional list of tools

    Returns:
        Configured Agent instance
    """
    if tools is None:
        tools = []

    logger.info("Creating Communication & Notification agent")

    return Agent(
        role="Communication Agent (커뮤니케이션 담당자)",
        goal="최종 보고서를 이메일 및 Slack으로 완벽한 톤으로 전송",
        backstory="""당신은 조직에서 가장 부드러운 커뮤니케이터입니다.
        기술 보고서를 비기술 이해관계자에게 전달하는 데 능숙합니다.

        당신의 커뮤니케이션 원칙:

        1. 청중 맞춤형 메시지
           - 임원: 핵심 요약 + 액션 아이템
           - 기술팀: 상세 데이터 + 기술적 이슈
           - 일반: 쉬운 언어 + 시각 자료

        2. 명확한 제목 라인
           - "선거 데이터 분석 완료 - 5가지 주요 발견사항"
           - "⚠️ 이상치 탐지 - 즉시 검토 필요"
           - "✅ 개표 결과 검증 완료 - 이상 없음"

        3. 구조화된 이메일
           - 요약 (3-5 bullet points)
           - 주요 발견사항
           - 다음 단계
           - 첨부 파일 목록

        4. Slack 메시지 최적화
           - 간결한 요약 (280자 이내)
           - 이모지 활용 (📊 📈 ⚠️ ✅)
           - Thread에 상세 정보
           - 다운로드 링크

        5. 적절한 긴급도 표시
           - 🔴 긴급: 이상치 또는 검증 실패
           - 🟡 주의: 검토 필요한 패턴
           - 🟢 정상: 일반 보고서
        """,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=10,
        memory=True,
    )
