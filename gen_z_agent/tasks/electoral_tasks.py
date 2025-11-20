"""
Electoral Analysis Task Definitions
선거 데이터 분석 작업 정의
"""

from crewai import Task, Agent
from typing import List
import logging

logger = logging.getLogger(__name__)


def create_extraction_task(agent: Agent, file_path: str) -> Task:
    """
    Create data extraction task.

    Args:
        agent: Extractor agent
        file_path: Path to file to extract

    Returns:
        Configured Task
    """
    logger.info(f"Creating extraction task for: {file_path}")

    return Task(
        description=f"""
        다음 파일을 읽고 데이터를 추출하세요: '{file_path}'

        파일이 선거 개표상황표인 경우:
        - 투표소 정보 (읍면동, 투표소명)
        - 후보자별 득표수 (기계분류 ②③, 인간확인 a, b)
        - 총 득표수 및 검증 데이터
        - 불일치 사항 (①-②-③)

        파일이 청구서인 경우:
        - invoice_number, date, vendor_name, line_items, total_amount

        추출 지침:
        1. 한국어 텍스트를 정확히 인식하세요
        2. 테이블 구조를 파악하고 행/열을 올바르게 매핑하세요
        3. 숫자 데이터를 정수로 변환하세요 (쉼표 제거)
        4. 누락된 데이터는 null로 표시하세요
        5. 추출 신뢰도를 평가하세요 (0-100)

        출력 형식: JSON
        {{
            "document_type": "election_count_sheet" | "invoice",
            "extraction_confidence": 85,
            "voting_location": {{"district": "...", "location": "..."}},
            "candidates": [
                {{"name": "이재명", "machine_count": 1234, "human_count": 1234}},
                ...
            ],
            "total_votes": 12345,
            "discrepancies": [],
            "raw_tables": [...]
        }}
        """,
        expected_output="모든 필드가 포함된 깨끗한 JSON 객체",
        agent=agent
    )


def create_validation_task(agent: Agent) -> Task:
    """
    Create validation task.

    Args:
        agent: Validator agent

    Returns:
        Configured Task
    """
    logger.info("Creating validation task")

    return Task(
        description="""
        Task 1의 JSON 데이터를 가져와서 다음을 수행하세요:

        1. 합계 검증 (Sum Validation)
           - 각 후보자 득표수 합계 = 총 득표수
           - 기계분류 vs 인간확인 일치 여부
           - 오차가 있을 경우 플래그 설정

        2. 데이터 무결성 검사
           - 모든 필수 필드 존재 확인
           - 날짜 형식 검증 (YYYY-MM-DD)
           - 숫자 범위 검증 (음수 불가, 비현실적 값)

        3. 통계적 이상치 탐지
           - 각 후보자 득표수의 평균, 표준편차 계산
           - Z-score > 2.0인 값을 이상치로 플래그
           - 의심스러운 패턴 탐지

        4. 데이터 보강 (Enrichment)
           - 후보자 번호 → 이름, 정당 매핑
           - 투표 유형 한글 → 영문 키 변환
           - 역사적 데이터 조회 (가능한 경우)

        5. 신뢰도 점수 계산
           - extraction_confidence 고려
           - 검증 통과율 기반 confidence_score 계산 (0-100)

        추가할 필드:
        - category: 투표 유형 (관외사전투표, 관내사전투표, 선거일투표)
        - is_anomaly: boolean
        - validation_status: "passed" | "warning" | "failed"
        - confidence_score: 0-100
        - anomalies_detected: []
        - enriched_candidates: [...]

        출력 형식: 검증되고 보강된 JSON
        """,
        expected_output="검증되고 보강된 JSON (validation_status, confidence_score, anomalies 포함)",
        agent=agent
    )


def create_analysis_task(agent: Agent) -> Task:
    """
    Create analysis task.

    Args:
        agent: Analyst agent

    Returns:
        Configured Task
    """
    logger.info("Creating analysis task")

    return Task(
        description="""
        보강된 JSON을 사용하여 다음 분석을 수행하세요:

        1. 기술 통계 (Descriptive Statistics)
           - 총 투표수, 평균 득표수, 중앙값
           - 최소/최대 득표수
           - 표준편차, 변동계수
           - 득표율 계산 (각 후보 / 총 투표수 * 100)

        2. 후보자별 분석
           - 각 후보자의 총 득표수 및 득표율
           - 순위 (1위, 2위, ...)
           - 1위와의 득표 차이
           - 지역별 득표 패턴 (가능한 경우)
           - 강세 지역 / 약세 지역

        3. 투표 유형별 분석
           - 관외사전투표 vs 관내사전투표 vs 선거일투표
           - 각 유형별 총 투표수 및 비율
           - 각 후보자의 유형별 득표 차이
           - 이상 패턴 (특정 유형에서만 급증)

        4. 이상 패턴 탐지
           - Z-score > 2.0인 데이터 포인트
           - 기계분류 vs 인간확인 불일치 (> 5%)
           - 비현실적 득표율 (> 90% 또는 < 1%)
           - 의심스러운 패턴 설명

        5. 과거 데이터와 비교 (./historical/ 디렉토리 확인)
           - 이전 선거 대비 득표 변화율
           - 투표율 변화
           - 후보자별 성과 비교

        6. 인사이트 도출
           - 3-5개의 핵심 발견사항
           - 데이터 기반 결론
           - 추가 조사가 필요한 영역

        출력 형식: 상세 Markdown 분석 보고서
        - 표와 차트 포함
        - 핵심 메트릭 강조
        - 명확한 섹션 구분
        """,
        expected_output="""
        상세 Markdown 분석:
        - ## 기술 통계
        - ## 후보자별 분석 (표 포함)
        - ## 투표 유형별 분석
        - ## 이상 패턴 탐지
        - ## 과거 데이터 비교
        - ## 핵심 인사이트
        """,
        agent=agent
    )


def create_report_task(agent: Agent, invoice_id: str) -> Task:
    """
    Create report generation task.

    Args:
        agent: Reporter agent
        invoice_id: Analysis ID

    Returns:
        Configured Task
    """
    logger.info(f"Creating report task for: {invoice_id}")

    return Task(
        description=f"""
        다음 보고서 파일들을 생성하세요:

        1. **Excel 파일**: './gen_z_agent/output/Analysis_{invoice_id}.xlsx'

           생성 방법:
           - gen_z_agent.tools.excel_generator.ExcelReportGenerator 사용
           - 다음 시트 포함:
             * Summary: 요약 정보, 핵심 메트릭
             * Raw Data: 원본 추출 데이터
             * Enriched Data: 검증/보강된 데이터
             * Analysis: 분석 결과 (후보자별, 투표유형별)

           스타일링:
           - 헤더: 파란색 배경, 흰색 텍스트, 볼드
           - 이상치: 노란색 배경
           - 자동 열 너비 조정
           - 숫자: 천 단위 쉼표, 소수점 2자리

        2. **Markdown 보고서**: './gen_z_agent/output/Report_{invoice_id}.md'

           생성 방법:
           - gen_z_agent.tools.markdown_generator.MarkdownReportGenerator 사용
           - 다음 섹션 포함:
             * # 선거 데이터 분석 보고서
             * ## 📊 임원 요약 (Executive Summary)
             * ## 🗳️ 후보자별 분석
             * ## 📮 투표 유형별 분석
             * ## ⚠️ 이상치 및 주의사항
             * ## 💡 권장사항
             * ## 📎 부록 (Appendix)

           서식:
           - 표 사용 (후보자별 득표수)
           - 이모지 활용
           - 명확한 제목 계층
           - 코드 블록 (JSON 데이터)

        3. **이메일용 요약**: './gen_z_agent/output/Email_Summary_{invoice_id}.md'

           - 한 페이지 분량 (500자 이내)
           - 핵심 발견사항 3-5개
           - 액션 아이템
           - 첨부 파일 목록

        실행 지침:
        - 실제 파일 생성 (템플릿이 아닌 실제 데이터 사용)
        - 한국어 인코딩 (UTF-8)
        - 오류 처리 (파일 쓰기 실패 시 로그)
        - 생성된 파일 경로 반환

        중요: 이전 작업(Task 1, 2, 3)의 데이터를 활용하세요.
        """,
        expected_output="""
        생성된 파일 경로 목록:
        - Excel: ./gen_z_agent/output/Analysis_{invoice_id}.xlsx
        - Markdown: ./gen_z_agent/output/Report_{invoice_id}.md
        - Email: ./gen_z_agent/output/Email_Summary_{invoice_id}.md

        각 파일의 생성 상태 (성공/실패)
        """,
        agent=agent
    )


def create_notification_task(agent: Agent, recipients: List[str]) -> Task:
    """
    Create notification task.

    Args:
        agent: Communicator agent
        recipients: List of email recipients

    Returns:
        Configured Task
    """
    logger.info(f"Creating notification task for: {recipients}")

    return Task(
        description=f"""
        다음 채널로 분석 완료 알림을 전송하세요:

        1. **이메일 알림**:

           수신자: {', '.join(recipients)}

           제목 형식:
           - 정상: "✅ 선거 데이터 분석 완료 - {{분석_ID}}"
           - 이상치 탐지: "⚠️ 선거 데이터 분석 완료 (이상치 탐지) - {{분석_ID}}"
           - 검증 실패: "🔴 선거 데이터 분석 실패 - {{분석_ID}}"

           본문 구성:
           ```
           안녕하세요,

           Gen Z Agent 시스템이 [문서명]에 대한 자동 분석을 완료했습니다.

           📊 핵심 결과:
           - 총 투표수: [숫자]
           - 후보자 수: [숫자]
           - 이상치 탐지: [개수]
           - 검증 상태: [통과/경고/실패]

           🔍 주요 발견사항:
           1. [발견사항 1]
           2. [발견사항 2]
           3. [발견사항 3]

           📎 첨부 파일:
           - Analysis_[ID].xlsx - 상세 데이터 및 분석
           - Report_[ID].md - 종합 보고서

           자세한 내용은 첨부 파일을 참조해주세요.

           감사합니다.

           ---
           Gen Z Agent - Automated Analysis System
           Powered by Anthropic Claude
           ```

        2. **Slack 알림** (선택사항):

           채널: #election-analysis

           메시지 형식:
           ```
           📊 *선거 데이터 분석 완료*

           *분석 ID*: {{ID}}
           *문서*: {{문서명}}
           *상태*: ✅ 완료

           *핵심 메트릭*:
           • 총 투표수: {{숫자:,}}
           • 후보자: {{명단}}
           • 이상치: {{개수}}

           [상세 보고서 다운로드](링크)
           ```

        실행 모드:
        - DRY RUN: 실제 전송 없이 로그만 출력
        - PRODUCTION: 실제 이메일/Slack 전송

        현재는 DRY RUN 모드로 실행하고, 전송될 내용을 출력하세요.

        출력 형식:
        - 전송 대상 (이메일 주소, Slack 채널)
        - 메시지 내용
        - 첨부 파일 목록
        - 전송 상태 (DRY RUN / SENT / FAILED)
        """,
        expected_output="""
        이메일 및 Slack 전송 확인:

        ✅ Email (DRY RUN):
           - To: [수신자 목록]
           - Subject: [제목]
           - Body: [본문 미리보기]
           - Attachments: [파일 목록]

        ✅ Slack (DRY RUN):
           - Channel: #election-analysis
           - Message: [메시지 미리보기]

        실제 전송하려면 --production 플래그를 사용하세요.
        """,
        agent=agent
    )
