"""
═════════════════════════════════════════════════════════════
Multi-Agent Invoice Automation System (CrewAI + Anthropic Claude)
Gen Z Agent - Korean Election Invoice Analysis System
═════════════════════════════════════════════════════════════
"""

import os
from crewai import Agent, Task, Crew, Process
from langchain_anthropic import ChatAnthropic
from crewai_tools import FileReadTool, SerperDevTool
from dotenv import load_dotenv
import json
from pathlib import Path

# Load environment variables
load_dotenv()

# Initialize Claude LLM (using Claude Sonnet 4.5)
llm = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    temperature=0,
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# Initialize tools
file_reader = FileReadTool()
search_tool = SerperDevTool() if os.getenv("SERPER_API_KEY") else None

# ═════════════════════════════════════════════════════════════
# AGENT DEFINITIONS
# ═════════════════════════════════════════════════════════════

# 1. Data Extraction Agent (reads PDF/HTML invoices)
extractor = Agent(
    role="Invoice Data Extractor (청구서 데이터 추출 전문가)",
    goal="한국어 선거 개표상황표 및 청구서에서 모든 구조화된 데이터를 완벽하게 추출",
    backstory="""당신은 OCR 및 스캔 문서 읽기 전문가입니다.
    특히 한국어 선거 개표상황표의 복잡한 테이블 구조를 이해하고,
    후보자별 득표수, 투표소 정보, 검증 데이터를 정확히 추출할 수 있습니다.""",
    tools=[file_reader] if file_reader else [],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# 2. Data Validator & Enricher Agent
validator = Agent(
    role="Data Validator & Enricher (데이터 검증 및 보강 전문가)",
    goal="추출된 데이터를 검증하고 외부 컨텍스트로 보강 (후보자 정당, 투표율, 이상치 등)",
    backstory="""당신은 법의학 회계사이자 선거 데이터 분석 전문가입니다.
    잘못된 데이터를 절대 통과시키지 않으며, 득표수 합계 검증, 투표율 계산,
    통계적 이상치 탐지를 수행합니다.""",
    tools=[search_tool] if search_tool else [],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# 3. Financial/Electoral Analyst Agent
analyst = Agent(
    role="Electoral Data Analyst (선거 데이터 분석가)",
    goal="투표 패턴 분석, 후보별 지역별 득표 분석, 이상 패턴 탐지, 통계적 인사이트 도출",
    backstory="""당신은 선거 데이터 과학자입니다.
    후보자별 득표 추세, 지역별 투표 패턴, YoY 비교,
    통계적 이상치를 찾아내는 것을 즐깁니다.""",
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# 4. Report Generator Agent
reporter = Agent(
    role="Executive Report Writer (보고서 작성 전문가)",
    goal="Markdown, Excel, PDF 형식의 세련된 클라이언트급 보고서 작성",
    backstory="""당신은 임원급 보고서를 작성하는 전문가입니다.
    선거 개표 결과를 명확하고 시각적으로 표현하며,
    데이터 기반 인사이트를 비즈니스 언어로 번역합니다.""",
    tools=[file_reader] if file_reader else [],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# 5. Communication & Notification Agent
notifier = Agent(
    role="Communication Agent (커뮤니케이션 담당자)",
    goal="최종 보고서를 이메일 및 Slack으로 완벽한 톤으로 전송",
    backstory="""당신은 조직에서 가장 부드러운 커뮤니케이터입니다.
    기술 보고서를 비기술 이해관계자에게 전달하는 데 능숙합니다.""",
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# ═════════════════════════════════════════════════════════════
# TASK DEFINITIONS
# ═════════════════════════════════════════════════════════════

def create_extraction_task(file_path: str) -> Task:
    """Create data extraction task"""
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

        출력 형식: JSON
        """,
        expected_output="모든 필드가 포함된 깨끗한 JSON 객체",
        agent=extractor
    )

def create_validation_task() -> Task:
    """Create validation task"""
    return Task(
        description="""
        Task 1의 JSON 데이터를 가져와서:
        1. 합계 검증 (기계분류 vs 인간확인)
        2. 날짜 형식 검증
        3. 통계적 이상치 탐지 (2 std dev 초과)
        4. 누락된 경우 카테고리 추가

        추가할 필드:
        - category (예: 관외사전투표, 관내사전투표, 선거일투표)
        - is_anomaly (true/false)
        - validation_status (passed/failed/warning)
        - confidence_score (0-100)
        """,
        expected_output="검증되고 보강된 JSON",
        agent=validator
    )

def create_analysis_task() -> Task:
    """Create analysis task"""
    return Task(
        description="""
        보강된 JSON을 사용하여 다음 분석 수행:

        1. 후보자별 득표 분석:
           - 각 후보자의 총 득표수
           - 득표율 계산
           - 지역별 득표 패턴

        2. 투표 유형별 분석:
           - 관외사전투표 vs 관내사전투표 vs 선거일투표
           - 각 유형별 득표 차이

        3. 이상 패턴 탐지:
           - 평균에서 2 표준편차 이상 벗어난 값
           - 기계분류와 인간확인 불일치 사항

        4. 과거 데이터와 비교 (있는 경우):
           - ./historical/ 디렉토리의 과거 선거 데이터
           - YoY 또는 QoQ 변화율
        """,
        expected_output="테이블이 포함된 상세 Markdown 분석",
        agent=analyst
    )

def create_report_task(invoice_id: str) -> Task:
    """Create report generation task"""
    return Task(
        description=f"""
        다음 파일들을 생성하세요:

        1. Excel 파일: './gen_z_agent/output/Analysis_{invoice_id}.xlsx'
           - Sheet 1: 원본 데이터 (raw_data)
           - Sheet 2: 보강 데이터 (enriched_data)
           - Sheet 3: 분석 결과 (analysis)
           - Sheet 4: 요약 (summary)

        2. Markdown 보고서: './gen_z_agent/output/Report_{invoice_id}.md'
           - 임원 요약 (Executive Summary)
           - 주요 발견사항 (Key Findings)
           - 상세 분석 (Detailed Analysis)
           - 권장사항 (Recommendations)

        3. 이메일용 한 페이지 요약: './gen_z_agent/output/Email_Summary_{invoice_id}.md'
        """,
        expected_output="생성된 Excel, Markdown 파일 경로",
        agent=reporter
    )

def create_notification_task(recipients: list) -> Task:
    """Create notification task"""
    return Task(
        description=f"""
        다음 채널로 알림을 전송하세요:

        1. 이메일:
           - To: {', '.join(recipients)}
           - Subject: "자동 선거 데이터 분석 완료 - {{분석_ID}} - {{날짜}}"
           - Body: Email_Summary 내용
           - 첨부: Excel 및 Markdown 파일

        2. Slack (선택사항):
           - 채널: #election-analysis
           - 메시지: 분석 완료 요약 및 다운로드 링크

        참고: 실제 전송은 dry-run 모드에서 로그만 출력
        """,
        expected_output="이메일 및 Slack 전송 확인 메시지",
        agent=notifier
    )

# ═════════════════════════════════════════════════════════════
# MAIN ORCHESTRATION FUNCTION
# ═════════════════════════════════════════════════════════════

def run_invoice_analysis(
    invoice_path: str,
    invoice_id: str = None,
    recipients: list = None,
    dry_run: bool = True
):
    """
    Run the complete multi-agent invoice analysis pipeline

    Args:
        invoice_path: Path to invoice/election data file
        invoice_id: Unique identifier for this analysis
        recipients: List of email recipients
        dry_run: If True, don't actually send emails/notifications
    """
    # Set defaults
    if invoice_id is None:
        invoice_id = Path(invoice_path).stem
    if recipients is None:
        recipients = ["client@example.com"]

    # Create tasks
    task1 = create_extraction_task(invoice_path)
    task2 = create_validation_task()
    task3 = create_analysis_task()
    task4 = create_report_task(invoice_id)
    task5 = create_notification_task(recipients)

    # Create crew
    crew = Crew(
        agents=[extractor, validator, analyst, reporter, notifier],
        tasks=[task1, task2, task3, task4, task5],
        process=Process.sequential,
        verbose=2
    )

    # Run the analysis
    print(f"\n{'='*60}")
    print(f"Starting Gen Z Agent Analysis")
    print(f"File: {invoice_path}")
    print(f"ID: {invoice_id}")
    print(f"Dry Run: {dry_run}")
    print(f"{'='*60}\n")

    result = crew.kickoff()

    print(f"\n{'='*60}")
    print(f"Analysis Complete!")
    print(f"{'='*60}\n")
    print(result)

    return result

# ═════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Gen Z Agent - Multi-Agent Invoice/Election Analysis System"
    )
    parser.add_argument(
        "file",
        help="Path to invoice or election data file"
    )
    parser.add_argument(
        "--id",
        help="Analysis ID (default: filename)",
        default=None
    )
    parser.add_argument(
        "--recipients",
        help="Comma-separated email recipients",
        default="client@example.com"
    )
    parser.add_argument(
        "--production",
        help="Run in production mode (actually send emails)",
        action="store_true"
    )

    args = parser.parse_args()

    # Parse recipients
    recipients = [r.strip() for r in args.recipients.split(",")]

    # Run analysis
    run_invoice_analysis(
        invoice_path=args.file,
        invoice_id=args.id,
        recipients=recipients,
        dry_run=not args.production
    )
