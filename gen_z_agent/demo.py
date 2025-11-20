#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════
Multi-Agent Invoice Automation System
Adapted for GenZ Project - Korean Electoral Data Analysis
═══════════════════════════════════════════════════════════════

This system uses 5 specialized AI agents to automate the analysis
of Korean election count sheets (개표상황표) and invoices.

Author: GenZ Project Team
License: MIT
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

try:
    from crewai import Agent, Task, Crew, Process
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("\n📦 Install required packages:")
    print("pip install crewai langchain-anthropic langchain-openai python-dotenv")
    sys.exit(1)

# Load environment variables
load_dotenv()


class InvoiceAutomationSystem:
    """Multi-agent system for automated invoice and electoral data analysis."""

    def __init__(self, model_provider: str = "anthropic"):
        """
        Initialize the automation system.

        Args:
            model_provider: "anthropic" or "openai"
        """
        self.model_provider = model_provider
        self.llm = self._initialize_llm()
        self.agents = self._create_agents()

    def _initialize_llm(self):
        """Initialize the LLM based on provider."""
        if self.model_provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
            return ChatAnthropic(
                model="claude-sonnet-4-20250514",
                anthropic_api_key=api_key,
                temperature=0
            )
        elif self.model_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            return ChatOpenAI(
                model="gpt-4o",
                openai_api_key=api_key,
                temperature=0
            )
        else:
            raise ValueError(f"Unknown provider: {self.model_provider}")

    def _create_agents(self) -> Dict[str, Agent]:
        """Create the 5 specialized agents."""

        # Agent 1: Data Extractor (데이터 추출 전문가)
        extractor = Agent(
            role="Invoice Data Extractor",
            goal="Extract structured data from Korean invoice/election documents",
            backstory="""You are an expert in reading Korean documents including
            개표상황표 (election count sheets) and invoices. You understand Korean
            text patterns and can accurately extract candidate names (후보자),
            vote counts (득표수), and regional data (지역).""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

        # Agent 2: Data Validator (데이터 검증 전문가)
        validator = Agent(
            role="Data Validator & Enricher",
            goal="Validate data integrity and enrich with external context",
            backstory="""You are a meticulous data analyst who ensures accuracy.
            You validate vote counts, check sum calculations, and verify Korean
            names and regions. You never let invalid data pass through.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

        # Agent 3: Data Analyst (데이터 분석가)
        analyst = Agent(
            role="Electoral Data Analyst",
            goal="Perform statistical analysis and detect anomalies",
            backstory="""You are an expert in electoral data analysis. You can
            calculate statistics, identify outliers using standard deviation,
            detect unusual voting patterns, and compare results across regions.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

        # Agent 4: Report Generator (보고서 작성 전문가)
        reporter = Agent(
            role="Executive Report Writer",
            goal="Create professional Korean/English bilingual reports",
            backstory="""You write executive-level reports in both Korean and
            English. You create clear visualizations, tables, and summaries
            that decision-makers love.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

        # Agent 5: Communication Agent (커뮤니케이션 담당자)
        notifier = Agent(
            role="Communication Agent",
            goal="Communicate findings clearly to stakeholders",
            backstory="""You excel at distilling complex analysis into clear,
            actionable messages for email and Slack. You understand both
            Korean and English business communication.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

        return {
            "extractor": extractor,
            "validator": validator,
            "analyst": analyst,
            "reporter": reporter,
            "notifier": notifier
        }

    def analyze_invoice(
        self,
        invoice_data: str,
        document_type: str = "invoice",
        output_dir: Optional[Path] = None
    ) -> str:
        """
        Analyze invoice or electoral data using the multi-agent system.

        Args:
            invoice_data: Raw text or path to document
            document_type: "invoice" or "election"
            output_dir: Directory to save reports (default: ./output)

        Returns:
            Final analysis result as string
        """
        if output_dir is None:
            output_dir = Path("./output")
        output_dir.mkdir(exist_ok=True)

        # Define tasks based on document type
        if document_type == "election":
            tasks = self._create_election_tasks(invoice_data, output_dir)
        else:
            tasks = self._create_invoice_tasks(invoice_data, output_dir)

        # Create and execute crew
        crew = Crew(
            agents=list(self.agents.values()),
            tasks=tasks,
            process=Process.sequential,
            verbose=2
        )

        print("=" * 70)
        print("🚀 Starting Multi-Agent Analysis System")
        print(f"📄 Document Type: {document_type}")
        print(f"🤖 Model: {self.model_provider}")
        print(f"📊 Agents: {len(self.agents)}")
        print("=" * 70)

        result = crew.kickoff()

        print("\n" + "=" * 70)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 70)

        return result

    def _create_invoice_tasks(
        self,
        invoice_data: str,
        output_dir: Path
    ) -> list:
        """Create tasks for invoice analysis."""

        task1 = Task(
            description=f"""
            Analyze this invoice data and extract structured information:
            {invoice_data}

            Extract in JSON format:
            - invoice_number
            - date
            - vendor_name
            - vendor_gstin (if available)
            - line_items (description, quantity, rate, amount, tax)
            - subtotal
            - total_tax
            - total_amount

            Ensure all calculations are verified.
            """,
            expected_output="Clean JSON object with all invoice fields",
            agent=self.agents["extractor"]
        )

        task2 = Task(
            description="""
            Take the JSON from task 1 and validate:
            - All totals add up correctly
            - Date format is valid
            - GSTIN format (if present)
            - Categorize vendor (e.g., Technology, Office Supplies)
            - Flag any anomalies (amounts > 2σ from historical mean)

            Add fields: category, is_anomaly, validation_status
            """,
            expected_output="Validated and enriched JSON",
            agent=self.agents["validator"]
        )

        task3 = Task(
            description="""
            Perform financial analysis:
            - Category-wise spend breakdown
            - Identify overpriced items (compare to market rates)
            - Calculate savings opportunities
            - Flag unusual patterns
            - Provide 3 actionable recommendations
            """,
            expected_output="Detailed analysis with findings and recommendations",
            agent=self.agents["analyst"]
        )

        task4 = Task(
            description=f"""
            Create executive summary report:
            1. Invoice overview (vendor, date, total)
            2. Validation results
            3. Analysis insights (category breakdown, anomalies)
            4. Top 3 recommendations
            5. Summary statistics

            Save report to: {output_dir}/analysis_report.md
            Format in clear markdown with tables.
            """,
            expected_output="Professional executive summary in markdown",
            agent=self.agents["reporter"]
        )

        task5 = Task(
            description="""
            Create email-style notification:
            - Subject line
            - Executive summary (2-3 sentences)
            - Key highlights (bullet points)
            - Next steps

            Keep it concise and actionable.
            """,
            expected_output="Email-ready summary",
            agent=self.agents["notifier"]
        )

        return [task1, task2, task3, task4, task5]

    def _create_election_tasks(
        self,
        election_data: str,
        output_dir: Path
    ) -> list:
        """Create tasks for election data analysis."""

        task1 = Task(
            description=f"""
            Extract election data from this 개표상황표 (election count sheet):
            {election_data}

            Extract in JSON format:
            - election_date (선거일)
            - region (지역)
            - candidates (후보자) with vote counts (득표수)
            - total_votes (총 득표수)
            - turnout (투표율) if available
            - invalid_votes (무효표) if available

            Handle Korean text properly (UTF-8).
            """,
            expected_output="Clean JSON with all election data fields",
            agent=self.agents["extractor"]
        )

        task2 = Task(
            description="""
            Validate the election data:
            - Sum of candidate votes matches total_votes
            - All Korean names are valid (2-4 Hangul characters)
            - Dates are in valid format
            - Vote counts are reasonable (no negative numbers)
            - Flag any statistical anomalies

            Add validation_status and anomaly_flags.
            """,
            expected_output="Validated election data with flags",
            agent=self.agents["validator"]
        )

        task3 = Task(
            description="""
            Perform electoral analysis:
            - Calculate vote share percentages for each candidate
            - Identify winner and margin of victory
            - Compute statistical measures (mean, median, std dev)
            - Detect anomalies (votes > 2σ from regional mean)
            - Compare turnout to historical averages
            - Provide insights on voting patterns
            """,
            expected_output="Statistical analysis with insights",
            agent=self.agents["analyst"]
        )

        task4 = Task(
            description=f"""
            Create bilingual (Korean/English) election analysis report:

            Include:
            1. 선거 개요 / Election Overview
            2. 후보자별 득표 현황 / Vote Distribution by Candidate
            3. 통계 분석 / Statistical Analysis
            4. 이상 징후 / Anomalies (if any)
            5. 주요 발견사항 / Key Findings

            Format with tables showing:
            - Candidate | Votes | Percentage | Winner

            Save to: {output_dir}/election_analysis.md
            """,
            expected_output="Bilingual election analysis report",
            agent=self.agents["reporter"]
        )

        task5 = Task(
            description="""
            Create notification summary in Korean and English:

            Subject: 선거 분석 완료 / Election Analysis Complete

            Include:
            - 주요 결과 요약 / Executive Summary
            - 당선자 정보 / Winner Information
            - 통계 하이라이트 / Statistical Highlights
            - 이상 징후 (있을 경우) / Anomalies (if any)
            """,
            expected_output="Bilingual notification summary",
            agent=self.agents["notifier"]
        )

        return [task1, task2, task3, task4, task5]


def main():
    """Main execution function with examples."""

    # Example 1: Invoice Analysis
    sample_invoice = """
    INVOICE
    Invoice Number: INV-2025-001
    Date: January 15, 2025
    Vendor: TechSupplies Korea
    GSTIN: 29ABCDE1234F1Z5

    Line Items:
    1. Dell Laptop (XPS 15) - Qty: 5 - Rate: ₩2,500,000
       Amount: ₩12,500,000 - Tax (10%): ₩1,250,000
    2. Microsoft Office 365 - Qty: 10 - Rate: ₩300,000
       Amount: ₩3,000,000 - Tax (10%): ₩300,000
    3. Network Equipment - Qty: 2 - Rate: ₩1,500,000
       Amount: ₩3,000,000 - Tax (10%): ₩300,000

    Subtotal: ₩18,500,000
    Total Tax: ₩1,850,000
    Total Amount: ₩20,350,000
    """

    # Example 2: Election Data
    sample_election = """
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

    print("\n" + "=" * 70)
    print("📋 GenZ Multi-Agent System - Demo")
    print("=" * 70)
    print("\nAvailable examples:")
    print("1. Invoice Analysis (영수증 분석)")
    print("2. Election Data Analysis (선거 데이터 분석)")
    print("3. Exit")

    choice = input("\nSelect example (1-3): ").strip()

    try:
        if choice == "1":
            system = InvoiceAutomationSystem(model_provider="anthropic")
            result = system.analyze_invoice(
                invoice_data=sample_invoice,
                document_type="invoice"
            )
            print("\n" + "=" * 70)
            print("📄 FINAL RESULT")
            print("=" * 70)
            print(result)

        elif choice == "2":
            system = InvoiceAutomationSystem(model_provider="anthropic")
            result = system.analyze_invoice(
                invoice_data=sample_election,
                document_type="election"
            )
            print("\n" + "=" * 70)
            print("📊 FINAL RESULT")
            print("=" * 70)
            print(result)

        elif choice == "3":
            print("👋 Goodbye!")
            return
        else:
            print("❌ Invalid choice")

    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\n💡 Make sure to set your API key:")
        print("   export ANTHROPIC_API_KEY='your_key_here'")
        print("   # or")
        print("   export OPENAI_API_KEY='your_key_here'")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
