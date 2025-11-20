"""
Markdown Report Generator
Markdown 형식 보고서 생성기
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MarkdownReportGenerator:
    """
    Generate professional Markdown reports.
    """

    @staticmethod
    def generate_header(title: str, analysis_id: str, timestamp: str, document_name: str) -> str:
        """
        Generate report header.

        Args:
            title: Report title
            analysis_id: Analysis ID
            timestamp: Analysis timestamp
            document_name: Source document name

        Returns:
            Markdown header
        """
        return f"""# {title}

**분석 ID**: {analysis_id}
**분석 일시**: {timestamp}
**문서**: {document_name}

---

"""

    @staticmethod
    def generate_executive_summary(key_findings: List[str], metrics: Dict[str, Any]) -> str:
        """
        Generate executive summary section.

        Args:
            key_findings: List of key findings
            metrics: Key metrics dictionary

        Returns:
            Markdown executive summary
        """
        md = "## 📊 임원 요약 (Executive Summary)\n\n"

        # Key metrics
        md += "### 핵심 메트릭\n\n"
        for key, value in metrics.items():
            md += f"- **{key}**: {value}\n"
        md += "\n"

        # Key findings
        md += "### 주요 발견사항\n\n"
        for i, finding in enumerate(key_findings, 1):
            md += f"{i}. {finding}\n"
        md += "\n---\n\n"

        return md

    @staticmethod
    def generate_candidate_analysis(candidates: List[Dict[str, Any]]) -> str:
        """
        Generate candidate analysis section.

        Args:
            candidates: List of candidate data

        Returns:
            Markdown candidate analysis
        """
        md = "## 🗳️ 후보자별 분석\n\n"

        # Create table
        md += "| 후보자 | 득표수 | 득표율 (%) | 순위 |\n"
        md += "|--------|--------|-----------|------|\n"

        for candidate in candidates:
            name = candidate.get("name", "N/A")
            votes = candidate.get("votes", 0)
            percentage = candidate.get("vote_percentage", 0)
            rank = candidate.get("rank", "-")
            md += f"| {name} | {votes:,} | {percentage:.2f} | {rank} |\n"

        md += "\n"

        # Detailed analysis for each candidate
        for candidate in candidates:
            name = candidate.get("name", "N/A")
            party = candidate.get("party", "N/A")
            md += f"### {name} ({party})\n\n"

            if "regional_analysis" in candidate:
                md += "**지역별 득표 분석**:\n\n"
                for region, data in candidate["regional_analysis"].items():
                    md += f"- {region}: {data.get('votes', 0):,}표 ({data.get('percentage', 0):.1f}%)\n"
                md += "\n"

        md += "---\n\n"
        return md

    @staticmethod
    def generate_vote_type_analysis(vote_types: Dict[str, Any]) -> str:
        """
        Generate vote type analysis section.

        Args:
            vote_types: Vote type data

        Returns:
            Markdown vote type analysis
        """
        md = "## 📮 투표 유형별 분석\n\n"

        md += "| 투표 유형 | 총 투표수 | 비율 (%) |\n"
        md += "|----------|----------|----------|\n"

        for vote_type, data in vote_types.items():
            count = data.get("count", 0)
            percentage = data.get("percentage", 0)
            md += f"| {vote_type} | {count:,} | {percentage:.2f} |\n"

        md += "\n---\n\n"
        return md

    @staticmethod
    def generate_anomaly_section(anomalies: List[Dict[str, Any]]) -> str:
        """
        Generate anomaly detection section.

        Args:
            anomalies: List of detected anomalies

        Returns:
            Markdown anomaly section
        """
        md = "## ⚠️ 이상치 및 주의사항\n\n"

        if not anomalies:
            md += "✅ 이상치가 탐지되지 않았습니다.\n\n"
        else:
            md += f"🔴 **{len(anomalies)}개의 이상치가 탐지되었습니다.**\n\n"

            for i, anomaly in enumerate(anomalies, 1):
                severity = anomaly.get("severity", "medium")
                emoji = "🔴" if severity == "high" else "🟡" if severity == "medium" else "🟢"

                md += f"{i}. {emoji} **{anomaly.get('type', 'Unknown')}**\n"
                md += f"   - 설명: {anomaly.get('description', 'N/A')}\n"
                md += f"   - 위치: {anomaly.get('location', 'N/A')}\n"
                md += f"   - 값: {anomaly.get('value', 'N/A')}\n"
                if "recommendation" in anomaly:
                    md += f"   - 권장사항: {anomaly['recommendation']}\n"
                md += "\n"

        md += "---\n\n"
        return md

    @staticmethod
    def generate_recommendations(recommendations: List[str]) -> str:
        """
        Generate recommendations section.

        Args:
            recommendations: List of recommendations

        Returns:
            Markdown recommendations
        """
        md = "## 💡 권장사항\n\n"

        for i, rec in enumerate(recommendations, 1):
            md += f"{i}. {rec}\n"

        md += "\n---\n\n"
        return md

    @staticmethod
    def generate_appendix(raw_data: Optional[Any] = None) -> str:
        """
        Generate appendix with raw data.

        Args:
            raw_data: Raw data to include

        Returns:
            Markdown appendix
        """
        md = "## 📎 부록 (Appendix)\n\n"
        md += "### 원본 데이터\n\n"

        if raw_data:
            md += "```json\n"
            import json
            md += json.dumps(raw_data, ensure_ascii=False, indent=2)[:5000]  # Truncate if too long
            md += "\n```\n\n"
        else:
            md += "원본 데이터가 제공되지 않았습니다.\n\n"

        return md

    @staticmethod
    def generate_report(
        output_path: str,
        title: str = "선거 데이터 분석 보고서",
        analysis_id: str = "",
        document_name: str = "",
        key_findings: Optional[List[str]] = None,
        key_metrics: Optional[Dict[str, Any]] = None,
        candidates: Optional[List[Dict[str, Any]]] = None,
        vote_types: Optional[Dict[str, Any]] = None,
        anomalies: Optional[List[Dict[str, Any]]] = None,
        recommendations: Optional[List[str]] = None,
        raw_data: Optional[Any] = None
    ) -> str:
        """
        Generate complete Markdown report.

        Args:
            output_path: Path to save Markdown file
            title: Report title
            analysis_id: Analysis ID
            document_name: Source document name
            key_findings: List of key findings
            key_metrics: Key metrics dictionary
            candidates: Candidate analysis data
            vote_types: Vote type analysis data
            anomalies: List of anomalies
            recommendations: List of recommendations
            raw_data: Raw data for appendix

        Returns:
            Path to generated file
        """
        logger.info(f"Generating Markdown report: {output_path}")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build report
        report = MarkdownReportGenerator.generate_header(
            title, analysis_id or "N/A", timestamp, document_name or "N/A"
        )

        if key_findings or key_metrics:
            report += MarkdownReportGenerator.generate_executive_summary(
                key_findings or [], key_metrics or {}
            )

        if candidates:
            report += MarkdownReportGenerator.generate_candidate_analysis(candidates)

        if vote_types:
            report += MarkdownReportGenerator.generate_vote_type_analysis(vote_types)

        if anomalies is not None:
            report += MarkdownReportGenerator.generate_anomaly_section(anomalies)

        if recommendations:
            report += MarkdownReportGenerator.generate_recommendations(recommendations)

        if raw_data:
            report += MarkdownReportGenerator.generate_appendix(raw_data)

        # Footer
        report += f"\n---\n\n*Generated by Gen Z Agent - {timestamp}*\n"
        report += "*Powered by Anthropic Claude*\n"

        # Save report
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"Markdown report generated successfully: {output_path}")
        return output_path
