"""
Tests for Markdown Report Generator
"""

import pytest
from pathlib import Path
from gen_z_agent.tools.markdown_generator import MarkdownReportGenerator


class TestMarkdownReportGenerator:
    """Test Markdown report generation."""

    def test_generate_header(self):
        """Test generating report header."""
        header = MarkdownReportGenerator.generate_header(
            title="Test Report",
            analysis_id="TEST001",
            timestamp="2024-01-01 12:00:00",
            document_name="test.pdf"
        )

        assert "Test Report" in header
        assert "TEST001" in header
        assert "test.pdf" in header

    def test_generate_executive_summary(self):
        """Test generating executive summary."""
        findings = ["Finding 1", "Finding 2", "Finding 3"]
        metrics = {"Total Votes": 10000, "Candidates": 3}

        summary = MarkdownReportGenerator.generate_executive_summary(findings, metrics)

        assert "임원 요약" in summary
        assert "Finding 1" in summary
        assert "Total Votes" in summary

    def test_generate_candidate_analysis(self):
        """Test generating candidate analysis section."""
        candidates = [
            {
                "name": "이재명",
                "votes": 15234,
                "vote_percentage": 44.1,
                "rank": 1,
                "party": "더불어민주당"
            }
        ]

        analysis = MarkdownReportGenerator.generate_candidate_analysis(candidates)

        assert "후보자별 분석" in analysis
        assert "이재명" in analysis
        assert "15,234" in analysis or "15234" in analysis

    def test_generate_anomaly_section_no_anomalies(self):
        """Test anomaly section with no anomalies."""
        section = MarkdownReportGenerator.generate_anomaly_section([])

        assert "이상치" in section
        assert "탐지되지 않았습니다" in section or "✅" in section

    def test_generate_anomaly_section_with_anomalies(self):
        """Test anomaly section with detected anomalies."""
        anomalies = [
            {
                "type": "Statistical Outlier",
                "severity": "high",
                "description": "Unusual vote count",
                "location": "Seoul",
                "value": 99999
            }
        ]

        section = MarkdownReportGenerator.generate_anomaly_section(anomalies)

        assert "이상치" in section
        assert "Statistical Outlier" in section
        assert "🔴" in section  # High severity emoji

    def test_generate_complete_report(self, temp_dir, sample_analysis_data):
        """Test generating complete Markdown report."""
        output_path = temp_dir / "test_report.md"

        result_path = MarkdownReportGenerator.generate_report(
            output_path=str(output_path),
            title="Test Report",
            analysis_id="TEST001",
            document_name="test.pdf",
            key_findings=["Finding 1", "Finding 2"],
            key_metrics={"Total": 1000},
            candidates=sample_analysis_data["candidates"]
        )

        assert Path(result_path).exists()
        content = Path(result_path).read_text(encoding='utf-8')
        assert "Test Report" in content
        assert "이재명" in content

    def test_report_has_proper_markdown_structure(self, temp_dir):
        """Test that generated report has proper Markdown structure."""
        output_path = temp_dir / "test_report.md"

        MarkdownReportGenerator.generate_report(
            output_path=str(output_path),
            title="Test",
            key_findings=["Test finding"]
        )

        content = Path(output_path).read_text(encoding='utf-8')

        # Check for Markdown headers
        assert content.count("# ") >= 1
        assert content.count("## ") >= 1
