"""
Tests for Excel Report Generator
"""

import pytest
from pathlib import Path
from gen_z_agent.tools.excel_generator import ExcelReportGenerator


class TestExcelReportGenerator:
    """Test Excel report generation."""

    def test_create_workbook(self):
        """Test creating Excel workbook."""
        try:
            wb = ExcelReportGenerator.create_workbook()
            assert wb is not None
        except ImportError:
            pytest.skip("openpyxl not available")

    def test_generate_report(self, temp_dir, sample_analysis_data):
        """Test generating complete Excel report."""
        try:
            output_path = temp_dir / "test_report.xlsx"

            result_path = ExcelReportGenerator.generate_report(
                output_path=str(output_path),
                summary=sample_analysis_data,
                raw_data=[{"test": "data"}],
                analysis=sample_analysis_data
            )

            assert Path(result_path).exists()
            assert Path(result_path).suffix == ".xlsx"
        except ImportError:
            pytest.skip("openpyxl not available")

    def test_generate_report_creates_directory(self, temp_dir, sample_analysis_data):
        """Test that report generation creates parent directories."""
        try:
            output_path = temp_dir / "subdir" / "test_report.xlsx"

            ExcelReportGenerator.generate_report(
                output_path=str(output_path),
                summary=sample_analysis_data
            )

            assert output_path.exists()
            assert output_path.parent.exists()
        except ImportError:
            pytest.skip("openpyxl not available")

    def test_style_header(self):
        """Test header styling."""
        try:
            wb = ExcelReportGenerator.create_workbook()
            ws = wb.active
            ws.append(["Header 1", "Header 2", "Header 3"])

            ExcelReportGenerator.style_header(ws, row=1)

            # Check that styling was applied
            assert ws['A1'].font.bold is True
        except ImportError:
            pytest.skip("openpyxl not available")
