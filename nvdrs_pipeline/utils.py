"""
Utility Functions for NVDRS Pipeline
Helper functions for data loading, visualization, and analysis
"""

import logging
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def create_sample_nvdrs_data():
    """
    Create sample NVDRS data for testing and demonstration

    Returns:
        List of dictionaries representing NVDRS records
    """
    sample_data = [
        {
            "record_id": "NVDRS_2024_001",
            "state": "CA",
            "year": 2024,
            "CME_Narrative": """
                Decedent was a 45-year-old male found unresponsive at home.
                Toxicology report revealed presence of fentanyl and oxycodone.
                Medical history included chronic pain and depression.
                Family reported recent job loss and financial difficulties.
            """,
            "LE_Narrative": """
                Officers responded to a welfare check at the residence.
                Found eviction notice on the door dated two weeks prior.
                Neighbor reported decedent had mentioned financial problems.
                No signs of foul play observed.
            """
        },
        {
            "record_id": "NVDRS_2024_002",
            "state": "NY",
            "year": 2024,
            "CME_Narrative": """
                34-year-old female decedent with history of anxiety and depression.
                Prescription bottles for sertraline and alprazolam found at scene.
                Previous suicide attempt documented in medical records from 2022.
                Toxicology pending.
            """,
            "LE_Narrative": """
                Suicide note found mentioning relationship problems and recent divorce.
                Family confirmed history of mental health treatment.
                Decedent had recently lost custody of children.
            """
        },
        {
            "record_id": "NVDRS_2024_003",
            "state": "TX",
            "year": 2024,
            "CME_Narrative": """
                56-year-old male with terminal cancer diagnosis.
                Hospice care records indicate severe chronic pain management.
                Multiple prescription opioids found at scene.
                Morphine and hydrocodone levels elevated on toxicology.
            """,
            "LE_Narrative": """
                Family present at scene, stated decedent had received terminal diagnosis 3 months prior.
                Medical records confirm stage 4 pancreatic cancer.
                No indication of foul play.
            """
        },
    ]

    return sample_data


def load_nvdrs_csv(file_path: str, spark_session):
    """
    Load NVDRS data from CSV file into Spark DataFrame

    Args:
        file_path: Path to CSV file
        spark_session: Active SparkSession

    Returns:
        Spark DataFrame
    """
    logger.info(f"Loading NVDRS data from {file_path}")

    df = (
        spark_session.read
        .option("header", True)
        .option("inferSchema", True)
        .option("multiLine", True)
        .option("escape", '"')
        .csv(file_path)
    )

    logger.info(f"Loaded {df.count()} records")
    return df


def export_to_excel(records: List, output_path: str):
    """
    Export processed records to Excel with formatting

    Args:
        records: List of NVDRSRecord objects
        output_path: Output Excel file path
    """
    try:
        import pandas as pd
        import openpyxl
        from openpyxl.styles import Font, PatternFill
    except ImportError:
        logger.error("pandas and openpyxl required for Excel export")
        raise

    logger.info(f"Exporting {len(records)} records to Excel: {output_path}")

    # Convert to flat dictionaries for DataFrame
    flat_records = []
    for record in records:
        flat_record = {
            "record_id": record.record_id,
            "state": record.state,
            "year": record.year,
            # Risk factors - substance use
            "opioid_mentioned": record.risk_factors.substance_use.opioid_mentioned,
            "alcohol_mentioned": record.risk_factors.substance_use.alcohol_mentioned,
            "overdose_indicated": record.risk_factors.substance_use.overdose_indicated,
            # Risk factors - mental health
            "depression": record.risk_factors.mental_health.depression_mentioned,
            "anxiety": record.risk_factors.mental_health.anxiety_mentioned,
            "suicide_history": record.risk_factors.mental_health.suicide_history,
            # Risk factors - social stressors
            "financial_crisis": record.risk_factors.social_stressors.financial_crisis,
            "relationship_problems": record.risk_factors.social_stressors.relationship_problems,
            "legal_problems": record.risk_factors.social_stressors.legal_problems,
            # Overall risk
            "risk_score": record.risk_factors.risk_score,
            # Narratives (truncated for readability)
            "cme_narrative_excerpt": record.cme_narrative[:100] + "..." if record.cme_narrative else "",
        }
        flat_records.append(flat_record)

    # Create DataFrame
    df = pd.DataFrame(flat_records)

    # Write to Excel with formatting
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='NVDRS_Analysis', index=False)

        # Get workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['NVDRS_Analysis']

        # Format header
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF")

        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font

        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column = [cell for cell in column]
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column[0].column_letter].width = adjusted_width

    logger.info(f"Excel export complete: {output_path}")


def generate_summary_report(records: List, output_path: str):
    """
    Generate a markdown summary report

    Args:
        records: List of NVDRSRecord objects
        output_path: Output markdown file path
    """
    logger.info(f"Generating summary report: {output_path}")

    total = len(records)

    # Calculate statistics
    opioid_count = sum(1 for r in records if r.risk_factors.substance_use.opioid_mentioned)
    depression_count = sum(1 for r in records if r.risk_factors.mental_health.depression_mentioned)
    financial_count = sum(1 for r in records if r.risk_factors.social_stressors.financial_crisis)
    high_risk_count = sum(1 for r in records if r.risk_factors.risk_score > 0.6)

    avg_risk_score = sum(r.risk_factors.risk_score for r in records) / total if total > 0 else 0

    report = f"""# NVDRS NLP Pipeline - Analysis Summary

**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Records Analyzed**: {total}

## Executive Summary

This report summarizes the analysis of {total} NVDRS (National Violent Death Reporting System)
records processed through the NLP pipeline to extract risk factors from coroner and law
enforcement narratives.

## Key Findings

### Overall Risk Distribution
- **High Risk Cases** (score > 0.6): {high_risk_count} ({high_risk_count/total*100:.1f}%)
- **Average Risk Score**: {avg_risk_score:.3f}

### Risk Factor Prevalence

#### Substance Use
- **Opioid Mentions**: {opioid_count} ({opioid_count/total*100:.1f}%)
- Includes: Fentanyl, Oxycodone, Heroin, etc.

#### Mental Health
- **Depression Mentioned**: {depression_count} ({depression_count/total*100:.1f}%)
- Indicates documented mental health concerns in narratives

#### Social Stressors
- **Financial Crisis**: {financial_count} ({financial_count/total*100:.1f}%)
- Includes: Eviction, Job Loss, Debt, Bankruptcy

## Methodology

### Data Processing Pipeline
1. **PII Redaction**: Personal identifiable information removed for HIPAA compliance
2. **Text Preprocessing**: Narratives cleaned and normalized
3. **Named Entity Recognition**: Clinical entities extracted using BioBERT/ClinicalBERT
4. **Risk Factor Extraction**: Pattern matching and NLP classification
5. **Scoring**: Aggregate risk score calculation

### Models Used
- **Clinical NER**: Spark NLP pre-trained clinical models
- **Text Processing**: Apache Spark distributed processing
- **Classification**: Hugging Face transformers (Bio_ClinicalBERT)

## Risk Factor Categories

The pipeline extracts the following risk factor categories:

1. **Substance Use**: Opioids, alcohol, other drugs, overdose indicators
2. **Mental Health**: Depression, anxiety, PTSD, suicide history
3. **Social Stressors**: Financial, relationship, legal, health crises

## Recommendations

1. **Prevention Focus**: High-risk cases should be prioritized for intervention strategy development
2. **Multi-Factor Approach**: Most cases show multiple risk factors, suggesting need for comprehensive prevention
3. **Data Quality**: Narrative completeness varies; standardized reporting could improve analysis

## Technical Notes

- Pipeline Version: 1.0.0
- PII Redaction: Enabled
- Processing Environment: Apache Spark + Spark NLP
- Output Format: Structured risk factors for statistical analysis

---

*This report was generated automatically by the NVDRS NLP Pipeline.*
*For questions or technical details, see the pipeline documentation.*
"""

    Path(output_path).write_text(report)
    logger.info(f"Summary report generated: {output_path}")


if __name__ == "__main__":
    # Example: Create sample data
    print("Creating sample NVDRS data...")
    sample_data = create_sample_nvdrs_data()

    print(f"\nGenerated {len(sample_data)} sample records:")
    for record in sample_data:
        print(f"  - {record['record_id']}: {record['state']} {record['year']}")

    print("\nSample CME Narrative:")
    print(sample_data[0]['CME_Narrative'])
