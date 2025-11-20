"""
NVDRS NLP Pipeline - Complete Example
Demonstrates full workflow from data loading to analysis
"""

import sys
from pathlib import Path

# Add parent directory to path to import nvdrs_pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyspark.sql import SparkSession
from nvdrs_pipeline import NVDRSPipeline
from nvdrs_pipeline.config import NVDRSConfig
from nvdrs_pipeline.utils import create_sample_nvdrs_data, export_to_excel, generate_summary_report


def main():
    """
    Complete example of NVDRS NLP Pipeline usage

    This demonstrates the "Narrative-to-Data" transformation:
    1. Load unstructured coroner narratives
    2. Apply PII redaction (HIPAA compliance)
    3. Extract entities using Clinical NER (Spark NLP + Hugging Face)
    4. Identify risk factors (substance use, mental health, social stressors)
    5. Generate structured output for statistical analysis
    """

    print("=" * 80)
    print("NVDRS NLP PIPELINE - DEMONSTRATION")
    print("Transforming Narrative to Data for Suicide Prevention Research")
    print("=" * 80)

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 0: Configuration
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n📋 STAGE 0: Configuration")
    print("-" * 80)

    config = NVDRSConfig()
    print(f"✓ Spark Master: {config.SPARK_MASTER}")
    print(f"✓ NER Model: {config.NER_MODEL}")
    print(f"✓ PII Redaction: {'Enabled' if config.REDACT_PII else 'Disabled'}")
    print(f"✓ Output Directory: {config.OUTPUT_DIR}")

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 1: Data Ingestion & PII Redaction
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n📥 STAGE 1: Ingestion & PII Redaction")
    print("-" * 80)

    # Create Spark session
    print("Creating Spark session...")
    spark = SparkSession.builder \
        .appName("NVDRS_Example") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    # Load sample data
    print("Loading sample NVDRS data...")
    sample_data = create_sample_nvdrs_data()
    df = spark.createDataFrame(sample_data)

    print(f"✓ Loaded {df.count()} NVDRS records")
    print("\nSample Record (Before Processing):")
    print("-" * 40)
    first_record = df.first()
    print(f"Record ID: {first_record['record_id']}")
    print(f"State: {first_record['state']}")
    print(f"CME Narrative (excerpt): {first_record['CME_Narrative'][:100]}...")

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 2: NLP Pipeline Construction
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n🧠 STAGE 2: NLP Pipeline Construction")
    print("-" * 80)
    print("Building Spark NLP Pipeline with the following stages:")
    print("  1. Document Assembler - Prepare text for processing")
    print("  2. Sentence Detector - Split narratives into sentences")
    print("  3. Tokenizer - Break sentences into tokens")
    print("  4. Normalizer - Clean and standardize text")
    print("  5. Clinical NER (BioBERT) - Extract medical entities")
    print("  6. NER Converter - Format entities for analysis")

    pipeline = NVDRSPipeline(spark=spark, config=config, enable_pii_redaction=True)
    print("✓ Pipeline initialized")

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 3: Entity Extraction (The "Magic")
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n✨ STAGE 3: Entity Extraction & Risk Factor Identification")
    print("-" * 80)
    print("Processing narratives through Clinical NER model...")
    print("This is where unstructured text becomes structured data.")

    # Process the narratives
    results = pipeline.process_narratives(
        df,
        narrative_columns=['CME_Narrative', 'LE_Narrative']
    )

    print(f"\n✓ Processing complete!")
    print(f"  Total Records: {results.total_records}")
    print(f"  Successful: {results.successful}")
    print(f"  Processing Time: {results.processing_time_seconds:.2f} seconds")

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 4: Risk Factor Analysis
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n📊 STAGE 4: Risk Factor Analysis")
    print("-" * 80)
    print("Summary Statistics:")
    print(f"  Opioid Mentions: {results.summary_statistics['substance_use']['opioid_mentions']} "
          f"({results.summary_statistics['substance_use']['opioid_mentions_percent']:.1f}%)")
    print(f"  Depression Mentions: {results.summary_statistics['mental_health']['depression']} "
          f"({results.summary_statistics['mental_health']['depression_percent']:.1f}%)")
    print(f"  Financial Crisis: {results.summary_statistics['social_stressors']['financial_crisis']} "
          f"({results.summary_statistics['social_stressors']['financial_crisis_percent']:.1f}%)")
    print(f"  High Risk Cases: {results.summary_statistics['risk_scores']['high_risk_count']}")
    print(f"  Average Risk Score: {results.summary_statistics['risk_scores']['mean']:.3f}")

    # Display individual record results
    print("\n📝 Individual Record Analysis:")
    print("-" * 80)
    for i, record in enumerate(results.records[:3], 1):  # Show first 3
        print(f"\n{i}. Record {record.record_id}")
        print(f"   Risk Score: {record.risk_factors.risk_score:.2f}")

        # Substance use
        if record.risk_factors.substance_use.opioid_mentioned:
            print(f"   ⚠️  Opioid Use: {', '.join(record.risk_factors.substance_use.opioid_types)}")

        # Mental health
        if record.risk_factors.mental_health.depression_mentioned:
            print(f"   ⚠️  Mental Health: Depression mentioned")
        if record.risk_factors.mental_health.suicide_history:
            print(f"   ⚠️  Previous suicide attempt documented")

        # Social stressors
        if record.risk_factors.social_stressors.financial_crisis:
            print(f"   ⚠️  Financial Crisis indicators present")
        if record.risk_factors.social_stressors.relationship_problems:
            print(f"   ⚠️  Relationship problems indicated")

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 5: Output Generation
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n💾 STAGE 5: Output Generation")
    print("-" * 80)

    # Ensure output directory exists
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save as Parquet (for big data analytics)
    parquet_path = config.OUTPUT_DIR / "nvdrs_processed.parquet"
    print(f"Saving to Parquet: {parquet_path}")
    pipeline.save_results(results, str(parquet_path), format="parquet")
    print("✓ Parquet saved")

    # Export to Excel (for human review)
    excel_path = config.OUTPUT_DIR / "nvdrs_analysis.xlsx"
    print(f"Exporting to Excel: {excel_path}")
    try:
        export_to_excel(results.records, str(excel_path))
        print("✓ Excel exported")
    except ImportError:
        print("⚠️  Excel export requires pandas and openpyxl")

    # Generate summary report
    report_path = config.OUTPUT_DIR / "nvdrs_summary_report.md"
    print(f"Generating summary report: {report_path}")
    try:
        generate_summary_report(results.records, str(report_path))
        print("✓ Summary report generated")
    except Exception as e:
        print(f"⚠️  Report generation failed: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 6: The "So What?" - Use Cases
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n🎯 STAGE 6: Practical Applications")
    print("-" * 80)
    print("This structured data enables:")
    print("  1. Statistical Analysis - Identify risk factor prevalence and correlations")
    print("  2. Predictive Modeling - Build ML models for suicide risk prediction")
    print("  3. Public Health Policy - Evidence-based prevention strategies")
    print("  4. Geographic Analysis - Regional patterns and targeted interventions")
    print("  5. Trend Detection - Monitor changes in risk factors over time")

    print("\n💡 Example Insights from this Data:")
    opioid_pct = results.summary_statistics['substance_use']['opioid_mentions_percent']
    if opioid_pct > 50:
        print(f"  • HIGH ALERT: Opioids mentioned in {opioid_pct:.0f}% of cases")
        print(f"    → Suggests opioid epidemic correlation with suicide deaths")
        print(f"    → Recommendation: Integrate substance abuse treatment with mental health services")

    financial_pct = results.summary_statistics['social_stressors']['financial_crisis_percent']
    if financial_pct > 30:
        print(f"  • Economic stress present in {financial_pct:.0f}% of cases")
        print(f"    → Economic downturns may increase suicide risk")
        print(f"    → Recommendation: Financial counseling as part of crisis intervention")

    # ═══════════════════════════════════════════════════════════════════════════
    # Cleanup
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n🧹 Cleanup")
    print("-" * 80)
    pipeline.stop()
    print("✓ Spark session stopped")

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print(f"\nOutput files created in: {config.OUTPUT_DIR}")
    print("  • nvdrs_processed.parquet - Structured data for analytics")
    print("  • nvdrs_analysis.xlsx - Human-readable Excel format")
    print("  • nvdrs_summary_report.md - Executive summary report")
    print("\nThese files can now be used for:")
    print("  - Statistical analysis in R/Python")
    print("  - Machine learning model training")
    print("  - Public health research publications")
    print("  - Policy recommendations")
    print("\n✅ Pipeline validated successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
