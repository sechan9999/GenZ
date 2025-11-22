#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════
ETL Quality Validation - Complete Example
Demonstrates A/B testing of pre/post pipeline data quality
═══════════════════════════════════════════════════════════════════

This example shows:
1. Loading pre/post ingestion data
2. Creating 30+ quality expectations
3. Running validation
4. Comparing results
5. Generating Excel report

Usage:
    python examples/etl_quality_validation_example.py

Author: Gen Z Agent Team
Created: 2025-11-22
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gen_z_agent.etl_quality_validator import ETLQualityValidator


def create_sample_data():
    """
    Create sample pre/post ingestion datasets for demonstration

    Simulates a healthcare ETL pipeline processing patient observations.
    """
    print("📊 Creating sample pre/post ingestion data...")

    np.random.seed(42)
    n_records = 5000

    # ═══════════════════════════════════════════════════════════════
    # PRE-INGESTION DATA (Bronze Layer - Raw FHIR data)
    # ═══════════════════════════════════════════════════════════════

    pre_df = pd.DataFrame({
        # Identifiers
        'observation_id': [f"OBS-{i:06d}" for i in range(1, n_records + 1)],
        'patient_id': [f"PAT-{np.random.randint(1, 1000):05d}" for _ in range(n_records)],
        'encounter_id': [f"ENC-{np.random.randint(1, 2000):06d}" for _ in range(n_records)],

        # Observation details
        'status': np.random.choice(
            ['final', 'preliminary', 'amended', 'corrected'],
            n_records,
            p=[0.7, 0.2, 0.08, 0.02]
        ),

        'category_code': np.random.choice(
            ['vital-signs', 'laboratory', 'imaging', 'procedure'],
            n_records,
            p=[0.4, 0.35, 0.15, 0.1]
        ),

        'code': np.random.choice(
            ['8867-4', '8310-5', '8480-6', '8462-4', '2339-0'],  # LOINC codes
            n_records
        ),

        # Values
        'value_quantity': np.random.normal(120, 20, n_records).round(2),
        'value_unit': np.random.choice(['mmHg', 'mg/dL', 'celsius'], n_records),

        # Reference ranges
        'reference_range_low': np.random.uniform(90, 100, n_records).round(2),
        'reference_range_high': np.random.uniform(140, 150, n_records).round(2),

        # Temporal data
        'effective_datetime': pd.date_range(
            start='2024-01-01',
            periods=n_records,
            freq='5min'
        ),
        'issued': pd.date_range(
            start='2024-01-01 00:05:00',
            periods=n_records,
            freq='5min'
        ),

        # Metadata
        'performer': np.random.choice(
            ['Dr. Smith', 'Dr. Jones', 'Dr. Williams', 'Dr. Brown'],
            n_records
        ),

        'device_id': [
            f"DEV-{np.random.randint(1, 50):03d}" if np.random.random() > 0.1 else None
            for _ in range(n_records)
        ],

        # Quality indicators
        'data_quality_flag': np.random.choice(
            ['good', 'fair', 'poor'],
            n_records,
            p=[0.85, 0.12, 0.03]
        )
    })

    # Add some intentional nulls (2-3%)
    null_mask = np.random.random(n_records) < 0.02
    pre_df.loc[null_mask, 'value_quantity'] = np.nan

    null_mask = np.random.random(n_records) < 0.03
    pre_df.loc[null_mask, 'performer'] = None

    # ═══════════════════════════════════════════════════════════════
    # POST-INGESTION DATA (Silver Layer - Transformed data)
    # ═══════════════════════════════════════════════════════════════

    post_df = pre_df.copy()

    # Simulate ETL transformations with intentional issues for demonstration

    # Issue 1: Some records filtered out (deduplication)
    post_df = post_df.sample(frac=0.98, random_state=42).reset_index(drop=True)

    # Issue 2: Increased nulls in value_quantity (data quality degradation)
    additional_null_mask = np.random.random(len(post_df)) < 0.05
    post_df.loc[additional_null_mask, 'value_quantity'] = np.nan

    # Issue 3: Mean shift in value_quantity (unit conversion issue?)
    post_df['value_quantity'] = post_df['value_quantity'] * 1.08  # 8% increase

    # Issue 4: New categorical value (data integration)
    post_df.loc[np.random.random(len(post_df)) < 0.01, 'status'] = 'registered'

    # Issue 5: Outliers introduced
    outlier_mask = np.random.random(len(post_df)) < 0.005
    post_df.loc[outlier_mask, 'value_quantity'] = np.random.uniform(500, 1000, outlier_mask.sum())

    # Transformation: Normalize column names (expected change)
    post_df.columns = [col.replace('_', '').lower() for col in post_df.columns]

    # Revert for comparison (in real scenario, you'd map columns)
    post_df.columns = pre_df.columns[:len(post_df.columns)]

    print(f"✅ Created pre-ingestion data: {len(pre_df):,} records")
    print(f"✅ Created post-ingestion data: {len(post_df):,} records")
    print(f"   Row change: {((len(post_df) - len(pre_df)) / len(pre_df) * 100):.2f}%\n")

    return pre_df, post_df


def run_full_validation_example():
    """
    Run complete ETL quality validation workflow
    """
    print("\n" + "="*70)
    print("ETL QUALITY VALIDATION - COMPLETE EXAMPLE")
    print("="*70 + "\n")

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: Create sample data
    # ═══════════════════════════════════════════════════════════════

    pre_df, post_df = create_sample_data()

    # ═══════════════════════════════════════════════════════════════
    # STEP 2: Initialize validator
    # ═══════════════════════════════════════════════════════════════

    print("🔧 Initializing ETL Quality Validator...")

    validator = ETLQualityValidator(
        project_name="fhir_etl_quality_validation",
        output_dir="./gen_z_agent/output/quality_reports",
        ge_context_dir="./gen_z_agent/ge_context"
    )

    print("✅ Validator initialized\n")

    # ═══════════════════════════════════════════════════════════════
    # STEP 3: Define column types
    # ═══════════════════════════════════════════════════════════════

    print("📋 Categorizing columns...")

    numeric_columns = [
        'value_quantity',
        'reference_range_low',
        'reference_range_high'
    ]

    categorical_columns = [
        'status',
        'category_code',
        'code',
        'value_unit',
        'performer',
        'data_quality_flag'
    ]

    date_columns = [
        'effective_datetime',
        'issued'
    ]

    key_columns = [
        'observation_id',
        'patient_id'
    ]

    print(f"   Numeric columns: {len(numeric_columns)}")
    print(f"   Categorical columns: {len(categorical_columns)}")
    print(f"   Date columns: {len(date_columns)}")
    print(f"   Key columns: {len(key_columns)}\n")

    # ═══════════════════════════════════════════════════════════════
    # STEP 4: Create expectation suite
    # ═══════════════════════════════════════════════════════════════

    if validator.context:
        print("📝 Creating expectation suite...")

        suite = validator.create_expectation_suite(
            suite_name="fhir_comprehensive_quality_suite"
        )

        # Get validator object
        try:
            datasource = validator.context.sources.add_pandas("fhir_datasource")
            data_asset = datasource.add_dataframe_asset(name="fhir_observations")
            batch_request = data_asset.build_batch_request(dataframe=pre_df)

            ge_validator = validator.context.get_validator(
                batch_request=batch_request,
                expectation_suite_name="fhir_comprehensive_quality_suite"
            )

            # Add comprehensive expectations
            expectations_count = validator.add_comprehensive_expectations(
                validator=ge_validator,
                columns=list(pre_df.columns),
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
                date_columns=date_columns,
                key_columns=key_columns
            )

            # Add custom FHIR-specific expectations
            ge_validator.expect_column_values_to_be_in_set(
                column="status",
                value_set=['final', 'preliminary', 'amended', 'corrected', 'registered'],
                mostly=0.99,
                comment="FHIR Observation status values"
            )
            expectations_count += 1

            ge_validator.save_expectation_suite(discard_failed_expectations=False)

            print(f"✅ Added {expectations_count} expectations to suite\n")

        except Exception as e:
            print(f"⚠️  Could not create GE expectations: {e}")
            print("   Continuing with basic comparison...\n")
    else:
        print("⚠️  Great Expectations not available, using basic comparison\n")

    # ═══════════════════════════════════════════════════════════════
    # STEP 5: Compare pre/post data
    # ═══════════════════════════════════════════════════════════════

    print("🔍 Comparing pre-ingestion and post-ingestion data...\n")

    comparison_results = validator.compare_pre_post_data(
        pre_df=pre_df,
        post_df=post_df,
        comparison_name="bronze_to_silver_validation"
    )

    # ═══════════════════════════════════════════════════════════════
    # STEP 6: Display results
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)

    print(f"\n📊 Overall Quality Score: {comparison_results['quality_score']:.2f}%")

    if comparison_results['passed']:
        print("✅ Status: PASSED")
    else:
        print("❌ Status: FAILED")

    print(f"\n📈 Row Count:")
    print(f"   Pre-ingestion:  {comparison_results['pre_stats']['row_count']:,}")
    print(f"   Post-ingestion: {comparison_results['post_stats']['row_count']:,}")
    print(f"   Change: {comparison_results['differences']['row_count_change_pct']:.2f}%")

    print(f"\n📁 Column Count:")
    print(f"   Pre-ingestion:  {comparison_results['pre_stats']['column_count']}")
    print(f"   Post-ingestion: {comparison_results['post_stats']['column_count']}")

    if comparison_results['differences']['missing_columns']:
        print(f"\n⚠️  Missing Columns:")
        for col in comparison_results['differences']['missing_columns']:
            print(f"   - {col}")

    if comparison_results['differences']['column_differences']:
        print(f"\n➕ New Columns:")
        for col in comparison_results['differences']['column_differences']:
            print(f"   + {col}")

    # Show null percentage changes
    significant_null_changes = {
        k: v for k, v in comparison_results['differences']['null_changes'].items()
        if abs(v) > 2.0
    }

    if significant_null_changes:
        print(f"\n⚠️  Significant Null Percentage Changes:")
        for col, change in sorted(
            significant_null_changes.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ):
            pre_null = comparison_results['pre_stats']['null_percentages'].get(col, 0)
            post_null = comparison_results['post_stats']['null_percentages'].get(col, 0)
            print(f"   {col:30} {pre_null:6.2f}% → {post_null:6.2f}% ({change:+.2f}%)")

    # Show numeric value changes
    significant_numeric_changes = {
        k: v for k, v in comparison_results['differences']['numeric_changes'].items()
        if abs(v.get('mean_change_pct', 0)) > 5.0
    }

    if significant_numeric_changes:
        print(f"\n⚠️  Significant Mean Value Changes:")
        for col, change_info in sorted(
            significant_numeric_changes.items(),
            key=lambda x: abs(x[1].get('mean_change_pct', 0)),
            reverse=True
        ):
            pct_change = change_info.get('mean_change_pct', 0)
            pre_mean = comparison_results['pre_stats']['numeric_stats'][col]['mean']
            post_mean = comparison_results['post_stats']['numeric_stats'][col]['mean']
            print(f"   {col:30} {pre_mean:8.2f} → {post_mean:8.2f} ({pct_change:+.2f}%)")

    # Show all issues
    if comparison_results['differences']['issues']:
        print(f"\n🚨 Issues Detected ({len(comparison_results['differences']['issues'])}):")
        for i, issue in enumerate(comparison_results['differences']['issues'], 1):
            print(f"   {i}. {issue}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 7: Generate Excel report
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "="*70)
    print("GENERATING EXCEL REPORT")
    print("="*70 + "\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"FHIR_ETL_Quality_Report_{timestamp}.xlsx"

    try:
        report_path = validator.generate_excel_report(
            output_filename=report_filename,
            include_comparison=True
        )

        print(f"✅ Excel report generated successfully!")
        print(f"📁 Location: {report_path}")

        # Show report structure
        print(f"\n📊 Report Contents:")
        print(f"   Sheet 1: Executive Summary")
        print(f"   Sheet 2: Pre-Ingestion Stats")
        print(f"   Sheet 3: Post-Ingestion Stats")
        print(f"   Sheet 4: Comparison Results")
        print(f"   Sheet 5: Issues & Recommendations")

    except Exception as e:
        print(f"❌ Failed to generate Excel report: {e}")
        report_path = None

    # ═══════════════════════════════════════════════════════════════
    # STEP 8: Summary and recommendations
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "="*70)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*70 + "\n")

    quality_score = comparison_results['quality_score']

    if quality_score >= 90:
        print("✅ EXCELLENT - Pipeline quality is very high")
        print("   Recommendation: Continue normal operations and monitoring")
    elif quality_score >= 80:
        print("✅ GOOD - Pipeline quality is acceptable")
        print("   Recommendation: Review warnings but likely OK to proceed")
    elif quality_score >= 70:
        print("⚠️  ACCEPTABLE - Pipeline has some issues")
        print("   Recommendation: Investigation recommended before production")
    else:
        print("❌ NEEDS ATTENTION - Pipeline has significant issues")
        print("   Recommendation: DO NOT PROCEED - Fix issues immediately")

    print(f"\n📧 Next Steps:")
    print(f"   1. Review the Excel report: {report_path}")
    print(f"   2. Investigate flagged issues")
    print(f"   3. Validate ETL transformation logic")
    print(f"   4. Re-run validation after fixes")

    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70 + "\n")

    return comparison_results, report_path


if __name__ == "__main__":
    # Run the complete example
    try:
        results, report_path = run_full_validation_example()

        # Exit with appropriate code
        exit_code = 0 if results['passed'] else 1
        sys.exit(exit_code)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
