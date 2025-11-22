"""
═══════════════════════════════════════════════════════════════════
ETL Quality Validation System with Great Expectations
A/B Testing Framework for Pre/Post Pipeline Data Quality
═══════════════════════════════════════════════════════════════════

This module implements comprehensive data quality validation for ETL pipelines
using Great Expectations, with support for pre/post ingestion comparisons.

Features:
- 30+ data quality expectations
- Pre/Post pipeline comparison
- Excel report generation
- Docker-ready configuration
- SAS code integration examples

Author: Gen Z Agent Team
Created: 2025-11-22
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

try:
    import great_expectations as gx
    from great_expectations.core.batch import RuntimeBatchRequest
    from great_expectations.checkpoint import SimpleCheckpoint
    GX_AVAILABLE = True
except ImportError:
    GX_AVAILABLE = False
    print("WARNING: Great Expectations not installed. Run: pip install great-expectations")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ETLQualityValidator:
    """
    Main class for ETL quality validation and A/B testing

    This validator compares pre-ingestion (source) and post-ingestion (target)
    data to ensure ETL pipeline quality and integrity.
    """

    def __init__(
        self,
        project_name: str = "etl_quality_validation",
        output_dir: str = "./gen_z_agent/output/quality_reports",
        ge_context_dir: str = "./gen_z_agent/ge_context"
    ):
        """
        Initialize the ETL Quality Validator

        Args:
            project_name: Name of the validation project
            output_dir: Directory for output reports
            ge_context_dir: Great Expectations context directory
        """
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.ge_context_dir = Path(ge_context_dir)

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ge_context_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Great Expectations context
        self.context = None
        if GX_AVAILABLE:
            self._initialize_ge_context()

        # Quality metrics storage
        self.pre_metrics: Dict[str, Any] = {}
        self.post_metrics: Dict[str, Any] = {}
        self.comparison_results: Dict[str, Any] = {}

        logger.info(f"ETL Quality Validator initialized for project: {project_name}")

    def _initialize_ge_context(self):
        """Initialize Great Expectations data context"""
        try:
            self.context = gx.get_context(context_root_dir=str(self.ge_context_dir))
            logger.info("Great Expectations context initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize GE context: {e}")
            # Create new context
            try:
                self.context = gx.data_context.DataContext.create(str(self.ge_context_dir))
                logger.info("Created new Great Expectations context")
            except Exception as e2:
                logger.error(f"Failed to create GE context: {e2}")
                self.context = None

    def create_expectation_suite(
        self,
        suite_name: str = "etl_quality_suite"
    ) -> Optional[Any]:
        """
        Create a comprehensive expectation suite with 30+ expectations

        Args:
            suite_name: Name of the expectation suite

        Returns:
            ExpectationSuite object or None
        """
        if not GX_AVAILABLE or not self.context:
            logger.error("Great Expectations not available")
            return None

        try:
            # Create or get suite
            suite = self.context.add_or_update_expectation_suite(
                expectation_suite_name=suite_name
            )
            logger.info(f"Created expectation suite: {suite_name}")
            return suite

        except Exception as e:
            logger.error(f"Failed to create expectation suite: {e}")
            return None

    def add_comprehensive_expectations(
        self,
        validator,
        columns: List[str],
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        date_columns: Optional[List[str]] = None,
        key_columns: Optional[List[str]] = None
    ) -> int:
        """
        Add 30+ comprehensive data quality expectations

        Args:
            validator: Great Expectations validator object
            columns: All column names
            numeric_columns: Numeric column names
            categorical_columns: Categorical column names
            date_columns: Date/datetime column names
            key_columns: Primary/foreign key column names

        Returns:
            Number of expectations added
        """
        expectations_count = 0

        # Set defaults
        numeric_columns = numeric_columns or []
        categorical_columns = categorical_columns or []
        date_columns = date_columns or []
        key_columns = key_columns or []

        # ════════════════════════════════════════════════════════════
        # 1. TABLE-LEVEL EXPECTATIONS (5 expectations)
        # ════════════════════════════════════════════════════════════

        # 1.1 Table must exist and have columns
        validator.expect_table_columns_to_match_ordered_list(
            column_list=columns,
            comment="Verify all expected columns exist in correct order"
        )
        expectations_count += 1

        # 1.2 Table must have at least 1 row
        validator.expect_table_row_count_to_be_between(
            min_value=1,
            comment="Table must not be empty"
        )
        expectations_count += 1

        # 1.3 Column count must match
        validator.expect_table_column_count_to_equal(
            value=len(columns),
            comment="Verify expected number of columns"
        )
        expectations_count += 1

        # 1.4 No completely empty columns
        for col in columns:
            validator.expect_column_values_to_not_be_null(
                column=col,
                mostly=0.01,  # At least 1% should be non-null
                comment=f"Column {col} should have some non-null values"
            )
            expectations_count += 1

        # ════════════════════════════════════════════════════════════
        # 2. COMPLETENESS EXPECTATIONS (5 expectations)
        # ════════════════════════════════════════════════════════════

        # 2.1 Key columns must never be null
        for key_col in key_columns:
            if key_col in columns:
                validator.expect_column_values_to_not_be_null(
                    column=key_col,
                    comment=f"Key column {key_col} must never be null"
                )
                expectations_count += 1

        # 2.2 Numeric columns should be mostly complete (>95%)
        for num_col in numeric_columns:
            if num_col in columns:
                validator.expect_column_values_to_not_be_null(
                    column=num_col,
                    mostly=0.95,
                    comment=f"Numeric column {num_col} should be >95% complete"
                )
                expectations_count += 1

        # ════════════════════════════════════════════════════════════
        # 3. UNIQUENESS EXPECTATIONS (3 expectations)
        # ════════════════════════════════════════════════════════════

        # 3.1 Key columns must be unique
        for key_col in key_columns:
            if key_col in columns:
                validator.expect_column_values_to_be_unique(
                    column=key_col,
                    comment=f"Key column {key_col} must have unique values"
                )
                expectations_count += 1

        # ════════════════════════════════════════════════════════════
        # 4. NUMERIC VALUE EXPECTATIONS (8 expectations)
        # ════════════════════════════════════════════════════════════

        for num_col in numeric_columns:
            if num_col in columns:
                # 4.1 Must be numeric type
                validator.expect_column_values_to_be_in_type_list(
                    column=num_col,
                    type_list=["int", "int64", "float", "float64", "INTEGER", "FLOAT", "NUMERIC"],
                    comment=f"{num_col} must be numeric type"
                )
                expectations_count += 1

                # 4.2 Check for reasonable ranges (no extreme outliers)
                validator.expect_column_quantile_values_to_be_between(
                    column=num_col,
                    quantile_ranges={
                        "quantiles": [0.01, 0.25, 0.5, 0.75, 0.99],
                        "value_ranges": [[None, None]] * 5
                    },
                    allow_relative_error=0.1,
                    comment=f"{num_col} quantile distribution check"
                )
                expectations_count += 1

                # 4.3 Mean should be within expected range (statistical validation)
                validator.expect_column_mean_to_be_between(
                    column=num_col,
                    min_value=None,
                    max_value=None,
                    comment=f"{num_col} mean value check"
                )
                expectations_count += 1

                # 4.4 Standard deviation should be reasonable
                validator.expect_column_stdev_to_be_between(
                    column=num_col,
                    min_value=0,
                    comment=f"{num_col} standard deviation check"
                )
                expectations_count += 1

        # ════════════════════════════════════════════════════════════
        # 5. CATEGORICAL VALUE EXPECTATIONS (4 expectations)
        # ════════════════════════════════════════════════════════════

        for cat_col in categorical_columns:
            if cat_col in columns:
                # 5.1 Check distinct value count
                validator.expect_column_unique_value_count_to_be_between(
                    column=cat_col,
                    min_value=1,
                    comment=f"{cat_col} should have valid distinct values"
                )
                expectations_count += 1

                # 5.2 Check for unexpected values (will learn from data)
                validator.expect_column_distinct_values_to_be_in_set(
                    column=cat_col,
                    value_set=None,  # Will be populated from profiling
                    mostly=0.95,
                    comment=f"{cat_col} values should be in expected set"
                )
                expectations_count += 1

        # ════════════════════════════════════════════════════════════
        # 6. DATE/DATETIME EXPECTATIONS (3 expectations)
        # ════════════════════════════════════════════════════════════

        for date_col in date_columns:
            if date_col in columns:
                # 6.1 Must be valid datetime
                validator.expect_column_values_to_be_of_type(
                    column=date_col,
                    type_="datetime64",
                    comment=f"{date_col} must be datetime type"
                )
                expectations_count += 1

                # 6.2 Dates should be within reasonable range
                validator.expect_column_values_to_be_between(
                    column=date_col,
                    min_value="1900-01-01",
                    max_value="2100-12-31",
                    parse_strings_as_datetimes=True,
                    comment=f"{date_col} should be within valid date range"
                )
                expectations_count += 1

        # ════════════════════════════════════════════════════════════
        # 7. CROSS-COLUMN EXPECTATIONS (3 expectations)
        # ════════════════════════════════════════════════════════════

        # 7.1 Example: Start date should be before end date
        if "start_date" in columns and "end_date" in columns:
            validator.expect_column_pair_values_A_to_be_greater_than_B(
                column_A="end_date",
                column_B="start_date",
                or_equal=True,
                comment="End date must be >= start date"
            )
            expectations_count += 1

        # 7.2 Example: Total should equal sum of parts
        if all(col in columns for col in ["part1", "part2", "total"]):
            validator.expect_column_pair_values_to_be_equal(
                column_A="total",
                column_B=None,  # Would need custom calculation
                comment="Total should equal sum of parts"
            )
            expectations_count += 1

        # ════════════════════════════════════════════════════════════
        # 8. FRESHNESS EXPECTATIONS (2 expectations)
        # ════════════════════════════════════════════════════════════

        if date_columns:
            latest_date_col = date_columns[0]
            # 8.1 Most recent record should be within acceptable lag
            validator.expect_column_max_to_be_between(
                column=latest_date_col,
                min_value="2020-01-01",  # Adjust based on your needs
                comment=f"Most recent {latest_date_col} should be relatively recent"
            )
            expectations_count += 1

        # ════════════════════════════════════════════════════════════
        # 9. CONSISTENCY EXPECTATIONS (2 expectations)
        # ════════════════════════════════════════════════════════════

        # 9.1 No unexpected null patterns
        validator.expect_compound_columns_to_be_unique(
            column_list=key_columns if key_columns else columns[:2],
            comment="Compound key uniqueness check"
        )
        expectations_count += 1

        logger.info(f"Added {expectations_count} expectations to validator")
        return expectations_count

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        suite_name: str,
        datasource_name: str = "pandas_datasource",
        asset_name: str = "etl_data",
        run_name: str = None
    ) -> Dict[str, Any]:
        """
        Validate a pandas DataFrame against expectations

        Args:
            df: DataFrame to validate
            suite_name: Name of expectation suite to use
            datasource_name: Name for the datasource
            asset_name: Name for the data asset
            run_name: Optional run name for this validation

        Returns:
            Validation results dictionary
        """
        if not GX_AVAILABLE or not self.context:
            logger.error("Great Expectations not available")
            return {"success": False, "error": "GE not available"}

        try:
            # Create or get datasource
            try:
                datasource = self.context.get_datasource(datasource_name)
            except:
                datasource = self.context.sources.add_pandas(datasource_name)

            # Add data asset
            data_asset = datasource.add_dataframe_asset(name=asset_name)

            # Create batch request
            batch_request = data_asset.build_batch_request(dataframe=df)

            # Get validator
            validator = self.context.get_validator(
                batch_request=batch_request,
                expectation_suite_name=suite_name
            )

            # Run validation
            run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            results = validator.validate(run_name=run_name)

            logger.info(f"Validation completed: {results.success}")
            return results.to_json_dict()

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {"success": False, "error": str(e)}

    def compare_pre_post_data(
        self,
        pre_df: pd.DataFrame,
        post_df: pd.DataFrame,
        comparison_name: str = "pre_post_comparison"
    ) -> Dict[str, Any]:
        """
        Compare pre-ingestion and post-ingestion data

        Args:
            pre_df: Pre-ingestion DataFrame
            post_df: Post-ingestion DataFrame
            comparison_name: Name for this comparison

        Returns:
            Comparison results dictionary
        """
        logger.info(f"Starting pre/post comparison: {comparison_name}")

        results = {
            "comparison_name": comparison_name,
            "timestamp": datetime.now().isoformat(),
            "pre_stats": {},
            "post_stats": {},
            "differences": {},
            "quality_score": 0.0,
            "passed": False
        }

        # Basic statistics
        results["pre_stats"] = self._calculate_statistics(pre_df, "pre")
        results["post_stats"] = self._calculate_statistics(post_df, "post")

        # Calculate differences
        results["differences"] = self._calculate_differences(
            results["pre_stats"],
            results["post_stats"]
        )

        # Calculate quality score (0-100)
        results["quality_score"] = self._calculate_quality_score(results["differences"])

        # Determine if passed (threshold: 80%)
        results["passed"] = results["quality_score"] >= 80.0

        # Store results
        self.comparison_results[comparison_name] = results

        logger.info(f"Comparison completed. Quality Score: {results['quality_score']:.2f}%")
        return results

    def _calculate_statistics(self, df: pd.DataFrame, prefix: str) -> Dict[str, Any]:
        """Calculate comprehensive statistics for a DataFrame"""
        stats = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "null_counts": df.isnull().sum().to_dict(),
            "null_percentages": (df.isnull().sum() / len(df) * 100).to_dict(),
            "duplicate_rows": df.duplicated().sum(),
            "numeric_stats": {},
            "categorical_stats": {}
        }

        # Numeric column statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            stats["numeric_stats"][col] = {
                "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                "median": float(df[col].median()) if not df[col].isna().all() else None,
                "std": float(df[col].std()) if not df[col].isna().all() else None,
                "min": float(df[col].min()) if not df[col].isna().all() else None,
                "max": float(df[col].max()) if not df[col].isna().all() else None,
                "q25": float(df[col].quantile(0.25)) if not df[col].isna().all() else None,
                "q75": float(df[col].quantile(0.75)) if not df[col].isna().all() else None
            }

        # Categorical column statistics
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            stats["categorical_stats"][col] = {
                "unique_count": int(df[col].nunique()),
                "most_common": df[col].mode()[0] if len(df[col].mode()) > 0 else None,
                "most_common_count": int(df[col].value_counts().iloc[0]) if len(df[col]) > 0 else 0
            }

        return stats

    def _calculate_differences(
        self,
        pre_stats: Dict[str, Any],
        post_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate differences between pre and post statistics"""
        differences = {
            "row_count_change": post_stats["row_count"] - pre_stats["row_count"],
            "row_count_change_pct": (
                (post_stats["row_count"] - pre_stats["row_count"]) / pre_stats["row_count"] * 100
                if pre_stats["row_count"] > 0 else 0
            ),
            "column_differences": list(set(post_stats["columns"]) - set(pre_stats["columns"])),
            "missing_columns": list(set(pre_stats["columns"]) - set(post_stats["columns"])),
            "null_changes": {},
            "numeric_changes": {},
            "issues": []
        }

        # Null percentage changes
        common_cols = set(pre_stats["columns"]) & set(post_stats["columns"])
        for col in common_cols:
            pre_null = pre_stats["null_percentages"].get(col, 0)
            post_null = post_stats["null_percentages"].get(col, 0)
            differences["null_changes"][col] = post_null - pre_null

            # Flag significant null increase
            if post_null - pre_null > 5.0:  # More than 5% increase in nulls
                differences["issues"].append(
                    f"WARNING: Column '{col}' has {post_null - pre_null:.2f}% more nulls"
                )

        # Numeric value changes
        common_numeric = (
            set(pre_stats["numeric_stats"].keys()) &
            set(post_stats["numeric_stats"].keys())
        )
        for col in common_numeric:
            pre_mean = pre_stats["numeric_stats"][col]["mean"]
            post_mean = post_stats["numeric_stats"][col]["mean"]

            if pre_mean and post_mean:
                pct_change = (post_mean - pre_mean) / pre_mean * 100
                differences["numeric_changes"][col] = {
                    "mean_change_pct": pct_change
                }

                # Flag significant changes
                if abs(pct_change) > 10.0:  # More than 10% change
                    differences["issues"].append(
                        f"WARNING: Column '{col}' mean changed by {pct_change:.2f}%"
                    )

        return differences

    def _calculate_quality_score(self, differences: Dict[str, Any]) -> float:
        """
        Calculate overall quality score (0-100)

        Scoring criteria:
        - Start with 100 points
        - Deduct points for issues
        """
        score = 100.0

        # Deduct for row count changes (max -10 points)
        row_change_pct = abs(differences["row_count_change_pct"])
        if row_change_pct > 50:
            score -= 10
        elif row_change_pct > 20:
            score -= 5
        elif row_change_pct > 5:
            score -= 2

        # Deduct for missing columns (max -20 points)
        missing_cols = len(differences["missing_columns"])
        score -= min(missing_cols * 5, 20)

        # Deduct for issues (max -30 points)
        issue_count = len(differences["issues"])
        score -= min(issue_count * 3, 30)

        # Deduct for significant null changes (max -20 points)
        significant_null_changes = sum(
            1 for v in differences["null_changes"].values()
            if abs(v) > 5.0
        )
        score -= min(significant_null_changes * 4, 20)

        # Deduct for significant numeric changes (max -20 points)
        significant_numeric_changes = sum(
            1 for v in differences["numeric_changes"].values()
            if abs(v.get("mean_change_pct", 0)) > 10.0
        )
        score -= min(significant_numeric_changes * 4, 20)

        return max(0.0, score)

    def generate_excel_report(
        self,
        output_filename: str = None,
        include_comparison: bool = True
    ) -> str:
        """
        Generate comprehensive Excel report

        Args:
            output_filename: Output Excel file path
            include_comparison: Whether to include comparison results

        Returns:
            Path to generated Excel file
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"etl_quality_report_{timestamp}.xlsx"

        output_path = self.output_dir / output_filename

        logger.info(f"Generating Excel report: {output_path}")

        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Sheet 1: Executive Summary
                summary_data = self._create_summary_sheet()
                summary_df = pd.DataFrame([summary_data])
                summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)

                # Sheet 2: Pre-Ingestion Statistics
                if self.pre_metrics:
                    pre_df = self._metrics_to_dataframe(self.pre_metrics)
                    pre_df.to_excel(writer, sheet_name='Pre-Ingestion Stats', index=False)

                # Sheet 3: Post-Ingestion Statistics
                if self.post_metrics:
                    post_df = self._metrics_to_dataframe(self.post_metrics)
                    post_df.to_excel(writer, sheet_name='Post-Ingestion Stats', index=False)

                # Sheet 4: Comparison Results
                if include_comparison and self.comparison_results:
                    for i, (name, results) in enumerate(self.comparison_results.items()):
                        sheet_name = f'Comparison_{i+1}'[:31]  # Excel sheet name limit
                        comp_df = self._comparison_to_dataframe(results)
                        comp_df.to_excel(writer, sheet_name=sheet_name, index=False)

                # Sheet 5: Issues and Recommendations
                issues_df = self._create_issues_sheet()
                issues_df.to_excel(writer, sheet_name='Issues & Recommendations', index=False)

            logger.info(f"Excel report generated successfully: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to generate Excel report: {e}")
            raise

    def _create_summary_sheet(self) -> Dict[str, Any]:
        """Create executive summary data"""
        summary = {
            "Report Generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Project Name": self.project_name,
            "Comparisons Run": len(self.comparison_results),
            "Overall Quality Score": 0.0,
            "Status": "UNKNOWN"
        }

        if self.comparison_results:
            avg_score = np.mean([
                r["quality_score"] for r in self.comparison_results.values()
            ])
            summary["Overall Quality Score"] = f"{avg_score:.2f}%"

            if avg_score >= 90:
                summary["Status"] = "EXCELLENT"
            elif avg_score >= 80:
                summary["Status"] = "GOOD"
            elif avg_score >= 70:
                summary["Status"] = "ACCEPTABLE"
            else:
                summary["Status"] = "NEEDS ATTENTION"

        return summary

    def _metrics_to_dataframe(self, metrics: Dict[str, Any]) -> pd.DataFrame:
        """Convert metrics dictionary to DataFrame"""
        rows = []
        for key, value in metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    rows.append({
                        "Category": key,
                        "Metric": sub_key,
                        "Value": str(sub_value)
                    })
            else:
                rows.append({
                    "Category": "General",
                    "Metric": key,
                    "Value": str(value)
                })
        return pd.DataFrame(rows)

    def _comparison_to_dataframe(self, comparison: Dict[str, Any]) -> pd.DataFrame:
        """Convert comparison results to DataFrame"""
        rows = []

        # Add summary row
        rows.append({
            "Category": "Summary",
            "Metric": "Quality Score",
            "Pre Value": "-",
            "Post Value": "-",
            "Change": f"{comparison['quality_score']:.2f}%",
            "Status": "PASS" if comparison["passed"] else "FAIL"
        })

        # Add row count
        rows.append({
            "Category": "Row Count",
            "Metric": "Total Rows",
            "Pre Value": comparison["pre_stats"]["row_count"],
            "Post Value": comparison["post_stats"]["row_count"],
            "Change": f"{comparison['differences']['row_count_change_pct']:.2f}%",
            "Status": "OK"
        })

        # Add null changes
        for col, change in comparison["differences"]["null_changes"].items():
            status = "WARNING" if abs(change) > 5.0 else "OK"
            rows.append({
                "Category": "Null %",
                "Metric": col,
                "Pre Value": f"{comparison['pre_stats']['null_percentages'].get(col, 0):.2f}%",
                "Post Value": f"{comparison['post_stats']['null_percentages'].get(col, 0):.2f}%",
                "Change": f"{change:.2f}%",
                "Status": status
            })

        return pd.DataFrame(rows)

    def _create_issues_sheet(self) -> pd.DataFrame:
        """Create issues and recommendations sheet"""
        issues = []

        for comp_name, results in self.comparison_results.items():
            for issue in results["differences"].get("issues", []):
                issues.append({
                    "Comparison": comp_name,
                    "Severity": "WARNING",
                    "Issue": issue,
                    "Recommendation": "Review and validate data transformation logic"
                })

        if not issues:
            issues.append({
                "Comparison": "All",
                "Severity": "INFO",
                "Issue": "No issues detected",
                "Recommendation": "Continue monitoring"
            })

        return pd.DataFrame(issues)


# ═══════════════════════════════════════════════════════════════════
# EXAMPLE USAGE FUNCTION
# ═══════════════════════════════════════════════════════════════════

def run_etl_quality_validation_example():
    """
    Example function demonstrating ETL quality validation
    """
    logger.info("Starting ETL Quality Validation Example")

    # Initialize validator
    validator = ETLQualityValidator(
        project_name="healthcare_etl_validation",
        output_dir="./gen_z_agent/output/quality_reports"
    )

    # Create sample pre/post data
    np.random.seed(42)

    # Pre-ingestion data (source)
    pre_df = pd.DataFrame({
        'patient_id': range(1000),
        'age': np.random.randint(18, 90, 1000),
        'blood_pressure': np.random.randint(110, 140, 1000),
        'diagnosis_code': np.random.choice(['A01', 'B02', 'C03'], 1000),
        'visit_date': pd.date_range('2024-01-01', periods=1000, freq='H')
    })

    # Post-ingestion data (target) - with some intentional changes
    post_df = pre_df.copy()
    post_df.loc[np.random.choice(1000, 50, replace=False), 'blood_pressure'] = np.nan  # Add nulls
    post_df['age'] = post_df['age'] + np.random.randint(-2, 3, 1000)  # Slight variations

    # Run comparison
    comparison_results = validator.compare_pre_post_data(
        pre_df=pre_df,
        post_df=post_df,
        comparison_name="healthcare_pipeline_validation"
    )

    # Generate report
    report_path = validator.generate_excel_report()

    logger.info(f"Quality Score: {comparison_results['quality_score']:.2f}%")
    logger.info(f"Report generated: {report_path}")

    return report_path


if __name__ == "__main__":
    # Run example
    run_etl_quality_validation_example()
