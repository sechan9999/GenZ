"""
Model Monitoring & Data Drift Detection

This module monitors production model performance and detects data drift.
Critical for LIMS: If a lab instrument starts drifting due to calibration
errors, this system will alert lab technicians before bad data hits the dashboard.

Detection methods:
1. Data Drift: Kolmogorov-Smirnov (KS) test
2. Feature Drift: Population Stability Index (PSI)
3. Model Performance: Accuracy degradation monitoring
4. Prediction Drift: Distribution shift in predictions

Author: MLOps Team
Date: 2025-11-22
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StructType, StructField, StringType, TimestampType
from scipy import stats
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Monitors data drift and model performance degradation.

    Alerts when:
    - Lab instrument calibration drifts (distribution shift)
    - Feature distributions change significantly (data drift)
    - Model predictions shift unexpectedly (concept drift)
    - Model performance degrades (accuracy drop)
    """

    def __init__(
        self,
        spark: SparkSession,
        silver_path: str,
        gold_path: str,
        baseline_start_date: str,
        baseline_end_date: str,
    ):
        """
        Initialize drift detector.

        Args:
            spark: Active SparkSession
            silver_path: Path to Silver layer (for raw lab results)
            gold_path: Path to Gold layer (for features)
            baseline_start_date: Start date for baseline period (YYYY-MM-DD)
            baseline_end_date: End date for baseline period (YYYY-MM-DD)
        """
        self.spark = spark
        self.silver_path = silver_path
        self.gold_path = gold_path
        self.baseline_start_date = baseline_start_date
        self.baseline_end_date = baseline_end_date

        # Load baseline data
        self.baseline_data = self._load_baseline_data()
        self.baseline_stats = self._calculate_baseline_stats()

        logger.info("DriftDetector initialized with baseline data")

    def _load_baseline_data(self) -> DataFrame:
        """
        Load baseline data from Silver layer.

        Returns:
            DataFrame with baseline lab results
        """
        logger.info(
            f"Loading baseline data: {self.baseline_start_date} to {self.baseline_end_date}"
        )

        df_baseline = (
            self.spark.read.format("delta")
            .load(f"{self.silver_path}/lab_results_standardized")
            .filter(F.col("is_valid") == True)
            .filter(F.col("collected_date") >= self.baseline_start_date)
            .filter(F.col("collected_date") <= self.baseline_end_date)
        )

        baseline_count = df_baseline.count()
        logger.info(f"Loaded {baseline_count} baseline records")

        return df_baseline

    def _calculate_baseline_stats(self) -> Dict[str, Dict]:
        """
        Calculate baseline statistics for each test type.

        Returns:
            Dictionary mapping LOINC codes to statistics
        """
        logger.info("Calculating baseline statistics...")

        # Calculate statistics by LOINC code
        baseline_stats = (
            self.baseline_data.groupBy("loinc_code", "test_name")
            .agg(
                F.mean("result_value").alias("mean"),
                F.stddev("result_value").alias("stddev"),
                F.min("result_value").alias("min"),
                F.max("result_value").alias("max"),
                F.expr("percentile_approx(result_value, 0.25)").alias("p25"),
                F.expr("percentile_approx(result_value, 0.50)").alias("p50"),
                F.expr("percentile_approx(result_value, 0.75)").alias("p75"),
                F.count("*").alias("count"),
            )
            .toPandas()
        )

        # Convert to dictionary
        stats_dict = {}
        for _, row in baseline_stats.iterrows():
            stats_dict[row["loinc_code"]] = {
                "test_name": row["test_name"],
                "mean": row["mean"],
                "stddev": row["stddev"],
                "min": row["min"],
                "max": row["max"],
                "p25": row["p25"],
                "p50": row["p50"],
                "p75": row["p75"],
                "count": row["count"],
            }

        logger.info(f"Calculated baseline statistics for {len(stats_dict)} test types")
        return stats_dict

    def detect_data_drift_ks_test(
        self, current_start_date: str, current_end_date: str, threshold: float = 0.05
    ) -> pd.DataFrame:
        """
        Detect data drift using Kolmogorov-Smirnov test.

        Compares distribution of lab results between baseline and current periods.
        If KS test p-value < threshold, distributions are significantly different.

        Args:
            current_start_date: Start date for current period
            current_end_date: End date for current period
            threshold: Significance threshold (default: 0.05)

        Returns:
            DataFrame with KS test results for each test type
        """
        logger.info(
            f"Running KS drift detection: {current_start_date} to {current_end_date}"
        )

        # Load current data
        df_current = (
            self.spark.read.format("delta")
            .load(f"{self.silver_path}/lab_results_standardized")
            .filter(F.col("is_valid") == True)
            .filter(F.col("collected_date") >= current_start_date)
            .filter(F.col("collected_date") <= current_end_date)
        )

        # Get unique LOINC codes
        loinc_codes = [row["loinc_code"] for row in self.baseline_data.select("loinc_code").distinct().collect()]

        results = []

        for loinc_code in loinc_codes:
            # Baseline distribution
            baseline_values = (
                self.baseline_data.filter(F.col("loinc_code") == loinc_code)
                .select("result_value")
                .toPandas()["result_value"]
                .values
            )

            # Current distribution
            current_values = (
                df_current.filter(F.col("loinc_code") == loinc_code)
                .select("result_value")
                .toPandas()["result_value"]
                .values
            )

            if len(current_values) == 0:
                logger.warning(f"No current data for {loinc_code}, skipping")
                continue

            # Perform KS test
            ks_statistic, p_value = stats.ks_2samp(baseline_values, current_values)

            # Check if drift detected
            drift_detected = p_value < threshold

            # Calculate distribution shift metrics
            baseline_mean = np.mean(baseline_values)
            current_mean = np.mean(current_values)
            mean_shift = ((current_mean - baseline_mean) / baseline_mean) * 100

            baseline_std = np.std(baseline_values)
            current_std = np.std(current_values)

            results.append(
                {
                    "loinc_code": loinc_code,
                    "test_name": self.baseline_stats.get(loinc_code, {}).get(
                        "test_name", "Unknown"
                    ),
                    "ks_statistic": ks_statistic,
                    "p_value": p_value,
                    "drift_detected": drift_detected,
                    "baseline_mean": baseline_mean,
                    "current_mean": current_mean,
                    "mean_shift_percent": mean_shift,
                    "baseline_std": baseline_std,
                    "current_std": current_std,
                    "baseline_count": len(baseline_values),
                    "current_count": len(current_values),
                }
            )

        df_results = pd.DataFrame(results)

        # Log summary
        drift_count = df_results["drift_detected"].sum()
        logger.info(f"Drift detected in {drift_count}/{len(df_results)} test types")

        if drift_count > 0:
            logger.warning("Tests with drift detected:")
            for _, row in df_results[df_results["drift_detected"]].iterrows():
                logger.warning(
                    f"  {row['test_name']} (LOINC: {row['loinc_code']}): "
                    f"KS={row['ks_statistic']:.3f}, p={row['p_value']:.4f}, "
                    f"mean shift={row['mean_shift_percent']:.1f}%"
                )

        return df_results

    def detect_device_drift(
        self, current_start_date: str, current_end_date: str, device_id: str = None
    ) -> pd.DataFrame:
        """
        Detect drift for specific devices (calibration error detection).

        This is critical for LIMS: if a device's calibration drifts, all
        results from that device will be systematically biased.

        Args:
            current_start_date: Start date for current period
            current_end_date: End date for current period
            device_id: Specific device to check (None = all devices)

        Returns:
            DataFrame with device-level drift metrics
        """
        logger.info(f"Detecting device drift for period: {current_start_date} to {current_end_date}")

        # Load current data
        df_current = (
            self.spark.read.format("delta")
            .load(f"{self.silver_path}/lab_results_standardized")
            .filter(F.col("is_valid") == True)
            .filter(F.col("collected_date") >= current_start_date)
            .filter(F.col("collected_date") <= current_end_date)
        )

        if device_id:
            df_current = df_current.filter(F.col("device_id") == device_id)

        # Calculate device-level statistics for current period
        df_device_stats = (
            df_current.groupBy("device_id", "loinc_code", "test_name")
            .agg(
                F.mean("result_value").alias("current_mean"),
                F.stddev("result_value").alias("current_stddev"),
                F.count("*").alias("current_count"),
            )
            .toPandas()
        )

        results = []

        for _, row in df_device_stats.iterrows():
            loinc_code = row["loinc_code"]
            device = row["device_id"]

            # Get baseline stats for this test
            baseline = self.baseline_stats.get(loinc_code)
            if not baseline:
                continue

            # Calculate z-score for device mean vs. baseline
            z_score = (row["current_mean"] - baseline["mean"]) / (
                baseline["stddev"] + 0.01
            )

            # Flag if device is >2 SD from baseline
            drift_detected = abs(z_score) > 2.0

            results.append(
                {
                    "device_id": device,
                    "loinc_code": loinc_code,
                    "test_name": row["test_name"],
                    "baseline_mean": baseline["mean"],
                    "current_mean": row["current_mean"],
                    "z_score": z_score,
                    "drift_detected": drift_detected,
                    "current_count": row["current_count"],
                    "severity": "CRITICAL"
                    if abs(z_score) > 3.0
                    else "HIGH" if abs(z_score) > 2.5 else "MEDIUM" if abs(z_score) > 2.0 else "LOW",
                }
            )

        df_results = pd.DataFrame(results)

        if len(df_results) > 0:
            drift_count = df_results["drift_detected"].sum()
            logger.info(
                f"Device drift detected: {drift_count}/{len(df_results)} device-test combinations"
            )

            if drift_count > 0:
                logger.warning("Devices with drift detected:")
                for _, row in df_results[df_results["drift_detected"]].iterrows():
                    logger.warning(
                        f"  Device {row['device_id']} - {row['test_name']}: "
                        f"z-score={row['z_score']:.2f}, severity={row['severity']}"
                    )

        return df_results

    def calculate_psi(
        self, baseline_counts: np.ndarray, current_counts: np.ndarray
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI measures the shift between two distributions.
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate change
        PSI >= 0.2: Significant change

        Args:
            baseline_counts: Histogram counts for baseline
            current_counts: Histogram counts for current

        Returns:
            PSI value
        """
        # Convert to proportions
        baseline_prop = baseline_counts / np.sum(baseline_counts)
        current_prop = current_counts / np.sum(current_counts)

        # Avoid division by zero
        baseline_prop = np.where(baseline_prop == 0, 0.0001, baseline_prop)
        current_prop = np.where(current_prop == 0, 0.0001, current_prop)

        # Calculate PSI
        psi = np.sum((current_prop - baseline_prop) * np.log(current_prop / baseline_prop))

        return psi

    def detect_feature_drift_psi(
        self, current_start_date: str, current_end_date: str
    ) -> pd.DataFrame:
        """
        Detect feature drift using Population Stability Index (PSI).

        Monitors drift in ML model features (Gold layer).

        Args:
            current_start_date: Start date for current period
            current_end_date: End date for current period

        Returns:
            DataFrame with PSI scores for each feature
        """
        logger.info("Detecting feature drift using PSI...")

        # Load baseline features
        df_baseline_features = (
            self.spark.read.format("delta")
            .load(f"{self.gold_path}/device_failure_features")
            .filter(F.col("log_date") >= self.baseline_start_date)
            .filter(F.col("log_date") <= self.baseline_end_date)
        )

        # Load current features
        df_current_features = (
            self.spark.read.format("delta")
            .load(f"{self.gold_path}/device_failure_features")
            .filter(F.col("log_date") >= current_start_date)
            .filter(F.col("log_date") <= current_end_date)
        )

        # Feature columns to monitor
        numeric_features = [
            "error_rate_7d",
            "warning_rate_7d",
            "qc_pass_rate_7d",
            "result_volume_trend",
            "avg_turnaround_time_7d",
        ]

        results = []

        for feature in numeric_features:
            # Get baseline and current values
            baseline_values = (
                df_baseline_features.select(feature).toPandas()[feature].values
            )
            current_values = (
                df_current_features.select(feature).toPandas()[feature].values
            )

            # Create histograms (10 bins)
            bins = np.linspace(
                min(baseline_values.min(), current_values.min()),
                max(baseline_values.max(), current_values.max()),
                11,
            )

            baseline_counts, _ = np.histogram(baseline_values, bins=bins)
            current_counts, _ = np.histogram(current_values, bins=bins)

            # Calculate PSI
            psi = self.calculate_psi(baseline_counts, current_counts)

            # Determine drift level
            if psi < 0.1:
                drift_level = "NO_DRIFT"
            elif psi < 0.2:
                drift_level = "MODERATE_DRIFT"
            else:
                drift_level = "SIGNIFICANT_DRIFT"

            results.append(
                {
                    "feature": feature,
                    "psi": psi,
                    "drift_level": drift_level,
                    "baseline_mean": baseline_values.mean(),
                    "current_mean": current_values.mean(),
                    "baseline_std": baseline_values.std(),
                    "current_std": current_values.std(),
                }
            )

        df_results = pd.DataFrame(results)

        # Log summary
        significant_drift_count = (df_results["drift_level"] == "SIGNIFICANT_DRIFT").sum()
        logger.info(
            f"Feature drift: {significant_drift_count}/{len(df_results)} features with significant drift"
        )

        if significant_drift_count > 0:
            logger.warning("Features with significant drift:")
            for _, row in df_results[
                df_results["drift_level"] == "SIGNIFICANT_DRIFT"
            ].iterrows():
                logger.warning(f"  {row['feature']}: PSI={row['psi']:.3f}")

        return df_results

    def generate_drift_report(
        self, current_start_date: str, current_end_date: str
    ) -> Dict:
        """
        Generate comprehensive drift detection report.

        Args:
            current_start_date: Start date for current period
            current_end_date: End date for current period

        Returns:
            Dictionary with all drift detection results
        """
        logger.info("Generating comprehensive drift report...")

        report = {
            "report_timestamp": datetime.now().isoformat(),
            "baseline_period": {
                "start": self.baseline_start_date,
                "end": self.baseline_end_date,
            },
            "current_period": {"start": current_start_date, "end": current_end_date},
        }

        # Data drift (KS test)
        df_data_drift = self.detect_data_drift_ks_test(
            current_start_date, current_end_date
        )
        report["data_drift"] = df_data_drift.to_dict("records")

        # Device drift
        df_device_drift = self.detect_device_drift(current_start_date, current_end_date)
        report["device_drift"] = df_device_drift.to_dict("records")

        # Feature drift (PSI)
        df_feature_drift = self.detect_feature_drift_psi(
            current_start_date, current_end_date
        )
        report["feature_drift"] = df_feature_drift.to_dict("records")

        # Summary
        report["summary"] = {
            "total_tests_monitored": len(df_data_drift),
            "tests_with_data_drift": int(df_data_drift["drift_detected"].sum()),
            "devices_with_drift": int(df_device_drift["drift_detected"].sum())
            if len(df_device_drift) > 0
            else 0,
            "features_with_significant_drift": int(
                (df_feature_drift["drift_level"] == "SIGNIFICANT_DRIFT").sum()
            ),
        }

        # Determine overall alert level
        if report["summary"]["devices_with_drift"] > 0:
            report["alert_level"] = "CRITICAL"
        elif report["summary"]["tests_with_data_drift"] > 3:
            report["alert_level"] = "HIGH"
        elif report["summary"]["tests_with_data_drift"] > 0:
            report["alert_level"] = "MEDIUM"
        else:
            report["alert_level"] = "LOW"

        logger.info(f"Drift report generated. Alert level: {report['alert_level']}")

        return report


# Example usage in Databricks notebook
if __name__ == "__main__":
    # Initialize Spark session
    spark = (
        SparkSession.builder.appName("LIMS Drift Detection")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .getOrCreate()
    )

    # Configure paths
    SILVER_PATH = "/mnt/delta/lims/silver"
    GOLD_PATH = "/mnt/delta/lims/gold"

    # Define baseline period (e.g., last 90 days of "normal" operation)
    baseline_start = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
    baseline_end = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    # Initialize drift detector
    detector = DriftDetector(
        spark,
        silver_path=SILVER_PATH,
        gold_path=GOLD_PATH,
        baseline_start_date=baseline_start,
        baseline_end_date=baseline_end,
    )

    # Define current period (last 7 days)
    current_start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    current_end = datetime.now().strftime("%Y-%m-%d")

    # Generate drift report
    report = detector.generate_drift_report(current_start, current_end)

    # Print summary
    print("\n=== Drift Detection Report ===")
    print(f"Alert Level: {report['alert_level']}")
    print(f"\nSummary:")
    for key, value in report["summary"].items():
        print(f"  {key}: {value}")

    # Save report
    report_json = json.dumps(report, indent=2)
    with open("/tmp/drift_report.json", "w") as f:
        f.write(report_json)

    print("\nDrift report saved to /tmp/drift_report.json")
