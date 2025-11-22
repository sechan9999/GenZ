"""
Gold Layer: Feature Engineering for ML Models

This module creates ML-ready features for:
1. Device Failure Prediction
2. Outbreak Risk Detection
3. Quality Anomaly Detection

Author: Data Engineering Team
Date: 2025-11-22
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoldLayerFeatureEngineering:
    """
    Creates ML-ready features from Silver layer data.

    Feature sets:
    - Device failure features: Predict device failure in next 7 days
    - Outbreak risk features: Detect potential disease outbreaks
    - Quality anomaly features: Detect data quality issues
    """

    def __init__(self, spark: SparkSession, silver_path: str, gold_path: str):
        """
        Initialize Gold layer feature engineering.

        Args:
            spark: Active SparkSession
            silver_path: Base path for Silver layer Delta tables
            gold_path: Base path for Gold layer Delta tables
        """
        self.spark = spark
        self.silver_path = silver_path
        self.gold_path = gold_path
        logger.info("Initialized GoldLayerFeatureEngineering")

    def create_device_failure_features(self) -> Dict[str, int]:
        """
        Create features for device failure prediction model.

        Features:
        - calibration_overdue: Boolean, >30 days since calibration
        - maintenance_overdue: Boolean, >90 days since maintenance
        - error_rate_7d: Rolling 7-day error rate
        - qc_fail_rate_7d: Rolling 7-day QC failure rate
        - result_volume_trend: Increasing/decreasing trend
        - device_age_months: Age of device
        - last_failure_days: Days since last failure (if any)
        - avg_turnaround_time_7d: Average test turnaround time

        Returns:
            Dictionary with statistics
        """
        logger.info("Creating device failure features")

        # Read Silver layer data
        df_metrics = self.spark.read.format("delta").load(
            f"{self.silver_path}/device_metrics_normalized"
        )
        df_results = self.spark.read.format("delta").load(
            f"{self.silver_path}/lab_results_standardized"
        )
        df_qc = self.spark.read.format("delta").load(
            f"{self.bronze_path.replace('bronze', 'bronze')}/quality_control_raw"
        )

        # Calculate rolling metrics (7-day windows)
        window_7d = (
            Window.partitionBy("device_id")
            .orderBy(F.col("log_date").cast("long"))
            .rangeBetween(-7 * 86400, 0)
        )

        df_rolling = df_metrics.withColumn(
            "error_rate_7d",
            F.sum("error_count").over(window_7d)
            / (F.sum("result_volume").over(window_7d) + 1),
        ).withColumn(
            "warning_rate_7d",
            F.sum("warning_count").over(window_7d)
            / (F.sum("result_volume").over(window_7d) + 1),
        )

        # Calculate QC pass rate (7-day rolling)
        df_qc_agg = (
            df_qc.withColumn("qc_date", F.to_date("timestamp"))
            .groupBy("device_id", "qc_date")
            .agg(
                F.count("*").alias("qc_total"),
                F.sum(F.when(F.col("passed") == True, 1).otherwise(0)).alias(
                    "qc_passed"
                ),
            )
            .withColumn("qc_pass_rate", F.col("qc_passed") / F.col("qc_total"))
        )

        window_qc_7d = (
            Window.partitionBy("device_id")
            .orderBy(F.col("qc_date").cast("long"))
            .rangeBetween(-7 * 86400, 0)
        )

        df_qc_rolling = df_qc_agg.withColumn(
            "qc_pass_rate_7d", F.avg("qc_pass_rate").over(window_qc_7d)
        )

        # Join rolling metrics with QC data
        df_features = df_rolling.join(
            df_qc_rolling.select("device_id", "qc_date", "qc_pass_rate_7d"),
            (df_rolling.device_id == df_qc_rolling.device_id)
            & (df_rolling.log_date == df_qc_rolling.qc_date),
            how="left",
        ).select(df_rolling["*"], df_qc_rolling.qc_pass_rate_7d)

        # Calculate result volume trend (linear regression slope)
        window_30d = (
            Window.partitionBy("device_id")
            .orderBy(F.col("log_date").cast("long"))
            .rangeBetween(-30 * 86400, 0)
        )

        df_features = df_features.withColumn(
            "result_volume_mean_30d", F.avg("result_volume").over(window_30d)
        ).withColumn(
            "result_volume_trend",
            (F.col("result_volume") - F.col("result_volume_mean_30d"))
            / (F.col("result_volume_mean_30d") + 1),
        )

        # Flag overdue maintenance/calibration
        df_features = (
            df_features.withColumn(
                "calibration_overdue", F.col("calibration_days_since") > 30
            )
            .withColumn("maintenance_overdue", F.col("maintenance_days_since") > 90)
            .withColumn(
                "calibration_overdue_days",
                F.greatest(F.col("calibration_days_since") - 30, F.lit(0)),
            )
            .withColumn(
                "maintenance_overdue_days",
                F.greatest(F.col("maintenance_days_since") - 90, F.lit(0)),
            )
        )

        # Calculate average turnaround time (7-day rolling)
        df_turnaround = (
            df_results.filter(F.col("is_valid") == True)
            .groupBy("device_id", "collected_date")
            .agg(F.avg("turnaround_time_hours").alias("avg_turnaround_time"))
        )

        window_tat_7d = (
            Window.partitionBy("device_id")
            .orderBy(F.col("collected_date").cast("long"))
            .rangeBetween(-7 * 86400, 0)
        )

        df_turnaround = df_turnaround.withColumn(
            "avg_turnaround_time_7d", F.avg("avg_turnaround_time").over(window_tat_7d)
        )

        # Join with turnaround time
        df_features = df_features.join(
            df_turnaround.select(
                "device_id", "collected_date", "avg_turnaround_time_7d"
            ),
            (df_features.device_id == df_turnaround.device_id)
            & (df_features.log_date == df_turnaround.collected_date),
            how="left",
        ).select(df_features["*"], df_turnaround.avg_turnaround_time_7d)

        # Select final features
        df_gold = df_features.select(
            "device_id",
            "log_date",
            "calibration_overdue",
            "maintenance_overdue",
            "calibration_overdue_days",
            "maintenance_overdue_days",
            F.coalesce("error_rate_7d", F.lit(0.0)).alias("error_rate_7d"),
            F.coalesce("warning_rate_7d", F.lit(0.0)).alias("warning_rate_7d"),
            F.coalesce("qc_pass_rate_7d", F.lit(1.0)).alias("qc_pass_rate_7d"),
            F.coalesce("result_volume_trend", F.lit(0.0)).alias("result_volume_trend"),
            F.coalesce("avg_turnaround_time_7d", F.lit(0.0)).alias(
                "avg_turnaround_time_7d"
            ),
            "result_volume",
        )

        # Write to Gold layer
        delta_path = f"{self.gold_path}/device_failure_features"

        df_gold.write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).partitionBy("log_date").save(delta_path)

        records_written = df_gold.count()
        logger.info(
            f"Device failure features created: {records_written} device-days"
        )

        return {
            "table": "device_failure_features",
            "records_written": records_written,
            "features_count": len(df_gold.columns) - 2,  # Exclude device_id, log_date
        }

    def create_outbreak_risk_features(self) -> Dict[str, int]:
        """
        Create features for outbreak risk detection.

        Features:
        - positive_rate_7d: % positive tests for specific pathogens
        - case_count_delta: Week-over-week change in positive cases
        - geographic_cluster_score: Spatial clustering metric
        - temporal_cluster_score: Temporal clustering metric
        - seasonal_baseline_deviation: Deviation from seasonal baseline

        Note: This example focuses on temporal features. Geographic features
        would require patient location data (not included in standard LIMS).

        Returns:
            Dictionary with statistics
        """
        logger.info("Creating outbreak risk features")

        # Read Silver layer data
        df_results = self.spark.read.format("delta").load(
            f"{self.silver_path}/lab_results_standardized"
        )

        # Focus on infectious disease tests (example LOINC codes)
        infectious_disease_codes = [
            "94500-6",  # COVID-19 RNA
            "85478-6",  # Influenza A+B
            "11268-0",  # RSV
            "6349-5",  # Hepatitis C
            "5196-1",  # Hepatitis B
        ]

        df_infectious = df_results.filter(
            F.col("loinc_code").isin(infectious_disease_codes)
        )

        # Determine positive results (assuming result_value > threshold or specific codes)
        # This is simplified - actual logic depends on test type
        df_infectious = df_infectious.withColumn(
            "is_positive",
            F.when(
                (F.col("loinc_code").isin(["94500-6", "85478-6", "11268-0"]))
                & (F.col("result_value") > 0),
                True,
            ).otherwise(False),
        )

        # Aggregate by test type and date
        df_daily = (
            df_infectious.groupBy("loinc_code", "test_name", "collected_date")
            .agg(
                F.count("*").alias("total_tests"),
                F.sum(F.when(F.col("is_positive") == True, 1).otherwise(0)).alias(
                    "positive_tests"
                ),
            )
            .withColumn(
                "positive_rate", F.col("positive_tests") / F.col("total_tests")
            )
        )

        # Calculate rolling 7-day metrics
        window_7d = (
            Window.partitionBy("loinc_code")
            .orderBy(F.col("collected_date").cast("long"))
            .rangeBetween(-7 * 86400, 0)
        )

        df_rolling = df_daily.withColumn(
            "positive_rate_7d", F.avg("positive_rate").over(window_7d)
        ).withColumn(
            "total_tests_7d", F.sum("total_tests").over(window_7d)
        ).withColumn("positive_tests_7d", F.sum("positive_tests").over(window_7d))

        # Calculate week-over-week change
        window_lag_7d = (
            Window.partitionBy("loinc_code")
            .orderBy("collected_date")
            .rowsBetween(-7, -7)
        )

        df_delta = df_rolling.withColumn(
            "positive_tests_7d_prev", F.lag("positive_tests_7d", 7).over(window_lag_7d)
        ).withColumn(
            "case_count_delta",
            (F.col("positive_tests_7d") - F.col("positive_tests_7d_prev"))
            / (F.col("positive_tests_7d_prev") + 1),
        )

        # Calculate seasonal baseline (52-week rolling average)
        window_52w = (
            Window.partitionBy("loinc_code")
            .orderBy(F.col("collected_date").cast("long"))
            .rangeBetween(-365 * 86400, 0)
        )

        df_seasonal = df_delta.withColumn(
            "seasonal_baseline", F.avg("positive_rate").over(window_52w)
        ).withColumn(
            "seasonal_deviation",
            (F.col("positive_rate_7d") - F.col("seasonal_baseline"))
            / (F.col("seasonal_baseline") + 0.01),
        )

        # Calculate outbreak risk score (composite metric)
        df_outbreak = df_seasonal.withColumn(
            "outbreak_risk_score",
            (
                F.when(F.col("positive_rate_7d") > 0.1, 25)
                .when(F.col("positive_rate_7d") > 0.05, 15)
                .when(F.col("positive_rate_7d") > 0.02, 5)
                .otherwise(0)
                + F.when(F.col("case_count_delta") > 0.5, 25)
                .when(F.col("case_count_delta") > 0.2, 15)
                .when(F.col("case_count_delta") > 0.1, 5)
                .otherwise(0)
                + F.when(F.col("seasonal_deviation") > 2.0, 25)
                .when(F.col("seasonal_deviation") > 1.0, 15)
                .when(F.col("seasonal_deviation") > 0.5, 5)
                .otherwise(0)
            ),
        ).withColumn(
            "risk_level",
            F.when(F.col("outbreak_risk_score") >= 50, "HIGH")
            .when(F.col("outbreak_risk_score") >= 25, "MEDIUM")
            .otherwise("LOW"),
        )

        # Select final features
        df_gold = df_outbreak.select(
            "loinc_code",
            "test_name",
            "collected_date",
            "positive_rate_7d",
            "total_tests_7d",
            "positive_tests_7d",
            F.coalesce("case_count_delta", F.lit(0.0)).alias("case_count_delta"),
            F.coalesce("seasonal_baseline", F.lit(0.0)).alias("seasonal_baseline"),
            F.coalesce("seasonal_deviation", F.lit(0.0)).alias("seasonal_deviation"),
            "outbreak_risk_score",
            "risk_level",
        )

        # Write to Gold layer
        delta_path = f"{self.gold_path}/outbreak_risk_features"

        df_gold.write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).partitionBy("collected_date").save(delta_path)

        records_written = df_gold.count()
        high_risk_count = df_gold.filter(F.col("risk_level") == "HIGH").count()

        logger.info(f"Outbreak risk features created: {records_written} test-days")
        logger.info(f"High-risk alerts: {high_risk_count}")

        return {
            "table": "outbreak_risk_features",
            "records_written": records_written,
            "high_risk_alerts": high_risk_count,
        }

    def create_quality_anomaly_features(self) -> Dict[str, int]:
        """
        Create features for quality anomaly detection.

        Features:
        - result_distribution_shift: Statistical test for distribution change
        - outlier_rate: % results > 3 standard deviations
        - missing_data_rate: % missing/null results
        - device_consistency_score: Variance across devices for same test

        Returns:
            Dictionary with statistics
        """
        logger.info("Creating quality anomaly features")

        # Read Silver layer data
        df_results = self.spark.read.format("delta").load(
            f"{self.silver_path}/lab_results_standardized"
        )

        # Calculate baseline statistics for each test (90-day rolling)
        window_90d = (
            Window.partitionBy("loinc_code")
            .orderBy(F.col("collected_date").cast("long"))
            .rangeBetween(-90 * 86400, 0)
        )

        df_baseline = (
            df_results.filter(F.col("is_valid") == True)
            .withColumn("result_mean_90d", F.avg("result_value").over(window_90d))
            .withColumn("result_stddev_90d", F.stddev("result_value").over(window_90d))
            .withColumn(
                "z_score",
                (F.col("result_value") - F.col("result_mean_90d"))
                / (F.col("result_stddev_90d") + 0.01),
            )
            .withColumn("is_outlier", F.abs(F.col("z_score")) > 3)
        )

        # Aggregate by test and date
        df_daily_qm = (
            df_baseline.groupBy("loinc_code", "test_name", "collected_date")
            .agg(
                F.count("*").alias("total_results"),
                F.sum(F.when(F.col("is_outlier") == True, 1).otherwise(0)).alias(
                    "outlier_count"
                ),
                F.avg("result_value").alias("daily_mean"),
                F.stddev("result_value").alias("daily_stddev"),
            )
            .withColumn("outlier_rate", F.col("outlier_count") / F.col("total_results"))
        )

        # Calculate distribution shift (compare daily mean to 90-day baseline)
        window_baseline = (
            Window.partitionBy("loinc_code")
            .orderBy(F.col("collected_date").cast("long"))
            .rangeBetween(-90 * 86400, -1)
        )

        df_shift = df_daily_qm.withColumn(
            "baseline_mean", F.avg("daily_mean").over(window_baseline)
        ).withColumn(
            "baseline_stddev", F.stddev("daily_mean").over(window_baseline)
        ).withColumn(
            "distribution_shift_zscore",
            (F.col("daily_mean") - F.col("baseline_mean"))
            / (F.col("baseline_stddev") + 0.01),
        )

        # Calculate device consistency (variance across devices)
        df_device_variance = (
            df_results.filter(F.col("is_valid") == True)
            .groupBy("loinc_code", "collected_date", "device_id")
            .agg(F.avg("result_value").alias("device_mean"))
            .groupBy("loinc_code", "collected_date")
            .agg(F.variance("device_mean").alias("device_variance"))
        )

        # Join all metrics
        df_gold = df_shift.join(
            df_device_variance, on=["loinc_code", "collected_date"], how="left"
        ).select(
            "loinc_code",
            "test_name",
            "collected_date",
            "total_results",
            F.coalesce("outlier_rate", F.lit(0.0)).alias("outlier_rate"),
            F.coalesce("distribution_shift_zscore", F.lit(0.0)).alias(
                "distribution_shift"
            ),
            F.coalesce("device_variance", F.lit(0.0)).alias(
                "device_consistency_score"
            ),
        )

        # Flag anomalies
        df_gold = df_gold.withColumn(
            "quality_anomaly",
            (F.col("outlier_rate") > 0.05)
            | (F.abs(F.col("distribution_shift")) > 2.0)
            | (F.col("device_consistency_score") > 100),
        )

        # Write to Gold layer
        delta_path = f"{self.gold_path}/quality_anomaly_features"

        df_gold.write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).partitionBy("collected_date").save(delta_path)

        records_written = df_gold.count()
        anomalies_detected = df_gold.filter(F.col("quality_anomaly") == True).count()

        logger.info(f"Quality anomaly features created: {records_written} test-days")
        logger.info(f"Anomalies detected: {anomalies_detected}")

        return {
            "table": "quality_anomaly_features",
            "records_written": records_written,
            "anomalies_detected": anomalies_detected,
        }

    def run_feature_engineering_pipeline(self) -> Dict[str, Dict]:
        """
        Run complete Gold layer feature engineering pipeline.

        Returns:
            Dictionary with statistics for all feature tables
        """
        logger.info("Starting Gold layer feature engineering pipeline")

        results = {}

        # Create device failure features
        results["device_failure"] = self.create_device_failure_features()

        # Create outbreak risk features
        results["outbreak_risk"] = self.create_outbreak_risk_features()

        # Create quality anomaly features
        results["quality_anomaly"] = self.create_quality_anomaly_features()

        logger.info("Gold layer feature engineering pipeline completed")
        return results


# Example usage in Databricks notebook
if __name__ == "__main__":
    # Initialize Spark session
    spark = (
        SparkSession.builder.appName("LIMS Gold Feature Engineering")
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

    # Initialize feature engineering
    feature_eng = GoldLayerFeatureEngineering(spark, SILVER_PATH, GOLD_PATH)

    # Run feature engineering pipeline
    stats = feature_eng.run_feature_engineering_pipeline()

    # Print statistics
    for feature_set, result in stats.items():
        print(f"\n{feature_set}:")
        for key, value in result.items():
            print(f"  {key}: {value}")

    print("\nGold layer feature engineering completed successfully!")
