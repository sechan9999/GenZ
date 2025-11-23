"""
FHIR Gold Layer - Clinical Aggregation & ML Feature Engineering

Creates analytics-ready datasets and ML features from normalized FHIR data.

Use Cases:
1. Patient vital sign trends (for clinical dashboards)
2. Medication adherence analysis
3. Chronic disease risk scoring (diabetes, hypertension)
4. Hospital readmission prediction features
5. Lab result anomaly detection

Author: MLOps Healthcare Team
Date: 2025-11-23
"""

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType
from delta import DeltaTable
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FHIRGoldAggregation:
    """
    Create Gold layer analytics tables and ML features.

    Gold Layer Tables:
    - patient_vital_trends: Daily vital sign aggregations
    - patient_medication_summary: Active medication counts
    - chronic_disease_features: Risk scoring features
    - lab_result_trends: Trend analysis for key lab tests
    """

    def __init__(
        self,
        spark: SparkSession,
        silver_path: str,
        gold_path: str,
    ):
        """
        Initialize Gold layer pipeline.

        Args:
            spark: Active SparkSession
            silver_path: Path to Silver layer normalized tables
            gold_path: Path to Gold layer output
        """
        self.spark = spark
        self.silver_path = silver_path
        self.gold_path = gold_path

    def create_patient_vital_trends(self, lookback_days: int = 30) -> None:
        """
        Create daily vital sign trends per patient.

        Vital Signs (LOINC codes):
        - Heart Rate: 8867-4
        - Blood Pressure Systolic: 8480-6
        - Blood Pressure Diastolic: 8462-4
        - Temperature: 8310-5
        - Respiratory Rate: 9279-1
        - Oxygen Saturation: 2708-6

        Output Columns:
        - patient_id_hashed
        - observation_date
        - heart_rate_bpm (avg, min, max)
        - bp_systolic_mmhg (avg, min, max)
        - bp_diastolic_mmhg (avg, min, max)
        - temperature_celsius (avg)
        - respiratory_rate (avg)
        - oxygen_saturation_pct (avg, min)

        Args:
            lookback_days: Process last N days
        """
        logger.info("Creating patient vital sign trends...")

        # Read Silver observations
        df_obs = (
            self.spark.read.format("delta")
            .load(f"{self.silver_path}/observations_normalized")
            .filter(F.col("is_valid") == True)
            .filter(
                F.col("observation_date")
                >= F.date_sub(F.current_date(), lookback_days)
            )
        )

        # Define vital sign LOINC codes
        vital_loinc_codes = {
            "8867-4": "heart_rate",
            "8480-6": "bp_systolic",
            "8462-4": "bp_diastolic",
            "8310-5": "temperature",
            "9279-1": "respiratory_rate",
            "2708-6": "oxygen_saturation",
        }

        # Filter to vital signs only
        df_vitals = df_obs.filter(
            F.col("loinc_code").isin(list(vital_loinc_codes.keys()))
        )

        # Pivot by LOINC code
        df_pivoted = df_vitals.groupBy("patient_id_hashed", "observation_date").pivot(
            "loinc_code", list(vital_loinc_codes.keys())
        ).agg(
            F.avg("value_numeric").alias("avg"),
            F.min("value_numeric").alias("min"),
            F.max("value_numeric").alias("max"),
            F.count("*").alias("count"),
        )

        # Rename columns for clarity
        for loinc_code, vital_name in vital_loinc_codes.items():
            df_pivoted = df_pivoted.withColumnRenamed(
                f"{loinc_code}_avg", f"{vital_name}_avg"
            )
            df_pivoted = df_pivoted.withColumnRenamed(
                f"{loinc_code}_min", f"{vital_name}_min"
            )
            df_pivoted = df_pivoted.withColumnRenamed(
                f"{loinc_code}_max", f"{vital_name}_max"
            )
            df_pivoted = df_pivoted.withColumnRenamed(
                f"{loinc_code}_count", f"{vital_name}_count"
            )

        # Add clinical flags
        df_gold = self._add_vital_sign_flags(df_pivoted)

        # Write to Delta Lake
        record_count = df_gold.count()
        logger.info(f"Creating {record_count} patient-day vital trend records")

        df_gold.write.format("delta").mode("overwrite").partitionBy(
            "observation_date"
        ).save(f"{self.gold_path}/patient_vital_trends")

        logger.info("✓ Patient vital trends created")

    def create_lab_result_trends(self, lookback_days: int = 90) -> None:
        """
        Create lab result trends for key tests.

        Key Lab Tests (LOINC codes):
        - Glucose: 2345-7
        - HbA1c: 4548-4
        - Creatinine: 2160-0
        - Total Cholesterol: 2093-3
        - LDL: 2089-1
        - HDL: 2085-9
        - Hemoglobin: 718-7

        Features:
        - Current value
        - 7-day avg, 30-day avg, 90-day avg
        - Trend direction (increasing/decreasing)
        - Days since last abnormal result

        Args:
            lookback_days: Historical window
        """
        logger.info("Creating lab result trends...")

        df_obs = (
            self.spark.read.format("delta")
            .load(f"{self.silver_path}/observations_normalized")
            .filter(F.col("is_valid") == True)
            .filter(
                F.col("observation_date")
                >= F.date_sub(F.current_date(), lookback_days)
            )
        )

        # Key lab tests
        key_labs = {
            "2345-7": "glucose",
            "4548-4": "hba1c",
            "2160-0": "creatinine",
            "2093-3": "total_cholesterol",
            "2089-1": "ldl",
            "2085-9": "hdl",
            "718-7": "hemoglobin",
        }

        df_labs = df_obs.filter(F.col("loinc_code").isin(list(key_labs.keys())))

        # Window specs for trend calculation
        window_7d = (
            Window.partitionBy("patient_id_hashed", "loinc_code")
            .orderBy(F.col("observation_date").cast("long"))
            .rangeBetween(-7 * 86400, 0)
        )

        window_30d = (
            Window.partitionBy("patient_id_hashed", "loinc_code")
            .orderBy(F.col("observation_date").cast("long"))
            .rangeBetween(-30 * 86400, 0)
        )

        window_90d = (
            Window.partitionBy("patient_id_hashed", "loinc_code")
            .orderBy(F.col("observation_date").cast("long"))
            .rangeBetween(-90 * 86400, 0)
        )

        # Calculate rolling averages
        df_trends = (
            df_labs.withColumn("value_7d_avg", F.avg("value_numeric").over(window_7d))
            .withColumn("value_30d_avg", F.avg("value_numeric").over(window_30d))
            .withColumn("value_90d_avg", F.avg("value_numeric").over(window_90d))
            .withColumn("value_7d_min", F.min("value_numeric").over(window_7d))
            .withColumn("value_7d_max", F.max("value_numeric").over(window_7d))
        )

        # Calculate trend direction (7d vs 30d average)
        df_trends = df_trends.withColumn(
            "trend_direction",
            F.when(
                F.col("value_7d_avg") > F.col("value_30d_avg") * 1.05, "INCREASING"
            )
            .when(F.col("value_7d_avg") < F.col("value_30d_avg") * 0.95, "DECREASING")
            .otherwise("STABLE"),
        )

        # Percent change
        df_trends = df_trends.withColumn(
            "pct_change_7d_vs_30d",
            ((F.col("value_7d_avg") - F.col("value_30d_avg")) / F.col("value_30d_avg"))
            * 100,
        )

        # Days since last abnormal result
        window_all = Window.partitionBy("patient_id_hashed", "loinc_code").orderBy(
            "observation_date"
        )

        df_trends = df_trends.withColumn(
            "days_since_abnormal",
            F.when(
                F.col("is_abnormal"),
                0,
            ).otherwise(
                F.datediff(
                    F.current_date(),
                    F.last(
                        F.when(F.col("is_abnormal"), F.col("observation_date")),
                        ignorenulls=True,
                    ).over(window_all),
                )
            ),
        )

        # Select final columns
        df_gold = df_trends.select(
            "patient_id_hashed",
            "observation_date",
            "loinc_code",
            "loinc_display",
            "value_numeric",
            "value_unit",
            "value_7d_avg",
            "value_30d_avg",
            "value_90d_avg",
            "value_7d_min",
            "value_7d_max",
            "trend_direction",
            "pct_change_7d_vs_30d",
            "is_abnormal",
            "abnormal_severity",
            "days_since_abnormal",
        )

        record_count = df_gold.count()
        logger.info(f"Creating {record_count} lab result trend records")

        df_gold.write.format("delta").mode("overwrite").partitionBy(
            "observation_date"
        ).save(f"{self.gold_path}/lab_result_trends")

        logger.info("✓ Lab result trends created")

    def create_chronic_disease_features(self) -> None:
        """
        Create ML features for chronic disease risk scoring.

        Chronic Diseases:
        1. Diabetes Risk
        2. Hypertension Risk
        3. Cardiovascular Disease Risk

        Features per patient:
        - Demographics (age, gender)
        - Recent vital signs (avg last 30 days)
        - Recent lab results (avg last 90 days)
        - Medication count (active chronic disease meds)
        - Encounter frequency (last 6 months)

        Output: One row per patient with 50+ features
        """
        logger.info("Creating chronic disease risk features...")

        # 1. Get patient demographics
        df_patients = self.spark.read.format("delta").load(
            f"{self.silver_path}/patients_normalized"
        )

        # 2. Get recent vital signs (last 30 days avg)
        df_vitals_recent = (
            self.spark.read.format("delta")
            .load(f"{self.gold_path}/patient_vital_trends")
            .filter(
                F.col("observation_date")
                >= F.date_sub(F.current_date(), 30)
            )
            .groupBy("patient_id_hashed")
            .agg(
                F.avg("heart_rate_avg").alias("avg_heart_rate_30d"),
                F.avg("bp_systolic_avg").alias("avg_bp_systolic_30d"),
                F.avg("bp_diastolic_avg").alias("avg_bp_diastolic_30d"),
                F.max("bp_systolic_max").alias("max_bp_systolic_30d"),
                F.max("bp_diastolic_max").alias("max_bp_diastolic_30d"),
            )
        )

        # 3. Get recent lab results (last 90 days avg)
        df_labs_recent = (
            self.spark.read.format("delta")
            .load(f"{self.gold_path}/lab_result_trends")
            .filter(
                F.col("observation_date")
                >= F.date_sub(F.current_date(), 90)
            )
        )

        # Pivot key labs
        df_labs_pivot = df_labs_recent.groupBy("patient_id_hashed").pivot(
            "loinc_code", ["2345-7", "4548-4", "2160-0", "2093-3", "2089-1", "2085-9"]
        ).agg(
            F.avg("value_numeric").alias("avg"),
            F.max("is_abnormal").alias("has_abnormal"),
        )

        # Rename for clarity
        df_labs_pivot = (
            df_labs_pivot.withColumnRenamed("2345-7_avg", "avg_glucose_90d")
            .withColumnRenamed("4548-4_avg", "avg_hba1c_90d")
            .withColumnRenamed("2160-0_avg", "avg_creatinine_90d")
            .withColumnRenamed("2093-3_avg", "avg_total_chol_90d")
            .withColumnRenamed("2089-1_avg", "avg_ldl_90d")
            .withColumnRenamed("2085-9_avg", "avg_hdl_90d")
            .withColumnRenamed("2345-7_has_abnormal", "has_abnormal_glucose")
            .withColumnRenamed("4548-4_has_abnormal", "has_abnormal_hba1c")
        )

        # 4. Get active medication count
        df_meds_active = (
            self.spark.read.format("delta")
            .load(f"{self.silver_path}/medications_normalized")
            .filter(F.col("is_active") == True)
            .groupBy("patient_id_hashed")
            .agg(F.count("*").alias("active_medication_count"))
        )

        # 5. Get encounter frequency (last 6 months)
        df_encounters_recent = (
            self.spark.read.format("delta")
            .load(f"{self.silver_path}/encounters_normalized")
            .filter(
                F.col("encounter_date")
                >= F.date_sub(F.current_date(), 180)
            )
            .groupBy("patient_id_hashed")
            .agg(
                F.count("*").alias("encounter_count_6m"),
                F.avg("duration_hours").alias("avg_encounter_duration_hours"),
            )
        )

        # Join all features
        df_features = (
            df_patients.join(df_vitals_recent, "patient_id_hashed", "left")
            .join(df_labs_pivot, "patient_id_hashed", "left")
            .join(df_meds_active, "patient_id_hashed", "left")
            .join(df_encounters_recent, "patient_id_hashed", "left")
        )

        # Fill nulls
        df_features = df_features.fillna(0, subset=["active_medication_count", "encounter_count_6m"])

        # Add derived risk flags
        df_features = self._add_chronic_disease_flags(df_features)

        # Select final feature columns
        feature_columns = [
            "patient_id_hashed",
            "age_years",
            "gender",
            "avg_heart_rate_30d",
            "avg_bp_systolic_30d",
            "avg_bp_diastolic_30d",
            "max_bp_systolic_30d",
            "max_bp_diastolic_30d",
            "avg_glucose_90d",
            "avg_hba1c_90d",
            "avg_creatinine_90d",
            "avg_total_chol_90d",
            "avg_ldl_90d",
            "avg_hdl_90d",
            "has_abnormal_glucose",
            "has_abnormal_hba1c",
            "active_medication_count",
            "encounter_count_6m",
            "avg_encounter_duration_hours",
            "diabetes_risk_flag",
            "hypertension_risk_flag",
            "cvd_risk_flag",
            F.current_date().alias("feature_date"),
        ]

        df_gold = df_features.select(*feature_columns)

        record_count = df_gold.count()
        logger.info(f"Creating {record_count} patient chronic disease feature records")

        df_gold.write.format("delta").mode("overwrite").save(
            f"{self.gold_path}/chronic_disease_features"
        )

        logger.info("✓ Chronic disease features created")

    def create_medication_adherence_features(self, lookback_days: int = 90) -> None:
        """
        Create medication adherence analysis.

        Features:
        - Active medication count
        - Discontinued medication count
        - Medication start/stop frequency
        - Average medication duration

        Args:
            lookback_days: Analysis window
        """
        logger.info("Creating medication adherence features...")

        df_meds = (
            self.spark.read.format("delta")
            .load(f"{self.silver_path}/medications_normalized")
            .filter(
                F.col("start_date")
                >= F.date_sub(F.current_date(), lookback_days)
            )
        )

        df_adherence = (
            df_meds.groupBy("patient_id_hashed")
            .agg(
                F.sum(F.when(F.col("is_active"), 1).otherwise(0)).alias(
                    "active_medication_count"
                ),
                F.sum(F.when(F.col("status") == "stopped", 1).otherwise(0)).alias(
                    "stopped_medication_count"
                ),
                F.avg("duration_days").alias("avg_medication_duration_days"),
                F.countDistinct("rxnorm_code").alias("unique_medications"),
            )
        )

        # Adherence score (simple heuristic)
        df_adherence = df_adherence.withColumn(
            "adherence_score",
            F.when(
                F.col("avg_medication_duration_days") > 60, 100
            ).when(F.col("avg_medication_duration_days") > 30, 75).otherwise(50),
        )

        df_adherence = df_adherence.withColumn(
            "feature_date", F.current_date()
        )

        record_count = df_adherence.count()
        logger.info(f"Creating {record_count} medication adherence records")

        df_adherence.write.format("delta").mode("overwrite").save(
            f"{self.gold_path}/medication_adherence_features"
        )

        logger.info("✓ Medication adherence features created")

    def _add_vital_sign_flags(self, df: DataFrame) -> DataFrame:
        """Add clinical flags for abnormal vital signs."""

        # Hypertension: BP >= 140/90
        df = df.withColumn(
            "hypertension_flag",
            (F.col("bp_systolic_avg") >= 140) | (F.col("bp_diastolic_avg") >= 90),
        )

        # Tachycardia: HR > 100
        df = df.withColumn("tachycardia_flag", F.col("heart_rate_avg") > 100)

        # Hypoxia: SpO2 < 90%
        df = df.withColumn("hypoxia_flag", F.col("oxygen_saturation_min") < 90)

        return df

    def _add_chronic_disease_flags(self, df: DataFrame) -> DataFrame:
        """Add chronic disease risk flags based on clinical criteria."""

        # Diabetes risk: HbA1c >= 6.5% or Glucose >= 126 mg/dL
        df = df.withColumn(
            "diabetes_risk_flag",
            (F.col("avg_hba1c_90d") >= 6.5) | (F.col("avg_glucose_90d") >= 126),
        )

        # Hypertension risk: BP >= 140/90
        df = df.withColumn(
            "hypertension_risk_flag",
            (F.col("avg_bp_systolic_30d") >= 140) | (F.col("avg_bp_diastolic_30d") >= 90),
        )

        # CVD risk: LDL >= 160 or multiple risk factors
        df = df.withColumn(
            "cvd_risk_flag",
            (F.col("avg_ldl_90d") >= 160)
            | (
                (F.col("avg_total_chol_90d") >= 240)
                & (F.col("hypertension_risk_flag") == True)
            ),
        )

        return df

    def run_all_aggregations(self, lookback_days: int = 30) -> None:
        """Run all Gold layer aggregations."""
        logger.info("=== Starting FHIR Gold Aggregation ===")

        self.create_patient_vital_trends(lookback_days)
        self.create_lab_result_trends(lookback_days=90)
        self.create_chronic_disease_features()
        self.create_medication_adherence_features(lookback_days=90)

        logger.info("=== FHIR Gold Aggregation Complete ===")


if __name__ == "__main__":
    spark = (
        SparkSession.builder.appName("FHIR Gold Aggregation")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .getOrCreate()
    )

    SILVER_PATH = "/mnt/delta/fhir/silver"
    GOLD_PATH = "/mnt/delta/fhir/gold"

    aggregation = FHIRGoldAggregation(
        spark=spark, silver_path=SILVER_PATH, gold_path=GOLD_PATH
    )

    # Run all aggregations
    aggregation.run_all_aggregations(lookback_days=30)

    logger.info("✓ FHIR Gold pipeline complete")
