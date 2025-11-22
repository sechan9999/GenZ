"""
Silver Layer: LIMS Data Standardization Pipeline

This module handles data cleaning, validation, and standardization:
- LOINC code mapping (local test codes → standard codes)
- Unit normalization (convert all units to standard)
- PII hashing (protect patient identifiers)
- Data quality validation

Author: Data Engineering Team
Date: 2025-11-22
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, DoubleType, BooleanType
from pyspark.sql.window import Window
from delta.tables import DeltaTable
from typing import Dict, List, Optional
import logging
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SilverLayerStandardization:
    """
    Handles standardization of Bronze layer data into Silver layer.

    Transformations:
    1. LOINC mapping: Local test codes → standard LOINC codes
    2. Unit normalization: Convert all units to standard (mg/dL, mmol/L)
    3. PII hashing: SHA-256 hash patient_id, technician_id
    4. Data validation: Remove nulls, check ranges, flag anomalies
    5. Deduplication: Remove duplicate records
    """

    def __init__(
        self,
        spark: SparkSession,
        bronze_path: str,
        silver_path: str,
        loinc_mapping_path: Optional[str] = None,
    ):
        """
        Initialize Silver layer standardization.

        Args:
            spark: Active SparkSession
            bronze_path: Base path for Bronze layer Delta tables
            silver_path: Base path for Silver layer Delta tables
            loinc_mapping_path: Path to LOINC mapping reference table
        """
        self.spark = spark
        self.bronze_path = bronze_path
        self.silver_path = silver_path
        self.loinc_mapping_path = loinc_mapping_path

        # Salt for PII hashing (in production, load from Azure Key Vault)
        self.hash_salt = "CHANGE_THIS_IN_PRODUCTION"

        logger.info(f"Initialized SilverLayerStandardization")

    def get_loinc_mapping(self) -> DataFrame:
        """
        Load LOINC mapping table.

        In production, this would be a curated reference table mapping
        local lab test codes to standard LOINC codes.

        Returns:
            DataFrame with columns: local_code, loinc_code, test_name, standard_unit
        """
        if self.loinc_mapping_path:
            # Load from curated mapping table
            df_loinc = self.spark.read.format("delta").load(self.loinc_mapping_path)
        else:
            # Create sample mapping for demonstration
            logger.warning("Using sample LOINC mapping. In production, use curated reference.")

            sample_data = [
                # (local_code, loinc_code, test_name, standard_unit, conversion_factor)
                ("GLU", "2345-7", "Glucose", "mg/dL", 1.0),
                ("GLU_MMOL", "2345-7", "Glucose", "mg/dL", 18.0),  # mmol/L to mg/dL
                ("HBA1C", "4548-4", "Hemoglobin A1c", "%", 1.0),
                ("CHOL", "2093-3", "Cholesterol", "mg/dL", 1.0),
                ("TRIG", "2571-8", "Triglycerides", "mg/dL", 1.0),
                ("HDL", "2085-9", "HDL Cholesterol", "mg/dL", 1.0),
                ("LDL", "2089-1", "LDL Cholesterol", "mg/dL", 1.0),
                ("CREAT", "2160-0", "Creatinine", "mg/dL", 1.0),
                ("BUN", "3094-0", "Blood Urea Nitrogen", "mg/dL", 1.0),
                ("NA", "2951-2", "Sodium", "mmol/L", 1.0),
                ("K", "2823-3", "Potassium", "mmol/L", 1.0),
                ("WBC", "6690-2", "White Blood Cell Count", "10*3/uL", 1.0),
                ("RBC", "789-8", "Red Blood Cell Count", "10*6/uL", 1.0),
                ("HGB", "718-7", "Hemoglobin", "g/dL", 1.0),
                ("PLT", "777-3", "Platelet Count", "10*3/uL", 1.0),
            ]

            df_loinc = self.spark.createDataFrame(
                sample_data,
                schema="local_code STRING, loinc_code STRING, test_name STRING, standard_unit STRING, conversion_factor DOUBLE",
            )

        return df_loinc

    def hash_pii(self, col_name: str) -> F.Column:
        """
        Create SHA-256 hash of PII field with salt.

        Args:
            col_name: Column name to hash

        Returns:
            PySpark column expression for hashed value
        """
        # UDF for SHA-256 hashing with salt
        @F.udf(returnType=StringType())
        def sha256_hash(value: str) -> str:
            if value is None:
                return None
            salted = f"{value}_{self.hash_salt}"
            return hashlib.sha256(salted.encode()).hexdigest()

        return sha256_hash(F.col(col_name))

    def standardize_lab_results(self) -> Dict[str, int]:
        """
        Standardize lab results from Bronze to Silver layer.

        Transformations:
        1. Join with LOINC mapping
        2. Convert units to standard
        3. Hash PII fields
        4. Validate data quality
        5. Flag critical values
        6. Remove duplicates

        Returns:
            Dictionary with processing statistics
        """
        logger.info("Starting lab results standardization")

        # Read Bronze layer
        df_bronze = self.spark.read.format("delta").load(
            f"{self.bronze_path}/lab_results_raw"
        )

        # Get LOINC mapping
        df_loinc = self.get_loinc_mapping()

        # Join with LOINC mapping
        df_mapped = df_bronze.join(
            df_loinc, df_bronze.test_code == df_loinc.local_code, how="left"
        )

        # Standardize units and convert result_value to numeric
        df_standardized = df_mapped.withColumn(
            # Convert result_value to double, handling non-numeric values
            "result_value_numeric",
            F.when(
                F.col("result_value").rlike(r"^\d+\.?\d*$"),
                F.col("result_value").cast(DoubleType()),
            ).otherwise(None),
        ).withColumn(
            # Apply unit conversion
            "result_value_standard",
            F.when(
                F.col("result_value_numeric").isNotNull(),
                F.col("result_value_numeric") * F.col("conversion_factor"),
            ).otherwise(None),
        )

        # Hash PII fields
        df_hashed = (
            df_standardized.withColumn("patient_hash", self.hash_pii("patient_id"))
            .withColumn("technician_hash", self.hash_pii("technician_id"))
            .drop("patient_id", "technician_id")  # Drop original PII
        )

        # Extract date and time features
        df_features = (
            df_hashed.withColumn(
                "collected_date", F.to_date(F.col("collected_timestamp"))
            )
            .withColumn("resulted_date", F.to_date(F.col("resulted_timestamp")))
            .withColumn("collected_hour", F.hour(F.col("collected_timestamp")))
            .withColumn(
                "turnaround_time_hours",
                (
                    F.unix_timestamp("resulted_timestamp")
                    - F.unix_timestamp("collected_timestamp")
                )
                / 3600,
            )
        )

        # Flag critical values (example ranges)
        df_flagged = df_features.withColumn(
            "is_critical",
            F.when(
                (F.col("loinc_code") == "2345-7")
                & (  # Glucose
                    (F.col("result_value_standard") < 60)
                    | (F.col("result_value_standard") > 300)
                ),
                True,
            )
            .when(
                (F.col("loinc_code") == "2823-3")
                & (  # Potassium
                    (F.col("result_value_standard") < 2.5)
                    | (F.col("result_value_standard") > 6.0)
                ),
                True,
            )
            .otherwise(False),
        )

        # Validate data quality
        df_validated = df_flagged.withColumn(
            "is_valid",
            (F.col("result_id").isNotNull())
            & (F.col("patient_hash").isNotNull())
            & (F.col("loinc_code").isNotNull())  # Must have LOINC mapping
            & (F.col("result_value_standard").isNotNull())  # Must be numeric
            & (F.col("device_id").isNotNull())
            & (F.col("collected_timestamp").isNotNull()),
        )

        # Remove duplicates (keep most recent ingestion)
        window_spec = Window.partitionBy("result_id").orderBy(
            F.col("ingestion_timestamp").desc()
        )
        df_deduped = (
            df_validated.withColumn("row_num", F.row_number().over(window_spec))
            .filter(F.col("row_num") == 1)
            .drop("row_num")
        )

        # Select final columns for Silver layer
        df_silver = df_deduped.select(
            "result_id",
            "patient_hash",
            "loinc_code",
            "test_name",
            F.col("result_value_standard").alias("result_value"),
            F.col("standard_unit").alias("result_unit"),
            "device_id",
            "collected_date",
            "resulted_date",
            "collected_hour",
            "turnaround_time_hours",
            "technician_hash",
            "is_critical",
            "is_valid",
            "ingestion_timestamp",
        )

        # Write to Silver layer Delta Lake
        delta_path = f"{self.silver_path}/lab_results_standardized"

        df_silver.write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).partitionBy("collected_date").save(delta_path)

        total_records = df_bronze.count()
        valid_records = df_silver.filter(F.col("is_valid") == True).count()
        invalid_records = df_silver.filter(F.col("is_valid") == False).count()
        critical_records = df_silver.filter(F.col("is_critical") == True).count()

        logger.info(f"Lab results standardization completed:")
        logger.info(f"  Total records: {total_records}")
        logger.info(f"  Valid records: {valid_records}")
        logger.info(f"  Invalid records: {invalid_records}")
        logger.info(f"  Critical values: {critical_records}")

        return {
            "table": "lab_results_standardized",
            "total_records": total_records,
            "valid_records": valid_records,
            "invalid_records": invalid_records,
            "critical_records": critical_records,
            "data_quality_rate": round(valid_records / total_records * 100, 2)
            if total_records > 0
            else 0,
        }

    def standardize_device_metrics(self) -> Dict[str, int]:
        """
        Aggregate device logs into normalized device metrics (Silver layer).

        Creates daily device metrics including:
        - Error counts
        - Warning counts
        - Days since last calibration/maintenance
        - Daily result volume

        Returns:
            Dictionary with processing statistics
        """
        logger.info("Starting device metrics standardization")

        # Read Bronze layer device logs
        df_logs = self.spark.read.format("delta").load(
            f"{self.bronze_path}/device_logs_raw"
        )

        # Extract date from timestamp
        df_logs = df_logs.withColumn("log_date", F.to_date("log_timestamp")).withColumn(
            "log_hour", F.hour("log_timestamp")
        )

        # Aggregate by device and date
        df_metrics = df_logs.groupBy("device_id", "log_date").agg(
            F.sum(F.when(F.col("event_type") == "ERROR", 1).otherwise(0)).alias(
                "error_count"
            ),
            F.sum(F.when(F.col("event_type") == "WARNING", 1).otherwise(0)).alias(
                "warning_count"
            ),
            F.max("calibration_date").alias("last_calibration_date"),
            F.max("maintenance_date").alias("last_maintenance_date"),
        )

        # Calculate days since last calibration/maintenance
        df_metrics = df_metrics.withColumn(
            "calibration_days_since",
            F.datediff(F.col("log_date"), F.col("last_calibration_date")),
        ).withColumn(
            "maintenance_days_since",
            F.datediff(F.col("log_date"), F.col("last_maintenance_date")),
        )

        # Get result volume from lab results
        df_results = self.spark.read.format("delta").load(
            f"{self.silver_path}/lab_results_standardized"
        )

        df_volume = df_results.groupBy("device_id", "collected_date").agg(
            F.count("*").alias("result_volume")
        )

        # Join with volume data
        df_final = df_metrics.join(
            df_volume,
            (df_metrics.device_id == df_volume.device_id)
            & (df_metrics.log_date == df_volume.collected_date),
            how="left",
        ).select(
            df_metrics.device_id,
            df_metrics.log_date,
            df_metrics.error_count,
            df_metrics.warning_count,
            df_metrics.calibration_days_since,
            df_metrics.maintenance_days_since,
            F.coalesce(df_volume.result_volume, F.lit(0)).alias("result_volume"),
        )

        # Write to Silver layer
        delta_path = f"{self.silver_path}/device_metrics_normalized"

        df_final.write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).partitionBy("log_date").save(delta_path)

        records_written = df_final.count()
        logger.info(
            f"Device metrics standardization completed: {records_written} records"
        )

        return {
            "table": "device_metrics_normalized",
            "records_written": records_written,
        }

    def run_standardization_pipeline(self) -> Dict[str, Dict]:
        """
        Run complete Silver layer standardization pipeline.

        Returns:
            Dictionary with statistics for all tables
        """
        logger.info("Starting Silver layer standardization pipeline")

        results = {}

        # Standardize lab results
        results["lab_results"] = self.standardize_lab_results()

        # Standardize device metrics
        results["device_metrics"] = self.standardize_device_metrics()

        # Optimize Silver tables
        self.optimize_silver_tables()

        logger.info("Silver layer standardization pipeline completed")
        return results

    def optimize_silver_tables(self) -> None:
        """Optimize Silver layer Delta tables."""
        logger.info("Starting Silver layer optimization")

        tables = ["lab_results_standardized", "device_metrics_normalized"]

        for table in tables:
            delta_path = f"{self.silver_path}/{table}"
            logger.info(f"Optimizing {table}...")

            self.spark.sql(f"""
                OPTIMIZE delta.`{delta_path}`
                ZORDER BY (device_id, loinc_code)
            """)

            self.spark.sql(f"""
                VACUUM delta.`{delta_path}` RETAIN 720 HOURS
            """)

            logger.info(f"Optimized {table}")

        logger.info("Silver layer optimization completed")


# Example usage in Databricks notebook
if __name__ == "__main__":
    # Initialize Spark session
    spark = (
        SparkSession.builder.appName("LIMS Silver Standardization")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .getOrCreate()
    )

    # Configure paths
    BRONZE_PATH = "/mnt/delta/lims/bronze"
    SILVER_PATH = "/mnt/delta/lims/silver"
    LOINC_MAPPING_PATH = None  # Use sample mapping

    # Initialize standardization
    standardization = SilverLayerStandardization(
        spark, BRONZE_PATH, SILVER_PATH, LOINC_MAPPING_PATH
    )

    # Run standardization pipeline
    stats = standardization.run_standardization_pipeline()

    # Print statistics
    for table, result in stats.items():
        print(f"\n{table}:")
        for key, value in result.items():
            print(f"  {key}: {value}")

    print("\nSilver layer standardization completed successfully!")
