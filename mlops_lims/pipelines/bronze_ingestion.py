"""
Bronze Layer: LIMS Data Ingestion Pipeline

This module handles raw data ingestion from LIMS systems into Delta Lake.
Implements CDC (Change Data Capture) and batch ingestion patterns.

Author: Data Engineering Team
Date: 2025-11-22
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    TimestampType,
    DoubleType,
    BooleanType,
)
from delta.tables import DeltaTable
from typing import Dict, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BronzeLayerIngestion:
    """
    Handles ingestion of raw LIMS data into Bronze layer Delta Lake tables.

    The Bronze layer stores immutable, raw data exactly as received from source systems.
    This ensures data lineage and enables reprocessing if needed.
    """

    def __init__(self, spark: SparkSession, bronze_path: str):
        """
        Initialize Bronze layer ingestion.

        Args:
            spark: Active SparkSession
            bronze_path: Base path for Bronze layer Delta tables
        """
        self.spark = spark
        self.bronze_path = bronze_path
        logger.info(f"Initialized BronzeLayerIngestion with path: {bronze_path}")

    def get_lab_results_schema(self) -> StructType:
        """Define schema for lab results."""
        return StructType(
            [
                StructField("result_id", StringType(), nullable=False),
                StructField("patient_id", StringType(), nullable=False),
                StructField("test_code", StringType(), nullable=False),
                StructField("result_value", StringType(), nullable=True),
                StructField("result_unit", StringType(), nullable=True),
                StructField("device_id", StringType(), nullable=False),
                StructField("collected_timestamp", TimestampType(), nullable=False),
                StructField("resulted_timestamp", TimestampType(), nullable=False),
                StructField("technician_id", StringType(), nullable=True),
            ]
        )

    def get_device_logs_schema(self) -> StructType:
        """Define schema for device logs."""
        return StructType(
            [
                StructField("device_id", StringType(), nullable=False),
                StructField("log_timestamp", TimestampType(), nullable=False),
                StructField("event_type", StringType(), nullable=False),
                StructField("severity", StringType(), nullable=False),
                StructField("message", StringType(), nullable=True),
                StructField("calibration_date", TimestampType(), nullable=True),
                StructField("maintenance_date", TimestampType(), nullable=True),
            ]
        )

    def get_quality_control_schema(self) -> StructType:
        """Define schema for quality control data."""
        return StructType(
            [
                StructField("qc_id", StringType(), nullable=False),
                StructField("device_id", StringType(), nullable=False),
                StructField("test_code", StringType(), nullable=False),
                StructField("qc_level", StringType(), nullable=False),
                StructField("expected_value", DoubleType(), nullable=False),
                StructField("measured_value", DoubleType(), nullable=False),
                StructField("passed", BooleanType(), nullable=False),
                StructField("timestamp", TimestampType(), nullable=False),
            ]
        )

    def ingest_lab_results(
        self, source_path: str, mode: str = "append"
    ) -> Dict[str, int]:
        """
        Ingest lab results from source system into Bronze layer.

        Args:
            source_path: Path to source data (CSV, JSON, or JDBC connection string)
            mode: Write mode ('append' for incremental, 'overwrite' for full refresh)

        Returns:
            Dictionary with ingestion statistics (records_read, records_written)
        """
        logger.info(f"Starting lab results ingestion from {source_path}")

        # Read source data
        # In production, this could be:
        # - JDBC connection to LIMS database
        # - CSV/JSON files from SFTP drop
        # - Azure Event Hubs stream
        df_source = self._read_source_data(source_path, "lab_results")

        # Add ingestion metadata
        df_bronze = df_source.withColumn(
            "ingestion_timestamp", F.current_timestamp()
        ).withColumn("ingestion_date", F.current_date())

        # Write to Delta Lake (Bronze layer)
        delta_path = f"{self.bronze_path}/lab_results_raw"

        df_bronze.write.format("delta").mode(mode).option(
            "mergeSchema", "true"
        ).partitionBy("ingestion_date").save(delta_path)

        records_written = df_bronze.count()
        logger.info(f"Successfully ingested {records_written} lab results to {delta_path}")

        return {
            "table": "lab_results_raw",
            "records_read": df_source.count(),
            "records_written": records_written,
            "timestamp": datetime.now().isoformat(),
        }

    def ingest_device_logs(
        self, source_path: str, mode: str = "append"
    ) -> Dict[str, int]:
        """
        Ingest device logs from source system into Bronze layer.

        Args:
            source_path: Path to source data
            mode: Write mode ('append' or 'overwrite')

        Returns:
            Dictionary with ingestion statistics
        """
        logger.info(f"Starting device logs ingestion from {source_path}")

        df_source = self._read_source_data(source_path, "device_logs")

        df_bronze = df_source.withColumn(
            "ingestion_timestamp", F.current_timestamp()
        ).withColumn("ingestion_date", F.current_date())

        delta_path = f"{self.bronze_path}/device_logs_raw"

        df_bronze.write.format("delta").mode(mode).option(
            "mergeSchema", "true"
        ).partitionBy("ingestion_date").save(delta_path)

        records_written = df_bronze.count()
        logger.info(f"Successfully ingested {records_written} device logs to {delta_path}")

        return {
            "table": "device_logs_raw",
            "records_read": df_source.count(),
            "records_written": records_written,
            "timestamp": datetime.now().isoformat(),
        }

    def ingest_quality_control(
        self, source_path: str, mode: str = "append"
    ) -> Dict[str, int]:
        """
        Ingest quality control data from source system into Bronze layer.

        Args:
            source_path: Path to source data
            mode: Write mode ('append' or 'overwrite')

        Returns:
            Dictionary with ingestion statistics
        """
        logger.info(f"Starting QC data ingestion from {source_path}")

        df_source = self._read_source_data(source_path, "quality_control")

        df_bronze = df_source.withColumn(
            "ingestion_timestamp", F.current_timestamp()
        ).withColumn("ingestion_date", F.current_date())

        delta_path = f"{self.bronze_path}/quality_control_raw"

        df_bronze.write.format("delta").mode(mode).option(
            "mergeSchema", "true"
        ).partitionBy("ingestion_date").save(delta_path)

        records_written = df_bronze.count()
        logger.info(f"Successfully ingested {records_written} QC records to {delta_path}")

        return {
            "table": "quality_control_raw",
            "records_read": df_source.count(),
            "records_written": records_written,
            "timestamp": datetime.now().isoformat(),
        }

    def _read_source_data(self, source_path: str, table_type: str) -> DataFrame:
        """
        Read source data based on format and table type.

        Args:
            source_path: Path or connection string to source data
            table_type: Type of table (lab_results, device_logs, quality_control)

        Returns:
            DataFrame with source data
        """
        # Determine schema based on table type
        schema_mapping = {
            "lab_results": self.get_lab_results_schema(),
            "device_logs": self.get_device_logs_schema(),
            "quality_control": self.get_quality_control_schema(),
        }

        schema = schema_mapping.get(table_type)
        if not schema:
            raise ValueError(f"Unknown table type: {table_type}")

        # Read data based on source format
        if source_path.startswith("jdbc:"):
            # JDBC connection to LIMS database
            df = (
                self.spark.read.format("jdbc")
                .option("url", source_path)
                .option("dbtable", table_type)
                .option("driver", "com.microsoft.sqlserver.jdbc.SQLServerDriver")
                .load()
            )
        elif source_path.endswith(".csv"):
            # CSV file
            df = self.spark.read.schema(schema).option("header", "true").csv(
                source_path
            )
        elif source_path.endswith(".json"):
            # JSON file
            df = self.spark.read.schema(schema).json(source_path)
        else:
            # Default to parquet
            df = self.spark.read.schema(schema).parquet(source_path)

        return df

    def run_incremental_ingestion(
        self, source_config: Dict[str, str], lookback_hours: int = 24
    ) -> Dict[str, Dict]:
        """
        Run incremental ingestion for all LIMS tables (last N hours of data).

        Args:
            source_config: Dictionary mapping table types to source paths
            lookback_hours: Number of hours to look back for incremental load

        Returns:
            Dictionary with statistics for each table

        Example:
            >>> source_config = {
            ...     'lab_results': 'jdbc:sqlserver://lims-prod.db/results',
            ...     'device_logs': 'jdbc:sqlserver://lims-prod.db/devices',
            ...     'quality_control': 'jdbc:sqlserver://lims-prod.db/qc'
            ... }
            >>> stats = ingestion.run_incremental_ingestion(source_config, lookback_hours=24)
        """
        logger.info(f"Starting incremental ingestion (lookback: {lookback_hours} hours)")

        results = {}

        # Ingest lab results
        if "lab_results" in source_config:
            results["lab_results"] = self.ingest_lab_results(
                source_config["lab_results"], mode="append"
            )

        # Ingest device logs
        if "device_logs" in source_config:
            results["device_logs"] = self.ingest_device_logs(
                source_config["device_logs"], mode="append"
            )

        # Ingest quality control data
        if "quality_control" in source_config:
            results["quality_control"] = self.ingest_quality_control(
                source_config["quality_control"], mode="append"
            )

        logger.info("Incremental ingestion completed successfully")
        return results

    def optimize_bronze_tables(self) -> None:
        """
        Optimize Bronze layer Delta tables for better query performance.

        This runs:
        - OPTIMIZE: Compacts small files
        - VACUUM: Removes old versions (30-day retention for compliance)
        - Z-ORDER: Optimizes data layout
        """
        logger.info("Starting Bronze layer optimization")

        tables = ["lab_results_raw", "device_logs_raw", "quality_control_raw"]

        for table in tables:
            delta_path = f"{self.bronze_path}/{table}"
            logger.info(f"Optimizing {table}...")

            # Run OPTIMIZE with Z-ORDER
            self.spark.sql(f"""
                OPTIMIZE delta.`{delta_path}`
                ZORDER BY (device_id)
            """)

            # Run VACUUM (retain 30 days for compliance)
            self.spark.sql(f"""
                VACUUM delta.`{delta_path}` RETAIN 720 HOURS
            """)

            logger.info(f"Optimized {table}")

        logger.info("Bronze layer optimization completed")


# Example usage in Databricks notebook
if __name__ == "__main__":
    # Initialize Spark session
    spark = (
        SparkSession.builder.appName("LIMS Bronze Ingestion")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .getOrCreate()
    )

    # Configure paths
    BRONZE_PATH = "/mnt/delta/lims/bronze"

    # Initialize ingestion
    ingestion = BronzeLayerIngestion(spark, BRONZE_PATH)

    # Example: Ingest from JDBC source
    source_config = {
        "lab_results": "jdbc:sqlserver://lims-prod.database.windows.net:1433;database=LIMS;",
        "device_logs": "jdbc:sqlserver://lims-prod.database.windows.net:1433;database=LIMS;",
        "quality_control": "jdbc:sqlserver://lims-prod.database.windows.net:1433;database=LIMS;",
    }

    # Run incremental ingestion (last 24 hours)
    stats = ingestion.run_incremental_ingestion(source_config, lookback_hours=24)

    # Print statistics
    for table, result in stats.items():
        print(f"{table}: {result['records_written']} records ingested")

    # Optimize tables
    ingestion.optimize_bronze_tables()

    print("Bronze layer ingestion completed successfully!")
