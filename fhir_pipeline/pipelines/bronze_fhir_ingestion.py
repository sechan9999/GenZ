"""
FHIR Bronze Layer Ingestion Pipeline

Reads FHIR JSON streams from Azure Event Hubs or Azure Blob Storage
and writes raw data to Delta Lake Bronze layer.

Supported FHIR Resources:
- Observation (Lab results, vital signs)
- MedicationStatement (Current and historical medications)
- Patient (Demographics)
- Encounter (Clinical visits)

Author: MLOps Healthcare Team
Date: 2025-11-23
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    TimestampType,
    ArrayType,
    MapType,
)
from delta import DeltaTable
from typing import Dict, List, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FHIRBronzeIngestion:
    """
    Ingest FHIR JSON streams into Delta Lake Bronze layer.

    Data Sources:
    - Azure Event Hubs (real-time streaming)
    - Azure Blob Storage (batch files)
    - Azure Data Lake Gen2 (landing zone)

    Output:
    - Delta Lake Bronze tables (one per FHIR resource type)
    - Full FHIR JSON preserved
    - Ingestion metadata added
    """

    def __init__(
        self,
        spark: SparkSession,
        bronze_path: str,
        checkpoint_path: str,
    ):
        """
        Initialize FHIR Bronze ingestion pipeline.

        Args:
            spark: Active SparkSession with Delta Lake configured
            bronze_path: Base path for Bronze layer Delta tables
            checkpoint_path: Path for streaming checkpoints
        """
        self.spark = spark
        self.bronze_path = bronze_path
        self.checkpoint_path = checkpoint_path

    def ingest_from_event_hub(
        self,
        connection_string: str,
        event_hub_name: str,
        consumer_group: str = "$Default",
        starting_position: str = "latest",
    ) -> None:
        """
        Ingest FHIR streams from Azure Event Hubs (real-time).

        Architecture:
        Event Hub → Spark Structured Streaming → Delta Lake Bronze

        Args:
            connection_string: Event Hub connection string
            event_hub_name: Name of the Event Hub
            consumer_group: Consumer group name
            starting_position: 'earliest' or 'latest'
        """
        logger.info(f"Starting Event Hub ingestion: {event_hub_name}")

        # Event Hub configuration
        eh_conf = {
            "eventhubs.connectionString": connection_string,
            "eventhubs.consumerGroup": consumer_group,
            "eventhubs.startingPosition": starting_position,
        }

        # Read from Event Hub
        df_stream = (
            self.spark.readStream.format("eventhubs")
            .options(**eh_conf)
            .load()
            .select(
                F.col("body").cast("string").alias("fhir_json"),
                F.col("enqueuedTime").alias("event_hub_timestamp"),
                F.col("offset").alias("event_hub_offset"),
                F.col("sequenceNumber").alias("event_hub_sequence"),
                F.col("properties").alias("event_hub_properties"),
            )
        )

        # Parse FHIR resource type from JSON
        df_stream = df_stream.withColumn(
            "resource_type",
            F.get_json_object(F.col("fhir_json"), "$.resourceType"),
        )

        # Add ingestion metadata
        df_stream = self._add_ingestion_metadata(df_stream)

        # Write to Delta Lake (partitioned by resource type and date)
        query = (
            df_stream.writeStream.format("delta")
            .outputMode("append")
            .option(
                "checkpointLocation",
                f"{self.checkpoint_path}/event_hub_bronze",
            )
            .partitionBy("ingestion_date", "resource_type")
            .trigger(processingTime="30 seconds")  # Micro-batch every 30 seconds
            .start(f"{self.bronze_path}/fhir_raw")
        )

        logger.info("Event Hub streaming query started")
        logger.info(f"Checkpoint: {self.checkpoint_path}/event_hub_bronze")
        logger.info(f"Output: {self.bronze_path}/fhir_raw")

        # Return query handle for monitoring
        return query

    def ingest_from_blob_storage(
        self,
        storage_account: str,
        container_name: str,
        folder_path: str,
        file_format: str = "json",
        mode: str = "append",
    ) -> None:
        """
        Ingest FHIR files from Azure Blob Storage (batch).

        Use Case:
        - Daily bulk extracts from EHR systems
        - Historical data backfill
        - Batch reconciliation

        Args:
            storage_account: Azure storage account name
            container_name: Blob container name
            folder_path: Path to FHIR JSON files
            file_format: 'json' or 'ndjson' (newline-delimited JSON)
            mode: 'append' or 'overwrite'
        """
        logger.info(f"Starting Blob Storage ingestion: {folder_path}")

        # Construct blob path
        blob_path = (
            f"wasbs://{container_name}@{storage_account}.blob.core.windows.net/{folder_path}"
        )

        logger.info(f"Reading from: {blob_path}")

        # Read FHIR JSON files
        if file_format == "ndjson":
            # Newline-delimited JSON (one FHIR resource per line)
            df_raw = (
                self.spark.read.text(blob_path)
                .withColumnRenamed("value", "fhir_json")
            )
        else:
            # Standard JSON array
            df_raw = (
                self.spark.read.option("multiLine", "true")
                .json(blob_path)
                .select(F.to_json(F.struct("*")).alias("fhir_json"))
            )

        # Parse resource type
        df_raw = df_raw.withColumn(
            "resource_type",
            F.get_json_object(F.col("fhir_json"), "$.resourceType"),
        )

        # Add ingestion metadata
        df_bronze = self._add_ingestion_metadata(df_raw)

        # Add source metadata
        df_bronze = df_bronze.withColumn("source_type", F.lit("blob_storage"))
        df_bronze = df_bronze.withColumn("source_path", F.lit(blob_path))

        # Write to Delta Lake
        record_count = df_bronze.count()
        logger.info(f"Ingesting {record_count} FHIR resources")

        df_bronze.write.format("delta").mode(mode).partitionBy(
            "ingestion_date", "resource_type"
        ).save(f"{self.bronze_path}/fhir_raw")

        logger.info(f"✓ Ingested {record_count} records to Delta Lake")

    def ingest_from_adls_gen2(
        self,
        storage_account: str,
        file_system: str,
        directory_path: str,
        mode: str = "append",
    ) -> None:
        """
        Ingest FHIR files from Azure Data Lake Storage Gen2.

        Recommended for:
        - Large-scale data lakes
        - Hierarchical namespace benefits
        - Integration with Azure Synapse

        Args:
            storage_account: ADLS Gen2 storage account
            file_system: File system (container) name
            directory_path: Path to FHIR JSON files
            mode: 'append' or 'overwrite'
        """
        logger.info(f"Starting ADLS Gen2 ingestion: {directory_path}")

        # Construct ADLS Gen2 path
        adls_path = (
            f"abfss://{file_system}@{storage_account}.dfs.core.windows.net/{directory_path}"
        )

        logger.info(f"Reading from: {adls_path}")

        # Read FHIR JSON files (supports nested directories)
        df_raw = (
            self.spark.read.option("recursiveFileLookup", "true")
            .option("multiLine", "true")
            .json(adls_path)
            .select(F.to_json(F.struct("*")).alias("fhir_json"))
        )

        # Parse resource type
        df_raw = df_raw.withColumn(
            "resource_type",
            F.get_json_object(F.col("fhir_json"), "$.resourceType"),
        )

        # Add ingestion metadata
        df_bronze = self._add_ingestion_metadata(df_raw)
        df_bronze = df_bronze.withColumn("source_type", F.lit("adls_gen2"))
        df_bronze = df_bronze.withColumn("source_path", F.lit(adls_path))

        # Write to Delta Lake
        record_count = df_bronze.count()
        logger.info(f"Ingesting {record_count} FHIR resources")

        df_bronze.write.format("delta").mode(mode).partitionBy(
            "ingestion_date", "resource_type"
        ).save(f"{self.bronze_path}/fhir_raw")

        logger.info(f"✓ Ingested {record_count} records to Delta Lake")

    def _add_ingestion_metadata(self, df: DataFrame) -> DataFrame:
        """
        Add ingestion metadata columns.

        Metadata:
        - ingestion_timestamp: When record was ingested
        - ingestion_date: Partition key (YYYY-MM-DD)
        - pipeline_run_id: Unique ID for this pipeline run

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with metadata columns
        """
        pipeline_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        df_with_metadata = (
            df.withColumn("ingestion_timestamp", F.current_timestamp())
            .withColumn(
                "ingestion_date",
                F.date_format(F.current_timestamp(), "yyyy-MM-dd"),
            )
            .withColumn("pipeline_run_id", F.lit(pipeline_run_id))
        )

        return df_with_metadata

    def create_resource_specific_tables(self) -> None:
        """
        Create resource-specific views/tables from raw FHIR data.

        Creates separate Delta tables for:
        - Observation (lab results, vitals)
        - MedicationStatement (medications)
        - Patient (demographics)
        - Encounter (visits)

        Benefits:
        - Faster queries (no resource_type filter needed)
        - Better partitioning strategies per resource
        - Easier schema evolution
        """
        logger.info("Creating resource-specific Bronze tables...")

        # Read raw FHIR data
        df_raw = self.spark.read.format("delta").load(f"{self.bronze_path}/fhir_raw")

        resource_types = ["Observation", "MedicationStatement", "Patient", "Encounter"]

        for resource_type in resource_types:
            logger.info(f"Processing {resource_type}...")

            # Filter by resource type
            df_resource = df_raw.filter(F.col("resource_type") == resource_type)

            # Write to resource-specific table
            output_path = f"{self.bronze_path}/{resource_type.lower()}_raw"

            record_count = df_resource.count()
            if record_count > 0:
                df_resource.write.format("delta").mode("overwrite").partitionBy(
                    "ingestion_date"
                ).save(output_path)

                logger.info(f"✓ Created {resource_type} table: {record_count} records")
            else:
                logger.warning(f"⚠ No {resource_type} records found")

    def optimize_bronze_tables(self) -> None:
        """
        Optimize Bronze layer Delta tables.

        Operations:
        - OPTIMIZE: Compact small files
        - VACUUM: Remove old files (retention: 7 days)
        - Z-ORDER: Optimize by resource_type for queries

        Run this periodically (e.g., daily) to maintain performance.
        """
        logger.info("Optimizing Bronze layer tables...")

        # Optimize main raw table
        logger.info("Optimizing fhir_raw table...")
        self.spark.sql(
            f"""
            OPTIMIZE delta.`{self.bronze_path}/fhir_raw`
            ZORDER BY (resource_type)
        """
        )

        # Vacuum old files (retain 7 days)
        logger.info("Vacuuming old files (7 day retention)...")
        self.spark.sql(
            f"""
            VACUUM delta.`{self.bronze_path}/fhir_raw` RETAIN 168 HOURS
        """
        )

        logger.info("✓ Bronze layer optimization complete")


# Example usage in Databricks notebook or Azure Synapse
if __name__ == "__main__":
    # Initialize Spark with Delta Lake
    spark = (
        SparkSession.builder.appName("FHIR Bronze Ingestion")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.databricks.delta.properties.defaults.enableChangeDataFeed", "true")
        .getOrCreate()
    )

    # Configure paths
    BRONZE_PATH = "/mnt/delta/fhir/bronze"
    CHECKPOINT_PATH = "/mnt/delta/fhir/checkpoints"

    # Initialize ingestion pipeline
    ingestion = FHIRBronzeIngestion(
        spark=spark,
        bronze_path=BRONZE_PATH,
        checkpoint_path=CHECKPOINT_PATH,
    )

    # === OPTION 1: Real-time streaming from Event Hub ===
    """
    EVENT_HUB_CONNECTION_STRING = dbutils.secrets.get(
        scope="fhir-secrets", key="event-hub-connection-string"
    )

    query = ingestion.ingest_from_event_hub(
        connection_string=EVENT_HUB_CONNECTION_STRING,
        event_hub_name="fhir-observations",
        consumer_group="$Default",
        starting_position="latest"
    )

    # Monitor streaming query
    query.awaitTermination()
    """

    # === OPTION 2: Batch ingestion from Blob Storage ===
    ingestion.ingest_from_blob_storage(
        storage_account="myhealthdatalake",
        container_name="fhir-landing",
        folder_path="daily_extract/2025-11-23/*.json",
        file_format="json",
        mode="append",
    )

    # === OPTION 3: Batch ingestion from ADLS Gen2 ===
    """
    ingestion.ingest_from_adls_gen2(
        storage_account="myhealthdatalake",
        file_system="fhir-raw",
        directory_path="extracts/2025-11-23",
        mode="append"
    )
    """

    # Create resource-specific tables
    ingestion.create_resource_specific_tables()

    # Optimize tables (run daily)
    ingestion.optimize_bronze_tables()

    logger.info("✓ FHIR Bronze ingestion complete")
