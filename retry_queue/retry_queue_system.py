"""
Exponential Backoff Retry Queue for PySpark Data Pipelines

This module implements a resilient retry mechanism for processing records that may fail
due to transient errors (API timeouts, temporary service unavailability, etc.).

Key Features:
- Exponential backoff delays (2^retry_count minutes)
- Time-locked retry queue prevents premature retries
- Dead Letter Queue (DLQ) for permanently failed records
- Delta Lake integration for ACID transactions
- Separates transient from permanent failures

Author: Generated for GenZ Agent Project
Date: 2025-11-24
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, lit, expr
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, IntegerType
from delta.tables import DeltaTable
import datetime
from typing import List, Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RetryQueueConfig:
    """Configuration for the retry queue system."""

    def __init__(
        self,
        retry_table_path: str = "/mnt/delta/lims_retry_queue",
        final_table_path: str = "/mnt/delta/lims_verified_results",
        dlq_table_path: str = "/mnt/delta/lims_dead_letter_queue",
        max_retries: int = 5,
        base_backoff_minutes: int = 2
    ):
        self.retry_table_path = retry_table_path
        self.final_table_path = final_table_path
        self.dlq_table_path = dlq_table_path
        self.max_retries = max_retries
        self.base_backoff_minutes = base_backoff_minutes


class RetryQueueSchema:
    """Schema definitions for retry queue tables."""

    @staticmethod
    def get_queue_schema() -> StructType:
        """
        Schema for the retry queue table.

        Fields:
            sample_id: Unique identifier for the record
            payload: The actual data to process
            error_msg: Last error message encountered
            retry_count: Number of times this record has been retried
            process_after: Timestamp when record becomes eligible for retry (TIME LOCK)
        """
        return StructType([
            StructField("sample_id", StringType(), False),
            StructField("payload", StringType(), True),
            StructField("error_msg", StringType(), True),
            StructField("retry_count", IntegerType(), True),
            StructField("process_after", TimestampType(), True)
        ])

    @staticmethod
    def get_success_schema() -> StructType:
        """Schema for successfully processed records."""
        return StructType([
            StructField("sample_id", StringType(), False),
            StructField("data", StringType(), True),
            StructField("processed_at", TimestampType(), True)
        ])

    @staticmethod
    def get_dlq_schema() -> StructType:
        """Schema for Dead Letter Queue (permanently failed records)."""
        return StructType([
            StructField("sample_id", StringType(), False),
            StructField("payload", StringType(), True),
            StructField("error_msg", StringType(), True),
            StructField("failed_at", TimestampType(), True),
            StructField("retry_count", IntegerType(), True)
        ])


class ExponentialBackoffRetryQueue:
    """
    Main retry queue processor with exponential backoff.

    This class handles the complete lifecycle of record processing:
    1. Reads new records and ready-to-retry records
    2. Processes each record through external system
    3. Implements exponential backoff for transient failures
    4. Routes successes to final table, failures to retry queue or DLQ
    """

    def __init__(self, spark: SparkSession, config: RetryQueueConfig):
        self.spark = spark
        self.config = config
        self.schema = RetryQueueSchema()

    def initialize_tables(self):
        """
        Initialize Delta tables if they don't exist.

        Call this once during system setup.
        """
        # Retry Queue
        DeltaTable.createIfNotExists(self.spark) \
            .location(self.config.retry_table_path) \
            .addColumns(self.schema.get_queue_schema()) \
            .execute()
        logger.info(f"Retry queue initialized at {self.config.retry_table_path}")

        # Final Results Table
        DeltaTable.createIfNotExists(self.spark) \
            .location(self.config.final_table_path) \
            .addColumns(self.schema.get_success_schema()) \
            .execute()
        logger.info(f"Final table initialized at {self.config.final_table_path}")

        # Dead Letter Queue
        DeltaTable.createIfNotExists(self.spark) \
            .location(self.config.dlq_table_path) \
            .addColumns(self.schema.get_dlq_schema()) \
            .execute()
        logger.info(f"DLQ initialized at {self.config.dlq_table_path}")

    def calculate_backoff_delay(self, retry_count: int) -> datetime.datetime:
        """
        Calculate next retry time using exponential backoff.

        Formula: wait_time = base * (2 ^ retry_count)

        Args:
            retry_count: Current number of retries

        Returns:
            Timestamp when record should be retried

        Example:
            retry_count=0 → 2 minutes from now
            retry_count=1 → 4 minutes from now
            retry_count=2 → 8 minutes from now
            retry_count=3 → 16 minutes from now
            retry_count=4 → 32 minutes from now
        """
        backoff_minutes = self.config.base_backoff_minutes ** retry_count
        next_try_time = datetime.datetime.now() + datetime.timedelta(minutes=backoff_minutes)

        logger.debug(f"Backoff delay: {backoff_minutes} minutes (retry #{retry_count})")
        return next_try_time

    def process_record(self, row: Dict[str, Any]) -> Tuple[str, str]:
        """
        Process a single record through external system.

        THIS IS WHERE YOU INTEGRATE YOUR ACTUAL BUSINESS LOGIC:
        - API calls to external systems
        - Database writes
        - File uploads
        - Any I/O operation that might fail

        Args:
            row: Record dictionary with 'sample_id' and 'payload'

        Returns:
            Tuple of (status, message) where status is:
            - 'SUCCESS': Record processed successfully
            - 'TRANSIENT_FAIL': Temporary error, can retry
            - 'PERMANENT_FAIL': Bad data, cannot retry
        """
        # VALIDATION: Check for bad data (permanent failures)
        if row['payload'] is None or row['payload'] == "":
            return "PERMANENT_FAIL", "Missing or invalid payload"

        # TODO: Replace this with your actual processing logic
        # Example integrations:
        #   - requests.post("https://api.example.com/submit", json=row)
        #   - jdbc_connection.execute(f"INSERT INTO table VALUES (...)")
        #   - s3_client.put_object(Bucket='my-bucket', Key=row['sample_id'], Body=row['payload'])

        try:
            # Simulated processing (replace with real logic)
            result = self._external_api_call(row)
            return "SUCCESS", "Processed successfully"

        except TransientError as e:
            # Temporary failures: API timeout, 503 errors, connection issues
            return "TRANSIENT_FAIL", str(e)

        except PermanentError as e:
            # Permanent failures: validation errors, 400 errors, data issues
            return "PERMANENT_FAIL", str(e)

    def _external_api_call(self, row: Dict[str, Any]) -> Any:
        """
        Placeholder for external system integration.

        Replace this with your actual API/database/file system call.
        """
        # This is a simulation - replace with real logic
        import random
        if random.random() < 0.3:  # 30% failure rate for demo
            raise TransientError("API Timeout / 503 Service Unavailable")
        return {"status": "ok"}

    def process_batch(self, new_records_df):
        """
        Main processing loop for a batch of records.

        Workflow:
        1. Read new records and ready-to-retry records
        2. Process each record
        3. Route to success, retry queue, or DLQ based on result
        4. Update Delta tables using ACID transactions

        Args:
            new_records_df: DataFrame with new records to process
        """
        logger.info(f"Starting batch processing at {datetime.datetime.now()}")

        # STEP 1: Read retry queue (only records past their time lock)
        retry_table = DeltaTable.forPath(self.spark, self.config.retry_table_path)

        ready_to_retry_df = retry_table.toDF() \
            .filter(col("process_after") <= current_timestamp())

        retry_count = ready_to_retry_df.count()
        logger.info(f"Found {retry_count} records ready for retry")

        # STEP 2: Combine new records with retry records
        batch_df = new_records_df.unionByName(ready_to_retry_df, allowMissingColumns=True)

        total_records = batch_df.count()
        logger.info(f"Processing {total_records} total records")

        # STEP 3: Process records
        # Note: For production with large datasets, use mapPartitions instead of collect
        rows_to_process = batch_df.collect()

        success_list = []
        retry_list = []
        permanent_fail_list = []

        for row in rows_to_process:
            row_dict = row.asDict()
            status, msg = self.process_record(row_dict)

            if status == "SUCCESS":
                success_list.append((
                    row_dict['sample_id'],
                    row_dict['payload'],
                    datetime.datetime.now()
                ))

            elif status == "TRANSIENT_FAIL":
                current_retries = row_dict.get('retry_count', 0) + 1

                # Check if max retries exceeded
                if current_retries > self.config.max_retries:
                    logger.warning(f"Sample {row_dict['sample_id']} exceeded max retries")
                    permanent_fail_list.append((
                        row_dict['sample_id'],
                        row_dict['payload'],
                        f"Max retries exceeded: {msg}",
                        datetime.datetime.now(),
                        current_retries
                    ))
                else:
                    next_try_time = self.calculate_backoff_delay(current_retries)

                    logger.info(
                        f"Sample {row_dict['sample_id']} failed (attempt {current_retries}). "
                        f"Next retry at {next_try_time}"
                    )

                    retry_list.append((
                        row_dict['sample_id'],
                        row_dict['payload'],
                        msg,
                        current_retries,
                        next_try_time
                    ))

            elif status == "PERMANENT_FAIL":
                permanent_fail_list.append((
                    row_dict['sample_id'],
                    row_dict['payload'],
                    msg,
                    datetime.datetime.now(),
                    row_dict.get('retry_count', 0)
                ))

        # STEP 4: Persist results to Delta tables
        self._persist_results(success_list, retry_list, permanent_fail_list, retry_table)

        # STEP 5: Log summary
        logger.info(
            f"Batch complete: {len(success_list)} succeeded, "
            f"{len(retry_list)} queued for retry, "
            f"{len(permanent_fail_list)} permanently failed"
        )

    def _persist_results(
        self,
        success_list: List[Tuple],
        retry_list: List[Tuple],
        permanent_fail_list: List[Tuple],
        retry_table: DeltaTable
    ):
        """
        Persist processing results to Delta tables using ACID transactions.

        This method ensures data consistency:
        - Successful records written to final table
        - Failed records updated in retry queue with new timestamps
        - Permanently failed records written to DLQ
        - Successful records removed from retry queue
        """
        # A. Write successes to final table
        if success_list:
            success_df = self.spark.createDataFrame(
                success_list,
                self.schema.get_success_schema()
            )
            success_df.write.format("delta").mode("append").save(self.config.final_table_path)
            logger.info(f"✅ Saved {len(success_list)} successful records")

        # B. Update retry queue with failed records
        if retry_list:
            retry_update_df = self.spark.createDataFrame(
                retry_list,
                self.schema.get_queue_schema()
            )

            # UPSERT: Update existing records or insert new ones
            (retry_table.alias("target")
             .merge(
                 retry_update_df.alias("source"),
                 "target.sample_id = source.sample_id"
             )
             .whenMatchedUpdateAll()  # Update timestamp and retry_count
             .whenNotMatchedInsertAll()  # Insert new failures
             .execute()
            )
            logger.info(f"🔄 Queued {len(retry_list)} records for retry")

        # C. Remove successful records from retry queue
        successful_ids = [x[0] for x in success_list]
        if successful_ids:
            retry_table.delete(col("sample_id").isin(successful_ids))
            logger.info(f"🧹 Cleaned up {len(successful_ids)} records from retry queue")

        # D. Write permanently failed records to DLQ
        if permanent_fail_list:
            dlq_df = self.spark.createDataFrame(
                permanent_fail_list,
                self.schema.get_dlq_schema()
            )
            dlq_df.write.format("delta").mode("append").save(self.config.dlq_table_path)
            logger.error(f"❌ Moved {len(permanent_fail_list)} records to Dead Letter Queue")


# Custom exception classes for clear error handling
class TransientError(Exception):
    """Temporary error that warrants a retry (e.g., API timeout, 503 error)."""
    pass


class PermanentError(Exception):
    """Permanent error that should not be retried (e.g., validation error, 400 error)."""
    pass


# ==========================================
# USAGE EXAMPLE
# ==========================================
def main():
    """Example usage of the retry queue system."""

    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("RetryQueueExample") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()

    # Configure retry queue
    config = RetryQueueConfig(
        retry_table_path="/tmp/delta/retry_queue",
        final_table_path="/tmp/delta/final_results",
        dlq_table_path="/tmp/delta/dead_letter_queue",
        max_retries=5,
        base_backoff_minutes=2
    )

    # Create retry queue processor
    retry_queue = ExponentialBackoffRetryQueue(spark, config)

    # Initialize tables (run once)
    retry_queue.initialize_tables()

    # Create sample input data
    new_data = spark.createDataFrame([
        ("S-101", "Valid Result A", "", 0, datetime.datetime.now()),
        ("S-102", "", "", 0, datetime.datetime.now()),  # Bad data
        ("S-103", "Valid Result B", "", 0, datetime.datetime.now())
    ], schema=RetryQueueSchema.get_queue_schema())

    # Process batch
    retry_queue.process_batch(new_data)

    spark.stop()


if __name__ == "__main__":
    main()
