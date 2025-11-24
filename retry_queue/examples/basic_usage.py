"""
Basic Usage Example for Exponential Backoff Retry Queue

This script demonstrates how to use the retry queue system with sample data.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import datetime
import sys
sys.path.append('..')

from retry_queue_system import (
    ExponentialBackoffRetryQueue,
    RetryQueueConfig,
    RetryQueueSchema
)


def create_spark_session():
    """Create Spark session with Delta Lake configuration."""
    return SparkSession.builder \
        .appName("RetryQueueBasicExample") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()


def main():
    print("=" * 60)
    print("Exponential Backoff Retry Queue - Basic Example")
    print("=" * 60)

    # Initialize Spark
    spark = create_spark_session()

    # Configure retry queue with local paths
    config = RetryQueueConfig(
        retry_table_path="/tmp/delta/retry_queue",
        final_table_path="/tmp/delta/final_results",
        dlq_table_path="/tmp/delta/dead_letter_queue",
        max_retries=5,
        base_backoff_minutes=2
    )

    # Create processor
    processor = ExponentialBackoffRetryQueue(spark, config)

    # Initialize tables (run once)
    print("\n📦 Initializing Delta tables...")
    processor.initialize_tables()

    # Create sample input data
    print("\n📝 Creating sample data...")
    schema = RetryQueueSchema.get_queue_schema()

    sample_data = spark.createDataFrame([
        ("SAMPLE-001", "Lab Result: Hemoglobin 14.5 g/dL", "", 0, datetime.datetime.now()),
        ("SAMPLE-002", "Lab Result: Glucose 95 mg/dL", "", 0, datetime.datetime.now()),
        ("SAMPLE-003", "", "", 0, datetime.datetime.now()),  # Bad data - empty payload
        ("SAMPLE-004", "Lab Result: Creatinine 1.2 mg/dL", "", 0, datetime.datetime.now()),
        ("SAMPLE-005", "Lab Result: WBC 7500 cells/µL", "", 0, datetime.datetime.now()),
    ], schema=schema)

    print(f"Created {sample_data.count()} sample records")

    # Process the batch
    print("\n🚀 Processing batch...")
    processor.process_batch(sample_data)

    # Display results
    print("\n" + "=" * 60)
    print("📊 PROCESSING RESULTS")
    print("=" * 60)

    # Check successful records
    success_df = spark.read.format("delta").load(config.final_table_path)
    success_count = success_df.count()
    print(f"\n✅ Successfully processed: {success_count} records")
    if success_count > 0:
        print("\nSuccessful records:")
        success_df.select("sample_id", "processed_at").show(truncate=False)

    # Check retry queue
    retry_df = spark.read.format("delta").load(config.retry_table_path)
    retry_count = retry_df.count()
    print(f"\n🔄 In retry queue: {retry_count} records")
    if retry_count > 0:
        print("\nRecords waiting for retry:")
        retry_df.select("sample_id", "retry_count", "process_after", "error_msg").show(truncate=False)

    # Check DLQ
    dlq_df = spark.read.format("delta").load(config.dlq_table_path)
    dlq_count = dlq_df.count()
    print(f"\n❌ In Dead Letter Queue: {dlq_count} records")
    if dlq_count > 0:
        print("\nPermanently failed records:")
        dlq_df.select("sample_id", "error_msg", "failed_at").show(truncate=False)

    print("\n" + "=" * 60)
    print("✨ Example completed!")
    print("=" * 60)

    # Cleanup
    spark.stop()


if __name__ == "__main__":
    main()
