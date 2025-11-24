"""
Databricks Job Example for Exponential Backoff Retry Queue

This script is designed to run as a Databricks job.
Configure as a notebook or Python script job.

Databricks Job Configuration:
- Cluster: Standard cluster with Delta Lake support
- Libraries: Install from requirements.txt
- Schedule: Every 5 minutes
- Timeout: 10 minutes
- Max concurrent runs: 1
"""

# Databricks notebook source
# MAGIC %md
# MAGIC # Retry Queue Processor - Databricks Job
# MAGIC
# MAGIC This job processes the retry queue every 5 minutes.

# COMMAND ----------

import sys
from datetime import datetime

# Add your module path if needed
# sys.path.append("/dbfs/mnt/scripts/retry_queue")

from retry_queue_system import (
    ExponentialBackoffRetryQueue,
    RetryQueueConfig
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Get configuration from Databricks widgets or secrets
try:
    # Option 1: Use Databricks widgets for parameters
    dbutils.widgets.text("retry_table_path", "/mnt/delta/lims_retry_queue")
    dbutils.widgets.text("final_table_path", "/mnt/delta/lims_verified_results")
    dbutils.widgets.text("dlq_table_path", "/mnt/delta/lims_dead_letter_queue")
    dbutils.widgets.text("max_retries", "5")
    dbutils.widgets.text("base_backoff_minutes", "2")

    # Read widget values
    retry_table_path = dbutils.widgets.get("retry_table_path")
    final_table_path = dbutils.widgets.get("final_table_path")
    dlq_table_path = dbutils.widgets.get("dlq_table_path")
    max_retries = int(dbutils.widgets.get("max_retries"))
    base_backoff_minutes = int(dbutils.widgets.get("base_backoff_minutes"))

except Exception as e:
    # Option 2: Use default values for development
    print(f"Widget configuration not available: {e}")
    print("Using default configuration for development")

    retry_table_path = "/mnt/delta/lims_retry_queue"
    final_table_path = "/mnt/delta/lims_verified_results"
    dlq_table_path = "/mnt/delta/lims_dead_letter_queue"
    max_retries = 5
    base_backoff_minutes = 2

# Create configuration
config = RetryQueueConfig(
    retry_table_path=retry_table_path,
    final_table_path=final_table_path,
    dlq_table_path=dlq_table_path,
    max_retries=max_retries,
    base_backoff_minutes=base_backoff_minutes
)

print(f"Configuration loaded:")
print(f"  Retry Table: {config.retry_table_path}")
print(f"  Final Table: {config.final_table_path}")
print(f"  DLQ Table: {config.dlq_table_path}")
print(f"  Max Retries: {config.max_retries}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize Processor

# COMMAND ----------

# Create processor (spark session is already available in Databricks)
processor = ExponentialBackoffRetryQueue(spark, config)

# Initialize tables if this is first run
try:
    processor.initialize_tables()
    print("✅ Tables initialized successfully")
except Exception as e:
    print(f"Tables already exist or initialization failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read New Data from Source

# COMMAND ----------

# Read new data from your source system
# Replace this with your actual data source
# Examples:
#   - Event Hub: spark.readStream.format("eventhubs")...
#   - JDBC: spark.read.format("jdbc").option(...)
#   - Delta: spark.read.format("delta").load(...)
#   - Kafka: spark.readStream.format("kafka")...

# For this example, we'll simulate reading from a staging table
try:
    # Read from staging table (replace with your source)
    staging_table = "/mnt/delta/lims_staging_data"

    new_data = spark.read.format("delta") \
        .load(staging_table) \
        .filter("processing_status = 'PENDING'")

    new_count = new_data.count()
    print(f"📥 Found {new_count} new records to process")

    # Transform to retry queue schema if needed
    from retry_queue_system import RetryQueueSchema
    from pyspark.sql.functions import lit, current_timestamp

    new_data_formatted = new_data.select(
        "sample_id",
        "payload",
        lit("").alias("error_msg"),
        lit(0).alias("retry_count"),
        current_timestamp().alias("process_after")
    )

except Exception as e:
    print(f"⚠️ No new data available: {e}")
    new_data_formatted = spark.createDataFrame([], RetryQueueSchema.get_queue_schema())
    new_count = 0

# COMMAND ----------

# MAGIC %md
# MAGIC ## Process Batch

# COMMAND ----------

if new_count > 0 or True:  # Always check retry queue
    print(f"\n🚀 Starting batch processing at {datetime.now()}")

    try:
        processor.process_batch(new_data_formatted)
        print("✅ Batch processing completed successfully")

    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        raise  # Re-raise to fail the Databricks job

# COMMAND ----------

# MAGIC %md
# MAGIC ## Health Checks and Monitoring

# COMMAND ----------

# Monitor retry queue health
retry_df = spark.read.format("delta").load(config.retry_table_path)
retry_count = retry_df.count()

# Monitor DLQ
dlq_df = spark.read.format("delta").load(config.dlq_table_path)
dlq_count = dlq_df.count()

# Monitor success rate
success_df = spark.read.format("delta").load(config.final_table_path)
success_count = success_df.count()

print("\n" + "=" * 60)
print("📊 SYSTEM HEALTH REPORT")
print("=" * 60)
print(f"✅ Successful records: {success_count}")
print(f"🔄 Retry queue size: {retry_count}")
print(f"❌ DLQ size: {dlq_count}")

# Alert thresholds
RETRY_QUEUE_THRESHOLD = 1000
DLQ_THRESHOLD = 100

if retry_count > RETRY_QUEUE_THRESHOLD:
    print(f"\n⚠️ WARNING: Retry queue exceeds threshold ({retry_count} > {RETRY_QUEUE_THRESHOLD})")
    # Send alert via Databricks notification or external service

if dlq_count > DLQ_THRESHOLD:
    print(f"\n🚨 ALERT: DLQ exceeds threshold ({dlq_count} > {DLQ_THRESHOLD})")
    # Send critical alert for manual review

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display Recent DLQ Entries (for monitoring)

# COMMAND ----------

if dlq_count > 0:
    print("\n❌ Recent Dead Letter Queue Entries:")
    dlq_df.orderBy("failed_at", ascending=False) \
        .select("sample_id", "error_msg", "retry_count", "failed_at") \
        .limit(10) \
        .display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Job Summary

# COMMAND ----------

print("\n" + "=" * 60)
print("✨ Job completed successfully")
print(f"Execution time: {datetime.now()}")
print("=" * 60)

# Return summary for Databricks job monitoring
dbutils.notebook.exit({
    "status": "success",
    "records_processed": new_count,
    "retry_queue_size": retry_count,
    "dlq_size": dlq_count,
    "timestamp": str(datetime.now())
})
