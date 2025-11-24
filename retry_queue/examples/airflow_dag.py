"""
Apache Airflow DAG for Exponential Backoff Retry Queue

This DAG runs the retry queue processor every 5 minutes.

Installation:
1. Place this file in your Airflow DAGs folder
2. Ensure retry_queue_system.py is in PYTHONPATH
3. Configure Spark connection in Airflow UI

DAG Configuration:
- Schedule: Every 5 minutes
- Max active runs: 1
- Catchup: False (don't backfill)
- SLA: 10 minutes
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import logging

# DAG default arguments
default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'email': ['alerts@yourcompany.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=2),
    'execution_timeout': timedelta(minutes=10),
}

# Create DAG
dag = DAG(
    'retry_queue_processor',
    default_args=default_args,
    description='Process retry queue with exponential backoff',
    schedule_interval='*/5 * * * *',  # Every 5 minutes
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=['retry-queue', 'data-pipeline', 'lims'],
)


def check_system_health(**context):
    """
    Pre-flight check before processing.
    Ensures external systems are available.
    """
    from pyspark.sql import SparkSession

    logger = logging.getLogger(__name__)
    logger.info("Performing health checks...")

    # Create Spark session
    spark = SparkSession.builder \
        .appName("RetryQueueHealthCheck") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()

    try:
        # Check if Delta tables exist
        retry_table_path = "/mnt/delta/lims_retry_queue"

        try:
            retry_df = spark.read.format("delta").load(retry_table_path)
            queue_size = retry_df.count()
            logger.info(f"✅ Retry queue accessible: {queue_size} records")
        except Exception as e:
            logger.error(f"❌ Cannot access retry queue: {e}")
            raise

        # Check external API availability (optional)
        # import requests
        # response = requests.get("https://api.yourcompany.com/health", timeout=5)
        # if response.status_code != 200:
        #     raise Exception("External API unhealthy")

        logger.info("✅ All health checks passed")

    finally:
        spark.stop()


def process_retry_queue(**context):
    """
    Main task: Process the retry queue.
    """
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import lit, current_timestamp
    import sys
    sys.path.append('/opt/airflow/dags/retry_queue')  # Adjust path

    from retry_queue_system import (
        ExponentialBackoffRetryQueue,
        RetryQueueConfig,
        RetryQueueSchema
    )

    logger = logging.getLogger(__name__)

    # Create Spark session
    spark = SparkSession.builder \
        .appName("RetryQueueProcessor") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()

    try:
        # Configure
        config = RetryQueueConfig(
            retry_table_path="/mnt/delta/lims_retry_queue",
            final_table_path="/mnt/delta/lims_verified_results",
            dlq_table_path="/mnt/delta/lims_dead_letter_queue",
            max_retries=5,
            base_backoff_minutes=2
        )

        # Create processor
        processor = ExponentialBackoffRetryQueue(spark, config)

        # Read new data from staging
        staging_path = "/mnt/delta/lims_staging_data"

        try:
            new_data = spark.read.format("delta") \
                .load(staging_path) \
                .filter("processing_status = 'PENDING'")

            # Transform to retry queue schema
            new_data_formatted = new_data.select(
                "sample_id",
                "payload",
                lit("").alias("error_msg"),
                lit(0).alias("retry_count"),
                current_timestamp().alias("process_after")
            )

            new_count = new_data_formatted.count()
            logger.info(f"📥 Found {new_count} new records")

        except Exception as e:
            logger.warning(f"No new data: {e}")
            new_data_formatted = spark.createDataFrame([], RetryQueueSchema.get_queue_schema())

        # Process batch
        logger.info("🚀 Starting batch processing")
        processor.process_batch(new_data_formatted)
        logger.info("✅ Batch processing complete")

        # Push metrics to XCom for monitoring
        retry_count = spark.read.format("delta").load(config.retry_table_path).count()
        dlq_count = spark.read.format("delta").load(config.dlq_table_path).count()

        context['task_instance'].xcom_push(key='retry_queue_size', value=retry_count)
        context['task_instance'].xcom_push(key='dlq_size', value=dlq_count)

    finally:
        spark.stop()


def monitor_and_alert(**context):
    """
    Post-processing monitoring and alerting.
    """
    logger = logging.getLogger(__name__)

    # Get metrics from XCom
    ti = context['task_instance']
    retry_queue_size = ti.xcom_pull(task_ids='process_queue', key='retry_queue_size')
    dlq_size = ti.xcom_pull(task_ids='process_queue', key='dlq_size')

    logger.info(f"📊 Metrics - Retry Queue: {retry_queue_size}, DLQ: {dlq_size}")

    # Alert thresholds
    RETRY_QUEUE_THRESHOLD = 1000
    DLQ_THRESHOLD = 100

    alerts = []

    if retry_queue_size and retry_queue_size > RETRY_QUEUE_THRESHOLD:
        message = f"⚠️ Retry queue size ({retry_queue_size}) exceeds threshold"
        alerts.append(message)
        logger.warning(message)

    if dlq_size and dlq_size > DLQ_THRESHOLD:
        message = f"🚨 DLQ size ({dlq_size}) exceeds threshold - manual review needed"
        alerts.append(message)
        logger.error(message)

    if alerts:
        # Send alerts via Slack, PagerDuty, etc.
        # from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
        # SlackWebhookOperator(
        #     task_id='send_slack_alert',
        #     slack_webhook_conn_id='slack_alerts',
        #     message='\n'.join(alerts)
        # ).execute(context)
        pass

    logger.info("✅ Monitoring complete")


# Define tasks
health_check = PythonOperator(
    task_id='health_check',
    python_callable=check_system_health,
    dag=dag,
)

process_queue = PythonOperator(
    task_id='process_queue',
    python_callable=process_retry_queue,
    dag=dag,
)

monitor = PythonOperator(
    task_id='monitor_and_alert',
    python_callable=monitor_and_alert,
    dag=dag,
)

# Alternative: Use SparkSubmitOperator for better resource management
# process_queue_spark = SparkSubmitOperator(
#     task_id='process_queue_spark',
#     application='/path/to/retry_queue_job.py',
#     conn_id='spark_default',
#     conf={
#         'spark.sql.extensions': 'io.delta.sql.DeltaSparkSessionExtension',
#         'spark.sql.catalog.spark_catalog': 'org.apache.spark.sql.delta.catalog.DeltaCatalog',
#     },
#     dag=dag,
# )

# Define task dependencies
health_check >> process_queue >> monitor
