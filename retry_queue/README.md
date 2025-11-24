# 🔄 Exponential Backoff Retry Queue for PySpark

A production-ready retry queue system for handling transient failures in data pipelines using PySpark and Delta Lake.

## 📋 Overview

This system handles transient errors in data processing pipelines by automatically retrying failed records with exponentially increasing delays. It distinguishes between temporary failures (which should be retried) and permanent failures (which should be reviewed manually).

### Key Features

- ✅ **Exponential Backoff**: Smart retry delays that give failing systems time to recover
- ✅ **Time Lock Pattern**: Prevents premature retries that would fail again
- ✅ **Dead Letter Queue (DLQ)**: Captures permanently failed records for investigation
- ✅ **ACID Transactions**: Delta Lake ensures data consistency
- ✅ **Separation of Concerns**: Clear distinction between transient and permanent failures
- ✅ **Production-Ready**: Comprehensive logging, monitoring, and error handling

## 🏗️ Architecture

```
┌─────────────────┐
│   New Records   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         Retry Queue Processor           │
│                                         │
│  1. Read new + ready-to-retry records  │
│  2. Process each record                │
│  3. Calculate backoff delays           │
│  4. Route to appropriate destination   │
└─────────┬───────────────────────────────┘
          │
    ┌─────┴──────┬──────────────┐
    ▼            ▼              ▼
┌────────┐  ┌─────────┐  ┌──────────┐
│Success │  │  Retry  │  │   DLQ    │
│ Table  │  │  Queue  │  │ (Manual  │
│        │  │(Locked) │  │ Review)  │
└────────┘  └─────────┘  └──────────┘
```

## 🔧 How It Works

### The Time Lock Pattern

Each failed record in the retry queue has a `process_after` timestamp:

```python
# Record structure
{
    "sample_id": "S-101",
    "payload": "data to process",
    "error_msg": "API Timeout",
    "retry_count": 2,
    "process_after": "2025-11-24 14:30:00"  # TIME LOCK
}
```

**The system only processes records where `process_after <= current_time`**

This prevents the system from immediately retrying a failed operation, which would likely fail again.

### Exponential Backoff Formula

```
wait_time = base_backoff * (2 ^ retry_count)
```

**Example with base_backoff = 2 minutes:**

| Retry # | Wait Time | Cumulative Wait |
|---------|-----------|-----------------|
| 1       | 2 min     | 2 min           |
| 2       | 4 min     | 6 min           |
| 3       | 8 min     | 14 min          |
| 4       | 16 min    | 30 min          |
| 5       | 32 min    | 62 min          |

After 5 retries, the record moves to the Dead Letter Queue.

### Processing Flow

```python
for record in batch:
    status, message = process_record(record)

    if status == "SUCCESS":
        ✅ Write to final results table
        ✅ Remove from retry queue

    elif status == "TRANSIENT_FAIL":
        if retry_count < max_retries:
            🔄 Update retry queue with new timestamp
            🔄 Increment retry_count
        else:
            ❌ Move to Dead Letter Queue

    elif status == "PERMANENT_FAIL":
        ❌ Move to Dead Letter Queue immediately
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install pyspark delta-spark pyyaml
```

### Installation

```bash
# Clone the repository
git clone <repository_url>
cd retry_queue

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from pyspark.sql import SparkSession
from retry_queue_system import ExponentialBackoffRetryQueue, RetryQueueConfig
import datetime

# Initialize Spark with Delta Lake
spark = SparkSession.builder \
    .appName("RetryQueueExample") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# Configure the retry queue
config = RetryQueueConfig(
    retry_table_path="/mnt/delta/retry_queue",
    final_table_path="/mnt/delta/results",
    dlq_table_path="/mnt/delta/dlq",
    max_retries=5,
    base_backoff_minutes=2
)

# Create processor
processor = ExponentialBackoffRetryQueue(spark, config)

# Initialize tables (run once)
processor.initialize_tables()

# Create sample data
from retry_queue_system import RetryQueueSchema
schema = RetryQueueSchema.get_queue_schema()

new_data = spark.createDataFrame([
    ("sample-001", "data payload", "", 0, datetime.datetime.now()),
    ("sample-002", "another payload", "", 0, datetime.datetime.now())
], schema=schema)

# Process the batch
processor.process_batch(new_data)
```

## 🎯 Integrating with Your System

### Step 1: Implement Your Processing Logic

Replace the `process_record()` method with your actual business logic:

```python
def process_record(self, row: Dict[str, Any]) -> Tuple[str, str]:
    """Process record through your external system."""

    # Example: REST API call
    try:
        response = requests.post(
            "https://api.yourcompany.com/endpoint",
            json={"sample_id": row['sample_id'], "data": row['payload']},
            timeout=30
        )

        if response.status_code == 200:
            return "SUCCESS", "Processed OK"
        elif response.status_code in [500, 502, 503, 504]:
            # Transient server errors
            return "TRANSIENT_FAIL", f"Server error: {response.status_code}"
        else:
            # Client errors (bad data)
            return "PERMANENT_FAIL", f"Client error: {response.status_code}"

    except requests.exceptions.Timeout:
        return "TRANSIENT_FAIL", "Request timeout"
    except requests.exceptions.ConnectionError:
        return "TRANSIENT_FAIL", "Connection failed"
    except ValueError as e:
        return "PERMANENT_FAIL", f"Invalid data: {e}"
```

### Step 2: Configure Paths and Settings

Edit `config.yaml`:

```yaml
paths:
  retry_queue: "/mnt/your-storage/retry_queue"
  final_results: "/mnt/your-storage/results"
  dead_letter_queue: "/mnt/your-storage/dlq"

retry:
  max_retries: 5
  base_backoff_minutes: 2
```

### Step 3: Schedule the Job

**Using Databricks Jobs:**

```python
# Databricks notebook
import sys
sys.path.append("/dbfs/mnt/scripts")

from retry_queue_system import ExponentialBackoffRetryQueue, RetryQueueConfig

# Load config
config = RetryQueueConfig(
    retry_table_path=dbutils.widgets.get("retry_table_path"),
    final_table_path=dbutils.widgets.get("final_table_path"),
    # ...
)

processor = ExponentialBackoffRetryQueue(spark, config)

# Read new data from your source
new_data = spark.read.format("delta").load("/path/to/new/data")

processor.process_batch(new_data)
```

**Using Apache Airflow:**

```python
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

with DAG('retry_queue_processor', schedule_interval='*/5 * * * *') as dag:
    process_task = SparkSubmitOperator(
        task_id='process_retry_queue',
        application='/path/to/retry_queue_job.py',
        conf={
            'spark.sql.extensions': 'io.delta.sql.DeltaSparkSessionExtension',
            'spark.sql.catalog.spark_catalog':
                'org.apache.spark.sql.delta.catalog.DeltaCatalog'
        }
    )
```

## 📊 Monitoring and Observability

### Key Metrics to Track

1. **Success Rate**: `successful_records / total_records`
2. **Retry Queue Size**: Number of records waiting for retry
3. **DLQ Size**: Number of permanently failed records
4. **Average Retry Count**: How many retries records typically need
5. **Processing Time**: Time to process each batch

### Example Monitoring Query

```sql
-- Check retry queue health
SELECT
    COUNT(*) as total_waiting,
    AVG(retry_count) as avg_retries,
    MAX(retry_count) as max_retries,
    MIN(process_after) as next_ready_time
FROM delta.`/mnt/delta/retry_queue`

-- Check DLQ for manual review
SELECT
    sample_id,
    error_msg,
    retry_count,
    failed_at
FROM delta.`/mnt/delta/dead_letter_queue`
ORDER BY failed_at DESC
LIMIT 10
```

### Setting Up Alerts

```python
def check_health_and_alert():
    """Monitor system health and send alerts."""

    # Check DLQ size
    dlq_count = spark.read.format("delta") \
        .load("/mnt/delta/dlq") \
        .count()

    if dlq_count > 100:
        send_alert(f"⚠️ DLQ has {dlq_count} records. Manual review needed!")

    # Check retry queue backlog
    retry_count = spark.read.format("delta") \
        .load("/mnt/delta/retry_queue") \
        .count()

    if retry_count > 1000:
        send_alert(f"⚠️ Retry queue backlog: {retry_count} records")
```

## 🔍 Troubleshooting

### Problem: Records stuck in retry queue

**Symptoms**: Records never succeed, keep retrying indefinitely

**Diagnosis**:
```sql
SELECT sample_id, retry_count, error_msg, process_after
FROM delta.`/mnt/delta/retry_queue`
WHERE retry_count >= 3
ORDER BY retry_count DESC
```

**Solutions**:
1. Check if external system is actually down
2. Verify network connectivity
3. Check if error classification is correct (transient vs permanent)
4. Review `process_record()` logic for bugs

### Problem: Too many records in DLQ

**Symptoms**: High number of permanently failed records

**Diagnosis**:
```sql
SELECT error_msg, COUNT(*) as count
FROM delta.`/mnt/delta/dead_letter_queue`
GROUP BY error_msg
ORDER BY count DESC
```

**Solutions**:
1. Fix data validation issues in source systems
2. Add better error handling for edge cases
3. Review permanent failure classification logic
4. Consider data quality checks before ingestion

### Problem: Slow processing

**Symptoms**: Batches take too long to process

**Diagnosis**:
- Check Spark UI for bottlenecks
- Review external API response times
- Analyze partition distribution

**Solutions**:
1. Increase parallelism: `spark.conf.set("spark.sql.shuffle.partitions", "200")`
2. Use `mapPartitions` instead of `collect()` for large datasets
3. Batch API calls instead of one-by-one
4. Add connection pooling for external systems

## 📁 File Structure

```
retry_queue/
├── retry_queue_system.py    # Main implementation
├── config.yaml               # Configuration file
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── examples/
│   ├── basic_usage.py       # Simple example
│   ├── databricks_job.py    # Databricks integration
│   └── airflow_dag.py       # Airflow integration
└── tests/
    ├── test_retry_logic.py  # Unit tests
    └── test_integration.py  # Integration tests
```

## 🧪 Testing

### Unit Tests

```python
import pytest
from retry_queue_system import ExponentialBackoffRetryQueue

def test_backoff_calculation():
    """Test exponential backoff delays."""
    config = RetryQueueConfig(base_backoff_minutes=2)
    processor = ExponentialBackoffRetryQueue(spark, config)

    # Test backoff calculations
    assert processor.calculate_backoff_delay(0) == 2  # 2^0 * 2 = 2 min
    assert processor.calculate_backoff_delay(1) == 4  # 2^1 * 2 = 4 min
    assert processor.calculate_backoff_delay(2) == 8  # 2^2 * 2 = 8 min
```

### Integration Tests

```python
def test_full_workflow():
    """Test complete retry workflow."""

    # Create test data with intentional failures
    test_data = create_test_dataframe([
        ("S-001", "valid data", "", 0, now()),
        ("S-002", None, "", 0, now()),  # Will fail permanently
        ("S-003", "valid data", "", 0, now())
    ])

    processor.process_batch(test_data)

    # Verify results
    success_count = spark.read.format("delta") \
        .load(config.final_table_path).count()
    assert success_count >= 1

    dlq_count = spark.read.format("delta") \
        .load(config.dlq_table_path).count()
    assert dlq_count >= 1
```

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Add tests for new functionality
4. Ensure all tests pass: `pytest`
5. Submit a pull request

## 📄 License

This project is part of the GenZ Agent repository.

## 🔗 Related Documentation

- [Delta Lake Documentation](https://docs.delta.io/)
- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)
- [Exponential Backoff Algorithm](https://en.wikipedia.org/wiki/Exponential_backoff)

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Refer to the main GenZ Agent documentation
- Check the troubleshooting section above

---

**Last Updated**: 2025-11-24
**Version**: 1.0.0
**Author**: GenZ Agent Project
