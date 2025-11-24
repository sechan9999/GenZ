# 🔧 LIMS Data Migration Troubleshooting Guide for Azure Databricks

Complete guide for diagnosing and fixing LIMS data migration issues on Azure Databricks.

---

## 📋 Table of Contents

1. [Quick Diagnostics Checklist](#quick-diagnostics-checklist)
2. [Problem: Incomplete Migration (Missing Records)](#problem-1-incomplete-migration-missing-records)
3. [Problem: Slow Migration Performance](#problem-2-slow-migration-performance)
4. [Problem: Out of Memory Errors](#problem-3-out-of-memory-errors)
5. [Problem: Connection Timeouts (JDBC)](#problem-4-connection-timeouts-jdbc)
6. [Problem: Data Skew](#problem-5-data-skew)
7. [Problem: Delta Lake Errors](#problem-6-delta-lake-errors)
8. [Problem: Duplicate Records](#problem-7-duplicate-records)
9. [Azure Databricks-Specific Issues](#azure-databricks-specific-issues)
10. [Performance Optimization Checklist](#performance-optimization-checklist)

---

## Quick Diagnostics Checklist

Run these checks first to identify the root cause:

```python
from lims_migration_diagnostics import LIMSMigrationDiagnostics

diagnostics = LIMSMigrationDiagnostics(spark)

# 1. Compare counts
diagnostics.compare_source_target_counts(
    source_path="/mnt/source",
    target_path="/mnt/target"
)

# 2. Find missing records
missing_df = diagnostics.find_missing_records(
    source_path="/mnt/source",
    target_path="/mnt/target"
)

# 3. Check performance
diagnostics.analyze_migration_performance(
    target_path="/mnt/target",
    migration_start_time=migration_start
)

# 4. Data quality check
diagnostics.check_data_quality(target_path="/mnt/target")
```

---

## Problem 1: Incomplete Migration (Missing Records)

### Symptoms
- Source has 1,000,000 records
- Target only has 850,000 records
- Some samples never made it to target table

### Root Causes

#### 1.1 Silent Failures in Batch Processing

**Diagnosis:**
```sql
-- Check checkpoint table for failures
SELECT batch_id, status, records_processed
FROM delta.`/mnt/delta/checkpoint`
WHERE status LIKE 'failed%'
ORDER BY timestamp DESC
```

**Solution:**
```python
# Use the optimized migration pipeline with error handling
from optimized_migration_pipeline import OptimizedLIMSMigration, MigrationConfig

config = MigrationConfig(
    source_path="/mnt/source",
    target_path="/mnt/target",
    batch_size=10000,  # Smaller batches = better fault tolerance
    checkpoint_path="/mnt/checkpoint"
)

migration = OptimizedLIMSMigration(spark, config)
stats = migration.migrate_full_dataset()

# If batches failed, resume them
migration.resume_failed_batches()
```

#### 1.2 JDBC Read Timeouts

**Diagnosis:**
```python
# Check Spark logs for timeout errors
%sh grep -i "timeout" /databricks/driver/logs/stderr
```

**Solution:**
```python
# Increase JDBC read timeout and fetch size
df = spark.read \
    .format("jdbc") \
    .option("url", jdbc_url) \
    .option("dbtable", table_name) \
    .option("sessionInitStatement", "SET statement_timeout = 3600000") \
    .option("queryTimeout", "3600") \
    .option("fetchsize", "10000") \
    .option("batchsize", "10000") \
    .load()
```

#### 1.3 Data Type Conversion Errors

**Diagnosis:**
```python
# Look for records with problematic data types
# Example: Invalid dates, NULL in NOT NULL columns
source_df = spark.read.format("delta").load("/mnt/source")

# Check for NULL values in key columns
source_df.select([count(when(col(c).isNull(), c)).alias(c) for c in source_df.columns]).show()
```

**Solution:**
```python
# Add data cleansing transformation
def clean_lims_data(df):
    """Clean common LIMS data issues."""
    from pyspark.sql.functions import coalesce, lit, to_timestamp

    return df \
        .withColumn("sample_id", coalesce(col("sample_id"), lit("UNKNOWN"))) \
        .withColumn("test_code", coalesce(col("test_code"), lit("MISSING"))) \
        .withColumn("collection_date",
                    to_timestamp(col("collection_date"), "yyyy-MM-dd HH:mm:ss")) \
        .filter(col("sample_id") != "UNKNOWN")  # Filter out invalid records

# Use in migration
migration = OptimizedLIMSMigration(spark, config)
stats = migration.migrate_full_dataset(transformation_func=clean_lims_data)
```

#### 1.4 Network Interruptions

**Diagnosis:**
```python
# Check for network errors in migration checkpoint
checkpoint_df = spark.read.format("delta").load("/mnt/checkpoint")
checkpoint_df.filter(col("status").like("%network%") | col("status").like("%connection%")).show()
```

**Solution:**
```python
# Integrate with retry queue for automatic retry
import sys
sys.path.append('/Workspace/retry_queue')

from retry_queue_system import ExponentialBackoffRetryQueue, RetryQueueConfig

# Configure retry queue
retry_config = RetryQueueConfig(
    retry_table_path="/mnt/delta/lims_retry_queue",
    final_table_path="/mnt/delta/lims_target",
    dlq_table_path="/mnt/delta/lims_dlq",
    max_retries=5,
    base_backoff_minutes=2
)

retry_processor = ExponentialBackoffRetryQueue(spark, retry_config)
retry_processor.initialize_tables()

# Process failed records through retry queue
failed_records = spark.read.format("delta").load("/mnt/migration_failed")
retry_processor.process_batch(failed_records)
```

---

## Problem 2: Slow Migration Performance

### Symptoms
- Migration takes hours instead of minutes
- Cluster resources underutilized (low CPU/memory usage)
- Throughput < 500 records/minute

### Root Causes & Solutions

#### 2.1 Insufficient Parallelism

**Diagnosis:**
```python
# Check current partition count
df = spark.read.format("delta").load("/mnt/source")
print(f"Number of partitions: {df.rdd.getNumPartitions()}")

# Check shuffle partitions
print(f"Shuffle partitions: {spark.conf.get('spark.sql.shuffle.partitions')}")
```

**Solution:**
```python
# Increase parallelism
spark.conf.set("spark.sql.shuffle.partitions", "400")  # Default is 200

# Repartition source data
df_repartitioned = df.repartition(400, "sample_id")

# Or use range partitioning for better distribution
df_repartitioned = df.repartition(400, "sample_id", "test_code")
```

#### 2.2 Small Cluster Size

**Diagnosis:**
```bash
# Check cluster configuration
# Databricks UI → Clusters → Your Cluster → Configuration
```

**Solution:**
```python
# For large LIMS migrations, recommend:
# - Worker type: Standard_D32s_v3 or better
# - Min workers: 2
# - Max workers: 10-20 (with autoscaling)
# - Databricks Runtime: 13.3 LTS or later

# If using Jobs API, configure programmatically:
cluster_config = {
    "spark_version": "13.3.x-scala2.12",
    "node_type_id": "Standard_D32s_v3",
    "autoscale": {
        "min_workers": 2,
        "max_workers": 20
    },
    "spark_conf": {
        "spark.sql.adaptive.enabled": "true",
        "spark.databricks.delta.optimizeWrite.enabled": "true"
    }
}
```

#### 2.3 Single-Threaded JDBC Reads

**Diagnosis:**
```python
# Check if JDBC read is using partitions
# If you see only 1 Spark task, JDBC is reading single-threaded
```

**Solution:**
```python
# Enable parallel JDBC reads with partitioning
df = spark.read \
    .format("jdbc") \
    .option("url", "jdbc:sqlserver://lims-server:1433;database=LIMS") \
    .option("dbtable", "lab_results") \
    .option("numPartitions", "200") \
    .option("partitionColumn", "sample_id") \
    .option("lowerBound", "1") \
    .option("upperBound", "10000000") \
    .load()

# For non-numeric partition columns, use query pushdown
query = """
    (SELECT *,
            NTILE(200) OVER (ORDER BY sample_id) as partition_id
     FROM lab_results) as partitioned_data
"""

df = spark.read \
    .format("jdbc") \
    .option("url", jdbc_url) \
    .option("dbtable", query) \
    .option("numPartitions", "200") \
    .option("partitionColumn", "partition_id") \
    .option("lowerBound", "1") \
    .option("upperBound", "200") \
    .load()
```

#### 2.4 Writing to Single Large File

**Diagnosis:**
```bash
# Check file sizes in target directory
%sh ls -lh /dbfs/mnt/delta/target/_delta_log/
%sh du -h /dbfs/mnt/delta/target/ | sort -h | tail -20
```

**Solution:**
```python
# Enable Delta Lake write optimizations
spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", "true")
spark.conf.set("spark.databricks.delta.autoCompact.enabled", "true")

# Repartition before writing
df.repartition(200).write \
    .format("delta") \
    .mode("append") \
    .save("/mnt/delta/target")
```

#### 2.5 Adaptive Query Execution (AQE) Not Enabled

**Solution:**
```python
# Enable AQE for automatic optimizations
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.sql.adaptive.localShuffleReader.enabled", "true")
```

---

## Problem 3: Out of Memory Errors

### Symptoms
- `java.lang.OutOfMemoryError: Java heap space`
- Tasks failing with executor memory errors
- Cluster crashes during migration

### Solutions

#### 3.1 Reduce Batch Size

```python
# Instead of processing entire table at once
config = MigrationConfig(
    source_path="/mnt/source",
    target_path="/mnt/target",
    batch_size=10000,  # Smaller batches = less memory
)
```

#### 3.2 Increase Executor Memory

```python
# Databricks cluster configuration
spark.conf.set("spark.executor.memory", "16g")
spark.conf.set("spark.driver.memory", "8g")
spark.conf.set("spark.executor.memoryOverhead", "4g")
```

#### 3.3 Disable Broadcast Joins for Large Tables

```python
# If joining with large dimension tables
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")

# Or explicitly use shuffle join
large_df1.join(large_df2, "sample_id", "inner").show()
```

#### 3.4 Use Streaming for Incremental Processing

```python
# For very large tables, use structured streaming
stream_df = spark.readStream \
    .format("delta") \
    .load("/mnt/source")

query = stream_df.writeStream \
    .format("delta") \
    .option("checkpointLocation", "/mnt/checkpoint") \
    .outputMode("append") \
    .start("/mnt/target")

query.awaitTermination()
```

---

## Problem 4: Connection Timeouts (JDBC)

### Symptoms
- `java.sql.SQLException: Connection timeout`
- `The driver could not establish a secure connection to SQL Server`
- Migration fails after a few minutes

### Solutions

#### 4.1 Configure Connection Pooling

```python
# Increase connection timeout
jdbc_options = {
    "url": "jdbc:sqlserver://lims-server:1433;database=LIMS",
    "dbtable": "lab_results",
    "user": dbutils.secrets.get(scope="lims", key="username"),
    "password": dbutils.secrets.get(scope="lims", key="password"),
    "loginTimeout": "300",  # 5 minutes
    "queryTimeout": "3600",  # 1 hour
    "socketTimeout": "3600000"  # 1 hour in ms
}

df = spark.read.format("jdbc").options(**jdbc_options).load()
```

#### 4.2 Use Service Principal Authentication (Azure SQL)

```python
# More reliable than SQL authentication
jdbc_url = f"jdbc:sqlserver://{server}:1433;database={database};encrypt=true;trustServerCertificate=false;hostNameInCertificate=*.database.windows.net;authentication=ActiveDirectoryServicePrincipal"

df = spark.read \
    .format("jdbc") \
    .option("url", jdbc_url) \
    .option("dbtable", table_name) \
    .option("user", client_id) \
    .option("password", client_secret) \
    .load()
```

#### 4.3 Enable Connection Retry Logic

```python
# Add retry logic for JDBC connections
from time import sleep

def read_jdbc_with_retry(spark, options, max_retries=5):
    """Read from JDBC with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            return spark.read.format("jdbc").options(**options).load()
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"JDBC read failed, retrying in {wait_time}s: {e}")
                sleep(wait_time)
            else:
                raise

df = read_jdbc_with_retry(spark, jdbc_options)
```

---

## Problem 5: Data Skew

### Symptoms
- Some tasks take 10x longer than others
- Uneven CPU utilization across executors
- A few tasks process 80% of data

### Diagnosis

```python
# Check partition distribution
from pyspark.sql.functions import spark_partition_id, count

df = spark.read.format("delta").load("/mnt/source")

df.groupBy(spark_partition_id()).count().orderBy("count", ascending=False).show(20)
```

### Solutions

#### 5.1 Salt Skewed Keys

```python
from pyspark.sql.functions import concat_ws, hash, abs as spark_abs

# Add salt to skewed column
df_salted = df.withColumn(
    "test_code_salted",
    concat_ws("_", col("test_code"), (spark_abs(hash(col("sample_id"))) % 100))
)

# Use salted column for partitioning
df_salted.repartition(200, "test_code_salted").write.format("delta").save("/mnt/target")
```

#### 5.2 Enable AQE Skew Join Optimization

```python
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "5")
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256MB")
```

#### 5.3 Use Iterative Broadcasting

```python
# For joins with skewed keys
from pyspark.sql.functions import broadcast

# Split data into skewed and non-skewed
skewed_values = ["TEST_CODE_A", "TEST_CODE_B"]  # Identify from profiling

df_skewed = df.filter(col("test_code").isin(skewed_values))
df_normal = df.filter(~col("test_code").isin(skewed_values))

# Process separately
result_normal = df_normal.join(broadcast(dim_table), "test_code")
result_skewed = df_skewed.join(dim_table, "test_code")  # No broadcast

result = result_normal.union(result_skewed)
```

---

## Problem 6: Delta Lake Errors

### Common Delta Errors

#### 6.1 ConcurrentAppendException

**Error:**
```
org.apache.spark.sql.delta.ConcurrentAppendException: Files were added to the root of the table by a concurrent update
```

**Solution:**
```python
# Enable optimistic concurrency control
spark.conf.set("spark.databricks.delta.optimisticTransaction.enabled", "true")

# Use merge instead of append for concurrent writes
from delta.tables import DeltaTable

target_table = DeltaTable.forPath(spark, "/mnt/delta/target")

target_table.alias("target").merge(
    new_data.alias("source"),
    "target.sample_id = source.sample_id"
).whenNotMatchedInsertAll().execute()
```

#### 6.2 Protocol Version Errors

**Error:**
```
Unsupported Delta table version: minReaderVersion 2
```

**Solution:**
```python
# Upgrade Delta table protocol
from delta.tables import DeltaTable

delta_table = DeltaTable.forPath(spark, "/mnt/delta/target")
delta_table.upgradeTableProtocol(1, 3)
```

#### 6.3 Too Many Small Files

**Diagnosis:**
```python
# Check file count
file_count = spark.sql(f"SELECT COUNT(*) FROM delta.`/mnt/delta/target`").first()[0]
file_list = dbutils.fs.ls("/mnt/delta/target")
print(f"Number of files: {len(file_list)}")
```

**Solution:**
```python
# Run OPTIMIZE to compact small files
from delta.tables import DeltaTable

delta_table = DeltaTable.forPath(spark, "/mnt/delta/target")
delta_table.optimize().executeCompaction()

# Enable auto-optimize for future writes
spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", "true")
spark.conf.set("spark.databricks.delta.autoCompact.enabled", "true")
```

---

## Problem 7: Duplicate Records

### Symptoms
- Target table has more records than source
- Same `sample_id` appears multiple times

### Diagnosis

```python
# Find duplicates
df = spark.read.format("delta").load("/mnt/delta/target")

duplicates = df.groupBy("sample_id").count().filter("count > 1")
duplicate_count = duplicates.count()

print(f"Number of duplicate sample_ids: {duplicate_count}")
duplicates.show(20)
```

### Solutions

#### 7.1 Deduplicate During Migration

```python
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, desc

# Keep only latest record per sample_id
window_spec = Window.partitionBy("sample_id").orderBy(desc("modified_date"))

df_deduped = df.withColumn("row_num", row_number().over(window_spec)) \
    .filter(col("row_num") == 1) \
    .drop("row_num")

df_deduped.write.format("delta").mode("overwrite").save("/mnt/delta/target")
```

#### 7.2 Use MERGE for Upsert Semantics

```python
from delta.tables import DeltaTable

target_table = DeltaTable.forPath(spark, "/mnt/delta/target")

target_table.alias("target").merge(
    new_data.alias("source"),
    "target.sample_id = source.sample_id"
).whenMatchedUpdateAll() \
 .whenNotMatchedInsertAll() \
 .execute()
```

---

## Azure Databricks-Specific Issues

### Issue 1: Mounting Azure Storage

**Problem:** Can't read from `/mnt/` paths

**Solution:**
```python
# Mount Azure Data Lake Storage Gen2
configs = {
    "fs.azure.account.auth.type": "OAuth",
    "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
    "fs.azure.account.oauth2.client.id": client_id,
    "fs.azure.account.oauth2.client.secret": client_secret,
    "fs.azure.account.oauth2.client.endpoint": f"https://login.microsoftonline.com/{tenant_id}/oauth2/token"
}

dbutils.fs.mount(
    source=f"abfss://container@storageaccount.dfs.core.windows.net/",
    mount_point="/mnt/datalake",
    extra_configs=configs
)
```

### Issue 2: Access Secrets

**Solution:**
```python
# Store credentials in Databricks Secrets
# CLI: databricks secrets create-scope --scope lims
# CLI: databricks secrets put --scope lims --key jdbc-password

jdbc_password = dbutils.secrets.get(scope="lims", key="jdbc-password")
```

### Issue 3: Job Fails with Exit Code 1

**Diagnosis:**
```bash
# Check driver logs
%sh tail -100 /databricks/driver/logs/stderr

# Check executor logs in Spark UI
```

**Common Causes:**
- Out of memory
- Missing dependencies
- Authentication failures

---

## Performance Optimization Checklist

Use this checklist for every LIMS migration:

### Pre-Migration

- [ ] Profile source data (row count, size, skew)
- [ ] Choose appropriate cluster size
- [ ] Configure mount points for Azure storage
- [ ] Test connectivity to source database
- [ ] Create sample data for testing

### During Migration

- [ ] Enable Adaptive Query Execution (AQE)
- [ ] Configure parallelism (shuffle partitions)
- [ ] Use checkpointing for fault tolerance
- [ ] Monitor migration progress
- [ ] Watch for skew in Spark UI

### Post-Migration

- [ ] Run OPTIMIZE on Delta tables
- [ ] Apply Z-ORDER for query performance
- [ ] Validate record counts
- [ ] Check data quality
- [ ] Run VACUUM to clean up old files

### Spark Configuration (Copy-Paste Ready)

```python
# Optimal configuration for LIMS migration on Databricks
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.sql.shuffle.partitions", "400")
spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", "true")
spark.conf.set("spark.databricks.delta.autoCompact.enabled", "true")
spark.conf.set("spark.sql.files.maxPartitionBytes", "134217728")  # 128MB
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10485760")  # 10MB
```

---

## Quick Reference Commands

### Check Migration Status
```python
# Count comparison
source_count = spark.read.format("delta").load("/mnt/source").count()
target_count = spark.read.format("delta").load("/mnt/target").count()
print(f"Progress: {target_count}/{source_count} ({target_count/source_count*100:.1f}%)")
```

### Find Missing Records
```python
source_df = spark.read.format("delta").load("/mnt/source")
target_df = spark.read.format("delta").load("/mnt/target")

missing = source_df.join(target_df.select("sample_id"), "sample_id", "left_anti")
print(f"Missing: {missing.count()}")
```

### Check Performance
```python
# View Spark UI for detailed metrics
displayHTML(f'<a href="/driver-proxy-api/o/0/{spark.sparkContext.uiWebUrl.split("http://")[-1]}" target="_blank">Open Spark UI</a>')
```

---

## Getting Help

If issues persist:

1. **Check Databricks documentation**: [docs.databricks.com](https://docs.databricks.com)
2. **Review Spark UI**: Look for failed tasks, skew, memory issues
3. **Enable DEBUG logging**:
   ```python
   spark.sparkContext.setLogLevel("DEBUG")
   ```
4. **Contact support**: Provide cluster ID, job ID, and error logs

---

**Last Updated:** 2025-11-24
**Version:** 1.0.0
