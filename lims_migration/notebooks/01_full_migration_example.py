# Databricks notebook source
# MAGIC %md
# MAGIC # LIMS Full Data Migration - Example Notebook
# MAGIC
# MAGIC This notebook demonstrates a complete LIMS data migration from SQL Server to Delta Lake.
# MAGIC
# MAGIC **Use Case:** Migrate entire LIMS database to Azure Data Lake
# MAGIC
# MAGIC **Steps:**
# MAGIC 1. Configure connection to source LIMS database
# MAGIC 2. Run diagnostics to assess data
# MAGIC 3. Execute migration with optimizations
# MAGIC 4. Validate results
# MAGIC 5. Optimize Delta tables

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

# Import required libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from datetime import datetime
import sys

# Add custom modules to path
sys.path.append('/Workspace/Shared/lims_migration')

from lims_migration_diagnostics import LIMSMigrationDiagnostics, MigrationHealthMonitor
from optimized_migration_pipeline import OptimizedLIMSMigration, MigrationConfig

# COMMAND ----------

# Configuration parameters (use widgets for flexibility)
dbutils.widgets.text("source_server", "lims-sql-server.database.windows.net")
dbutils.widgets.text("source_database", "LIMS_PROD")
dbutils.widgets.text("source_table", "lab_results")
dbutils.widgets.text("target_path", "/mnt/datalake/lims/lab_results")
dbutils.widgets.dropdown("migration_mode", "full", ["full", "incremental", "resume"])

# Get widget values
source_server = dbutils.widgets.get("source_server")
source_database = dbutils.widgets.get("source_database")
source_table = dbutils.widgets.get("source_table")
target_path = dbutils.widgets.get("target_path")
migration_mode = dbutils.widgets.get("migration_mode")

# Derived paths
checkpoint_path = f"{target_path}_checkpoint"
failed_records_path = f"{target_path}_failed"

print(f"📋 Migration Configuration:")
print(f"   Source: {source_server}/{source_database}/{source_table}")
print(f"   Target: {target_path}")
print(f"   Mode: {migration_mode}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Connect to Source Database

# COMMAND ----------

# Retrieve credentials from Databricks Secrets
# Setup: databricks secrets create-scope --scope lims
#        databricks secrets put --scope lims --key sql-username
#        databricks secrets put --scope lims --key sql-password

username = dbutils.secrets.get(scope="lims", key="sql-username")
password = dbutils.secrets.get(scope="lims", key="sql-password")

# Build JDBC URL
jdbc_url = f"jdbc:sqlserver://{source_server}:1433;" \
           f"database={source_database};" \
           f"encrypt=true;" \
           f"trustServerCertificate=false;" \
           f"hostNameInCertificate=*.database.windows.net;" \
           f"loginTimeout=30;"

# Test connection
try:
    test_query = f"(SELECT TOP 10 * FROM {source_table}) as test"
    test_df = spark.read \
        .format("jdbc") \
        .option("url", jdbc_url) \
        .option("dbtable", test_query) \
        .option("user", username) \
        .option("password", password) \
        .load()

    print(f"✅ Connection successful! Sample data:")
    test_df.show(5)
except Exception as e:
    print(f"❌ Connection failed: {e}")
    dbutils.notebook.exit({"status": "failed", "error": str(e)})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Pre-Migration Diagnostics

# COMMAND ----------

# Get source data profile
source_query = f"(SELECT * FROM {source_table}) as source_data"

source_df = spark.read \
    .format("jdbc") \
    .option("url", jdbc_url) \
    .option("dbtable", source_query) \
    .option("user", username) \
    .option("password", password) \
    .option("numPartitions", "10") \
    .option("partitionColumn", "sample_id") \
    .option("lowerBound", "1") \
    .option("upperBound", "10000000") \
    .load()

# Profile source data
print("📊 SOURCE DATA PROFILE:")
print(f"   Total rows: {source_df.count():,}")
print(f"   Columns: {len(source_df.columns)}")
print(f"   Partitions: {source_df.rdd.getNumPartitions()}")
print(f"\n📋 Schema:")
source_df.printSchema()

# Check for data quality issues
print("\n🔬 DATA QUALITY CHECK:")

# Null counts
null_counts = source_df.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in source_df.columns
])
print("Null counts:")
null_counts.show()

# Duplicate sample_ids
duplicate_count = source_df.groupBy("sample_id").count().filter("count > 1").count()
print(f"Duplicate sample_ids: {duplicate_count}")

# Data distribution (check for skew)
print("\n📈 Top 10 Test Types (checking for skew):")
source_df.groupBy("test_code").count().orderBy(col("count").desc()).show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Configure Migration Pipeline

# COMMAND ----------

# Determine optimal configuration based on data size
source_count = source_df.count()

if source_count < 100000:
    # Small dataset
    batch_size = 10000
    num_partitions = 50
elif source_count < 1000000:
    # Medium dataset
    batch_size = 50000
    num_partitions = 200
else:
    # Large dataset
    batch_size = 100000
    num_partitions = 400

print(f"📊 Optimizing for {source_count:,} records:")
print(f"   Batch size: {batch_size:,}")
print(f"   Partitions: {num_partitions}")

# Create migration configuration
config = MigrationConfig(
    source_path=f"jdbc:{jdbc_url};table={source_table};user={username};password={password}",
    target_path=target_path,
    checkpoint_path=checkpoint_path,
    failed_records_path=failed_records_path,
    batch_size=batch_size,
    num_partitions=num_partitions,
    enable_adaptive_execution=True,
    enable_z_ordering=True,
    z_order_columns=["test_code", "collection_date"]
)

# Initialize migration pipeline
migration = OptimizedLIMSMigration(spark, config)

print("✅ Migration pipeline configured")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Define Data Transformations (Optional)

# COMMAND ----------

def transform_lims_data(df):
    """
    Apply LIMS-specific transformations during migration.

    Examples:
    - Standardize date formats
    - Clean text fields
    - Derive calculated columns
    - Handle NULL values
    """
    from pyspark.sql.functions import (
        coalesce, lit, trim, upper, to_timestamp, regexp_replace
    )

    return df \
        .withColumn("sample_id", trim(col("sample_id"))) \
        .withColumn("test_code", upper(trim(col("test_code")))) \
        .withColumn("collection_date",
                    to_timestamp(col("collection_date"), "yyyy-MM-dd HH:mm:ss")) \
        .withColumn("result_value",
                    regexp_replace(col("result_value"), "[^0-9.]", "")) \
        .withColumn("migrated_at", current_timestamp())

# Test transformation on sample
print("🧪 Testing transformation on sample data:")
transformed_sample = transform_lims_data(test_df)
transformed_sample.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Execute Migration

# COMMAND ----------

print("=" * 70)
print("🚀 STARTING LIMS DATA MIGRATION")
print("=" * 70)

migration_start_time = datetime.now()

try:
    if migration_mode == "full":
        # Full migration
        print("📦 Mode: FULL MIGRATION")
        stats = migration.migrate_full_dataset(
            transformation_func=transform_lims_data
        )

    elif migration_mode == "incremental":
        # Incremental migration
        print("🔄 Mode: INCREMENTAL MIGRATION")
        stats = migration.migrate_incremental(
            watermark_column="modified_date"
        )

    elif migration_mode == "resume":
        # Resume failed batches
        print("⏯️  Mode: RESUME FAILED BATCHES")
        migration.resume_failed_batches()
        stats = {"mode": "resume"}

    print("\n✅ MIGRATION COMPLETED SUCCESSFULLY")
    print(f"📊 Statistics: {stats}")

except Exception as e:
    print(f"\n❌ MIGRATION FAILED: {e}")
    import traceback
    traceback.print_exc()

    dbutils.notebook.exit({
        "status": "failed",
        "error": str(e),
        "timestamp": str(datetime.now())
    })

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Post-Migration Validation

# COMMAND ----------

print("🔍 VALIDATING MIGRATION RESULTS...")

# Initialize diagnostics
diagnostics = LIMSMigrationDiagnostics(spark)

# Create temporary source Delta table for comparison
# (In production, source would be the actual SQL Server table)
print("\n📊 Comparing source and target counts...")

source_count_actual = source_df.count()
target_df = spark.read.format("delta").load(target_path)
target_count_actual = target_df.count()

print(f"   Source: {source_count_actual:,}")
print(f"   Target: {target_count_actual:,}")
print(f"   Missing: {source_count_actual - target_count_actual:,}")
print(f"   Success Rate: {(target_count_actual / source_count_actual * 100):.2f}%")

# Validate data quality in target
print("\n🔬 Checking target data quality...")
quality_results = diagnostics.check_data_quality(target_path)

# Sample target data
print("\n📋 Sample migrated records:")
target_df.select(
    "sample_id", "test_code", "result_value", "collection_date", "migrated_at"
).show(10, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Optimize Delta Table

# COMMAND ----------

print("🔧 OPTIMIZING DELTA TABLE...")

from delta.tables import DeltaTable

target_table = DeltaTable.forPath(spark, target_path)

# Run OPTIMIZE for compaction
print("⚡ Running OPTIMIZE (compaction)...")
target_table.optimize().executeCompaction()

# Run Z-ORDER for query performance
print("📊 Running Z-ORDER on test_code and collection_date...")
spark.sql(f"""
    OPTIMIZE delta.`{target_path}`
    ZORDER BY (test_code, collection_date)
""")

# Get table statistics
print("\n📈 Target Table Statistics:")
display(spark.sql(f"DESCRIBE DETAIL delta.`{target_path}`"))

print("\n✅ Optimization complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Generate Migration Report

# COMMAND ----------

migration_end_time = datetime.now()
duration_minutes = (migration_end_time - migration_start_time).total_seconds() / 60

# Generate comprehensive report
report = {
    "migration_id": f"lims_{migration_start_time.strftime('%Y%m%d_%H%M%S')}",
    "source": f"{source_server}/{source_database}/{source_table}",
    "target": target_path,
    "mode": migration_mode,
    "start_time": str(migration_start_time),
    "end_time": str(migration_end_time),
    "duration_minutes": duration_minutes,
    "source_count": source_count_actual,
    "target_count": target_count_actual,
    "success_rate": (target_count_actual / source_count_actual * 100) if source_count_actual > 0 else 0,
    "throughput_per_minute": source_count_actual / duration_minutes if duration_minutes > 0 else 0,
    "status": "completed"
}

print("\n" + "=" * 70)
print("📋 MIGRATION REPORT")
print("=" * 70)
for key, value in report.items():
    print(f"{key:25s}: {value}")
print("=" * 70)

# Save report to Delta table
report_df = spark.createDataFrame([report])
report_path = "/mnt/datalake/lims/migration_reports"
report_df.write.format("delta").mode("append").save(report_path)

print(f"\n💾 Report saved to: {report_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Exit Notebook with Status

# COMMAND ----------

# Return status for job orchestration
dbutils.notebook.exit(report)
