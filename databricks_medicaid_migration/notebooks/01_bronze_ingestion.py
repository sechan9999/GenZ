# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Layer: Raw Claims Data Ingestion
# MAGIC
# MAGIC This notebook ingests raw Medicaid claims data into Delta Lake bronze layer.
# MAGIC
# MAGIC **Data Sources:**
# MAGIC - Claims data (medical, pharmacy, dental)
# MAGIC - Member eligibility data
# MAGIC - Provider data
# MAGIC - Diagnosis and procedure codes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and Configuration

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from delta.tables import DeltaTable
import json
from datetime import datetime, timedelta

# Configure Spark for optimal performance
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", "true")
spark.conf.set("spark.databricks.delta.autoCompact.enabled", "true")

# Define paths
BRONZE_PATH = "/mnt/medicaid/bronze"
LANDING_PATH = "/mnt/medicaid/landing"
CHECKPOINT_PATH = "/mnt/medicaid/checkpoints/bronze"

print("✓ Configuration loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Data Schemas

# COMMAND ----------

# Medical Claims Schema
medical_claims_schema = StructType([
    StructField("claim_id", StringType(), False),
    StructField("member_id", StringType(), False),
    StructField("provider_id", StringType(), True),
    StructField("claim_date", DateType(), False),
    StructField("service_from_date", DateType(), False),
    StructField("service_to_date", DateType(), True),
    StructField("admit_date", DateType(), True),
    StructField("discharge_date", DateType(), True),
    StructField("diagnosis_code_1", StringType(), True),
    StructField("diagnosis_code_2", StringType(), True),
    StructField("diagnosis_code_3", StringType(), True),
    StructField("diagnosis_code_4", StringType(), True),
    StructField("procedure_code_1", StringType(), True),
    StructField("procedure_code_2", StringType(), True),
    StructField("claim_type", StringType(), True),  # IP, OP, ER, etc.
    StructField("claim_status", StringType(), True),
    StructField("allowed_amount", DecimalType(10, 2), True),
    StructField("paid_amount", DecimalType(10, 2), True),
    StructField("member_liability", DecimalType(10, 2), True),
    StructField("service_units", IntegerType(), True),
    StructField("drg_code", StringType(), True),
    StructField("revenue_code", StringType(), True),
    StructField("place_of_service", StringType(), True),
])

# Pharmacy Claims Schema
pharmacy_claims_schema = StructType([
    StructField("claim_id", StringType(), False),
    StructField("member_id", StringType(), False),
    StructField("pharmacy_id", StringType(), True),
    StructField("prescriber_id", StringType(), True),
    StructField("fill_date", DateType(), False),
    StructField("ndc_code", StringType(), True),
    StructField("drug_name", StringType(), True),
    StructField("days_supply", IntegerType(), True),
    StructField("quantity", DecimalType(10, 2), True),
    StructField("refill_number", IntegerType(), True),
    StructField("allowed_amount", DecimalType(10, 2), True),
    StructField("paid_amount", DecimalType(10, 2), True),
    StructField("member_copay", DecimalType(10, 2), True),
    StructField("generic_indicator", StringType(), True),
    StructField("therapeutic_class", StringType(), True),
])

# Member Eligibility Schema
member_eligibility_schema = StructType([
    StructField("member_id", StringType(), False),
    StructField("effective_date", DateType(), False),
    StructField("term_date", DateType(), True),
    StructField("birth_date", DateType(), True),
    StructField("gender", StringType(), True),
    StructField("race", StringType(), True),
    StructField("ethnicity", StringType(), True),
    StructField("zip_code", StringType(), True),
    StructField("county", StringType(), True),
    StructField("eligibility_category", StringType(), True),
    StructField("aid_category", StringType(), True),
    StructField("dual_eligible", BooleanType(), True),
    StructField("ltss_indicator", BooleanType(), True),
])

# Provider Data Schema
provider_schema = StructType([
    StructField("provider_id", StringType(), False),
    StructField("provider_name", StringType(), True),
    StructField("provider_type", StringType(), True),
    StructField("specialty", StringType(), True),
    StructField("npi", StringType(), True),
    StructField("taxonomy_code", StringType(), True),
    StructField("address_line1", StringType(), True),
    StructField("city", StringType(), True),
    StructField("state", StringType(), True),
    StructField("zip_code", StringType(), True),
])

print("✓ Schemas defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingestion Functions

# COMMAND ----------

def ingest_to_bronze(source_path: str, target_table: str, schema: StructType,
                      partition_cols: list = None, merge_keys: list = None):
    """
    Ingest data from landing zone to bronze layer with incremental processing.

    Args:
        source_path: Path to source data files
        target_table: Target Delta table name
        schema: Expected schema for the data
        partition_cols: Columns to partition by
        merge_keys: Keys for merge operation (upsert)
    """
    print(f"Starting ingestion for {target_table}...")

    # Read source data
    df = (spark.readStream
          .format("cloudFiles")
          .option("cloudFiles.format", "csv")
          .option("cloudFiles.schemaLocation", f"{CHECKPOINT_PATH}/{target_table}/schema")
          .option("cloudFiles.inferColumnTypes", "true")
          .option("header", "true")
          .schema(schema)
          .load(source_path))

    # Add metadata columns
    df = (df
          .withColumn("ingestion_timestamp", current_timestamp())
          .withColumn("source_file", input_file_name())
          .withColumn("ingestion_date", current_date()))

    # Write to Delta Lake
    query = (df.writeStream
             .format("delta")
             .outputMode("append")
             .option("checkpointLocation", f"{CHECKPOINT_PATH}/{target_table}")
             .trigger(availableNow=True))

    if partition_cols:
        query = query.partitionBy(partition_cols)

    query.toTable(f"bronze.{target_table}")

    print(f"✓ Ingestion completed for {target_table}")


def batch_ingest_to_bronze(source_path: str, target_table: str, schema: StructType,
                           partition_cols: list = None, mode: str = "append"):
    """
    Batch ingestion for static or historical data.

    Args:
        source_path: Path to source data files
        target_table: Target Delta table name
        schema: Expected schema
        partition_cols: Partition columns
        mode: Write mode (append, overwrite)
    """
    print(f"Starting batch ingestion for {target_table}...")

    # Read source data
    df = (spark.read
          .format("csv")
          .option("header", "true")
          .schema(schema)
          .load(source_path))

    # Add metadata
    df = (df
          .withColumn("ingestion_timestamp", current_timestamp())
          .withColumn("ingestion_date", current_date()))

    # Write to Delta
    writer = df.write.format("delta").mode(mode)

    if partition_cols:
        writer = writer.partitionBy(partition_cols)

    writer.saveAsTable(f"bronze.{target_table}")

    record_count = df.count()
    print(f"✓ Batch ingestion completed for {target_table}: {record_count:,} records")

    return record_count

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Bronze Database and Tables

# COMMAND ----------

# Create database
spark.sql("CREATE DATABASE IF NOT EXISTS bronze")
spark.sql("USE bronze")

print("✓ Bronze database ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest Medical Claims

# COMMAND ----------

# Batch load historical medical claims
medical_claims_count = batch_ingest_to_bronze(
    source_path=f"{LANDING_PATH}/medical_claims/historical/*.csv",
    target_table="medical_claims",
    schema=medical_claims_schema,
    partition_cols=["claim_date"],
    mode="overwrite"
)

print(f"Medical claims ingested: {medical_claims_count:,}")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Verify medical claims ingestion
# MAGIC SELECT
# MAGIC   COUNT(*) as total_claims,
# MAGIC   COUNT(DISTINCT member_id) as unique_members,
# MAGIC   MIN(claim_date) as earliest_claim,
# MAGIC   MAX(claim_date) as latest_claim,
# MAGIC   SUM(paid_amount) as total_paid
# MAGIC FROM bronze.medical_claims

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest Pharmacy Claims

# COMMAND ----------

pharmacy_claims_count = batch_ingest_to_bronze(
    source_path=f"{LANDING_PATH}/pharmacy_claims/historical/*.csv",
    target_table="pharmacy_claims",
    schema=pharmacy_claims_schema,
    partition_cols=["fill_date"],
    mode="overwrite"
)

print(f"Pharmacy claims ingested: {pharmacy_claims_count:,}")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Verify pharmacy claims
# MAGIC SELECT
# MAGIC   COUNT(*) as total_rx,
# MAGIC   COUNT(DISTINCT member_id) as unique_members,
# MAGIC   COUNT(DISTINCT ndc_code) as unique_drugs,
# MAGIC   AVG(days_supply) as avg_days_supply,
# MAGIC   SUM(paid_amount) as total_paid
# MAGIC FROM bronze.pharmacy_claims

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest Member Eligibility

# COMMAND ----------

eligibility_count = batch_ingest_to_bronze(
    source_path=f"{LANDING_PATH}/member_eligibility/*.csv",
    target_table="member_eligibility",
    schema=member_eligibility_schema,
    partition_cols=["effective_date"],
    mode="overwrite"
)

print(f"Eligibility records ingested: {eligibility_count:,}")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Member demographics summary
# MAGIC SELECT
# MAGIC   COUNT(DISTINCT member_id) as total_members,
# MAGIC   COUNT(CASE WHEN dual_eligible THEN 1 END) as dual_eligible_count,
# MAGIC   COUNT(CASE WHEN ltss_indicator THEN 1 END) as ltss_count,
# MAGIC   ROUND(AVG(DATEDIFF(CURRENT_DATE(), birth_date) / 365.25), 1) as avg_age
# MAGIC FROM bronze.member_eligibility
# MAGIC WHERE term_date IS NULL OR term_date >= CURRENT_DATE()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest Provider Data

# COMMAND ----------

provider_count = batch_ingest_to_bronze(
    source_path=f"{LANDING_PATH}/providers/*.csv",
    target_table="providers",
    schema=provider_schema,
    mode="overwrite"
)

print(f"Provider records ingested: {provider_count:,}")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Provider summary
# MAGIC SELECT
# MAGIC   provider_type,
# MAGIC   COUNT(*) as provider_count
# MAGIC FROM bronze.providers
# MAGIC GROUP BY provider_type
# MAGIC ORDER BY provider_count DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Checks

# COMMAND ----------

def run_data_quality_checks():
    """Run basic data quality checks on bronze tables."""

    checks = []

    # Check 1: No null primary keys in medical claims
    null_claim_ids = spark.sql("""
        SELECT COUNT(*) as null_count
        FROM bronze.medical_claims
        WHERE claim_id IS NULL
    """).collect()[0]['null_count']

    checks.append({
        'check': 'Medical Claims - No Null Claim IDs',
        'passed': null_claim_ids == 0,
        'detail': f"{null_claim_ids} null values found"
    })

    # Check 2: Valid date ranges
    invalid_dates = spark.sql("""
        SELECT COUNT(*) as invalid_count
        FROM bronze.medical_claims
        WHERE service_from_date > service_to_date
        OR claim_date < '2020-01-01'
        OR claim_date > CURRENT_DATE()
    """).collect()[0]['invalid_count']

    checks.append({
        'check': 'Medical Claims - Valid Date Ranges',
        'passed': invalid_dates == 0,
        'detail': f"{invalid_dates} invalid dates found"
    })

    # Check 3: Member IDs exist in eligibility
    orphan_members = spark.sql("""
        SELECT COUNT(DISTINCT mc.member_id) as orphan_count
        FROM bronze.medical_claims mc
        LEFT JOIN bronze.member_eligibility me ON mc.member_id = me.member_id
        WHERE me.member_id IS NULL
    """).collect()[0]['orphan_count']

    checks.append({
        'check': 'Member IDs Reference Integrity',
        'passed': orphan_members == 0,
        'detail': f"{orphan_members} members without eligibility records"
    })

    # Check 4: Reasonable claim amounts
    extreme_amounts = spark.sql("""
        SELECT COUNT(*) as extreme_count
        FROM bronze.medical_claims
        WHERE paid_amount < 0 OR paid_amount > 1000000
    """).collect()[0]['extreme_count']

    checks.append({
        'check': 'Medical Claims - Reasonable Amounts',
        'passed': extreme_amounts == 0,
        'detail': f"{extreme_amounts} claims with extreme amounts"
    })

    # Display results
    print("\n=== Data Quality Check Results ===\n")
    for check in checks:
        status = "✓ PASS" if check['passed'] else "✗ FAIL"
        print(f"{status}: {check['check']}")
        print(f"   Detail: {check['detail']}\n")

    return checks

# Run checks
quality_results = run_data_quality_checks()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimize Bronze Tables

# COMMAND ----------

# Optimize tables for better query performance
tables_to_optimize = [
    "medical_claims",
    "pharmacy_claims",
    "member_eligibility",
    "providers"
]

for table in tables_to_optimize:
    print(f"Optimizing {table}...")
    spark.sql(f"OPTIMIZE bronze.{table}")

    # Z-order by commonly filtered columns
    if table == "medical_claims":
        spark.sql(f"OPTIMIZE bronze.{table} ZORDER BY (member_id, provider_id, claim_date)")
    elif table == "pharmacy_claims":
        spark.sql(f"OPTIMIZE bronze.{table} ZORDER BY (member_id, fill_date)")
    elif table == "member_eligibility":
        spark.sql(f"OPTIMIZE bronze.{table} ZORDER BY (member_id)")

    print(f"✓ {table} optimized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingestion Summary

# COMMAND ----------

# Generate ingestion summary
summary = spark.sql("""
SELECT
    'Medical Claims' as data_source,
    COUNT(*) as record_count,
    COUNT(DISTINCT member_id) as unique_members,
    MIN(claim_date) as min_date,
    MAX(claim_date) as max_date
FROM bronze.medical_claims

UNION ALL

SELECT
    'Pharmacy Claims' as data_source,
    COUNT(*) as record_count,
    COUNT(DISTINCT member_id) as unique_members,
    MIN(fill_date) as min_date,
    MAX(fill_date) as max_date
FROM bronze.pharmacy_claims

UNION ALL

SELECT
    'Member Eligibility' as data_source,
    COUNT(*) as record_count,
    COUNT(DISTINCT member_id) as unique_members,
    MIN(effective_date) as min_date,
    MAX(effective_date) as max_date
FROM bronze.member_eligibility

UNION ALL

SELECT
    'Providers' as data_source,
    COUNT(*) as record_count,
    NULL as unique_members,
    NULL as min_date,
    NULL as max_date
FROM bronze.providers
""")

display(summary)

# COMMAND ----------

print("=" * 60)
print("BRONZE LAYER INGESTION COMPLETE")
print("=" * 60)
print("\nAll raw data successfully loaded into Delta Lake bronze layer.")
print("Ready for silver layer transformation.")
