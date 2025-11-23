# Databricks notebook source
# MAGIC %md
# MAGIC # Silver Layer: Data Cleaning and Validation
# MAGIC
# MAGIC This notebook cleans and validates bronze layer data, creating high-quality silver tables.
# MAGIC
# MAGIC **Transformations:**
# MAGIC - Data quality validation and cleansing
# MAGIC - Deduplication
# MAGIC - Standardization and normalization
# MAGIC - Business rule validation
# MAGIC - Derived fields creation

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import *
from pyspark.sql.types import *
from delta.tables import DeltaTable
import re

# Paths
BRONZE_PATH = "/mnt/medicaid/bronze"
SILVER_PATH = "/mnt/medicaid/silver"

# Create silver database
spark.sql("CREATE DATABASE IF NOT EXISTS silver")
spark.sql("USE silver")

print("✓ Setup complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions for Data Cleaning

# COMMAND ----------

def standardize_gender(gender_col):
    """Standardize gender values."""
    return when(upper(col(gender_col)).isin(['M', 'MALE', '1']), 'M')\
           .when(upper(col(gender_col)).isin(['F', 'FEMALE', '2']), 'F')\
           .when(upper(col(gender_col)).isin(['U', 'UNKNOWN', 'UNK', 'O']), 'U')\
           .otherwise('U')


def clean_zip_code(zip_col):
    """Clean and validate ZIP codes (5 or 9 digit)."""
    return when(regexp_extract(col(zip_col), r'^(\d{5})(-\d{4})?$', 1) != '',
                regexp_extract(col(zip_col), r'^(\d{5})(-\d{4})?$', 1))\
           .otherwise(None)


def calculate_age(birth_date_col, reference_date='current_date()'):
    """Calculate age from birth date."""
    return floor(datediff(eval(reference_date), col(birth_date_col)) / 365.25)


def flag_duplicate_claims(df, partition_cols, order_col):
    """Flag duplicate claims, keeping most recent."""
    window_spec = Window.partitionBy(partition_cols).orderBy(col(order_col).desc())
    return df.withColumn("row_num", row_number().over(window_spec))\
             .withColumn("is_duplicate", when(col("row_num") > 1, True).otherwise(False))


def validate_date_range(from_date_col, to_date_col):
    """Validate that from_date <= to_date."""
    return when(col(to_date_col).isNull(), True)\
           .when(col(from_date_col) <= col(to_date_col), True)\
           .otherwise(False)


print("✓ Helper functions loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean Medical Claims

# COMMAND ----------

# Read bronze medical claims
medical_claims_bronze = spark.table("bronze.medical_claims")

# Apply cleaning transformations
medical_claims_silver = (
    medical_claims_bronze
    # Remove exact duplicates
    .dropDuplicates()

    # Flag logical duplicates (same member, provider, service date)
    .transform(lambda df: flag_duplicate_claims(
        df,
        partition_cols=["member_id", "provider_id", "service_from_date", "claim_type"],
        order_col="ingestion_timestamp"
    ))

    # Filter out duplicates
    .filter(col("is_duplicate") == False)
    .drop("row_num", "is_duplicate")

    # Validate date ranges
    .withColumn("valid_service_dates",
                validate_date_range("service_from_date", "service_to_date"))
    .withColumn("valid_admission_dates",
                validate_date_range("admit_date", "discharge_date"))

    # Correct invalid date ranges (swap if reversed)
    .withColumn("service_from_date_clean",
                when(col("valid_service_dates"), col("service_from_date"))
                .otherwise(least(col("service_from_date"), col("service_to_date"))))
    .withColumn("service_to_date_clean",
                when(col("valid_service_dates"), col("service_to_date"))
                .when(col("service_to_date").isNotNull(),
                      greatest(col("service_from_date"), col("service_to_date")))
                .otherwise(None))

    # Standardize claim type
    .withColumn("claim_type_clean",
                upper(trim(col("claim_type"))))

    # Standardize claim status
    .withColumn("claim_status_clean",
                upper(trim(col("claim_status"))))

    # Calculate length of stay for inpatient
    .withColumn("length_of_stay",
                when(col("claim_type_clean") == "IP",
                     datediff(col("discharge_date"), col("admit_date")) + 1)
                .otherwise(None))

    # Validate amounts (set negative to null)
    .withColumn("allowed_amount_clean",
                when(col("allowed_amount") >= 0, col("allowed_amount")).otherwise(None))
    .withColumn("paid_amount_clean",
                when(col("paid_amount") >= 0, col("paid_amount")).otherwise(None))
    .withColumn("member_liability_clean",
                when(col("member_liability") >= 0, col("member_liability")).otherwise(None))

    # Create derived fields
    .withColumn("claim_year", year(col("claim_date")))
    .withColumn("claim_month", month(col("claim_date")))
    .withColumn("claim_quarter", quarter(col("claim_date")))
    .withColumn("service_year", year(col("service_from_date_clean")))

    # Flag high-cost claims (> $10,000)
    .withColumn("high_cost_flag",
                when(col("paid_amount_clean") > 10000, True).otherwise(False))

    # Flag emergency claims
    .withColumn("emergency_flag",
                when(col("claim_type_clean") == "ER", True)
                .when(col("revenue_code").like("045%"), True)  # ER revenue codes
                .otherwise(False))

    # Flag inpatient admissions
    .withColumn("inpatient_flag",
                when(col("claim_type_clean") == "IP", True).otherwise(False))

    # Data quality score (percentage of filled key fields)
    .withColumn("data_quality_score",
                (when(col("provider_id").isNotNull(), 1).otherwise(0) +
                 when(col("diagnosis_code_1").isNotNull(), 1).otherwise(0) +
                 when(col("procedure_code_1").isNotNull(), 1).otherwise(0) +
                 when(col("paid_amount_clean").isNotNull(), 1).otherwise(0) +
                 when(col("claim_status_clean").isNotNull(), 1).otherwise(0)) / 5.0)

    # Add processing metadata
    .withColumn("silver_processed_timestamp", current_timestamp())
    .withColumn("silver_processed_date", current_date())
)

# Select final columns
medical_claims_silver = medical_claims_silver.select(
    "claim_id",
    "member_id",
    "provider_id",
    "claim_date",
    col("service_from_date_clean").alias("service_from_date"),
    col("service_to_date_clean").alias("service_to_date"),
    "admit_date",
    "discharge_date",
    "length_of_stay",
    "diagnosis_code_1",
    "diagnosis_code_2",
    "diagnosis_code_3",
    "diagnosis_code_4",
    "procedure_code_1",
    "procedure_code_2",
    col("claim_type_clean").alias("claim_type"),
    col("claim_status_clean").alias("claim_status"),
    col("allowed_amount_clean").alias("allowed_amount"),
    col("paid_amount_clean").alias("paid_amount"),
    col("member_liability_clean").alias("member_liability"),
    "service_units",
    "drg_code",
    "revenue_code",
    "place_of_service",
    "claim_year",
    "claim_month",
    "claim_quarter",
    "service_year",
    "high_cost_flag",
    "emergency_flag",
    "inpatient_flag",
    "data_quality_score",
    "ingestion_timestamp",
    "silver_processed_timestamp"
)

# Write to silver layer
(medical_claims_silver.write
    .format("delta")
    .mode("overwrite")
    .partitionBy("claim_year", "claim_month")
    .option("overwriteSchema", "true")
    .saveAsTable("silver.medical_claims"))

print(f"✓ Medical claims cleaned: {medical_claims_silver.count():,} records")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Medical claims cleaning summary
# MAGIC SELECT
# MAGIC   COUNT(*) as total_claims,
# MAGIC   COUNT(DISTINCT member_id) as unique_members,
# MAGIC   AVG(data_quality_score) as avg_quality_score,
# MAGIC   SUM(CASE WHEN high_cost_flag THEN 1 ELSE 0 END) as high_cost_claims,
# MAGIC   SUM(CASE WHEN emergency_flag THEN 1 ELSE 0 END) as emergency_visits,
# MAGIC   SUM(CASE WHEN inpatient_flag THEN 1 ELSE 0 END) as inpatient_admissions,
# MAGIC   AVG(CASE WHEN inpatient_flag THEN length_of_stay END) as avg_los,
# MAGIC   SUM(paid_amount) as total_paid
# MAGIC FROM silver.medical_claims

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean Pharmacy Claims

# COMMAND ----------

pharmacy_claims_bronze = spark.table("bronze.pharmacy_claims")

pharmacy_claims_silver = (
    pharmacy_claims_bronze
    # Remove duplicates
    .dropDuplicates()

    # Flag and remove logical duplicates
    .transform(lambda df: flag_duplicate_claims(
        df,
        partition_cols=["member_id", "ndc_code", "fill_date", "pharmacy_id"],
        order_col="ingestion_timestamp"
    ))
    .filter(col("is_duplicate") == False)
    .drop("row_num", "is_duplicate")

    # Validate and clean amounts
    .withColumn("allowed_amount_clean",
                when(col("allowed_amount") >= 0, col("allowed_amount")).otherwise(None))
    .withColumn("paid_amount_clean",
                when(col("paid_amount") >= 0, col("paid_amount")).otherwise(None))
    .withColumn("member_copay_clean",
                when(col("member_copay") >= 0, col("member_copay")).otherwise(None))

    # Validate days supply (1-365 days)
    .withColumn("days_supply_clean",
                when((col("days_supply") >= 1) & (col("days_supply") <= 365),
                     col("days_supply"))
                .otherwise(None))

    # Calculate expected end date
    .withColumn("expected_end_date",
                expr("date_add(fill_date, days_supply_clean)"))

    # Standardize generic indicator
    .withColumn("generic_indicator_clean",
                upper(trim(col("generic_indicator"))))
    .withColumn("is_generic",
                when(col("generic_indicator_clean").isin(['G', 'GENERIC', 'Y']), True)
                .when(col("generic_indicator_clean").isin(['B', 'BRAND', 'N']), False)
                .otherwise(None))

    # Extract year/month for partitioning
    .withColumn("fill_year", year(col("fill_date")))
    .withColumn("fill_month", month(col("fill_date")))
    .withColumn("fill_quarter", quarter(col("fill_date")))

    # Flag high-cost prescriptions (> $500)
    .withColumn("high_cost_rx_flag",
                when(col("paid_amount_clean") > 500, True).otherwise(False))

    # Flag controlled substances (Schedule II-V) based on therapeutic class
    .withColumn("controlled_substance_flag",
                when(col("therapeutic_class").like("%OPIOID%"), True)
                .when(col("therapeutic_class").like("%BENZODIAZEPINE%"), True)
                .when(col("therapeutic_class").like("%STIMULANT%"), True)
                .otherwise(False))

    # Data quality score
    .withColumn("data_quality_score",
                (when(col("pharmacy_id").isNotNull(), 1).otherwise(0) +
                 when(col("prescriber_id").isNotNull(), 1).otherwise(0) +
                 when(col("ndc_code").isNotNull(), 1).otherwise(0) +
                 when(col("days_supply_clean").isNotNull(), 1).otherwise(0) +
                 when(col("paid_amount_clean").isNotNull(), 1).otherwise(0)) / 5.0)

    # Processing metadata
    .withColumn("silver_processed_timestamp", current_timestamp())
)

# Select final columns
pharmacy_claims_silver = pharmacy_claims_silver.select(
    "claim_id",
    "member_id",
    "pharmacy_id",
    "prescriber_id",
    "fill_date",
    "expected_end_date",
    "ndc_code",
    "drug_name",
    col("days_supply_clean").alias("days_supply"),
    "quantity",
    "refill_number",
    col("allowed_amount_clean").alias("allowed_amount"),
    col("paid_amount_clean").alias("paid_amount"),
    col("member_copay_clean").alias("member_copay"),
    "is_generic",
    "therapeutic_class",
    "fill_year",
    "fill_month",
    "fill_quarter",
    "high_cost_rx_flag",
    "controlled_substance_flag",
    "data_quality_score",
    "ingestion_timestamp",
    "silver_processed_timestamp"
)

# Write to silver
(pharmacy_claims_silver.write
    .format("delta")
    .mode("overwrite")
    .partitionBy("fill_year", "fill_month")
    .option("overwriteSchema", "true")
    .saveAsTable("silver.pharmacy_claims"))

print(f"✓ Pharmacy claims cleaned: {pharmacy_claims_silver.count():,} records")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Pharmacy claims summary
# MAGIC SELECT
# MAGIC   COUNT(*) as total_rx,
# MAGIC   COUNT(DISTINCT member_id) as unique_members,
# MAGIC   COUNT(DISTINCT ndc_code) as unique_drugs,
# MAGIC   AVG(days_supply) as avg_days_supply,
# MAGIC   AVG(data_quality_score) as avg_quality_score,
# MAGIC   SUM(CASE WHEN is_generic THEN 1 ELSE 0 END) as generic_fills,
# MAGIC   SUM(CASE WHEN high_cost_rx_flag THEN 1 ELSE 0 END) as high_cost_rx,
# MAGIC   SUM(CASE WHEN controlled_substance_flag THEN 1 ELSE 0 END) as controlled_substances,
# MAGIC   SUM(paid_amount) as total_paid
# MAGIC FROM silver.pharmacy_claims

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean Member Eligibility

# COMMAND ----------

eligibility_bronze = spark.table("bronze.member_eligibility")

eligibility_silver = (
    eligibility_bronze
    # Remove exact duplicates
    .dropDuplicates()

    # For overlapping eligibility periods, keep most recent
    .transform(lambda df: flag_duplicate_claims(
        df,
        partition_cols=["member_id", "effective_date"],
        order_col="ingestion_timestamp"
    ))
    .filter(col("is_duplicate") == False)
    .drop("row_num", "is_duplicate")

    # Standardize gender
    .withColumn("gender_clean", standardize_gender("gender"))

    # Clean ZIP code
    .withColumn("zip_code_clean", clean_zip_code("zip_code"))

    # Standardize race
    .withColumn("race_clean",
                when(col("race").like("%WHITE%"), "White")
                .when(col("race").like("%BLACK%"), "Black")
                .when(col("race").like("%ASIAN%"), "Asian")
                .when(col("race").like("%HISPANIC%"), "Hispanic")
                .when(col("race").like("%NATIVE%"), "Native American")
                .when(col("race").like("%PACIFIC%"), "Pacific Islander")
                .when(col("race").like("%TWO%"), "Two or More Races")
                .otherwise("Unknown"))

    # Standardize ethnicity
    .withColumn("ethnicity_clean",
                when(col("ethnicity").like("%HISPANIC%"), "Hispanic")
                .when(col("ethnicity").like("%NOT HISPANIC%"), "Not Hispanic")
                .otherwise("Unknown"))

    # Calculate age
    .withColumn("age", calculate_age("birth_date"))

    # Age groups
    .withColumn("age_group",
                when(col("age") < 18, "0-17")
                .when(col("age") < 45, "18-44")
                .when(col("age") < 65, "45-64")
                .otherwise("65+"))

    # Validate eligibility span
    .withColumn("valid_eligibility_span",
                validate_date_range("effective_date", "term_date"))

    # Calculate eligibility duration (days)
    .withColumn("eligibility_duration_days",
                when(col("term_date").isNotNull(),
                     datediff(col("term_date"), col("effective_date")))
                .otherwise(datediff(current_date(), col("effective_date"))))

    # Currently eligible flag
    .withColumn("currently_eligible",
                when(col("term_date").isNull(), True)
                .when(col("term_date") >= current_date(), True)
                .otherwise(False))

    # Risk flags
    .withColumn("high_risk_flag",
                when(col("dual_eligible") | col("ltss_indicator"), True)
                .otherwise(False))

    # Processing metadata
    .withColumn("silver_processed_timestamp", current_timestamp())
)

# Select final columns
eligibility_silver = eligibility_silver.select(
    "member_id",
    "effective_date",
    "term_date",
    "birth_date",
    "age",
    "age_group",
    col("gender_clean").alias("gender"),
    col("race_clean").alias("race"),
    col("ethnicity_clean").alias("ethnicity"),
    col("zip_code_clean").alias("zip_code"),
    "county",
    "eligibility_category",
    "aid_category",
    "dual_eligible",
    "ltss_indicator",
    "high_risk_flag",
    "currently_eligible",
    "eligibility_duration_days",
    "ingestion_timestamp",
    "silver_processed_timestamp"
)

# Write to silver
(eligibility_silver.write
    .format("delta")
    .mode("overwrite")
    .partitionBy("age_group")
    .option("overwriteSchema", "true")
    .saveAsTable("silver.member_eligibility"))

print(f"✓ Member eligibility cleaned: {eligibility_silver.count():,} records")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Member demographics summary
# MAGIC SELECT
# MAGIC   COUNT(DISTINCT member_id) as total_members,
# MAGIC   age_group,
# MAGIC   gender,
# MAGIC   race,
# MAGIC   COUNT(*) as member_count,
# MAGIC   SUM(CASE WHEN dual_eligible THEN 1 ELSE 0 END) as dual_eligible_count,
# MAGIC   SUM(CASE WHEN ltss_indicator THEN 1 ELSE 0 END) as ltss_count
# MAGIC FROM silver.member_eligibility
# MAGIC WHERE currently_eligible = true
# MAGIC GROUP BY age_group, gender, race
# MAGIC ORDER BY age_group, gender, race

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean Provider Data

# COMMAND ----------

providers_bronze = spark.table("bronze.providers")

providers_silver = (
    providers_bronze
    # Remove duplicates
    .dropDuplicates(["provider_id"])

    # Standardize provider type
    .withColumn("provider_type_clean",
                upper(trim(col("provider_type"))))

    # Clean ZIP code
    .withColumn("zip_code_clean", clean_zip_code("zip_code"))

    # Standardize state
    .withColumn("state_clean", upper(trim(col("state"))))

    # Validate NPI (10 digits)
    .withColumn("npi_clean",
                when(regexp_extract(col("npi"), r'^\d{10}$', 0) != '', col("npi"))
                .otherwise(None))

    # Data quality score
    .withColumn("data_quality_score",
                (when(col("provider_name").isNotNull(), 1).otherwise(0) +
                 when(col("provider_type_clean").isNotNull(), 1).otherwise(0) +
                 when(col("specialty").isNotNull(), 1).otherwise(0) +
                 when(col("npi_clean").isNotNull(), 1).otherwise(0) +
                 when(col("zip_code_clean").isNotNull(), 1).otherwise(0)) / 5.0)

    # Processing metadata
    .withColumn("silver_processed_timestamp", current_timestamp())
)

# Select final columns
providers_silver = providers_silver.select(
    "provider_id",
    "provider_name",
    col("provider_type_clean").alias("provider_type"),
    "specialty",
    col("npi_clean").alias("npi"),
    "taxonomy_code",
    "address_line1",
    "city",
    col("state_clean").alias("state"),
    col("zip_code_clean").alias("zip_code"),
    "data_quality_score",
    "ingestion_timestamp",
    "silver_processed_timestamp"
)

# Write to silver
(providers_silver.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("silver.providers"))

print(f"✓ Providers cleaned: {providers_silver.count():,} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Summary

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Overall data quality across silver tables
# MAGIC SELECT
# MAGIC   'Medical Claims' as table_name,
# MAGIC   COUNT(*) as record_count,
# MAGIC   AVG(data_quality_score) as avg_quality_score,
# MAGIC   MIN(silver_processed_timestamp) as processed_at
# MAGIC FROM silver.medical_claims
# MAGIC
# MAGIC UNION ALL
# MAGIC
# MAGIC SELECT
# MAGIC   'Pharmacy Claims' as table_name,
# MAGIC   COUNT(*) as record_count,
# MAGIC   AVG(data_quality_score) as avg_quality_score,
# MAGIC   MIN(silver_processed_timestamp) as processed_at
# MAGIC FROM silver.pharmacy_claims
# MAGIC
# MAGIC UNION ALL
# MAGIC
# MAGIC SELECT
# MAGIC   'Providers' as table_name,
# MAGIC   COUNT(*) as record_count,
# MAGIC   AVG(data_quality_score) as avg_quality_score,
# MAGIC   MIN(silver_processed_timestamp) as processed_at
# MAGIC FROM silver.providers

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimize Silver Tables

# COMMAND ----------

# Optimize for query performance
for table in ["medical_claims", "pharmacy_claims", "member_eligibility", "providers"]:
    print(f"Optimizing silver.{table}...")
    spark.sql(f"OPTIMIZE silver.{table}")

    if table == "medical_claims":
        spark.sql(f"OPTIMIZE silver.{table} ZORDER BY (member_id, claim_date)")
    elif table == "pharmacy_claims":
        spark.sql(f"OPTIMIZE silver.{table} ZORDER BY (member_id, fill_date)")
    elif table == "member_eligibility":
        spark.sql(f"OPTIMIZE silver.{table} ZORDER BY (member_id, effective_date)")

    print(f"✓ silver.{table} optimized")

# COMMAND ----------

print("=" * 60)
print("SILVER LAYER CLEANING COMPLETE")
print("=" * 60)
print("\nCleaned data ready for feature engineering in gold layer.")
