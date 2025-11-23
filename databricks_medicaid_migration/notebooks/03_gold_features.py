# Databricks notebook source
# MAGIC %md
# MAGIC # Gold Layer: Feature Engineering for ML Models
# MAGIC
# MAGIC This notebook creates analytical features for predictive modeling using Databricks Feature Store.
# MAGIC
# MAGIC **Feature Categories:**
# MAGIC - Clinical features (diagnoses, procedures, chronic conditions)
# MAGIC - Utilization features (service counts, costs, patterns)
# MAGIC - Demographic features (age, gender, social determinants)
# MAGIC - Temporal features (trends, seasonality, recency)
# MAGIC - Risk scores (calculated risk indicators)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import *
from pyspark.sql.types import *
from delta.tables import DeltaTable
from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup
from datetime import datetime, timedelta
import numpy as np

# Initialize Feature Store
fs = feature_store.FeatureStoreClient()

# Create gold database
spark.sql("CREATE DATABASE IF NOT EXISTS gold")
spark.sql("USE gold")

# Reference date for feature calculation
REFERENCE_DATE = current_date()
LOOKBACK_PERIOD_DAYS = 365  # 1 year lookback

print("✓ Setup complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Member-Level Feature Set

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Demographic Features

# COMMAND ----------

# Get active member demographics
demographics = spark.sql(f"""
SELECT
    member_id,
    age,
    age_group,
    gender,
    race,
    ethnicity,
    zip_code,
    county,
    dual_eligible,
    ltss_indicator,
    high_risk_flag,
    eligibility_duration_days / 365.0 as eligibility_years
FROM silver.member_eligibility
WHERE currently_eligible = true
   OR term_date >= date_sub(current_date(), 90)  -- Include recently termed
""")

print(f"Demographics: {demographics.count():,} members")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Clinical Features - Chronic Conditions

# COMMAND ----------

# Define chronic condition logic based on ICD-10 codes
chronic_conditions_sql = f"""
WITH diagnosis_claims AS (
    SELECT DISTINCT
        member_id,
        CASE
            WHEN diagnosis_code_1 LIKE 'E11%' OR diagnosis_code_2 LIKE 'E11%'
                 OR diagnosis_code_3 LIKE 'E11%' OR diagnosis_code_4 LIKE 'E11%'
            THEN 'diabetes'
            WHEN diagnosis_code_1 LIKE 'I10%' OR diagnosis_code_1 LIKE 'I11%'
                 OR diagnosis_code_2 LIKE 'I10%' OR diagnosis_code_2 LIKE 'I11%'
            THEN 'hypertension'
            WHEN diagnosis_code_1 LIKE 'J44%' OR diagnosis_code_1 LIKE 'J45%'
                 OR diagnosis_code_2 LIKE 'J44%' OR diagnosis_code_2 LIKE 'J45%'
            THEN 'copd_asthma'
            WHEN diagnosis_code_1 LIKE 'I50%' OR diagnosis_code_2 LIKE 'I50%'
            THEN 'heart_failure'
            WHEN diagnosis_code_1 LIKE 'N18%' OR diagnosis_code_2 LIKE 'N18%'
            THEN 'ckd'
            WHEN diagnosis_code_1 LIKE 'F%' OR diagnosis_code_2 LIKE 'F%'
            THEN 'mental_health'
            WHEN diagnosis_code_1 LIKE 'E66%' OR diagnosis_code_2 LIKE 'E66%'
            THEN 'obesity'
            WHEN diagnosis_code_1 LIKE 'C%' OR diagnosis_code_2 LIKE 'C%'
            THEN 'cancer'
        END as condition
    FROM silver.medical_claims
    WHERE service_from_date >= date_sub(current_date(), {LOOKBACK_PERIOD_DAYS})
      AND claim_status = 'PAID'
)
SELECT
    member_id,
    MAX(CASE WHEN condition = 'diabetes' THEN 1 ELSE 0 END) as has_diabetes,
    MAX(CASE WHEN condition = 'hypertension' THEN 1 ELSE 0 END) as has_hypertension,
    MAX(CASE WHEN condition = 'copd_asthma' THEN 1 ELSE 0 END) as has_copd_asthma,
    MAX(CASE WHEN condition = 'heart_failure' THEN 1 ELSE 0 END) as has_heart_failure,
    MAX(CASE WHEN condition = 'ckd' THEN 1 ELSE 0 END) as has_ckd,
    MAX(CASE WHEN condition = 'mental_health' THEN 1 ELSE 0 END) as has_mental_health,
    MAX(CASE WHEN condition = 'obesity' THEN 1 ELSE 0 END) as has_obesity,
    MAX(CASE WHEN condition = 'cancer' THEN 1 ELSE 0 END) as has_cancer
FROM diagnosis_claims
WHERE condition IS NOT NULL
GROUP BY member_id
"""

chronic_conditions = spark.sql(chronic_conditions_sql)

# Calculate chronic condition count
chronic_conditions = chronic_conditions.withColumn(
    "chronic_condition_count",
    col("has_diabetes") + col("has_hypertension") + col("has_copd_asthma") +
    col("has_heart_failure") + col("has_ckd") + col("has_mental_health") +
    col("has_obesity") + col("has_cancer")
)

print(f"Chronic conditions identified for: {chronic_conditions.count():,} members")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Utilization Features

# COMMAND ----------

# Medical utilization features
medical_utilization = spark.sql(f"""
SELECT
    member_id,
    COUNT(*) as claim_count_12mo,
    COUNT(DISTINCT claim_date) as service_days_12mo,
    SUM(paid_amount) as total_paid_12mo,
    AVG(paid_amount) as avg_claim_amount,
    MAX(paid_amount) as max_claim_amount,
    SUM(CASE WHEN emergency_flag THEN 1 ELSE 0 END) as er_visit_count_12mo,
    SUM(CASE WHEN inpatient_flag THEN 1 ELSE 0 END) as inpatient_admit_count_12mo,
    SUM(CASE WHEN high_cost_flag THEN 1 ELSE 0 END) as high_cost_claim_count,
    AVG(CASE WHEN inpatient_flag THEN length_of_stay END) as avg_length_of_stay,
    SUM(length_of_stay) as total_inpatient_days,
    COUNT(DISTINCT provider_id) as unique_provider_count
FROM silver.medical_claims
WHERE service_from_date >= date_sub(current_date(), {LOOKBACK_PERIOD_DAYS})
  AND claim_status = 'PAID'
GROUP BY member_id
""")

# Pharmacy utilization features
pharmacy_utilization = spark.sql(f"""
SELECT
    member_id,
    COUNT(*) as rx_fill_count_12mo,
    COUNT(DISTINCT ndc_code) as unique_drug_count,
    COUNT(DISTINCT therapeutic_class) as unique_therapy_class_count,
    SUM(paid_amount) as total_rx_paid_12mo,
    AVG(paid_amount) as avg_rx_cost,
    AVG(days_supply) as avg_days_supply,
    SUM(CASE WHEN is_generic THEN 1 ELSE 0 END) as generic_fill_count,
    SUM(CASE WHEN controlled_substance_flag THEN 1 ELSE 0 END) as controlled_substance_count,
    SUM(CASE WHEN high_cost_rx_flag THEN 1 ELSE 0 END) as high_cost_rx_count
FROM silver.pharmacy_claims
WHERE fill_date >= date_sub(current_date(), {LOOKBACK_PERIOD_DAYS})
GROUP BY member_id
""")

print(f"Medical utilization: {medical_utilization.count():,} members")
print(f"Pharmacy utilization: {pharmacy_utilization.count():,} members")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4: Temporal Features - Recent Activity

# COMMAND ----------

# Recent utilization trends (last 90 days vs previous 275 days)
recent_trends = spark.sql("""
WITH recent_period AS (
    SELECT
        member_id,
        COUNT(*) as recent_claim_count,
        SUM(paid_amount) as recent_paid_amount,
        SUM(CASE WHEN emergency_flag THEN 1 ELSE 0 END) as recent_er_visits
    FROM silver.medical_claims
    WHERE service_from_date >= date_sub(current_date(), 90)
      AND service_from_date < current_date()
      AND claim_status = 'PAID'
    GROUP BY member_id
),
prior_period AS (
    SELECT
        member_id,
        COUNT(*) as prior_claim_count,
        SUM(paid_amount) as prior_paid_amount,
        SUM(CASE WHEN emergency_flag THEN 1 ELSE 0 END) as prior_er_visits
    FROM silver.medical_claims
    WHERE service_from_date >= date_sub(current_date(), 365)
      AND service_from_date < date_sub(current_date(), 90)
      AND claim_status = 'PAID'
    GROUP BY member_id
)
SELECT
    COALESCE(r.member_id, p.member_id) as member_id,
    COALESCE(r.recent_claim_count, 0) as recent_claim_count_90d,
    COALESCE(p.prior_claim_count, 0) as prior_claim_count_275d,
    COALESCE(r.recent_paid_amount, 0) as recent_paid_90d,
    COALESCE(p.prior_paid_amount, 0) as prior_paid_275d,
    COALESCE(r.recent_er_visits, 0) as recent_er_visits_90d,
    COALESCE(p.prior_er_visits, 0) as prior_er_visits_275d,
    -- Trend indicators
    CASE
        WHEN p.prior_claim_count > 0 THEN
            (r.recent_claim_count * (275.0/90.0) - p.prior_claim_count) / p.prior_claim_count
        ELSE 0
    END as claim_count_trend_pct,
    CASE
        WHEN p.prior_paid_amount > 0 THEN
            (r.recent_paid_amount * (275.0/90.0) - p.prior_paid_amount) / p.prior_paid_amount
        ELSE 0
    END as cost_trend_pct
FROM recent_period r
FULL OUTER JOIN prior_period p ON r.member_id = p.member_id
""")

print(f"Temporal trends: {recent_trends.count():,} members")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5: Recency Features

# COMMAND ----------

# Days since last services
recency_features = spark.sql(f"""
SELECT
    member_id,
    datediff(current_date(), MAX(service_from_date)) as days_since_last_medical_claim,
    datediff(current_date(), MAX(CASE WHEN emergency_flag THEN service_from_date END)) as days_since_last_er_visit,
    datediff(current_date(), MAX(CASE WHEN inpatient_flag THEN service_from_date END)) as days_since_last_admission
FROM silver.medical_claims
WHERE service_from_date >= date_sub(current_date(), {LOOKBACK_PERIOD_DAYS})
GROUP BY member_id
""")

rx_recency = spark.sql(f"""
SELECT
    member_id,
    datediff(current_date(), MAX(fill_date)) as days_since_last_rx_fill
FROM silver.pharmacy_claims
WHERE fill_date >= date_sub(current_date(), {LOOKBACK_PERIOD_DAYS})
GROUP BY member_id
""")

print(f"Recency features calculated")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6: Risk Scores

# COMMAND ----------

# Calculate composite risk scores
def calculate_risk_score(df):
    """Calculate composite risk score based on multiple factors."""
    return (df
        # Age risk (higher for very young and elderly)
        .withColumn("age_risk_score",
                    when(col("age") < 1, 3)
                    .when(col("age") < 5, 2)
                    .when(col("age") >= 75, 3)
                    .when(col("age") >= 65, 2)
                    .otherwise(1))

        # Chronic condition risk
        .withColumn("chronic_condition_risk_score",
                    when(col("chronic_condition_count") >= 3, 3)
                    .when(col("chronic_condition_count") >= 2, 2)
                    .when(col("chronic_condition_count") >= 1, 1)
                    .otherwise(0))

        # Utilization risk (high ER/IP usage)
        .withColumn("utilization_risk_score",
                    when((col("er_visit_count_12mo") >= 4) | (col("inpatient_admit_count_12mo") >= 2), 3)
                    .when((col("er_visit_count_12mo") >= 2) | (col("inpatient_admit_count_12mo") >= 1), 2)
                    .when(col("er_visit_count_12mo") >= 1, 1)
                    .otherwise(0))

        # Cost risk
        .withColumn("cost_risk_score",
                    when(col("total_paid_12mo") >= 50000, 3)
                    .when(col("total_paid_12mo") >= 25000, 2)
                    .when(col("total_paid_12mo") >= 10000, 1)
                    .otherwise(0))

        # Composite risk score (0-12)
        .withColumn("composite_risk_score",
                    col("age_risk_score") + col("chronic_condition_risk_score") +
                    col("utilization_risk_score") + col("cost_risk_score"))

        # Risk stratification
        .withColumn("risk_tier",
                    when(col("composite_risk_score") >= 9, "High")
                    .when(col("composite_risk_score") >= 6, "Medium")
                    .when(col("composite_risk_score") >= 3, "Low")
                    .otherwise("Very Low"))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 7: Combine All Features

# COMMAND ----------

# Start with demographics
member_features = demographics

# Join chronic conditions
member_features = member_features.join(chronic_conditions, "member_id", "left")

# Join utilization features
member_features = member_features.join(medical_utilization, "member_id", "left")
member_features = member_features.join(pharmacy_utilization, "member_id", "left")

# Join temporal trends
member_features = member_features.join(recent_trends, "member_id", "left")

# Join recency features
member_features = member_features.join(recency_features, "member_id", "left")
member_features = member_features.join(rx_recency, "member_id", "left")

# Fill nulls for members with no claims
member_features = member_features.fillna(0, subset=[
    "claim_count_12mo", "service_days_12mo", "total_paid_12mo", "er_visit_count_12mo",
    "inpatient_admit_count_12mo", "high_cost_claim_count", "rx_fill_count_12mo",
    "unique_drug_count", "total_rx_paid_12mo", "chronic_condition_count"
])

# Calculate risk scores
member_features = calculate_risk_score(member_features)

# Add feature timestamp
member_features = member_features.withColumn("feature_timestamp", current_timestamp())
member_features = member_features.withColumn("feature_date", current_date())

print(f"✓ Combined features for {member_features.count():,} members")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Feature summary statistics
# MAGIC SELECT
# MAGIC   COUNT(*) as total_members,
# MAGIC   AVG(age) as avg_age,
# MAGIC   AVG(chronic_condition_count) as avg_chronic_conditions,
# MAGIC   AVG(claim_count_12mo) as avg_claims_12mo,
# MAGIC   AVG(total_paid_12mo) as avg_total_paid,
# MAGIC   AVG(er_visit_count_12mo) as avg_er_visits,
# MAGIC   AVG(composite_risk_score) as avg_risk_score
# MAGIC FROM gold.member_features

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Feature Store Table

# COMMAND ----------

# Define feature table with primary key
feature_table_name = "gold.member_features"

# Create or replace feature table
fs.create_table(
    name=feature_table_name,
    primary_keys=["member_id"],
    df=member_features,
    schema=member_features.schema,
    description="Member-level features for risk prediction and utilization forecasting"
)

print(f"✓ Feature Store table created: {feature_table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Immunization Gap Features

# COMMAND ----------

# Identify immunization gaps based on age and chronic conditions
immunization_features = spark.sql("""
SELECT
    mf.member_id,
    mf.age,
    mf.age_group,
    mf.chronic_condition_count,
    mf.has_diabetes,
    mf.has_copd_asthma,
    mf.has_heart_failure,
    mf.has_ckd,
    mf.composite_risk_score,
    mf.risk_tier,
    -- Immunization eligibility flags
    CASE
        WHEN mf.age >= 65 OR mf.chronic_condition_count >= 1 THEN true
        ELSE false
    END as flu_vaccine_eligible,
    CASE
        WHEN mf.age >= 65 OR mf.has_diabetes = 1 OR mf.has_copd_asthma = 1
             OR mf.has_heart_failure = 1 THEN true
        ELSE false
    END as pneumococcal_vaccine_eligible,
    CASE
        WHEN mf.age >= 50 THEN true
        ELSE false
    END as shingles_vaccine_eligible,
    CASE
        WHEN mf.age >= 19 AND mf.age <= 26 THEN true
        ELSE false
    END as hpv_vaccine_eligible,
    -- Priority score for outreach (0-100)
    CASE
        WHEN mf.risk_tier = 'High' THEN 80
        WHEN mf.risk_tier = 'Medium' THEN 60
        WHEN mf.risk_tier = 'Low' THEN 40
        ELSE 20
    END +
    CASE WHEN mf.age >= 65 THEN 15 ELSE 0 END +
    CASE WHEN mf.chronic_condition_count >= 3 THEN 5 ELSE 0 END as immunization_priority_score
FROM gold.member_features mf
WHERE mf.risk_tier IN ('High', 'Medium')
   OR mf.age >= 65
   OR mf.chronic_condition_count >= 1
""")

# Save immunization features
(immunization_features.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("gold.immunization_targeting"))

print(f"✓ Immunization targeting features: {immunization_features.count():,} members")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Immunization targeting summary
# MAGIC SELECT
# MAGIC   risk_tier,
# MAGIC   COUNT(*) as member_count,
# MAGIC   SUM(CASE WHEN flu_vaccine_eligible THEN 1 ELSE 0 END) as flu_eligible,
# MAGIC   SUM(CASE WHEN pneumococcal_vaccine_eligible THEN 1 ELSE 0 END) as pneumo_eligible,
# MAGIC   SUM(CASE WHEN shingles_vaccine_eligible THEN 1 ELSE 0 END) as shingles_eligible,
# MAGIC   AVG(immunization_priority_score) as avg_priority_score
# MAGIC FROM gold.immunization_targeting
# MAGIC GROUP BY risk_tier
# MAGIC ORDER BY avg_priority_score DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimize Gold Tables

# COMMAND ----------

# Optimize feature tables
for table in ["member_features", "immunization_targeting"]:
    print(f"Optimizing gold.{table}...")
    spark.sql(f"OPTIMIZE gold.{table} ZORDER BY (member_id)")
    print(f"✓ gold.{table} optimized")

# COMMAND ----------

print("=" * 60)
print("GOLD LAYER FEATURE ENGINEERING COMPLETE")
print("=" * 60)
print("\nFeatures ready for ML model training.")
print(f"Feature Store table: {feature_table_name}")
