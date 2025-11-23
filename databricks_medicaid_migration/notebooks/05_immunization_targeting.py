# Databricks notebook source
# MAGIC %md
# MAGIC # Immunization Program Targeting Analytics
# MAGIC
# MAGIC This notebook combines risk predictions with immunization targeting to:
# MAGIC 1. Identify high-priority members for immunization outreach
# MAGIC 2. Generate targeted outreach lists
# MAGIC 3. Analyze immunization gaps by risk tier
# MAGIC 4. Create actionable dashboards for program management

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("✓ Setup complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

# Load member features
member_features = spark.table("gold.member_features")

# Load immunization targeting
immunization_targeting = spark.table("gold.immunization_targeting")

# Load risk predictions
risk_predictions = spark.table("gold.member_risk_predictions")

# Join all data
comprehensive_view = (
    immunization_targeting
    .join(risk_predictions, "member_id", "inner")
    .join(
        member_features.select(
            "member_id", "gender", "race", "ethnicity", "zip_code", "county",
            "total_paid_12mo", "er_visit_count_12mo", "inpatient_admit_count_12mo"
        ),
        "member_id",
        "inner"
    )
)

print(f"Comprehensive view: {comprehensive_view.count():,} members")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Priority Targeting Lists

# COMMAND ----------

# Enhanced priority scoring combining clinical risk and ML predictions
priority_outreach = comprehensive_view.withColumn(
    "outreach_priority_score",
    # Base immunization priority
    col("immunization_priority_score") * 0.4 +
    # ML-based risk scores
    (col("high_risk_score") * 100) * 0.3 +
    (col("er_utilization_risk_score") * 100) * 0.2 +
    (col("high_cost_risk_score") * 100) * 0.1
).withColumn(
    "outreach_tier",
    when(col("outreach_priority_score") >= 80, "Tier 1 - Immediate")
    .when(col("outreach_priority_score") >= 60, "Tier 2 - High Priority")
    .when(col("outreach_priority_score") >= 40, "Tier 3 - Standard")
    .otherwise("Tier 4 - Routine")
)

# Add outreach recommendations
priority_outreach = priority_outreach.withColumn(
    "recommended_outreach_method",
    when(col("outreach_tier") == "Tier 1 - Immediate", "Phone Call + Letter")
    .when(col("outreach_tier") == "Tier 2 - High Priority", "Phone Call")
    .when(col("outreach_tier") == "Tier 3 - Standard", "Letter")
    .otherwise("Email/Portal Message")
)

# Calculate expected impact (estimated prevented costs)
priority_outreach = priority_outreach.withColumn(
    "estimated_preventable_cost",
    when(col("flu_vaccine_eligible") & (col("er_visit_count_12mo") >= 2), 3500)
    .when(col("pneumococcal_vaccine_eligible") & (col("chronic_condition_count") >= 2), 5000)
    .otherwise(1500)
)

print("✓ Priority scoring complete")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW priority_outreach
# MAGIC AS SELECT * FROM priority_outreach_temp;

# COMMAND ----------

# Register as temp view for SQL
priority_outreach.createOrReplaceTempView("priority_outreach_temp")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Outreach List Generation

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Tier 1 Immediate Outreach List
# MAGIC SELECT
# MAGIC   member_id,
# MAGIC   age,
# MAGIC   gender,
# MAGIC   risk_tier,
# MAGIC   chronic_condition_count,
# MAGIC   er_visit_count_12mo,
# MAGIC   high_risk_score,
# MAGIC   er_utilization_risk_score,
# MAGIC   outreach_priority_score,
# MAGIC   CASE
# MAGIC     WHEN flu_vaccine_eligible THEN 'Flu'
# MAGIC     ELSE ''
# MAGIC   END ||
# MAGIC   CASE
# MAGIC     WHEN pneumococcal_vaccine_eligible THEN ', Pneumococcal'
# MAGIC     ELSE ''
# MAGIC   END ||
# MAGIC   CASE
# MAGIC     WHEN shingles_vaccine_eligible THEN ', Shingles'
# MAGIC     ELSE ''
# MAGIC   END as recommended_vaccines,
# MAGIC   recommended_outreach_method,
# MAGIC   estimated_preventable_cost,
# MAGIC   zip_code,
# MAGIC   county
# MAGIC FROM priority_outreach_temp
# MAGIC WHERE outreach_tier = 'Tier 1 - Immediate'
# MAGIC ORDER BY outreach_priority_score DESC
# MAGIC LIMIT 1000

# COMMAND ----------

# Export Tier 1 list for outreach team
tier1_outreach = spark.sql("""
SELECT
    member_id,
    age,
    age_group,
    gender,
    race,
    county,
    zip_code,
    risk_tier,
    chronic_condition_count,
    has_diabetes,
    has_copd_asthma,
    has_heart_failure,
    er_visit_count_12mo,
    inpatient_admit_count_12mo,
    total_paid_12mo,
    high_risk_score,
    er_utilization_risk_score,
    high_cost_risk_score,
    outreach_priority_score,
    flu_vaccine_eligible,
    pneumococcal_vaccine_eligible,
    shingles_vaccine_eligible,
    recommended_outreach_method,
    estimated_preventable_cost
FROM priority_outreach_temp
WHERE outreach_tier = 'Tier 1 - Immediate'
ORDER BY outreach_priority_score DESC
""")

# Save to gold layer
(tier1_outreach.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("gold.immunization_tier1_outreach"))

print(f"✓ Tier 1 outreach list: {tier1_outreach.count():,} members")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analytics Dashboard Data

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Summary statistics by outreach tier
# MAGIC SELECT
# MAGIC   outreach_tier,
# MAGIC   COUNT(*) as member_count,
# MAGIC   AVG(age) as avg_age,
# MAGIC   AVG(chronic_condition_count) as avg_chronic_conditions,
# MAGIC   AVG(outreach_priority_score) as avg_priority_score,
# MAGIC   SUM(CASE WHEN flu_vaccine_eligible THEN 1 ELSE 0 END) as flu_eligible,
# MAGIC   SUM(CASE WHEN pneumococcal_vaccine_eligible THEN 1 ELSE 0 END) as pneumo_eligible,
# MAGIC   SUM(estimated_preventable_cost) as total_preventable_cost,
# MAGIC   AVG(high_risk_score) as avg_risk_score,
# MAGIC   AVG(er_utilization_risk_score) as avg_er_risk,
# MAGIC   SUM(CASE WHEN er_utilization_prediction = 1 THEN 1 ELSE 0 END) as predicted_high_er_users
# MAGIC FROM priority_outreach_temp
# MAGIC GROUP BY outreach_tier
# MAGIC ORDER BY avg_priority_score DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Geographic Analysis

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Immunization targeting by county (top 10 counties)
# MAGIC SELECT
# MAGIC   county,
# MAGIC   COUNT(*) as total_members,
# MAGIC   SUM(CASE WHEN outreach_tier = 'Tier 1 - Immediate' THEN 1 ELSE 0 END) as tier1_members,
# MAGIC   AVG(outreach_priority_score) as avg_priority_score,
# MAGIC   SUM(CASE WHEN flu_vaccine_eligible THEN 1 ELSE 0 END) as flu_eligible,
# MAGIC   SUM(CASE WHEN pneumococcal_vaccine_eligible THEN 1 ELSE 0 END) as pneumo_eligible,
# MAGIC   SUM(estimated_preventable_cost) as potential_savings,
# MAGIC   AVG(er_visit_count_12mo) as avg_er_visits
# MAGIC FROM priority_outreach_temp
# MAGIC GROUP BY county
# MAGIC ORDER BY tier1_members DESC
# MAGIC LIMIT 10

# COMMAND ----------

# Geographic targeting summary
geo_summary = spark.sql("""
SELECT
    county,
    COUNT(*) as member_count,
    SUM(CASE WHEN outreach_tier IN ('Tier 1 - Immediate', 'Tier 2 - High Priority') THEN 1 ELSE 0 END) as high_priority_count,
    AVG(outreach_priority_score) as avg_priority,
    SUM(estimated_preventable_cost) as total_preventable_cost
FROM priority_outreach_temp
GROUP BY county
HAVING COUNT(*) >= 10
ORDER BY high_priority_count DESC
""")

# Save for mapping/visualization
(geo_summary.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("gold.immunization_geographic_summary"))

print("✓ Geographic summary saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Demographic Disparities Analysis

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Immunization gaps by race/ethnicity
# MAGIC SELECT
# MAGIC   race,
# MAGIC   ethnicity,
# MAGIC   COUNT(*) as member_count,
# MAGIC   AVG(outreach_priority_score) as avg_priority_score,
# MAGIC   SUM(CASE WHEN flu_vaccine_eligible THEN 1 ELSE 0 END) as flu_eligible,
# MAGIC   SUM(CASE WHEN pneumococcal_vaccine_eligible THEN 1 ELSE 0 END) as pneumo_eligible,
# MAGIC   AVG(chronic_condition_count) as avg_chronic_conditions,
# MAGIC   AVG(high_risk_score) as avg_risk_score,
# MAGIC   SUM(CASE WHEN outreach_tier = 'Tier 1 - Immediate' THEN 1 ELSE 0 END) as tier1_count
# MAGIC FROM priority_outreach_temp
# MAGIC GROUP BY race, ethnicity
# MAGIC ORDER BY avg_priority_score DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Risk Factor Analysis

# COMMAND ----------

# Key risk factors for immunization targeting
risk_factor_analysis = spark.sql("""
SELECT
    'Chronic Conditions' as risk_factor_category,
    CASE
        WHEN chronic_condition_count >= 3 THEN '3+ conditions'
        WHEN chronic_condition_count = 2 THEN '2 conditions'
        WHEN chronic_condition_count = 1 THEN '1 condition'
        ELSE 'No chronic conditions'
    END as risk_factor,
    COUNT(*) as member_count,
    AVG(outreach_priority_score) as avg_priority,
    SUM(CASE WHEN outreach_tier = 'Tier 1 - Immediate' THEN 1 ELSE 0 END) as tier1_count,
    AVG(er_visit_count_12mo) as avg_er_visits,
    AVG(total_paid_12mo) as avg_total_cost
FROM priority_outreach_temp
GROUP BY
    CASE
        WHEN chronic_condition_count >= 3 THEN '3+ conditions'
        WHEN chronic_condition_count = 2 THEN '2 conditions'
        WHEN chronic_condition_count = 1 THEN '1 condition'
        ELSE 'No chronic conditions'
    END

UNION ALL

SELECT
    'Age Group' as risk_factor_category,
    age_group as risk_factor,
    COUNT(*) as member_count,
    AVG(outreach_priority_score) as avg_priority,
    SUM(CASE WHEN outreach_tier = 'Tier 1 - Immediate' THEN 1 ELSE 0 END) as tier1_count,
    AVG(er_visit_count_12mo) as avg_er_visits,
    AVG(total_paid_12mo) as avg_total_cost
FROM priority_outreach_temp
GROUP BY age_group

UNION ALL

SELECT
    'ER Utilization' as risk_factor_category,
    CASE
        WHEN er_visit_count_12mo >= 4 THEN '4+ ER visits'
        WHEN er_visit_count_12mo >= 2 THEN '2-3 ER visits'
        WHEN er_visit_count_12mo = 1 THEN '1 ER visit'
        ELSE 'No ER visits'
    END as risk_factor,
    COUNT(*) as member_count,
    AVG(outreach_priority_score) as avg_priority,
    SUM(CASE WHEN outreach_tier = 'Tier 1 - Immediate' THEN 1 ELSE 0 END) as tier1_count,
    AVG(er_visit_count_12mo) as avg_er_visits,
    AVG(total_paid_12mo) as avg_total_cost
FROM priority_outreach_temp
GROUP BY
    CASE
        WHEN er_visit_count_12mo >= 4 THEN '4+ ER visits'
        WHEN er_visit_count_12mo >= 2 THEN '2-3 ER visits'
        WHEN er_visit_count_12mo = 1 THEN '1 ER visit'
        ELSE 'No ER visits'
    END

ORDER BY risk_factor_category, avg_priority DESC
""")

display(risk_factor_analysis)

# Save for reporting
(risk_factor_analysis.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("gold.immunization_risk_factor_analysis"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Program ROI Estimation

# COMMAND ----------

# Calculate expected ROI from immunization program
roi_analysis = spark.sql("""
WITH outreach_costs AS (
    SELECT
        outreach_tier,
        COUNT(*) as member_count,
        CASE outreach_tier
            WHEN 'Tier 1 - Immediate' THEN COUNT(*) * 25.0  -- Phone + letter cost
            WHEN 'Tier 2 - High Priority' THEN COUNT(*) * 15.0  -- Phone call cost
            WHEN 'Tier 3 - Standard' THEN COUNT(*) * 5.0   -- Letter cost
            ELSE COUNT(*) * 1.0  -- Email cost
        END as total_outreach_cost,
        SUM(estimated_preventable_cost) as total_preventable_cost
    FROM priority_outreach_temp
    GROUP BY outreach_tier
),
vaccine_costs AS (
    SELECT
        SUM(CASE WHEN flu_vaccine_eligible THEN 1 ELSE 0 END) * 30.0 as flu_vaccine_cost,
        SUM(CASE WHEN pneumococcal_vaccine_eligible THEN 1 ELSE 0 END) * 150.0 as pneumo_vaccine_cost,
        SUM(CASE WHEN shingles_vaccine_eligible THEN 1 ELSE 0 END) * 200.0 as shingles_vaccine_cost
    FROM priority_outreach_temp
)
SELECT
    oc.outreach_tier,
    oc.member_count,
    oc.total_outreach_cost,
    oc.total_preventable_cost,
    -- Assume 40% vaccination uptake for high priority, 20% for others
    CASE
        WHEN oc.outreach_tier IN ('Tier 1 - Immediate', 'Tier 2 - High Priority') THEN 0.4
        ELSE 0.2
    END as assumed_uptake_rate,
    -- Expected prevented costs
    oc.total_preventable_cost *
    CASE
        WHEN oc.outreach_tier IN ('Tier 1 - Immediate', 'Tier 2 - High Priority') THEN 0.4
        ELSE 0.2
    END as expected_prevented_cost,
    -- Net ROI
    (oc.total_preventable_cost *
     CASE
        WHEN oc.outreach_tier IN ('Tier 1 - Immediate', 'Tier 2 - High Priority') THEN 0.4
        ELSE 0.2
     END - oc.total_outreach_cost) as estimated_net_savings,
    -- ROI ratio
    (oc.total_preventable_cost *
     CASE
        WHEN oc.outreach_tier IN ('Tier 1 - Immediate', 'Tier 2 - High Priority') THEN 0.4
        ELSE 0.2
     END / NULLIF(oc.total_outreach_cost, 0)) as roi_ratio
FROM outreach_costs oc
CROSS JOIN vaccine_costs vc
ORDER BY oc.outreach_tier
""")

display(roi_analysis)

# Save ROI analysis
(roi_analysis.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("gold.immunization_program_roi"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Final Outreach Master List

# COMMAND ----------

# Comprehensive outreach list with all fields
final_outreach_list = spark.sql("""
SELECT
    member_id,
    -- Demographics
    age,
    age_group,
    gender,
    race,
    ethnicity,
    zip_code,
    county,
    -- Clinical profile
    risk_tier,
    chronic_condition_count,
    has_diabetes,
    has_hypertension,
    has_copd_asthma,
    has_heart_failure,
    has_ckd,
    -- Utilization
    er_visit_count_12mo,
    inpatient_admit_count_12mo,
    total_paid_12mo,
    -- Risk scores
    ROUND(high_risk_score * 100, 1) as high_risk_score_pct,
    ROUND(er_utilization_risk_score * 100, 1) as er_risk_score_pct,
    ROUND(high_cost_risk_score * 100, 1) as cost_risk_score_pct,
    -- Predictions
    high_risk_prediction,
    er_utilization_prediction,
    high_cost_prediction,
    ROUND(predicted_total_cost, 2) as predicted_total_cost,
    -- Immunization targeting
    flu_vaccine_eligible,
    pneumococcal_vaccine_eligible,
    shingles_vaccine_eligible,
    hpv_vaccine_eligible,
    ROUND(immunization_priority_score, 1) as immunization_priority_score,
    ROUND(outreach_priority_score, 1) as outreach_priority_score,
    outreach_tier,
    recommended_outreach_method,
    estimated_preventable_cost,
    -- Metadata
    current_date() as list_generation_date
FROM priority_outreach_temp
ORDER BY outreach_priority_score DESC
""")

# Save master outreach list
(final_outreach_list.write
    .format("delta")
    .mode("overwrite")
    .partitionBy("outreach_tier")
    .option("overwriteSchema", "true")
    .saveAsTable("gold.immunization_master_outreach_list"))

print(f"✓ Master outreach list created: {final_outreach_list.count():,} members")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Final summary
# MAGIC SELECT
# MAGIC   'Total Members' as metric,
# MAGIC   COUNT(*) as value
# MAGIC FROM gold.immunization_master_outreach_list
# MAGIC
# MAGIC UNION ALL
# MAGIC
# MAGIC SELECT
# MAGIC   'Tier 1 - Immediate' as metric,
# MAGIC   COUNT(*) as value
# MAGIC FROM gold.immunization_master_outreach_list
# MAGIC WHERE outreach_tier = 'Tier 1 - Immediate'
# MAGIC
# MAGIC UNION ALL
# MAGIC
# MAGIC SELECT
# MAGIC   'Tier 2 - High Priority' as metric,
# MAGIC   COUNT(*) as value
# MAGIC FROM gold.immunization_master_outreach_list
# MAGIC WHERE outreach_tier = 'Tier 2 - High Priority'
# MAGIC
# MAGIC UNION ALL
# MAGIC
# MAGIC SELECT
# MAGIC   'Total Estimated Preventable Cost' as metric,
# MAGIC   CAST(SUM(estimated_preventable_cost) as BIGINT) as value
# MAGIC FROM gold.immunization_master_outreach_list
# MAGIC WHERE outreach_tier IN ('Tier 1 - Immediate', 'Tier 2 - High Priority')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export Lists for Outreach Team

# COMMAND ----------

# Export to CSV for outreach teams (save to DBFS)
output_path = "/dbfs/mnt/medicaid/outreach_exports"

# Tier 1 export
tier1_export = final_outreach_list.filter(col("outreach_tier") == "Tier 1 - Immediate")
tier1_pdf = tier1_export.toPandas()
tier1_pdf.to_csv(f"{output_path}/tier1_immunization_outreach.csv", index=False)
print(f"✓ Tier 1 exported: {len(tier1_pdf):,} members")

# Tier 2 export
tier2_export = final_outreach_list.filter(col("outreach_tier") == "Tier 2 - High Priority")
tier2_pdf = tier2_export.toPandas()
tier2_pdf.to_csv(f"{output_path}/tier2_immunization_outreach.csv", index=False)
print(f"✓ Tier 2 exported: {len(tier2_pdf):,} members")

# County-specific exports (top 5 counties)
top_counties = spark.sql("""
SELECT county
FROM gold.immunization_master_outreach_list
WHERE outreach_tier IN ('Tier 1 - Immediate', 'Tier 2 - High Priority')
GROUP BY county
ORDER BY COUNT(*) DESC
LIMIT 5
""").toPandas()['county'].tolist()

for county in top_counties:
    county_data = final_outreach_list.filter(
        (col("county") == county) &
        (col("outreach_tier").isin(["Tier 1 - Immediate", "Tier 2 - High Priority"]))
    ).toPandas()
    county_safe = county.replace(" ", "_").replace("/", "_")
    county_data.to_csv(f"{output_path}/county_{county_safe}_outreach.csv", index=False)
    print(f"✓ {county} exported: {len(county_data):,} members")

# COMMAND ----------

print("=" * 60)
print("IMMUNIZATION TARGETING ANALYTICS COMPLETE")
print("=" * 60)
print("\nKey Outputs:")
print("  ✓ Master outreach list: gold.immunization_master_outreach_list")
print("  ✓ Tier 1 priority list: gold.immunization_tier1_outreach")
print("  ✓ Geographic summary: gold.immunization_geographic_summary")
print("  ✓ Risk factor analysis: gold.immunization_risk_factor_analysis")
print("  ✓ Program ROI: gold.immunization_program_roi")
print(f"\n  ✓ CSV exports saved to: {output_path}/")
print("\nReady for program deployment!")
