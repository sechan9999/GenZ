"""
Databricks Quick Start: MLOps LIMS Pipeline

This notebook demonstrates the complete end-to-end pipeline in Databricks.
Run each cell sequentially to set up and test the MLOps pipeline.

Author: MLOps Team
Date: 2025-11-22
"""

# COMMAND ----------
# MAGIC %md
# MAGIC # MLOps LIMS Pipeline - Quick Start
# MAGIC
# MAGIC This notebook sets up the complete MLOps pipeline for LIMS data.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Databricks workspace with ML Runtime 14.0+
# MAGIC - LIMS database JDBC connection configured
# MAGIC - Delta Lake storage mounted at `/mnt/delta/lims/`

# COMMAND ----------
# Install dependencies (if not in cluster libraries)
%pip install mlflow scikit-learn scipy fastapi uvicorn

# COMMAND ----------
# Import libraries
from pyspark.sql import SparkSession
from datetime import datetime, timedelta
import mlflow
import os

# Configure paths
BRONZE_PATH = "/mnt/delta/lims/bronze"
SILVER_PATH = "/mnt/delta/lims/silver"
GOLD_PATH = "/mnt/delta/lims/gold"
OUTPUT_PATH = "/mnt/delta/lims/predictions"

print("✓ Configuration loaded")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: Bronze Layer - Raw Data Ingestion
# MAGIC
# MAGIC Ingest raw data from LIMS database into Delta Lake.

# COMMAND ----------
# Copy pipeline code to Databricks (in production, use Git integration)
# For demo, we'll define inline

from pipelines.bronze_ingestion import BronzeLayerIngestion

# Initialize Bronze ingestion
bronze = BronzeLayerIngestion(spark, BRONZE_PATH)

# Configure LIMS source
source_config = {
    'lab_results': dbutils.secrets.get(scope="lims", key="jdbc_url"),
    'device_logs': dbutils.secrets.get(scope="lims", key="jdbc_url"),
    'quality_control': dbutils.secrets.get(scope="lims", key="jdbc_url")
}

# Run incremental ingestion (last 24 hours)
stats = bronze.run_incremental_ingestion(source_config, lookback_hours=24)

print("\n=== Bronze Layer Ingestion Results ===")
for table, result in stats.items():
    print(f"{table}: {result['records_written']} records ingested")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: Silver Layer - Data Standardization
# MAGIC
# MAGIC Clean, validate, and standardize data with LOINC mapping and PII hashing.

# COMMAND ----------
from pipelines.silver_standardization import SilverLayerStandardization

# Initialize Silver standardization
silver = SilverLayerStandardization(
    spark,
    bronze_path=BRONZE_PATH,
    silver_path=SILVER_PATH,
    loinc_mapping_path=None  # Using built-in sample mapping
)

# Run standardization pipeline
stats = silver.run_standardization_pipeline()

print("\n=== Silver Layer Standardization Results ===")
for table, result in stats.items():
    print(f"\n{table}:")
    for key, value in result.items():
        print(f"  {key}: {value}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: Gold Layer - Feature Engineering
# MAGIC
# MAGIC Create ML-ready features for device failure, outbreak risk, and quality anomaly detection.

# COMMAND ----------
from pipelines.gold_feature_engineering import GoldLayerFeatureEngineering

# Initialize Gold feature engineering
gold = GoldLayerFeatureEngineering(
    spark,
    silver_path=SILVER_PATH,
    gold_path=GOLD_PATH
)

# Run feature engineering pipeline
stats = gold.run_feature_engineering_pipeline()

print("\n=== Gold Layer Feature Engineering Results ===")
for feature_set, result in stats.items():
    print(f"\n{feature_set}:")
    for key, value in result.items():
        print(f"  {key}: {value}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Model Training with MLflow
# MAGIC
# MAGIC Train device failure prediction model and track with MLflow.

# COMMAND ----------
from models.train_device_failure_model import DeviceFailureModelTrainer

# Set MLflow experiment
mlflow.set_experiment("/LIMS/device_failure_prediction")

# Initialize trainer
trainer = DeviceFailureModelTrainer(
    spark,
    gold_path=GOLD_PATH,
    mlflow_experiment_name="/LIMS/device_failure_prediction"
)

# Prepare training data
print("Preparing training data...")
X, y = trainer.prepare_training_data(lookback_days=90)

print(f"Training samples: {len(X)}")
print(f"Failure rate: {y.mean():.2%}")

# Train model
print("\nTraining model...")
model, metrics = trainer.train_model(X, y)

print("\n=== Model Training Results ===")
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1 Score: {metrics['f1']:.3f}")
print(f"ROC-AUC: {metrics['roc_auc']:.3f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 5: Register Model to MLflow Model Registry
# MAGIC
# MAGIC Manually register the best model from the MLflow UI:
# MAGIC 1. Go to Machine Learning → Experiments
# MAGIC 2. Select the experiment "/LIMS/device_failure_prediction"
# MAGIC 3. Find the best run (highest F1 score)
# MAGIC 4. Click "Register Model" → Name: "device_failure_predictor"
# MAGIC 5. Transition to "Production" stage

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 6: Batch Scoring
# MAGIC
# MAGIC Score all devices and generate daily report.

# COMMAND ----------
from deployment.batch_scoring import BatchScoringJob

# Initialize batch scoring
batch = BatchScoringJob(
    spark,
    gold_path=GOLD_PATH,
    output_path=OUTPUT_PATH,
    model_name="device_failure_predictor",
    model_stage="Production"
)

# Run batch scoring (for yesterday)
print("Running batch scoring...")
stats = batch.run(send_alerts=True)

print("\n=== Batch Scoring Results ===")
print(f"Status: {stats['status']}")
print(f"Devices scored: {stats['devices_scored']}")
print(f"Risk distribution: {stats['risk_distribution']}")
print(f"Excel report: {stats['excel_report']}")
print(f"Duration: {stats['duration_seconds']:.2f} seconds")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 7: Data Drift Monitoring
# MAGIC
# MAGIC Detect data drift and calibration errors.

# COMMAND ----------
from monitoring.drift_detection import DriftDetector

# Define baseline period (90 days of "normal" data)
baseline_start = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
baseline_end = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

print(f"Baseline period: {baseline_start} to {baseline_end}")

# Initialize drift detector
detector = DriftDetector(
    spark,
    silver_path=SILVER_PATH,
    gold_path=GOLD_PATH,
    baseline_start_date=baseline_start,
    baseline_end_date=baseline_end
)

# Check last 7 days for drift
current_start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
current_end = datetime.now().strftime("%Y-%m-%d")

print(f"Current period: {current_start} to {current_end}")

# Generate drift report
report = detector.generate_drift_report(current_start, current_end)

print("\n=== Drift Detection Report ===")
print(f"Alert Level: {report['alert_level']}")
print(f"\nSummary:")
for key, value in report['summary'].items():
    print(f"  {key}: {value}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 8: Query Results
# MAGIC
# MAGIC Query the Delta Lake tables to view results.

# COMMAND ----------
# MAGIC %sql
# MAGIC -- View high-risk devices
# MAGIC SELECT
# MAGIC   device_id,
# MAGIC   risk_score,
# MAGIC   risk_level,
# MAGIC   failure_probability,
# MAGIC   recommendation
# MAGIC FROM delta.`/mnt/delta/lims/predictions/device_risk_scores`
# MAGIC WHERE risk_level IN ('CRITICAL', 'HIGH')
# MAGIC ORDER BY risk_score DESC
# MAGIC LIMIT 10

# COMMAND ----------
# MAGIC %sql
# MAGIC -- View data drift alerts
# MAGIC SELECT
# MAGIC   loinc_code,
# MAGIC   test_name,
# MAGIC   drift_detected,
# MAGIC   ks_statistic,
# MAGIC   p_value,
# MAGIC   mean_shift_percent
# MAGIC FROM drift_report_data_drift
# MAGIC WHERE drift_detected = true
# MAGIC ORDER BY ks_statistic DESC

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 9: Create Databricks SQL Dashboard
# MAGIC
# MAGIC 1. Go to SQL → Dashboards
# MAGIC 2. Create new dashboard "LIMS MLOps Dashboard"
# MAGIC 3. Add visualizations:
# MAGIC    - Device Risk Heatmap
# MAGIC    - Drift Detection Timeline
# MAGIC    - Model Performance Metrics
# MAGIC    - Daily QC Pass Rates

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 10: Schedule Jobs
# MAGIC
# MAGIC Create Databricks Jobs to run the pipeline on a schedule:
# MAGIC
# MAGIC 1. **Bronze Ingestion**: Hourly
# MAGIC 2. **Silver Standardization**: Hourly (after Bronze)
# MAGIC 3. **Gold Feature Engineering**: Daily at 1 AM
# MAGIC 4. **Model Training**: Weekly (Sunday 3 AM)
# MAGIC 5. **Batch Scoring**: Daily at 2 AM
# MAGIC 6. **Drift Monitoring**: Daily at 4 AM

# COMMAND ----------
print("""
✅ MLOps LIMS Pipeline Setup Complete!

Next Steps:
1. Register model in MLflow Model Registry
2. Set up Databricks Jobs for automation
3. Create SQL dashboards for monitoring
4. Configure email/Slack alerts
5. Deploy API (optional) for real-time predictions

For documentation, see: /dbfs/mnt/delta/lims/README.md
""")
