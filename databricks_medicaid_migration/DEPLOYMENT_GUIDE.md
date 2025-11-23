# Medicaid Claims Analysis - Azure Databricks Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the Medicaid claims analysis pipeline to Azure Databricks.

## Prerequisites

### 1. Azure Resources
- Azure Databricks workspace (Premium tier for MLflow Model Registry)
- Azure Data Lake Storage Gen2 (ADLS Gen2) account
- Azure Key Vault for secrets management
- Appropriate RBAC permissions

### 2. Local Development Environment
- Python 3.9+
- Databricks CLI
- Git

### 3. Credentials
- Databricks personal access token
- Azure storage account key or SAS token
- MLflow tracking URI credentials

## Deployment Steps

### Step 1: Configure Databricks CLI

```bash
# Install Databricks CLI
pip install databricks-cli

# Configure authentication
databricks configure --token

# Enter your workspace URL and personal access token when prompted
# Workspace URL: https://<your-workspace>.azuredatabricks.net
# Token: <your-personal-access-token>

# Verify configuration
databricks workspace ls /
```

### Step 2: Set Up Storage Mounts

Create storage mounts in Databricks for data access:

```python
# Run this in a Databricks notebook

# Configure storage credentials
storage_account_name = "<your-storage-account>"
storage_account_key = dbutils.secrets.get(scope="medicaid", key="storage-account-key")

# Mount landing zone
dbutils.fs.mount(
    source=f"wasbs://landing@{storage_account_name}.blob.core.windows.net",
    mount_point="/mnt/medicaid/landing",
    extra_configs={
        f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net": storage_account_key
    }
)

# Mount bronze layer
dbutils.fs.mount(
    source=f"wasbs://bronze@{storage_account_name}.blob.core.windows.net",
    mount_point="/mnt/medicaid/bronze",
    extra_configs={
        f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net": storage_account_key
    }
)

# Mount silver layer
dbutils.fs.mount(
    source=f"wasbs://silver@{storage_account_name}.blob.core.windows.net",
    mount_point="/mnt/medicaid/silver",
    extra_configs={
        f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net": storage_account_key
    }
)

# Mount gold layer
dbutils.fs.mount(
    source=f"wasbs://gold@{storage_account_name}.blob.core.windows.net",
    mount_point="/mnt/medicaid/gold",
    extra_configs={
        f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net": storage_account_key
    }
)

# Verify mounts
display(dbutils.fs.mounts())
```

### Step 3: Configure Secrets

Set up Azure Key Vault-backed secrets scope:

```bash
# Create secret scope (if not exists)
databricks secrets create-scope --scope medicaid --initial-manage-principal users

# Add secrets
databricks secrets put --scope medicaid --key storage-account-key
databricks secrets put --scope medicaid --key mlflow-tracking-uri

# Verify secrets
databricks secrets list --scope medicaid
```

### Step 4: Deploy Notebooks

```bash
# Clone repository
git clone <repository-url>
cd databricks_medicaid_migration

# Set environment variables
export DATABRICKS_HOST=https://<your-workspace>.azuredatabricks.net
export DATABRICKS_TOKEN=<your-token>

# Deploy notebooks
python scripts/deploy_notebooks.py
```

Expected output:
```
============================================================
DEPLOYING NOTEBOOKS TO DATABRICKS
============================================================
Found 5 notebooks to deploy

✓ Uploaded: 01_bronze_ingestion
✓ Uploaded: 02_silver_cleaning
✓ Uploaded: 03_gold_features
✓ Uploaded: 04_ml_risk_models
✓ Uploaded: 05_immunization_targeting

============================================================
DEPLOYMENT COMPLETE: 5/5 notebooks uploaded
============================================================
```

### Step 5: Create Databases

Run this SQL in Databricks SQL editor or notebook:

```sql
-- Create databases
CREATE DATABASE IF NOT EXISTS bronze
    COMMENT 'Raw ingested data'
    LOCATION '/mnt/medicaid/bronze';

CREATE DATABASE IF NOT EXISTS silver
    COMMENT 'Cleaned and validated data'
    LOCATION '/mnt/medicaid/silver';

CREATE DATABASE IF NOT EXISTS gold
    COMMENT 'Feature-engineered analytical data'
    LOCATION '/mnt/medicaid/gold';

-- Verify
SHOW DATABASES;
```

### Step 6: Initial Data Load

Upload sample data to landing zone:

```bash
# Using Azure CLI
az storage blob upload-batch \
    --account-name <storage-account-name> \
    --destination landing/medical_claims/historical \
    --source ./sample_data/medical_claims \
    --auth-mode key

az storage blob upload-batch \
    --account-name <storage-account-name> \
    --destination landing/pharmacy_claims/historical \
    --source ./sample_data/pharmacy_claims \
    --auth-mode key

az storage blob upload-batch \
    --account-name <storage-account-name> \
    --destination landing/member_eligibility \
    --source ./sample_data/eligibility \
    --auth-mode key
```

### Step 7: Run Initial Pipeline

Execute notebooks in sequence:

```bash
# Option 1: Manual execution
# Go to Databricks workspace and run notebooks in order:
# 1. /Medicaid/Notebooks/01_bronze_ingestion
# 2. /Medicaid/Notebooks/02_silver_cleaning
# 3. /Medicaid/Notebooks/03_gold_features
# 4. /Medicaid/Notebooks/04_ml_risk_models
# 5. /Medicaid/Notebooks/05_immunization_targeting

# Option 2: Using Databricks CLI
databricks runs submit --json '{
  "run_name": "Initial Pipeline Run",
  "new_cluster": {
    "spark_version": "13.3.x-scala2.12",
    "node_type_id": "i3.xlarge",
    "num_workers": 2
  },
  "notebook_task": {
    "notebook_path": "/Medicaid/Notebooks/01_bronze_ingestion"
  }
}'
```

### Step 8: Create Scheduled Jobs

```bash
# Create production jobs
python scripts/create_jobs.py
```

Expected output:
```
============================================================
CREATING DATABRICKS JOBS
============================================================

Creating Bronze Ingestion job...
✓ Created job: Medicaid - Bronze Layer Ingestion (ID: 123456)

Creating Silver Cleaning job...
✓ Created job: Medicaid - Silver Layer Cleaning (ID: 123457)

Creating Feature Engineering job...
✓ Created job: Medicaid - Feature Engineering (ID: 123458)

Creating Model Training job...
✓ Created job: Medicaid - ML Model Training (ID: 123459)

Creating Risk Scoring job...
✓ Created job: Medicaid - Risk Scoring & Immunization Targeting (ID: 123460)

Creating End-to-End Pipeline job...
✓ Created job: Medicaid - End-to-End Pipeline (ID: 123461)

============================================================
JOB CREATION COMPLETE: 6/6 jobs created
============================================================
```

### Step 9: Configure MLflow Model Registry

```python
# Run in Databricks notebook
import mlflow
from mlflow.tracking import MlflowClient

# Set up MLflow
mlflow.set_tracking_uri("databricks")

# Verify experiment
experiment = mlflow.get_experiment_by_name("/Medicaid/Risk_Prediction_Models")
print(f"Experiment ID: {experiment.experiment_id}")

# Configure model registry permissions
client = MlflowClient()

# List registered models (after first training run)
registered_models = client.search_registered_models()
for model in registered_models:
    print(f"Model: {model.name}")
```

### Step 10: Validate Deployment

Run validation checks:

```sql
-- Check data volumes
SELECT 'bronze.medical_claims' as table_name, COUNT(*) as record_count
FROM bronze.medical_claims
UNION ALL
SELECT 'silver.medical_claims', COUNT(*) FROM silver.medical_claims
UNION ALL
SELECT 'gold.member_features', COUNT(*) FROM gold.member_features
UNION ALL
SELECT 'gold.member_risk_predictions', COUNT(*) FROM gold.member_risk_predictions;

-- Check model predictions
SELECT
    COUNT(*) as total_members,
    SUM(CASE WHEN high_risk_prediction = 1 THEN 1 ELSE 0 END) as high_risk_count,
    AVG(high_risk_score) as avg_risk_score,
    AVG(predicted_total_cost) as avg_predicted_cost
FROM gold.member_risk_predictions;

-- Check immunization targeting
SELECT
    outreach_tier,
    COUNT(*) as member_count,
    AVG(outreach_priority_score) as avg_priority
FROM gold.immunization_master_outreach_list
GROUP BY outreach_tier
ORDER BY avg_priority DESC;
```

## Performance Tuning

### Delta Lake Optimization

```sql
-- Optimize all tables
OPTIMIZE bronze.medical_claims ZORDER BY (member_id, claim_date);
OPTIMIZE silver.medical_claims ZORDER BY (member_id, claim_date);
OPTIMIZE gold.member_features ZORDER BY (member_id);
OPTIMIZE gold.member_risk_predictions ZORDER BY (member_id);

-- Vacuum old versions (be careful in production)
VACUUM bronze.medical_claims RETAIN 168 HOURS;  -- 7 days
VACUUM silver.medical_claims RETAIN 168 HOURS;
```

### Cluster Configuration

Recommended cluster configurations:

**Bronze Ingestion**:
- Workers: 2-4 i3.xlarge
- Driver: i3.xlarge
- Auto-scaling: Enabled
- Spot instances: Yes

**Silver Cleaning**:
- Workers: 2-4 i3.xlarge
- Driver: i3.xlarge
- Auto-scaling: Enabled

**Gold Features**:
- Workers: 4-8 i3.xlarge
- Driver: i3.2xlarge
- Auto-scaling: Enabled

**ML Training**:
- Workers: 4-8 i3.2xlarge
- Driver: i3.2xlarge
- GPU: Optional (for deep learning)

### Caching Strategy

```python
# Cache frequently accessed tables
spark.sql("CACHE TABLE gold.member_features")
spark.sql("CACHE TABLE gold.member_risk_predictions")

# Verify cached tables
spark.sql("SHOW TABLES IN gold").filter("isTemporary = true").show()
```

## Monitoring and Maintenance

### Daily Checks

1. **Data Quality Dashboard**
   - Check row counts for each layer
   - Review data quality scores
   - Verify no null primary keys

2. **Job Success Rates**
   - Monitor job completion times
   - Review failure alerts
   - Check retry patterns

3. **Model Performance**
   - Review prediction distributions
   - Check drift metrics
   - Validate score ranges

### Weekly Tasks

1. **Feature Store Health**
   - Verify feature freshness
   - Check for staleness
   - Review feature usage

2. **Cost Optimization**
   - Review cluster utilization
   - Optimize job schedules
   - Clean up old data

3. **Security Audit**
   - Review access logs
   - Validate permissions
   - Update secrets if needed

### Monthly Tasks

1. **Model Retraining**
   - Review model performance
   - Retrain if needed
   - Update model registry

2. **Pipeline Optimization**
   - Analyze query performance
   - Optimize slow queries
   - Review partitioning strategy

3. **Capacity Planning**
   - Review data growth
   - Plan storage expansion
   - Adjust cluster sizes

## Troubleshooting

### Common Issues

**Issue 1: Mount point errors**
```
Error: Directory already mounted
Solution: Unmount first with dbutils.fs.unmount("/mnt/medicaid/landing")
```

**Issue 2: Out of memory errors**
```
Error: java.lang.OutOfMemoryError
Solution:
1. Increase cluster size
2. Add more workers
3. Enable adaptive query execution
4. Partition data appropriately
```

**Issue 3: Delta table not found**
```
Error: Table or view not found: gold.member_features
Solution: Ensure previous notebooks have run successfully
Check with: SHOW TABLES IN gold;
```

**Issue 4: MLflow tracking issues**
```
Error: Cannot connect to MLflow tracking server
Solution:
1. Verify MLflow tracking URI
2. Check network connectivity
3. Validate credentials in secrets
```

## Security Best Practices

1. **Never hard-code credentials**
   - Use Azure Key Vault-backed secrets
   - Rotate secrets regularly

2. **Implement RBAC**
   - Grant minimal required permissions
   - Use service principals for automation
   - Audit access regularly

3. **Encrypt data**
   - Enable encryption at rest (ADLS Gen2)
   - Enable encryption in transit (HTTPS)
   - Use Delta Lake encryption features

4. **PHI Handling**
   - De-identify data before exporting
   - Restrict access to PII fields
   - Implement audit logging
   - Follow HIPAA compliance guidelines

## Cost Optimization

1. **Use spot instances for non-critical workloads**
2. **Enable auto-scaling clusters**
3. **Schedule jobs during off-peak hours**
4. **Set up cluster auto-termination (120 minutes)**
5. **Use Delta Lake caching and optimization**
6. **Review and delete unused tables/files**

## Support and Documentation

- **Internal Documentation**: Confluence space (link)
- **Runbook**: Azure DevOps Wiki (link)
- **Escalation**: data-team@example.com
- **On-call**: PagerDuty rotation

## Appendix

### A. Sample Data Schema

See `config/data_schema.json` for complete schema definitions.

### B. Configuration Files

- `config/model_config.yaml`: Model hyperparameters
- `config/feature_config.yaml`: Feature engineering settings

### C. Useful SQL Queries

```sql
-- Data lineage check
SELECT
    COUNT(DISTINCT mc.member_id) as bronze_members,
    COUNT(DISTINCT sc.member_id) as silver_members,
    COUNT(DISTINCT gf.member_id) as gold_members
FROM bronze.medical_claims mc
LEFT JOIN silver.medical_claims sc ON mc.member_id = sc.member_id
LEFT JOIN gold.member_features gf ON mc.member_id = gf.member_id;

-- Performance metrics
SELECT
    table_name,
    num_files,
    size_in_bytes / 1024 / 1024 / 1024 as size_gb,
    num_partitions
FROM (
    DESCRIBE DETAIL gold.member_features
);
```

---

**Deployment Guide Version**: 1.0.0
**Last Updated**: 2025-11-23
**Next Review Date**: 2025-12-23
