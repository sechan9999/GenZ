# Azure Databricks Medicaid Claims Analysis Migration

## Overview

This project migrates Medicaid claims analysis to Azure Databricks with MLOps practices for predictive risk modeling and immunization program targeting.

## Architecture

```
Claims Data Sources
      ↓
Bronze Layer (Raw Ingestion)
      ↓
Silver Layer (Cleaned & Validated)
      ↓
Gold Layer (Feature Engineering)
      ↓
ML Models (Risk Prediction & Utilization Forecasting)
      ↓
Dashboards & Immunization Targeting
```

## Key Features

- **Delta Lake Architecture**: Bronze/Silver/Gold medallion architecture
- **MLOps Integration**: MLflow tracking, Feature Store, Model Registry
- **Predictive Models**: Service utilization forecasting and risk stratification
- **Immunization Targeting**: Risk-based targeting for immunization programs
- **Performance Optimization**: Optimized Delta tables with Z-ordering and caching

## Components

### 1. Data Ingestion (`notebooks/01_bronze_ingestion.py`)
- Ingest claims, eligibility, provider, and diagnosis data
- Raw data validation and schema enforcement
- Incremental loading with change data capture

### 2. Data Cleaning (`notebooks/02_silver_cleaning.py`)
- Data quality checks and cleansing
- Deduplication and standardization
- Business rule validation

### 3. Feature Engineering (`notebooks/03_gold_features.py`)
- Clinical features (chronic conditions, utilization patterns)
- Demographic features (age groups, social determinants)
- Temporal features (seasonality, trends)
- Feature Store integration

### 4. Predictive Modeling (`notebooks/04_ml_risk_models.py`)
- Service utilization forecasting
- Readmission risk prediction
- High-cost member identification
- MLflow experiment tracking

### 5. MLOps Pipeline (`mlops/ml_pipeline.py`)
- Automated retraining
- Model versioning and registry
- A/B testing framework
- Model monitoring

### 6. Analytics & Targeting (`notebooks/05_immunization_targeting.py`)
- Risk stratification
- Immunization gap analysis
- Targeted outreach lists
- Program effectiveness tracking

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure Databricks CLI
databricks configure --token

# Deploy notebooks
python scripts/deploy_notebooks.py

# Run initial ingestion
databricks jobs run-now --job-id <job-id>
```

## MLflow Tracking

All models are tracked in MLflow with:
- Parameters (hyperparameters, feature sets)
- Metrics (AUC, precision, recall, RMSE)
- Artifacts (model files, feature importance)
- Tags (model version, deployment stage)

## Performance Improvements

- **Query Performance**: 10-50x faster with Delta Lake optimization
- **Feature Engineering**: Reproducible features via Feature Store
- **Model Deployment**: Automated retraining reduces staleness
- **Scalability**: Handles millions of claims records efficiently

## Directory Structure

```
databricks_medicaid_migration/
├── notebooks/              # Databricks notebooks
│   ├── 01_bronze_ingestion.py
│   ├── 02_silver_cleaning.py
│   ├── 03_gold_features.py
│   ├── 04_ml_risk_models.py
│   └── 05_immunization_targeting.py
├── src/                    # Reusable Python modules
│   ├── data_ingestion.py
│   ├── data_quality.py
│   ├── feature_engineering.py
│   └── model_utils.py
├── mlops/                  # MLOps pipeline
│   ├── ml_pipeline.py
│   ├── model_deployment.py
│   └── monitoring.py
├── config/                 # Configuration files
│   ├── data_schema.json
│   ├── feature_config.yaml
│   └── model_config.yaml
├── scripts/                # Deployment scripts
│   ├── deploy_notebooks.py
│   ├── create_jobs.py
│   └── setup_feature_store.py
├── tests/                  # Unit tests
└── requirements.txt
```

## License

Copyright 2025 - Medicaid Analytics Team
