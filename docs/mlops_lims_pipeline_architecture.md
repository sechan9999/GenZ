# MLOps Pipeline for LIMS Data - Architecture Document

**Version**: 1.0.0
**Date**: 2025-11-22
**Status**: Production-Ready Design

## Executive Summary

This document describes a production-grade MLOps pipeline for Laboratory Information Management System (LIMS) data using Azure Databricks, Delta Lake, and MLflow. The pipeline transforms static lab data into a living, continuously-learning system that predicts device failures, detects outbreaks, and monitors data quality.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LIMS Data Source                             │
│                    (Production Lab Systems)                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             │ Automated Ingestion
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      BRONZE LAYER (Raw Data)                         │
│                         Delta Lake Tables                            │
│  - lab_results_raw                                                   │
│  - device_logs_raw                                                   │
│  - quality_control_raw                                               │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             │ Standardization & Cleaning
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     SILVER LAYER (Cleaned Data)                      │
│                         Delta Lake Tables                            │
│  - lab_results_standardized (LOINC codes)                           │
│  - device_metrics_normalized                                         │
│  - pii_hashed (patient identifiers)                                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             │ Feature Engineering
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      GOLD LAYER (Features)                           │
│                         Delta Lake Tables                            │
│  - device_failure_features                                           │
│  - outbreak_risk_features                                            │
│  - quality_metrics_aggregated                                        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             │ Model Training (MLflow)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        ML MODELS (MLflow)                            │
│  - Device Failure Predictor (Random Forest)                         │
│  - Outbreak Risk Detector (XGBoost)                                 │
│  - Quality Anomaly Detector (Isolation Forest)                      │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             │ Deployment
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      MODEL SERVING                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  Real-Time API   │  │   Batch Jobs     │  │   Monitoring     │  │
│  │  (REST/FastAPI)  │  │  (Daily Reports) │  │  (Data Drift)    │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Design Principles

### 1. Decoupling
- **Problem**: LIMS systems are critical production systems; data extraction must not impact lab operations
- **Solution**: Use change data capture (CDC) or scheduled batch exports to Delta Lake
- **Implementation**: Azure Data Factory pipelines with incremental loads

### 2. Data Quality & Standardization
- **Problem**: LIMS data has inconsistent units, local codes, and varying formats
- **Solution**: Silver layer enforces LOINC standardization and unit normalization
- **Implementation**: PySpark transformations with validation rules

### 3. Privacy & Security
- **Problem**: Lab data contains PHI (Protected Health Information)
- **Solution**: Hash PII at ingestion, implement row-level security
- **Implementation**: SHA-256 hashing with salt, Delta Lake access controls

### 4. Model Lifecycle Management
- **Problem**: Models must be reproducible, versioned, and auditable
- **Solution**: MLflow tracks all experiments, parameters, and data versions
- **Implementation**: Every model run links to Delta Lake version/timestamp

### 5. Continuous Monitoring
- **Problem**: Lab instrument drift can corrupt data before humans notice
- **Solution**: Automated data drift detection with alerting
- **Implementation**: Statistical tests (KS, PSI) + alerting via Azure Monitor

## Data Flow Details

### Bronze Layer: Raw Ingestion
**Purpose**: Immutable copy of source data
**Frequency**: Hourly (CDC) or Daily (batch)
**Format**: Delta Lake (schema enforcement enabled)

**Tables**:
```sql
lab_results_raw (
    result_id STRING,
    patient_id STRING,
    test_code STRING,          -- Local lab codes
    result_value STRING,       -- Mixed types/units
    result_unit STRING,
    device_id STRING,
    collected_timestamp TIMESTAMP,
    resulted_timestamp TIMESTAMP,
    technician_id STRING,
    ingestion_timestamp TIMESTAMP
)

device_logs_raw (
    device_id STRING,
    log_timestamp TIMESTAMP,
    event_type STRING,
    severity STRING,
    message STRING,
    calibration_date TIMESTAMP,
    maintenance_date TIMESTAMP
)

quality_control_raw (
    qc_id STRING,
    device_id STRING,
    test_code STRING,
    qc_level STRING,
    expected_value DOUBLE,
    measured_value DOUBLE,
    passed BOOLEAN,
    timestamp TIMESTAMP
)
```

### Silver Layer: Standardization
**Purpose**: Clean, validated, standardized data
**Transformations**:
1. **LOINC Mapping**: Local test codes → standard LOINC codes
2. **Unit Normalization**: Convert all units to standard (mg/dL, mmol/L, etc.)
3. **PII Hashing**: Hash patient_id, technician_id
4. **Data Validation**: Remove nulls, check ranges, flag anomalies
5. **Deduplication**: Remove duplicate records

**Tables**:
```sql
lab_results_standardized (
    result_id STRING,
    patient_hash STRING,        -- SHA-256 hashed
    loinc_code STRING,          -- e.g., "2345-7" for glucose
    test_name STRING,           -- Human-readable
    result_value DOUBLE,        -- Numeric only
    result_unit_standard STRING,-- Standardized unit
    device_id STRING,
    collected_date DATE,
    resulted_date DATE,
    collected_hour INT,         -- For time-based features
    technician_hash STRING,
    is_critical BOOLEAN,        -- Flagged critical values
    is_valid BOOLEAN,           -- Passed validation
    ingestion_timestamp TIMESTAMP
)

device_metrics_normalized (
    device_id STRING,
    device_type STRING,
    log_date DATE,
    log_hour INT,
    error_count INT,
    warning_count INT,
    calibration_days_since INT,
    maintenance_days_since INT,
    qc_pass_rate DOUBLE,        -- Daily QC pass rate
    result_volume INT           -- Tests run per day
)
```

### Gold Layer: Feature Engineering
**Purpose**: ML-ready features for specific use cases

**Device Failure Features**:
```python
# Features for predicting device failure in next 7 days
- calibration_overdue (boolean)
- maintenance_overdue (boolean)
- error_rate_7d (rolling 7-day error rate)
- qc_fail_rate_7d (rolling QC failure rate)
- result_volume_trend (increasing = wearing out)
- device_age_months
- last_failure_days (days since last failure)
```

**Outbreak Risk Features**:
```python
# Features for detecting potential disease outbreaks
- positive_rate_7d (% positive tests for pathogen)
- case_count_delta (week-over-week change)
- geographic_cluster_score (spatial clustering)
- temporal_cluster_score (temporal clustering)
- seasonal_baseline_deviation
```

**Quality Anomaly Features**:
```python
# Features for detecting data quality issues
- result_distribution_shift (KS test p-value)
- outlier_rate (% results > 3 SD)
- missing_data_rate
- device_consistency_score (variance across devices)
```

## MLflow Model Lifecycle

### Experiment Tracking
Every model training run records:
- **Data Version**: Delta Lake version number or timestamp
- **Parameters**: All hyperparameters (learning rate, max_depth, etc.)
- **Metrics**: Accuracy, precision, recall, F1, AUC-ROC
- **Artifacts**: Trained model, feature importance, confusion matrix

### Model Registry
- **Staging**: Newly trained models for validation
- **Production**: Validated models serving predictions
- **Archived**: Deprecated models for audit trail

### Model Versioning Strategy
```
device_failure_v1.0.0 → Baseline (Logistic Regression)
device_failure_v2.0.0 → Improved (Random Forest)
device_failure_v2.1.0 → Tuned hyperparameters
device_failure_v3.0.0 → New features (maintenance logs)
```

## Model Deployment Architecture

### Real-Time API (FastAPI)
**Use Case**: Instant alerts when device shows failure risk
**Latency**: < 100ms
**Infrastructure**: Azure Container Instances or AKS
**Endpoints**:
- `POST /predict/device-failure` - Returns risk score 0-100
- `POST /predict/outbreak-risk` - Returns risk level (low/medium/high)
- `GET /health` - Health check

### Batch Scoring
**Use Case**: Daily reports for lab managers and governors
**Schedule**: Daily at 2 AM
**Infrastructure**: Databricks Jobs
**Output**:
- CSV/Excel reports to Azure Blob Storage
- Power BI refresh trigger
- Email alerts for high-risk devices

## Model Monitoring & Drift Detection

### Data Drift Detection
**Method**: Kolmogorov-Smirnov (KS) test comparing training vs. production data
**Frequency**: Daily
**Thresholds**:
- KS statistic > 0.1 → Warning
- KS statistic > 0.2 → Critical alert

**Example Scenario**:
```
Device XYZ starts drifting due to calibration error:
- Training data: glucose mean=95 mg/dL, std=15
- Production data (Week 1): mean=95, std=15 ✓
- Production data (Week 4): mean=105, std=25 ⚠️
→ Alert triggered: "Device XYZ shows data drift, check calibration"
```

### Model Performance Monitoring
**Metrics**:
- Prediction accuracy (if ground truth available)
- Prediction distribution (should match training)
- Feature importance drift (are features behaving differently?)

### Alerting Rules
1. **Critical**: Data drift KS > 0.2 → Alert lab manager immediately
2. **Warning**: QC fail rate > 10% → Email lab technicians
3. **Info**: Model retrained → Notify data science team

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Data Storage | Azure Delta Lake | ACID transactions, time travel, schema evolution |
| Compute | Azure Databricks | Distributed processing, notebooks, jobs |
| ML Tracking | MLflow | Experiment tracking, model registry |
| Orchestration | Databricks Workflows | Schedule jobs, manage dependencies |
| Serving (Real-time) | FastAPI + Azure Container Instances | Low-latency API |
| Serving (Batch) | Databricks Jobs | Daily/weekly reports |
| Monitoring | Azure Monitor + Databricks | Alerts, dashboards |
| Visualization | Power BI | Executive dashboards |
| Security | Azure Key Vault | Secrets management |
| IAM | Azure AD + Delta Lake ACLs | Row-level security |

## Security & Compliance

### HIPAA Compliance
- **Encryption at Rest**: Azure Storage encryption (AES-256)
- **Encryption in Transit**: TLS 1.2+
- **Access Control**: Azure AD + Delta Lake row-level security
- **Audit Logging**: All data access logged (7-year retention)
- **PHI Handling**: All PII hashed before Silver layer

### Data Governance
- **Data Lineage**: Delta Lake + Unity Catalog track data flow
- **Data Quality**: Great Expectations validation rules
- **Change Management**: All schema changes versioned

## Operational Runbooks

### Daily Operations
1. **8 AM**: Check overnight batch job status
2. **9 AM**: Review data drift alerts
3. **10 AM**: Validate QC metrics
4. **As needed**: Respond to real-time API alerts

### Weekly Operations
1. **Monday**: Review model performance metrics
2. **Wednesday**: Check data quality reports
3. **Friday**: Review device maintenance schedule vs. predictions

### Monthly Operations
1. **Week 1**: Retrain models with latest data
2. **Week 2**: A/B test new model version
3. **Week 3**: Promote to production if improved
4. **Week 4**: Review architecture and costs

## Cost Optimization

### Databricks Cluster Strategy
- **Interactive Clusters**: Autoscaling 2-8 workers (data scientists)
- **Job Clusters**: Right-sized for each job (ephemeral)
- **Photon**: Enabled for SQL workloads (3x faster)

### Delta Lake Optimization
- **Partition Strategy**: Partition by `result_date` (daily)
- **Z-Order**: Optimize by `device_id`, `loinc_code`
- **Vacuum**: Retain 30 days (compliance requirement)

### Estimated Monthly Cost (10M records/month)
- Databricks Compute: $2,000
- Delta Lake Storage: $500
- Container Instances (API): $300
- Azure Monitor: $100
- **Total**: ~$3,000/month

## Success Metrics

### Technical Metrics
- **Data Freshness**: < 1 hour lag from LIMS to Bronze
- **Data Quality**: > 99% records pass validation
- **Model Accuracy**: > 85% for device failure prediction
- **API Latency**: p95 < 100ms
- **Data Drift Detection**: < 1% false positive rate

### Business Metrics
- **Device Downtime**: Reduced by 40% (predictive maintenance)
- **Lab Efficiency**: 20% fewer repeat tests (quality improvement)
- **Outbreak Detection**: 5-day earlier detection vs. manual review
- **Cost Savings**: $500K/year (reduced device failures + efficiency)

## Future Enhancements

### Phase 2: Advanced Analytics
- Real-time streaming from lab devices (Kafka + Spark Streaming)
- Deep learning for anomaly detection (autoencoders)
- Causal inference for root cause analysis

### Phase 3: Integration
- Integration with EHR systems (HL7 FHIR)
- Integration with public health surveillance (CDC reporting)
- Mobile app for lab managers (real-time alerts)

### Phase 4: AI-Driven Insights
- Natural language query interface (Databricks SQL + GPT-4)
- Automated report generation (GPT-4 + data)
- Prescriptive analytics (not just "what" but "why" and "what to do")

## Appendix

### Glossary
- **LIMS**: Laboratory Information Management System
- **LOINC**: Logical Observation Identifiers Names and Codes (standard lab test codes)
- **PHI**: Protected Health Information (HIPAA-regulated)
- **CDC**: Change Data Capture
- **KS Test**: Kolmogorov-Smirnov test for distribution comparison
- **PSI**: Population Stability Index (drift metric)

### References
- [LOINC Database](https://loinc.org/)
- [Delta Lake Documentation](https://docs.delta.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Databricks ML Best Practices](https://docs.databricks.com/machine-learning/)
- [HIPAA Compliance on Azure](https://docs.microsoft.com/en-us/azure/compliance/hipaa)

---

**Document Owner**: Data Engineering Team
**Reviewers**: Lab Operations, Data Science, Security/Compliance
**Next Review**: 2026-02-22
