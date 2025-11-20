# Palantir Foundry EHR Integration - Examples

This directory contains configuration examples for integrating Palantir Foundry with the [EHR Streaming Pipeline](https://github.com/sechan9999/ehr-streaming-pipeline) project.

## Overview

The EHR Streaming Pipeline processes FHIR healthcare data through Azure Event Hubs and Databricks Delta Lake. These examples show how to connect Palantir Foundry to this pipeline for advanced analytics and healthcare applications.

## Files in This Directory

- **`foundry_ehr_quickstart.yml`** - Minimal configuration to get started quickly
- **`../docs/palantir_foundry_ehr_integration.md`** - Comprehensive integration guide with all configurations

## Architecture

```
┌─────────────────────┐
│  Azure Event Hubs   │
│   (FHIR Messages)   │
└──────────┬──────────┘
           │
           ├──────────────────────────┐
           │                          │
           ▼                          ▼
┌─────────────────────┐    ┌─────────────────────┐
│    Databricks       │    │ Palantir Foundry    │
│  (Delta Lake)       │◄───┤   (Analytics)       │
│  Bronze/Silver/Gold │    │                     │
└─────────────────────┘    └─────────────────────┘
```

## Quick Start

### Prerequisites

1. **EHR Streaming Pipeline** running on Databricks
2. **Palantir Foundry** instance with appropriate permissions
3. **Azure Event Hubs** with FHIR message stream
4. **Foundry CLI** installed: `pip install palantir-foundry-cli`

### Step 1: Set Environment Variables

Create a `.env.foundry` file:

```bash
# Azure
EVENTHUB_CONNECTION_STRING="Endpoint=sb://your-namespace.servicebus.windows.net/..."
AZURE_TENANT_ID="your-tenant-id"
AZURE_CLIENT_ID="your-client-id"
AZURE_CLIENT_SECRET="your-secret"

# Foundry
FOUNDRY_STACK_URL="https://your-stack.palantirfoundry.com"
FOUNDRY_TOKEN="your-token"
```

### Step 2: Authenticate with Foundry

```bash
foundry-cli auth login --stack your-stack.palantirfoundry.com
```

### Step 3: Deploy Quick Start Configuration

```bash
# Apply the quick start configuration
foundry-cli apply -f foundry_ehr_quickstart.yml

# Start the streaming pipeline
foundry-cli build start ehr-foundry-pipeline
```

### Step 4: Verify Data Flow

```bash
# Check pipeline status
foundry-cli build status ehr-foundry-pipeline

# View recent records
foundry-cli dataset preview ehr_fhir_bronze --limit 10
```

## Configuration Options

### Integration Approaches

**Option 1: Direct Delta Lake Connection** (Recommended)
- Connect Foundry to existing Databricks Delta tables
- Reuse Silver/Gold layers from EHR pipeline
- Best for batch analytics and reporting

**Option 2: Parallel Event Hubs Consumer**
- Foundry consumes from Event Hubs independently
- Parallel processing with Databricks
- Best for real-time dashboards and alerts

**Option 3: Databricks Lakehouse Federation**
- Federated queries across Databricks and Foundry
- Unified query interface
- Best for hybrid analytical workloads

### Data Layers

Following the medallion architecture:

1. **Bronze Layer** - Raw FHIR messages from Event Hubs
2. **Silver Layer** - Normalized FHIR resources (Observations, Medications)
3. **Gold Layer** - Aggregated clinical summaries and analytics

## Common Use Cases

### 1. Clinical Dashboard

Query aggregated patient vitals from Gold layer:

```python
from foundry_workspace import Dataset

summary = Dataset.get("ehr_patient_clinical_summary_gold")
df = summary.read().toPandas()

# Get patients with hypertension
hypertensive = df[df['most_recent_vital_signs.blood_pressure_systolic'] >= 140]
```

### 2. Real-Time Alerts

Set up alerts on streaming Bronze layer:

```yaml
alert:
  name: critical-vital-alert
  dataset: ehr_fhir_bronze
  condition: |
    resourceType = 'Observation' AND
    code.coding[0].code = '8480-6' AND
    valueQuantity.value > 180
  action:
    slack: "#clinical-alerts"
```

### 3. Research Cohort Building

Build patient cohorts from Silver layer:

```python
observations = Dataset.get("ehr_observations_silver")
medications = Dataset.get("ehr_medication_statements_silver")

# Find diabetic patients on metformin
diabetic_cohort = observations.join(
    medications,
    on='patient_id'
).filter(
    (observations.observation_code == '15074-8') &  # HbA1c
    (medications.medication_code.contains('6809'))   # Metformin RxNorm
)
```

## Security & Compliance

### HIPAA Compliance

All datasets are classified as PHI (Protected Health Information):

- **Encryption**: AES-256 at rest, TLS 1.2 in transit
- **Access Control**: Role-based access (RBAC)
- **Audit Logging**: 7-year retention for HIPAA compliance
- **De-identification**: Available for research use cases

### Data Access Roles

1. **ehr-data-engineer** - Full pipeline access
2. **clinical-analyst** - Read Silver/Gold layers
3. **healthcare-researcher** - De-identified Gold layer only

## Monitoring

### Pipeline Health

```bash
# Check streaming lag
foundry-cli metrics get ehr-foundry-pipeline --metric lag

# View data quality checks
foundry-cli quality-check list ehr_observations_silver

# Export audit logs
foundry-cli audit export --dataset ehr_fhir_bronze --days 30
```

### Key Metrics

- **Ingestion Rate**: Messages/second from Event Hubs
- **Processing Lag**: Time between event and processing
- **Data Quality**: Null rates, validation failures
- **Completeness**: Record counts by layer

## Troubleshooting

### Issue: Streaming Pipeline Not Starting

```bash
# Check connection status
foundry-cli connection test ehr-eventhub-streaming

# Verify Event Hub credentials
echo $EVENTHUB_CONNECTION_STRING

# Check Foundry permissions
foundry-cli auth whoami
```

### Issue: High Processing Lag

```bash
# Increase worker count
foundry-cli build scale ehr-foundry-pipeline --workers 8

# Check checkpoint health
foundry-cli checkpoint status /checkpoints/ehr-stream
```

### Issue: Data Quality Failures

```bash
# View failed records
foundry-cli quality-check results ehr_observations_silver --status FAILED

# Reprocess with corrections
foundry-cli build reprocess ehr-foundry-pipeline --from-checkpoint
```

## Advanced Topics

### Custom Transformations

See `/home/user/GenZ/docs/palantir_foundry_ehr_integration.md` for:

- PySpark transformation code
- FHIR parsing logic
- Data quality checks
- Aggregation pipelines

### Ontology Development

Build healthcare object types:

- Patient
- Observation
- MedicationStatement
- Encounter
- Condition

### Application Development

Use Foundry's Object Explorer and Workshop for:

- Clinical decision support
- Population health analytics
- Predictive modeling
- Adverse event detection

## Resources

### Documentation

- [Palantir Foundry Docs](https://www.palantir.com/docs/foundry/)
- [FHIR Specification](https://hl7.org/fhir/)
- [EHR Streaming Pipeline](https://github.com/sechan9999/ehr-streaming-pipeline)
- [Delta Lake Documentation](https://docs.delta.io/)

### Related Projects

- **GenZ Agent** - Multi-agent system for election data (this repo)
- **EHR Streaming Pipeline** - Source FHIR data pipeline
- **K21 Election** - Original election analysis project

## Support

For questions or issues:

1. Check the comprehensive guide: `../docs/palantir_foundry_ehr_integration.md`
2. Review EHR pipeline docs: https://github.com/sechan9999/ehr-streaming-pipeline
3. Consult Foundry documentation: https://www.palantir.com/docs/foundry/

---

**Last Updated**: 2025-11-20
**Version**: 1.0.0
**Maintainer**: Data Engineering Team
