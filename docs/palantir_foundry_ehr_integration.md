# Palantir Foundry Integration with EHR Streaming Pipeline

## Overview

This document describes how to configure Palantir Foundry to integrate with the EHR Streaming Pipeline project that processes FHIR healthcare data from Azure Event Hubs through Databricks Delta Lake.

## Architecture

```
Azure Event Hubs → Databricks (Bronze/Silver/Gold) → Palantir Foundry
                                                    ↓
                                            Healthcare Analytics
                                            Clinical Dashboards
                                            ML/AI Models
```

## Integration Approaches

### Approach 1: Delta Lake Direct Connection (Recommended)

Connect Foundry directly to the Delta Lake tables created by the Databricks pipeline.

### Approach 2: Event Hubs Streaming

Configure Foundry to consume from the same Azure Event Hubs for parallel processing.

### Approach 3: Databricks Lakehouse Federation

Use Foundry's Databricks integration to federate queries.

---

## Configuration Examples

### 1. Data Connection Configuration

#### 1.1 Azure Data Lake Storage Gen2 Connection

**File**: `foundry_connections/adls_gen2_connection.yml`

```yaml
apiVersion: v1
kind: DataConnection
metadata:
  name: ehr-delta-lake-connection
  description: Connection to EHR streaming pipeline Delta Lake storage
spec:
  connectionType: AZURE_DATA_LAKE_STORAGE_GEN2
  authentication:
    type: SERVICE_PRINCIPAL
    tenantId: ${AZURE_TENANT_ID}
    clientId: ${AZURE_CLIENT_ID}
    clientSecret: ${secret:azure-sp-secret}
  configuration:
    storageAccountName: ${ADLS_STORAGE_ACCOUNT}
    fileSystemName: ${ADLS_CONTAINER_NAME}
    basePath: /ehr-pipeline
  capabilities:
    - READ
    - WRITE
  tags:
    - healthcare
    - ehr
    - fhir
```

#### 1.2 Azure Event Hubs Connection

**File**: `foundry_connections/eventhub_connection.yml`

```yaml
apiVersion: v1
kind: DataConnection
metadata:
  name: ehr-eventhub-connection
  description: Connection to EHR FHIR message stream
spec:
  connectionType: AZURE_EVENT_HUBS
  authentication:
    type: SHARED_ACCESS_KEY
    connectionString: ${secret:eventhub-connection-string}
  configuration:
    namespace: ${EVENTHUB_NAMESPACE}
    eventHubName: ${EVENTHUB_NAME}
    consumerGroup: foundry-consumer-group
  streaming:
    enabled: true
    checkpointLocation: /foundry/checkpoints/ehr-stream
  tags:
    - healthcare
    - streaming
    - real-time
```

---

### 2. Dataset Definitions

#### 2.1 Bronze Layer - Raw FHIR Messages

**File**: `foundry_datasets/bronze_fhir_raw.yml`

```yaml
apiVersion: v1
kind: Dataset
metadata:
  name: ehr_fhir_raw_bronze
  description: Raw FHIR messages from Event Hubs (Bronze layer)
  rid: ri.foundry.main.dataset.bronze-fhir-raw
spec:
  source:
    connectionRid: ri.foundry.main.connection.ehr-eventhub-connection
    format: JSON
    schema:
      type: INFERRED
      mode: PERMISSIVE
  storage:
    type: DELTA
    location: /bronze/fhir_raw
    partitionBy:
      - _eventDate
  streaming:
    enabled: true
    trigger:
      type: CONTINUOUS
      interval: 30 seconds
  compliance:
    classification: PHI
    encryption: AES-256
    auditLogging: ENABLED
```

#### 2.2 Silver Layer - Normalized FHIR Observations

**File**: `foundry_datasets/silver_observations.yml`

```yaml
apiVersion: v1
kind: Dataset
metadata:
  name: ehr_observations_silver
  description: Normalized FHIR Observation resources (Silver layer)
  rid: ri.foundry.main.dataset.silver-observations
spec:
  source:
    type: DELTA_TABLE
    connectionRid: ri.foundry.main.connection.ehr-delta-lake-connection
    path: /silver/observations
  schema:
    fields:
      - name: observation_id
        type: STRING
        description: Unique FHIR Observation identifier
        nullable: false
        primaryKey: true

      - name: patient_id
        type: STRING
        description: Reference to Patient resource
        nullable: false
        indexed: true

      - name: encounter_id
        type: STRING
        description: Reference to Encounter resource
        nullable: true

      - name: observation_code
        type: STRING
        description: LOINC or SNOMED code
        nullable: false
        indexed: true

      - name: observation_display
        type: STRING
        description: Human-readable observation name
        nullable: true

      - name: value_quantity
        type: DOUBLE
        description: Numeric observation value
        nullable: true

      - name: value_unit
        type: STRING
        description: Unit of measurement (UCUM)
        nullable: true

      - name: effective_datetime
        type: TIMESTAMP
        description: Clinically relevant date/time
        nullable: false
        indexed: true

      - name: status
        type: STRING
        description: Observation status (registered, preliminary, final, etc.)
        nullable: false

      - name: ingestion_timestamp
        type: TIMESTAMP
        description: Pipeline ingestion time
        nullable: false

      - name: _source_system
        type: STRING
        description: Source EHR system identifier
        nullable: true

  partitioning:
    type: DATE
    column: effective_datetime
    format: YYYY-MM-DD

  optimization:
    zOrderBy:
      - patient_id
      - observation_code

  quality:
    constraints:
      - name: valid_status
        type: CHECK
        expression: status IN ('registered', 'preliminary', 'final', 'amended', 'corrected', 'cancelled')

      - name: future_date_check
        type: CHECK
        expression: effective_datetime <= CURRENT_TIMESTAMP()

  compliance:
    classification: PHI
    retention:
      duration: 7 YEARS
      policy: HIPAA_COMPLIANT
```

#### 2.3 Silver Layer - Medication Statements

**File**: `foundry_datasets/silver_medications.yml`

```yaml
apiVersion: v1
kind: Dataset
metadata:
  name: ehr_medication_statements_silver
  description: Normalized FHIR MedicationStatement resources (Silver layer)
  rid: ri.foundry.main.dataset.silver-medications
spec:
  source:
    type: DELTA_TABLE
    connectionRid: ri.foundry.main.connection.ehr-delta-lake-connection
    path: /silver/medication_statements
  schema:
    fields:
      - name: medication_statement_id
        type: STRING
        nullable: false
        primaryKey: true

      - name: patient_id
        type: STRING
        nullable: false
        indexed: true

      - name: medication_code
        type: STRING
        description: RxNorm or NDC code
        nullable: false
        indexed: true

      - name: medication_display
        type: STRING
        description: Medication name
        nullable: true

      - name: status
        type: STRING
        description: active, completed, entered-in-error, intended, stopped, on-hold
        nullable: false

      - name: effective_start
        type: TIMESTAMP
        nullable: true

      - name: effective_end
        type: TIMESTAMP
        nullable: true

      - name: dosage_text
        type: STRING
        nullable: true

      - name: dosage_route
        type: STRING
        description: Route of administration
        nullable: true

      - name: ingestion_timestamp
        type: TIMESTAMP
        nullable: false

  partitioning:
    type: DATE
    column: effective_start
    format: YYYY-MM

  compliance:
    classification: PHI
```

#### 2.4 Gold Layer - Patient Clinical Summary

**File**: `foundry_datasets/gold_patient_summary.yml`

```yaml
apiVersion: v1
kind: Dataset
metadata:
  name: ehr_patient_clinical_summary_gold
  description: Aggregated patient clinical summary (Gold layer)
  rid: ri.foundry.main.dataset.gold-patient-summary
spec:
  source:
    type: DERIVED
    dependsOn:
      - ri.foundry.main.dataset.silver-observations
      - ri.foundry.main.dataset.silver-medications
  schema:
    fields:
      - name: patient_id
        type: STRING
        nullable: false
        primaryKey: true

      - name: total_observations
        type: INTEGER

      - name: total_medications
        type: INTEGER

      - name: first_observation_date
        type: DATE

      - name: last_observation_date
        type: DATE

      - name: active_medications_count
        type: INTEGER

      - name: most_recent_vital_signs
        type: STRUCT
        fields:
          - name: blood_pressure_systolic
            type: DOUBLE
          - name: blood_pressure_diastolic
            type: DOUBLE
          - name: heart_rate
            type: DOUBLE
          - name: temperature
            type: DOUBLE
          - name: measurement_date
            type: TIMESTAMP

      - name: last_updated
        type: TIMESTAMP

  materialization:
    type: INCREMENTAL
    schedule: "0 */6 * * *"  # Every 6 hours
```

---

### 3. Pipeline Builder Transforms

#### 3.1 Bronze to Silver - FHIR Observation Transformation

**File**: `foundry_transforms/bronze_to_silver_observations.py`

```python
"""
Transform raw FHIR Observation messages to normalized Silver layer
"""
from transforms.api import transform_df, Input, Output
from pyspark.sql import functions as F
from pyspark.sql.types import *


@transform_df(
    Output("ri.foundry.main.dataset.silver-observations"),
    raw_fhir=Input("ri.foundry.main.dataset.bronze-fhir-raw"),
)
def normalize_fhir_observations(raw_fhir):
    """
    Parse and normalize FHIR Observation resources from raw JSON.

    Handles nested FHIR structure and extracts key clinical elements.
    """

    # Filter for Observation resources only
    observations = raw_fhir.filter(
        F.col("resourceType") == "Observation"
    )

    # Extract and flatten nested FHIR elements
    normalized = observations.select(
        # Identifiers
        F.col("id").alias("observation_id"),
        F.col("subject.reference").alias("patient_reference"),
        F.regexp_extract(F.col("subject.reference"), r"Patient/(.+)", 1).alias("patient_id"),
        F.col("encounter.reference").alias("encounter_reference"),
        F.regexp_extract(F.col("encounter.reference"), r"Encounter/(.+)", 1).alias("encounter_id"),

        # Coding
        F.col("code.coding")[0]["code"].alias("observation_code"),
        F.col("code.coding")[0]["display"].alias("observation_display"),
        F.col("code.coding")[0]["system"].alias("code_system"),

        # Value
        F.col("valueQuantity.value").alias("value_quantity"),
        F.col("valueQuantity.unit").alias("value_unit"),
        F.col("valueQuantity.system").alias("value_system"),
        F.col("valueString").alias("value_string"),
        F.col("valueCodeableConcept.coding")[0]["code"].alias("value_code"),

        # Temporal
        F.coalesce(
            F.to_timestamp("effectiveDateTime"),
            F.to_timestamp("effectivePeriod.start")
        ).alias("effective_datetime"),

        # Status and metadata
        F.col("status").alias("status"),
        F.current_timestamp().alias("ingestion_timestamp"),
        F.col("meta.source").alias("_source_system"),

        # Original body for lineage
        F.to_json(F.struct("*")).alias("_raw_json")
    )

    # Data quality filters
    validated = normalized.filter(
        (F.col("observation_id").isNotNull()) &
        (F.col("patient_id").isNotNull()) &
        (F.col("observation_code").isNotNull()) &
        (F.col("effective_datetime").isNotNull()) &
        (F.col("status").isNotNull())
    )

    # Add data quality flags
    quality_checked = validated.withColumn(
        "quality_flags",
        F.struct(
            (F.col("value_quantity").isNull() &
             F.col("value_string").isNull() &
             F.col("value_code").isNull()).alias("missing_value"),
            (F.col("effective_datetime") > F.current_timestamp()).alias("future_date"),
            F.col("status").isin(["entered-in-error", "cancelled"]).alias("invalid_status")
        )
    )

    return quality_checked


@transform_df(
    Output("ri.foundry.main.dataset.silver-medications"),
    raw_fhir=Input("ri.foundry.main.dataset.bronze-fhir-raw"),
)
def normalize_fhir_medications(raw_fhir):
    """
    Parse and normalize FHIR MedicationStatement resources.
    """

    medications = raw_fhir.filter(
        F.col("resourceType") == "MedicationStatement"
    )

    normalized = medications.select(
        F.col("id").alias("medication_statement_id"),
        F.regexp_extract(F.col("subject.reference"), r"Patient/(.+)", 1).alias("patient_id"),

        # Medication coding
        F.col("medicationCodeableConcept.coding")[0]["code"].alias("medication_code"),
        F.col("medicationCodeableConcept.coding")[0]["display"].alias("medication_display"),
        F.col("medicationCodeableConcept.coding")[0]["system"].alias("code_system"),

        # Status
        F.col("status").alias("status"),

        # Effective period
        F.to_timestamp("effectivePeriod.start").alias("effective_start"),
        F.to_timestamp("effectivePeriod.end").alias("effective_end"),

        # Dosage
        F.col("dosage")[0]["text"].alias("dosage_text"),
        F.col("dosage")[0]["route.coding")[0]["display"].alias("dosage_route"),
        F.col("dosage")[0]["doseAndRate")[0]["doseQuantity.value"].alias("dose_value"),
        F.col("dosage")[0]["doseAndRate")[0]["doseQuantity.unit"].alias("dose_unit"),

        # Metadata
        F.current_timestamp().alias("ingestion_timestamp"),
        F.to_json(F.struct("*")).alias("_raw_json")
    )

    validated = normalized.filter(
        (F.col("medication_statement_id").isNotNull()) &
        (F.col("patient_id").isNotNull()) &
        (F.col("medication_code").isNotNull())
    )

    return validated
```

#### 3.2 Silver to Gold - Patient Clinical Summary

**File**: `foundry_transforms/silver_to_gold_patient_summary.py`

```python
"""
Aggregate Silver layer data into Gold patient clinical summaries
"""
from transforms.api import transform_df, Input, Output, incremental
from pyspark.sql import functions as F, Window


@incremental(snapshot_inputs=["observations", "medications"])
@transform_df(
    Output("ri.foundry.main.dataset.gold-patient-summary"),
    observations=Input("ri.foundry.main.dataset.silver-observations"),
    medications=Input("ri.foundry.main.dataset.silver-medications"),
)
def create_patient_summary(observations, medications):
    """
    Create comprehensive patient clinical summary from observations and medications.
    """

    # Aggregate observations per patient
    obs_summary = observations.groupBy("patient_id").agg(
        F.count("*").alias("total_observations"),
        F.min("effective_datetime").alias("first_observation_date"),
        F.max("effective_datetime").alias("last_observation_date")
    )

    # Get most recent vital signs
    vital_codes = {
        "8480-6": "blood_pressure_systolic",     # Systolic BP (LOINC)
        "8462-4": "blood_pressure_diastolic",    # Diastolic BP
        "8867-4": "heart_rate",                   # Heart rate
        "8310-5": "temperature"                   # Body temperature
    }

    # Window function to get latest vital per patient
    window_spec = Window.partitionBy("patient_id", "observation_code").orderBy(F.desc("effective_datetime"))

    latest_vitals = observations.filter(
        F.col("observation_code").isin(list(vital_codes.keys()))
    ).withColumn(
        "row_num", F.row_number().over(window_spec)
    ).filter(
        F.col("row_num") == 1
    ).select(
        "patient_id",
        "observation_code",
        "value_quantity",
        "effective_datetime"
    )

    # Pivot vitals to wide format
    vitals_wide = latest_vitals.groupBy("patient_id").pivot(
        "observation_code", list(vital_codes.keys())
    ).agg(
        F.first("value_quantity")
    ).select(
        "patient_id",
        F.col("8480-6").alias("blood_pressure_systolic"),
        F.col("8462-4").alias("blood_pressure_diastolic"),
        F.col("8867-4").alias("heart_rate"),
        F.col("8310-5").alias("temperature")
    )

    # Get vital measurement date
    latest_vital_date = latest_vitals.groupBy("patient_id").agg(
        F.max("effective_datetime").alias("measurement_date")
    )

    vitals_complete = vitals_wide.join(latest_vital_date, "patient_id", "left")

    # Create vital signs struct
    vitals_struct = vitals_complete.withColumn(
        "most_recent_vital_signs",
        F.struct(
            "blood_pressure_systolic",
            "blood_pressure_diastolic",
            "heart_rate",
            "temperature",
            "measurement_date"
        )
    ).select("patient_id", "most_recent_vital_signs")

    # Aggregate medications per patient
    med_summary = medications.groupBy("patient_id").agg(
        F.count("*").alias("total_medications"),
        F.sum(F.when(F.col("status") == "active", 1).otherwise(0)).alias("active_medications_count")
    )

    # Combine all summaries
    patient_summary = obs_summary.join(
        med_summary, "patient_id", "left"
    ).join(
        vitals_struct, "patient_id", "left"
    ).withColumn(
        "last_updated", F.current_timestamp()
    )

    # Fill nulls
    final_summary = patient_summary.fillna({
        "total_medications": 0,
        "active_medications_count": 0
    })

    return final_summary
```

---

### 4. Build Configuration

#### 4.1 Pipeline Build Specification

**File**: `foundry_build/ehr_pipeline_build.yml`

```yaml
apiVersion: v1
kind: PipelineBuild
metadata:
  name: ehr-streaming-pipeline
  description: EHR FHIR data processing pipeline from Event Hubs to Gold layer
spec:
  schedule:
    type: STREAMING
    checkpointInterval: 5 minutes

  stages:
    - name: bronze-ingestion
      type: STREAMING
      inputs: []
      outputs:
        - ri.foundry.main.dataset.bronze-fhir-raw
      resources:
        workerType: MEMORY_OPTIMIZED
        numWorkers: 4

    - name: silver-normalization
      type: STREAMING
      dependsOn:
        - bronze-ingestion
      inputs:
        - ri.foundry.main.dataset.bronze-fhir-raw
      outputs:
        - ri.foundry.main.dataset.silver-observations
        - ri.foundry.main.dataset.silver-medications
      transforms:
        - foundry_transforms/bronze_to_silver_observations.py
      resources:
        workerType: STANDARD
        numWorkers: 8

    - name: gold-aggregation
      type: BATCH
      schedule: "0 */6 * * *"  # Every 6 hours
      dependsOn:
        - silver-normalization
      inputs:
        - ri.foundry.main.dataset.silver-observations
        - ri.foundry.main.dataset.silver-medications
      outputs:
        - ri.foundry.main.dataset.gold-patient-summary
      transforms:
        - foundry_transforms/silver_to_gold_patient_summary.py
      resources:
        workerType: STANDARD
        numWorkers: 4

  monitoring:
    enabled: true
    alerts:
      - name: pipeline-failure
        condition: FAILURE
        channels:
          - slack: "#ehr-pipeline-alerts"
          - email: "data-team@example.com"

      - name: data-quality-issues
        condition: QUALITY_CHECK_FAILED
        channels:
          - slack: "#data-quality"

      - name: lag-alert
        condition: LAG > 15 minutes
        channels:
          - slack: "#ehr-pipeline-alerts"

  dataQuality:
    checks:
      - dataset: ri.foundry.main.dataset.silver-observations
        rules:
          - type: NOT_NULL
            columns: [observation_id, patient_id, observation_code]
          - type: UNIQUENESS
            columns: [observation_id]
          - type: FRESHNESS
            maxAge: 1 hour
          - type: CUSTOM_SQL
            name: valid_vital_ranges
            query: |
              SELECT COUNT(*) = 0 as passes
              FROM silver_observations
              WHERE observation_code = '8480-6' -- Systolic BP
                AND (value_quantity < 50 OR value_quantity > 300)

      - dataset: ri.foundry.main.dataset.silver-medications
        rules:
          - type: NOT_NULL
            columns: [medication_statement_id, patient_id]
          - type: UNIQUENESS
            columns: [medication_statement_id]
          - type: REFERENTIAL_INTEGRITY
            references:
              - table: silver_observations
                column: patient_id
```

---

### 5. Ontology Configuration

#### 5.1 FHIR Object Types

**File**: `foundry_ontology/fhir_object_types.yml`

```yaml
apiVersion: v1
kind: Ontology
metadata:
  name: ehr-fhir-ontology
  description: Healthcare ontology for FHIR resources

objectTypes:
  - name: Patient
    pluralName: Patients
    primaryKey: patient_id
    properties:
      - name: patient_id
        type: STRING
        required: true
      - name: mrn
        type: STRING
        description: Medical Record Number
      - name: date_of_birth
        type: DATE
      - name: gender
        type: STRING

    linkTypes:
      - name: has_observations
        targetObjectType: Observation
        cardinality: ONE_TO_MANY

      - name: has_medications
        targetObjectType: MedicationStatement
        cardinality: ONE_TO_MANY

  - name: Observation
    pluralName: Observations
    primaryKey: observation_id
    backingDataset: ri.foundry.main.dataset.silver-observations
    properties:
      - name: observation_id
        type: STRING
        datasetColumn: observation_id

      - name: patient_id
        type: STRING
        datasetColumn: patient_id

      - name: observation_code
        type: STRING
        datasetColumn: observation_code

      - name: observation_display
        type: STRING
        datasetColumn: observation_display

      - name: value_quantity
        type: DOUBLE
        datasetColumn: value_quantity

      - name: value_unit
        type: STRING
        datasetColumn: value_unit

      - name: effective_datetime
        type: TIMESTAMP
        datasetColumn: effective_datetime

      - name: status
        type: STRING
        datasetColumn: status

    searchableProperties:
      - observation_code
      - observation_display
      - patient_id

  - name: MedicationStatement
    pluralName: MedicationStatements
    primaryKey: medication_statement_id
    backingDataset: ri.foundry.main.dataset.silver-medications
    properties:
      - name: medication_statement_id
        type: STRING
        datasetColumn: medication_statement_id

      - name: patient_id
        type: STRING
        datasetColumn: patient_id

      - name: medication_code
        type: STRING
        datasetColumn: medication_code

      - name: medication_display
        type: STRING
        datasetColumn: medication_display

      - name: status
        type: STRING
        datasetColumn: status

      - name: effective_start
        type: TIMESTAMP
        datasetColumn: effective_start

      - name: effective_end
        type: TIMESTAMP
        datasetColumn: effective_end

    searchableProperties:
      - medication_code
      - medication_display
      - patient_id
```

---

### 6. Workshop & Applications

#### 6.1 Workshop Python Analysis

**File**: `foundry_workshop/ehr_clinical_analysis.py`

```python
"""
Foundry Workshop notebook for EHR clinical analysis
"""
from foundry_workspace import Dataset
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta


# Load Silver layer datasets
observations_ds = Dataset.get("ri.foundry.main.dataset.silver-observations")
medications_ds = Dataset.get("ri.foundry.main.dataset.silver-medications")

# Read as Pandas DataFrames
obs_df = observations_ds.read().toPandas()
med_df = medications_ds.read().toPandas()

# Example 1: Vital Signs Trend Analysis
def analyze_vital_trends(patient_id, vital_code, days=30):
    """
    Analyze vital sign trends for a specific patient.

    Args:
        patient_id: Patient identifier
        vital_code: LOINC code (e.g., '8480-6' for systolic BP)
        days: Number of days to analyze
    """
    cutoff_date = datetime.now() - timedelta(days=days)

    patient_vitals = obs_df[
        (obs_df['patient_id'] == patient_id) &
        (obs_df['observation_code'] == vital_code) &
        (obs_df['effective_datetime'] >= cutoff_date)
    ].sort_values('effective_datetime')

    # Create trend visualization
    fig = px.line(
        patient_vitals,
        x='effective_datetime',
        y='value_quantity',
        title=f'Vital Sign Trend: {patient_vitals["observation_display"].iloc[0]}',
        labels={'value_quantity': f'Value ({patient_vitals["value_unit"].iloc[0]})'}
    )

    return fig

# Example 2: Medication Adherence Check
def check_medication_gaps(patient_id):
    """
    Identify gaps in medication coverage.
    """
    patient_meds = med_df[
        (med_df['patient_id'] == patient_id) &
        (med_df['status'].isin(['active', 'completed']))
    ].sort_values('effective_start')

    gaps = []
    for i in range(len(patient_meds) - 1):
        current_end = patient_meds.iloc[i]['effective_end']
        next_start = patient_meds.iloc[i + 1]['effective_start']

        if pd.notna(current_end) and pd.notna(next_start):
            gap_days = (next_start - current_end).days
            if gap_days > 7:  # More than 7 days gap
                gaps.append({
                    'medication': patient_meds.iloc[i]['medication_display'],
                    'gap_start': current_end,
                    'gap_end': next_start,
                    'gap_days': gap_days
                })

    return pd.DataFrame(gaps)

# Example 3: Cohort Analysis
def create_hypertension_cohort():
    """
    Identify patients with hypertension based on BP readings.
    """
    # Get latest systolic BP for each patient
    systolic_readings = obs_df[
        obs_df['observation_code'] == '8480-6'
    ].sort_values('effective_datetime').groupby('patient_id').last()

    # Hypertension threshold: Systolic >= 140
    hypertension_patients = systolic_readings[
        systolic_readings['value_quantity'] >= 140
    ]['patient_id'].tolist()

    return hypertension_patients

# Example 4: Data Quality Report
def generate_quality_report():
    """
    Generate data quality metrics for EHR pipeline.
    """
    report = {
        'total_observations': len(obs_df),
        'total_medications': len(med_df),
        'unique_patients': obs_df['patient_id'].nunique(),
        'date_range': {
            'earliest': obs_df['effective_datetime'].min(),
            'latest': obs_df['effective_datetime'].max()
        },
        'null_values': {
            'obs_value_quantity': obs_df['value_quantity'].isna().sum(),
            'med_end_date': med_df['effective_end'].isna().sum()
        },
        'status_distribution': {
            'observations': obs_df['status'].value_counts().to_dict(),
            'medications': med_df['status'].value_counts().to_dict()
        }
    }

    return report
```

---

### 7. Security & Compliance Configuration

#### 7.1 Data Classification & Access Control

**File**: `foundry_security/data_governance.yml`

```yaml
apiVersion: v1
kind: DataGovernance
metadata:
  name: ehr-phi-governance
  description: HIPAA-compliant governance for EHR data

classification:
  - dataset: ri.foundry.main.dataset.bronze-fhir-raw
    level: PHI
    regulations:
      - HIPAA
      - HITECH
    encryption:
      atRest: AES-256
      inTransit: TLS-1.2

  - dataset: ri.foundry.main.dataset.silver-observations
    level: PHI
    regulations:
      - HIPAA

  - dataset: ri.foundry.main.dataset.gold-patient-summary
    level: PHI
    regulations:
      - HIPAA

accessControl:
  roles:
    - name: ehr-data-engineer
      description: Can read and write to all pipeline datasets
      permissions:
        - resource: "ri.foundry.main.dataset.*"
          actions: [READ, WRITE, BUILD]

    - name: clinical-analyst
      description: Can read Silver and Gold layers only
      permissions:
        - resource: "ri.foundry.main.dataset.silver-*"
          actions: [READ]
        - resource: "ri.foundry.main.dataset.gold-*"
          actions: [READ]

    - name: healthcare-researcher
      description: Read-only access to de-identified Gold layer
      permissions:
        - resource: "ri.foundry.main.dataset.gold-*"
          actions: [READ]
      conditions:
        - type: DE_IDENTIFICATION_APPLIED

auditLogging:
  enabled: true
  events:
    - DATA_ACCESS
    - DATA_EXPORT
    - SCHEMA_CHANGE
    - PERMISSION_CHANGE
  retention: 7 YEARS
  destination:
    type: AZURE_LOG_ANALYTICS
    workspaceId: ${AZURE_LOG_WORKSPACE_ID}
```

---

### 8. Environment Configuration

#### 8.1 Foundry Environment Variables

**File**: `.env.foundry`

```bash
# Azure Connection Details
AZURE_TENANT_ID=your-tenant-id
AZURE_CLIENT_ID=your-client-id
AZURE_CLIENT_SECRET=your-client-secret
ADLS_STORAGE_ACCOUNT=yourstorageaccount
ADLS_CONTAINER_NAME=ehr-pipeline

# Azure Event Hubs
EVENTHUB_NAMESPACE=your-eventhub-namespace
EVENTHUB_NAME=fhir-events
EVENTHUB_CONNECTION_STRING=Endpoint=sb://...

# Foundry Configuration
FOUNDRY_STACK_URL=https://your-stack.palantirfoundry.com
FOUNDRY_TOKEN=your-foundry-token

# Pipeline Settings
BRONZE_CHECKPOINT_PATH=/foundry/checkpoints/bronze
SILVER_CHECKPOINT_PATH=/foundry/checkpoints/silver
OUTPUT_BASE_PATH=/foundry/ehr-pipeline

# Data Quality
ANOMALY_THRESHOLD=2.0
DATA_QUALITY_ALERT_CHANNEL=#data-quality
```

#### 8.2 Foundry CLI Configuration

**File**: `foundry_cli/config.yml`

```yaml
version: 1
stack: your-stack.palantirfoundry.com
token: ${FOUNDRY_TOKEN}

defaults:
  project: EHR-Streaming-Pipeline
  branch: master

sync:
  include:
    - "foundry_transforms/**/*.py"
    - "foundry_datasets/**/*.yml"
    - "foundry_build/**/*.yml"
  exclude:
    - "**/__pycache__/**"
    - "**/*.pyc"
    - "**/test_*.py"
```

---

### 9. Deployment Instructions

#### Step-by-Step Deployment

```bash
# 1. Install Foundry CLI
pip install palantir-foundry-cli

# 2. Authenticate
foundry-cli auth login --stack your-stack.palantirfoundry.com

# 3. Create project
foundry-cli project create EHR-Streaming-Pipeline

# 4. Upload connections
foundry-cli connection apply -f foundry_connections/adls_gen2_connection.yml
foundry-cli connection apply -f foundry_connections/eventhub_connection.yml

# 5. Create datasets
foundry-cli dataset apply -f foundry_datasets/bronze_fhir_raw.yml
foundry-cli dataset apply -f foundry_datasets/silver_observations.yml
foundry-cli dataset apply -f foundry_datasets/silver_medications.yml
foundry-cli dataset apply -f foundry_datasets/gold_patient_summary.yml

# 6. Upload transforms
foundry-cli transform sync foundry_transforms/

# 7. Create pipeline build
foundry-cli build apply -f foundry_build/ehr_pipeline_build.yml

# 8. Apply security policies
foundry-cli governance apply -f foundry_security/data_governance.yml

# 9. Build ontology
foundry-cli ontology apply -f foundry_ontology/fhir_object_types.yml

# 10. Start streaming pipeline
foundry-cli build start ehr-streaming-pipeline
```

---

### 10. Monitoring & Observability

#### 10.1 Pipeline Monitoring Dashboard

**File**: `foundry_monitoring/pipeline_dashboard.yml`

```yaml
apiVersion: v1
kind: Dashboard
metadata:
  name: ehr-pipeline-monitoring
  description: Real-time monitoring for EHR streaming pipeline

widgets:
  - type: METRIC
    title: Ingestion Rate
    query: |
      SELECT
        COUNT(*) / 60 as messages_per_second,
        TUMBLE_END(event_time, INTERVAL '1' MINUTE) as window_end
      FROM bronze_fhir_raw
      GROUP BY TUMBLE(event_time, INTERVAL '1' MINUTE)
    visualization: LINE_CHART

  - type: METRIC
    title: Processing Lag
    query: |
      SELECT
        MAX(current_timestamp - ingestion_timestamp) as lag_seconds
      FROM silver_observations
    visualization: GAUGE
    thresholds:
      warning: 300   # 5 minutes
      critical: 900  # 15 minutes

  - type: TABLE
    title: Data Quality Issues
    query: |
      SELECT
        dataset_name,
        check_name,
        failures_count,
        last_check_time
      FROM data_quality_results
      WHERE status = 'FAILED'
      ORDER BY last_check_time DESC

  - type: METRIC
    title: Records by Layer
    query: |
      SELECT
        'Bronze' as layer, COUNT(*) as count FROM bronze_fhir_raw
      UNION ALL
      SELECT
        'Silver-Obs' as layer, COUNT(*) FROM silver_observations
      UNION ALL
      SELECT
        'Silver-Med' as layer, COUNT(*) FROM silver_medications
      UNION ALL
      SELECT
        'Gold' as layer, COUNT(*) FROM gold_patient_summary
    visualization: BAR_CHART

alerts:
  - name: high-lag-alert
    condition: lag_seconds > 900
    channels:
      - slack: "#ehr-pipeline-alerts"
    message: "EHR pipeline lag exceeds 15 minutes"

  - name: ingestion-stopped
    condition: messages_per_second = 0
    duration: 5 minutes
    channels:
      - slack: "#ehr-pipeline-alerts"
      - email: "oncall@example.com"
```

---

## Summary

This configuration demonstrates a complete integration between the EHR Streaming Pipeline (Databricks + Delta Lake) and Palantir Foundry, including:

1. **Data Connections**: Azure Event Hubs and ADLS Gen2
2. **Dataset Definitions**: Bronze, Silver, and Gold layers with schemas
3. **Transformations**: PySpark code for FHIR normalization and aggregation
4. **Pipeline Orchestration**: Streaming and batch build configurations
5. **Ontology Modeling**: FHIR object types for application development
6. **Security & Governance**: HIPAA-compliant data classification
7. **Monitoring**: Real-time dashboards and alerting
8. **Workshop Analytics**: Python notebooks for clinical analysis

The medallion architecture (Bronze → Silver → Gold) is preserved, and Foundry acts as both a parallel consumer of streaming data and an analytical layer on top of the Delta Lake tables.
