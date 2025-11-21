# Gen Z Agent - Healthcare Workflow Automation Guide

## 📋 Overview

The Gen Z Healthcare Agent system provides AI-driven automation for clinical workflows using a multi-agent architecture powered by CrewAI and Anthropic Claude. The system is designed to be **HIPAA-compliant** and handles Protected Health Information (PHI) with appropriate security controls.

## 🏥 Key Features

### 1. Multi-Agent Clinical Workflows
- **FHIR Data Ingestion**: Parse and normalize HL7 FHIR R4 resources
- **Clinical Validation**: Validate data quality and clinical plausibility
- **Risk Analysis**: Identify high-risk patients and clinical concerns
- **Report Generation**: Create actionable clinical reports
- **Care Coordination**: Route alerts and coordinate care team actions

### 2. HIPAA Compliance
- **PHI Classification**: Automatic identification and marking of PHI
- **AES-256 Encryption**: Encrypt PHI data at rest and in transit
- **Audit Logging**: Comprehensive 7-year audit trail
- **De-identification**: HIPAA Safe Harbor de-identification method
- **Access Control**: Role-based access control (RBAC)

### 3. Clinical Intelligence
- **Vital Signs Monitoring**: Real-time monitoring with critical thresholds
- **Medication Safety**: Drug interaction checking, polypharmacy detection
- **Risk Stratification**: Evidence-based patient risk scoring
- **Care Gap Identification**: Identify missing preventive care
- **Clinical Decision Support**: Evidence-based recommendations

## 🏗️ Architecture

### Multi-Agent System

```
┌─────────────────────────────────────────────────────────────┐
│                   Healthcare Workflow                        │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                     ▼                      ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ FHIR Ingestion│───▶│   Clinical    │───▶│  Risk Analysis│
│     Agent     │    │  Validation   │    │     Agent     │
└───────────────┘    └───────────────┘    └───────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                            ▼
┌───────────────┐                          ┌───────────────┐
│ Clinical Report│                          │Care Coordination│
│     Agent     │                          │     Agent     │
└───────────────┘                          └───────────────┘
```

### Agent Responsibilities

| Agent | Role | Key Functions |
|-------|------|---------------|
| **FHIR Ingestion** | Data Integration | Parse FHIR resources, validate structure, extract clinical data |
| **Clinical Validation** | Data Quality | Validate ranges, check coding systems, enrich with terminology |
| **Risk Analysis** | Clinical Intelligence | Calculate risk scores, identify patterns, detect care gaps |
| **Clinical Report** | Documentation | Generate reports, format for audiences, ensure compliance |
| **Care Coordination** | Workflow | Route alerts, notify care teams, coordinate interventions |

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** with required packages
2. **Anthropic API Key** for Claude access
3. **FHIR Data** (JSON files or API endpoint)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY

# Verify setup
python gen_z_agent/healthcare_agents.py --help
```

### Basic Usage

#### Patient Risk Assessment

```bash
python gen_z_agent/healthcare_agents.py \
  patient_risk_assessment \
  --patient-id PAT001 \
  --data-source gen_z_agent/healthcare/fhir_data/
```

#### Medication Review

```bash
python gen_z_agent/healthcare_agents.py \
  medication_review \
  --patient-id PAT001 \
  --production  # Send actual notifications
```

### Python API Usage

```python
from gen_z_agent.healthcare_agents import run_clinical_workflow

# Run patient risk assessment
result = run_clinical_workflow(
    workflow_type="patient_risk_assessment",
    patient_id="PAT001",
    data_source="./fhir_data/",
    dry_run=False
)

print(result)
```

## 📊 Supported Workflows

### 1. Patient Risk Assessment

**Purpose**: Comprehensive evaluation of patient clinical risk

**Inputs**:
- Patient demographics
- Vital signs (last 90 days)
- Active medications
- Diagnoses
- Recent encounters

**Outputs**:
- Risk score (0-100)
- Risk category (LOW, MEDIUM, HIGH, CRITICAL)
- Identified risk factors
- Care gaps
- Actionable recommendations

**Use Cases**:
- Hospital discharge planning
- Primary care panel management
- Care management program enrollment
- Quality improvement initiatives

### 2. Medication Review

**Purpose**: Medication safety and appropriateness review

**Checks**:
- Drug-drug interactions
- High-risk medications
- Polypharmacy (5+ medications)
- Medication-diagnosis appropriateness
- Duplicate therapies

**Outputs**:
- Medication list with risk flags
- Interaction alerts
- Pharmacist review recommendations
- Patient education needs

**Use Cases**:
- Transitions of care
- Medication therapy management (MTM)
- Polypharmacy management
- Adverse drug event prevention

### 3. Vital Signs Monitoring

**Purpose**: Continuous monitoring with intelligent alerting

**Monitored Vitals**:
- Blood Pressure (Systolic/Diastolic)
- Heart Rate
- Body Temperature
- Respiratory Rate
- Oxygen Saturation

**Alert Levels**:
- **CRITICAL**: Immediate intervention needed
- **ABNORMAL**: Outside normal range
- **TRENDING**: Moving toward abnormal

**Outputs**:
- Real-time alerts
- Trend analysis
- Early warning scores
- Escalation protocols

**Use Cases**:
- Remote patient monitoring
- Post-discharge surveillance
- Chronic disease management
- Clinical deterioration detection

### 4. Care Gap Analysis

**Purpose**: Identify missing preventive care and screenings

**Checks**:
- Preventive screenings (mammography, colonoscopy, etc.)
- Immunizations
- Disease-specific monitoring (HbA1c for diabetes, etc.)
- Medication adherence
- Follow-up appointments

**Outputs**:
- Prioritized care gap list
- Patient outreach recommendations
- Scheduling suggestions
- Quality measure impact

**Use Cases**:
- Quality measure improvement
- Value-based care programs
- Preventive care campaigns
- Population health management

## 🔒 Security and Compliance

### HIPAA Requirements

#### 1. PHI Identification and Classification

All data containing the following is automatically classified as PHI:
- Names
- Geographic subdivisions smaller than state
- Dates (except year)
- Phone numbers, email addresses
- Medical record numbers (MRN)
- Account numbers
- Device identifiers
- IP addresses
- Any unique identifying numbers

#### 2. Encryption

**At Rest**:
- AES-256-GCM encryption for all PHI files
- Encrypted temporary storage
- Secure key management

**In Transit**:
- TLS 1.2+ for all network communications
- Encrypted API calls
- Secure webhook delivery

#### 3. Audit Logging

All PHI access is logged with:
- User ID
- Patient ID (hashed)
- Timestamp
- Action performed
- Resource accessed
- IP address
- Success/failure status

Logs are retained for **7 years** per HIPAA requirements.

#### 4. Access Control

**Role-Based Access**:

| Role | PHI Read | PHI Write | PHI Export | Resources |
|------|----------|-----------|------------|-----------|
| Physician | ✓ | ✓ | ✓ | All |
| Nurse | ✓ | ✓ | ✗ | Patient, Obs, Med, Encounter |
| Pharmacist | ✓ | ✓ | ✗ | Patient, Med |
| Researcher | ✗ | ✗ | ✗ | De-identified only |
| Admin | ✓ | ✓ | ✓ | All |

#### 5. De-identification

**HIPAA Safe Harbor Method**:
- Removes all 18 types of PHI identifiers
- Generalizes ages >89 years
- Removes geographic subdivisions
- Keeps only state-level geography
- Generates synthetic identifiers

**Usage**:
```python
from gen_z_agent.healthcare_security import phi_deidentifier

# De-identify patient data
deidentified = phi_deidentifier.deidentify_patient_data(patient_fhir)
```

### Security Best Practices

1. **API Keys**: Store in environment variables, never commit to code
2. **File Permissions**: Set restrictive permissions (0o600) on PHI files
3. **Network Security**: Use VPN/secure networks when accessing PHI
4. **Session Management**: 15-minute timeout, require re-authentication
5. **MFA**: Enable multi-factor authentication for production use
6. **Backups**: Encrypted backups with secure offsite storage
7. **Incident Response**: Document and report breaches within 60 days

## 📈 Clinical Decision Support

### Risk Scoring Methodology

**Overall Risk Score** (0-100 scale):

```
Risk Score = (
    Vital Signs Weight × Vital Signs Subscore +
    Medications Weight × Medications Subscore +
    Conditions Weight × Conditions Subscore +
    Care Gaps Weight × Care Gaps Subscore
)

Weights:
- Vital Signs: 30%
- Medications: 25%
- Conditions: 25%
- Care Gaps: 20%
```

**Risk Categories**:
- **LOW** (0-49): Routine care appropriate
- **MEDIUM** (50-74): Enhanced monitoring recommended
- **HIGH** (75-89): Intensive case management needed
- **CRITICAL** (90-100): Immediate intervention required

### Vital Signs Thresholds

| Vital Sign | Normal Range | Critical Low | Critical High |
|------------|--------------|--------------|---------------|
| Systolic BP | 90-120 mmHg | <70 | >180 |
| Diastolic BP | 60-80 mmHg | <40 | >120 |
| Heart Rate | 60-100 bpm | <40 | >140 |
| Temperature | 36.1-37.2°C | <35.0 | >39.5 |
| Resp Rate | 12-20 /min | <8 | >30 |
| O2 Saturation | 95-100% | <88 | N/A |

### Medication Risk Assessment

**High-Risk Medications**:
- Warfarin and anticoagulants
- Insulin and other diabetes medications
- Opioids
- Chemotherapy agents
- Immunosuppressants

**Polypharmacy**:
- **Definition**: 5+ concurrent medications
- **Risks**: Drug interactions, adherence issues, adverse events
- **Intervention**: Pharmacist medication review

## 🔌 Integration Options

### 1. Azure Event Hubs (Real-time Streaming)

```python
from gen_z_agent.healthcare_config import HealthcareConfig

# Configure Event Hubs connection
HealthcareConfig.EVENTHUB_CONNECTION_STRING = "Endpoint=sb://..."
HealthcareConfig.EVENTHUB_NAME = "fhir-events"

# Run streaming workflow
run_clinical_workflow(
    workflow_type="vitals_monitoring",
    data_source="eventhub",
    dry_run=False
)
```

### 2. Databricks Delta Lake (Batch Processing)

```python
# Configure Delta Lake
HealthcareConfig.DATABRICKS_HOST = "https://..."
HealthcareConfig.DATABRICKS_TOKEN = "..."
HealthcareConfig.DELTA_LAKE_PATH = "/ehr-pipeline/silver"

# Run batch analysis
run_clinical_workflow(
    workflow_type="care_gap_analysis",
    data_source="delta_lake",
    dry_run=False
)
```

### 3. FHIR API (Direct Integration)

```python
# Configure FHIR endpoint
HealthcareConfig.FHIR_BASE_URL = "https://fhir.example.com/api"

# Query and analyze
run_clinical_workflow(
    workflow_type="patient_risk_assessment",
    patient_id="PAT001",
    data_source="fhir_api",
    dry_run=False
)
```

### 4. Palantir Foundry (Advanced Analytics)

See `docs/palantir_foundry_ehr_integration.md` for comprehensive Foundry integration guide.

## 📝 Example Outputs

### Patient Risk Assessment Report

```markdown
# Patient Risk Assessment Report

**Patient ID**: PAT001
**Assessment Date**: 2025-11-21 10:30:00
**Risk Score**: 78/100
**Risk Category**: HIGH

## Clinical Summary

68-year-old male with Type 2 Diabetes, Hypertension, and Hyperlipidemia.
Recent vitals show uncontrolled blood pressure (165/95 mmHg). Currently on 3 medications.

## Risk Factors Identified

- ⚠️ **Uncontrolled Hypertension**: BP 165/95 (Goal: <140/90)
- ⚠️ **Diabetes**: On metformin, HbA1c monitoring due
- ⚠️ **Polypharmacy Risk**: 3 active medications
- ⚠️ **Age >65**: Increased fall risk, frequent monitoring needed

## Recent Vital Signs

| Vital Sign | Value | Normal Range | Status |
|------------|-------|--------------|--------|
| Systolic BP | 165 mmHg | 90-120 | ⚠️ HIGH |
| Diastolic BP | 95 mmHg | 60-80 | ⚠️ HIGH |
| Heart Rate | 88 bpm | 60-100 | ✓ Normal |
| O2 Saturation | 96% | 95-100 | ✓ Normal |

## Active Medications

1. **Metformin 500mg** - Twice daily (Diabetes)
2. **Lisinopril 10mg** - Once daily (Hypertension)
3. **Atorvastatin 20mg** - Once daily (Hyperlipidemia)

## Recommendations

### URGENT (Within 24-48 hours)
1. ❗ Reassess blood pressure control - Consider lisinopril dose increase
2. ❗ Check medication adherence - Patient education on BP medication importance
3. ❗ Order HbA1c test (last result >90 days ago)

### HIGH PRIORITY (Within 1 week)
4. Schedule follow-up appointment for BP recheck
5. Refer to diabetes educator for self-management education
6. Consider adding HCTZ for better BP control

### ROUTINE
7. Annual diabetic eye exam due
8. Update cardiovascular risk assessment

## Care Team Actions

**Primary Care Provider**:
- Review BP medication regimen
- Consider medication adjustment
- Order HbA1c lab test

**Nurse Care Manager**:
- Patient education: BP monitoring at home
- Medication adherence counseling
- Schedule 2-week follow-up call

**Pharmacist**:
- Medication therapy management review
- Check for any barriers to adherence
- Provide BP monitoring technique education

**Patient**:
- Monitor BP at home daily
- Bring BP log to next visit
- Continue all medications as prescribed

---
*This report contains Protected Health Information (PHI) and must be handled according to HIPAA regulations.*
*Generated by Gen Z Agent Healthcare System*
*Report ID: RISK_PAT001_20251121_103000*
```

## 🧪 Testing

### Unit Tests

```bash
# Run healthcare module tests
pytest tests/test_healthcare_agents.py
pytest tests/test_healthcare_security.py
pytest tests/test_healthcare_models.py

# With coverage
pytest --cov=gen_z_agent/healthcare tests/
```

### Integration Tests

```bash
# Test full workflow with sample data
python tests/integration/test_clinical_workflow.py
```

### Sample Data

Sample FHIR bundles are provided in `gen_z_agent/healthcare/fhir_data/`:
- `sample_patient_bundle.json` - Complete patient with vitals, meds, conditions

## 🔧 Configuration

### Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Optional - FHIR Integration
FHIR_BASE_URL=https://fhir.example.com/api
FHIR_VERSION=R4

# Optional - Azure Integration
EVENTHUB_CONNECTION_STRING=Endpoint=sb://...
DATABRICKS_HOST=https://...
DATABRICKS_TOKEN=...

# Optional - Security
PHI_ENCRYPTION_KEY=<base64-encoded-key>
ENABLE_DE_IDENTIFICATION=true

# Optional - Notifications
CRITICAL_ALERT_RECIPIENTS=oncall@example.com,ed@example.com
SMTP_HOST=smtp.gmail.com
SMTP_USERNAME=alerts@example.com
SMTP_PASSWORD=...
```

### Healthcare Configuration

Edit `gen_z_agent/healthcare_config.py` to customize:
- Vital sign thresholds
- Risk scoring weights
- Medication risk categories
- Care gap definitions
- Alert routing logic

## 📚 Additional Resources

- **FHIR Documentation**: https://hl7.org/fhir/
- **HIPAA Compliance**: https://www.hhs.gov/hipaa/
- **LOINC Codes**: https://loinc.org/
- **SNOMED CT**: https://www.snomed.org/
- **RxNorm**: https://www.nlm.nih.gov/research/umls/rxnorm/

## 🆘 Troubleshooting

### Issue: "ANTHROPIC_API_KEY is required"
**Solution**: Set API key in `.env` file or environment variable

### Issue: "PHI encryption key not found"
**Solution**: Generate key and set `PHI_ENCRYPTION_KEY` environment variable
```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

### Issue: "Access denied for user role"
**Solution**: Check user role permissions in `healthcare_security.py` AccessControl

### Issue: "FHIR validation failed"
**Solution**: Ensure FHIR resources have required fields (id, resourceType, status)

## 📞 Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/sechan9999/GenZ/issues
- Documentation: `docs/healthcare_automation_guide.md`
- CLAUDE.md: AI assistant guide

---

**Last Updated**: 2025-11-21
**Version**: 1.0.0
**Maintainer**: Healthcare Automation Team
