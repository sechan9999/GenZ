# Healthcare Workflow Automation Module

## 📋 Overview

This module provides **AI-driven healthcare workflow automation** using LLM-based agents with HIPAA compliance and clinical decision support capabilities.

## 🏥 Key Features

- **Multi-Agent Clinical Workflows**: 5 specialized AI agents for healthcare automation
- **FHIR R4 Support**: Full support for HL7 FHIR resources (Patient, Observation, Medication, etc.)
- **HIPAA Compliant**: PHI encryption, audit logging, de-identification, access control
- **Clinical Intelligence**: Evidence-based risk scoring, vital signs monitoring, medication safety
- **Integration Ready**: Azure Event Hubs, Databricks Delta Lake, Palantir Foundry

## 🤖 The 5 Healthcare Agents

| Agent | Purpose | Key Capabilities |
|-------|---------|------------------|
| **FHIR Ingestion** | Data integration | Parse FHIR resources, validate structure, extract clinical data |
| **Clinical Validation** | Data quality | Validate ranges, check terminologies (LOINC, SNOMED, RxNorm) |
| **Risk Analysis** | Patient risk assessment | Calculate risk scores, identify care gaps, detect patterns |
| **Clinical Report** | Documentation | Generate clinical reports for care teams |
| **Care Coordination** | Workflow management | Route alerts, notify teams, coordinate interventions |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd gen_z_agent
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add:
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 3. Run Demo

```bash
# Run interactive demo
python ../examples/healthcare_demo.py

# Or run specific workflow
python healthcare_agents.py patient_risk_assessment --patient-id PAT001
```

## 📊 Supported Workflows

### 1. Patient Risk Assessment
Comprehensive evaluation of patient clinical risk including:
- Vital signs analysis
- Medication review
- Chronic disease management
- Care gap identification
- Risk scoring (0-100 scale)

```bash
python healthcare_agents.py patient_risk_assessment --patient-id PAT001
```

### 2. Medication Review
Medication safety and appropriateness review:
- Drug-drug interaction checking
- High-risk medication identification
- Polypharmacy assessment
- Duplicate therapy detection

```bash
python healthcare_agents.py medication_review --patient-id PAT001
```

### 3. Vital Signs Monitoring
Real-time vital signs monitoring with intelligent alerting:
- Blood pressure, heart rate, temperature, O2 saturation
- Critical threshold detection
- Trend analysis
- Automated care team alerts

```bash
python healthcare_agents.py vitals_monitoring --patient-id PAT001
```

### 4. Care Gap Analysis
Identify missing preventive care and screenings:
- Preventive screening status
- Immunization compliance
- Disease-specific monitoring
- Quality measure tracking

```bash
python healthcare_agents.py care_gap_analysis --patient-id PAT001
```

## 🔒 HIPAA Compliance

### PHI Protection

All PHI is protected with:
- **AES-256 encryption** at rest and in transit
- **Audit logging** with 7-year retention
- **Access control** (role-based permissions)
- **De-identification** (HIPAA Safe Harbor method)

### Security Features

```python
from healthcare_security import phi_encryption, phi_deidentifier

# Encrypt PHI
encrypted = phi_encryption.encrypt_phi("Patient: John Doe")

# De-identify for research
deidentified = phi_deidentifier.deidentify_patient_data(patient_fhir)
```

### Audit Trail

All PHI access is automatically logged:
```python
from healthcare_security import audit_logger

audit_logger.log_phi_access(
    user_id="dr_smith",
    patient_id="PAT001",
    action="read",
    resource_type="Patient"
)
```

## 📂 Module Structure

```
healthcare/
├── README.md                    # This file
├── fhir_data/                   # FHIR resource storage
│   └── sample_patient_bundle.json
├── clinical_reports/            # Generated clinical reports
├── audit_logs/                  # HIPAA audit logs (7-year retention)
├── de_identified/               # De-identified data exports
└── phi_temp/                    # Encrypted temporary storage
```

## 🔧 Configuration

### Healthcare Config

Edit `healthcare_config.py` to customize:

```python
# Vital sign thresholds
VITAL_SIGNS = {
    "blood_pressure_systolic": {
        "critical_low": 70,
        "critical_high": 180,
        ...
    }
}

# Risk scoring weights
RISK_SCORE_HIGH_THRESHOLD = 75.0

# Medication risk categories
HIGH_RISK_MEDICATIONS = ["warfarin", "insulin", ...]
```

### Integration Settings

```python
# Azure Event Hubs
EVENTHUB_CONNECTION_STRING = "..."
EVENTHUB_NAME = "fhir-events"

# Databricks Delta Lake
DATABRICKS_HOST = "https://..."
DELTA_LAKE_PATH = "/ehr-pipeline/silver"

# FHIR API
FHIR_BASE_URL = "https://fhir.example.com/api"
```

## 📈 Clinical Decision Support

### Risk Scoring

Patient risk calculated from:
- **Vital Signs** (30%): Critical values, trends
- **Medications** (25%): High-risk meds, interactions
- **Conditions** (25%): Chronic disease control
- **Care Gaps** (20%): Missing preventive care

**Risk Categories**:
- LOW (0-49): Routine care
- MEDIUM (50-74): Enhanced monitoring
- HIGH (75-89): Case management
- CRITICAL (90-100): Immediate intervention

### Vital Signs Thresholds

| Vital Sign | Normal | Critical Low | Critical High |
|------------|--------|--------------|---------------|
| Systolic BP | 90-120 | <70 | >180 |
| Heart Rate | 60-100 | <40 | >140 |
| O2 Sat | 95-100% | <88 | N/A |

## 🧪 Testing

```bash
# Unit tests
pytest tests/test_healthcare_*.py

# Integration tests
python tests/integration/test_clinical_workflow.py

# Run with coverage
pytest --cov=healthcare tests/
```

## 📚 Documentation

- **Full Guide**: `../../docs/healthcare_automation_guide.md`
- **Foundry Integration**: `../../docs/palantir_foundry_ehr_integration.md`
- **FHIR Resources**: https://hl7.org/fhir/
- **HIPAA Compliance**: https://www.hhs.gov/hipaa/

## 🔌 Integration Examples

### Azure Event Hubs Streaming

```python
from healthcare_agents import run_clinical_workflow

result = run_clinical_workflow(
    workflow_type="vitals_monitoring",
    data_source="eventhub",
    patient_id=None,  # Stream all patients
    dry_run=False
)
```

### Databricks Delta Lake Batch

```python
result = run_clinical_workflow(
    workflow_type="care_gap_analysis",
    data_source="delta_lake",
    dry_run=False
)
```

### FHIR API Direct

```python
result = run_clinical_workflow(
    workflow_type="patient_risk_assessment",
    patient_id="PAT001",
    data_source="fhir_api",
    dry_run=False
)
```

## 🆘 Troubleshooting

### Common Issues

**Issue**: `ANTHROPIC_API_KEY is required`
```bash
# Solution: Set API key in .env
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
```

**Issue**: `PHI encryption key not found`
```bash
# Solution: Generate and set encryption key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# Add to .env: PHI_ENCRYPTION_KEY=<generated-key>
```

**Issue**: `FHIR validation failed`
- Ensure FHIR resources have required fields: `id`, `resourceType`, `status`
- Check date formats are ISO-8601
- Verify references are valid (e.g., `Patient/PAT001`)

## 📞 Support

- **GitHub Issues**: https://github.com/sechan9999/GenZ/issues
- **Documentation**: `../../docs/healthcare_automation_guide.md`
- **AI Assistant Guide**: `../../CLAUDE.md`

## ⚠️ Important Notes

### Production Deployment

Before deploying to production:
1. ✅ Configure proper encryption key management (not env variable)
2. ✅ Enable multi-factor authentication (MFA)
3. ✅ Set up monitoring and alerting
4. ✅ Configure actual SMTP/Slack for notifications
5. ✅ Review and customize access control roles
6. ✅ Perform security audit
7. ✅ Sign Business Associate Agreement (BAA) for HIPAA
8. ✅ Document incident response procedures

### Compliance

This module implements HIPAA technical safeguards, but full HIPAA compliance requires:
- Administrative safeguards (policies, training)
- Physical safeguards (facility security)
- Business Associate Agreements (BAAs)
- Risk assessments and security audits

Consult with your compliance team before processing real PHI.

---

**Version**: 1.0.0
**Last Updated**: 2025-11-21
**License**: MIT (Note: Separate licensing may be required for production healthcare use)
