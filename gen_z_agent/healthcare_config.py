"""
Healthcare-Specific Configuration for Gen Z Agent
HIPAA-compliant settings and clinical workflow configuration
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from datetime import timedelta

load_dotenv()


class HealthcareConfig:
    """Healthcare and HIPAA-compliant configuration"""

    # ═════════════════════════════════════════════════════════════
    # FHIR and Data Integration
    # ═════════════════════════════════════════════════════════════
    FHIR_BASE_URL: str = os.getenv("FHIR_BASE_URL", "https://fhir.example.com/api")
    FHIR_VERSION: str = os.getenv("FHIR_VERSION", "R4")

    # Azure Event Hubs
    EVENTHUB_CONNECTION_STRING: Optional[str] = os.getenv("EVENTHUB_CONNECTION_STRING")
    EVENTHUB_NAMESPACE: str = os.getenv("EVENTHUB_NAMESPACE", "")
    EVENTHUB_NAME: str = os.getenv("EVENTHUB_NAME", "fhir-events")

    # Databricks / Delta Lake
    DATABRICKS_HOST: Optional[str] = os.getenv("DATABRICKS_HOST")
    DATABRICKS_TOKEN: Optional[str] = os.getenv("DATABRICKS_TOKEN")
    DELTA_LAKE_PATH: str = os.getenv("DELTA_LAKE_PATH", "/ehr-pipeline")

    # Palantir Foundry
    FOUNDRY_STACK_URL: Optional[str] = os.getenv("FOUNDRY_STACK_URL")
    FOUNDRY_TOKEN: Optional[str] = os.getenv("FOUNDRY_TOKEN")

    # ═════════════════════════════════════════════════════════════
    # HIPAA Compliance Settings
    # ═════════════════════════════════════════════════════════════
    PHI_CLASSIFICATION_LEVEL: str = "RESTRICTED"  # PHI data classification
    ENCRYPTION_ALGORITHM: str = "AES-256-GCM"
    AUDIT_LOG_RETENTION_YEARS: int = 7  # HIPAA requirement

    # De-identification settings
    ENABLE_DE_IDENTIFICATION: bool = os.getenv("ENABLE_DE_IDENTIFICATION", "false").lower() == "true"
    DE_IDENTIFICATION_METHOD: str = os.getenv("DE_IDENTIFICATION_METHOD", "SAFE_HARBOR")

    # Access control
    REQUIRE_MFA: bool = True
    SESSION_TIMEOUT_MINUTES: int = int(os.getenv("SESSION_TIMEOUT_MINUTES", "15"))
    MAX_FAILED_AUTH_ATTEMPTS: int = 3

    # ═════════════════════════════════════════════════════════════
    # Healthcare Directory Structure
    # ═════════════════════════════════════════════════════════════
    BASE_DIR: Path = Path(__file__).parent
    HEALTHCARE_DIR: Path = BASE_DIR / "healthcare"
    FHIR_DATA_DIR: Path = HEALTHCARE_DIR / "fhir_data"
    CLINICAL_REPORTS_DIR: Path = HEALTHCARE_DIR / "clinical_reports"
    AUDIT_LOGS_DIR: Path = HEALTHCARE_DIR / "audit_logs"
    DE_IDENTIFIED_DIR: Path = HEALTHCARE_DIR / "de_identified"
    PHI_TEMP_DIR: Path = HEALTHCARE_DIR / "phi_temp"  # Encrypted temporary storage

    # ═════════════════════════════════════════════════════════════
    # FHIR Resource Types
    # ═════════════════════════════════════════════════════════════
    SUPPORTED_FHIR_RESOURCES: List[str] = [
        "Patient",
        "Observation",
        "MedicationStatement",
        "MedicationRequest",
        "Condition",
        "Encounter",
        "Procedure",
        "DiagnosticReport",
        "AllergyIntolerance",
        "Immunization",
    ]

    # ═════════════════════════════════════════════════════════════
    # Clinical Coding Systems
    # ═════════════════════════════════════════════════════════════
    CODING_SYSTEMS = {
        "LOINC": "http://loinc.org",
        "SNOMED_CT": "http://snomed.info/sct",
        "ICD10": "http://hl7.org/fhir/sid/icd-10",
        "RXNORM": "http://www.nlm.nih.gov/research/umls/rxnorm",
        "CPT": "http://www.ama-assn.org/go/cpt",
        "NDC": "http://hl7.org/fhir/sid/ndc",
        "UCUM": "http://unitsofmeasure.org",
    }

    # ═════════════════════════════════════════════════════════════
    # Vital Signs Configuration (LOINC Codes)
    # ═════════════════════════════════════════════════════════════
    VITAL_SIGNS = {
        "blood_pressure_systolic": {
            "loinc": "8480-6",
            "display": "Systolic Blood Pressure",
            "unit": "mm[Hg]",
            "normal_range": (90, 120),
            "critical_low": 70,
            "critical_high": 180,
        },
        "blood_pressure_diastolic": {
            "loinc": "8462-4",
            "display": "Diastolic Blood Pressure",
            "unit": "mm[Hg]",
            "normal_range": (60, 80),
            "critical_low": 40,
            "critical_high": 120,
        },
        "heart_rate": {
            "loinc": "8867-4",
            "display": "Heart Rate",
            "unit": "/min",
            "normal_range": (60, 100),
            "critical_low": 40,
            "critical_high": 140,
        },
        "body_temperature": {
            "loinc": "8310-5",
            "display": "Body Temperature",
            "unit": "Cel",
            "normal_range": (36.1, 37.2),
            "critical_low": 35.0,
            "critical_high": 39.5,
        },
        "respiratory_rate": {
            "loinc": "9279-1",
            "display": "Respiratory Rate",
            "unit": "/min",
            "normal_range": (12, 20),
            "critical_low": 8,
            "critical_high": 30,
        },
        "oxygen_saturation": {
            "loinc": "2708-6",
            "display": "Oxygen Saturation",
            "unit": "%",
            "normal_range": (95, 100),
            "critical_low": 88,
            "critical_high": 100,
        },
    }

    # ═════════════════════════════════════════════════════════════
    # Clinical Risk Thresholds
    # ═════════════════════════════════════════════════════════════
    RISK_THRESHOLDS = {
        "hypertension": {
            "systolic": 140,
            "diastolic": 90,
        },
        "hypotension": {
            "systolic": 90,
            "diastolic": 60,
        },
        "tachycardia": 100,
        "bradycardia": 60,
        "fever": 38.0,
        "hypothermia": 35.0,
        "hypoxia": 92,
    }

    # ═════════════════════════════════════════════════════════════
    # Medication Risk Categories
    # ═════════════════════════════════════════════════════════════
    HIGH_RISK_MEDICATIONS: List[str] = [
        "warfarin",
        "insulin",
        "opioid",
        "chemotherapy",
        "anticoagulant",
        "immunosuppressant",
    ]

    # Drug interaction checking
    ENABLE_DRUG_INTERACTION_CHECK: bool = True
    DRUG_INTERACTION_API_URL: Optional[str] = os.getenv("DRUG_INTERACTION_API_URL")

    # ═════════════════════════════════════════════════════════════
    # Clinical Workflow Settings
    # ═════════════════════════════════════════════════════════════

    # Patient risk assessment
    RISK_ASSESSMENT_LOOKBACK_DAYS: int = 90
    RISK_SCORE_HIGH_THRESHOLD: float = 75.0
    RISK_SCORE_MEDIUM_THRESHOLD: float = 50.0

    # Medication review
    MEDICATION_REVIEW_LOOKBACK_DAYS: int = 180
    POLYPHARMACY_THRESHOLD: int = 5  # 5+ concurrent medications

    # Vitals monitoring
    VITALS_MONITORING_INTERVAL_HOURS: int = 4
    VITALS_ALERT_ENABLED: bool = True

    # Care gap identification
    CARE_GAP_LOOKBACK_DAYS: int = 365
    SCREENING_DUE_WINDOW_DAYS: int = 30

    # ═════════════════════════════════════════════════════════════
    # Agent Behavior Configuration
    # ═════════════════════════════════════════════════════════════

    # LLM settings for clinical agents
    CLINICAL_LLM_TEMPERATURE: float = 0.1  # Low temperature for clinical accuracy
    CLINICAL_LLM_MAX_TOKENS: int = 8192

    # Agent verbosity
    AGENT_VERBOSE: bool = os.getenv("AGENT_VERBOSE", "true").lower() == "true"

    # Evidence requirements
    REQUIRE_CLINICAL_EVIDENCE: bool = True
    MIN_CONFIDENCE_SCORE: float = 0.85

    # ═════════════════════════════════════════════════════════════
    # Notification Settings
    # ═════════════════════════════════════════════════════════════

    # Critical alerts
    CRITICAL_ALERT_CHANNELS: List[str] = ["email", "sms", "pager"]
    CRITICAL_ALERT_RECIPIENTS: List[str] = os.getenv(
        "CRITICAL_ALERT_RECIPIENTS", ""
    ).split(",")

    # Care team notifications
    CARE_TEAM_EMAIL_ENABLED: bool = True
    CARE_TEAM_SLACK_CHANNEL: str = "#clinical-alerts"

    # Patient notifications
    PATIENT_NOTIFICATION_ENABLED: bool = False  # Requires patient consent

    # ═════════════════════════════════════════════════════════════
    # Report Templates
    # ═════════════════════════════════════════════════════════════
    CLINICAL_REPORT_TEMPLATES = {
        "patient_risk_assessment": """
# Patient Risk Assessment Report

**Patient ID**: {patient_id}
**Assessment Date**: {assessment_date}
**Risk Score**: {risk_score}/100
**Risk Category**: {risk_category}

## Clinical Summary

{clinical_summary}

## Risk Factors Identified

{risk_factors}

## Recent Vital Signs

{vital_signs_table}

## Active Medications

{medications_table}

## Recent Diagnoses

{diagnoses_table}

## Recommendations

{recommendations}

## Care Team Actions

{care_team_actions}

---
*This report contains Protected Health Information (PHI) and must be handled according to HIPAA regulations.*
*Generated by Gen Z Agent Healthcare System*
*Report ID: {report_id}*
        """,

        "medication_review": """
# Medication Review Report

**Patient ID**: {patient_id}
**Review Date**: {review_date}
**Total Active Medications**: {medication_count}
**Polypharmacy Risk**: {polypharmacy_risk}

## Medication List

{medications_table}

## Drug Interactions Detected

{interactions}

## High-Risk Medications

{high_risk_medications}

## Recommendations

{recommendations}

## Pharmacist Review Required

{pharmacist_review_items}

---
*PHI - HIPAA Protected*
*Report ID: {report_id}*
        """,

        "vitals_monitoring_alert": """
# Vital Signs Monitoring Alert

**Patient ID**: {patient_id}
**Alert Time**: {alert_time}
**Severity**: {severity}

## Critical Vital Signs

{critical_vitals}

## Vital Signs Trend

{vitals_trend}

## Recommended Actions

{recommended_actions}

## Care Team Notified

{care_team_notified}

---
*URGENT - Requires immediate clinical review*
*Alert ID: {alert_id}*
        """,
    }

    # ═════════════════════════════════════════════════════════════
    # Utility Methods
    # ═════════════════════════════════════════════════════════════

    @classmethod
    def ensure_directories(cls):
        """Create healthcare directory structure with proper permissions"""
        for dir_path in [
            cls.HEALTHCARE_DIR,
            cls.FHIR_DATA_DIR,
            cls.CLINICAL_REPORTS_DIR,
            cls.AUDIT_LOGS_DIR,
            cls.DE_IDENTIFIED_DIR,
            cls.PHI_TEMP_DIR,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True, mode=0o700)  # Restricted permissions

    @classmethod
    def validate(cls) -> bool:
        """Validate healthcare-specific configuration"""
        errors = []

        # Check required settings for production
        if os.getenv("PRODUCTION", "false").lower() == "true":
            if not cls.EVENTHUB_CONNECTION_STRING:
                errors.append("EVENTHUB_CONNECTION_STRING required in production")
            if not cls.CRITICAL_ALERT_RECIPIENTS:
                errors.append("CRITICAL_ALERT_RECIPIENTS required in production")

        if errors:
            raise ValueError(f"Healthcare configuration errors: {', '.join(errors)}")

        return True

    @classmethod
    def get_vital_sign_config(cls, vital_name: str) -> Optional[Dict]:
        """Get configuration for a specific vital sign"""
        return cls.VITAL_SIGNS.get(vital_name)

    @classmethod
    def is_critical_vital(cls, vital_name: str, value: float) -> bool:
        """Check if a vital sign value is in critical range"""
        config = cls.get_vital_sign_config(vital_name)
        if not config:
            return False

        return value < config["critical_low"] or value > config["critical_high"]

    @classmethod
    def calculate_risk_category(cls, risk_score: float) -> str:
        """Calculate risk category from numeric score"""
        if risk_score >= cls.RISK_SCORE_HIGH_THRESHOLD:
            return "HIGH"
        elif risk_score >= cls.RISK_SCORE_MEDIUM_THRESHOLD:
            return "MEDIUM"
        else:
            return "LOW"

    @classmethod
    def info(cls) -> Dict:
        """Return healthcare configuration summary"""
        return {
            "fhir_version": cls.FHIR_VERSION,
            "phi_classification": cls.PHI_CLASSIFICATION_LEVEL,
            "encryption": cls.ENCRYPTION_ALGORITHM,
            "audit_retention_years": cls.AUDIT_LOG_RETENTION_YEARS,
            "supported_resources": len(cls.SUPPORTED_FHIR_RESOURCES),
            "de_identification_enabled": cls.ENABLE_DE_IDENTIFICATION,
            "drug_interaction_check_enabled": cls.ENABLE_DRUG_INTERACTION_CHECK,
            "vitals_alert_enabled": cls.VITALS_ALERT_ENABLED,
            "session_timeout_minutes": cls.SESSION_TIMEOUT_MINUTES,
        }


# Initialize healthcare directories on import
HealthcareConfig.ensure_directories()


if __name__ == "__main__":
    import json

    print("Gen Z Agent - Healthcare Configuration")
    print("=" * 60)
    print(json.dumps(HealthcareConfig.info(), indent=2))
    print("\nSupported FHIR Resources:")
    for resource in HealthcareConfig.SUPPORTED_FHIR_RESOURCES:
        print(f"  - {resource}")
    print("\nVital Signs Monitoring:")
    for vital_name, config in HealthcareConfig.VITAL_SIGNS.items():
        print(f"  - {config['display']} ({config['loinc']})")
    print("\nDirectory Structure:")
    for dir_name in ["HEALTHCARE_DIR", "FHIR_DATA_DIR", "CLINICAL_REPORTS_DIR", "AUDIT_LOGS_DIR"]:
        print(f"  {dir_name}: {getattr(HealthcareConfig, dir_name)}")
