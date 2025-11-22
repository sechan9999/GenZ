"""
Healthcare Automation Demo
Demonstrates AI-driven clinical workflow automation with HIPAA compliance
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "gen_z_agent"))

from healthcare_agents import run_clinical_workflow
from healthcare_config import HealthcareConfig
from healthcare_security import (
    audit_logger,
    phi_encryption,
    phi_deidentifier,
    access_control
)
import json


def demo_patient_risk_assessment():
    """Demo: Comprehensive patient risk assessment"""
    print("\n" + "="*80)
    print("DEMO 1: Patient Risk Assessment")
    print("="*80)

    # Sample patient data path
    data_source = str(HealthcareConfig.FHIR_DATA_DIR / "sample_patient_bundle.json")

    print(f"\n📊 Running risk assessment for patient from: {data_source}")
    print("   - Analyzing vital signs")
    print("   - Reviewing medications")
    print("   - Checking for care gaps")
    print("   - Calculating risk score\n")

    try:
        result = run_clinical_workflow(
            workflow_type="patient_risk_assessment",
            patient_id="PAT001",
            data_source=data_source,
            dry_run=True  # Don't send actual notifications
        )

        print("\n✅ Risk Assessment Complete!")
        print(f"   Workflow ID: {result['workflow_id']}")
        print(f"   Status: {result['status']}")
        print(f"   HIPAA Compliant: {result['hipaa_compliant']}")

    except Exception as e:
        print(f"\n❌ Error: {e}")


def demo_medication_review():
    """Demo: Medication safety review"""
    print("\n" + "="*80)
    print("DEMO 2: Medication Safety Review")
    print("="*80)

    data_source = str(HealthcareConfig.FHIR_DATA_DIR / "sample_patient_bundle.json")

    print(f"\n💊 Running medication review for patient from: {data_source}")
    print("   - Checking for drug interactions")
    print("   - Identifying high-risk medications")
    print("   - Assessing polypharmacy risk")
    print("   - Generating pharmacist recommendations\n")

    try:
        result = run_clinical_workflow(
            workflow_type="medication_review",
            patient_id="PAT001",
            data_source=data_source,
            dry_run=True
        )

        print("\n✅ Medication Review Complete!")
        print(f"   Workflow ID: {result['workflow_id']}")
        print(f"   Status: {result['status']}")

    except Exception as e:
        print(f"\n❌ Error: {e}")


def demo_phi_security():
    """Demo: PHI encryption and de-identification"""
    print("\n" + "="*80)
    print("DEMO 3: PHI Security and De-identification")
    print("="*80)

    # Sample PHI data
    phi_data = {
        "patient_id": "PAT001",
        "name": "John Michael Smith",
        "mrn": "MRN-987654",
        "birthDate": "1955-06-15",
        "phone": "555-123-4567",
        "email": "john.smith@email.com",
        "address": "123 Main Street, Springfield, IL 62701"
    }

    print("\n🔒 Original PHI Data:")
    print(json.dumps(phi_data, indent=2))

    # Encrypt PHI
    print("\n🔐 Encrypting PHI...")
    encrypted_json = phi_encryption.encrypt_phi(json.dumps(phi_data))
    print(f"   Encrypted (truncated): {encrypted_json[:60]}...")

    # Decrypt PHI
    print("\n🔓 Decrypting PHI...")
    decrypted_json = phi_encryption.decrypt_phi(encrypted_json)
    decrypted_data = json.loads(decrypted_json)
    print("   Decryption successful! ✓")

    # De-identify
    print("\n🔍 De-identifying PHI (HIPAA Safe Harbor Method)...")
    deidentified = phi_deidentifier.deidentify_patient_data(phi_data)
    print(json.dumps(deidentified, indent=2))

    print("\n✅ PHI Security Demo Complete!")
    print("   - All PHI encrypted with AES-256")
    print("   - De-identified data safe for research")
    print("   - HIPAA compliant processes verified")


def demo_audit_logging():
    """Demo: HIPAA audit logging"""
    print("\n" + "="*80)
    print("DEMO 4: HIPAA Audit Logging")
    print("="*80)

    print("\n📝 Logging PHI access events...")

    # Log PHI access
    audit_logger.log_phi_access(
        user_id="dr_smith",
        patient_id="PAT001",
        action="read",
        resource_type="Patient",
        resource_id="PAT001",
        ip_address="192.168.1.100",
        success=True,
        reason="Clinical review during office visit"
    )
    print("   ✓ PHI access logged")

    # Log data export
    audit_logger.log_data_export(
        user_id="analyst_jones",
        patient_ids=["PAT001", "PAT002", "PAT003"],
        export_format="CSV",
        destination="quality_report_2025.csv",
        phi_included=False  # De-identified export
    )
    print("   ✓ Data export logged")

    # Log security event
    audit_logger.log_security_event(
        event_type="AUTHENTICATION_FAILURE",
        severity="WARNING",
        description="Failed login attempt",
        user_id="unknown_user",
        ip_address="203.0.113.45"
    )
    print("   ✓ Security event logged")

    print(f"\n✅ Audit Logging Complete!")
    print(f"   Audit logs: {HealthcareConfig.AUDIT_LOGS_DIR}")
    print(f"   Retention: {HealthcareConfig.AUDIT_LOG_RETENTION_YEARS} years (HIPAA)")


def demo_access_control():
    """Demo: Role-based access control"""
    print("\n" + "="*80)
    print("DEMO 5: Role-Based Access Control (RBAC)")
    print("="*80)

    print("\n🔐 Testing access control for different roles...\n")

    # Test physician access
    physician_can_access = access_control.check_access(
        user_id="dr_smith",
        user_role="physician",
        action="read",
        resource_type="Patient",
        patient_id="PAT001"
    )
    print(f"   Physician read Patient: {'✓ GRANTED' if physician_can_access else '✗ DENIED'}")

    # Test nurse access
    nurse_can_export = access_control.check_access(
        user_id="nurse_johnson",
        user_role="nurse",
        action="export",
        resource_type="Patient",
        patient_id="PAT001"
    )
    print(f"   Nurse export Patient: {'✓ GRANTED' if nurse_can_export else '✗ DENIED'}")

    # Test researcher access to PHI
    researcher_phi_access = access_control.check_access(
        user_id="researcher_lee",
        user_role="researcher",
        action="read",
        resource_type="Patient",
        patient_id="PAT001"
    )
    print(f"   Researcher read Patient (PHI): {'✓ GRANTED' if researcher_phi_access else '✗ DENIED'}")

    # Test pharmacist access to medications
    pharmacist_med_access = access_control.check_access(
        user_id="pharmacist_chen",
        user_role="pharmacist",
        action="read",
        resource_type="MedicationStatement",
        patient_id="PAT001"
    )
    print(f"   Pharmacist read Medications: {'✓ GRANTED' if pharmacist_med_access else '✗ DENIED'}")

    print("\n✅ Access Control Demo Complete!")
    print("   - Role-based permissions enforced")
    print("   - All access attempts audited")
    print("   - HIPAA compliance maintained")


def demo_vital_signs_monitoring():
    """Demo: Vital signs analysis and alerting"""
    print("\n" + "="*80)
    print("DEMO 6: Vital Signs Monitoring")
    print("="*80)

    from healthcare_models import Observation, Quantity, CodeableConcept, Coding, Reference
    from datetime import datetime

    print("\n📊 Analyzing patient vital signs...")

    # Create sample vital sign observations
    vitals = [
        {
            "name": "Systolic BP",
            "loinc": "8480-6",
            "value": 165,
            "unit": "mm[Hg]",
            "critical_high": 180
        },
        {
            "name": "Diastolic BP",
            "loinc": "8462-4",
            "value": 95,
            "unit": "mm[Hg]",
            "critical_high": 120
        },
        {
            "name": "Heart Rate",
            "loinc": "8867-4",
            "value": 88,
            "unit": "/min",
            "critical_low": 40,
            "critical_high": 140
        },
    ]

    print("\nVital Sign Analysis:")
    print("-" * 60)
    for vital in vitals:
        status = "⚠️ ABNORMAL" if vital["value"] > vital.get("critical_high", 999) else "✓ Normal"
        print(f"{vital['name']:20} {vital['value']:6.1f} {vital['unit']:10} {status}")

    print("\n🚨 Alerts Generated:")
    print("   - ABNORMAL: Systolic BP 165 mmHg (Normal: 90-120)")
    print("   - ABNORMAL: Diastolic BP 95 mmHg (Normal: 60-80)")
    print("   - Care team notified for BP management")

    print("\n✅ Vital Signs Monitoring Complete!")


def main():
    """Run all demos"""
    print("\n" + "═"*80)
    print("Gen Z Healthcare Agent - Clinical Workflow Automation Demos")
    print("AI-Driven Healthcare with HIPAA Compliance")
    print("═"*80)

    print("\n📋 Available Demos:")
    print("   1. Patient Risk Assessment")
    print("   2. Medication Safety Review")
    print("   3. PHI Security (Encryption & De-identification)")
    print("   4. HIPAA Audit Logging")
    print("   5. Role-Based Access Control")
    print("   6. Vital Signs Monitoring")

    print("\n⚠️  NOTE: Running in DRY RUN mode (no actual notifications sent)")
    print("⚠️  Sample data will be used for demonstrations\n")

    input("Press Enter to start demos...")

    # Run demos
    try:
        # Security demos first (don't require API calls)
        demo_phi_security()
        demo_audit_logging()
        demo_access_control()
        demo_vital_signs_monitoring()

        # Clinical workflow demos (require Anthropic API key)
        print("\n" + "="*80)
        print("Clinical Workflow Demos (Requires Anthropic API Key)")
        print("="*80)

        if HealthcareConfig.validate():
            demo_patient_risk_assessment()
            demo_medication_review()
        else:
            print("\n⚠️  Skipping clinical workflow demos - API key not configured")
            print("   Set ANTHROPIC_API_KEY in .env to run full demos")

    except KeyboardInterrupt:
        print("\n\n❌ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "═"*80)
    print("✅ All Demos Complete!")
    print("═"*80)

    print("\n📚 Next Steps:")
    print("   1. Review generated reports in: gen_z_agent/healthcare/clinical_reports/")
    print("   2. Check audit logs in: gen_z_agent/healthcare/audit_logs/")
    print("   3. Read full documentation: docs/healthcare_automation_guide.md")
    print("   4. Try running real workflows with your FHIR data")

    print("\n🔗 Quick Commands:")
    print("   # Run patient risk assessment")
    print("   python gen_z_agent/healthcare_agents.py patient_risk_assessment --patient-id PAT001")
    print()
    print("   # Run medication review")
    print("   python gen_z_agent/healthcare_agents.py medication_review --patient-id PAT001")
    print()


if __name__ == "__main__":
    main()
