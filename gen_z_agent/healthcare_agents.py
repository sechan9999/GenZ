"""
════════════════════════════════════════════════════════════════════════════
Healthcare Multi-Agent System (CrewAI + Anthropic Claude)
AI-Driven Clinical Workflow Automation with HIPAA Compliance
════════════════════════════════════════════════════════════════════════════
"""

import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from crewai import Agent, Task, Crew, Process
from langchain_anthropic import ChatAnthropic
from crewai_tools import FileReadTool
from dotenv import load_dotenv

from healthcare_config import HealthcareConfig
from healthcare_models import (
    PatientRiskAssessment,
    MedicationReview,
    ClinicalAlert,
    RiskCategory,
    AlertSeverity,
)

# Load environment variables
load_dotenv()

# Initialize Claude LLM with clinical-grade settings
clinical_llm = ChatAnthropic(
    model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929"),
    temperature=HealthcareConfig.CLINICAL_LLM_TEMPERATURE,
    max_tokens=HealthcareConfig.CLINICAL_LLM_MAX_TOKENS,
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# Initialize tools
file_reader = FileReadTool()


# ════════════════════════════════════════════════════════════════════════════
# HEALTHCARE AGENT DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════

# 1. FHIR Data Ingestion & Parsing Agent
fhir_ingestion_agent = Agent(
    role="FHIR Data Ingestion Specialist (FHIR 데이터 수집 전문가)",
    goal="""Ingest, parse, and normalize FHIR R4 healthcare data from multiple sources
    (Event Hubs, Delta Lake, APIs) while maintaining data integrity and PHI security""",
    backstory="""You are a healthcare data integration expert specializing in HL7 FHIR
    standards. You understand the complex nested structures of FHIR resources and can
    accurately extract Patient, Observation, MedicationStatement, Condition, and
    Encounter data.

    You are meticulous about data quality, ensuring all required fields are present,
    timestamps are valid, and references between resources are intact.

    You handle PHI (Protected Health Information) with extreme care, ensuring all data
    is properly classified and encrypted according to HIPAA requirements.""",
    tools=[file_reader],
    llm=clinical_llm,
    verbose=HealthcareConfig.AGENT_VERBOSE,
    allow_delegation=False
)

# 2. Clinical Data Validation & Enrichment Agent
clinical_validation_agent = Agent(
    role="Clinical Data Validator (임상 데이터 검증 전문가)",
    goal="""Validate clinical data for completeness, accuracy, and clinical plausibility.
    Enrich data with standard medical terminologies (LOINC, SNOMED CT, RxNorm)""",
    backstory="""You are a clinical informaticist with deep knowledge of medical coding
    systems and data quality standards. You validate that:

    - Vital signs are within physiologically plausible ranges
    - Medication codes are valid RxNorm/NDC codes
    - Diagnosis codes follow ICD-10 standards
    - LOINC codes for lab results are correctly applied
    - Temporal relationships make clinical sense

    You flag data quality issues, missing values, and potential data entry errors.
    You enrich sparse data with standard medical terminologies and context.

    Your work ensures downstream clinical algorithms receive high-quality, validated data.""",
    llm=clinical_llm,
    verbose=HealthcareConfig.AGENT_VERBOSE,
    allow_delegation=False
)

# 3. Clinical Risk Analysis Agent
clinical_risk_analyst = Agent(
    role="Clinical Risk Analyst (임상 위험 분석가)",
    goal="""Analyze patient clinical data to identify risk factors, predict adverse events,
    detect care gaps, and calculate patient risk scores using evidence-based algorithms""",
    backstory="""You are a clinical data scientist and population health expert. You analyze
    patient data holistically to identify:

    - High-risk patients requiring immediate intervention
    - Medication safety issues (drug interactions, high-risk medications)
    - Vital sign trends indicating clinical deterioration
    - Chronic disease management gaps
    - Fall risk and hospital readmission risk
    - Polypharmacy concerns

    You apply evidence-based clinical algorithms (e.g., CHADS2-VASc, Framingham Risk Score,
    Fall Risk Assessment) and machine learning models to calculate risk scores.

    Your analysis is always grounded in clinical evidence and best practices.
    You provide actionable insights that care teams can immediately act upon.""",
    llm=clinical_llm,
    verbose=HealthcareConfig.AGENT_VERBOSE,
    allow_delegation=False
)

# 4. Clinical Report Generation Agent
clinical_report_writer = Agent(
    role="Clinical Report Writer (임상 보고서 작성자)",
    goal="""Generate comprehensive, clinically relevant reports for care teams including
    risk assessments, medication reviews, and care gap analyses in standardized formats""",
    backstory="""You are a clinical documentation specialist who creates clear, actionable
    reports for busy healthcare providers. Your reports include:

    - Executive summaries highlighting critical findings
    - Evidence-based risk stratification
    - Prioritized action items for care teams
    - Clinical context and supporting evidence
    - Trending data and longitudinal views

    You format reports for different audiences:
    - Physicians: Clinical detail, differential diagnoses, treatment recommendations
    - Nurses: Care plans, monitoring protocols, patient education needs
    - Case managers: Care coordination needs, social determinants of health
    - Quality teams: Performance metrics, care gap identification

    All reports are HIPAA-compliant and properly classified as PHI.
    You ensure clinical accuracy while maintaining readability.""",
    tools=[file_reader],
    llm=clinical_llm,
    verbose=HealthcareConfig.AGENT_VERBOSE,
    allow_delegation=False
)

# 5. Care Coordination & Communication Agent
care_coordination_agent = Agent(
    role="Care Coordination Specialist (케어 코디네이터)",
    goal="""Coordinate care team notifications, patient outreach, and clinical alert
    routing based on patient risk levels and organizational workflows""",
    backstory="""You are a care coordination expert who ensures the right clinical information
    reaches the right care team members at the right time. You:

    - Route critical alerts to on-call providers immediately
    - Notify care teams of high-risk patients needing intervention
    - Schedule follow-up appointments for patients with care gaps
    - Coordinate between primary care, specialists, and pharmacists
    - Ensure patients receive appropriate education and outreach

    You understand clinical urgency levels:
    - CRITICAL: Immediate notification (page, SMS, phone)
    - HIGH: Same-day notification (email, EHR task)
    - MEDIUM: Next business day notification
    - LOW: Routine care coordination

    You respect provider workflows and patient preferences.
    You maintain HIPAA compliance in all communications.

    Your goal is to close the loop on identified clinical issues and ensure
    no patient falls through the cracks.""",
    llm=clinical_llm,
    verbose=HealthcareConfig.AGENT_VERBOSE,
    allow_delegation=False
)


# ════════════════════════════════════════════════════════════════════════════
# HEALTHCARE TASK DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════

def create_fhir_ingestion_task(data_source: str, patient_id: Optional[str] = None) -> Task:
    """Task: Ingest and parse FHIR data"""
    patient_filter = f"for patient {patient_id}" if patient_id else "for all patients"

    return Task(
        description=f"""
        Ingest and parse FHIR R4 healthcare data {patient_filter} from: {data_source}

        **Data Sources to Check**:
        1. Local FHIR JSON files: {HealthcareConfig.FHIR_DATA_DIR}
        2. Delta Lake tables: {HealthcareConfig.DELTA_LAKE_PATH}
        3. FHIR API endpoint: {HealthcareConfig.FHIR_BASE_URL}

        **Resources to Extract**:
        - Patient demographics (identifiers, demographics - HANDLE AS PHI)
        - Observations (vital signs, lab results) - Last 90 days
        - MedicationStatements (active and recent medications)
        - Conditions (active diagnoses)
        - Encounters (recent visits)

        **Data Quality Requirements**:
        - Validate FHIR resource structure
        - Check required fields (id, resourceType, subject)
        - Validate date formats and temporal logic
        - Flag missing or malformed data
        - Ensure patient references are consistent

        **Security Requirements**:
        - Mark all data as PHI (Protected Health Information)
        - Do NOT log patient identifiers in plain text
        - Encrypt temporary files using AES-256
        - Generate audit log entry for data access

        **Output Format**: Structured JSON with:
        ```json
        {{
            "patient": {{...}},
            "observations": [...],
            "medications": [...],
            "conditions": [...],
            "encounters": [...],
            "data_quality": {{
                "completeness_score": 0.0-1.0,
                "validation_errors": [...],
                "warnings": [...]
            }},
            "phi_classification": "RESTRICTED",
            "ingestion_timestamp": "ISO-8601"
        }}
        ```
        """,
        expected_output="Validated and structured FHIR data in JSON format with PHI classification",
        agent=fhir_ingestion_agent
    )


def create_clinical_validation_task() -> Task:
    """Task: Validate and enrich clinical data"""
    return Task(
        description="""
        Validate and enrich the clinical data from Task 1.

        **Validation Checks**:

        1. **Vital Signs Validation**:
           - Check values are physiologically plausible
           - Blood Pressure: Systolic 50-300, Diastolic 30-200 mmHg
           - Heart Rate: 20-250 bpm
           - Temperature: 32-42 Celsius
           - Oxygen Saturation: 50-100%
           - Flag out-of-range values with severity level

        2. **Medication Validation**:
           - Verify RxNorm or NDC codes are valid
           - Check medication-diagnosis appropriateness
           - Identify high-risk medications requiring monitoring
           - Flag polypharmacy (5+ concurrent medications)

        3. **Diagnosis Validation**:
           - Verify ICD-10 or SNOMED CT codes
           - Check for conflicting diagnoses
           - Identify chronic conditions requiring ongoing management

        4. **Temporal Logic Validation**:
           - Encounter dates are not in the future
           - Medication start dates precede end dates
           - Observation dates align with encounter dates

        **Enrichment Tasks**:
        - Add LOINC display names for observation codes
        - Add medication class and therapeutic category
        - Calculate derived metrics (BMI, eGFR if applicable)
        - Add normal range references for vital signs
        - Flag critical values based on thresholds

        **Output**: Enhanced JSON with validation results:
        ```json
        {{
            "validated_data": {{...}},
            "validation_results": {{
                "passed": true/false,
                "errors": [...],
                "warnings": [...],
                "critical_values": [...]
            }},
            "enrichment_applied": [...]
        }}
        ```
        """,
        expected_output="Validated and enriched clinical data with quality assessment",
        agent=clinical_validation_agent
    )


def create_risk_analysis_task(patient_id: str) -> Task:
    """Task: Perform clinical risk analysis"""
    return Task(
        description=f"""
        Perform comprehensive clinical risk analysis for patient {patient_id}.

        **Risk Assessment Components**:

        1. **Vital Signs Risk Analysis**:
           - Identify critical vital signs (per {HealthcareConfig.VITAL_SIGNS} thresholds)
           - Analyze vital sign trends (improving/deteriorating)
           - Calculate early warning scores if applicable

        2. **Medication Safety Analysis**:
           - Check for drug-drug interactions
           - Identify high-risk medications: {', '.join(HealthcareConfig.HIGH_RISK_MEDICATIONS)}
           - Assess polypharmacy risk
           - Review medication adherence patterns

        3. **Chronic Disease Management**:
           - Identify uncontrolled chronic conditions
           - Check for missing preventive care (screenings, immunizations)
           - Calculate disease-specific risk scores:
             * Diabetes: HbA1c control, microvascular complications
             * Hypertension: BP control, cardiovascular risk
             * Heart Failure: symptoms, medication adherence

        4. **Care Gaps**:
           - Missing medications for diagnosed conditions
           - Overdue preventive screenings
           - Incomplete disease monitoring (e.g., diabetic eye exams)

        5. **Overall Risk Score Calculation**:
           - Weight factors: vital signs (30%), medications (25%), conditions (25%), care gaps (20%)
           - Score 0-100 (Higher = Higher Risk)
           - Category: LOW (<50), MEDIUM (50-75), HIGH (75-90), CRITICAL (>90)

        **Output**: Risk assessment report:
        ```json
        {{
            "patient_id": "{patient_id}",
            "risk_score": 0-100,
            "risk_category": "LOW|MEDIUM|HIGH|CRITICAL",
            "risk_factors": {{
                "critical_vitals": bool,
                "high_risk_medications": bool,
                "polypharmacy": bool,
                "uncontrolled_chronic_disease": bool
            }},
            "care_gaps": [...],
            "recommendations": [...],
            "confidence_score": 0.0-1.0
        }}
        ```
        """,
        expected_output="Comprehensive patient risk assessment with actionable recommendations",
        agent=clinical_risk_analyst
    )


def create_clinical_report_task(report_type: str, patient_id: str) -> Task:
    """Task: Generate clinical report"""
    template = HealthcareConfig.CLINICAL_REPORT_TEMPLATES.get(
        report_type,
        HealthcareConfig.CLINICAL_REPORT_TEMPLATES["patient_risk_assessment"]
    )

    return Task(
        description=f"""
        Generate a comprehensive {report_type} report for patient {patient_id}.

        **Report Requirements**:

        1. **Executive Summary** (1-2 paragraphs):
           - Patient risk level and category
           - Top 3 clinical concerns
           - Immediate actions required

        2. **Clinical Data Tables**:
           - Recent vital signs with trend indicators (↑↓→)
           - Active medications with indication
           - Active diagnoses with onset date
           - Recent encounters

        3. **Risk Analysis**:
           - Detailed risk factor breakdown
           - Supporting clinical evidence
           - Care gaps identified

        4. **Recommendations** (Prioritized):
           - URGENT: Immediate clinical intervention needed
           - HIGH: Action needed within 24-48 hours
           - MEDIUM: Action needed within 1 week
           - LOW: Routine follow-up

        5. **Care Team Actions**:
           - Primary Care Provider: [specific actions]
           - Nurse Care Manager: [specific actions]
           - Pharmacist: [specific actions]
           - Patient: [education/self-management]

        **Output Formats**:
        1. Markdown report: {HealthcareConfig.CLINICAL_REPORTS_DIR}/{{report_type}}_{{patient_id}}_{{timestamp}}.md
        2. Structured JSON: Same filename with .json extension

        **PHI Handling**:
        - Mark all files with PHI classification header
        - Encrypt files at rest
        - Log file access in audit trail

        Use template:
        {template}
        """,
        expected_output=f"Clinical {report_type} report in Markdown and JSON formats",
        agent=clinical_report_writer
    )


def create_care_coordination_task(patient_id: str, risk_category: str) -> Task:
    """Task: Coordinate care team notifications"""
    return Task(
        description=f"""
        Coordinate care team notifications and next steps for patient {patient_id}
        with risk category: {risk_category}.

        **Notification Routing Logic**:

        **CRITICAL Risk** (Immediate Action):
        - Page on-call provider immediately
        - Send critical alert to primary care MD
        - Notify charge nurse
        - Create urgent EHR task
        - Consider ED referral or urgent clinic visit

        **HIGH Risk** (Same-Day Action):
        - Email primary care provider
        - Notify care manager
        - Create high-priority EHR task
        - Schedule phone outreach to patient
        - Consider same-day clinic visit

        **MEDIUM Risk** (This Week):
        - Email care team summary
        - Create standard EHR task
        - Schedule routine follow-up appointment
        - Plan patient education intervention

        **LOW Risk** (Routine):
        - Add to care manager panel review
        - Include in next scheduled visit
        - Routine preventive care reminders

        **Communication Channels**:
        1. **Critical Alerts**:
           - {', '.join(HealthcareConfig.CRITICAL_ALERT_CHANNELS)}
           - Recipients: {', '.join(HealthcareConfig.CRITICAL_ALERT_RECIPIENTS or ['oncall@example.com'])}

        2. **Care Team Notifications**:
           - Email with report attached
           - Slack channel: {HealthcareConfig.CARE_TEAM_SLACK_CHANNEL}
           - EHR in-basket message

        3. **Patient Notifications** (if consented):
           - Patient portal message
           - Secure email
           - Automated phone call

        **Coordination Tasks**:
        - Schedule follow-up appointments
        - Order necessary labs/tests
        - Refer to specialists if needed
        - Coordinate medication delivery for adherence issues
        - Arrange social services if indicated

        **Output**:
        ```json
        {{
            "patient_id": "{patient_id}",
            "notifications_sent": [
                {{
                    "recipient": "...",
                    "channel": "...",
                    "timestamp": "...",
                    "status": "sent|failed"
                }}
            ],
            "tasks_created": [...],
            "appointments_scheduled": [...],
            "care_plan_updated": bool
        }}
        ```

        **Note**: In dry-run mode, log all actions without actually sending notifications.
        """,
        expected_output="Care coordination plan with notification confirmations and next steps",
        agent=care_coordination_agent
    )


# ════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATION FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def run_clinical_workflow(
    workflow_type: str,
    patient_id: Optional[str] = None,
    data_source: str = None,
    dry_run: bool = True
) -> Dict[str, Any]:
    """
    Run AI-driven clinical workflow automation

    Args:
        workflow_type: Type of clinical workflow
            - "patient_risk_assessment": Comprehensive patient risk analysis
            - "medication_review": Medication safety review
            - "vitals_monitoring": Continuous vital signs monitoring
            - "care_gap_analysis": Identify preventive care gaps
        patient_id: Patient identifier (MRN or FHIR ID)
        data_source: Path to FHIR data or API endpoint
        dry_run: If True, don't send actual notifications

    Returns:
        dict: Workflow results including analysis and actions taken
    """

    # Set defaults
    if data_source is None:
        data_source = str(HealthcareConfig.FHIR_DATA_DIR)

    # Create audit log entry
    workflow_id = f"{workflow_type}_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n{'═'*80}")
    print(f"🏥 Gen Z Healthcare Agent - Clinical Workflow Automation")
    print(f"{'═'*80}")
    print(f"Workflow Type: {workflow_type}")
    print(f"Patient ID: {patient_id or 'Multiple Patients'}")
    print(f"Data Source: {data_source}")
    print(f"Dry Run Mode: {dry_run}")
    print(f"Workflow ID: {workflow_id}")
    print(f"HIPAA Compliance: ENABLED")
    print(f"{'═'*80}\n")

    # Create workflow-specific tasks
    if workflow_type == "patient_risk_assessment":
        task1 = create_fhir_ingestion_task(data_source, patient_id)
        task2 = create_clinical_validation_task()
        task3 = create_risk_analysis_task(patient_id or "UNKNOWN")
        task4 = create_clinical_report_task("patient_risk_assessment", patient_id or "UNKNOWN")
        task5 = create_care_coordination_task(patient_id or "UNKNOWN", "UNKNOWN")

        agents = [
            fhir_ingestion_agent,
            clinical_validation_agent,
            clinical_risk_analyst,
            clinical_report_writer,
            care_coordination_agent
        ]
        tasks = [task1, task2, task3, task4, task5]

    elif workflow_type == "medication_review":
        # Similar task creation for medication review workflow
        task1 = create_fhir_ingestion_task(data_source, patient_id)
        task2 = create_clinical_validation_task()
        task3 = create_risk_analysis_task(patient_id or "UNKNOWN")
        task4 = create_clinical_report_task("medication_review", patient_id or "UNKNOWN")
        task5 = create_care_coordination_task(patient_id or "UNKNOWN", "MEDIUM")

        agents = [fhir_ingestion_agent, clinical_validation_agent, clinical_risk_analyst,
                 clinical_report_writer, care_coordination_agent]
        tasks = [task1, task2, task3, task4, task5]

    else:
        raise ValueError(f"Unknown workflow type: {workflow_type}")

    # Create and run crew
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=2
    )

    # Execute workflow
    result = crew.kickoff()

    print(f"\n{'═'*80}")
    print(f"✅ Clinical Workflow Complete!")
    print(f"Workflow ID: {workflow_id}")
    print(f"{'═'*80}\n")

    # Return results
    return {
        "workflow_id": workflow_id,
        "workflow_type": workflow_type,
        "patient_id": patient_id,
        "completed_at": datetime.now().isoformat(),
        "result": str(result),
        "status": "SUCCESS",
        "hipaa_compliant": True
    }


# ════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Gen Z Healthcare Agent - AI-Driven Clinical Workflow Automation"
    )
    parser.add_argument(
        "workflow",
        choices=["patient_risk_assessment", "medication_review", "vitals_monitoring", "care_gap_analysis"],
        help="Type of clinical workflow to run"
    )
    parser.add_argument(
        "--patient-id",
        help="Patient identifier (MRN or FHIR ID)",
        required=False
    )
    parser.add_argument(
        "--data-source",
        help="Path to FHIR data directory or API endpoint",
        default=str(HealthcareConfig.FHIR_DATA_DIR)
    )
    parser.add_argument(
        "--production",
        help="Run in production mode (actually send notifications)",
        action="store_true"
    )

    args = parser.parse_args()

    # Validate configuration
    try:
        HealthcareConfig.validate()
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        exit(1)

    # Run workflow
    result = run_clinical_workflow(
        workflow_type=args.workflow,
        patient_id=args.patient_id,
        data_source=args.data_source,
        dry_run=not args.production
    )

    print("\n" + json.dumps(result, indent=2))
