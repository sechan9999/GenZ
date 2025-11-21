"""
FHIR Data Models for Healthcare Workflows
Pydantic models for type-safe clinical data processing
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ═════════════════════════════════════════════════════════════
# Enumerations
# ═════════════════════════════════════════════════════════════

class ObservationStatus(str, Enum):
    REGISTERED = "registered"
    PRELIMINARY = "preliminary"
    FINAL = "final"
    AMENDED = "amended"
    CORRECTED = "corrected"
    CANCELLED = "cancelled"
    ENTERED_IN_ERROR = "entered-in-error"


class MedicationStatementStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ENTERED_IN_ERROR = "entered-in-error"
    INTENDED = "intended"
    STOPPED = "stopped"
    ON_HOLD = "on-hold"
    UNKNOWN = "unknown"
    NOT_TAKEN = "not-taken"


class RiskCategory(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ═════════════════════════════════════════════════════════════
# Base FHIR Models
# ═════════════════════════════════════════════════════════════

class Coding(BaseModel):
    """FHIR Coding datatype"""
    system: Optional[str] = None
    code: Optional[str] = None
    display: Optional[str] = None
    version: Optional[str] = None


class CodeableConcept(BaseModel):
    """FHIR CodeableConcept datatype"""
    coding: List[Coding] = Field(default_factory=list)
    text: Optional[str] = None

    def get_primary_code(self) -> Optional[str]:
        """Get the primary code from coding list"""
        return self.coding[0].code if self.coding else None

    def get_primary_display(self) -> Optional[str]:
        """Get the primary display from coding list"""
        return self.coding[0].display if self.coding else self.text


class Quantity(BaseModel):
    """FHIR Quantity datatype"""
    value: Optional[float] = None
    unit: Optional[str] = None
    system: Optional[str] = None
    code: Optional[str] = None


class Reference(BaseModel):
    """FHIR Reference datatype"""
    reference: Optional[str] = None
    display: Optional[str] = None

    def extract_id(self) -> Optional[str]:
        """Extract resource ID from reference (e.g., 'Patient/123' -> '123')"""
        if self.reference and "/" in self.reference:
            return self.reference.split("/")[-1]
        return None


class Period(BaseModel):
    """FHIR Period datatype"""
    start: Optional[datetime] = None
    end: Optional[datetime] = None


# ═════════════════════════════════════════════════════════════
# FHIR Resource Models
# ═════════════════════════════════════════════════════════════

class Patient(BaseModel):
    """Simplified FHIR Patient resource"""
    id: str
    resourceType: str = "Patient"
    identifier: List[Dict[str, Any]] = Field(default_factory=list)
    name: List[Dict[str, Any]] = Field(default_factory=list)
    gender: Optional[str] = None
    birthDate: Optional[str] = None
    address: List[Dict[str, Any]] = Field(default_factory=list)
    telecom: List[Dict[str, Any]] = Field(default_factory=list)

    # PHI flag
    contains_phi: bool = True

    def get_mrn(self) -> Optional[str]:
        """Extract Medical Record Number"""
        for ident in self.identifier:
            if ident.get("type", {}).get("text") == "MRN":
                return ident.get("value")
        return None

    def get_full_name(self) -> str:
        """Get patient's full name"""
        if self.name:
            name = self.name[0]
            given = " ".join(name.get("given", []))
            family = name.get("family", "")
            return f"{given} {family}".strip()
        return "Unknown"


class Observation(BaseModel):
    """FHIR Observation resource"""
    id: str
    resourceType: str = "Observation"
    status: ObservationStatus
    code: CodeableConcept
    subject: Reference
    encounter: Optional[Reference] = None
    effectiveDateTime: Optional[datetime] = None
    effectivePeriod: Optional[Period] = None
    issued: Optional[datetime] = None
    valueQuantity: Optional[Quantity] = None
    valueString: Optional[str] = None
    valueCodeableConcept: Optional[CodeableConcept] = None
    interpretation: Optional[List[CodeableConcept]] = None
    note: Optional[List[Dict[str, Any]]] = None

    # Metadata
    meta: Optional[Dict[str, Any]] = None
    contains_phi: bool = True

    def get_patient_id(self) -> Optional[str]:
        """Extract patient ID from subject reference"""
        return self.subject.extract_id() if self.subject else None

    def get_observation_code(self) -> Optional[str]:
        """Get primary observation code (e.g., LOINC)"""
        return self.code.get_primary_code()

    def get_observation_display(self) -> Optional[str]:
        """Get human-readable observation name"""
        return self.code.get_primary_display()

    def get_numeric_value(self) -> Optional[float]:
        """Get numeric value if available"""
        return self.valueQuantity.value if self.valueQuantity else None

    def get_value_unit(self) -> Optional[str]:
        """Get unit of measurement"""
        return self.valueQuantity.unit if self.valueQuantity else None

    def is_critical(self, vital_config: Dict) -> bool:
        """Check if observation value is in critical range"""
        if not self.valueQuantity or not vital_config:
            return False

        value = self.valueQuantity.value
        if value is None:
            return False

        critical_low = vital_config.get("critical_low")
        critical_high = vital_config.get("critical_high")

        return value < critical_low or value > critical_high


class MedicationStatement(BaseModel):
    """FHIR MedicationStatement resource"""
    id: str
    resourceType: str = "MedicationStatement"
    status: MedicationStatementStatus
    medicationCodeableConcept: Optional[CodeableConcept] = None
    subject: Reference
    effectivePeriod: Optional[Period] = None
    dateAsserted: Optional[datetime] = None
    informationSource: Optional[Reference] = None
    derivedFrom: Optional[List[Reference]] = None
    reasonCode: Optional[List[CodeableConcept]] = None
    note: Optional[List[Dict[str, Any]]] = None
    dosage: Optional[List[Dict[str, Any]]] = None

    # Metadata
    meta: Optional[Dict[str, Any]] = None
    contains_phi: bool = True

    def get_patient_id(self) -> Optional[str]:
        """Extract patient ID from subject reference"""
        return self.subject.extract_id() if self.subject else None

    def get_medication_code(self) -> Optional[str]:
        """Get medication code (RxNorm)"""
        if self.medicationCodeableConcept:
            return self.medicationCodeableConcept.get_primary_code()
        return None

    def get_medication_name(self) -> Optional[str]:
        """Get medication name"""
        if self.medicationCodeableConcept:
            return self.medicationCodeableConcept.get_primary_display()
        return None

    def is_active(self) -> bool:
        """Check if medication is currently active"""
        return self.status == MedicationStatementStatus.ACTIVE

    def get_dosage_text(self) -> Optional[str]:
        """Get dosage instructions"""
        if self.dosage and len(self.dosage) > 0:
            return self.dosage[0].get("text")
        return None


class Condition(BaseModel):
    """FHIR Condition resource"""
    id: str
    resourceType: str = "Condition"
    clinicalStatus: Optional[CodeableConcept] = None
    verificationStatus: Optional[CodeableConcept] = None
    category: Optional[List[CodeableConcept]] = None
    severity: Optional[CodeableConcept] = None
    code: CodeableConcept
    subject: Reference
    encounter: Optional[Reference] = None
    onsetDateTime: Optional[datetime] = None
    recordedDate: Optional[datetime] = None

    # Metadata
    contains_phi: bool = True

    def get_patient_id(self) -> Optional[str]:
        """Extract patient ID"""
        return self.subject.extract_id() if self.subject else None

    def get_condition_code(self) -> Optional[str]:
        """Get condition code (ICD-10, SNOMED)"""
        return self.code.get_primary_code()

    def get_condition_name(self) -> Optional[str]:
        """Get condition name"""
        return self.code.get_primary_display()


class Encounter(BaseModel):
    """FHIR Encounter resource"""
    id: str
    resourceType: str = "Encounter"
    status: str
    class_: Optional[CodeableConcept] = Field(None, alias="class")
    type: Optional[List[CodeableConcept]] = None
    subject: Reference
    period: Optional[Period] = None
    reasonCode: Optional[List[CodeableConcept]] = None

    # Metadata
    contains_phi: bool = True

    def get_patient_id(self) -> Optional[str]:
        """Extract patient ID"""
        return self.subject.extract_id() if self.subject else None


# ═════════════════════════════════════════════════════════════
# Clinical Analysis Models
# ═════════════════════════════════════════════════════════════

class VitalSignReading(BaseModel):
    """Processed vital sign reading"""
    vital_name: str
    loinc_code: str
    value: float
    unit: str
    timestamp: datetime
    status: ObservationStatus
    is_critical: bool = False
    is_abnormal: bool = False
    interpretation: Optional[str] = None


class PatientRiskFactors(BaseModel):
    """Identified patient risk factors"""
    hypertension: bool = False
    diabetes: bool = False
    chronic_kidney_disease: bool = False
    heart_disease: bool = False
    high_fall_risk: bool = False
    polypharmacy: bool = False
    critical_vitals: bool = False
    medication_adherence_issues: bool = False

    custom_risk_factors: List[str] = Field(default_factory=list)


class PatientRiskAssessment(BaseModel):
    """Comprehensive patient risk assessment"""
    patient_id: str
    assessment_date: datetime
    risk_score: float = Field(..., ge=0, le=100)
    risk_category: RiskCategory

    # Clinical data
    vital_signs: List[VitalSignReading] = Field(default_factory=list)
    active_medications: List[Dict[str, Any]] = Field(default_factory=list)
    conditions: List[Dict[str, Any]] = Field(default_factory=list)
    recent_encounters: List[Dict[str, Any]] = Field(default_factory=list)

    # Risk analysis
    risk_factors: PatientRiskFactors
    identified_gaps: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)

    # Confidence and metadata
    confidence_score: float = Field(..., ge=0, le=1)
    analyzed_by: str = "GenZ-Healthcare-Agent"
    analysis_version: str = "1.0.0"


class MedicationReview(BaseModel):
    """Medication review results"""
    patient_id: str
    review_date: datetime
    total_medications: int
    active_medications: int

    # Analysis
    polypharmacy_risk: bool
    high_risk_medications: List[Dict[str, Any]] = Field(default_factory=list)
    drug_interactions: List[Dict[str, Any]] = Field(default_factory=list)
    duplicate_therapies: List[Dict[str, Any]] = Field(default_factory=list)

    # Recommendations
    recommendations: List[str] = Field(default_factory=list)
    requires_pharmacist_review: bool = False

    # Metadata
    confidence_score: float
    reviewed_by: str = "GenZ-Medication-Review-Agent"


class ClinicalAlert(BaseModel):
    """Clinical alert/notification"""
    alert_id: str
    patient_id: str
    alert_type: str
    severity: AlertSeverity
    timestamp: datetime

    # Alert content
    title: str
    description: str
    clinical_context: Optional[str] = None

    # Actionable information
    recommended_actions: List[str] = Field(default_factory=list)
    care_team_notified: List[str] = Field(default_factory=list)

    # Metadata
    triggered_by: str
    requires_immediate_action: bool = False
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


class ClinicalReport(BaseModel):
    """Base clinical report model"""
    report_id: str
    report_type: str
    patient_id: str
    generated_at: datetime
    generated_by: str

    # Report content
    summary: str
    detailed_findings: Dict[str, Any]
    recommendations: List[str]

    # Classification
    contains_phi: bool = True
    classification_level: str = "RESTRICTED"

    # Audit
    accessed_by: List[str] = Field(default_factory=list)
    access_log: List[Dict[str, Any]] = Field(default_factory=list)


# ═════════════════════════════════════════════════════════════
# Data Quality Models
# ═════════════════════════════════════════════════════════════

class DataQualityCheck(BaseModel):
    """Data quality validation result"""
    check_name: str
    passed: bool
    severity: AlertSeverity
    message: str
    checked_at: datetime
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None


class FHIRValidationResult(BaseModel):
    """FHIR resource validation result"""
    valid: bool
    resource_type: str
    resource_id: str

    # Validation details
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    quality_checks: List[DataQualityCheck] = Field(default_factory=list)

    # Completeness
    completeness_score: float = Field(..., ge=0, le=1)
    required_fields_present: bool

    # Metadata
    validated_at: datetime
    validator_version: str


# ═════════════════════════════════════════════════════════════
# Utility Functions
# ═════════════════════════════════════════════════════════════

def parse_fhir_observation(fhir_dict: Dict[str, Any]) -> Observation:
    """Parse FHIR Observation from dictionary"""
    return Observation(**fhir_dict)


def parse_fhir_medication(fhir_dict: Dict[str, Any]) -> MedicationStatement:
    """Parse FHIR MedicationStatement from dictionary"""
    return MedicationStatement(**fhir_dict)


if __name__ == "__main__":
    # Example usage
    print("Healthcare Data Models Loaded")
    print("=" * 60)
    print("Available FHIR Resource Models:")
    for model in [Patient, Observation, MedicationStatement, Condition, Encounter]:
        print(f"  - {model.__name__}")
    print("\nClinical Analysis Models:")
    for model in [PatientRiskAssessment, MedicationReview, ClinicalAlert, ClinicalReport]:
        print(f"  - {model.__name__}")
