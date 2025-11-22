"""
Patient Data Models for COPD Telehealth Intervention

Defines Pydantic models for patient demographics, clinical data, and equity stratification.
"""

from datetime import date, datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, validator, EmailStr


class RaceEthnicity(str, Enum):
    """NIH-compliant race and ethnicity categories"""
    WHITE = "Non-Hispanic White"
    BLACK = "Non-Hispanic Black/African American"
    HISPANIC = "Hispanic/Latino (any race)"
    ASIAN = "Asian"
    AIAN = "American Indian/Alaska Native"
    NHPI = "Native Hawaiian/Pacific Islander"
    MULTIPLE = "Multiple races"
    OTHER = "Other"
    UNKNOWN = "Unknown/Not reported"


class RuralityCategory(str, Enum):
    """USDA Rural-Urban Continuum Code categories"""
    URBAN = "Urban (RUCC 1-3)"
    RURAL = "Rural (RUCC 4-6)"
    HIGHLY_RURAL = "Highly Rural (RUCC 7-9)"


class DigitalLiteracyLevel(str, Enum):
    """Digital literacy classification based on eHEALS score"""
    LOW = "Low (<24)"
    MODERATE = "Moderate (24-32)"
    HIGH = "High (>32)"


class COPDSeverity(str, Enum):
    """GOLD classification for COPD severity"""
    GOLD_1 = "GOLD 1 (Mild, FEV1 ≥80%)"
    GOLD_2 = "GOLD 2 (Moderate, 50% ≤ FEV1 < 80%)"
    GOLD_3 = "GOLD 3 (Severe, 30% ≤ FEV1 < 50%)"
    GOLD_4 = "GOLD 4 (Very Severe, FEV1 < 30%)"


class PatientDemographics(BaseModel):
    """Patient demographic information"""
    patient_id: str = Field(..., description="Unique patient identifier (de-identified)")
    date_of_birth: date = Field(..., description="Patient date of birth")
    sex: str = Field(..., description="Biological sex", regex="^(Male|Female|Other)$")
    race_ethnicity: RaceEthnicity
    zip_code: str = Field(..., description="5-digit ZIP code", regex="^\d{5}$")
    rurality: RuralityCategory
    insurance_type: str = Field(..., description="Insurance", regex="^(Medicare|Medicaid|Commercial|Uninsured|Other)$")

    @property
    def age(self) -> int:
        """Calculate age from date of birth"""
        today = date.today()
        return today.year - self.date_of_birth.year - (
            (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
        )

    class Config:
        schema_extra = {
            "example": {
                "patient_id": "PT-00123",
                "date_of_birth": "1955-03-15",
                "sex": "Female",
                "race_ethnicity": "Non-Hispanic White",
                "zip_code": "12345",
                "rurality": "Rural (RUCC 4-6)",
                "insurance_type": "Medicare"
            }
        }


class ClinicalProfile(BaseModel):
    """Patient clinical information and COPD severity"""
    patient_id: str
    copd_diagnosis_date: date
    copd_severity: COPDSeverity
    fev1_percent: float = Field(..., ge=0, le=150, description="FEV1 as % predicted")
    fvc_percent: float = Field(..., ge=0, le=150, description="FVC as % predicted")
    charlson_comorbidity_index: int = Field(..., ge=0, le=33, description="Charlson comorbidity score")
    comorbidities: List[str] = Field(default_factory=list, description="ICD-10 codes for comorbidities")

    # Baseline clinical metrics
    baseline_cat_score: Optional[int] = Field(None, ge=0, le=40, description="Baseline CAT score")
    exacerbations_past_year: int = Field(..., ge=0, description="Exacerbations in past 12 months")
    hospitalizations_past_year: int = Field(..., ge=0, description="Hospitalizations in past 12 months")

    # Oxygen use
    on_home_oxygen: bool = False
    oxygen_flow_rate: Optional[float] = Field(None, ge=0, description="L/min if on oxygen")

    @validator('fev1_percent', 'fvc_percent')
    def validate_pulmonary_function(cls, v):
        if v < 10 or v > 150:
            raise ValueError('Pulmonary function value outside plausible range')
        return v

    class Config:
        schema_extra = {
            "example": {
                "patient_id": "PT-00123",
                "copd_diagnosis_date": "2018-06-10",
                "copd_severity": "GOLD 2 (Moderate, 50% ≤ FEV1 < 80%)",
                "fev1_percent": 65.2,
                "fvc_percent": 72.1,
                "charlson_comorbidity_index": 4,
                "comorbidities": ["I10", "E11.9", "I50.9"],
                "baseline_cat_score": 22,
                "exacerbations_past_year": 2,
                "hospitalizations_past_year": 1,
                "on_home_oxygen": False
            }
        }


class DigitalLiteracyAssessment(BaseModel):
    """eHealth Literacy Scale (eHEALS) assessment"""
    patient_id: str
    assessment_date: date

    # eHEALS items (1-5 scale each)
    eheals_1: int = Field(..., ge=1, le=5, description="Know what health resources are online")
    eheals_2: int = Field(..., ge=1, le=5, description="Know where to find helpful resources")
    eheals_3: int = Field(..., ge=1, le=5, description="Know how to find helpful resources")
    eheals_4: int = Field(..., ge=1, le=5, description="Know how to use Internet for health questions")
    eheals_5: int = Field(..., ge=1, le=5, description="Know how to use health info found online")
    eheals_6: int = Field(..., ge=1, le=5, description="Have skills to evaluate resources")
    eheals_7: int = Field(..., ge=1, le=5, description="Can tell high from low quality")
    eheals_8: int = Field(..., ge=1, le=5, description="Feel confident using info for decisions")

    @property
    def total_score(self) -> int:
        """Calculate total eHEALS score (8-40)"""
        return (self.eheals_1 + self.eheals_2 + self.eheals_3 + self.eheals_4 +
                self.eheals_5 + self.eheals_6 + self.eheals_7 + self.eheals_8)

    @property
    def literacy_level(self) -> DigitalLiteracyLevel:
        """Classify digital literacy level"""
        score = self.total_score
        if score < 24:
            return DigitalLiteracyLevel.LOW
        elif score <= 32:
            return DigitalLiteracyLevel.MODERATE
        else:
            return DigitalLiteracyLevel.HIGH

    class Config:
        schema_extra = {
            "example": {
                "patient_id": "PT-00123",
                "assessment_date": "2025-01-15",
                "eheals_1": 3,
                "eheals_2": 3,
                "eheals_3": 2,
                "eheals_4": 3,
                "eheals_5": 3,
                "eheals_6": 2,
                "eheals_7": 2,
                "eheals_8": 3
            }
        }


class PatientEnrollment(BaseModel):
    """Patient enrollment and randomization information"""
    patient_id: str
    enrollment_date: date
    cluster_id: str = Field(..., description="Clinic/site cluster identifier")

    # Study design fields
    study_arm: str = Field(..., description="Study arm", regex="^(Intervention|Control|Waitlist)$")
    randomization_date: Optional[date] = None
    intervention_start_date: Optional[date] = None

    # Stepped-wedge specific
    wedge_step: Optional[int] = Field(None, ge=0, le=10, description="Which step cluster crossed over")

    # Eligibility
    meets_inclusion_criteria: bool = True
    exclusion_reasons: List[str] = Field(default_factory=list)

    # Consent
    informed_consent_date: Optional[date] = None
    consent_version: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "patient_id": "PT-00123",
                "enrollment_date": "2025-02-01",
                "cluster_id": "CLINIC-05",
                "study_arm": "Intervention",
                "randomization_date": "2025-01-15",
                "intervention_start_date": "2025-02-01",
                "wedge_step": 2,
                "meets_inclusion_criteria": True,
                "informed_consent_date": "2025-01-28",
                "consent_version": "v1.2"
            }
        }


class EquityStratificationProfile(BaseModel):
    """Complete equity stratification profile combining multiple dimensions"""
    patient_id: str
    race_ethnicity: RaceEthnicity
    rurality: RuralityCategory
    digital_literacy: DigitalLiteracyLevel

    # Additional equity-relevant factors
    primary_language: str = "English"
    needs_interpreter: bool = False
    has_reliable_internet: bool = True
    has_smartphone: bool = True
    has_caregiver_support: bool = False

    # Socioeconomic proxy measures
    medicaid_eligible: bool = False
    lives_in_hpsa: bool = Field(False, description="Health Professional Shortage Area")

    @property
    def intersectionality_score(self) -> int:
        """
        Calculate intersectionality burden score (0-7)
        Higher = more potential barriers
        """
        score = 0
        if self.race_ethnicity != RaceEthnicity.WHITE:
            score += 1
        if self.rurality != RuralityCategory.URBAN:
            score += 1
        if self.digital_literacy == DigitalLiteracyLevel.LOW:
            score += 1
        if not self.has_reliable_internet:
            score += 1
        if not self.has_smartphone:
            score += 1
        if self.medicaid_eligible:
            score += 1
        if self.lives_in_hpsa:
            score += 1
        return score

    @property
    def risk_category(self) -> str:
        """Categorize equity risk"""
        score = self.intersectionality_score
        if score <= 1:
            return "Low barrier"
        elif score <= 3:
            return "Moderate barrier"
        else:
            return "High barrier (intensive support needed)"

    class Config:
        schema_extra = {
            "example": {
                "patient_id": "PT-00123",
                "race_ethnicity": "Non-Hispanic Black/African American",
                "rurality": "Rural (RUCC 4-6)",
                "digital_literacy": "Low (<24)",
                "primary_language": "English",
                "needs_interpreter": False,
                "has_reliable_internet": False,
                "has_smartphone": True,
                "has_caregiver_support": True,
                "medicaid_eligible": True,
                "lives_in_hpsa": True
            }
        }


class CompletePatientProfile(BaseModel):
    """Complete integrated patient profile"""
    demographics: PatientDemographics
    clinical: ClinicalProfile
    digital_literacy: DigitalLiteracyAssessment
    enrollment: PatientEnrollment
    equity: EquityStratificationProfile

    @property
    def patient_id(self) -> str:
        """Unified patient ID across all components"""
        return self.demographics.patient_id

    def to_analysis_dict(self) -> dict:
        """
        Convert to flat dictionary for statistical analysis

        Returns:
            Dictionary with all relevant variables for regression models
        """
        return {
            # Demographics
            "patient_id": self.patient_id,
            "age": self.demographics.age,
            "sex": self.demographics.sex,
            "race_ethnicity": self.demographics.race_ethnicity.value,
            "rurality": self.demographics.rurality.value,

            # Clinical
            "copd_severity": self.clinical.copd_severity.value,
            "fev1_percent": self.clinical.fev1_percent,
            "charlson_index": self.clinical.charlson_comorbidity_index,
            "baseline_cat": self.clinical.baseline_cat_score,
            "exacerbations_past_year": self.clinical.exacerbations_past_year,

            # Digital literacy
            "eheals_score": self.digital_literacy.total_score,
            "digital_literacy": self.digital_literacy.literacy_level.value,

            # Study design
            "cluster_id": self.enrollment.cluster_id,
            "study_arm": self.enrollment.study_arm,
            "wedge_step": self.enrollment.wedge_step,

            # Equity
            "intersectionality_score": self.equity.intersectionality_score,
            "equity_risk": self.equity.risk_category,
            "has_reliable_internet": self.equity.has_reliable_internet,
            "has_caregiver_support": self.equity.has_caregiver_support
        }

    class Config:
        schema_extra = {
            "example": {
                "demographics": {
                    "patient_id": "PT-00123",
                    "date_of_birth": "1955-03-15",
                    "sex": "Female",
                    "race_ethnicity": "Non-Hispanic White",
                    "zip_code": "12345",
                    "rurality": "Rural (RUCC 4-6)",
                    "insurance_type": "Medicare"
                },
                "clinical": {
                    "patient_id": "PT-00123",
                    "copd_diagnosis_date": "2018-06-10",
                    "copd_severity": "GOLD 2 (Moderate, 50% ≤ FEV1 < 80%)",
                    "fev1_percent": 65.2,
                    "fvc_percent": 72.1,
                    "charlson_comorbidity_index": 4,
                    "comorbidities": ["I10", "E11.9"],
                    "baseline_cat_score": 22,
                    "exacerbations_past_year": 2,
                    "hospitalizations_past_year": 1
                },
                "digital_literacy": {
                    "patient_id": "PT-00123",
                    "assessment_date": "2025-01-15",
                    "eheals_1": 3, "eheals_2": 3, "eheals_3": 2, "eheals_4": 3,
                    "eheals_5": 3, "eheals_6": 2, "eheals_7": 2, "eheals_8": 3
                },
                "enrollment": {
                    "patient_id": "PT-00123",
                    "enrollment_date": "2025-02-01",
                    "cluster_id": "CLINIC-05",
                    "study_arm": "Intervention",
                    "wedge_step": 2
                },
                "equity": {
                    "patient_id": "PT-00123",
                    "race_ethnicity": "Non-Hispanic White",
                    "rurality": "Rural (RUCC 4-6)",
                    "digital_literacy": "Low (<24)",
                    "has_reliable_internet": False,
                    "has_smartphone": True,
                    "medicaid_eligible": False
                }
            }
        }
