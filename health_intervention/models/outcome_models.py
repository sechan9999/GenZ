"""
Clinical Outcome Models for COPD Telehealth Intervention

Defines models for symptom assessments, adherence tracking, and KPI measurements.
"""

from datetime import date, datetime
from enum import Enum
from typing import Optional, List, Dict
from pydantic import BaseModel, Field, validator


class CATAssessment(BaseModel):
    """
    COPD Assessment Test (CAT) - Primary symptom measure

    8-item questionnaire, each scored 0-5, total 0-40
    Higher scores indicate worse health status
    MCID (Minimal Clinically Important Difference) = 2 points
    """
    patient_id: str
    assessment_date: datetime
    source: str = Field(..., description="Data source", regex="^(App|Survey|EHR|Research)$")

    # CAT items (0-5 each)
    cat_1_cough: int = Field(..., ge=0, le=5, description="Cough frequency")
    cat_2_phlegm: int = Field(..., ge=0, le=5, description="Phlegm in chest")
    cat_3_chest_tightness: int = Field(..., ge=0, le=5, description="Chest tightness")
    cat_4_breathlessness: int = Field(..., ge=0, le=5, description="Breathlessness on exertion")
    cat_5_activity_limitation: int = Field(..., ge=0, le=5, description="Activity limitation at home")
    cat_6_confidence: int = Field(..., ge=0, le=5, description="Confidence leaving home")
    cat_7_sleep: int = Field(..., ge=0, le=5, description="Sleep quality")
    cat_8_energy: int = Field(..., ge=0, le=5, description="Energy level")

    @property
    def total_score(self) -> int:
        """Calculate total CAT score (0-40)"""
        return (self.cat_1_cough + self.cat_2_phlegm + self.cat_3_chest_tightness +
                self.cat_4_breathlessness + self.cat_5_activity_limitation +
                self.cat_6_confidence + self.cat_7_sleep + self.cat_8_energy)

    @property
    def impact_category(self) -> str:
        """Classify symptom impact"""
        score = self.total_score
        if score <= 10:
            return "Low impact"
        elif score <= 20:
            return "Medium impact"
        elif score <= 30:
            return "High impact"
        else:
            return "Very high impact"

    def change_from_baseline(self, baseline_score: int) -> Dict[str, any]:
        """
        Calculate change from baseline and determine clinical significance

        Args:
            baseline_score: Baseline CAT score

        Returns:
            Dictionary with change metrics
        """
        change = self.total_score - baseline_score
        mcid_achieved = abs(change) >= 2  # MCID = 2 points

        return {
            "baseline_score": baseline_score,
            "current_score": self.total_score,
            "change": change,
            "percent_change": (change / baseline_score * 100) if baseline_score > 0 else 0,
            "mcid_achieved": mcid_achieved,
            "improved": change < 0 and mcid_achieved,
            "worsened": change > 0 and mcid_achieved,
            "stable": not mcid_achieved
        }

    class Config:
        schema_extra = {
            "example": {
                "patient_id": "PT-00123",
                "assessment_date": "2025-03-15T09:30:00",
                "source": "App",
                "cat_1_cough": 3,
                "cat_2_phlegm": 2,
                "cat_3_chest_tightness": 3,
                "cat_4_breathlessness": 4,
                "cat_5_activity_limitation": 3,
                "cat_6_confidence": 2,
                "cat_7_sleep": 3,
                "cat_8_energy": 2
            }
        }


class DailySymptomCheck(BaseModel):
    """Daily symptom monitoring (simplified, app-based)"""
    patient_id: str
    check_date: date
    timestamp: datetime

    # Quick symptom assessment (0-10 scales)
    breathlessness: int = Field(..., ge=0, le=10, description="Breathlessness severity today")
    cough: int = Field(..., ge=0, le=10, description="Cough severity today")
    sputum: int = Field(..., ge=0, le=10, description="Sputum production today")
    energy: int = Field(..., ge=0, le=10, description="Energy level today")

    # Boolean symptom flags
    worsening_symptoms: bool = False
    increased_sputum_purulence: bool = False
    fever: bool = False
    chest_pain: bool = False

    # Vital signs (if available)
    oxygen_saturation: Optional[int] = Field(None, ge=0, le=100, description="SpO2 %")
    heart_rate: Optional[int] = Field(None, ge=30, le=200, description="Beats per minute")

    @property
    def exacerbation_risk(self) -> str:
        """
        Assess risk of COPD exacerbation based on symptom changes

        Uses modified Anthonisen criteria
        """
        major_symptoms = sum([
            self.breathlessness >= 7,
            self.cough >= 7,
            self.increased_sputum_purulence
        ])

        minor_symptoms = sum([
            self.fever,
            self.worsening_symptoms,
            self.oxygen_saturation is not None and self.oxygen_saturation < 90
        ])

        if major_symptoms >= 2:
            return "High risk - Consider immediate clinical contact"
        elif major_symptoms == 1 and minor_symptoms >= 1:
            return "Moderate risk - Monitor closely"
        else:
            return "Low risk"

    class Config:
        schema_extra = {
            "example": {
                "patient_id": "PT-00123",
                "check_date": "2025-03-15",
                "timestamp": "2025-03-15T08:00:00",
                "breathlessness": 4,
                "cough": 3,
                "sputum": 2,
                "energy": 6,
                "worsening_symptoms": False,
                "increased_sputum_purulence": False,
                "oxygen_saturation": 94,
                "heart_rate": 78
            }
        }


class MedicationAdherence(BaseModel):
    """Medication adherence tracking"""
    patient_id: str
    date: date
    timestamp: datetime

    # Medication confirmations
    controller_inhaler_taken: bool = Field(..., description="Long-acting bronchodilator/ICS taken")
    rescue_inhaler_uses: int = Field(..., ge=0, le=20, description="Short-acting rescue inhaler uses")

    # Technique
    correct_inhaler_technique: Optional[bool] = Field(None, description="Self-reported correct technique")

    # Barriers to adherence
    forgot: bool = False
    side_effects: bool = False
    cost_barrier: bool = False
    ran_out: bool = False

    class Config:
        schema_extra = {
            "example": {
                "patient_id": "PT-00123",
                "date": "2025-03-15",
                "timestamp": "2025-03-15T20:00:00",
                "controller_inhaler_taken": True,
                "rescue_inhaler_uses": 2,
                "correct_inhaler_technique": True,
                "forgot": False
            }
        }


class AdherenceMetrics(BaseModel):
    """
    Calculated adherence metrics over a time period

    Primary KPI: ≥80% adherence to daily monitoring activities
    """
    patient_id: str
    start_date: date
    end_date: date
    total_days: int

    # Activity completion counts
    days_with_symptom_check: int = Field(..., ge=0)
    days_with_medication_confirmation: int = Field(..., ge=0)
    days_with_spirometry: int = Field(..., ge=0)
    days_with_activity_sync: int = Field(..., ge=0)

    # Composite adherence (≥3 of 4 activities)
    days_with_high_engagement: int = Field(..., ge=0, description="Days with ≥3 activities completed")

    @property
    def adherence_rate(self) -> float:
        """Calculate overall adherence rate (%)"""
        return (self.days_with_high_engagement / self.total_days * 100) if self.total_days > 0 else 0

    @property
    def adherence_category(self) -> str:
        """Classify adherence level"""
        rate = self.adherence_rate
        if rate >= 80:
            return "High adherence (≥80%)"
        elif rate >= 50:
            return "Moderate adherence (50-79%)"
        else:
            return "Low adherence (<50%)"

    @property
    def meets_kpi_threshold(self) -> bool:
        """Check if patient meets primary KPI (≥80% adherence)"""
        return self.adherence_rate >= 80

    def engagement_by_activity(self) -> Dict[str, float]:
        """Calculate adherence rate for each activity type"""
        return {
            "symptom_check": (self.days_with_symptom_check / self.total_days * 100) if self.total_days > 0 else 0,
            "medication_confirmation": (self.days_with_medication_confirmation / self.total_days * 100) if self.total_days > 0 else 0,
            "spirometry": (self.days_with_spirometry / self.total_days * 100) if self.total_days > 0 else 0,
            "activity_sync": (self.days_with_activity_sync / self.total_days * 100) if self.total_days > 0 else 0
        }

    class Config:
        schema_extra = {
            "example": {
                "patient_id": "PT-00123",
                "start_date": "2025-02-01",
                "end_date": "2025-02-28",
                "total_days": 28,
                "days_with_symptom_check": 25,
                "days_with_medication_confirmation": 26,
                "days_with_spirometry": 20,
                "days_with_activity_sync": 22,
                "days_with_high_engagement": 24
            }
        }


class HospitalizationEvent(BaseModel):
    """Hospital readmission or ED visit event"""
    patient_id: str
    event_id: str = Field(..., description="Unique event identifier")

    # Event details
    event_type: str = Field(..., description="Type", regex="^(Hospitalization|ED Visit|Observation)$")
    admission_date: date
    discharge_date: Optional[date] = None
    length_of_stay: Optional[int] = Field(None, ge=0, description="Days")

    # Clinical details
    primary_diagnosis: str = Field(..., description="ICD-10 code for primary diagnosis")
    copd_related: bool = Field(..., description="COPD as primary or secondary diagnosis")

    # Readmission tracking
    index_discharge_date: Optional[date] = Field(None, description="Date of index hospitalization discharge")
    days_since_discharge: Optional[int] = Field(None, ge=0)

    @property
    def is_30day_readmission(self) -> bool:
        """Determine if event is 30-day readmission (primary KPI)"""
        if self.index_discharge_date and self.days_since_discharge is not None:
            return self.days_since_discharge <= 30
        return False

    @property
    def is_preventable(self) -> bool:
        """
        Flag potentially preventable readmissions

        COPD-related readmissions within 30 days are potentially preventable
        """
        return self.is_30day_readmission and self.copd_related and self.event_type == "Hospitalization"

    class Config:
        schema_extra = {
            "example": {
                "patient_id": "PT-00123",
                "event_id": "HOSP-00456",
                "event_type": "Hospitalization",
                "admission_date": "2025-03-20",
                "discharge_date": "2025-03-24",
                "length_of_stay": 4,
                "primary_diagnosis": "J44.1",
                "copd_related": True,
                "index_discharge_date": "2025-02-28",
                "days_since_discharge": 20
            }
        }


class ReadmissionOutcome(BaseModel):
    """
    30-day readmission outcome (Primary KPI)

    Binary outcome for stepped-wedge analysis
    """
    patient_id: str
    index_discharge_date: date
    follow_up_period_days: int = 30

    # Outcome
    readmitted: bool = Field(..., description="Any readmission within 30 days")
    readmission_date: Optional[date] = None
    days_to_readmission: Optional[int] = Field(None, ge=0, le=30)

    # Risk adjustment variables
    age: int
    copd_severity: str
    charlson_index: int
    prior_hospitalizations_1yr: int

    @validator('days_to_readmission')
    def validate_readmission_timing(cls, v, values):
        if values.get('readmitted') and v is None:
            raise ValueError('days_to_readmission required if readmitted=True')
        if not values.get('readmitted') and v is not None:
            raise ValueError('days_to_readmission should be None if readmitted=False')
        return v

    class Config:
        schema_extra = {
            "example": {
                "patient_id": "PT-00123",
                "index_discharge_date": "2025-02-28",
                "follow_up_period_days": 30,
                "readmitted": True,
                "readmission_date": "2025-03-20",
                "days_to_readmission": 20,
                "age": 70,
                "copd_severity": "GOLD 2",
                "charlson_index": 4,
                "prior_hospitalizations_1yr": 2
            }
        }


class QualityOfLifeAssessment(BaseModel):
    """EQ-5D-5L health-related quality of life assessment"""
    patient_id: str
    assessment_date: date

    # EQ-5D-5L dimensions (1-5 scale)
    mobility: int = Field(..., ge=1, le=5)
    self_care: int = Field(..., ge=1, le=5)
    usual_activities: int = Field(..., ge=1, le=5)
    pain_discomfort: int = Field(..., ge=1, le=5)
    anxiety_depression: int = Field(..., ge=1, le=5)

    # Visual analogue scale (0-100)
    vas_score: int = Field(..., ge=0, le=100, description="Self-rated health today")

    @property
    def health_state(self) -> str:
        """Generate 5-digit health state profile"""
        return f"{self.mobility}{self.self_care}{self.usual_activities}{self.pain_discomfort}{self.anxiety_depression}"

    def calculate_utility(self, value_set: str = "US") -> float:
        """
        Calculate utility score using EQ-5D-5L value set

        Note: Simplified calculation. In practice, use published value sets.

        Args:
            value_set: Country-specific value set (US, UK, etc.)

        Returns:
            Utility score (0=death, 1=perfect health)
        """
        # Simplified US value set approximation
        # In production, use actual EQ-5D-5L US crosswalk or value set
        base_utility = 1.0
        dimension_weights = {
            "mobility": 0.05,
            "self_care": 0.04,
            "usual_activities": 0.05,
            "pain_discomfort": 0.06,
            "anxiety_depression": 0.05
        }

        for dimension, weight in dimension_weights.items():
            level = getattr(self, dimension)
            base_utility -= weight * (level - 1)

        return max(base_utility, -0.5)  # US value set range: -0.573 to 1.0

    class Config:
        schema_extra = {
            "example": {
                "patient_id": "PT-00123",
                "assessment_date": "2025-03-01",
                "mobility": 2,
                "self_care": 1,
                "usual_activities": 2,
                "pain_discomfort": 3,
                "anxiety_depression": 2,
                "vas_score": 65
            }
        }


class StudyKPIs(BaseModel):
    """
    Complete KPI measurements for a patient during follow-up period

    Aligns with pre-specified primary and secondary outcomes
    """
    patient_id: str
    follow_up_start: date
    follow_up_end: date
    follow_up_months: int

    # PRIMARY KPIs
    # 1. 30-day readmission
    readmitted_30day: bool
    readmission_count: int = Field(0, ge=0)

    # 2. Symptom score reduction
    baseline_cat_score: int = Field(..., ge=0, le=40)
    final_cat_score: int = Field(..., ge=0, le=40)
    cat_score_change: int  # Calculated: final - baseline
    achieved_mcid: bool  # MCID = ≥2 point reduction

    # 3. Adherence rate
    adherence_rate_percent: float = Field(..., ge=0, le=100)
    high_adherence: bool  # ≥80%

    # SECONDARY KPIs
    ed_visits_copd: int = Field(0, ge=0, description="COPD-related ED visits")
    exacerbations_moderate: int = Field(0, ge=0)
    exacerbations_severe: int = Field(0, ge=0)

    qol_baseline: Optional[float] = Field(None, description="EQ-5D utility at baseline")
    qol_final: Optional[float] = Field(None, description="EQ-5D utility at end")
    qol_change: Optional[float] = None

    # Healthcare utilization
    total_inpatient_days: int = Field(0, ge=0)
    total_healthcare_cost: Optional[float] = Field(None, ge=0, description="USD")

    @property
    def composite_success(self) -> bool:
        """
        Define composite success outcome:
        - No 30-day readmission AND
        - Achieved MCID in CAT score AND
        - High adherence (≥80%)
        """
        return (not self.readmitted_30day and
                self.achieved_mcid and
                self.high_adherence)

    @validator('cat_score_change', always=True)
    def calculate_cat_change(cls, v, values):
        if 'final_cat_score' in values and 'baseline_cat_score' in values:
            return values['final_cat_score'] - values['baseline_cat_score']
        return v

    @validator('achieved_mcid', always=True)
    def check_mcid(cls, v, values):
        if 'cat_score_change' in values:
            return values['cat_score_change'] <= -2  # Reduction of ≥2 points
        return v

    @validator('high_adherence', always=True)
    def check_adherence_threshold(cls, v, values):
        if 'adherence_rate_percent' in values:
            return values['adherence_rate_percent'] >= 80
        return v

    class Config:
        schema_extra = {
            "example": {
                "patient_id": "PT-00123",
                "follow_up_start": "2025-02-01",
                "follow_up_end": "2025-08-01",
                "follow_up_months": 6,
                "readmitted_30day": False,
                "readmission_count": 0,
                "baseline_cat_score": 24,
                "final_cat_score": 18,
                "cat_score_change": -6,
                "achieved_mcid": True,
                "adherence_rate_percent": 85.2,
                "high_adherence": True,
                "ed_visits_copd": 1,
                "exacerbations_moderate": 1,
                "exacerbations_severe": 0,
                "qol_baseline": 0.62,
                "qol_final": 0.72,
                "qol_change": 0.10,
                "total_inpatient_days": 0,
                "total_healthcare_cost": 3250.00
            }
        }
