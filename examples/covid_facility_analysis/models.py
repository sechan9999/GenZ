"""
Data models for COVID-19 facility data analysis.

This module defines Pydantic models for EHR records, staffing rosters,
PPE inventory, and merged facility data.
"""

from datetime import date, datetime
from typing import Optional, List
from pydantic import BaseModel, Field, validator
from enum import Enum


class FacilityType(str, Enum):
    """Type of VA facility."""
    MEDICAL_CENTER = "medical_center"
    OUTPATIENT_CLINIC = "outpatient_clinic"
    LONG_TERM_CARE = "long_term_care"
    COMMUNITY_LIVING_CENTER = "community_living_center"


class CovidTestResult(str, Enum):
    """COVID-19 test result values."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    PENDING = "pending"
    INCONCLUSIVE = "inconclusive"


# ============================================================================
# Electronic Health Record (EHR) Data Model
# ============================================================================

class EHRRecord(BaseModel):
    """Electronic Health Record for COVID-19 patient data."""

    record_id: str = Field(..., description="Unique EHR record identifier")
    facility_id: str = Field(..., description="VA facility identifier (e.g., 'VA-528')")
    facility_name: str = Field(..., description="Full facility name")
    facility_type: FacilityType

    record_date: date = Field(..., description="Date of record")

    # Patient metrics (de-identified aggregate counts)
    total_patients: int = Field(ge=0, description="Total patients seen")
    covid_positive_count: int = Field(ge=0, description="COVID-19 positive patients")
    covid_hospitalized: int = Field(ge=0, description="COVID-19 hospitalizations")
    covid_icu_count: int = Field(ge=0, description="Patients in ICU")
    covid_ventilator_count: int = Field(ge=0, description="Patients on ventilators")
    covid_deaths: int = Field(ge=0, description="COVID-19 deaths")

    # Testing metrics
    tests_conducted: int = Field(ge=0, description="COVID-19 tests conducted")
    tests_positive: int = Field(ge=0, description="Positive tests")
    tests_pending: int = Field(ge=0, description="Pending results")

    # Vaccination metrics
    first_dose_administered: int = Field(ge=0, description="First vaccine doses given")
    second_dose_administered: int = Field(ge=0, description="Second vaccine doses given")
    booster_dose_administered: int = Field(ge=0, description="Booster doses given")

    # Metadata
    data_source: str = Field(default="VistA EHR", description="Source system")
    last_updated: datetime = Field(default_factory=datetime.now)

    @validator('covid_positive_count')
    def positive_not_exceed_total(cls, v, values):
        """Validate that COVID positive count doesn't exceed total patients."""
        if 'total_patients' in values and v > values['total_patients']:
            raise ValueError(f"COVID positive count ({v}) cannot exceed total patients ({values['total_patients']})")
        return v

    @validator('tests_positive')
    def positive_tests_not_exceed_total(cls, v, values):
        """Validate that positive tests don't exceed total tests."""
        if 'tests_conducted' in values and v > values['tests_conducted']:
            raise ValueError(f"Positive tests ({v}) cannot exceed total tests ({values['tests_conducted']})")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "record_id": "EHR-20210115-VA528-001",
                "facility_id": "VA-528",
                "facility_name": "VA Western New York Healthcare System",
                "facility_type": "medical_center",
                "record_date": "2021-01-15",
                "total_patients": 1247,
                "covid_positive_count": 34,
                "covid_hospitalized": 12,
                "covid_icu_count": 4,
                "covid_ventilator_count": 2,
                "covid_deaths": 0,
                "tests_conducted": 156,
                "tests_positive": 18,
                "tests_pending": 5,
                "first_dose_administered": 423,
                "second_dose_administered": 187,
                "booster_dose_administered": 0,
            }
        }


# ============================================================================
# Staffing Roster Data Model
# ============================================================================

class StaffingRoster(BaseModel):
    """Daily staffing roster for VA facility."""

    roster_id: str = Field(..., description="Unique roster identifier")
    facility_id: str = Field(..., description="VA facility identifier")
    facility_name: str = Field(..., description="Full facility name")

    roster_date: date = Field(..., description="Date of roster")

    # Staffing levels
    physicians_scheduled: int = Field(ge=0, description="Physicians scheduled")
    physicians_present: int = Field(ge=0, description="Physicians present")
    nurses_scheduled: int = Field(ge=0, description="Nurses scheduled")
    nurses_present: int = Field(ge=0, description="Nurses present")
    respiratory_therapists_scheduled: int = Field(ge=0)
    respiratory_therapists_present: int = Field(ge=0)
    support_staff_scheduled: int = Field(ge=0)
    support_staff_present: int = Field(ge=0)

    # Staff health status
    staff_covid_positive: int = Field(ge=0, description="Staff testing positive")
    staff_quarantined: int = Field(ge=0, description="Staff in quarantine")
    staff_vaccinated_full: int = Field(ge=0, description="Fully vaccinated staff")

    # Capacity metrics
    total_beds: int = Field(ge=0, description="Total bed capacity")
    occupied_beds: int = Field(ge=0, description="Occupied beds")
    covid_beds_available: int = Field(ge=0, description="COVID-designated beds available")
    icu_beds_total: int = Field(ge=0)
    icu_beds_occupied: int = Field(ge=0)

    # Metadata
    data_source: str = Field(default="HRIS", description="HR Information System")
    last_updated: datetime = Field(default_factory=datetime.now)

    @validator('physicians_present')
    def physicians_present_validation(cls, v, values):
        """Validate physicians present doesn't exceed scheduled."""
        if 'physicians_scheduled' in values and v > values['physicians_scheduled']:
            raise ValueError(f"Physicians present ({v}) exceeds scheduled ({values['physicians_scheduled']})")
        return v

    @validator('occupied_beds')
    def beds_validation(cls, v, values):
        """Validate occupied beds don't exceed total."""
        if 'total_beds' in values and v > values['total_beds']:
            raise ValueError(f"Occupied beds ({v}) exceeds total beds ({values['total_beds']})")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "roster_id": "ROSTER-20210115-VA528",
                "facility_id": "VA-528",
                "facility_name": "VA Western New York Healthcare System",
                "roster_date": "2021-01-15",
                "physicians_scheduled": 45,
                "physicians_present": 42,
                "nurses_scheduled": 156,
                "nurses_present": 148,
                "respiratory_therapists_scheduled": 12,
                "respiratory_therapists_present": 11,
                "support_staff_scheduled": 89,
                "support_staff_present": 85,
                "staff_covid_positive": 3,
                "staff_quarantined": 7,
                "staff_vaccinated_full": 234,
                "total_beds": 250,
                "occupied_beds": 187,
                "covid_beds_available": 15,
                "icu_beds_total": 24,
                "icu_beds_occupied": 18,
            }
        }


# ============================================================================
# PPE Inventory Data Model
# ============================================================================

class PPEInventory(BaseModel):
    """Personal Protective Equipment inventory tracking."""

    inventory_id: str = Field(..., description="Unique inventory record identifier")
    facility_id: str = Field(..., description="VA facility identifier")
    facility_name: str = Field(..., description="Full facility name")

    inventory_date: date = Field(..., description="Date of inventory count")

    # PPE quantities (units)
    n95_masks_count: int = Field(ge=0, description="N95 respirator masks")
    surgical_masks_count: int = Field(ge=0, description="Surgical masks")
    face_shields_count: int = Field(ge=0, description="Face shields")
    gowns_count: int = Field(ge=0, description="Isolation gowns")
    gloves_boxes: int = Field(ge=0, description="Boxes of gloves")
    hand_sanitizer_bottles: int = Field(ge=0, description="Hand sanitizer bottles")
    disinfectant_wipes_count: int = Field(ge=0, description="Disinfectant wipes")

    # Ventilator equipment
    ventilators_total: int = Field(ge=0, description="Total ventilators")
    ventilators_in_use: int = Field(ge=0, description="Ventilators currently in use")

    # Days of supply (calculated)
    n95_days_supply: Optional[float] = Field(None, ge=0, description="Days of N95 supply remaining")
    surgical_mask_days_supply: Optional[float] = Field(None, ge=0)
    gown_days_supply: Optional[float] = Field(None, ge=0)

    # Reorder status
    critical_shortage: bool = Field(default=False, description="Critical shortage flag")
    reorder_needed: List[str] = Field(default_factory=list, description="Items needing reorder")

    # Metadata
    data_source: str = Field(default="MERS (Medical Equipment Reporting System)", description="Source system")
    last_updated: datetime = Field(default_factory=datetime.now)

    @validator('ventilators_in_use')
    def ventilators_validation(cls, v, values):
        """Validate ventilators in use doesn't exceed total."""
        if 'ventilators_total' in values and v > values['ventilators_total']:
            raise ValueError(f"Ventilators in use ({v}) exceeds total ({values['ventilators_total']})")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "inventory_id": "PPE-20210115-VA528",
                "facility_id": "VA-528",
                "facility_name": "VA Western New York Healthcare System",
                "inventory_date": "2021-01-15",
                "n95_masks_count": 4500,
                "surgical_masks_count": 12000,
                "face_shields_count": 800,
                "gowns_count": 3200,
                "gloves_boxes": 450,
                "hand_sanitizer_bottles": 350,
                "disinfectant_wipes_count": 280,
                "ventilators_total": 18,
                "ventilators_in_use": 12,
                "n95_days_supply": 15.2,
                "surgical_mask_days_supply": 30.1,
                "gown_days_supply": 12.5,
                "critical_shortage": False,
                "reorder_needed": ["gowns", "face_shields"],
            }
        }


# ============================================================================
# Merged Facility Data Model
# ============================================================================

class FacilityDailySnapshot(BaseModel):
    """Merged daily snapshot of facility data from all sources."""

    snapshot_id: str = Field(..., description="Unique snapshot identifier")
    facility_id: str = Field(..., description="VA facility identifier")
    facility_name: str = Field(..., description="Standardized facility name")
    facility_type: FacilityType

    snapshot_date: date = Field(..., description="Date of snapshot")

    # EHR metrics
    ehr_data: Optional[EHRRecord] = None

    # Staffing metrics
    staffing_data: Optional[StaffingRoster] = None

    # PPE metrics
    ppe_data: Optional[PPEInventory] = None

    # Derived metrics
    occupancy_rate: Optional[float] = Field(None, ge=0, le=1, description="Bed occupancy rate")
    staff_shortage: Optional[bool] = Field(None, description="Staff shortage indicator")
    ppe_critical: Optional[bool] = Field(None, description="PPE critical shortage")
    covid_positivity_rate: Optional[float] = Field(None, ge=0, le=1, description="Test positivity rate")

    # Data quality indicators
    data_completeness: float = Field(ge=0, le=1, description="Completeness score (0-1)")
    is_duplicate: bool = Field(default=False, description="Duplicate record flag")
    merge_confidence: float = Field(ge=0, le=1, description="Merge confidence score")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    data_sources_used: List[str] = Field(default_factory=list)

    def calculate_derived_metrics(self):
        """Calculate derived metrics from component data."""
        if self.staffing_data:
            if self.staffing_data.total_beds > 0:
                self.occupancy_rate = self.staffing_data.occupied_beds / self.staffing_data.total_beds

            # Staff shortage if >10% absence rate
            total_scheduled = (self.staffing_data.physicians_scheduled +
                             self.staffing_data.nurses_scheduled)
            total_present = (self.staffing_data.physicians_present +
                           self.staffing_data.nurses_present)
            if total_scheduled > 0:
                absence_rate = 1 - (total_present / total_scheduled)
                self.staff_shortage = absence_rate > 0.10

        if self.ppe_data:
            self.ppe_critical = self.ppe_data.critical_shortage

        if self.ehr_data and self.ehr_data.tests_conducted > 0:
            self.covid_positivity_rate = (self.ehr_data.tests_positive /
                                         self.ehr_data.tests_conducted)

        # Calculate data completeness
        completeness_score = 0
        if self.ehr_data:
            completeness_score += 0.4
        if self.staffing_data:
            completeness_score += 0.3
        if self.ppe_data:
            completeness_score += 0.3
        self.data_completeness = completeness_score

    class Config:
        json_schema_extra = {
            "example": {
                "snapshot_id": "SNAPSHOT-20210115-VA528",
                "facility_id": "VA-528",
                "facility_name": "VA Western New York Healthcare System",
                "facility_type": "medical_center",
                "snapshot_date": "2021-01-15",
                "occupancy_rate": 0.748,
                "staff_shortage": False,
                "ppe_critical": False,
                "covid_positivity_rate": 0.115,
                "data_completeness": 1.0,
                "is_duplicate": False,
                "merge_confidence": 0.98,
                "data_sources_used": ["VistA EHR", "HRIS", "MERS"],
            }
        }


# ============================================================================
# Validation and Data Quality Models
# ============================================================================

class DataQualityIssue(BaseModel):
    """Data quality issue tracking."""

    issue_id: str
    facility_id: str
    issue_date: date
    issue_type: str  # "missing_data", "outlier", "duplicate", "validation_error"
    severity: str  # "critical", "high", "medium", "low"
    description: str
    affected_fields: List[str]
    resolution_status: str = "open"  # "open", "investigating", "resolved", "false_positive"

    created_at: datetime = Field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None


class DeduplicationMatch(BaseModel):
    """Record of fuzzy matching deduplication."""

    match_id: str
    record_1_id: str
    record_2_id: str
    match_score: float = Field(ge=0, le=1, description="Fuzzy match score")
    matched_on: List[str] = Field(description="Fields used for matching")
    action_taken: str  # "merged", "kept_both", "manual_review"

    created_at: datetime = Field(default_factory=datetime.now)
