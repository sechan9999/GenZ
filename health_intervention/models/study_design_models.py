"""
Study Design Models for Quasi-Experimental Designs

Defines models for stepped-wedge cluster RCT and interrupted time-series designs.
"""

from datetime import date, datetime
from enum import Enum
from typing import Optional, List, Dict
from pydantic import BaseModel, Field, validator


class StudyDesignType(str, Enum):
    """Type of quasi-experimental design"""
    STEPPED_WEDGE = "Stepped-Wedge Cluster RCT"
    INTERRUPTED_TIME_SERIES = "Interrupted Time-Series"
    PARALLEL_CRT = "Parallel Cluster RCT"
    BEFORE_AFTER = "Before-After"


class ClusterCharacteristics(BaseModel):
    """Characteristics of a study cluster (clinic/site)"""
    cluster_id: str = Field(..., description="Unique cluster identifier")
    cluster_name: str
    cluster_type: str = Field(..., description="Type", regex="^(Academic|Community|FQHC|VA|Rural Health|Other)$")

    # Geographic
    state: str
    county: str
    rurality: str = Field(..., regex="^(Urban|Rural|Highly Rural)$")

    # Size and capacity
    copd_panel_size: int = Field(..., ge=0, description="Number of COPD patients in clinic panel")
    total_patient_panel: int = Field(..., ge=0)
    number_of_providers: int = Field(..., ge=1)

    # Baseline metrics (for stratification)
    baseline_readmission_rate: float = Field(..., ge=0, le=100, description="% readmitted within 30 days")
    ehr_system: str = Field(..., description="EHR vendor (Epic, Cerner, etc.)")
    has_patient_portal: bool = True

    # Organizational readiness
    organizational_readiness_score: Optional[float] = Field(None, ge=1, le=5, description="ORC score")

    class Config:
        schema_extra = {
            "example": {
                "cluster_id": "CLINIC-01",
                "cluster_name": "Memorial Health Clinic",
                "cluster_type": "Community",
                "state": "Ohio",
                "county": "Franklin",
                "rurality": "Urban",
                "copd_panel_size": 145,
                "total_patient_panel": 3200,
                "number_of_providers": 8,
                "baseline_readmission_rate": 23.5,
                "ehr_system": "Epic",
                "has_patient_portal": True,
                "organizational_readiness_score": 4.2
            }
        }


class SteppedWedgeDesign(BaseModel):
    """
    Stepped-wedge cluster randomized trial configuration

    All clusters start in control, sequentially cross over to intervention
    """
    design_id: str
    study_start_date: date
    study_end_date: date

    # Design parameters
    number_of_clusters: int = Field(..., ge=3, le=100)
    number_of_steps: int = Field(..., ge=2, le=20, description="Number of crossover steps")
    step_duration_months: int = Field(..., ge=1, le=6, description="Length of each step")

    # Randomization
    randomization_date: date
    randomization_method: str = Field(
        ...,
        description="Method",
        regex="^(Simple|Stratified|Restricted|Constrained)$"
    )
    stratification_variables: List[str] = Field(
        default_factory=list,
        description="Variables used for stratification (e.g., 'size', 'rurality', 'baseline_rate')"
    )

    # Clusters per step
    clusters_per_step: int = Field(..., ge=1, description="Number of clusters crossing over each step")

    # Washout/transition period
    washout_period_weeks: int = Field(0, ge=0, le=8, description="Washout period after crossover")

    @property
    def total_study_duration_months(self) -> int:
        """Calculate total study duration"""
        return self.step_duration_months * (self.number_of_steps + 1)  # +1 for baseline period

    @property
    def expected_total_measurements(self) -> int:
        """Calculate expected number of cluster-time observations"""
        return self.number_of_clusters * (self.number_of_steps + 1)

    def get_intervention_status(self, cluster_step: int, current_step: int) -> str:
        """
        Determine intervention status for a cluster at given time point

        Args:
            cluster_step: Step at which cluster crosses over (1-indexed)
            current_step: Current step in study (0=baseline, 1-N=steps)

        Returns:
            "Control" or "Intervention"
        """
        if current_step >= cluster_step:
            return "Intervention"
        else:
            return "Control"

    class Config:
        schema_extra = {
            "example": {
                "design_id": "SW-COPD-2025",
                "study_start_date": "2025-01-01",
                "study_end_date": "2026-06-30",
                "number_of_clusters": 20,
                "number_of_steps": 5,
                "step_duration_months": 2,
                "randomization_date": "2024-12-15",
                "randomization_method": "Stratified",
                "stratification_variables": ["size", "rurality", "baseline_rate"],
                "clusters_per_step": 4,
                "washout_period_weeks": 2
            }
        }


class SteppedWedgeAllocation(BaseModel):
    """Allocation of clusters to steps in stepped-wedge design"""
    design_id: str
    cluster_id: str
    step_allocated: int = Field(..., ge=1, description="Step at which cluster crosses over")
    allocation_date: date
    crossover_date: date = Field(..., description="Date cluster begins intervention")

    # Allocation concealment
    allocation_concealed_until: date

    class Config:
        schema_extra = {
            "example": {
                "design_id": "SW-COPD-2025",
                "cluster_id": "CLINIC-01",
                "step_allocated": 2,
                "allocation_date": "2024-12-15",
                "crossover_date": "2025-05-01",
                "allocation_concealed_until": "2025-04-15"
            }
        }


class InterruptedTimeSeriesDesign(BaseModel):
    """
    Interrupted time-series design configuration

    Single intervention point, multiple pre- and post-intervention measurements
    """
    design_id: str
    intervention_name: str
    intervention_date: date = Field(..., description="Date intervention implemented")

    # Study period
    study_start_date: date
    study_end_date: date

    # Measurement frequency
    measurement_frequency: str = Field(
        ...,
        description="Frequency of outcome measurement",
        regex="^(Daily|Weekly|Monthly|Quarterly)$"
    )

    # Time points
    pre_intervention_periods: int = Field(..., ge=8, description="Number of pre-intervention time points")
    post_intervention_periods: int = Field(..., ge=8, description="Number of post-intervention time points")

    # Control for secular trends
    control_series_available: bool = Field(
        False,
        description="Whether control time series available (e.g., different geographic region)"
    )
    control_series_id: Optional[str] = None

    # Seasonality
    adjust_for_seasonality: bool = True
    seasonal_period: Optional[int] = Field(None, description="Seasonal cycle length (e.g., 12 for monthly data)")

    @property
    def total_time_points(self) -> int:
        """Total number of time points"""
        return self.pre_intervention_periods + self.post_intervention_periods

    @property
    def adequate_power(self) -> bool:
        """
        Check if design has adequate time points for power

        Rule of thumb: ≥8 pre and ≥8 post for 80% power
        """
        return (self.pre_intervention_periods >= 8 and
                self.post_intervention_periods >= 8)

    def time_point_to_date(self, time_point: int) -> date:
        """
        Convert time point index to calendar date

        Args:
            time_point: Time point index (0 = first measurement)

        Returns:
            Date corresponding to that time point
        """
        from datetime import timedelta

        if self.measurement_frequency == "Monthly":
            days = time_point * 30  # Approximate
        elif self.measurement_frequency == "Weekly":
            days = time_point * 7
        elif self.measurement_frequency == "Daily":
            days = time_point
        elif self.measurement_frequency == "Quarterly":
            days = time_point * 91  # Approximate
        else:
            days = 0

        return self.study_start_date + timedelta(days=days)

    class Config:
        schema_extra = {
            "example": {
                "design_id": "ITS-COPD-2025",
                "intervention_name": "State-wide Telehealth COPD Program",
                "intervention_date": "2025-07-01",
                "study_start_date": "2023-07-01",
                "study_end_date": "2027-06-30",
                "measurement_frequency": "Monthly",
                "pre_intervention_periods": 24,
                "post_intervention_periods": 24,
                "control_series_available": True,
                "control_series_id": "Adjacent State",
                "adjust_for_seasonality": True,
                "seasonal_period": 12
            }
        }


class TimeSeriesObservation(BaseModel):
    """Single observation in interrupted time-series"""
    design_id: str
    time_point: int = Field(..., ge=0, description="Sequential time point (0, 1, 2, ...)")
    observation_date: date

    # Timing variables
    time: int = Field(..., description="Continuous time variable (months from study start)")
    intervention: int = Field(..., ge=0, le=1, description="Binary: 0=pre-intervention, 1=post-intervention")
    time_after_intervention: int = Field(..., ge=0, description="Months since intervention (0 if pre-intervention)")

    # Outcome
    outcome_value: float = Field(..., description="Outcome at this time point (e.g., readmission rate per 100)")
    outcome_count: Optional[int] = Field(None, description="Count if outcome is count/rate")
    outcome_denominator: Optional[int] = Field(None, description="Denominator for rate calculation")

    # Seasonality controls
    month: int = Field(..., ge=1, le=12, description="Calendar month (1-12)")
    quarter: int = Field(..., ge=1, le=4, description="Calendar quarter (1-4)")
    season: str = Field(..., regex="^(Winter|Spring|Summer|Fall)$")

    @validator('time_after_intervention')
    def validate_time_after(cls, v, values):
        if values.get('intervention') == 0 and v != 0:
            raise ValueError('time_after_intervention must be 0 for pre-intervention periods')
        return v

    class Config:
        schema_extra = {
            "example": {
                "design_id": "ITS-COPD-2025",
                "time_point": 24,
                "observation_date": "2025-07-01",
                "time": 24,
                "intervention": 1,
                "time_after_intervention": 0,
                "outcome_value": 22.3,
                "outcome_count": 45,
                "outcome_denominator": 202,
                "month": 7,
                "quarter": 3,
                "season": "Summer"
            }
        }


class SegmentedRegressionModel(BaseModel):
    """
    Segmented regression model specification for ITS analysis

    Model: Y_t = β0 + β1*Time + β2*Intervention + β3*Time_after + ε_t
    """
    model_id: str
    design_id: str
    outcome_name: str
    outcome_type: str = Field(..., regex="^(Continuous|Count|Binary|Rate)$")

    # Model specification
    includes_time_trend: bool = True
    includes_level_change: bool = True
    includes_slope_change: bool = True

    # Adjustments
    seasonal_adjustment: bool = False
    autocorrelation_adjustment: bool = True
    lag_order: int = Field(1, ge=0, le=12, description="AR lag order (0=no autocorrelation)")

    # Covariates
    covariates: List[str] = Field(default_factory=list, description="Additional adjustment variables")

    # Model fit (populated after fitting)
    beta_0_intercept: Optional[float] = None
    beta_1_baseline_trend: Optional[float] = None
    beta_2_level_change: Optional[float] = None
    beta_3_slope_change: Optional[float] = None

    r_squared: Optional[float] = Field(None, ge=0, le=1)
    aic: Optional[float] = None
    bic: Optional[float] = None

    # Diagnostics
    durbin_watson: Optional[float] = Field(None, description="Test for autocorrelation")
    ljung_box_p: Optional[float] = Field(None, description="Ljung-Box test p-value")

    @property
    def immediate_effect(self) -> Optional[float]:
        """Immediate level change at intervention point"""
        return self.beta_2_level_change

    @property
    def trend_change(self) -> Optional[float]:
        """Change in slope after intervention"""
        return self.beta_3_slope_change

    def predicted_counterfactual(self, time_point: int) -> Optional[float]:
        """
        Calculate counterfactual prediction (what would have happened without intervention)

        Args:
            time_point: Time point for prediction (post-intervention)

        Returns:
            Predicted outcome value if intervention had not occurred
        """
        if self.beta_0_intercept is None or self.beta_1_baseline_trend is None:
            return None

        return self.beta_0_intercept + (self.beta_1_baseline_trend * time_point)

    class Config:
        schema_extra = {
            "example": {
                "model_id": "ITS-MODEL-001",
                "design_id": "ITS-COPD-2025",
                "outcome_name": "30-day readmission rate",
                "outcome_type": "Rate",
                "includes_time_trend": True,
                "includes_level_change": True,
                "includes_slope_change": True,
                "seasonal_adjustment": True,
                "autocorrelation_adjustment": True,
                "lag_order": 2,
                "beta_0_intercept": 22.5,
                "beta_1_baseline_trend": 0.1,
                "beta_2_level_change": -3.2,
                "beta_3_slope_change": -0.3,
                "r_squared": 0.78,
                "durbin_watson": 1.95
            }
        }


class SampleSizeCalculation(BaseModel):
    """Sample size and power calculation for study designs"""
    calculation_id: str
    design_type: StudyDesignType
    calculation_date: date

    # Parameters
    alpha: float = Field(0.05, gt=0, lt=1, description="Type I error rate (two-sided)")
    power: float = Field(0.80, gt=0, lt=1, description="Statistical power (1 - β)")

    # Effect size
    control_rate: float = Field(..., ge=0, le=1, description="Control group event rate")
    intervention_rate: float = Field(..., ge=0, le=1, description="Intervention group event rate")
    effect_size_absolute: float = Field(..., description="Absolute difference")
    effect_size_relative: float = Field(..., description="Relative reduction (e.g., 0.30 = 30%)")

    # Design-specific parameters
    icc: Optional[float] = Field(None, ge=0, le=1, description="Intracluster correlation coefficient")
    cluster_size: Optional[int] = Field(None, ge=1, description="Average cluster size")
    number_of_steps: Optional[int] = Field(None, ge=1, description="For stepped-wedge")

    # Results
    required_clusters: Optional[int] = Field(None, description="Required number of clusters")
    required_patients: Optional[int] = Field(None, description="Required total sample size")
    design_effect: Optional[float] = Field(None, description="Design effect for clustering")

    @validator('effect_size_absolute', always=True)
    def calculate_absolute_effect(cls, v, values):
        if 'control_rate' in values and 'intervention_rate' in values:
            return values['control_rate'] - values['intervention_rate']
        return v

    @validator('effect_size_relative', always=True)
    def calculate_relative_effect(cls, v, values):
        if 'control_rate' in values and 'intervention_rate' in values and values['control_rate'] > 0:
            return (values['control_rate'] - values['intervention_rate']) / values['control_rate']
        return v

    def calculate_stepped_wedge_sample_size(self) -> Dict[str, int]:
        """
        Calculate sample size for stepped-wedge design using Hussey & Hughes method

        Simplified calculation - use specialized software for production
        """
        import math

        if self.icc is None or self.cluster_size is None or self.number_of_steps is None:
            raise ValueError("ICC, cluster_size, and number_of_steps required")

        # Z-scores
        z_alpha = 1.96  # Two-sided α=0.05
        z_beta = 0.84   # Power=0.80

        # Variance components
        p1 = self.control_rate
        p2 = self.intervention_rate
        var_p1 = p1 * (1 - p1)
        var_p2 = p2 * (1 - p2)

        # Design effect (simplified)
        k = self.number_of_steps
        m = self.cluster_size
        rho = self.icc
        de = 1 + ((k + 1) / 2) * m * rho  # Approximate SW design effect

        # Sample size per arm (individual-level)
        n_per_arm = ((z_alpha + z_beta) ** 2 * (var_p1 + var_p2)) / ((p1 - p2) ** 2)

        # Adjust for clustering and SW design
        n_total = n_per_arm * 2 * de

        # Number of clusters
        clusters = math.ceil(n_total / (m * k))

        return {
            "required_clusters": clusters,
            "required_patients": clusters * m,
            "design_effect": de
        }

    class Config:
        schema_extra = {
            "example": {
                "calculation_id": "CALC-001",
                "design_type": "Stepped-Wedge Cluster RCT",
                "calculation_date": "2024-10-01",
                "alpha": 0.05,
                "power": 0.80,
                "control_rate": 0.22,
                "intervention_rate": 0.15,
                "effect_size_absolute": 0.07,
                "effect_size_relative": 0.32,
                "icc": 0.03,
                "cluster_size": 40,
                "number_of_steps": 5,
                "required_clusters": 20,
                "required_patients": 800,
                "design_effect": 1.5
            }
        }


class DataMonitoringPlan(BaseModel):
    """Data Safety Monitoring Board (DSMB) plan"""
    plan_id: str
    study_design_id: str

    # DSMB composition
    dsmb_members: List[str] = Field(..., description="Names/roles of DSMB members")
    dsmb_charter_date: date

    # Monitoring frequency
    interim_analyses_planned: int = Field(..., ge=1, description="Number of interim analyses")
    interim_analysis_timepoints: List[str] = Field(..., description="E.g., ['6 months', '12 months']")

    # Stopping rules
    efficacy_stopping_boundary: Optional[float] = Field(None, description="O'Brien-Fleming or similar")
    futility_stopping_boundary: Optional[float] = None
    harm_stopping_rule: str = Field(..., description="Criteria for stopping due to harm")

    # Safety monitoring
    adverse_event_monitoring: bool = True
    serious_adverse_event_reporting_timeframe: str = "24 hours"

    class Config:
        schema_extra = {
            "example": {
                "plan_id": "DSMB-001",
                "study_design_id": "SW-COPD-2025",
                "dsmb_members": ["Dr. Smith (Chair, Biostatistician)", "Dr. Jones (Pulmonologist)", "Dr. Lee (Ethicist)"],
                "dsmb_charter_date": "2024-11-01",
                "interim_analyses_planned": 2,
                "interim_analysis_timepoints": ["Month 9 (after step 3)", "Month 15 (after step 5)"],
                "efficacy_stopping_boundary": 0.001,
                "harm_stopping_rule": "Stop if readmission rate >25% in intervention arm",
                "adverse_event_monitoring": True,
                "serious_adverse_event_reporting_timeframe": "24 hours"
            }
        }
