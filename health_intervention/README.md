# Digital Health Intervention Evaluation Framework
## COPD Telehealth Remote Patient Monitoring

![Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 📋 Overview

This framework provides a comprehensive toolkit for evaluating digital health interventions, specifically remote patient monitoring applications for Chronic Obstructive Pulmonary Disease (COPD). It implements rigorous quasi-experimental study designs with equity-focused analyses.

**Key Features:**
- ✅ **Stepped-Wedge Cluster RCT** implementation
- ✅ **Interrupted Time-Series** with segmented regression
- ✅ **Equity Stratification** by race, rurality, and digital literacy
- ✅ **KPI Tracking**: 30-day readmission, symptom scores, adherence
- ✅ **COPD Monitoring App** with symptom tracking
- ✅ **Visualization Dashboard** for study outcomes

---

## 🏗️ Architecture

```
health_intervention/
├── models/                      # Data models (Pydantic)
│   ├── patient_models.py        # Patient demographics, clinical profiles
│   ├── outcome_models.py        # Outcomes, KPIs, symptom assessments
│   └── study_design_models.py   # Stepped-wedge, ITS designs
├── app/                         # COPD Monitoring Application
│   └── copd_monitoring_app.py   # Main patient-facing app
├── analysis/                    # Statistical analysis
│   ├── statistical_analysis.py # Stepped-wedge, ITS, equity analyses
│   └── visualization_dashboard.py # Outcome visualizations
├── data/                        # Patient data storage (gitignored)
├── tests/                       # Unit and integration tests
└── utils/                       # Utility functions

```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/sechan9999/GenZ.git
cd GenZ/health_intervention

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the COPD Monitoring App

```python
from models.patient_models import CompletePatientProfile
from app.copd_monitoring_app import COPDMonitoringApp

# Load patient profile (see demo in app/copd_monitoring_app.py)
patient_profile = CompletePatientProfile(...)

# Initialize app
app = COPDMonitoringApp(patient_profile, data_dir="./data")

# Daily symptom check-in
result = app.daily_symptom_checkin(
    breathlessness=6,
    cough=5,
    sputum=4,
    energy=4,
    worsening_symptoms=True,
    oxygen_saturation=92
)

# Record medication adherence
result = app.record_medication_adherence(
    controller_inhaler_taken=True,
    rescue_inhaler_uses=2,
    correct_inhaler_technique=True
)

# Generate patient report
report = app.generate_patient_report(months=6)
```

### Running Statistical Analyses

#### Stepped-Wedge Analysis

```python
from analysis.statistical_analysis import SteppedWedgeAnalysis
from models.study_design_models import SteppedWedgeDesign
import pandas as pd

# Define study design
design = SteppedWedgeDesign(
    design_id="SW-COPD-2025",
    study_start_date=date(2025, 1, 1),
    study_end_date=date(2026, 6, 30),
    number_of_clusters=20,
    number_of_steps=5,
    step_duration_months=2,
    randomization_method="Stratified",
    clusters_per_step=4
)

# Prepare data
analysis = SteppedWedgeAnalysis(design)
analysis.prepare_data(df, outcome_col="readmitted_30day")

# Analyze binary outcome (readmission)
results = analysis.analyze_binary_outcome(covariates=["age", "copd_severity"])

print(f"Odds Ratio: {results['odds_ratio']:.3f}")
print(f"95% CI: ({results['or_95ci_lower']:.3f}, {results['or_95ci_upper']:.3f})")
print(f"P-value: {results['intervention_pvalue']:.4f}")
```

#### Interrupted Time-Series Analysis

```python
from analysis.statistical_analysis import InterruptedTimeSeriesAnalysis
from models.study_design_models import InterruptedTimeSeriesDesign

# Define ITS design
design = InterruptedTimeSeriesDesign(
    design_id="ITS-COPD-2025",
    intervention_name="Statewide Telehealth Program",
    intervention_date=date(2025, 7, 1),
    study_start_date=date(2023, 7, 1),
    study_end_date=date(2027, 6, 30),
    measurement_frequency="Monthly",
    pre_intervention_periods=24,
    post_intervention_periods=24
)

# Prepare and analyze
its_analysis = InterruptedTimeSeriesAnalysis(design)
its_analysis.prepare_its_data(ts_df, outcome_col="readmission_rate")

# Fit segmented regression
seg_model = its_analysis.fit_segmented_regression(
    autocorrelation=True,
    seasonal_adjustment=True
)

# Interpret results
interpretation = its_analysis.interpret_its_results(seg_model)
print(f"Immediate level change: {interpretation['immediate_level_change']:.2f}")
print(f"Slope change: {interpretation['slope_change']:.2f}")

# Plot ITS
its_analysis.plot_its(save_path="its_plot.png")
```

#### Equity Stratification Analysis

```python
from analysis.statistical_analysis import EquityStratificationAnalysis

# Initialize equity analysis
equity = EquityStratificationAnalysis(data=df)

# Analyze by race/ethnicity
race_results = equity.analyze_by_race_ethnicity(
    outcome_col="readmitted_30day",
    race_col="race_ethnicity",
    intervention_col="intervention"
)

# Test for interaction
interaction = equity.test_interaction(
    outcome_col="readmitted_30day",
    intervention_col="intervention",
    strata_col="race_ethnicity"
)

print(f"Interaction significant: {interaction['interaction_significant']}")
print(f"P-value: {interaction['lr_pvalue']:.4f}")
```

### Creating Visualizations

```python
from analysis.visualization_dashboard import InterventionDashboard

# Initialize dashboard
dashboard = InterventionDashboard(output_dir="./visualizations")

# Create comprehensive dashboard
study_data = {
    'kpi_summary': kpi_data,
    'equity_results': equity_results,
    'patient_longitudinal_data': patient_df,
    'adherence_data': adherence_df,
    'n_clusters': 20,
    'n_steps': 5
}

dashboard.create_comprehensive_dashboard(study_data)

# Open visualizations/index.html in browser to view
```

---

## 📊 Key Performance Indicators (KPIs)

### Primary KPIs

| KPI | Definition | Target | Measurement |
|-----|------------|--------|-------------|
| **30-Day Readmission** | Hospital readmission within 30 days | <16% (30% reduction) | EHR data |
| **Symptom Score Reduction** | CAT score change from baseline | ≥2 point reduction (MCID) | Weekly surveys |
| **Adherence Rate** | Days with ≥3 of 4 activities completed | ≥80% for ≥70% of patients | App engagement |

### Secondary KPIs

- All-cause mortality
- ED visits (COPD-related)
- Health-related quality of life (EQ-5D-5L)
- Medication adherence (PDC ≥80%)
- Exacerbation rate
- Healthcare costs

---

## 🎯 Study Designs

### 1. Stepped-Wedge Cluster RCT

**When to use:**
- Phased rollout required due to resource constraints
- Intervention deemed beneficial (ethical to delay, not withhold)
- Need to account for temporal trends
- Cluster-level intervention (clinics, hospital systems)

**Design:**
```
Time:      T0    T1    T2    T3    T4    T5
Cluster 1  C     I     I     I     I     I
Cluster 2  C     C     I     I     I     I
Cluster 3  C     C     C     I     I     I
Cluster 4  C     C     C     C     I     I
Cluster 5  C     C     C     C     C     I

C = Control, I = Intervention
```

**Analysis:**
- Mixed-effects regression (logistic for binary, linear for continuous)
- Accounts for clustering (ICC), time trends, intervention effect
- Risk-adjusted for patient-level covariates

### 2. Interrupted Time-Series (ITS)

**When to use:**
- Policy-level intervention with clear implementation date
- No randomization feasible
- Sufficient pre-intervention data (≥8 time points)
- Need to control for secular trends

**Model:**
```
Y_t = β0 + β1*Time + β2*Intervention + β3*Time_after + ε_t
```

**Interpretation:**
- β1: Baseline trend (pre-intervention slope)
- β2: Immediate level change
- β3: Change in trend (slope difference post-intervention)

---

## 🌍 Equity Stratification

### Why Equity Matters

Digital health interventions risk exacerbating disparities if not designed inclusively. This framework pre-specifies equity analyses to ensure benefits are distributed fairly.

### Stratification Variables

1. **Race/Ethnicity** (NIH categories)
   - Non-Hispanic White
   - Non-Hispanic Black/African American
   - Hispanic/Latino
   - Asian, AIAN, NHPI, Multiple, Other

2. **Rurality** (USDA RUCC codes)
   - Urban (RUCC 1-3)
   - Rural (RUCC 4-6)
   - Highly Rural (RUCC 7-9)

3. **Digital Literacy** (eHEALS score)
   - Low (<24)
   - Moderate (24-32)
   - High (>32)

### Equity Metrics

**Disparity Ratio:**
```
DR = (Outcome rate in minoritized group) / (Outcome rate in White group)
```

**Goal:** DR ≤ 1.0 post-intervention (eliminate disparity)

**Intersectionality Score:**
- Captures cumulative burden across multiple dimensions
- Range: 0-7 (higher = more barriers)
- Informs tailored support strategies

---

## 📈 Data Models

### Patient Profile

```python
CompletePatientProfile:
  - demographics: PatientDemographics
  - clinical: ClinicalProfile (COPD severity, comorbidities)
  - digital_literacy: DigitalLiteracyAssessment (eHEALS)
  - enrollment: PatientEnrollment (study arm, cluster)
  - equity: EquityStratificationProfile
```

### Outcomes

```python
StudyKPIs:
  - readmitted_30day: bool
  - baseline_cat_score: int (0-40)
  - final_cat_score: int
  - cat_score_change: int
  - achieved_mcid: bool (≥2 point reduction)
  - adherence_rate_percent: float
  - high_adherence: bool (≥80%)
  - composite_success: bool (all 3 KPIs met)
```

### Study Design

```python
SteppedWedgeDesign:
  - number_of_clusters: int
  - number_of_steps: int
  - step_duration_months: int
  - randomization_method: str
  - stratification_variables: List[str]

InterruptedTimeSeriesDesign:
  - intervention_date: date
  - pre_intervention_periods: int (≥8)
  - post_intervention_periods: int (≥8)
  - measurement_frequency: str
  - adjust_for_seasonality: bool
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=health_intervention --cov-report=html

# Run specific test module
pytest tests/test_patient_models.py -v
```

---

## 📚 Documentation

Comprehensive documentation is available in `/docs`:

1. **[Digital Health Intervention Evaluation Framework](../docs/digital_health_intervention_evaluation.md)** (1,000+ lines)
   - Study design details
   - Statistical methods
   - Sample size calculations
   - Implementation protocols

2. **[Palantir Foundry EHR Integration](../docs/palantir_foundry_ehr_integration.md)**
   - FHIR data integration
   - Azure Event Hubs configuration
   - Databricks Delta Lake setup

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-analysis`)
3. Commit changes (`git commit -m 'Add new equity analysis'`)
4. Push to branch (`git push origin feature/new-analysis`)
5. Open a Pull Request

**Development guidelines:**
- Follow PEP 8 style guide
- Add docstrings (Google style)
- Write tests for new features
- Update documentation

---

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@software{genz_health_intervention_2025,
  title={Digital Health Intervention Evaluation Framework},
  author={GenZ Project Team},
  year={2025},
  url={https://github.com/sechan9999/GenZ/tree/main/health_intervention},
  version={1.0}
}
```

---

## 📖 References

### Methodological Papers

1. Hussey MA, Hughes JP. Design and analysis of stepped wedge cluster randomized trials. *Contemporary Clinical Trials*. 2007;28(2):182-191.

2. Bernal JL, Cummins S, Gasparrini A. Interrupted time series regression for the evaluation of public health interventions: a tutorial. *Int J Epidemiol*. 2017;46(1):348-355.

3. Hemming K, Taljaard M, Forbes AB. Analysis of cluster randomised stepped wedge trials with repeated cross-sectional samples. *Trials*. 2017;18(1):101.

### COPD & Digital Health

4. Noah B, Keller MS, Mosadeghi S, et al. Impact of remote patient monitoring on clinical outcomes: an updated meta-analysis of randomized controlled trials. *NPJ Digit Med*. 2018;1:20172.

5. Jones PW, et al. Development and first validation of the COPD Assessment Test. *Eur Respir J*. 2009;34(3):648-654.

### Health Equity

6. Ye S, Kronish I, Fleck E, et al. Telemedicine Expansion During the COVID-19 Pandemic and the Potential for Technology-Driven Disparities. *J Gen Intern Med*. 2021;36(1):256-258.

7. Norman CD, Skinner HA. eHEALS: The eHealth Literacy Scale. *J Med Internet Res*. 2006;8(4):e27.

---

## 📞 Contact

**Project Lead:** Gen Z Health Intervention Team
**Email:** [project-email@institution.edu]
**GitHub:** [https://github.com/sechan9999/GenZ](https://github.com/sechan9999/GenZ)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

---

## 🙏 Acknowledgments

- National Institutes of Health (NIH) for methodological guidance
- Patient Advisory Board for co-design input
- Participating clinics and healthcare systems
- Open-source community (Pandas, StatsModels, Matplotlib)

---

**Version:** 1.0
**Last Updated:** 2025-11-22
**Status:** Active Development
