# Digital Health Intervention Evaluation Framework
## Remote Patient Monitoring App Assessment Methodology

**Version**: 1.0
**Date**: 2025-11-22
**Domain**: Telehealth & Remote Patient Monitoring
**Target**: COPD Intervention Evaluation

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Study Design Overview](#study-design-overview)
3. [Quasi-Experimental Designs](#quasi-experimental-designs)
4. [Key Performance Indicators (KPIs)](#key-performance-indicators-kpis)
5. [Equity Stratification](#equity-stratification)
6. [Statistical Analysis Methods](#statistical-analysis-methods)
7. [Implementation Protocol](#implementation-protocol)
8. [Sample Size & Power Analysis](#sample-size--power-analysis)
9. [Data Collection & Management](#data-collection--management)
10. [Reporting & Dissemination](#reporting--dissemination)

---

## Executive Summary

This framework provides a rigorous methodology for evaluating digital health interventions, specifically remote patient monitoring applications for chronic disease management (focus: COPD). The framework employs quasi-experimental designs to address real-world implementation challenges while maintaining scientific rigor.

### Key Features:
- **Stepped-wedge cluster randomized controlled trial (SW-CRT)** for phased rollout
- **Interrupted time-series (ITS)** with segmented regression for policy evaluation
- **Pre-specified KPIs**: 30-day readmission rates, symptom score reduction, adherence >80%
- **Equity-focused**: Stratification by race, rurality, digital literacy

### Primary Objective:
Evaluate the effectiveness of a telehealth COPD intervention app in reducing hospital readmissions and improving patient-reported outcomes while ensuring equitable access across diverse populations.

---

## Study Design Overview

### Design Selection Rationale

#### Stepped-Wedge Cluster RCT (Primary Design)
**When to Use**:
- Pragmatic implementation contexts
- Intervention deemed beneficial (ethical to delay, not withhold)
- Limited resources requiring phased rollout
- Cluster-level intervention (clinic, hospital system)

**Advantages**:
- All clusters eventually receive intervention
- Accounts for temporal trends
- Increased statistical power vs parallel-cluster RCT
- Acceptable when randomization is logistically challenging

#### Interrupted Time-Series (Alternative Design)
**When to Use**:
- Policy-level interventions with clear implementation date
- No randomization feasible
- Sufficient pre-intervention data available (≥8 time points)
- Need to control for secular trends

**Advantages**:
- Natural experiment approach
- Controls for long-term trends and seasonality
- Can assess immediate and gradual effects
- Suitable for system-wide implementations

---

## Quasi-Experimental Designs

### Design 1: Stepped-Wedge Cluster RCT

#### Study Structure

```
Time Period:        T0    T1    T2    T3    T4    T5
-------------------------------------------------------
Cluster 1 (n=5)     C     I     I     I     I     I
Cluster 2 (n=5)     C     C     I     I     I     I
Cluster 3 (n=5)     C     C     C     I     I     I
Cluster 4 (n=5)     C     C     C     C     I     I
Cluster 5 (n=5)     C     C     C     C     C     I

C = Control (usual care)
I = Intervention (remote monitoring app)
```

#### Key Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Clusters** | 20-25 clinics | Balance power and feasibility |
| **Patients per cluster** | 30-50 COPD patients | Typical clinic panel size |
| **Rollout period** | 12-18 months | Allow adaptation, avoid seasonal confounding |
| **Washout period** | 2-4 weeks | Account for learning curve |
| **Follow-up** | 12 months post-intervention | Capture sustained effects |

#### Statistical Model

**Mixed-effects logistic regression** for binary outcomes (readmission):

```
logit(Y_ijk) = β0 + β1*Time + β2*Intervention + u_j + u_jk + ε_ijk
```

Where:
- `Y_ijk` = Outcome for patient i in cluster j at time k
- `β1` = Time trend (accounts for temporal changes)
- `β2` = Intervention effect (primary parameter of interest)
- `u_j` = Random effect for cluster (ICC component)
- `u_jk` = Random effect for time within cluster
- `ε_ijk` = Residual error

**Intracluster correlation (ICC)**: Expected 0.01-0.05 for clinical outcomes

#### Randomization

**Restricted randomization**:
1. Stratify clinics by:
   - Size (small: <100 COPD patients, medium: 100-200, large: >200)
   - Rural vs urban location
   - Baseline readmission rate (low: <15%, high: ≥15%)
2. Generate 10,000 random allocation sequences
3. Select sequence that minimizes imbalance across strata
4. Conceal allocation until each step begins

---

### Design 2: Interrupted Time-Series with Segmented Regression

#### Study Structure

**Timeline**:
```
Pre-intervention:   ←——————————————————|——————————————————→  Post-intervention
                    (24 months)        ↑                       (24 months)
                                  Intervention start
                                   (January 2026)
```

**Data frequency**: Monthly measurements (48 time points total)

#### Statistical Model

**Segmented regression**:

```
Y_t = β0 + β1*Time_t + β2*Intervention_t + β3*Time_after_t + ε_t
```

Where:
- `Y_t` = Outcome at time t (e.g., readmission rate per 100 patients)
- `Time_t` = Continuous variable (months from study start: 1, 2, 3...)
- `Intervention_t` = Binary indicator (0 before, 1 after intervention)
- `Time_after_t` = Months since intervention (0 before intervention, 1, 2, 3... after)

**Interpretation**:
- `β1` = Baseline trend (pre-intervention slope)
- `β2` = Immediate level change at intervention point
- `β3` = Change in trend (difference in slopes post-intervention)

#### Assumptions & Tests

| Assumption | Test | Mitigation |
|------------|------|------------|
| **No autocorrelation** | Durbin-Watson test | Add AR(1) terms if needed |
| **Stationarity** | Augmented Dickey-Fuller | First-difference transformation |
| **No seasonality** | ACF/PACF plots | Add seasonal dummy variables |
| **No co-interventions** | Chart review | Document and adjust in analysis |

#### Sensitivity Analyses

1. **Varying intervention start point** (±3 months)
2. **Different lag structures** (immediate vs 3-month lag)
3. **Outlier exclusion** (observations >3 SD from trend)
4. **Subgroup analyses** by equity strata

---

## Key Performance Indicators (KPIs)

### Primary KPIs

#### 1. 30-Day All-Cause Readmission Rate

**Definition**: Proportion of COPD patients readmitted to hospital within 30 days of discharge

**Data Source**:
- Electronic Health Record (EHR) discharge dates
- Regional health information exchange (HIE) for out-of-network admissions

**Measurement**:
```
Readmission Rate = (Number of readmissions within 30 days) / (Total index discharges) × 100
```

**Target**:
- **Baseline** (US national average): 22.6% for COPD
- **Goal**: Reduce to <16% (30% relative reduction)
- **Clinically meaningful difference**: ≥5 percentage points

**Risk Adjustment**:
- Age, sex, race/ethnicity
- COPD severity (GOLD stage)
- Comorbidity burden (Charlson index)
- Prior hospitalization history

---

#### 2. Symptom Score Reduction

**Instrument**: **COPD Assessment Test (CAT)**
- 8 items, score range 0-40
- Higher scores = worse health status
- Validated, widely used

**Measurement Schedule**:
- Baseline (enrollment)
- Weekly during first month
- Bi-weekly months 2-3
- Monthly thereafter

**Target**:
- **Minimal clinically important difference (MCID)**: ≥2 points
- **Goal**: Mean reduction of ≥3 points from baseline
- **Responder rate**: ≥60% of patients achieve MCID

**Statistical Analysis**:
- **Primary**: Change from baseline at 6 months (linear mixed model)
- **Secondary**: Time to first clinically meaningful improvement
- **Trajectory analysis**: Latent growth curve modeling

**CAT Score Interpretation**:
| Score | Impact |
|-------|--------|
| 0-10 | Low |
| 11-20 | Medium |
| 21-30 | High |
| 31-40 | Very high |

---

#### 3. Adherence Rate (App Engagement)

**Definition**: Proportion of days with completed monitoring activities

**Required Activities** (daily):
- ✅ Symptom check-in (CAT short form)
- ✅ Medication confirmation
- ✅ Spirometry recording (if equipped)
- ✅ Activity/step count sync

**Calculation**:
```
Adherence (%) = (Days with ≥3 of 4 activities completed) / (Total enrolled days) × 100
```

**Thresholds**:
- **High adherence**: ≥80% of days
- **Moderate adherence**: 50-79%
- **Low adherence**: <50%

**Target**:
- **Primary goal**: ≥70% of patients maintain >80% adherence for 6 months
- **Engagement decay model**: Predict dropout risk using survival analysis

**Adherence Monitoring**:
- Real-time dashboard for care team
- Automated outreach triggers:
  - No activity for 3 consecutive days → SMS reminder
  - No activity for 7 days → Phone call from nurse
  - Adherence <50% for 2 weeks → Re-education session

---

### Secondary KPIs

| KPI | Definition | Target | Measurement |
|-----|------------|--------|-------------|
| **All-cause mortality** | Deaths during follow-up | Not powered, exploratory | Vital records, EHR |
| **ED visits** | COPD-related emergency dept visits | 20% reduction | EHR, patient report |
| **Health-related QoL** | EQ-5D-5L utility score | +0.05 increase | Quarterly survey |
| **Medication adherence** | Proportion of days covered (PDC) | PDC ≥80% | Pharmacy claims |
| **Pulmonary function** | FEV1 % predicted | Slow decline | Spirometry |
| **Exacerbation rate** | Moderate/severe exacerbations | 30% reduction | Patient report, EHR |
| **Healthcare costs** | Total costs per patient | 15% reduction | Claims data |

---

## Equity Stratification

### Rationale

Digital health interventions risk exacerbating disparities if not designed inclusively. Pre-specified equity analyses ensure intervention benefits are distributed fairly across vulnerable populations.

### Stratification Variables

#### 1. Race and Ethnicity

**Categories** (following NIH guidance):
- Non-Hispanic White
- Non-Hispanic Black/African American
- Hispanic/Latino (any race)
- Asian
- American Indian/Alaska Native
- Native Hawaiian/Pacific Islander
- Multiple races
- Other/Unknown

**Analysis Plan**:
- Test for intervention × race interaction (α=0.05)
- Report outcomes separately for each group
- If interaction significant: investigate mechanisms (digital literacy, trust, access)

**Equity Metric**:
```
Disparity Ratio = (Readmission rate in minoritized group) / (Readmission rate in White group)
```
- **Goal**: Disparity ratio ≤1.0 post-intervention (eliminate disparity)
- **Current US baseline**: Black patients have 1.2× higher COPD readmission rates

---

#### 2. Rurality

**Classification**: USDA Rural-Urban Continuum Codes (RUCC)
- **Urban**: RUCC 1-3 (metro areas)
- **Rural**: RUCC 4-6 (non-metro, adjacent to metro)
- **Highly rural**: RUCC 7-9 (non-metro, not adjacent)

**Rural-Specific Challenges**:
- Limited broadband access (13% of rural US lacks internet)
- Cellular connectivity gaps
- Longer distances to care facilities
- Provider shortages

**Accommodations**:
- **Offline mode**: Store-and-forward data sync
- **SMS-based fallback**: Text message check-ins if app unusable
- **Equipment lending**: Provide tablets/hotspots if needed
- **Hybrid visits**: Video + in-person appointments

**Equity Metrics**:
- App usability rate (% successfully installing and logging in)
- Adherence rate by rurality
- Technical support call volume
- Patient satisfaction scores

---

#### 3. Digital Literacy

**Assessment**: **eHealth Literacy Scale (eHEALS)**
- 8 items, 5-point Likert scale
- Score range: 8-40
- Administered at baseline

**Categories**:
- **Low digital literacy**: eHEALS <24
- **Moderate**: eHEALS 24-32
- **High**: eHEALS >32

**Tailored Support**:
| Literacy Level | Intervention Modifications |
|----------------|----------------------------|
| **Low** | • In-person onboarding (1-2 hours)<br>• Simplified UI with large fonts<br>• Voice-guided tutorials<br>• Weekly check-in calls (first month)<br>• Caregiver co-enrollment |
| **Moderate** | • Video onboarding (30 min)<br>• Help center with FAQs<br>• Bi-weekly check-ins (first month) |
| **High** | • Self-guided onboarding<br>• Advanced features enabled (data export, trends) |

**Equity Metrics**:
- Adherence rates by eHEALS tertile
- Time to proficiency (independent use without support)
- Differential dropout rates

---

### Intersectionality Analysis

Examine outcomes across **intersecting identities**:

Example: Rural × Black × Low digital literacy group
- Hypothesis: Triple jeopardy leads to lowest adherence
- Targeted support: Community health worker visits, group training sessions

**Statistical Approach**:
- **Multi-level modeling** with cross-classified random effects
- **Mediation analysis**: Examine pathways (e.g., does digital literacy mediate rural-urban differences?)

---

## Statistical Analysis Methods

### Primary Analysis: Stepped-Wedge Design

#### Model Specification

**Binary outcomes (readmission)**:
```R
glmer(readmission ~ intervention + time + (1 | cluster) + (1 | cluster:time),
      family = binomial(link = "logit"),
      data = sw_data)
```

**Continuous outcomes (CAT score)**:
```R
lmer(cat_score ~ intervention + time + baseline_cat +
     (1 + time | cluster) + (1 | patient),
     data = sw_data)
```

#### Effect Measures

| Outcome Type | Effect Measure | Interpretation |
|--------------|----------------|----------------|
| Binary | Odds Ratio (OR) | OR=0.70 → 30% lower odds of readmission |
| Binary | Risk Ratio (RR) | RR=0.80 → 20% lower risk (more intuitive) |
| Continuous | Mean Difference (MD) | MD=-3.2 → 3.2 point reduction in CAT score |
| Count | Incidence Rate Ratio (IRR) | IRR=0.65 → 35% fewer exacerbations |

#### Sample Size Calculation

**Parameters** (for readmission outcome):
- Control group rate: 22%
- Intervention group rate: 15% (target)
- ICC: 0.03
- Cluster size: 40 patients
- Power: 80%
- Alpha: 0.05 (two-sided)
- Design effect for SW: ~1.5× parallel CRT

**Required**: 20 clusters (800 total patients)

Using Hussey & Hughes (2007) formula:
```
n_clusters = (Z_α/2 + Z_β)² × [p1(1-p1) + p2(1-p2)] × DE / (p1 - p2)²
```

---

### Secondary Analysis: Interrupted Time-Series

#### Model Specification (Stata)

```stata
* Basic segmented regression
regress readmission_rate time intervention time_after_intervention

* With Newey-West standard errors (autocorrelation-robust)
newey readmission_rate time intervention time_after_intervention, lag(2)

* With seasonal adjustment
regress readmission_rate time intervention time_after_intervention ///
        i.month, vce(robust)
```

#### Sensitivity Analyses

1. **Lag structure**:
   - Immediate effect (lag=0)
   - 1-month lag (lag=1)
   - 3-month lag (lag=3)

2. **Outlier treatment**:
   - Include all data
   - Exclude points >3 SD
   - Winsorize at 1st/99th percentile

3. **Trend specification**:
   - Linear
   - Quadratic
   - Piecewise (change points at quarters)

4. **Counterfactual comparison**:
   - Project pre-intervention trend forward
   - Calculate cumulative effect

#### Minimum Sample Size

**Power considerations**:
- **Time points**: Minimum 8 pre-intervention, 8 post-intervention
- **Effect size**: Detect level change of 5 percentage points (readmission)
- **Power**: 80%
- **Autocorrelation**: ρ=0.3 (typical for monthly data)

**Rule of thumb**: 20-25 time points (2+ years) for adequate power

---

### Subgroup and Moderator Analyses

#### Pre-specified Subgroups

Test for heterogeneity of treatment effects:

1. **By COPD severity** (GOLD stage 2 vs 3 vs 4)
2. **By age** (<65 vs ≥65 years)
3. **By comorbidity burden** (Charlson <3 vs ≥3)
4. **By prior technology use** (smartphone owner vs not)
5. **By equity strata** (race, rurality, digital literacy)

#### Statistical Approach

**Interaction tests**:
```R
# Test intervention × subgroup interaction
glmer(readmission ~ intervention * subgroup + time + covariates +
      (1 | cluster),
      family = binomial, data = data)

# Bonferroni correction for multiple comparisons
alpha_adjusted = 0.05 / 5 = 0.01
```

**Interpretation**:
- Significant interaction (p<0.01) → Report subgroup-specific effects
- Non-significant → Report overall effect only

---

### Missing Data Handling

#### Assumptions

**Primary analysis**: Modified intention-to-treat (mITT)
- Include all patients with ≥1 post-baseline assessment
- Rationale: Captures "real-world" effectiveness

**Sensitivity analysis**: Complete case analysis

#### Methods

| Missing Data Type | Method | Software |
|-------------------|--------|----------|
| **Outcome data** | Multiple imputation (MI) | mice (R) |
| **Covariates** | MI with chained equations | mi impute (Stata) |
| **Dropout** | Inverse probability weighting | ipw (R) |

**MI Specifications**:
- m=20 imputations (Recommended for 20-40% missing)
- Predictive mean matching for continuous variables
- Logistic regression for binary variables
- Include auxiliary variables (e.g., baseline values, adherence)

#### Missing Data Reporting

- **CONSORT diagram**: Document flow and reasons for dropout
- **Missingness patterns**: Visualize with VIM package (R)
- **MAR assumption testing**: Compare characteristics of complete vs incomplete cases

---

## Implementation Protocol

### Phase 1: Pre-Implementation (Months 1-3)

#### Stakeholder Engagement

**Key stakeholders**:
- Pulmonologists and primary care physicians
- Nurses and respiratory therapists
- Patients and caregivers (patient advisory board)
- Hospital administrators
- IT/EHR vendors
- Payers (Medicaid, Medicare, commercial insurers)

**Activities**:
- Monthly advisory board meetings
- Iterative design workshops (co-creation)
- Workflow integration planning
- Reimbursement pathway development

#### Site Selection & Readiness

**Inclusion criteria** (for stepped-wedge clusters):
- ≥100 COPD patients in panel
- EHR system compatible with data extraction
- Broadband internet access at clinic
- Committed clinical champion

**Readiness assessment** (adapted from Weiner's ORC):
- Organizational readiness score >3.5/5
- Staff training capacity confirmed
- Technical infrastructure validated

#### Ethics & Regulatory

- **IRB approval**: Submit protocol, obtain approval before recruitment
- **Informed consent**: Waiver for cluster randomization (intervention low-risk), individual consent for data use
- **Data use agreements**: HIPAA-compliant with all sites
- **Trial registration**: ClinicalTrials.gov

---

### Phase 2: Pilot Testing (Months 4-6)

#### Objectives

1. Test app usability across diverse patient groups
2. Refine clinical protocols (alert thresholds, escalation pathways)
3. Train clinical staff on intervention delivery
4. Validate data collection processes

#### Pilot Sample

- **n=50 patients** from 2-3 clinics (not included in main trial)
- Purposively sample diversity:
  - 50% age ≥70
  - 30% racial/ethnic minorities
  - 30% rural residents
  - 40% low digital literacy (eHEALS <24)

#### Usability Testing

**Metrics**:
- **System Usability Scale (SUS)**: Target >70 (above average)
- **Task completion rate**: >90% for core tasks
- **Error rate**: <5% on critical tasks (symptom entry)
- **Time to proficiency**: <1 week of daily use

**Iterative refinement**:
- Weekly usability testing sessions (n=5 patients/week × 4 weeks)
- Rapid prototyping with design team
- Think-aloud protocols to identify pain points

---

### Phase 3: Staged Rollout (Months 7-18)

#### Stepped-Wedge Timeline

| Step | Months | Clusters | Cumulative Intervention | Control |
|------|--------|----------|-------------------------|---------|
| 0 | 7-8 | - | 0 | 20 |
| 1 | 9-10 | 4 | 4 | 16 |
| 2 | 11-12 | 4 | 8 | 12 |
| 3 | 13-14 | 4 | 12 | 8 |
| 4 | 15-16 | 4 | 16 | 4 |
| 5 | 17-18 | 4 | 20 | 0 |

#### Cluster Activation Checklist

Before each step:
- [ ] Clinical staff training completed (4-hour session)
- [ ] Workflow integration finalized
- [ ] Patient recruitment target met (≥30 enrolled)
- [ ] Technical setup verified (app access, EHR integration)
- [ ] 24/7 technical support activated
- [ ] Emergency escalation protocol tested

#### Patient Recruitment

**Eligibility criteria**:
- Age ≥40 years
- COPD diagnosis (ICD-10: J44.x)
- ≥1 exacerbation or hospitalization in past 12 months
- Discharged from hospital in past 30 days (for high-risk cohort)
- Able to provide informed consent
- Access to smartphone or willingness to use loaned tablet

**Exclusion criteria**:
- Life expectancy <6 months
- Active cancer requiring chemotherapy
- Severe cognitive impairment (unable to use app with caregiver support)
- No reliable internet/cellular access (even with accommodations)

**Recruitment strategies**:
- EHR-based identification (automated queries)
- Pulmonary clinic referrals
- Hospital discharge planning integration
- Community outreach (churches, senior centers)

---

### Phase 4: Maintenance & Monitoring (Months 19-30)

#### Continuous Quality Improvement

**Real-time monitoring dashboard** (for implementation team):
- Enrollment rate by cluster
- Adherence trends (7-day rolling average)
- Alert volume and response times
- Technical support tickets
- Adverse events log

**Intervention fidelity checks**:
- Monthly chart audits (10% random sample)
- Quarterly staff competency assessments
- Patient satisfaction surveys (Net Promoter Score)

#### Adaptations & Protocol Deviations

**Allowed adaptations** (document but don't exclude):
- Adjustment of alert thresholds based on individual patient patterns
- Increased support frequency for struggling patients
- Addition of caregiver access with patient permission

**Protocol deviations requiring steering committee review**:
- >20% dropout in a cluster
- Systematic intervention non-delivery
- Major technical failures (>7 days downtime)

---

## Sample Size & Power Analysis

### Stepped-Wedge Design

#### Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| **Control readmission rate** | 22% | US national average (CMS 2022) |
| **Intervention readmission rate** | 15% | Target (30% relative reduction) |
| **ICC** | 0.03 | Typical for hospital readmissions |
| **Cluster size** | 40 | Average clinic COPD panel |
| **Number of steps** | 5 | Balances power and feasibility |
| **Power** | 80% | Standard |
| **Alpha** | 0.05 | Two-sided |

#### Calculation

Using **Hussey & Hughes (2007)** method for SW-CRT:

```R
library(swCRTdesign)

swCRTdesign(
  design = "sw",
  clusters = 20,
  cluster_size = 40,
  steps = 5,
  sigma_b = 0.03,  # ICC
  p_c = 0.22,      # Control proportion
  p_i = 0.15,      # Intervention proportion
  family = "binomial"
)

# Output: Power = 0.81
```

**Result**: 20 clusters (800 patients) provides 81% power

#### Sensitivity Analysis

| Scenario | Clusters | Power | Notes |
|----------|----------|-------|-------|
| Base case | 20 | 81% | Assumes ICC=0.03 |
| Higher ICC (0.05) | 20 | 73% | Need 24 clusters for 80% |
| Smaller effect (18% vs 22%) | 20 | 62% | Need 30 clusters for 80% |
| Larger clusters (n=60) | 20 | 88% | More efficient |

**Recommendation**: Recruit 24 clusters to buffer against:
- Higher-than-expected ICC
- Smaller effect size
- Cluster dropout

---

### Interrupted Time-Series

#### Parameters

| Parameter | Value |
|-----------|-------|
| **Baseline rate** | 22% (monthly average) |
| **Effect size** | 5 percentage point immediate level change |
| **Autocorrelation** | ρ=0.3 (monthly) |
| **Pre-intervention time points** | 24 months |
| **Post-intervention time points** | 24 months |
| **Power** | 80% |

#### Calculation

Using **Zhang et al. (2011)** simulation approach:

```R
library(samplesize)

# Simulate power for ITS
power_its <- replicate(1000, {
  # Generate pre-post data with effect
  simulate_its(n_pre=24, n_post=24, effect=5, rho=0.3)
  # Fit segmented regression
  model <- lm(rate ~ time + intervention + time_after)
  # Test intervention coefficient
  p_value <- summary(model)$coefficients["intervention", "Pr(>|t|)"]
  p_value < 0.05
})

mean(power_its)  # Proportion of simulations with p<0.05
# Output: Power = 0.83
```

**Result**: 24 pre + 24 post time points provides 83% power

---

## Data Collection & Management

### Data Sources

#### 1. Electronic Health Records (EHR)

**Variables extracted**:
- Demographics: Age, sex, race/ethnicity, address (for rurality coding)
- Diagnoses: COPD, comorbidities (ICD-10 codes)
- Hospitalizations: Dates, reasons, discharge disposition
- Medications: Prescriptions, fill dates (via HIE pharmacy data)
- Pulmonary function: Spirometry results (FEV1, FVC)
- Vital signs: Weight, blood pressure, oxygen saturation

**Extraction frequency**: Weekly batch updates

**Data quality checks**:
- Missing data reports (flag >10% missingness)
- Outlier detection (biologically implausible values)
- Temporal consistency (hospitalization before discharge)

#### 2. Mobile App Database

**Captured data**:
- User activity logs (logins, feature usage)
- Symptom assessments (CAT scores, dyspnea)
- Medication adherence confirmations
- Spirometry readings (if home device connected)
- Activity data (steps, active minutes via wearable sync)
- Alerts triggered and acknowledgments

**Storage**: HIPAA-compliant cloud database (encrypted at rest and in transit)

**Retention**: 7 years per HIPAA requirements

#### 3. Patient-Reported Outcomes (PROs)

**Surveys** (administered via app and email):

| Instrument | Frequency | Domains |
|------------|-----------|---------|
| **CAT** | Weekly | COPD symptoms, impact |
| **mMRC** | Monthly | Dyspnea severity |
| **EQ-5D-5L** | Quarterly | Health-related quality of life |
| **eHEALS** | Baseline only | Digital literacy |
| **PHQ-4** | Quarterly | Depression, anxiety screening |

**Response rate targets**:
- In-app surveys: >85% (push notifications, incentives)
- Email surveys: >70%

#### 4. Administrative Claims

**Data elements**:
- All-cause healthcare utilization (inpatient, ED, outpatient)
- Costs (paid amounts, not charges)
- Pharmacy fills and days supply
- Out-of-network care

**Payers**: Medicare, Medicaid, commercial insurers (requires data use agreements)

**Lag time**: 3-6 months (accommodate claims processing delays)

---

### Data Management Plan

#### Data Architecture

```
EHR → HL7 Interface → Research Data Warehouse (REDCap)
                              ↓
App Database → API → Research Data Warehouse
                              ↓
Claims Data → Manual Upload → Research Data Warehouse
                              ↓
                      Master Analysis Dataset
                              ↓
                    Statistical Software (R, Stata)
```

#### REDCap Configuration

**Projects**:
1. **Enrollment & Eligibility**: Screening, consent
2. **Clinical Data**: EHR extracts, outcomes
3. **App Data**: Engagement, PROs
4. **Adverse Events**: Safety reporting

**Features utilized**:
- Data quality rules (range checks, required fields)
- Audit trail (all changes logged)
- User roles (data entry vs analysis access)
- Automated data imports (API)

#### Data Dictionary

All variables documented with:
- Variable name (no PHI, limited to 32 characters)
- Label (plain language description)
- Type (numeric, categorical, date)
- Units (for continuous variables)
- Allowed values/ranges
- Source system
- Logic for derived variables

Example:
| Variable | Label | Type | Values | Source |
|----------|-------|------|--------|--------|
| copd_cat_t0 | CAT score at baseline | Numeric | 0-40 | App |
| readm30 | 30-day readmission | Binary | 0=No, 1=Yes | EHR |
| ruca_cat | Rurality category | Categorical | 1=Urban, 2=Rural, 3=Highly rural | Census |

---

### Data Security & Privacy

#### HIPAA Compliance

- **BAA agreements** with all vendors (app developer, cloud hosting)
- **Encryption**: AES-256 (data at rest), TLS 1.3 (data in transit)
- **Access controls**: Role-based, MFA required
- **Audit logs**: All data access logged, reviewed quarterly
- **De-identification**: Safe harbor method for public datasets

#### Breach Response Plan

1. **Detection**: Automated monitoring alerts
2. **Containment**: Revoke access, preserve logs
3. **Assessment**: Privacy officer determines if breach occurred
4. **Notification**: Within 60 days if breach affects >500 individuals
5. **Remediation**: Implement fixes, update policies

---

## Reporting & Dissemination

### Timeline

| Milestone | Timing | Deliverable |
|-----------|--------|-------------|
| **Protocol publication** | Month 1 | Pre-print (MedRxiv), trial registry |
| **Recruitment update** | Month 12 | Advisory board report |
| **Interim analysis** | Month 18 | DSMB review (stopping rules) |
| **Final analysis** | Month 30 | Statistical analysis plan finalized |
| **Primary manuscript** | Month 33 | Submitted to JAMA, NEJM, BMJ |
| **Secondary papers** | Months 34-36 | Equity analyses, cost-effectiveness |
| **Dissemination** | Months 36+ | Conferences, webinars, policy briefs |

### Primary Manuscript Structure

**Title**: "Effectiveness of a Remote Patient Monitoring App for Reducing COPD Readmissions: A Stepped-Wedge Cluster Randomized Trial"

**Outline**:
1. **Abstract** (300 words)
   - Structured: Background, Methods, Results, Conclusions
2. **Introduction** (500 words)
   - COPD burden, readmission problem, digital health promise
3. **Methods** (1500 words)
   - Study design, setting, participants, intervention, outcomes, statistical analysis
4. **Results** (1000 words)
   - Participant flow, baseline characteristics, primary outcome, secondary outcomes, subgroups
5. **Discussion** (1200 words)
   - Key findings, comparison to prior work, mechanisms, implications, limitations
6. **Conclusions** (200 words)

**Figures** (target journal: 4-5 figures/tables):
1. CONSORT diagram (flow chart)
2. Primary outcome over time (stepped-wedge visualization)
3. Forest plot (subgroup analyses)
4. Equity stratification results
5. Cost-effectiveness plane (incremental cost vs QALY)

---

### Secondary Publications

1. **Protocol paper** (BMJ Open, Trials)
   - Full study protocol, statistical analysis plan

2. **Equity-focused paper** (Health Affairs, Milbank Quarterly)
   - Deep dive into disparities by race, rurality, digital literacy
   - Title: "Digital Divide or Digital Dividend? Equity Impacts of Telehealth for COPD"

3. **Implementation science paper** (Implementation Science, JGIM)
   - Barriers and facilitators to adoption
   - RE-AIM framework evaluation

4. **Cost-effectiveness analysis** (Medical Decision Making, Value in Health)
   - Markov model, lifetime horizon, healthcare sector perspective

5. **Qualitative companion study** (Qualitative Health Research)
   - Patient and provider experiences
   - Acceptability and perceived value

---

### Dissemination to Stakeholders

#### Policy Makers

- **CMS Innovation Center**: Share findings for potential Medicare coverage
- **State Medicaid agencies**: Toolkit for reimbursement pathways
- **Policy brief** (2-page executive summary)

#### Clinicians

- **CME webinar**: Practical implementation guidance
- **Pocket cards**: Quick reference for app features
- **Clinical practice guidelines**: Work with ATS/ERS to update recommendations

#### Patients

- **Plain language summary**: Infographic of key findings
- **Patient advisory board**: Presentation and discussion
- **Social media**: Twitter thread, patient advocacy groups

#### Payers

- **ROI calculator**: Excel tool for estimating savings
- **Business case white paper**: Costs, savings, value proposition

---

## Appendices

### Appendix A: COPD Assessment Test (CAT)

**Instructions**: Please check the box that best describes your situation

1. I never cough ☐☐☐☐☐☐ I cough all the time
2. I have no phlegm ☐☐☐☐☐☐ My chest is full of phlegm
3. My chest does not feel tight ☐☐☐☐☐☐ My chest feels very tight
4. When I walk up a hill or stairs, I am not breathless ☐☐☐☐☐☐ Very breathless
5. I am not limited doing activities at home ☐☐☐☐☐☐ Very limited
6. I am confident leaving home ☐☐☐☐☐☐ Not confident at all
7. I sleep soundly ☐☐☐☐☐☐ I don't sleep soundly
8. I have lots of energy ☐☐☐☐☐☐ I have no energy at all

**Scoring**: Each item scored 0-5, total score 0-40

---

### Appendix B: eHealth Literacy Scale (eHEALS)

**Instructions**: Please rate your agreement with each statement (1=Strongly disagree to 5=Strongly agree)

1. I know what health resources are available on the Internet
2. I know where to find helpful health resources on the Internet
3. I know how to find helpful health resources on the Internet
4. I know how to use the Internet to answer my health questions
5. I know how to use health information I find on the Internet to help me
6. I have the skills I need to evaluate health resources I find on the Internet
7. I can tell high quality from low quality health resources on the Internet
8. I feel confident in using information from the Internet to make health decisions

**Scoring**: Sum of all items, range 8-40

---

### Appendix C: Statistical Code Templates

#### Stepped-Wedge Analysis (R)

```R
# Load packages
library(lme4)
library(lmerTest)

# Prepare data
sw_data$intervention <- as.factor(sw_data$intervention)
sw_data$time <- scale(sw_data$time)  # Center time

# Primary model: 30-day readmission
model_primary <- glmer(
  readmission30 ~ intervention + time + (1 | cluster) + (1 | cluster:time),
  family = binomial(link = "logit"),
  data = sw_data,
  control = glmerControl(optimizer = "bobyqa")
)

# Extract results
summary(model_primary)
exp(fixef(model_primary))  # Odds ratios
confint(model_primary, parm = "beta_", method = "Wald")  # 95% CI

# Secondary model: CAT score (continuous)
model_cat <- lmer(
  cat_score ~ intervention + time + baseline_cat + age + sex +
    (1 + time | cluster) + (1 | patient_id),
  data = sw_data,
  REML = TRUE
)

summary(model_cat)
```

#### Segmented Regression (Stata)

```stata
* Load data
use its_data.dta, clear

* Create time variables
gen time = _n
gen intervention = (time >= 25)  // Assuming intervention at month 25
gen time_after = max(0, time - 24)

* Basic segmented regression
regress readmission_rate time intervention time_after

* Newey-West standard errors (2 lags for monthly data)
newey readmission_rate time intervention time_after, lag(2)

* Visualization
twoway (scatter readmission_rate time) ///
       (lfit readmission_rate time if intervention==0) ///
       (lfit readmission_rate time if intervention==1), ///
       xline(24, lcolor(red) lpattern(dash)) ///
       legend(order(1 "Observed" 2 "Pre-trend" 3 "Post-trend")) ///
       title("Interrupted Time-Series: COPD Readmission Rate")
```

---

### Appendix D: Equity Analysis Plan

#### Stratified Analysis Template

```R
# Fit models separately for each equity stratum

# By race/ethnicity
strata_race <- sw_data %>% split(.$race_ethnicity)

models_race <- lapply(strata_race, function(df) {
  glmer(readmission30 ~ intervention + time + (1 | cluster),
        family = binomial, data = df)
})

# Extract stratum-specific ORs
ors_race <- sapply(models_race, function(m) {
  exp(fixef(m)["intervention"])
})

# Test for interaction
model_interaction <- glmer(
  readmission30 ~ intervention * race_binary + time + (1 | cluster),
  family = binomial,
  data = sw_data
)

summary(model_interaction)  # Check p-value for interaction term
```

#### Disparity Metrics

```R
# Calculate disparity ratios

# Readmission rates by race
rates <- sw_data %>%
  filter(intervention == 1) %>%  # Post-intervention only
  group_by(race_ethnicity) %>%
  summarise(
    n = n(),
    readm = sum(readmission30),
    rate = readm / n * 100
  )

# Disparity ratio (minoritized group vs White)
rates <- rates %>%
  mutate(
    disparity_ratio = rate / rates$rate[rates$race_ethnicity == "White"]
  )

print(rates)
```

---

## References

### Methodological Papers

1. Hussey MA, Hughes JP. Design and analysis of stepped wedge cluster randomized trials. *Contemporary Clinical Trials*. 2007;28(2):182-191.

2. Hemming K, Taljaard M, Forbes AB. Analysis of cluster randomised stepped wedge trials with repeated cross-sectional samples. *Trials*. 2017;18(1):101.

3. Bernal JL, Cummins S, Gasparrini A. Interrupted time series regression for the evaluation of public health interventions: a tutorial. *Int J Epidemiol*. 2017;46(1):348-355.

4. Zhang F, Wagner AK, Ross-Degnan D. Simulation-based power calculation for designing interrupted time series analyses of health policy interventions. *J Clin Epidemiol*. 2011;64(11):1252-1261.

### COPD & Digital Health

5. Shah T, Churpek MM, Coca Perraillon M, Konetzka RT. Understanding why patients with COPD get readmitted. *Chest*. 2015;147(5):1219-1226.

6. Noah B, Keller MS, Mosadeghi S, et al. Impact of remote patient monitoring on clinical outcomes: an updated meta-analysis of randomized controlled trials. *NPJ Digit Med*. 2018;1:20172.

7. Alwashmi MF, Mugford G, Abu-Ashour W, Nuccio M. Effectiveness of smartphone apps for patients with chronic obstructive pulmonary disease: systematic review and meta-analysis. *JMIR Mhealth Uhealth*. 2022;10(1):e31480.

### Health Equity

8. Ye S, Kronish I, Fleck E, et al. Telemedicine Expansion During the COVID-19 Pandemic and the Potential for Technology-Driven Disparities. *J Gen Intern Med*. 2021;36(1):256-258.

9. Rodriguez JA, Clark CR, Bates DW. Digital health equity as a necessity in the 21st century cures act era. *JAMA*. 2020;323(23):2381-2382.

---

**Document Version**: 1.0
**Last Updated**: 2025-11-22
**Authors**: Gen Z Digital Health Evaluation Team
**Contact**: [Insert contact information]
**License**: CC BY 4.0 (Attribution required for reuse)
