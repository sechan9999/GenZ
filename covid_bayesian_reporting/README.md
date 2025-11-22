# COVID-19 Bayesian Nowcasting and Data Imputation System

A comprehensive statistical framework for addressing three critical challenges in COVID-19 pandemic data reporting:

1. **Severe Reporting Lags (14-21 days)** → Bayesian Hierarchical Nowcasting
2. **30-40% Missing Race/Ethnicity Data** → MICE with Census Tract Proxies
3. **Inconsistent Lab Positivity Definitions** → Bayesian Standardization

## 🎯 Overview

During the COVID-19 pandemic response, public health agencies faced severe data quality challenges that hindered real-time decision making:

- **Reporting lags**: Cases were reported 14-21 days after they occurred, making current situation assessment difficult
- **Missing demographics**: 30-40% of cases lacked race/ethnicity information, obscuring health equity issues
- **Inconsistent testing**: Labs used different Ct thresholds (Quest: 37, LabCorp: 35) and test types, making positivity rates incomparable

This system addresses all three challenges using state-of-the-art Bayesian statistical methods, similar to approaches used by the Delphi COVIDcast project at CMU.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    COVID-19 Data Sources                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Case Reports │  │ Patient Demo │  │ Lab Tests    │         │
│  │ (with lags)  │  │ (30% missing)│  │ (inconsist.) │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
└─────────┼──────────────────┼──────────────────┼─────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Bayesian      │  │  MICE Imputation│  │   Positivity    │
│  Nowcasting     │  │  with Census    │  │ Standardization │
│                 │  │  Tract Proxies  │  │                 │
│ • Hierarchical  │  │ • 5 imputations │  │ • Lab-specific  │
│ • AR(1) temporal│  │ • Chained eqs   │  │   adjustments   │
│ • Gamma delays  │  │ • Diagnostics   │  │ • Latent class  │
│ • 95% CI        │  │ • Pooling       │  │ • Uncertainty   │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Integrated Reporting                           │
│  • Nowcast estimates with uncertainty                           │
│  • Multiple imputed datasets                                    │
│  • Standardized positivity rates                                │
│  • Comprehensive diagnostics                                    │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
cd GenZ/covid_bayesian_reporting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment (optional - for Census API)
cp .env.example .env
# Edit .env and add your CENSUS_API_KEY
```

### Run Demo

```bash
# Run with synthetic demo data
python main.py --demo

# Outputs will be saved to output/ directory
ls output/
# nowcast_estimates.csv
# nowcast_plot.png
# imputed_dataset_1.csv ... imputed_dataset_5.csv
# mice_diagnostics.csv
# standardized_positivity.csv
# positivity_comparison.png
# report.md
```

### Run with Real Data

```bash
python main.py \
  --cases data/case_reports.csv \
  --patients data/patient_demographics.csv \
  --tests data/lab_test_results.csv \
  --output results/2024-01-15/
```

## 📊 Data Requirements

### 1. Case Reports (for Nowcasting)

CSV with columns:
- `event_date`: Date when case occurred (test date)
- `report_date`: Date when case was reported
- `count`: Number of cases
- Optional: `state`, `county`, `facility` (for hierarchical modeling)

Example:
```csv
event_date,report_date,count,state,county
2024-01-01,2024-01-08,45,CA,Los Angeles
2024-01-01,2024-01-15,12,CA,Los Angeles
2024-01-02,2024-01-09,52,CA,Los Angeles
```

### 2. Patient Demographics (for MICE Imputation)

CSV with columns:
- `patient_id`: Unique identifier
- `race_ethnicity`: Race/ethnicity (can have missing values)
- `age`: Age in years
- `gender`: Gender (M/F)
- `zip_code`: ZIP code (for census tract linkage)
- Optional: `facility_type`, `admission_date`

Example:
```csv
patient_id,race_ethnicity,age,gender,zip_code,facility_type
P001,white_non_hispanic,45,F,90210,Hospital
P002,,52,M,90211,Clinic
P003,black_non_hispanic,38,F,90212,Hospital
P004,,61,M,90210,Testing Site
```

Note: Empty `race_ethnicity` values will be imputed.

### 3. Lab Test Results (for Positivity Standardization)

CSV with columns:
- `test_id`: Unique identifier
- `lab_name`: Lab that performed test (Quest, LabCorp, etc.)
- `test_type`: Type of test (pcr, antigen)
- `result`: Test result (positive, negative)
- Optional: `ct_value` (for PCR tests)

Example:
```csv
test_id,lab_name,test_type,result,ct_value
T001,Quest,pcr,positive,28.5
T002,LabCorp,pcr,negative,
T003,Local_PCR,pcr,positive,36.2
T004,Rapid_Antigen,antigen,negative,
```

## 📖 Module Documentation

### 1. Bayesian Nowcasting (`nowcasting.py`)

Addresses reporting lags using hierarchical Bayesian model.

**Key Features:**
- Hierarchical structure (state → county → facility)
- Temporal correlation via AR(1) process
- Gamma distribution for reporting delays
- Weekend/holiday effects
- Negative Binomial likelihood (overdispersion)
- MCMC inference via PyMC

**Usage:**
```python
from nowcasting import BayesianNowcaster

# Initialize
nowcaster = BayesianNowcaster(case_data)

# Build and fit model
nowcaster.build_model()
nowcaster.fit(draws=2000, tune=1000, chains=4)

# Get nowcast estimates
nowcast = nowcaster.get_nowcast()
print(nowcast)
#          date  nowcast_median  ci_lower_95  ci_upper_95
# 2024-01-14            245.3        198.2        302.1
# 2024-01-15            263.8        215.4        321.5

# Diagnostics
diagnostics = nowcaster.diagnose()
print(f"Max R-hat: {diagnostics['rhat_max']}")  # Should be < 1.01

# Plot
nowcaster.plot_nowcast(save_path="nowcast.png")
```

**Model Specification:**

```
True counts:        log(λ_t) ~ AR(1) process
Reporting delay:    D ~ Gamma(α, β)
Weekend effect:     β_weekend ~ N(1.5, 0.3)
Observed counts:    Y_t ~ NegativeBinomial(λ_t * P(reported), φ)
```

Where:
- `λ_t`: True case count at time t
- `D`: Delay in days
- `P(reported)`: Fraction of cases reported so far
- `φ`: Overdispersion parameter

### 2. MICE Imputation (`imputation.py`)

Handles 30-40% missing race/ethnicity data using multiple imputation.

**Key Features:**
- Multiple Imputation by Chained Equations (MICE)
- Census tract demographic proxies (from US Census API)
- 5 imputed datasets for uncertainty quantification
- Comprehensive diagnostics (FMI, λ, KL divergence)
- Rubin's rules for pooling estimates

**Usage:**
```python
from imputation import CensusTractProxy, MICEImputer, validate_imputation

# Fetch census data
census_proxy = CensusTractProxy(api_key="YOUR_KEY")
census_data = census_proxy.fetch_census_data(state_fips='06')  # California

# Link patients to census tracts
patient_data = census_proxy.link_to_census_tract(patient_data)

# Run MICE
imputer = MICEImputer(n_imputations=5, max_iter=10)
imputed_datasets = imputer.fit_transform(
    patient_data,
    census_data,
    target_col='race_ethnicity'
)

# Check quality
diagnostics = imputer.get_diagnostics()
print(f"FMI: {diagnostics['fmi']:.3f}")  # Should be < 0.3

# Validate
validation = validate_imputation(patient_data, imputed_datasets)
print(f"KL divergence: {validation['kl_divergence']:.4f}")

# Pool estimates across imputations
estimates = [df['age'].mean() for df in imputed_datasets]
std_errors = [df['age'].std() / np.sqrt(len(df)) for df in imputed_datasets]
pooled_est, pooled_se, ci_width = imputer.pool_estimates(estimates, std_errors)
```

**Census Proxy Variables:**
- `pct_white`, `pct_black`, `pct_hispanic`, `pct_asian`
- `median_income`, `poverty_rate`
- `pct_bachelors`, `population_density`

**Quality Metrics:**
- **FMI (Fraction of Missing Information)**: Should be < 0.3
- **λ (Relative Increase in Variance)**: Between/within variance ratio
- **KL Divergence**: Distribution similarity (lower is better)

### 3. Positivity Standardization (`positivity_standardization.py`)

Standardizes positivity rates across labs with different protocols.

**Key Features:**
- Adjusts for different PCR Ct thresholds
- Accounts for antigen test sensitivity differences
- Bayesian latent class model for true prevalence
- Lab-specific sensitivity/specificity parameters
- Standardization to CDC reference definition

**Usage:**
```python
from positivity_standardization import PositivityStandardizer

# Initialize
standardizer = PositivityStandardizer()

# Standardize positivity rates
standardized_rates = standardizer.standardize_positivity(
    test_data,
    lab_col='lab_name',
    test_type_col='test_type',
    result_col='result',
    ct_value_col='ct_value'
)

print(standardized_rates)
#   lab_name  n_total  observed_positivity  standardized_positivity  adjustment_factor
#   Quest       1200            0.095                  0.102               1.074
#   LabCorp     1500            0.088                  0.102               1.159

# Plot comparison
standardizer.plot_comparison(save_path="positivity_comparison.png")
```

**Lab-Specific Adjustments:**

| Lab | Ct Threshold | Adjustment Direction |
|-----|--------------|---------------------|
| Quest | 37 | Reference (CDC standard) |
| LabCorp | 35 | Upward (more conservative threshold) |
| Local Health | 40 | Downward (more permissive threshold) |

**Antigen Test Sensitivity:**

| Test | Sensitivity | Specificity |
|------|-------------|-------------|
| BinaxNOW | 85% | 99% |
| Sofia | 80% | 99% |
| BD Veritor | 84% | 99% |
| Generic | 75% | 99% |

## 🔬 Statistical Methods

### Bayesian Hierarchical Nowcasting

**Model Structure:**
```
Level 1 (State):     β_state ~ N(μ, τ_state)
Level 2 (County):    β_county ~ N(β_state, τ_county)
Level 3 (Facility):  β_facility ~ N(β_county, τ_facility)

Temporal:            λ_t = λ_{t-1}^ρ * exp(ε_t),  ε_t ~ N(0, σ²)

Reporting:           D ~ Gamma(α, β)
                     P(reported by day d) = GammaCDF(d; α, β)

Observation:         Y_t ~ NegBin(λ_t * P(reported), φ)
```

**Inference:**
- MCMC sampling via NUTS (No-U-Turn Sampler)
- 4 chains × 2000 draws = 8000 posterior samples
- R-hat convergence diagnostic (< 1.01)
- Effective sample size (ESS > 400)

### MICE Imputation

**Algorithm:**
1. Initialize missing values with simple imputation (mean/mode)
2. For iteration = 1 to max_iter:
   - For each variable with missing values:
     - Regress on all other variables (observed + imputed)
     - Predict missing values using fitted model
     - Update imputed values
3. Repeat to generate M imputed datasets (M = 5)

**Pooling (Rubin's Rules):**
```
Q̄ = (1/M) Σ Q_m                     # Pooled estimate
W = (1/M) Σ SE_m²                    # Within-imputation variance
B = (1/(M-1)) Σ (Q_m - Q̄)²         # Between-imputation variance
T = W + (1 + 1/M) B                 # Total variance
SE_pooled = √T
```

### Bayesian Latent Class Model

**For Positivity Standardization:**
```
True infection:      I_i ~ Bernoulli(π)       # Latent
Test result (PCR):   T_i ~ Bernoulli(I_i * Se + (1-I_i) * (1-Sp))
Test result (Ag):    T_i ~ Bernoulli(I_i * Se_ag + (1-I_i) * (1-Sp_ag))

Priors:
  π ~ Beta(1, 9)                # Prevalence ~10%
  Se_pcr ~ Beta(19, 1)          # PCR sensitivity ~95%
  Sp_pcr ~ Beta(199, 1)         # PCR specificity ~99.5%
  Se_ag ~ Beta(8, 2)            # Antigen sensitivity ~80%
  Sp_ag ~ Beta(99, 1)           # Antigen specificity ~99%
```

## 📈 Validation & Diagnostics

### Nowcasting Validation

**Convergence Diagnostics:**
- **R-hat (Gelman-Rubin)**: < 1.01 (good), < 1.05 (acceptable)
- **Effective Sample Size (ESS)**: > 400 (good), > 100 (acceptable)
- **Divergences**: 0 (ideal), < 1% of samples (acceptable)
- **Bayesian Fraction of Missing Information (BFMI)**: > 0.3

**Nowcast Quality:**
- **Coverage**: 95% credible intervals should contain ~95% of true values
- **Sharpness**: Narrow credible intervals indicate precision
- **Calibration**: Compare to eventual complete data

### MICE Imputation Validation

**Quality Metrics:**
- **FMI (Fraction of Missing Information)**: < 0.3 (good), < 0.5 (acceptable)
- **λ (Relative Variance Increase)**: Measure of between-imputation variability
- **Convergence**: < 1% change in parameter estimates across iterations

**Distribution Checks:**
- **KL Divergence**: Observed vs imputed distribution similarity (< 0.1 is good)
- **Chi-square test**: Test for distributional differences
- **Visual**: Compare observed and imputed value distributions

### Positivity Standardization Validation

**Model Diagnostics:**
- **Posterior Predictive Checks**: Simulated data should match observed
- **Sensitivity Analysis**: Vary priors to check robustness
- **External Validation**: Compare to gold-standard PCR prevalence surveys

## 🛠️ Configuration

Edit `config.py` to customize model parameters:

```python
# Nowcasting
Config.MAX_REPORTING_LAG_DAYS = 21
Config.NOWCAST_HORIZON_DAYS = 14
Config.NOWCAST_MODEL_CONFIG['mcmc_draws'] = 2000

# MICE
Config.MICE_CONFIG['n_imputations'] = 5
Config.MICE_CONFIG['max_iterations'] = 10

# Positivity
Config.POSITIVITY_STANDARDIZATION['lab_definitions']['pcr_ct_threshold']['quest'] = 37
```

## 📚 References

### Scientific Literature

1. **Nowcasting:**
   - McGough et al. (2020). "Nowcasting by Bayesian Smoothing: A flexible, generalizable model for real-time epidemic tracking." *PLoS Computational Biology*
   - Delphi COVIDcast: https://delphi.cmu.edu/covidcast/
   - Günther et al. (2021). "Nowcasting the COVID-19 pandemic in Bavaria." *Biometrical Journal*

2. **MICE Imputation:**
   - van Buuren & Groothuis-Oudshoorn (2011). "mice: Multivariate Imputation by Chained Equations in R." *Journal of Statistical Software*
   - Little & Rubin (2019). "Statistical Analysis with Missing Data" (3rd ed.)
   - Rubin (1987). "Multiple Imputation for Nonresponse in Surveys"

3. **Latent Class Models:**
   - Branscum et al. (2005). "Estimation of diagnostic-test sensitivity and specificity through Bayesian modeling." *Preventive Veterinary Medicine*
   - Dendukuri & Joseph (2001). "Bayesian approaches to modeling the conditional dependence between multiple diagnostic tests." *Biometrics*

### Software & Tools

- **PyMC**: https://www.pymc.io/
- **ArviZ**: https://python.arviz.org/
- **scikit-learn MICE**: https://scikit-learn.org/stable/modules/impute.html
- **US Census API**: https://www.census.gov/data/developers.html

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is part of the GenZ Agent repository. See main repository LICENSE for details.

## 🙏 Acknowledgments

This implementation draws inspiration from:
- **Delphi Research Group** (CMU) for COVIDcast nowcasting methodology
- **CDC COVID-19 Response Team** for public health surveillance best practices
- **US Census Bureau** for demographic data infrastructure

## 📞 Support

For questions or issues:
- Open an issue in the GitHub repository
- Contact: [sechan9999/GenZ]

---

**Built with ❤️ for public health**

*Last Updated: 2025-11-22*
