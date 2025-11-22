{% docs __overview__ %}

# COVID-19 Surveillance dbt Project

## Project Purpose

This dbt project provides **production-grade data models** for COVID-19 public health surveillance systems. It transforms raw laboratory results, genomic sequencing data, and patient demographics into analytics-ready tables used by state Departments of Health and the Veterans Affairs (VA) National COVID Surveillance program.

## Business Context

### Stakeholders

- **Public Health Officials**: Track case trends, positivity rates, variant prevalence
- **Laboratory Directors**: Monitor test volumes, turnaround times, quality metrics
- **Epidemiologists**: Analyze demographic patterns, geographic hotspots
- **CDC/CSTE**: Receive standardized weekly lineage reports (SPHERES)
- **Clinical Teams**: Access test results for patient care

### Key Use Cases

1. **Daily Case Surveillance**
   - Test volumes and positivity rates by geography
   - Demographic analysis (age, race, ethnicity)
   - Turnaround time monitoring

2. **Variant Surveillance**
   - SARS-CoV-2 lineage tracking (Omicron sublineages, etc.)
   - Variants of Concern (VOC) early detection
   - Mutation frequency analysis

3. **CDC Reporting**
   - Weekly lineage proportions by county/state
   - SPHERES genomic surveillance submission
   - MMWR epi week calculations

4. **Quality Assurance**
   - Assay performance monitoring
   - Data completeness checks
   - Anomaly detection

## Data Sources

### Raw LIMS (Laboratory Information Management System)

**Source**: `raw_lims.laboratory_results`

- PCR test results (CT values by target: N, ORF1ab, E, S)
- Antigen test results (qualitative only)
- Specimen and test metadata
- Timestamps (collection, receipt, result, release)

**Volume**: ~50,000 tests/day

### Raw Sequencing Results

**Source**: `raw_sequencing.sequence_analysis_results`

- SARS-CoV-2 whole genome sequences (FASTA)
- Pangolin lineage assignments (Pango nomenclature)
- Nextclade classifications (clades and QC)
- Mutation profiles (JSON arrays)
- Quality metrics (coverage, depth, N content)

**Volume**: ~5,000 sequences/week

### Raw Patient Demographics

**Source**: `raw_ehr.patient_demographics`

- Patient identifiers (PAT_ID, MRN)
- Demographics (age, sex, race, ethnicity)
- Geographic data (address, county FIPS)
- Contact information (PHI - restricted access)

**Volume**: ~2M active patients

## Model Architecture

### Layers

1. **Staging** (`stg_*`)
   - Clean and standardize raw data
   - Materialized as views (no data duplication)
   - Apply data type conversions, trimming, standardization
   - Filter to valid records only

2. **Dimensions** (`dim_*`)
   - Reference tables (patients, assay rules)
   - Materialized as tables
   - Slowly changing dimensions (SCD) via snapshots

3. **Facts** (`covid_tests`, `covid_variants`)
   - Primary business process tables
   - Incremental materialization for performance
   - Partitioned by date

4. **Aggregates** (`covid_lineage_summary`)
   - Pre-aggregated reports
   - Optimized for common queries
   - Incrementally updated

5. **Snapshots** (`snap_*`)
   - Historical tracking (SCD Type 2)
   - Captures changes to assay rules, WHO classifications

## Key Business Logic

### PCR Test Interpretation

Multi-target PCR tests require complex interpretation logic:

1. **Positive (Detected)**: ≥2 targets below CT cutoff
2. **Negative (Not Detected)**: All targets above cutoff or undetermined
3. **Inconclusive**: Single weak positive (CT 35-40)

CT cutoffs vary by:
- Assay manufacturer (Cepheid, Roche, Abbott, etc.)
- Reagent lot
- Effective date (rules change over time)

**Implementation**: `dim_assay_interpretation_rules` + logic in `covid_tests.sql`

### Variant Classification

Lineage assignment uses multiple tools:
- **Pangolin**: Pango nomenclature (most granular: BA.2.86.1)
- **Nextclade**: WHO clade system (broader: 23F)
- **Custom logic**: Maps Pango → WHO variant (Alpha, Delta, Omicron, etc.)

Key mutations flagged:
- `E484K`, `N501Y`, `L452R` - Immune escape, transmissibility
- `del69-70` - S-gene dropout marker
- `P681H/R` - Furin cleavage site

**Implementation**: `covid_variants.sql` with JSON mutation parsing

### Epidemiological Weeks

CDC uses MMWR (Morbidity and Mortality Weekly Report) weeks:
- Start: Sunday
- End: Saturday (reported as week date)
- Span year boundaries (week 52/53 → week 1)

**Implementation**: `epi_week()` macro

## Data Quality

### Critical Tests (Errors)

- **Primary keys**: Accession numbers unique
- **Referential integrity**: All tests link to valid patients
- **CT value ranges**: 0-50 only
- **Date sequences**: Collection before result release
- **Required fields**: Result, assay, patient ID

### Warnings

- **Demographics >90% complete**: County FIPS, race, ethnicity
- **Lineage consistency**: Pango matches WHO classification
- **Volume anomalies**: >3 std devs from mean

### Monitoring

dbt Cloud (or Airflow) alerts on:
- Test failures (Slack, email)
- Data freshness >24h
- Positivity rate >30% (likely data issue)

## Performance

### Optimization Strategies

1. **Incremental models**: Only process new records
   - `covid_tests`: Filter on `result_released_dttm`
   - `covid_lineage_summary`: Filter on `test_event_date`

2. **Partitioning**: Daily partitions on `test_event_date`
   - Prunes 99% of data for daily queries
   - Critical for 15M+ record tables

3. **Clustering**: Geographic columns (`county_fips`, `state`)
   - Optimizes common groupings

4. **Materialization choices**:
   - Staging: Views (cheap, always fresh)
   - Dimensions: Tables (small, rarely change)
   - Facts: Incremental (large, frequent updates)

### Typical Runtimes

| Job | Frequency | Runtime |
|-----|-----------|---------|
| Full refresh | Weekly | 15 min |
| Daily incremental | Daily | 2 min |
| Hourly incremental | Hourly | 45 sec |

## Security & Compliance

### PHI/PII

- **Patient identifiers**: PAT_ID, MRN (restricted)
- **Contact info**: Phone, email (restricted)
- **Addresses**: Full address restricted, ZIP3/county OK
- **Access control**: Warehouse RBAC required

### De-identification

Public reports use:
- ZIP3 (first 3 digits) instead of ZIP5
- County FIPS instead of street addresses
- Age groups instead of date of birth

### Audit Trail

- **SCD snapshots**: Historical tracking of all rule changes
- **dbt metadata**: Full lineage of transformations
- **Query logs**: `query-comment` tags all queries
- **Git history**: Code changes tracked

### Compliance

- **HIPAA**: Encryption, access logs, minimum necessary
- **FDA**: Audit-ready (snapshots, validation tests)
- **CLIA**: Test interpretation logic documented

## Maintenance

### Weekly Tasks

1. **Review test failures**: Fix data issues or adjust tests
2. **Check data freshness**: Ensure sources updating
3. **Monitor volumes**: Detect processing delays

### Monthly Tasks

1. **Update assay rules**: New EUAs, changed cutoffs
2. **Update WHO variants**: New VOC/VOI designations
3. **Review performance**: Optimize slow models

### As Needed

1. **Schema changes**: Add new columns, tests
2. **New assays**: Add to `assay_rules.csv`
3. **Bug fixes**: Address edge cases

## Getting Help

- **dbt docs**: `dbt docs serve`
- **Model code**: Check SQL and comments
- **Schema tests**: Review `schema.yml` files
- **Data team**: Contact for access, issues

---

*This overview is automatically generated in dbt docs. For more details, see individual model documentation.*

{% enddocs %}
