# NVDRS NLP Pipeline

**Transform Tragedy into Prevention: Converting Coroner Narratives into Actionable Suicide Risk Factors**

## 🎯 Project Overview

The **NVDRS NLP Pipeline** is a production-ready system that analyzes unstructured text from the [National Violent Death Reporting System](https://www.cdc.gov/violenceprevention/datasources/nvdrs/) to extract structured risk factors for suicide prevention research.

### The Challenge

NVDRS contains rich qualitative data in coroner/medical examiner narratives and law enforcement reports, but this unstructured text is difficult to analyze at scale. Each narrative is a unique, free-text description of circumstances surrounding a violent death.

### The Solution

This pipeline uses **Spark NLP** + **Hugging Face transformers** to:
1. Process thousands of narratives in parallel (Spark distributed computing)
2. Extract medical entities (BioBERT/ClinicalBERT)
3. Identify risk factors (substance use, mental health, social stressors)
4. Generate structured data for statistical analysis

### The Impact

Enables researchers and public health officials to:
- Identify high-prevalence risk factors across populations
- Build predictive models for suicide risk
- Develop evidence-based prevention strategies
- Track trends over time and across geographic regions

---

## 🏗️ Architecture

### The "Narrative-to-Data" Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: Ingestion & PII Redaction                             │
│  • Load NVDRS CSV/Parquet data into Spark DataFrame            │
│  • Redact names, dates, SSNs (HIPAA compliance)                │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: NLP Pipeline Construction (Spark NLP)                 │
│  1. Document Assembler - Prepare text                          │
│  2. Sentence Detector - Split into sentences                   │
│  3. Tokenizer - Break into tokens                              │
│  4. Normalizer - Clean text                                    │
│  5. Clinical NER (BioBERT) - Extract medical entities          │
│  6. NER Converter - Format results                             │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3: Entity Extraction & Classification (Hugging Face)     │
│  • Named Entity Recognition (medications, conditions)          │
│  • Intent Classification (suicide vs accidental vs undetermined)│
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 4: Risk Factor Transformation                            │
│  • Substance Use: Opioids, alcohol, overdose indicators        │
│  • Mental Health: Depression, anxiety, suicide history         │
│  • Social Stressors: Financial, relationship, legal, health    │
│  • Risk Scoring: Aggregate 0-1 risk score                      │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 5: Structured Output                                     │
│  • Parquet files (big data analytics)                          │
│  • Excel spreadsheets (human review)                           │
│  • Summary reports (executive briefings)                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Java 8 or 11** (required for Apache Spark)
   ```bash
   # Verify Java installation
   java -version

   # Install if needed:
   # Ubuntu/Debian: sudo apt-get install openjdk-11-jdk
   # macOS: brew install openjdk@11
   ```

3. **8GB+ RAM** (16GB recommended for large datasets)

### Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd GenZ

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements-nvdrs.txt

# 4. Verify installation
python -c "import sparknlp; print('Spark NLP version:', sparknlp.version())"
```

### Run the Example

```bash
# Run the complete demonstration
python examples/nvdrs_example.py
```

This will:
- Load sample NVDRS data (3 synthetic records)
- Apply PII redaction
- Extract risk factors using Clinical NER
- Generate structured output files
- Display summary statistics

**Expected Output:**
```
✓ Loaded 3 NVDRS records
✓ Processing complete!
  Opioid Mentions: 2 (66.7%)
  Depression Mentions: 3 (100.0%)
  Financial Crisis: 2 (66.7%)
  Average Risk Score: 0.625
✓ Output files created in output/nvdrs/
```

---

## 📊 Example: From Narrative to Data

### Input (Unstructured Text)

```text
CME_Narrative: "Decedent was a 45-year-old male found unresponsive
at home. Toxicology report revealed presence of fentanyl and
oxycodone. Medical history included chronic pain and depression.
Family reported recent job loss and financial difficulties."

LE_Narrative: "Officers responded to a welfare check. Found eviction
notice on the door. Neighbor reported decedent had mentioned
financial problems. No signs of foul play."
```

### Output (Structured Data)

| Field | Value |
|-------|-------|
| `record_id` | NVDRS_2024_001 |
| `opioid_mentioned` | ✓ True |
| `opioid_types` | ['fentanyl', 'oxycodone'] |
| `depression` | ✓ True |
| `financial_crisis` | ✓ True |
| `risk_score` | 0.75 |
| `predicted_intent` | suicide |
| `intent_confidence` | 0.82 |

### What Just Happened?

The pipeline:
1. **Detected** mentions of specific opioids (fentanyl, oxycodone)
2. **Identified** mental health indicator (depression)
3. **Recognized** financial stressor (job loss, eviction)
4. **Calculated** aggregate risk score (0.75 = high risk)
5. **Classified** intent as suicide (82% confidence)

All of this from free-text narratives!

---

## 🔬 Technical Details

### Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Compute** | Apache Spark | Distributed processing for big data |
| **NLP** | Spark NLP | Industrial-strength NLP pipelines |
| **Models** | Hugging Face Transformers | BioBERT/ClinicalBERT for medical text |
| **Data Models** | Pydantic | Type-safe data structures |
| **Output** | Parquet, Excel, Markdown | Multi-format analytics |

### Models Used

1. **Clinical NER**: `bert_token_classifier_ner_clinical`
   - Pre-trained on medical text
   - Extracts: PROBLEM, TREATMENT, TEST, etc.
   - Source: Spark NLP Model Hub

2. **Classification**: `emilyalsentzer/Bio_ClinicalBERT`
   - BioBERT variant trained on clinical notes
   - Better than generic BERT for medical context
   - Source: Hugging Face Model Hub

### Risk Factor Categories

The pipeline extracts **7 categories** of risk factors:

1. **Opioids** (13 keywords): fentanyl, oxycodone, heroin, morphine, etc.
2. **Other Substances** (14 keywords): alcohol, cocaine, methamphetamine, etc.
3. **Mental Health** (16 keywords): depression, anxiety, PTSD, suicide history, etc.
4. **Financial Stressors** (14 keywords): eviction, bankruptcy, job loss, debt, etc.
5. **Relationship Issues** (9 keywords): divorce, domestic violence, custody, etc.
6. **Legal Problems** (9 keywords): arrest, incarceration, criminal charges, etc.
7. **Health Crises** (8 keywords): terminal illness, chronic pain, cancer, etc.

See [`nvdrs_pipeline/config.py`](nvdrs_pipeline/config.py) for complete keyword lists.

---

## 📁 Project Structure

```
GenZ/
├── nvdrs_pipeline/              # Main pipeline module
│   ├── __init__.py             # Package initialization
│   ├── pipeline.py             # Core NLP pipeline (Spark NLP + HF)
│   ├── config.py               # Configuration & keywords
│   ├── models.py               # Pydantic data models
│   ├── pii_redaction.py        # HIPAA-compliant PII removal
│   └── utils.py                # Helper functions
│
├── examples/
│   ├── nvdrs_example.py        # Complete demonstration
│   └── nvdrs_data/             # Sample data (gitignored)
│
├── tests/
│   └── test_nvdrs/             # Unit & integration tests
│       ├── test_pipeline.py
│       ├── test_pii_redaction.py
│       └── test_models.py
│
├── output/nvdrs/               # Generated outputs (gitignored)
│   ├── nvdrs_processed.parquet # Structured data
│   ├── nvdrs_analysis.xlsx     # Excel export
│   └── nvdrs_summary_report.md # Summary report
│
├── requirements-nvdrs.txt      # Dependencies
└── README_NVDRS.md            # This file
```

---

## 🔐 Privacy & Compliance

### HIPAA Compliance

The pipeline includes **robust PII redaction**:

- ✓ **Names**: Mr./Mrs./Dr. + name patterns
- ✓ **Dates**: All date formats
- ✓ **SSNs**: 123-45-6789 patterns
- ✓ **Phone Numbers**: Various formats
- ✓ **Email Addresses**: Standard email patterns
- ✓ **Street Addresses**: Numeric + street type

**Ages are preserved** (important for epidemiological analysis).

### Configuration

```python
from nvdrs_pipeline import NVDRSPipeline

pipeline = NVDRSPipeline(
    enable_pii_redaction=True  # Default: True
)
```

See [`nvdrs_pipeline/pii_redaction.py`](nvdrs_pipeline/pii_redaction.py) for redaction logic.

---

## 💻 Usage

### Basic Usage

```python
from nvdrs_pipeline import NVDRSPipeline
from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder \
    .appName("NVDRS_Analysis") \
    .master("local[*]") \
    .getOrCreate()

# Load your NVDRS data
df = spark.read.csv("nvdrs_data.csv", header=True)

# Initialize pipeline
pipeline = NVDRSPipeline(spark=spark)

# Process narratives
results = pipeline.process_narratives(
    df,
    narrative_columns=['CME_Narrative', 'LE_Narrative']
)

# Save results
pipeline.save_results(results, "output.parquet", format="parquet")

# Generate summary
print(f"Processed {results.total_records} records")
print(f"High risk cases: {results.summary_statistics['risk_scores']['high_risk_count']}")
```

### Advanced: Custom Keywords

```python
from nvdrs_pipeline.config import NVDRSConfig

# Extend opioid keyword list
config = NVDRSConfig()
config.OPIOID_KEYWORDS.extend(['carfentanil', 'sufentanil'])

# Use custom config
pipeline = NVDRSPipeline(config=config)
```

### Running on Databricks

```python
# In Databricks notebook

# Install Spark NLP
%pip install spark-nlp

from nvdrs_pipeline import NVDRSPipeline

# Use existing Databricks Spark session
pipeline = NVDRSPipeline(spark=spark)

# Load from Delta table
df = spark.table("nvdrs_raw_data")

# Process
results = pipeline.process_narratives(df)

# Save to Delta
pipeline.save_results(results, "/mnt/nvdrs/processed", format="parquet")
```

---

## 📈 Output Formats

### 1. Parquet (For Big Data Analytics)

**Best for**: Spark, Hive, Presto, Athena

```python
pipeline.save_results(results, "output.parquet", format="parquet")
```

Use in analytics:
```python
# Load in Spark
df = spark.read.parquet("output.parquet")
df.groupBy("state").agg(avg("risk_score")).show()

# Load in Pandas
import pandas as pd
df = pd.read_parquet("output.parquet")
df['risk_score'].describe()
```

### 2. Excel (For Human Review)

**Best for**: Manual review, presentations

```python
from nvdrs_pipeline.utils import export_to_excel
export_to_excel(results.records, "analysis.xlsx")
```

Creates formatted spreadsheet with:
- Color-coded headers
- Auto-sized columns
- Boolean flags for risk factors
- Risk score column

### 3. Markdown Summary (For Reports)

**Best for**: Documentation, executive briefings

```python
from nvdrs_pipeline.utils import generate_summary_report
generate_summary_report(results.records, "summary.md")
```

Includes:
- Executive summary
- Risk factor prevalence statistics
- Methodology description
- Recommendations

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/test_nvdrs/

# With coverage
pytest --cov=nvdrs_pipeline tests/test_nvdrs/

# Specific test
pytest tests/test_nvdrs/test_pii_redaction.py -v
```

---

## 🎓 Use Cases

### 1. Public Health Research

**Goal**: Identify risk factor prevalence across populations

```python
# Calculate opioid involvement rate
opioid_rate = sum(
    1 for r in results.records
    if r.risk_factors.substance_use.opioid_mentioned
) / len(results.records)

print(f"Opioid involvement: {opioid_rate:.1%}")
```

### 2. Predictive Modeling

**Goal**: Build ML models for suicide risk prediction

```python
# Extract features for ML
import pandas as pd

features = pd.DataFrame([
    {
        'opioid': r.risk_factors.substance_use.opioid_mentioned,
        'depression': r.risk_factors.mental_health.depression_mentioned,
        'financial_crisis': r.risk_factors.social_stressors.financial_crisis,
        # ... more features
        'target': r.predicted_intent == 'suicide'
    }
    for r in results.records
])

# Train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(features.drop('target', axis=1), features['target'])
```

### 3. Geographic Analysis

**Goal**: Identify regional patterns

```python
# Group by state
from collections import defaultdict

state_stats = defaultdict(list)
for record in results.records:
    state_stats[record.state].append(record.risk_factors.risk_score)

# Calculate averages
for state, scores in state_stats.items():
    avg_score = sum(scores) / len(scores)
    print(f"{state}: avg risk = {avg_score:.3f}")
```

### 4. Trend Analysis

**Goal**: Monitor changes over time

```python
# Group by year
import pandas as pd

df = pd.DataFrame([
    {
        'year': r.year,
        'opioid': r.risk_factors.substance_use.opioid_mentioned,
        'risk_score': r.risk_factors.risk_score
    }
    for r in results.records
])

# Calculate yearly trends
yearly = df.groupby('year').agg({
    'opioid': 'mean',
    'risk_score': 'mean'
})

print(yearly)
```

---

## 🔧 Configuration

All configuration is in [`nvdrs_pipeline/config.py`](nvdrs_pipeline/config.py).

### Environment Variables

Create a `.env` file:

```bash
# Spark Configuration
SPARK_MASTER=local[*]
SPARK_DRIVER_MEMORY=8g
SPARK_EXECUTOR_MEMORY=8g

# NLP Models
NER_MODEL=bert_token_classifier_ner_clinical
HF_CLASSIFIER_MODEL=emilyalsentzer/Bio_ClinicalBERT

# PII Redaction
REDACT_PII=True
REDACT_NAMES=True
REDACT_DATES=True

# Output
OUTPUT_FORMAT=parquet
OUTPUT_DIR=./output/nvdrs
```

### Customizing Risk Keywords

Edit `NVDRSConfig` class:

```python
class NVDRSConfig:
    # Add your keywords
    OPIOID_KEYWORDS = [
        "fentanyl", "oxycodone", "heroin",
        # Add more...
    ]

    # New category
    VETERAN_KEYWORDS = [
        "veteran", "ptsd", "combat", "deployment"
    ]
```

---

## 🤝 Contributing

We welcome contributions! Areas for improvement:

1. **Additional Risk Categories**: Veteran status, housing instability, etc.
2. **Better Intent Classification**: Fine-tune transformer for suicide vs accidental
3. **Multilingual Support**: Spanish, Chinese for diverse populations
4. **Real-time Processing**: Streaming pipeline for live data
5. **Dashboard**: Interactive visualization of results

See `CONTRIBUTING.md` for guidelines.

---

## 📚 References

### NVDRS
- [CDC NVDRS Overview](https://www.cdc.gov/violenceprevention/datasources/nvdrs/)
- [NVDRS Data Dictionary](https://www.cdc.gov/nvdrs/resources/datadictionary.html)

### Spark NLP
- [Spark NLP Documentation](https://nlp.johnsnowlabs.com/)
- [Clinical NER Models](https://nlp.johnsnowlabs.com/models?task=Named+Entity+Recognition&edition=Healthcare)

### Hugging Face
- [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)

### Research
- Stone DM, et al. (2018). "Vital Signs: Trends in State Suicide Rates — United States, 1999–2016 and Circumstances Contributing to Suicide — 27 States, 2015." MMWR.
- Rossen LM, et al. (2019). "Drug Poisoning Mortality: United States, 1999-2017." National Center for Health Statistics.

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## ⚠️ Disclaimer

This pipeline is designed for **research purposes** to support public health initiatives. It should not be used as the sole basis for clinical decisions or individual risk assessment. Always consult qualified mental health professionals for suicide risk evaluation.

---

## 📧 Contact

For questions, issues, or collaboration:

- **GitHub Issues**: [Report a bug or request a feature](https://github.com/your-org/GenZ/issues)
- **Email**: your-email@example.com
- **Documentation**: See `docs/` directory

---

## 🌟 Acknowledgments

This pipeline was developed to support suicide prevention research using data from the National Violent Death Reporting System (NVDRS), a collaboration between CDC and state health departments.

**Technology Stack**:
- [Apache Spark](https://spark.apache.org/) - Distributed computing
- [Spark NLP by John Snow Labs](https://www.johnsnowlabs.com/spark-nlp/) - Clinical NLP
- [Hugging Face](https://huggingface.co/) - Transformer models
- [BioBERT](https://github.com/dmis-lab/biobert) - Biomedical language representation

---

**Built with ❤️ by the Gen Z Agent Team**

*Transforming tragedy into prevention, one narrative at a time.*
