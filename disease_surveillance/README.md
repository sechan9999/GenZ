# Disease Hotspot Detection System

**Real-time multi-stream surveillance for emerging disease outbreak detection**

This system combines multiple data streams (syndromic surveillance, wastewater monitoring, search trends, and mobility data) to identify emerging disease hotspots using advanced temporal and spatial analytics.

## 🎯 Key Features

### Multi-Stream Data Integration
- **ESSENCE Syndromic Surveillance**: Primary clinical signal from emergency department visits
- **Wastewater Viral Loads**: Early warning indicator (typically 5-7 days ahead of clinical)
- **Google/Search Trends**: Population-level health-seeking behavior
- **Mobility Data**: Movement patterns correlating with disease spread

### Advanced Analytics
- **STL Decomposition**: Seasonal-Trend decomposition using LOESS for temporal pattern extraction
- **Modified Z-Score Detection**: Robust anomaly detection using Median Absolute Deviation (MAD)
- **SaTScan Integration**: Industry-standard spatiotemporal cluster detection
- **Cross-Validation**: Multi-signal agreement scoring for high-confidence alerts

### Intelligent Alerting
- **Severity Assessment**: Critical, High, Moderate, Low based on multiple factors
- **Trend Analysis**: Increasing, stable, or decreasing outbreak trajectories
- **Actionable Recommendations**: Evidence-based public health response actions

## 📋 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Ingestion Layer                      │
│  ┌──────────┐  ┌───────────┐  ┌────────┐  ┌──────────┐    │
│  │ ESSENCE  │  │Wastewater │  │ Search │  │ Mobility │    │
│  │   Data   │  │   Data    │  │ Trends │  │   Data   │    │
│  └────┬─────┘  └─────┬─────┘  └───┬────┘  └────┬─────┘    │
└───────┼──────────────┼────────────┼────────────┼───────────┘
        │              │            │            │
        └──────────────┴────────────┴────────────┘
                       │
        ┌──────────────▼───────────────┐
        │  Temporal Anomaly Detection  │
        │  • STL Decomposition         │
        │  • Modified Z-Score          │
        │  • Per-stream thresholds     │
        └──────────────┬───────────────┘
                       │
        ┌──────────────▼───────────────┐
        │ Spatiotemporal Clustering    │
        │  • SaTScan (optional)        │
        │  • DBSCAN (fallback)         │
        │  • Geographic clustering     │
        └──────────────┬───────────────┘
                       │
        ┌──────────────▼───────────────┐
        │   Signal Cross-Validation    │
        │  • Temporal alignment        │
        │  • Spatial alignment         │
        │  • Weighted agreement score  │
        └──────────────┬───────────────┘
                       │
        ┌──────────────▼───────────────┐
        │     Hotspot Identification   │
        │  • Severity assessment       │
        │  • Trend analysis            │
        │  • Risk quantification       │
        └──────────────┬───────────────┘
                       │
        ┌──────────────▼───────────────┐
        │  Alerting & Reporting        │
        │  • JSON reports              │
        │  • Markdown summaries        │
        │  • Email/Slack notifications │
        └──────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Navigate to disease surveillance module
cd disease_surveillance

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from disease_surveillance import DiseaseHotspotPipeline
from pathlib import Path
from datetime import datetime, timedelta

# Initialize pipeline
pipeline = DiseaseHotspotPipeline(
    stl_seasonal_period=7,           # Weekly seasonality
    modified_zscore_threshold=3.5,   # Anomaly threshold
    min_observations=28,             # Minimum 4 weeks data
    use_satscan=False,               # Use simplified clustering
    spatial_radius_km=50,            # Cluster radius
    min_confirming_signals=2,        # Minimum signals to confirm
    essence_weight=1.0,              # Primary signal weight
    wastewater_weight=0.9,           # Early warning weight
    search_trends_weight=0.6,        # Behavioral signal weight
    mobility_weight=0.5,             # Supporting signal weight
    output_dir=Path("./output")
)

# Run detection
report = pipeline.run(
    essence_path=Path("data/essence.csv"),
    wastewater_path=Path("data/wastewater.csv"),
    search_trends_path=Path("data/search_trends.csv"),
    mobility_path=Path("data/mobility.csv"),
    start_date=datetime.now() - timedelta(days=90),
    end_date=datetime.now()
)

# Access results
print(f"Hotspots detected: {len(report.hotspots)}")
print(f"Critical alerts: {len(report.critical_alerts)}")

for hotspot in report.critical_alerts:
    print(f"\nRegion: {hotspot.location.region_name}")
    print(f"Severity: {hotspot.severity.value}")
    print(f"Confirming signals: {hotspot.num_confirming_signals}")
    print(f"Actions: {hotspot.recommended_actions}")
```

### Run Example with Synthetic Data

```bash
cd examples
python disease_hotspot_detection_example.py
```

This generates synthetic outbreak data and runs the complete detection pipeline, demonstrating:
- Multi-stream data generation with realistic outbreak patterns
- Temporal anomaly detection with STL + modified Z-scores
- Spatial clustering of anomalies
- Cross-validation and hotspot identification
- Report generation

## 📊 Data Format Requirements

### ESSENCE Syndromic Data

CSV/Excel with columns:
- `timestamp`: Date/datetime
- `region_name`: Region identifier
- `chief_complaint_category`: ILI, Respiratory, Fever, etc.
- `visit_count`: Number of visits
- `total_visits`: Total ED visits (optional)
- `latitude`, `longitude`: Coordinates (optional)
- `population`: Region population (optional)

### Wastewater Data

CSV/Excel with columns:
- `timestamp`: Collection date
- `collection_site_id`: Site identifier
- `viral_load`: Copies per liter
- `pathogen`: SARS-CoV-2, Influenza, etc.
- `region_name`: Region identifier
- `latitude`, `longitude`: Coordinates (optional)
- `population_served`: Population served (optional)

### Search Trends Data

CSV/Excel/JSON with columns:
- `timestamp`: Date/week
- `region_name`: Region identifier
- `search_term`: Search query
- `normalized_interest`: 0-100 scale
- `latitude`, `longitude`: Coordinates (optional)

### Mobility Data

CSV/Excel with columns:
- `timestamp`: Date
- `region_name`: Region identifier
- `category`: retail, transit, workplaces, etc.
- `baseline_comparison`: % change from baseline
- `latitude`, `longitude`: Coordinates (optional)

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the `disease_surveillance` directory:

```bash
# Temporal Detection
STL_SEASONAL_PERIOD=7
MODIFIED_ZSCORE_THRESHOLD=3.5
MIN_OBSERVATIONS=28
STL_ROBUST=true

# Stream-specific thresholds
ESSENCE_THRESHOLD=3.5
WASTEWATER_THRESHOLD=3.0
SEARCH_TRENDS_THRESHOLD=3.5
MOBILITY_THRESHOLD=4.0

# Spatial Clustering
USE_SATSCAN=false
SATSCAN_EXECUTABLE=/usr/local/bin/satscan
MAX_CLUSTER_SIZE=0.5
SPATIAL_RADIUS_KM=50
DBSCAN_MIN_SAMPLES=3

# Cross-Validation
MIN_CONFIRMING_SIGNALS=2
TIME_ALIGNMENT_WINDOW_DAYS=7

# Signal weights (0-1)
ESSENCE_WEIGHT=1.0
WASTEWATER_WEIGHT=0.9
SEARCH_TRENDS_WEIGHT=0.6
MOBILITY_WEIGHT=0.5

# Alerting
ENABLE_EMAIL_ALERTS=false
EMAIL_RECIPIENTS=epidemiology@health.gov
ENABLE_SLACK_ALERTS=false
SLACK_WEBHOOK_URL=https://hooks.slack.com/...

# Logging
LOG_LEVEL=INFO
ENABLE_VERBOSE_LOGGING=false
```

### Python Configuration

```python
from disease_surveillance.config import SurveillanceConfig

# Set custom thresholds
SurveillanceConfig.MODIFIED_ZSCORE_THRESHOLD = 3.0
SurveillanceConfig.MIN_CONFIRMING_SIGNALS = 3

# Set data paths
SurveillanceConfig.set_data_paths(
    essence_path="path/to/essence.csv",
    wastewater_path="path/to/wastewater.csv"
)

# Validate configuration
if SurveillanceConfig.validate():
    print("Configuration valid ✓")
```

## 🧪 Algorithm Details

### STL Decomposition

Seasonal-Trend decomposition using LOESS (Locally Estimated Scatterplot Smoothing):

```
Y(t) = T(t) + S(t) + R(t)
```

Where:
- `Y(t)`: Observed time series
- `T(t)`: Trend component
- `S(t)`: Seasonal component
- `R(t)`: Residual component

Anomalies are detected in the residual component.

### Modified Z-Score

Uses Median Absolute Deviation (MAD) instead of standard deviation for robustness:

```
Modified Z-Score = 0.6745 × (X - median(X)) / MAD
MAD = median(|X - median(X)|)
```

Threshold typically set at 3.5 (equivalent to ~3σ for normal distributions).

### Signal Agreement Score

Weighted cross-validation score:

```
Agreement Score = Σ(w_i × c_i) / Σ(w_i)
```

Where:
- `w_i`: Weight for signal i (ESSENCE=1.0, Wastewater=0.9, etc.)
- `c_i`: Confidence for signal i (0-1)

### Severity Assessment

Multi-factor scoring system:

| Factor | Points |
|--------|--------|
| 4 confirming signals | +3 |
| Agreement score ≥ 0.8 | +3 |
| Modified Z-score ≥ 5.0 | +3 |
| Relative risk ≥ 3.0 | +3 |

**Severity Levels**:
- **Critical**: Score ≥ 9
- **High**: Score ≥ 6
- **Moderate**: Score ≥ 3
- **Low**: Score < 3

## 📈 Output Reports

### JSON Report

```json
{
  "report_id": "surveillance_report_20250122_143022",
  "generation_time": "2025-01-22T14:30:22",
  "time_period_start": "2024-10-23",
  "time_period_end": "2025-01-22",
  "summary": {
    "total_hotspots": 3,
    "critical_alerts": 1,
    "high_priority_alerts": 2,
    "temporal_anomalies": 47,
    "spatial_clusters": 2
  },
  "hotspots": [
    {
      "hotspot_id": "hotspot_Region_3_0_1705932622",
      "severity": "critical",
      "location": {
        "region_name": "Region_3",
        "latitude": 38.5,
        "longitude": -120.5
      },
      "num_confirming_signals": 4,
      "signal_agreement_score": 0.87,
      "trend_direction": "increasing",
      "recommended_actions": [
        "Activate emergency response protocols",
        "Deploy mobile testing units",
        "Issue public health advisory"
      ]
    }
  ]
}
```

### Markdown Summary

```markdown
# Disease Surveillance Report

**Report ID**: surveillance_report_20250122_143022
**Period**: 2024-10-23 to 2025-01-22

## Executive Summary

- **Total Hotspots**: 3
- **Critical Alerts**: 1
- **High Priority**: 2

## Critical Alerts

### Region_3 (hotspot_Region_3_0_1705932622)

- **Severity**: CRITICAL
- **Confirming Signals**: 4 (ESSENCE, Wastewater, Search Trends, Mobility)
- **Agreement**: 87%
- **Trend**: Increasing

**Recommended Actions**:
- Activate emergency response protocols
- Deploy mobile testing units to affected area
- Issue public health advisory
```

## 🔬 Use Cases

### 1. COVID-19 Surveillance

```python
pipeline = DiseaseHotspotPipeline(
    essence_weight=1.0,
    wastewater_weight=0.95,  # High weight for COVID wastewater
    search_trends_weight=0.7,
    mobility_weight=0.6
)
```

### 2. Influenza Season Monitoring

```python
pipeline = DiseaseHotspotPipeline(
    stl_seasonal_period=365,  # Annual seasonality
    essence_weight=1.0,
    wastewater_weight=0.5,  # Less reliable for flu
    search_trends_weight=0.8
)
```

### 3. Foodborne Outbreak Detection

```python
pipeline = DiseaseHotspotPipeline(
    spatial_radius_km=25,  # Smaller clusters
    min_confirming_signals=3,  # Higher confidence
    essence_weight=1.0,
    wastewater_weight=0.0,  # Not applicable
    search_trends_weight=0.4
)
```

## 🧩 Integration Examples

### Real-Time Monitoring

```python
import schedule
import time

def run_surveillance():
    report = pipeline.run(
        essence_path=get_latest_essence_data(),
        wastewater_path=get_latest_wastewater_data(),
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now()
    )

    if report.critical_alerts:
        send_alert_email(report.critical_alerts)

# Run every 24 hours
schedule.every(24).hours.do(run_surveillance)

while True:
    schedule.run_pending()
    time.sleep(3600)
```

### API Integration

```python
from fastapi import FastAPI
from disease_surveillance import DiseaseHotspotPipeline

app = FastAPI()
pipeline = DiseaseHotspotPipeline()

@app.post("/api/v1/detect-hotspots")
async def detect_hotspots(
    essence_url: str,
    wastewater_url: str
):
    # Download data
    essence_data = download_data(essence_url)
    wastewater_data = download_data(wastewater_url)

    # Run detection
    report = pipeline.run(
        essence_path=essence_data,
        wastewater_path=wastewater_data
    )

    return {
        "hotspots": len(report.hotspots),
        "critical": len(report.critical_alerts),
        "report_id": report.report_id
    }
```

## 🔧 Advanced Features

### Custom Anomaly Detectors

```python
from disease_surveillance.temporal_detection import TemporalAnomalyDetector

custom_detector = TemporalAnomalyDetector(
    seasonal_period=14,  # Bi-weekly
    threshold=3.0,       # More sensitive
    min_observations=42  # 6 weeks minimum
)

pipeline = DiseaseHotspotPipeline(
    essence_detector=custom_detector
)
```

### SaTScan Integration

Requires [SaTScan](https://www.satscan.org/) to be installed separately.

```python
pipeline = DiseaseHotspotPipeline(
    use_satscan=True,
    satscan_executable="/usr/local/bin/satscan",
    max_cluster_size=0.5,  # 50% of population max
    min_cases=5
)
```

### Custom Signal Weights

```python
# Example: Prioritize wastewater for early warning
pipeline = DiseaseHotspotPipeline(
    essence_weight=0.9,
    wastewater_weight=1.0,  # Highest weight
    search_trends_weight=0.5,
    mobility_weight=0.4
)
```

## 📚 References

### Scientific Literature

1. **STL Decomposition**: Cleveland et al. (1990). "STL: A Seasonal-Trend Decomposition Procedure Based on Loess"
2. **Modified Z-Score**: Iglewicz & Hoaglin (1993). "How to Detect and Handle Outliers"
3. **SaTScan**: Kulldorff (1997). "A spatial scan statistic"
4. **Wastewater Surveillance**: Peccia et al. (2020). "Measurement of SARS-CoV-2 RNA in wastewater"

### Data Sources

- **NSSP ESSENCE**: [National Syndromic Surveillance Program](https://www.cdc.gov/nssp/index.html)
- **NWSS**: [National Wastewater Surveillance System](https://www.cdc.gov/nwss/)
- **Google Trends**: [Google Trends API](https://trends.google.com/)
- **Google Mobility**: [COVID-19 Community Mobility Reports](https://www.google.com/covid19/mobility/)

## 🛠️ Development

### Running Tests

```bash
pytest tests/ -v --cov=disease_surveillance
```

### Code Quality

```bash
# Format
black disease_surveillance/

# Lint
flake8 disease_surveillance/

# Type check
mypy disease_surveillance/
```

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

## 📞 Support

For issues or questions:
- GitHub Issues: [https://github.com/yourusername/GenZ/issues](https://github.com/yourusername/GenZ/issues)
- Email: support@example.com

## 🙏 Acknowledgments

This system builds on methodologies from:
- CDC National Syndromic Surveillance Program (NSSP)
- CDC National Wastewater Surveillance System (NWSS)
- Johns Hopkins Center for Systems Science and Engineering (CSSE)
- SaTScan™ software for spatial statistics

---

**Version**: 1.0.0
**Last Updated**: 2025-01-22
