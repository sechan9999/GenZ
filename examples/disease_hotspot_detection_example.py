"""
Example Usage of Disease Hotspot Detection System
Demonstrates multi-stream surveillance with synthetic data
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from disease_surveillance import DiseaseHotspotPipeline
from disease_surveillance.config import SurveillanceConfig


def generate_synthetic_essence_data(
    n_days: int = 90,
    n_regions: int = 5,
    outbreak_region: str = "Region_3",
    outbreak_start_day: int = 60,
    outbreak_duration: int = 21
) -> pd.DataFrame:
    """
    Generate synthetic ESSENCE syndromic surveillance data

    Args:
        n_days: Number of days of data
        n_regions: Number of regions
        outbreak_region: Region with outbreak
        outbreak_start_day: Day when outbreak starts
        outbreak_duration: Duration of outbreak in days

    Returns:
        DataFrame with ESSENCE data
    """
    print("Generating synthetic ESSENCE data...")

    data = []
    base_date = datetime.now() - timedelta(days=n_days)

    regions = [f"Region_{i+1}" for i in range(n_regions)]
    chief_complaints = ["ILI", "Respiratory", "Fever"]

    for day in range(n_days):
        current_date = base_date + timedelta(days=day)

        for region_idx, region in enumerate(regions):
            for complaint in chief_complaints:
                # Base visit count with weekly seasonality
                base_visits = 50 + 20 * np.sin(2 * np.pi * day / 7)

                # Add random variation
                base_visits += np.random.normal(0, 5)

                # Add outbreak signal
                if (region == outbreak_region and
                    outbreak_start_day <= day < outbreak_start_day + outbreak_duration):
                    days_into_outbreak = day - outbreak_start_day
                    # Exponential growth followed by plateau
                    if days_into_outbreak < 10:
                        outbreak_multiplier = 1 + 2 * (days_into_outbreak / 10)
                    else:
                        outbreak_multiplier = 3.0
                    base_visits *= outbreak_multiplier

                visit_count = max(0, int(base_visits))
                total_visits = int(visit_count / 0.1)  # Assume 10% are ILI-related

                data.append({
                    'timestamp': current_date,
                    'region_name': region,
                    'region_code': region,
                    'latitude': 37.0 + region_idx * 0.5,
                    'longitude': -122.0 + region_idx * 0.5,
                    'chief_complaint_category': complaint,
                    'visit_count': visit_count,
                    'total_visits': total_visits,
                    'percentage': (visit_count / total_visits * 100) if total_visits > 0 else 0,
                    'population': 100000 + region_idx * 50000
                })

    return pd.DataFrame(data)


def generate_synthetic_wastewater_data(
    n_days: int = 90,
    n_regions: int = 5,
    outbreak_region: str = "Region_3",
    outbreak_start_day: int = 55,  # Earlier than clinical
    outbreak_duration: int = 28
) -> pd.DataFrame:
    """
    Generate synthetic wastewater viral load data

    Note: Wastewater signals typically appear ~5-7 days before clinical signals
    """
    print("Generating synthetic wastewater data...")

    data = []
    base_date = datetime.now() - timedelta(days=n_days)

    regions = [f"Region_{i+1}" for i in range(n_regions)]

    for day in range(n_days):
        current_date = base_date + timedelta(days=day)

        for region_idx, region in enumerate(regions):
            # Base viral load with weekly seasonality
            base_load = 1000 + 300 * np.sin(2 * np.pi * day / 7)

            # Add random variation
            base_load += np.random.normal(0, 200)

            # Add outbreak signal (appears earlier than clinical)
            if (region == outbreak_region and
                outbreak_start_day <= day < outbreak_start_day + outbreak_duration):
                days_into_outbreak = day - outbreak_start_day
                # Exponential growth
                if days_into_outbreak < 12:
                    outbreak_multiplier = 1 + 4 * (days_into_outbreak / 12)
                else:
                    outbreak_multiplier = 5.0
                base_load *= outbreak_multiplier

            viral_load = max(100, base_load)

            data.append({
                'timestamp': current_date,
                'region_name': region,
                'region_code': region,
                'latitude': 37.0 + region_idx * 0.5,
                'longitude': -122.0 + region_idx * 0.5,
                'collection_site_id': f"site_{region}",
                'viral_load': viral_load,
                'pathogen': 'SARS-CoV-2',
                'population_served': 100000 + region_idx * 50000
            })

    return pd.DataFrame(data)


def generate_synthetic_search_trends_data(
    n_days: int = 90,
    n_regions: int = 5,
    outbreak_region: str = "Region_3",
    outbreak_start_day: int = 58,
    outbreak_duration: int = 25
) -> pd.DataFrame:
    """
    Generate synthetic search trends data

    Note: Search trends appear slightly before clinical but after wastewater
    """
    print("Generating synthetic search trends data...")

    data = []
    base_date = datetime.now() - timedelta(days=n_days)

    regions = [f"Region_{i+1}" for i in range(n_regions)]
    search_terms = ["flu symptoms", "covid test", "fever"]

    for day in range(n_days):
        current_date = base_date + timedelta(days=day)

        for region_idx, region in enumerate(regions):
            for term in search_terms:
                # Base interest with weekly seasonality
                base_interest = 20 + 10 * np.sin(2 * np.pi * day / 7)

                # Add random variation
                base_interest += np.random.normal(0, 5)

                # Add outbreak signal
                if (region == outbreak_region and
                    outbreak_start_day <= day < outbreak_start_day + outbreak_duration):
                    days_into_outbreak = day - outbreak_start_day
                    if days_into_outbreak < 10:
                        outbreak_multiplier = 1 + 2.5 * (days_into_outbreak / 10)
                    else:
                        outbreak_multiplier = 3.5
                    base_interest *= outbreak_multiplier

                normalized_interest = np.clip(base_interest, 0, 100)

                data.append({
                    'timestamp': current_date,
                    'region_name': region,
                    'region_code': region,
                    'latitude': 37.0 + region_idx * 0.5,
                    'longitude': -122.0 + region_idx * 0.5,
                    'search_term': term,
                    'normalized_interest': normalized_interest
                })

    return pd.DataFrame(data)


def generate_synthetic_mobility_data(
    n_days: int = 90,
    n_regions: int = 5,
    outbreak_region: str = "Region_3",
    outbreak_start_day: int = 62,
    outbreak_duration: int = 21
) -> pd.DataFrame:
    """
    Generate synthetic mobility data

    Note: Mobility changes appear around the same time as clinical signals
    """
    print("Generating synthetic mobility data...")

    data = []
    base_date = datetime.now() - timedelta(days=n_days)

    regions = [f"Region_{i+1}" for i in range(n_regions)]
    categories = ["retail", "transit", "workplaces"]

    for day in range(n_days):
        current_date = base_date + timedelta(days=day)

        for region_idx, region in enumerate(regions):
            for category in categories:
                # Base mobility (% change from baseline)
                base_change = -5 + 3 * np.sin(2 * np.pi * day / 7)

                # Add random variation
                base_change += np.random.normal(0, 2)

                # Add outbreak effect (people stay home)
                if (region == outbreak_region and
                    outbreak_start_day <= day < outbreak_start_day + outbreak_duration):
                    days_into_outbreak = day - outbreak_start_day
                    if days_into_outbreak < 10:
                        reduction = -20 * (days_into_outbreak / 10)
                    else:
                        reduction = -20
                    base_change += reduction

                baseline_comparison = base_change

                data.append({
                    'timestamp': current_date,
                    'region_name': region,
                    'region_code': region,
                    'latitude': 37.0 + region_idx * 0.5,
                    'longitude': -122.0 + region_idx * 0.5,
                    'category': category,
                    'baseline_comparison': baseline_comparison,
                    'mobility_index': 100 + baseline_comparison
                })

    return pd.DataFrame(data)


def main():
    """Run disease hotspot detection example"""
    print("=" * 80)
    print("DISEASE HOTSPOT DETECTION SYSTEM - EXAMPLE")
    print("=" * 80)
    print()

    # Create data directory
    data_dir = Path("./disease_surveillance/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic data
    print("Step 1: Generating synthetic surveillance data...")
    print()

    essence_df = generate_synthetic_essence_data(
        n_days=90,
        n_regions=5,
        outbreak_region="Region_3",
        outbreak_start_day=60,
        outbreak_duration=21
    )

    wastewater_df = generate_synthetic_wastewater_data(
        n_days=90,
        n_regions=5,
        outbreak_region="Region_3",
        outbreak_start_day=55,  # Earlier warning
        outbreak_duration=28
    )

    search_trends_df = generate_synthetic_search_trends_data(
        n_days=90,
        n_regions=5,
        outbreak_region="Region_3",
        outbreak_start_day=58,
        outbreak_duration=25
    )

    mobility_df = generate_synthetic_mobility_data(
        n_days=90,
        n_regions=5,
        outbreak_region="Region_3",
        outbreak_start_day=62,
        outbreak_duration=21
    )

    # Save synthetic data
    print("\nStep 2: Saving synthetic data to files...")
    essence_path = data_dir / "essence_data.csv"
    wastewater_path = data_dir / "wastewater_data.csv"
    search_trends_path = data_dir / "search_trends_data.csv"
    mobility_path = data_dir / "mobility_data.csv"

    essence_df.to_csv(essence_path, index=False)
    wastewater_df.to_csv(wastewater_path, index=False)
    search_trends_df.to_csv(search_trends_path, index=False)
    mobility_df.to_csv(mobility_path, index=False)

    print(f"  ✓ ESSENCE data: {essence_path} ({len(essence_df)} records)")
    print(f"  ✓ Wastewater data: {wastewater_path} ({len(wastewater_df)} records)")
    print(f"  ✓ Search trends data: {search_trends_path} ({len(search_trends_df)} records)")
    print(f"  ✓ Mobility data: {mobility_path} ({len(mobility_df)} records)")

    # Initialize pipeline
    print("\nStep 3: Initializing disease hotspot detection pipeline...")
    pipeline = DiseaseHotspotPipeline(
        stl_seasonal_period=7,
        modified_zscore_threshold=3.5,
        min_observations=28,
        use_satscan=False,  # Use simplified clustering
        spatial_radius_km=50,
        min_confirming_signals=2,
        essence_weight=1.0,
        wastewater_weight=0.9,
        search_trends_weight=0.6,
        mobility_weight=0.5,
        output_dir=Path("./disease_surveillance/output")
    )

    print("  ✓ Pipeline initialized")

    # Run pipeline
    print("\nStep 4: Running disease hotspot detection pipeline...")
    print()

    report = pipeline.run(
        essence_path=essence_path,
        wastewater_path=wastewater_path,
        search_trends_path=search_trends_path,
        mobility_path=mobility_path,
        start_date=datetime.now() - timedelta(days=90),
        end_date=datetime.now()
    )

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()
    print(f"Report ID: {report.report_id}")
    print(f"Period: {report.time_period_start.date()} to {report.time_period_end.date()}")
    print()
    print(f"📊 Total Hotspots Detected: {len(report.hotspots)}")
    print(f"🚨 Critical Alerts: {len(report.critical_alerts)}")
    print(f"⚠️  High Priority Alerts: {len(report.high_priority_alerts)}")
    print()
    print(f"📈 Temporal Anomalies: {report.temporal_anomalies_detected}")
    print(f"🗺️  Spatial Clusters: {report.spatial_clusters_detected}")
    print()

    if report.critical_alerts:
        print("CRITICAL ALERTS:")
        print("-" * 80)
        for hotspot in report.critical_alerts:
            print(f"\n🚨 {hotspot.location.region_name} (ID: {hotspot.hotspot_id})")
            print(f"   Severity: {hotspot.severity.value.upper()}")
            print(f"   Confirming Signals: {hotspot.num_confirming_signals}")
            print(f"   Agreement Score: {hotspot.signal_agreement_score:.2%}")
            print(f"   Trend: {hotspot.trend_direction}")
            print(f"   Description: {hotspot.description}")
            print(f"   Recommended Actions:")
            for action in hotspot.recommended_actions:
                print(f"     • {action}")

    if report.high_priority_alerts:
        print("\nHIGH PRIORITY ALERTS:")
        print("-" * 80)
        for hotspot in report.high_priority_alerts:
            print(f"\n⚠️  {hotspot.location.region_name} (ID: {hotspot.hotspot_id})")
            print(f"   Confirming Signals: {hotspot.num_confirming_signals}")
            print(f"   Agreement Score: {hotspot.signal_agreement_score:.2%}")
            print(f"   Trend: {hotspot.trend_direction}")

    print("\n" + "=" * 80)
    print(f"✓ Reports saved to: {pipeline.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
