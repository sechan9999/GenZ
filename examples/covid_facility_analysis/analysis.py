"""
Statistical Analysis and Aggregation for COVID-19 Facility Data.

This module performs:
- Time series analysis of COVID-19 trends
- Regional and facility-type comparisons
- Capacity and resource utilization analysis
- Outbreak detection and anomaly identification
- Executive-level KPI calculations
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging
from scipy import stats
from collections import defaultdict

from models import FacilityDailySnapshot, FacilityType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CovidFacilityAnalyzer:
    """
    Analyzer for COVID-19 facility data.

    Performs statistical analysis, trend detection, and generates
    executive-level insights from merged facility snapshots.
    """

    def __init__(self, snapshots: List[FacilityDailySnapshot]):
        """
        Initialize analyzer with facility snapshots.

        Args:
            snapshots: List of FacilityDailySnapshot objects
        """
        self.snapshots = snapshots
        self.df = self._convert_to_dataframe()
        logger.info(f"Analyzer initialized with {len(self.snapshots)} snapshots")

    def _convert_to_dataframe(self) -> pd.DataFrame:
        """Convert snapshots to pandas DataFrame for analysis."""
        records = []

        for snapshot in self.snapshots:
            record = {
                'snapshot_id': snapshot.snapshot_id,
                'facility_id': snapshot.facility_id,
                'facility_name': snapshot.facility_name,
                'facility_type': snapshot.facility_type,
                'snapshot_date': snapshot.snapshot_date,
                'occupancy_rate': snapshot.occupancy_rate,
                'staff_shortage': snapshot.staff_shortage,
                'ppe_critical': snapshot.ppe_critical,
                'covid_positivity_rate': snapshot.covid_positivity_rate,
                'data_completeness': snapshot.data_completeness,
                'merge_confidence': snapshot.merge_confidence,
            }

            # EHR metrics
            if snapshot.ehr_data:
                record.update({
                    'total_patients': snapshot.ehr_data.total_patients,
                    'covid_positive': snapshot.ehr_data.covid_positive_count,
                    'covid_hospitalized': snapshot.ehr_data.covid_hospitalized,
                    'covid_icu': snapshot.ehr_data.covid_icu_count,
                    'covid_ventilator': snapshot.ehr_data.covid_ventilator_count,
                    'covid_deaths': snapshot.ehr_data.covid_deaths,
                    'tests_conducted': snapshot.ehr_data.tests_conducted,
                    'tests_positive': snapshot.ehr_data.tests_positive,
                })

            # Staffing metrics
            if snapshot.staffing_data:
                record.update({
                    'total_beds': snapshot.staffing_data.total_beds,
                    'occupied_beds': snapshot.staffing_data.occupied_beds,
                    'physicians_present': snapshot.staffing_data.physicians_present,
                    'nurses_present': snapshot.staffing_data.nurses_present,
                    'staff_covid_positive': snapshot.staffing_data.staff_covid_positive,
                })

            # PPE metrics
            if snapshot.ppe_data:
                record.update({
                    'n95_masks': snapshot.ppe_data.n95_masks_count,
                    'n95_days_supply': snapshot.ppe_data.n95_days_supply,
                })

            records.append(record)

        df = pd.DataFrame(records)
        df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
        return df

    # ========================================================================
    # EXECUTIVE SUMMARY METRICS
    # ========================================================================

    def calculate_system_wide_kpis(self) -> Dict[str, any]:
        """
        Calculate system-wide KPIs across all VA facilities.

        Returns:
            Dictionary with executive-level KPIs
        """
        logger.info("Calculating system-wide KPIs...")

        # Get most recent date
        latest_date = self.df['snapshot_date'].max()
        latest_data = self.df[self.df['snapshot_date'] == latest_date]

        kpis = {
            'report_date': latest_date.date(),
            'facilities_reporting': len(latest_data),

            # COVID-19 Patient Metrics
            'total_covid_positive': int(latest_data['covid_positive'].sum()),
            'total_covid_hospitalized': int(latest_data['covid_hospitalized'].sum()),
            'total_covid_icu': int(latest_data['covid_icu'].sum()),
            'total_covid_ventilator': int(latest_data['covid_ventilator'].sum()),
            'total_covid_deaths': int(latest_data['covid_deaths'].sum()),

            # Testing Metrics
            'total_tests_conducted': int(latest_data['tests_conducted'].sum()),
            'total_tests_positive': int(latest_data['tests_positive'].sum()),
            'avg_positivity_rate': float(latest_data['covid_positivity_rate'].mean()),

            # Capacity Metrics
            'total_beds': int(latest_data['total_beds'].sum()),
            'total_occupied_beds': int(latest_data['occupied_beds'].sum()),
            'avg_occupancy_rate': float(latest_data['occupancy_rate'].mean()),

            # Staffing Metrics
            'facilities_staff_shortage': int(latest_data['staff_shortage'].sum()),
            'total_staff_covid_positive': int(latest_data['staff_covid_positive'].sum()),

            # PPE Status
            'facilities_ppe_critical': int(latest_data['ppe_critical'].sum()),
            'avg_n95_days_supply': float(latest_data['n95_days_supply'].mean()),

            # Data Quality
            'avg_data_completeness': float(latest_data['data_completeness'].mean()),
        }

        # Calculate 7-day trends
        week_ago = latest_date - timedelta(days=7)
        week_ago_data = self.df[self.df['snapshot_date'] == week_ago]

        if not week_ago_data.empty:
            kpis['covid_positive_7d_change'] = (
                int(latest_data['covid_positive'].sum()) -
                int(week_ago_data['covid_positive'].sum())
            )
            kpis['covid_deaths_7d_change'] = (
                int(latest_data['covid_deaths'].sum()) -
                int(week_ago_data['covid_deaths'].sum())
            )

        logger.info(f"System-wide KPIs calculated for {latest_date.date()}")
        return kpis

    # ========================================================================
    # TIME SERIES ANALYSIS
    # ========================================================================

    def analyze_time_trends(
        self,
        metric: str,
        window_days: int = 7
    ) -> pd.DataFrame:
        """
        Analyze time trends for a specific metric.

        Args:
            metric: Column name to analyze
            window_days: Rolling average window size

        Returns:
            DataFrame with daily aggregates and rolling averages
        """
        logger.info(f"Analyzing time trends for '{metric}' with {window_days}-day window")

        # Daily aggregates
        daily = self.df.groupby('snapshot_date').agg({
            metric: ['sum', 'mean', 'std', 'min', 'max', 'count']
        }).reset_index()

        daily.columns = [
            'date',
            f'{metric}_total',
            f'{metric}_avg',
            f'{metric}_std',
            f'{metric}_min',
            f'{metric}_max',
            'facility_count'
        ]

        # Calculate rolling averages
        daily[f'{metric}_rolling_avg'] = (
            daily[f'{metric}_total'].rolling(window=window_days, min_periods=1).mean()
        )

        # Calculate day-over-day change
        daily[f'{metric}_daily_change'] = daily[f'{metric}_total'].diff()

        # Calculate percent change
        daily[f'{metric}_pct_change'] = (
            daily[f'{metric}_total'].pct_change() * 100
        )

        return daily

    def detect_outbreaks(
        self,
        threshold_std: float = 2.0,
        min_cases: int = 10
    ) -> pd.DataFrame:
        """
        Detect potential COVID-19 outbreaks at facilities.

        An outbreak is flagged when:
        - COVID positive count exceeds mean + threshold_std * std
        - Absolute count exceeds min_cases

        Args:
            threshold_std: Number of standard deviations for outlier detection
            min_cases: Minimum absolute case count to flag

        Returns:
            DataFrame with flagged outbreak facilities
        """
        logger.info(f"Detecting outbreaks (threshold={threshold_std} std, min_cases={min_cases})")

        # Calculate statistics per facility
        facility_stats = self.df.groupby('facility_id').agg({
            'covid_positive': ['mean', 'std', 'max'],
            'facility_name': 'first',
            'facility_type': 'first'
        }).reset_index()

        facility_stats.columns = [
            'facility_id',
            'covid_positive_mean',
            'covid_positive_std',
            'covid_positive_max',
            'facility_name',
            'facility_type'
        ]

        # Calculate outbreak threshold for each facility
        facility_stats['outbreak_threshold'] = (
            facility_stats['covid_positive_mean'] +
            threshold_std * facility_stats['covid_positive_std']
        )

        # Get latest data
        latest_date = self.df['snapshot_date'].max()
        latest = self.df[self.df['snapshot_date'] == latest_date][
            ['facility_id', 'covid_positive', 'snapshot_date']
        ]

        # Merge and flag outbreaks
        outbreaks = facility_stats.merge(latest, on='facility_id')

        outbreaks['is_outbreak'] = (
            (outbreaks['covid_positive'] > outbreaks['outbreak_threshold']) &
            (outbreaks['covid_positive'] >= min_cases)
        )

        # Filter to only outbreak facilities
        outbreak_facilities = outbreaks[outbreaks['is_outbreak']].sort_values(
            'covid_positive', ascending=False
        )

        logger.info(f"Detected {len(outbreak_facilities)} potential outbreaks")
        return outbreak_facilities[[
            'facility_id',
            'facility_name',
            'facility_type',
            'covid_positive',
            'covid_positive_mean',
            'outbreak_threshold',
            'snapshot_date'
        ]]

    # ========================================================================
    # FACILITY TYPE COMPARISONS
    # ========================================================================

    def compare_by_facility_type(self) -> pd.DataFrame:
        """
        Compare metrics across facility types.

        Returns:
            DataFrame with aggregated metrics by facility type
        """
        logger.info("Comparing metrics by facility type...")

        # Get latest data
        latest_date = self.df['snapshot_date'].max()
        latest = self.df[self.df['snapshot_date'] == latest_date]

        # Group by facility type
        comparison = latest.groupby('facility_type').agg({
            'facility_id': 'count',
            'covid_positive': ['sum', 'mean', 'std'],
            'covid_hospitalized': ['sum', 'mean'],
            'covid_icu': ['sum', 'mean'],
            'covid_deaths': ['sum', 'mean'],
            'occupancy_rate': 'mean',
            'covid_positivity_rate': 'mean',
            'staff_shortage': 'sum',
            'ppe_critical': 'sum',
        }).reset_index()

        comparison.columns = [
            'facility_type',
            'facility_count',
            'covid_positive_total',
            'covid_positive_avg',
            'covid_positive_std',
            'covid_hospitalized_total',
            'covid_hospitalized_avg',
            'covid_icu_total',
            'covid_icu_avg',
            'covid_deaths_total',
            'covid_deaths_avg',
            'avg_occupancy_rate',
            'avg_positivity_rate',
            'facilities_with_staff_shortage',
            'facilities_with_ppe_critical',
        ]

        return comparison

    # ========================================================================
    # CAPACITY ANALYSIS
    # ========================================================================

    def analyze_capacity_constraints(
        self,
        occupancy_threshold: float = 0.85,
        icu_threshold: float = 0.90
    ) -> Dict[str, any]:
        """
        Analyze capacity constraints across facilities.

        Args:
            occupancy_threshold: Occupancy rate threshold for concern
            icu_threshold: ICU occupancy threshold for concern

        Returns:
            Dictionary with capacity constraint analysis
        """
        logger.info("Analyzing capacity constraints...")

        # Get latest data
        latest_date = self.df['snapshot_date'].max()
        latest = self.df[self.df['snapshot_date'] == latest_date]

        # Calculate ICU occupancy rate
        latest['icu_occupancy_rate'] = np.where(
            latest['total_beds'] > 0,
            latest['occupied_beds'] / latest['total_beds'],
            0
        )

        # Identify constrained facilities
        high_occupancy = latest[latest['occupancy_rate'] >= occupancy_threshold]

        analysis = {
            'report_date': latest_date.date(),
            'total_facilities': len(latest),

            # High occupancy
            'facilities_high_occupancy': len(high_occupancy),
            'high_occupancy_list': high_occupancy[
                ['facility_id', 'facility_name', 'occupancy_rate', 'total_beds', 'occupied_beds']
            ].to_dict('records'),

            # Capacity statistics
            'avg_occupancy_rate': float(latest['occupancy_rate'].mean()),
            'median_occupancy_rate': float(latest['occupancy_rate'].median()),
            'max_occupancy_rate': float(latest['occupancy_rate'].max()),

            # Bed availability
            'total_beds_system': int(latest['total_beds'].sum()),
            'total_occupied_beds': int(latest['occupied_beds'].sum()),
            'total_available_beds': int(latest['total_beds'].sum() - latest['occupied_beds'].sum()),

            # Staffing constraints
            'facilities_with_staff_shortage': int(latest['staff_shortage'].sum()),

            # PPE constraints
            'facilities_with_ppe_critical': int(latest['ppe_critical'].sum()),
        }

        return analysis

    # ========================================================================
    # STATISTICAL TESTS AND CORRELATIONS
    # ========================================================================

    def correlation_analysis(self) -> pd.DataFrame:
        """
        Perform correlation analysis between key metrics.

        Returns:
            Correlation matrix DataFrame
        """
        logger.info("Performing correlation analysis...")

        metrics = [
            'covid_positive',
            'covid_hospitalized',
            'covid_icu',
            'covid_ventilator',
            'occupancy_rate',
            'covid_positivity_rate',
            'staff_covid_positive',
        ]

        # Filter to metrics that exist
        available_metrics = [m for m in metrics if m in self.df.columns]

        # Calculate correlation matrix
        corr_matrix = self.df[available_metrics].corr()

        return corr_matrix

    def test_facility_type_differences(
        self,
        metric: str = 'covid_positive'
    ) -> Dict[str, any]:
        """
        Statistical test for differences between facility types.

        Performs ANOVA test to determine if metric differs significantly
        across facility types.

        Args:
            metric: Metric to test

        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing facility type differences for '{metric}'")

        # Get latest data
        latest_date = self.df['snapshot_date'].max()
        latest = self.df[self.df['snapshot_date'] == latest_date]

        # Group by facility type
        groups = [
            group[metric].dropna().values
            for name, group in latest.groupby('facility_type')
            if len(group) > 0
        ]

        # Perform ANOVA
        if len(groups) >= 2:
            f_stat, p_value = stats.f_oneway(*groups)

            result = {
                'metric': metric,
                'test': 'ANOVA',
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'interpretation': (
                    f"Significant difference found (p={p_value:.4f})" if p_value < 0.05
                    else f"No significant difference (p={p_value:.4f})"
                )
            }
        else:
            result = {
                'metric': metric,
                'test': 'ANOVA',
                'error': 'Insufficient groups for comparison'
            }

        return result

    # ========================================================================
    # ANOMALY DETECTION
    # ========================================================================

    def detect_data_anomalies(
        self,
        z_threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Detect statistical anomalies in facility data.

        Uses Z-score method to identify outliers.

        Args:
            z_threshold: Z-score threshold for anomaly detection

        Returns:
            DataFrame with anomalous records
        """
        logger.info(f"Detecting data anomalies (z_threshold={z_threshold})...")

        # Calculate Z-scores for key metrics
        metrics = [
            'covid_positive',
            'covid_hospitalized',
            'covid_deaths',
            'occupancy_rate',
        ]

        anomalies = []

        for metric in metrics:
            if metric not in self.df.columns:
                continue

            # Calculate Z-scores
            mean = self.df[metric].mean()
            std = self.df[metric].std()

            self.df[f'{metric}_zscore'] = (self.df[metric] - mean) / std

            # Find anomalies
            anomaly_records = self.df[
                abs(self.df[f'{metric}_zscore']) > z_threshold
            ].copy()

            anomaly_records['anomaly_metric'] = metric
            anomaly_records['anomaly_zscore'] = anomaly_records[f'{metric}_zscore']

            anomalies.append(anomaly_records[[
                'snapshot_id',
                'facility_id',
                'facility_name',
                'snapshot_date',
                'anomaly_metric',
                metric,
                'anomaly_zscore'
            ]])

        if anomalies:
            anomalies_df = pd.concat(anomalies, ignore_index=True)
            logger.info(f"Detected {len(anomalies_df)} anomalies")
            return anomalies_df
        else:
            return pd.DataFrame()

    # ========================================================================
    # FORECASTING
    # ========================================================================

    def calculate_trend_direction(
        self,
        metric: str,
        days: int = 7
    ) -> Dict[str, any]:
        """
        Calculate trend direction for a metric.

        Args:
            metric: Metric to analyze
            days: Number of days to analyze

        Returns:
            Dictionary with trend analysis
        """
        logger.info(f"Calculating {days}-day trend for '{metric}'")

        # Get recent data
        latest_date = self.df['snapshot_date'].max()
        start_date = latest_date - timedelta(days=days)

        recent = self.df[
            (self.df['snapshot_date'] >= start_date) &
            (self.df['snapshot_date'] <= latest_date)
        ]

        # Daily totals
        daily = recent.groupby('snapshot_date')[metric].sum().reset_index()
        daily = daily.sort_values('snapshot_date')

        if len(daily) < 2:
            return {'error': 'Insufficient data for trend analysis'}

        # Calculate linear regression
        x = np.arange(len(daily))
        y = daily[metric].values

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Determine trend direction
        if slope > 0 and p_value < 0.05:
            direction = "increasing"
        elif slope < 0 and p_value < 0.05:
            direction = "decreasing"
        else:
            direction = "stable"

        return {
            'metric': metric,
            'days_analyzed': days,
            'trend_direction': direction,
            'slope': float(slope),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'start_value': float(y[0]),
            'end_value': float(y[-1]),
            'total_change': float(y[-1] - y[0]),
            'percent_change': float((y[-1] - y[0]) / y[0] * 100) if y[0] != 0 else 0,
        }

    # ========================================================================
    # REPORT GENERATION
    # ========================================================================

    def generate_executive_summary(self) -> Dict[str, any]:
        """
        Generate comprehensive executive summary.

        Returns:
            Dictionary with all key metrics and insights
        """
        logger.info("Generating executive summary...")

        summary = {
            'generated_at': datetime.now(),
            'kpis': self.calculate_system_wide_kpis(),
            'capacity_analysis': self.analyze_capacity_constraints(),
            'outbreaks_detected': len(self.detect_outbreaks()),
            'trends': {
                'covid_positive': self.calculate_trend_direction('covid_positive', days=7),
                'covid_deaths': self.calculate_trend_direction('covid_deaths', days=7),
                'occupancy_rate': self.calculate_trend_direction('occupancy_rate', days=7),
            },
            'facility_type_comparison': self.compare_by_facility_type().to_dict('records'),
        }

        logger.info("Executive summary generated")
        return summary


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from etl_pipeline import FacilityDataETL

    # Run ETL pipeline
    etl = FacilityDataETL()
    etl.run_pipeline(
        ehr_file="data/ehr_data.csv",
        staffing_file="data/staffing_roster.csv",
        ppe_file="data/ppe_inventory.csv",
        output_file="output/merged_data.csv"
    )

    # Initialize analyzer
    analyzer = CovidFacilityAnalyzer(etl.merged_snapshots)

    # Generate executive summary
    summary = analyzer.generate_executive_summary()

    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY - VA COVID-19 Response")
    print("=" * 80)
    print(f"\nReport Date: {summary['kpis']['report_date']}")
    print(f"Facilities Reporting: {summary['kpis']['facilities_reporting']}")
    print(f"\nCOVID-19 Patient Metrics:")
    print(f"  Total Positive: {summary['kpis']['total_covid_positive']:,}")
    print(f"  Hospitalized: {summary['kpis']['total_covid_hospitalized']:,}")
    print(f"  ICU: {summary['kpis']['total_covid_icu']:,}")
    print(f"  Ventilator: {summary['kpis']['total_covid_ventilator']:,}")
    print(f"  Deaths: {summary['kpis']['total_covid_deaths']:,}")
    print(f"\nCapacity:")
    print(f"  Average Occupancy: {summary['kpis']['avg_occupancy_rate']:.1%}")
    print(f"  Facilities with Staff Shortage: {summary['kpis']['facilities_staff_shortage']}")
    print(f"  Facilities with PPE Critical: {summary['kpis']['facilities_ppe_critical']}")
    print(f"\nOutbreaks Detected: {summary['outbreaks_detected']}")
    print("=" * 80)
