"""
Main Disease Hotspot Detection Pipeline
Orchestrates data ingestion, temporal detection, spatial clustering, and cross-validation
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import logging
import json

from .models import (
    DataStreamType,
    TemporalAnomaly,
    SpatialCluster,
    Hotspot,
    SurveillanceReport,
    SurveillanceDataFrame,
    SeverityLevel
)
from .data_ingestion import MultiStreamDataIngester
from .temporal_detection import (
    TemporalAnomalyDetector,
    MultiStreamAnomalyDetector,
    align_temporal_anomalies
)
from .spatial_clustering import SaTScanIntegration, SimplifiedSpatialClustering
from .cross_validation import SignalCrossValidator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiseaseHotspotPipeline:
    """
    End-to-end pipeline for disease hotspot detection

    Workflow:
    1. Ingest multi-stream data (ESSENCE, wastewater, search trends, mobility)
    2. Apply STL + modified Z-score temporal anomaly detection
    3. Perform SaTScan spatiotemporal clustering
    4. Cross-validate signals across streams
    5. Generate hotspot alerts and reports
    """

    def __init__(
        self,
        # Temporal detection parameters
        stl_seasonal_period: int = 7,
        modified_zscore_threshold: float = 3.5,
        min_observations: int = 28,

        # Spatial clustering parameters
        use_satscan: bool = False,
        satscan_executable: Optional[str] = None,
        max_cluster_size: float = 0.5,
        spatial_radius_km: float = 50,

        # Cross-validation parameters
        min_confirming_signals: int = 2,
        essence_weight: float = 1.0,
        wastewater_weight: float = 0.9,
        search_trends_weight: float = 0.6,
        mobility_weight: float = 0.5,

        # Output configuration
        output_dir: Optional[Path] = None
    ):
        """
        Initialize disease hotspot detection pipeline

        Args:
            stl_seasonal_period: Period for seasonal decomposition (default: 7 days)
            modified_zscore_threshold: Threshold for anomaly detection (default: 3.5)
            min_observations: Minimum observations required for STL (default: 28)
            use_satscan: Whether to use SaTScan (requires separate installation)
            satscan_executable: Path to SaTScan executable
            max_cluster_size: Maximum spatial cluster size as % of population
            spatial_radius_km: Radius for simplified spatial clustering
            min_confirming_signals: Minimum signals to confirm hotspot
            essence_weight: Weight for ESSENCE signal (0-1)
            wastewater_weight: Weight for wastewater signal (0-1)
            search_trends_weight: Weight for search trends signal (0-1)
            mobility_weight: Weight for mobility signal (0-1)
            output_dir: Directory for output files
        """
        # Initialize data ingestion
        self.data_ingester = MultiStreamDataIngester()

        # Initialize temporal detection
        self.temporal_detector = MultiStreamAnomalyDetector(
            essence_detector=TemporalAnomalyDetector(
                seasonal_period=stl_seasonal_period,
                threshold=modified_zscore_threshold,
                min_observations=min_observations
            ),
            wastewater_detector=TemporalAnomalyDetector(
                seasonal_period=stl_seasonal_period,
                threshold=modified_zscore_threshold * 0.85,  # More sensitive
                min_observations=min_observations
            ),
            search_detector=TemporalAnomalyDetector(
                seasonal_period=stl_seasonal_period,
                threshold=modified_zscore_threshold,
                min_observations=min_observations
            ),
            mobility_detector=TemporalAnomalyDetector(
                seasonal_period=stl_seasonal_period,
                threshold=modified_zscore_threshold * 1.15,  # Less sensitive
                min_observations=min_observations
            )
        )

        # Initialize spatial clustering
        self.use_satscan = use_satscan
        if use_satscan:
            self.spatial_clusterer = SaTScanIntegration(
                satscan_executable=satscan_executable,
                max_cluster_size=max_cluster_size,
                min_cases=5
            )
        else:
            self.spatial_clusterer = SimplifiedSpatialClustering(
                max_distance_km=spatial_radius_km,
                min_samples=3
            )

        # Initialize cross-validation
        self.cross_validator = SignalCrossValidator(
            essence_weight=essence_weight,
            wastewater_weight=wastewater_weight,
            search_trends_weight=search_trends_weight,
            mobility_weight=mobility_weight,
            min_confirming_signals=min_confirming_signals,
            spatial_radius_km=spatial_radius_km
        )

        # Output configuration
        self.output_dir = output_dir or Path("./surveillance_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        essence_path: Optional[Path] = None,
        wastewater_path: Optional[Path] = None,
        search_trends_path: Optional[Path] = None,
        mobility_path: Optional[Path] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> SurveillanceReport:
        """
        Run the complete disease hotspot detection pipeline

        Args:
            essence_path: Path to ESSENCE syndromic data
            wastewater_path: Path to wastewater viral load data
            search_trends_path: Path to search trends data
            mobility_path: Path to mobility data
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            SurveillanceReport with detected hotspots
        """
        logger.info("="*80)
        logger.info("DISEASE HOTSPOT DETECTION PIPELINE")
        logger.info("="*80)

        # Set default date range
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=90)

        logger.info(f"Analysis period: {start_date.date()} to {end_date.date()}")

        # Step 1: Ingest multi-stream data
        logger.info("\n[Step 1/5] Ingesting multi-stream surveillance data...")
        data_streams = self.data_ingester.ingest_all(
            essence_path=essence_path,
            wastewater_path=wastewater_path,
            search_trends_path=search_trends_path,
            mobility_path=mobility_path
        )

        logger.info(f"  Loaded {len(data_streams)} data streams:")
        for stream_type, df in data_streams.items():
            logger.info(f"    - {stream_type.value}: {len(df)} records")

        if not data_streams:
            logger.warning("No data streams loaded. Exiting.")
            return self._create_empty_report(start_date, end_date)

        # Filter by date range
        data_streams = self._filter_by_date_range(data_streams, start_date, end_date)

        # Step 2: Detect temporal anomalies
        logger.info("\n[Step 2/5] Detecting temporal anomalies using STL + modified Z-scores...")
        all_anomalies = self._detect_temporal_anomalies(data_streams)

        essence_anomalies, wastewater_anomalies, search_anomalies, mobility_anomalies = all_anomalies

        logger.info(f"  Temporal anomalies detected:")
        logger.info(f"    - ESSENCE: {len(essence_anomalies)}")
        logger.info(f"    - Wastewater: {len(wastewater_anomalies)}")
        logger.info(f"    - Search Trends: {len(search_anomalies)}")
        logger.info(f"    - Mobility: {len(mobility_anomalies)}")

        total_anomalies = sum(len(a) for a in all_anomalies)

        if total_anomalies == 0:
            logger.info("No temporal anomalies detected.")
            return self._create_empty_report(start_date, end_date, data_streams)

        # Step 3: Perform spatial clustering
        logger.info("\n[Step 3/5] Performing spatiotemporal clustering...")
        spatial_clusters = self._detect_spatial_clusters(
            essence_anomalies,
            wastewater_anomalies,
            search_anomalies,
            mobility_anomalies,
            start_date,
            end_date
        )

        logger.info(f"  Spatial clusters detected: {len(spatial_clusters)}")

        # Step 4: Cross-validate signals and create hotspots
        logger.info("\n[Step 4/5] Cross-validating signals across data streams...")
        hotspots = self.cross_validator.validate_and_create_hotspots(
            essence_anomalies,
            wastewater_anomalies,
            search_anomalies,
            mobility_anomalies,
            spatial_clusters
        )

        logger.info(f"  Validated hotspots: {len(hotspots)}")

        # Categorize by severity
        critical = [h for h in hotspots if h.severity == SeverityLevel.CRITICAL]
        high = [h for h in hotspots if h.severity == SeverityLevel.HIGH]
        moderate = [h for h in hotspots if h.severity == SeverityLevel.MODERATE]
        low = [h for h in hotspots if h.severity == SeverityLevel.LOW]

        logger.info(f"    - Critical: {len(critical)}")
        logger.info(f"    - High: {len(high)}")
        logger.info(f"    - Moderate: {len(moderate)}")
        logger.info(f"    - Low: {len(low)}")

        # Step 5: Generate report
        logger.info("\n[Step 5/5] Generating surveillance report...")
        report = self._create_report(
            hotspots,
            data_streams,
            total_anomalies,
            len(spatial_clusters),
            start_date,
            end_date
        )

        # Save report
        self._save_report(report)

        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*80)

        return report

    def _filter_by_date_range(
        self,
        data_streams: Dict[DataStreamType, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[DataStreamType, pd.DataFrame]:
        """Filter data streams by date range"""
        filtered = {}

        for stream_type, df in data_streams.items():
            if 'timestamp' in df.columns:
                mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
                filtered[stream_type] = df[mask].copy()
            else:
                filtered[stream_type] = df.copy()

        return filtered

    def _detect_temporal_anomalies(
        self,
        data_streams: Dict[DataStreamType, pd.DataFrame]
    ) -> Tuple[List[TemporalAnomaly], List[TemporalAnomaly], List[TemporalAnomaly], List[TemporalAnomaly]]:
        """Detect temporal anomalies across all streams"""
        essence_df = data_streams.get(DataStreamType.ESSENCE_SYNDROMIC)
        wastewater_df = data_streams.get(DataStreamType.WASTEWATER)
        search_df = data_streams.get(DataStreamType.SEARCH_TRENDS)
        mobility_df = data_streams.get(DataStreamType.MOBILITY)

        # Get unique regions
        all_regions = set()
        for df in [essence_df, wastewater_df, search_df, mobility_df]:
            if df is not None and 'region_code' in df.columns:
                all_regions.update(df['region_code'].unique())

        # Detect anomalies per region
        all_essence_anomalies = []
        all_wastewater_anomalies = []
        all_search_anomalies = []
        all_mobility_anomalies = []

        for region_code in all_regions:
            # Filter data for this region
            region_essence = essence_df[essence_df['region_code'] == region_code] if essence_df is not None else None
            region_wastewater = wastewater_df[wastewater_df['region_code'] == region_code] if wastewater_df is not None else None
            region_search = search_df[search_df['region_code'] == region_code] if search_df is not None else None
            region_mobility = mobility_df[mobility_df['region_code'] == region_code] if mobility_df is not None else None

            # Get location from first available record
            location = self._extract_location(
                region_essence, region_wastewater, region_search, region_mobility, region_code
            )

            # Detect anomalies
            try:
                e_anom, w_anom, s_anom, m_anom = self.temporal_detector.detect_all_streams(
                    region_essence,
                    region_wastewater,
                    region_search,
                    region_mobility,
                    location
                )

                all_essence_anomalies.extend(e_anom)
                all_wastewater_anomalies.extend(w_anom)
                all_search_anomalies.extend(s_anom)
                all_mobility_anomalies.extend(m_anom)

            except Exception as e:
                logger.warning(f"Error detecting anomalies for region {region_code}: {str(e)}")
                continue

        return (
            all_essence_anomalies,
            all_wastewater_anomalies,
            all_search_anomalies,
            all_mobility_anomalies
        )

    def _extract_location(
        self,
        essence_df: Optional[pd.DataFrame],
        wastewater_df: Optional[pd.DataFrame],
        search_df: Optional[pd.DataFrame],
        mobility_df: Optional[pd.DataFrame],
        region_code: str
    ):
        """Extract geographic location from available data"""
        from .models import GeographicLocation

        for df in [essence_df, wastewater_df, search_df, mobility_df]:
            if df is not None and not df.empty:
                row = df.iloc[0]
                return GeographicLocation(
                    latitude=row.get('latitude', 0.0),
                    longitude=row.get('longitude', 0.0),
                    region_name=row.get('region_name', region_code),
                    region_code=region_code,
                    population=row.get('population')
                )

        # Default location
        return GeographicLocation(
            latitude=0.0,
            longitude=0.0,
            region_name=region_code,
            region_code=region_code
        )

    def _detect_spatial_clusters(
        self,
        essence_anomalies: List[TemporalAnomaly],
        wastewater_anomalies: List[TemporalAnomaly],
        search_anomalies: List[TemporalAnomaly],
        mobility_anomalies: List[TemporalAnomaly],
        start_date: datetime,
        end_date: datetime
    ) -> List[SpatialCluster]:
        """Detect spatial clusters from anomalies"""
        # Combine all anomalies
        all_anomalies = (
            essence_anomalies +
            wastewater_anomalies +
            search_anomalies +
            mobility_anomalies
        )

        if not all_anomalies:
            return []

        # Use simplified clustering (SaTScan would require more complex setup)
        if not self.use_satscan:
            return self.spatial_clusterer.detect_clusters(all_anomalies)
        else:
            # Would need to convert anomalies to SaTScan format
            # This is more complex and requires actual SaTScan installation
            logger.warning("SaTScan integration requires additional setup. Using simplified clustering.")
            simplified_clusterer = SimplifiedSpatialClustering()
            return simplified_clusterer.detect_clusters(all_anomalies)

    def _create_report(
        self,
        hotspots: List[Hotspot],
        data_streams: Dict[DataStreamType, pd.DataFrame],
        total_anomalies: int,
        spatial_clusters_count: int,
        start_date: datetime,
        end_date: datetime
    ) -> SurveillanceReport:
        """Create surveillance report"""
        report_id = f"surveillance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Calculate data quality scores
        data_quality_scores = {}
        for stream_type, df in data_streams.items():
            # Simple quality score based on completeness
            if not df.empty:
                completeness = df.notna().mean().mean()
                data_quality_scores[stream_type] = completeness
            else:
                data_quality_scores[stream_type] = 0.0

        # Get critical and high priority alerts
        critical_alerts = [h for h in hotspots if h.severity == SeverityLevel.CRITICAL]
        high_alerts = [h for h in hotspots if h.severity == SeverityLevel.HIGH]

        # Create report
        report = SurveillanceReport(
            report_id=report_id,
            generation_time=datetime.now(),
            time_period_start=start_date,
            time_period_end=end_date,
            hotspots=hotspots,
            total_records_processed={
                stream_type: len(df) for stream_type, df in data_streams.items()
            },
            temporal_anomalies_detected=total_anomalies,
            spatial_clusters_detected=spatial_clusters_count,
            data_quality_scores=data_quality_scores,
            coverage_by_region={},  # Would need to calculate
            critical_alerts=critical_alerts,
            high_priority_alerts=high_alerts
        )

        return report

    def _create_empty_report(
        self,
        start_date: datetime,
        end_date: datetime,
        data_streams: Optional[Dict[DataStreamType, pd.DataFrame]] = None
    ) -> SurveillanceReport:
        """Create empty report when no data or anomalies"""
        return SurveillanceReport(
            report_id=f"surveillance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generation_time=datetime.now(),
            time_period_start=start_date,
            time_period_end=end_date,
            hotspots=[],
            total_records_processed={s: len(df) for s, df in data_streams.items()} if data_streams else {},
            temporal_anomalies_detected=0,
            spatial_clusters_detected=0,
            data_quality_scores={},
            coverage_by_region={},
            critical_alerts=[],
            high_priority_alerts=[]
        )

    def _save_report(self, report: SurveillanceReport):
        """Save report to files"""
        # Save JSON report
        json_path = self.output_dir / f"{report.report_id}.json"
        with open(json_path, 'w') as f:
            json.dump(self._report_to_dict(report), f, indent=2, default=str)

        logger.info(f"  Report saved to: {json_path}")

        # Save summary markdown
        md_path = self.output_dir / f"{report.report_id}_summary.md"
        with open(md_path, 'w') as f:
            f.write(self._generate_markdown_summary(report))

        logger.info(f"  Summary saved to: {md_path}")

    def _report_to_dict(self, report: SurveillanceReport) -> dict:
        """Convert report to dictionary for JSON serialization"""
        return {
            'report_id': report.report_id,
            'generation_time': report.generation_time.isoformat(),
            'time_period_start': report.time_period_start.isoformat(),
            'time_period_end': report.time_period_end.isoformat(),
            'summary': {
                'total_hotspots': len(report.hotspots),
                'critical_alerts': len(report.critical_alerts),
                'high_priority_alerts': len(report.high_priority_alerts),
                'temporal_anomalies': report.temporal_anomalies_detected,
                'spatial_clusters': report.spatial_clusters_detected
            },
            'hotspots': [
                {
                    'hotspot_id': h.hotspot_id,
                    'severity': h.severity.value,
                    'location': {
                        'region_name': h.location.region_name,
                        'region_code': h.location.region_code,
                        'latitude': h.location.latitude,
                        'longitude': h.location.longitude
                    },
                    'detection_time': h.detection_time.isoformat(),
                    'num_confirming_signals': h.num_confirming_signals,
                    'signal_agreement_score': h.signal_agreement_score,
                    'trend_direction': h.trend_direction,
                    'description': h.description,
                    'recommended_actions': h.recommended_actions
                }
                for h in report.hotspots
            ]
        }

    def _generate_markdown_summary(self, report: SurveillanceReport) -> str:
        """Generate markdown summary"""
        md = f"""# Disease Surveillance Report

**Report ID**: {report.report_id}
**Generated**: {report.generation_time.strftime('%Y-%m-%d %H:%M:%S')}
**Period**: {report.time_period_start.date()} to {report.time_period_end.date()}

## Executive Summary

- **Total Hotspots Detected**: {len(report.hotspots)}
- **Critical Alerts**: {len(report.critical_alerts)}
- **High Priority Alerts**: {len(report.high_priority_alerts)}
- **Temporal Anomalies**: {report.temporal_anomalies_detected}
- **Spatial Clusters**: {report.spatial_clusters_detected}

## Critical Alerts

"""
        if report.critical_alerts:
            for hotspot in report.critical_alerts:
                md += f"""
### {hotspot.location.region_name} ({hotspot.hotspot_id})

- **Severity**: {hotspot.severity.value.upper()}
- **Confirming Signals**: {hotspot.num_confirming_signals}
- **Agreement Score**: {hotspot.signal_agreement_score:.2%}
- **Trend**: {hotspot.trend_direction}
- **Description**: {hotspot.description}

**Recommended Actions**:
"""
                for action in hotspot.recommended_actions:
                    md += f"- {action}\n"
        else:
            md += "No critical alerts.\n"

        md += "\n## High Priority Alerts\n\n"

        if report.high_priority_alerts:
            for hotspot in report.high_priority_alerts:
                md += f"""
### {hotspot.location.region_name} ({hotspot.hotspot_id})

- **Confirming Signals**: {hotspot.num_confirming_signals}
- **Agreement Score**: {hotspot.signal_agreement_score:.2%}
- **Trend**: {hotspot.trend_direction}
"""
        else:
            md += "No high priority alerts.\n"

        return md
