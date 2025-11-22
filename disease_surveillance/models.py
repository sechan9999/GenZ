"""
Data Models for Disease Hotspot Detection System
Handles multi-stream surveillance data structures
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd


class DataStreamType(Enum):
    """Types of surveillance data streams"""
    ESSENCE_SYNDROMIC = "essence_syndromic"
    WASTEWATER = "wastewater"
    SEARCH_TRENDS = "search_trends"
    MOBILITY = "mobility"


class AnomalyType(Enum):
    """Types of detected anomalies"""
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    SPATIOTEMPORAL = "spatiotemporal"


class SeverityLevel(Enum):
    """Severity levels for hotspot alerts"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class GeographicLocation:
    """Geographic location data"""
    latitude: float
    longitude: float
    region_name: str
    region_code: str
    population: Optional[int] = None
    zip_code: Optional[str] = None
    county: Optional[str] = None
    state: Optional[str] = None


@dataclass
class TimeSeriesDataPoint:
    """Single time series observation"""
    timestamp: datetime
    value: float
    location: GeographicLocation
    data_source: DataStreamType
    metadata: Dict = field(default_factory=dict)


@dataclass
class ESSENCERecord:
    """ESSENCE syndromic surveillance record"""
    timestamp: datetime
    location: GeographicLocation
    chief_complaint_category: str  # ILI, respiratory, GI, etc.
    visit_count: int
    total_visits: int
    percentage: float
    age_group: Optional[str] = None
    sex: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class WastewaterRecord:
    """Wastewater viral load measurement"""
    timestamp: datetime
    location: GeographicLocation
    viral_load: float  # copies per liter
    pathogen: str  # SARS-CoV-2, Influenza, Norovirus, etc.
    collection_site_id: str
    flow_rate: Optional[float] = None
    population_served: Optional[int] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class SearchTrendsRecord:
    """Google/Search trends data"""
    timestamp: datetime
    location: GeographicLocation
    search_term: str
    normalized_interest: float  # 0-100 scale
    related_queries: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class MobilityRecord:
    """Mobility/movement data"""
    timestamp: datetime
    location: GeographicLocation
    mobility_index: float
    baseline_comparison: float  # % change from baseline
    category: str  # retail, transit, workplaces, residential, etc.
    metadata: Dict = field(default_factory=dict)


@dataclass
class STLDecomposition:
    """STL decomposition results"""
    trend: np.ndarray
    seasonal: np.ndarray
    residual: np.ndarray
    timestamps: List[datetime]
    original: np.ndarray


@dataclass
class TemporalAnomaly:
    """Detected temporal anomaly"""
    timestamp: datetime
    location: GeographicLocation
    observed_value: float
    expected_value: float
    modified_z_score: float
    is_anomaly: bool
    data_source: DataStreamType
    confidence: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class SpatialCluster:
    """Spatial cluster detected by SaTScan"""
    cluster_id: str
    center_location: GeographicLocation
    radius_km: float
    locations: List[GeographicLocation]
    start_date: datetime
    end_date: datetime
    observed_cases: int
    expected_cases: float
    relative_risk: float
    p_value: float
    log_likelihood_ratio: float


@dataclass
class Hotspot:
    """Identified disease hotspot"""
    hotspot_id: str
    detection_time: datetime
    location: GeographicLocation
    severity: SeverityLevel
    anomaly_type: AnomalyType

    # Multi-stream signals
    essence_signal: Optional[TemporalAnomaly] = None
    wastewater_signal: Optional[TemporalAnomaly] = None
    search_trends_signal: Optional[TemporalAnomaly] = None
    mobility_signal: Optional[TemporalAnomaly] = None

    # Spatial clustering
    spatial_cluster: Optional[SpatialCluster] = None

    # Cross-validation
    num_confirming_signals: int = 0
    signal_agreement_score: float = 0.0

    # Risk assessment
    estimated_cases: Optional[int] = None
    population_at_risk: Optional[int] = None
    trend_direction: str = "unknown"  # increasing, decreasing, stable

    # Metadata
    description: str = ""
    recommended_actions: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class SurveillanceReport:
    """Comprehensive surveillance report"""
    report_id: str
    generation_time: datetime
    time_period_start: datetime
    time_period_end: datetime

    # Detected hotspots
    hotspots: List[Hotspot]

    # Data summary
    total_records_processed: Dict[DataStreamType, int]
    temporal_anomalies_detected: int
    spatial_clusters_detected: int

    # System metrics
    data_quality_scores: Dict[DataStreamType, float]
    coverage_by_region: Dict[str, float]

    # Alerts
    critical_alerts: List[Hotspot]
    high_priority_alerts: List[Hotspot]

    metadata: Dict = field(default_factory=dict)


class SurveillanceDataFrame:
    """
    Unified DataFrame wrapper for multi-stream surveillance data
    Provides convenience methods for data manipulation and analysis
    """

    def __init__(self):
        self.essence_data: Optional[pd.DataFrame] = None
        self.wastewater_data: Optional[pd.DataFrame] = None
        self.search_trends_data: Optional[pd.DataFrame] = None
        self.mobility_data: Optional[pd.DataFrame] = None

    def load_essence_data(self, records: List[ESSENCERecord]) -> pd.DataFrame:
        """Convert ESSENCE records to DataFrame"""
        data = []
        for record in records:
            data.append({
                'timestamp': record.timestamp,
                'latitude': record.location.latitude,
                'longitude': record.location.longitude,
                'region_name': record.location.region_name,
                'region_code': record.location.region_code,
                'chief_complaint': record.chief_complaint_category,
                'visit_count': record.visit_count,
                'total_visits': record.total_visits,
                'percentage': record.percentage,
                'age_group': record.age_group,
                'sex': record.sex
            })
        self.essence_data = pd.DataFrame(data)
        return self.essence_data

    def load_wastewater_data(self, records: List[WastewaterRecord]) -> pd.DataFrame:
        """Convert wastewater records to DataFrame"""
        data = []
        for record in records:
            data.append({
                'timestamp': record.timestamp,
                'latitude': record.location.latitude,
                'longitude': record.location.longitude,
                'region_name': record.location.region_name,
                'region_code': record.location.region_code,
                'viral_load': record.viral_load,
                'pathogen': record.pathogen,
                'collection_site_id': record.collection_site_id,
                'flow_rate': record.flow_rate,
                'population_served': record.population_served
            })
        self.wastewater_data = pd.DataFrame(data)
        return self.wastewater_data

    def load_search_trends_data(self, records: List[SearchTrendsRecord]) -> pd.DataFrame:
        """Convert search trends records to DataFrame"""
        data = []
        for record in records:
            data.append({
                'timestamp': record.timestamp,
                'latitude': record.location.latitude,
                'longitude': record.location.longitude,
                'region_name': record.location.region_name,
                'region_code': record.location.region_code,
                'search_term': record.search_term,
                'normalized_interest': record.normalized_interest
            })
        self.search_trends_data = pd.DataFrame(data)
        return self.search_trends_data

    def load_mobility_data(self, records: List[MobilityRecord]) -> pd.DataFrame:
        """Convert mobility records to DataFrame"""
        data = []
        for record in records:
            data.append({
                'timestamp': record.timestamp,
                'latitude': record.location.latitude,
                'longitude': record.location.longitude,
                'region_name': record.location.region_name,
                'region_code': record.location.region_code,
                'mobility_index': record.mobility_index,
                'baseline_comparison': record.baseline_comparison,
                'category': record.category
            })
        self.mobility_data = pd.DataFrame(data)
        return self.mobility_data

    def get_unified_timeseries(self, region_code: str) -> pd.DataFrame:
        """
        Get unified time series for a specific region across all data streams
        """
        unified = pd.DataFrame()

        if self.essence_data is not None:
            essence = self.essence_data[
                self.essence_data['region_code'] == region_code
            ][['timestamp', 'percentage']].rename(columns={'percentage': 'essence_signal'})
            unified = essence if unified.empty else unified.merge(essence, on='timestamp', how='outer')

        if self.wastewater_data is not None:
            wastewater = self.wastewater_data[
                self.wastewater_data['region_code'] == region_code
            ][['timestamp', 'viral_load']].rename(columns={'viral_load': 'wastewater_signal'})
            unified = wastewater if unified.empty else unified.merge(wastewater, on='timestamp', how='outer')

        if self.search_trends_data is not None:
            search = self.search_trends_data[
                self.search_trends_data['region_code'] == region_code
            ][['timestamp', 'normalized_interest']].rename(columns={'normalized_interest': 'search_signal'})
            unified = search if unified.empty else unified.merge(search, on='timestamp', how='outer')

        if self.mobility_data is not None:
            mobility = self.mobility_data[
                self.mobility_data['region_code'] == region_code
            ][['timestamp', 'baseline_comparison']].rename(columns={'baseline_comparison': 'mobility_signal'})
            unified = mobility if unified.empty else unified.merge(mobility, on='timestamp', how='outer')

        return unified.sort_values('timestamp')
