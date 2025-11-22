"""
Disease Hotspot Detection System
Multi-stream surveillance using STL decomposition, modified Z-scores, and SaTScan clustering
"""

from .models import (
    DataStreamType,
    AnomalyType,
    SeverityLevel,
    GeographicLocation,
    TimeSeriesDataPoint,
    ESSENCERecord,
    WastewaterRecord,
    SearchTrendsRecord,
    MobilityRecord,
    STLDecomposition,
    TemporalAnomaly,
    SpatialCluster,
    Hotspot,
    SurveillanceReport,
    SurveillanceDataFrame
)

from .temporal_detection import (
    TemporalAnomalyDetector,
    MultiStreamAnomalyDetector,
    align_temporal_anomalies
)

from .spatial_clustering import (
    SaTScanIntegration,
    SimplifiedSpatialClustering
)

from .cross_validation import SignalCrossValidator

from .data_ingestion import (
    ESSENCEIngester,
    WastewaterIngester,
    SearchTrendsIngester,
    MobilityIngester,
    MultiStreamDataIngester
)

from .pipeline import DiseaseHotspotPipeline

__version__ = "1.0.0"

__all__ = [
    # Models
    "DataStreamType",
    "AnomalyType",
    "SeverityLevel",
    "GeographicLocation",
    "TimeSeriesDataPoint",
    "ESSENCERecord",
    "WastewaterRecord",
    "SearchTrendsRecord",
    "MobilityRecord",
    "STLDecomposition",
    "TemporalAnomaly",
    "SpatialCluster",
    "Hotspot",
    "SurveillanceReport",
    "SurveillanceDataFrame",

    # Temporal Detection
    "TemporalAnomalyDetector",
    "MultiStreamAnomalyDetector",
    "align_temporal_anomalies",

    # Spatial Clustering
    "SaTScanIntegration",
    "SimplifiedSpatialClustering",

    # Cross Validation
    "SignalCrossValidator",

    # Data Ingestion
    "ESSENCEIngester",
    "WastewaterIngester",
    "SearchTrendsIngester",
    "MobilityIngester",
    "MultiStreamDataIngester",

    # Pipeline
    "DiseaseHotspotPipeline"
]
