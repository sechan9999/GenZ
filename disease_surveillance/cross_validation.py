"""
Cross-Validation of Multiple Data Streams
Combines signals from ESSENCE, wastewater, search trends, and mobility data
to validate and strengthen hotspot detections
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import warnings

from .models import (
    TemporalAnomaly,
    SpatialCluster,
    Hotspot,
    GeographicLocation,
    DataStreamType,
    AnomalyType,
    SeverityLevel
)


class SignalCrossValidator:
    """
    Cross-validates disease outbreak signals across multiple data streams

    Implements a weighted scoring system where multiple confirming signals
    increase confidence in hotspot detection
    """

    def __init__(
        self,
        essence_weight: float = 1.0,  # Primary signal
        wastewater_weight: float = 0.9,  # High confidence early indicator
        search_trends_weight: float = 0.6,  # Moderate confidence
        mobility_weight: float = 0.5,  # Supporting indicator
        min_confirming_signals: int = 2,  # Minimum signals to confirm hotspot
        time_alignment_window_days: int = 7,  # Window for temporal alignment
        spatial_radius_km: float = 50  # Radius for spatial alignment
    ):
        """
        Initialize signal cross-validator

        Args:
            essence_weight: Weight for ESSENCE syndromic signal (0-1)
            wastewater_weight: Weight for wastewater viral load signal (0-1)
            search_trends_weight: Weight for search trends signal (0-1)
            mobility_weight: Weight for mobility signal (0-1)
            min_confirming_signals: Minimum number of signals to confirm hotspot
            time_alignment_window_days: Days within which signals are considered aligned
            spatial_radius_km: Distance within which signals are spatially aligned
        """
        self.weights = {
            DataStreamType.ESSENCE_SYNDROMIC: essence_weight,
            DataStreamType.WASTEWATER: wastewater_weight,
            DataStreamType.SEARCH_TRENDS: search_trends_weight,
            DataStreamType.MOBILITY: mobility_weight
        }
        self.min_confirming_signals = min_confirming_signals
        self.time_alignment_window = timedelta(days=time_alignment_window_days)
        self.spatial_radius_km = spatial_radius_km

    def haversine_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """Calculate distance between two points in km"""
        R = 6371  # Earth radius in km

        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    def are_spatially_aligned(
        self,
        loc1: GeographicLocation,
        loc2: GeographicLocation
    ) -> bool:
        """Check if two locations are within spatial alignment radius"""
        distance = self.haversine_distance(
            loc1.latitude,
            loc1.longitude,
            loc2.latitude,
            loc2.longitude
        )
        return distance <= self.spatial_radius_km

    def are_temporally_aligned(
        self,
        time1: datetime,
        time2: datetime
    ) -> bool:
        """Check if two timestamps are within temporal alignment window"""
        return abs(time1 - time2) <= self.time_alignment_window

    def group_anomalies_by_location(
        self,
        essence_anomalies: List[TemporalAnomaly],
        wastewater_anomalies: List[TemporalAnomaly],
        search_trends_anomalies: List[TemporalAnomaly],
        mobility_anomalies: List[TemporalAnomaly]
    ) -> Dict[str, Dict[DataStreamType, List[TemporalAnomaly]]]:
        """
        Group anomalies by region code

        Returns:
            Dictionary mapping region_code to dict of anomalies by stream type
        """
        grouped = defaultdict(lambda: {
            DataStreamType.ESSENCE_SYNDROMIC: [],
            DataStreamType.WASTEWATER: [],
            DataStreamType.SEARCH_TRENDS: [],
            DataStreamType.MOBILITY: []
        })

        for anomaly in essence_anomalies:
            grouped[anomaly.location.region_code][DataStreamType.ESSENCE_SYNDROMIC].append(anomaly)

        for anomaly in wastewater_anomalies:
            grouped[anomaly.location.region_code][DataStreamType.WASTEWATER].append(anomaly)

        for anomaly in search_trends_anomalies:
            grouped[anomaly.location.region_code][DataStreamType.SEARCH_TRENDS].append(anomaly)

        for anomaly in mobility_anomalies:
            grouped[anomaly.location.region_code][DataStreamType.MOBILITY].append(anomaly)

        return dict(grouped)

    def find_aligned_signals(
        self,
        anomalies_by_stream: Dict[DataStreamType, List[TemporalAnomaly]]
    ) -> List[Dict[DataStreamType, TemporalAnomaly]]:
        """
        Find temporally aligned anomalies across data streams

        Returns:
            List of aligned signal groups (one group per time window)
        """
        aligned_groups = []

        # Use ESSENCE as anchor if available, otherwise use first available stream
        anchor_stream = None
        anchor_anomalies = []

        if anomalies_by_stream[DataStreamType.ESSENCE_SYNDROMIC]:
            anchor_stream = DataStreamType.ESSENCE_SYNDROMIC
            anchor_anomalies = anomalies_by_stream[DataStreamType.ESSENCE_SYNDROMIC]
        else:
            # Find first non-empty stream
            for stream_type, anomalies in anomalies_by_stream.items():
                if anomalies:
                    anchor_stream = stream_type
                    anchor_anomalies = anomalies
                    break

        if not anchor_anomalies:
            return []

        # For each anchor anomaly, find aligned signals from other streams
        for anchor in anchor_anomalies:
            aligned = {anchor_stream: anchor}

            # Check each other stream
            for stream_type, anomalies in anomalies_by_stream.items():
                if stream_type == anchor_stream:
                    continue

                # Find temporally aligned anomaly
                for anomaly in anomalies:
                    if self.are_temporally_aligned(anchor.timestamp, anomaly.timestamp):
                        if self.are_spatially_aligned(anchor.location, anomaly.location):
                            # Take the strongest signal if multiple match
                            if stream_type not in aligned:
                                aligned[stream_type] = anomaly
                            elif anomaly.modified_z_score > aligned[stream_type].modified_z_score:
                                aligned[stream_type] = anomaly

            aligned_groups.append(aligned)

        return aligned_groups

    def calculate_agreement_score(
        self,
        aligned_signals: Dict[DataStreamType, TemporalAnomaly]
    ) -> Tuple[float, int]:
        """
        Calculate signal agreement score

        Returns:
            Tuple of (agreement_score, num_confirming_signals)
        """
        if not aligned_signals:
            return 0.0, 0

        # Calculate weighted agreement score
        total_weight = 0.0
        num_signals = len(aligned_signals)

        for stream_type, anomaly in aligned_signals.items():
            weight = self.weights.get(stream_type, 0.0)
            confidence = anomaly.confidence
            total_weight += weight * confidence

        # Normalize by maximum possible weight
        max_possible_weight = sum(self.weights.values())
        agreement_score = total_weight / max_possible_weight if max_possible_weight > 0 else 0.0

        return agreement_score, num_signals

    def assess_severity(
        self,
        num_signals: int,
        agreement_score: float,
        max_z_score: float,
        spatial_cluster: Optional[SpatialCluster]
    ) -> SeverityLevel:
        """
        Assess hotspot severity based on multiple factors

        Args:
            num_signals: Number of confirming signals
            agreement_score: Signal agreement score
            max_z_score: Maximum modified Z-score across signals
            spatial_cluster: Associated spatial cluster if any

        Returns:
            Severity level
        """
        severity_score = 0

        # Factor 1: Number of confirming signals
        if num_signals >= 4:
            severity_score += 3
        elif num_signals >= 3:
            severity_score += 2
        elif num_signals >= 2:
            severity_score += 1

        # Factor 2: Agreement score
        if agreement_score >= 0.8:
            severity_score += 3
        elif agreement_score >= 0.6:
            severity_score += 2
        elif agreement_score >= 0.4:
            severity_score += 1

        # Factor 3: Z-score magnitude
        if max_z_score >= 5.0:
            severity_score += 3
        elif max_z_score >= 4.0:
            severity_score += 2
        elif max_z_score >= 3.5:
            severity_score += 1

        # Factor 4: Spatial cluster
        if spatial_cluster:
            if spatial_cluster.relative_risk >= 3.0:
                severity_score += 3
            elif spatial_cluster.relative_risk >= 2.0:
                severity_score += 2
            elif spatial_cluster.relative_risk >= 1.5:
                severity_score += 1

        # Map score to severity level
        if severity_score >= 9:
            return SeverityLevel.CRITICAL
        elif severity_score >= 6:
            return SeverityLevel.HIGH
        elif severity_score >= 3:
            return SeverityLevel.MODERATE
        else:
            return SeverityLevel.LOW

    def determine_trend_direction(
        self,
        aligned_signals: Dict[DataStreamType, TemporalAnomaly]
    ) -> str:
        """
        Determine trend direction from signals

        Returns:
            "increasing", "decreasing", or "stable"
        """
        if not aligned_signals:
            return "unknown"

        # Check if observed > expected for majority of signals
        increasing_count = 0
        decreasing_count = 0

        for anomaly in aligned_signals.values():
            if anomaly.observed_value > anomaly.expected_value:
                increasing_count += 1
            elif anomaly.observed_value < anomaly.expected_value:
                decreasing_count += 1

        if increasing_count > decreasing_count:
            return "increasing"
        elif decreasing_count > increasing_count:
            return "decreasing"
        else:
            return "stable"

    def generate_recommended_actions(
        self,
        severity: SeverityLevel,
        num_signals: int,
        trend_direction: str
    ) -> List[str]:
        """Generate recommended public health actions"""
        actions = []

        if severity == SeverityLevel.CRITICAL:
            actions.extend([
                "Activate emergency response protocols",
                "Deploy mobile testing units to affected area",
                "Increase hospital capacity and supplies",
                "Issue public health advisory",
                "Coordinate with local health departments"
            ])
        elif severity == SeverityLevel.HIGH:
            actions.extend([
                "Enhance surveillance in affected area",
                "Increase testing availability",
                "Notify healthcare providers",
                "Prepare public health communication",
                "Monitor hospital capacity"
            ])
        elif severity == SeverityLevel.MODERATE:
            actions.extend([
                "Continue enhanced monitoring",
                "Review and update response plans",
                "Coordinate with regional partners",
                "Prepare communication materials"
            ])
        else:
            actions.extend([
                "Maintain routine surveillance",
                "Document findings for trend analysis"
            ])

        if trend_direction == "increasing":
            actions.append("Monitor closely for escalation")

        if num_signals >= 3:
            actions.append("High confidence detection - prioritize investigation")

        return actions

    def create_hotspot(
        self,
        hotspot_id: str,
        aligned_signals: Dict[DataStreamType, TemporalAnomaly],
        spatial_cluster: Optional[SpatialCluster] = None
    ) -> Hotspot:
        """
        Create hotspot from aligned signals

        Args:
            hotspot_id: Unique identifier
            aligned_signals: Aligned anomalies across streams
            spatial_cluster: Associated spatial cluster

        Returns:
            Hotspot object
        """
        if not aligned_signals:
            raise ValueError("Cannot create hotspot without aligned signals")

        # Get primary location (from anchor signal)
        primary_signal = list(aligned_signals.values())[0]
        location = primary_signal.location

        # Calculate metrics
        agreement_score, num_signals = self.calculate_agreement_score(aligned_signals)

        # Get maximum Z-score
        max_z_score = max(
            anomaly.modified_z_score
            for anomaly in aligned_signals.values()
        )

        # Assess severity
        severity = self.assess_severity(
            num_signals,
            agreement_score,
            abs(max_z_score),
            spatial_cluster
        )

        # Determine trend
        trend_direction = self.determine_trend_direction(aligned_signals)

        # Get detection time (earliest signal)
        detection_time = min(
            anomaly.timestamp
            for anomaly in aligned_signals.values()
        )

        # Generate description
        signal_names = [s.value for s in aligned_signals.keys()]
        description = (
            f"Disease hotspot detected with {num_signals} confirming signals: "
            f"{', '.join(signal_names)}. "
            f"Signal agreement: {agreement_score:.2%}. "
            f"Trend: {trend_direction}."
        )

        # Estimate cases (use spatial cluster if available)
        estimated_cases = None
        population_at_risk = None

        if spatial_cluster:
            estimated_cases = spatial_cluster.observed_cases
            population_at_risk = sum(
                loc.population or 0
                for loc in spatial_cluster.locations
            )

        # Generate recommended actions
        recommended_actions = self.generate_recommended_actions(
            severity,
            num_signals,
            trend_direction
        )

        # Create hotspot
        hotspot = Hotspot(
            hotspot_id=hotspot_id,
            detection_time=detection_time,
            location=location,
            severity=severity,
            anomaly_type=(
                AnomalyType.SPATIOTEMPORAL if spatial_cluster
                else AnomalyType.TEMPORAL
            ),
            essence_signal=aligned_signals.get(DataStreamType.ESSENCE_SYNDROMIC),
            wastewater_signal=aligned_signals.get(DataStreamType.WASTEWATER),
            search_trends_signal=aligned_signals.get(DataStreamType.SEARCH_TRENDS),
            mobility_signal=aligned_signals.get(DataStreamType.MOBILITY),
            spatial_cluster=spatial_cluster,
            num_confirming_signals=num_signals,
            signal_agreement_score=agreement_score,
            estimated_cases=estimated_cases,
            population_at_risk=population_at_risk,
            trend_direction=trend_direction,
            description=description,
            recommended_actions=recommended_actions
        )

        return hotspot

    def validate_and_create_hotspots(
        self,
        essence_anomalies: List[TemporalAnomaly],
        wastewater_anomalies: List[TemporalAnomaly],
        search_trends_anomalies: List[TemporalAnomaly],
        mobility_anomalies: List[TemporalAnomaly],
        spatial_clusters: Optional[List[SpatialCluster]] = None
    ) -> List[Hotspot]:
        """
        Cross-validate anomalies and create confirmed hotspots

        Args:
            essence_anomalies: ESSENCE syndromic anomalies
            wastewater_anomalies: Wastewater viral load anomalies
            search_trends_anomalies: Search trends anomalies
            mobility_anomalies: Mobility anomalies
            spatial_clusters: Pre-detected spatial clusters

        Returns:
            List of validated hotspots
        """
        hotspots = []

        # Group anomalies by region
        grouped_anomalies = self.group_anomalies_by_location(
            essence_anomalies,
            wastewater_anomalies,
            search_trends_anomalies,
            mobility_anomalies
        )

        # Process each region
        for region_code, anomalies_by_stream in grouped_anomalies.items():
            # Find aligned signals
            aligned_groups = self.find_aligned_signals(anomalies_by_stream)

            # Create hotspot for each aligned group
            for i, aligned_signals in enumerate(aligned_groups):
                agreement_score, num_signals = self.calculate_agreement_score(aligned_signals)

                # Only create hotspot if minimum signals met
                if num_signals < self.min_confirming_signals:
                    continue

                # Try to associate with spatial cluster
                spatial_cluster = None
                if spatial_clusters:
                    primary_signal = list(aligned_signals.values())[0]
                    for cluster in spatial_clusters:
                        # Check if signal is within cluster
                        for cluster_loc in cluster.locations:
                            if self.are_spatially_aligned(
                                primary_signal.location,
                                cluster_loc
                            ):
                                spatial_cluster = cluster
                                break
                        if spatial_cluster:
                            break

                # Create hotspot
                hotspot_id = f"hotspot_{region_code}_{i}_{int(datetime.now().timestamp())}"
                hotspot = self.create_hotspot(
                    hotspot_id,
                    aligned_signals,
                    spatial_cluster
                )
                hotspots.append(hotspot)

        return hotspots
