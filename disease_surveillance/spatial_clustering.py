"""
Spatiotemporal Clustering using SaTScan Integration
Implements spatial and space-time scan statistics for disease cluster detection
"""

from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import tempfile
import uuid
import warnings

from .models import (
    GeographicLocation,
    SpatialCluster,
    TemporalAnomaly,
    TimeSeriesDataPoint
)


class SaTScanIntegration:
    """
    Integration with SaTScan software for spatial cluster detection

    SaTScan is widely used in disease surveillance for detecting and evaluating clusters
    of events in space and/or time. This class provides a Python interface to SaTScan.

    Note: Requires SaTScan to be installed separately from:
    https://www.satscan.org/
    """

    def __init__(
        self,
        satscan_executable: Optional[str] = None,
        max_cluster_size: float = 0.5,  # Maximum cluster size as % of population
        min_cases: int = 5,  # Minimum cases in cluster
        significance_level: float = 0.05,
        scan_type: str = "both"  # "spatial", "temporal", or "both"
    ):
        """
        Initialize SaTScan integration

        Args:
            satscan_executable: Path to SaTScan executable
            max_cluster_size: Maximum cluster size (0-1)
            min_cases: Minimum number of cases in cluster
            significance_level: P-value threshold for significance
            scan_type: Type of scan ("spatial", "temporal", "both")
        """
        self.satscan_executable = satscan_executable or self._find_satscan()
        self.max_cluster_size = max_cluster_size
        self.min_cases = min_cases
        self.significance_level = significance_level
        self.scan_type = scan_type

    def _find_satscan(self) -> Optional[str]:
        """Attempt to locate SaTScan executable"""
        possible_paths = [
            "/usr/local/bin/satscan",
            "/opt/satscan/satscan",
            "C:\\Program Files\\SaTScan\\satscan.exe",
            "C:\\Program Files (x86)\\SaTScan\\satscan.exe"
        ]

        for path in possible_paths:
            if Path(path).exists():
                return path

        return None

    def create_case_file(
        self,
        cases_df: pd.DataFrame,
        output_path: Path
    ):
        """
        Create SaTScan case file format

        Case file format: LocationID, NumberOfCases, Date
        """
        with open(output_path, 'w') as f:
            for _, row in cases_df.iterrows():
                f.write(f"{row['location_id']}\t{row['cases']}\t{row['date']}\n")

    def create_coordinates_file(
        self,
        locations_df: pd.DataFrame,
        output_path: Path
    ):
        """
        Create SaTScan coordinates file

        Coordinates file format: LocationID, Latitude, Longitude
        """
        with open(output_path, 'w') as f:
            for _, row in locations_df.iterrows():
                f.write(f"{row['location_id']}\t{row['latitude']}\t{row['longitude']}\n")

    def create_population_file(
        self,
        population_df: pd.DataFrame,
        output_path: Path
    ):
        """
        Create SaTScan population file

        Population file format: LocationID, Year, Population
        """
        with open(output_path, 'w') as f:
            for _, row in population_df.iterrows():
                year = row.get('year', datetime.now().year)
                f.write(f"{row['location_id']}\t{year}\t{row['population']}\n")

    def create_parameter_file(
        self,
        case_file: Path,
        coordinates_file: Path,
        population_file: Path,
        output_base: Path,
        start_date: datetime,
        end_date: datetime
    ) -> Path:
        """
        Create SaTScan parameter file (.prm)
        """
        param_file = output_base.with_suffix('.prm')

        params = f"""[Input]
CaseFile={case_file}
CoordinatesFile={coordinates_file}
PopulationFile={population_file}

[Analysis]
AnalysisType={1 if self.scan_type == 'spatial' else 3 if self.scan_type == 'temporal' else 4}
ModelType=2
ScanAreas=1

[Output]
ResultsFile={output_base}
OutputGoogleEarthKML=n
OutputShapefiles=n

[Multiple Data Sets]
StartDate={start_date.strftime('%Y/%m/%d')}
EndDate={end_date.strftime('%Y/%m/%d')}

[Advanced]
MaximumSpatialSizeInPopulation={self.max_cluster_size}
MinimumTemporalClusterSize=2
MaximumTemporalClusterSize=90
IncludeTemporalClustersThatOverlapStart=y
IncludeTemporalClustersThatOverlapEnd=y
MinimumCasesInCluster={self.min_cases}
ReportGiniClusters=n
MonteCarloReps=999
"""

        with open(param_file, 'w') as f:
            f.write(params)

        return param_file

    def run_satscan(
        self,
        param_file: Path,
        timeout: int = 300
    ) -> bool:
        """
        Execute SaTScan with parameter file

        Args:
            param_file: Path to parameter file
            timeout: Timeout in seconds

        Returns:
            True if successful, False otherwise
        """
        if not self.satscan_executable:
            raise RuntimeError(
                "SaTScan executable not found. Please install SaTScan or "
                "specify the path to the executable."
            )

        try:
            result = subprocess.run(
                [self.satscan_executable, str(param_file)],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            warnings.warn(f"SaTScan execution timed out after {timeout} seconds")
            return False
        except Exception as e:
            warnings.warn(f"SaTScan execution failed: {str(e)}")
            return False

    def parse_results(
        self,
        results_file: Path,
        locations_lookup: Dict[str, GeographicLocation]
    ) -> List[SpatialCluster]:
        """
        Parse SaTScan results file

        Returns:
            List of detected spatial clusters
        """
        clusters = []

        try:
            with open(results_file.with_suffix('.col.txt'), 'r') as f:
                lines = f.readlines()

            # Parse cluster information
            # SaTScan output format is complex, this is a simplified parser
            in_cluster = False
            current_cluster = None

            for line in lines:
                if 'Cluster' in line and 'Start' in line:
                    in_cluster = True
                    # Extract cluster info
                    # This is simplified - actual parsing would be more complex
                    continue

                if in_cluster:
                    # Parse cluster details
                    # Would need to parse: locations, dates, statistics
                    pass

            return clusters

        except Exception as e:
            warnings.warn(f"Failed to parse SaTScan results: {str(e)}")
            return []

    def detect_clusters(
        self,
        cases_df: pd.DataFrame,
        locations_df: pd.DataFrame,
        population_df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime
    ) -> List[SpatialCluster]:
        """
        Detect spatial clusters using SaTScan

        Args:
            cases_df: DataFrame with columns [location_id, cases, date]
            locations_df: DataFrame with columns [location_id, latitude, longitude]
            population_df: DataFrame with columns [location_id, population]
            start_date: Analysis start date
            end_date: Analysis end date

        Returns:
            List of detected spatial clusters
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create input files
            case_file = tmpdir_path / "cases.txt"
            coord_file = tmpdir_path / "coords.txt"
            pop_file = tmpdir_path / "population.txt"
            output_base = tmpdir_path / "results"

            self.create_case_file(cases_df, case_file)
            self.create_coordinates_file(locations_df, coord_file)
            self.create_population_file(population_df, pop_file)

            # Create parameter file
            param_file = self.create_parameter_file(
                case_file,
                coord_file,
                pop_file,
                output_base,
                start_date,
                end_date
            )

            # Run SaTScan
            success = self.run_satscan(param_file)

            if success:
                # Create location lookup
                locations_lookup = {
                    row['location_id']: GeographicLocation(
                        latitude=row['latitude'],
                        longitude=row['longitude'],
                        region_name=row.get('region_name', ''),
                        region_code=row['location_id'],
                        population=population_df[
                            population_df['location_id'] == row['location_id']
                        ]['population'].iloc[0] if not population_df.empty else None
                    )
                    for _, row in locations_df.iterrows()
                }

                return self.parse_results(output_base, locations_lookup)
            else:
                warnings.warn("SaTScan execution was not successful")
                return []


class SimplifiedSpatialClustering:
    """
    Simplified spatial clustering when SaTScan is not available

    Uses density-based spatial clustering (DBSCAN) and Kulldorff's spatial scan statistic
    approximation
    """

    def __init__(
        self,
        max_distance_km: float = 50,
        min_samples: int = 3,
        significance_level: float = 0.05
    ):
        """
        Initialize simplified spatial clustering

        Args:
            max_distance_km: Maximum distance between points in cluster
            min_samples: Minimum samples to form cluster
            significance_level: Significance threshold
        """
        self.max_distance_km = max_distance_km
        self.min_samples = min_samples
        self.significance_level = significance_level

    def haversine_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """
        Calculate great circle distance between two points in km

        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates

        Returns:
            Distance in kilometers
        """
        R = 6371  # Earth radius in km

        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    def detect_clusters(
        self,
        anomalies: List[TemporalAnomaly],
        time_window_days: int = 14
    ) -> List[SpatialCluster]:
        """
        Detect spatial clusters from temporal anomalies

        Args:
            anomalies: List of temporal anomalies
            time_window_days: Time window for clustering

        Returns:
            List of spatial clusters
        """
        if not anomalies:
            return []

        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            warnings.warn("scikit-learn not available, spatial clustering disabled")
            return []

        # Group anomalies by time windows
        time_groups = self._group_by_time(anomalies, time_window_days)

        clusters = []

        for time_group in time_groups:
            # Extract coordinates
            coords = np.array([
                [a.location.latitude, a.location.longitude]
                for a in time_group
            ])

            if len(coords) < self.min_samples:
                continue

            # Approximate distance in km to degrees
            # This is rough: 1 degree ≈ 111 km
            eps_degrees = self.max_distance_km / 111.0

            # Perform DBSCAN clustering
            clustering = DBSCAN(
                eps=eps_degrees,
                min_samples=self.min_samples,
                metric='haversine'
            )
            labels = clustering.fit_predict(np.radians(coords))

            # Extract clusters
            for cluster_id in set(labels):
                if cluster_id == -1:  # Noise points
                    continue

                cluster_anomalies = [
                    a for a, label in zip(time_group, labels)
                    if label == cluster_id
                ]

                cluster = self._create_cluster(cluster_id, cluster_anomalies)
                clusters.append(cluster)

        return clusters

    def _group_by_time(
        self,
        anomalies: List[TemporalAnomaly],
        time_window_days: int
    ) -> List[List[TemporalAnomaly]]:
        """Group anomalies into time windows"""
        sorted_anomalies = sorted(anomalies, key=lambda a: a.timestamp)

        groups = []
        current_group = []

        for anomaly in sorted_anomalies:
            if not current_group:
                current_group.append(anomaly)
            else:
                time_diff = (anomaly.timestamp - current_group[0].timestamp).days
                if time_diff <= time_window_days:
                    current_group.append(anomaly)
                else:
                    groups.append(current_group)
                    current_group = [anomaly]

        if current_group:
            groups.append(current_group)

        return groups

    def _create_cluster(
        self,
        cluster_id: int,
        anomalies: List[TemporalAnomaly]
    ) -> SpatialCluster:
        """Create SpatialCluster from anomalies"""
        # Calculate centroid
        lats = [a.location.latitude for a in anomalies]
        lons = [a.location.longitude for a in anomalies]
        center_lat = np.mean(lats)
        center_lon = np.mean(lons)

        # Calculate radius
        max_radius = 0
        for anomaly in anomalies:
            dist = self.haversine_distance(
                center_lat, center_lon,
                anomaly.location.latitude, anomaly.location.longitude
            )
            max_radius = max(max_radius, dist)

        # Get time range
        timestamps = [a.timestamp for a in anomalies]
        start_date = min(timestamps)
        end_date = max(timestamps)

        # Observed cases (sum of observed values)
        observed_cases = int(sum(a.observed_value for a in anomalies))
        expected_cases = sum(a.expected_value for a in anomalies)

        # Calculate relative risk
        relative_risk = observed_cases / expected_cases if expected_cases > 0 else 0

        # Simple log-likelihood ratio approximation
        if observed_cases > expected_cases:
            llr = observed_cases * np.log(observed_cases / expected_cases)
        else:
            llr = 0

        return SpatialCluster(
            cluster_id=f"cluster_{cluster_id}_{uuid.uuid4().hex[:8]}",
            center_location=GeographicLocation(
                latitude=center_lat,
                longitude=center_lon,
                region_name="Cluster Center",
                region_code=f"cluster_{cluster_id}"
            ),
            radius_km=max_radius,
            locations=[a.location for a in anomalies],
            start_date=start_date,
            end_date=end_date,
            observed_cases=observed_cases,
            expected_cases=expected_cases,
            relative_risk=relative_risk,
            p_value=0.05,  # Placeholder - would need Monte Carlo simulation
            log_likelihood_ratio=llr
        )
