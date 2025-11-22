"""
Temporal Anomaly Detection using STL Decomposition and Modified Z-Scores
Implements Seasonal-Trend decomposition with LOESS (STL) followed by modified Z-score anomaly detection
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime
from statsmodels.tsa.seasonal import STL
from scipy import stats
import warnings

from .models import (
    TimeSeriesDataPoint,
    STLDecomposition,
    TemporalAnomaly,
    GeographicLocation,
    DataStreamType
)


class TemporalAnomalyDetector:
    """
    Detects temporal anomalies using STL decomposition and modified Z-scores

    The modified Z-score uses median absolute deviation (MAD) instead of
    standard deviation, making it more robust to outliers.
    """

    def __init__(
        self,
        seasonal_period: int = 7,  # Weekly seasonality
        threshold: float = 3.5,  # Modified Z-score threshold
        min_observations: int = 28,  # Minimum 4 weeks of data
        stl_robust: bool = True
    ):
        """
        Initialize temporal anomaly detector

        Args:
            seasonal_period: Period for seasonal decomposition (7 for weekly)
            threshold: Modified Z-score threshold for anomaly detection
            min_observations: Minimum number of observations required
            stl_robust: Use robust STL decomposition (handles outliers better)
        """
        self.seasonal_period = seasonal_period
        self.threshold = threshold
        self.min_observations = min_observations
        self.stl_robust = stl_robust

    def decompose_timeseries(
        self,
        timeseries: pd.Series,
        timestamps: List[datetime]
    ) -> STLDecomposition:
        """
        Perform STL decomposition on time series

        Args:
            timeseries: Time series values
            timestamps: Corresponding timestamps

        Returns:
            STLDecomposition object with trend, seasonal, and residual components
        """
        if len(timeseries) < self.min_observations:
            raise ValueError(
                f"Insufficient data: {len(timeseries)} observations, "
                f"need at least {self.min_observations}"
            )

        # Ensure no NaN values
        if timeseries.isna().any():
            warnings.warn("Time series contains NaN values, interpolating...")
            timeseries = timeseries.interpolate(method='linear')

        # Perform STL decomposition
        try:
            stl = STL(
                timeseries,
                seasonal=self.seasonal_period,
                robust=self.stl_robust
            )
            result = stl.fit()

            return STLDecomposition(
                trend=result.trend.values,
                seasonal=result.seasonal.values,
                residual=result.resid.values,
                timestamps=timestamps,
                original=timeseries.values
            )
        except Exception as e:
            raise RuntimeError(f"STL decomposition failed: {str(e)}")

    def calculate_modified_zscore(self, residuals: np.ndarray) -> np.ndarray:
        """
        Calculate modified Z-scores using Median Absolute Deviation (MAD)

        Modified Z-score = 0.6745 * (x - median(x)) / MAD
        where MAD = median(|x - median(x)|)

        The factor 0.6745 makes the MAD estimate consistent with the standard
        deviation for normally distributed data.

        Args:
            residuals: Residual values from STL decomposition

        Returns:
            Array of modified Z-scores
        """
        median = np.median(residuals)
        mad = np.median(np.abs(residuals - median))

        # Avoid division by zero
        if mad == 0:
            # If MAD is 0, use standard deviation as fallback
            std = np.std(residuals)
            if std == 0:
                return np.zeros_like(residuals)
            return (residuals - median) / std

        # Calculate modified Z-scores
        modified_zscores = 0.6745 * (residuals - median) / mad

        return modified_zscores

    def detect_anomalies(
        self,
        timeseries: pd.Series,
        timestamps: List[datetime],
        location: GeographicLocation,
        data_source: DataStreamType
    ) -> List[TemporalAnomaly]:
        """
        Detect temporal anomalies in time series data

        Args:
            timeseries: Time series values
            timestamps: Corresponding timestamps
            location: Geographic location of the data
            data_source: Source of the surveillance data

        Returns:
            List of detected temporal anomalies
        """
        # Perform STL decomposition
        decomposition = self.decompose_timeseries(timeseries, timestamps)

        # Calculate modified Z-scores on residuals
        modified_zscores = self.calculate_modified_zscore(decomposition.residual)

        # Identify anomalies
        anomalies = []
        for i, (timestamp, z_score) in enumerate(zip(timestamps, modified_zscores)):
            is_anomaly = abs(z_score) > self.threshold

            if is_anomaly or abs(z_score) > self.threshold * 0.7:  # Include near-anomalies
                # Calculate expected value (trend + seasonal)
                expected_value = decomposition.trend[i] + decomposition.seasonal[i]
                observed_value = decomposition.original[i]

                # Calculate confidence based on how far above threshold
                confidence = min(1.0, abs(z_score) / (self.threshold * 2))

                anomaly = TemporalAnomaly(
                    timestamp=timestamp,
                    location=location,
                    observed_value=observed_value,
                    expected_value=expected_value,
                    modified_z_score=float(z_score),
                    is_anomaly=is_anomaly,
                    data_source=data_source,
                    confidence=confidence,
                    metadata={
                        'trend_component': float(decomposition.trend[i]),
                        'seasonal_component': float(decomposition.seasonal[i]),
                        'residual': float(decomposition.residual[i])
                    }
                )
                anomalies.append(anomaly)

        return anomalies

    def detect_anomalies_dataframe(
        self,
        df: pd.DataFrame,
        value_column: str,
        timestamp_column: str,
        location: GeographicLocation,
        data_source: DataStreamType
    ) -> List[TemporalAnomaly]:
        """
        Detect anomalies from a pandas DataFrame

        Args:
            df: DataFrame containing time series data
            value_column: Name of column with values
            timestamp_column: Name of column with timestamps
            location: Geographic location
            data_source: Data stream type

        Returns:
            List of detected temporal anomalies
        """
        df = df.sort_values(timestamp_column)
        timeseries = df[value_column]
        timestamps = df[timestamp_column].tolist()

        return self.detect_anomalies(
            timeseries,
            timestamps,
            location,
            data_source
        )


class MultiStreamAnomalyDetector:
    """
    Detects anomalies across multiple data streams simultaneously
    """

    def __init__(
        self,
        essence_detector: Optional[TemporalAnomalyDetector] = None,
        wastewater_detector: Optional[TemporalAnomalyDetector] = None,
        search_detector: Optional[TemporalAnomalyDetector] = None,
        mobility_detector: Optional[TemporalAnomalyDetector] = None
    ):
        """
        Initialize multi-stream anomaly detector

        Args:
            essence_detector: Detector for ESSENCE syndromic data
            wastewater_detector: Detector for wastewater viral loads
            search_detector: Detector for search trends
            mobility_detector: Detector for mobility data
        """
        self.essence_detector = essence_detector or TemporalAnomalyDetector(
            seasonal_period=7, threshold=3.5
        )
        self.wastewater_detector = wastewater_detector or TemporalAnomalyDetector(
            seasonal_period=7, threshold=3.0  # More sensitive for wastewater
        )
        self.search_detector = search_detector or TemporalAnomalyDetector(
            seasonal_period=7, threshold=3.5
        )
        self.mobility_detector = mobility_detector or TemporalAnomalyDetector(
            seasonal_period=7, threshold=4.0  # Less sensitive for mobility
        )

    def detect_all_streams(
        self,
        essence_data: Optional[pd.DataFrame],
        wastewater_data: Optional[pd.DataFrame],
        search_trends_data: Optional[pd.DataFrame],
        mobility_data: Optional[pd.DataFrame],
        location: GeographicLocation
    ) -> Tuple[
        List[TemporalAnomaly],
        List[TemporalAnomaly],
        List[TemporalAnomaly],
        List[TemporalAnomaly]
    ]:
        """
        Detect anomalies across all available data streams

        Returns:
            Tuple of (essence_anomalies, wastewater_anomalies, search_anomalies, mobility_anomalies)
        """
        essence_anomalies = []
        wastewater_anomalies = []
        search_anomalies = []
        mobility_anomalies = []

        # Detect ESSENCE anomalies
        if essence_data is not None and not essence_data.empty:
            try:
                essence_anomalies = self.essence_detector.detect_anomalies_dataframe(
                    essence_data,
                    value_column='percentage',
                    timestamp_column='timestamp',
                    location=location,
                    data_source=DataStreamType.ESSENCE_SYNDROMIC
                )
            except Exception as e:
                warnings.warn(f"ESSENCE anomaly detection failed: {str(e)}")

        # Detect wastewater anomalies
        if wastewater_data is not None and not wastewater_data.empty:
            try:
                wastewater_anomalies = self.wastewater_detector.detect_anomalies_dataframe(
                    wastewater_data,
                    value_column='viral_load',
                    timestamp_column='timestamp',
                    location=location,
                    data_source=DataStreamType.WASTEWATER
                )
            except Exception as e:
                warnings.warn(f"Wastewater anomaly detection failed: {str(e)}")

        # Detect search trends anomalies
        if search_trends_data is not None and not search_trends_data.empty:
            try:
                search_anomalies = self.search_detector.detect_anomalies_dataframe(
                    search_trends_data,
                    value_column='normalized_interest',
                    timestamp_column='timestamp',
                    location=location,
                    data_source=DataStreamType.SEARCH_TRENDS
                )
            except Exception as e:
                warnings.warn(f"Search trends anomaly detection failed: {str(e)}")

        # Detect mobility anomalies
        if mobility_data is not None and not mobility_data.empty:
            try:
                mobility_anomalies = self.mobility_detector.detect_anomalies_dataframe(
                    mobility_data,
                    value_column='baseline_comparison',
                    timestamp_column='timestamp',
                    location=location,
                    data_source=DataStreamType.MOBILITY
                )
            except Exception as e:
                warnings.warn(f"Mobility anomaly detection failed: {str(e)}")

        return (
            essence_anomalies,
            wastewater_anomalies,
            search_anomalies,
            mobility_anomalies
        )


def align_temporal_anomalies(
    anomalies: List[TemporalAnomaly],
    time_window_days: int = 7
) -> List[List[TemporalAnomaly]]:
    """
    Group temporal anomalies that occur within a time window

    Args:
        anomalies: List of temporal anomalies from all sources
        time_window_days: Time window for grouping anomalies

    Returns:
        List of anomaly groups (each group is anomalies within time window)
    """
    if not anomalies:
        return []

    # Sort by timestamp
    sorted_anomalies = sorted(anomalies, key=lambda a: a.timestamp)

    groups = []
    current_group = [sorted_anomalies[0]]

    for anomaly in sorted_anomalies[1:]:
        time_diff = (anomaly.timestamp - current_group[0].timestamp).days

        if time_diff <= time_window_days:
            current_group.append(anomaly)
        else:
            groups.append(current_group)
            current_group = [anomaly]

    # Add last group
    if current_group:
        groups.append(current_group)

    return groups
