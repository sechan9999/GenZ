"""
Configuration for Disease Hotspot Detection System
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class SurveillanceConfig:
    """Configuration for disease surveillance system"""

    # ═══════════════════════════════════════════════════════════
    # Directory Paths
    # ═══════════════════════════════════════════════════════════
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "output"
    TEMP_DIR = BASE_DIR / "temp"

    # ═══════════════════════════════════════════════════════════
    # Data Source Paths
    # ═══════════════════════════════════════════════════════════
    ESSENCE_DATA_PATH: Optional[Path] = None
    WASTEWATER_DATA_PATH: Optional[Path] = None
    SEARCH_TRENDS_DATA_PATH: Optional[Path] = None
    MOBILITY_DATA_PATH: Optional[Path] = None

    # ═══════════════════════════════════════════════════════════
    # Temporal Anomaly Detection Parameters
    # ═══════════════════════════════════════════════════════════
    STL_SEASONAL_PERIOD = int(os.getenv("STL_SEASONAL_PERIOD", "7"))
    MODIFIED_ZSCORE_THRESHOLD = float(os.getenv("MODIFIED_ZSCORE_THRESHOLD", "3.5"))
    MIN_OBSERVATIONS = int(os.getenv("MIN_OBSERVATIONS", "28"))
    STL_ROBUST = os.getenv("STL_ROBUST", "true").lower() == "true"

    # Stream-specific thresholds
    ESSENCE_THRESHOLD = float(os.getenv("ESSENCE_THRESHOLD", "3.5"))
    WASTEWATER_THRESHOLD = float(os.getenv("WASTEWATER_THRESHOLD", "3.0"))
    SEARCH_TRENDS_THRESHOLD = float(os.getenv("SEARCH_TRENDS_THRESHOLD", "3.5"))
    MOBILITY_THRESHOLD = float(os.getenv("MOBILITY_THRESHOLD", "4.0"))

    # ═══════════════════════════════════════════════════════════
    # Spatial Clustering Parameters
    # ═══════════════════════════════════════════════════════════
    USE_SATSCAN = os.getenv("USE_SATSCAN", "false").lower() == "true"
    SATSCAN_EXECUTABLE: Optional[str] = os.getenv("SATSCAN_EXECUTABLE")
    MAX_CLUSTER_SIZE = float(os.getenv("MAX_CLUSTER_SIZE", "0.5"))
    MIN_CLUSTER_CASES = int(os.getenv("MIN_CLUSTER_CASES", "5"))
    SPATIAL_RADIUS_KM = float(os.getenv("SPATIAL_RADIUS_KM", "50"))
    DBSCAN_MIN_SAMPLES = int(os.getenv("DBSCAN_MIN_SAMPLES", "3"))

    # ═══════════════════════════════════════════════════════════
    # Cross-Validation Parameters
    # ═══════════════════════════════════════════════════════════
    MIN_CONFIRMING_SIGNALS = int(os.getenv("MIN_CONFIRMING_SIGNALS", "2"))
    TIME_ALIGNMENT_WINDOW_DAYS = int(os.getenv("TIME_ALIGNMENT_WINDOW_DAYS", "7"))

    # Signal weights (0-1)
    ESSENCE_WEIGHT = float(os.getenv("ESSENCE_WEIGHT", "1.0"))
    WASTEWATER_WEIGHT = float(os.getenv("WASTEWATER_WEIGHT", "0.9"))
    SEARCH_TRENDS_WEIGHT = float(os.getenv("SEARCH_TRENDS_WEIGHT", "0.6"))
    MOBILITY_WEIGHT = float(os.getenv("MOBILITY_WEIGHT", "0.5"))

    # ═══════════════════════════════════════════════════════════
    # Severity Assessment
    # ═══════════════════════════════════════════════════════════
    CRITICAL_SEVERITY_THRESHOLD = int(os.getenv("CRITICAL_SEVERITY_THRESHOLD", "9"))
    HIGH_SEVERITY_THRESHOLD = int(os.getenv("HIGH_SEVERITY_THRESHOLD", "6"))
    MODERATE_SEVERITY_THRESHOLD = int(os.getenv("MODERATE_SEVERITY_THRESHOLD", "3"))

    # ═══════════════════════════════════════════════════════════
    # Real-Time Monitoring
    # ═══════════════════════════════════════════════════════════
    ENABLE_REAL_TIME_MONITORING = os.getenv("ENABLE_REAL_TIME_MONITORING", "false").lower() == "true"
    MONITORING_INTERVAL_HOURS = int(os.getenv("MONITORING_INTERVAL_HOURS", "24"))
    LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "90"))

    # ═══════════════════════════════════════════════════════════
    # Alerting Configuration
    # ═══════════════════════════════════════════════════════════
    ENABLE_EMAIL_ALERTS = os.getenv("ENABLE_EMAIL_ALERTS", "false").lower() == "true"
    EMAIL_RECIPIENTS = os.getenv("EMAIL_RECIPIENTS", "").split(",")

    ENABLE_SLACK_ALERTS = os.getenv("ENABLE_SLACK_ALERTS", "false").lower() == "true"
    SLACK_WEBHOOK_URL: Optional[str] = os.getenv("SLACK_WEBHOOK_URL")

    # Alert on severity levels
    ALERT_ON_CRITICAL = True
    ALERT_ON_HIGH = True
    ALERT_ON_MODERATE = False
    ALERT_ON_LOW = False

    # ═══════════════════════════════════════════════════════════
    # Logging
    # ═══════════════════════════════════════════════════════════
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = BASE_DIR / "surveillance.log"
    ENABLE_VERBOSE_LOGGING = os.getenv("ENABLE_VERBOSE_LOGGING", "false").lower() == "true"

    # ═══════════════════════════════════════════════════════════
    # Data Quality
    # ═══════════════════════════════════════════════════════════
    MIN_DATA_QUALITY_SCORE = float(os.getenv("MIN_DATA_QUALITY_SCORE", "0.7"))
    INTERPOLATE_MISSING_DATA = os.getenv("INTERPOLATE_MISSING_DATA", "true").lower() == "true"
    MAX_INTERPOLATION_GAP_DAYS = int(os.getenv("MAX_INTERPOLATION_GAP_DAYS", "7"))

    # ═══════════════════════════════════════════════════════════
    # Report Configuration
    # ═══════════════════════════════════════════════════════════
    GENERATE_VISUALIZATIONS = os.getenv("GENERATE_VISUALIZATIONS", "true").lower() == "true"
    REPORT_FORMAT = os.getenv("REPORT_FORMAT", "markdown,json")  # comma-separated
    INCLUDE_RAW_DATA = os.getenv("INCLUDE_RAW_DATA", "false").lower() == "true"

    @classmethod
    def ensure_directories(cls):
        """Create necessary directories"""
        for dir_path in [cls.DATA_DIR, cls.OUTPUT_DIR, cls.TEMP_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def set_data_paths(
        cls,
        essence_path: Optional[Path] = None,
        wastewater_path: Optional[Path] = None,
        search_trends_path: Optional[Path] = None,
        mobility_path: Optional[Path] = None
    ):
        """Set data source paths"""
        if essence_path:
            cls.ESSENCE_DATA_PATH = Path(essence_path)
        if wastewater_path:
            cls.WASTEWATER_DATA_PATH = Path(wastewater_path)
        if search_trends_path:
            cls.SEARCH_TRENDS_DATA_PATH = Path(search_trends_path)
        if mobility_path:
            cls.MOBILITY_DATA_PATH = Path(mobility_path)

    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        errors = []

        # Check signal weights sum
        total_weight = (
            cls.ESSENCE_WEIGHT +
            cls.WASTEWATER_WEIGHT +
            cls.SEARCH_TRENDS_WEIGHT +
            cls.MOBILITY_WEIGHT
        )
        if total_weight == 0:
            errors.append("Signal weights cannot all be zero")

        # Check thresholds
        if cls.MODIFIED_ZSCORE_THRESHOLD <= 0:
            errors.append("Modified Z-score threshold must be positive")

        if cls.MIN_OBSERVATIONS < 14:
            errors.append("Minimum observations should be at least 14 (2 weeks)")

        # Check SaTScan configuration
        if cls.USE_SATSCAN and not cls.SATSCAN_EXECUTABLE:
            errors.append("SATSCAN_EXECUTABLE must be set when USE_SATSCAN is true")

        # Check alerting configuration
        if cls.ENABLE_EMAIL_ALERTS and not cls.EMAIL_RECIPIENTS:
            errors.append("EMAIL_RECIPIENTS must be set when ENABLE_EMAIL_ALERTS is true")

        if cls.ENABLE_SLACK_ALERTS and not cls.SLACK_WEBHOOK_URL:
            errors.append("SLACK_WEBHOOK_URL must be set when ENABLE_SLACK_ALERTS is true")

        if errors:
            for error in errors:
                print(f"⚠️  Configuration Error: {error}")
            return False

        return True

    @classmethod
    def summary(cls) -> dict:
        """Return configuration summary"""
        return {
            "temporal_detection": {
                "seasonal_period": cls.STL_SEASONAL_PERIOD,
                "zscore_threshold": cls.MODIFIED_ZSCORE_THRESHOLD,
                "min_observations": cls.MIN_OBSERVATIONS,
                "robust_stl": cls.STL_ROBUST
            },
            "spatial_clustering": {
                "use_satscan": cls.USE_SATSCAN,
                "max_cluster_size": cls.MAX_CLUSTER_SIZE,
                "spatial_radius_km": cls.SPATIAL_RADIUS_KM
            },
            "cross_validation": {
                "min_confirming_signals": cls.MIN_CONFIRMING_SIGNALS,
                "time_alignment_window_days": cls.TIME_ALIGNMENT_WINDOW_DAYS,
                "signal_weights": {
                    "essence": cls.ESSENCE_WEIGHT,
                    "wastewater": cls.WASTEWATER_WEIGHT,
                    "search_trends": cls.SEARCH_TRENDS_WEIGHT,
                    "mobility": cls.MOBILITY_WEIGHT
                }
            },
            "alerting": {
                "email_enabled": cls.ENABLE_EMAIL_ALERTS,
                "slack_enabled": cls.ENABLE_SLACK_ALERTS,
                "alert_on_critical": cls.ALERT_ON_CRITICAL,
                "alert_on_high": cls.ALERT_ON_HIGH
            }
        }


# Initialize on import
SurveillanceConfig.ensure_directories()

if __name__ == "__main__":
    import json
    print("Disease Surveillance Configuration")
    print("=" * 60)
    print(json.dumps(SurveillanceConfig.summary(), indent=2))
    print("\nValidation:", "✓ PASSED" if SurveillanceConfig.validate() else "✗ FAILED")
