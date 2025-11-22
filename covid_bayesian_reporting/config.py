"""
Configuration for COVID-19 Bayesian Nowcasting and Imputation System.

This module manages all configuration settings for the nowcasting model,
MICE imputation, and data preprocessing.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()


class Config:
    """Central configuration for COVID-19 reporting model."""

    # ========================================================================
    # PROJECT PATHS
    # ========================================================================
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "output"
    MODELS_DIR = PROJECT_ROOT / "models"

    # Ensure directories exist
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    # ========================================================================
    # BAYESIAN NOWCASTING PARAMETERS
    # ========================================================================

    # Reporting lag parameters
    MAX_REPORTING_LAG_DAYS = 21  # Maximum reporting lag to model
    MEDIAN_REPORTING_LAG = 7  # Median reporting lag observed
    NOWCAST_HORIZON_DAYS = 14  # Days into the past to nowcast

    # Model hyperparameters
    NOWCAST_MODEL_CONFIG = {
        "hierarchical_levels": ["state", "county", "facility"],
        "prior_distribution": "negative_binomial",  # For count data
        "overdispersion_prior": "half_cauchy",  # For variance
        "temporal_correlation": "ar1",  # AR(1) for time series
        "spatial_correlation": "car",  # Conditional autoregressive

        # MCMC sampling parameters
        "mcmc_draws": 2000,
        "mcmc_tune": 1000,
        "mcmc_chains": 4,
        "mcmc_cores": 4,
        "target_accept": 0.95,  # High acceptance for better convergence

        # Regularization
        "shrinkage_prior": "horseshoe",  # Sparse hierarchical shrinkage
        "temporal_smoothing": 0.1,  # Smoothness penalty
    }

    # Reporting curve parameters (similar to Delphi COVIDcast)
    REPORTING_CURVE_CONFIG = {
        "functional_form": "gamma",  # Gamma distribution for delays
        "shape_prior_mean": 2.0,
        "shape_prior_sd": 0.5,
        "rate_prior_mean": 0.3,
        "rate_prior_sd": 0.1,
        "allow_weekday_effects": True,  # Weekend reporting lag
    }

    # ========================================================================
    # MICE IMPUTATION PARAMETERS
    # ========================================================================

    MICE_CONFIG = {
        "max_iterations": 10,
        "n_imputations": 5,  # Multiple imputation datasets
        "convergence_threshold": 0.01,
        "random_state": 42,

        # Imputation methods by variable type
        "imputation_methods": {
            "race_ethnicity": "predictive_mean_matching",
            "age": "bayesian_ridge",
            "comorbidities": "random_forest",
            "continuous_vars": "bayesian_ridge",
            "categorical_vars": "logistic_regression",
        },

        # Census tract proxy variables
        "census_proxy_vars": [
            "pct_white",
            "pct_black",
            "pct_hispanic",
            "pct_asian",
            "pct_native_american",
            "pct_pacific_islander",
            "median_income",
            "poverty_rate",
            "education_level",
            "population_density",
        ],

        # Imputation model features
        "predictor_vars": [
            "age",
            "gender",
            "zip_code",
            "census_tract",
            "facility_type",
            "admission_date",
            "test_type",
            "symptom_onset_date",
            # Census proxies will be added dynamically
        ],
    }

    # Race/ethnicity categories (OMB standards)
    RACE_ETHNICITY_CATEGORIES = [
        "white_non_hispanic",
        "black_non_hispanic",
        "hispanic",
        "asian",
        "native_american",
        "pacific_islander",
        "multiracial",
        "other",
        "unknown",  # Pre-imputation
    ]

    # ========================================================================
    # POSITIVITY RATE STANDARDIZATION
    # ========================================================================

    POSITIVITY_STANDARDIZATION = {
        # Lab-specific positivity definitions
        "lab_definitions": {
            "pcr_ct_threshold": {
                "quest": 37.0,  # Quest Diagnostics Ct cutoff
                "labcorp": 35.0,  # LabCorp Ct cutoff
                "local_health": 40.0,  # Local health departments
                "clinical_lab": 38.0,  # Clinical labs
            },

            # Antigen test sensitivity adjustments
            "antigen_sensitivity": {
                "binax_now": 0.85,
                "sofia": 0.80,
                "bd_veritor": 0.84,
                "generic": 0.75,
            },
        },

        # Standardization approach
        "standardization_method": "bayesian_latent_class",

        # Target positivity definition (CDC standard)
        "reference_definition": {
            "test_type": "pcr",
            "ct_threshold": 37.0,
            "include_retest": False,
            "denominator": "total_tests_excluding_retests",
        },

        # Sensitivity/specificity priors
        "test_performance_priors": {
            "pcr_sensitivity": (0.95, 0.03),  # (mean, sd)
            "pcr_specificity": (0.995, 0.005),
            "antigen_sensitivity": (0.80, 0.10),
            "antigen_specificity": (0.99, 0.01),
        },
    }

    # ========================================================================
    # CENSUS DATA API
    # ========================================================================

    CENSUS_API_KEY = os.getenv("CENSUS_API_KEY", "")
    CENSUS_YEAR = 2020  # ACS 5-year estimates
    CENSUS_DATASET = "acs/acs5"

    # Census variables to fetch (ACS 5-year)
    CENSUS_VARIABLES = {
        # Race/ethnicity counts
        "B02001_001E": "total_population",
        "B02001_002E": "white_alone",
        "B02001_003E": "black_alone",
        "B02001_005E": "asian_alone",
        "B02001_004E": "native_american_alone",
        "B02001_006E": "pacific_islander_alone",
        "B03003_003E": "hispanic_latino",

        # Socioeconomic
        "B19013_001E": "median_household_income",
        "B17001_002E": "poverty_count",
        "B15003_022E": "bachelors_degree_count",
        "B01003_001E": "total_pop_for_density",
    }

    # ========================================================================
    # VALIDATION AND QUALITY CHECKS
    # ========================================================================

    VALIDATION_CONFIG = {
        # Data quality thresholds
        "min_completeness": 0.6,  # 60% completeness required
        "max_missing_race": 0.45,  # 45% missing race/ethnicity allowed
        "min_reporting_facilities": 100,

        # Nowcasting validation
        "nowcast_credible_interval": 0.95,  # 95% CI
        "max_nowcast_uncertainty": 0.5,  # 50% relative uncertainty

        # Imputation validation
        "max_mice_iterations": 20,
        "convergence_tolerance": 0.001,
        "min_imputation_quality": 0.7,  # FMI < 0.3

        # Cross-validation
        "cv_folds": 5,
        "holdout_fraction": 0.2,
    }

    # ========================================================================
    # OUTPUT AND REPORTING
    # ========================================================================

    OUTPUT_CONFIG = {
        "save_mcmc_trace": True,
        "save_imputed_datasets": True,
        "save_diagnostics": True,
        "export_formats": ["csv", "parquet", "json"],

        # Reporting outputs
        "generate_plots": True,
        "plot_format": "png",
        "plot_dpi": 300,

        # Report sections
        "report_sections": [
            "nowcast_estimates",
            "imputation_summary",
            "positivity_trends",
            "uncertainty_quantification",
            "model_diagnostics",
            "data_quality_assessment",
        ],
    }

    # ========================================================================
    # LOGGING
    # ========================================================================

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    LOG_FILE = OUTPUT_DIR / "covid_nowcasting.log"

    @classmethod
    def load_from_yaml(cls, config_path: str) -> Dict:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Dictionary with configuration values
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    @classmethod
    def validate_config(cls) -> bool:
        """
        Validate configuration settings.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        # Check required directories
        if not cls.DATA_DIR.exists():
            raise ValueError(f"Data directory does not exist: {cls.DATA_DIR}")

        # Validate nowcasting parameters
        if cls.MAX_REPORTING_LAG_DAYS < cls.NOWCAST_HORIZON_DAYS:
            raise ValueError(
                "MAX_REPORTING_LAG_DAYS must be >= NOWCAST_HORIZON_DAYS"
            )

        # Validate MCMC parameters
        mcmc_config = cls.NOWCAST_MODEL_CONFIG
        if mcmc_config["mcmc_draws"] < 1000:
            raise ValueError("MCMC draws should be >= 1000 for reliable inference")

        # Validate MICE parameters
        if cls.MICE_CONFIG["n_imputations"] < 3:
            raise ValueError("Number of imputations should be >= 3")

        return True

    @classmethod
    def get_model_params(cls, model_type: str) -> Dict:
        """
        Get parameters for specific model type.

        Args:
            model_type: Type of model ('nowcast', 'mice', 'positivity')

        Returns:
            Dictionary with model parameters
        """
        model_configs = {
            "nowcast": cls.NOWCAST_MODEL_CONFIG,
            "mice": cls.MICE_CONFIG,
            "positivity": cls.POSITIVITY_STANDARDIZATION,
        }

        if model_type not in model_configs:
            raise ValueError(f"Unknown model type: {model_type}")

        return model_configs[model_type]


# Validate configuration on module load
try:
    Config.validate_config()
except Exception as e:
    print(f"Warning: Configuration validation failed: {e}")
