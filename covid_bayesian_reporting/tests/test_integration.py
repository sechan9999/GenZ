"""
Integration tests for COVID-19 Bayesian Nowcasting and Imputation System.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nowcasting import BayesianNowcaster, create_synthetic_lagged_data
from imputation import CensusTractProxy, MICEImputer, validate_imputation
from positivity_standardization import PositivityStandardizer, simulate_heterogeneous_lab_data
from config import Config


class TestNowcasting:
    """Test Bayesian nowcasting functionality."""

    def test_synthetic_data_creation(self):
        """Test synthetic lagged data generation."""
        n_days = 30
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]
        true_counts = np.random.poisson(100, n_days)

        data = create_synthetic_lagged_data(true_counts, dates)

        assert isinstance(data, pd.DataFrame)
        assert 'event_date' in data.columns
        assert 'report_date' in data.columns
        assert 'count' in data.columns
        assert len(data) > 0

    def test_nowcaster_initialization(self):
        """Test nowcaster initialization."""
        # Create minimal test data
        data = pd.DataFrame({
            'event_date': ['2024-01-01', '2024-01-02'],
            'report_date': ['2024-01-08', '2024-01-09'],
            'count': [10, 15]
        })

        nowcaster = BayesianNowcaster(data)

        assert nowcaster.data is not None
        assert len(nowcaster.data) == 2
        assert 'lag_days' in nowcaster.data.columns


class TestMICEImputation:
    """Test MICE imputation functionality."""

    def test_census_proxy_synthetic_data(self):
        """Test synthetic census data generation."""
        census_proxy = CensusTractProxy()
        census_data = census_proxy.fetch_census_data('01')

        assert isinstance(census_data, pd.DataFrame)
        assert 'census_tract' in census_data.columns
        assert 'pct_white' in census_data.columns
        assert len(census_data) > 0

    def test_mice_imputer_initialization(self):
        """Test MICE imputer initialization."""
        imputer = MICEImputer(n_imputations=3, max_iter=5)

        assert imputer.n_imputations == 3
        assert imputer.max_iter == 5

    def test_census_tract_linkage(self):
        """Test linking patients to census tracts."""
        census_proxy = CensusTractProxy()
        census_data = census_proxy.fetch_census_data('01')

        patient_data = pd.DataFrame({
            'patient_id': [1, 2, 3],
            'zip_code': ['10001', '10002', '10003']
        })

        linked_data = census_proxy.link_to_census_tract(patient_data)

        assert 'census_tract' in linked_data.columns
        assert len(linked_data) == len(patient_data)


class TestPositivityStandardization:
    """Test positivity standardization functionality."""

    def test_synthetic_lab_data(self):
        """Test synthetic lab data generation."""
        test_data = simulate_heterogeneous_lab_data(
            n_tests=100,
            true_prevalence=0.10
        )

        assert isinstance(test_data, pd.DataFrame)
        assert 'lab_name' in test_data.columns
        assert 'test_type' in test_data.columns
        assert 'result' in test_data.columns
        assert len(test_data) == 100

    def test_standardizer_initialization(self):
        """Test positivity standardizer initialization."""
        standardizer = PositivityStandardizer()

        assert standardizer.config is not None
        assert 'lab_definitions' in standardizer.config


class TestConfiguration:
    """Test configuration settings."""

    def test_config_validation(self):
        """Test configuration validation."""
        # Should not raise exception
        Config.validate_config()

    def test_config_paths(self):
        """Test configuration paths exist."""
        assert Config.DATA_DIR.exists()
        assert Config.OUTPUT_DIR.exists()
        assert Config.MODELS_DIR.exists()

    def test_model_params(self):
        """Test retrieving model parameters."""
        nowcast_params = Config.get_model_params('nowcast')
        mice_params = Config.get_model_params('mice')
        positivity_params = Config.get_model_params('positivity')

        assert isinstance(nowcast_params, dict)
        assert isinstance(mice_params, dict)
        assert isinstance(positivity_params, dict)


class TestIntegration:
    """Integration tests for full pipeline."""

    def test_end_to_end_small_dataset(self):
        """Test full pipeline with small dataset."""
        # Create small synthetic datasets
        n_days = 10
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]
        true_counts = np.random.poisson(50, n_days)
        case_data = create_synthetic_lagged_data(true_counts, dates)

        n_patients = 50
        patient_data = pd.DataFrame({
            'patient_id': range(n_patients),
            'age': np.random.normal(55, 15, n_patients).clip(18, 100),
            'gender': np.random.choice(['M', 'F'], n_patients),
            'zip_code': np.random.choice(['10001', '10002'], n_patients),
            'facility_type': np.random.choice(['Hospital', 'Clinic'], n_patients),
        })

        races = ['white', 'black', 'hispanic', 'asian']
        true_race = np.random.choice(races, n_patients)
        missing_mask = np.random.rand(n_patients) < 0.3
        patient_data['race_ethnicity'] = true_race
        patient_data.loc[missing_mask, 'race_ethnicity'] = np.nan

        test_data = simulate_heterogeneous_lab_data(n_tests=100, true_prevalence=0.10)

        # Verify data structures
        assert len(case_data) > 0
        assert len(patient_data) == n_patients
        assert len(test_data) == 100
        assert patient_data['race_ethnicity'].isna().sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
