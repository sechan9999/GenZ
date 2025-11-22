"""
Multiple Imputation with Chained Equations (MICE) for COVID-19 Race/Ethnicity Data.

This module implements MICE imputation to handle 30-40% missing race/ethnicity
data using census tract proxies and demographic predictors.

The approach:
1. Fetch census tract demographics from US Census API
2. Link patient records to census tracts via ZIP code/geolocation
3. Use census demographics as auxiliary variables in imputation
4. Generate multiple imputed datasets for uncertainty quantification
5. Validate imputation quality using diagnostic metrics

Reference:
- van Buuren & Groothuis-Oudshoorn (2011). "mice: Multivariate Imputation by Chained Equations"
- Little & Rubin (2019). "Statistical Analysis with Missing Data"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats
import warnings
from loguru import logger

from config import Config

warnings.filterwarnings('ignore', category=FutureWarning)


class CensusTractProxy:
    """
    Fetch and manage census tract demographic data for imputation.

    Uses US Census API to retrieve race/ethnicity distributions
    by census tract, which serve as proxy variables for missing data.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize census data fetcher.

        Args:
            api_key: US Census API key (optional, uses env var if not provided)
        """
        self.api_key = api_key or Config.CENSUS_API_KEY

        if not self.api_key:
            logger.warning(
                "No Census API key provided. Using synthetic proxy data."
            )

        self.census_data = None

    def fetch_census_data(
        self,
        state_fips: Union[str, List[str]],
        year: int = 2020
    ) -> pd.DataFrame:
        """
        Fetch census tract demographics from US Census API.

        Args:
            state_fips: State FIPS code(s) to fetch
            year: Census year (ACS 5-year estimates)

        Returns:
            DataFrame with census tract demographics
        """
        if not self.api_key:
            logger.warning("Using synthetic census data (no API key)")
            return self._generate_synthetic_census_data()

        try:
            from census import Census
            from us import states

            c = Census(self.api_key)

            # Fetch ACS 5-year data
            logger.info(f"Fetching census data for year {year}...")

            if isinstance(state_fips, str):
                state_fips = [state_fips]

            all_tracts = []

            for fips in state_fips:
                # Fetch census variables
                data = c.acs5.state_county_tract(
                    fields=list(Config.CENSUS_VARIABLES.keys()),
                    state_fips=fips,
                    county_fips=Census.ALL,
                    tract=Census.ALL,
                    year=year
                )

                all_tracts.extend(data)

            # Convert to DataFrame
            df = pd.DataFrame(all_tracts)

            # Rename columns
            df = df.rename(columns=Config.CENSUS_VARIABLES)

            # Create census tract ID
            df['census_tract'] = (
                df['state'].astype(str).str.zfill(2) +
                df['county'].astype(str).str.zfill(3) +
                df['tract'].astype(str).str.zfill(6)
            )

            # Calculate percentages
            total_pop = df['total_population'].replace(0, np.nan)

            df['pct_white'] = df['white_alone'] / total_pop
            df['pct_black'] = df['black_alone'] / total_pop
            df['pct_asian'] = df['asian_alone'] / total_pop
            df['pct_native_american'] = df['native_american_alone'] / total_pop
            df['pct_pacific_islander'] = df['pacific_islander_alone'] / total_pop
            df['pct_hispanic'] = df['hispanic_latino'] / total_pop

            # Poverty rate
            df['poverty_rate'] = df['poverty_count'] / total_pop

            # Education (% with bachelor's or higher)
            df['pct_bachelors'] = df['bachelors_degree_count'] / total_pop

            # Population density (per sq mile - simplified)
            df['population_density'] = df['total_pop_for_density']

            self.census_data = df

            logger.info(f"Fetched data for {len(df)} census tracts")

            return df

        except ImportError:
            logger.error("census package not installed. Install with: pip install census")
            return self._generate_synthetic_census_data()

        except Exception as e:
            logger.error(f"Failed to fetch census data: {e}")
            return self._generate_synthetic_census_data()

    def _generate_synthetic_census_data(self, n_tracts: int = 1000) -> pd.DataFrame:
        """Generate synthetic census tract data for testing."""
        np.random.seed(42)

        logger.info(f"Generating {n_tracts} synthetic census tracts...")

        # Create synthetic tracts
        df = pd.DataFrame({
            'census_tract': [f"{i:011d}" for i in range(n_tracts)],
            'state': np.random.choice(['01', '06', '36', '48'], n_tracts),
        })

        # Simulate race/ethnicity distributions (using Dirichlet)
        alphas = np.array([10, 5, 3, 2, 1, 1])  # Concentrations
        race_dist = np.random.dirichlet(alphas, size=n_tracts)

        df['pct_white'] = race_dist[:, 0]
        df['pct_black'] = race_dist[:, 1]
        df['pct_hispanic'] = race_dist[:, 2]
        df['pct_asian'] = race_dist[:, 3]
        df['pct_native_american'] = race_dist[:, 4]
        df['pct_pacific_islander'] = race_dist[:, 5]

        # Socioeconomic variables
        df['median_income'] = np.random.lognormal(11, 0.5, n_tracts)
        df['poverty_rate'] = np.random.beta(2, 10, n_tracts)
        df['pct_bachelors'] = np.random.beta(3, 7, n_tracts)
        df['population_density'] = np.random.lognormal(6, 2, n_tracts)

        self.census_data = df

        return df

    def link_to_census_tract(
        self,
        patient_data: pd.DataFrame,
        zip_col: str = 'zip_code'
    ) -> pd.DataFrame:
        """
        Link patient data to census tracts.

        Args:
            patient_data: DataFrame with patient records
            zip_col: Column name containing ZIP codes

        Returns:
            DataFrame with census tract identifiers added
        """
        # Simplified: Use ZIP code to census tract crosswalk
        # In production, use HUD USPS ZIP-Tract crosswalk file

        logger.info("Linking patients to census tracts via ZIP code...")

        # For simplicity, assign random census tract within state
        # In production, use actual crosswalk

        if self.census_data is None:
            self.fetch_census_data('01')  # Fetch sample data

        patient_data = patient_data.copy()

        # Randomly assign census tracts (for demo)
        patient_data['census_tract'] = np.random.choice(
            self.census_data['census_tract'].values,
            size=len(patient_data),
            replace=True
        )

        logger.info(f"Linked {len(patient_data)} patients to census tracts")

        return patient_data


class MICEImputer:
    """
    Multiple Imputation with Chained Equations for race/ethnicity data.

    Handles 30-40% missing race/ethnicity data using:
    - Census tract demographic proxies
    - Patient demographic variables (age, gender, location)
    - Clinical variables (facility type, admission date)
    - Predictive mean matching for categorical variables
    """

    def __init__(
        self,
        n_imputations: int = 5,
        max_iter: int = 10,
        random_state: int = 42
    ):
        """
        Initialize MICE imputer.

        Args:
            n_imputations: Number of imputed datasets to create
            max_iter: Maximum MICE iterations
            random_state: Random seed for reproducibility
        """
        self.n_imputations = n_imputations
        self.max_iter = max_iter
        self.random_state = random_state

        self.imputers = []
        self.encoders = {}
        self.scalers = {}
        self.imputed_datasets = []
        self.imputation_diagnostics = {}

        logger.info(
            f"Initialized MICE imputer: {n_imputations} imputations, "
            f"{max_iter} iterations"
        )

    def fit_transform(
        self,
        data: pd.DataFrame,
        census_data: pd.DataFrame,
        target_col: str = 'race_ethnicity',
        predictor_cols: Optional[List[str]] = None
    ) -> List[pd.DataFrame]:
        """
        Fit MICE model and generate multiple imputed datasets.

        Args:
            data: Patient data with missing race/ethnicity
            census_data: Census tract demographics
            target_col: Column to impute
            predictor_cols: Predictor variable columns (optional)

        Returns:
            List of imputed DataFrames
        """
        logger.info("Starting MICE imputation...")

        # Merge with census data
        data_with_census = data.merge(
            census_data,
            on='census_tract',
            how='left'
        )

        # Select predictors
        if predictor_cols is None:
            predictor_cols = self._select_predictors(data_with_census, target_col)

        # Prepare data
        X, y, missing_mask = self._prepare_data(
            data_with_census,
            target_col,
            predictor_cols
        )

        logger.info(
            f"Imputing {missing_mask.sum()} missing values "
            f"({missing_mask.mean()*100:.1f}%)"
        )

        # Generate multiple imputations
        self.imputed_datasets = []

        for m in range(self.n_imputations):
            logger.info(f"Generating imputation {m+1}/{self.n_imputations}...")

            # Create imputer
            imputer = self._create_imputer(m)

            # Fit and transform
            X_imputed = imputer.fit_transform(X)

            # Convert back to DataFrame
            df_imputed = data_with_census.copy()

            # Decode imputed values
            y_imputed = X_imputed[:, predictor_cols.index(target_col)]

            # Round to nearest category for categorical
            if target_col in self.encoders:
                y_imputed = np.round(y_imputed).astype(int)
                y_imputed = np.clip(
                    y_imputed,
                    0,
                    len(self.encoders[target_col].classes_) - 1
                )
                y_imputed = self.encoders[target_col].inverse_transform(y_imputed)

            df_imputed[target_col] = y_imputed

            self.imputed_datasets.append(df_imputed)
            self.imputers.append(imputer)

        # Calculate diagnostics
        self._calculate_diagnostics(data_with_census, target_col, missing_mask)

        logger.info("MICE imputation complete")

        return self.imputed_datasets

    def _select_predictors(
        self,
        data: pd.DataFrame,
        target_col: str
    ) -> List[str]:
        """Select predictor variables for imputation."""
        # Use all census proxy variables
        census_vars = [
            col for col in Config.MICE_CONFIG["census_proxy_vars"]
            if col in data.columns
        ]

        # Add patient demographics
        patient_vars = [
            col for col in ['age', 'gender', 'zip_code', 'facility_type']
            if col in data.columns
        ]

        # Add target variable
        predictors = census_vars + patient_vars + [target_col]

        logger.info(f"Selected {len(predictors)} predictor variables")

        return predictors

    def _prepare_data(
        self,
        data: pd.DataFrame,
        target_col: str,
        predictor_cols: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for MICE imputation."""
        # Extract predictors
        df_subset = data[predictor_cols].copy()

        # Encode categorical variables
        for col in df_subset.columns:
            if df_subset[col].dtype == 'object' or df_subset[col].dtype.name == 'category':
                encoder = LabelEncoder()
                # Handle missing values
                valid_mask = df_subset[col].notna()
                if valid_mask.any():
                    df_subset.loc[valid_mask, col] = encoder.fit_transform(
                        df_subset.loc[valid_mask, col]
                    )
                    self.encoders[col] = encoder

        # Convert to numeric
        df_subset = df_subset.apply(pd.to_numeric, errors='coerce')

        # Extract arrays
        X = df_subset.values
        y = df_subset[target_col].values
        missing_mask = pd.isna(y)

        return X, y, missing_mask

    def _create_imputer(self, seed: int) -> IterativeImputer:
        """Create MICE imputer with specified random seed."""
        # Use Bayesian Ridge for continuous, will handle mixed types
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=self.max_iter,
            tol=Config.MICE_CONFIG["convergence_threshold"],
            random_state=self.random_state + seed,
            verbose=0
        )

        return imputer

    def _calculate_diagnostics(
        self,
        original_data: pd.DataFrame,
        target_col: str,
        missing_mask: np.ndarray
    ):
        """Calculate imputation quality diagnostics."""
        logger.info("Calculating imputation diagnostics...")

        # Extract imputed values across all datasets
        imputed_values = []

        for df in self.imputed_datasets:
            imputed_values.append(df.loc[missing_mask, target_col].values)

        imputed_values = np.array(imputed_values)  # (n_imputations, n_missing)

        # Calculate variance metrics
        # Within-imputation variance
        within_var = np.mean([
            df.loc[missing_mask, target_col].var()
            for df in self.imputed_datasets
        ])

        # Between-imputation variance
        means = np.array([
            df.loc[missing_mask, target_col].mean()
            for df in self.imputed_datasets
        ])
        between_var = means.var()

        # Total variance
        total_var = within_var + between_var * (1 + 1/self.n_imputations)

        # Fraction of Missing Information (FMI)
        fmi = (between_var * (1 + 1/self.n_imputations)) / total_var

        # Relative increase in variance
        lambda_val = between_var / within_var

        self.imputation_diagnostics = {
            'n_missing': int(missing_mask.sum()),
            'missing_fraction': float(missing_mask.mean()),
            'within_variance': float(within_var),
            'between_variance': float(between_var),
            'total_variance': float(total_var),
            'fmi': float(fmi),
            'lambda': float(lambda_val),
        }

        logger.info(f"Imputation diagnostics:")
        logger.info(f"  Missing fraction: {self.imputation_diagnostics['missing_fraction']:.1%}")
        logger.info(f"  FMI: {self.imputation_diagnostics['fmi']:.3f}")
        logger.info(f"  Lambda: {self.imputation_diagnostics['lambda']:.3f}")

        # Check quality
        if fmi > 0.3:
            logger.warning(
                f"High FMI ({fmi:.2f}) indicates substantial missing information. "
                "Consider increasing n_imputations."
            )

    def get_diagnostics(self) -> Dict:
        """Get imputation quality diagnostics."""
        return self.imputation_diagnostics.copy()

    def pool_estimates(
        self,
        estimates: List[float],
        std_errors: List[float]
    ) -> Tuple[float, float, float]:
        """
        Pool estimates across imputed datasets using Rubin's rules.

        Args:
            estimates: Point estimates from each imputed dataset
            std_errors: Standard errors from each imputed dataset

        Returns:
            Tuple of (pooled_estimate, pooled_se, pooled_ci_width)
        """
        m = len(estimates)

        # Pooled estimate (average)
        pooled_est = np.mean(estimates)

        # Within-imputation variance
        W = np.mean(np.array(std_errors) ** 2)

        # Between-imputation variance
        B = np.var(estimates, ddof=1)

        # Total variance
        T = W + B * (1 + 1/m)

        # Pooled standard error
        pooled_se = np.sqrt(T)

        # Degrees of freedom (Barnard & Rubin, 1999)
        df = (m - 1) * (1 + W / (B * (1 + 1/m))) ** 2

        # 95% CI width
        t_crit = stats.t.ppf(0.975, df)
        ci_width = 2 * t_crit * pooled_se

        logger.info(
            f"Pooled estimate: {pooled_est:.3f} ± {pooled_se:.3f} "
            f"(95% CI width: {ci_width:.3f})"
        )

        return pooled_est, pooled_se, ci_width


# ============================================================================
# Validation Functions
# ============================================================================

def validate_imputation(
    original_data: pd.DataFrame,
    imputed_datasets: List[pd.DataFrame],
    target_col: str = 'race_ethnicity'
) -> Dict:
    """
    Validate imputation quality.

    Args:
        original_data: Original data with missing values
        imputed_datasets: List of imputed datasets
        target_col: Column that was imputed

    Returns:
        Dictionary with validation metrics
    """
    logger.info("Validating imputation quality...")

    missing_mask = original_data[target_col].isna()

    # 1. Distribution comparison (observed vs imputed)
    observed_dist = original_data[target_col].value_counts(normalize=True)

    imputed_dist = pd.concat([
        df.loc[missing_mask, target_col]
        for df in imputed_datasets
    ]).value_counts(normalize=True)

    # KL divergence
    # Ensure same categories
    all_cats = set(observed_dist.index) | set(imputed_dist.index)

    obs_probs = np.array([observed_dist.get(cat, 1e-10) for cat in all_cats])
    imp_probs = np.array([imputed_dist.get(cat, 1e-10) for cat in all_cats])

    # Normalize
    obs_probs /= obs_probs.sum()
    imp_probs /= imp_probs.sum()

    kl_div = stats.entropy(obs_probs, imp_probs)

    # 2. Variance across imputations
    imputed_values = pd.concat([
        df.loc[missing_mask, target_col]
        for df in imputed_datasets
    ], keys=range(len(imputed_datasets)))

    variance_ratio = (
        imputed_values.groupby(level=0).value_counts(normalize=True).var() /
        observed_dist.var()
    )

    validation = {
        'kl_divergence': float(kl_div),
        'variance_ratio': float(variance_ratio),
        'n_imputed': int(missing_mask.sum()),
        'observed_distribution': observed_dist.to_dict(),
        'imputed_distribution': imputed_dist.to_dict(),
    }

    logger.info(f"Validation metrics:")
    logger.info(f"  KL divergence: {kl_div:.4f}")
    logger.info(f"  Variance ratio: {variance_ratio:.4f}")

    return validation


if __name__ == "__main__":
    # Example usage
    logger.info("MICE Imputation Example")

    # Create synthetic patient data with missing race/ethnicity
    np.random.seed(42)
    n_patients = 1000

    # Simulate data
    data = pd.DataFrame({
        'patient_id': range(n_patients),
        'age': np.random.normal(55, 15, n_patients).clip(18, 100),
        'gender': np.random.choice(['M', 'F'], n_patients),
        'zip_code': np.random.choice([f'{i:05d}' for i in range(10000, 10100)], n_patients),
        'facility_type': np.random.choice(['Hospital', 'Clinic', 'Testing Site'], n_patients),
    })

    # Add race/ethnicity with 35% missing
    races = Config.RACE_ETHNICITY_CATEGORIES[:-1]  # Exclude 'unknown'
    true_race = np.random.choice(races, n_patients, p=[0.5, 0.15, 0.2, 0.08, 0.05, 0.01, 0.008, 0.002])

    # Introduce missingness (MCAR for demo)
    missing_mask = np.random.rand(n_patients) < 0.35
    data['race_ethnicity'] = true_race
    data.loc[missing_mask, 'race_ethnicity'] = np.nan

    logger.info(f"Created dataset with {missing_mask.sum()} ({missing_mask.mean()*100:.1f}%) missing race/ethnicity")

    # Fetch census data
    census_proxy = CensusTractProxy()
    census_data = census_proxy.fetch_census_data('01')  # Alabama

    # Link to census tracts
    data = census_proxy.link_to_census_tract(data)

    # Run MICE imputation
    imputer = MICEImputer(n_imputations=5, max_iter=10)
    imputed_datasets = imputer.fit_transform(
        data,
        census_data,
        target_col='race_ethnicity'
    )

    # Get diagnostics
    diagnostics = imputer.get_diagnostics()
    print("\nImputation Diagnostics:")
    for key, value in diagnostics.items():
        print(f"  {key}: {value}")

    # Validate
    validation = validate_imputation(data, imputed_datasets)
    print("\nValidation Metrics:")
    for key, value in validation.items():
        if key not in ['observed_distribution', 'imputed_distribution']:
            print(f"  {key}: {value}")

    logger.info("Done!")
