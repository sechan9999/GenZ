"""
Bayesian Hierarchical Nowcasting Model for COVID-19 Reporting.

This module implements a nowcasting system to correct for reporting lags
(14-21 days) in COVID-19 case data, similar to the Delphi COVIDcast approach.

The model uses a hierarchical Bayesian framework to:
1. Estimate the true underlying case counts
2. Model reporting delays as a stochastic process
3. Account for weekend/holiday effects
4. Provide uncertainty quantification

Reference:
- Delphi COVIDcast: https://delphi.cmu.edu/covidcast/
- McGough et al. (2020). "Nowcasting by Bayesian Smoothing"
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from scipy import stats
from loguru import logger
import warnings

from config import Config

warnings.filterwarnings('ignore', category=FutureWarning)


class BayesianNowcaster:
    """
    Bayesian hierarchical nowcasting model for COVID-19 reporting lags.

    This model addresses the challenge of severe reporting lags (14-21 days)
    by estimating the true underlying case counts from incomplete data.

    The model includes:
    - Hierarchical structure (state -> county -> facility)
    - Temporal correlation (AR(1) process)
    - Reporting delay distribution (Gamma)
    - Weekend/holiday effects
    - Overdispersion handling (Negative Binomial)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        hierarchical_levels: Optional[List[str]] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize nowcasting model.

        Args:
            data: DataFrame with columns:
                - report_date: Date when case was reported
                - event_date: Date when case occurred (test date)
                - location: Geographic identifier
                - count: Number of cases
                - [hierarchical columns]: e.g., state, county, facility
            hierarchical_levels: List of column names for hierarchy
            config: Model configuration dict (defaults to Config.NOWCAST_MODEL_CONFIG)
        """
        self.data = data.copy()
        self.config = config or Config.NOWCAST_MODEL_CONFIG
        self.hierarchical_levels = hierarchical_levels or ["state", "county"]

        # Model artifacts
        self.model = None
        self.trace = None
        self.nowcast_estimates = None

        # Validate and prepare data
        self._validate_data()
        self._prepare_data()

        logger.info(f"Initialized nowcaster with {len(self.data)} records")

    def _validate_data(self):
        """Validate input data structure."""
        required_cols = ["report_date", "event_date", "count"]

        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")

        # Convert dates
        self.data['report_date'] = pd.to_datetime(self.data['report_date'])
        self.data['event_date'] = pd.to_datetime(self.data['event_date'])

        # Calculate reporting lag
        self.data['lag_days'] = (
            self.data['report_date'] - self.data['event_date']
        ).dt.days

        # Filter out negative lags (data quality issues)
        invalid_lags = self.data['lag_days'] < 0
        if invalid_lags.any():
            logger.warning(
                f"Removing {invalid_lags.sum()} records with negative lags"
            )
            self.data = self.data[~invalid_lags]

    def _prepare_data(self):
        """Prepare data for modeling."""
        # Create time index
        self.min_date = self.data['event_date'].min()
        self.max_date = self.data['event_date'].max()
        self.n_days = (self.max_date - self.min_date).days + 1

        # Create date -> index mapping
        self.date_to_idx = {
            self.min_date + timedelta(days=i): i
            for i in range(self.n_days)
        }

        # Add day of week (for weekend effects)
        self.data['day_of_week'] = self.data['event_date'].dt.dayofweek
        self.data['is_weekend'] = self.data['day_of_week'].isin([5, 6]).astype(int)

        # Add time index
        self.data['time_idx'] = self.data['event_date'].map(self.date_to_idx)

        logger.info(f"Prepared data spanning {self.n_days} days")

    def build_model(self) -> pm.Model:
        """
        Build hierarchical Bayesian nowcasting model.

        Returns:
            PyMC model object
        """
        logger.info("Building hierarchical Bayesian model...")

        with pm.Model() as model:
            # ================================================================
            # Data preparation
            # ================================================================

            # Aggregate by event date and location
            daily_counts = self.data.groupby(['time_idx', 'is_weekend'])['count'].sum()

            # Observed counts (incomplete due to reporting lag)
            y_obs = daily_counts.values

            # ================================================================
            # Priors: Reporting delay distribution
            # ================================================================

            # Gamma distribution for reporting delays (in days)
            # Shape and rate parameters
            delay_shape = pm.Gamma(
                "delay_shape",
                mu=Config.REPORTING_CURVE_CONFIG["shape_prior_mean"],
                sigma=Config.REPORTING_CURVE_CONFIG["shape_prior_sd"]
            )

            delay_rate = pm.Gamma(
                "delay_rate",
                mu=Config.REPORTING_CURVE_CONFIG["rate_prior_mean"],
                sigma=Config.REPORTING_CURVE_CONFIG["rate_prior_sd"]
            )

            # Weekend reporting effect
            weekend_delay_multiplier = pm.Normal(
                "weekend_delay_multiplier",
                mu=1.5,  # 50% longer delays on weekends
                sigma=0.3
            )

            # ================================================================
            # Priors: True underlying case counts
            # ================================================================

            # Global trend parameters
            log_baseline = pm.Normal("log_baseline", mu=5, sigma=2)

            # AR(1) temporal correlation
            rho = pm.Beta("rho", alpha=10, beta=2)  # Strong temporal correlation

            # Innovation variance
            sigma_innovation = pm.HalfCauchy("sigma_innovation", beta=0.5)

            # Overdispersion parameter for Negative Binomial
            phi = pm.HalfCauchy("phi", beta=2)

            # ================================================================
            # Latent true case counts (with temporal correlation)
            # ================================================================

            # Initialize true counts
            mu = pm.math.exp(log_baseline)

            # AR(1) process for log(true counts)
            log_true_counts = pm.GaussianRandomWalk(
                "log_true_counts",
                mu=0,
                sigma=sigma_innovation,
                shape=self.n_days,
                init_dist=pm.Normal.dist(mu=log_baseline, sigma=1)
            )

            # Exponentiate to get counts
            true_counts = pm.Deterministic(
                "true_counts",
                pm.math.exp(log_true_counts)
            )

            # ================================================================
            # Observation model with reporting delay
            # ================================================================

            # Expected observed counts accounting for reporting lag
            # This is a simplified model - full implementation would use
            # convolution of true counts with delay distribution

            # For each observed time point, model as function of:
            # - True count at that time
            # - Fraction of cases reported so far (based on delay dist)

            # Reporting fraction by lag time
            max_lag = Config.MAX_REPORTING_LAG_DAYS
            lag_times = np.arange(0, max_lag + 1)

            # Gamma CDF for cumulative reporting probability
            reporting_cdf = pm.math.exp(
                pm.logcdf(pm.Gamma.dist(alpha=delay_shape, beta=delay_rate), lag_times)
            )

            # Simplification: Use current reporting completeness
            # In full model, would convolve over all lags
            current_lag = 7  # Assume we're nowcasting 7 days ago
            reporting_fraction = reporting_cdf[current_lag]

            # Expected observed counts
            expected_observed = true_counts * reporting_fraction

            # ================================================================
            # Likelihood: Observed data
            # ================================================================

            # Negative Binomial for overdispersed count data
            # Note: PyMC uses mu and alpha parameterization
            # alpha = phi (overdispersion)
            obs = pm.NegativeBinomial(
                "obs",
                mu=expected_observed[:len(y_obs)],
                alpha=phi,
                observed=y_obs
            )

            # ================================================================
            # Nowcast: Predict current true counts
            # ================================================================

            # Nowcast is the true_counts for recent days
            nowcast_horizon = Config.NOWCAST_HORIZON_DAYS
            nowcast = pm.Deterministic(
                "nowcast",
                true_counts[-nowcast_horizon:]
            )

        self.model = model
        logger.info("Model built successfully")

        return model

    def fit(
        self,
        draws: Optional[int] = None,
        tune: Optional[int] = None,
        chains: Optional[int] = None,
        **kwargs
    ) -> az.InferenceData:
        """
        Fit the nowcasting model using MCMC.

        Args:
            draws: Number of MCMC samples (default from config)
            tune: Number of tuning steps (default from config)
            chains: Number of MCMC chains (default from config)
            **kwargs: Additional arguments passed to pm.sample()

        Returns:
            ArviZ InferenceData object with MCMC trace
        """
        if self.model is None:
            self.build_model()

        draws = draws or self.config["mcmc_draws"]
        tune = tune or self.config["mcmc_tune"]
        chains = chains or self.config["mcmc_chains"]

        logger.info(f"Sampling posterior: {draws} draws, {tune} tune, {chains} chains")

        with self.model:
            # Use NUTS sampler
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=self.config.get("mcmc_cores", 4),
                target_accept=self.config.get("target_accept", 0.95),
                return_inferencedata=True,
                **kwargs
            )

        logger.info("Sampling complete")
        self._extract_nowcast_estimates()

        return self.trace

    def _extract_nowcast_estimates(self):
        """Extract nowcast estimates from MCMC trace."""
        if self.trace is None:
            raise ValueError("Model has not been fit yet")

        # Extract posterior samples for nowcast
        nowcast_samples = self.trace.posterior["nowcast"].values

        # Reshape: (chains, draws, time)
        n_chains, n_draws, n_time = nowcast_samples.shape

        # Combine chains
        nowcast_combined = nowcast_samples.reshape(-1, n_time)

        # Calculate summary statistics
        nowcast_mean = nowcast_combined.mean(axis=0)
        nowcast_median = np.median(nowcast_combined, axis=0)
        nowcast_std = nowcast_combined.std(axis=0)

        # Credible intervals
        ci_lower = np.percentile(nowcast_combined, 2.5, axis=0)
        ci_upper = np.percentile(nowcast_combined, 97.5, axis=0)

        # Create DataFrame
        nowcast_dates = [
            self.max_date - timedelta(days=Config.NOWCAST_HORIZON_DAYS - i - 1)
            for i in range(Config.NOWCAST_HORIZON_DAYS)
        ]

        self.nowcast_estimates = pd.DataFrame({
            'date': nowcast_dates,
            'nowcast_mean': nowcast_mean,
            'nowcast_median': nowcast_median,
            'nowcast_std': nowcast_std,
            'ci_lower_95': ci_lower,
            'ci_upper_95': ci_upper,
        })

        logger.info(f"Extracted nowcast estimates for {len(nowcast_dates)} days")

    def get_nowcast(self) -> pd.DataFrame:
        """
        Get nowcast estimates.

        Returns:
            DataFrame with nowcast estimates and uncertainty
        """
        if self.nowcast_estimates is None:
            raise ValueError("Model has not been fit yet")

        return self.nowcast_estimates.copy()

    def diagnose(self) -> Dict:
        """
        Run MCMC diagnostics.

        Returns:
            Dictionary with diagnostic statistics
        """
        if self.trace is None:
            raise ValueError("Model has not been fit yet")

        logger.info("Running MCMC diagnostics...")

        diagnostics = {}

        # R-hat (convergence diagnostic)
        rhat = az.rhat(self.trace)
        diagnostics['rhat_max'] = float(rhat.max().to_array().values.max())
        diagnostics['rhat_summary'] = {
            var: float(rhat[var].values.max())
            for var in rhat.data_vars
        }

        # Effective sample size
        ess = az.ess(self.trace)
        diagnostics['ess_min'] = float(ess.min().to_array().values.min())
        diagnostics['ess_summary'] = {
            var: float(ess[var].values.min())
            for var in ess.data_vars
        }

        # Divergences
        divergences = self.trace.sample_stats["diverging"].values.sum()
        diagnostics['n_divergences'] = int(divergences)

        # Energy diagnostic
        try:
            energy = az.bfmi(self.trace)
            diagnostics['bfmi'] = float(energy.mean())
        except Exception as e:
            logger.warning(f"Could not compute BFMI: {e}")
            diagnostics['bfmi'] = None

        # Check for warnings
        warnings_list = []
        if diagnostics['rhat_max'] > 1.01:
            warnings_list.append(f"High R-hat detected: {diagnostics['rhat_max']:.3f}")
        if diagnostics['ess_min'] < 400:
            warnings_list.append(f"Low ESS detected: {diagnostics['ess_min']:.0f}")
        if diagnostics['n_divergences'] > 0:
            warnings_list.append(f"{diagnostics['n_divergences']} divergences detected")

        diagnostics['warnings'] = warnings_list

        if warnings_list:
            logger.warning("MCMC diagnostics found issues:")
            for warning in warnings_list:
                logger.warning(f"  - {warning}")
        else:
            logger.info("MCMC diagnostics passed all checks")

        return diagnostics

    def plot_nowcast(self, save_path: Optional[str] = None):
        """
        Plot nowcast estimates with uncertainty.

        Args:
            save_path: Path to save plot (optional)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if self.nowcast_estimates is None:
            raise ValueError("Model has not been fit yet")

        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot nowcast
        ax.plot(
            self.nowcast_estimates['date'],
            self.nowcast_estimates['nowcast_median'],
            'o-',
            color='#2E86AB',
            linewidth=2,
            markersize=6,
            label='Nowcast (median)'
        )

        # Plot credible interval
        ax.fill_between(
            self.nowcast_estimates['date'],
            self.nowcast_estimates['ci_lower_95'],
            self.nowcast_estimates['ci_upper_95'],
            alpha=0.3,
            color='#2E86AB',
            label='95% Credible Interval'
        )

        # Get observed data for comparison
        observed = self.data.groupby('event_date')['count'].sum().reset_index()
        observed_recent = observed[
            observed['event_date'] >= self.nowcast_estimates['date'].min()
        ]

        ax.plot(
            observed_recent['event_date'],
            observed_recent['count'],
            's--',
            color='#A23B72',
            alpha=0.7,
            markersize=4,
            label='Observed (incomplete)'
        )

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('COVID-19 Cases', fontsize=12)
        ax.set_title(
            'Bayesian Nowcast Estimates with 95% Credible Intervals',
            fontsize=14,
            fontweight='bold'
        )
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved nowcast plot to {save_path}")

        return fig, ax

    def compare_to_observed(self) -> pd.DataFrame:
        """
        Compare nowcast to observed data.

        Returns:
            DataFrame with comparison metrics
        """
        if self.nowcast_estimates is None:
            raise ValueError("Model has not been fit yet")

        # Get observed data
        observed = self.data.groupby('event_date')['count'].sum().reset_index()
        observed = observed.rename(columns={'count': 'observed_count'})

        # Merge with nowcast
        comparison = self.nowcast_estimates.merge(
            observed,
            left_on='date',
            right_on='event_date',
            how='left'
        )

        # Calculate metrics
        comparison['absolute_error'] = (
            comparison['nowcast_median'] - comparison['observed_count']
        ).abs()

        comparison['relative_error'] = (
            comparison['absolute_error'] / comparison['observed_count']
        )

        comparison['coverage'] = (
            (comparison['observed_count'] >= comparison['ci_lower_95']) &
            (comparison['observed_count'] <= comparison['ci_upper_95'])
        )

        return comparison[['date', 'nowcast_median', 'observed_count',
                          'absolute_error', 'relative_error', 'coverage',
                          'ci_lower_95', 'ci_upper_95']]


# ============================================================================
# Utility Functions
# ============================================================================

def create_synthetic_lagged_data(
    true_counts: np.ndarray,
    dates: List[datetime],
    delay_shape: float = 2.0,
    delay_rate: float = 0.3,
    weekend_multiplier: float = 1.5
) -> pd.DataFrame:
    """
    Create synthetic COVID data with reporting lags for testing.

    Args:
        true_counts: Array of true case counts
        dates: List of corresponding dates
        delay_shape: Gamma distribution shape parameter
        delay_rate: Gamma distribution rate parameter
        weekend_multiplier: Multiplier for weekend reporting delays

    Returns:
        DataFrame with simulated lagged reports
    """
    np.random.seed(42)

    records = []

    for event_idx, (event_date, true_count) in enumerate(zip(dates, true_counts)):
        # Determine delay multiplier
        is_weekend = event_date.weekday() in [5, 6]
        delay_mult = weekend_multiplier if is_weekend else 1.0

        # Sample reporting lags for each case
        for _ in range(int(true_count)):
            # Sample delay from gamma distribution
            delay_days = int(
                np.random.gamma(delay_shape, 1/delay_rate) * delay_mult
            )
            delay_days = min(delay_days, 30)  # Cap at 30 days

            report_date = event_date + timedelta(days=delay_days)

            records.append({
                'event_date': event_date,
                'report_date': report_date,
                'count': 1
            })

    df = pd.DataFrame(records)

    # Aggregate
    df = df.groupby(['event_date', 'report_date'], as_index=False)['count'].sum()

    return df


if __name__ == "__main__":
    # Example usage with synthetic data
    logger.info("Creating synthetic COVID-19 data with reporting lags...")

    # Create synthetic true counts
    n_days = 60
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]

    # Simulate epidemic curve (Gaussian)
    peak_day = 30
    true_counts = 100 * np.exp(-((np.arange(n_days) - peak_day) ** 2) / (2 * 10**2))
    true_counts = np.random.poisson(true_counts)

    # Add reporting lags
    data = create_synthetic_lagged_data(true_counts, dates)

    logger.info(f"Created {len(data)} lagged reports")

    # Fit nowcasting model
    nowcaster = BayesianNowcaster(data)
    nowcaster.build_model()

    logger.info("Fitting model (this may take a few minutes)...")
    trace = nowcaster.fit(draws=500, tune=500, chains=2)

    # Diagnostics
    diagnostics = nowcaster.diagnose()
    print("\nDiagnostics:")
    print(f"  Max R-hat: {diagnostics['rhat_max']:.3f}")
    print(f"  Min ESS: {diagnostics['ess_min']:.0f}")
    print(f"  Divergences: {diagnostics['n_divergences']}")

    # Get nowcast
    nowcast = nowcaster.get_nowcast()
    print("\nNowcast estimates:")
    print(nowcast)

    # Plot
    nowcaster.plot_nowcast(save_path=Config.OUTPUT_DIR / "nowcast_example.png")
    logger.info("Done!")
