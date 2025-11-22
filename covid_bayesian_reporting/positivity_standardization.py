"""
Positivity Rate Standardization for Inconsistent Lab Definitions.

This module standardizes COVID-19 positivity rates across labs with
different testing protocols and threshold definitions.

Challenges addressed:
1. Different PCR Ct thresholds (Quest: 37, LabCorp: 35, etc.)
2. Different antigen test sensitivities
3. Varying denominators (total tests, unique individuals, etc.)
4. Test type mixing (PCR + antigen)

Approach:
- Bayesian latent class model for true infection status
- Lab-specific sensitivity/specificity parameters
- Standardization to CDC reference definition
- Uncertainty propagation

Reference:
- Branscum et al. (2005). "Estimation of diagnostic-test sensitivity and specificity through Bayesian modeling"
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from typing import Dict, List, Tuple, Optional
from scipy import stats
from loguru import logger
import warnings

from config import Config

warnings.filterwarnings('ignore')


class PositivityStandardizer:
    """
    Standardize COVID-19 positivity rates across heterogeneous lab protocols.

    Uses Bayesian latent class modeling to estimate true positivity rate
    accounting for lab-specific test characteristics.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize positivity standardizer.

        Args:
            config: Configuration dict (defaults to Config.POSITIVITY_STANDARDIZATION)
        """
        self.config = config or Config.POSITIVITY_STANDARDIZATION

        self.model = None
        self.trace = None
        self.standardized_rates = None

        logger.info("Initialized positivity standardizer")

    def standardize_positivity(
        self,
        test_data: pd.DataFrame,
        lab_col: str = 'lab_name',
        test_type_col: str = 'test_type',
        result_col: str = 'result',
        ct_value_col: Optional[str] = 'ct_value'
    ) -> pd.DataFrame:
        """
        Standardize positivity rates across labs.

        Args:
            test_data: DataFrame with test results
                Required columns: lab_name, test_type, result
                Optional: ct_value (for PCR tests)
            lab_col: Column with lab identifier
            test_type_col: Column with test type (PCR, antigen)
            result_col: Column with test result (positive/negative)
            ct_value_col: Column with Ct values (optional)

        Returns:
            DataFrame with standardized positivity rates
        """
        logger.info(f"Standardizing positivity rates for {len(test_data)} tests")

        # Prepare data
        df = test_data.copy()

        # Apply lab-specific standardization
        df_standardized = self._apply_lab_adjustments(
            df, lab_col, test_type_col, result_col, ct_value_col
        )

        # Fit Bayesian model for uncertainty
        self._fit_latent_class_model(df_standardized, lab_col, test_type_col)

        # Calculate standardized rates
        self.standardized_rates = self._calculate_standardized_rates(
            df_standardized, lab_col
        )

        logger.info("Positivity standardization complete")

        return self.standardized_rates

    def _apply_lab_adjustments(
        self,
        data: pd.DataFrame,
        lab_col: str,
        test_type_col: str,
        result_col: str,
        ct_value_col: Optional[str]
    ) -> pd.DataFrame:
        """Apply lab-specific adjustments to test results."""
        df = data.copy()

        # Initialize standardized result
        df['standardized_result'] = df[result_col]
        df['adjustment_applied'] = 'none'

        # Get reference Ct threshold
        ref_ct = self.config['reference_definition']['ct_threshold']

        # PCR Ct threshold adjustments
        if ct_value_col and ct_value_col in df.columns:
            logger.info("Applying Ct threshold standardization...")

            pcr_mask = df[test_type_col].str.lower() == 'pcr'
            pcr_data = df[pcr_mask].copy()

            if len(pcr_data) > 0:
                # Get lab-specific thresholds
                lab_thresholds = self.config['lab_definitions']['pcr_ct_threshold']

                for lab, lab_ct in lab_thresholds.items():
                    lab_mask = (pcr_mask) & (df[lab_col] == lab)

                    if lab_mask.sum() == 0:
                        continue

                    logger.info(f"  {lab}: Ct {lab_ct} -> {ref_ct}")

                    # Re-classify based on reference threshold
                    # Original: positive if Ct < lab_ct
                    # Standardized: positive if Ct < ref_ct

                    ct_values = df.loc[lab_mask, ct_value_col]

                    # Cases where lab called positive but ref would call negative
                    # (Ct between ref_ct and lab_ct)
                    if lab_ct > ref_ct:
                        reclassify_neg = (
                            (ct_values >= ref_ct) &
                            (ct_values < lab_ct) &
                            (df.loc[lab_mask, result_col] == 'positive')
                        )

                        df.loc[lab_mask & reclassify_neg, 'standardized_result'] = 'negative'
                        df.loc[lab_mask & reclassify_neg, 'adjustment_applied'] = 'ct_threshold'

                    # Cases where lab called negative but ref would call positive
                    elif lab_ct < ref_ct:
                        reclassify_pos = (
                            (ct_values >= lab_ct) &
                            (ct_values < ref_ct) &
                            (df.loc[lab_mask, result_col] == 'negative')
                        )

                        df.loc[lab_mask & reclassify_pos, 'standardized_result'] = 'positive'
                        df.loc[lab_mask & reclassify_pos, 'adjustment_applied'] = 'ct_threshold'

        # Antigen test sensitivity adjustments
        antigen_mask = df[test_type_col].str.lower() == 'antigen'
        if antigen_mask.sum() > 0:
            logger.info("Applying antigen test sensitivity adjustments...")

            # Get test-specific sensitivities
            antigen_sens = self.config['lab_definitions']['antigen_sensitivity']

            for test_name, sensitivity in antigen_sens.items():
                test_mask = (
                    antigen_mask &
                    df[lab_col].str.contains(test_name, case=False, na=False)
                )

                if test_mask.sum() == 0:
                    continue

                logger.info(f"  {test_name}: sensitivity = {sensitivity:.2f}")

                # For negative results, probabilistically reclassify based on
                # false negative rate (1 - sensitivity)
                neg_results = test_mask & (df[result_col] == 'negative')

                if neg_results.sum() > 0:
                    # Estimate false negatives
                    # Assume PCR reference has ~95% sensitivity
                    pcr_sens = 0.95
                    fnr_adjustment = (1 - sensitivity) / (1 - pcr_sens)

                    # Mark these for probabilistic adjustment
                    df.loc[neg_results, 'fnr_multiplier'] = fnr_adjustment
                    df.loc[neg_results, 'adjustment_applied'] = 'antigen_sensitivity'

        logger.info(
            f"Applied {(df['adjustment_applied'] != 'none').sum()} adjustments"
        )

        return df

    def _fit_latent_class_model(
        self,
        data: pd.DataFrame,
        lab_col: str,
        test_type_col: str
    ):
        """
        Fit Bayesian latent class model for true infection prevalence.

        This model estimates:
        - True underlying prevalence
        - Lab-specific sensitivity/specificity
        - Uncertainty in all parameters
        """
        logger.info("Fitting Bayesian latent class model...")

        # Aggregate by lab and test type
        summary = data.groupby([lab_col, test_type_col])['standardized_result'].agg([
            ('n_positive', lambda x: (x == 'positive').sum()),
            ('n_total', 'count')
        ]).reset_index()

        n_labs = len(summary)

        if n_labs == 0:
            logger.warning("No data to fit latent class model")
            return

        # Extract data
        n_pos = summary['n_positive'].values
        n_total = summary['n_total'].values
        test_types = summary[test_type_col].values

        with pm.Model() as model:
            # ================================================================
            # Prior: True prevalence
            # ================================================================

            # True underlying prevalence (logit scale for better sampling)
            logit_prevalence = pm.Normal("logit_prevalence", mu=0, sigma=2)
            prevalence = pm.Deterministic(
                "prevalence",
                pm.math.invlogit(logit_prevalence)
            )

            # ================================================================
            # Priors: Test characteristics
            # ================================================================

            # Sensitivity priors (from config)
            sens_priors = self.config['test_performance_priors']

            # PCR sensitivity
            sens_pcr = pm.Beta(
                "sens_pcr",
                mu=sens_priors['pcr_sensitivity'][0],
                sigma=sens_priors['pcr_sensitivity'][1]
            )

            # PCR specificity
            spec_pcr = pm.Beta(
                "spec_pcr",
                mu=sens_priors['pcr_specificity'][0],
                sigma=sens_priors['pcr_specificity'][1]
            )

            # Antigen sensitivity
            sens_antigen = pm.Beta(
                "sens_antigen",
                mu=sens_priors['antigen_sensitivity'][0],
                sigma=sens_priors['antigen_sensitivity'][1]
            )

            # Antigen specificity
            spec_antigen = pm.Beta(
                "spec_antigen",
                mu=sens_priors['antigen_specificity'][0],
                sigma=sens_priors['antigen_specificity'][1]
            )

            # ================================================================
            # Likelihood: Observed test results
            # ================================================================

            # Expected positive results by test type
            for i in range(n_labs):
                if test_types[i].lower() == 'pcr':
                    sensitivity = sens_pcr
                    specificity = spec_pcr
                else:
                    sensitivity = sens_antigen
                    specificity = spec_antigen

                # Expected number of positives
                # = true_positives + false_positives
                expected_pos = (
                    prevalence * sensitivity * n_total[i] +  # True positives
                    (1 - prevalence) * (1 - specificity) * n_total[i]  # False positives
                )

                # Binomial likelihood
                pm.Binomial(
                    f"obs_{i}",
                    n=n_total[i],
                    p=expected_pos / n_total[i],
                    observed=n_pos[i]
                )

        self.model = model

        # Sample posterior
        logger.info("Sampling posterior (this may take a moment)...")

        with model:
            self.trace = pm.sample(
                draws=1000,
                tune=1000,
                chains=2,
                cores=2,
                target_accept=0.95,
                return_inferencedata=True
            )

        # Diagnostics
        rhat = az.rhat(self.trace)
        max_rhat = float(rhat.max().to_array().values.max())

        logger.info(f"Model fit complete (max R-hat: {max_rhat:.3f})")

    def _calculate_standardized_rates(
        self,
        data: pd.DataFrame,
        lab_col: str
    ) -> pd.DataFrame:
        """Calculate standardized positivity rates by lab."""
        # Calculate observed and standardized rates
        rates = data.groupby(lab_col).agg({
            'result': [
                ('n_positive_observed', lambda x: (x == 'positive').sum()),
                ('n_total', 'count')
            ],
            'standardized_result': [
                ('n_positive_standardized', lambda x: (x == 'positive').sum()),
            ]
        }).reset_index()

        rates.columns = [
            'lab_name', 'n_positive_observed', 'n_total', 'n_positive_standardized'
        ]

        # Calculate rates
        rates['observed_positivity'] = (
            rates['n_positive_observed'] / rates['n_total']
        )

        rates['standardized_positivity'] = (
            rates['n_positive_standardized'] / rates['n_total']
        )

        # Calculate adjustment
        rates['adjustment_factor'] = (
            rates['standardized_positivity'] / rates['observed_positivity']
        )

        # Add uncertainty from Bayesian model
        if self.trace is not None:
            prevalence_samples = self.trace.posterior['prevalence'].values.flatten()

            rates['bayesian_prevalence_mean'] = prevalence_samples.mean()
            rates['bayesian_prevalence_std'] = prevalence_samples.std()
            rates['bayesian_prevalence_ci_lower'] = np.percentile(prevalence_samples, 2.5)
            rates['bayesian_prevalence_ci_upper'] = np.percentile(prevalence_samples, 97.5)

        logger.info(f"Calculated standardized rates for {len(rates)} labs")

        return rates

    def get_standardized_rates(self) -> pd.DataFrame:
        """Get standardized positivity rates."""
        if self.standardized_rates is None:
            raise ValueError("Standardization has not been performed yet")

        return self.standardized_rates.copy()

    def plot_comparison(self, save_path: Optional[str] = None):
        """Plot observed vs standardized positivity rates."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        if self.standardized_rates is None:
            raise ValueError("Standardization has not been performed yet")

        rates = self.standardized_rates.sort_values('observed_positivity')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Observed vs Standardized
        x = np.arange(len(rates))
        width = 0.35

        ax1.bar(
            x - width/2,
            rates['observed_positivity'],
            width,
            label='Observed',
            color='#E76F51',
            alpha=0.8
        )

        ax1.bar(
            x + width/2,
            rates['standardized_positivity'],
            width,
            label='Standardized',
            color='#2A9D8F',
            alpha=0.8
        )

        ax1.set_xlabel('Lab', fontsize=11)
        ax1.set_ylabel('Positivity Rate', fontsize=11)
        ax1.set_title('Observed vs Standardized Positivity Rates', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(rates['lab_name'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: Adjustment factors
        ax2.barh(
            x,
            rates['adjustment_factor'],
            color=['#E63946' if f < 1 else '#06A77D' for f in rates['adjustment_factor']],
            alpha=0.8
        )

        ax2.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_yticks(x)
        ax2.set_yticklabels(rates['lab_name'])
        ax2.set_xlabel('Adjustment Factor', fontsize=11)
        ax2.set_title('Standardization Adjustment Factors', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {save_path}")

        return fig, (ax1, ax2)


# ============================================================================
# Utility Functions
# ============================================================================

def simulate_heterogeneous_lab_data(
    n_tests: int = 5000,
    true_prevalence: float = 0.10,
    n_labs: int = 4
) -> pd.DataFrame:
    """
    Simulate COVID test data with heterogeneous lab protocols.

    Args:
        n_tests: Total number of tests
        true_prevalence: True underlying infection prevalence
        n_labs: Number of labs with different protocols

    Returns:
        DataFrame with simulated test results
    """
    np.random.seed(42)

    # Lab definitions
    labs = {
        'Quest': {'ct_threshold': 37, 'test_type': 'pcr', 'sensitivity': 0.95, 'specificity': 0.995},
        'LabCorp': {'ct_threshold': 35, 'test_type': 'pcr', 'sensitivity': 0.95, 'specificity': 0.995},
        'Local_PCR': {'ct_threshold': 40, 'test_type': 'pcr', 'sensitivity': 0.95, 'specificity': 0.995},
        'Rapid_Antigen': {'ct_threshold': None, 'test_type': 'antigen', 'sensitivity': 0.75, 'specificity': 0.99},
    }

    records = []

    for _ in range(n_tests):
        # Random lab
        lab_name = np.random.choice(list(labs.keys()))
        lab_config = labs[lab_name]

        # True infection status
        is_infected = np.random.rand() < true_prevalence

        # Test result based on sensitivity/specificity
        if is_infected:
            test_positive = np.random.rand() < lab_config['sensitivity']
        else:
            test_positive = np.random.rand() > lab_config['specificity']  # False positive

        # Generate Ct value for PCR tests
        ct_value = None
        if lab_config['test_type'] == 'pcr' and test_positive:
            # Simulate Ct values (lower = higher viral load)
            ct_value = np.random.normal(25, 5) if is_infected else np.random.normal(35, 3)
            ct_value = max(15, min(40, ct_value))  # Clip to realistic range

            # Apply lab-specific threshold
            test_positive = ct_value < lab_config['ct_threshold']

        records.append({
            'test_id': len(records),
            'lab_name': lab_name,
            'test_type': lab_config['test_type'],
            'result': 'positive' if test_positive else 'negative',
            'ct_value': ct_value,
            'true_infection': is_infected  # Ground truth (normally unknown)
        })

    df = pd.DataFrame(records)

    logger.info(f"Simulated {len(df)} tests from {n_labs} labs")
    logger.info(f"True prevalence: {true_prevalence:.1%}")
    logger.info(f"Observed positivity: {(df['result'] == 'positive').mean():.1%}")

    return df


if __name__ == "__main__":
    # Example usage
    logger.info("Positivity Standardization Example")

    # Simulate heterogeneous lab data
    test_data = simulate_heterogeneous_lab_data(
        n_tests=5000,
        true_prevalence=0.10
    )

    # Standardize positivity rates
    standardizer = PositivityStandardizer()
    standardized_rates = standardizer.standardize_positivity(
        test_data,
        lab_col='lab_name',
        test_type_col='test_type',
        result_col='result',
        ct_value_col='ct_value'
    )

    print("\n" + "="*80)
    print("STANDARDIZED POSITIVITY RATES")
    print("="*80)
    print(standardized_rates[[
        'lab_name',
        'n_total',
        'observed_positivity',
        'standardized_positivity',
        'adjustment_factor'
    ]].to_string(index=False))

    # Plot comparison
    standardizer.plot_comparison(
        save_path=Config.OUTPUT_DIR / "positivity_standardization.png"
    )

    # Calculate true accuracy (using ground truth)
    true_prev = test_data['true_infection'].mean()
    print(f"\nTrue prevalence: {true_prev:.1%}")

    if standardizer.standardized_rates is not None:
        bayesian_prev = standardizer.standardized_rates['bayesian_prevalence_mean'].iloc[0]
        print(f"Bayesian estimate: {bayesian_prev:.1%}")
        print(f"Error: {abs(bayesian_prev - true_prev):.1%}")

    logger.info("Done!")
