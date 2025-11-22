"""
Statistical Analysis for Quasi-Experimental Designs

Implements analysis methods for:
1. Stepped-wedge cluster RCT (mixed-effects models)
2. Interrupted time-series (segmented regression)
3. Equity stratification analyses
4. KPI calculations
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from datetime import date, datetime
import logging
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.power import zt_ind_solve_power
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.study_design_models import (
    SteppedWedgeDesign,
    InterruptedTimeSeriesDesign,
    SegmentedRegressionModel
)
from models.outcome_models import StudyKPIs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SteppedWedgeAnalysis:
    """
    Statistical analysis for stepped-wedge cluster randomized trials

    Implements mixed-effects regression models accounting for:
    - Clustering (ICC)
    - Time trends
    - Intervention effect
    """

    def __init__(self, design: SteppedWedgeDesign):
        """
        Initialize stepped-wedge analysis

        Args:
            design: Stepped-wedge design configuration
        """
        self.design = design
        self.data: Optional[pd.DataFrame] = None
        self.model_binary: Optional[sm.GLM] = None
        self.model_continuous: Optional[MixedLM] = None
        logger.info(f"Initialized stepped-wedge analysis for design {design.design_id}")

    def prepare_data(
        self,
        data: pd.DataFrame,
        outcome_col: str,
        patient_id_col: str = "patient_id",
        cluster_id_col: str = "cluster_id",
        time_col: str = "time",
        intervention_col: str = "intervention"
    ) -> pd.DataFrame:
        """
        Prepare data for stepped-wedge analysis

        Args:
            data: Raw data DataFrame
            outcome_col: Column name for outcome variable
            patient_id_col: Patient identifier column
            cluster_id_col: Cluster identifier column
            time_col: Time period column
            intervention_col: Binary intervention indicator (0/1)

        Returns:
            Prepared DataFrame with required columns
        """
        required_cols = [patient_id_col, cluster_id_col, time_col, intervention_col, outcome_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Standardize column names
        df = data.copy()
        df = df.rename(columns={
            patient_id_col: "patient_id",
            cluster_id_col: "cluster_id",
            time_col: "time",
            intervention_col: "intervention",
            outcome_col: "outcome"
        })

        # Center time variable (improves convergence)
        df['time_centered'] = df['time'] - df['time'].mean()

        # Create cluster-time identifier
        df['cluster_time'] = df['cluster_id'].astype(str) + "_" + df['time'].astype(str)

        self.data = df
        logger.info(f"Prepared data: {len(df)} observations, {df['cluster_id'].nunique()} clusters")

        return df

    def analyze_binary_outcome(
        self,
        covariates: Optional[List[str]] = None,
        random_effects: str = "cluster"
    ) -> Dict:
        """
        Analyze binary outcome (e.g., 30-day readmission) using logistic regression

        Args:
            covariates: List of covariate column names for adjustment
            random_effects: Random effects structure ("cluster" or "cluster_time")

        Returns:
            Dictionary with model results
        """
        if self.data is None:
            raise ValueError("Must call prepare_data() first")

        # Build formula
        formula = "outcome ~ intervention + time_centered"
        if covariates:
            formula += " + " + " + ".join(covariates)

        # For demonstration, using GLM with cluster-robust SE
        # In production, use glmer from R via rpy2 for proper mixed-effects logistic
        model = smf.glm(
            formula=formula,
            data=self.data,
            family=sm.families.Binomial()
        ).fit(cov_type='cluster', cov_kwds={'groups': self.data['cluster_id']})

        # Extract results
        intervention_coef = model.params['intervention']
        intervention_se = model.bse['intervention']
        intervention_pval = model.pvalues['intervention']

        # Calculate odds ratio and 95% CI
        or_estimate = np.exp(intervention_coef)
        or_ci_lower = np.exp(intervention_coef - 1.96 * intervention_se)
        or_ci_upper = np.exp(intervention_coef + 1.96 * intervention_se)

        self.model_binary = model

        results = {
            "model_type": "Logistic regression (cluster-robust SE)",
            "outcome": "Binary",
            "n_observations": len(self.data),
            "n_clusters": self.data['cluster_id'].nunique(),
            "intervention_coefficient": intervention_coef,
            "intervention_se": intervention_se,
            "intervention_pvalue": intervention_pval,
            "odds_ratio": or_estimate,
            "or_95ci_lower": or_ci_lower,
            "or_95ci_upper": or_ci_upper,
            "significant": intervention_pval < 0.05,
            "aic": model.aic,
            "bic": model.bic
        }

        logger.info(f"Binary outcome analysis complete: OR={or_estimate:.3f}, p={intervention_pval:.4f}")

        return results

    def analyze_continuous_outcome(
        self,
        covariates: Optional[List[str]] = None
    ) -> Dict:
        """
        Analyze continuous outcome (e.g., CAT score) using linear mixed-effects model

        Args:
            covariates: List of covariate column names

        Returns:
            Dictionary with model results
        """
        if self.data is None:
            raise ValueError("Must call prepare_data() first")

        # Build formula
        formula = "outcome ~ intervention + time_centered"
        if covariates:
            formula += " + " + " + ".join(covariates)

        # Fit mixed-effects model with random intercept for cluster
        try:
            model = smf.mixedlm(
                formula=formula,
                data=self.data,
                groups=self.data['cluster_id']
            ).fit()

            self.model_continuous = model

            # Extract results
            intervention_coef = model.params['intervention']
            intervention_se = model.bse['intervention']
            intervention_pval = model.pvalues['intervention']
            intervention_ci_lower = model.conf_int().loc['intervention', 0]
            intervention_ci_upper = model.conf_int().loc['intervention', 1]

            # Calculate ICC
            var_cluster = model.cov_re.iloc[0, 0]
            var_residual = model.scale
            icc = var_cluster / (var_cluster + var_residual)

            results = {
                "model_type": "Linear mixed-effects model",
                "outcome": "Continuous",
                "n_observations": len(self.data),
                "n_clusters": self.data['cluster_id'].nunique(),
                "intervention_coefficient": intervention_coef,
                "intervention_se": intervention_se,
                "intervention_pvalue": intervention_pval,
                "mean_difference": intervention_coef,
                "ci_95_lower": intervention_ci_lower,
                "ci_95_upper": intervention_ci_upper,
                "significant": intervention_pval < 0.05,
                "icc": icc,
                "aic": model.aic,
                "bic": model.bic
            }

            logger.info(f"Continuous outcome analysis complete: MD={intervention_coef:.3f}, p={intervention_pval:.4f}, ICC={icc:.3f}")

            return results

        except Exception as e:
            logger.error(f"Mixed-effects model failed: {e}")
            return {"error": str(e)}

    def calculate_adjusted_effect_sizes(self) -> Dict:
        """
        Calculate clinically meaningful effect size metrics

        Returns:
            Dictionary with effect sizes
        """
        if self.data is None:
            raise ValueError("No data available")

        # Calculate unadjusted means by intervention group
        control_mean = self.data[self.data['intervention'] == 0]['outcome'].mean()
        intervention_mean = self.data[self.data['intervention'] == 1]['outcome'].mean()
        pooled_sd = self.data['outcome'].std()

        # Cohen's d
        cohens_d = (intervention_mean - control_mean) / pooled_sd

        # Absolute and relative effect
        absolute_effect = intervention_mean - control_mean
        relative_effect = absolute_effect / control_mean if control_mean != 0 else 0

        return {
            "control_mean": control_mean,
            "intervention_mean": intervention_mean,
            "absolute_difference": absolute_effect,
            "relative_difference_percent": relative_effect * 100,
            "cohens_d": cohens_d,
            "effect_size_interpretation": self._interpret_cohens_d(cohens_d)
        }

    @staticmethod
    def _interpret_cohens_d(d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "Negligible"
        elif abs_d < 0.5:
            return "Small"
        elif abs_d < 0.8:
            return "Medium"
        else:
            return "Large"


class InterruptedTimeSeriesAnalysis:
    """
    Interrupted time-series analysis with segmented regression

    Model: Y_t = β0 + β1*Time + β2*Intervention + β3*Time_after + ε_t
    """

    def __init__(self, design: InterruptedTimeSeriesDesign):
        """
        Initialize ITS analysis

        Args:
            design: ITS design configuration
        """
        self.design = design
        self.data: Optional[pd.DataFrame] = None
        self.model: Optional[sm.OLS] = None
        logger.info(f"Initialized ITS analysis for design {design.design_id}")

    def prepare_its_data(
        self,
        data: pd.DataFrame,
        time_col: str = "time",
        outcome_col: str = "outcome",
        intervention_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Prepare data for interrupted time-series analysis

        Args:
            data: Raw time series data
            time_col: Time variable column (sequential: 0, 1, 2, ...)
            outcome_col: Outcome variable column
            intervention_date: Date of intervention (if not in design)

        Returns:
            Prepared DataFrame
        """
        df = data.copy()

        # Determine intervention time point
        if intervention_date is None:
            intervention_date = self.design.intervention_date

        # Create time variables
        if time_col not in df.columns:
            df['time'] = range(len(df))
        else:
            df = df.rename(columns={time_col: 'time'})

        # Create intervention indicator (0 before, 1 after)
        df['intervention'] = (df['time'] >= self.design.pre_intervention_periods).astype(int)

        # Create time after intervention (0 before, 1, 2, 3... after)
        df['time_after'] = np.where(
            df['intervention'] == 1,
            df['time'] - self.design.pre_intervention_periods + 1,
            0
        )

        # Rename outcome column
        if outcome_col != 'outcome':
            df = df.rename(columns={outcome_col: 'outcome'})

        # Add seasonality variables if needed
        if self.design.adjust_for_seasonality:
            df['month'] = (df['time'] % 12) + 1
            df['quarter'] = ((df['time'] % 12) // 3) + 1

        self.data = df
        logger.info(f"Prepared ITS data: {len(df)} time points, intervention at t={self.design.pre_intervention_periods}")

        return df

    def fit_segmented_regression(
        self,
        autocorrelation: bool = True,
        seasonal_adjustment: bool = False
    ) -> SegmentedRegressionModel:
        """
        Fit segmented regression model for ITS

        Args:
            autocorrelation: Whether to adjust for autocorrelation (Newey-West SE)
            seasonal_adjustment: Whether to include seasonal dummy variables

        Returns:
            SegmentedRegressionModel with fitted parameters
        """
        if self.data is None:
            raise ValueError("Must call prepare_its_data() first")

        # Build formula
        formula = "outcome ~ time + intervention + time_after"

        if seasonal_adjustment and 'month' in self.data.columns:
            formula += " + C(month)"

        # Fit OLS model
        model = smf.ols(formula=formula, data=self.data).fit()

        # Apply Newey-West robust standard errors if autocorrelation
        if autocorrelation:
            # Determine lag order (rule of thumb: sqrt(T))
            lag_order = int(np.sqrt(len(self.data)))
            model = smf.ols(formula=formula, data=self.data).fit(
                cov_type='HAC',
                cov_kwds={'maxlags': lag_order}
            )

        self.model = model

        # Extract coefficients
        beta_0 = model.params['Intercept']
        beta_1 = model.params['time']
        beta_2 = model.params['intervention']
        beta_3 = model.params['time_after']

        # Test for autocorrelation (Durbin-Watson)
        from statsmodels.stats.stattools import durbin_watson
        dw_stat = durbin_watson(model.resid)

        # Create SegmentedRegressionModel
        seg_model = SegmentedRegressionModel(
            model_id=f"ITS-{self.design.design_id}",
            design_id=self.design.design_id,
            outcome_name="outcome",
            outcome_type="Continuous",
            includes_time_trend=True,
            includes_level_change=True,
            includes_slope_change=True,
            seasonal_adjustment=seasonal_adjustment,
            autocorrelation_adjustment=autocorrelation,
            lag_order=lag_order if autocorrelation else 0,
            beta_0_intercept=beta_0,
            beta_1_baseline_trend=beta_1,
            beta_2_level_change=beta_2,
            beta_3_slope_change=beta_3,
            r_squared=model.rsquared,
            aic=model.aic,
            bic=model.bic,
            durbin_watson=dw_stat
        )

        logger.info(f"Fitted segmented regression: Level change={beta_2:.3f}, Slope change={beta_3:.3f}, R²={model.rsquared:.3f}")

        return seg_model

    def interpret_its_results(self, seg_model: SegmentedRegressionModel) -> Dict:
        """
        Interpret segmented regression results

        Args:
            seg_model: Fitted segmented regression model

        Returns:
            Dictionary with interpretation
        """
        # Immediate effect
        immediate_effect = seg_model.beta_2_level_change
        immediate_pval = self.model.pvalues['intervention']

        # Slope change
        slope_change = seg_model.beta_3_slope_change
        slope_pval = self.model.pvalues['time_after']

        # Calculate cumulative effect over post-intervention period
        post_periods = self.design.post_intervention_periods
        cumulative_effect = immediate_effect + (slope_change * post_periods)

        # Counterfactual at end of study
        end_time = len(self.data) - 1
        counterfactual = seg_model.predicted_counterfactual(end_time)
        actual = self.data.iloc[-1]['outcome']
        total_effect = actual - counterfactual if counterfactual else None

        interpretation = {
            "immediate_level_change": immediate_effect,
            "immediate_significant": immediate_pval < 0.05,
            "immediate_interpretation": "Increase" if immediate_effect > 0 else "Decrease",

            "slope_change": slope_change,
            "slope_significant": slope_pval < 0.05,
            "slope_interpretation": "Accelerating" if slope_change > 0 else "Decelerating",

            "cumulative_effect": cumulative_effect,
            "total_effect_vs_counterfactual": total_effect,

            "model_fit": {
                "r_squared": seg_model.r_squared,
                "aic": seg_model.aic,
                "durbin_watson": seg_model.durbin_watson,
                "autocorrelation_detected": seg_model.durbin_watson < 1.5 or seg_model.durbin_watson > 2.5
            }
        }

        return interpretation

    def plot_its(self, save_path: Optional[str] = None):
        """
        Plot interrupted time-series with fitted model

        Args:
            save_path: Path to save figure (optional)
        """
        import matplotlib.pyplot as plt

        if self.data is None or self.model is None:
            raise ValueError("Must fit model first")

        # Get predictions
        self.data['predicted'] = self.model.predict(self.data)

        # Create counterfactual (extend pre-intervention trend)
        intervention_point = self.design.pre_intervention_periods
        pre_data = self.data[self.data['time'] < intervention_point]
        pre_model = smf.ols('outcome ~ time', data=pre_data).fit()
        self.data['counterfactual'] = pre_model.predict(self.data)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Actual data
        ax.scatter(self.data['time'], self.data['outcome'], alpha=0.6, label='Observed', color='black')

        # Pre-intervention trend
        pre_times = self.data[self.data['time'] < intervention_point]['time']
        ax.plot(pre_times, self.data.loc[pre_times.index, 'predicted'],
                color='blue', linewidth=2, label='Pre-intervention trend')

        # Post-intervention trend
        post_times = self.data[self.data['time'] >= intervention_point]['time']
        ax.plot(post_times, self.data.loc[post_times.index, 'predicted'],
                color='red', linewidth=2, label='Post-intervention trend')

        # Counterfactual
        ax.plot(post_times, self.data.loc[post_times.index, 'counterfactual'],
                color='gray', linestyle='--', linewidth=2, label='Counterfactual')

        # Intervention line
        ax.axvline(x=intervention_point, color='red', linestyle='--', alpha=0.7, label='Intervention')

        ax.set_xlabel('Time Period')
        ax.set_ylabel('Outcome')
        ax.set_title('Interrupted Time-Series Analysis')
        ax.legend()
        ax.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ITS plot saved to {save_path}")

        plt.tight_layout()
        return fig


class EquityStratificationAnalysis:
    """
    Equity-focused stratified analyses

    Examines differential intervention effects across:
    - Race/ethnicity
    - Rurality
    - Digital literacy
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize equity analysis

        Args:
            data: DataFrame with outcomes and equity variables
        """
        self.data = data
        logger.info(f"Initialized equity analysis with {len(data)} observations")

    def analyze_by_race_ethnicity(
        self,
        outcome_col: str = "outcome",
        race_col: str = "race_ethnicity",
        intervention_col: str = "intervention"
    ) -> Dict:
        """
        Stratified analysis by race/ethnicity

        Args:
            outcome_col: Outcome variable column
            race_col: Race/ethnicity column
            intervention_col: Intervention indicator column

        Returns:
            Dictionary with stratified results and disparity metrics
        """
        results = {}

        # Get unique race/ethnicity categories
        categories = self.data[race_col].unique()

        for category in categories:
            subset = self.data[self.data[race_col] == category]

            # Calculate outcome rates by intervention status
            control = subset[subset[intervention_col] == 0][outcome_col]
            intervention = subset[subset[intervention_col] == 1][outcome_col]

            if len(control) > 0 and len(intervention) > 0:
                # Calculate means
                control_mean = control.mean()
                intervention_mean = intervention.mean()

                # Statistical test
                if subset[outcome_col].dtype == 'bool' or subset[outcome_col].nunique() == 2:
                    # Binary outcome - chi-square test
                    contingency = pd.crosstab(subset[intervention_col], subset[outcome_col])
                    chi2, pval, dof, expected = chi2_contingency(contingency)
                    test_statistic = chi2
                else:
                    # Continuous outcome - t-test
                    t_stat, pval = ttest_ind(intervention, control)
                    test_statistic = t_stat

                results[category] = {
                    "n": len(subset),
                    "control_mean": control_mean,
                    "intervention_mean": intervention_mean,
                    "absolute_difference": intervention_mean - control_mean,
                    "relative_difference_percent": ((intervention_mean - control_mean) / control_mean * 100) if control_mean != 0 else 0,
                    "test_statistic": test_statistic,
                    "p_value": pval,
                    "significant": pval < 0.05
                }

        # Calculate disparity ratios (relative to White non-Hispanic if available)
        if "Non-Hispanic White" in results:
            reference_rate = results["Non-Hispanic White"]["intervention_mean"]
            for category in results:
                if category != "Non-Hispanic White":
                    results[category]["disparity_ratio"] = results[category]["intervention_mean"] / reference_rate if reference_rate != 0 else None

        logger.info(f"Race/ethnicity stratified analysis complete for {len(results)} categories")

        return results

    def analyze_by_rurality(
        self,
        outcome_col: str = "outcome",
        rurality_col: str = "rurality",
        intervention_col: str = "intervention"
    ) -> Dict:
        """
        Stratified analysis by rurality

        Args:
            outcome_col: Outcome variable
            rurality_col: Rurality category column
            intervention_col: Intervention indicator

        Returns:
            Dictionary with stratified results
        """
        return self._generic_stratified_analysis(
            outcome_col=outcome_col,
            strata_col=rurality_col,
            intervention_col=intervention_col,
            strata_name="Rurality"
        )

    def analyze_by_digital_literacy(
        self,
        outcome_col: str = "outcome",
        literacy_col: str = "digital_literacy",
        intervention_col: str = "intervention"
    ) -> Dict:
        """
        Stratified analysis by digital literacy

        Args:
            outcome_col: Outcome variable
            literacy_col: Digital literacy level column
            intervention_col: Intervention indicator

        Returns:
            Dictionary with stratified results
        """
        return self._generic_stratified_analysis(
            outcome_col=outcome_col,
            strata_col=literacy_col,
            intervention_col=intervention_col,
            strata_name="Digital Literacy"
        )

    def _generic_stratified_analysis(
        self,
        outcome_col: str,
        strata_col: str,
        intervention_col: str,
        strata_name: str
    ) -> Dict:
        """Generic stratified analysis function"""
        results = {"strata_name": strata_name, "categories": {}}

        categories = self.data[strata_col].unique()

        for category in categories:
            subset = self.data[self.data[strata_col] == category]

            control = subset[subset[intervention_col] == 0][outcome_col]
            intervention = subset[subset[intervention_col] == 1][outcome_col]

            if len(control) > 0 and len(intervention) > 0:
                control_mean = control.mean()
                intervention_mean = intervention.mean()

                # T-test
                t_stat, pval = ttest_ind(intervention, control, equal_var=False)

                results["categories"][category] = {
                    "n": len(subset),
                    "n_control": len(control),
                    "n_intervention": len(intervention),
                    "control_mean": control_mean,
                    "intervention_mean": intervention_mean,
                    "absolute_difference": intervention_mean - control_mean,
                    "test_statistic": t_stat,
                    "p_value": pval,
                    "significant": pval < 0.05
                }

        logger.info(f"{strata_name} stratified analysis complete for {len(results['categories'])} categories")

        return results

    def test_interaction(
        self,
        outcome_col: str,
        intervention_col: str,
        strata_col: str
    ) -> Dict:
        """
        Test for intervention × strata interaction

        Args:
            outcome_col: Outcome variable
            intervention_col: Intervention indicator
            strata_col: Stratification variable

        Returns:
            Dictionary with interaction test results
        """
        # Fit model with interaction term
        formula = f"{outcome_col} ~ {intervention_col} * C({strata_col})"

        try:
            model = smf.ols(formula=formula, data=self.data).fit()

            # Extract interaction term p-values
            interaction_terms = [param for param in model.params.index if ':' in param]
            interaction_pvals = [model.pvalues[term] for term in interaction_terms]

            # Overall interaction test (F-test)
            # Compare model with vs without interaction
            reduced_formula = f"{outcome_col} ~ {intervention_col} + C({strata_col})"
            reduced_model = smf.ols(formula=reduced_formula, data=self.data).fit()

            # Likelihood ratio test
            lr_stat = 2 * (model.llf - reduced_model.llf)
            df = len(interaction_terms)
            lr_pval = 1 - stats.chi2.cdf(lr_stat, df)

            results = {
                "interaction_significant": lr_pval < 0.05,
                "lr_statistic": lr_stat,
                "lr_pvalue": lr_pval,
                "degrees_of_freedom": df,
                "individual_interactions": dict(zip(interaction_terms, interaction_pvals)),
                "interpretation": "Heterogeneous treatment effects detected" if lr_pval < 0.05 else "No significant interaction"
            }

            logger.info(f"Interaction test: {strata_col}, p={lr_pval:.4f}")

            return results

        except Exception as e:
            logger.error(f"Interaction test failed: {e}")
            return {"error": str(e)}


def calculate_kpi_summary(patient_data: List[StudyKPIs]) -> Dict:
    """
    Calculate summary statistics for primary KPIs across all patients

    Args:
        patient_data: List of StudyKPIs for all patients

    Returns:
        Dictionary with KPI summary statistics
    """
    if not patient_data:
        return {"error": "No patient data provided"}

    df = pd.DataFrame([kpi.dict() for kpi in patient_data])

    summary = {
        "n_patients": len(df),

        # Primary KPI 1: 30-day readmission
        "readmission_rate_percent": (df['readmitted_30day'].sum() / len(df) * 100),
        "readmission_count": df['readmission_count'].sum(),

        # Primary KPI 2: CAT score reduction
        "mean_cat_change": df['cat_score_change'].mean(),
        "median_cat_change": df['cat_score_change'].median(),
        "percent_achieved_mcid": (df['achieved_mcid'].sum() / len(df) * 100),

        # Primary KPI 3: Adherence
        "mean_adherence_percent": df['adherence_rate_percent'].mean(),
        "median_adherence_percent": df['adherence_rate_percent'].median(),
        "percent_high_adherence": (df['high_adherence'].sum() / len(df) * 100),

        # Composite success
        "percent_composite_success": (df.apply(lambda row: (
            not row['readmitted_30day'] and
            row['achieved_mcid'] and
            row['high_adherence']
        ), axis=1).sum() / len(df) * 100),

        # Secondary outcomes
        "mean_ed_visits": df['ed_visits_copd'].mean(),
        "mean_exacerbations": (df['exacerbations_moderate'] + df['exacerbations_severe']).mean(),
        "mean_qol_change": df['qol_change'].mean()
    }

    logger.info(f"KPI summary calculated for {len(df)} patients")

    return summary


if __name__ == "__main__":
    # Demo: Generate synthetic data and run analyses
    print("\n" + "="*60)
    print("Statistical Analysis Demo")
    print("="*60)

    # Generate synthetic stepped-wedge data
    np.random.seed(42)
    n_clusters = 20
    n_patients_per_cluster = 40
    n_time_points = 6

    sw_data = []
    for cluster in range(n_clusters):
        wedge_step = (cluster // 4) + 1  # 4 clusters per step
        for patient in range(n_patients_per_cluster):
            for time in range(n_time_points):
                intervention = 1 if time >= wedge_step else 0
                # Simulate readmission (baseline 22%, intervention reduces to 15%)
                prob = 0.22 if intervention == 0 else 0.15
                outcome = np.random.binomial(1, prob)

                sw_data.append({
                    "patient_id": f"PT-{cluster:02d}-{patient:03d}",
                    "cluster_id": f"CLINIC-{cluster:02d}",
                    "time": time,
                    "intervention": intervention,
                    "outcome": outcome
                })

    df_sw = pd.DataFrame(sw_data)

    print(f"\nGenerated synthetic stepped-wedge data: {len(df_sw)} observations")
    print(f"Clusters: {df_sw['cluster_id'].nunique()}, Time points: {df_sw['time'].nunique()}")

    # Run stepped-wedge analysis
    from models.study_design_models import SteppedWedgeDesign
    design = SteppedWedgeDesign(
        design_id="DEMO-SW",
        study_start_date=date(2025, 1, 1),
        study_end_date=date(2025, 12, 31),
        number_of_clusters=20,
        number_of_steps=5,
        step_duration_months=2,
        randomization_date=date(2024, 12, 1),
        randomization_method="Stratified",
        clusters_per_step=4
    )

    sw_analysis = SteppedWedgeAnalysis(design)
    sw_analysis.prepare_data(df_sw, outcome_col="outcome")
    results = sw_analysis.analyze_binary_outcome()

    print("\nStepped-Wedge Analysis Results:")
    print(f"  Odds Ratio: {results['odds_ratio']:.3f} (95% CI: {results['or_95ci_lower']:.3f}-{results['or_95ci_upper']:.3f})")
    print(f"  P-value: {results['intervention_pvalue']:.4f}")
    print(f"  Significant: {results['significant']}")

    print("\n" + "="*60)
