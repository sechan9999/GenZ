"""
COVID-19 Bayesian Nowcasting and Data Imputation System - Main Entry Point.

This module orchestrates the three major components:
1. Bayesian hierarchical nowcasting (reporting lags)
2. MICE imputation (missing race/ethnicity)
3. Positivity rate standardization (inconsistent lab definitions)

Usage:
    python main.py --input data/covid_data.csv --output output/results/
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger

from config import Config
from nowcasting import BayesianNowcaster, create_synthetic_lagged_data
from imputation import CensusTractProxy, MICEImputer, validate_imputation
from positivity_standardization import PositivityStandardizer, simulate_heterogeneous_lab_data


# ============================================================================
# Main Pipeline
# ============================================================================

class COVID19ReportingPipeline:
    """
    End-to-end pipeline for COVID-19 data processing.

    Integrates nowcasting, imputation, and standardization into a
    unified workflow with comprehensive reporting.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        save_outputs: bool = True
    ):
        """
        Initialize pipeline.

        Args:
            output_dir: Directory for output files
            save_outputs: Whether to save intermediate outputs
        """
        self.output_dir = output_dir or Config.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.save_outputs = save_outputs

        # Pipeline components
        self.nowcaster = None
        self.imputer = None
        self.standardizer = None

        # Results
        self.results = {}

        logger.info(f"Initialized pipeline (output: {self.output_dir})")

    def run_full_pipeline(
        self,
        case_data: pd.DataFrame,
        patient_data: pd.DataFrame,
        test_data: pd.DataFrame,
        census_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Run complete pipeline on COVID-19 data.

        Args:
            case_data: Case reports with event_date, report_date, count
            patient_data: Patient records with demographics
            test_data: Test results with lab info
            census_data: Census tract data (optional, will fetch if not provided)

        Returns:
            Dictionary with all results
        """
        logger.info("="*80)
        logger.info("COVID-19 BAYESIAN REPORTING PIPELINE")
        logger.info("="*80)

        start_time = datetime.now()

        # ====================================================================
        # 1. BAYESIAN NOWCASTING (Reporting Lags)
        # ====================================================================

        logger.info("\n[1/3] BAYESIAN NOWCASTING")
        logger.info("-" * 80)

        try:
            self.nowcaster = BayesianNowcaster(case_data)
            self.nowcaster.build_model()
            self.nowcaster.fit(draws=1000, tune=1000, chains=2)

            # Get nowcast estimates
            nowcast = self.nowcaster.get_nowcast()
            diagnostics_nowcast = self.nowcaster.diagnose()

            self.results['nowcast'] = {
                'estimates': nowcast,
                'diagnostics': diagnostics_nowcast
            }

            logger.success(f"✓ Nowcasting complete ({len(nowcast)} days)")

            # Save outputs
            if self.save_outputs:
                nowcast.to_csv(self.output_dir / "nowcast_estimates.csv", index=False)
                self.nowcaster.plot_nowcast(
                    save_path=self.output_dir / "nowcast_plot.png"
                )

        except Exception as e:
            logger.error(f"✗ Nowcasting failed: {e}")
            self.results['nowcast'] = {'error': str(e)}

        # ====================================================================
        # 2. MICE IMPUTATION (Missing Race/Ethnicity)
        # ====================================================================

        logger.info("\n[2/3] MICE IMPUTATION")
        logger.info("-" * 80)

        try:
            # Fetch census data if not provided
            if census_data is None:
                census_proxy = CensusTractProxy()
                census_data = census_proxy.fetch_census_data('01')  # Example state

                # Link patients to census tracts
                patient_data = census_proxy.link_to_census_tract(patient_data)

            # Run MICE imputation
            self.imputer = MICEImputer(
                n_imputations=Config.MICE_CONFIG['n_imputations'],
                max_iter=Config.MICE_CONFIG['max_iterations']
            )

            imputed_datasets = self.imputer.fit_transform(
                patient_data,
                census_data,
                target_col='race_ethnicity'
            )

            # Validation
            diagnostics_mice = self.imputer.get_diagnostics()
            validation_metrics = validate_imputation(
                patient_data,
                imputed_datasets,
                target_col='race_ethnicity'
            )

            self.results['imputation'] = {
                'n_imputations': len(imputed_datasets),
                'diagnostics': diagnostics_mice,
                'validation': validation_metrics
            }

            logger.success(
                f"✓ MICE imputation complete "
                f"(FMI: {diagnostics_mice['fmi']:.3f})"
            )

            # Save outputs
            if self.save_outputs:
                for i, df in enumerate(imputed_datasets):
                    df.to_csv(
                        self.output_dir / f"imputed_dataset_{i+1}.csv",
                        index=False
                    )

                # Save diagnostics
                pd.DataFrame([diagnostics_mice]).to_csv(
                    self.output_dir / "mice_diagnostics.csv",
                    index=False
                )

        except Exception as e:
            logger.error(f"✗ MICE imputation failed: {e}")
            self.results['imputation'] = {'error': str(e)}

        # ====================================================================
        # 3. POSITIVITY STANDARDIZATION (Lab Heterogeneity)
        # ====================================================================

        logger.info("\n[3/3] POSITIVITY STANDARDIZATION")
        logger.info("-" * 80)

        try:
            self.standardizer = PositivityStandardizer()
            standardized_rates = self.standardizer.standardize_positivity(
                test_data,
                lab_col='lab_name',
                test_type_col='test_type',
                result_col='result',
                ct_value_col='ct_value'
            )

            self.results['positivity'] = {
                'standardized_rates': standardized_rates
            }

            logger.success(
                f"✓ Positivity standardization complete "
                f"({len(standardized_rates)} labs)"
            )

            # Save outputs
            if self.save_outputs:
                standardized_rates.to_csv(
                    self.output_dir / "standardized_positivity.csv",
                    index=False
                )

                self.standardizer.plot_comparison(
                    save_path=self.output_dir / "positivity_comparison.png"
                )

        except Exception as e:
            logger.error(f"✗ Positivity standardization failed: {e}")
            self.results['positivity'] = {'error': str(e)}

        # ====================================================================
        # SUMMARY
        # ====================================================================

        elapsed = (datetime.now() - start_time).total_seconds()

        logger.info("\n" + "="*80)
        logger.info("PIPELINE SUMMARY")
        logger.info("="*80)
        logger.info(f"Execution time: {elapsed:.1f} seconds")
        logger.info(f"Output directory: {self.output_dir}")

        # Print summary stats
        if 'nowcast' in self.results and 'estimates' in self.results['nowcast']:
            nowcast_latest = self.results['nowcast']['estimates'].iloc[-1]
            logger.info(
                f"\nNowcast (latest): {nowcast_latest['nowcast_median']:.0f} cases "
                f"[95% CI: {nowcast_latest['ci_lower_95']:.0f}-{nowcast_latest['ci_upper_95']:.0f}]"
            )

        if 'imputation' in self.results and 'diagnostics' in self.results['imputation']:
            fmi = self.results['imputation']['diagnostics']['fmi']
            logger.info(
                f"\nImputation quality: FMI = {fmi:.3f} "
                f"({'Good' if fmi < 0.3 else 'Moderate' if fmi < 0.5 else 'Poor'})"
            )

        if 'positivity' in self.results and 'standardized_rates' in self.results['positivity']:
            rates = self.results['positivity']['standardized_rates']
            mean_adj = rates['adjustment_factor'].mean()
            logger.info(
                f"\nPositivity standardization: Mean adjustment = {mean_adj:.3f}"
            )

        logger.success("\n✓ Pipeline complete!")

        return self.results

    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate comprehensive markdown report.

        Args:
            output_path: Path to save report (optional)

        Returns:
            Report as markdown string
        """
        report_lines = [
            "# COVID-19 Bayesian Nowcasting and Imputation Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
        ]

        # Nowcasting section
        if 'nowcast' in self.results:
            report_lines.extend([
                "## 1. Bayesian Nowcasting (Reporting Lags)",
                "",
            ])

            if 'estimates' in self.results['nowcast']:
                nowcast = self.results['nowcast']['estimates']
                diagnostics = self.results['nowcast']['diagnostics']

                report_lines.extend([
                    f"**Status:** ✓ Success",
                    f"**Horizon:** {len(nowcast)} days",
                    f"**MCMC Diagnostics:**",
                    f"- Max R-hat: {diagnostics.get('rhat_max', 'N/A')}",
                    f"- Min ESS: {diagnostics.get('ess_min', 'N/A')}",
                    f"- Divergences: {diagnostics.get('n_divergences', 'N/A')}",
                    "",
                    "**Latest Nowcast Estimates:**",
                    "",
                    "| Date | Median | 95% CI Lower | 95% CI Upper |",
                    "|------|--------|--------------|--------------|",
                ])

                for _, row in nowcast.tail(7).iterrows():
                    report_lines.append(
                        f"| {row['date'].strftime('%Y-%m-%d')} | "
                        f"{row['nowcast_median']:.0f} | "
                        f"{row['ci_lower_95']:.0f} | "
                        f"{row['ci_upper_95']:.0f} |"
                    )
            else:
                report_lines.append(f"**Status:** ✗ Failed - {self.results['nowcast'].get('error', 'Unknown error')}")

            report_lines.extend(["", "---", ""])

        # Imputation section
        if 'imputation' in self.results:
            report_lines.extend([
                "## 2. MICE Imputation (Missing Race/Ethnicity)",
                "",
            ])

            if 'diagnostics' in self.results['imputation']:
                diag = self.results['imputation']['diagnostics']
                val = self.results['imputation']['validation']

                report_lines.extend([
                    f"**Status:** ✓ Success",
                    f"**Number of Imputations:** {self.results['imputation']['n_imputations']}",
                    f"**Missing Rate:** {diag['missing_fraction']*100:.1f}%",
                    "",
                    "**Quality Metrics:**",
                    f"- Fraction of Missing Information (FMI): {diag['fmi']:.3f}",
                    f"- Relative Increase in Variance (λ): {diag['lambda']:.3f}",
                    f"- KL Divergence: {val['kl_divergence']:.4f}",
                ])
            else:
                report_lines.append(f"**Status:** ✗ Failed - {self.results['imputation'].get('error', 'Unknown error')}")

            report_lines.extend(["", "---", ""])

        # Positivity section
        if 'positivity' in self.results:
            report_lines.extend([
                "## 3. Positivity Rate Standardization",
                "",
            ])

            if 'standardized_rates' in self.results['positivity']:
                rates = self.results['positivity']['standardized_rates']

                report_lines.extend([
                    f"**Status:** ✓ Success",
                    f"**Labs Analyzed:** {len(rates)}",
                    "",
                    "**Standardized Positivity Rates:**",
                    "",
                    "| Lab | Tests | Observed | Standardized | Adjustment |",
                    "|-----|-------|----------|--------------|------------|",
                ])

                for _, row in rates.iterrows():
                    report_lines.append(
                        f"| {row['lab_name']} | "
                        f"{row['n_total']} | "
                        f"{row['observed_positivity']*100:.2f}% | "
                        f"{row['standardized_positivity']*100:.2f}% | "
                        f"{row['adjustment_factor']:.3f} |"
                    )
            else:
                report_lines.append(f"**Status:** ✗ Failed - {self.results['positivity'].get('error', 'Unknown error')}")

        report_md = "\n".join(report_lines)

        # Save if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_md)
            logger.info(f"Saved report to {output_path}")

        return report_md


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="COVID-19 Bayesian Nowcasting and Data Imputation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with synthetic demo data
  python main.py --demo

  # Run with real data
  python main.py --cases data/cases.csv --patients data/patients.csv --tests data/tests.csv

  # Customize output location
  python main.py --demo --output results/2024-01-15/
        """
    )

    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run with synthetic demo data'
    )

    parser.add_argument(
        '--cases',
        type=str,
        help='Path to case data CSV (columns: event_date, report_date, count)'
    )

    parser.add_argument(
        '--patients',
        type=str,
        help='Path to patient data CSV (with race_ethnicity column)'
    )

    parser.add_argument(
        '--tests',
        type=str,
        help='Path to test data CSV (columns: lab_name, test_type, result)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=str(Config.OUTPUT_DIR),
        help=f'Output directory (default: {Config.OUTPUT_DIR})'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save intermediate outputs'
    )

    args = parser.parse_args()

    # Configure logger
    logger.remove()
    logger.add(
        sys.stdout,
        format=Config.LOG_FORMAT,
        level=Config.LOG_LEVEL
    )
    logger.add(
        Config.LOG_FILE,
        format=Config.LOG_FORMAT,
        level="DEBUG"
    )

    # Load data
    if args.demo:
        logger.info("Generating synthetic demo data...")

        # Create synthetic case data
        n_days = 60
        dates = [datetime(2024, 1, 1) + pd.Timedelta(days=i) for i in range(n_days)]
        true_counts = 100 * np.exp(-((np.arange(n_days) - 30) ** 2) / (2 * 10**2))
        true_counts = np.random.poisson(true_counts)
        case_data = create_synthetic_lagged_data(true_counts, dates)

        # Create synthetic patient data
        n_patients = 1000
        patient_data = pd.DataFrame({
            'patient_id': range(n_patients),
            'age': np.random.normal(55, 15, n_patients).clip(18, 100),
            'gender': np.random.choice(['M', 'F'], n_patients),
            'zip_code': np.random.choice([f'{i:05d}' for i in range(10000, 10100)], n_patients),
            'facility_type': np.random.choice(['Hospital', 'Clinic'], n_patients),
        })

        # Add race with 35% missing
        races = Config.RACE_ETHNICITY_CATEGORIES[:-1]
        true_race = np.random.choice(
            races, n_patients,
            p=[0.5, 0.15, 0.2, 0.08, 0.05, 0.01, 0.008, 0.002]
        )
        missing_mask = np.random.rand(n_patients) < 0.35
        patient_data['race_ethnicity'] = true_race
        patient_data.loc[missing_mask, 'race_ethnicity'] = np.nan

        # Create synthetic test data
        test_data = simulate_heterogeneous_lab_data(n_tests=5000, true_prevalence=0.10)

    else:
        if not all([args.cases, args.patients, args.tests]):
            logger.error("Must provide --cases, --patients, and --tests, or use --demo")
            sys.exit(1)

        logger.info("Loading data from files...")
        case_data = pd.read_csv(args.cases)
        patient_data = pd.read_csv(args.patients)
        test_data = pd.read_csv(args.tests)

    # Run pipeline
    pipeline = COVID19ReportingPipeline(
        output_dir=Path(args.output),
        save_outputs=not args.no_save
    )

    results = pipeline.run_full_pipeline(
        case_data=case_data,
        patient_data=patient_data,
        test_data=test_data
    )

    # Generate report
    report_path = Path(args.output) / "report.md"
    pipeline.generate_report(output_path=report_path)

    logger.info(f"\n✓ All outputs saved to: {args.output}")


if __name__ == "__main__":
    main()
