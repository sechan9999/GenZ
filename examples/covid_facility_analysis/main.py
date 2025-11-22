#!/usr/bin/env python3
"""
VA COVID-19 Facility Data Analysis Pipeline

This is the main orchestration script that automates the complete
analysis pipeline:
1. Extract data from EHR, staffing, and PPE sources
2. Perform fuzzy matching deduplication
3. Merge data on facility + date keys
4. Analyze trends and detect anomalies
5. Generate executive dashboards and reports

Usage:
    python main.py --ehr data/ehr.csv --staffing data/staffing.csv --ppe data/ppe.csv
    python main.py --config config.yaml
    python main.py --help
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import yaml

from etl_pipeline import FacilityDataETL
from analysis import CovidFacilityAnalyzer
from reporting import DashboardGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('covid_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class CovidAnalysisPipeline:
    """
    Complete COVID-19 facility data analysis pipeline.

    Orchestrates ETL, analysis, and reporting.
    """

    def __init__(self, config: Dict):
        """
        Initialize pipeline with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.etl = None
        self.analyzer = None
        self.dashboard_gen = None

        logger.info("Pipeline initialized")

    def run(self) -> Dict[str, any]:
        """
        Run complete pipeline.

        Returns:
            Dictionary with pipeline results and output paths
        """
        logger.info("=" * 100)
        logger.info("VA COVID-19 FACILITY DATA ANALYSIS PIPELINE")
        logger.info("=" * 100)

        start_time = datetime.now()
        results = {
            'start_time': start_time,
            'status': 'running'
        }

        try:
            # Step 1: ETL
            logger.info("\n" + "=" * 100)
            logger.info("STEP 1: EXTRACT, TRANSFORM, LOAD (ETL)")
            logger.info("=" * 100)

            etl_stats = self._run_etl()
            results['etl'] = etl_stats

            # Step 2: Analysis
            logger.info("\n" + "=" * 100)
            logger.info("STEP 2: STATISTICAL ANALYSIS")
            logger.info("=" * 100)

            analysis_results = self._run_analysis()
            results['analysis'] = analysis_results

            # Step 3: Reporting
            logger.info("\n" + "=" * 100)
            logger.info("STEP 3: REPORT AND DASHBOARD GENERATION")
            logger.info("=" * 100)

            report_paths = self._run_reporting()
            results['reports'] = report_paths

            # Pipeline complete
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            results['end_time'] = end_time
            results['duration_seconds'] = duration
            results['status'] = 'completed'

            self._print_final_summary(results)

            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            results['status'] = 'failed'
            results['error'] = str(e)
            raise

    def _run_etl(self) -> Dict:
        """Run ETL pipeline."""
        logger.info("Initializing ETL pipeline...")

        self.etl = FacilityDataETL(
            fuzzy_match_threshold=self.config.get('fuzzy_match_threshold', 85),
            date_tolerance_days=self.config.get('date_tolerance_days', 0)
        )

        stats = self.etl.run_pipeline(
            ehr_file=self.config['ehr_file'],
            staffing_file=self.config['staffing_file'],
            ppe_file=self.config['ppe_file'],
            output_file=self.config.get('merged_output', 'output/merged_data.csv')
        )

        # Export data quality reports
        dq_report = self.etl.get_data_quality_report()
        if not dq_report.empty:
            dq_path = Path(self.config.get('output_dir', 'output')) / 'data_quality_issues.csv'
            dq_report.to_csv(dq_path, index=False)
            logger.info(f"Data quality report: {dq_path} ({len(dq_report)} issues)")

        dedup_report = self.etl.get_deduplication_report()
        if not dedup_report.empty:
            dedup_path = Path(self.config.get('output_dir', 'output')) / 'deduplication_matches.csv'
            dedup_report.to_csv(dedup_path, index=False)
            logger.info(f"Deduplication report: {dedup_path} ({len(dedup_report)} matches)")

        return stats

    def _run_analysis(self) -> Dict:
        """Run statistical analysis."""
        logger.info("Initializing analyzer...")

        self.analyzer = CovidFacilityAnalyzer(self.etl.merged_snapshots)

        # Generate executive summary
        summary = self.analyzer.generate_executive_summary()

        # Log key findings
        kpis = summary['kpis']
        logger.info("\n" + "-" * 80)
        logger.info("EXECUTIVE SUMMARY")
        logger.info("-" * 80)
        logger.info(f"Report Date: {kpis['report_date']}")
        logger.info(f"Facilities: {kpis['facilities_reporting']}")
        logger.info(f"COVID-19 Positive: {kpis['total_covid_positive']:,}")
        logger.info(f"Hospitalized: {kpis['total_covid_hospitalized']:,}")
        logger.info(f"ICU: {kpis['total_covid_icu']:,}")
        logger.info(f"Deaths: {kpis['total_covid_deaths']:,}")
        logger.info(f"Avg Occupancy: {kpis['avg_occupancy_rate']:.1%}")
        logger.info(f"Avg Positivity Rate: {kpis['avg_positivity_rate']:.1%}")
        logger.info("-" * 80)

        # Detect outbreaks
        outbreaks = self.analyzer.detect_outbreaks(
            threshold_std=self.config.get('outbreak_threshold_std', 2.0),
            min_cases=self.config.get('outbreak_min_cases', 10)
        )

        if not outbreaks.empty:
            logger.warning(f"\n⚠️  {len(outbreaks)} potential outbreaks detected:")
            for _, outbreak in outbreaks.head(5).iterrows():
                logger.warning(
                    f"  - {outbreak['facility_name']}: "
                    f"{int(outbreak['covid_positive'])} cases "
                    f"(threshold: {outbreak['outbreak_threshold']:.1f})"
                )

            outbreak_path = Path(self.config.get('output_dir', 'output')) / 'outbreak_alerts.csv'
            outbreaks.to_csv(outbreak_path, index=False)
            logger.info(f"\nOutbreak alerts saved: {outbreak_path}")

        # Capacity analysis
        capacity = self.analyzer.analyze_capacity_constraints(
            occupancy_threshold=self.config.get('occupancy_threshold', 0.85)
        )

        if capacity['facilities_high_occupancy'] > 0:
            logger.warning(
                f"\n⚠️  {capacity['facilities_high_occupancy']} facilities "
                f"at high occupancy (>{self.config.get('occupancy_threshold', 0.85):.0%})"
            )

        return {
            'summary': summary,
            'outbreaks_detected': len(outbreaks),
            'high_occupancy_facilities': capacity['facilities_high_occupancy']
        }

    def _run_reporting(self) -> Dict[str, str]:
        """Generate reports and dashboards."""
        logger.info("Initializing report generator...")

        output_dir = self.config.get('output_dir', 'output')
        self.dashboard_gen = DashboardGenerator(
            self.analyzer,
            output_dir=output_dir
        )

        # Generate all reports
        report_paths = self.dashboard_gen.generate_all_reports()

        return report_paths

    def _print_final_summary(self, results: Dict) -> None:
        """Print final pipeline summary."""
        logger.info("\n" + "=" * 100)
        logger.info("PIPELINE EXECUTION COMPLETE")
        logger.info("=" * 100)

        logger.info(f"\nExecution Time: {results['duration_seconds']:.2f} seconds")

        logger.info("\n📊 ETL Statistics:")
        etl = results['etl']
        logger.info(f"  Records Extracted: {etl['records_extracted']['total']:,}")
        logger.info(f"  Duplicates Removed: {etl['deduplication']['total_duplicates']:,}")
        logger.info(f"  Snapshots Created: {etl['snapshots_created']:,}")
        logger.info(f"  Data Quality Issues: {etl['data_quality_issues']:,}")

        logger.info("\n📈 Analysis Results:")
        analysis = results['analysis']
        logger.info(f"  Outbreaks Detected: {analysis['outbreaks_detected']}")
        logger.info(f"  High Occupancy Facilities: {analysis['high_occupancy_facilities']}")

        logger.info("\n📄 Generated Reports:")
        for report_type, path in results['reports'].items():
            logger.info(f"  {report_type}: {path}")

        logger.info("\n" + "=" * 100)
        logger.info("✅ All tasks completed successfully!")
        logger.info("=" * 100)


def load_config(config_file: Optional[str] = None) -> Dict:
    """
    Load configuration from YAML file or command-line arguments.

    Args:
        config_file: Path to YAML config file (optional)

    Returns:
        Configuration dictionary
    """
    if config_file and Path(config_file).exists():
        logger.info(f"Loading configuration from {config_file}")
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    else:
        return {}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='VA COVID-19 Facility Data Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using individual data files
  python main.py --ehr data/ehr.csv --staffing data/staffing.csv --ppe data/ppe.csv

  # Using configuration file
  python main.py --config config.yaml

  # With custom output directory
  python main.py --config config.yaml --output reports/

  # With custom thresholds
  python main.py --config config.yaml --fuzzy-threshold 90 --outbreak-threshold 2.5
        """
    )

    # Configuration file
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file'
    )

    # Data files
    parser.add_argument(
        '--ehr',
        type=str,
        help='Path to EHR data CSV file'
    )
    parser.add_argument(
        '--staffing',
        type=str,
        help='Path to staffing roster CSV file'
    )
    parser.add_argument(
        '--ppe',
        type=str,
        help='Path to PPE inventory CSV file'
    )

    # Output options
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Output directory for reports (default: output/)'
    )
    parser.add_argument(
        '--merged-output',
        type=str,
        help='Path to save merged data CSV (default: output/merged_data.csv)'
    )

    # Analysis parameters
    parser.add_argument(
        '--fuzzy-threshold',
        type=int,
        default=85,
        help='Fuzzy matching threshold 0-100 (default: 85)'
    )
    parser.add_argument(
        '--date-tolerance',
        type=int,
        default=0,
        help='Date tolerance in days for matching (default: 0)'
    )
    parser.add_argument(
        '--outbreak-threshold',
        type=float,
        default=2.0,
        help='Standard deviations for outbreak detection (default: 2.0)'
    )
    parser.add_argument(
        '--outbreak-min-cases',
        type=int,
        default=10,
        help='Minimum cases to flag outbreak (default: 10)'
    )
    parser.add_argument(
        '--occupancy-threshold',
        type=float,
        default=0.85,
        help='Occupancy rate threshold for alerts (default: 0.85)'
    )

    # Logging
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    config = load_config(args.config)

    # Override config with command-line arguments
    if args.ehr:
        config['ehr_file'] = args.ehr
    if args.staffing:
        config['staffing_file'] = args.staffing
    if args.ppe:
        config['ppe_file'] = args.ppe

    config['output_dir'] = args.output

    if args.merged_output:
        config['merged_output'] = args.merged_output

    config['fuzzy_match_threshold'] = args.fuzzy_threshold
    config['date_tolerance_days'] = args.date_tolerance
    config['outbreak_threshold_std'] = args.outbreak_threshold
    config['outbreak_min_cases'] = args.outbreak_min_cases
    config['occupancy_threshold'] = args.occupancy_threshold

    # Validate required parameters
    required = ['ehr_file', 'staffing_file', 'ppe_file']
    missing = [r for r in required if r not in config]

    if missing:
        logger.error(f"Missing required parameters: {', '.join(missing)}")
        logger.error("Provide them via --config file or command-line arguments")
        sys.exit(1)

    # Ensure output directory exists
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)

    # Run pipeline
    try:
        pipeline = CovidAnalysisPipeline(config)
        results = pipeline.run()

        # Exit with success
        sys.exit(0)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
