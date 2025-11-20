"""
═════════════════════════════════════════════════════════════
Gen Z Agent - Korean Election Invoice Analysis System
Multi-Agent System (CrewAI + Anthropic Claude)
Version 2.0.0 - Refactored Modular Architecture
═════════════════════════════════════════════════════════════

This is the main entry point for the Gen Z Agent system.
Run with: python -m gen_z_agent.main_refactored <file_path>
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional
import sys

from .crew import run_electoral_analysis
from .config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    """
    Main CLI entry point for Gen Z Agent.
    """
    parser = argparse.ArgumentParser(
        description="Gen Z Agent - Multi-Agent Invoice/Election Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze election data file
  python -m gen_z_agent.main_refactored data/election_2024.pdf

  # Specify analysis ID and recipients
  python -m gen_z_agent.main_refactored data/election.pdf --id ELEC001 --recipients "admin@example.com,analyst@example.com"

  # Run in production mode (actually send emails)
  python -m gen_z_agent.main_refactored data/election.pdf --production

  # Verbose output
  python -m gen_z_agent.main_refactored data/election.pdf -v

For more information, see README.md or CLAUDE.md
        """
    )

    parser.add_argument(
        "file",
        help="Path to invoice or election data file (PDF, HTML, or text)"
    )

    parser.add_argument(
        "--id",
        dest="analysis_id",
        help="Analysis ID (default: filename without extension)",
        default=None
    )

    parser.add_argument(
        "--recipients",
        help="Comma-separated email recipients (default: client@example.com)",
        default="client@example.com"
    )

    parser.add_argument(
        "--production",
        help="Run in production mode (actually send emails/notifications)",
        action="store_true"
    )

    parser.add_argument(
        "-v", "--verbose",
        help="Verbose output (use -vv for more verbosity)",
        action="count",
        default=1
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"Gen Z Agent v2.0.0"
    )

    args = parser.parse_args()

    # Validate file exists
    file_path = Path(args.file)
    if not file_path.exists():
        logger.error(f"File not found: {args.file}")
        print(f"❌ Error: File not found: {args.file}")
        sys.exit(1)

    # Parse recipients
    recipients: List[str] = [r.strip() for r in args.recipients.split(",")]

    # Set analysis ID
    analysis_id: Optional[str] = args.analysis_id or file_path.stem

    # Display configuration
    print("\n" + "="*70)
    print("Gen Z Agent - Electoral Analysis System v2.0.0")
    print("="*70)
    print(f"📄 File:       {args.file}")
    print(f"🆔 Analysis:   {analysis_id}")
    print(f"📧 Recipients: {', '.join(recipients)}")
    print(f"⚙️  Mode:       {'Production' if args.production else 'Dry Run'}")
    print(f"🗣️  Verbosity:  {args.verbose}")
    print("="*70)
    print()

    # Validate configuration
    try:
        Config.validate()
        logger.info("Configuration validated successfully")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print(f"❌ Configuration Error: {e}")
        print("\nPlease check your .env file and ensure ANTHROPIC_API_KEY is set.")
        sys.exit(1)

    # Run analysis
    try:
        logger.info(f"Starting analysis for: {args.file}")

        result = run_electoral_analysis(
            file_path=str(file_path),
            invoice_id=analysis_id,
            recipients=recipients,
            dry_run=not args.production,
            verbose=args.verbose
        )

        logger.info("Analysis completed successfully")
        print("\n" + "="*70)
        print("✅ Analysis completed successfully!")
        print("="*70)

        # Display output locations
        print(f"\n📊 Output files:")
        print(f"   - Excel:    {Config.OUTPUT_DIR}/Analysis_{analysis_id}.xlsx")
        print(f"   - Markdown: {Config.OUTPUT_DIR}/Report_{analysis_id}.md")
        print(f"   - Summary:  {Config.OUTPUT_DIR}/Email_Summary_{analysis_id}.md")
        print()

        return 0

    except KeyboardInterrupt:
        logger.warning("Analysis interrupted by user")
        print("\n\n⚠️  Analysis interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        print(f"\n❌ Analysis failed: {e}")
        print("\nCheck the log file for details: {}".format(Config.LOG_FILE))
        return 1


if __name__ == "__main__":
    sys.exit(main())
