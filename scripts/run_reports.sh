#!/bin/bash
################################################################################
# Quick-Start Script for State Line-List Report Automation
#
# Purpose: One-command execution of the complete reporting workflow
#
# Usage:
#   ./scripts/run_reports.sh                    # Use sample data
#   ./scripts/run_reports.sh /path/to/data.csv  # Use custom data
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     State Line-List Report Automation - Quick Start           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if R is installed
if ! command -v Rscript &> /dev/null; then
    echo -e "${RED}ERROR: R is not installed or not in PATH${NC}"
    echo "Please install R from: https://www.r-project.org/"
    exit 1
fi

echo -e "${GREEN}✓${NC} R is installed: $(Rscript --version 2>&1 | head -n1)"
echo ""

# Check for required R packages
echo -e "${YELLOW}Checking R package dependencies...${NC}"

Rscript -e "
packages <- c('tidyverse', 'readxl', 'writexl', 'openxlsx', 'lubridate', 'glue', 'logger')
missing <- packages[!(packages %in% installed.packages()[,'Package'])]

if(length(missing) > 0) {
  cat('Missing packages:', paste(missing, collapse=', '), '\n')
  cat('Installing missing packages...\n')
  install.packages(missing, repos='https://cloud.r-project.org/', quiet=TRUE)
  cat('✓ All packages installed successfully\n')
} else {
  cat('✓ All required packages are installed\n')
}
" || {
    echo -e "${RED}ERROR: Failed to install R packages${NC}"
    exit 1
}

echo ""

# Determine input file
if [ $# -eq 0 ]; then
    # No arguments - generate and use sample data
    echo -e "${YELLOW}No input file specified. Generating sample data...${NC}"
    echo ""

    # Create directories
    mkdir -p data/raw
    mkdir -p output/state_reports
    mkdir -p logs

    # Generate sample data
    Rscript scripts/generate_sample_data.R || {
        echo -e "${RED}ERROR: Failed to generate sample data${NC}"
        exit 1
    }

    INPUT_FILE="data/raw/surveillance_sample.csv"
    echo ""
    echo -e "${GREEN}✓${NC} Sample data generated: ${INPUT_FILE}"

else
    # Use provided input file
    INPUT_FILE="$1"

    if [ ! -f "$INPUT_FILE" ]; then
        echo -e "${RED}ERROR: Input file not found: ${INPUT_FILE}${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓${NC} Using input file: ${INPUT_FILE}"
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Starting report generation...${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Run the automation
START_TIME=$(date +%s)

Rscript scripts/automate_state_reports.R "$INPUT_FILE" || {
    echo ""
    echo -e "${RED}ERROR: Report generation failed${NC}"
    echo "Check logs in the logs/ directory for details"
    exit 1
}

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Report Generation Complete!${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "Total time: ${DURATION} seconds"
echo ""

# Count output files
REPORT_COUNT=$(find output/state_reports -name "*.xlsx" -type f 2>/dev/null | wc -l)
echo -e "${GREEN}Generated ${REPORT_COUNT} Excel reports${NC}"
echo ""

# Show output directory
echo -e "Output location: ${BLUE}output/state_reports/${NC}"
echo ""

# Show sample of generated files
echo -e "${YELLOW}Sample of generated reports:${NC}"
find output/state_reports -name "*.xlsx" -type f 2>/dev/null | head -10 | while read file; do
    SIZE=$(ls -lh "$file" | awk '{print $5}')
    FILENAME=$(basename "$file")
    echo -e "  • ${FILENAME} (${SIZE})"
done

if [ $REPORT_COUNT -gt 10 ]; then
    echo -e "  ... and $((REPORT_COUNT - 10)) more reports"
fi

echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "  1. Review reports in: ${BLUE}output/state_reports/${NC}"
echo -e "  2. Check logs in: ${BLUE}logs/${NC}"
echo -e "  3. Distribute reports to stakeholders"
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Automation Complete! 🎉                     ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

exit 0
