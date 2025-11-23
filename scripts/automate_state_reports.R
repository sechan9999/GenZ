#!/usr/bin/env Rscript
################################################################################
# State Line-List Report Automation
#
# Purpose: Automate generation of state-specific COVID-19 surveillance reports
#          for 53 jurisdictions (50 states + DC + PR + GU) and State Veterans
#          Homes (SVH)
#
# Author: Data Analytics Team
# Date: 2025-11-23
#
# Description:
#   This script automates the weekly reporting cycle by:
#   - Ingesting raw surveillance data
#   - Applying state-specific filters (Ventilation, Booster counts, etc.)
#   - Formatting reports with jurisdiction-specific requirements
#   - Exporting 26+ distinct reports in a single batch
#
# Impact: Reduces weekly reporting time from days to minutes
################################################################################

# Load required libraries
library(tidyverse)    # Data manipulation and visualization
library(readxl)       # Reading Excel files
library(writexl)      # Writing Excel files
library(openxlsx)     # Advanced Excel formatting
library(lubridate)    # Date handling
library(glue)         # String interpolation
library(logger)       # Logging

# Set up logging
log_threshold(INFO)
log_info("Starting State Line-List Report Automation")

################################################################################
# Configuration
################################################################################

# Define jurisdictions
JURISDICTIONS <- c(
  # 50 States
  "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
  "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
  "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
  "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
  "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
  # Territories
  "DC", "PR", "GU"
)

# Report types and their filters
REPORT_TYPES <- list(
  ventilation = list(
    name = "Ventilation_Report",
    filter_col = "ventilation_status",
    filter_value = "ON_VENTILATOR"
  ),
  booster = list(
    name = "Booster_Report",
    filter_col = "booster_status",
    filter_value = c("BOOSTED_1", "BOOSTED_2", "BOOSTED_3")
  ),
  hospitalization = list(
    name = "Hospitalization_Report",
    filter_col = "patient_status",
    filter_value = "HOSPITALIZED"
  ),
  icu_admission = list(
    name = "ICU_Report",
    filter_col = "icu_status",
    filter_value = "ICU_ADMITTED"
  ),
  death = list(
    name = "Death_Report",
    filter_col = "outcome",
    filter_value = "DECEASED"
  ),
  svh_facility = list(
    name = "SVH_Facility_Report",
    filter_col = "facility_type",
    filter_value = "STATE_VETERANS_HOME"
  )
)

# Directories
INPUT_DIR <- "data/raw"
OUTPUT_DIR <- "output/state_reports"
LOG_DIR <- "logs"

# Create directories if they don't exist
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(LOG_DIR, recursive = TRUE, showWarnings = FALSE)

# Report date (use current week ending date)
REPORT_DATE <- floor_date(Sys.Date(), "week") + days(6)  # Saturday
REPORT_WEEK <- week(REPORT_DATE)
REPORT_YEAR <- year(REPORT_DATE)

log_info(glue("Report Date: {REPORT_DATE} (Week {REPORT_WEEK}, {REPORT_YEAR})"))

################################################################################
# Data Ingestion Functions
################################################################################

#' Load raw surveillance data from source
#'
#' @param file_path Path to the raw data file
#' @return Tibble with surveillance data
load_surveillance_data <- function(file_path) {
  log_info(glue("Loading data from: {file_path}"))

  tryCatch({
    # Read data (adjust based on your data source format)
    data <- read_csv(file_path, col_types = cols(
      case_id = col_character(),
      state = col_character(),
      facility_name = col_character(),
      facility_type = col_character(),
      patient_status = col_character(),
      ventilation_status = col_character(),
      icu_status = col_character(),
      booster_status = col_character(),
      outcome = col_character(),
      report_date = col_date(format = "%Y-%m-%d"),
      age = col_integer(),
      sex = col_character(),
      race = col_character(),
      ethnicity = col_character()
    ))

    log_info(glue("Loaded {nrow(data)} records"))
    return(data)

  }, error = function(e) {
    log_error(glue("Failed to load data: {e$message}"))
    stop(e)
  })
}

################################################################################
# Data Processing Functions
################################################################################

#' Filter data for specific state/jurisdiction
#'
#' @param data Full surveillance data
#' @param state_code Two-letter state code
#' @return Filtered data for the state
filter_by_state <- function(data, state_code) {
  data %>%
    filter(state == state_code) %>%
    arrange(desc(report_date), case_id)
}

#' Apply report-specific filters
#'
#' @param data State-filtered data
#' @param report_config Report configuration from REPORT_TYPES
#' @return Filtered data based on report type
apply_report_filter <- function(data, report_config) {
  filter_col <- report_config$filter_col
  filter_value <- report_config$filter_value

  data %>%
    filter(.data[[filter_col]] %in% filter_value)
}

#' Calculate summary statistics for a state report
#'
#' @param data Filtered state data
#' @return Tibble with summary statistics
calculate_summary_stats <- function(data) {
  tibble(
    total_cases = nrow(data),
    avg_age = round(mean(data$age, na.rm = TRUE), 1),
    median_age = median(data$age, na.rm = TRUE),
    male_pct = round(sum(data$sex == "M", na.rm = TRUE) / nrow(data) * 100, 1),
    female_pct = round(sum(data$sex == "F", na.rm = TRUE) / nrow(data) * 100, 1),
    facilities = n_distinct(data$facility_name)
  )
}

################################################################################
# Report Formatting Functions
################################################################################

#' Create formatted Excel workbook with multiple sheets
#'
#' @param state_code State code
#' @param report_type Report type configuration
#' @param data Filtered data for the report
#' @param summary Summary statistics
#' @return Workbook object
create_formatted_workbook <- function(state_code, report_type, data, summary) {
  wb <- createWorkbook()

  # Sheet 1: Line List
  addWorksheet(wb, "Line_List")
  writeData(wb, "Line_List", data, startRow = 1, startCol = 1)

  # Format header row
  headerStyle <- createStyle(
    fontSize = 11,
    fontName = "Calibri",
    fontColour = "#FFFFFF",
    fgFill = "#4F81BD",
    halign = "center",
    valign = "center",
    textDecoration = "bold",
    border = "TopBottomLeftRight"
  )
  addStyle(wb, "Line_List", headerStyle, rows = 1, cols = 1:ncol(data), gridExpand = TRUE)

  # Freeze top row
  freezePane(wb, "Line_List", firstRow = TRUE)

  # Auto-size columns
  setColWidths(wb, "Line_List", cols = 1:ncol(data), widths = "auto")

  # Sheet 2: Summary
  addWorksheet(wb, "Summary")

  # Create summary table
  summary_data <- tibble(
    Metric = c(
      "Report Date",
      "State/Jurisdiction",
      "Report Type",
      "Total Cases",
      "Average Age",
      "Median Age",
      "% Male",
      "% Female",
      "Facilities Reporting"
    ),
    Value = c(
      as.character(REPORT_DATE),
      state_code,
      report_type$name,
      as.character(summary$total_cases),
      as.character(summary$avg_age),
      as.character(summary$median_age),
      paste0(summary$male_pct, "%"),
      paste0(summary$female_pct, "%"),
      as.character(summary$facilities)
    )
  )

  writeData(wb, "Summary", summary_data, startRow = 2, startCol = 2)

  # Format summary
  summaryHeaderStyle <- createStyle(
    fontSize = 12,
    fontName = "Calibri",
    fontColour = "#000000",
    fgFill = "#D9E1F2",
    halign = "left",
    valign = "center",
    textDecoration = "bold",
    border = "TopBottomLeftRight"
  )
  addStyle(wb, "Summary", summaryHeaderStyle, rows = 2, cols = 2:3, gridExpand = TRUE)

  # Title
  writeData(wb, "Summary",
            glue("{state_code} - {report_type$name}"),
            startRow = 1, startCol = 2)
  titleStyle <- createStyle(fontSize = 14, textDecoration = "bold")
  addStyle(wb, "Summary", titleStyle, rows = 1, cols = 2)

  return(wb)
}

################################################################################
# Export Functions
################################################################################

#' Export a single state report
#'
#' @param state_code State code
#' @param report_type Report type configuration
#' @param data Full surveillance data
#' @return TRUE if successful, FALSE otherwise
export_state_report <- function(state_code, report_type, data) {
  log_info(glue("Processing {state_code} - {report_type$name}"))

  tryCatch({
    # Filter data
    state_data <- filter_by_state(data, state_code)
    filtered_data <- apply_report_filter(state_data, report_type)

    # Check if there's data
    if (nrow(filtered_data) == 0) {
      log_warn(glue("No data for {state_code} - {report_type$name}, skipping"))
      return(FALSE)
    }

    # Calculate summary
    summary <- calculate_summary_stats(filtered_data)

    # Create workbook
    wb <- create_formatted_workbook(state_code, report_type, filtered_data, summary)

    # Generate filename
    filename <- glue("{state_code}_{report_type$name}_{REPORT_YEAR}W{REPORT_WEEK}.xlsx")
    filepath <- file.path(OUTPUT_DIR, filename)

    # Save workbook
    saveWorkbook(wb, filepath, overwrite = TRUE)

    log_info(glue("Exported: {filename} ({nrow(filtered_data)} records)"))
    return(TRUE)

  }, error = function(e) {
    log_error(glue("Failed to export {state_code} - {report_type$name}: {e$message}"))
    return(FALSE)
  })
}

#' Export State Veterans Home (SVH) consolidated report
#'
#' @param data Full surveillance data
#' @return TRUE if successful, FALSE otherwise
export_svh_report <- function(data) {
  log_info("Processing State Veterans Homes (SVH) Consolidated Report")

  tryCatch({
    # Filter for SVH facilities across all states
    svh_data <- data %>%
      filter(facility_type == "STATE_VETERANS_HOME") %>%
      arrange(state, facility_name, desc(report_date))

    if (nrow(svh_data) == 0) {
      log_warn("No SVH data found, skipping")
      return(FALSE)
    }

    # Create workbook with breakdown by state
    wb <- createWorkbook()

    # Sheet 1: All SVH Data
    addWorksheet(wb, "All_SVH_Facilities")
    writeData(wb, "All_SVH_Facilities", svh_data)

    # Format header
    headerStyle <- createStyle(
      fontSize = 11,
      fontColour = "#FFFFFF",
      fgFill = "#4F81BD",
      halign = "center",
      textDecoration = "bold"
    )
    addStyle(wb, "All_SVH_Facilities", headerStyle, rows = 1, cols = 1:ncol(svh_data), gridExpand = TRUE)
    freezePane(wb, "All_SVH_Facilities", firstRow = TRUE)
    setColWidths(wb, "All_SVH_Facilities", cols = 1:ncol(svh_data), widths = "auto")

    # Sheet 2: Summary by State
    addWorksheet(wb, "Summary_by_State")

    svh_summary <- svh_data %>%
      group_by(state) %>%
      summarise(
        facilities = n_distinct(facility_name),
        total_cases = n(),
        avg_age = round(mean(age, na.rm = TRUE), 1),
        hospitalized = sum(patient_status == "HOSPITALIZED", na.rm = TRUE),
        on_ventilator = sum(ventilation_status == "ON_VENTILATOR", na.rm = TRUE),
        in_icu = sum(icu_status == "ICU_ADMITTED", na.rm = TRUE),
        deceased = sum(outcome == "DECEASED", na.rm = TRUE)
      ) %>%
      arrange(desc(total_cases))

    writeData(wb, "Summary_by_State", svh_summary)
    addStyle(wb, "Summary_by_State", headerStyle, rows = 1, cols = 1:ncol(svh_summary), gridExpand = TRUE)
    freezePane(wb, "Summary_by_State", firstRow = TRUE)
    setColWidths(wb, "Summary_by_State", cols = 1:ncol(svh_summary), widths = "auto")

    # Save
    filename <- glue("SVH_Consolidated_Report_{REPORT_YEAR}W{REPORT_WEEK}.xlsx")
    filepath <- file.path(OUTPUT_DIR, filename)
    saveWorkbook(wb, filepath, overwrite = TRUE)

    log_info(glue("Exported SVH Report: {filename} ({nrow(svh_data)} records from {nrow(svh_summary)} states)"))
    return(TRUE)

  }, error = function(e) {
    log_error(glue("Failed to export SVH report: {e$message}"))
    return(FALSE)
  })
}

################################################################################
# Main Execution Function
################################################################################

#' Main function to orchestrate all report generation
#'
#' @param input_file Path to raw surveillance data file
#' @param report_types List of report types to generate (default: all)
#' @param jurisdictions List of jurisdictions to process (default: all 53)
main <- function(input_file = NULL,
                 report_types = REPORT_TYPES,
                 jurisdictions = JURISDICTIONS) {

  start_time <- Sys.time()
  log_info("="*80)
  log_info("State Line-List Report Automation - START")
  log_info("="*80)

  # Use default input file if not specified
  if (is.null(input_file)) {
    input_file <- file.path(INPUT_DIR, glue("surveillance_data_{REPORT_DATE}.csv"))
  }

  # Load data
  data <- load_surveillance_data(input_file)

  # Track export statistics
  total_reports <- 0
  successful_exports <- 0
  failed_exports <- 0

  # Generate reports for each jurisdiction and report type
  for (state in jurisdictions) {
    for (report_type in report_types) {
      total_reports <- total_reports + 1

      success <- export_state_report(state, report_type, data)

      if (success) {
        successful_exports <- successful_exports + 1
      } else {
        failed_exports <- failed_exports + 1
      }
    }
  }

  # Generate SVH consolidated report
  total_reports <- total_reports + 1
  svh_success <- export_svh_report(data)

  if (svh_success) {
    successful_exports <- successful_exports + 1
  } else {
    failed_exports <- failed_exports + 1
  }

  # Summary
  end_time <- Sys.time()
  duration <- round(as.numeric(difftime(end_time, start_time, units = "mins")), 2)

  log_info("="*80)
  log_info("REPORT GENERATION SUMMARY")
  log_info("="*80)
  log_info(glue("Total Reports Attempted: {total_reports}"))
  log_info(glue("Successful Exports: {successful_exports}"))
  log_info(glue("Failed Exports: {failed_exports}"))
  log_info(glue("Success Rate: {round(successful_exports/total_reports*100, 1)}%"))
  log_info(glue("Duration: {duration} minutes"))
  log_info(glue("Output Directory: {OUTPUT_DIR}"))
  log_info("="*80)

  # Return summary
  list(
    total = total_reports,
    successful = successful_exports,
    failed = failed_exports,
    duration_mins = duration
  )
}

################################################################################
# Command-Line Execution
################################################################################

# Check if script is being run directly (not sourced)
if (!interactive()) {
  # Parse command-line arguments
  args <- commandArgs(trailingOnly = TRUE)

  if (length(args) > 0) {
    input_file <- args[1]
    log_info(glue("Using input file from command line: {input_file}"))
    result <- main(input_file = input_file)
  } else {
    log_info("No input file specified, using default")
    result <- main()
  }

  # Exit with appropriate code
  if (result$failed > 0) {
    quit(status = 1)
  } else {
    quit(status = 0)
  }
}

################################################################################
# Example Usage (when sourced interactively)
################################################################################

# Example 1: Generate all reports for all jurisdictions
# source("scripts/automate_state_reports.R")
# result <- main(input_file = "data/raw/surveillance_2025_11_23.csv")

# Example 2: Generate only ventilation reports for select states
# result <- main(
#   input_file = "data/raw/surveillance_2025_11_23.csv",
#   report_types = list(REPORT_TYPES$ventilation),
#   jurisdictions = c("CA", "NY", "TX", "FL")
# )

# Example 3: Generate only SVH report
# data <- load_surveillance_data("data/raw/surveillance_2025_11_23.csv")
# export_svh_report(data)
