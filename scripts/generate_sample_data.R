#!/usr/bin/env Rscript
################################################################################
# Sample Surveillance Data Generator
#
# Purpose: Generate realistic sample COVID-19 surveillance data for testing
#          the state report automation scripts
#
# Author: Data Analytics Team
# Date: 2025-11-23
################################################################################

library(tidyverse)
library(lubridate)

# Set seed for reproducibility
set.seed(42)

# Configuration
N_RECORDS <- 5000  # Total number of records to generate
OUTPUT_FILE <- "data/raw/surveillance_sample.csv"

# Jurisdictions
STATES <- c(
  "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
  "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
  "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
  "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
  "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
  "DC", "PR", "GU"
)

# Generate sample data
generate_sample_data <- function(n_records = N_RECORDS) {

  cat("Generating", n_records, "sample records...\n")

  # Generate base data
  data <- tibble(
    case_id = paste0("CASE-", str_pad(1:n_records, 8, pad = "0")),

    # State distribution (weighted by population)
    state = sample(STATES, n_records, replace = TRUE, prob = c(
      # Higher weights for populous states
      rep(2, 5),   # CA, TX, FL, NY, PA
      rep(1.5, 10), # Other large states
      rep(1, 38)   # Remaining states/territories
    )),

    # Facility information
    facility_name = sample(c(
      paste0("Regional Hospital ", 1:20),
      paste0("Community Medical Center ", 1:15),
      paste0("Veterans Home ", 1:10),
      paste0("Nursing Facility ", 1:25),
      paste0("Rehabilitation Center ", 1:10)
    ), n_records, replace = TRUE),

    facility_type = sample(c(
      "HOSPITAL",
      "NURSING_HOME",
      "STATE_VETERANS_HOME",
      "REHAB_FACILITY",
      "COMMUNITY_HEALTH"
    ), n_records, replace = TRUE, prob = c(0.40, 0.25, 0.10, 0.15, 0.10)),

    # Patient status
    patient_status = sample(c(
      "OUTPATIENT",
      "HOSPITALIZED",
      "DISCHARGED",
      "TRANSFERRED"
    ), n_records, replace = TRUE, prob = c(0.50, 0.30, 0.15, 0.05)),

    # Ventilation status (higher probability if hospitalized)
    ventilation_status = NA_character_,

    # ICU status
    icu_status = NA_character_,

    # Booster status
    booster_status = sample(c(
      "UNVACCINATED",
      "VACCINATED_NO_BOOSTER",
      "BOOSTED_1",
      "BOOSTED_2",
      "BOOSTED_3",
      "UNKNOWN"
    ), n_records, replace = TRUE, prob = c(0.15, 0.20, 0.35, 0.20, 0.05, 0.05)),

    # Outcome
    outcome = sample(c(
      "RECOVERED",
      "RECOVERING",
      "DECEASED",
      "UNKNOWN"
    ), n_records, replace = TRUE, prob = c(0.60, 0.25, 0.08, 0.07)),

    # Report date (last 30 days)
    report_date = sample(seq(Sys.Date() - days(30), Sys.Date(), by = "day"),
                        n_records, replace = TRUE),

    # Demographics
    age = pmax(18, pmin(100, round(rnorm(n_records, mean = 65, sd = 18)))),

    sex = sample(c("M", "F", "U"), n_records, replace = TRUE, prob = c(0.48, 0.48, 0.04)),

    race = sample(c(
      "WHITE",
      "BLACK",
      "ASIAN",
      "NATIVE_AMERICAN",
      "PACIFIC_ISLANDER",
      "MULTIRACIAL",
      "UNKNOWN"
    ), n_records, replace = TRUE, prob = c(0.60, 0.18, 0.08, 0.03, 0.02, 0.05, 0.04)),

    ethnicity = sample(c(
      "NOT_HISPANIC",
      "HISPANIC",
      "UNKNOWN"
    ), n_records, replace = TRUE, prob = c(0.75, 0.20, 0.05))
  )

  # Set ventilation status based on patient status
  data <- data %>%
    mutate(
      ventilation_status = case_when(
        patient_status == "HOSPITALIZED" ~ sample(c("ON_VENTILATOR", "NOT_ON_VENTILATOR"),
                                                   n(), replace = TRUE, prob = c(0.15, 0.85)),
        TRUE ~ "NOT_APPLICABLE"
      )
    )

  # Set ICU status based on patient status
  data <- data %>%
    mutate(
      icu_status = case_when(
        patient_status == "HOSPITALIZED" ~ sample(c("ICU_ADMITTED", "NOT_IN_ICU"),
                                                   n(), replace = TRUE, prob = c(0.20, 0.80)),
        TRUE ~ "NOT_APPLICABLE"
      )
    )

  cat("Sample data generation complete!\n")
  cat("Total records:", nrow(data), "\n")
  cat("Date range:", min(data$report_date), "to", max(data$report_date), "\n")
  cat("States represented:", n_distinct(data$state), "\n")
  cat("Facilities:", n_distinct(data$facility_name), "\n")
  cat("SVH facilities:", sum(data$facility_type == "STATE_VETERANS_HOME"), "records\n")

  return(data)
}

# Main execution
main <- function() {
  # Create output directory
  dir.create(dirname(OUTPUT_FILE), recursive = TRUE, showWarnings = FALSE)

  # Generate data
  sample_data <- generate_sample_data()

  # Save to CSV
  write_csv(sample_data, OUTPUT_FILE)
  cat("\nSample data saved to:", OUTPUT_FILE, "\n")

  # Display summary
  cat("\n=== DATA SUMMARY ===\n")
  cat("Ventilation cases:", sum(sample_data$ventilation_status == "ON_VENTILATOR"), "\n")
  cat("ICU admissions:", sum(sample_data$icu_status == "ICU_ADMITTED"), "\n")
  cat("Hospitalizations:", sum(sample_data$patient_status == "HOSPITALIZED"), "\n")
  cat("Deaths:", sum(sample_data$outcome == "DECEASED"), "\n")
  cat("Boosted (any):", sum(grepl("BOOSTED", sample_data$booster_status)), "\n")

  return(sample_data)
}

# Run if executed directly
if (!interactive()) {
  main()
}
