"""
LIMS Quality Monitoring - Local Demo Script

This script demonstrates real-time quality monitoring for LIMS data
with automatic anomaly detection and Power BI dashboard simulation.

In production, this runs hourly via Azure Data Factory and updates
a Power BI dashboard in Direct Query mode.

Author: MLOps Team
Date: 2025-11-22
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import os
import json

# Optional: For visualization (comment out if not available)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Skipping visualizations.")

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_DIR = "./lims_monitoring_output"
DASHBOARD_CSV = os.path.join(OUTPUT_DIR, "dashboard_data.csv")
ALERTS_CSV = os.path.join(OUTPUT_DIR, "critical_alerts.csv")
REPORT_JSON = os.path.join(OUTPUT_DIR, "monitoring_report.json")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# STEP 1: EXTRACT (Simulating Ingestion)
# ==========================================
def extract_lims_data(num_samples=50):
    """
    Simulates pulling raw batch data from a LIMS SQL DB or FTP CSV.

    In production, this would be:
        conn = pyodbc.connect('DRIVER={SQL Server};SERVER=lims-prod;DATABASE=LIMS')
        df = pd.read_sql("SELECT * FROM batch_results WHERE timestamp > ?", conn, params=[last_run])

    Args:
        num_samples: Number of samples to generate for demo

    Returns:
        DataFrame with raw LIMS data
    """
    print("\n" + "="*60)
    print(">>> STEP 1: EXTRACTING DATA FROM LIMS")
    print("="*60)

    # Generate realistic sample data
    np.random.seed(42)

    facilities = ['GA_LTC_01', 'GA_LTC_02', 'GA_LTC_03', 'GA_LTC_04', 'GA_LTC_05']

    data = {
        'sample_id': [f'S{i:04d}' for i in range(1, num_samples + 1)],
        'facility_id': np.random.choice(facilities, num_samples),
        'batch_id': [f'BATCH_{datetime.now().strftime("%Y%m%d")}_{i//10:03d}' for i in range(num_samples)],
        'ph_level': np.random.normal(7.2, 0.15, num_samples),  # Normal: 7.0-7.4
        'storage_temp_c': np.random.normal(4.0, 0.3, num_samples),  # Normal: 2-8°C
        'dissolved_oxygen_ppm': np.random.normal(8.5, 0.5, num_samples),  # Normal: 7-10 ppm
        'turbidity_ntu': np.random.gamma(2, 0.5, num_samples),  # Normal: 0-5 NTU
        'timestamp': [datetime.now() - timedelta(hours=num_samples-i) for i in range(num_samples)]
    }

    df = pd.DataFrame(data)

    # INJECT REALISTIC ANOMALIES (this is what we want to catch!)
    # Anomaly 1: Machine error codes (-999)
    df.loc[12, 'ph_level'] = -999.0  # Machine error
    df.loc[23, 'dissolved_oxygen_ppm'] = -999.0  # Sensor failure

    # Anomaly 2: Equipment malfunction (extreme values)
    df.loc[35, 'storage_temp_c'] = 85.0  # Refrigerator heater malfunction! CRITICAL
    df.loc[36, 'storage_temp_c'] = 82.5  # Same unit degrading

    # Anomaly 3: Contamination event (multiple parameters off)
    df.loc[42:45, 'turbidity_ntu'] = np.random.uniform(15, 25, 4)  # Contamination
    df.loc[42:45, 'dissolved_oxygen_ppm'] -= 3.0  # Same samples

    # Anomaly 4: Calibration drift (subtle but systematic)
    df.loc[18:22, 'ph_level'] += 0.8  # pH sensor needs recalibration

    print(f"✓ Extracted {len(df)} samples from LIMS database")
    print(f"✓ Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"✓ Facilities: {df['facility_id'].nunique()} facilities")

    # Show a sample
    print("\nSample of raw data (first 5 rows):")
    print(df.head())

    return df


# ==========================================
# STEP 2: CLEAN (The Silver Layer)
# ==========================================
def clean_data(df):
    """
    Standardizes data for the model.
    Handles missing values and machine error codes.

    This is the "Silver Layer" in Delta Lake terminology:
    - Raw data (Bronze) → Cleaned data (Silver) → Features (Gold)

    Args:
        df: Raw DataFrame from LIMS

    Returns:
        Cleaned DataFrame with error codes handled
    """
    print("\n" + "="*60)
    print(">>> STEP 2: CLEANING AND STANDARDIZING DATA")
    print("="*60)

    df_clean = df.copy()

    # Track data quality issues
    quality_issues = {
        'machine_errors': 0,
        'out_of_range': 0,
        'missing_values': 0
    }

    # Fix machine error codes (-999 is a common LIMS error code)
    for col in ['ph_level', 'storage_temp_c', 'dissolved_oxygen_ppm', 'turbidity_ntu']:
        error_mask = df_clean[col] == -999.0
        quality_issues['machine_errors'] += error_mask.sum()
        df_clean.loc[error_mask, col] = np.nan

    # Detect out-of-range values (before imputation)
    out_of_range_conditions = [
        (df_clean['ph_level'] < 6.0) | (df_clean['ph_level'] > 9.0),
        (df_clean['storage_temp_c'] < -10) | (df_clean['storage_temp_c'] > 50),
        (df_clean['dissolved_oxygen_ppm'] < 0) | (df_clean['dissolved_oxygen_ppm'] > 20),
        (df_clean['turbidity_ntu'] < 0) | (df_clean['turbidity_ntu'] > 100)
    ]

    for condition in out_of_range_conditions:
        quality_issues['out_of_range'] += condition.sum()

    # Impute missing values with median (standard MLOps practice)
    numeric_cols = ['ph_level', 'storage_temp_c', 'dissolved_oxygen_ppm', 'turbidity_ntu']
    for col in numeric_cols:
        missing_count = df_clean[col].isna().sum()
        if missing_count > 0:
            quality_issues['missing_values'] += missing_count
            median_value = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_value)
            print(f"  - Imputed {missing_count} missing values in {col} with median: {median_value:.2f}")

    print(f"\n✓ Data quality issues found:")
    print(f"  - Machine errors (-999): {quality_issues['machine_errors']}")
    print(f"  - Out-of-range values: {quality_issues['out_of_range']}")
    print(f"  - Missing values: {quality_issues['missing_values']}")
    print(f"✓ All issues cleaned and standardized")

    return df_clean, quality_issues


# ==========================================
# STEP 3: MODEL (Anomaly Detection)
# ==========================================
def detect_anomalies(df, contamination=0.1):
    """
    Uses Isolation Forest to detect data that 'looks wrong'
    compared to the rest of the batch.

    Isolation Forest is ideal for LIMS because:
    - It doesn't need labeled training data
    - It catches multivariate anomalies (multiple params off together)
    - It's fast enough for real-time processing

    Args:
        df: Cleaned DataFrame
        contamination: Expected proportion of anomalies (0.1 = 10%)

    Returns:
        DataFrame with anomaly scores and alerts
    """
    print("\n" + "="*60)
    print(">>> STEP 3: RUNNING ANOMALY DETECTION MODEL")
    print("="*60)

    # Select features for the model
    feature_cols = ['ph_level', 'storage_temp_c', 'dissolved_oxygen_ppm', 'turbidity_ntu']
    features = df[feature_cols]

    print(f"✓ Model features: {', '.join(feature_cols)}")
    print(f"✓ Expected anomaly rate: {contamination*100:.1f}%")

    # Initialize Isolation Forest
    # n_estimators=100: Use 100 decision trees (more = more accurate but slower)
    # contamination=0.1: We expect roughly 10% of data might be anomalies
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )

    # Fit and Predict
    # -1 = Anomaly, 1 = Normal
    df['anomaly_score'] = iso_forest.fit_predict(features)

    # Get anomaly scores (lower = more anomalous)
    df['anomaly_score_value'] = iso_forest.score_samples(features)

    # Create user-friendly alert levels
    def classify_alert(row):
        if row['anomaly_score'] == -1:
            # Critical if multiple extreme values
            if (abs(row['storage_temp_c'] - 4.0) > 10 or
                abs(row['ph_level'] - 7.2) > 1.0):
                return 'CRITICAL_FAILURE'
            else:
                return 'WARNING'
        return 'OK'

    df['QA_ALERT'] = df.apply(classify_alert, axis=1)

    # Summary
    alert_counts = df['QA_ALERT'].value_counts()
    print(f"\n✓ Anomaly detection complete:")
    for alert_type, count in alert_counts.items():
        print(f"  - {alert_type}: {count} samples ({count/len(df)*100:.1f}%)")

    # Detailed anomaly report
    critical = df[df['QA_ALERT'] == 'CRITICAL_FAILURE']
    if not critical.empty:
        print(f"\n⚠️  ALERT: {len(critical)} CRITICAL FAILURES detected!")
        print("\nCritical samples:")
        print(critical[['sample_id', 'facility_id', 'storage_temp_c', 'ph_level', 'QA_ALERT']])

        # This is where you would send email/Slack alerts in production
        # send_alert_email(critical)

    return df


# ==========================================
# STEP 4: DASHBOARD UPDATE (The Gold Layer)
# ==========================================
def update_dashboard_source(df):
    """
    Writes the tagged data to files that simulate the Production Database.

    In production, this would be:
        engine = create_engine('postgresql://user:password@dashboard_db')
        df.to_sql('lims_daily_monitoring', engine, if_exists='replace', index=False)

    Power BI Dashboard would be set to 'Direct Query' mode:
    - As soon as this script finishes, the dashboard refreshes automatically
    - No manual refresh needed
    - Red 'CRITICAL_FAILURE' flags appear instantly

    Args:
        df: Final DataFrame with anomaly scores
    """
    print("\n" + "="*60)
    print(">>> STEP 4: UPDATING DASHBOARD DATA SOURCE")
    print("="*60)

    # Prepare dashboard dataset (only columns needed for viz)
    dashboard_df = df[[
        'sample_id', 'facility_id', 'batch_id', 'timestamp',
        'ph_level', 'storage_temp_c', 'dissolved_oxygen_ppm', 'turbidity_ntu',
        'QA_ALERT', 'anomaly_score_value'
    ]].copy()

    # Write to CSV (simulates SQL database for Power BI)
    dashboard_df.to_csv(DASHBOARD_CSV, index=False)
    print(f"✓ Dashboard data written to: {DASHBOARD_CSV}")

    # Write critical alerts to separate file (for email alerts)
    critical_df = df[df['QA_ALERT'] == 'CRITICAL_FAILURE'].copy()
    if not critical_df.empty:
        critical_df.to_csv(ALERTS_CSV, index=False)
        print(f"✓ Critical alerts written to: {ALERTS_CSV}")

    # Generate monitoring report (JSON for API/logging)
    report = {
        'run_timestamp': datetime.now().isoformat(),
        'total_samples': len(df),
        'alert_summary': df['QA_ALERT'].value_counts().to_dict(),
        'critical_samples': critical_df['sample_id'].tolist() if not critical_df.empty else [],
        'facilities_affected': critical_df['facility_id'].unique().tolist() if not critical_df.empty else [],
        'data_quality_score': (df['QA_ALERT'] == 'OK').sum() / len(df) * 100
    }

    with open(REPORT_JSON, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✓ Monitoring report written to: {REPORT_JSON}")

    print("\n--- DASHBOARD PREVIEW (Top 10 Rows) ---")
    print(dashboard_df[['sample_id', 'facility_id', 'storage_temp_c', 'ph_level', 'QA_ALERT']].head(10))

    return dashboard_df


# ==========================================
# STEP 5: VISUALIZATION (Optional)
# ==========================================
def create_visualizations(df):
    """
    Generate visualizations for the monitoring report.

    These would appear in:
    - Power BI dashboard
    - Email alerts (as attachments)
    - Slack notifications

    Args:
        df: Final DataFrame with anomaly scores
    """
    if not HAS_MATPLOTLIB:
        print("\n⚠️  Skipping visualizations (matplotlib not available)")
        return

    print("\n" + "="*60)
    print(">>> STEP 5: GENERATING VISUALIZATIONS")
    print("="*60)

    # Set style
    sns.set_style("whitegrid")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('LIMS Quality Monitoring Dashboard', fontsize=16, fontweight='bold')

    # Plot 1: Temperature distribution with anomalies
    ax1 = axes[0, 0]
    normal = df[df['QA_ALERT'] == 'OK']
    critical = df[df['QA_ALERT'] == 'CRITICAL_FAILURE']
    warning = df[df['QA_ALERT'] == 'WARNING']

    ax1.scatter(normal.index, normal['storage_temp_c'], alpha=0.6, label='OK', color='green', s=50)
    ax1.scatter(warning.index, warning['storage_temp_c'], alpha=0.8, label='Warning', color='orange', s=100)
    ax1.scatter(critical.index, critical['storage_temp_c'], alpha=0.9, label='CRITICAL', color='red', s=150, marker='X')
    ax1.axhline(y=4.0, color='blue', linestyle='--', alpha=0.5, label='Target (4°C)')
    ax1.axhline(y=8.0, color='orange', linestyle='--', alpha=0.5, label='Upper Limit')
    ax1.axhline(y=2.0, color='orange', linestyle='--', alpha=0.5, label='Lower Limit')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Storage Temperature (°C)')
    ax1.set_title('Storage Temperature Monitoring')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: pH levels over time
    ax2 = axes[0, 1]
    ax2.scatter(normal.index, normal['ph_level'], alpha=0.6, label='OK', color='green', s=50)
    ax2.scatter(warning.index, warning['ph_level'], alpha=0.8, label='Warning', color='orange', s=100)
    ax2.scatter(critical.index, critical['ph_level'], alpha=0.9, label='CRITICAL', color='red', s=150, marker='X')
    ax2.axhline(y=7.2, color='blue', linestyle='--', alpha=0.5, label='Target (pH 7.2)')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('pH Level')
    ax2.set_title('pH Level Monitoring')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Alert distribution by facility
    ax3 = axes[1, 0]
    alert_by_facility = df.groupby(['facility_id', 'QA_ALERT']).size().unstack(fill_value=0)
    alert_by_facility.plot(kind='bar', stacked=True, ax=ax3, color=['green', 'red', 'orange'])
    ax3.set_xlabel('Facility')
    ax3.set_ylabel('Sample Count')
    ax3.set_title('Alert Distribution by Facility')
    ax3.legend(title='Alert Level')
    ax3.tick_params(axis='x', rotation=45)

    # Plot 4: Anomaly score distribution
    ax4 = axes[1, 1]
    ax4.hist(df['anomaly_score_value'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax4.axvline(x=df[df['QA_ALERT'] != 'OK']['anomaly_score_value'].max(),
                color='red', linestyle='--', linewidth=2, label='Anomaly Threshold')
    ax4.set_xlabel('Anomaly Score (lower = more anomalous)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Anomaly Score Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    viz_path = os.path.join(OUTPUT_DIR, 'monitoring_dashboard.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {viz_path}")

    # Show plot (comment out in production/automated runs)
    plt.show()


# ==========================================
# EXECUTION PIPELINE
# ==========================================
def run_monitoring_pipeline():
    """
    Main execution function.

    In production (Azure Data Factory):
    - This runs every hour via scheduled trigger
    - Takes ~30 seconds to process 1000 samples
    - Writes to SQL database that Power BI queries
    - Sends email/Slack alerts for critical failures
    """
    print("\n" + "="*70)
    print(" LIMS QUALITY MONITORING PIPELINE - AUTOMATED RUN ".center(70, "="))
    print("="*70)
    print(f"Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = datetime.now()

    # Step 1: Extract
    raw_df = extract_lims_data(num_samples=50)

    # Step 2: Clean
    clean_df, quality_issues = clean_data(raw_df)

    # Step 3: Detect anomalies
    final_df = detect_anomalies(clean_df, contamination=0.1)

    # Step 4: Update dashboard
    dashboard_df = update_dashboard_source(final_df)

    # Step 5: Generate visualizations
    create_visualizations(final_df)

    # Summary
    duration = (datetime.now() - start_time).total_seconds()

    print("\n" + "="*70)
    print(" PIPELINE EXECUTION COMPLETE ".center(70, "="))
    print("="*70)
    print(f"Total samples processed: {len(final_df)}")
    print(f"Processing time: {duration:.2f} seconds")
    print(f"Samples per second: {len(final_df)/duration:.1f}")
    print(f"\nOutput files:")
    print(f"  - Dashboard data: {DASHBOARD_CSV}")
    print(f"  - Critical alerts: {ALERTS_CSV}")
    print(f"  - Monitoring report: {REPORT_JSON}")
    if HAS_MATPLOTLIB:
        print(f"  - Visualization: {os.path.join(OUTPUT_DIR, 'monitoring_dashboard.png')}")
    print("\n💡 In production, Power BI dashboard would refresh automatically now!")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_monitoring_pipeline()
