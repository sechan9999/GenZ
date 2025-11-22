"""
Batch Model Scoring Job

Daily batch job to score all devices and generate reports for lab managers.
Runs as Databricks Job or scheduled workflow.

Outputs:
- Risk scores for all devices (Delta Lake table)
- Excel report for lab managers
- High-risk device alerts (email/Slack)

Author: MLOps Team
Date: 2025-11-22
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType
import mlflow
import mlflow.pyfunc
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchScoringJob:
    """
    Batch scoring job for device failure predictions.

    Workflow:
    1. Load production model from MLflow
    2. Load latest features from Gold layer
    3. Score all devices
    4. Write scores to Delta Lake
    5. Generate Excel report
    6. Send alerts for high-risk devices
    """

    def __init__(
        self,
        spark: SparkSession,
        gold_path: str,
        output_path: str,
        model_name: str = "device_failure_predictor",
        model_stage: str = "Production",
    ):
        """
        Initialize batch scoring job.

        Args:
            spark: Active SparkSession
            gold_path: Path to Gold layer features
            output_path: Path to write scoring results
            model_name: MLflow model name
            model_stage: Model stage (Production, Staging)
        """
        self.spark = spark
        self.gold_path = gold_path
        self.output_path = output_path
        self.model_name = model_name
        self.model_stage = model_stage

        # Load model
        self.model = self._load_model()

    def _load_model(self) -> mlflow.pyfunc.PyFuncModel:
        """Load model from MLflow Model Registry."""
        logger.info(f"Loading model: {self.model_name} ({self.model_stage})")

        model_uri = f"models:/{self.model_name}/{self.model_stage}"
        model = mlflow.pyfunc.load_model(model_uri)

        logger.info("Model loaded successfully")
        return model

    def load_features(self, scoring_date: str = None) -> DataFrame:
        """
        Load latest features for all devices.

        Args:
            scoring_date: Date to score (YYYY-MM-DD), defaults to yesterday

        Returns:
            DataFrame with device features
        """
        if scoring_date is None:
            scoring_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        logger.info(f"Loading features for date: {scoring_date}")

        df_features = (
            self.spark.read.format("delta")
            .load(f"{self.gold_path}/device_failure_features")
            .filter(F.col("log_date") == scoring_date)
        )

        record_count = df_features.count()
        logger.info(f"Loaded {record_count} device records")

        return df_features

    def score_devices(self, df_features: DataFrame) -> DataFrame:
        """
        Score all devices using the production model.

        Args:
            df_features: DataFrame with device features

        Returns:
            DataFrame with predictions
        """
        logger.info("Scoring devices...")

        # Convert PySpark DataFrame to Pandas for model prediction
        pdf_features = df_features.select(
            "device_id",
            "log_date",
            F.col("calibration_overdue").cast("int").alias("calibration_overdue"),
            F.col("maintenance_overdue").cast("int").alias("maintenance_overdue"),
            "calibration_overdue_days",
            "maintenance_overdue_days",
            "error_rate_7d",
            "warning_rate_7d",
            "qc_pass_rate_7d",
            "result_volume_trend",
            "avg_turnaround_time_7d",
            "result_volume",
        ).toPandas()

        # Prepare feature columns
        feature_cols = [
            "calibration_overdue",
            "maintenance_overdue",
            "calibration_overdue_days",
            "maintenance_overdue_days",
            "error_rate_7d",
            "warning_rate_7d",
            "qc_pass_rate_7d",
            "result_volume_trend",
            "avg_turnaround_time_7d",
            "result_volume",
        ]

        X = pdf_features[feature_cols]

        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model._model_impl.predict_proba(X)[:, 1]

        # Add predictions to DataFrame
        pdf_features["failure_prediction"] = predictions
        pdf_features["failure_probability"] = probabilities
        pdf_features["risk_score"] = (probabilities * 100).astype(int)

        # Determine risk level
        pdf_features["risk_level"] = pdf_features["risk_score"].apply(
            lambda x: "CRITICAL"
            if x >= 80
            else "HIGH" if x >= 60 else "MEDIUM" if x >= 30 else "LOW"
        )

        # Add recommendations
        pdf_features["recommendation"] = pdf_features["risk_level"].apply(
            lambda x: "IMMEDIATE ACTION REQUIRED: Emergency maintenance"
            if x == "CRITICAL"
            else "Schedule urgent maintenance within 24 hours"
            if x == "HIGH"
            else "Schedule maintenance within 7 days"
            if x == "MEDIUM"
            else "Continue normal operations"
        )

        # Add scoring metadata
        pdf_features["scoring_timestamp"] = datetime.now().isoformat()
        pdf_features["model_name"] = self.model_name
        pdf_features["model_stage"] = self.model_stage

        # Convert back to PySpark DataFrame
        df_scored = self.spark.createDataFrame(pdf_features)

        logger.info(f"Scored {len(pdf_features)} devices")

        # Log risk distribution
        risk_distribution = pdf_features["risk_level"].value_counts().to_dict()
        logger.info(f"Risk distribution: {risk_distribution}")

        return df_scored

    def write_scores(self, df_scored: DataFrame) -> None:
        """
        Write scoring results to Delta Lake.

        Args:
            df_scored: DataFrame with predictions
        """
        logger.info("Writing scores to Delta Lake...")

        output_path = f"{self.output_path}/device_risk_scores"

        df_scored.write.format("delta").mode("append").partitionBy("log_date").save(
            output_path
        )

        logger.info(f"Scores written to {output_path}")

    def generate_excel_report(
        self, df_scored: DataFrame, report_path: str
    ) -> str:
        """
        Generate Excel report for lab managers.

        Args:
            df_scored: DataFrame with predictions
            report_path: Path to save Excel file

        Returns:
            Path to generated Excel file
        """
        logger.info("Generating Excel report...")

        # Convert to Pandas
        pdf = df_scored.toPandas()

        # Create Excel file with multiple sheets
        excel_file = f"{report_path}/device_risk_report_{datetime.now().strftime('%Y%m%d')}.xlsx"

        with pd.ExcelWriter(excel_file, engine="xlsxwriter") as writer:
            # Sheet 1: High-risk devices
            high_risk = pdf[pdf["risk_level"].isin(["CRITICAL", "HIGH"])].sort_values(
                "risk_score", ascending=False
            )
            high_risk[
                [
                    "device_id",
                    "risk_score",
                    "risk_level",
                    "failure_probability",
                    "recommendation",
                    "calibration_overdue_days",
                    "maintenance_overdue_days",
                    "error_rate_7d",
                    "qc_pass_rate_7d",
                ]
            ].to_excel(writer, sheet_name="High Risk Devices", index=False)

            # Sheet 2: All devices
            pdf[
                [
                    "device_id",
                    "risk_score",
                    "risk_level",
                    "failure_probability",
                    "calibration_overdue",
                    "maintenance_overdue",
                    "error_rate_7d",
                    "qc_pass_rate_7d",
                ]
            ].to_excel(writer, sheet_name="All Devices", index=False)

            # Sheet 3: Risk summary
            summary = (
                pdf.groupby("risk_level")
                .agg(
                    {
                        "device_id": "count",
                        "risk_score": "mean",
                        "failure_probability": "mean",
                    }
                )
                .rename(
                    columns={
                        "device_id": "device_count",
                        "risk_score": "avg_risk_score",
                        "failure_probability": "avg_failure_probability",
                    }
                )
            )
            summary.to_excel(writer, sheet_name="Summary")

        logger.info(f"Excel report generated: {excel_file}")
        return excel_file

    def send_alerts(self, df_scored: DataFrame) -> None:
        """
        Send email alerts for high-risk devices.

        Args:
            df_scored: DataFrame with predictions
        """
        # Get high-risk devices
        high_risk = df_scored.filter(
            F.col("risk_level").isin(["CRITICAL", "HIGH"])
        ).toPandas()

        if len(high_risk) == 0:
            logger.info("No high-risk devices found. No alerts sent.")
            return

        logger.info(f"Sending alerts for {len(high_risk)} high-risk devices")

        # Prepare email content
        critical_count = len(high_risk[high_risk["risk_level"] == "CRITICAL"])
        high_count = len(high_risk[high_risk["risk_level"] == "HIGH"])

        subject = f"⚠️ LIMS Device Risk Alert: {critical_count} Critical, {high_count} High Risk Devices"

        body = f"""
        <html>
        <body>
        <h2>LIMS Device Failure Risk Alert</h2>
        <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h3>Summary</h3>
        <ul>
            <li><strong>Critical Risk:</strong> {critical_count} devices</li>
            <li><strong>High Risk:</strong> {high_count} devices</li>
        </ul>

        <h3>Critical Risk Devices (Immediate Action Required)</h3>
        <table border="1" style="border-collapse: collapse;">
            <tr>
                <th>Device ID</th>
                <th>Risk Score</th>
                <th>Failure Probability</th>
                <th>Recommendation</th>
            </tr>
        """

        for _, row in high_risk[high_risk["risk_level"] == "CRITICAL"].iterrows():
            body += f"""
            <tr>
                <td>{row['device_id']}</td>
                <td>{row['risk_score']}</td>
                <td>{row['failure_probability']:.2%}</td>
                <td>{row['recommendation']}</td>
            </tr>
            """

        body += """
        </table>

        <p><strong>Action Required:</strong> Please review the attached Excel report and schedule maintenance for high-risk devices.</p>

        <p>This is an automated alert from the LIMS MLOps system.</p>
        </body>
        </html>
        """

        # Send email (configuration from environment variables)
        self._send_email(
            subject=subject,
            body=body,
            recipients=os.getenv("ALERT_EMAIL_RECIPIENTS", "lab-managers@example.com").split(","),
        )

    def _send_email(self, subject: str, body: str, recipients: List[str]) -> None:
        """
        Send email notification.

        Args:
            subject: Email subject
            body: Email body (HTML)
            recipients: List of recipient email addresses
        """
        try:
            # Email configuration (from environment variables)
            smtp_server = os.getenv("SMTP_SERVER", "smtp.office365.com")
            smtp_port = int(os.getenv("SMTP_PORT", "587"))
            smtp_user = os.getenv("SMTP_USER")
            smtp_password = os.getenv("SMTP_PASSWORD")

            if not smtp_user or not smtp_password:
                logger.warning("Email credentials not configured. Skipping email.")
                return

            # Create message
            msg = MIMEMultipart()
            msg["From"] = smtp_user
            msg["To"] = ", ".join(recipients)
            msg["Subject"] = subject

            msg.attach(MIMEText(body, "html"))

            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)

            logger.info(f"Email sent to {len(recipients)} recipients")

        except Exception as e:
            logger.error(f"Failed to send email: {e}")

    def run(self, scoring_date: str = None, send_alerts: bool = True) -> Dict:
        """
        Run complete batch scoring job.

        Args:
            scoring_date: Date to score (YYYY-MM-DD)
            send_alerts: Whether to send email alerts

        Returns:
            Dictionary with job statistics
        """
        logger.info("=" * 60)
        logger.info("Starting Batch Scoring Job")
        logger.info("=" * 60)

        start_time = datetime.now()

        # Load features
        df_features = self.load_features(scoring_date)
        device_count = df_features.count()

        # Score devices
        df_scored = self.score_devices(df_features)

        # Write scores
        self.write_scores(df_scored)

        # Generate Excel report
        report_path = "/dbfs/mnt/reports/lims"  # Databricks file system
        excel_file = self.generate_excel_report(df_scored, report_path)

        # Send alerts
        if send_alerts:
            self.send_alerts(df_scored)

        # Calculate statistics
        risk_distribution = (
            df_scored.groupBy("risk_level").count().toPandas().set_index("risk_level")["count"].to_dict()
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        stats = {
            "scoring_date": scoring_date or (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
            "devices_scored": device_count,
            "risk_distribution": risk_distribution,
            "excel_report": excel_file,
            "duration_seconds": duration,
            "status": "success",
        }

        logger.info("=" * 60)
        logger.info("Batch Scoring Job Complete")
        logger.info(f"Devices scored: {device_count}")
        logger.info(f"Risk distribution: {risk_distribution}")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info("=" * 60)

        return stats


# Example usage in Databricks notebook
if __name__ == "__main__":
    # Initialize Spark session
    spark = (
        SparkSession.builder.appName("LIMS Batch Scoring")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .getOrCreate()
    )

    # Configure paths
    GOLD_PATH = "/mnt/delta/lims/gold"
    OUTPUT_PATH = "/mnt/delta/lims/predictions"

    # Initialize batch scoring
    batch_job = BatchScoringJob(
        spark,
        gold_path=GOLD_PATH,
        output_path=OUTPUT_PATH,
        model_name="device_failure_predictor",
        model_stage="Production",
    )

    # Run job
    stats = batch_job.run(send_alerts=True)

    print("\n=== Batch Scoring Complete ===")
    print(f"Status: {stats['status']}")
    print(f"Devices scored: {stats['devices_scored']}")
    print(f"Risk distribution: {stats['risk_distribution']}")
    print(f"Excel report: {stats['excel_report']}")
    print(f"Duration: {stats['duration_seconds']:.2f} seconds")
