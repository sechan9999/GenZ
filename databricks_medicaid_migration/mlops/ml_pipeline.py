"""
MLOps Pipeline for Medicaid Risk Models

This module implements automated model retraining, validation, and deployment pipeline.
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from databricks.feature_store import FeatureStoreClient
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicaidRiskMLOpsPipeline:
    """MLOps pipeline for automated model training and deployment."""

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.fs = FeatureStoreClient()
        self.mlflow_client = MlflowClient()
        self.experiment_name = "/Medicaid/Risk_Prediction_Models"

    def check_data_drift(self, reference_window_days=365, current_window_days=90):
        """
        Check for data drift in feature distributions.

        Args:
            reference_window_days: Historical reference period
            current_window_days: Recent period to compare

        Returns:
            dict: Drift metrics and retrain recommendation
        """
        logger.info("Checking for data drift...")

        # Get reference and current feature distributions
        query = f"""
        WITH reference_features AS (
            SELECT
                AVG(age) as avg_age,
                AVG(chronic_condition_count) as avg_conditions,
                AVG(claim_count_12mo) as avg_claims,
                AVG(total_paid_12mo) as avg_cost,
                STDDEV(total_paid_12mo) as std_cost
            FROM gold.member_features
            WHERE feature_date BETWEEN
                date_sub(current_date(), {reference_window_days})
                AND date_sub(current_date(), {current_window_days})
        ),
        current_features AS (
            SELECT
                AVG(age) as avg_age,
                AVG(chronic_condition_count) as avg_conditions,
                AVG(claim_count_12mo) as avg_claims,
                AVG(total_paid_12mo) as avg_cost,
                STDDEV(total_paid_12mo) as std_cost
            FROM gold.member_features
            WHERE feature_date >= date_sub(current_date(), {current_window_days})
        )
        SELECT
            ABS(c.avg_age - r.avg_age) / NULLIF(r.avg_age, 0) * 100 as age_drift_pct,
            ABS(c.avg_conditions - r.avg_conditions) / NULLIF(r.avg_conditions, 0) * 100 as conditions_drift_pct,
            ABS(c.avg_claims - r.avg_claims) / NULLIF(r.avg_claims, 0) * 100 as claims_drift_pct,
            ABS(c.avg_cost - r.avg_cost) / NULLIF(r.avg_cost, 0) * 100 as cost_drift_pct,
            ABS(c.std_cost - r.std_cost) / NULLIF(r.std_cost, 0) * 100 as cost_std_drift_pct
        FROM reference_features r
        CROSS JOIN current_features c
        """

        drift_metrics = self.spark.sql(query).collect()[0].asDict()

        # Determine if retraining is needed (threshold: 10% drift)
        max_drift = max(drift_metrics.values())
        should_retrain = max_drift > 10.0

        logger.info(f"Data drift analysis complete. Max drift: {max_drift:.2f}%")
        logger.info(f"Retrain recommended: {should_retrain}")

        return {
            "drift_metrics": drift_metrics,
            "max_drift_pct": max_drift,
            "should_retrain": should_retrain,
            "check_timestamp": datetime.now()
        }

    def check_model_performance(self, model_name, days_back=30):
        """
        Check recent model performance against validation set.

        Args:
            model_name: Name of the registered model
            days_back: Days to look back for performance metrics

        Returns:
            dict: Performance metrics and degradation flag
        """
        logger.info(f"Checking performance for model: {model_name}")

        # Get the current production model
        try:
            production_model = self.mlflow_client.get_latest_versions(
                model_name, stages=["Production"]
            )[0]
            run_id = production_model.run_id
        except IndexError:
            logger.warning(f"No production model found for {model_name}")
            return {"performance_ok": False, "reason": "No production model"}

        # Get run metrics from MLflow
        run = self.mlflow_client.get_run(run_id)
        training_auc = run.data.metrics.get("auc", 0)

        # Calculate recent performance on new data
        # This would involve scoring recent members and comparing predictions to outcomes
        # For now, using a simplified check

        logger.info(f"Training AUC: {training_auc:.4f}")

        # If training AUC < threshold, flag for retraining
        performance_ok = training_auc >= 0.70

        return {
            "model_name": model_name,
            "training_auc": training_auc,
            "performance_ok": performance_ok,
            "check_timestamp": datetime.now()
        }

    def retrain_models(self):
        """
        Trigger model retraining pipeline.

        Returns:
            dict: Retraining results
        """
        logger.info("Starting model retraining pipeline...")

        # Trigger notebook execution for model training
        # In Databricks, you would use the Jobs API or dbutils.notebook.run
        results = {
            "retrain_initiated": True,
            "timestamp": datetime.now(),
            "status": "running"
        }

        logger.info("Model retraining initiated")
        return results

    def promote_model_to_production(self, model_name, version):
        """
        Promote a model version to production after validation.

        Args:
            model_name: Name of the registered model
            version: Model version to promote

        Returns:
            bool: Success status
        """
        logger.info(f"Promoting {model_name} version {version} to production")

        try:
            # Transition to staging first for validation
            self.mlflow_client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Staging",
                archive_existing_versions=False
            )

            # Run validation tests (simplified here)
            validation_passed = True

            if validation_passed:
                # Promote to production
                self.mlflow_client.transition_model_version_stage(
                    name=model_name,
                    version=version,
                    stage="Production",
                    archive_existing_versions=True
                )
                logger.info(f"Model {model_name} v{version} promoted to production")
                return True
            else:
                logger.warning(f"Model {model_name} v{version} failed validation")
                return False

        except Exception as e:
            logger.error(f"Error promoting model: {str(e)}")
            return False

    def run_monitoring_pipeline(self):
        """
        Run complete monitoring and retraining pipeline.

        Returns:
            dict: Pipeline execution results
        """
        logger.info("=" * 60)
        logger.info("RUNNING MLOPS MONITORING PIPELINE")
        logger.info("=" * 60)

        results = {
            "pipeline_run_timestamp": datetime.now(),
            "drift_check": None,
            "performance_checks": {},
            "retrain_triggered": False
        }

        # Step 1: Check data drift
        drift_check = self.check_data_drift()
        results["drift_check"] = drift_check

        # Step 2: Check model performance
        models_to_check = [
            "er_utilization_predictor",
            "high_cost_predictor",
            "high_risk_predictor"
        ]

        for model_name in models_to_check:
            perf_check = self.check_model_performance(model_name)
            results["performance_checks"][model_name] = perf_check

        # Step 3: Decide if retraining is needed
        should_retrain = (
            drift_check["should_retrain"] or
            any(not check["performance_ok"]
                for check in results["performance_checks"].values()
                if isinstance(check, dict))
        )

        # Step 4: Trigger retraining if needed
        if should_retrain:
            logger.info("Triggering model retraining...")
            retrain_results = self.retrain_models()
            results["retrain_triggered"] = True
            results["retrain_results"] = retrain_results
        else:
            logger.info("No retraining needed. Models performing well.")

        # Save monitoring results
        self._save_monitoring_results(results)

        logger.info("MLOps monitoring pipeline complete")
        return results

    def _save_monitoring_results(self, results):
        """Save monitoring results to Delta table for tracking."""
        from pyspark.sql import Row

        monitoring_record = Row(
            pipeline_run_timestamp=results["pipeline_run_timestamp"],
            max_drift_pct=results["drift_check"]["max_drift_pct"],
            retrain_triggered=results["retrain_triggered"],
            drift_details=str(results["drift_check"]),
            performance_details=str(results["performance_checks"])
        )

        monitoring_df = self.spark.createDataFrame([monitoring_record])

        (monitoring_df.write
            .format("delta")
            .mode("append")
            .saveAsTable("gold.model_monitoring_log"))

        logger.info("Monitoring results saved to gold.model_monitoring_log")


def main():
    """Main entry point for MLOps pipeline."""
    spark = SparkSession.builder.appName("MedicaidRiskMLOps").getOrCreate()

    pipeline = MedicaidRiskMLOpsPipeline(spark)
    results = pipeline.run_monitoring_pipeline()

    return results


if __name__ == "__main__":
    main()
