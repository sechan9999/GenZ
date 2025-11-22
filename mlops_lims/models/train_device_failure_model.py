"""
MLflow Model Training: Device Failure Prediction

This module trains a model to predict device failure within the next 7 days
using MLflow for experiment tracking and model registry.

Model: Random Forest Classifier
Target: Device will fail within 7 days (binary classification)
Features: Gold layer device_failure_features

Author: Data Science Team
Date: 2025-11-22
"""

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeviceFailureModelTrainer:
    """
    Trains and evaluates device failure prediction models using MLflow.
    """

    def __init__(
        self,
        spark: SparkSession,
        gold_path: str,
        mlflow_experiment_name: str = "/LIMS/device_failure_prediction",
    ):
        """
        Initialize model trainer.

        Args:
            spark: Active SparkSession
            gold_path: Path to Gold layer Delta tables
            mlflow_experiment_name: MLflow experiment name
        """
        self.spark = spark
        self.gold_path = gold_path
        self.mlflow_experiment_name = mlflow_experiment_name

        # Set MLflow experiment
        mlflow.set_experiment(mlflow_experiment_name)
        logger.info(f"MLflow experiment: {mlflow_experiment_name}")

    def prepare_training_data(
        self, lookback_days: int = 90
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data with labels.

        Labels are created by looking ahead 7 days:
        - If device had errors/failures in next 7 days, label = 1
        - Otherwise, label = 0

        Args:
            lookback_days: Number of days of historical data to use

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info(f"Preparing training data (lookback: {lookback_days} days)")

        # Read Gold layer features
        df_features = self.spark.read.format("delta").load(
            f"{self.gold_path}/device_failure_features"
        )

        # Filter to recent data
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        df_features = df_features.filter(F.col("log_date") >= cutoff_date)

        # Create labels: device will fail in next 7 days
        # For this example, we define "failure" as:
        # - error_rate > 0.1 OR qc_pass_rate < 0.8 in next 7 days
        df_future = df_features.select(
            F.col("device_id"),
            F.col("log_date").alias("future_date"),
            F.col("error_rate_7d").alias("future_error_rate"),
            F.col("qc_pass_rate_7d").alias("future_qc_pass_rate"),
        )

        df_labeled = df_features.join(
            df_future,
            (df_features.device_id == df_future.device_id)
            & (
                df_future.future_date
                == F.date_add(df_features.log_date, 7)  # 7 days in future
            ),
            how="left",
        ).withColumn(
            "will_fail",
            (
                (F.col("future_error_rate") > 0.1)
                | (F.col("future_qc_pass_rate") < 0.8)
            ).cast("int"),
        )

        # Select features for training
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

        # Convert boolean columns to int
        df_model = df_labeled.withColumn(
            "calibration_overdue", F.col("calibration_overdue").cast("int")
        ).withColumn("maintenance_overdue", F.col("maintenance_overdue").cast("int"))

        # Convert to Pandas (for sklearn)
        pdf = df_model.select(feature_cols + ["will_fail"]).toPandas()

        # Remove rows with null labels (no future data)
        pdf = pdf.dropna(subset=["will_fail"])

        X = pdf[feature_cols]
        y = pdf["will_fail"]

        logger.info(f"Training data prepared: {len(X)} samples, {len(feature_cols)} features")
        logger.info(f"Failure rate: {y.mean():.2%}")

        return X, y

    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        hyperparameters: Dict = None,
    ) -> Tuple[RandomForestClassifier, Dict]:
        """
        Train Random Forest model with MLflow tracking.

        Args:
            X: Feature DataFrame
            y: Target Series
            hyperparameters: Model hyperparameters (default: balanced RF)

        Returns:
            Tuple of (trained model, metrics dict)
        """
        # Default hyperparameters
        if hyperparameters is None:
            hyperparameters = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "class_weight": "balanced",  # Handle imbalanced data
                "random_state": 42,
            }

        logger.info(f"Training model with hyperparameters: {hyperparameters}")

        # Start MLflow run
        with mlflow.start_run() as run:
            # Log hyperparameters
            mlflow.log_params(hyperparameters)

            # Log data version (Delta Lake version)
            delta_version = self._get_delta_version(
                f"{self.gold_path}/device_failure_features"
            )
            mlflow.log_param("data_version", delta_version)
            mlflow.log_param("training_date", datetime.now().isoformat())

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))

            # Train model
            logger.info("Training Random Forest model...")
            model = RandomForestClassifier(**hyperparameters)
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_pred_proba),
            }

            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            logger.info(f"Model metrics: {metrics}")

            # Feature importance
            feature_importance = pd.DataFrame(
                {
                    "feature": X.columns,
                    "importance": model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

            logger.info("Top 5 features:")
            logger.info(feature_importance.head())

            # Log artifacts
            feature_importance.to_csv("/tmp/feature_importance.csv", index=False)
            mlflow.log_artifact("/tmp/feature_importance.csv")

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(
                cm,
                index=["Actual No Failure", "Actual Failure"],
                columns=["Predicted No Failure", "Predicted Failure"],
            )
            cm_df.to_csv("/tmp/confusion_matrix.csv")
            mlflow.log_artifact("/tmp/confusion_matrix.csv")

            # Log model
            signature = infer_signature(X_train, y_pred)
            mlflow.sklearn.log_model(
                model,
                "model",
                signature=signature,
                registered_model_name="device_failure_predictor",
            )

            logger.info(f"Model logged to MLflow. Run ID: {run.info.run_id}")

            return model, metrics

    def evaluate_model(
        self, model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict:
        """
        Evaluate model performance on test set.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of metrics
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }

        return metrics

    def _get_delta_version(self, delta_path: str) -> int:
        """
        Get current version of Delta table.

        Args:
            delta_path: Path to Delta table

        Returns:
            Current version number
        """
        try:
            history = (
                self.spark.sql(f"DESCRIBE HISTORY delta.`{delta_path}`")
                .select("version")
                .first()
            )
            return int(history["version"])
        except Exception as e:
            logger.warning(f"Could not get Delta version: {e}")
            return -1

    def hyperparameter_tuning(
        self, X: pd.DataFrame, y: pd.Series, param_grid: Dict
    ) -> Tuple[Dict, Dict]:
        """
        Perform hyperparameter tuning with MLflow tracking.

        Args:
            X: Feature DataFrame
            y: Target Series
            param_grid: Grid of hyperparameters to try

        Returns:
            Tuple of (best_params, best_metrics)

        Example:
            >>> param_grid = {
            ...     'n_estimators': [50, 100, 200],
            ...     'max_depth': [5, 10, 15],
            ...     'min_samples_split': [5, 10, 20]
            ... }
            >>> best_params, best_metrics = trainer.hyperparameter_tuning(X, y, param_grid)
        """
        logger.info("Starting hyperparameter tuning...")

        best_f1 = 0
        best_params = None
        best_metrics = None

        # Grid search (simplified - use GridSearchCV for production)
        for n_est in param_grid.get("n_estimators", [100]):
            for max_d in param_grid.get("max_depth", [10]):
                for min_split in param_grid.get("min_samples_split", [10]):
                    params = {
                        "n_estimators": n_est,
                        "max_depth": max_d,
                        "min_samples_split": min_split,
                        "min_samples_leaf": 5,
                        "class_weight": "balanced",
                        "random_state": 42,
                    }

                    _, metrics = self.train_model(X, y, params)

                    if metrics["f1"] > best_f1:
                        best_f1 = metrics["f1"]
                        best_params = params
                        best_metrics = metrics

        logger.info(f"Best hyperparameters: {best_params}")
        logger.info(f"Best metrics: {best_metrics}")

        return best_params, best_metrics


# Example usage in Databricks notebook
if __name__ == "__main__":
    # Initialize Spark session
    spark = (
        SparkSession.builder.appName("Device Failure Model Training")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .getOrCreate()
    )

    # Configure MLflow (Databricks automatically configures this)
    # For local development:
    # mlflow.set_tracking_uri("databricks")

    # Configure paths
    GOLD_PATH = "/mnt/delta/lims/gold"

    # Initialize trainer
    trainer = DeviceFailureModelTrainer(
        spark, GOLD_PATH, mlflow_experiment_name="/LIMS/device_failure_prediction"
    )

    # Prepare training data
    X, y = trainer.prepare_training_data(lookback_days=90)

    # Train model
    model, metrics = trainer.train_model(X, y)

    print("\n=== Model Training Complete ===")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1']:.3f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")

    print("\nModel registered to MLflow Model Registry as 'device_failure_predictor'")
    print("To deploy: Transition model to 'Production' stage in MLflow UI")
