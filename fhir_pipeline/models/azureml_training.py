"""
Azure ML Integration for FHIR Predictive Modeling

Trains ML models on FHIR Gold layer features using Azure Machine Learning.

Use Cases:
1. Diabetes Risk Prediction (binary classification)
2. Hospital Readmission Prediction (30-day readmission)
3. Hypertension Onset Prediction
4. Chronic Disease Progression Scoring

Integration:
- Reads features from Delta Lake Gold layer
- Trains models using Azure ML SDK
- Registers models in Azure ML Model Registry
- Deploys to Azure ML endpoints for real-time scoring

Author: MLOps Healthcare Team
Date: 2025-11-23
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import mlflow
import mlflow.spark
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# Azure ML imports
try:
    from azureml.core import Workspace, Dataset, Experiment, Environment, Run
    from azureml.core.model import Model
    from azureml.core.compute import ComputeTarget, AmlCompute
    from azureml.train.automl import AutoMLConfig
    AZUREML_AVAILABLE = True
except ImportError:
    AZUREML_AVAILABLE = False
    logger.warning("Azure ML SDK not available. Install with: pip install azureml-sdk")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FHIRAzureMLTraining:
    """
    Train and deploy ML models using Azure ML + MLflow.

    Architecture:
    Delta Lake (Gold) → Spark ML Training → MLflow Tracking → Azure ML Registry → Deployment
    """

    def __init__(
        self,
        spark: SparkSession,
        gold_path: str,
        mlflow_tracking_uri: str = "databricks",
        azure_ml_workspace: Optional[Workspace] = None,
    ):
        """
        Initialize Azure ML training pipeline.

        Args:
            spark: Active SparkSession
            gold_path: Path to Gold layer feature tables
            mlflow_tracking_uri: MLflow tracking server URI
            azure_ml_workspace: Azure ML Workspace object (optional)
        """
        self.spark = spark
        self.gold_path = gold_path

        # Configure MLflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.spark.autolog()

        self.azure_ml_workspace = azure_ml_workspace

    def train_diabetes_risk_model(
        self,
        experiment_name: str = "fhir_diabetes_risk",
        model_type: str = "random_forest",
    ) -> Dict:
        """
        Train diabetes risk prediction model.

        Target Variable:
        - diabetes_risk_flag (binary: 0 = low risk, 1 = high risk)

        Features:
        - Age, gender
        - Avg glucose (90 days)
        - Avg HbA1c (90 days)
        - BMI (if available)
        - Blood pressure
        - Active medication count

        Args:
            experiment_name: MLflow experiment name
            model_type: 'random_forest' or 'logistic_regression'

        Returns:
            Dictionary with model metrics and run info
        """
        logger.info(f"Training diabetes risk model ({model_type})...")

        # Set MLflow experiment
        mlflow.set_experiment(experiment_name)

        # Load features
        df_features = self.spark.read.format("delta").load(
            f"{self.gold_path}/chronic_disease_features"
        )

        # Filter to patients with complete data
        df_model_data = df_features.filter(
            F.col("avg_glucose_90d").isNotNull()
            & F.col("avg_hba1c_90d").isNotNull()
            & F.col("age_years").isNotNull()
        )

        # Feature columns
        feature_cols = [
            "age_years",
            "avg_glucose_90d",
            "avg_hba1c_90d",
            "avg_bp_systolic_30d",
            "avg_bp_diastolic_30d",
            "avg_total_chol_90d",
            "avg_ldl_90d",
            "avg_hdl_90d",
            "active_medication_count",
            "encounter_count_6m",
        ]

        # Handle nulls
        df_model_data = df_model_data.fillna(0, subset=feature_cols)

        # Encode gender (male=1, female=0)
        df_model_data = df_model_data.withColumn(
            "gender_encoded", F.when(F.col("gender") == "male", 1).otherwise(0)
        )
        feature_cols.append("gender_encoded")

        # Train/test split (80/20)
        df_train, df_test = df_model_data.randomSplit([0.8, 0.2], seed=42)

        logger.info(f"Training set: {df_train.count()} records")
        logger.info(f"Test set: {df_test.count()} records")

        # Start MLflow run
        with mlflow.start_run(run_name=f"{model_type}_diabetes_risk") as run:
            # Log parameters
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("feature_count", len(feature_cols))
            mlflow.log_param("train_count", df_train.count())
            mlflow.log_param("test_count", df_test.count())

            # Create feature vector
            assembler = VectorAssembler(
                inputCols=feature_cols, outputCol="features_raw"
            )

            # Standardize features
            scaler = StandardScaler(
                inputCol="features_raw",
                outputCol="features",
                withStd=True,
                withMean=False,
            )

            # Choose classifier
            if model_type == "random_forest":
                classifier = RandomForestClassifier(
                    labelCol="diabetes_risk_flag",
                    featuresCol="features",
                    numTrees=100,
                    maxDepth=10,
                    minInstancesPerNode=5,
                    seed=42,
                )
                mlflow.log_param("num_trees", 100)
                mlflow.log_param("max_depth", 10)
            else:  # logistic_regression
                classifier = LogisticRegression(
                    labelCol="diabetes_risk_flag",
                    featuresCol="features",
                    maxIter=100,
                    regParam=0.1,
                )
                mlflow.log_param("max_iter", 100)
                mlflow.log_param("reg_param", 0.1)

            # Create pipeline
            pipeline = Pipeline(stages=[assembler, scaler, classifier])

            # Train model
            logger.info("Training model...")
            model = pipeline.fit(df_train)

            # Evaluate on test set
            df_predictions = model.transform(df_test)

            # Binary classification metrics
            evaluator_auc = BinaryClassificationEvaluator(
                labelCol="diabetes_risk_flag",
                rawPredictionCol="rawPrediction",
                metricName="areaUnderROC",
            )

            evaluator_accuracy = MulticlassClassificationEvaluator(
                labelCol="diabetes_risk_flag",
                predictionCol="prediction",
                metricName="accuracy",
            )

            evaluator_precision = MulticlassClassificationEvaluator(
                labelCol="diabetes_risk_flag",
                predictionCol="prediction",
                metricName="weightedPrecision",
            )

            evaluator_recall = MulticlassClassificationEvaluator(
                labelCol="diabetes_risk_flag",
                predictionCol="prediction",
                metricName="weightedRecall",
            )

            # Calculate metrics
            auc = evaluator_auc.evaluate(df_predictions)
            accuracy = evaluator_accuracy.evaluate(df_predictions)
            precision = evaluator_precision.evaluate(df_predictions)
            recall = evaluator_recall.evaluate(df_predictions)
            f1_score = 2 * (precision * recall) / (precision + recall)

            # Log metrics
            mlflow.log_metric("test_auc", auc)
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_precision", precision)
            mlflow.log_metric("test_recall", recall)
            mlflow.log_metric("test_f1_score", f1_score)

            logger.info(f"Model Performance:")
            logger.info(f"  AUC: {auc:.4f}")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1 Score: {f1_score:.4f}")

            # Log model to MLflow
            mlflow.spark.log_model(
                model,
                "diabetes_risk_model",
                registered_model_name="fhir_diabetes_risk_predictor",
            )

            # Log feature importance (Random Forest only)
            if model_type == "random_forest":
                rf_model = model.stages[-1]
                feature_importance = rf_model.featureImportances.toArray()

                importance_dict = dict(zip(feature_cols, feature_importance))
                sorted_importance = sorted(
                    importance_dict.items(), key=lambda x: x[1], reverse=True
                )

                logger.info("Top 5 Important Features:")
                for feat, imp in sorted_importance[:5]:
                    logger.info(f"  {feat}: {imp:.4f}")
                    mlflow.log_metric(f"feature_importance_{feat}", imp)

            # Add governance tags
            mlflow.set_tag("use_case", "diabetes_risk_prediction")
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("training_date", datetime.now().strftime("%Y-%m-%d"))
            mlflow.set_tag("data_source", "fhir_gold_layer")
            mlflow.set_tag("hipaa_compliant", "true")

            run_id = run.info.run_id
            logger.info(f"✓ Model training complete. MLflow Run ID: {run_id}")

        # Register to Azure ML (if workspace provided)
        if self.azure_ml_workspace and AZUREML_AVAILABLE:
            self._register_to_azure_ml(
                model_name="fhir_diabetes_risk_predictor",
                mlflow_run_id=run_id,
                tags={
                    "model_type": model_type,
                    "use_case": "diabetes_risk",
                    "auc": f"{auc:.4f}",
                },
            )

        return {
            "run_id": run_id,
            "auc": auc,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

    def train_readmission_risk_model(
        self, experiment_name: str = "fhir_readmission_risk"
    ) -> Dict:
        """
        Train 30-day hospital readmission prediction model.

        Target Variable:
        - readmission_30d (binary: 0 = no readmission, 1 = readmitted)

        Features:
        - Encounter frequency (last 6 months)
        - Avg encounter duration
        - Active medication count
        - Chronic disease flags
        - Recent lab abnormalities

        Args:
            experiment_name: MLflow experiment name

        Returns:
            Dictionary with model metrics
        """
        logger.info("Training readmission risk model...")

        mlflow.set_experiment(experiment_name)

        # Load features
        df_features = self.spark.read.format("delta").load(
            f"{self.gold_path}/chronic_disease_features"
        )

        # Create synthetic readmission target (in production, join with actual readmission data)
        # High-risk patients: frequent encounters + multiple chronic conditions
        df_features = df_features.withColumn(
            "readmission_30d",
            (
                (F.col("encounter_count_6m") >= 3)
                & (
                    F.col("diabetes_risk_flag")
                    | F.col("hypertension_risk_flag")
                    | F.col("cvd_risk_flag")
                )
            ).cast("int"),
        )

        feature_cols = [
            "age_years",
            "encounter_count_6m",
            "avg_encounter_duration_hours",
            "active_medication_count",
            "avg_bp_systolic_30d",
            "avg_glucose_90d",
        ]

        df_model_data = df_features.fillna(0, subset=feature_cols)
        df_train, df_test = df_model_data.randomSplit([0.8, 0.2], seed=42)

        with mlflow.start_run(run_name="random_forest_readmission") as run:
            mlflow.log_param("model_type", "random_forest")
            mlflow.log_param("target", "readmission_30d")

            assembler = VectorAssembler(
                inputCols=feature_cols, outputCol="features_raw"
            )
            scaler = StandardScaler(
                inputCol="features_raw",
                outputCol="features",
                withStd=True,
                withMean=False,
            )
            classifier = RandomForestClassifier(
                labelCol="readmission_30d",
                featuresCol="features",
                numTrees=100,
                maxDepth=10,
                seed=42,
            )

            pipeline = Pipeline(stages=[assembler, scaler, classifier])
            model = pipeline.fit(df_train)

            df_predictions = model.transform(df_test)

            evaluator_auc = BinaryClassificationEvaluator(
                labelCol="readmission_30d",
                rawPredictionCol="rawPrediction",
                metricName="areaUnderROC",
            )

            auc = evaluator_auc.evaluate(df_predictions)
            mlflow.log_metric("test_auc", auc)

            logger.info(f"Readmission Model AUC: {auc:.4f}")

            mlflow.spark.log_model(
                model,
                "readmission_risk_model",
                registered_model_name="fhir_readmission_risk_predictor",
            )

            mlflow.set_tag("use_case", "hospital_readmission")
            mlflow.set_tag("hipaa_compliant", "true")

        logger.info("✓ Readmission risk model training complete")

        return {"auc": auc}

    def _register_to_azure_ml(
        self,
        model_name: str,
        mlflow_run_id: str,
        tags: Dict[str, str],
    ) -> None:
        """
        Register MLflow model to Azure ML Model Registry.

        Args:
            model_name: Name for Azure ML model
            mlflow_run_id: MLflow run ID
            tags: Model tags
        """
        if not AZUREML_AVAILABLE:
            logger.warning("Azure ML SDK not available. Skipping Azure ML registration.")
            return

        logger.info(f"Registering model to Azure ML: {model_name}")

        try:
            # Get MLflow model URI
            model_uri = f"runs:/{mlflow_run_id}/diabetes_risk_model"

            # Register to Azure ML
            registered_model = Model.register(
                workspace=self.azure_ml_workspace,
                model_name=model_name,
                model_path=model_uri,
                tags=tags,
                description="FHIR-based diabetes risk prediction model",
                model_framework="MLflow",
            )

            logger.info(f"✓ Model registered to Azure ML: {registered_model.name} v{registered_model.version}")

        except Exception as e:
            logger.error(f"Failed to register to Azure ML: {e}")

    def deploy_to_azure_ml_endpoint(
        self,
        model_name: str,
        endpoint_name: str,
        compute_target: str = "aci",
    ) -> None:
        """
        Deploy model to Azure ML real-time endpoint.

        Args:
            model_name: Registered model name
            endpoint_name: Deployment endpoint name
            compute_target: 'aci' (Azure Container Instances) or 'aks' (Azure Kubernetes Service)
        """
        if not AZUREML_AVAILABLE:
            logger.warning("Azure ML SDK not available. Cannot deploy endpoint.")
            return

        logger.info(f"Deploying {model_name} to Azure ML endpoint: {endpoint_name}")

        try:
            # Get latest model version
            model = Model(self.azure_ml_workspace, name=model_name)

            # Create inference configuration
            env = Environment.from_conda_specification(
                name="fhir_model_env",
                file_path="./config/conda_env.yml",  # You'll need to create this
            )

            # Deploy based on compute target
            if compute_target == "aci":
                from azureml.core.webservice import AciWebservice

                deployment_config = AciWebservice.deploy_configuration(
                    cpu_cores=1,
                    memory_gb=2,
                    auth_enabled=True,
                    tags={"model": model_name, "use_case": "diabetes_risk"},
                )
            else:  # aks
                from azureml.core.webservice import AksWebservice

                deployment_config = AksWebservice.deploy_configuration(
                    cpu_cores=2,
                    memory_gb=4,
                    autoscale_enabled=True,
                    autoscale_min_replicas=1,
                    autoscale_max_replicas=10,
                    auth_enabled=True,
                )

            # Note: Actual deployment requires inference script (score.py)
            # This is a template - full implementation needs scoring script

            logger.info(
                f"Deployment configuration created for {endpoint_name} on {compute_target}"
            )
            logger.info("⚠️  Complete deployment requires scoring script (score.py)")

        except Exception as e:
            logger.error(f"Deployment failed: {e}")


# Example usage
if __name__ == "__main__":
    # Initialize Spark
    spark = (
        SparkSession.builder.appName("FHIR Azure ML Training")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .getOrCreate()
    )

    GOLD_PATH = "/mnt/delta/fhir/gold"

    # Initialize Azure ML workspace (optional)
    azure_ml_workspace = None
    if AZUREML_AVAILABLE:
        try:
            azure_ml_workspace = Workspace.from_config("./config/azureml_config.json")
            logger.info(f"Connected to Azure ML Workspace: {azure_ml_workspace.name}")
        except Exception as e:
            logger.warning(f"Azure ML workspace not configured: {e}")

    # Initialize training pipeline
    trainer = FHIRAzureMLTraining(
        spark=spark,
        gold_path=GOLD_PATH,
        mlflow_tracking_uri="databricks",
        azure_ml_workspace=azure_ml_workspace,
    )

    # Train diabetes risk model
    results_rf = trainer.train_diabetes_risk_model(
        experiment_name="fhir_diabetes_risk", model_type="random_forest"
    )

    logger.info(f"Random Forest Results: {results_rf}")

    # Train readmission model
    results_readmission = trainer.train_readmission_risk_model()

    logger.info(f"Readmission Model Results: {results_readmission}")

    logger.info("✓ Azure ML training pipeline complete")
