"""
LIMS Quality Model Training with MLflow - Local Demo

This script demonstrates MLflow experiment tracking for a LIMS contamination
detection model. Runs locally without Databricks.

Purpose: Train a Random Forest model to predict sample contamination based on
         pH level, temperature, turbidity, and processing time.

MLflow Features Demonstrated:
- Experiment tracking
- Parameter logging
- Metric logging
- Model registration
- Artifact storage
- Model versioning
- Tags for governance

Author: MLOps Team
Date: 2025-11-22
"""

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import pandas as pd
import numpy as np
from datetime import datetime
import os

# ==========================================
# CONFIGURATION
# ==========================================
MLFLOW_TRACKING_URI = "./mlruns"  # Local MLflow tracking directory
EXPERIMENT_NAME = "LIMS_Quality_Control"
MODEL_NAME = "lims_contamination_detector"

# Set MLflow tracking URI (local directory)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print(f"✓ MLflow tracking URI: {MLFLOW_TRACKING_URI}")

# Create or get experiment
try:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
except:
    experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

mlflow.set_experiment(EXPERIMENT_NAME)
print(f"✓ MLflow experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")


# ==========================================
# DATA GENERATION
# ==========================================
def generate_lims_training_data(n_samples=1000, contamination_rate=0.15):
    """
    Generate synthetic LIMS quality control data.

    Features:
    - ph_level: pH measurement (normal: 7.0-7.4)
    - temperature: Storage temperature in °C (normal: 2-8°C)
    - turbidity: Turbidity in NTU (normal: 0-5)
    - processing_time: Sample processing time in hours (normal: 1-4)

    Target:
    - is_contaminated: Binary (1 = contaminated, 0 = clean)

    Args:
        n_samples: Number of samples to generate
        contamination_rate: Proportion of contaminated samples

    Returns:
        DataFrame with training data
    """
    np.random.seed(42)

    n_contaminated = int(n_samples * contamination_rate)
    n_clean = n_samples - n_contaminated

    # Generate clean samples (normal distributions)
    clean_samples = pd.DataFrame(
        {
            "ph_level": np.random.normal(7.2, 0.15, n_clean),
            "temperature": np.random.normal(4.0, 1.0, n_clean),
            "turbidity": np.random.gamma(2, 0.5, n_clean),
            "processing_time": np.random.normal(2.5, 0.5, n_clean),
            "is_contaminated": 0,
        }
    )

    # Generate contaminated samples (shifted distributions)
    contaminated_samples = pd.DataFrame(
        {
            "ph_level": np.random.normal(
                6.5, 0.5, n_contaminated
            ),  # Lower pH (acidic)
            "temperature": np.random.normal(
                12.0, 3.0, n_contaminated
            ),  # Higher temp
            "turbidity": np.random.gamma(
                5, 2.0, n_contaminated
            ),  # Higher turbidity
            "processing_time": np.random.normal(
                5.0, 1.5, n_contaminated
            ),  # Longer processing
            "is_contaminated": 1,
        }
    )

    # Combine and shuffle
    df = pd.concat([clean_samples, contaminated_samples], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Add sample metadata
    df["sample_id"] = [f"S{i:05d}" for i in range(len(df))]
    df["batch_id"] = [
        f"BATCH_{datetime.now().strftime('%Y%m%d')}_{i//50:03d}"
        for i in range(len(df))
    ]
    df["facility_id"] = np.random.choice(
        ["GA_LTC_01", "GA_LTC_02", "GA_LTC_03"], len(df)
    )

    return df


# ==========================================
# MODEL TRAINING WITH MLFLOW
# ==========================================
def train_lims_quality_model(
    train_df, version_note="Initial training run", hyperparameters=None
):
    """
    Trains a QC model and logs it to the MLflow registry.

    This function demonstrates best practices for MLOps:
    1. Parameter tracking (reproducibility)
    2. Metric logging (performance monitoring)
    3. Model versioning (governance)
    4. Artifact storage (deployment)
    5. Tags (metadata for tracking)

    Args:
        train_df: DataFrame with training data
        version_note: Human-readable note for this model version
        hyperparameters: Dict of model hyperparameters (optional)

    Returns:
        Tuple of (trained_model, metrics_dict)

    Interview Talking Point:
    "Using MLflow allows me to track every single experiment.
    If a model fails in production, I can trace it back to the exact code
    and LIMS data snapshot that created it."
    """

    # Default hyperparameters
    if hyperparameters is None:
        hyperparameters = {
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "class_weight": "balanced",  # Handle imbalanced data
            "random_state": 42,
        }

    print("\n" + "=" * 60)
    print(">>> STARTING MLFLOW TRAINING RUN")
    print("=" * 60)

    # 1. Start an MLflow Run
    with mlflow.start_run(run_name="LIMS_Contamination_Detector") as run:

        print(f"✓ MLflow Run ID: {run.info.run_id}")
        print(f"✓ Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 2. Log Parameters (Hyperparameters)
        print("\n--- Logging Parameters ---")
        for param_name, param_value in hyperparameters.items():
            mlflow.log_param(param_name, param_value)
            print(f"  {param_name}: {param_value}")

        # Log data source information
        mlflow.log_param("data_source", "LIMS_Synthetic_v1")
        mlflow.log_param("training_samples", len(train_df))
        mlflow.log_param(
            "contamination_rate", train_df["is_contaminated"].mean()
        )

        # 3. Prepare Training Data
        print("\n--- Preparing Training Data ---")
        feature_cols = ["ph_level", "temperature", "turbidity", "processing_time"]
        X = train_df[feature_cols]
        y = train_df["is_contaminated"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"  Train samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Features: {feature_cols}")

        # 4. Train the Model
        print("\n--- Training Model ---")
        rf = RandomForestClassifier(**hyperparameters)
        rf.fit(X_train, y_train)
        print("  ✓ Model training complete")

        # 5. Evaluate on Training Set
        print("\n--- Evaluating Model (Training Set) ---")
        train_predictions = rf.predict(X_train)
        train_acc = accuracy_score(y_train, train_predictions)
        print(f"  Training Accuracy: {train_acc:.4f}")

        # 6. Evaluate on Test Set
        print("\n--- Evaluating Model (Test Set) ---")
        test_predictions = rf.predict(X_test)
        test_proba = rf.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            "test_accuracy": accuracy_score(y_test, test_predictions),
            "test_precision": precision_score(y_test, test_predictions, zero_division=0),
            "test_recall": recall_score(y_test, test_predictions, zero_division=0),
            "test_f1": f1_score(y_test, test_predictions, zero_division=0),
        }

        # Log metrics to MLflow
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            print(f"  {metric_name}: {metric_value:.4f}")

        # 7. Feature Importance
        print("\n--- Feature Importance ---")
        feature_importance = pd.DataFrame(
            {"feature": feature_cols, "importance": rf.feature_importances_}
        ).sort_values("importance", ascending=False)

        for _, row in feature_importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        # Save feature importance as artifact
        feature_importance.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")
        os.remove("feature_importance.csv")  # Clean up

        # 8. Confusion Matrix
        print("\n--- Confusion Matrix ---")
        cm = confusion_matrix(y_test, test_predictions)
        print(f"  True Negatives:  {cm[0][0]}")
        print(f"  False Positives: {cm[0][1]}")
        print(f"  False Negatives: {cm[1][0]}")
        print(f"  True Positives:  {cm[1][1]}")

        # Save confusion matrix
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Clean", "Actual Contaminated"],
            columns=["Predicted Clean", "Predicted Contaminated"],
        )
        cm_df.to_csv("confusion_matrix.csv")
        mlflow.log_artifact("confusion_matrix.csv")
        os.remove("confusion_matrix.csv")

        # 9. Classification Report
        print("\n--- Classification Report ---")
        report = classification_report(y_test, test_predictions, target_names=["Clean", "Contaminated"])
        print(report)

        # Save classification report
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")
        os.remove("classification_report.txt")

        # 10. Register the Model
        print("\n--- Registering Model ---")
        mlflow.sklearn.log_model(
            rf,
            "lims_qc_model",
            registered_model_name=MODEL_NAME,
        )
        print(f"  ✓ Model registered as: {MODEL_NAME}")

        # 11. Set Tags for Governance
        print("\n--- Adding Metadata Tags ---")
        tags = {
            "Project": "Lab_Modernization",
            "Analyst": "Senior_DS_Lead",
            "Model_Type": "Random_Forest",
            "Use_Case": "Contamination_Detection",
            "Note": version_note,
            "Training_Date": datetime.now().strftime("%Y-%m-%d"),
        }

        for tag_key, tag_value in tags.items():
            mlflow.set_tag(tag_key, tag_value)
            print(f"  {tag_key}: {tag_value}")

        print("\n" + "=" * 60)
        print(">>> MLFLOW RUN COMPLETE")
        print("=" * 60)
        print(f"Run ID: {run.info.run_id}")
        print(f"Model URI: runs:/{run.info.run_id}/lims_qc_model")

        return rf, metrics


# ==========================================
# MODEL LOADING AND PREDICTION
# ==========================================
def load_and_predict(run_id, new_samples):
    """
    Load a trained model from MLflow and make predictions.

    This demonstrates how to use MLflow models in production.

    Args:
        run_id: MLflow run ID
        new_samples: DataFrame with new samples to predict

    Returns:
        DataFrame with predictions
    """
    print("\n" + "=" * 60)
    print(">>> LOADING MODEL FROM MLFLOW")
    print("=" * 60)

    # Load model
    model_uri = f"runs:/{run_id}/lims_qc_model"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"✓ Model loaded from: {model_uri}")

    # Make predictions
    feature_cols = ["ph_level", "temperature", "turbidity", "processing_time"]
    X_new = new_samples[feature_cols]

    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)[:, 1]

    # Add predictions to DataFrame
    result = new_samples.copy()
    result["predicted_contamination"] = predictions
    result["contamination_probability"] = probabilities
    result["risk_level"] = result["contamination_probability"].apply(
        lambda x: "HIGH" if x > 0.7 else "MEDIUM" if x > 0.3 else "LOW"
    )

    print(f"✓ Predictions made for {len(result)} samples")

    return result


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":

    print("\n" + "=" * 70)
    print(" LIMS QUALITY MODEL TRAINING WITH MLFLOW - LOCAL DEMO ".center(70, "="))
    print("=" * 70)

    # 1. Generate Training Data
    print("\n>>> STEP 1: Generating Training Data")
    print("-" * 60)
    train_df = generate_lims_training_data(n_samples=1000, contamination_rate=0.15)
    print(f"✓ Generated {len(train_df)} samples")
    print(f"✓ Contamination rate: {train_df['is_contaminated'].mean():.2%}")
    print(f"✓ Features: ph_level, temperature, turbidity, processing_time")

    print("\nSample data (first 5 rows):")
    print(
        train_df[
            [
                "sample_id",
                "ph_level",
                "temperature",
                "turbidity",
                "processing_time",
                "is_contaminated",
            ]
        ].head()
    )

    # 2. Train Model with MLflow
    print("\n>>> STEP 2: Training Model with MLflow Tracking")
    print("-" * 60)

    hyperparameters = {
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "class_weight": "balanced",
        "random_state": 42,
    }

    model, metrics = train_lims_quality_model(
        train_df,
        version_note="Initial production model - contamination detection",
        hyperparameters=hyperparameters,
    )

    # 3. Generate Test Samples
    print("\n>>> STEP 3: Testing Model on New Samples")
    print("-" * 60)

    # Create some test samples (3 clean, 2 contaminated)
    test_samples = pd.DataFrame(
        {
            "sample_id": ["TEST001", "TEST002", "TEST003", "TEST004", "TEST005"],
            "ph_level": [7.2, 7.1, 6.3, 7.3, 6.5],  # TEST003, TEST005 suspicious
            "temperature": [4.0, 3.8, 15.0, 4.2, 12.5],  # TEST003, TEST005 high
            "turbidity": [1.2, 0.8, 8.5, 1.0, 7.2],  # TEST003, TEST005 high
            "processing_time": [2.5, 2.3, 6.0, 2.4, 5.5],  # TEST003, TEST005 long
        }
    )

    print("Test samples:")
    print(test_samples)

    # Get the latest run ID
    latest_run = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["start_time DESC"]).iloc[0]
    run_id = latest_run.run_id

    # Make predictions
    predictions = load_and_predict(run_id, test_samples)

    print("\n--- Prediction Results ---")
    print(
        predictions[
            [
                "sample_id",
                "ph_level",
                "temperature",
                "predicted_contamination",
                "contamination_probability",
                "risk_level",
            ]
        ]
    )

    # 4. Summary
    print("\n" + "=" * 70)
    print(" DEMO COMPLETE ".center(70, "="))
    print("=" * 70)
    print("\n✓ Model trained and logged to MLflow")
    print(f"✓ Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"✓ Test F1 Score: {metrics['test_f1']:.4f}")
    print(f"✓ MLflow Run ID: {run_id}")
    print(f"\nMLflow UI: Run 'mlflow ui' and open http://localhost:5000")
    print(f"Model Registry: {MLFLOW_TRACKING_URI}")

    # Check for high-risk samples
    high_risk = predictions[predictions["risk_level"] == "HIGH"]
    if len(high_risk) > 0:
        print(f"\n⚠️  WARNING: {len(high_risk)} high-risk samples detected!")
        print("Samples requiring immediate attention:")
        for _, sample in high_risk.iterrows():
            print(
                f"  - {sample['sample_id']}: {sample['contamination_probability']:.2%} contamination probability"
            )
    else:
        print("\n✓ No high-risk samples detected")

    print("\n" + "=" * 70)
