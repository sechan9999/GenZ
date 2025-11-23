# Databricks notebook source
# MAGIC %md
# MAGIC # ML Risk Models: Service Utilization Prediction
# MAGIC
# MAGIC This notebook trains predictive models for:
# MAGIC 1. **High utilization forecasting** - Predict members at risk of high service use
# MAGIC 2. **ER visit prediction** - Forecast emergency department visits
# MAGIC 3. **Readmission risk** - Identify inpatient readmission risk
# MAGIC 4. **Cost prediction** - Forecast total healthcare costs
# MAGIC
# MAGIC All models are tracked with MLflow and registered in the Model Registry.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and Imports

# COMMAND ----------

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as SKRFClassifier, GradientBoostingClassifier as SKGBClassifier
from sklearn.ensemble import RandomForestRegressor as SKRFRegressor, GradientBoostingRegressor as SKGBRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler as SKStandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

# Set MLflow experiment
mlflow.set_experiment("/Medicaid/Risk_Prediction_Models")

print("✓ Imports complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Training Data

# COMMAND ----------

# Load member features from Feature Store
from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

# Load features
member_features_df = spark.table("gold.member_features")

# Convert to Pandas for sklearn
member_features_pdf = member_features_df.toPandas()

print(f"Training data shape: {member_features_pdf.shape}")
print(f"Features: {member_features_pdf.columns.tolist()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Selection

# COMMAND ----------

# Define feature sets for different models

# Demographic features
demographic_features = [
    "age",
    "eligibility_years",
    "dual_eligible",
    "ltss_indicator",
    "high_risk_flag"
]

# Chronic condition features
chronic_condition_features = [
    "has_diabetes",
    "has_hypertension",
    "has_copd_asthma",
    "has_heart_failure",
    "has_ckd",
    "has_mental_health",
    "has_obesity",
    "has_cancer",
    "chronic_condition_count"
]

# Utilization features
utilization_features = [
    "claim_count_12mo",
    "service_days_12mo",
    "er_visit_count_12mo",
    "inpatient_admit_count_12mo",
    "high_cost_claim_count",
    "unique_provider_count",
    "rx_fill_count_12mo",
    "unique_drug_count",
    "controlled_substance_count"
]

# Temporal features
temporal_features = [
    "recent_claim_count_90d",
    "claim_count_trend_pct",
    "cost_trend_pct",
    "days_since_last_medical_claim",
    "days_since_last_er_visit",
    "days_since_last_rx_fill"
]

# Cost features
cost_features = [
    "total_paid_12mo",
    "avg_claim_amount",
    "total_rx_paid_12mo",
    "avg_rx_cost"
]

# All features
all_features = (demographic_features + chronic_condition_features +
                utilization_features + temporal_features + cost_features)

print(f"Total features: {len(all_features)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Target Variables

# COMMAND ----------

# Prepare training dataset with target variables
training_data = member_features_pdf.copy()

# Handle nulls
training_data = training_data.fillna(0)

# Create target variables

# Target 1: High ER utilization (>= 3 ER visits in next period)
# For demo, using current period as proxy
training_data['high_er_utilization'] = (training_data['er_visit_count_12mo'] >= 3).astype(int)

# Target 2: High cost (>= $25,000 in next period)
training_data['high_cost'] = (training_data['total_paid_12mo'] >= 25000).astype(int)

# Target 3: High risk tier
training_data['high_risk'] = (training_data['risk_tier'].isin(['High', 'Medium'])).astype(int)

# Target 4: Total cost (for regression)
training_data['total_cost'] = training_data['total_paid_12mo'] + training_data['total_rx_paid_12mo']

# Encode categorical variables
if 'gender' in training_data.columns:
    training_data['gender_encoded'] = training_data['gender'].map({'M': 1, 'F': 0, 'U': 2}).fillna(2)
    all_features.append('gender_encoded')

print("✓ Target variables created")
print(f"High ER Utilization: {training_data['high_er_utilization'].sum()} / {len(training_data)}")
print(f"High Cost: {training_data['high_cost'].sum()} / {len(training_data)}")
print(f"High Risk: {training_data['high_risk'].sum()} / {len(training_data)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model 1: High ER Utilization Prediction

# COMMAND ----------

# Prepare data
X = training_data[all_features].fillna(0)
y = training_data['high_er_utilization']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = SKStandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
print(f"Class distribution - Train: {y_train.value_counts().to_dict()}")

# COMMAND ----------

# Train Random Forest model with MLflow tracking
with mlflow.start_run(run_name="ER_Utilization_RandomForest") as run:
    # Log parameters
    params = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 20,
        "random_state": 42,
        "target": "high_er_utilization"
    }
    mlflow.log_params(params)

    # Train model
    rf_model = SKRFClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        random_state=params["random_state"],
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = rf_model.predict(X_test_scaled)
    y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metrics({
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': all_features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Log feature importance
    mlflow.log_table(feature_importance.head(20), "feature_importance.json")

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance.head(15)['feature'], feature_importance.head(15)['importance'])
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importances - ER Utilization Model')
    plt.tight_layout()
    plt.savefig('/tmp/er_feature_importance.png')
    mlflow.log_artifact('/tmp/er_feature_importance.png')
    plt.close()

    # Log model
    signature = infer_signature(X_train_scaled, y_pred_proba)
    mlflow.sklearn.log_model(
        rf_model,
        "model",
        signature=signature,
        registered_model_name="er_utilization_predictor"
    )

    print(f"✓ ER Utilization Model trained")
    print(f"  AUC: {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  MLflow Run ID: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model 2: High Cost Prediction

# COMMAND ----------

# Prepare data for high cost prediction
y_cost = training_data['high_cost']
X_train_cost, X_test_cost, y_train_cost, y_test_cost = train_test_split(
    X, y_cost, test_size=0.2, random_state=42, stratify=y_cost
)

# Standardize
X_train_cost_scaled = scaler.fit_transform(X_train_cost)
X_test_cost_scaled = scaler.transform(X_test_cost)

# Train Gradient Boosting model
with mlflow.start_run(run_name="High_Cost_GradientBoosting") as run:
    params = {
        "model_type": "GradientBoostingClassifier",
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "random_state": 42,
        "target": "high_cost"
    }
    mlflow.log_params(params)

    # Train
    gb_model = SKGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        random_state=params["random_state"]
    )
    gb_model.fit(X_train_cost_scaled, y_train_cost)

    # Predict
    y_pred_cost = gb_model.predict(X_test_cost_scaled)
    y_pred_proba_cost = gb_model.predict_proba(X_test_cost_scaled)[:, 1]

    # Metrics
    auc_cost = roc_auc_score(y_test_cost, y_pred_proba_cost)
    precision_cost = precision_score(y_test_cost, y_pred_cost)
    recall_cost = recall_score(y_test_cost, y_pred_cost)
    f1_cost = f1_score(y_test_cost, y_pred_cost)

    mlflow.log_metrics({
        "auc": auc_cost,
        "precision": precision_cost,
        "recall": recall_cost,
        "f1_score": f1_cost
    })

    # Feature importance
    feature_importance_cost = pd.DataFrame({
        'feature': all_features,
        'importance': gb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    mlflow.log_table(feature_importance_cost.head(20), "feature_importance.json")

    # Log model
    signature = infer_signature(X_train_cost_scaled, y_pred_proba_cost)
    mlflow.sklearn.log_model(
        gb_model,
        "model",
        signature=signature,
        registered_model_name="high_cost_predictor"
    )

    print(f"✓ High Cost Model trained")
    print(f"  AUC: {auc_cost:.4f}")
    print(f"  Precision: {precision_cost:.4f}")
    print(f"  Recall: {recall_cost:.4f}")
    print(f"  F1: {f1_cost:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model 3: Total Cost Regression

# COMMAND ----------

# Prepare data for cost regression
y_total_cost = training_data['total_cost']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_total_cost, test_size=0.2, random_state=42
)

# Standardize
X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)

# Train Gradient Boosting Regressor
with mlflow.start_run(run_name="Total_Cost_Regression") as run:
    params = {
        "model_type": "GradientBoostingRegressor",
        "n_estimators": 150,
        "max_depth": 6,
        "learning_rate": 0.05,
        "random_state": 42,
        "target": "total_cost"
    }
    mlflow.log_params(params)

    # Train
    gbr_model = SKGBRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        random_state=params["random_state"]
    )
    gbr_model.fit(X_train_reg_scaled, y_train_reg)

    # Predict
    y_pred_reg = gbr_model.predict(X_test_reg_scaled)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
    r2 = r2_score(y_test_reg, y_pred_reg)
    mae = np.mean(np.abs(y_test_reg - y_pred_reg))

    mlflow.log_metrics({
        "rmse": rmse,
        "r2_score": r2,
        "mae": mae
    })

    # Residual plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_reg, y_pred_reg, alpha=0.5)
    plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
    plt.xlabel('Actual Cost ($)')
    plt.ylabel('Predicted Cost ($)')
    plt.title(f'Total Cost Prediction (R² = {r2:.3f})')
    plt.tight_layout()
    plt.savefig('/tmp/cost_prediction_scatter.png')
    mlflow.log_artifact('/tmp/cost_prediction_scatter.png')
    plt.close()

    # Log model
    signature = infer_signature(X_train_reg_scaled, y_pred_reg)
    mlflow.sklearn.log_model(
        gbr_model,
        "model",
        signature=signature,
        registered_model_name="total_cost_predictor"
    )

    print(f"✓ Total Cost Regression Model trained")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAE: ${mae:,.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model 4: Composite Risk Score Prediction

# COMMAND ----------

# Use a subset of features that don't include derived risk scores
risk_features = [f for f in all_features if not any(x in f.lower() for x in ['risk', 'score'])]

X_risk = training_data[risk_features].fillna(0)
y_risk = training_data['high_risk']

X_train_risk, X_test_risk, y_train_risk, y_test_risk = train_test_split(
    X_risk, y_risk, test_size=0.2, random_state=42, stratify=y_risk
)

# Standardize
scaler_risk = SKStandardScaler()
X_train_risk_scaled = scaler_risk.fit_transform(X_train_risk)
X_test_risk_scaled = scaler_risk.transform(X_test_risk)

# Train Logistic Regression (interpretable baseline)
with mlflow.start_run(run_name="High_Risk_LogisticRegression") as run:
    params = {
        "model_type": "LogisticRegression",
        "C": 1.0,
        "max_iter": 1000,
        "random_state": 42,
        "target": "high_risk"
    }
    mlflow.log_params(params)

    # Train
    lr_model = LogisticRegression(
        C=params["C"],
        max_iter=params["max_iter"],
        random_state=params["random_state"]
    )
    lr_model.fit(X_train_risk_scaled, y_train_risk)

    # Predict
    y_pred_risk = lr_model.predict(X_test_risk_scaled)
    y_pred_proba_risk = lr_model.predict_proba(X_test_risk_scaled)[:, 1]

    # Metrics
    auc_risk = roc_auc_score(y_test_risk, y_pred_proba_risk)
    precision_risk = precision_score(y_test_risk, y_pred_risk)
    recall_risk = recall_score(y_test_risk, y_pred_risk)
    f1_risk = f1_score(y_test_risk, y_pred_risk)

    mlflow.log_metrics({
        "auc": auc_risk,
        "precision": precision_risk,
        "recall": recall_risk,
        "f1_score": f1_risk
    })

    # Coefficient analysis
    coefficients = pd.DataFrame({
        'feature': risk_features,
        'coefficient': lr_model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)

    mlflow.log_table(coefficients.head(20), "coefficients.json")

    # Log model
    signature = infer_signature(X_train_risk_scaled, y_pred_proba_risk)
    mlflow.sklearn.log_model(
        lr_model,
        "model",
        signature=signature,
        registered_model_name="high_risk_predictor"
    )

    print(f"✓ High Risk Model trained")
    print(f"  AUC: {auc_risk:.4f}")
    print(f"  Precision: {precision_risk:.4f}")
    print(f"  Recall: {recall_risk:.4f}")
    print(f"  F1: {f1_risk:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Comparison

# COMMAND ----------

# Compare all models
model_comparison = pd.DataFrame({
    'Model': ['ER Utilization (RF)', 'High Cost (GB)', 'High Risk (LR)'],
    'AUC': [auc, auc_cost, auc_risk],
    'Precision': [precision, precision_cost, precision_risk],
    'Recall': [recall, recall_cost, recall_risk],
    'F1': [f1, f1_cost, f1_risk]
})

display(model_comparison)

# COMMAND ----------

# Visualize model comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = ['AUC', 'Precision', 'Recall']
for idx, metric in enumerate(metrics):
    axes[idx].bar(model_comparison['Model'], model_comparison[metric])
    axes[idx].set_ylabel(metric)
    axes[idx].set_title(f'{metric} Comparison')
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].set_ylim([0, 1])

plt.tight_layout()
plt.savefig('/tmp/model_comparison.png')
mlflow.log_artifact('/tmp/model_comparison.png')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Model Predictions to Gold Layer

# COMMAND ----------

# Generate predictions for all members
all_predictions = training_data[['member_id']].copy()

# ER Utilization predictions
X_all_scaled = scaler.transform(training_data[all_features].fillna(0))
all_predictions['er_utilization_risk_score'] = rf_model.predict_proba(X_all_scaled)[:, 1]
all_predictions['er_utilization_prediction'] = rf_model.predict(X_all_scaled)

# High Cost predictions
all_predictions['high_cost_risk_score'] = gb_model.predict_proba(X_all_scaled)[:, 1]
all_predictions['high_cost_prediction'] = gb_model.predict(X_all_scaled)

# Total Cost predictions
all_predictions['predicted_total_cost'] = gbr_model.predict(X_all_scaled)

# High Risk predictions
X_all_risk_scaled = scaler_risk.transform(training_data[risk_features].fillna(0))
all_predictions['high_risk_score'] = lr_model.predict_proba(X_all_risk_scaled)[:, 1]
all_predictions['high_risk_prediction'] = lr_model.predict(X_all_risk_scaled)

# Add timestamp
all_predictions['prediction_timestamp'] = pd.Timestamp.now()

# Convert to Spark DataFrame and save
predictions_sdf = spark.createDataFrame(all_predictions)

(predictions_sdf.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("gold.member_risk_predictions"))

print(f"✓ Predictions saved to gold.member_risk_predictions")
print(f"  Total members scored: {len(all_predictions):,}")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Risk prediction summary
# MAGIC SELECT
# MAGIC   COUNT(*) as total_members,
# MAGIC   SUM(CASE WHEN high_risk_prediction = 1 THEN 1 ELSE 0 END) as high_risk_members,
# MAGIC   SUM(CASE WHEN er_utilization_prediction = 1 THEN 1 ELSE 0 END) as high_er_utilization_members,
# MAGIC   SUM(CASE WHEN high_cost_prediction = 1 THEN 1 ELSE 0 END) as high_cost_members,
# MAGIC   AVG(predicted_total_cost) as avg_predicted_cost,
# MAGIC   AVG(high_risk_score) as avg_risk_score
# MAGIC FROM gold.member_risk_predictions

# COMMAND ----------

print("=" * 60)
print("ML MODEL TRAINING COMPLETE")
print("=" * 60)
print("\nAll models trained and registered in MLflow Model Registry:")
print("  1. ER Utilization Predictor")
print("  2. High Cost Predictor")
print("  3. Total Cost Regressor")
print("  4. High Risk Predictor")
print("\nPredictions saved to: gold.member_risk_predictions")
