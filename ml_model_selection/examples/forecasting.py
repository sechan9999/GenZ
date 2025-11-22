"""
Example 2: Operational Forecasting

Use Case: Hospital patient readmission prediction
Requirements:
- HIGH performance (maximize prediction accuracy)
- MEDIUM interpretability (clinical insights needed)
- Handle missing values (real-world clinical data)
- Handle interactions (complex medical patterns)

Model Choice: XGBoost/LightGBM + SHAP global summaries
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('..')

from eda_analyzer import EDAAnalyzer
from model_selector import InterpretabilityMatrix
from automl_pipeline import AutoMLPipeline
from explainability import ModelExplainer


def create_readmission_dataset(n_samples=10000):
    """
    Simulate hospital readmission dataset.

    Features:
    - Patient demographics (age, gender)
    - Vital signs (BP, heart rate, temperature)
    - Lab results (glucose, creatinine, hemoglobin)
    - Medications (num_medications, insulin)
    - Prior admissions
    - Length of stay
    - Diagnosis codes

    Target: readmitted_30days (0 = no, 1 = yes)
    """
    print("=" * 80)
    print("CREATING HOSPITAL READMISSION DATASET")
    print("=" * 80)

    # Generate base features with interactions
    X, y = make_classification(
        n_samples=n_samples,
        n_features=25,
        n_informative=18,
        n_redundant=4,
        n_classes=2,
        weights=[0.75, 0.25],  # 25% readmission rate
        random_state=42
    )

    feature_names = [
        'age',
        'gender',
        'bmi',
        'systolic_bp',
        'diastolic_bp',
        'heart_rate',
        'temperature',
        'glucose_level',
        'creatinine',
        'hemoglobin',
        'wbc_count',
        'platelet_count',
        'num_medications',
        'on_insulin',
        'num_prior_admissions',
        'length_of_stay',
        'num_diagnoses',
        'num_procedures',
        'emergency_admission',
        'icu_days',
        'comorbidity_score',
        'discharge_to_facility',
        'has_diabetes',
        'has_hypertension',
        'has_heart_disease'
    ]

    df = pd.DataFrame(X, columns=feature_names)

    # Transform to realistic ranges
    df['age'] = ((df['age'] - df['age'].min()) /
                 (df['age'].max() - df['age'].min()) * 75 + 18).round(0)

    df['gender'] = (df['gender'] > df['gender'].median()).astype(int)  # Binary

    df['bmi'] = (np.abs(df['bmi']) * 3 + 22).clip(15, 50).round(1)
    df['systolic_bp'] = (np.abs(df['systolic_bp']) * 15 + 110).clip(90, 200).round(0)
    df['diastolic_bp'] = (np.abs(df['diastolic_bp']) * 10 + 70).clip(50, 120).round(0)
    df['heart_rate'] = (np.abs(df['heart_rate']) * 10 + 70).clip(40, 150).round(0)
    df['temperature'] = (df['temperature'] * 0.5 + 37).clip(35, 41).round(1)

    df['glucose_level'] = np.exp(df['glucose_level'] * 0.2 + 4.5).round(0)
    df['creatinine'] = (np.abs(df['creatinine']) * 0.3 + 0.8).clip(0.5, 5.0).round(2)
    df['hemoglobin'] = (np.abs(df['hemoglobin']) * 2 + 12).clip(8, 18).round(1)

    df['num_medications'] = (np.abs(df['num_medications']) * 3).clip(0, 30).round(0)
    df['on_insulin'] = (df['on_insulin'] > df['on_insulin'].median()).astype(int)
    df['num_prior_admissions'] = (np.abs(df['num_prior_admissions']) * 2).clip(0, 20).round(0)
    df['length_of_stay'] = (np.abs(df['length_of_stay']) * 2 + 1).clip(1, 30).round(0)
    df['num_diagnoses'] = (np.abs(df['num_diagnoses']) * 2 + 1).clip(1, 15).round(0)
    df['num_procedures'] = (np.abs(df['num_procedures']) * 1.5).clip(0, 10).round(0)

    df['emergency_admission'] = (df['emergency_admission'] > df['emergency_admission'].median()).astype(int)
    df['icu_days'] = (np.abs(df['icu_days']) * 1.5).clip(0, 15).round(0)
    df['comorbidity_score'] = (np.abs(df['comorbidity_score']) * 1.5).clip(0, 10).round(0)
    df['discharge_to_facility'] = (df['discharge_to_facility'] > df['discharge_to_facility'].median()).astype(int)

    df['has_diabetes'] = (df['has_diabetes'] > df['has_diabetes'].median()).astype(int)
    df['has_hypertension'] = (df['has_hypertension'] > df['has_hypertension'].median()).astype(int)
    df['has_heart_disease'] = (df['has_heart_disease'] > df['has_heart_disease'].median()).astype(int)

    df['readmitted_30days'] = y

    # Add realistic missing values (common in clinical data)
    np.random.seed(42)
    missing_patterns = {
        'glucose_level': 0.10,  # 10% missing
        'creatinine': 0.08,
        'hemoglobin': 0.05,
        'wbc_count': 0.07,
        'platelet_count': 0.06,
        'icu_days': 0.15,  # More missing (not all patients go to ICU)
        'num_procedures': 0.12
    }

    for col, missing_pct in missing_patterns.items():
        n_missing = int(missing_pct * len(df))
        missing_idx = np.random.choice(df.index, size=n_missing, replace=False)
        df.loc[missing_idx, col] = np.nan

    print(f"✓ Dataset created: {len(df)} samples, {len(feature_names)} features")
    print(f"  Readmission rate: {df['readmitted_30days'].mean():.1%}")
    print(f"  Missing values: {df.isnull().sum().sum()} ({df.isnull().sum().sum() / df.size * 100:.1f}%)")
    print("=" * 80 + "\n")

    return df


def main():
    """Run complete operational forecasting pipeline."""

    print("\n" + "=" * 80)
    print("OPERATIONAL FORECASTING: HOSPITAL READMISSION PREDICTION")
    print("=" * 80 + "\n")

    # Step 1: Create dataset
    df = create_readmission_dataset(n_samples=10000)

    # Step 2: EDA
    print("\n" + "=" * 80)
    print("STEP 1: EXPLORATORY DATA ANALYSIS")
    print("=" * 80 + "\n")

    eda = EDAAnalyzer(df, target='readmitted_30days', task_type='classification')
    eda_results = eda.run_full_analysis()

    # Step 3: Model selection recommendation
    print("\n" + "=" * 80)
    print("STEP 2: MODEL SELECTION")
    print("=" * 80 + "\n")

    matrix = InterpretabilityMatrix()

    data_characteristics = {
        'has_missing': True,
        'has_categorical': True,
        'is_linear': False,  # Based on EDA
        'has_interactions': True,
        'sample_size': len(df)
    }

    matrix.print_recommendation(
        use_case='operational_forecasting',
        data_characteristics=data_characteristics,
        top_n=3
    )

    # Step 4: Automated model selection with FLAML
    print("\n" + "=" * 80)
    print("STEP 3: AUTOMATED MODEL SELECTION (FLAML)")
    print("=" * 80 + "\n")

    # Prepare data
    X = df.drop(columns=['readmitted_30days'])
    y = df['readmitted_30days']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Note: Tree models handle missing values natively, no imputation needed!

    # Initialize AutoML pipeline
    pipeline = AutoMLPipeline(
        task='classification',
        metric='roc_auc',  # Optimize for AUC (good for imbalanced data)
        time_budget=180,  # 3 minutes for demo (use 3600 for production)
        estimator_list=['lgbm', 'xgboost', 'rf', 'extra_tree', 'catboost'],
        ensemble_method='stacking',  # Build stacking ensemble
        verbose=1
    )

    # Train (FLAML will automatically select best model and hyperparameters)
    pipeline.fit(X_train, y_train)

    # Step 5: Evaluate
    print("\n" + "=" * 80)
    print("STEP 4: MODEL EVALUATION")
    print("=" * 80 + "\n")

    results = pipeline.evaluate(X_test, y_test, use_ensemble=True)

    # Additional metrics
    y_pred_proba = pipeline.predict_proba(X_test, use_ensemble=True)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Plot ROC curve
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc_score(y_test, y_pred_proba):.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Hospital Readmission Prediction')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('forecasting_roc_curve.png', dpi=300)
    print("\n✓ ROC curve saved to: forecasting_roc_curve.png")
    plt.close()

    # Plot precision-recall curve
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, linewidth=2, label=f'PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Hospital Readmission Prediction')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('forecasting_pr_curve.png', dpi=300)
    print("✓ PR curve saved to: forecasting_pr_curve.png")
    plt.close()

    # Step 6: Explainability with SHAP (global summaries for clinical insights)
    print("\n" + "=" * 80)
    print("STEP 5: MODEL EXPLAINABILITY (SHAP GLOBAL SUMMARIES)")
    print("=" * 80 + "\n")

    # Get best model from ensemble
    best_model = pipeline.best_model

    explainer = ModelExplainer(
        model=best_model,
        X_train=X_train.fillna(0).values,  # SHAP needs no missing values
        feature_names=X.columns.tolist(),
        class_names=['No Readmission', 'Readmission'],
        task='classification'
    )

    # SHAP summary (global feature importance)
    explainer.setup_shap(explainer_type='auto')
    explainer.compute_shap_values(X_test.fillna(0).values, max_samples=1000)
    explainer.plot_shap_summary(max_display=20)

    # Feature importance
    explainer.plot_feature_importance(method='shap', top_n=20)

    # Dependence plots for top clinical factors
    top_features = ['num_prior_admissions', 'length_of_stay', 'age']
    for feat in top_features:
        if feat in X.columns:
            explainer.plot_shap_dependence(feat, X_test.fillna(0).values)

    # Step 7: Clinical insights
    print("\n" + "=" * 80)
    print("STEP 6: CLINICAL INSIGHTS & ACTIONABLE RECOMMENDATIONS")
    print("=" * 80 + "\n")

    # Get feature importance from SHAP
    shap_importance = np.abs(explainer.shap_values).mean(axis=0) if not isinstance(explainer.shap_values, list) else np.abs(explainer.shap_values[1]).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': shap_importance
    }).sort_values('importance', ascending=False)

    print("Top 10 Readmission Risk Factors:")
    print("-" * 80)
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {idx+1}. {row['feature']:30s} | SHAP importance: {row['importance']:.4f}")

    print("\nActionable Clinical Interventions:")
    print("-" * 80)

    # Analyze top factors
    if 'num_prior_admissions' in importance_df.head(5)['feature'].values:
        print("  ✓ INTERVENTION 1: Intensive care management for patients with 2+ prior admissions")

    if 'length_of_stay' in importance_df.head(5)['feature'].values:
        print("  ✓ INTERVENTION 2: Enhanced discharge planning for long-stay patients (>7 days)")

    if 'num_medications' in importance_df.head(5)['feature'].values:
        print("  ✓ INTERVENTION 3: Medication reconciliation for polypharmacy patients (>10 meds)")

    if 'discharge_to_facility' in importance_df.head(5)['feature'].values:
        print("  ✓ INTERVENTION 4: Follow-up coordination for patients discharged to facilities")

    if 'comorbidity_score' in importance_df.head(5)['feature'].values:
        print("  ✓ INTERVENTION 5: Disease management programs for high comorbidity scores")

    # Step 8: Save model
    print("\n" + "=" * 80)
    print("STEP 7: SAVE MODEL FOR DEPLOYMENT")
    print("=" * 80 + "\n")

    pipeline.save('saved_models/readmission_model', save_ensemble=True)

    print("\n" + "=" * 80)
    print("✓ OPERATIONAL FORECASTING COMPLETE")
    print("=" * 80)
    print("\nModel Performance Summary:")
    print(f"  - ROC-AUC: {results['ensemble']['roc_auc']:.4f}" if 'ensemble' in results else f"  - ROC-AUC: {results['automl']['roc_auc']:.4f}")
    print(f"  - Precision: {results['ensemble']['precision']:.4f}" if 'ensemble' in results else f"  - Precision: {results['automl']['precision']:.4f}")
    print(f"  - Recall: {results['ensemble']['recall']:.4f}" if 'ensemble' in results else f"  - Recall: {results['automl']['recall']:.4f}")
    print(f"  - F1-Score: {results['ensemble']['f1']:.4f}" if 'ensemble' in results else f"  - F1-Score: {results['automl']['f1']:.4f}")
    print("\nBest Model:", pipeline.automl.best_estimator)
    print("\nGenerated Artifacts:")
    print("  - EDA plots (distribution_analysis.png, missingness_heatmap.png, etc.)")
    print("  - ROC/PR curves (forecasting_roc_curve.png, forecasting_pr_curve.png)")
    print("  - SHAP summaries (shap_summary.png, feature_importance_shap.png)")
    print("  - SHAP dependence plots (shap_dependence_*.png)")
    print("  - Saved model (saved_models/readmission_model/)")
    print("\nDeployment: Ready for integration with hospital EHR system")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
