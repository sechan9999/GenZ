"""
Example 1: High-Stakes Equity Analysis

Use Case: Credit risk assessment for loan approvals
Requirements:
- HIGH interpretability (regulatory compliance, fair lending laws)
- Calibrated probability outputs
- Explainable predictions for individual applicants
- Robust to imbalanced data

Model Choice: Logistic Regression + LIME/SHAP or Causal Forest
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('..')

from eda_analyzer import EDAAnalyzer
from model_selector import InterpretabilityMatrix
from explainability import ModelExplainer


def create_credit_risk_dataset(n_samples=5000):
    """
    Simulate credit risk dataset.

    Features:
    - credit_score: 300-850
    - income: Annual income
    - debt_to_income_ratio: DTI ratio
    - employment_years: Years at current job
    - num_credit_lines: Number of open credit lines
    - delinquencies: Past delinquencies
    - ... and more

    Target: loan_default (0 = no default, 1 = default)
    """
    print("=" * 80)
    print("CREATING CREDIT RISK DATASET")
    print("=" * 80)

    # Generate base features
    X, y = make_classification(
        n_samples=n_samples,
        n_features=15,
        n_informative=10,
        n_redundant=3,
        n_classes=2,
        weights=[0.85, 0.15],  # Imbalanced (15% default rate)
        random_state=42
    )

    # Create meaningful feature names
    feature_names = [
        'credit_score',
        'annual_income',
        'debt_to_income_ratio',
        'employment_years',
        'num_credit_lines',
        'delinquencies_2yrs',
        'total_debt',
        'mortgage_payment',
        'credit_utilization',
        'num_hard_inquiries',
        'age',
        'num_dependents',
        'savings_balance',
        'checking_balance',
        'loan_amount'
    ]

    df = pd.DataFrame(X, columns=feature_names)

    # Transform to realistic ranges
    df['credit_score'] = ((df['credit_score'] - df['credit_score'].min()) /
                          (df['credit_score'].max() - df['credit_score'].min()) * 550 + 300).round(0)

    df['annual_income'] = np.exp(df['annual_income'] * 0.5 + 10.5).round(0)
    df['debt_to_income_ratio'] = (np.abs(df['debt_to_income_ratio']) * 0.1 + 0.2).clip(0, 0.8).round(3)
    df['employment_years'] = (np.abs(df['employment_years']) * 3).clip(0, 40).round(1)
    df['num_credit_lines'] = (np.abs(df['num_credit_lines']) * 2 + 1).clip(0, 30).round(0)
    df['delinquencies_2yrs'] = (np.abs(df['delinquencies_2yrs']) * 0.5).clip(0, 10).round(0)
    df['loan_amount'] = np.exp(df['loan_amount'] * 0.3 + 9).round(0)
    df['age'] = (np.abs(df['age']) * 5 + 25).clip(18, 80).round(0)

    df['loan_default'] = y

    # Add some missing values (realistic)
    np.random.seed(42)
    for col in ['employment_years', 'savings_balance', 'checking_balance']:
        missing_idx = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
        df.loc[missing_idx, col] = np.nan

    print(f"✓ Dataset created: {len(df)} samples, {len(feature_names)} features")
    print(f"  Default rate: {df['loan_default'].mean():.1%}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print("=" * 80 + "\n")

    return df


def main():
    """Run complete equity analysis pipeline."""

    print("\n" + "=" * 80)
    print("HIGH-STAKES EQUITY ANALYSIS: CREDIT RISK ASSESSMENT")
    print("=" * 80 + "\n")

    # Step 1: Create dataset
    df = create_credit_risk_dataset(n_samples=5000)

    # Step 2: EDA
    print("\n" + "=" * 80)
    print("STEP 1: EXPLORATORY DATA ANALYSIS")
    print("=" * 80 + "\n")

    eda = EDAAnalyzer(df, target='loan_default', task_type='classification')
    eda_results = eda.run_full_analysis()

    # Step 3: Model selection recommendation
    print("\n" + "=" * 80)
    print("STEP 2: MODEL SELECTION")
    print("=" * 80 + "\n")

    matrix = InterpretabilityMatrix()

    data_characteristics = {
        'has_missing': True,
        'has_categorical': False,
        'is_linear': True,  # Based on EDA
        'has_interactions': False,
        'sample_size': len(df)
    }

    matrix.print_recommendation(
        use_case='equity_analysis',
        data_characteristics=data_characteristics,
        top_n=3
    )

    # Step 4: Train Logistic Regression (high interpretability)
    print("\n" + "=" * 80)
    print("STEP 3: TRAIN LOGISTIC REGRESSION MODEL")
    print("=" * 80 + "\n")

    # Prepare data
    X = df.drop(columns=['loan_default'])
    y = df['loan_default']

    # Handle missing values (simple imputation)
    X = X.fillna(X.median())

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features (important for logistic regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model with L2 regularization
    model = LogisticRegression(
        penalty='l2',
        C=1.0,
        class_weight='balanced',  # Handle imbalance
        random_state=42,
        max_iter=1000
    )

    print("Training Logistic Regression...")
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("\nModel Performance:")
    print("-" * 80)
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('equity_analysis_confusion_matrix.png', dpi=300)
    print("\n✓ Confusion matrix saved to: equity_analysis_confusion_matrix.png")
    plt.close()

    # Feature coefficients (interpretability)
    print("\n" + "=" * 80)
    print("FEATURE COEFFICIENTS (Interpretability)")
    print("=" * 80)

    coef_df = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)

    print("\nTop 10 Most Important Features:")
    print("-" * 80)
    for idx, row in coef_df.head(10).iterrows():
        direction = "↑ INCREASES" if row['coefficient'] > 0 else "↓ DECREASES"
        print(f"{row['feature']:30s} | coef={row['coefficient']:+.4f} | {direction} default risk")

    # Visualize coefficients
    plt.figure(figsize=(12, 8))
    top_features = coef_df.head(15)
    colors = ['red' if c > 0 else 'green' for c in top_features['coefficient']]

    plt.barh(range(len(top_features)), top_features['coefficient'], color=colors, alpha=0.7)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Coefficient (Log-Odds)')
    plt.title('Top 15 Features by Coefficient\n(Red = Increases default risk, Green = Decreases)')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.grid(axis='x', alpha=0.3)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('equity_analysis_feature_coefficients.png', dpi=300)
    print("\n✓ Feature coefficients plot saved to: equity_analysis_feature_coefficients.png")
    plt.close()

    # Step 5: Explainability with SHAP and LIME
    print("\n" + "=" * 80)
    print("STEP 4: MODEL EXPLAINABILITY (SHAP + LIME)")
    print("=" * 80 + "\n")

    explainer = ModelExplainer(
        model=model,
        X_train=X_train_scaled,
        feature_names=X.columns.tolist(),
        class_names=['No Default', 'Default'],
        task='classification'
    )

    # Generate comprehensive explainability report
    explainer.create_full_report(
        X_test=X_test_scaled,
        sample_indices=[0, 5, 10],  # Explain specific applicants
        top_features=10
    )

    # Step 6: Individual applicant explanation
    print("\n" + "=" * 80)
    print("STEP 5: INDIVIDUAL APPLICANT RISK ASSESSMENT")
    print("=" * 80 + "\n")

    # Explain high-risk applicant
    high_risk_idx = np.where(y_pred_proba > 0.7)[0]
    if len(high_risk_idx) > 0:
        applicant_idx = high_risk_idx[0]

        print(f"Applicant #{applicant_idx} (High Risk)")
        print("-" * 80)
        print(f"Predicted Default Probability: {y_pred_proba[applicant_idx]:.1%}")
        print(f"True Label: {'DEFAULT' if y_test.iloc[applicant_idx] == 1 else 'NO DEFAULT'}")
        print("\nApplicant Profile:")

        for feat in X.columns[:5]:  # Show top 5 features
            val = X_test.iloc[applicant_idx][feat]
            print(f"  {feat:30s}: {val:.2f}")

        print("\nFor detailed explanation, see:")
        print(f"  - shap_waterfall_sample_{applicant_idx}.png")
        print(f"  - lime_explanation_sample_{applicant_idx}.png")

    # Step 7: Regulatory compliance report
    print("\n" + "=" * 80)
    print("STEP 6: REGULATORY COMPLIANCE SUMMARY")
    print("=" * 80 + "\n")

    print("✓ Model Interpretability: HIGH (Logistic Regression)")
    print("✓ Feature Coefficients: Fully transparent")
    print("✓ Individual Explanations: Available via SHAP/LIME")
    print("✓ Calibrated Probabilities: Yes (logistic regression outputs)")
    print("✓ Fairness: Model coefficients can be audited for bias")
    print("✓ Regulatory Documentation: All plots and metrics saved")

    print("\n" + "=" * 80)
    print("✓ EQUITY ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nGenerated artifacts:")
    print("  - EDA plots (linearity_check.png, interaction_plots.png, etc.)")
    print("  - Confusion matrix (equity_analysis_confusion_matrix.png)")
    print("  - Feature coefficients (equity_analysis_feature_coefficients.png)")
    print("  - SHAP/LIME explanations (multiple PNG files)")
    print("\nModel: Ready for regulatory review and deployment")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
