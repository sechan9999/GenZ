"""
Healthcare Model Interpretability Framework
===========================================

Demonstrates the trade-off between model complexity and interpretability
in clinical and policy settings.

Rule of Thumb:
1. Individual patient decisions → Logistic Regression or Monotonic GBM + SHAP
2. Population surveillance → XGBoost + SHAP global + PDP
3. Black-box gain >20% AUC → Deploy with strict monitoring + model cards

Author: Gen Z Agent
Date: 2025-11-22
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Any
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import PartialDependenceDisplay

# XGBoost with monotonic constraints
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

# SHAP for interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Install with: pip install shap")


# =============================================================================
# DATA CLASSES FOR MODEL CARDS AND MONITORING
# =============================================================================

@dataclass
class ModelCard:
    """Model documentation following Google's Model Card framework."""
    model_name: str
    model_type: str
    intended_use: str
    training_date: str
    performance_metrics: Dict[str, float]
    fairness_metrics: Dict[str, Any]
    limitations: List[str]
    ethical_considerations: List[str]

    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'intended_use': self.intended_use,
            'training_date': self.training_date,
            'performance_metrics': self.performance_metrics,
            'fairness_metrics': self.fairness_metrics,
            'limitations': self.limitations,
            'ethical_considerations': self.ethical_considerations
        }

    def print_card(self):
        """Print formatted model card."""
        print("=" * 80)
        print(f"MODEL CARD: {self.model_name}")
        print("=" * 80)
        print(f"\nModel Type: {self.model_type}")
        print(f"Intended Use: {self.intended_use}")
        print(f"Training Date: {self.training_date}")

        print("\n--- PERFORMANCE METRICS ---")
        for metric, value in self.performance_metrics.items():
            print(f"  {metric}: {value:.4f}")

        print("\n--- FAIRNESS METRICS ---")
        for metric, value in self.fairness_metrics.items():
            print(f"  {metric}: {value}")

        print("\n--- LIMITATIONS ---")
        for limitation in self.limitations:
            print(f"  • {limitation}")

        print("\n--- ETHICAL CONSIDERATIONS ---")
        for consideration in self.ethical_considerations:
            print(f"  • {consideration}")
        print("=" * 80 + "\n")


@dataclass
class ModelMonitor:
    """Continuous monitoring for deployed models."""
    alert_threshold: float = 0.05  # AUC drop threshold

    def __init__(self, baseline_auc: float, alert_threshold: float = 0.05):
        self.baseline_auc = baseline_auc
        self.alert_threshold = alert_threshold
        self.monitoring_log = []

    def check_performance(self, current_auc: float, timestamp: datetime) -> bool:
        """Check if current performance is acceptable."""
        drop = self.baseline_auc - current_auc
        alert = drop > self.alert_threshold

        self.monitoring_log.append({
            'timestamp': timestamp,
            'current_auc': current_auc,
            'baseline_auc': self.baseline_auc,
            'drop': drop,
            'alert': alert
        })

        if alert:
            print(f"⚠️  ALERT: AUC dropped by {drop:.4f} (threshold: {self.alert_threshold})")
            print(f"    Baseline: {self.baseline_auc:.4f} → Current: {current_auc:.4f}")

        return not alert

    def get_monitoring_summary(self) -> pd.DataFrame:
        """Get monitoring log as DataFrame."""
        return pd.DataFrame(self.monitoring_log)


# =============================================================================
# SCENARIO 1: INDIVIDUAL PATIENT DECISIONS
# Use interpretable models (Logistic Regression or Monotonic GBM)
# =============================================================================

class ClinicalDecisionModel:
    """
    For individual patient decisions (e.g., hospitalization risk, treatment selection).
    Prioritizes interpretability over marginal performance gains.
    """

    def __init__(self, use_monotonic_gbm: bool = False):
        self.use_monotonic_gbm = use_monotonic_gbm
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.model_card = None

    def train(self, X: pd.DataFrame, y: pd.Series,
              monotonic_constraints: Dict[str, int] = None):
        """
        Train interpretable model for clinical decisions.

        Args:
            X: Features
            y: Target (binary outcome)
            monotonic_constraints: Dict mapping feature names to constraints
                                   1 = positive monotonic, -1 = negative, 0 = none
        """
        self.feature_names = X.columns.tolist()

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        if self.use_monotonic_gbm and XGBOOST_AVAILABLE:
            # Monotonic Gradient Boosting with constraints
            # E.g., age → positive risk, BP medication → negative risk
            if monotonic_constraints:
                constraint_list = [monotonic_constraints.get(f, 0) for f in self.feature_names]
            else:
                constraint_list = [0] * len(self.feature_names)

            self.model = xgb.XGBClassifier(
                max_depth=3,  # Keep shallow for interpretability
                n_estimators=50,
                learning_rate=0.1,
                monotone_constraints=tuple(constraint_list),
                random_state=42
            )
            print("Training Monotonic XGBoost for clinical decisions...")
        else:
            # Logistic Regression - most interpretable
            self.model = LogisticRegression(
                penalty='l2',
                C=1.0,
                max_iter=1000,
                random_state=42
            )
            print("Training Logistic Regression for clinical decisions...")

        self.model.fit(X_scaled, y)

        # Create model card
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
        auc = roc_auc_score(y, y_pred_proba)

        self.model_card = ModelCard(
            model_name="Clinical Decision Support Model",
            model_type="Monotonic XGBoost" if self.use_monotonic_gbm else "Logistic Regression",
            intended_use="Individual patient risk assessment for clinical decision-making",
            training_date=datetime.now().strftime("%Y-%m-%d"),
            performance_metrics={'AUC-ROC': auc},
            fairness_metrics={'Subgroup analysis': 'Required before deployment'},
            limitations=[
                "Model trained on historical data - may not generalize to new populations",
                "Requires regular recalibration as clinical practices evolve",
                "Should be used as decision support, not autonomous decision-making"
            ],
            ethical_considerations=[
                "Must be validated across demographic subgroups",
                "Clinicians should understand feature importance before using predictions",
                "Patient consent required for algorithmic risk assessment"
            ]
        )

        return self

    def explain_prediction(self, X: pd.DataFrame, patient_idx: int = 0):
        """
        Explain prediction for a specific patient using SHAP.
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available. Install with: pip install shap")
            return

        X_scaled = self.scaler.transform(X)

        # SHAP explainer
        if isinstance(self.model, LogisticRegression):
            explainer = shap.LinearExplainer(self.model, X_scaled)
        else:
            explainer = shap.TreeExplainer(self.model)

        shap_values = explainer.shap_values(X_scaled)

        # For binary classification, shap_values might be a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class

        # Explain single patient
        print(f"\n--- SHAP Explanation for Patient {patient_idx} ---")

        # Waterfall plot for individual prediction
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[patient_idx],
                base_values=explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value[1],
                data=X.iloc[patient_idx].values,
                feature_names=self.feature_names
            )
        )
        plt.title(f"SHAP Waterfall Plot - Patient {patient_idx}")
        plt.tight_layout()
        plt.savefig(f'/home/user/GenZ/output/shap_patient_{patient_idx}_waterfall.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Feature importance
        print("\nTop features contributing to this prediction:")
        feature_impact = pd.DataFrame({
            'Feature': self.feature_names,
            'SHAP Value': shap_values[patient_idx],
            'Feature Value': X.iloc[patient_idx].values
        }).sort_values('SHAP Value', key=abs, ascending=False)

        print(feature_impact.head(10))

        return shap_values

    def get_feature_importance(self):
        """Get global feature importance."""
        if isinstance(self.model, LogisticRegression):
            # Logistic regression coefficients
            importance = pd.DataFrame({
                'Feature': self.feature_names,
                'Coefficient': self.model.coef_[0],
                'Abs_Coefficient': np.abs(self.model.coef_[0])
            }).sort_values('Abs_Coefficient', ascending=False)

            print("\n--- Logistic Regression Coefficients ---")
            print(importance)

            # Visualize
            plt.figure(figsize=(10, 6))
            plt.barh(importance['Feature'][:10], importance['Coefficient'][:10])
            plt.xlabel('Coefficient (log-odds)')
            plt.title('Top 10 Features - Logistic Regression')
            plt.tight_layout()
            plt.savefig('/home/user/GenZ/output/logreg_coefficients.png', dpi=300, bbox_inches='tight')
            plt.close()

            return importance
        else:
            # Tree-based feature importance
            importance = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)

            print("\n--- XGBoost Feature Importance ---")
            print(importance)

            # Visualize
            plt.figure(figsize=(10, 6))
            plt.barh(importance['Feature'][:10], importance['Importance'][:10])
            plt.xlabel('Importance')
            plt.title('Top 10 Features - XGBoost')
            plt.tight_layout()
            plt.savefig('/home/user/GenZ/output/xgb_importance.png', dpi=300, bbox_inches='tight')
            plt.close()

            return importance


# =============================================================================
# SCENARIO 2: POPULATION SURVEILLANCE
# Use XGBoost + SHAP Global + Partial Dependence Plots
# =============================================================================

class PopulationSurveillanceModel:
    """
    For population-level surveillance and policy decisions.
    Allows more complex models with comprehensive interpretability tools.
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.model_card = None

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train XGBoost for population surveillance."""
        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)

        if XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(
                max_depth=6,  # Can be deeper for population models
                n_estimators=100,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        else:
            # Fallback to sklearn GBM
            self.model = GradientBoostingClassifier(
                max_depth=6,
                n_estimators=100,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )

        print("Training XGBoost for population surveillance...")
        self.model.fit(X_scaled, y)

        # Model card
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
        auc = roc_auc_score(y, y_pred_proba)

        self.model_card = ModelCard(
            model_name="Population Surveillance Model",
            model_type="XGBoost Classifier",
            intended_use="Population-level disease surveillance and policy planning",
            training_date=datetime.now().strftime("%Y-%m-%d"),
            performance_metrics={'AUC-ROC': auc},
            fairness_metrics={'Geographic equity': 'To be assessed', 'Age stratification': 'To be assessed'},
            limitations=[
                "Designed for aggregate predictions, not individual diagnoses",
                "May not capture rapid epidemiological changes",
                "Requires periodic retraining with recent surveillance data"
            ],
            ethical_considerations=[
                "Population-level predictions should not stigmatize subgroups",
                "Model outputs should inform resource allocation equitably",
                "Privacy-preserving aggregation required for reporting"
            ]
        )

        return self

    def global_interpretability(self, X: pd.DataFrame):
        """
        Generate global interpretability artifacts:
        1. SHAP summary plot (global feature importance)
        2. SHAP dependence plots (feature interactions)
        3. Partial Dependence Plots (marginal effects)
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available for global interpretability.")
            return

        X_scaled = self.scaler.transform(X)

        # SHAP TreeExplainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_scaled)

        # For binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # 1. SHAP Summary Plot (beeswarm)
        print("\n--- Generating SHAP Summary Plot ---")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, feature_names=self.feature_names, show=False)
        plt.title('SHAP Summary Plot - Global Feature Importance')
        plt.tight_layout()
        plt.savefig('/home/user/GenZ/output/shap_summary_population.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. SHAP Bar Plot (mean absolute SHAP)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, feature_names=self.feature_names, plot_type='bar', show=False)
        plt.title('SHAP Feature Importance - Mean Absolute Impact')
        plt.tight_layout()
        plt.savefig('/home/user/GenZ/output/shap_bar_population.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. SHAP Dependence Plots for top 3 features
        print("\n--- Generating SHAP Dependence Plots ---")
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_features_idx = np.argsort(mean_abs_shap)[-3:][::-1]

        for idx in top_features_idx:
            feature_name = self.feature_names[idx]
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                idx, shap_values, X,
                feature_names=self.feature_names,
                show=False
            )
            plt.title(f'SHAP Dependence Plot - {feature_name}')
            plt.tight_layout()
            plt.savefig(f'/home/user/GenZ/output/shap_dependence_{feature_name}.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 4. Partial Dependence Plots
        print("\n--- Generating Partial Dependence Plots ---")
        fig, ax = plt.subplots(figsize=(14, 8))

        # PDP for top 4 features
        top_4_idx = np.argsort(mean_abs_shap)[-4:][::-1].tolist()

        display = PartialDependenceDisplay.from_estimator(
            self.model,
            X_scaled,
            features=top_4_idx,
            feature_names=self.feature_names,
            ax=ax,
            n_cols=2,
            grid_resolution=50
        )
        plt.suptitle('Partial Dependence Plots - Top 4 Features', fontsize=16)
        plt.tight_layout()
        plt.savefig('/home/user/GenZ/output/pdp_population.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ Global interpretability artifacts saved to /home/user/GenZ/output/")

    def subgroup_analysis(self, X: pd.DataFrame, y: pd.Series,
                          subgroup_column: str):
        """
        Analyze model performance across subgroups (e.g., age groups, regions).
        Critical for ensuring equitable population surveillance.
        """
        # Separate subgroup column from features
        subgroup_values = X[subgroup_column]
        X_features = X.drop(subgroup_column, axis=1)

        X_scaled = self.scaler.transform(X_features)
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]

        # Group by subgroup
        subgroups = subgroup_values.unique()

        print(f"\n--- Subgroup Analysis by {subgroup_column} ---")
        results = []

        for subgroup in subgroups:
            mask = subgroup_values == subgroup
            if mask.sum() < 10:  # Skip small groups
                continue

            y_true_sub = y[mask]
            y_pred_sub = y_pred_proba[mask]

            auc = roc_auc_score(y_true_sub, y_pred_sub)
            n_samples = mask.sum()
            prevalence = y_true_sub.mean()

            results.append({
                'Subgroup': subgroup,
                'N': n_samples,
                'Prevalence': prevalence,
                'AUC': auc
            })

        results_df = pd.DataFrame(results).sort_values('AUC')
        print(results_df)

        # Visualize
        plt.figure(figsize=(10, 6))
        plt.barh(results_df['Subgroup'].astype(str), results_df['AUC'])
        plt.xlabel('AUC-ROC')
        plt.title(f'Model Performance by {subgroup_column}')
        plt.axvline(x=0.7, color='r', linestyle='--', label='Minimum Acceptable (0.70)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'/home/user/GenZ/output/subgroup_auc_{subgroup_column}.png', dpi=300, bbox_inches='tight')
        plt.close()

        return results_df


# =============================================================================
# SCENARIO 3: BLACK-BOX MODEL WITH STRICT MONITORING
# Deploy complex models if AUC gain > 20% with comprehensive monitoring
# =============================================================================

class MonitoredBlackBoxModel:
    """
    For cases where black-box models (e.g., deep learning, ensemble)
    provide >20% AUC improvement. Requires strict monitoring and model cards.
    """

    def __init__(self, baseline_auc: float):
        self.baseline_auc = baseline_auc
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.model_card = None
        self.monitor = None

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train complex ensemble model."""
        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)

        # Complex ensemble (Random Forest as proxy for black-box)
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )

        print("Training complex ensemble model (black-box)...")
        self.model.fit(X_scaled, y)

        # Evaluate
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
        current_auc = roc_auc_score(y, y_pred_proba)

        # Check if deployment is justified
        auc_gain = (current_auc - self.baseline_auc) / self.baseline_auc
        print(f"\n📊 AUC Comparison:")
        print(f"   Baseline (interpretable): {self.baseline_auc:.4f}")
        print(f"   Black-box model: {current_auc:.4f}")
        print(f"   Relative gain: {auc_gain*100:.1f}%")

        if auc_gain < 0.20:
            print(f"\n⚠️  WARNING: AUC gain ({auc_gain*100:.1f}%) < 20% threshold")
            print("   Consider using interpretable model instead.")
        else:
            print(f"\n✅ AUC gain ({auc_gain*100:.1f}%) exceeds 20% threshold")
            print("   Black-box deployment justified with strict monitoring.")

        # Create comprehensive model card
        self.model_card = ModelCard(
            model_name="High-Performance Black-Box Model",
            model_type="Random Forest Ensemble (200 trees, depth=10)",
            intended_use="High-stakes prediction where >20% AUC gain justifies complexity",
            training_date=datetime.now().strftime("%Y-%m-%d"),
            performance_metrics={
                'AUC-ROC': current_auc,
                'Baseline AUC': self.baseline_auc,
                'Relative Gain': auc_gain
            },
            fairness_metrics={
                'Subgroup parity': 'Required - must test across demographics',
                'Calibration': 'Required - must be well-calibrated',
                'Disparate impact': 'Must be <1.25 across protected groups'
            },
            limitations=[
                "Complex model - limited interpretability at individual prediction level",
                "Requires SHAP/LIME for post-hoc explanations",
                "Higher risk of overfitting - requires careful validation",
                "May not generalize well to distribution shifts",
                "Computationally expensive for real-time inference"
            ],
            ethical_considerations=[
                "MANDATORY: Continuous monitoring of performance degradation",
                "MANDATORY: Regular fairness audits across demographic subgroups",
                "MANDATORY: Human oversight for high-stakes decisions",
                "MANDATORY: Documented escalation protocol for anomalies",
                "MANDATORY: Annual model recertification",
                "Patients must be informed when algorithmic decisions are involved",
                "Must have fail-safe mechanism to revert to baseline model"
            ]
        )

        # Set up monitoring
        self.monitor = ModelMonitor(
            baseline_auc=current_auc,
            alert_threshold=0.05  # Alert if AUC drops >5%
        )

        return self

    def deploy_with_monitoring(self, X_validation: pd.DataFrame,
                               y_validation: pd.Series):
        """
        Simulate deployment with continuous monitoring.
        """
        print("\n" + "="*80)
        print("DEPLOYING BLACK-BOX MODEL WITH STRICT MONITORING")
        print("="*80)

        # Print model card
        self.model_card.print_card()

        # Initial validation
        X_val_scaled = self.scaler.transform(X_validation)
        y_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]
        validation_auc = roc_auc_score(y_validation, y_pred_proba)

        print(f"Initial Validation AUC: {validation_auc:.4f}")

        # Check against monitor
        is_healthy = self.monitor.check_performance(
            validation_auc,
            datetime.now()
        )

        if is_healthy:
            print("✅ Model performance within acceptable range - DEPLOYED")
        else:
            print("🚨 Model performance degraded - DEPLOYMENT BLOCKED")

        return is_healthy

    def explain_with_shap(self, X: pd.DataFrame, n_samples: int = 100):
        """
        Generate SHAP explanations for black-box model.
        Use TreeExplainer for tree-based, KernelExplainer for true black-box.
        """
        if not SHAP_AVAILABLE:
            print("SHAP required for black-box explanations.")
            return

        X_scaled = self.scaler.transform(X)

        # For large datasets, use subsample
        if len(X) > n_samples:
            sample_idx = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X_scaled[sample_idx]
            X_df_sample = X.iloc[sample_idx]
        else:
            X_sample = X_scaled
            X_df_sample = X

        print(f"\n--- Generating SHAP Explanations (n={len(X_sample)}) ---")
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_df_sample, feature_names=self.feature_names, show=False)
        plt.title('SHAP Explanations - Black-Box Model')
        plt.tight_layout()
        plt.savefig('/home/user/GenZ/output/shap_blackbox_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ SHAP explanations saved.")


# =============================================================================
# SYNTHETIC HEALTHCARE DATA GENERATOR
# =============================================================================

def generate_synthetic_patient_data(n_samples: int = 1000,
                                   scenario: str = 'hospital_readmission') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate synthetic healthcare data for demonstration.

    Scenarios:
    - 'hospital_readmission': Predict 30-day readmission risk
    - 'diabetes_progression': Predict diabetes complications
    - 'sepsis_risk': Predict sepsis in ICU patients
    """
    np.random.seed(42)

    if scenario == 'hospital_readmission':
        # Features with clinical meaning
        age = np.random.normal(65, 15, n_samples).clip(18, 95)
        prior_admissions = np.random.poisson(2, n_samples).clip(0, 10)
        comorbidity_count = np.random.poisson(3, n_samples).clip(0, 15)
        length_of_stay = np.random.exponential(5, n_samples).clip(1, 30)
        num_medications = np.random.poisson(7, n_samples).clip(0, 25)
        emergency_admit = np.random.binomial(1, 0.3, n_samples)
        icu_stay = np.random.binomial(1, 0.2, n_samples)

        # Labs
        hemoglobin = np.random.normal(13, 2, n_samples).clip(7, 18)
        creatinine = np.random.lognormal(0.3, 0.5, n_samples).clip(0.5, 10)
        sodium = np.random.normal(140, 3, n_samples).clip(125, 155)

        # Socioeconomic
        low_income = np.random.binomial(1, 0.25, n_samples)
        rural = np.random.binomial(1, 0.15, n_samples)

        # Target: readmission (with realistic relationships)
        logit = (
            -5.0 +
            0.05 * age +
            0.4 * prior_admissions +
            0.3 * comorbidity_count +
            0.1 * length_of_stay +
            0.05 * num_medications +
            0.8 * emergency_admit +
            0.6 * icu_stay +
            -0.3 * hemoglobin +
            0.5 * np.log(creatinine) +
            0.5 * low_income +
            0.3 * rural
        )

        prob = 1 / (1 + np.exp(-logit))
        readmitted = np.random.binomial(1, prob, n_samples)

        X = pd.DataFrame({
            'age': age,
            'prior_admissions': prior_admissions,
            'comorbidity_count': comorbidity_count,
            'length_of_stay': length_of_stay,
            'num_medications': num_medications,
            'emergency_admit': emergency_admit,
            'icu_stay': icu_stay,
            'hemoglobin': hemoglobin,
            'creatinine': creatinine,
            'sodium': sodium,
            'low_income': low_income,
            'rural': rural,
            'age_group': pd.cut(age, bins=[0, 50, 65, 80, 100], labels=['<50', '50-65', '65-80', '80+'])
        })

        y = pd.Series(readmitted, name='readmitted')

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return X, y


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================

def main():
    """
    Demonstrate the three scenarios for balancing complexity vs interpretability.
    """
    print("\n" + "="*80)
    print("HEALTHCARE MODEL INTERPRETABILITY FRAMEWORK")
    print("Balancing Complexity vs. Interpretability in Clinical Settings")
    print("="*80 + "\n")

    # Generate synthetic data
    print("📊 Generating synthetic hospital readmission data...")
    X, y = generate_synthetic_patient_data(n_samples=1000, scenario='hospital_readmission')

    print(f"   Dataset: {len(X)} patients, {y.sum()} readmissions ({y.mean()*100:.1f}%)")
    print(f"   Features: {', '.join(X.columns[:8])}...\n")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X.drop('age_group', axis=1), y, test_size=0.3, random_state=42, stratify=y
    )

    # Keep age_group for subgroup analysis
    age_group_test = X.loc[X_test.index, 'age_group']

    # =========================================================================
    # SCENARIO 1: Individual Patient Decisions
    # =========================================================================
    print("\n" + "="*80)
    print("SCENARIO 1: INDIVIDUAL PATIENT DECISIONS")
    print("Use Case: Predict readmission risk to guide discharge planning")
    print("Approach: Logistic Regression (most interpretable)")
    print("="*80 + "\n")

    clinical_model = ClinicalDecisionModel(use_monotonic_gbm=False)
    clinical_model.train(X_train, y_train)

    # Evaluate
    X_test_scaled = clinical_model.scaler.transform(X_test)
    y_pred_proba = clinical_model.model.predict_proba(X_test_scaled)[:, 1]
    auc_interpretable = roc_auc_score(y_test, y_pred_proba)
    print(f"\n✅ Logistic Regression AUC: {auc_interpretable:.4f}")

    # Model card
    clinical_model.model_card.print_card()

    # Feature importance
    clinical_model.get_feature_importance()

    # Explain single patient
    if SHAP_AVAILABLE:
        clinical_model.explain_prediction(X_test, patient_idx=0)

    # =========================================================================
    # SCENARIO 2: Population Surveillance
    # =========================================================================
    print("\n" + "="*80)
    print("SCENARIO 2: POPULATION SURVEILLANCE")
    print("Use Case: Monitor regional readmission trends for policy decisions")
    print("Approach: XGBoost + SHAP Global + PDP")
    print("="*80 + "\n")

    if XGBOOST_AVAILABLE:
        pop_model = PopulationSurveillanceModel()
        pop_model.train(X_train, y_train)

        # Evaluate
        X_test_scaled_pop = pop_model.scaler.transform(X_test)
        y_pred_proba_pop = pop_model.model.predict_proba(X_test_scaled_pop)[:, 1]
        auc_population = roc_auc_score(y_test, y_pred_proba_pop)
        print(f"\n✅ XGBoost AUC: {auc_population:.4f}")

        # Model card
        pop_model.model_card.print_card()

        # Global interpretability
        pop_model.global_interpretability(X_test)

        # Subgroup analysis
        X_test_with_group = X_test.copy()
        X_test_with_group['age_group'] = age_group_test
        pop_model.subgroup_analysis(X_test_with_group, y_test, 'age_group')
    else:
        print("⚠️  XGBoost not available - skipping population surveillance demo")
        auc_population = auc_interpretable  # Use baseline

    # =========================================================================
    # SCENARIO 3: Black-Box with Monitoring (if justified)
    # =========================================================================
    print("\n" + "="*80)
    print("SCENARIO 3: BLACK-BOX MODEL WITH STRICT MONITORING")
    print("Use Case: Deploy complex model if AUC gain > 20%")
    print("Approach: Ensemble + Model Card + Continuous Monitoring")
    print("="*80 + "\n")

    blackbox_model = MonitoredBlackBoxModel(baseline_auc=auc_interpretable)
    blackbox_model.train(X_train, y_train)

    # Deploy with monitoring
    is_deployed = blackbox_model.deploy_with_monitoring(X_test, y_test)

    if SHAP_AVAILABLE and is_deployed:
        blackbox_model.explain_with_shap(X_test, n_samples=100)

    # =========================================================================
    # FINAL RECOMMENDATIONS
    # =========================================================================
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR YOUR HEALTHCARE PROJECT")
    print("="*80 + "\n")

    print("1️⃣  FOR INDIVIDUAL PATIENT DECISIONS:")
    print("   → Use Logistic Regression or Monotonic GBM")
    print("   → Provide SHAP waterfall plots to clinicians")
    print("   → Feature importance must align with clinical knowledge")
    print(f"   → Your baseline: AUC = {auc_interpretable:.3f}\n")

    print("2️⃣  FOR POPULATION SURVEILLANCE:")
    print("   → XGBoost is acceptable with interpretability tools")
    print("   → Generate SHAP summary + PDP for policymakers")
    print("   → Conduct subgroup analysis (age, region, SES)")
    print(f"   → Your performance: AUC = {auc_population:.3f}\n")

    print("3️⃣  FOR BLACK-BOX MODELS:")
    print("   → Only if AUC gain > 20% over interpretable baseline")
    print("   → MANDATORY: Comprehensive model cards")
    print("   → MANDATORY: Continuous performance monitoring")
    print("   → MANDATORY: Fairness audits across demographics")
    print("   → MANDATORY: Human oversight for high-stakes decisions\n")

    print("📁 Outputs saved to: /home/user/GenZ/output/")
    print("   - shap_*.png (SHAP plots)")
    print("   - logreg_coefficients.png (feature importance)")
    print("   - pdp_population.png (partial dependence)")
    print("   - subgroup_auc_*.png (fairness analysis)")

    print("\n" + "="*80)
    print("✅ DEMONSTRATION COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Create output directory if needed
    import os
    os.makedirs('/home/user/GenZ/output', exist_ok=True)

    main()
