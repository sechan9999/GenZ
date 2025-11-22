"""
Model Selector with Interpretability Matrix

Decision framework for selecting ML models based on:
- Use case requirements (interpretability, performance, latency)
- Data characteristics (linearity, interactions, missingness)
- Operational constraints (deployment environment, compliance)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class UseCase(Enum):
    """Pre-defined use cases with different requirements."""
    EQUITY_ANALYSIS = "equity_analysis"  # High-stakes, need interpretability
    OPERATIONAL_FORECASTING = "operational_forecasting"  # Performance + some interpretability
    LOW_LATENCY = "low_latency"  # Ultra-fast inference
    EXPLORATORY = "exploratory"  # General purpose
    CAUSAL_INFERENCE = "causal_inference"  # Causal effect estimation


@dataclass
class ModelRequirements:
    """Requirements for model selection."""
    interpretability: str  # 'high', 'medium', 'low'
    performance_priority: str  # 'high', 'medium', 'low'
    latency_requirement: str  # 'ultra_low', 'low', 'medium', 'high'
    handles_missing: bool = False
    handles_categorical: bool = False
    handles_interactions: bool = False
    requires_calibration: bool = False
    compliance_required: bool = False  # GDPR, healthcare, finance


@dataclass
class ModelProfile:
    """Profile for each model type."""
    name: str
    interpretability: str  # 'high', 'medium', 'low'
    performance: str  # 'high', 'medium', 'low'
    latency: str  # 'ultra_low', 'low', 'medium', 'high'
    handles_missing: bool
    handles_categorical: bool
    handles_interactions: bool
    requires_feature_engineering: bool
    supports_incremental_learning: bool
    typical_use_cases: List[str]
    pros: List[str]
    cons: List[str]


class InterpretabilityMatrix:
    """
    Decision matrix for model selection based on use case and data characteristics.

    The matrix considers:
    1. Use case requirements (interpretability, performance, latency)
    2. Data characteristics (linearity, missing values, interactions)
    3. Operational constraints (compliance, deployment environment)
    """

    def __init__(self):
        """Initialize model profiles."""
        self.model_profiles = self._build_model_profiles()
        self.use_case_requirements = self._build_use_case_requirements()

    def _build_model_profiles(self) -> Dict[str, ModelProfile]:
        """Build comprehensive model profiles."""
        return {
            'logistic_regression': ModelProfile(
                name='Logistic Regression',
                interpretability='high',
                performance='medium',
                latency='ultra_low',
                handles_missing=False,
                handles_categorical=False,
                handles_interactions=False,
                requires_feature_engineering=True,
                supports_incremental_learning=True,
                typical_use_cases=['equity_analysis', 'causal_inference', 'compliance'],
                pros=[
                    'Highly interpretable (coefficients = feature importance)',
                    'Fast training and inference',
                    'Probabilistic outputs (calibrated)',
                    'Works well with linear relationships',
                    'Regularization (L1/L2) for feature selection'
                ],
                cons=[
                    'Assumes linearity',
                    'Requires feature engineering for interactions',
                    'Poor with non-linear patterns',
                    'Sensitive to outliers',
                    'Needs manual handling of missing values'
                ]
            ),

            'random_forest': ModelProfile(
                name='Random Forest',
                interpretability='medium',
                performance='high',
                latency='medium',
                handles_missing=True,
                handles_categorical=True,
                handles_interactions=True,
                requires_feature_engineering=False,
                supports_incremental_learning=False,
                typical_use_cases=['operational_forecasting', 'exploratory'],
                pros=[
                    'Handles non-linear relationships',
                    'Built-in feature importance',
                    'Robust to outliers',
                    'Handles missing values (surrogate splits)',
                    'No need for feature scaling'
                ],
                cons=[
                    'Can overfit with small datasets',
                    'Large model size (many trees)',
                    'Slower inference than linear models',
                    'Less interpretable than single trees',
                    'Biased toward high-cardinality features'
                ]
            ),

            'xgboost': ModelProfile(
                name='XGBoost',
                interpretability='medium',
                performance='high',
                latency='low',
                handles_missing=True,
                handles_categorical=True,
                handles_interactions=True,
                requires_feature_engineering=False,
                supports_incremental_learning=True,
                typical_use_cases=['operational_forecasting', 'kaggle_competitions'],
                pros=[
                    'State-of-the-art performance',
                    'Built-in regularization',
                    'Handles missing values natively',
                    'Fast training (parallelized)',
                    'SHAP values for interpretability',
                    'Early stopping to prevent overfitting'
                ],
                cons=[
                    'Many hyperparameters to tune',
                    'Can overfit if not regularized',
                    'Requires more memory than linear models',
                    'Less interpretable than linear models',
                    'Sensitive to hyperparameter choices'
                ]
            ),

            'lightgbm': ModelProfile(
                name='LightGBM',
                interpretability='medium',
                performance='high',
                latency='ultra_low',
                handles_missing=True,
                handles_categorical=True,
                handles_interactions=True,
                requires_feature_engineering=False,
                supports_incremental_learning=True,
                typical_use_cases=['operational_forecasting', 'low_latency'],
                pros=[
                    'Extremely fast training and inference',
                    'Memory efficient (histogram-based)',
                    'Native categorical support',
                    'Handles large datasets well',
                    'SHAP values for interpretability',
                    'Built-in early stopping'
                ],
                cons=[
                    'Can overfit on small datasets',
                    'Sensitive to hyperparameters',
                    'Less mature than XGBoost',
                    'May need careful tuning for optimal results'
                ]
            ),

            'catboost': ModelProfile(
                name='CatBoost',
                interpretability='medium',
                performance='high',
                latency='low',
                handles_missing=True,
                handles_categorical=True,
                handles_interactions=True,
                requires_feature_engineering=False,
                supports_incremental_learning=True,
                typical_use_cases=['operational_forecasting', 'categorical_heavy'],
                pros=[
                    'Best-in-class categorical handling',
                    'Ordered boosting (less overfitting)',
                    'Fewer hyperparameters to tune',
                    'Good default parameters',
                    'Built-in cross-validation',
                    'SHAP values for interpretability'
                ],
                cons=[
                    'Slower training than LightGBM',
                    'Larger model size',
                    'Less community support than XGBoost',
                    'May be overkill for simple problems'
                ]
            ),

            'causal_forest': ModelProfile(
                name='Causal Forest (EconML)',
                interpretability='high',
                performance='medium',
                latency='high',
                handles_missing=False,
                handles_categorical=False,
                handles_interactions=True,
                requires_feature_engineering=True,
                supports_incremental_learning=False,
                typical_use_cases=['causal_inference', 'equity_analysis', 'policy_evaluation'],
                pros=[
                    'Estimates heterogeneous treatment effects',
                    'Provides confidence intervals',
                    'Handles non-linear treatment effects',
                    'Robust to confounding (with proper design)',
                    'Interpretable treatment effects'
                ],
                cons=[
                    'Requires treatment/control groups',
                    'Computationally expensive',
                    'Needs large sample sizes',
                    'Assumes no unmeasured confounding',
                    'Complex to implement and validate'
                ]
            ),

            'neural_network': ModelProfile(
                name='Neural Network (Deep Learning)',
                interpretability='low',
                performance='high',
                latency='medium',
                handles_missing=False,
                handles_categorical=True,
                handles_interactions=True,
                requires_feature_engineering=False,
                supports_incremental_learning=True,
                typical_use_cases=['image_classification', 'nlp', 'complex_patterns'],
                pros=[
                    'Learns complex non-linear patterns',
                    'State-of-the-art for images/text',
                    'Flexible architecture',
                    'Can handle very large datasets',
                    'Transfer learning possible'
                ],
                cons=[
                    'Black box (low interpretability)',
                    'Requires large datasets',
                    'Computationally expensive',
                    'Many hyperparameters',
                    'Prone to overfitting',
                    'Requires GPU for efficiency'
                ]
            ),

            'svm': ModelProfile(
                name='Support Vector Machine',
                interpretability='medium',
                performance='medium',
                latency='low',
                handles_missing=False,
                handles_categorical=False,
                handles_interactions=True,
                requires_feature_engineering=True,
                supports_incremental_learning=False,
                typical_use_cases=['small_datasets', 'high_dimensional'],
                pros=[
                    'Works well with high-dimensional data',
                    'Effective with small datasets',
                    'Kernel trick for non-linearity',
                    'Robust to overfitting (with right kernel)'
                ],
                cons=[
                    'Slow on large datasets',
                    'Sensitive to feature scaling',
                    'Difficult to interpret',
                    'Many kernel choices',
                    'No probabilistic outputs (unless calibrated)'
                ]
            )
        }

    def _build_use_case_requirements(self) -> Dict[str, ModelRequirements]:
        """Build requirements for each use case."""
        return {
            UseCase.EQUITY_ANALYSIS.value: ModelRequirements(
                interpretability='high',
                performance_priority='medium',
                latency_requirement='medium',
                requires_calibration=True,
                compliance_required=True
            ),

            UseCase.OPERATIONAL_FORECASTING.value: ModelRequirements(
                interpretability='medium',
                performance_priority='high',
                latency_requirement='low',
                handles_missing=True,
                handles_categorical=True,
                handles_interactions=True
            ),

            UseCase.LOW_LATENCY.value: ModelRequirements(
                interpretability='low',
                performance_priority='high',
                latency_requirement='ultra_low',
                handles_missing=False
            ),

            UseCase.EXPLORATORY.value: ModelRequirements(
                interpretability='medium',
                performance_priority='medium',
                latency_requirement='medium'
            ),

            UseCase.CAUSAL_INFERENCE.value: ModelRequirements(
                interpretability='high',
                performance_priority='medium',
                latency_requirement='high',
                compliance_required=True
            )
        }

    def select_models(
        self,
        use_case: str,
        data_characteristics: Optional[Dict[str, Any]] = None,
        top_n: int = 3
    ) -> List[Tuple[str, ModelProfile, float]]:
        """
        Select top N models based on use case and data characteristics.

        Args:
            use_case: Use case key (e.g., 'equity_analysis')
            data_characteristics: Dict with keys like 'has_missing', 'has_categorical',
                                'is_linear', 'has_interactions', 'sample_size'
            top_n: Number of top models to return

        Returns:
            List of (model_key, model_profile, score) tuples, sorted by score
        """
        if use_case not in self.use_case_requirements:
            raise ValueError(f"Unknown use case: {use_case}. Available: {list(self.use_case_requirements.keys())}")

        requirements = self.use_case_requirements[use_case]

        # Score each model
        scores = {}
        for model_key, profile in self.model_profiles.items():
            score = self._score_model(profile, requirements, data_characteristics)
            scores[model_key] = score

        # Sort by score
        ranked_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Return top N with profiles
        results = [
            (model_key, self.model_profiles[model_key], score)
            for model_key, score in ranked_models[:top_n]
        ]

        return results

    def _score_model(
        self,
        profile: ModelProfile,
        requirements: ModelRequirements,
        data_characteristics: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Score a model based on requirements and data characteristics.

        Scoring logic:
        - Base score: match on interpretability, performance, latency
        - Bonus: handles data characteristics (missing, categorical, interactions)
        - Penalty: doesn't meet requirements
        """
        score = 0.0

        # Interpretability match (weight: 30%)
        interp_map = {'high': 3, 'medium': 2, 'low': 1}
        if profile.interpretability == requirements.interpretability:
            score += 30
        elif interp_map[profile.interpretability] >= interp_map[requirements.interpretability]:
            score += 20  # Higher interpretability than required is ok
        else:
            score += 10  # Lower interpretability is a penalty

        # Performance match (weight: 25%)
        perf_map = {'high': 3, 'medium': 2, 'low': 1}
        if requirements.performance_priority == 'high':
            score += perf_map[profile.performance] * 8
        elif requirements.performance_priority == 'medium':
            score += perf_map[profile.performance] * 5
        else:
            score += 10  # Performance not critical

        # Latency match (weight: 25%)
        latency_map = {'ultra_low': 4, 'low': 3, 'medium': 2, 'high': 1}
        required_latency_score = latency_map[requirements.latency_requirement]
        profile_latency_score = latency_map[profile.latency]

        if profile_latency_score >= required_latency_score:
            score += 25  # Meets latency requirement
        else:
            score += 10  # Doesn't meet latency

        # Data characteristics bonus (weight: 20%)
        if data_characteristics:
            if data_characteristics.get('has_missing', False) and profile.handles_missing:
                score += 5
            if data_characteristics.get('has_categorical', False) and profile.handles_categorical:
                score += 5
            if data_characteristics.get('has_interactions', False) and profile.handles_interactions:
                score += 5
            if data_characteristics.get('is_linear', False) and profile.name in ['Logistic Regression', 'Linear Regression']:
                score += 5
            if data_characteristics.get('sample_size', 1000) < 1000 and profile.name in ['Logistic Regression', 'SVM']:
                score += 5  # Small data bonus for simpler models

        # Compliance bonus
        if requirements.compliance_required and profile.interpretability == 'high':
            score += 10

        return score

    def print_recommendation(
        self,
        use_case: str,
        data_characteristics: Optional[Dict[str, Any]] = None,
        top_n: int = 3
    ) -> None:
        """
        Print detailed model recommendations.

        Args:
            use_case: Use case key
            data_characteristics: Data characteristics dict
            top_n: Number of recommendations to show
        """
        print("=" * 80)
        print("MODEL SELECTION RECOMMENDATION")
        print("=" * 80)
        print(f"\n📋 Use Case: {use_case.upper().replace('_', ' ')}")

        if use_case in self.use_case_requirements:
            req = self.use_case_requirements[use_case]
            print(f"\n📌 Requirements:")
            print(f"   - Interpretability: {req.interpretability.upper()}")
            print(f"   - Performance Priority: {req.performance_priority.upper()}")
            print(f"   - Latency Requirement: {req.latency_requirement.upper().replace('_', ' ')}")
            print(f"   - Compliance Required: {req.compliance_required}")

        if data_characteristics:
            print(f"\n📊 Data Characteristics:")
            for key, value in data_characteristics.items():
                print(f"   - {key.replace('_', ' ').title()}: {value}")

        print(f"\n🏆 Top {top_n} Recommended Models:")
        print("=" * 80)

        recommendations = self.select_models(use_case, data_characteristics, top_n)

        for rank, (model_key, profile, score) in enumerate(recommendations, 1):
            print(f"\n#{rank}. {profile.name.upper()} (Score: {score:.1f}/100)")
            print("-" * 80)
            print(f"   Interpretability: {profile.interpretability.upper()}")
            print(f"   Performance: {profile.performance.upper()}")
            print(f"   Latency: {profile.latency.upper().replace('_', ' ')}")
            print(f"\n   ✅ Pros:")
            for pro in profile.pros[:3]:
                print(f"      • {pro}")
            print(f"\n   ⚠️  Cons:")
            for con in profile.cons[:3]:
                print(f"      • {con}")
            print(f"\n   💡 Typical Use Cases: {', '.join(profile.typical_use_cases)}")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    # Example usage
    matrix = InterpretabilityMatrix()

    # Example 1: Equity Analysis (High-stakes decision making)
    print("\n" + "=" * 80)
    print("EXAMPLE 1: EQUITY ANALYSIS")
    print("=" * 80)

    data_chars_equity = {
        'has_missing': True,
        'has_categorical': False,
        'is_linear': True,
        'has_interactions': False,
        'sample_size': 5000
    }

    matrix.print_recommendation(
        use_case='equity_analysis',
        data_characteristics=data_chars_equity,
        top_n=3
    )

    # Example 2: Operational Forecasting
    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: OPERATIONAL FORECASTING")
    print("=" * 80)

    data_chars_forecast = {
        'has_missing': True,
        'has_categorical': True,
        'is_linear': False,
        'has_interactions': True,
        'sample_size': 50000
    }

    matrix.print_recommendation(
        use_case='operational_forecasting',
        data_characteristics=data_chars_forecast,
        top_n=3
    )

    # Example 3: Low Latency Deployment
    print("\n\n" + "=" * 80)
    print("EXAMPLE 3: LOW LATENCY DEPLOYMENT")
    print("=" * 80)

    data_chars_latency = {
        'has_missing': False,
        'has_categorical': False,
        'is_linear': False,
        'has_interactions': True,
        'sample_size': 100000
    }

    matrix.print_recommendation(
        use_case='low_latency',
        data_characteristics=data_chars_latency,
        top_n=3
    )
