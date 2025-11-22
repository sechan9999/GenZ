"""
Model Explainability with SHAP and LIME

Comprehensive model interpretation tools:
- SHAP values (global and local explanations)
- LIME explanations (local interpretable model-agnostic)
- Feature importance visualizations
- Partial dependence plots
- Individual prediction explanations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Union, Any, Dict
import warnings
warnings.filterwarnings('ignore')

# SHAP
import shap

# LIME
from lime import lime_tabular

# Scikit-learn
from sklearn.inspection import partial_dependence, PartialDependenceDisplay


class ModelExplainer:
    """
    Comprehensive model explainability toolkit.

    Supports:
    - SHAP (TreeExplainer, KernelExplainer)
    - LIME
    - Feature importance
    - Partial dependence plots
    """

    def __init__(
        self,
        model: Any,
        X_train: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        task: str = 'classification'
    ):
        """
        Initialize explainer.

        Args:
            model: Trained model
            X_train: Training data (used as background for SHAP)
            feature_names: Feature names (auto-detected from DataFrame)
            class_names: Class names for classification
            task: 'classification' or 'regression'
        """
        self.model = model
        self.task = task

        # Handle pandas DataFrame
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            self.X_train = X_train.values
        else:
            self.X_train = X_train
            if feature_names is not None:
                self.feature_names = feature_names
            else:
                self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

        self.class_names = class_names

        # Initialize SHAP explainer
        self.shap_explainer = None
        self.shap_values = None

        # Initialize LIME explainer
        self.lime_explainer = None

        print("=" * 80)
        print("Model Explainer Initialized")
        print("=" * 80)
        print(f"Task: {task}")
        print(f"Features: {len(self.feature_names)}")
        print(f"Training samples: {len(self.X_train):,}")
        print("=" * 80 + "\n")

    def setup_shap(
        self,
        explainer_type: str = 'auto',  # 'auto', 'tree', 'kernel', 'linear'
        n_background: int = 100
    ) -> None:
        """
        Setup SHAP explainer.

        Args:
            explainer_type: Type of SHAP explainer
            n_background: Number of background samples for KernelExplainer
        """
        print(f"Setting up SHAP explainer (type: {explainer_type})...")

        if explainer_type == 'auto':
            # Auto-detect based on model type
            model_type = type(self.model).__name__

            if 'XGB' in model_type or 'LGBM' in model_type or 'CatBoost' in model_type or 'Forest' in model_type or 'Tree' in model_type:
                explainer_type = 'tree'
            else:
                explainer_type = 'kernel'

        if explainer_type == 'tree':
            # TreeExplainer (fast for tree-based models)
            self.shap_explainer = shap.TreeExplainer(self.model)
            print("✓ Using TreeExplainer (fast, exact for tree models)")

        elif explainer_type == 'kernel':
            # KernelExplainer (model-agnostic, slower)
            background = shap.sample(self.X_train, n_background)

            if self.task == 'classification':
                if hasattr(self.model, 'predict_proba'):
                    predict_fn = lambda x: self.model.predict_proba(x)
                else:
                    predict_fn = lambda x: self.model.predict(x)
            else:
                predict_fn = lambda x: self.model.predict(x)

            self.shap_explainer = shap.KernelExplainer(predict_fn, background)
            print(f"✓ Using KernelExplainer (model-agnostic, background: {n_background} samples)")

        elif explainer_type == 'linear':
            # LinearExplainer (for linear models)
            self.shap_explainer = shap.LinearExplainer(self.model, self.X_train)
            print("✓ Using LinearExplainer (fast for linear models)")

        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")

    def compute_shap_values(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        max_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute SHAP values for given samples.

        Args:
            X: Samples to explain
            max_samples: Limit number of samples (for performance)

        Returns:
            SHAP values array
        """
        if self.shap_explainer is None:
            self.setup_shap()

        if isinstance(X, pd.DataFrame):
            X = X.values

        if max_samples is not None and len(X) > max_samples:
            print(f"Limiting to {max_samples} samples for SHAP computation...")
            X = X[:max_samples]

        print(f"Computing SHAP values for {len(X)} samples...")
        self.shap_values = self.shap_explainer.shap_values(X)

        print("✓ SHAP values computed\n")
        return self.shap_values

    def plot_shap_summary(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        max_display: int = 20,
        plot_type: str = 'dot'  # 'dot', 'bar', 'violin'
    ) -> None:
        """
        Create SHAP summary plot (global feature importance).

        Args:
            X: Samples (uses X_train if None)
            max_display: Max features to display
            plot_type: Plot type ('dot', 'bar', 'violin')
        """
        if X is None:
            X = self.X_train
        elif isinstance(X, pd.DataFrame):
            X = X.values

        if self.shap_values is None:
            self.compute_shap_values(X, max_samples=1000)

        print("Creating SHAP summary plot...")

        # Handle multi-class classification
        shap_vals = self.shap_values
        if isinstance(shap_vals, list):  # Multi-class
            # Use class 1 for binary, or average for multi-class
            if len(shap_vals) == 2:
                shap_vals = shap_vals[1]
            else:
                shap_vals = np.mean(shap_vals, axis=0)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_vals,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            plot_type=plot_type,
            show=False
        )
        plt.tight_layout()
        plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
        print("✓ SHAP summary plot saved to: shap_summary.png\n")
        plt.close()

    def plot_shap_waterfall(
        self,
        sample_idx: int,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None
    ) -> None:
        """
        Create SHAP waterfall plot for a single prediction.

        Args:
            sample_idx: Index of sample to explain
            X: Samples (uses X_train if None)
        """
        if X is None:
            X = self.X_train
        elif isinstance(X, pd.DataFrame):
            X = X.values

        if self.shap_values is None:
            self.compute_shap_values(X)

        print(f"Creating SHAP waterfall plot for sample {sample_idx}...")

        # Handle multi-class
        shap_vals = self.shap_values
        if isinstance(shap_vals, list):
            if len(shap_vals) == 2:
                shap_vals = shap_vals[1]
            else:
                shap_vals = shap_vals[1]  # Use class 1

        # Create explanation object
        explanation = shap.Explanation(
            values=shap_vals[sample_idx],
            base_values=self.shap_explainer.expected_value if not isinstance(self.shap_explainer.expected_value, list) else self.shap_explainer.expected_value[1],
            data=X[sample_idx],
            feature_names=self.feature_names
        )

        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(explanation, show=False)
        plt.tight_layout()
        plt.savefig(f'shap_waterfall_sample_{sample_idx}.png', dpi=300, bbox_inches='tight')
        print(f"✓ SHAP waterfall plot saved to: shap_waterfall_sample_{sample_idx}.png\n")
        plt.close()

    def plot_shap_force(
        self,
        sample_idx: int,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None
    ) -> None:
        """
        Create SHAP force plot for a single prediction.

        Args:
            sample_idx: Index of sample to explain
            X: Samples (uses X_train if None)
        """
        if X is None:
            X = self.X_train
        elif isinstance(X, pd.DataFrame):
            X = X.values

        if self.shap_values is None:
            self.compute_shap_values(X)

        print(f"Creating SHAP force plot for sample {sample_idx}...")

        # Handle multi-class
        shap_vals = self.shap_values
        base_value = self.shap_explainer.expected_value

        if isinstance(shap_vals, list):
            if len(shap_vals) == 2:
                shap_vals = shap_vals[1]
                base_value = base_value[1] if isinstance(base_value, list) else base_value
            else:
                shap_vals = shap_vals[1]
                base_value = base_value[1]

        # Generate force plot
        shap.force_plot(
            base_value,
            shap_vals[sample_idx],
            X[sample_idx],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        plt.tight_layout()
        plt.savefig(f'shap_force_sample_{sample_idx}.png', dpi=300, bbox_inches='tight')
        print(f"✓ SHAP force plot saved to: shap_force_sample_{sample_idx}.png\n")
        plt.close()

    def plot_shap_dependence(
        self,
        feature_name: str,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        interaction_feature: Optional[str] = 'auto'
    ) -> None:
        """
        Create SHAP dependence plot for a feature.

        Args:
            feature_name: Feature to plot
            X: Samples (uses X_train if None)
            interaction_feature: Feature to color by ('auto' for automatic selection)
        """
        if X is None:
            X = self.X_train
        elif isinstance(X, pd.DataFrame):
            X = X.values

        if self.shap_values is None:
            self.compute_shap_values(X, max_samples=1000)

        print(f"Creating SHAP dependence plot for '{feature_name}'...")

        # Get feature index
        feature_idx = self.feature_names.index(feature_name)

        # Handle multi-class
        shap_vals = self.shap_values
        if isinstance(shap_vals, list):
            if len(shap_vals) == 2:
                shap_vals = shap_vals[1]
            else:
                shap_vals = np.mean(shap_vals, axis=0)

        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx,
            shap_vals,
            X,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False
        )
        plt.tight_layout()
        plt.savefig(f'shap_dependence_{feature_name}.png', dpi=300, bbox_inches='tight')
        print(f"✓ SHAP dependence plot saved to: shap_dependence_{feature_name}.png\n")
        plt.close()

    def setup_lime(
        self,
        mode: str = 'classification',  # 'classification' or 'regression'
        discretize_continuous: bool = True
    ) -> None:
        """
        Setup LIME explainer.

        Args:
            mode: 'classification' or 'regression'
            discretize_continuous: Discretize continuous features
        """
        print("Setting up LIME explainer...")

        self.lime_explainer = lime_tabular.LimeTabularExplainer(
            self.X_train,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode=mode,
            discretize_continuous=discretize_continuous
        )

        print("✓ LIME explainer ready\n")

    def explain_lime(
        self,
        sample_idx: int,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        num_features: int = 10,
        num_samples: int = 5000
    ) -> None:
        """
        Create LIME explanation for a single prediction.

        Args:
            sample_idx: Index of sample to explain
            X: Samples (uses X_train if None)
            num_features: Number of features to show
            num_samples: Number of samples for LIME
        """
        if self.lime_explainer is None:
            self.setup_lime(mode=self.task)

        if X is None:
            X = self.X_train
        elif isinstance(X, pd.DataFrame):
            X = X.values

        print(f"Creating LIME explanation for sample {sample_idx}...")

        # Get prediction function
        if self.task == 'classification':
            if hasattr(self.model, 'predict_proba'):
                predict_fn = self.model.predict_proba
            else:
                # Wrap predict to return probabilities
                predict_fn = lambda x: np.column_stack([1 - self.model.predict(x), self.model.predict(x)])
        else:
            predict_fn = self.model.predict

        # Generate explanation
        explanation = self.lime_explainer.explain_instance(
            X[sample_idx],
            predict_fn,
            num_features=num_features,
            num_samples=num_samples
        )

        # Save to file
        fig = explanation.as_pyplot_figure()
        plt.tight_layout()
        plt.savefig(f'lime_explanation_sample_{sample_idx}.png', dpi=300, bbox_inches='tight')
        print(f"✓ LIME explanation saved to: lime_explanation_sample_{sample_idx}.png\n")
        plt.close()

        # Print explanation
        print("LIME Feature Weights:")
        print("-" * 60)
        for feature, weight in explanation.as_list():
            print(f"  {feature:40s}: {weight:+.4f}")
        print()

    def plot_feature_importance(
        self,
        method: str = 'shap',  # 'shap', 'permutation', or 'builtin'
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        top_n: int = 20
    ) -> None:
        """
        Plot feature importance.

        Args:
            method: Importance method ('shap', 'permutation', 'builtin')
            X: Samples for SHAP/permutation importance
            top_n: Number of top features to show
        """
        print(f"Computing feature importance ({method})...")

        if method == 'shap':
            if X is None:
                X = self.X_train
            elif isinstance(X, pd.DataFrame):
                X = X.values

            if self.shap_values is None:
                self.compute_shap_values(X, max_samples=1000)

            # Handle multi-class
            shap_vals = self.shap_values
            if isinstance(shap_vals, list):
                if len(shap_vals) == 2:
                    shap_vals = shap_vals[1]
                else:
                    shap_vals = np.mean(shap_vals, axis=0)

            # Mean absolute SHAP values
            importance = np.abs(shap_vals).mean(axis=0)

        elif method == 'builtin':
            # Use model's built-in feature importance
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                importance = np.abs(self.model.coef_).flatten()
            else:
                print("⚠️  Model doesn't have built-in feature importance")
                return

        else:
            print(f"⚠️  Unknown method: {method}")
            return

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        # Plot
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(top_n)

        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance ({method.upper()})')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'feature_importance_{method}.png', dpi=300, bbox_inches='tight')
        print(f"✓ Feature importance plot saved to: feature_importance_{method}.png\n")
        plt.close()

        # Print top features
        print(f"Top {min(10, top_n)} Features:")
        print("-" * 60)
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']:40s}: {row['importance']:.6f}")
        print()

    def create_full_report(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        sample_indices: List[int] = [0, 1, 2],
        top_features: int = 10
    ) -> None:
        """
        Create comprehensive explainability report.

        Args:
            X_test: Test samples
            sample_indices: Indices of samples to explain in detail
            top_features: Number of top features to analyze
        """
        print("=" * 80)
        print("CREATING COMPREHENSIVE EXPLAINABILITY REPORT")
        print("=" * 80 + "\n")

        # 1. SHAP global summary
        print("1. SHAP Global Summary...")
        self.setup_shap()
        self.compute_shap_values(X_test, max_samples=1000)
        self.plot_shap_summary(max_display=top_features)

        # 2. Feature importance
        print("2. Feature Importance...")
        self.plot_feature_importance(method='shap', top_n=top_features)

        # 3. Individual SHAP explanations
        print("3. Individual SHAP Explanations...")
        for idx in sample_indices:
            self.plot_shap_waterfall(idx, X_test)

        # 4. LIME explanations
        print("4. LIME Explanations...")
        for idx in sample_indices:
            self.explain_lime(idx, X_test, num_features=top_features)

        # 5. Dependence plots for top features
        print("5. SHAP Dependence Plots...")
        importance = np.abs(self.shap_values).mean(axis=0) if not isinstance(self.shap_values, list) else np.abs(self.shap_values[1]).mean(axis=0)
        top_feature_indices = np.argsort(importance)[-min(3, len(self.feature_names)):]

        for feat_idx in top_feature_indices:
            feat_name = self.feature_names[feat_idx]
            self.plot_shap_dependence(feat_name, X_test)

        print("=" * 80)
        print("✓ EXPLAINABILITY REPORT COMPLETE")
        print("=" * 80)
        print("\nGenerated files:")
        print("  - shap_summary.png")
        print("  - feature_importance_shap.png")
        for idx in sample_indices:
            print(f"  - shap_waterfall_sample_{idx}.png")
            print(f"  - lime_explanation_sample_{idx}.png")
        for feat_idx in top_feature_indices:
            print(f"  - shap_dependence_{self.feature_names[feat_idx]}.png")
        print()


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    print("Creating sample dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        random_state=42
    )

    feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training model...")
    model = XGBClassifier(random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)

    print("Creating explainer...")
    explainer = ModelExplainer(
        model=model,
        X_train=X_train,
        feature_names=feature_names,
        class_names=['Class 0', 'Class 1'],
        task='classification'
    )

    # Create full report
    explainer.create_full_report(X_test, sample_indices=[0, 1, 2], top_features=10)

    print("\n✓ Example complete!")
