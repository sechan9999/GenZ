"""
Automated ML Pipeline with FLAML

Features:
- Automated model selection and hyperparameter tuning
- Ensemble methods (stacking, blending, voting)
- Time-budgeted optimization
- Custom metric support
- Model persistence and logging
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
import warnings
import time
import joblib
from pathlib import Path

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score,
    average_precision_score
)
from sklearn.ensemble import StackingClassifier, StackingRegressor, VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

# AutoML
from flaml import AutoML

# Gradient boosting models
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

# Traditional models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

warnings.filterwarnings('ignore')


class AutoMLPipeline:
    """
    Automated ML pipeline with FLAML integration.

    Workflow:
    1. Auto feature type detection
    2. FLAML auto model selection and hyperparameter tuning
    3. Ensemble construction (stacking/voting)
    4. Model evaluation and comparison
    """

    def __init__(
        self,
        task: str = 'classification',  # 'classification' or 'regression'
        metric: Optional[str] = None,
        time_budget: int = 3600,  # seconds
        estimator_list: Optional[List[str]] = None,
        ensemble_method: str = 'stacking',  # 'stacking', 'voting', or 'none'
        n_splits: int = 5,
        random_state: int = 42,
        verbose: int = 1
    ):
        """
        Initialize AutoML pipeline.

        Args:
            task: 'classification' or 'regression'
            metric: Evaluation metric (auto-selected if None)
            time_budget: Time budget for AutoML in seconds
            estimator_list: List of estimators to try (default: ['lgbm', 'xgboost', 'rf', 'extra_tree', 'catboost'])
            ensemble_method: 'stacking', 'voting', or 'none'
            n_splits: Number of cross-validation folds
            random_state: Random seed
            verbose: Verbosity level
        """
        self.task = task
        self.time_budget = time_budget
        self.ensemble_method = ensemble_method
        self.n_splits = n_splits
        self.random_state = random_state
        self.verbose = verbose

        # Set default estimator list
        if estimator_list is None:
            self.estimator_list = ['lgbm', 'xgboost', 'rf', 'extra_tree', 'catboost']
        else:
            self.estimator_list = estimator_list

        # Set default metric
        if metric is None:
            if task == 'classification':
                self.metric = 'roc_auc'  # or 'accuracy', 'f1', 'log_loss', 'aucpr'
            else:
                self.metric = 'r2'  # or 'mse', 'mae', 'rmse'
        else:
            self.metric = metric

        # Initialize FLAML AutoML
        self.automl = AutoML()

        # Storage
        self.best_model = None
        self.ensemble_model = None
        self.base_models = {}
        self.results = {
            'automl_results': {},
            'ensemble_results': {},
            'training_time': 0
        }

        print("=" * 80)
        print("AutoML Pipeline Initialized")
        print("=" * 80)
        print(f"Task: {task}")
        print(f"Metric: {self.metric}")
        print(f"Time Budget: {time_budget}s ({time_budget/60:.1f} min)")
        print(f"Estimators: {', '.join(self.estimator_list)}")
        print(f"Ensemble Method: {ensemble_method}")
        print("=" * 80 + "\n")

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> 'AutoMLPipeline':
        """
        Fit AutoML pipeline.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional, will split from X_train if not provided)
            y_val: Validation target
            **kwargs: Additional arguments passed to FLAML

        Returns:
            self
        """
        start_time = time.time()

        # Convert to numpy if needed
        if isinstance(X_train, pd.DataFrame):
            X_train_array = X_train.values
        else:
            X_train_array = X_train

        if isinstance(y_train, pd.Series):
            y_train_array = y_train.values
        else:
            y_train_array = y_train

        print("=" * 80)
        print("STEP 1: FLAML AUTOMATED MODEL SELECTION")
        print("=" * 80)
        print(f"Training samples: {len(X_train_array):,}")
        print(f"Features: {X_train_array.shape[1]}")
        print(f"Starting AutoML optimization...")
        print("-" * 80)

        # FLAML auto model selection and HPO
        self.automl.fit(
            X_train=X_train_array,
            y_train=y_train_array,
            task=self.task,
            metric=self.metric,
            time_budget=self.time_budget,
            estimator_list=self.estimator_list,
            n_splits=self.n_splits,
            verbose=self.verbose,
            **kwargs
        )

        self.best_model = self.automl.model

        print("\n" + "=" * 80)
        print("FLAML RESULTS")
        print("=" * 80)
        print(f"Best Estimator: {self.automl.best_estimator}")
        print(f"Best Config: {self.automl.best_config}")
        print(f"Best {self.metric}: {self.automl.best_loss:.6f}")
        print(f"Training Time: {self.automl.time:.2f}s")

        # Store results
        self.results['automl_results'] = {
            'best_estimator': self.automl.best_estimator,
            'best_config': self.automl.best_config,
            'best_loss': self.automl.best_loss,
            'time': self.automl.time
        }

        # Build ensemble if requested
        if self.ensemble_method != 'none':
            print("\n" + "=" * 80)
            print(f"STEP 2: BUILDING {self.ensemble_method.upper()} ENSEMBLE")
            print("=" * 80)

            self._build_ensemble(X_train_array, y_train_array)

        self.results['training_time'] = time.time() - start_time

        print("\n" + "=" * 80)
        print("✓ TRAINING COMPLETE")
        print("=" * 80)
        print(f"Total Time: {self.results['training_time']:.2f}s")
        print("=" * 80 + "\n")

        return self

    def _build_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> None:
        """
        Build ensemble model (stacking or voting).

        Args:
            X_train: Training features
            y_train: Training target
        """
        # Train multiple base models
        print("Training base models...")

        if self.task == 'classification':
            self.base_models = {
                'lgbm': LGBMClassifier(random_state=self.random_state, verbose=-1),
                'xgboost': XGBClassifier(random_state=self.random_state, eval_metric='logloss', verbosity=0),
                'rf': RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1),
                'extra_tree': ExtraTreesClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1),
                'catboost': CatBoostClassifier(random_state=self.random_state, verbose=0)
            }
        else:
            self.base_models = {
                'lgbm': LGBMRegressor(random_state=self.random_state, verbose=-1),
                'xgboost': XGBRegressor(random_state=self.random_state, verbosity=0),
                'rf': RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1),
                'extra_tree': ExtraTreesRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1),
                'catboost': CatBoostRegressor(random_state=self.random_state, verbose=0)
            }

        # Filter to requested estimators
        self.base_models = {
            k: v for k, v in self.base_models.items()
            if k in self.estimator_list
        }

        # Train base models
        for name, model in self.base_models.items():
            print(f"  - Training {name}...")
            model.fit(X_train, y_train)

        # Build ensemble
        estimators = [(name, model) for name, model in self.base_models.items()]

        if self.ensemble_method == 'stacking':
            print("\nBuilding stacking ensemble...")

            if self.task == 'classification':
                meta_learner = LogisticRegression(random_state=self.random_state, max_iter=1000)
                self.ensemble_model = StackingClassifier(
                    estimators=estimators,
                    final_estimator=meta_learner,
                    cv=self.n_splits,
                    n_jobs=-1,
                    verbose=0
                )
            else:
                meta_learner = Ridge(random_state=self.random_state)
                self.ensemble_model = StackingRegressor(
                    estimators=estimators,
                    final_estimator=meta_learner,
                    cv=self.n_splits,
                    n_jobs=-1,
                    verbose=0
                )

            print("Training stacking meta-learner...")
            self.ensemble_model.fit(X_train, y_train)

        elif self.ensemble_method == 'voting':
            print("\nBuilding voting ensemble...")

            if self.task == 'classification':
                self.ensemble_model = VotingClassifier(
                    estimators=estimators,
                    voting='soft',  # Use predicted probabilities
                    n_jobs=-1
                )
            else:
                self.ensemble_model = VotingRegressor(
                    estimators=estimators,
                    n_jobs=-1
                )

            print("Training voting ensemble...")
            self.ensemble_model.fit(X_train, y_train)

        print("✓ Ensemble model created\n")

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        use_ensemble: bool = True
    ) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features
            use_ensemble: Use ensemble model if available

        Returns:
            Predictions
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        if use_ensemble and self.ensemble_model is not None:
            return self.ensemble_model.predict(X)
        else:
            return self.automl.predict(X)

    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        use_ensemble: bool = True
    ) -> np.ndarray:
        """
        Predict class probabilities (classification only).

        Args:
            X: Features
            use_ensemble: Use ensemble model if available

        Returns:
            Class probabilities
        """
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")

        if isinstance(X, pd.DataFrame):
            X = X.values

        if use_ensemble and self.ensemble_model is not None:
            return self.ensemble_model.predict_proba(X)
        else:
            return self.automl.predict_proba(X)

    def evaluate(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        use_ensemble: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            X_test: Test features
            y_test: Test target
            use_ensemble: Use ensemble model if available

        Returns:
            Dictionary of evaluation metrics
        """
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        if isinstance(y_test, pd.Series):
            y_test = y_test.values

        print("=" * 80)
        print("MODEL EVALUATION")
        print("=" * 80)

        results = {}

        # Evaluate AutoML model
        print("\n1. FLAML Best Model:")
        print("-" * 80)
        y_pred_automl = self.automl.predict(X_test)

        if self.task == 'classification':
            y_pred_proba_automl = self.automl.predict_proba(X_test)

            results['automl'] = {
                'accuracy': accuracy_score(y_test, y_pred_automl),
                'precision': precision_score(y_test, y_pred_automl, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred_automl, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred_automl, average='weighted', zero_division=0),
            }

            # Add AUC if binary classification
            if len(np.unique(y_test)) == 2:
                results['automl']['roc_auc'] = roc_auc_score(y_test, y_pred_proba_automl[:, 1])
                results['automl']['aucpr'] = average_precision_score(y_test, y_pred_proba_automl[:, 1])

            for metric, value in results['automl'].items():
                print(f"   {metric:12s}: {value:.6f}")

        else:  # regression
            results['automl'] = {
                'mse': mean_squared_error(y_test, y_pred_automl),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_automl)),
                'mae': mean_absolute_error(y_test, y_pred_automl),
                'r2': r2_score(y_test, y_pred_automl)
            }

            for metric, value in results['automl'].items():
                print(f"   {metric:12s}: {value:.6f}")

        # Evaluate ensemble if available
        if use_ensemble and self.ensemble_model is not None:
            print(f"\n2. {self.ensemble_method.capitalize()} Ensemble:")
            print("-" * 80)

            y_pred_ensemble = self.ensemble_model.predict(X_test)

            if self.task == 'classification':
                y_pred_proba_ensemble = self.ensemble_model.predict_proba(X_test)

                results['ensemble'] = {
                    'accuracy': accuracy_score(y_test, y_pred_ensemble),
                    'precision': precision_score(y_test, y_pred_ensemble, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred_ensemble, average='weighted', zero_division=0),
                    'f1': f1_score(y_test, y_pred_ensemble, average='weighted', zero_division=0),
                }

                if len(np.unique(y_test)) == 2:
                    results['ensemble']['roc_auc'] = roc_auc_score(y_test, y_pred_proba_ensemble[:, 1])
                    results['ensemble']['aucpr'] = average_precision_score(y_test, y_pred_proba_ensemble[:, 1])

                for metric, value in results['ensemble'].items():
                    print(f"   {metric:12s}: {value:.6f}")

            else:
                results['ensemble'] = {
                    'mse': mean_squared_error(y_test, y_pred_ensemble),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_ensemble)),
                    'mae': mean_absolute_error(y_test, y_pred_ensemble),
                    'r2': r2_score(y_test, y_pred_ensemble)
                }

                for metric, value in results['ensemble'].items():
                    print(f"   {metric:12s}: {value:.6f}")

        print("\n" + "=" * 80)

        self.results['test_results'] = results
        return results

    def save(self, path: str, save_ensemble: bool = True) -> None:
        """
        Save model to disk.

        Args:
            path: Save path (directory)
            save_ensemble: Save ensemble model if available
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FLAML model
        automl_path = path / 'automl_model.pkl'
        joblib.dump(self.automl, automl_path)
        print(f"✓ AutoML model saved to: {automl_path}")

        # Save ensemble if available
        if save_ensemble and self.ensemble_model is not None:
            ensemble_path = path / 'ensemble_model.pkl'
            joblib.dump(self.ensemble_model, ensemble_path)
            print(f"✓ Ensemble model saved to: {ensemble_path}")

        # Save results
        results_path = path / 'results.pkl'
        joblib.dump(self.results, results_path)
        print(f"✓ Results saved to: {results_path}")

    @classmethod
    def load(cls, path: str, load_ensemble: bool = True) -> 'AutoMLPipeline':
        """
        Load model from disk.

        Args:
            path: Load path (directory)
            load_ensemble: Load ensemble model if available

        Returns:
            AutoMLPipeline instance
        """
        path = Path(path)

        # Load AutoML model
        automl_path = path / 'automl_model.pkl'
        automl = joblib.load(automl_path)

        # Create instance
        pipeline = cls(task=automl.task)
        pipeline.automl = automl
        pipeline.best_model = automl.model

        # Load ensemble if available
        if load_ensemble:
            ensemble_path = path / 'ensemble_model.pkl'
            if ensemble_path.exists():
                pipeline.ensemble_model = joblib.load(ensemble_path)

        # Load results
        results_path = path / 'results.pkl'
        if results_path.exists():
            pipeline.results = joblib.load(results_path)

        print(f"✓ Model loaded from: {path}")
        return pipeline


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification

    print("Creating sample dataset...")
    X, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

    # Initialize and train AutoML pipeline
    pipeline = AutoMLPipeline(
        task='classification',
        metric='roc_auc',
        time_budget=60,  # 1 minute for demo
        estimator_list=['lgbm', 'xgboost', 'rf', 'extra_tree'],
        ensemble_method='stacking',
        verbose=0
    )

    pipeline.fit(X_train, y_train)

    # Evaluate
    results = pipeline.evaluate(X_test, y_test)

    # Save
    pipeline.save('saved_models/')

    print("\n✓ Example complete!")
