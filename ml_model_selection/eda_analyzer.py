"""
EDA Analyzer for ML Model Selection

Provides comprehensive exploratory data analysis including:
- Linearity checks (residual plots, correlation analysis)
- Interaction plots (2-way and 3-way)
- Missingness heatmaps and patterns
- Distribution analysis
- Multicollinearity detection (VIF)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, normaltest
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from typing import List, Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class EDAAnalyzer:
    """Automated EDA for model selection decision-making."""

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        task_type: str = 'auto',  # 'regression', 'classification', or 'auto'
        figsize: Tuple[int, int] = (15, 10)
    ):
        """
        Initialize EDA Analyzer.

        Args:
            df: Input dataframe
            target: Target variable column name
            task_type: Type of ML task ('regression', 'classification', 'auto')
            figsize: Default figure size for plots
        """
        self.df = df.copy()
        self.target = target
        self.figsize = figsize

        # Auto-detect task type
        if task_type == 'auto':
            self.task_type = self._detect_task_type()
        else:
            self.task_type = task_type

        # Separate features and target
        self.X = self.df.drop(columns=[target])
        self.y = self.df[target]

        # Identify feature types
        self.numeric_features = self.X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = self.X.select_dtypes(exclude=[np.number]).columns.tolist()

        # Results storage
        self.results = {
            'linearity': {},
            'interactions': {},
            'missingness': {},
            'distributions': {},
            'multicollinearity': {},
            'recommendations': []
        }

    def _detect_task_type(self) -> str:
        """Auto-detect if regression or classification task."""
        nunique = self.df[self.target].nunique()
        if nunique <= 10:
            return 'classification'
        else:
            return 'regression'

    def check_linearity(self, features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Check linearity assumptions for features.

        Creates:
        - Scatter plots with regression lines
        - Residual plots
        - Correlation matrix
        - Statistical tests (Pearson correlation, p-values)

        Args:
            features: List of features to check (default: all numeric features)

        Returns:
            Dictionary with linearity metrics and recommendations
        """
        if features is None:
            features = self.numeric_features

        print("=" * 80)
        print("LINEARITY ANALYSIS")
        print("=" * 80)

        # Correlation analysis
        correlations = {}
        for feat in features:
            if self.task_type == 'regression':
                corr, pval = stats.pearsonr(self.X[feat].dropna(),
                                           self.y[self.X[feat].notna()])
            else:
                # Point-biserial correlation for binary classification
                corr, pval = stats.pointbiserialr(self.y[self.X[feat].notna()],
                                                 self.X[feat].dropna())

            correlations[feat] = {
                'correlation': corr,
                'p_value': pval,
                'is_linear': abs(corr) > 0.3 and pval < 0.05
            }

        # Sort by absolute correlation
        sorted_corr = sorted(correlations.items(),
                           key=lambda x: abs(x[1]['correlation']),
                           reverse=True)

        print(f"\nTop 10 Features by Correlation with {self.target}:")
        print("-" * 60)
        for feat, metrics in sorted_corr[:10]:
            print(f"{feat:30s} | r={metrics['correlation']:7.4f} | p={metrics['p_value']:.4e} | Linear: {metrics['is_linear']}")

        # Visualize top features
        n_plots = min(6, len(features))
        top_features = [f[0] for f in sorted_corr[:n_plots]]

        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        axes = axes.flatten()

        for idx, feat in enumerate(top_features):
            ax = axes[idx]

            # Scatter plot with regression line
            ax.scatter(self.X[feat], self.y, alpha=0.5, s=20)

            # Fit regression line
            X_feat = self.X[feat].values.reshape(-1, 1)
            mask = ~np.isnan(X_feat.flatten()) & ~np.isnan(self.y)
            if mask.sum() > 0:
                lr = LinearRegression()
                lr.fit(X_feat[mask], self.y[mask])
                x_line = np.linspace(X_feat[mask].min(), X_feat[mask].max(), 100)
                y_line = lr.predict(x_line.reshape(-1, 1))
                ax.plot(x_line, y_line, 'r--', linewidth=2)

            ax.set_xlabel(feat)
            ax.set_ylabel(self.target)
            ax.set_title(f"{feat}\nr={correlations[feat]['correlation']:.3f}, p={correlations[feat]['p_value']:.3e}")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('linearity_check.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Linearity plots saved to: linearity_check.png")
        plt.close()

        # Residual plots for top features (regression only)
        if self.task_type == 'regression':
            fig, axes = plt.subplots(2, 3, figsize=self.figsize)
            axes = axes.flatten()

            for idx, feat in enumerate(top_features):
                ax = axes[idx]

                X_feat = self.X[feat].values.reshape(-1, 1)
                mask = ~np.isnan(X_feat.flatten()) & ~np.isnan(self.y)

                if mask.sum() > 0:
                    lr = LinearRegression()
                    lr.fit(X_feat[mask], self.y[mask])
                    y_pred = lr.predict(X_feat[mask])
                    residuals = self.y[mask] - y_pred

                    ax.scatter(y_pred, residuals, alpha=0.5, s=20)
                    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
                    ax.set_xlabel('Fitted values')
                    ax.set_ylabel('Residuals')
                    ax.set_title(f"{feat} - Residual Plot")
                    ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('residual_plots.png', dpi=300, bbox_inches='tight')
            print(f"✓ Residual plots saved to: residual_plots.png")
            plt.close()

        # Store results
        self.results['linearity'] = correlations

        # Recommendations
        linear_features = [f for f, m in correlations.items() if m['is_linear']]
        nonlinear_features = [f for f, m in correlations.items() if not m['is_linear']]

        print(f"\n📊 Summary:")
        print(f"  - Linear features: {len(linear_features)}/{len(features)}")
        print(f"  - Non-linear features: {len(nonlinear_features)}/{len(features)}")

        if len(nonlinear_features) > len(linear_features):
            self.results['recommendations'].append(
                "⚠️  Many non-linear relationships detected → Consider tree-based models (XGBoost, RandomForest)"
            )
        else:
            self.results['recommendations'].append(
                "✓ Many linear relationships detected → Linear models (Logistic/Linear Regression) may perform well"
            )

        return correlations

    def plot_interactions(
        self,
        top_n: int = 5,
        interaction_type: str = 'pairwise'  # 'pairwise' or 'threeway'
    ) -> None:
        """
        Create interaction plots for top features.

        Args:
            top_n: Number of top features to analyze
            interaction_type: 'pairwise' for 2-way, 'threeway' for 3-way interactions
        """
        print("\n" + "=" * 80)
        print("INTERACTION ANALYSIS")
        print("=" * 80)

        # Get top features by correlation
        if not self.results['linearity']:
            self.check_linearity()

        sorted_features = sorted(
            self.results['linearity'].items(),
            key=lambda x: abs(x[1]['correlation']),
            reverse=True
        )
        top_features = [f[0] for f in sorted_features[:top_n]]

        if interaction_type == 'pairwise':
            # 2-way interactions
            n_pairs = min(6, len(top_features) * (len(top_features) - 1) // 2)
            fig, axes = plt.subplots(2, 3, figsize=self.figsize)
            axes = axes.flatten()

            pair_idx = 0
            for i in range(len(top_features)):
                for j in range(i + 1, len(top_features)):
                    if pair_idx >= 6:
                        break

                    feat1, feat2 = top_features[i], top_features[j]
                    ax = axes[pair_idx]

                    # Create interaction bins
                    X1_binned = pd.qcut(self.X[feat1], q=3, labels=['Low', 'Med', 'High'], duplicates='drop')

                    for bin_label in X1_binned.unique():
                        mask = X1_binned == bin_label
                        ax.scatter(
                            self.X.loc[mask, feat2],
                            self.y[mask],
                            label=f"{feat1}={bin_label}",
                            alpha=0.6,
                            s=30
                        )

                    ax.set_xlabel(feat2)
                    ax.set_ylabel(self.target)
                    ax.set_title(f"Interaction: {feat1} × {feat2}")
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)

                    pair_idx += 1

                if pair_idx >= 6:
                    break

            plt.tight_layout()
            plt.savefig('interaction_plots_2way.png', dpi=300, bbox_inches='tight')
            print(f"✓ 2-way interaction plots saved to: interaction_plots_2way.png")
            plt.close()

        else:  # threeway
            # 3-way interaction with color and size
            if len(top_features) >= 3:
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))

                feat1, feat2, feat3 = top_features[:3]

                # Plot 1: Color by third variable
                ax = axes[0]
                scatter = ax.scatter(
                    self.X[feat1],
                    self.X[feat2],
                    c=self.X[feat3],
                    s=50,
                    alpha=0.6,
                    cmap='viridis'
                )
                ax.set_xlabel(feat1)
                ax.set_ylabel(feat2)
                ax.set_title(f"3-way Interaction (color={feat3})")
                plt.colorbar(scatter, ax=ax, label=feat3)
                ax.grid(True, alpha=0.3)

                # Plot 2: Size by target
                ax = axes[1]
                if self.task_type == 'regression':
                    sizes = (self.y - self.y.min()) / (self.y.max() - self.y.min()) * 200 + 20
                else:
                    sizes = self.y * 100 + 20

                scatter = ax.scatter(
                    self.X[feat1],
                    self.X[feat2],
                    c=self.X[feat3],
                    s=sizes,
                    alpha=0.6,
                    cmap='plasma'
                )
                ax.set_xlabel(feat1)
                ax.set_ylabel(feat2)
                ax.set_title(f"3-way Interaction (size={self.target})")
                plt.colorbar(scatter, ax=ax, label=feat3)
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig('interaction_plots_3way.png', dpi=300, bbox_inches='tight')
                print(f"✓ 3-way interaction plots saved to: interaction_plots_3way.png")
                plt.close()

        self.results['recommendations'].append(
            "✓ Interaction plots created → Review for non-additive effects (indicates need for interaction terms or tree models)"
        )

    def plot_missingness_heatmap(self) -> Dict[str, Any]:
        """
        Create missingness heatmap and analyze missing data patterns.

        Returns:
            Dictionary with missingness statistics
        """
        print("\n" + "=" * 80)
        print("MISSINGNESS ANALYSIS")
        print("=" * 80)

        # Calculate missingness
        missing_counts = self.df.isnull().sum()
        missing_pct = (missing_counts / len(self.df) * 100).round(2)
        missing_df = pd.DataFrame({
            'count': missing_counts,
            'percent': missing_pct
        }).sort_values('percent', ascending=False)

        # Filter to features with missing values
        missing_features = missing_df[missing_df['count'] > 0]

        if len(missing_features) == 0:
            print("✓ No missing values detected!")
            self.results['missingness'] = {'has_missing': False}
            return {'has_missing': False}

        print(f"\nFeatures with Missing Values ({len(missing_features)}/{len(self.df.columns)}):")
        print("-" * 60)
        print(missing_features.to_string())

        # Heatmap of missingness
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Plot 1: Missingness heatmap
        ax = axes[0]
        missing_matrix = self.df[missing_features.index].isnull().astype(int)

        if len(missing_matrix) > 1000:
            # Sample for visualization if too large
            missing_matrix = missing_matrix.sample(1000, random_state=42)

        sns.heatmap(
            missing_matrix.T,
            cmap='YlOrRd',
            cbar_kws={'label': 'Missing (1) vs Present (0)'},
            ax=ax,
            yticklabels=True
        )
        ax.set_title('Missingness Heatmap')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Features')

        # Plot 2: Missingness correlation
        ax = axes[1]
        missing_corr = missing_matrix.corr()
        sns.heatmap(
            missing_corr,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            ax=ax,
            square=True
        )
        ax.set_title('Missingness Pattern Correlation\n(High values = features missing together)')

        plt.tight_layout()
        plt.savefig('missingness_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Missingness heatmap saved to: missingness_heatmap.png")
        plt.close()

        # Missingness correlation with target
        print("\nMissingness Correlation with Target:")
        print("-" * 60)

        missing_target_corr = {}
        for col in missing_features.index:
            is_missing = self.df[col].isnull().astype(int)
            if self.task_type == 'classification':
                corr, pval = stats.pointbiserialr(self.y, is_missing)
            else:
                # Use t-test to check if target differs when feature is missing
                present = self.y[~self.df[col].isnull()]
                missing = self.y[self.df[col].isnull()]
                if len(missing) > 0 and len(present) > 0:
                    tstat, pval = stats.ttest_ind(present, missing)
                    corr = tstat  # Use t-statistic as proxy
                else:
                    corr, pval = 0, 1

            missing_target_corr[col] = {'correlation': corr, 'p_value': pval}
            print(f"{col:30s} | corr={corr:7.4f} | p={pval:.4e}")

        # Store results
        self.results['missingness'] = {
            'has_missing': True,
            'summary': missing_df,
            'target_correlation': missing_target_corr
        }

        # Recommendations
        high_missing = missing_features[missing_features['percent'] > 20]
        if len(high_missing) > 0:
            self.results['recommendations'].append(
                f"⚠️  {len(high_missing)} features have >20% missing values → Consider imputation strategies or tree-based models (handle missing natively)"
            )

        # Check for MCAR (Missing Completely At Random)
        significant_corr = [k for k, v in missing_target_corr.items() if v['p_value'] < 0.05]
        if len(significant_corr) > 0:
            self.results['recommendations'].append(
                f"⚠️  Missingness is NOT random for {len(significant_corr)} features → Missing values are informative (MAR/MNAR)"
            )
        else:
            self.results['recommendations'].append(
                "✓ Missingness appears random (MCAR) → Simple imputation may suffice"
            )

        return self.results['missingness']

    def check_multicollinearity(self, threshold: float = 10.0) -> Dict[str, float]:
        """
        Check multicollinearity using Variance Inflation Factor (VIF).

        Args:
            threshold: VIF threshold (default: 10.0, values >10 indicate high multicollinearity)

        Returns:
            Dictionary with VIF scores for each feature
        """
        print("\n" + "=" * 80)
        print("MULTICOLLINEARITY ANALYSIS (VIF)")
        print("=" * 80)

        if len(self.numeric_features) < 2:
            print("⚠️  Need at least 2 numeric features for VIF calculation")
            return {}

        # Prepare data (drop missing values for VIF calculation)
        X_numeric = self.X[self.numeric_features].dropna()

        if len(X_numeric) == 0:
            print("⚠️  No complete cases for VIF calculation")
            return {}

        # Calculate VIF
        vif_data = {}
        for i, col in enumerate(X_numeric.columns):
            try:
                vif = variance_inflation_factor(X_numeric.values, i)
                vif_data[col] = vif
            except:
                vif_data[col] = np.nan

        # Sort by VIF
        vif_sorted = sorted(vif_data.items(), key=lambda x: x[1] if not np.isnan(x[1]) else 0, reverse=True)

        print(f"\nVIF Scores (threshold={threshold}):")
        print("-" * 60)
        print(f"{'Feature':<30s} | VIF Score | Interpretation")
        print("-" * 60)

        for feat, vif in vif_sorted:
            if np.isnan(vif):
                status = "ERROR"
            elif vif > threshold:
                status = "HIGH (multicollinear)"
            elif vif > 5:
                status = "MODERATE"
            else:
                status = "LOW (ok)"

            print(f"{feat:<30s} | {vif:8.2f}  | {status}")

        # Store results
        self.results['multicollinearity'] = vif_data

        # Recommendations
        high_vif = [f for f, v in vif_data.items() if not np.isnan(v) and v > threshold]
        if len(high_vif) > 0:
            self.results['recommendations'].append(
                f"⚠️  {len(high_vif)} features have VIF > {threshold} → Consider feature selection, PCA, or regularization (Ridge/Lasso)"
            )
        else:
            self.results['recommendations'].append(
                f"✓ No severe multicollinearity detected (all VIF < {threshold})"
            )

        return vif_data

    def analyze_distributions(self, top_n: int = 12) -> None:
        """
        Analyze and visualize feature distributions.

        Args:
            top_n: Number of features to visualize
        """
        print("\n" + "=" * 80)
        print("DISTRIBUTION ANALYSIS")
        print("=" * 80)

        # Get top features
        if not self.results['linearity']:
            self.check_linearity()

        sorted_features = sorted(
            self.results['linearity'].items(),
            key=lambda x: abs(x[1]['correlation']),
            reverse=True
        )
        top_features = [f[0] for f in sorted_features[:top_n]]

        # Create distribution plots
        n_cols = 3
        n_rows = (len(top_features) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else []

        for idx, feat in enumerate(top_features):
            ax = axes[idx]
            data = self.X[feat].dropna()

            # Histogram with KDE
            ax.hist(data, bins=50, alpha=0.6, density=True, edgecolor='black')

            # KDE overlay
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 100)
                ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            except:
                pass

            # Normality test
            if len(data) >= 20:
                _, p_shapiro = shapiro(data[:5000])  # Shapiro-Wilk (max 5000 samples)
                _, p_normal = normaltest(data)  # D'Agostino-Pearson

                normality_text = f"Shapiro p={p_shapiro:.3e}\nNormal p={p_normal:.3e}"
                ax.text(0.02, 0.98, normality_text, transform=ax.transAxes,
                       verticalalignment='top', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_xlabel(feat)
            ax.set_ylabel('Density')
            ax.set_title(f"{feat}\nSkew={stats.skew(data):.2f}, Kurt={stats.kurtosis(data):.2f}")
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Hide empty subplots
        for idx in range(len(top_features), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig('distribution_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✓ Distribution plots saved to: distribution_analysis.png")
        plt.close()

        # Check for skewness
        skewed_features = []
        for feat in self.numeric_features:
            skew = stats.skew(self.X[feat].dropna())
            if abs(skew) > 1:
                skewed_features.append((feat, skew))

        if skewed_features:
            print(f"\nHighly Skewed Features (|skew| > 1): {len(skewed_features)}")
            for feat, skew in sorted(skewed_features, key=lambda x: abs(x[1]), reverse=True)[:10]:
                print(f"  - {feat}: skew={skew:.2f}")

            self.results['recommendations'].append(
                f"⚠️  {len(skewed_features)} features are highly skewed → Consider log/sqrt transformations or tree-based models"
            )

    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run complete EDA pipeline.

        Returns:
            Dictionary with all analysis results
        """
        print("\n" + "=" * 80)
        print(f"COMPREHENSIVE EDA FOR MODEL SELECTION")
        print(f"Task Type: {self.task_type.upper()}")
        print(f"Target: {self.target}")
        print(f"Samples: {len(self.df):,}")
        print(f"Features: {len(self.X.columns)} ({len(self.numeric_features)} numeric, {len(self.categorical_features)} categorical)")
        print("=" * 80)

        # Run all analyses
        self.check_linearity()
        self.plot_interactions(top_n=min(5, len(self.numeric_features)))
        self.plot_missingness_heatmap()

        if len(self.numeric_features) >= 2:
            self.check_multicollinearity()

        self.analyze_distributions(top_n=min(12, len(self.numeric_features)))

        # Print recommendations
        print("\n" + "=" * 80)
        print("MODEL SELECTION RECOMMENDATIONS")
        print("=" * 80)
        for i, rec in enumerate(self.results['recommendations'], 1):
            print(f"{i}. {rec}")

        print("\n" + "=" * 80)
        print("EDA COMPLETE ✓")
        print("=" * 80)

        return self.results


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification, make_regression

    print("Creating sample dataset...")

    # Classification example
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42
    )

    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y

    # Add some missing values
    np.random.seed(42)
    for col in df.columns[:5]:
        missing_idx = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
        df.loc[missing_idx, col] = np.nan

    # Run EDA
    eda = EDAAnalyzer(df, target='target', task_type='classification')
    results = eda.run_full_analysis()

    print("\n✓ Example EDA complete! Check generated PNG files.")
