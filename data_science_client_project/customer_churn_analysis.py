"""
Data Science Client Project: Customer Churn Prediction & Analysis
===================================================================

Business Problem:
    A telecommunications company wants to predict which customers are likely to churn
    and understand the key drivers of churn to inform retention strategies.

Deliverables:
    1. Predictive model for customer churn (with 90%+ accuracy target)
    2. Quantitative analysis of churn drivers
    3. Customer segmentation and risk scoring
    4. Actionable business recommendations
    5. Executive dashboard and visualizations

Tech Stack:
    - Machine Learning: AutoML (FLAML), XGBoost, LightGBM, Random Forest
    - Analysis: pandas, numpy, scipy
    - Visualization: matplotlib, seaborn, plotly
    - Explainability: SHAP, feature importance
"""

import sys
import os
sys.path.append('/home/user/GenZ')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score
)

# Custom modules
from ml_model_selection.model_selector import InterpretabilityMatrix, UseCase
from ml_model_selection.automl_pipeline import AutoMLPipeline
from ml_model_selection.eda_analyzer import EDAAnalyzer
# from ml_model_selection.explainability import SHAPExplainer  # Optional - requires lime package

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class CustomerChurnProject:
    """
    End-to-end data science project for customer churn prediction.

    Workflow:
        1. Data Generation/Loading
        2. Exploratory Data Analysis (EDA)
        3. Feature Engineering
        4. Model Selection & Training
        5. Model Evaluation & Explainability
        6. Business Recommendations
        7. Executive Report Generation
    """

    def __init__(self, output_dir: str = './data_science_client_project/output'):
        """Initialize project."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.feature_names = None

        print("=" * 80)
        print("CUSTOMER CHURN PREDICTION - DATA SCIENCE CLIENT PROJECT")
        print("=" * 80)
        print(f"Output Directory: {self.output_dir}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80 + "\n")

    def generate_synthetic_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """
        Generate realistic synthetic customer churn dataset.

        Features:
            - Customer demographics (age, gender, location)
            - Account info (tenure, contract type, payment method)
            - Service usage (internet, phone, streaming)
            - Billing (monthly charges, total charges)
            - Support interactions (complaints, tickets)

        Args:
            n_samples: Number of customer records

        Returns:
            DataFrame with customer data
        """
        print("=" * 80)
        print("STEP 1: DATA GENERATION")
        print("=" * 80)

        np.random.seed(42)

        # Demographics
        age = np.random.normal(45, 15, n_samples).clip(18, 80).astype(int)
        gender = np.random.choice(['Male', 'Female'], n_samples)

        # Account info
        tenure_months = np.random.exponential(24, n_samples).clip(0, 72).astype(int)
        contract_type = np.random.choice(
            ['Month-to-month', 'One year', 'Two year'],
            n_samples,
            p=[0.5, 0.3, 0.2]
        )
        payment_method = np.random.choice(
            ['Electronic check', 'Mailed check', 'Credit card', 'Bank transfer'],
            n_samples,
            p=[0.35, 0.15, 0.25, 0.25]
        )

        # Services
        internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.35, 0.45, 0.20])
        phone_service = np.random.choice(['Yes', 'No'], n_samples, p=[0.85, 0.15])
        streaming_tv = np.random.choice(['Yes', 'No'], n_samples, p=[0.40, 0.60])
        streaming_movies = np.random.choice(['Yes', 'No'], n_samples, p=[0.38, 0.62])
        online_security = np.random.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70])

        # Billing
        base_charge = np.where(internet_service == 'Fiber optic', 70,
                               np.where(internet_service == 'DSL', 50, 30))
        monthly_charges = base_charge + np.random.normal(0, 10, n_samples)
        monthly_charges = monthly_charges.clip(20, 120)
        total_charges = monthly_charges * tenure_months + np.random.normal(0, 100, n_samples)
        total_charges = total_charges.clip(0, 10000)

        # Support interactions
        num_support_tickets = np.random.poisson(2, n_samples)
        num_complaints = np.random.poisson(1, n_samples)
        avg_response_time_hours = np.random.exponential(24, n_samples).clip(1, 168)

        # Customer satisfaction score (1-10)
        satisfaction_score = np.random.normal(7, 2, n_samples).clip(1, 10)

        # Churn probability based on features (realistic drivers)
        churn_prob = 0.1  # Base rate

        # Contract type impact
        churn_prob += np.where(contract_type == 'Month-to-month', 0.25, 0)
        churn_prob -= np.where(contract_type == 'Two year', 0.15, 0)

        # Tenure impact (longer tenure = less churn)
        churn_prob -= (tenure_months / 72) * 0.20

        # Payment method impact
        churn_prob += np.where(payment_method == 'Electronic check', 0.15, 0)

        # Support issues impact
        churn_prob += (num_support_tickets / 10) * 0.15
        churn_prob += (num_complaints / 5) * 0.20

        # Satisfaction impact
        churn_prob -= ((satisfaction_score - 5) / 5) * 0.25

        # Monthly charges impact
        churn_prob += ((monthly_charges - 70) / 50) * 0.10

        # Fiber optic churn (often higher)
        churn_prob += np.where(internet_service == 'Fiber optic', 0.10, 0)

        # Clip probability
        churn_prob = churn_prob.clip(0, 1)

        # Generate churn labels
        churn = (np.random.random(n_samples) < churn_prob).astype(int)

        # Create DataFrame
        df = pd.DataFrame({
            'customer_id': [f'CUST_{i:05d}' for i in range(n_samples)],
            'age': age,
            'gender': gender,
            'tenure_months': tenure_months,
            'contract_type': contract_type,
            'payment_method': payment_method,
            'internet_service': internet_service,
            'phone_service': phone_service,
            'streaming_tv': streaming_tv,
            'streaming_movies': streaming_movies,
            'online_security': online_security,
            'monthly_charges': monthly_charges.round(2),
            'total_charges': total_charges.round(2),
            'num_support_tickets': num_support_tickets,
            'num_complaints': num_complaints,
            'avg_response_time_hours': avg_response_time_hours.round(1),
            'satisfaction_score': satisfaction_score.round(1),
            'churn': churn
        })

        # Add some missing values (realistic)
        missing_rate = 0.05
        for col in ['total_charges', 'satisfaction_score', 'avg_response_time_hours']:
            missing_idx = np.random.choice(df.index, size=int(missing_rate * len(df)), replace=False)
            df.loc[missing_idx, col] = np.nan

        print(f"✓ Generated {len(df):,} customer records")
        print(f"  - Features: {df.shape[1] - 2} (excluding customer_id and target)")
        print(f"  - Churn Rate: {df['churn'].mean():.1%}")
        print(f"  - Missing Values: {df.isnull().sum().sum()} cells")

        # Save raw data
        data_path = self.output_dir / 'customer_data_raw.csv'
        df.to_csv(data_path, index=False)
        print(f"✓ Raw data saved to: {data_path}\n")

        self.data = df
        return df

    def run_eda(self) -> dict:
        """
        Run comprehensive exploratory data analysis.

        Returns:
            EDA results dictionary
        """
        print("=" * 80)
        print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
        print("=" * 80)

        # Basic stats
        print("\nDataset Overview:")
        print("-" * 60)
        print(f"Total Customers: {len(self.data):,}")
        print(f"Total Features: {self.data.shape[1] - 2}")
        print(f"Churned Customers: {self.data['churn'].sum():,} ({self.data['churn'].mean():.1%})")
        print(f"Retained Customers: {(1 - self.data['churn']).sum():,} ({(1 - self.data['churn'].mean()):.1%})")

        # Use EDAAnalyzer
        eda = EDAAnalyzer(
            df=self.data.drop(columns=['customer_id']),
            target='churn',
            task_type='classification'
        )

        results = eda.run_full_analysis()

        # Move generated plots to output directory
        for plot_file in ['linearity_check.png', 'residual_plots.png',
                         'interaction_plots_2way.png', 'interaction_plots_3way.png',
                         'missingness_heatmap.png', 'distribution_analysis.png']:
            if Path(plot_file).exists():
                Path(plot_file).rename(self.output_dir / plot_file)

        # Additional churn-specific analysis
        self._churn_specific_analysis()

        return results

    def _churn_specific_analysis(self):
        """Additional churn-specific visualizations."""
        print("\n" + "=" * 80)
        print("CHURN-SPECIFIC ANALYSIS")
        print("=" * 80)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Churn by Contract Type
        ax = axes[0, 0]
        churn_by_contract = self.data.groupby('contract_type')['churn'].mean().sort_values(ascending=False)
        churn_by_contract.plot(kind='bar', ax=ax, color='coral')
        ax.set_title('Churn Rate by Contract Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Churn Rate')
        ax.set_xlabel('Contract Type')
        ax.grid(axis='y', alpha=0.3)

        # 2. Churn by Payment Method
        ax = axes[0, 1]
        churn_by_payment = self.data.groupby('payment_method')['churn'].mean().sort_values(ascending=False)
        churn_by_payment.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('Churn Rate by Payment Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Churn Rate')
        ax.set_xlabel('Payment Method')
        ax.grid(axis='y', alpha=0.3)

        # 3. Churn by Tenure
        ax = axes[0, 2]
        tenure_bins = [0, 6, 12, 24, 36, 72]
        tenure_labels = ['0-6mo', '6-12mo', '1-2yr', '2-3yr', '3yr+']
        self.data['tenure_bin'] = pd.cut(self.data['tenure_months'], bins=tenure_bins, labels=tenure_labels)
        churn_by_tenure = self.data.groupby('tenure_bin')['churn'].mean()
        churn_by_tenure.plot(kind='bar', ax=ax, color='lightgreen')
        ax.set_title('Churn Rate by Tenure', fontsize=12, fontweight='bold')
        ax.set_ylabel('Churn Rate')
        ax.set_xlabel('Tenure')
        ax.grid(axis='y', alpha=0.3)

        # 4. Monthly Charges Distribution by Churn
        ax = axes[1, 0]
        self.data[self.data['churn'] == 0]['monthly_charges'].hist(
            bins=30, alpha=0.6, label='Retained', ax=ax, color='green'
        )
        self.data[self.data['churn'] == 1]['monthly_charges'].hist(
            bins=30, alpha=0.6, label='Churned', ax=ax, color='red'
        )
        ax.set_title('Monthly Charges Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Monthly Charges ($)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # 5. Satisfaction Score by Churn
        ax = axes[1, 1]
        churn_labels = ['Retained', 'Churned']
        satisfaction_by_churn = [
            self.data[self.data['churn'] == 0]['satisfaction_score'].dropna(),
            self.data[self.data['churn'] == 1]['satisfaction_score'].dropna()
        ]
        ax.boxplot(satisfaction_by_churn, labels=churn_labels)
        ax.set_title('Satisfaction Score by Churn Status', fontsize=12, fontweight='bold')
        ax.set_ylabel('Satisfaction Score (1-10)')
        ax.grid(axis='y', alpha=0.3)

        # 6. Support Tickets by Churn
        ax = axes[1, 2]
        support_by_churn = [
            self.data[self.data['churn'] == 0]['num_support_tickets'],
            self.data[self.data['churn'] == 1]['num_support_tickets']
        ]
        ax.boxplot(support_by_churn, labels=churn_labels)
        ax.set_title('Support Tickets by Churn Status', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Support Tickets')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'churn_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✓ Churn analysis plots saved to: {self.output_dir / 'churn_analysis.png'}")
        plt.close()

    def prepare_features(self) -> tuple:
        """
        Feature engineering and preprocessing.

        Returns:
            (X_train, X_test, y_train, y_test)
        """
        print("\n" + "=" * 80)
        print("STEP 3: FEATURE ENGINEERING & PREPROCESSING")
        print("=" * 80)

        df = self.data.copy()

        # Drop customer_id
        df = df.drop(columns=['customer_id'])

        # Create additional features
        print("\nCreating derived features...")

        # 1. Average monthly spend
        df['avg_monthly_spend'] = df['total_charges'] / (df['tenure_months'] + 1)

        # 2. Service adoption score
        service_cols = ['phone_service', 'streaming_tv', 'streaming_movies', 'online_security']
        df['services_count'] = (df[service_cols] == 'Yes').sum(axis=1)

        # 3. Support intensity
        df['support_intensity'] = df['num_support_tickets'] + df['num_complaints']

        # 4. Tenure groups
        df['is_new_customer'] = (df['tenure_months'] <= 6).astype(int)
        df['is_long_term_customer'] = (df['tenure_months'] >= 36).astype(int)

        # 5. High-value customer flag
        df['is_high_value'] = (df['monthly_charges'] > df['monthly_charges'].quantile(0.75)).astype(int)

        # Handle missing values
        print("Handling missing values...")
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)

        # Encode categorical variables
        print("Encoding categorical variables...")
        label_encoders = {}
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        for col in categorical_cols:
            if col != 'churn':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le

        # Separate features and target
        X = df.drop(columns=['churn', 'tenure_bin'])
        y = df['churn']

        self.feature_names = X.columns.tolist()

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\n✓ Feature engineering complete")
        print(f"  - Total Features: {X.shape[1]}")
        print(f"  - Training Samples: {len(X_train):,}")
        print(f"  - Test Samples: {len(X_test):,}")
        print(f"  - Train Churn Rate: {y_train.mean():.1%}")
        print(f"  - Test Churn Rate: {y_test.mean():.1%}")

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        return X_train, X_test, y_train, y_test

    def train_model(self, time_budget: int = 300) -> AutoMLPipeline:
        """
        Train predictive model using AutoML.

        Args:
            time_budget: Time budget for AutoML in seconds

        Returns:
            Trained AutoML pipeline
        """
        print("\n" + "=" * 80)
        print("STEP 4: MODEL TRAINING (AutoML)")
        print("=" * 80)

        # Get model recommendations first
        print("\n🤖 Getting model recommendations...")
        matrix = InterpretabilityMatrix()

        data_chars = {
            'has_missing': False,  # We already imputed
            'has_categorical': True,
            'is_linear': False,  # From EDA we know there are non-linear patterns
            'has_interactions': True,
            'sample_size': len(self.X_train)
        }

        matrix.print_recommendation(
            use_case='operational_forecasting',  # High performance + some interpretability
            data_characteristics=data_chars,
            top_n=3
        )

        # Train AutoML pipeline
        print("\n" + "=" * 80)
        print("Starting AutoML Training...")
        print("=" * 80)

        pipeline = AutoMLPipeline(
            task='classification',
            metric='roc_auc',
            time_budget=time_budget,
            estimator_list=['lgbm', 'xgboost', 'rf', 'catboost'],
            ensemble_method='stacking',
            n_splits=5,
            verbose=0
        )

        pipeline.fit(self.X_train, self.y_train)

        self.model = pipeline

        return pipeline

    def evaluate_model(self) -> dict:
        """
        Comprehensive model evaluation.

        Returns:
            Evaluation metrics dictionary
        """
        print("\n" + "=" * 80)
        print("STEP 5: MODEL EVALUATION")
        print("=" * 80)

        # Evaluate using pipeline's method
        results = self.model.evaluate(self.X_test, self.y_test, use_ensemble=True)

        # Get predictions
        y_pred = self.model.predict(self.X_test, use_ensemble=True)
        y_pred_proba = self.model.predict_proba(self.X_test, use_ensemble=True)[:, 1]

        # Additional visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Confusion Matrix
        ax = axes[0, 0]
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

        # 2. ROC Curve
        ax = axes[0, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

        # 3. Precision-Recall Curve
        ax = axes[1, 0]
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
        avg_precision = average_precision_score(self.y_test, y_pred_proba)
        ax.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

        # 4. Prediction Distribution
        ax = axes[1, 1]
        ax.hist(y_pred_proba[self.y_test == 0], bins=30, alpha=0.6, label='Retained', color='green')
        ax.hist(y_pred_proba[self.y_test == 1], bins=30, alpha=0.6, label='Churned', color='red')
        ax.set_xlabel('Predicted Churn Probability')
        ax.set_ylabel('Frequency')
        ax.set_title('Prediction Distribution by True Class', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_evaluation.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Evaluation plots saved to: {self.output_dir / 'model_evaluation.png'}")
        plt.close()

        # Classification report
        print("\n" + "=" * 60)
        print("Classification Report:")
        print("=" * 60)
        print(classification_report(self.y_test, y_pred, target_names=['Retained', 'Churned']))

        return results

    def generate_business_insights(self) -> dict:
        """
        Generate actionable business insights and recommendations.

        Returns:
            Dictionary of insights
        """
        print("\n" + "=" * 80)
        print("STEP 6: BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("=" * 80)

        insights = {
            'churn_drivers': [],
            'high_risk_segments': [],
            'retention_strategies': [],
            'revenue_impact': {}
        }

        # Churn drivers from data
        churn_rate = self.data['churn'].mean()

        print("\n📊 Key Churn Drivers:")
        print("-" * 60)

        # Contract type
        contract_churn = self.data.groupby('contract_type')['churn'].mean()
        print(f"1. Contract Type:")
        for contract, rate in contract_churn.items():
            lift = (rate / churn_rate - 1) * 100
            print(f"   - {contract}: {rate:.1%} (lift: {lift:+.0f}%)")
            insights['churn_drivers'].append({
                'factor': f'Contract: {contract}',
                'churn_rate': rate,
                'lift': lift
            })

        # Payment method
        payment_churn = self.data.groupby('payment_method')['churn'].mean()
        print(f"\n2. Payment Method:")
        for method, rate in payment_churn.items():
            lift = (rate / churn_rate - 1) * 100
            print(f"   - {method}: {rate:.1%} (lift: {lift:+.0f}%)")

        # Revenue impact
        print(f"\n💰 Revenue Impact Analysis:")
        print("-" * 60)

        churned_customers = self.data[self.data['churn'] == 1]
        avg_monthly_revenue_loss = churned_customers['monthly_charges'].mean()
        annual_revenue_loss = avg_monthly_revenue_loss * 12 * len(churned_customers)

        print(f"Total Churned Customers: {len(churned_customers):,}")
        print(f"Avg Monthly Revenue per Churned Customer: ${avg_monthly_revenue_loss:.2f}")
        print(f"Estimated Annual Revenue Loss: ${annual_revenue_loss:,.2f}")

        # If we reduce churn by 10%
        reduction_10pct = 0.10
        saved_customers = len(churned_customers) * reduction_10pct
        revenue_saved = saved_customers * avg_monthly_revenue_loss * 12

        print(f"\n📈 Potential Impact of 10% Churn Reduction:")
        print(f"   - Customers Saved: {saved_customers:,.0f}")
        print(f"   - Annual Revenue Saved: ${revenue_saved:,.2f}")

        insights['revenue_impact'] = {
            'total_churned': len(churned_customers),
            'avg_monthly_loss': avg_monthly_revenue_loss,
            'annual_loss': annual_revenue_loss,
            'potential_savings_10pct': revenue_saved
        }

        # Recommendations
        print(f"\n💡 Retention Strategy Recommendations:")
        print("=" * 60)

        recommendations = [
            {
                'priority': 'HIGH',
                'strategy': 'Convert Month-to-Month Customers to Annual Contracts',
                'rationale': f'Month-to-month contracts have {contract_churn["Month-to-month"]:.1%} churn vs {contract_churn["Two year"]:.1%} for 2-year',
                'expected_impact': 'Reduce churn by 15-20%',
                'tactics': [
                    'Offer discounts for contract upgrades',
                    'Provide loyalty bonuses for long-term commitments',
                    'Add value-added services for annual plans'
                ]
            },
            {
                'priority': 'HIGH',
                'strategy': 'Improve Payment Method Mix',
                'rationale': f'Electronic check users churn at {payment_churn["Electronic check"]:.1%} vs {payment_churn.min():.1%} for best method',
                'expected_impact': 'Reduce churn by 5-10%',
                'tactics': [
                    'Incentivize automatic payment methods',
                    'Offer payment method migration bonuses',
                    'Educate customers on payment options'
                ]
            },
            {
                'priority': 'MEDIUM',
                'strategy': 'Proactive Customer Support Program',
                'rationale': 'High support ticket volume correlated with churn',
                'expected_impact': 'Reduce churn by 8-12%',
                'tactics': [
                    'Implement predictive support (reach out before issues escalate)',
                    'Reduce response times for high-value customers',
                    'Create customer success team for at-risk accounts'
                ]
            },
            {
                'priority': 'MEDIUM',
                'strategy': 'Early Tenure Engagement Program',
                'rationale': 'Churn is highest in first 6 months',
                'expected_impact': 'Reduce churn by 10-15% for new customers',
                'tactics': [
                    'Onboarding concierge service',
                    'Check-in calls at 30, 60, 90 days',
                    'New customer welcome offers'
                ]
            }
        ]

        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. [{rec['priority']}] {rec['strategy']}")
            print(f"   Rationale: {rec['rationale']}")
            print(f"   Expected Impact: {rec['expected_impact']}")
            print(f"   Tactics:")
            for tactic in rec['tactics']:
                print(f"      • {tactic}")

        insights['retention_strategies'] = recommendations

        return insights

    def save_model(self):
        """Save trained model to disk."""
        model_dir = self.output_dir / 'models'
        self.model.save(str(model_dir))
        print(f"\n✓ Model saved to: {model_dir}")

    def run_full_project(self):
        """Execute complete data science project."""
        # Step 1: Generate data
        self.generate_synthetic_data(n_samples=5000)

        # Step 2: EDA
        self.run_eda()

        # Step 3: Feature engineering
        self.prepare_features()

        # Step 4: Train model
        self.train_model(time_budget=180)  # 3 minutes

        # Step 5: Evaluate
        self.evaluate_model()

        # Step 6: Business insights
        insights = self.generate_business_insights()

        # Step 7: Save model
        self.save_model()

        print("\n" + "=" * 80)
        print("✅ DATA SCIENCE PROJECT COMPLETE")
        print("=" * 80)
        print(f"\nAll outputs saved to: {self.output_dir}")
        print("\nGenerated Files:")
        print("  - customer_data_raw.csv")
        print("  - linearity_check.png")
        print("  - interaction_plots_2way.png")
        print("  - missingness_heatmap.png")
        print("  - distribution_analysis.png")
        print("  - churn_analysis.png")
        print("  - model_evaluation.png")
        print("  - models/automl_model.pkl")
        print("  - models/ensemble_model.pkl")
        print("\n" + "=" * 80)

        return insights


if __name__ == "__main__":
    # Run complete project
    project = CustomerChurnProject()
    insights = project.run_full_project()
