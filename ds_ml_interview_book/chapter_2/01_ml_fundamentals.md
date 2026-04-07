# Chapter 2: Machine Learning Fundamentals

## Supervised Learning

### Q1: Explain bias-variance tradeoff with a real example

**A:** Bias-variance decomposition of expected test error:

```
E[(y - ŷ)²] = Bias² + Variance + Irreducible Error
```

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Generate data
np.random.seed(42)
X = np.sort(np.random.rand(100,1) * 10, axis=0)
y = np.sin(X).ravel() + np.random.randn(100) * 0.5

# Test different model complexities
degrees = [1, 4, 15]
plt.figure(figsize=(14, 4))

for i, degree in enumerate(degrees, 1):
    plt.subplot(1, 3, i)
    
    # Fit polynomial
    poly_features = PolynomialFeatures(degree=degree)
    model = Pipeline([
        ('poly', poly_features),
        ('linear', LinearRegression())
    ])
    model.fit(X, y)
    
    # Plot
    X_test = np.linspace(0, 10, 300).reshape(-1, 1)
    y_pred = model.predict(X_test)
    
    plt.scatter(X, y, s=20, alpha=0.5)
    plt.plot(X_test, y_pred, 'r-', linewidth=2)
    
    if degree == 1:
        plt.title(f'Degree {degree}: HIGH BIAS\n(underfitting)')
    elif degree == 4:
        plt.title(f'Degree {degree}: BALANCED')
    else:
        plt.title(f'Degree {degree}: HIGH VARIANCE\n(overfitting)')

plt.tight_layout()
plt.savefig('bias_variance.png')
```

**Interview answer structure:**

| Model Complexity | Bias | Variance | Example |
|---|---|---|---|
| Too simple | High | Low | Linear model on non-linear data |
| Just right | Low | Low | Polynomial degree 4 |
| Too complex | Low | High | Degree 15 — fits noise |

**Real-world example:**
- **High bias**: Logistic regression on image classification (assumes linear separability)
- **High variance**: Deep neural network with few samples (memorizes training set)
- **Sweet spot**: Cross-validated regularized model

**Follow-up: How do you detect which problem you have?**

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

# Diagnosis:
if train_mean > 0.9 and val_mean < 0.7:
    print("HIGH VARIANCE (overfitting)")
    # Fix: More data, regularization, simpler model
elif train_mean < 0.7 and val_mean < 0.7:
    print("HIGH BIAS (underfitting)")
    # Fix: More features, complex model, reduce regularization
else:
    print("BALANCED")
```

---

### Q2: Walk through building an end-to-end ML pipeline for production

**A:**
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib

# Define feature types
numeric_features = ['age', 'income', 'credit_score']
categorical_features = ['occupation', 'education']

# Numeric pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Cross-validation
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')
print(f"Cross-val AUC: {scores.mean():.3f} ± {scores.std():.3f}")

# Fit on full training data
pipeline.fit(X_train, y_train)

# Save for production
joblib.dump(pipeline, 'model_pipeline.pkl')

# Load and predict in production
loaded_pipeline = joblib.load('model_pipeline.pkl')
predictions = loaded_pipeline.predict(X_new)
probabilities = loaded_pipeline.predict_proba(X_new)
```

**Why pipelines matter:**
1. **Prevent data leakage**: Scaling fitted only on train, applied to test
2. **Reproducibility**: Same transforms in training and production
3. **Easy deployment**: Single object to serialize
4. **Hyperparameter tuning**: Can tune preprocessing + model together

---

### Q3: How do you handle class imbalance? Compare all approaches.

**A:**
```python
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report, confusion_matrix

# Dataset: 95% class 0, 5% class 1 (imbalanced)
# Assume X_train, y_train, X_test, y_test

## Approach 1: Class weights (in-model)
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
weight_dict = {0: class_weights[0], 1: class_weights[1]}

model = RandomForestClassifier(class_weight=weight_dict)
model.fit(X_train, y_train)

## Approach 2: SMOTE (synthetic minority oversampling)
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

model = RandomForestClassifier()
model.fit(X_train_sm, y_train_sm)

## Approach 3: Random undersampling
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

model = RandomForestClassifier()
model.fit(X_train_rus, y_train_rus)

## Approach 4: Threshold tuning
from sklearn.metrics import precision_recall_curve

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# Find threshold where F1 is maximized
f1_scores = 2 * (precision * recall) / (precision + recall)
best_threshold = thresholds[np.argmax(f1_scores)]

y_pred_tuned = (y_proba >= best_threshold).astype(int)

## Approach 5: Ensemble with different sampling
from sklearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier

# Automatically balances each bootstrap sample
model = BalancedRandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

## Approach 6: Anomaly detection (if extremely imbalanced)
from sklearn.ensemble import IsolationForest

# Treat minority class as anomalies
model = IsolationForest(contamination=0.05)  # 5% anomalies
model.fit(X_train[y_train == 0])  # Fit only on majority class

anomaly_preds = model.predict(X_test)  # -1 = anomaly (minority class)
```

**Comparison table:**

| Method | Pros | Cons | When to use |
|---|---|---|---|
| Class weights | No data modification | May not help tree models | Default choice |
| SMOTE | Synthetic samples, no loss | Can create noise, slow | <20:1 imbalance |
| Undersampling | Fast | Loses data | Lots of data (>100k) |
| Threshold tuning | Post-hoc, simple | Doesn't help training | Any model |
| Ensemble resampling | Robust | Complex | Tree-based models |
| Anomaly detection | Unsupervised option | Only for extreme cases | >99:1 imbalance |

**My recommendation in interviews:**
1. Start with class weights (easiest)
2. If not enough, add SMOTE
3. Tune threshold for deployment
4. Monitor precision/recall trade-off in production

---

### Q4: Explain cross-validation strategies. When would you use each?

**A:**
```python
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit,
    GroupKFold, LeaveOneOut
)

## Strategy 1: K-Fold (standard)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in kfold.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    # Train and evaluate

# Use when: IID data, balanced classes

## Strategy 2: Stratified K-Fold (classification)
stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in stratified.split(X, y):
    # Maintains class distribution in each fold
    pass

# Use when: Imbalanced classes (ALWAYS for classification)

## Strategy 3: Time Series Split
tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X):
    # Validation set always AFTER training set
    # No shuffle, respects temporal order
    pass

# Use when: Time series, sequential data (stock prices, sales)

## Strategy 4: Group K-Fold
groups = df['user_id'].values  # Same user shouldn't be in train & val
group_kfold = GroupKFold(n_splits=5)

for train_idx, val_idx in group_kfold.split(X, y, groups=groups):
    pass

# Use when: Multiple samples per entity (patient visits, user sessions)

## Strategy 5: Leave-One-Out (LOO)
loo = LeaveOneOut()

# Trains n times (one per sample)
# Use when: Very small dataset (<100 samples)

## Strategy 6: Nested CV (for hyperparameter tuning)
from sklearn.model_selection import GridSearchCV

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Inner loop: hyperparameter tuning
    grid_search = GridSearchCV(
        model, param_grid, cv=inner_cv, scoring='roc_auc'
    )
    grid_search.fit(X_train, y_train)
    
    # Outer loop: unbiased performance estimate
    score = grid_search.score(X_test, y_test)
```

**Decision tree:**
```
Is it time series?
├─ Yes → TimeSeriesSplit
└─ No → Are there groups/clusters?
    ├─ Yes → GroupKFold
    └─ No → Is it classification?
        ├─ Yes → StratifiedKFold
        └─ No → KFold
```

---

### Q5: Feature selection: filter, wrapper, embedded. Explain with code.

**A:**
```python
from sklearn.feature_selection import (
    SelectKBest, f_classif,  # Filter
    RFE,  # Wrapper
    SelectFromModel  # Embedded
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV

## Method 1: Filter (statistical tests, fast)
# ANOVA F-test for classification
selector = SelectKBest(score_func=f_classif, k=10)
X_new = selector.fit_transform(X_train, y_train)

# Get selected feature names
selected_features = X_train.columns[selector.get_support()]
print(f"Selected features: {list(selected_features)}")

# Pros: Fast, model-agnostic
# Cons: Ignores feature interactions

## Method 2: Wrapper (recursive feature elimination)
# Uses model to iteratively remove worst features
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=rf, n_features_to_select=10, step=1)
rfe.fit(X_train, y_train)

selected_features = X_train.columns[rfe.support_]
print(f"Selected features: {list(selected_features)}")

# Pros: Considers feature interactions
# Cons: Slow (trains model n_features times)

## Method 3: Embedded (L1 regularization)
# Lasso automatically zeros out irrelevant features
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train, y_train)

# Features with non-zero coefficients
selected_features = X_train.columns[lasso.coef_ != 0]
print(f"Selected features: {list(selected_features)}")

# Or: Use tree-based feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
selector = SelectFromModel(rf, threshold='median')
selector.fit(X_train, y_train)

X_new = selector.transform(X_train)

# Pros: Fast, considers interactions
# Cons: Model-dependent

## Method 4: Permutation importance (best for understanding)
from sklearn.inspection import permutation_importance

rf.fit(X_train, y_train)
perm_importance = permutation_importance(
    rf, X_test, y_test, n_repeats=10, random_state=42
)

# Sort features by importance
importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': perm_importance.importances_mean,
    'std': perm_importance.importances_std
}).sort_values('importance', ascending=False)

print(importances.head(10))
```

**My production approach:**
```python
# Step 1: Remove zero-variance and high-correlation features
from sklearn.feature_selection import VarianceThreshold

# Remove features with <1% variance
selector = VarianceThreshold(threshold=0.01)
X_filtered = selector.fit_transform(X_train)

# Remove highly correlated features (>0.95)
corr_matrix = pd.DataFrame(X_filtered).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X_filtered = X_filtered.drop(columns=to_drop)

# Step 2: Use embedded method (fast)
rf = RandomForestClassifier()
selector = SelectFromModel(rf, threshold='median')
X_final = selector.fit_transform(X_filtered, y_train)

# Step 3: Validate with permutation importance on final model
```

---

## Model Evaluation

### Q6: ROC-AUC vs PR-AUC: when to use which?

**A:**
```python
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    average_precision_score, roc_auc_score
)
import matplotlib.pyplot as plt

# Simulate imbalanced dataset
y_test = np.array([0]*950 + [1]*50)  # 5% positive class
y_proba = np.random.rand(1000)
y_proba[950:] += 0.3  # Make positives slightly higher

## ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

## Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)

plt.subplot(1, 2, 2)
plt.plot(recall, precision, label=f'PR (AUC = {pr_auc:.2f})')
plt.axhline(y=0.05, color='k', linestyle='--', label='Baseline')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()

plt.tight_layout()
plt.savefig('roc_vs_pr.png')
```

**Key differences:**

| Metric | When to use | Why |
|---|---|---|
| ROC-AUC | Balanced classes | Considers true negatives (denominator in FPR) |
| PR-AUC | Imbalanced classes | Ignores true negatives, focuses on positive class |

**Example showing the difference:**
- Dataset: 1% fraud (99 benign, 1 fraud)
- Baseline classifier: Predict all as benign
  - ROC-AUC = 0.50 (looks bad)
  - PR-AUC = 0.01 (terrible, shows the real problem)

**Interview answer:**
> "For imbalanced problems like fraud detection or rare disease diagnosis, I use PR-AUC because ROC-AUC can be misleadingly high. A classifier that achieves 0.99 accuracy by predicting everything as negative might have ROC-AUC = 0.90 but PR-AUC = 0.10, revealing it's useless. PR-AUC ranges from the baseline (% positive class) to 1.0, making it more interpretable for stakeholders."

---

### Q7: Explain calibration and how to check if your model probabilities are reliable

**A:**
```python
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import matplotlib.pyplot as plt

# Train a model (uncalibrated)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]

## Check calibration
prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)

plt.figure(figsize=(10, 5))

# Plot 1: Calibration curve
plt.subplot(1, 2, 1)
plt.plot(prob_pred, prob_true, marker='o', label='Model')
plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration Curve')
plt.legend()

# Interpretation:
# - If curve is ABOVE diagonal: Model is under-confident
# - If curve is BELOW diagonal: Model is over-confident
# - Perfect calibration: Points lie on diagonal

## Calibrate the model
calibrated = CalibratedClassifierCV(model, method='isotonic', cv=5)
calibrated.fit(X_train, y_train)

y_proba_cal = calibrated.predict_proba(X_test)[:, 1]
prob_true_cal, prob_pred_cal = calibration_curve(y_test, y_proba_cal, n_bins=10)

plt.subplot(1, 2, 2)
plt.plot(prob_pred, prob_true, marker='o', label='Uncalibrated')
plt.plot(prob_pred_cal, prob_true_cal, marker='s', label='Calibrated')
plt.plot([0, 1], [0, 1], 'k--', label='Perfect')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Before vs After Calibration')
plt.legend()

plt.tight_layout()
plt.savefig('calibration.png')
```

**Why calibration matters:**
- **Medical diagnosis**: "You have a 70% chance of disease" must be accurate
- **Credit scoring**: Interest rates based on default probability
- **Cost-sensitive decisions**: Expected value calculations require true probabilities

**Calibration methods:**
- **Platt scaling (sigmoid)**: Fits logistic regression to predictions (parametric)
- **Isotonic regression**: Non-parametric, monotonic (more flexible)

**Models that need calibration:**
- ✅ Naive Bayes (badly calibrated)
- ✅ SVM (needs calibration)
- ✅ Boosting (XGBoost, LightGBM)
- ❌ Logistic Regression (well-calibrated by design)
- ❌ Random Forest (reasonably calibrated)

---

## Summary: ML Fundamentals Checklist

| Concept | Key Insight | Interview Signal |
|---|---|---|
| Bias-variance | Trade-off between model complexity and generalization | "Learning curves show high bias (underfitting) vs high variance (overfitting)" |
| Pipeline | Prevent data leakage, ensure reproducibility | "Fit transforms on train only, apply to test" |
| Class imbalance | SMOTE, class weights, threshold tuning | "Start with class weights, use PR-AUC metric" |
| Cross-validation | Stratified for classification, time-split for time series | "Nested CV for unbiased hyperparameter tuning" |
| Feature selection | Filter (fast), wrapper (accurate), embedded (balanced) | "Use permutation importance for understanding" |
| ROC vs PR | PR-AUC for imbalanced data | "PR-AUC reveals performance on minority class" |
| Calibration | Probabilities must reflect reality | "Isotonic regression for calibration, check calibration curve" |
