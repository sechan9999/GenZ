# ML Model Selection Framework - Quick Start

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Complete Decision Framework (Your Example)

```python
from eda_analyzer import EDAAnalyzer
from model_selector import InterpretabilityMatrix
from automl_pipeline import AutoMLPipeline
from explainability import ModelExplainer
from onnx_converter import ONNXConverter

# STEP 1: EDA → Linearity checks, interaction plots, missingness heatmaps
eda = EDAAnalyzer(df, target='outcome')
results = eda.run_full_analysis()

# Output:
# - linearity_check.png
# - residual_plots.png
# - interaction_plots_2way.png
# - interaction_plots_3way.png
# - missingness_heatmap.png
# - distribution_analysis.png

# STEP 2: Interpretability Matrix
matrix = InterpretabilityMatrix()

# Use case: High-stakes equity analysis → Logistic + LIME/SHAP
matrix.print_recommendation(
    use_case='equity_analysis',
    data_characteristics={
        'has_missing': True,
        'is_linear': True,
        'sample_size': 5000
    },
    top_n=3
)

# Use case: Operational forecasting → XGBoost/LightGBM + SHAP summaries
matrix.print_recommendation(
    use_case='operational_forecasting',
    data_characteristics={
        'has_missing': True,
        'has_interactions': True,
        'sample_size': 50000
    }
)

# Use case: Ultra-low latency → ONNX-converted models in Azure Functions
matrix.print_recommendation(
    use_case='low_latency',
    data_characteristics={
        'sample_size': 100000
    }
)

# STEP 3: Automated Model Selection + Stacking
pipeline = AutoMLPipeline(
    task='classification',
    metric='roc_auc',  # or 'accuracy', 'f1', 'aucpr'
    time_budget=3600,  # 1 hour
    estimator_list=['lgbm', 'xgboost', 'rf', 'extra_tree', 'catboost'],
    ensemble_method='stacking'  # or 'voting', 'none'
)

pipeline.fit(X_train, y_train)
results = pipeline.evaluate(X_test, y_test)

# STEP 4: Explainability (SHAP + LIME)
explainer = ModelExplainer(
    model=pipeline.best_model,
    X_train=X_train,
    feature_names=feature_names,
    task='classification'
)

# Generate comprehensive report
explainer.create_full_report(
    X_test=X_test,
    sample_indices=[0, 1, 2],
    top_features=10
)

# Output:
# - shap_summary.png
# - feature_importance_shap.png
# - shap_waterfall_sample_*.png
# - lime_explanation_sample_*.png
# - shap_dependence_*.png

# STEP 5: ONNX Conversion for Ultra-Low Latency
converter = ONNXConverter(
    model=pipeline.best_model,
    task='classification'
)

converter.convert(n_features=X.shape[1])
converter.save('model.onnx')

# Benchmark: Compare ONNX vs Python
benchmark = converter.benchmark(
    X_test=X_test,
    n_runs=100,
    compare_original=True
)

# Output:
# Python LightGBM: 2.5 ms/sample
# ONNX Runtime:    0.15 ms/sample
# Speedup: 16.7x faster ✓

# Deploy to Azure Functions
converter.deploy_azure_function(
    output_dir='azure_function',
    function_name='predict'
)
```

## Three Complete Use Case Examples

### Example 1: High-Stakes Equity Analysis (Credit Risk)

```bash
cd examples
python equity_analysis.py
```

**Use Case**: Loan approval decision-making
**Requirements**: HIGH interpretability, regulatory compliance
**Model**: Logistic Regression + LIME/SHAP
**Key Outputs**:
- Feature coefficients (transparent interpretation)
- SHAP/LIME explanations for individual applicants
- Regulatory compliance documentation

### Example 2: Operational Forecasting (Hospital Readmissions)

```bash
cd examples
python forecasting.py
```

**Use Case**: Predict 30-day hospital readmissions
**Requirements**: HIGH performance, MEDIUM interpretability
**Model**: FLAML AutoML (XGBoost/LightGBM) + Stacking Ensemble
**Key Outputs**:
- ROC/PR curves
- SHAP global summaries (clinical insights)
- Actionable intervention recommendations
- Saved model for EHR integration

### Example 3: Ultra-Low Latency (Real-Time Fraud Detection)

```bash
cd examples
python low_latency.py
```

**Use Case**: Real-time payment fraud detection
**Requirements**: <10ms latency, HIGH performance
**Model**: LightGBM → ONNX conversion
**Key Outputs**:
- Latency benchmarks (Python vs ONNX)
- Azure Function deployment package
- Production-ready ONNX model

## Module-by-Module Usage

### 1. EDA Analyzer

```python
from eda_analyzer import EDAAnalyzer

eda = EDAAnalyzer(df, target='outcome', task_type='classification')

# Individual analyses
eda.check_linearity()  # Correlation analysis + residual plots
eda.plot_interactions(top_n=5, interaction_type='pairwise')
eda.plot_missingness_heatmap()
eda.check_multicollinearity(threshold=10.0)
eda.analyze_distributions(top_n=12)

# Or run all at once
eda.run_full_analysis()
```

### 2. Model Selector (Interpretability Matrix)

```python
from model_selector import InterpretabilityMatrix

matrix = InterpretabilityMatrix()

# Get top 3 model recommendations
recommendations = matrix.select_models(
    use_case='equity_analysis',
    data_characteristics={
        'has_missing': True,
        'has_categorical': False,
        'is_linear': True,
        'sample_size': 5000
    },
    top_n=3
)

# Print detailed report
matrix.print_recommendation(
    use_case='operational_forecasting',
    data_characteristics=...,
    top_n=3
)
```

### 3. AutoML Pipeline (FLAML)

```python
from automl_pipeline import AutoMLPipeline

# Initialize
pipeline = AutoMLPipeline(
    task='classification',
    metric='roc_auc',
    time_budget=3600,  # seconds
    estimator_list=['lgbm', 'xgboost', 'rf', 'catboost'],
    ensemble_method='stacking',
    n_splits=5
)

# Train
pipeline.fit(X_train, y_train)

# Evaluate
results = pipeline.evaluate(X_test, y_test)

# Predict
predictions = pipeline.predict(X_test, use_ensemble=True)
probabilities = pipeline.predict_proba(X_test, use_ensemble=True)

# Save/Load
pipeline.save('saved_models/my_model')
loaded_pipeline = AutoMLPipeline.load('saved_models/my_model')
```

### 4. Explainability (SHAP + LIME)

```python
from explainability import ModelExplainer

explainer = ModelExplainer(
    model=trained_model,
    X_train=X_train,
    feature_names=feature_names,
    class_names=['Class 0', 'Class 1'],
    task='classification'
)

# SHAP analyses
explainer.setup_shap(explainer_type='auto')
explainer.compute_shap_values(X_test)
explainer.plot_shap_summary(max_display=20)
explainer.plot_shap_waterfall(sample_idx=0)
explainer.plot_shap_dependence('important_feature')

# LIME explanations
explainer.setup_lime()
explainer.explain_lime(sample_idx=0, num_features=10)

# Feature importance
explainer.plot_feature_importance(method='shap', top_n=20)

# Complete report
explainer.create_full_report(
    X_test=X_test,
    sample_indices=[0, 1, 2],
    top_features=10
)
```

### 5. ONNX Converter

```python
from onnx_converter import ONNXConverter

# Initialize
converter = ONNXConverter(
    model=trained_model,
    model_type='auto',  # or 'sklearn', 'xgboost', 'lightgbm', 'catboost'
    task='classification',
    feature_names=feature_names
)

# Convert to ONNX
converter.convert(n_features=X.shape[1])

# Save
converter.save('model.onnx')

# Create inference session
converter.create_inference_session(
    providers=['CPUExecutionProvider']  # or ['CUDAExecutionProvider', ...]
)

# Predict with ONNX
predictions = converter.predict(X_test)
probabilities = converter.predict_proba(X_test)

# Benchmark
benchmark = converter.benchmark(
    X_test=X_test,
    n_runs=100,
    compare_original=True
)

# Deploy to Azure Functions
converter.deploy_azure_function(
    output_dir='azure_function',
    function_name='predict'
)
```

## Decision Tree: Which Model to Use?

```
START
  │
  ├─ Need HIGH interpretability (compliance, regulatory)?
  │  │
  │  ├─ YES → Logistic Regression / Linear Regression
  │  │        + LIME/SHAP for individual explanations
  │  │        + Feature coefficients for global interpretation
  │  │
  │  └─ NO → Continue ↓
  │
  ├─ Need ULTRA-LOW latency (<10ms)?
  │  │
  │  ├─ YES → LightGBM (fast training + inference)
  │  │        → Convert to ONNX
  │  │        → Deploy to Azure Functions / AWS Lambda
  │  │
  │  └─ NO → Continue ↓
  │
  ├─ Need HIGHEST performance (accuracy/AUC)?
  │  │
  │  ├─ YES → FLAML AutoML
  │  │        → Stacking Ensemble (LGBM + XGBoost + RF + CatBoost)
  │  │        + SHAP global summaries for interpretability
  │  │
  │  └─ NO → Continue ↓
  │
  ├─ Need causal effect estimation?
  │  │
  │  └─ YES → Causal Forest (EconML)
  │           + Confidence intervals for treatment effects
  │
  └─ Default → XGBoost or LightGBM
               (good balance of performance, speed, interpretability)
```

## Key Metrics by Use Case

| Use Case | Primary Metric | Secondary Metrics | Threshold |
|----------|----------------|-------------------|-----------|
| **Equity Analysis** | ROC-AUC | Precision, Recall, Fairness | AUC > 0.75 |
| **Operational Forecasting** | ROC-AUC / F1 | Precision-Recall AUC | AUC > 0.80 |
| **Low Latency** | Latency (ms) | ROC-AUC, Throughput | <10ms/txn |
| **Causal Inference** | Treatment Effect | Confidence Interval Width | CI excludes 0 |

## Tips & Best Practices

### 1. EDA Before Modeling
- **Always** run EDA first to understand data characteristics
- Check for linearity → informs model choice
- Identify missing patterns → informs imputation strategy
- Detect multicollinearity → use regularization or feature selection

### 2. Model Selection
- **High-stakes decisions** → interpretable models (Logistic Regression)
- **Performance-critical** → ensemble methods (stacking, FLAML AutoML)
- **Latency-critical** → ONNX conversion (10-100x speedup)

### 3. Explainability
- **SHAP** for global summaries (feature importance across dataset)
- **LIME** for individual explanations (why this prediction?)
- **Feature coefficients** for linear models (most transparent)

### 4. Production Deployment
- **Always** benchmark ONNX vs Python before deployment
- **Monitor** latency, throughput, and model drift
- **Version** models and track metadata (hyperparameters, performance)

### 5. Handling Imbalanced Data
- Use `class_weight='balanced'` in sklearn models
- Optimize for **ROC-AUC** or **AUCPR** (not accuracy)
- Consider **SMOTE** or **undersampling** for extreme imbalance

## Troubleshooting

### Issue: FLAML AutoML runs too long
**Solution**: Reduce `time_budget` or limit `estimator_list`

```python
pipeline = AutoMLPipeline(
    time_budget=300,  # 5 minutes instead of 1 hour
    estimator_list=['lgbm', 'xgboost']  # Only fast models
)
```

### Issue: ONNX conversion fails
**Solution**: Check model compatibility, try different opset version

```python
converter.convert(n_features=X.shape[1], target_opset=11)
```

### Issue: SHAP computation too slow
**Solution**: Limit samples for SHAP computation

```python
explainer.compute_shap_values(X_test, max_samples=1000)
```

### Issue: Memory error with large datasets
**Solution**: Use batching or sampling

```python
# Sample data for EDA
df_sample = df.sample(10000, random_state=42)
eda = EDAAnalyzer(df_sample, target='outcome')
```

## Next Steps

1. **Run the examples**: `python examples/equity_analysis.py`
2. **Adapt to your data**: Replace sample datasets with real data
3. **Tune hyperparameters**: Use FLAML or manual tuning
4. **Deploy to production**: Use ONNX converter for deployment
5. **Monitor performance**: Track latency, accuracy, and drift

## Support

- **Documentation**: See `README.md` for full details
- **Examples**: Check `examples/` for complete workflows
- **Issues**: Open GitHub issues for bugs or questions

---

**Happy Modeling! 🚀**
