# ML Model Selection Framework

A comprehensive framework for automated machine learning model selection, interpretability analysis, and deployment optimization.

## Features

- **Automated EDA**: Linearity checks, interaction plots, missingness heatmaps
- **Interpretability Matrix**: Decision framework for model selection based on use case
- **Automated Model Selection**: FLAML-powered AutoML with ensemble methods
- **Model Explainability**: SHAP and LIME integration
- **Production Optimization**: ONNX conversion for ultra-low latency deployment

## Use Cases

1. **High-Stakes Equity Analysis**: Logistic Regression + LIME/SHAP or Causal Forest
2. **Operational Forecasting**: XGBoost/LightGBM + SHAP global summaries
3. **Ultra-Low Latency**: ONNX-converted models in Azure Functions

## Quick Start

```python
from ml_model_selection import ModelSelector, EDAAnalyzer

# Run EDA
eda = EDAAnalyzer(df, target='outcome')
eda.run_full_analysis()

# Select and train model
selector = ModelSelector(use_case='equity_analysis')
selector.fit(X_train, y_train)
selector.explain(X_test[:10])
```

## Installation

```bash
pip install -r requirements.txt
```

## Directory Structure

```
ml_model_selection/
├── README.md
├── requirements.txt
├── eda_analyzer.py          # EDA and linearity checks
├── model_selector.py        # Interpretability matrix and selection
├── automl_pipeline.py       # FLAML integration
├── explainability.py        # SHAP/LIME wrappers
├── onnx_converter.py        # Low-latency deployment
└── examples/
    ├── equity_analysis.py   # High-stakes classification
    ├── forecasting.py       # Operational forecasting
    └── low_latency.py       # ONNX deployment
```
