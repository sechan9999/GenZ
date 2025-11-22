"""
Example 3: Ultra-Low Latency Deployment

Use Case: Real-time fraud detection for payment transactions
Requirements:
- ULTRA-LOW latency (<10ms per prediction)
- HIGH performance (maximize fraud detection)
- LOW interpretability acceptable (focus on speed)
- Deployment to Azure Functions / AWS Lambda

Model Choice: LightGBM → ONNX conversion for ultra-fast inference
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
sys.path.append('..')

from eda_analyzer import EDAAnalyzer
from model_selector import InterpretabilityMatrix
from onnx_converter import ONNXConverter


def create_fraud_detection_dataset(n_samples=50000):
    """
    Simulate real-time fraud detection dataset.

    Features:
    - Transaction amount
    - Time since last transaction
    - Merchant category
    - Geographic distance from last transaction
    - Device fingerprint features
    - User behavior patterns
    - Historical fraud indicators

    Target: is_fraud (0 = legitimate, 1 = fraud)
    """
    print("=" * 80)
    print("CREATING FRAUD DETECTION DATASET")
    print("=" * 80)

    # Generate base features
    X, y = make_classification(
        n_samples=n_samples,
        n_features=30,
        n_informative=22,
        n_redundant=5,
        n_classes=2,
        weights=[0.98, 0.02],  # Highly imbalanced (2% fraud rate)
        random_state=42
    )

    feature_names = [
        'transaction_amount',
        'time_since_last_txn_mins',
        'merchant_category_code',
        'distance_from_home_km',
        'distance_from_last_txn_km',
        'device_age_days',
        'is_international',
        'is_online',
        'hour_of_day',
        'day_of_week',
        'num_txns_24h',
        'num_txns_7d',
        'avg_txn_amount_30d',
        'max_txn_amount_30d',
        'num_merchants_7d',
        'num_failed_txns_7d',
        'card_age_days',
        'num_cards_used_7d',
        'ip_address_age_days',
        'is_new_merchant',
        'merchant_fraud_rate',
        'velocity_score',
        'amount_deviation_score',
        'geographic_anomaly_score',
        'device_fingerprint_match',
        'billing_shipping_match',
        'cvv_match',
        'address_verification',
        'historical_chargebacks',
        'account_age_days'
    ]

    df = pd.DataFrame(X, columns=feature_names)

    # Transform to realistic ranges
    df['transaction_amount'] = np.exp(np.abs(df['transaction_amount']) * 0.5 + 3).round(2)
    df['time_since_last_txn_mins'] = (np.abs(df['time_since_last_txn_mins']) * 120 + 1).clip(0, 10080).round(0)  # Up to 1 week
    df['merchant_category_code'] = (np.abs(df['merchant_category_code']) * 100).round(0) % 100
    df['distance_from_home_km'] = (np.abs(df['distance_from_home_km']) * 50).clip(0, 5000).round(1)
    df['distance_from_last_txn_km'] = (np.abs(df['distance_from_last_txn_km']) * 30).clip(0, 3000).round(1)

    df['device_age_days'] = (np.abs(df['device_age_days']) * 100).clip(0, 3650).round(0)
    df['is_international'] = (df['is_international'] > df['is_international'].median()).astype(int)
    df['is_online'] = (df['is_online'] > df['is_online'].median()).astype(int)
    df['hour_of_day'] = ((df['hour_of_day'] - df['hour_of_day'].min()) /
                          (df['hour_of_day'].max() - df['hour_of_day'].min()) * 23).round(0).astype(int)
    df['day_of_week'] = ((df['day_of_week'] - df['day_of_week'].min()) /
                         (df['day_of_week'].max() - df['day_of_week'].min()) * 6).round(0).astype(int)

    df['num_txns_24h'] = (np.abs(df['num_txns_24h']) * 3).clip(0, 50).round(0)
    df['num_txns_7d'] = (np.abs(df['num_txns_7d']) * 10 + 1).clip(1, 200).round(0)
    df['avg_txn_amount_30d'] = np.exp(np.abs(df['avg_txn_amount_30d']) * 0.4 + 3).round(2)
    df['max_txn_amount_30d'] = df['avg_txn_amount_30d'] * (1 + np.abs(df['max_txn_amount_30d']) * 0.5)

    df['num_merchants_7d'] = (np.abs(df['num_merchants_7d']) * 3 + 1).clip(1, 30).round(0)
    df['num_failed_txns_7d'] = (np.abs(df['num_failed_txns_7d']) * 0.5).clip(0, 10).round(0)
    df['card_age_days'] = (np.abs(df['card_age_days']) * 200).clip(0, 3650).round(0)
    df['num_cards_used_7d'] = (np.abs(df['num_cards_used_7d']) * 0.3 + 1).clip(1, 5).round(0)

    df['ip_address_age_days'] = (np.abs(df['ip_address_age_days']) * 50).clip(0, 365).round(0)
    df['is_new_merchant'] = (df['is_new_merchant'] > df['is_new_merchant'].median()).astype(int)
    df['merchant_fraud_rate'] = (np.abs(df['merchant_fraud_rate']) * 0.05).clip(0, 0.5).round(4)

    df['velocity_score'] = (np.abs(df['velocity_score']) * 0.3).clip(0, 1).round(4)
    df['amount_deviation_score'] = (np.abs(df['amount_deviation_score']) * 0.3).clip(0, 1).round(4)
    df['geographic_anomaly_score'] = (np.abs(df['geographic_anomaly_score']) * 0.3).clip(0, 1).round(4)

    df['device_fingerprint_match'] = (df['device_fingerprint_match'] > df['device_fingerprint_match'].median()).astype(int)
    df['billing_shipping_match'] = (df['billing_shipping_match'] > df['billing_shipping_match'].median()).astype(int)
    df['cvv_match'] = (df['cvv_match'] > df['cvv_match'].median()).astype(int)
    df['address_verification'] = (df['address_verification'] > df['address_verification'].median()).astype(int)

    df['historical_chargebacks'] = (np.abs(df['historical_chargebacks']) * 0.2).clip(0, 10).round(0)
    df['account_age_days'] = (np.abs(df['account_age_days']) * 200).clip(0, 3650).round(0)

    df['is_fraud'] = y

    print(f"✓ Dataset created: {len(df):,} samples, {len(feature_names)} features")
    print(f"  Fraud rate: {df['is_fraud'].mean():.2%}")
    print(f"  Legitimate transactions: {(df['is_fraud'] == 0).sum():,}")
    print(f"  Fraudulent transactions: {(df['is_fraud'] == 1).sum():,}")
    print("=" * 80 + "\n")

    return df


def main():
    """Run complete ultra-low latency deployment pipeline."""

    print("\n" + "=" * 80)
    print("ULTRA-LOW LATENCY: REAL-TIME FRAUD DETECTION")
    print("=" * 80 + "\n")

    # Step 1: Create dataset
    df = create_fraud_detection_dataset(n_samples=50000)

    # Step 2: Quick EDA (abbreviated for speed)
    print("\n" + "=" * 80)
    print("STEP 1: QUICK EDA")
    print("=" * 80 + "\n")

    print("Dataset Summary:")
    print(f"  - Samples: {len(df):,}")
    print(f"  - Features: {len(df.columns) - 1}")
    print(f"  - Fraud rate: {df['is_fraud'].mean():.2%}")
    print(f"  - Missing values: {df.isnull().sum().sum()}")

    # Step 3: Model selection recommendation
    print("\n" + "=" * 80)
    print("STEP 2: MODEL SELECTION")
    print("=" * 80 + "\n")

    matrix = InterpretabilityMatrix()

    data_characteristics = {
        'has_missing': False,
        'has_categorical': False,
        'is_linear': False,
        'has_interactions': True,
        'sample_size': len(df)
    }

    matrix.print_recommendation(
        use_case='low_latency',
        data_characteristics=data_characteristics,
        top_n=3
    )

    # Step 4: Train LightGBM (fastest gradient boosting library)
    print("\n" + "=" * 80)
    print("STEP 3: TRAIN LIGHTGBM MODEL")
    print("=" * 80 + "\n")

    # Prepare data
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training LightGBM (optimized for speed)...")
    print("-" * 80)

    model = LGBMClassifier(
        n_estimators=100,  # Fewer trees for faster inference
        max_depth=6,  # Shallower trees for speed
        num_leaves=31,
        learning_rate=0.1,
        class_weight='balanced',  # Handle imbalance
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    train_start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - train_start

    print(f"✓ Training complete in {train_time:.2f}s")

    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("\nModel Performance:")
    print("-" * 80)
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"AUCPR: {average_precision_score(y_test, y_pred_proba):.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

    # Step 5: Benchmark Python inference speed
    print("\n" + "=" * 80)
    print("STEP 4: BENCHMARK PYTHON INFERENCE")
    print("=" * 80 + "\n")

    # Warmup
    for _ in range(10):
        model.predict(X_test[:100])

    # Benchmark
    n_runs = 1000
    times = []

    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(X_test[:10])  # Batch of 10 transactions
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    python_mean = np.mean(times)
    python_std = np.std(times)
    python_per_sample = python_mean / 10

    print(f"Python LightGBM Performance ({n_runs} runs):")
    print("-" * 80)
    print(f"  Mean latency (10 samples): {python_mean:.3f} ± {python_std:.3f} ms")
    print(f"  Latency per transaction: {python_per_sample:.3f} ms")
    print(f"  Throughput: {10 / (python_mean / 1000):.1f} txns/sec")

    # Step 6: Convert to ONNX
    print("\n" + "=" * 80)
    print("STEP 5: CONVERT TO ONNX FOR ULTRA-LOW LATENCY")
    print("=" * 80 + "\n")

    converter = ONNXConverter(
        model=model,
        model_type='lightgbm',
        task='classification',
        feature_names=X.columns.tolist()
    )

    # Convert
    converter.convert(n_features=X.shape[1])

    # Save ONNX model
    converter.save('fraud_detection_model.onnx')

    # Step 7: Benchmark ONNX inference
    print("\n" + "=" * 80)
    print("STEP 6: BENCHMARK ONNX INFERENCE")
    print("=" * 80 + "\n")

    benchmark_results = converter.benchmark(
        X_test=X_test.values,
        n_runs=1000,
        compare_original=True
    )

    # Detailed latency analysis
    print("\nLatency Breakdown:")
    print("-" * 80)

    onnx_per_sample = benchmark_results['onnx']['latency_per_sample_ms']
    python_per_sample_full = benchmark_results['original']['latency_per_sample_ms']

    print(f"  Python LightGBM: {python_per_sample_full:.4f} ms/transaction")
    print(f"  ONNX Runtime:    {onnx_per_sample:.4f} ms/transaction")
    print(f"\n  Speedup: {benchmark_results['speedup']:.2f}x faster")
    print(f"  Latency reduction: {benchmark_results['latency_reduction_pct']:.1f}%")

    # Determine if meets latency requirement
    latency_requirement_ms = 10.0
    meets_requirement = onnx_per_sample < latency_requirement_ms

    print(f"\n  Latency Requirement: <{latency_requirement_ms} ms/transaction")
    print(f"  Status: {'✓ MEETS REQUIREMENT' if meets_requirement else '⚠️ DOES NOT MEET'}")

    # Step 8: Visualize latency comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Latency comparison
    ax = axes[0]
    methods = ['Python\nLightGBM', 'ONNX\nRuntime']
    latencies = [python_per_sample_full, onnx_per_sample]
    colors = ['#1f77b4', '#2ca02c']

    bars = ax.bar(methods, latencies, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=latency_requirement_ms, color='red', linestyle='--', linewidth=2, label='Requirement (<10ms)')
    ax.set_ylabel('Latency (ms/transaction)')
    ax.set_title('Inference Latency Comparison')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, latency in zip(bars, latencies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{latency:.4f}ms',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 2: Throughput comparison
    ax = axes[1]
    throughputs = [
        benchmark_results['original']['throughput_per_sec'],
        benchmark_results['onnx']['throughput_per_sec']
    ]

    bars = ax.bar(methods, throughputs, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Throughput (transactions/second)')
    ax.set_title('Inference Throughput Comparison')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, tput in zip(bars, throughputs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{tput:,.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('low_latency_benchmark.png', dpi=300)
    print("\n✓ Benchmark plot saved to: low_latency_benchmark.png")
    plt.close()

    # Step 9: Deploy to Azure Functions
    print("\n" + "=" * 80)
    print("STEP 7: GENERATE AZURE FUNCTION DEPLOYMENT PACKAGE")
    print("=" * 80 + "\n")

    converter.deploy_azure_function(
        output_dir='azure_function_fraud_detection',
        function_name='predict_fraud'
    )

    # Step 10: Generate deployment guide
    print("\n" + "=" * 80)
    print("DEPLOYMENT GUIDE")
    print("=" * 80 + "\n")

    print("✓ Model Artifacts:")
    print("  - ONNX model: fraud_detection_model.onnx")
    print("  - Azure Function package: azure_function_fraud_detection/")
    print()

    print("✓ Performance Metrics:")
    print(f"  - Latency: {onnx_per_sample:.4f} ms/transaction")
    print(f"  - Throughput: {benchmark_results['onnx']['throughput_per_sec']:,.0f} txns/sec")
    print(f"  - ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print()

    print("✓ Deployment Steps:")
    print("  1. cd azure_function_fraud_detection")
    print("  2. func azure functionapp publish <FUNCTION_APP_NAME>")
    print("  3. Test endpoint:")
    print('     curl -X POST https://<APP>.azurewebsites.net/api/predict_fraud \\')
    print('          -H "Content-Type: application/json" \\')
    print('          -d \'{"features": [100.0, 15.0, ...]}\'')
    print()

    print("✓ Monitoring:")
    print("  - Enable Application Insights for latency tracking")
    print("  - Set up alerts for latency > 10ms")
    print("  - Monitor fraud detection rate and false positives")
    print()

    print("=" * 80)
    print("✓ ULTRA-LOW LATENCY DEPLOYMENT COMPLETE")
    print("=" * 80)
    print(f"\n🚀 Model ready for production deployment!")
    print(f"   - Inference latency: {onnx_per_sample:.4f} ms (target: <10ms)")
    print(f"   - Speedup vs Python: {benchmark_results['speedup']:.2f}x")
    print(f"   - Detection performance: ROC-AUC {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
