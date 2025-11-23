# MLflow Training Demo - LIMS Quality Control

## 📋 개요

이 스크립트는 LIMS 샘플 오염 탐지 모델을 MLflow로 학습하고 추적하는 방법을 보여줍니다.

**목적**:
- pH 레벨, 온도, 탁도, 처리 시간을 기반으로 샘플 오염을 예측
- MLflow를 사용한 실험 추적, 모델 버전 관리, 메타데이터 로깅

## 🎯 주요 기능

### MLflow 기능 시연
1. ✅ **실험 추적** (Experiment Tracking)
2. ✅ **파라미터 로깅** (Hyperparameter Logging)
3. ✅ **메트릭 로깅** (Metric Logging)
4. ✅ **모델 등록** (Model Registry)
5. ✅ **아티팩트 저장** (Artifact Storage)
6. ✅ **모델 버전 관리** (Model Versioning)
7. ✅ **거버넌스 태그** (Governance Tags)

## 🚀 실행 방법

### 1. 의존성 설치

```bash
# 기본 패키지
pip install pandas numpy scikit-learn

# MLflow (필수)
pip install mlflow

# 또는 requirements 파일 사용
pip install -r requirements_local.txt
```

**주의**: MLflow 설치 시 패키지 충돌이 발생하면:

```bash
# 가상 환경 사용 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install mlflow pandas numpy scikit-learn
```

### 2. 스크립트 실행

```bash
cd mlops_lims/examples
python mlflow_training_demo.py
```

### 3. MLflow UI 확인

```bash
# MLflow UI 실행 (별도 터미널)
mlflow ui

# 브라우저에서 확인
http://localhost:5000
```

## 📊 예상 출력

### Step 1: 데이터 생성

```
>>> STEP 1: Generating Training Data
------------------------------------------------------------
✓ Generated 1000 samples
✓ Contamination rate: 15.00%
✓ Features: ph_level, temperature, turbidity, processing_time

Sample data (first 5 rows):
  sample_id  ph_level  temperature  turbidity  processing_time  is_contaminated
0    S00000  7.234567         3.81       0.35             2.45                0
1    S00001  6.456789         13.2       8.23             5.78                1
2    S00002  7.189012         4.12       1.02             2.31                0
3    S00003  6.512345         11.5       7.89             5.12                1
4    S00004  7.298765         3.95       0.78             2.67                0
```

### Step 2: MLflow 학습

```
>>> STARTING MLFLOW TRAINING RUN
============================================================
✓ MLflow Run ID: a1b2c3d4e5f6g7h8i9j0
✓ Run started at: 2025-11-23 01:00:00

--- Logging Parameters ---
  n_estimators: 100
  max_depth: 5
  min_samples_split: 10
  min_samples_leaf: 5
  class_weight: balanced
  random_state: 42

--- Preparing Training Data ---
  Train samples: 800
  Test samples: 200
  Features: ['ph_level', 'temperature', 'turbidity', 'processing_time']

--- Training Model ---
  ✓ Model training complete

--- Evaluating Model (Test Set) ---
  test_accuracy: 0.9450
  test_precision: 0.8571
  test_recall: 0.9000
  test_f1: 0.8780

--- Feature Importance ---
  turbidity: 0.3521
  temperature: 0.3012
  ph_level: 0.2145
  processing_time: 0.1322

--- Confusion Matrix ---
  True Negatives:  167
  False Positives: 3
  False Negatives: 2
  True Positives:  28

--- Registering Model ---
  ✓ Model registered as: lims_contamination_detector

--- Adding Metadata Tags ---
  Project: Lab_Modernization
  Analyst: Senior_DS_Lead
  Model_Type: Random_Forest
  Use_Case: Contamination_Detection
  Training_Date: 2025-11-23
```

### Step 3: 새 샘플 예측

```
>>> STEP 3: Testing Model on New Samples
------------------------------------------------------------
Test samples:
  sample_id  ph_level  temperature  turbidity  processing_time
0   TEST001      7.2          4.0        1.2              2.5
1   TEST002      7.1          3.8        0.8              2.3
2   TEST003      6.3         15.0        8.5              6.0  ← 오염 의심
3   TEST004      7.3          4.2        1.0              2.4
4   TEST005      6.5         12.5        7.2              5.5  ← 오염 의심

--- Prediction Results ---
  sample_id  ph_level  temperature  predicted_contamination  contamination_probability risk_level
0   TEST001      7.2          4.0                        0                       0.05        LOW
1   TEST002      7.1          3.8                        0                       0.03        LOW
2   TEST003      6.3         15.0                        1                       0.92       HIGH  ⚠️
3   TEST004      7.3          4.2                        0                       0.08        LOW
4   TEST005      6.5         12.5                        1                       0.78       HIGH  ⚠️

⚠️  WARNING: 2 high-risk samples detected!
Samples requiring immediate attention:
  - TEST003: 92% contamination probability
  - TEST005: 78% contamination probability
```

## 📁 생성되는 파일

### MLflow 추적 디렉토리

```
mlops_lims/examples/
└── mlruns/
    └── 1/  (experiment_id)
        └── a1b2c3d4e5f6g7h8i9j0/  (run_id)
            ├── artifacts/
            │   ├── lims_qc_model/  (저장된 모델)
            │   ├── feature_importance.csv
            │   ├── confusion_matrix.csv
            │   └── classification_report.txt
            ├── metrics/
            │   ├── test_accuracy
            │   ├── test_precision
            │   ├── test_recall
            │   └── test_f1
            ├── params/
            │   ├── n_estimators
            │   ├── max_depth
            │   ├── data_source
            │   └── training_samples
            └── tags/
                ├── Project
                ├── Analyst
                └── Model_Type
```

## 🎨 MLflow UI 스크린샷

### 실험 목록
![MLflow Experiments](https://mlflow.org/docs/latest/_images/tracking-ui-1.png)

### 실행 상세
![MLflow Run Details](https://mlflow.org/docs/latest/_images/tracking-ui-2.png)

### 모델 레지스트리
![MLflow Model Registry](https://mlflow.org/docs/latest/_images/model-registry.png)

## 🔍 MLflow UI에서 확인할 수 있는 정보

### 1. Experiments 탭
- 모든 실험 목록
- 각 실험의 실행 수
- 최신 실행 시간

### 2. Runs 탭 (실험 내부)
- 각 실행의 메트릭 비교
- 하이퍼파라미터 비교
- 실행 시간, 상태

### 3. Run Detail (특정 실행)
- **Parameters**: 모든 하이퍼파라미터
- **Metrics**: 정확도, F1 등
- **Artifacts**: 저장된 모델, 차트, 리포트
- **Tags**: 프로젝트, 분석가, 날짜 등
- **Model**: 등록된 모델 정보

### 4. Models 탭
- 등록된 모델 목록
- 모델 버전 관리 (v1, v2, ...)
- 스테이징 (Staging, Production, Archived)
- 모델 전환 히스토리

## 💡 Interview Talking Points

### 1. 재현성 (Reproducibility)
```
"Using MLflow allows me to track every single experiment.
If a model fails in production, I can trace it back to the exact code
and LIMS data snapshot that created it."
```

**예시**:
- Run ID: `a1b2c3d4e5f6g7h8i9j0`
- 데이터 소스: `LIMS_Synthetic_v1`
- 학습 샘플: `1,000`
- 오염률: `15%`
- 하이퍼파라미터: `n_estimators=100, max_depth=5`

→ 3개월 후에도 정확히 동일한 모델 재현 가능!

### 2. 모델 거버넌스 (Governance)
```
"For government clients, we need full audit trails.
Every model in production has metadata tags showing who trained it,
when, and for what purpose."
```

**태그 예시**:
- Project: `Lab_Modernization`
- Analyst: `Senior_DS_Lead`
- Training_Date: `2025-11-23`
- Use_Case: `Contamination_Detection`

### 3. 모델 버전 관리
```
"When we deploy a new model, we don't delete the old one.
MLflow's model registry keeps all versions, so we can instantly
roll back if the new model underperforms."
```

**버전 예시**:
- v1.0: Baseline (F1: 0.82)
- v2.0: Added turbidity feature (F1: 0.88) ✓ Production
- v2.1: Hyperparameter tuning (F1: 0.89) ← 테스트 중

### 4. 팀 협업
```
"Our data science team can run 50 experiments in parallel,
and MLflow automatically tracks everything. We can compare
models side-by-side and pick the best one objectively."
```

## 🏭 프로덕션 배포 워크플로우

### 1. 개발 단계
```python
# 로컬에서 실험
with mlflow.start_run():
    model = train_model(data)
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_metric("f1_score", 0.88)
```

### 2. 모델 등록
```python
# MLflow UI에서 또는 코드로
mlflow.register_model(
    "runs:/a1b2c3d4e5f6g7h8i9j0/model",
    "lims_contamination_detector"
)
```

### 3. Staging으로 전환
```python
# 검증 환경에 배포
client = MlflowClient()
client.transition_model_version_stage(
    name="lims_contamination_detector",
    version=2,
    stage="Staging"
)
```

### 4. Production으로 승격
```python
# A/B 테스트 후 프로덕션 배포
client.transition_model_version_stage(
    name="lims_contamination_detector",
    version=2,
    stage="Production"
)
```

### 5. 프로덕션에서 모델 로드
```python
# API 서버에서 사용
model = mlflow.pyfunc.load_model(
    "models:/lims_contamination_detector/Production"
)
predictions = model.predict(new_data)
```

## 📈 실제 사용 사례

### Case 1: 하이퍼파라미터 튜닝

```python
# 여러 설정 자동 실험
for n_estimators in [50, 100, 200]:
    for max_depth in [3, 5, 7]:
        with mlflow.start_run():
            model = train_model(n_estimators, max_depth)
            # MLflow가 자동으로 추적

# MLflow UI에서 최고 성능 모델 찾기
# → F1 Score로 정렬 → 최고 점수 모델 선택
```

### Case 2: 데이터 드리프트 탐지

```python
# 매주 모델 재학습
with mlflow.start_run():
    model = train_model(current_week_data)
    mlflow.log_metric("f1_score", 0.85)  # 이전 주: 0.88

# MLflow UI에서 메트릭 추이 확인
# → F1 점수 감소 발견
# → 데이터 드리프트 조사
```

### Case 3: 모델 비교

```python
# 2개 알고리즘 비교
with mlflow.start_run(run_name="RandomForest"):
    rf_model = RandomForestClassifier()
    mlflow.log_metric("f1", 0.88)

with mlflow.start_run(run_name="XGBoost"):
    xgb_model = XGBClassifier()
    mlflow.log_metric("f1", 0.91)  # 더 좋음!

# MLflow UI에서 나란히 비교
# → XGBoost 선택
```

## 🔐 보안 및 규정 준수

### HIPAA/GDPR 준수
- ✅ 모든 모델 학습 이력 추적 (7년 보관)
- ✅ 감사 로그: 누가, 언제, 무엇을 학습했는지
- ✅ 모델 승인 워크플로우 (Staging → Production)
- ✅ 데이터 계보: 어떤 데이터로 학습했는지

### 태그 예시
```python
mlflow.set_tag("Data_Source", "LIMS_Silver_Layer_v2")
mlflow.set_tag("PHI_Included", "No")
mlflow.set_tag("Approval_Status", "Pending_Review")
mlflow.set_tag("Reviewer", "Clinical_Director")
```

## 🐛 문제 해결

### 문제 1: MLflow UI가 실행되지 않음
```bash
# 포트 충돌 확인
lsof -i :5000

# 다른 포트 사용
mlflow ui --port 5001
```

### 문제 2: 모델을 찾을 수 없음
```bash
# 추적 URI 확인
export MLFLOW_TRACKING_URI=./mlruns
python mlflow_training_demo.py
```

### 문제 3: 패키지 충돌
```bash
# 가상 환경 사용 (권장)
python -m venv venv
source venv/bin/activate
pip install mlflow pandas numpy scikit-learn
```

## 📚 추가 리소스

- [MLflow 공식 문서](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking API](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow Projects](https://mlflow.org/docs/latest/projects.html)

## 🎓 학습 경로

1. **기초**: MLflow Tracking (이 스크립트)
2. **중급**: MLflow Projects (재현 가능한 실험)
3. **고급**: MLflow Model Serving (API 배포)
4. **전문가**: MLflow + Databricks (엔터프라이즈)

---

**Last Updated**: 2025-11-23
**Version**: 1.0.0
**Author**: MLOps Team
