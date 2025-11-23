# LIMS Quality Monitoring - Local Examples

이 디렉토리는 로컬에서 실행 가능한 LIMS 품질 모니터링 예제를 포함합니다.

## 🚀 빠른 시작

### 1. 의존성 설치

```bash
# 기본 의존성 (필수)
pip install pandas numpy scikit-learn

# 시각화 의존성 (선택사항)
pip install matplotlib seaborn
```

또는 requirements 파일 사용:

```bash
pip install -r requirements_local.txt
```

### 2. 스크립트 실행

#### Option A: Power BI 대시보드 데모
```bash
cd /home/user/GenZ/mlops_lims/examples
python lims_quality_monitoring_local.py
```

#### Option B: MLflow 실험 추적 데모 ⭐ NEW
```bash
# MLflow 추가 설치
pip install mlflow

# 스크립트 실행
python mlflow_training_demo.py

# MLflow UI 확인
mlflow ui  # http://localhost:5000
```

## 📁 파일 설명

### `lims_quality_monitoring_local.py`

**목적**: Power BI 대시보드와 연동되는 LIMS 품질 모니터링 파이프라인 시뮬레이션

**주요 기능**:
1. **데이터 추출**: LIMS 데이터베이스에서 샘플 데이터 추출 시뮬레이션
2. **데이터 클리닝**: 기계 오류 코드(-999) 처리, 결측치 보완
3. **이상 탐지**: Isolation Forest 모델로 품질 이상 탐지
4. **대시보드 업데이트**: Power BI용 CSV 파일 생성
5. **시각화**: 모니터링 대시보드 차트 생성

**실제 시나리오**:
- Azure Data Factory에서 매시간 자동 실행
- Power BI 대시보드가 Direct Query 모드로 자동 새로고침
- 임계 실패 발생 시 이메일/Slack 알림

---

### `mlflow_training_demo.py` ⭐ NEW

**목적**: MLflow를 사용한 LIMS 샘플 오염 탐지 모델 학습 및 추적

**주요 기능**:
1. **합성 데이터 생성**: 1,000개 LIMS 샘플 (오염률 15%)
2. **모델 학습**: Random Forest 분류기
3. **MLflow 추적**: 실험, 파라미터, 메트릭, 아티팩트 로깅
4. **모델 레지스트리**: 버전 관리 및 거버넌스 태그
5. **예측 데모**: 새 샘플에 대한 오염 확률 계산

**특징**:
- pH 레벨, 온도, 탁도, 처리 시간 기반 예측
- 피처 중요도 분석
- Confusion Matrix 생성
- 고위험 샘플 자동 플래그

**출력**:
- MLflow 추적 데이터 (`./mlruns/`)
- 피처 중요도 CSV
- Confusion Matrix CSV
- 분류 리포트 TXT
- 모델 아티팩트 (재사용 가능)

**MLflow UI**:
```bash
mlflow ui
# 브라우저: http://localhost:5000
```

**상세 가이드**: [README_MLFLOW.md](README_MLFLOW.md)

**Interview Talking Point**:
> "Using MLflow allows me to track every single experiment.
> If a model fails in production, I can trace it back to the exact code
> and LIMS data snapshot that created it."

---

### 출력 파일

스크립트 실행 후 `./lims_monitoring_output/` 디렉토리에 생성:

```
lims_monitoring_output/
├── dashboard_data.csv           # Power BI 대시보드 데이터 소스
├── critical_alerts.csv          # 임계 실패 샘플 목록
├── monitoring_report.json       # 실행 요약 보고서
└── monitoring_dashboard.png     # 시각화 차트
```

## 📊 예제 출력

### 대시보드 데이터 (dashboard_data.csv)

```csv
sample_id,facility_id,batch_id,timestamp,ph_level,storage_temp_c,dissolved_oxygen_ppm,turbidity_ntu,QA_ALERT,anomaly_score_value
S0001,GA_LTC_01,BATCH_20251122_000,2025-11-22 10:00:00,7.15,4.2,8.4,1.2,OK,-0.123
S0035,GA_LTC_03,BATCH_20251122_003,2025-11-22 13:00:00,7.18,85.0,8.3,1.1,CRITICAL_FAILURE,-0.567
```

### 모니터링 보고서 (monitoring_report.json)

```json
{
  "run_timestamp": "2025-11-22T14:30:00",
  "total_samples": 50,
  "alert_summary": {
    "OK": 42,
    "WARNING": 5,
    "CRITICAL_FAILURE": 3
  },
  "critical_samples": ["S0035", "S0036", "S0042"],
  "facilities_affected": ["GA_LTC_03", "GA_LTC_01"],
  "data_quality_score": 84.0
}
```

## 🔍 주입된 이상 사례

스크립트는 다음 현실적인 이상 사례를 포함합니다:

1. **기계 오류 코드** (샘플 S0012, S0023)
   - pH 센서: -999 (센서 고장)
   - 용존 산소: -999 (센서 실패)

2. **장비 오작동** (샘플 S0035, S0036)
   - 저장 온도: 85°C (냉장고 히터 오작동) ⚠️ **CRITICAL**
   - 정상 범위: 2-8°C

3. **오염 이벤트** (샘플 S0042-S0045)
   - 탁도: 15-25 NTU (정상 범위: 0-5)
   - 용존 산소 감소

4. **캘리브레이션 드리프트** (샘플 S0018-S0022)
   - pH 센서 재보정 필요 (+0.8 편향)

## 🏭 프로덕션 배포

### Azure Data Factory 설정

1. **파이프라인 생성**:
   ```json
   {
     "name": "LIMS_Quality_Monitoring",
     "type": "PythonActivity",
     "schedule": {
       "frequency": "Hour",
       "interval": 1
     },
     "script": "lims_quality_monitoring_local.py"
   }
   ```

2. **데이터베이스 연결 구성**:
   ```python
   # 스크립트 내 수정 필요:
   from sqlalchemy import create_engine

   engine = create_engine(
       'mssql+pyodbc://username:password@lims-prod.database.windows.net:1433/LIMS'
   )

   # extract_lims_data() 함수에서:
   df = pd.read_sql(
       "SELECT * FROM batch_results WHERE timestamp > ?",
       engine,
       params=[last_run_timestamp]
   )
   ```

3. **Power BI 연결**:
   - 데이터 소스: SQL Server (Direct Query)
   - 테이블: `lims_daily_monitoring`
   - 새로고침: 자동 (Direct Query 모드)

### 알림 설정

**이메일 알림** (임계 실패 시):

```python
import smtplib
from email.mime.text import MIMEText

def send_alert_email(critical_df):
    msg = MIMEText(f"CRITICAL: {len(critical_df)} samples failed quality check")
    msg['Subject'] = '⚠️ LIMS Quality Alert'
    msg['From'] = 'lims-monitor@example.com'
    msg['To'] = 'lab-manager@example.com'

    with smtplib.SMTP('smtp.office365.com', 587) as server:
        server.starttls()
        server.login(user, password)
        server.send_message(msg)
```

**Slack 알림**:

```python
import requests

def send_slack_alert(critical_df):
    webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
    message = {
        "text": f"🚨 LIMS Quality Alert: {len(critical_df)} critical failures detected",
        "attachments": [{
            "color": "danger",
            "fields": [
                {"title": "Samples", "value": ", ".join(critical_df['sample_id'].tolist())},
                {"title": "Facilities", "value": ", ".join(critical_df['facility_id'].unique())}
            ]
        }]
    }
    requests.post(webhook_url, json=message)
```

## 📈 성능 벤치마크

| 샘플 수 | 처리 시간 | 초당 샘플 |
|---------|-----------|-----------|
| 50      | ~2초      | 25        |
| 500     | ~5초      | 100       |
| 5,000   | ~30초     | 167       |
| 50,000  | ~5분      | 167       |

**프로덕션 환경**:
- 시간당 1,000-2,000 샘플
- Azure VM: Standard_D4s_v3 (4 vCPU, 16GB RAM)
- 월 비용: ~$150

## 🔐 보안 및 규정 준수

### HIPAA 준수
- PHI 데이터 없음 (샘플 ID만)
- 환자 정보는 LIMS에서 분리 저장
- 모든 데이터 전송 시 TLS 1.2+ 사용

### 데이터 거버넌스
- 원시 데이터 보관: 7년 (규정 준수)
- 모니터링 로그: 90일
- 접근 제어: Azure AD + RBAC

## 🐛 문제 해결

### 문제 1: "ModuleNotFoundError: No module named 'sklearn'"
```bash
pip install scikit-learn
```

### 문제 2: matplotlib 시각화가 표시되지 않음
```bash
# macOS/Linux
export MPLBACKEND=TkAgg

# Windows
set MPLBACKEND=TkAgg
```

### 문제 3: Permission denied writing to output directory
```bash
chmod +w ./lims_monitoring_output
```

## 📚 추가 리소스

- [Isolation Forest 알고리즘 설명](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [Power BI Direct Query 가이드](https://learn.microsoft.com/en-us/power-bi/connect-data/desktop-directquery-about)
- [Azure Data Factory Python Activity](https://learn.microsoft.com/en-us/azure/data-factory/transform-data-using-python)

## 💡 추가 개선 사항

### Phase 2: 고급 기능
- [ ] 실시간 스트리밍 (Kafka + Spark Streaming)
- [ ] 딥러닝 기반 이상 탐지 (Autoencoder)
- [ ] 다변량 시계열 예측 (Prophet, LSTM)
- [ ] 자동 근본 원인 분석 (Causal AI)

### Phase 3: 통합
- [ ] LIMS API 직접 연동
- [ ] EHR 시스템 연동 (HL7 FHIR)
- [ ] 모바일 앱 (실시간 알림)

## 📞 문의

- 기술 지원: ops-team@example.com
- 데이터 과학: datascience-team@example.com

---

**Last Updated**: 2025-11-22
**Version**: 1.0.0
