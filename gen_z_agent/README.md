# Gen Z Agent - 다중 에이전트 청구서/선거 데이터 자동화 시스템

CrewAI + Anthropic Claude를 사용한 다중 에이전트 시스템으로, 청구서와 선거 개표상황표를 자동으로 분석합니다.

## 🎯 주요 기능

- **다중 에이전트 협업**: 5개의 전문화된 AI 에이전트가 순차적으로 협력
- **자동 데이터 추출**: PDF/HTML 문서에서 구조화된 데이터 추출
- **지능형 검증**: 데이터 무결성 검증 및 이상치 탐지
- **고급 분석**: 통계 분석, 패턴 인식, 트렌드 분석
- **자동 보고서 생성**: Excel, Markdown, PDF 형식의 전문가급 보고서
- **알림 시스템**: 이메일 및 Slack 통합

## 📋 5개 전문 에이전트

1. **Invoice Data Extractor (청구서 데이터 추출 전문가)**
   - PDF/HTML 문서 읽기
   - 한국어 선거 개표상황표 구조 이해
   - OCR 및 테이블 추출

2. **Data Validator & Enricher (데이터 검증 및 보강 전문가)**
   - 데이터 무결성 검증
   - 합계 계산 확인
   - 외부 데이터로 보강

3. **Electoral Data Analyst (선거 데이터 분석가)**
   - 통계 분석 수행
   - 이상 패턴 탐지
   - 트렌드 및 인사이트 도출

4. **Executive Report Writer (보고서 작성 전문가)**
   - 다중 형식 보고서 생성
   - 시각화 및 차트
   - 임원급 요약

5. **Communication Agent (커뮤니케이션 담당자)**
   - 이메일 발송
   - Slack 알림
   - 이해관계자 커뮤니케이션

## 🚀 빠른 시작

### 1. 설치

```bash
# 저장소 클론
git clone https://github.com/sechan9999/GenZ.git
cd GenZ/gen_z_agent

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집하여 API 키 입력
# ANTHROPIC_API_KEY=your_actual_key_here
```

### 3. 실행

```bash
# 기본 실행 (dry-run 모드)
python main.py ./invoices/sample_invoice.pdf

# 특정 ID 지정
python main.py ./invoices/sejong_page2.pdf --id sejong_2025_jan

# 여러 수신자에게 전송
python main.py ./invoices/invoice.pdf --recipients "client@example.com,manager@example.com"

# 프로덕션 모드 (실제 이메일 전송)
python main.py ./invoices/invoice.pdf --production
```

## 📂 프로젝트 구조

```
gen_z_agent/
├── main.py                      # 메인 애플리케이션
├── config.py                    # 설정 관리
├── requirements.txt             # Python 의존성
├── .env.example                 # 환경 변수 템플릿
├── README.md                    # 이 파일
│
├── invoices/                    # 입력 문서 (PDF, HTML)
│   └── sample_invoice.pdf
│
├── historical/                  # 과거 데이터 (비교 분석용)
│   └── spend_2024_q4.csv
│
├── output/                      # 생성된 보고서
│   ├── Analysis_{id}.xlsx
│   ├── Report_{id}.md
│   └── Email_Summary_{id}.md
│
└── utils/                       # 유틸리티 함수
    ├── pdf_processor.py
    ├── email_sender.py
    └── slack_notifier.py
```

## 🔧 환경 변수

| 변수 | 필수 | 설명 |
|------|------|------|
| `ANTHROPIC_API_KEY` | ✅ | Anthropic Claude API 키 |
| `SERPER_API_KEY` | ❌ | 웹 검색 보강용 (선택) |
| `SMTP_HOST` | ❌ | 이메일 서버 호스트 |
| `SMTP_PORT` | ❌ | 이메일 서버 포트 |
| `SMTP_USERNAME` | ❌ | 이메일 계정 |
| `SMTP_PASSWORD` | ❌ | 이메일 비밀번호 |
| `SLACK_WEBHOOK_URL` | ❌ | Slack 웹훅 URL |

## 📊 출력 형식

### 1. Excel 보고서 (`Analysis_{id}.xlsx`)
- **Sheet 1: raw_data** - 원본 추출 데이터
- **Sheet 2: enriched_data** - 검증 및 보강된 데이터
- **Sheet 3: analysis** - 상세 분석 결과
- **Sheet 4: summary** - 요약 및 핵심 지표

### 2. Markdown 보고서 (`Report_{id}.md`)
```markdown
# 선거 데이터 분석 보고서

## 임원 요약 (Executive Summary)
- 주요 발견사항
- 핵심 메트릭

## 후보자별 분석
- 득표수 및 득표율
- 지역별 패턴

## 이상치 및 권장사항
```

### 3. 이메일 요약 (`Email_Summary_{id}.md`)
한 페이지 분량의 간결한 요약

## 🎓 사용 예시

### 예시 1: 선거 개표상황표 분석

```python
from main import run_invoice_analysis

result = run_invoice_analysis(
    invoice_path="./invoices/sejong_election_page2.pdf",
    invoice_id="sejong_2025_jan",
    recipients=["election@example.com"],
    dry_run=True  # 테스트 모드
)
```

### 예시 2: 청구서 분석

```python
result = run_invoice_analysis(
    invoice_path="./invoices/client_invoice_jan2025.pdf",
    invoice_id="inv_2025_001",
    recipients=["finance@company.com", "manager@company.com"],
    dry_run=False  # 실제 전송
)
```

## 🔍 선거 개표상황표 특화 기능

본 시스템은 한국 선거 개표상황표를 위한 특화 기능을 포함합니다:

- **이중 검증 시스템**: 기계 분류(②③) vs 인간 확인(a, b)
- **5명 후보자 자동 인식**: 이재명, 김문수, 이준석, 권영국, 송진호
- **투표 유형별 분류**: 관외사전, 관내사전, 선거일투표
- **자동 합계 검증**: (a) + (b) = 최종 집계
- **불일치 추적**: ①-②-③ 계산 및 플래깅

## 🛠️ 고급 설정

### 커스텀 에이전트 추가

`main.py`에서 새로운 에이전트를 추가할 수 있습니다:

```python
custom_agent = Agent(
    role="Custom Analyst",
    goal="특정 작업 수행",
    backstory="에이전트 배경 스토리",
    llm=llm,
    verbose=True
)
```

### 모델 변경

`.env` 파일에서 Claude 모델을 변경할 수 있습니다:

```bash
# Sonnet (균형잡힌 성능)
CLAUDE_MODEL=claude-sonnet-4-5-20250929

# Opus (최고 성능, 더 느림)
CLAUDE_MODEL=claude-opus-4

# Haiku (빠른 속도, 저렴)
CLAUDE_MODEL=claude-3-5-haiku-20241022
```

## 🐛 문제 해결

### "ANTHROPIC_API_KEY not found" 오류
```bash
# .env 파일이 올바른 위치에 있는지 확인
ls -la .env

# API 키가 올바르게 설정되었는지 확인
cat .env | grep ANTHROPIC_API_KEY
```

### PDF 읽기 오류
```bash
# PDF 처리 라이브러리 재설치
pip install --upgrade pypdf pdf2image

# 시스템 의존성 확인 (Ubuntu/Debian)
sudo apt-get install poppler-utils
```

### 메모리 부족 오류
```bash
# 대용량 PDF의 경우 페이지별 처리
# config.py에서 BATCH_SIZE 조정
```

## 📈 성능 최적화

- **병렬 처리**: 여러 문서를 동시에 처리하려면 `Process.hierarchical` 사용
- **캐싱**: 반복되는 분석은 결과 캐싱
- **모델 선택**: 간단한 작업에는 Haiku, 복잡한 분석에는 Sonnet/Opus

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🙏 감사의 말

- **CrewAI**: 다중 에이전트 프레임워크
- **Anthropic**: Claude AI 모델
- **PwC Hackathon**: 원본 아키텍처 영감

## 📞 지원 및 문의

- **GitHub Issues**: https://github.com/sechan9999/GenZ/issues
- **Email**: sechan9999@example.com

## 🎯 로드맵

- [ ] 웹 UI 인터페이스 추가
- [ ] 실시간 대시보드
- [ ] 더 많은 문서 형식 지원 (Excel, Word)
- [ ] 다국어 지원 확장
- [ ] 기계학습 기반 이상치 탐지 개선
- [ ] API 엔드포인트 제공

---

**Version**: 1.0.0
**Last Updated**: 2025-11-20
**Status**: Production Ready ✅
