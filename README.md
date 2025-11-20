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
cd GenZ

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
