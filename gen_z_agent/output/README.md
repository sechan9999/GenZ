# Output Directory

생성된 분석 보고서가 저장되는 디렉토리입니다.

## 생성되는 파일

### 1. Excel 보고서 (`Analysis_{id}.xlsx`)
- Sheet 1: 원본 데이터 (raw_data)
- Sheet 2: 보강 데이터 (enriched_data)
- Sheet 3: 분석 결과 (analysis)
- Sheet 4: 요약 (summary)

### 2. Markdown 보고서 (`Report_{id}.md`)
- 임원 요약
- 주요 발견사항
- 상세 분석
- 권장사항

### 3. 이메일 요약 (`Email_Summary_{id}.md`)
- 간결한 한 페이지 요약
- 이메일 전송용

## 파일 보관

- 생성된 파일은 자동으로 이 디렉토리에 저장됩니다
- 파일명의 `{id}`는 분석 실행 시 지정한 ID입니다
- 정기적으로 오래된 파일을 정리하는 것을 권장합니다

## .gitignore

이 디렉토리의 파일들은 `.gitignore`에 포함되어 있습니다:
- `*.xlsx` - Excel 파일
- `*.pdf` - PDF 보고서
- `*.md` - Markdown 보고서 (README.md 제외)
