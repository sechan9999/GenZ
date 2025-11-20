# Historical Data Directory

과거 데이터를 저장하여 트렌드 분석 및 비교에 사용합니다.

## 사용 목적

- YoY (Year-over-Year) 비교
- QoQ (Quarter-over-Quarter) 비교
- 패턴 분석
- 이상치 탐지 기준선

## 파일 형식

- CSV 파일 (`.csv`)
- Excel 파일 (`.xlsx`)
- JSON 파일 (`.json`)

## 파일 명명 규칙

```
{category}_{year}_{period}.{extension}

예시:
- election_data_2024_q4.csv
- spend_2024_jan.xlsx
- votes_2023_annual.json
```

## 참고사항

과거 데이터가 없는 경우, 시스템은 현재 데이터만으로 분석을 수행합니다.
