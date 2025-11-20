# Utils Directory

유틸리티 함수 및 헬퍼 모듈을 저장하는 디렉토리입니다.

## 모듈 목록

### pdf_processor.py
- PDF 파일 읽기 및 파싱
- 테이블 추출
- OCR 처리

### email_sender.py
- SMTP를 통한 이메일 전송
- 첨부 파일 처리
- HTML 이메일 포맷팅

### slack_notifier.py
- Slack 웹훅 통합
- 메시지 포맷팅
- 알림 전송

## 사용 방법

```python
from utils.pdf_processor import extract_text_from_pdf
from utils.email_sender import send_email
from utils.slack_notifier import send_slack_notification

# PDF 처리
text = extract_text_from_pdf("document.pdf")

# 이메일 전송
send_email(
    to=["recipient@example.com"],
    subject="Analysis Complete",
    body="Report attached",
    attachments=["report.xlsx"]
)

# Slack 알림
send_slack_notification(
    channel="#analysis",
    message="Analysis completed successfully"
)
```

## 새로운 유틸리티 추가

새로운 유틸리티 함수가 필요한 경우:
1. 이 디렉토리에 새 `.py` 파일 생성
2. 명확한 함수명과 docstring 사용
3. `main.py`에서 import하여 사용
