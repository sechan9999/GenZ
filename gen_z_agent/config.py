"""
Configuration Management for Gen Z Agent
Handles all environment variables and application settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""

    # ═════════════════════════════════════════════════════════════
    # API Keys
    # ═════════════════════════════════════════════════════════════
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    SERPER_API_KEY: Optional[str] = os.getenv("SERPER_API_KEY")

    # ═════════════════════════════════════════════════════════════
    # Model Configuration
    # ═════════════════════════════════════════════════════════════
    CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
    CLAUDE_TEMPERATURE: float = float(os.getenv("CLAUDE_TEMPERATURE", "0"))
    CLAUDE_MAX_TOKENS: int = int(os.getenv("CLAUDE_MAX_TOKENS", "4096"))

    # ═════════════════════════════════════════════════════════════
    # Directory Paths
    # ═════════════════════════════════════════════════════════════
    BASE_DIR: Path = Path(__file__).parent
    OUTPUT_DIR: Path = BASE_DIR / "output"
    INVOICE_DIR: Path = BASE_DIR / "invoices"
    HISTORICAL_DIR: Path = BASE_DIR / "historical"
    TEMP_DIR: Path = BASE_DIR / "temp"
    UTILS_DIR: Path = BASE_DIR / "utils"

    # ═════════════════════════════════════════════════════════════
    # Email Configuration
    # ═════════════════════════════════════════════════════════════
    SMTP_HOST: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME: Optional[str] = os.getenv("SMTP_USERNAME")
    SMTP_PASSWORD: Optional[str] = os.getenv("SMTP_PASSWORD")

    # ═════════════════════════════════════════════════════════════
    # Slack Configuration
    # ═════════════════════════════════════════════════════════════
    SLACK_WEBHOOK_URL: Optional[str] = os.getenv("SLACK_WEBHOOK_URL")
    SLACK_CHANNEL: str = os.getenv("SLACK_CHANNEL", "#election-analysis")

    # ═════════════════════════════════════════════════════════════
    # Analysis Settings
    # ═════════════════════════════════════════════════════════════
    ANOMALY_THRESHOLD: float = float(os.getenv("ANOMALY_THRESHOLD", "2.0"))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.8"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "10"))

    # ═════════════════════════════════════════════════════════════
    # Logging
    # ═════════════════════════════════════════════════════════════
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Path = BASE_DIR / "gen_z_agent.log"

    # ═════════════════════════════════════════════════════════════
    # Korean Election Candidates (제21대 대통령선거)
    # ═════════════════════════════════════════════════════════════
    CANDIDATES = {
        1: {"name": "이재명", "party": "더불어민주당", "color": "#1976D2"},
        2: {"name": "김문수", "party": "국민의힘", "color": "#D32F2F"},
        4: {"name": "이준석", "party": "개혁신당", "color": "#F57C00"},
        5: {"name": "권영국", "party": "민주노동당", "color": "#7B1FA2"},
        8: {"name": "송진호", "party": "무소속", "color": "#558B2F"},
    }

    # ═════════════════════════════════════════════════════════════
    # Vote Types (투표 유형)
    # ═════════════════════════════════════════════════════════════
    VOTE_TYPES = {
        "관외사전투표": "out_of_area_early_voting",
        "관내사전투표": "in_area_early_voting",
        "선거일투표": "election_day_voting",
        "거소투표": "absentee_voting",
        "우편투표": "mail_voting",
    }

    # ═════════════════════════════════════════════════════════════
    # Report Templates
    # ═════════════════════════════════════════════════════════════
    REPORT_TEMPLATES = {
        "executive_summary": """
# {title} - 임원 요약

**분석 ID**: {analysis_id}
**분석 일시**: {timestamp}
**문서**: {document_name}

## 📊 핵심 메트릭

{key_metrics}

## 🔍 주요 발견사항

{key_findings}

## ⚠️ 이상치 및 주의사항

{anomalies}

## 💡 권장사항

{recommendations}
        """,
        "email_template": """
안녕하세요,

Gen Z Agent 시스템이 {document_name}에 대한 자동 분석을 완료했습니다.

주요 결과:
{summary}

상세 보고서는 첨부 파일을 참조해주세요.
- Excel 분석: {excel_file}
- Markdown 보고서: {markdown_file}

감사합니다.

---
Gen Z Agent - Automated Analysis System
Powered by Anthropic Claude
        """
    }

    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        for dir_path in [
            cls.OUTPUT_DIR,
            cls.INVOICE_DIR,
            cls.HISTORICAL_DIR,
            cls.TEMP_DIR,
            cls.UTILS_DIR,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        if not cls.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is required. "
                "Please set it in .env file or environment variables."
            )
        return True

    @classmethod
    def get_output_path(cls, filename: str) -> Path:
        """Get full path for output file"""
        return cls.OUTPUT_DIR / filename

    @classmethod
    def get_candidate_info(cls, candidate_number: int) -> dict:
        """Get candidate information by number"""
        return cls.CANDIDATES.get(candidate_number, {
            "name": f"후보자 {candidate_number}",
            "party": "알 수 없음",
            "color": "#999999"
        })

    @classmethod
    def get_vote_type_english(cls, korean_name: str) -> str:
        """Convert Korean vote type to English key"""
        return cls.VOTE_TYPES.get(korean_name, "unknown")

    @classmethod
    def info(cls) -> dict:
        """Return configuration summary"""
        return {
            "model": cls.CLAUDE_MODEL,
            "temperature": cls.CLAUDE_TEMPERATURE,
            "max_tokens": cls.CLAUDE_MAX_TOKENS,
            "output_dir": str(cls.OUTPUT_DIR),
            "anomaly_threshold": cls.ANOMALY_THRESHOLD,
            "confidence_threshold": cls.CONFIDENCE_THRESHOLD,
            "has_serper_key": bool(cls.SERPER_API_KEY),
            "has_smtp_config": bool(cls.SMTP_USERNAME and cls.SMTP_PASSWORD),
            "has_slack_webhook": bool(cls.SLACK_WEBHOOK_URL),
        }


# Initialize on import
Config.ensure_directories()

# Validate configuration (will raise error if API key is missing)
try:
    Config.validate()
except ValueError as e:
    print(f"⚠️  Configuration Warning: {e}")
    print("Please copy .env.example to .env and set ANTHROPIC_API_KEY")


if __name__ == "__main__":
    # Display configuration info
    import json
    print("Gen Z Agent Configuration")
    print("=" * 60)
    print(json.dumps(Config.info(), indent=2))
    print("\nDirectories:")
    for dir_name in ["OUTPUT_DIR", "INVOICE_DIR", "HISTORICAL_DIR", "TEMP_DIR"]:
        print(f"  {dir_name}: {getattr(Config, dir_name)}")
    print("\nCandidates:")
    for num, info in Config.CANDIDATES.items():
        print(f"  {num}. {info['name']} ({info['party']})")
