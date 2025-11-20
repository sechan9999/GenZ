"""
NVDRS Pipeline Configuration
Manages settings for Spark NLP and Hugging Face model integration
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()


class NVDRSConfig:
    """Configuration for NVDRS NLP Pipeline"""

    # ═════════════════════════════════════════════════════════════
    # Spark Configuration
    # ═════════════════════════════════════════════════════════════
    SPARK_APP_NAME: str = os.getenv("SPARK_APP_NAME", "NVDRS_NLP_Pipeline")
    SPARK_MASTER: str = os.getenv("SPARK_MASTER", "local[*]")
    SPARK_DRIVER_MEMORY: str = os.getenv("SPARK_DRIVER_MEMORY", "8g")
    SPARK_EXECUTOR_MEMORY: str = os.getenv("SPARK_EXECUTOR_MEMORY", "8g")

    # Databricks settings (if running on Azure Databricks)
    DATABRICKS_TOKEN: Optional[str] = os.getenv("DATABRICKS_TOKEN")
    DATABRICKS_HOST: Optional[str] = os.getenv("DATABRICKS_HOST")

    # ═════════════════════════════════════════════════════════════
    # Model Configuration
    # ═════════════════════════════════════════════════════════════
    # Pre-trained models from Spark NLP
    NER_MODEL: str = os.getenv(
        "NER_MODEL",
        "bert_token_classifier_ner_clinical"
    )
    NER_LANGUAGE: str = os.getenv("NER_LANGUAGE", "en")
    NER_SOURCE: str = os.getenv("NER_SOURCE", "clinical/models")

    # Alternative models (uncomment to use)
    # "ner_clinical" - General clinical NER
    # "ner_posology" - Drug dosages and prescriptions
    # "ner_events_clinical" - Clinical events
    # "bert_token_classifier_ner_jsl" - John Snow Labs clinical NER

    # Hugging Face model for classification
    HF_CLASSIFIER_MODEL: str = os.getenv(
        "HF_CLASSIFIER_MODEL",
        "emilyalsentzer/Bio_ClinicalBERT"
    )

    # ═════════════════════════════════════════════════════════════
    # Pipeline Settings
    # ═════════════════════════════════════════════════════════════
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    MAX_SENTENCE_LENGTH: int = int(os.getenv("MAX_SENTENCE_LENGTH", "512"))
    CASE_SENSITIVE: bool = os.getenv("CASE_SENSITIVE", "False").lower() == "true"

    # ═════════════════════════════════════════════════════════════
    # Directory Paths
    # ═════════════════════════════════════════════════════════════
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data" / "nvdrs"
    OUTPUT_DIR: Path = BASE_DIR / "output" / "nvdrs"
    MODEL_CACHE_DIR: Path = BASE_DIR / "models" / "cache"
    TEMP_DIR: Path = BASE_DIR / "temp" / "nvdrs"

    # ═════════════════════════════════════════════════════════════
    # PII Redaction Settings
    # ═════════════════════════════════════════════════════════════
    REDACT_PII: bool = os.getenv("REDACT_PII", "True").lower() == "true"
    REDACT_NAMES: bool = os.getenv("REDACT_NAMES", "True").lower() == "true"
    REDACT_DATES: bool = os.getenv("REDACT_DATES", "True").lower() == "true"
    REDACT_LOCATIONS: bool = os.getenv("REDACT_LOCATIONS", "False").lower() == "true"

    # ═════════════════════════════════════════════════════════════
    # Risk Factor Detection Keywords
    # ═════════════════════════════════════════════════════════════

    # Substance use keywords
    OPIOID_KEYWORDS: List[str] = [
        "fentanyl", "oxycodone", "hydrocodone", "morphine", "heroin",
        "opioid", "opiate", "methadone", "buprenorphine", "tramadol",
        "codeine", "oxycontin", "percocet", "vicodin"
    ]

    SUBSTANCE_KEYWORDS: List[str] = [
        "alcohol", "cocaine", "methamphetamine", "meth", "amphetamine",
        "marijuana", "cannabis", "benzodiazepine", "xanax", "valium",
        "prescription", "overdose", "intoxication", "substance abuse"
    ]

    # Mental health keywords
    MENTAL_HEALTH_KEYWORDS: List[str] = [
        "depression", "depressed", "anxiety", "ptsd", "bipolar",
        "schizophrenia", "psychosis", "mental health", "psychiatric",
        "suicide", "suicidal", "self-harm", "attempted suicide",
        "mental illness", "therapy", "counseling", "antidepressant"
    ]

    # Financial stressor keywords
    FINANCIAL_KEYWORDS: List[str] = [
        "eviction", "foreclosure", "bankruptcy", "debt", "unemployed",
        "job loss", "financial", "bills", "mortgage", "rent",
        "creditor", "collection", "repossession", "financial crisis"
    ]

    # Relationship stressor keywords
    RELATIONSHIP_KEYWORDS: List[str] = [
        "divorce", "separation", "breakup", "domestic violence",
        "restraining order", "custody", "relationship problems",
        "marital", "spouse", "partner conflict"
    ]

    # Legal stressor keywords
    LEGAL_KEYWORDS: List[str] = [
        "arrest", "incarceration", "jail", "prison", "legal problems",
        "lawsuit", "criminal charges", "probation", "parole"
    ]

    # Health crisis keywords
    HEALTH_CRISIS_KEYWORDS: List[str] = [
        "terminal illness", "cancer", "chronic pain", "disability",
        "medical condition", "diagnosis", "prognosis", "terminal"
    ]

    # ═════════════════════════════════════════════════════════════
    # Classification Settings
    # ═════════════════════════════════════════════════════════════
    INTENT_CLASSIFICATION_THRESHOLD: float = float(
        os.getenv("INTENT_CLASSIFICATION_THRESHOLD", "0.7")
    )

    # Intent categories
    INTENT_CATEGORIES: List[str] = [
        "suicide",
        "accidental_overdose",
        "undetermined",
        "homicide",
        "natural_causes"
    ]

    # ═════════════════════════════════════════════════════════════
    # Output Settings
    # ═════════════════════════════════════════════════════════════
    OUTPUT_FORMAT: str = os.getenv("OUTPUT_FORMAT", "parquet")  # parquet, csv, json
    SAVE_INTERMEDIATE: bool = os.getenv("SAVE_INTERMEDIATE", "False").lower() == "true"

    # ═════════════════════════════════════════════════════════════
    # Logging
    # ═════════════════════════════════════════════════════════════
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Path = BASE_DIR / "logs" / "nvdrs_pipeline.log"

    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        for dir_path in [
            cls.DATA_DIR,
            cls.OUTPUT_DIR,
            cls.MODEL_CACHE_DIR,
            cls.TEMP_DIR,
            cls.LOG_FILE.parent,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_all_risk_keywords(cls) -> Dict[str, List[str]]:
        """Return all risk factor keywords organized by category"""
        return {
            "opioids": cls.OPIOID_KEYWORDS,
            "substances": cls.SUBSTANCE_KEYWORDS,
            "mental_health": cls.MENTAL_HEALTH_KEYWORDS,
            "financial": cls.FINANCIAL_KEYWORDS,
            "relationship": cls.RELATIONSHIP_KEYWORDS,
            "legal": cls.LEGAL_KEYWORDS,
            "health_crisis": cls.HEALTH_CRISIS_KEYWORDS,
        }

    @classmethod
    def info(cls) -> Dict:
        """Return configuration summary"""
        return {
            "spark_app": cls.SPARK_APP_NAME,
            "spark_master": cls.SPARK_MASTER,
            "ner_model": cls.NER_MODEL,
            "batch_size": cls.BATCH_SIZE,
            "redact_pii": cls.REDACT_PII,
            "output_format": cls.OUTPUT_FORMAT,
            "data_dir": str(cls.DATA_DIR),
            "output_dir": str(cls.OUTPUT_DIR),
        }


# Initialize directories on import
NVDRSConfig.ensure_directories()


if __name__ == "__main__":
    import json
    print("NVDRS Pipeline Configuration")
    print("=" * 60)
    print(json.dumps(NVDRSConfig.info(), indent=2))
    print("\nRisk Factor Categories:")
    for category, keywords in NVDRSConfig.get_all_risk_keywords().items():
        print(f"  {category}: {len(keywords)} keywords")
