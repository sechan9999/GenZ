"""
PII Redaction Utilities
Removes personally identifiable information from NVDRS narratives for HIPAA compliance
"""

import re
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class PIIRedactor:
    """Redacts PII from medical examiner and law enforcement narratives"""

    # Common name patterns (conservative matching)
    NAME_PATTERNS = [
        r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Last
        r'\b[A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+\b',  # First M. Last
        r'\b(Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-z]+\b',  # Title Name
    ]

    # Date patterns
    DATE_PATTERNS = [
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY or M/D/YY
        r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',  # MM-DD-YYYY
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}\b',
    ]

    # Social Security Numbers
    SSN_PATTERN = r'\b\d{3}-\d{2}-\d{4}\b'

    # Phone numbers
    PHONE_PATTERNS = [
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        r'\(\d{3}\)\s*\d{3}[-.]?\d{4}\b',
    ]

    # Email addresses
    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    # Street addresses
    ADDRESS_PATTERNS = [
        r'\b\d+\s+[A-Z][a-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\.?\b',
    ]

    # Age patterns (sometimes considered PII in combination with other data)
    AGE_PATTERN = r'\b(\d{1,3})\s*(?:year|yr)s?\s*old\b'

    def __init__(
        self,
        redact_names: bool = True,
        redact_dates: bool = True,
        redact_ssn: bool = True,
        redact_phone: bool = True,
        redact_email: bool = True,
        redact_addresses: bool = True,
        redact_ages: bool = False,  # Often kept for analysis
    ):
        """
        Initialize PII redactor

        Args:
            redact_names: Remove person names
            redact_dates: Remove dates
            redact_ssn: Remove SSNs
            redact_phone: Remove phone numbers
            redact_email: Remove email addresses
            redact_addresses: Remove street addresses
            redact_ages: Remove age mentions (usually kept for analysis)
        """
        self.redact_names = redact_names
        self.redact_dates = redact_dates
        self.redact_ssn = redact_ssn
        self.redact_phone = redact_phone
        self.redact_email = redact_email
        self.redact_addresses = redact_addresses
        self.redact_ages = redact_ages

        # Terms to preserve (not redact even if they match patterns)
        self.preserve_terms = {
            # Medical terms that might look like names
            "Chronic", "Acute", "Primary", "Secondary",
            # Common words that might match patterns
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
            # Locations (might want to keep for geographic analysis)
            "County", "State", "City", "Hospital", "Medical", "Center"
        }

    def redact(self, text: str) -> Tuple[str, List[str]]:
        """
        Redact PII from text

        Args:
            text: Input text to redact

        Returns:
            Tuple of (redacted_text, list of redaction types applied)
        """
        if not text:
            return "", []

        redacted_text = text
        redactions_applied = []

        # SSN (do first as most sensitive)
        if self.redact_ssn:
            if re.search(self.SSN_PATTERN, redacted_text):
                redacted_text = re.sub(self.SSN_PATTERN, "[SSN_REDACTED]", redacted_text)
                redactions_applied.append("SSN")

        # Phone numbers
        if self.redact_phone:
            for pattern in self.PHONE_PATTERNS:
                if re.search(pattern, redacted_text):
                    redacted_text = re.sub(pattern, "[PHONE_REDACTED]", redacted_text)
                    if "phone" not in redactions_applied:
                        redactions_applied.append("phone")

        # Email addresses
        if self.redact_email:
            if re.search(self.EMAIL_PATTERN, redacted_text):
                redacted_text = re.sub(self.EMAIL_PATTERN, "[EMAIL_REDACTED]", redacted_text)
                redactions_applied.append("email")

        # Dates (do before names to avoid conflicts)
        if self.redact_dates:
            for pattern in self.DATE_PATTERNS:
                if re.search(pattern, redacted_text, re.IGNORECASE):
                    redacted_text = re.sub(pattern, "[DATE_REDACTED]", redacted_text, flags=re.IGNORECASE)
                    if "date" not in redactions_applied:
                        redactions_applied.append("date")

        # Names (conservative - only redact clear name patterns)
        if self.redact_names:
            redacted_text = self._redact_names_carefully(redacted_text)
            if "[NAME_REDACTED]" in redacted_text:
                redactions_applied.append("name")

        # Addresses
        if self.redact_addresses:
            for pattern in self.ADDRESS_PATTERNS:
                if re.search(pattern, redacted_text, re.IGNORECASE):
                    redacted_text = re.sub(pattern, "[ADDRESS_REDACTED]", redacted_text, flags=re.IGNORECASE)
                    if "address" not in redactions_applied:
                        redactions_applied.append("address")

        # Ages (optional)
        if self.redact_ages:
            if re.search(self.AGE_PATTERN, redacted_text, re.IGNORECASE):
                # Keep the fact that age was mentioned but redact the value
                redacted_text = re.sub(
                    self.AGE_PATTERN,
                    r"[AGE_REDACTED] years old",
                    redacted_text,
                    flags=re.IGNORECASE
                )
                redactions_applied.append("age")

        return redacted_text, redactions_applied

    def _redact_names_carefully(self, text: str) -> str:
        """
        Redact names while preserving medical terms and important context

        This is conservative to avoid redacting medical terminology
        """
        redacted = text

        # Look for common name indicators
        # "Mr./Mrs./Ms./Dr. [Name]" - very likely to be names
        redacted = re.sub(
            r'\b(Mr\.|Mrs\.|Ms\.|Dr\.)\s+([A-Z][a-z]+)\b',
            r'\1 [NAME_REDACTED]',
            redacted
        )

        # "[First] [Last] was/had/reported" - likely names
        redacted = re.sub(
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(was|had|reported|stated|told|called)\b',
            r'[NAME_REDACTED] \2',
            redacted
        )

        # "according to [First Last]" - likely names
        redacted = re.sub(
            r'\baccording to\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b',
            r'according to [NAME_REDACTED]',
            redacted,
            flags=re.IGNORECASE
        )

        return redacted

    def batch_redact(self, texts: List[str]) -> List[Tuple[str, List[str]]]:
        """
        Redact PII from multiple texts

        Args:
            texts: List of texts to redact

        Returns:
            List of (redacted_text, redactions_applied) tuples
        """
        return [self.redact(text) for text in texts]

    def get_redaction_stats(self, texts: List[str]) -> dict:
        """
        Get statistics on redactions across multiple texts

        Args:
            texts: List of texts to analyze

        Returns:
            Dictionary with redaction statistics
        """
        results = self.batch_redact(texts)

        total_texts = len(texts)
        redaction_counts = {
            "name": 0,
            "date": 0,
            "SSN": 0,
            "phone": 0,
            "email": 0,
            "address": 0,
            "age": 0,
        }

        for _, redactions in results:
            for redaction_type in redactions:
                if redaction_type in redaction_counts:
                    redaction_counts[redaction_type] += 1

        return {
            "total_texts": total_texts,
            "texts_with_redactions": sum(1 for _, r in results if r),
            "redaction_counts": redaction_counts,
            "redaction_percentages": {
                k: round(v / total_texts * 100, 2)
                for k, v in redaction_counts.items()
            }
        }


def redact_spark_dataframe(df, text_columns: List[str], redactor: PIIRedactor = None):
    """
    Redact PII from Spark DataFrame columns

    Args:
        df: PySpark DataFrame
        text_columns: List of column names to redact
        redactor: PIIRedactor instance (creates default if None)

    Returns:
        DataFrame with redacted columns
    """
    if redactor is None:
        redactor = PIIRedactor()

    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType

    # Create UDF for redaction
    def redact_udf(text):
        if text:
            redacted, _ = redactor.redact(text)
            return redacted
        return text

    redact_text = udf(redact_udf, StringType())

    # Apply redaction to each specified column
    redacted_df = df
    for col_name in text_columns:
        if col_name in df.columns:
            redacted_df = redacted_df.withColumn(
                f"{col_name}_redacted",
                redact_text(df[col_name])
            )

    return redacted_df


if __name__ == "__main__":
    # Example usage
    print("PII Redaction Example")
    print("=" * 60)

    redactor = PIIRedactor()

    sample_texts = [
        "John Smith was found deceased on 03/15/2024 at 123 Main Street. Contact: 555-123-4567.",
        "Dr. Jane Doe reported that the decedent, age 45 years old, had a history of depression.",
        "The victim's SSN was 123-45-6789. Email: john.doe@example.com was found in records.",
    ]

    print("\nOriginal texts:")
    for i, text in enumerate(sample_texts, 1):
        print(f"{i}. {text}")

    print("\nRedacted texts:")
    for i, (redacted, redactions) in enumerate(redactor.batch_redact(sample_texts), 1):
        print(f"{i}. {redacted}")
        print(f"   Redactions: {', '.join(redactions)}")

    print("\nRedaction statistics:")
    stats = redactor.get_redaction_stats(sample_texts)
    import json
    print(json.dumps(stats, indent=2))
