"""
Tests for PII Redaction Module
Ensures HIPAA-compliant redaction of personally identifiable information
"""

import pytest
from nvdrs_pipeline.pii_redaction import PIIRedactor


class TestPIIRedactor:
    """Test suite for PIIRedactor class"""

    @pytest.fixture
    def redactor(self):
        """Create default redactor instance"""
        return PIIRedactor()

    def test_ssn_redaction(self, redactor):
        """Test Social Security Number redaction"""
        text = "Patient SSN is 123-45-6789 according to records."
        redacted, redaction_types = redactor.redact(text)

        assert "123-45-6789" not in redacted
        assert "[SSN_REDACTED]" in redacted
        assert "SSN" in redaction_types

    def test_phone_redaction(self, redactor):
        """Test phone number redaction"""
        test_cases = [
            "Call 555-123-4567 for information.",
            "Phone: (555) 123-4567",
            "Contact: 5551234567",
        ]

        for text in test_cases:
            redacted, redaction_types = redactor.redact(text)
            assert "[PHONE_REDACTED]" in redacted
            assert "phone" in redaction_types

    def test_email_redaction(self, redactor):
        """Test email address redaction"""
        text = "Contact john.doe@example.com for details."
        redacted, redaction_types = redactor.redact(text)

        assert "john.doe@example.com" not in redacted
        assert "[EMAIL_REDACTED]" in redacted
        assert "email" in redaction_types

    def test_date_redaction(self, redactor):
        """Test date redaction"""
        test_cases = [
            "Incident occurred on 03/15/2024.",
            "Date: January 15, 2024",
            "On Jan 15, 2024 at noon",
        ]

        for text in test_cases:
            redacted, redaction_types = redactor.redact(text)
            assert "[DATE_REDACTED]" in redacted
            assert "date" in redaction_types

    def test_name_redaction(self, redactor):
        """Test name redaction"""
        test_cases = [
            "Dr. Smith examined the patient.",
            "According to John Doe, the victim had...",
            "Mrs. Johnson reported the incident.",
        ]

        for text in test_cases:
            redacted, redaction_types = redactor.redact(text)
            assert "[NAME_REDACTED]" in redacted
            assert "name" in redaction_types

    def test_address_redaction(self, redactor):
        """Test street address redaction"""
        text = "Found at 123 Main Street in the afternoon."
        redacted, redaction_types = redactor.redact(text)

        assert "[ADDRESS_REDACTED]" in redacted
        assert "address" in redaction_types

    def test_preserve_medical_terms(self, redactor):
        """Ensure medical terms are not incorrectly redacted"""
        text = "Patient diagnosed with Chronic depression and Acute anxiety."
        redacted, _ = redactor.redact(text)

        # These medical terms should be preserved
        assert "Chronic" in redacted or "[NAME_REDACTED]" not in redacted
        assert "depression" in redacted
        assert "anxiety" in redacted

    def test_empty_text(self, redactor):
        """Test handling of empty text"""
        redacted, redaction_types = redactor.redact("")
        assert redacted == ""
        assert redaction_types == []

    def test_none_text(self, redactor):
        """Test handling of None"""
        redacted, redaction_types = redactor.redact(None)
        assert redacted == ""
        assert redaction_types == []

    def test_no_pii_text(self, redactor):
        """Test text with no PII"""
        text = "Patient presented with symptoms of depression."
        redacted, redaction_types = redactor.redact(text)

        assert redacted == text
        assert redaction_types == []

    def test_multiple_redactions(self, redactor):
        """Test text with multiple PII types"""
        text = """
        John Smith (SSN: 123-45-6789) called 555-1234 on 01/15/2024
        from 123 Main Street. Email: john@example.com
        """
        redacted, redaction_types = redactor.redact(text)

        assert "John Smith" not in redacted or "[NAME_REDACTED]" in redacted
        assert "123-45-6789" not in redacted
        assert "555-1234" not in redacted
        assert "john@example.com" not in redacted

        # Multiple redaction types should be present
        assert len(redaction_types) >= 3

    def test_batch_redaction(self, redactor):
        """Test batch processing of multiple texts"""
        texts = [
            "SSN: 123-45-6789",
            "Call 555-1234",
            "Email: test@example.com",
        ]

        results = redactor.batch_redact(texts)

        assert len(results) == 3
        for redacted, redaction_types in results:
            assert len(redaction_types) > 0

    def test_redaction_statistics(self, redactor):
        """Test redaction statistics calculation"""
        texts = [
            "John Doe, SSN: 123-45-6789, Phone: 555-1234",
            "No PII in this text",
            "Jane Smith, Email: jane@example.com",
        ]

        stats = redactor.get_redaction_stats(texts)

        assert stats["total_texts"] == 3
        assert stats["texts_with_redactions"] == 2
        assert stats["redaction_counts"]["SSN"] == 1
        assert stats["redaction_counts"]["phone"] == 1
        assert stats["redaction_counts"]["email"] == 1

    def test_selective_redaction(self):
        """Test selective redaction (turning off certain types)"""
        # Redact SSN but not names
        redactor = PIIRedactor(redact_names=False, redact_ssn=True)

        text = "John Smith's SSN is 123-45-6789."
        redacted, redaction_types = redactor.redact(text)

        assert "John Smith" in redacted  # Name preserved
        assert "123-45-6789" not in redacted  # SSN redacted
        assert "name" not in redaction_types
        assert "SSN" in redaction_types

    def test_age_preservation(self):
        """Test that ages are preserved by default"""
        redactor = PIIRedactor(redact_ages=False)

        text = "Patient was 45 years old at time of death."
        redacted, redaction_types = redactor.redact(text)

        assert "45 years old" in redacted
        assert "age" not in redaction_types

    def test_age_redaction_when_enabled(self):
        """Test age redaction when enabled"""
        redactor = PIIRedactor(redact_ages=True)

        text = "Decedent was 45 years old."
        redacted, redaction_types = redactor.redact(text)

        assert "[AGE_REDACTED]" in redacted
        assert "age" in redaction_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
