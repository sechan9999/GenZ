"""
Tests for Data Models
Validates Pydantic models for NVDRS data structures
"""

import pytest
from datetime import datetime
from nvdrs_pipeline.models import (
    SubstanceUse,
    MentalHealthIndicators,
    SocialStressors,
    RiskFactors,
    NERResult,
    NVDRSRecord,
    PipelineResult,
)


class TestSubstanceUse:
    """Test SubstanceUse model"""

    def test_default_values(self):
        """Test default values for substance use"""
        su = SubstanceUse()

        assert su.opioid_mentioned is False
        assert su.opioid_types == []
        assert su.alcohol_mentioned is False
        assert su.overdose_indicated is False

    def test_with_values(self):
        """Test creating with values"""
        su = SubstanceUse(
            opioid_mentioned=True,
            opioid_types=["fentanyl", "oxycodone"],
            alcohol_mentioned=True,
            overdose_indicated=True
        )

        assert su.opioid_mentioned is True
        assert len(su.opioid_types) == 2
        assert "fentanyl" in su.opioid_types


class TestMentalHealthIndicators:
    """Test MentalHealthIndicators model"""

    def test_default_values(self):
        """Test default values"""
        mh = MentalHealthIndicators()

        assert mh.depression_mentioned is False
        assert mh.anxiety_mentioned is False
        assert mh.suicide_history is False

    def test_with_diagnoses(self):
        """Test with mental health diagnoses"""
        mh = MentalHealthIndicators(
            depression_mentioned=True,
            mental_health_diagnoses=["Major Depressive Disorder", "GAD"]
        )

        assert mh.depression_mentioned is True
        assert len(mh.mental_health_diagnoses) == 2


class TestSocialStressors:
    """Test SocialStressors model"""

    def test_default_values(self):
        """Test default values"""
        ss = SocialStressors()

        assert ss.financial_crisis is False
        assert ss.relationship_problems is False
        assert ss.legal_problems is False

    def test_with_details(self):
        """Test with specific stressor details"""
        ss = SocialStressors(
            financial_crisis=True,
            financial_details=["eviction", "bankruptcy"],
            relationship_problems=True,
            relationship_details=["divorce"]
        )

        assert ss.financial_crisis is True
        assert "eviction" in ss.financial_details
        assert ss.relationship_problems is True


class TestRiskFactors:
    """Test RiskFactors model"""

    def test_default_values(self):
        """Test default risk factors"""
        rf = RiskFactors()

        assert rf.risk_score == 0.0
        assert rf.high_risk_flags == []
        assert isinstance(rf.substance_use, SubstanceUse)
        assert isinstance(rf.mental_health, MentalHealthIndicators)
        assert isinstance(rf.social_stressors, SocialStressors)

    def test_risk_score_validation(self):
        """Test risk score validation (must be 0-1)"""
        # Valid scores
        rf1 = RiskFactors(risk_score=0.5)
        assert rf1.risk_score == 0.5

        rf2 = RiskFactors(risk_score=0.0)
        assert rf2.risk_score == 0.0

        rf3 = RiskFactors(risk_score=1.0)
        assert rf3.risk_score == 1.0

        # Invalid scores should raise validation error
        with pytest.raises(Exception):  # Pydantic ValidationError
            RiskFactors(risk_score=1.5)

        with pytest.raises(Exception):
            RiskFactors(risk_score=-0.1)

    def test_complete_risk_profile(self):
        """Test complete risk factor profile"""
        rf = RiskFactors(
            substance_use=SubstanceUse(
                opioid_mentioned=True,
                opioid_types=["fentanyl"]
            ),
            mental_health=MentalHealthIndicators(
                depression_mentioned=True,
                suicide_history=True
            ),
            social_stressors=SocialStressors(
                financial_crisis=True
            ),
            risk_score=0.85,
            high_risk_flags=["opioid_use", "prior_attempt", "financial_crisis"]
        )

        assert rf.risk_score == 0.85
        assert len(rf.high_risk_flags) == 3
        assert rf.substance_use.opioid_mentioned is True
        assert rf.mental_health.suicide_history is True


class TestNERResult:
    """Test NERResult model"""

    def test_ner_result_creation(self):
        """Test creating NER result"""
        ner = NERResult(
            entity_text="fentanyl",
            entity_type="PROBLEM",
            confidence=0.95,
            start_position=10,
            end_position=18,
            sentence_id=0
        )

        assert ner.entity_text == "fentanyl"
        assert ner.entity_type == "PROBLEM"
        assert ner.confidence == 0.95

    def test_confidence_validation(self):
        """Test confidence score validation"""
        # Valid
        ner1 = NERResult(
            entity_text="test",
            entity_type="TEST",
            confidence=0.5,
            start_position=0,
            end_position=4
        )
        assert ner1.confidence == 0.5

        # Invalid
        with pytest.raises(Exception):
            NERResult(
                entity_text="test",
                entity_type="TEST",
                confidence=1.5,  # > 1.0
                start_position=0,
                end_position=4
            )


class TestNVDRSRecord:
    """Test NVDRSRecord model"""

    def test_minimal_record(self):
        """Test creating minimal record"""
        record = NVDRSRecord(record_id="TEST_001")

        assert record.record_id == "TEST_001"
        assert record.cme_narrative is None
        assert record.entities == []
        assert isinstance(record.risk_factors, RiskFactors)
        assert isinstance(record.processed_date, datetime)

    def test_complete_record(self):
        """Test creating complete record"""
        risk_factors = RiskFactors(
            substance_use=SubstanceUse(opioid_mentioned=True),
            risk_score=0.75
        )

        record = NVDRSRecord(
            record_id="NVDRS_2024_001",
            state="CA",
            year=2024,
            cme_narrative="[REDACTED] Toxicology positive for fentanyl.",
            le_narrative="Officers responded to scene.",
            risk_factors=risk_factors,
            predicted_intent="suicide",
            intent_confidence=0.82
        )

        assert record.record_id == "NVDRS_2024_001"
        assert record.state == "CA"
        assert record.year == 2024
        assert record.risk_factors.substance_use.opioid_mentioned is True
        assert record.predicted_intent == "suicide"
        assert record.intent_confidence == 0.82

    def test_json_serialization(self):
        """Test JSON serialization"""
        record = NVDRSRecord(
            record_id="TEST_001",
            state="NY",
            year=2024
        )

        json_str = record.model_dump_json()
        assert "TEST_001" in json_str
        assert "NY" in json_str

        # Should be valid JSON
        import json
        data = json.loads(json_str)
        assert data["record_id"] == "TEST_001"


class TestPipelineResult:
    """Test PipelineResult model"""

    def test_pipeline_result_creation(self):
        """Test creating pipeline result"""
        records = [
            NVDRSRecord(record_id="TEST_001"),
            NVDRSRecord(record_id="TEST_002"),
        ]

        result = PipelineResult(
            total_records=2,
            successful=2,
            failed=0,
            records=records,
            processing_time_seconds=5.5
        )

        assert result.total_records == 2
        assert result.successful == 2
        assert result.failed == 0
        assert len(result.records) == 2
        assert result.processing_time_seconds == 5.5

    def test_with_errors(self):
        """Test pipeline result with errors"""
        result = PipelineResult(
            total_records=3,
            successful=2,
            failed=1,
            records=[],
            processing_time_seconds=10.0,
            errors=["Failed to process record TEST_003"]
        )

        assert result.failed == 1
        assert len(result.errors) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
