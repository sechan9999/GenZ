"""
Data Models for NVDRS NLP Pipeline
Pydantic models for structured data representation
"""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class SubstanceUse(BaseModel):
    """Substance use indicators extracted from narratives"""

    opioid_mentioned: bool = Field(False, description="Opioid mention detected")
    opioid_types: List[str] = Field(default_factory=list, description="Specific opioids mentioned")

    other_substances: List[str] = Field(default_factory=list, description="Other substances detected")

    alcohol_mentioned: bool = Field(False, description="Alcohol mention detected")

    prescription_drugs: List[str] = Field(default_factory=list, description="Prescription drugs mentioned")

    overdose_indicated: bool = Field(False, description="Overdose indication present")

    substance_abuse_history: bool = Field(False, description="History of substance abuse mentioned")


class MentalHealthIndicators(BaseModel):
    """Mental health indicators extracted from narratives"""

    depression_mentioned: bool = Field(False, description="Depression mentioned")

    anxiety_mentioned: bool = Field(False, description="Anxiety mentioned")

    ptsd_mentioned: bool = Field(False, description="PTSD mentioned")

    bipolar_mentioned: bool = Field(False, description="Bipolar disorder mentioned")

    schizophrenia_mentioned: bool = Field(False, description="Schizophrenia mentioned")

    suicide_history: bool = Field(False, description="Previous suicide attempt mentioned")

    psychiatric_treatment: bool = Field(False, description="Psychiatric treatment history mentioned")

    mental_health_diagnoses: List[str] = Field(
        default_factory=list,
        description="Specific mental health diagnoses mentioned"
    )


class SocialStressors(BaseModel):
    """Social and environmental stressor indicators"""

    financial_crisis: bool = Field(False, description="Financial crisis indicated")
    financial_details: List[str] = Field(default_factory=list, description="Specific financial issues")

    relationship_problems: bool = Field(False, description="Relationship problems indicated")
    relationship_details: List[str] = Field(default_factory=list, description="Specific relationship issues")

    legal_problems: bool = Field(False, description="Legal problems indicated")
    legal_details: List[str] = Field(default_factory=list, description="Specific legal issues")

    job_loss: bool = Field(False, description="Job loss or unemployment indicated")

    health_crisis: bool = Field(False, description="Health crisis indicated")
    health_details: List[str] = Field(default_factory=list, description="Specific health issues")


class RiskFactors(BaseModel):
    """Comprehensive risk factor profile"""

    substance_use: SubstanceUse = Field(
        default_factory=SubstanceUse,
        description="Substance use indicators"
    )

    mental_health: MentalHealthIndicators = Field(
        default_factory=MentalHealthIndicators,
        description="Mental health indicators"
    )

    social_stressors: SocialStressors = Field(
        default_factory=SocialStressors,
        description="Social stressor indicators"
    )

    risk_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Aggregate risk score (0-1)"
    )

    high_risk_flags: List[str] = Field(
        default_factory=list,
        description="List of high-risk indicators present"
    )


class NERResult(BaseModel):
    """Named Entity Recognition result"""

    entity_text: str = Field(..., description="The extracted entity text")
    entity_type: str = Field(..., description="Entity type (PROBLEM, TREATMENT, TEST, etc.)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    start_position: int = Field(..., description="Start position in text")
    end_position: int = Field(..., description="End position in text")
    sentence_id: int = Field(0, description="Sentence number in document")


class NVDRSRecord(BaseModel):
    """Complete NVDRS record with original data and extracted features"""

    # Original identifiers
    record_id: str = Field(..., description="Unique record identifier")
    state: Optional[str] = Field(None, description="State code")
    year: Optional[int] = Field(None, description="Year of incident")

    # Original narratives (potentially redacted)
    cme_narrative: Optional[str] = Field(None, description="Coroner/Medical Examiner narrative")
    le_narrative: Optional[str] = Field(None, description="Law Enforcement narrative")

    # Extracted entities
    entities: List[NERResult] = Field(
        default_factory=list,
        description="Named entities extracted from narratives"
    )

    # Risk factors
    risk_factors: RiskFactors = Field(
        default_factory=RiskFactors,
        description="Extracted risk factors"
    )

    # Classification
    predicted_intent: Optional[str] = Field(
        None,
        description="Predicted intent (suicide, accidental, undetermined, etc.)"
    )
    intent_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence in intent classification"
    )

    # Processing metadata
    processed_date: datetime = Field(
        default_factory=datetime.now,
        description="Date/time of processing"
    )
    pipeline_version: str = Field("1.0.0", description="Pipeline version")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PipelineResult(BaseModel):
    """Result of processing a batch of NVDRS records"""

    total_records: int = Field(..., description="Total number of records processed")
    successful: int = Field(..., description="Successfully processed records")
    failed: int = Field(0, description="Failed records")

    records: List[NVDRSRecord] = Field(
        default_factory=list,
        description="Processed records"
    )

    processing_time_seconds: float = Field(..., description="Total processing time")

    summary_statistics: Dict = Field(
        default_factory=dict,
        description="Summary statistics of risk factors"
    )

    errors: List[str] = Field(
        default_factory=list,
        description="Error messages for failed records"
    )


class ValidationResult(BaseModel):
    """Result of data validation"""

    is_valid: bool = Field(..., description="Whether data passed validation")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    record_count: int = Field(..., description="Number of records validated")
    missing_fields: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of missing values per field"
    )


if __name__ == "__main__":
    # Example usage and demonstration
    print("NVDRS Data Models Example")
    print("=" * 60)

    # Create a sample risk factor profile
    risk_factors = RiskFactors(
        substance_use=SubstanceUse(
            opioid_mentioned=True,
            opioid_types=["fentanyl", "oxycodone"],
            overdose_indicated=True
        ),
        mental_health=MentalHealthIndicators(
            depression_mentioned=True,
            suicide_history=True
        ),
        social_stressors=SocialStressors(
            financial_crisis=True,
            financial_details=["eviction notice", "bankruptcy"]
        ),
        risk_score=0.85,
        high_risk_flags=["opioid_overdose", "prior_suicide_attempt", "financial_crisis"]
    )

    # Create a sample NVDRS record
    record = NVDRSRecord(
        record_id="NVDRS_2024_001",
        state="CA",
        year=2024,
        cme_narrative="[REDACTED] was found deceased. Toxicology revealed fentanyl and oxycodone. History of depression.",
        le_narrative="Officers responded to welfare check. Decedent had recent eviction notice.",
        risk_factors=risk_factors,
        predicted_intent="suicide",
        intent_confidence=0.82
    )

    print("\nSample NVDRS Record:")
    print(record.model_dump_json(indent=2))

    print("\n✅ Data models validated successfully!")
