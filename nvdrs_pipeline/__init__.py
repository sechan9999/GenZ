"""
NVDRS NLP Pipeline
==================

A production-ready pipeline for analyzing National Violent Death Reporting System (NVDRS)
coroner and medical examiner narratives using Spark NLP and Hugging Face transformers.

This module extracts structured risk factors from unstructured text to support
suicide prevention and public health research.

Key Features:
- Distributed processing with Apache Spark
- Clinical NER using BioBERT/ClinicalBERT
- PII redaction for HIPAA compliance
- Risk factor extraction (substance use, mental health, financial stressors)
- Structured output for downstream analytics

Example:
    >>> from nvdrs_pipeline import NVDRSPipeline
    >>> pipeline = NVDRSPipeline()
    >>> results = pipeline.process_narratives(df)
    >>> risk_factors = pipeline.extract_risk_factors(results)

Author: Gen Z Agent Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Gen Z Agent Team"

from nvdrs_pipeline.pipeline import NVDRSPipeline
from nvdrs_pipeline.models import (
    NVDRSRecord,
    NERResult,
    RiskFactors,
    SubstanceUse,
    MentalHealthIndicators,
    SocialStressors,
)
from nvdrs_pipeline.config import NVDRSConfig

__all__ = [
    "NVDRSPipeline",
    "NVDRSRecord",
    "NERResult",
    "RiskFactors",
    "SubstanceUse",
    "MentalHealthIndicators",
    "SocialStressors",
    "NVDRSConfig",
]
