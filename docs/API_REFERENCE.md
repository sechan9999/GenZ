# Clinical Workflow Assistant - API Reference

**Version**: 2.0.0
**Last Updated**: 2025-11-22

---

## Table of Contents

1. [RAG System API](#rag-system-api)
2. [Guardrails API](#guardrails-api)
3. [HITL System API](#hitl-system-api)
4. [Healthcare Agents API](#healthcare-agents-api)
5. [Security & Compliance API](#security--compliance-api)
6. [Data Models](#data-models)
7. [Error Handling](#error-handling)
8. [Examples](#examples)

---

## RAG System API

### Module: `gen_z_agent.clinical_rag`

#### `ClinicalRAGSystem`

Main class for retrieval-augmented generation.

```python
from gen_z_agent.clinical_rag import ClinicalRAGSystem

rag = ClinicalRAGSystem(
    collection_name="clinical_knowledge",
    embedding_model="all-MiniLM-L6-v2",
    persist_directory=None  # Optional: defaults to healthcare/vector_db
)
```

**Parameters**:
- `collection_name` (str): ChromaDB collection name
- `embedding_model` (str): Sentence transformer model name
- `persist_directory` (Path, optional): Vector database storage location

---

#### `get_clinical_context(query, k=3, min_score=0.3, filter_domain=None)`

Retrieve clinical evidence for a query.

```python
context = rag.get_clinical_context(
    query="How should I manage hypertension in diabetic patients?",
    k=3,
    min_score=0.3,
    filter_domain="cardiology"
)
```

**Parameters**:
- `query` (str): Clinical question
- `k` (int, default=3): Number of documents to retrieve
- `min_score` (float, default=0.3): Minimum relevance score (0.0-1.0)
- `filter_domain` (str, optional): Filter by clinical domain

**Returns**: `ClinicalContext` object with:
- `query` (str): Original query
- `retrieved_docs` (List[RetrievalResult]): Retrieved documents with scores
- `context_text` (str): Formatted context for LLM
- `sources` (List[str]): Source citations
- `confidence` (float): Overall confidence score (0.0-1.0)

**Performance**: <200ms typical

---

#### `add_documents(documents)`

Add clinical documents to knowledge base.

```python
from gen_z_agent.clinical_rag import ClinicalDocument

doc = ClinicalDocument(
    id="guideline_001",
    content="ACC/AHA 2017 Hypertension Guidelines...",
    metadata={
        "source": "ACC/AHA",
        "clinical_domain": "cardiology",
        "last_updated": "2017-11-13"
    }
)

rag.add_documents([doc])
```

**Parameters**:
- `documents` (List[ClinicalDocument]): Documents to add

**Returns**: None

**Side Effects**: Rebuilds BM25 index, logs audit event

---

#### `retrieve_hybrid(query, k=5, alpha=0.7)`

Hybrid retrieval (dense + sparse).

```python
results = rag.retrieve_hybrid(
    query="hypertension management",
    k=5,
    alpha=0.7  # 70% dense, 30% sparse
)
```

**Parameters**:
- `query` (str): Search query
- `k` (int, default=5): Number of results
- `alpha` (float, default=0.7): Weight for dense retrieval

**Returns**: List[RetrievalResult]

---

### Helper Functions

#### `get_rag_system()`

Get or initialize global RAG system instance.

```python
from gen_z_agent.clinical_rag import get_rag_system

rag = get_rag_system()
```

**Returns**: `ClinicalRAGSystem` (singleton)

---

#### `format_rag_prompt(clinical_question, context, base_prompt)`

Format prompt with RAG context.

```python
from gen_z_agent.clinical_rag import format_rag_prompt

augmented_prompt = format_rag_prompt(
    clinical_question="What are the treatment options?",
    context=clinical_context,
    base_prompt="You are a clinical expert..."
)
```

**Returns**: str (augmented prompt)

---

## Guardrails API

### Module: `gen_z_agent.clinical_guardrails`

#### `GuardrailsEngine`

Orchestrate all guardrail checks.

```python
from gen_z_agent.clinical_guardrails import GuardrailsEngine

engine = GuardrailsEngine()
```

---

#### `validate_clinical_workflow(input_data, prompt, output, patient_context=None)`

Run all guardrails for a clinical workflow.

```python
passed, violations = engine.validate_clinical_workflow(
    input_data=fhir_observation,
    prompt=agent_prompt,
    output=llm_output,
    patient_context={"allergies": ["penicillin"]}
)

if not passed:
    print(engine.format_violations_report(violations))
```

**Parameters**:
- `input_data` (Dict): FHIR data or clinical input
- `prompt` (str): Prompt sent to LLM
- `output` (str): LLM output
- `patient_context` (Dict, optional): Additional patient context

**Returns**:
- `passed` (bool): True if no critical/error violations
- `violations` (List[GuardrailViolation]): List of violations

**Performance**: <100ms typical

---

#### `format_violations_report(violations)`

Format violations as human-readable report.

```python
report = engine.format_violations_report(violations)
print(report)
```

**Returns**: str (formatted report)

---

### Input Guardrails

#### `InputGuardrails.validate_fhir_data(data)`

Validate FHIR resource data.

```python
from gen_z_agent.clinical_guardrails import InputGuardrails

result = InputGuardrails.validate_fhir_data(observation_data)
```

**Returns**: `GuardrailResult`

**Checks**:
- Required fields (resourceType, id)
- Valid resource types
- Temporal consistency
- PHI classification

---

#### `InputGuardrails.validate_vital_signs(observation)`

Validate vital sign values.

```python
result = InputGuardrails.validate_vital_signs(observation)
```

**Parameters**:
- `observation` (Observation): FHIR Observation object

**Returns**: `GuardrailResult`

**Checks**:
- Physiologically plausible ranges
- Critical value detection
- Unit consistency

---

### Output Guardrails

#### `OutputGuardrails.detect_hallucination(output, confidence_threshold=0.8)`

Detect potential hallucinations.

```python
from gen_z_agent.clinical_guardrails import OutputGuardrails

result = OutputGuardrails.detect_hallucination(
    output=llm_response,
    confidence_threshold=0.8
)
```

**Returns**: `GuardrailResult` with confidence score

**Detection Methods**:
- Uncertainty marker counting
- Excessive specificity detection
- Confidence scoring

---

#### `OutputGuardrails.validate_medication_recommendation(medication_name, patient_allergies=None, current_medications=None)`

Validate medication safety.

```python
result = OutputGuardrails.validate_medication_recommendation(
    medication_name="lisinopril",
    patient_allergies=["ACE inhibitors"],
    current_medications=["enalapril"]
)
```

**Returns**: `GuardrailResult`

**Checks**:
- Allergy contraindications (CRITICAL)
- Duplicate medications (ERROR)
- High-risk medications (WARNING)

---

#### `OutputGuardrails.validate_risk_score(risk_score, risk_factors, min_factors=1)`

Validate risk score calculation.

```python
result = OutputGuardrails.validate_risk_score(
    risk_score=85,
    risk_factors={"hypertension": True, "diabetes": True},
    min_factors=2
)
```

**Returns**: `GuardrailResult`

---

### Compliance Guardrails

#### `ComplianceGuardrails.validate_phi_handling(data, is_encrypted=False, phi_classification="RESTRICTED")`

Validate PHI handling.

```python
from gen_z_agent.clinical_guardrails import ComplianceGuardrails

result = ComplianceGuardrails.validate_phi_handling(
    data=patient_data,
    is_encrypted=True,
    phi_classification="RESTRICTED"
)
```

**Returns**: `GuardrailResult`

---

## HITL System API

### Module: `gen_z_agent.clinical_hitl`

#### `ReviewManager`

Manage clinician reviews and feedback.

```python
from gen_z_agent.clinical_hitl import ReviewManager

manager = ReviewManager(storage_dir=None)  # Optional
```

---

#### `register_clinician(name, role, specialty=None, years_experience=None)`

Register a clinician.

```python
clinician = manager.register_clinician(
    name="Dr. Sarah Johnson",
    role="physician",
    specialty="cardiology",
    years_experience=15
)
```

**Returns**: `ClinicianProfile`

---

#### `create_review(report_id, patient_id, ai_output, ai_confidence, ai_risk_category, priority=ReviewPriority.MEDIUM)`

Create a review request.

```python
from gen_z_agent.clinical_hitl import ReviewPriority

review = manager.create_review(
    report_id="report_001",
    patient_id="PAT12345",
    ai_output="Patient has HIGH risk...",
    ai_confidence=0.87,
    ai_risk_category="HIGH",
    priority=ReviewPriority.HIGH
)
```

**Returns**: `ClinicalReview`

**Side Effects**: De-identifies patient ID, logs audit event

---

#### `submit_review(review_id, clinician_id, agreement_level, feedback=None, corrected_output=None, ...)`

Submit clinician review.

```python
from gen_z_agent.clinical_hitl import AgreementLevel

reviewed = manager.submit_review(
    review_id=review.review_id,
    clinician_id="clinician_123",
    agreement_level=AgreementLevel.PARTIALLY_AGREE,
    feedback="Should also recommend medication counseling",
    corrected_output="Updated recommendation...",
    accuracy_rating=4,
    completeness_rating=3,
    clinical_utility_rating=5
)
```

**Parameters**:
- `review_id` (str): Review identifier
- `clinician_id` (str): Clinician submitting review
- `agreement_level` (AgreementLevel): Agreement with AI
- `feedback` (str, optional): Free-text feedback
- `corrected_output` (str, optional): Corrected output
- `accuracy_rating` (int, optional): 1-5 stars
- `completeness_rating` (int, optional): 1-5 stars
- `clinical_utility_rating` (int, optional): 1-5 stars
- `requires_escalation` (bool, default=False): Escalation flag
- `escalation_reason` (str, optional): Reason for escalation

**Returns**: `ClinicalReview`

**Side Effects**:
- Updates clinician statistics
- Creates training example if corrections provided
- Logs PHI access event

---

#### `get_metrics(start_date=None, end_date=None)`

Get review metrics.

```python
from datetime import datetime, timedelta

metrics = manager.get_metrics(
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now()
)

print(f"Agreement rate: {metrics.fully_agree_rate:.1%}")
print(f"Average accuracy: {metrics.avg_accuracy_rating:.1f}/5.0")
```

**Returns**: `ReviewMetrics`

---

#### `export_training_data(min_agreement_score=0.7, output_file=None)`

Export training examples.

```python
examples = manager.export_training_data(
    min_agreement_score=0.7,
    output_file=Path("training_data.jsonl")
)

print(f"Exported {len(examples)} examples")
```

**Returns**: List[TrainingExample]

---

### Global Instance

```python
from gen_z_agent.clinical_hitl import review_manager

# Use global instance
review = review_manager.create_review(...)
```

---

## Healthcare Agents API

### Module: `gen_z_agent.healthcare_agents`

#### `run_clinical_workflow(workflow_type, patient_id=None, data_source=None, dry_run=True)`

Run AI-driven clinical workflow.

```python
from gen_z_agent.healthcare_agents import run_clinical_workflow

result = run_clinical_workflow(
    workflow_type="patient_risk_assessment",
    patient_id="PAT12345",
    data_source="/path/to/fhir_data",
    dry_run=False
)
```

**Parameters**:
- `workflow_type` (str): "patient_risk_assessment", "medication_review", "vitals_monitoring", "care_gap_analysis"
- `patient_id` (str, optional): Patient identifier
- `data_source` (str, optional): Path to FHIR data or API endpoint
- `dry_run` (bool, default=True): If True, don't send actual notifications

**Returns**: Dict with:
- `workflow_id` (str)
- `workflow_type` (str)
- `patient_id` (str)
- `completed_at` (str)
- `result` (str)
- `status` (str)
- `hipaa_compliant` (bool)

---

## Security & Compliance API

### Module: `gen_z_agent.healthcare_security`

#### `HIPAAAuditLogger`

HIPAA-compliant audit logging.

```python
from gen_z_agent.healthcare_security import audit_logger

audit_logger.log_phi_access(
    user_id="dr_smith",
    patient_id="PAT12345",
    action="read",
    resource_type="Patient",
    resource_id="patient_001",
    ip_address="10.0.0.1",
    success=True,
    reason="Clinical review"
)
```

---

#### `PHIEncryption`

AES-256-GCM encryption for PHI.

```python
from gen_z_agent.healthcare_security import phi_encryption

# Encrypt
encrypted = phi_encryption.encrypt_phi("Patient: John Doe")

# Decrypt
decrypted = phi_encryption.decrypt_phi(encrypted)

# File encryption
encrypted_file = phi_encryption.encrypt_file(
    file_path=Path("patient_data.json"),
    output_path=Path("patient_data.json.encrypted")
)
```

---

#### `PHIDeidentifier`

De-identify PHI according to HIPAA Safe Harbor.

```python
from gen_z_agent.healthcare_security import phi_deidentifier

# De-identify text
deidentified = phi_deidentifier.deidentify_text(
    "Contact patient at 555-123-4567 or john.doe@email.com"
)
# Returns: "Contact patient at [PHONE] or [EMAIL]"

# De-identify patient data
deidentified_patient = phi_deidentifier.deidentify_patient_data(patient_fhir)

# Generate synthetic ID
synthetic_id = phi_deidentifier.generate_synthetic_id(
    real_id="PAT12345",
    id_type="PATIENT"
)
```

---

#### `AccessControl`

Role-based access control.

```python
from gen_z_agent.healthcare_security import access_control

can_access = access_control.check_access(
    user_id="dr_smith",
    user_role="physician",
    action="read",
    resource_type="Patient",
    patient_id="PAT123"
)
```

**Roles**: physician, nurse, pharmacist, researcher, admin

---

## Data Models

### RAG Models

#### `ClinicalDocument`

```python
from gen_z_agent.clinical_rag import ClinicalDocument

doc = ClinicalDocument(
    id="guideline_001",
    content="Clinical guideline text...",
    metadata={
        "source": "ACC/AHA",
        "clinical_domain": "cardiology",
        "last_updated": "2023-01-01"
    }
)
```

#### `ClinicalContext`

```python
@dataclass
class ClinicalContext:
    query: str
    retrieved_docs: List[RetrievalResult]
    context_text: str
    sources: List[str]
    confidence: float
```

---

### Guardrails Models

#### `GuardrailResult`

```python
@dataclass
class GuardrailResult:
    passed: bool
    violations: List[GuardrailViolation]
    confidence: float = 1.0
    recommendations: List[str] = field(default_factory=list)

    def has_critical_violations(self) -> bool
    def has_errors(self) -> bool
    def get_highest_severity(self) -> Optional[GuardrailSeverity]
```

#### `GuardrailViolation`

```python
@dataclass
class GuardrailViolation:
    guardrail_name: str
    guardrail_type: GuardrailType
    severity: GuardrailSeverity
    message: str
    details: Dict[str, Any]
    timestamp: datetime
```

**Severity Levels**: INFO, WARNING, ERROR, CRITICAL

---

### HITL Models

#### `ClinicalReview`

```python
@dataclass
class ClinicalReview:
    review_id: str
    report_id: str
    patient_id: str
    clinician_id: str
    ai_output: str
    ai_confidence: float
    ai_risk_category: str
    agreement_level: AgreementLevel
    review_status: ReviewStatus
    feedback: Optional[str]
    corrected_output: Optional[str]
    accuracy_rating: Optional[int]
    completeness_rating: Optional[int]
    clinical_utility_rating: Optional[int]
    # ... timestamps and flags
```

#### `ReviewMetrics`

```python
@dataclass
class ReviewMetrics:
    total_reviews: int
    pending_reviews: int
    completed_reviews: int
    fully_agree_rate: float
    partially_agree_rate: float
    disagree_rate: float
    avg_accuracy_rating: float
    avg_completeness_rating: float
    avg_clinical_utility_rating: float
    agreement_by_risk: Dict[str, float]
    common_disagreements: List[Dict[str, Any]]
```

---

### Healthcare Models

See `gen_z_agent.healthcare_models` for FHIR resource models:
- `Patient`
- `Observation`
- `MedicationStatement`
- `Condition`
- `Encounter`
- `PatientRiskAssessment`
- `MedicationReview`
- `ClinicalAlert`

---

## Error Handling

### Common Exceptions

```python
# RAG errors
from gen_z_agent.clinical_rag import ClinicalRAGError

try:
    context = rag.get_clinical_context(query)
except ClinicalRAGError as e:
    logger.error(f"RAG retrieval failed: {e}")
    # Fallback to direct LLM without RAG

# Guardrails errors
from gen_z_agent.clinical_guardrails import GuardrailsError

try:
    passed, violations = engine.validate_clinical_workflow(...)
except GuardrailsError as e:
    logger.error(f"Guardrails check failed: {e}")
    # Escalate to human review

# HITL errors
from gen_z_agent.clinical_hitl import ReviewError

try:
    review = manager.create_review(...)
except ReviewError as e:
    logger.error(f"Review creation failed: {e}")
```

---

## Examples

### Example 1: Complete Clinical Workflow with RAG + Guardrails + HITL

```python
from gen_z_agent.clinical_rag import get_rag_system
from gen_z_agent.clinical_guardrails import GuardrailsEngine
from gen_z_agent.clinical_hitl import review_manager, ReviewPriority
from gen_z_agent.healthcare_agents import run_clinical_workflow

# Initialize systems
rag = get_rag_system()
guardrails = GuardrailsEngine()

# Get clinical context
context = rag.get_clinical_context(
    "How to manage hypertension in diabetic patient?",
    k=3
)

# Run workflow
result = run_clinical_workflow(
    workflow_type="patient_risk_assessment",
    patient_id="PAT12345",
    data_source="./fhir_data"
)

# Validate with guardrails
passed, violations = guardrails.validate_clinical_workflow(
    input_data=patient_fhir_data,
    prompt=agent_prompt,
    output=result['result']
)

# Create review if needed
if not passed or result.get('confidence', 1.0) < 0.9:
    review = review_manager.create_review(
        report_id=result['workflow_id'],
        patient_id="PAT12345",
        ai_output=result['result'],
        ai_confidence=result.get('confidence', 0.0),
        ai_risk_category="HIGH",
        priority=ReviewPriority.HIGH if not passed else ReviewPriority.MEDIUM
    )
    print(f"Created review: {review.review_id}")
```

---

### Example 2: Adding Custom Clinical Guidelines

```python
from gen_z_agent.clinical_rag import get_rag_system, ClinicalDocument

rag = get_rag_system()

# Add custom guideline
custom_guideline = ClinicalDocument(
    id="custom_001",
    content="""
    Hospital Protocol for Sepsis Management:

    1. Recognize sepsis within 1 hour
    2. Obtain blood cultures before antibiotics
    3. Administer broad-spectrum antibiotics within 1 hour
    4. Begin fluid resuscitation (30 mL/kg crystalloid)
    5. Measure lactate and remeasure if >2 mmol/L
    6. Apply vasopressors if hypotensive during or after fluid resuscitation
    """,
    metadata={
        "source": "Hospital Internal Protocol",
        "clinical_domain": "emergency_medicine",
        "last_updated": "2024-01-01",
        "version": "3.0"
    }
)

rag.add_documents([custom_guideline])
```

---

### Example 3: Custom Guardrail

```python
from gen_z_agent.clinical_guardrails import (
    GuardrailResult, GuardrailViolation,
    GuardrailType, GuardrailSeverity
)

def validate_custom_rule(data: Dict) -> GuardrailResult:
    """Custom guardrail: Check lab result is recent"""
    violations = []

    if "effectiveDateTime" in data:
        result_date = datetime.fromisoformat(data["effectiveDateTime"])
        days_old = (datetime.now() - result_date).days

        if days_old > 30:
            violations.append(GuardrailViolation(
                guardrail_name="lab_result_freshness",
                guardrail_type=GuardrailType.CLINICAL_SAFETY,
                severity=GuardrailSeverity.WARNING,
                message=f"Lab result is {days_old} days old",
                details={"days_old": days_old, "threshold": 30}
            ))

    return GuardrailResult(
        passed=len(violations) == 0,
        violations=violations
    )
```

---

### Example 4: Batch Review Processing

```python
from gen_z_agent.clinical_hitl import review_manager, AgreementLevel

# Get pending reviews
pending = review_manager.get_pending_reviews()

for review in pending:
    # Assign to clinician
    review_manager.assign_review(review.review_id, "clinician_123")

    # Simulate clinician review (in production, this would be via UI)
    review_manager.submit_review(
        review_id=review.review_id,
        clinician_id="clinician_123",
        agreement_level=AgreementLevel.FULLY_AGREE,
        accuracy_rating=5,
        completeness_rating=5,
        clinical_utility_rating=5
    )

# Get metrics
metrics = review_manager.get_metrics()
print(f"Processed {len(pending)} reviews")
print(f"Agreement rate: {metrics.fully_agree_rate:.1%}")
```

---

## Performance Tips

### 1. RAG Optimization

```python
# Cache frequently accessed contexts
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_context(query: str) -> ClinicalContext:
    return rag.get_clinical_context(query, k=3)
```

### 2. Batch Guardrail Checks

```python
# Process multiple outputs in parallel
from concurrent.futures import ThreadPoolExecutor

def validate_batch(outputs: List[str]) -> List[GuardrailResult]:
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(
            guardrails.output_guardrails.detect_hallucination,
            outputs
        ))
    return results
```

### 3. Async HITL Operations

```python
import asyncio

async def create_reviews_async(reports: List[Dict]):
    tasks = [
        asyncio.to_thread(
            review_manager.create_review,
            report_id=report['id'],
            ai_output=report['output'],
            ...
        )
        for report in reports
    ]
    return await asyncio.gather(*tasks)
```

---

## API Versioning

Current version: **v2.0.0**

Version format: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes

Check version:
```python
from gen_z_agent import __version__
print(__version__)  # "2.0.0"
```

---

## Rate Limits

No built-in rate limits, but consider:

- **Anthropic API**: 50 requests/minute (varies by tier)
- **RAG retrieval**: Unlimited (local)
- **Guardrails**: Unlimited (local)
- **HITL operations**: Unlimited (local storage)

---

## Support

- **Documentation**: `/docs/`
- **Issues**: GitHub Issues
- **Email**: support@example.com

---

**Last Updated**: 2025-11-22
**Version**: 2.0.0
