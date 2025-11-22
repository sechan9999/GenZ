"""
Human-in-the-Loop (HITL) Evaluation System
Clinician feedback collection, review management, and continuous improvement
"""

import os
import json
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

import numpy as np
from healthcare_config import HealthcareConfig
from healthcare_security import audit_logger, phi_deidentifier
from healthcare_models import RiskCategory, ClinicalReport

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════

class ReviewStatus(str, Enum):
    """Review status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ESCALATED = "escalated"


class AgreementLevel(str, Enum):
    """Clinician agreement with AI output"""
    FULLY_AGREE = "fully_agree"
    PARTIALLY_AGREE = "partially_agree"
    DISAGREE = "disagree"
    UNCERTAIN = "uncertain"


class ReviewPriority(str, Enum):
    """Priority for human review"""
    CRITICAL = "critical"  # Requires immediate review
    HIGH = "high"  # Review within 24 hours
    MEDIUM = "medium"  # Review within 1 week
    LOW = "low"  # Routine review


@dataclass
class ClinicianProfile:
    """Clinician reviewer profile"""
    id: str
    name: str
    role: str  # physician, nurse, pharmacist, etc.
    specialty: Optional[str] = None
    years_experience: Optional[int] = None
    review_count: int = 0
    average_review_time_seconds: float = 0.0


@dataclass
class ClinicalReview:
    """Clinician review of AI output"""
    review_id: str
    report_id: str
    patient_id: str  # De-identified
    clinician_id: str

    # AI output
    ai_output: str
    ai_confidence: float
    ai_risk_category: str

    # Clinician feedback
    agreement_level: AgreementLevel
    review_status: ReviewStatus
    feedback: Optional[str] = None
    corrected_output: Optional[str] = None

    # Ratings (1-5 stars)
    accuracy_rating: Optional[int] = None
    completeness_rating: Optional[int] = None
    clinical_utility_rating: Optional[int] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    reviewed_at: Optional[datetime] = None
    review_time_seconds: Optional[float] = None

    # Flags
    requires_escalation: bool = False
    escalation_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        # Convert enums to strings
        data["agreement_level"] = self.agreement_level.value if self.agreement_level else None
        data["review_status"] = self.review_status.value
        # Convert datetimes to ISO format
        data["created_at"] = self.created_at.isoformat()
        data["reviewed_at"] = self.reviewed_at.isoformat() if self.reviewed_at else None
        return data


@dataclass
class ReviewMetrics:
    """Metrics for review system"""
    total_reviews: int
    pending_reviews: int
    completed_reviews: int

    # Agreement rates
    fully_agree_rate: float
    partially_agree_rate: float
    disagree_rate: float

    # Average ratings
    avg_accuracy_rating: float
    avg_completeness_rating: float
    avg_clinical_utility_rating: float

    # By risk category
    agreement_by_risk: Dict[str, float] = field(default_factory=dict)

    # Common disagreement patterns
    common_disagreements: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TrainingExample:
    """Training example from clinician review"""
    example_id: str
    input_data: Dict[str, Any]
    expected_output: str  # Clinician-corrected output
    metadata: Dict[str, Any]
    agreement_score: float  # 0.0 (disagree) to 1.0 (fully agree)
    created_at: datetime = field(default_factory=datetime.now)


# ═══════════════════════════════════════════════════════════════════════════
# Review Manager
# ═══════════════════════════════════════════════════════════════════════════

class ReviewManager:
    """Manage clinician reviews and feedback"""

    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize review manager

        Args:
            storage_dir: Directory to store reviews
        """
        if storage_dir is None:
            storage_dir = HealthcareConfig.HEALTHCARE_DIR / "reviews"

        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

        self.reviews_file = self.storage_dir / "reviews.jsonl"
        self.clinicians_file = self.storage_dir / "clinicians.json"
        self.training_data_file = self.storage_dir / "training_data.jsonl"

        # Load clinicians
        self.clinicians: Dict[str, ClinicianProfile] = self._load_clinicians()

        logger.info(f"Review manager initialized: {self.storage_dir}")

    def _load_clinicians(self) -> Dict[str, ClinicianProfile]:
        """Load clinician profiles"""
        if not self.clinicians_file.exists():
            return {}

        with open(self.clinicians_file, 'r') as f:
            data = json.load(f)
            return {
                cid: ClinicianProfile(**cdata)
                for cid, cdata in data.items()
            }

    def _save_clinicians(self):
        """Save clinician profiles"""
        data = {
            cid: asdict(clinician)
            for cid, clinician in self.clinicians.items()
        }

        with open(self.clinicians_file, 'w') as f:
            json.dump(data, f, indent=2)

    def register_clinician(
        self,
        name: str,
        role: str,
        specialty: Optional[str] = None,
        years_experience: Optional[int] = None
    ) -> ClinicianProfile:
        """Register a new clinician"""
        clinician_id = f"clinician_{uuid.uuid4().hex[:8]}"

        clinician = ClinicianProfile(
            id=clinician_id,
            name=name,
            role=role,
            specialty=specialty,
            years_experience=years_experience
        )

        self.clinicians[clinician_id] = clinician
        self._save_clinicians()

        logger.info(f"Registered clinician: {clinician_id} ({name}, {role})")

        return clinician

    def create_review(
        self,
        report_id: str,
        patient_id: str,
        ai_output: str,
        ai_confidence: float,
        ai_risk_category: str,
        priority: ReviewPriority = ReviewPriority.MEDIUM
    ) -> ClinicalReview:
        """
        Create a review request

        Args:
            report_id: ID of the clinical report
            patient_id: De-identified patient ID
            ai_output: AI-generated output to review
            ai_confidence: AI confidence score
            ai_risk_category: Risk category assigned by AI
            priority: Review priority

        Returns:
            Created review object
        """
        review_id = f"review_{uuid.uuid4().hex}"

        # De-identify patient ID for review
        deidentified_patient_id = phi_deidentifier.generate_synthetic_id(
            patient_id,
            id_type="PATIENT"
        )

        review = ClinicalReview(
            review_id=review_id,
            report_id=report_id,
            patient_id=deidentified_patient_id,
            clinician_id="",  # Assigned when claimed
            ai_output=ai_output,
            ai_confidence=ai_confidence,
            ai_risk_category=ai_risk_category,
            agreement_level=AgreementLevel.UNCERTAIN,
            review_status=ReviewStatus.PENDING
        )

        # Save review
        self._save_review(review)

        # Audit log
        audit_logger.log_security_event(
            event_type="REVIEW_CREATED",
            severity="INFO",
            description=f"Created review {review_id} for report {report_id}"
        )

        logger.info(
            f"Created review {review_id} with priority {priority.value}"
        )

        return review

    def assign_review(
        self,
        review_id: str,
        clinician_id: str
    ) -> ClinicalReview:
        """Assign review to clinician"""
        review = self._load_review(review_id)

        if review.review_status != ReviewStatus.PENDING:
            raise ValueError(f"Review {review_id} is not pending")

        review.clinician_id = clinician_id
        review.review_status = ReviewStatus.IN_PROGRESS

        self._save_review(review)

        logger.info(f"Assigned review {review_id} to {clinician_id}")

        return review

    def submit_review(
        self,
        review_id: str,
        clinician_id: str,
        agreement_level: AgreementLevel,
        feedback: Optional[str] = None,
        corrected_output: Optional[str] = None,
        accuracy_rating: Optional[int] = None,
        completeness_rating: Optional[int] = None,
        clinical_utility_rating: Optional[int] = None,
        requires_escalation: bool = False,
        escalation_reason: Optional[str] = None
    ) -> ClinicalReview:
        """
        Submit clinician review

        Args:
            review_id: Review ID
            clinician_id: Clinician submitting review
            agreement_level: Agreement with AI output
            feedback: Free-text feedback
            corrected_output: Clinician-corrected output (if disagreed)
            accuracy_rating: 1-5 star rating for accuracy
            completeness_rating: 1-5 star rating for completeness
            clinical_utility_rating: 1-5 star rating for clinical utility
            requires_escalation: Flag for escalation
            escalation_reason: Reason for escalation

        Returns:
            Updated review object
        """
        review = self._load_review(review_id)

        if review.clinician_id != clinician_id:
            raise ValueError(f"Review {review_id} not assigned to {clinician_id}")

        # Calculate review time
        review_time = (datetime.now() - review.created_at).total_seconds()

        # Update review
        review.agreement_level = agreement_level
        review.feedback = feedback
        review.corrected_output = corrected_output
        review.accuracy_rating = accuracy_rating
        review.completeness_rating = completeness_rating
        review.clinical_utility_rating = clinical_utility_rating
        review.requires_escalation = requires_escalation
        review.escalation_reason = escalation_reason
        review.reviewed_at = datetime.now()
        review.review_time_seconds = review_time
        review.review_status = (
            ReviewStatus.ESCALATED if requires_escalation
            else ReviewStatus.COMPLETED
        )

        # Save review
        self._save_review(review)

        # Update clinician stats
        if clinician_id in self.clinicians:
            clinician = self.clinicians[clinician_id]
            clinician.review_count += 1

            # Update average review time
            if clinician.review_count == 1:
                clinician.average_review_time_seconds = review_time
            else:
                clinician.average_review_time_seconds = (
                    (clinician.average_review_time_seconds * (clinician.review_count - 1) +
                     review_time) / clinician.review_count
                )

            self._save_clinicians()

        # Create training example if there's a correction
        if corrected_output and agreement_level != AgreementLevel.FULLY_AGREE:
            self._create_training_example(review)

        # Audit log
        audit_logger.log_phi_access(
            user_id=clinician_id,
            patient_id=review.patient_id,
            action="review",
            resource_type="ClinicalReport",
            resource_id=review.report_id,
            success=True,
            reason="Clinician review completed"
        )

        logger.info(f"Review {review_id} submitted by {clinician_id}")

        return review

    def _create_training_example(self, review: ClinicalReview):
        """Create training example from review"""
        # Map agreement to score
        agreement_scores = {
            AgreementLevel.FULLY_AGREE: 1.0,
            AgreementLevel.PARTIALLY_AGREE: 0.7,
            AgreementLevel.DISAGREE: 0.0,
            AgreementLevel.UNCERTAIN: 0.5
        }

        agreement_score = agreement_scores.get(review.agreement_level, 0.5)

        example = TrainingExample(
            example_id=f"train_{uuid.uuid4().hex[:8]}",
            input_data={
                "report_id": review.report_id,
                "patient_id": review.patient_id,
                "ai_output": review.ai_output
            },
            expected_output=review.corrected_output or review.ai_output,
            metadata={
                "clinician_id": review.clinician_id,
                "agreement_level": review.agreement_level.value,
                "accuracy_rating": review.accuracy_rating,
                "completeness_rating": review.completeness_rating,
                "feedback": review.feedback
            },
            agreement_score=agreement_score
        )

        # Save to training data file
        with open(self.training_data_file, 'a') as f:
            f.write(json.dumps(asdict(example), default=str) + '\n')

        logger.info(f"Created training example from review {review.review_id}")

    def _save_review(self, review: ClinicalReview):
        """Save review to file"""
        with open(self.reviews_file, 'a') as f:
            f.write(json.dumps(review.to_dict()) + '\n')

    def _load_review(self, review_id: str) -> ClinicalReview:
        """Load review from file"""
        if not self.reviews_file.exists():
            raise ValueError(f"Review {review_id} not found")

        with open(self.reviews_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                if data["review_id"] == review_id:
                    # Convert strings back to enums
                    if data.get("agreement_level"):
                        data["agreement_level"] = AgreementLevel(data["agreement_level"])
                    data["review_status"] = ReviewStatus(data["review_status"])

                    # Convert ISO strings to datetimes
                    data["created_at"] = datetime.fromisoformat(data["created_at"])
                    if data.get("reviewed_at"):
                        data["reviewed_at"] = datetime.fromisoformat(data["reviewed_at"])

                    return ClinicalReview(**data)

        raise ValueError(f"Review {review_id} not found")

    def get_pending_reviews(
        self,
        clinician_id: Optional[str] = None,
        priority: Optional[ReviewPriority] = None
    ) -> List[ClinicalReview]:
        """Get pending reviews"""
        if not self.reviews_file.exists():
            return []

        reviews = []

        with open(self.reviews_file, 'r') as f:
            for line in f:
                data = json.loads(line)

                if data["review_status"] != ReviewStatus.PENDING.value:
                    continue

                if clinician_id and data.get("clinician_id") != clinician_id:
                    continue

                # Convert data
                if data.get("agreement_level"):
                    data["agreement_level"] = AgreementLevel(data["agreement_level"])
                data["review_status"] = ReviewStatus(data["review_status"])
                data["created_at"] = datetime.fromisoformat(data["created_at"])
                if data.get("reviewed_at"):
                    data["reviewed_at"] = datetime.fromisoformat(data["reviewed_at"])

                reviews.append(ClinicalReview(**data))

        return reviews

    def get_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> ReviewMetrics:
        """
        Get review metrics

        Args:
            start_date: Start date for metrics
            end_date: End date for metrics

        Returns:
            Review metrics
        """
        if not self.reviews_file.exists():
            return ReviewMetrics(
                total_reviews=0,
                pending_reviews=0,
                completed_reviews=0,
                fully_agree_rate=0.0,
                partially_agree_rate=0.0,
                disagree_rate=0.0,
                avg_accuracy_rating=0.0,
                avg_completeness_rating=0.0,
                avg_clinical_utility_rating=0.0
            )

        reviews = []

        with open(self.reviews_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                created_at = datetime.fromisoformat(data["created_at"])

                if start_date and created_at < start_date:
                    continue
                if end_date and created_at > end_date:
                    continue

                reviews.append(data)

        total_reviews = len(reviews)
        pending_reviews = sum(1 for r in reviews if r["review_status"] == ReviewStatus.PENDING.value)
        completed_reviews = sum(1 for r in reviews if r["review_status"] == ReviewStatus.COMPLETED.value)

        # Agreement rates
        completed = [r for r in reviews if r["review_status"] == ReviewStatus.COMPLETED.value]

        if completed:
            fully_agree = sum(1 for r in completed if r.get("agreement_level") == AgreementLevel.FULLY_AGREE.value)
            partially_agree = sum(1 for r in completed if r.get("agreement_level") == AgreementLevel.PARTIALLY_AGREE.value)
            disagree = sum(1 for r in completed if r.get("agreement_level") == AgreementLevel.DISAGREE.value)

            fully_agree_rate = fully_agree / len(completed)
            partially_agree_rate = partially_agree / len(completed)
            disagree_rate = disagree / len(completed)
        else:
            fully_agree_rate = partially_agree_rate = disagree_rate = 0.0

        # Average ratings
        accuracy_ratings = [r["accuracy_rating"] for r in completed if r.get("accuracy_rating")]
        completeness_ratings = [r["completeness_rating"] for r in completed if r.get("completeness_rating")]
        utility_ratings = [r["clinical_utility_rating"] for r in completed if r.get("clinical_utility_rating")]

        avg_accuracy_rating = np.mean(accuracy_ratings) if accuracy_ratings else 0.0
        avg_completeness_rating = np.mean(completeness_ratings) if completeness_ratings else 0.0
        avg_clinical_utility_rating = np.mean(utility_ratings) if utility_ratings else 0.0

        # Agreement by risk category
        agreement_by_risk = {}
        for risk_cat in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            cat_reviews = [r for r in completed if r.get("ai_risk_category") == risk_cat]
            if cat_reviews:
                cat_agree = sum(1 for r in cat_reviews if r.get("agreement_level") == AgreementLevel.FULLY_AGREE.value)
                agreement_by_risk[risk_cat] = cat_agree / len(cat_reviews)

        return ReviewMetrics(
            total_reviews=total_reviews,
            pending_reviews=pending_reviews,
            completed_reviews=completed_reviews,
            fully_agree_rate=fully_agree_rate,
            partially_agree_rate=partially_agree_rate,
            disagree_rate=disagree_rate,
            avg_accuracy_rating=float(avg_accuracy_rating),
            avg_completeness_rating=float(avg_completeness_rating),
            avg_clinical_utility_rating=float(avg_clinical_utility_rating),
            agreement_by_risk=agreement_by_risk
        )

    def export_training_data(
        self,
        min_agreement_score: float = 0.7,
        output_file: Optional[Path] = None
    ) -> List[TrainingExample]:
        """
        Export training examples for model fine-tuning

        Args:
            min_agreement_score: Minimum agreement score to include
            output_file: Optional file to export to

        Returns:
            List of training examples
        """
        if not self.training_data_file.exists():
            return []

        examples = []

        with open(self.training_data_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                data["created_at"] = datetime.fromisoformat(data["created_at"])

                example = TrainingExample(**data)

                if example.agreement_score >= min_agreement_score:
                    examples.append(example)

        if output_file:
            with open(output_file, 'w') as f:
                for example in examples:
                    f.write(json.dumps(asdict(example), default=str) + '\n')

            logger.info(f"Exported {len(examples)} training examples to {output_file}")

        return examples


# ═══════════════════════════════════════════════════════════════════════════
# Global instance
# ═══════════════════════════════════════════════════════════════════════════

review_manager = ReviewManager()


if __name__ == "__main__":
    print("=" * 80)
    print("Human-in-the-Loop Evaluation System Demo")
    print("=" * 80)

    # Register clinicians
    print("\n1. Registering Clinicians")
    print("-" * 80)

    clinician1 = review_manager.register_clinician(
        name="Dr. Sarah Johnson",
        role="physician",
        specialty="cardiology",
        years_experience=15
    )
    print(f"Registered: {clinician1.name} ({clinician1.id})")

    clinician2 = review_manager.register_clinician(
        name="Emily Chen, PharmD",
        role="pharmacist",
        specialty="clinical_pharmacy",
        years_experience=8
    )
    print(f"Registered: {clinician2.name} ({clinician2.id})")

    # Create review
    print("\n2. Creating Review Request")
    print("-" * 80)

    review = review_manager.create_review(
        report_id="report_001",
        patient_id="PAT12345",
        ai_output="Patient has elevated risk (score: 78/100) due to uncontrolled hypertension and polypharmacy. Recommend: 1) Adjust lisinopril dosage, 2) Schedule cardiology consult",
        ai_confidence=0.87,
        ai_risk_category="HIGH",
        priority=ReviewPriority.HIGH
    )

    print(f"Created review: {review.review_id}")
    print(f"AI Output: {review.ai_output[:100]}...")
    print(f"AI Confidence: {review.ai_confidence:.1%}")

    # Assign and submit review
    print("\n3. Submitting Clinician Review")
    print("-" * 80)

    review = review_manager.assign_review(review.review_id, clinician1.id)
    print(f"Assigned to: {clinician1.name}")

    reviewed = review_manager.submit_review(
        review_id=review.review_id,
        clinician_id=clinician1.id,
        agreement_level=AgreementLevel.PARTIALLY_AGREE,
        feedback="Generally accurate, but should also recommend medication adherence counseling",
        corrected_output="Patient has elevated risk (score: 78/100) due to uncontrolled hypertension and polypharmacy. Recommend: 1) Adjust lisinopril dosage, 2) Schedule cardiology consult, 3) Medication adherence counseling",
        accuracy_rating=4,
        completeness_rating=3,
        clinical_utility_rating=5
    )

    print(f"Review completed by: {clinician1.name}")
    print(f"Agreement: {reviewed.agreement_level.value}")
    print(f"Feedback: {reviewed.feedback}")

    # Get metrics
    print("\n4. Review Metrics")
    print("-" * 80)

    metrics = review_manager.get_metrics()
    print(f"Total Reviews: {metrics.total_reviews}")
    print(f"Completed: {metrics.completed_reviews}")
    print(f"Fully Agree Rate: {metrics.fully_agree_rate:.1%}")
    print(f"Partially Agree Rate: {metrics.partially_agree_rate:.1%}")
    print(f"Average Accuracy Rating: {metrics.avg_accuracy_rating:.1f}/5.0")

    print("\n" + "=" * 80)
    print("✅ Human-in-the-Loop System operational")
    print("=" * 80)
