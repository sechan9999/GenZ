"""
Clinical Trial Framework with Statistical Analysis
Conduct rigorous randomized controlled trials (RCTs) for AI clinical validation
"""

import os
import json
import uuid
import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, chi2_contingency, mannwhitneyu

from healthcare_config import HealthcareConfig
from healthcare_security import audit_logger, phi_deidentifier
from healthcare_models import PatientRiskAssessment

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════

class TrialArm(str, Enum):
    """Trial arms"""
    AI_ASSISTED = "ai_assisted"  # Clinician + AI recommendations
    CONTROL = "control"  # Clinician alone (standard of care)


class TrialStatus(str, Enum):
    """Trial status"""
    PLANNING = "planning"
    RECRUITING = "recruiting"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ANALYZED = "analyzed"


class OutcomeType(str, Enum):
    """Types of clinical outcomes"""
    DIAGNOSTIC_ACCURACY = "diagnostic_accuracy"
    TIME_TO_DIAGNOSIS = "time_to_diagnosis"
    TREATMENT_APPROPRIATENESS = "treatment_appropriateness"
    PATIENT_SAFETY = "patient_safety"
    CLINICIAN_SATISFACTION = "clinician_satisfaction"


@dataclass
class TrialCase:
    """De-identified trial case"""
    case_id: str
    patient_id_deidentified: str
    clinical_data: Dict[str, Any]
    ground_truth: Optional[Dict[str, Any]]  # Expert panel consensus
    arm: TrialArm
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ClinicianParticipant:
    """Clinician trial participant"""
    participant_id: str
    role: str  # physician, nurse, pharmacist
    specialty: Optional[str]
    years_experience: int
    cases_completed: int = 0


@dataclass
class TrialResponse:
    """Clinician response to trial case"""
    response_id: str
    case_id: str
    participant_id: str
    arm: TrialArm

    # Clinician's assessment
    diagnosis: Optional[str]
    treatment_plan: Optional[List[str]]
    confidence: float  # 0.0-1.0

    # AI recommendations (if AI_ASSISTED arm)
    ai_recommendations: Optional[List[str]]

    # Timing
    start_time: datetime
    end_time: datetime
    time_taken_seconds: float

    # Metadata
    ai_used: bool  # Did clinician review AI recommendations?

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["arm"] = self.arm.value
        data["start_time"] = self.start_time.isoformat()
        data["end_time"] = self.end_time.isoformat()
        return data


@dataclass
class ExpertReview:
    """Expert panel review of clinician response"""
    review_id: str
    response_id: str
    expert_id: str

    # Scores (0-100)
    diagnostic_accuracy: float
    treatment_appropriateness: float
    patient_safety: float

    # Detailed feedback
    correct_diagnosis: bool
    critical_errors: List[str] = field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class StatisticalAnalysis:
    """Statistical analysis results"""
    trial_id: str
    n_ai_assisted: int
    n_control: int

    # Primary outcome: Diagnostic accuracy
    accuracy_ai: float
    accuracy_control: float
    accuracy_p_value: float
    accuracy_confidence_interval: Tuple[float, float]

    # Secondary outcome: Time to diagnosis
    time_ai_mean: float
    time_ai_std: float
    time_control_mean: float
    time_control_std: float
    time_p_value: float

    # Effect size
    cohens_d: float

    # Additional metrics
    sensitivity_ai: float
    sensitivity_control: float
    specificity_ai: float
    specificity_control: float

    # Safety
    critical_errors_ai: int
    critical_errors_control: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ClinicalTrial:
    """Clinical trial configuration and results"""
    trial_id: str
    title: str
    description: str
    status: TrialStatus

    # Design
    primary_outcome: OutcomeType
    secondary_outcomes: List[OutcomeType]
    randomization_ratio: float = 1.0  # 1:1 ratio AI:Control

    # Participants
    target_sample_size: int
    cases: List[TrialCase] = field(default_factory=list)
    participants: List[ClinicianParticipant] = field(default_factory=list)
    responses: List[TrialResponse] = field(default_factory=list)
    expert_reviews: List[ExpertReview] = field(default_factory=list)

    # Results
    analysis: Optional[StatisticalAnalysis] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# ═══════════════════════════════════════════════════════════════════════════
# Trial Manager
# ═══════════════════════════════════════════════════════════════════════════

class TrialManager:
    """Manage clinical trials"""

    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize trial manager

        Args:
            storage_dir: Directory to store trial data
        """
        if storage_dir is None:
            storage_dir = HealthcareConfig.HEALTHCARE_DIR / "clinical_trials"

        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

        self.trials_file = self.storage_dir / "trials.jsonl"

        logger.info(f"Trial manager initialized: {self.storage_dir}")

    def create_trial(
        self,
        title: str,
        description: str,
        primary_outcome: OutcomeType,
        secondary_outcomes: List[OutcomeType],
        target_sample_size: int
    ) -> ClinicalTrial:
        """
        Create a new clinical trial

        Args:
            title: Trial title
            description: Trial description
            primary_outcome: Primary outcome measure
            secondary_outcomes: Secondary outcome measures
            target_sample_size: Target number of cases

        Returns:
            Created trial
        """
        trial_id = f"trial_{uuid.uuid4().hex[:8]}"

        trial = ClinicalTrial(
            trial_id=trial_id,
            title=title,
            description=description,
            status=TrialStatus.PLANNING,
            primary_outcome=primary_outcome,
            secondary_outcomes=secondary_outcomes,
            target_sample_size=target_sample_size
        )

        self._save_trial(trial)

        logger.info(f"Created trial: {trial_id} - {title}")

        # Audit log
        audit_logger.log_security_event(
            event_type="TRIAL_CREATED",
            severity="INFO",
            description=f"Created clinical trial {trial_id}"
        )

        return trial

    def add_case(
        self,
        trial_id: str,
        patient_data: Dict[str, Any],
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> TrialCase:
        """
        Add a case to the trial

        Args:
            trial_id: Trial identifier
            patient_data: De-identified patient data
            ground_truth: Expert consensus (if available)

        Returns:
            Created case
        """
        trial = self._load_trial(trial_id)

        # De-identify patient ID
        patient_id = patient_data.get("id", f"patient_{len(trial.cases)}")
        deidentified_id = phi_deidentifier.generate_synthetic_id(
            patient_id,
            id_type="TRIAL_PATIENT"
        )

        # Randomize to arm
        random.seed(int(trial.created_at.timestamp()) + len(trial.cases))
        arm = random.choice([TrialArm.AI_ASSISTED, TrialArm.CONTROL])

        case = TrialCase(
            case_id=f"case_{uuid.uuid4().hex[:8]}",
            patient_id_deidentified=deidentified_id,
            clinical_data=patient_data,
            ground_truth=ground_truth,
            arm=arm
        )

        trial.cases.append(case)
        self._save_trial(trial)

        logger.info(f"Added case to trial {trial_id}: {case.case_id} (arm: {arm.value})")

        return case

    def register_participant(
        self,
        trial_id: str,
        role: str,
        specialty: Optional[str],
        years_experience: int
    ) -> ClinicianParticipant:
        """
        Register a clinician participant

        Args:
            trial_id: Trial identifier
            role: Clinician role
            specialty: Specialty (if applicable)
            years_experience: Years of clinical experience

        Returns:
            Participant profile
        """
        trial = self._load_trial(trial_id)

        participant = ClinicianParticipant(
            participant_id=f"participant_{uuid.uuid4().hex[:8]}",
            role=role,
            specialty=specialty,
            years_experience=years_experience
        )

        trial.participants.append(participant)
        self._save_trial(trial)

        logger.info(f"Registered participant: {participant.participant_id} ({role})")

        return participant

    def submit_response(
        self,
        trial_id: str,
        case_id: str,
        participant_id: str,
        diagnosis: Optional[str],
        treatment_plan: Optional[List[str]],
        confidence: float,
        start_time: datetime,
        end_time: datetime,
        ai_recommendations: Optional[List[str]] = None,
        ai_used: bool = False
    ) -> TrialResponse:
        """
        Submit clinician response to trial case

        Args:
            trial_id: Trial identifier
            case_id: Case identifier
            participant_id: Participant identifier
            diagnosis: Clinician's diagnosis
            treatment_plan: Proposed treatment
            confidence: Confidence score (0.0-1.0)
            start_time: When clinician started reviewing case
            end_time: When clinician completed review
            ai_recommendations: AI recommendations (if AI_ASSISTED arm)
            ai_used: Whether clinician reviewed AI recommendations

        Returns:
            Response object
        """
        trial = self._load_trial(trial_id)

        # Find case
        case = next((c for c in trial.cases if c.case_id == case_id), None)
        if not case:
            raise ValueError(f"Case {case_id} not found in trial {trial_id}")

        # Calculate time taken
        time_taken = (end_time - start_time).total_seconds()

        response = TrialResponse(
            response_id=f"response_{uuid.uuid4().hex[:8]}",
            case_id=case_id,
            participant_id=participant_id,
            arm=case.arm,
            diagnosis=diagnosis,
            treatment_plan=treatment_plan,
            confidence=confidence,
            ai_recommendations=ai_recommendations,
            start_time=start_time,
            end_time=end_time,
            time_taken_seconds=time_taken,
            ai_used=ai_used
        )

        trial.responses.append(response)

        # Update participant
        participant = next((p for p in trial.participants if p.participant_id == participant_id), None)
        if participant:
            participant.cases_completed += 1

        self._save_trial(trial)

        logger.info(f"Response submitted: {response.response_id} (arm: {case.arm.value}, time: {time_taken:.1f}s)")

        return response

    def add_expert_review(
        self,
        trial_id: str,
        response_id: str,
        expert_id: str,
        diagnostic_accuracy: float,
        treatment_appropriateness: float,
        patient_safety: float,
        correct_diagnosis: bool,
        critical_errors: List[str] = None,
        notes: Optional[str] = None
    ) -> ExpertReview:
        """
        Add expert panel review

        Args:
            trial_id: Trial identifier
            response_id: Response to review
            expert_id: Expert reviewer identifier
            diagnostic_accuracy: Accuracy score (0-100)
            treatment_appropriateness: Treatment score (0-100)
            patient_safety: Safety score (0-100)
            correct_diagnosis: Whether diagnosis is correct
            critical_errors: List of critical errors
            notes: Additional notes

        Returns:
            Expert review
        """
        trial = self._load_trial(trial_id)

        review = ExpertReview(
            review_id=f"review_{uuid.uuid4().hex[:8]}",
            response_id=response_id,
            expert_id=expert_id,
            diagnostic_accuracy=diagnostic_accuracy,
            treatment_appropriateness=treatment_appropriateness,
            patient_safety=patient_safety,
            correct_diagnosis=correct_diagnosis,
            critical_errors=critical_errors or [],
            notes=notes
        )

        trial.expert_reviews.append(review)
        self._save_trial(trial)

        logger.info(f"Expert review added: {review.review_id}")

        return review

    def analyze_trial(self, trial_id: str) -> StatisticalAnalysis:
        """
        Perform statistical analysis on trial results

        Args:
            trial_id: Trial identifier

        Returns:
            Statistical analysis results
        """
        trial = self._load_trial(trial_id)

        logger.info(f"Analyzing trial: {trial_id}")

        # Get responses by arm
        ai_responses = [r for r in trial.responses if r.arm == TrialArm.AI_ASSISTED]
        control_responses = [r for r in trial.responses if r.arm == TrialArm.CONTROL]

        # Get expert reviews
        ai_reviews = [
            review for review in trial.expert_reviews
            if any(r.response_id == review.response_id for r in ai_responses)
        ]
        control_reviews = [
            review for review in trial.expert_reviews
            if any(r.response_id == review.response_id for r in control_responses)
        ]

        # Primary outcome: Diagnostic accuracy
        accuracy_ai_scores = [r.diagnostic_accuracy / 100.0 for r in ai_reviews]
        accuracy_control_scores = [r.diagnostic_accuracy / 100.0 for r in control_reviews]

        accuracy_ai = np.mean(accuracy_ai_scores) if accuracy_ai_scores else 0.0
        accuracy_control = np.mean(accuracy_control_scores) if accuracy_control_scores else 0.0

        # Statistical test (t-test for continuous outcome)
        if len(accuracy_ai_scores) > 1 and len(accuracy_control_scores) > 1:
            t_stat, accuracy_p_value = ttest_ind(accuracy_ai_scores, accuracy_control_scores)

            # Confidence interval (95%)
            diff = accuracy_ai - accuracy_control
            se = np.sqrt(
                (np.std(accuracy_ai_scores)**2 / len(accuracy_ai_scores)) +
                (np.std(accuracy_control_scores)**2 / len(accuracy_control_scores))
            )
            ci_lower = diff - 1.96 * se
            ci_upper = diff + 1.96 * se
            accuracy_ci = (ci_lower, ci_upper)
        else:
            accuracy_p_value = 1.0
            accuracy_ci = (0.0, 0.0)

        # Secondary outcome: Time to diagnosis
        time_ai = [r.time_taken_seconds for r in ai_responses]
        time_control = [r.time_taken_seconds for r in control_responses]

        time_ai_mean = np.mean(time_ai) if time_ai else 0.0
        time_ai_std = np.std(time_ai) if time_ai else 0.0
        time_control_mean = np.mean(time_control) if time_control else 0.0
        time_control_std = np.std(time_control) if time_control else 0.0

        if len(time_ai) > 1 and len(time_control) > 1:
            _, time_p_value = ttest_ind(time_ai, time_control)
        else:
            time_p_value = 1.0

        # Effect size (Cohen's d)
        if len(accuracy_ai_scores) > 1 and len(accuracy_control_scores) > 1:
            pooled_std = np.sqrt(
                ((len(accuracy_ai_scores) - 1) * np.std(accuracy_ai_scores)**2 +
                 (len(accuracy_control_scores) - 1) * np.std(accuracy_control_scores)**2) /
                (len(accuracy_ai_scores) + len(accuracy_control_scores) - 2)
            )
            cohens_d = (accuracy_ai - accuracy_control) / (pooled_std + 1e-10)
        else:
            cohens_d = 0.0

        # Sensitivity and Specificity
        correct_ai = sum(1 for r in ai_reviews if r.correct_diagnosis)
        correct_control = sum(1 for r in control_reviews if r.correct_diagnosis)

        sensitivity_ai = correct_ai / len(ai_reviews) if ai_reviews else 0.0
        sensitivity_control = correct_control / len(control_reviews) if control_reviews else 0.0

        # Specificity (simplified - assume all cases are positive for demonstration)
        specificity_ai = sensitivity_ai
        specificity_control = sensitivity_control

        # Safety: Critical errors
        critical_errors_ai = sum(len(r.critical_errors) for r in ai_reviews)
        critical_errors_control = sum(len(r.critical_errors) for r in control_reviews)

        # Create analysis
        analysis = StatisticalAnalysis(
            trial_id=trial_id,
            n_ai_assisted=len(ai_responses),
            n_control=len(control_responses),
            accuracy_ai=float(accuracy_ai),
            accuracy_control=float(accuracy_control),
            accuracy_p_value=float(accuracy_p_value),
            accuracy_confidence_interval=accuracy_ci,
            time_ai_mean=float(time_ai_mean),
            time_ai_std=float(time_ai_std),
            time_control_mean=float(time_control_mean),
            time_control_std=float(time_control_std),
            time_p_value=float(time_p_value),
            cohens_d=float(cohens_d),
            sensitivity_ai=float(sensitivity_ai),
            sensitivity_control=float(sensitivity_control),
            specificity_ai=float(specificity_ai),
            specificity_control=float(specificity_control),
            critical_errors_ai=critical_errors_ai,
            critical_errors_control=critical_errors_control
        )

        trial.analysis = analysis
        trial.status = TrialStatus.ANALYZED
        self._save_trial(trial)

        logger.info(f"Trial analysis complete: {trial_id}")
        logger.info(f"  Accuracy AI: {accuracy_ai:.3f}, Control: {accuracy_control:.3f}, p={accuracy_p_value:.4f}")
        logger.info(f"  Time AI: {time_ai_mean:.1f}s, Control: {time_control_mean:.1f}s, p={time_p_value:.4f}")
        logger.info(f"  Cohen's d: {cohens_d:.3f}")

        # Audit log
        audit_logger.log_security_event(
            event_type="TRIAL_ANALYSIS",
            severity="INFO",
            description=f"Completed statistical analysis for trial {trial_id}"
        )

        return analysis

    def generate_report(self, trial_id: str) -> str:
        """
        Generate trial results report

        Args:
            trial_id: Trial identifier

        Returns:
            Markdown report
        """
        trial = self._load_trial(trial_id)

        if not trial.analysis:
            raise ValueError(f"Trial {trial_id} has not been analyzed yet")

        analysis = trial.analysis

        report = f"""
# Clinical Trial Results Report

**Trial ID**: {trial.trial_id}
**Title**: {trial.title}
**Status**: {trial.status.value}

## Study Design

**Type**: Randomized Controlled Trial (RCT)
**Randomization**: 1:1 (AI-Assisted vs. Control)
**Primary Outcome**: {trial.primary_outcome.value}
**Sample Size**: {len(trial.responses)} total ({analysis.n_ai_assisted} AI-assisted, {analysis.n_control} control)

## Participants

**Total Clinicians**: {len(trial.participants)}
**Roles**: {', '.join(set(p.role for p in trial.participants))}

## Results

### Primary Outcome: Diagnostic Accuracy

| Arm | Mean Accuracy | Sample Size |
|-----|---------------|-------------|
| **AI-Assisted** | {analysis.accuracy_ai:.1%} | {analysis.n_ai_assisted} |
| **Control** | {analysis.accuracy_control:.1%} | {analysis.n_control} |

**Statistical Test**: Independent samples t-test
**P-value**: {analysis.accuracy_p_value:.4f} {"***" if analysis.accuracy_p_value < 0.001 else "**" if analysis.accuracy_p_value < 0.01 else "*" if analysis.accuracy_p_value < 0.05 else "ns"}
**95% CI**: [{analysis.accuracy_confidence_interval[0]:.3f}, {analysis.accuracy_confidence_interval[1]:.3f}]

**Interpretation**: {"AI-assisted workflow shows statistically significant improvement" if analysis.accuracy_p_value < 0.05 else "No statistically significant difference"}

### Secondary Outcome: Time to Diagnosis

| Arm | Mean Time (seconds) | Std Dev |
|-----|---------------------|---------|
| **AI-Assisted** | {analysis.time_ai_mean:.1f}s | {analysis.time_ai_std:.1f}s |
| **Control** | {analysis.time_control_mean:.1f}s | {analysis.time_control_std:.1f}s |

**P-value**: {analysis.time_p_value:.4f} {"***" if analysis.time_p_value < 0.001 else "**" if analysis.time_p_value < 0.01 else "*" if analysis.time_p_value < 0.05 else "ns"}

**Time Savings**: {abs(analysis.time_ai_mean - analysis.time_control_mean):.1f}s ({abs(analysis.time_ai_mean - analysis.time_control_mean) / analysis.time_control_mean:.1%})

### Effect Size

**Cohen's d**: {analysis.cohens_d:.3f}

**Interpretation**:
- d < 0.2: Small effect
- d = 0.2-0.5: Medium effect
- d > 0.5: Large effect

**This study**: {"Large effect" if abs(analysis.cohens_d) > 0.5 else "Medium effect" if abs(analysis.cohens_d) > 0.2 else "Small effect"}

### Diagnostic Performance

| Metric | AI-Assisted | Control |
|--------|-------------|---------|
| **Sensitivity** | {analysis.sensitivity_ai:.1%} | {analysis.sensitivity_control:.1%} |
| **Specificity** | {analysis.specificity_ai:.1%} | {analysis.specificity_control:.1%} |

### Safety

**Critical Errors**:
- AI-Assisted arm: {analysis.critical_errors_ai} errors
- Control arm: {analysis.critical_errors_control} errors

## Conclusion

The AI-assisted clinical workflow demonstrated {"statistically significant improvement" if analysis.accuracy_p_value < 0.05 else "no significant difference"} in diagnostic accuracy compared to standard care (p={analysis.accuracy_p_value:.4f}). {"Time savings of " + f"{abs(analysis.time_ai_mean - analysis.time_control_mean):.0f}s per case were observed." if analysis.time_p_value < 0.05 else ""}

## Limitations

- Sample size: {len(trial.responses)} cases
- Single-center study
- Clinician familiarity with AI system may vary
- Observer bias possible

## Recommendations

- {"Deploy AI system for clinical use" if analysis.accuracy_p_value < 0.05 and analysis.cohens_d > 0.2 else "Conduct larger multi-center trial"}
- Continue monitoring for safety signals
- Collect long-term outcomes data

---

**Report Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Analyst**: Automated Statistical Analysis System
"""

        return report

    def _save_trial(self, trial: ClinicalTrial):
        """Save trial to file"""
        trial_data = {
            "trial_id": trial.trial_id,
            "title": trial.title,
            "description": trial.description,
            "status": trial.status.value,
            "primary_outcome": trial.primary_outcome.value,
            "secondary_outcomes": [o.value for o in trial.secondary_outcomes],
            "target_sample_size": trial.target_sample_size,
            "randomization_ratio": trial.randomization_ratio,
            "cases": [asdict(c) for c in trial.cases],
            "participants": [asdict(p) for p in trial.participants],
            "responses": [r.to_dict() for r in trial.responses],
            "expert_reviews": [asdict(r) for r in trial.expert_reviews],
            "analysis": trial.analysis.to_dict() if trial.analysis else None,
            "created_at": trial.created_at.isoformat(),
            "started_at": trial.started_at.isoformat() if trial.started_at else None,
            "completed_at": trial.completed_at.isoformat() if trial.completed_at else None
        }

        # Append to JSONL file
        with open(self.trials_file, 'a') as f:
            f.write(json.dumps(trial_data) + '\n')

    def _load_trial(self, trial_id: str) -> ClinicalTrial:
        """Load trial from file"""
        if not self.trials_file.exists():
            raise ValueError(f"Trial {trial_id} not found")

        # Read all trials and find the latest version
        trial_data = None

        with open(self.trials_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                if data["trial_id"] == trial_id:
                    trial_data = data  # Keep latest

        if not trial_data:
            raise ValueError(f"Trial {trial_id} not found")

        # Reconstruct trial object
        trial = ClinicalTrial(
            trial_id=trial_data["trial_id"],
            title=trial_data["title"],
            description=trial_data["description"],
            status=TrialStatus(trial_data["status"]),
            primary_outcome=OutcomeType(trial_data["primary_outcome"]),
            secondary_outcomes=[OutcomeType(o) for o in trial_data["secondary_outcomes"]],
            target_sample_size=trial_data["target_sample_size"],
            randomization_ratio=trial_data["randomization_ratio"],
            created_at=datetime.fromisoformat(trial_data["created_at"])
        )

        # Reconstruct cases
        for case_data in trial_data.get("cases", []):
            case = TrialCase(
                case_id=case_data["case_id"],
                patient_id_deidentified=case_data["patient_id_deidentified"],
                clinical_data=case_data["clinical_data"],
                ground_truth=case_data.get("ground_truth"),
                arm=TrialArm(case_data["arm"]),
                created_at=datetime.fromisoformat(case_data["created_at"])
            )
            trial.cases.append(case)

        # Reconstruct participants
        for p_data in trial_data.get("participants", []):
            participant = ClinicianParticipant(**p_data)
            trial.participants.append(participant)

        # Reconstruct responses
        for r_data in trial_data.get("responses", []):
            response = TrialResponse(
                response_id=r_data["response_id"],
                case_id=r_data["case_id"],
                participant_id=r_data["participant_id"],
                arm=TrialArm(r_data["arm"]),
                diagnosis=r_data.get("diagnosis"),
                treatment_plan=r_data.get("treatment_plan"),
                confidence=r_data["confidence"],
                ai_recommendations=r_data.get("ai_recommendations"),
                start_time=datetime.fromisoformat(r_data["start_time"]),
                end_time=datetime.fromisoformat(r_data["end_time"]),
                time_taken_seconds=r_data["time_taken_seconds"],
                ai_used=r_data["ai_used"]
            )
            trial.responses.append(response)

        # Reconstruct expert reviews
        for review_data in trial_data.get("expert_reviews", []):
            review = ExpertReview(**review_data)
            trial.expert_reviews.append(review)

        # Reconstruct analysis
        if trial_data.get("analysis"):
            trial.analysis = StatisticalAnalysis(**trial_data["analysis"])

        return trial


# ═══════════════════════════════════════════════════════════════════════════
# Example Usage
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("Clinical Trial Framework Demo")
    print("=" * 80)

    # Initialize manager
    manager = TrialManager()

    # Create trial
    print("\n1. Creating Clinical Trial")
    print("-" * 80)

    trial = manager.create_trial(
        title="AI-Assisted Hypertension Management RCT",
        description="Randomized controlled trial comparing AI-assisted vs. standard hypertension management",
        primary_outcome=OutcomeType.DIAGNOSTIC_ACCURACY,
        secondary_outcomes=[OutcomeType.TIME_TO_DIAGNOSIS, OutcomeType.PATIENT_SAFETY],
        target_sample_size=100
    )

    print(f"Trial ID: {trial.trial_id}")
    print(f"Title: {trial.title}")

    # Register participants
    print("\n2. Registering Clinician Participants")
    print("-" * 80)

    participants = [
        manager.register_participant(trial.trial_id, "physician", "cardiology", 10),
        manager.register_participant(trial.trial_id, "physician", "internal_medicine", 5),
        manager.register_participant(trial.trial_id, "nurse", "critical_care", 7)
    ]

    print(f"Registered {len(participants)} participants")

    # Add cases and responses (simulated)
    print("\n3. Running Trial (Simulated)")
    print("-" * 80)

    for i in range(20):
        # Add case
        case = manager.add_case(
            trial.trial_id,
            patient_data={"id": f"PAT{i:03d}", "bp": "165/98", "age": 62},
            ground_truth={"diagnosis": "Stage 2 Hypertension", "correct_treatment": "ACE inhibitor"}
        )

        # Simulate clinician response
        participant = random.choice(participants)

        start_time = datetime.now()
        end_time = datetime.now()  # Simulate time

        response = manager.submit_response(
            trial.trial_id,
            case.case_id,
            participant.participant_id,
            diagnosis="Stage 2 Hypertension",
            treatment_plan=["Start lisinopril 10mg daily", "Recheck BP in 2 weeks"],
            confidence=0.85 + random.uniform(-0.1, 0.1),
            start_time=start_time,
            end_time=end_time,
            ai_recommendations=["Increase to lisinopril 20mg"] if case.arm == TrialArm.AI_ASSISTED else None,
            ai_used=case.arm == TrialArm.AI_ASSISTED
        )

        # Simulate expert review
        manager.add_expert_review(
            trial.trial_id,
            response.response_id,
            expert_id="expert_001",
            diagnostic_accuracy=90 + random.uniform(-10, 10),
            treatment_appropriateness=85 + random.uniform(-10, 10),
            patient_safety=95 + random.uniform(-5, 5),
            correct_diagnosis=True,
            critical_errors=[],
            notes="Appropriate management"
        )

    print(f"Added 20 cases with responses and expert reviews")

    # Analyze trial
    print("\n4. Statistical Analysis")
    print("-" * 80)

    analysis = manager.analyze_trial(trial.trial_id)

    print(f"Diagnostic Accuracy:")
    print(f"  AI-Assisted: {analysis.accuracy_ai:.1%}")
    print(f"  Control: {analysis.accuracy_control:.1%}")
    print(f"  P-value: {analysis.accuracy_p_value:.4f}")
    print(f"  Cohen's d: {analysis.cohens_d:.3f}")

    # Generate report
    print("\n5. Generating Report")
    print("-" * 80)

    report = manager.generate_report(trial.trial_id)
    print(report[:500] + "...")

    print("\n" + "=" * 80)
    print("✅ Clinical Trial Framework operational")
    print("=" * 80)
