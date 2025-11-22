"""
Clinical Guardrails Framework
Multi-layer safety, accuracy, and compliance checks for LLM outputs
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

import numpy as np
from healthcare_config import HealthcareConfig
from healthcare_models import (
    Observation, MedicationStatement, RiskCategory,
    AlertSeverity, ClinicalAlert
)
from healthcare_security import audit_logger

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════

class GuardrailType(str, Enum):
    """Types of guardrails"""
    INPUT_VALIDATION = "input_validation"
    PROMPT_SAFETY = "prompt_safety"
    OUTPUT_VALIDATION = "output_validation"
    CLINICAL_SAFETY = "clinical_safety"
    COMPLIANCE = "compliance"


class GuardrailSeverity(str, Enum):
    """Severity of guardrail violations"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class GuardrailViolation:
    """Guardrail violation"""
    guardrail_name: str
    guardrail_type: GuardrailType
    severity: GuardrailSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class GuardrailResult:
    """Result of guardrail check"""
    passed: bool
    violations: List[GuardrailViolation] = field(default_factory=list)
    confidence: float = 1.0
    recommendations: List[str] = field(default_factory=list)

    def has_critical_violations(self) -> bool:
        """Check if there are critical violations"""
        return any(v.severity == GuardrailSeverity.CRITICAL for v in self.violations)

    def has_errors(self) -> bool:
        """Check if there are error-level violations"""
        return any(
            v.severity in [GuardrailSeverity.ERROR, GuardrailSeverity.CRITICAL]
            for v in self.violations
        )

    def get_highest_severity(self) -> Optional[GuardrailSeverity]:
        """Get highest severity level"""
        if not self.violations:
            return None

        severity_order = [
            GuardrailSeverity.INFO,
            GuardrailSeverity.WARNING,
            GuardrailSeverity.ERROR,
            GuardrailSeverity.CRITICAL
        ]

        return max(
            (v.severity for v in self.violations),
            key=lambda s: severity_order.index(s)
        )


# ═══════════════════════════════════════════════════════════════════════════
# Input Guardrails
# ═══════════════════════════════════════════════════════════════════════════

class InputGuardrails:
    """Validate inputs before processing"""

    @staticmethod
    def validate_fhir_data(data: Dict[str, Any]) -> GuardrailResult:
        """
        Validate FHIR resource data

        Checks:
        - Required fields present
        - Valid resource types
        - Temporal consistency
        - PHI classification
        """
        violations = []

        # Check required fields
        if "resourceType" not in data:
            violations.append(GuardrailViolation(
                guardrail_name="fhir_required_fields",
                guardrail_type=GuardrailType.INPUT_VALIDATION,
                severity=GuardrailSeverity.ERROR,
                message="Missing required field: resourceType",
                details={"field": "resourceType"}
            ))

        if "id" not in data:
            violations.append(GuardrailViolation(
                guardrail_name="fhir_required_fields",
                guardrail_type=GuardrailType.INPUT_VALIDATION,
                severity=GuardrailSeverity.WARNING,
                message="Missing recommended field: id",
                details={"field": "id"}
            ))

        # Validate resource type
        resource_type = data.get("resourceType")
        if resource_type and resource_type not in HealthcareConfig.SUPPORTED_FHIR_RESOURCES:
            violations.append(GuardrailViolation(
                guardrail_name="fhir_resource_type",
                guardrail_type=GuardrailType.INPUT_VALIDATION,
                severity=GuardrailSeverity.WARNING,
                message=f"Unsupported FHIR resource type: {resource_type}",
                details={"resource_type": resource_type}
            ))

        # Check temporal consistency for Observations
        if resource_type == "Observation":
            if "effectiveDateTime" in data:
                try:
                    effective_time = datetime.fromisoformat(
                        data["effectiveDateTime"].replace("Z", "+00:00")
                    )
                    if effective_time > datetime.now():
                        violations.append(GuardrailViolation(
                            guardrail_name="temporal_consistency",
                            guardrail_type=GuardrailType.INPUT_VALIDATION,
                            severity=GuardrailSeverity.ERROR,
                            message="Observation date is in the future",
                            details={"effectiveDateTime": data["effectiveDateTime"]}
                        ))
                except:
                    violations.append(GuardrailViolation(
                        guardrail_name="date_format",
                        guardrail_type=GuardrailType.INPUT_VALIDATION,
                        severity=GuardrailSeverity.WARNING,
                        message="Invalid date format",
                        details={"effectiveDateTime": data.get("effectiveDateTime")}
                    ))

        passed = len([v for v in violations if v.severity in [
            GuardrailSeverity.ERROR, GuardrailSeverity.CRITICAL
        ]]) == 0

        return GuardrailResult(passed=passed, violations=violations)

    @staticmethod
    def validate_vital_signs(observation: Observation) -> GuardrailResult:
        """
        Validate vital sign values are physiologically plausible

        Checks against extreme values that are likely data errors
        """
        violations = []

        value = observation.get_numeric_value()
        if value is None:
            return GuardrailResult(passed=True)

        code = observation.get_observation_code()

        # Find matching vital sign config
        vital_config = None
        for vital_name, config in HealthcareConfig.VITAL_SIGNS.items():
            if config["loinc"] == code:
                vital_config = config
                break

        if not vital_config:
            return GuardrailResult(passed=True)  # Unknown vital sign

        # Check for impossible values
        critical_low = vital_config["critical_low"]
        critical_high = vital_config["critical_high"]

        # Extremely implausible values (likely errors)
        if value < critical_low * 0.5 or value > critical_high * 1.5:
            violations.append(GuardrailViolation(
                guardrail_name="vital_signs_plausibility",
                guardrail_type=GuardrailType.INPUT_VALIDATION,
                severity=GuardrailSeverity.ERROR,
                message=f"{vital_config['display']}: {value} {vital_config['unit']} is physiologically implausible",
                details={
                    "vital_sign": vital_config["display"],
                    "value": value,
                    "unit": vital_config["unit"],
                    "critical_range": (critical_low, critical_high)
                }
            ))

        # Critical values (plausible but dangerous)
        elif value < critical_low or value > critical_high:
            violations.append(GuardrailViolation(
                guardrail_name="vital_signs_critical",
                guardrail_type=GuardrailType.CLINICAL_SAFETY,
                severity=GuardrailSeverity.CRITICAL,
                message=f"{vital_config['display']}: {value} {vital_config['unit']} is in critical range",
                details={
                    "vital_sign": vital_config["display"],
                    "value": value,
                    "unit": vital_config["unit"],
                    "critical_range": (critical_low, critical_high),
                    "action_required": "immediate_clinical_review"
                }
            ))

        passed = len([v for v in violations if v.severity == GuardrailSeverity.ERROR]) == 0

        return GuardrailResult(passed=passed, violations=violations)


# ═══════════════════════════════════════════════════════════════════════════
# Prompt Guardrails
# ═══════════════════════════════════════════════════════════════════════════

class PromptGuardrails:
    """Ensure prompts are safe and effective"""

    PHI_PATTERNS = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
    ]

    @staticmethod
    def validate_prompt(prompt: str) -> GuardrailResult:
        """
        Validate prompt for safety

        Checks:
        - No PHI leakage in prompts
        - No adversarial patterns
        - Appropriate clinical framing
        """
        violations = []

        # Check for PHI patterns
        for pattern in PromptGuardrails.PHI_PATTERNS:
            if re.search(pattern, prompt):
                violations.append(GuardrailViolation(
                    guardrail_name="phi_in_prompt",
                    guardrail_type=GuardrailType.COMPLIANCE,
                    severity=GuardrailSeverity.CRITICAL,
                    message="Potential PHI detected in prompt",
                    details={"pattern": pattern}
                ))

        # Check for adversarial patterns
        adversarial_keywords = [
            "ignore previous instructions",
            "disregard",
            "forget",
            "system:",
            "admin:",
        ]

        for keyword in adversarial_keywords:
            if keyword.lower() in prompt.lower():
                violations.append(GuardrailViolation(
                    guardrail_name="adversarial_prompt",
                    guardrail_type=GuardrailType.PROMPT_SAFETY,
                    severity=GuardrailSeverity.ERROR,
                    message=f"Potential adversarial pattern detected: {keyword}",
                    details={"keyword": keyword}
                ))

        # Check prompt length (extremely long prompts can be attacks)
        if len(prompt) > 50000:
            violations.append(GuardrailViolation(
                guardrail_name="prompt_length",
                guardrail_type=GuardrailType.PROMPT_SAFETY,
                severity=GuardrailSeverity.WARNING,
                message="Prompt is unusually long",
                details={"length": len(prompt)}
            ))

        passed = len([v for v in violations if v.severity in [
            GuardrailSeverity.ERROR, GuardrailSeverity.CRITICAL
        ]]) == 0

        return GuardrailResult(passed=passed, violations=violations)


# ═══════════════════════════════════════════════════════════════════════════
# Output Guardrails
# ═══════════════════════════════════════════════════════════════════════════

class OutputGuardrails:
    """Validate LLM outputs before use"""

    @staticmethod
    def detect_hallucination(
        output: str,
        confidence_threshold: float = 0.8
    ) -> GuardrailResult:
        """
        Detect potential hallucinations

        Simple heuristics:
        - Check for uncertainty markers
        - Detect contradictions
        - Flag overly specific claims without evidence
        """
        violations = []
        confidence = 1.0

        # Uncertainty markers
        uncertainty_markers = [
            "i think", "i believe", "maybe", "possibly", "might be",
            "could be", "unclear", "uncertain", "not sure", "probably"
        ]

        uncertainty_count = sum(
            1 for marker in uncertainty_markers
            if marker in output.lower()
        )

        if uncertainty_count > 0:
            confidence -= 0.1 * uncertainty_count
            violations.append(GuardrailViolation(
                guardrail_name="uncertainty_detected",
                guardrail_type=GuardrailType.OUTPUT_VALIDATION,
                severity=GuardrailSeverity.WARNING,
                message=f"Output contains {uncertainty_count} uncertainty markers",
                details={
                    "uncertainty_count": uncertainty_count,
                    "reduced_confidence": confidence
                }
            ))

        # Check for specific numbers without context
        # (e.g., "patient has 17 medications" without basis)
        specific_numbers = re.findall(r'\b\d+\b', output)
        if len(specific_numbers) > 10:
            violations.append(GuardrailViolation(
                guardrail_name="excessive_specificity",
                guardrail_type=GuardrailType.OUTPUT_VALIDATION,
                severity=GuardrailSeverity.WARNING,
                message="Output contains many specific numbers; verify accuracy",
                details={"number_count": len(specific_numbers)}
            ))

        # Low confidence flag
        if confidence < confidence_threshold:
            violations.append(GuardrailViolation(
                guardrail_name="low_confidence",
                guardrail_type=GuardrailType.OUTPUT_VALIDATION,
                severity=GuardrailSeverity.WARNING,
                message=f"Output confidence ({confidence:.2f}) below threshold ({confidence_threshold})",
                details={"confidence": confidence, "threshold": confidence_threshold}
            ))

        passed = confidence >= confidence_threshold

        return GuardrailResult(
            passed=passed,
            violations=violations,
            confidence=confidence
        )

    @staticmethod
    def validate_medication_recommendation(
        medication_name: str,
        patient_allergies: List[str] = None,
        current_medications: List[str] = None
    ) -> GuardrailResult:
        """
        Validate medication recommendations for safety

        Checks:
        - Not in allergy list
        - No obvious contraindications
        - Not duplicating existing medication
        """
        violations = []
        recommendations = []

        if patient_allergies is None:
            patient_allergies = []
        if current_medications is None:
            current_medications = []

        # Check allergies
        for allergy in patient_allergies:
            if allergy.lower() in medication_name.lower():
                violations.append(GuardrailViolation(
                    guardrail_name="medication_allergy",
                    guardrail_type=GuardrailType.CLINICAL_SAFETY,
                    severity=GuardrailSeverity.CRITICAL,
                    message=f"CONTRAINDICATION: Patient is allergic to {allergy}",
                    details={
                        "medication": medication_name,
                        "allergy": allergy,
                        "action_required": "DO_NOT_PRESCRIBE"
                    }
                ))

        # Check for duplicates
        for current_med in current_medications:
            if current_med.lower() in medication_name.lower() or \
               medication_name.lower() in current_med.lower():
                violations.append(GuardrailViolation(
                    guardrail_name="duplicate_medication",
                    guardrail_type=GuardrailType.CLINICAL_SAFETY,
                    severity=GuardrailSeverity.ERROR,
                    message=f"Patient is already on similar medication: {current_med}",
                    details={
                        "new_medication": medication_name,
                        "existing_medication": current_med
                    }
                ))
                recommendations.append(
                    f"Review necessity of adding {medication_name} when patient "
                    f"is already on {current_med}"
                )

        # Check high-risk medications
        for high_risk_med in HealthcareConfig.HIGH_RISK_MEDICATIONS:
            if high_risk_med.lower() in medication_name.lower():
                violations.append(GuardrailViolation(
                    guardrail_name="high_risk_medication",
                    guardrail_type=GuardrailType.CLINICAL_SAFETY,
                    severity=GuardrailSeverity.WARNING,
                    message=f"{medication_name} is a high-risk medication",
                    details={
                        "medication": medication_name,
                        "risk_category": high_risk_med,
                        "action_required": "enhanced_monitoring"
                    }
                ))
                recommendations.append(
                    f"High-risk medication {medication_name} requires:\n"
                    f"  - Enhanced monitoring\n"
                    f"  - Patient education\n"
                    f"  - Regular follow-up"
                )

        passed = not any(
            v.severity == GuardrailSeverity.CRITICAL for v in violations
        )

        return GuardrailResult(
            passed=passed,
            violations=violations,
            recommendations=recommendations
        )

    @staticmethod
    def validate_risk_score(
        risk_score: float,
        risk_factors: Dict[str, Any],
        min_factors: int = 1
    ) -> GuardrailResult:
        """
        Validate risk score calculation

        Checks:
        - Score is in valid range (0-100)
        - Backed by sufficient risk factors
        - Category assignment is correct
        """
        violations = []

        # Check score range
        if not (0 <= risk_score <= 100):
            violations.append(GuardrailViolation(
                guardrail_name="risk_score_range",
                guardrail_type=GuardrailType.OUTPUT_VALIDATION,
                severity=GuardrailSeverity.ERROR,
                message=f"Risk score {risk_score} is out of valid range [0, 100]",
                details={"risk_score": risk_score}
            ))

        # Count identified risk factors
        factor_count = sum(1 for v in risk_factors.values() if v is True)

        if factor_count < min_factors and risk_score > 50:
            violations.append(GuardrailViolation(
                guardrail_name="risk_score_justification",
                guardrail_type=GuardrailType.OUTPUT_VALIDATION,
                severity=GuardrailSeverity.WARNING,
                message=f"High risk score ({risk_score}) with few risk factors ({factor_count})",
                details={
                    "risk_score": risk_score,
                    "factor_count": factor_count,
                    "risk_factors": risk_factors
                }
            ))

        # Validate category assignment
        category = HealthcareConfig.calculate_risk_category(risk_score)

        if risk_score >= 90 and category != "HIGH":
            violations.append(GuardrailViolation(
                guardrail_name="risk_category_mismatch",
                guardrail_type=GuardrailType.OUTPUT_VALIDATION,
                severity=GuardrailSeverity.ERROR,
                message=f"Risk score {risk_score} should be HIGH category, got {category}",
                details={"risk_score": risk_score, "category": category}
            ))

        passed = len([v for v in violations if v.severity == GuardrailSeverity.ERROR]) == 0

        return GuardrailResult(passed=passed, violations=violations)


# ═══════════════════════════════════════════════════════════════════════════
# Compliance Guardrails
# ═══════════════════════════════════════════════════════════════════════════

class ComplianceGuardrails:
    """HIPAA and regulatory compliance checks"""

    @staticmethod
    def validate_phi_handling(
        data: Dict[str, Any],
        is_encrypted: bool = False,
        phi_classification: str = "RESTRICTED"
    ) -> GuardrailResult:
        """
        Validate PHI is handled according to HIPAA

        Checks:
        - PHI is classified
        - Encrypted if at rest
        - Audit logging enabled
        """
        violations = []

        # Check PHI classification
        if phi_classification not in ["UNRESTRICTED", "LIMITED", "RESTRICTED"]:
            violations.append(GuardrailViolation(
                guardrail_name="phi_classification",
                guardrail_type=GuardrailType.COMPLIANCE,
                severity=GuardrailSeverity.ERROR,
                message=f"Invalid PHI classification: {phi_classification}",
                details={"classification": phi_classification}
            ))

        # Check encryption for PHI
        phi_fields = ["identifier", "name", "telecom", "address", "birthDate"]
        has_phi = any(field in data for field in phi_fields)

        if has_phi and not is_encrypted:
            violations.append(GuardrailViolation(
                guardrail_name="phi_encryption",
                guardrail_type=GuardrailType.COMPLIANCE,
                severity=GuardrailSeverity.CRITICAL,
                message="PHI data is not encrypted",
                details={"has_phi": True, "is_encrypted": False}
            ))

        passed = not any(
            v.severity == GuardrailSeverity.CRITICAL for v in violations
        )

        return GuardrailResult(passed=passed, violations=violations)


# ═══════════════════════════════════════════════════════════════════════════
# Guardrails Engine
# ═══════════════════════════════════════════════════════════════════════════

class GuardrailsEngine:
    """Orchestrate all guardrail checks"""

    def __init__(self):
        self.input_guardrails = InputGuardrails()
        self.prompt_guardrails = PromptGuardrails()
        self.output_guardrails = OutputGuardrails()
        self.compliance_guardrails = ComplianceGuardrails()

    def validate_clinical_workflow(
        self,
        input_data: Dict[str, Any],
        prompt: str,
        output: str,
        patient_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[GuardrailViolation]]:
        """
        Run all guardrails for a clinical workflow

        Returns:
            (passed, violations)
        """
        all_violations = []

        # Input validation
        input_result = self.input_guardrails.validate_fhir_data(input_data)
        all_violations.extend(input_result.violations)

        # Prompt validation
        prompt_result = self.prompt_guardrails.validate_prompt(prompt)
        all_violations.extend(prompt_result.violations)

        # Output validation
        output_result = self.output_guardrails.detect_hallucination(output)
        all_violations.extend(output_result.violations)

        # Compliance checks
        compliance_result = self.compliance_guardrails.validate_phi_handling(input_data)
        all_violations.extend(compliance_result.violations)

        # Check for critical violations
        has_critical = any(
            v.severity == GuardrailSeverity.CRITICAL for v in all_violations
        )
        has_errors = any(
            v.severity in [GuardrailSeverity.ERROR, GuardrailSeverity.CRITICAL]
            for v in all_violations
        )

        passed = not has_errors

        # Audit log
        audit_logger.log_security_event(
            event_type="GUARDRAILS_CHECK",
            severity="WARNING" if has_critical else "INFO",
            description=f"Guardrails check: {len(all_violations)} violations, Passed: {passed}"
        )

        return passed, all_violations

    def format_violations_report(
        self,
        violations: List[GuardrailViolation]
    ) -> str:
        """Format violations as human-readable report"""

        if not violations:
            return "✅ All guardrails passed - No violations detected"

        # Group by severity
        by_severity = {}
        for v in violations:
            if v.severity not in by_severity:
                by_severity[v.severity] = []
            by_severity[v.severity].append(v)

        report_lines = ["=" * 80]
        report_lines.append("⚠️  GUARDRAILS VIOLATIONS REPORT")
        report_lines.append("=" * 80)

        severity_order = [
            GuardrailSeverity.CRITICAL,
            GuardrailSeverity.ERROR,
            GuardrailSeverity.WARNING,
            GuardrailSeverity.INFO
        ]

        for severity in severity_order:
            if severity not in by_severity:
                continue

            report_lines.append(f"\n{severity.value.upper()} ({len(by_severity[severity])} violations):")
            report_lines.append("-" * 80)

            for violation in by_severity[severity]:
                report_lines.append(f"\n  • {violation.guardrail_name}")
                report_lines.append(f"    Type: {violation.guardrail_type.value}")
                report_lines.append(f"    Message: {violation.message}")
                if violation.details:
                    report_lines.append(f"    Details: {json.dumps(violation.details, indent=6)}")

        report_lines.append("\n" + "=" * 80)

        return "\n".join(report_lines)


# ═══════════════════════════════════════════════════════════════════════════
# Global instance
# ═══════════════════════════════════════════════════════════════════════════

guardrails_engine = GuardrailsEngine()


if __name__ == "__main__":
    print("=" * 80)
    print("Clinical Guardrails Framework Demo")
    print("=" * 80)

    # Test input validation
    print("\n1. Testing FHIR Data Validation")
    print("-" * 80)

    test_observation = {
        "resourceType": "Observation",
        "id": "obs-001",
        "status": "final",
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": "8480-6",
                "display": "Systolic Blood Pressure"
            }]
        },
        "valueQuantity": {
            "value": 220,
            "unit": "mm[Hg]"
        },
        "effectiveDateTime": "2024-01-15T10:30:00Z"
    }

    result = InputGuardrails.validate_fhir_data(test_observation)
    print(f"Passed: {result.passed}")
    print(f"Violations: {len(result.violations)}")

    # Test medication safety
    print("\n2. Testing Medication Safety")
    print("-" * 80)

    result = OutputGuardrails.validate_medication_recommendation(
        medication_name="penicillin",
        patient_allergies=["penicillin"],
        current_medications=["amoxicillin"]
    )

    print(guardrails_engine.format_violations_report(result.violations))

    # Test risk score validation
    print("\n3. Testing Risk Score Validation")
    print("-" * 80)

    result = OutputGuardrails.validate_risk_score(
        risk_score=95,
        risk_factors={"critical_vitals": True},
        min_factors=2
    )

    print(f"Passed: {result.passed}")
    print(f"Violations: {len(result.violations)}")

    print("\n" + "=" * 80)
    print("✅ Clinical Guardrails Framework operational")
    print("=" * 80)
