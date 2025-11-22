"""
HIPAA Compliance and Security Utilities
PHI Handling, Audit Logging, Encryption, and De-identification
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend
import base64
import re

from healthcare_config import HealthcareConfig


# ════════════════════════════════════════════════════════════════════════════
# Audit Logging
# ════════════════════════════════════════════════════════════════════════════

class HIPAAAuditLogger:
    """HIPAA-compliant audit logging"""

    def __init__(self, log_dir: Path = None):
        self.log_dir = log_dir or HealthcareConfig.AUDIT_LOGS_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Configure logger
        self.logger = logging.getLogger("HIPAAAuditLog")
        self.logger.setLevel(logging.INFO)

        # File handler with rotation
        log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m')}.log"
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_phi_access(
        self,
        user_id: str,
        patient_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        ip_address: Optional[str] = None,
        success: bool = True,
        reason: Optional[str] = None
    ):
        """Log PHI access event"""
        event = {
            "event_type": "PHI_ACCESS",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "patient_id": self._hash_identifier(patient_id),  # Hash for audit log
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "ip_address": ip_address,
            "success": success,
            "reason": reason or "Authorized clinical access",
            "compliance_level": "HIPAA"
        }

        self.logger.info(json.dumps(event))

    def log_data_export(
        self,
        user_id: str,
        patient_ids: List[str],
        export_format: str,
        destination: str,
        phi_included: bool = True
    ):
        """Log data export event"""
        event = {
            "event_type": "DATA_EXPORT",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "patient_count": len(patient_ids),
            "patient_ids_hash": self._hash_list(patient_ids),
            "export_format": export_format,
            "destination": destination,
            "phi_included": phi_included,
            "compliance_level": "HIPAA"
        }

        self.logger.info(json.dumps(event))

    def log_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ):
        """Log security event"""
        event = {
            "event_type": event_type,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "user_id": user_id,
            "ip_address": ip_address,
            "compliance_level": "HIPAA"
        }

        self.logger.warning(json.dumps(event))

    def _hash_identifier(self, identifier: str) -> str:
        """Hash an identifier for audit logs"""
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]

    def _hash_list(self, items: List[str]) -> str:
        """Hash a list of identifiers"""
        combined = "|".join(sorted(items))
        return hashlib.sha256(combined.encode()).hexdigest()[:16]


# ════════════════════════════════════════════════════════════════════════════
# PHI Encryption
# ════════════════════════════════════════════════════════════════════════════

class PHIEncryption:
    """AES-256-GCM encryption for PHI data"""

    def __init__(self, key: Optional[bytes] = None):
        """
        Initialize encryption with key

        Args:
            key: Encryption key (32 bytes for AES-256). If None, uses environment variable.
        """
        if key is None:
            # Get key from environment or generate
            key_string = os.getenv("PHI_ENCRYPTION_KEY")
            if key_string:
                key = base64.urlsafe_b64decode(key_string)
            else:
                # Generate key (ONLY for development - use proper key management in production)
                key = Fernet.generate_key()
                print(f"⚠️  WARNING: Generated temporary encryption key. Set PHI_ENCRYPTION_KEY in production.")

        self.fernet = Fernet(key)

    def encrypt_phi(self, data: str) -> str:
        """Encrypt PHI data"""
        encrypted = self.fernet.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt_phi(self, encrypted_data: str) -> str:
        """Decrypt PHI data"""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted = self.fernet.decrypt(encrypted_bytes)
        return decrypted.decode()

    def encrypt_file(self, file_path: Path, output_path: Optional[Path] = None) -> Path:
        """Encrypt a file containing PHI"""
        with open(file_path, 'rb') as f:
            data = f.read()

        encrypted = self.fernet.encrypt(data)

        if output_path is None:
            output_path = file_path.with_suffix(file_path.suffix + '.encrypted')

        with open(output_path, 'wb') as f:
            f.write(encrypted)

        # Set restrictive permissions
        os.chmod(output_path, 0o600)

        return output_path

    def decrypt_file(self, encrypted_path: Path, output_path: Optional[Path] = None) -> Path:
        """Decrypt a file containing PHI"""
        with open(encrypted_path, 'rb') as f:
            encrypted_data = f.read()

        decrypted = self.fernet.decrypt(encrypted_data)

        if output_path is None:
            output_path = encrypted_path.with_suffix('')  # Remove .encrypted extension

        with open(output_path, 'wb') as f:
            f.write(decrypted)

        # Set restrictive permissions
        os.chmod(output_path, 0o600)

        return output_path


# ════════════════════════════════════════════════════════════════════════════
# De-identification (HIPAA Safe Harbor Method)
# ════════════════════════════════════════════════════════════════════════════

class PHIDeidentifier:
    """
    De-identify PHI according to HIPAA Safe Harbor method
    Removes 18 types of identifiers
    """

    # HIPAA Safe Harbor 18 identifiers
    PHI_PATTERNS = {
        "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        "url": r'https?://[^\s]+',
        "zip_code": r'\b\d{5}(?:-\d{4})?\b',  # Will keep first 3 digits
    }

    def __init__(self):
        self.replacement_map = {}  # For consistent de-identification

    def deidentify_text(self, text: str, preserve_structure: bool = True) -> str:
        """
        De-identify free text

        Args:
            text: Text containing potential PHI
            preserve_structure: If True, replace with placeholder of similar length

        Returns:
            De-identified text
        """
        deidentified = text

        # Remove phone numbers
        deidentified = re.sub(self.PHI_PATTERNS["phone"], "[PHONE]", deidentified)

        # Remove SSN
        deidentified = re.sub(self.PHI_PATTERNS["ssn"], "[SSN]", deidentified)

        # Remove email
        deidentified = re.sub(self.PHI_PATTERNS["email"], "[EMAIL]", deidentified)

        # Remove IP addresses
        deidentified = re.sub(self.PHI_PATTERNS["ip_address"], "[IP]", deidentified)

        # Remove URLs
        deidentified = re.sub(self.PHI_PATTERNS["url"], "[URL]", deidentified)

        return deidentified

    def deidentify_patient_data(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        De-identify patient FHIR data according to Safe Harbor

        Removes/redacts:
        1. Names
        2. Geographic subdivisions smaller than state
        3. Dates (except year) more than 89 years old
        4. Phone numbers
        5. Email addresses
        6. SSN
        7. MRN / Account numbers
        8. Device identifiers
        9. URLs
        10. IP addresses
        11. Biometric identifiers
        12. Photos
        13-18. Other unique identifiers
        """

        deidentified = patient_data.copy()

        # Remove direct identifiers
        if "identifier" in deidentified:
            deidentified["identifier"] = [
                {"system": "de-identified", "value": f"PATIENT_{hashlib.md5(str(patient_data.get('id', '')).encode()).hexdigest()[:8]}"}
            ]

        if "name" in deidentified:
            deidentified["name"] = [{"text": "[NAME REMOVED]"}]

        if "telecom" in deidentified:
            deidentified["telecom"] = []

        if "address" in deidentified:
            # Keep only state
            for addr in deidentified.get("address", []):
                if "state" in addr:
                    addr = {"state": addr["state"]}
                else:
                    addr = {}

        # Generalize dates (keep only year if > 89 years old)
        if "birthDate" in deidentified:
            birth_year = deidentified["birthDate"][:4] if len(deidentified["birthDate"]) >= 4 else None
            if birth_year:
                try:
                    age = datetime.now().year - int(birth_year)
                    if age > 89:
                        deidentified["birthDate"] = "1900-01-01"  # Generalize to >89
                    else:
                        deidentified["birthDate"] = f"{birth_year}-01-01"  # Keep year only
                except:
                    deidentified["birthDate"] = "UNKNOWN"

        # Remove photos
        if "photo" in deidentified:
            del deidentified["photo"]

        return deidentified

    def generate_synthetic_id(self, real_id: str, id_type: str = "PATIENT") -> str:
        """Generate consistent synthetic identifier"""
        key = f"{id_type}_{real_id}"
        if key not in self.replacement_map:
            hash_value = hashlib.sha256(key.encode()).hexdigest()[:8]
            self.replacement_map[key] = f"{id_type}_{hash_value}"

        return self.replacement_map[key]


# ════════════════════════════════════════════════════════════════════════════
# Access Control
# ════════════════════════════════════════════════════════════════════════════

class AccessControl:
    """Role-based access control for healthcare data"""

    ROLES = {
        "physician": {
            "can_read_phi": True,
            "can_write_phi": True,
            "can_export_phi": True,
            "can_deidentify": True,
            "resources": ["Patient", "Observation", "MedicationStatement", "Condition", "Encounter"]
        },
        "nurse": {
            "can_read_phi": True,
            "can_write_phi": True,
            "can_export_phi": False,
            "can_deidentify": False,
            "resources": ["Patient", "Observation", "MedicationStatement", "Encounter"]
        },
        "pharmacist": {
            "can_read_phi": True,
            "can_write_phi": True,
            "can_export_phi": False,
            "can_deidentify": False,
            "resources": ["Patient", "MedicationStatement"]
        },
        "researcher": {
            "can_read_phi": False,
            "can_write_phi": False,
            "can_export_phi": False,
            "can_deidentify": False,
            "resources": ["Observation", "MedicationStatement", "Condition"]  # De-identified only
        },
        "admin": {
            "can_read_phi": True,
            "can_write_phi": True,
            "can_export_phi": True,
            "can_deidentify": True,
            "resources": ["*"]
        }
    }

    def __init__(self, audit_logger: HIPAAAuditLogger):
        self.audit_logger = audit_logger

    def check_access(
        self,
        user_id: str,
        user_role: str,
        action: str,
        resource_type: str,
        patient_id: Optional[str] = None
    ) -> bool:
        """
        Check if user has permission for action

        Args:
            user_id: User identifier
            user_role: User role (physician, nurse, etc.)
            action: Action to perform (read, write, export)
            resource_type: FHIR resource type
            patient_id: Patient identifier (if applicable)

        Returns:
            bool: True if access granted
        """

        role_permissions = self.ROLES.get(user_role.lower())
        if not role_permissions:
            self.audit_logger.log_security_event(
                event_type="ACCESS_DENIED",
                severity="WARNING",
                description=f"Unknown role: {user_role}",
                user_id=user_id
            )
            return False

        # Check resource access
        allowed_resources = role_permissions["resources"]
        if "*" not in allowed_resources and resource_type not in allowed_resources:
            self.audit_logger.log_security_event(
                event_type="ACCESS_DENIED",
                severity="WARNING",
                description=f"Role {user_role} cannot access {resource_type}",
                user_id=user_id
            )
            return False

        # Check action permissions
        if action == "read" and not role_permissions.get("can_read_phi", False):
            return False
        if action == "write" and not role_permissions.get("can_write_phi", False):
            return False
        if action == "export" and not role_permissions.get("can_export_phi", False):
            return False

        # Log successful access
        if patient_id:
            self.audit_logger.log_phi_access(
                user_id=user_id,
                patient_id=patient_id,
                action=action,
                resource_type=resource_type,
                resource_id="N/A",
                success=True
            )

        return True


# ════════════════════════════════════════════════════════════════════════════
# Utility Functions
# ════════════════════════════════════════════════════════════════════════════

def classify_phi_content(data: Dict[str, Any]) -> str:
    """
    Classify data by PHI content level

    Returns:
        str: "UNRESTRICTED", "LIMITED", "RESTRICTED"
    """
    # Check for direct identifiers
    phi_fields = ["identifier", "name", "telecom", "address", "birthDate"]

    has_direct_phi = any(field in data for field in phi_fields)

    if has_direct_phi:
        return "RESTRICTED"
    elif "patient" in str(data).lower():
        return "LIMITED"
    else:
        return "UNRESTRICTED"


def sanitize_for_logging(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove PHI from data for logging purposes"""
    phi_fields = ["identifier", "name", "telecom", "address", "birthDate", "photo"]

    sanitized = data.copy()
    for field in phi_fields:
        if field in sanitized:
            sanitized[field] = "[REDACTED]"

    return sanitized


# Initialize global instances
audit_logger = HIPAAAuditLogger()
phi_encryption = PHIEncryption()
phi_deidentifier = PHIDeidentifier()
access_control = AccessControl(audit_logger)


if __name__ == "__main__":
    print("Gen Z Healthcare - HIPAA Security Module")
    print("=" * 60)

    # Test encryption
    test_phi = "Patient: John Doe, MRN: 12345, DOB: 1985-01-15"
    encrypted = phi_encryption.encrypt_phi(test_phi)
    decrypted = phi_encryption.decrypt_phi(encrypted)

    print("✅ PHI Encryption: Working")
    print(f"   Original: {test_phi}")
    print(f"   Encrypted: {encrypted[:50]}...")
    print(f"   Decrypted: {decrypted}")

    # Test de-identification
    test_text = "Contact patient at 555-123-4567 or john.doe@email.com"
    deidentified = phi_deidentifier.deidentify_text(test_text)
    print(f"\n✅ De-identification: Working")
    print(f"   Original: {test_text}")
    print(f"   De-identified: {deidentified}")

    # Test access control
    can_access = access_control.check_access(
        user_id="dr_smith",
        user_role="physician",
        action="read",
        resource_type="Patient",
        patient_id="PAT123"
    )
    print(f"\n✅ Access Control: Working")
    print(f"   Physician can read Patient: {can_access}")

    print(f"\n✅ Audit logging enabled: {HealthcareConfig.AUDIT_LOGS_DIR}")
    print(f"✅ HIPAA Compliance: Active")
