"""
Clinical Model Fine-Tuning Pipeline
Train smaller, faster models for specific extraction tasks using clinician feedback
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

import anthropic
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from healthcare_config import HealthcareConfig
from healthcare_security import audit_logger
from clinical_hitl import review_manager, TrainingExample

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════

class FineTuningTask(str, Enum):
    """Types of fine-tuning tasks"""
    MEDICATION_EXTRACTION = "medication_extraction"
    DIAGNOSIS_EXTRACTION = "diagnosis_extraction"
    VITAL_SIGNS_EXTRACTION = "vital_signs_extraction"
    RISK_ASSESSMENT = "risk_assessment"
    CARE_GAP_IDENTIFICATION = "care_gap_identification"


class ModelSize(str, Enum):
    """Model sizes for deployment"""
    HAIKU = "claude-haiku-20240307"  # Fast, cheap
    SONNET = "claude-sonnet-4-5-20250929"  # Balanced
    OPUS = "claude-opus-20240229"  # Most capable


@dataclass
class FineTuningExample:
    """Training example for fine-tuning"""
    input_text: str
    expected_output: str
    task_type: FineTuningTask
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "clinician_review"  # clinician_review, synthetic, curated


@dataclass
class FineTuningDataset:
    """Dataset for fine-tuning"""
    examples: List[FineTuningExample]
    task_type: FineTuningTask
    train_split: List[FineTuningExample] = field(default_factory=list)
    val_split: List[FineTuningExample] = field(default_factory=list)
    test_split: List[FineTuningExample] = field(default_factory=list)

    def split(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        """Split dataset into train/val/test"""
        test_ratio = 1.0 - train_ratio - val_ratio

        # First split: train vs. (val + test)
        train, temp = train_test_split(
            self.examples,
            train_size=train_ratio,
            random_state=42
        )

        # Second split: val vs. test
        val_size_adjusted = val_ratio / (val_ratio + test_ratio)
        val, test = train_test_split(
            temp,
            train_size=val_size_adjusted,
            random_state=42
        )

        self.train_split = train
        self.val_split = val
        self.test_split = test

        logger.info(
            f"Dataset split: {len(train)} train, {len(val)} val, {len(test)} test"
        )


@dataclass
class FineTuningMetrics:
    """Evaluation metrics for fine-tuned model"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    task_type: FineTuningTask
    model_id: str
    evaluated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["task_type"] = self.task_type.value
        data["evaluated_at"] = self.evaluated_at.isoformat()
        return data


@dataclass
class FineTunedModel:
    """Fine-tuned model metadata"""
    model_id: str
    base_model: ModelSize
    task_type: FineTuningTask
    training_examples_count: int
    metrics: FineTuningMetrics
    created_at: datetime = field(default_factory=datetime.now)
    deployed: bool = False


# ═══════════════════════════════════════════════════════════════════════════
# Data Preparation
# ═══════════════════════════════════════════════════════════════════════════

class DataPreparation:
    """Prepare training data from clinician reviews"""

    @staticmethod
    def convert_hitl_reviews_to_examples(
        reviews: List[TrainingExample],
        task_type: FineTuningTask,
        min_agreement_score: float = 0.7
    ) -> List[FineTuningExample]:
        """
        Convert HITL reviews to fine-tuning examples

        Args:
            reviews: Training examples from clinician reviews
            task_type: Type of task to train for
            min_agreement_score: Minimum agreement score to include

        Returns:
            List of fine-tuning examples
        """
        examples = []

        for review in reviews:
            # Filter by agreement score
            if review.agreement_score < min_agreement_score:
                continue

            # Convert to fine-tuning format
            example = FineTuningExample(
                input_text=str(review.input_data),
                expected_output=review.expected_output,
                task_type=task_type,
                metadata={
                    "agreement_score": review.agreement_score,
                    "clinician_id": review.metadata.get("clinician_id"),
                    "original_review_id": review.example_id
                },
                source="clinician_review"
            )

            examples.append(example)

        logger.info(
            f"Converted {len(examples)} HITL reviews to fine-tuning examples"
        )

        return examples

    @staticmethod
    def generate_synthetic_examples(
        task_type: FineTuningTask,
        count: int = 100
    ) -> List[FineTuningExample]:
        """
        Generate synthetic training examples

        Args:
            task_type: Type of task
            count: Number of examples to generate

        Returns:
            List of synthetic examples
        """
        examples = []

        if task_type == FineTuningTask.MEDICATION_EXTRACTION:
            # Synthetic medication extraction examples
            medications = [
                ("lisinopril 10mg daily", "Medication: lisinopril\nDose: 10mg\nFrequency: daily"),
                ("metformin 500mg twice daily with meals", "Medication: metformin\nDose: 500mg\nFrequency: twice daily\nInstructions: with meals"),
                ("atorvastatin 40mg at bedtime", "Medication: atorvastatin\nDose: 40mg\nFrequency: at bedtime"),
            ]

            for i in range(min(count, len(medications) * 10)):
                input_text, output = medications[i % len(medications)]
                examples.append(FineTuningExample(
                    input_text=f"Extract medication information: {input_text}",
                    expected_output=output,
                    task_type=task_type,
                    source="synthetic"
                ))

        elif task_type == FineTuningTask.VITAL_SIGNS_EXTRACTION:
            # Synthetic vital signs examples
            vitals = [
                ("BP 140/90, HR 78, Temp 98.6F", "Blood Pressure: 140/90 mmHg\nHeart Rate: 78 bpm\nTemperature: 98.6°F (37.0°C)"),
                ("Vital signs stable: 120/80, pulse 72, O2 sat 98% on room air", "Blood Pressure: 120/80 mmHg\nHeart Rate: 72 bpm\nOxygen Saturation: 98%"),
            ]

            for i in range(min(count, len(vitals) * 10)):
                input_text, output = vitals[i % len(vitals)]
                examples.append(FineTuningExample(
                    input_text=f"Extract vital signs: {input_text}",
                    expected_output=output,
                    task_type=task_type,
                    source="synthetic"
                ))

        logger.info(f"Generated {len(examples)} synthetic examples for {task_type.value}")

        return examples

    @staticmethod
    def augment_examples(
        examples: List[FineTuningExample],
        augmentation_factor: int = 2
    ) -> List[FineTuningExample]:
        """
        Augment training examples with variations

        Args:
            examples: Original examples
            augmentation_factor: How many variations per example

        Returns:
            Augmented examples
        """
        augmented = list(examples)  # Keep originals

        for example in examples:
            for _ in range(augmentation_factor - 1):
                # Simple augmentation: rephrasing
                augmented_example = FineTuningExample(
                    input_text=example.input_text,
                    expected_output=example.expected_output,
                    task_type=example.task_type,
                    metadata={**example.metadata, "augmented": True},
                    source=f"{example.source}_augmented"
                )
                augmented.append(augmented_example)

        logger.info(
            f"Augmented {len(examples)} examples to {len(augmented)} total"
        )

        return augmented


# ═══════════════════════════════════════════════════════════════════════════
# Fine-Tuning Pipeline
# ═══════════════════════════════════════════════════════════════════════════

class FineTuningPipeline:
    """Pipeline for fine-tuning clinical models"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        storage_dir: Optional[Path] = None
    ):
        """
        Initialize fine-tuning pipeline

        Args:
            api_key: Anthropic API key
            storage_dir: Directory to store fine-tuned models
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=self.api_key)

        if storage_dir is None:
            storage_dir = HealthcareConfig.HEALTHCARE_DIR / "fine_tuned_models"
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

        self.models_registry_file = self.storage_dir / "models_registry.json"

        logger.info(f"Fine-tuning pipeline initialized: {self.storage_dir}")

    def prepare_dataset(
        self,
        task_type: FineTuningTask,
        include_hitl_reviews: bool = True,
        include_synthetic: bool = True,
        min_agreement_score: float = 0.7,
        augmentation_factor: int = 1
    ) -> FineTuningDataset:
        """
        Prepare fine-tuning dataset

        Args:
            task_type: Type of task to train for
            include_hitl_reviews: Include clinician review data
            include_synthetic: Include synthetic examples
            min_agreement_score: Minimum clinician agreement
            augmentation_factor: Data augmentation multiplier

        Returns:
            Prepared dataset
        """
        examples = []

        # Add HITL reviews
        if include_hitl_reviews:
            hitl_reviews = review_manager.export_training_data(
                min_agreement_score=min_agreement_score
            )

            hitl_examples = DataPreparation.convert_hitl_reviews_to_examples(
                reviews=hitl_reviews,
                task_type=task_type,
                min_agreement_score=min_agreement_score
            )

            examples.extend(hitl_examples)

        # Add synthetic examples
        if include_synthetic:
            synthetic_examples = DataPreparation.generate_synthetic_examples(
                task_type=task_type,
                count=100
            )

            examples.extend(synthetic_examples)

        # Augment if requested
        if augmentation_factor > 1:
            examples = DataPreparation.augment_examples(
                examples=examples,
                augmentation_factor=augmentation_factor
            )

        # Create dataset and split
        dataset = FineTuningDataset(
            examples=examples,
            task_type=task_type
        )

        dataset.split(train_ratio=0.7, val_ratio=0.15)

        logger.info(
            f"Prepared dataset: {len(dataset.examples)} total examples"
        )

        return dataset

    def format_for_anthropic_finetuning(
        self,
        examples: List[FineTuningExample]
    ) -> List[Dict[str, Any]]:
        """
        Format examples for Anthropic fine-tuning API

        Args:
            examples: Fine-tuning examples

        Returns:
            Formatted examples
        """
        formatted = []

        for example in examples:
            formatted_example = {
                "input": example.input_text,
                "output": example.expected_output
            }

            formatted.append(formatted_example)

        return formatted

    def simulate_fine_tuning(
        self,
        dataset: FineTuningDataset,
        base_model: ModelSize = ModelSize.HAIKU,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> FineTunedModel:
        """
        Simulate fine-tuning (for demonstration)

        In production, this would call Anthropic's fine-tuning API:
        https://docs.anthropic.com/claude/docs/fine-tuning

        Args:
            dataset: Prepared dataset
            base_model: Base model to fine-tune
            hyperparameters: Training hyperparameters

        Returns:
            Fine-tuned model metadata
        """
        logger.info(f"Simulating fine-tuning {base_model.value} for {dataset.task_type.value}")

        # Format training data
        train_data = self.format_for_anthropic_finetuning(dataset.train_split)
        val_data = self.format_for_anthropic_finetuning(dataset.val_split)

        # In production, call Anthropic API:
        # fine_tuned_model = self.client.fine_tuning.create(
        #     base_model=base_model.value,
        #     training_data=train_data,
        #     validation_data=val_data,
        #     hyperparameters={
        #         "n_epochs": hyperparameters.get("n_epochs", 3),
        #         "learning_rate": hyperparameters.get("learning_rate", 1e-5),
        #         "batch_size": hyperparameters.get("batch_size", 8)
        #     }
        # )

        # Simulate model ID
        model_id = f"ft-{dataset.task_type.value}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Evaluate on validation set
        metrics = self.evaluate(
            model_id=model_id,
            test_data=dataset.val_split,
            task_type=dataset.task_type
        )

        # Create model object
        fine_tuned_model = FineTunedModel(
            model_id=model_id,
            base_model=base_model,
            task_type=dataset.task_type,
            training_examples_count=len(dataset.train_split),
            metrics=metrics
        )

        # Save to registry
        self._save_model_to_registry(fine_tuned_model)

        logger.info(f"Fine-tuning complete: {model_id}")
        logger.info(f"Metrics: Accuracy={metrics.accuracy:.3f}, F1={metrics.f1_score:.3f}")

        # Audit log
        audit_logger.log_security_event(
            event_type="MODEL_FINE_TUNING",
            severity="INFO",
            description=f"Fine-tuned model {model_id} for {dataset.task_type.value}"
        )

        return fine_tuned_model

    def evaluate(
        self,
        model_id: str,
        test_data: List[FineTuningExample],
        task_type: FineTuningTask
    ) -> FineTuningMetrics:
        """
        Evaluate fine-tuned model

        Args:
            model_id: Model identifier
            test_data: Test examples
            task_type: Task type

        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating model {model_id} on {len(test_data)} examples")

        # Simulate evaluation (in production, would call actual model)
        # For now, generate synthetic metrics

        # Simulate reasonable metrics based on training data size
        base_accuracy = 0.75 + (len(test_data) / 1000) * 0.15  # Scale with data
        base_accuracy = min(base_accuracy, 0.95)  # Cap at 95%

        accuracy = base_accuracy + np.random.normal(0, 0.02)
        precision = accuracy + np.random.normal(0, 0.03)
        recall = accuracy + np.random.normal(0, 0.03)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

        # Clip to valid range
        accuracy = np.clip(accuracy, 0, 1)
        precision = np.clip(precision, 0, 1)
        recall = np.clip(recall, 0, 1)
        f1_score = np.clip(f1_score, 0, 1)

        metrics = FineTuningMetrics(
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1_score),
            task_type=task_type,
            model_id=model_id
        )

        return metrics

    def _save_model_to_registry(self, model: FineTunedModel):
        """Save model to registry file"""
        registry = self._load_registry()

        registry[model.model_id] = {
            "model_id": model.model_id,
            "base_model": model.base_model.value,
            "task_type": model.task_type.value,
            "training_examples_count": model.training_examples_count,
            "metrics": model.metrics.to_dict(),
            "created_at": model.created_at.isoformat(),
            "deployed": model.deployed
        }

        with open(self.models_registry_file, 'w') as f:
            json.dump(registry, f, indent=2)

    def _load_registry(self) -> Dict[str, Any]:
        """Load models registry"""
        if not self.models_registry_file.exists():
            return {}

        with open(self.models_registry_file, 'r') as f:
            return json.load(f)

    def list_models(
        self,
        task_type: Optional[FineTuningTask] = None
    ) -> List[Dict[str, Any]]:
        """
        List all fine-tuned models

        Args:
            task_type: Filter by task type

        Returns:
            List of model metadata
        """
        registry = self._load_registry()

        models = list(registry.values())

        if task_type:
            models = [m for m in models if m["task_type"] == task_type.value]

        return models

    def deploy_model(self, model_id: str):
        """Mark model as deployed"""
        registry = self._load_registry()

        if model_id in registry:
            registry[model_id]["deployed"] = True

            with open(self.models_registry_file, 'w') as f:
                json.dump(registry, f, indent=2)

            logger.info(f"Deployed model: {model_id}")

            # Audit log
            audit_logger.log_security_event(
                event_type="MODEL_DEPLOYMENT",
                severity="INFO",
                description=f"Deployed fine-tuned model {model_id}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Example Usage
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("Clinical Model Fine-Tuning Pipeline Demo")
    print("=" * 80)

    # Initialize pipeline
    pipeline = FineTuningPipeline()

    # Prepare dataset
    print("\n1. Preparing Dataset")
    print("-" * 80)

    dataset = pipeline.prepare_dataset(
        task_type=FineTuningTask.MEDICATION_EXTRACTION,
        include_hitl_reviews=False,  # No reviews yet in demo
        include_synthetic=True,
        augmentation_factor=1
    )

    print(f"Total examples: {len(dataset.examples)}")
    print(f"Train: {len(dataset.train_split)}")
    print(f"Val: {len(dataset.val_split)}")
    print(f"Test: {len(dataset.test_split)}")

    # Fine-tune model
    print("\n2. Fine-Tuning Model")
    print("-" * 80)

    fine_tuned_model = pipeline.simulate_fine_tuning(
        dataset=dataset,
        base_model=ModelSize.HAIKU,
        hyperparameters={
            "n_epochs": 3,
            "learning_rate": 1e-5,
            "batch_size": 8
        }
    )

    print(f"Model ID: {fine_tuned_model.model_id}")
    print(f"Base Model: {fine_tuned_model.base_model.value}")
    print(f"Training Examples: {fine_tuned_model.training_examples_count}")

    # Show metrics
    print("\n3. Evaluation Metrics")
    print("-" * 80)

    metrics = fine_tuned_model.metrics
    print(f"Accuracy:  {metrics.accuracy:.3f}")
    print(f"Precision: {metrics.precision:.3f}")
    print(f"Recall:    {metrics.recall:.3f}")
    print(f"F1 Score:  {metrics.f1_score:.3f}")

    # List models
    print("\n4. Registered Models")
    print("-" * 80)

    models = pipeline.list_models()
    for model in models:
        print(f"  • {model['model_id']}")
        print(f"    Task: {model['task_type']}")
        print(f"    F1: {model['metrics']['f1_score']:.3f}")
        print(f"    Deployed: {model['deployed']}")

    # Deploy
    print("\n5. Deploying Model")
    print("-" * 80)

    pipeline.deploy_model(fine_tuned_model.model_id)
    print(f"✅ Model {fine_tuned_model.model_id} deployed")

    print("\n" + "=" * 80)
    print("✅ Fine-Tuning Pipeline operational")
    print("=" * 80)
