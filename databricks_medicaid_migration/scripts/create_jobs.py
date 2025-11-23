"""
Create Databricks jobs for automated pipeline execution.

This script creates Databricks Jobs for:
1. Daily bronze layer ingestion
2. Daily silver layer processing
3. Weekly feature engineering
4. Monthly model retraining
5. Daily risk scoring
"""

import os
import sys
import json
from databricks_cli.sdk import ApiClient, JobsService
from databricks_cli.jobs.api import JobsApi
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JobCreator:
    """Create and manage Databricks jobs."""

    def __init__(self, host, token):
        """Initialize job creator."""
        self.api_client = ApiClient(host=host, token=token)
        self.jobs_api = JobsApi(self.api_client)

    def create_bronze_ingestion_job(self, cluster_config):
        """Create daily bronze layer ingestion job."""
        job_config = {
            "name": "Medicaid - Bronze Layer Ingestion",
            "tags": {
                "layer": "bronze",
                "environment": "production"
            },
            "tasks": [
                {
                    "task_key": "ingest_bronze",
                    "description": "Ingest raw claims data to bronze layer",
                    "notebook_task": {
                        "notebook_path": "/Medicaid/Notebooks/01_bronze_ingestion",
                        "source": "WORKSPACE"
                    },
                    "new_cluster": cluster_config,
                    "timeout_seconds": 7200,
                    "max_retries": 2,
                    "retry_on_timeout": True
                }
            ],
            "schedule": {
                "quartz_cron_expression": "0 0 1 * * ?",  # Daily at 1 AM
                "timezone_id": "America/New_York",
                "pause_status": "UNPAUSED"
            },
            "max_concurrent_runs": 1,
            "email_notifications": {
                "on_failure": ["data-team@example.com"],
                "on_success": ["data-team@example.com"]
            }
        }

        return self._create_job(job_config)

    def create_silver_cleaning_job(self, cluster_config):
        """Create daily silver layer cleaning job."""
        job_config = {
            "name": "Medicaid - Silver Layer Cleaning",
            "tags": {
                "layer": "silver",
                "environment": "production"
            },
            "tasks": [
                {
                    "task_key": "clean_silver",
                    "description": "Clean and validate data in silver layer",
                    "notebook_task": {
                        "notebook_path": "/Medicaid/Notebooks/02_silver_cleaning",
                        "source": "WORKSPACE"
                    },
                    "new_cluster": cluster_config,
                    "timeout_seconds": 7200,
                    "max_retries": 2
                }
            ],
            "schedule": {
                "quartz_cron_expression": "0 0 2 * * ?",  # Daily at 2 AM
                "timezone_id": "America/New_York",
                "pause_status": "UNPAUSED"
            },
            "max_concurrent_runs": 1,
            "email_notifications": {
                "on_failure": ["data-team@example.com"]
            }
        }

        return self._create_job(job_config)

    def create_feature_engineering_job(self, cluster_config):
        """Create weekly feature engineering job."""
        job_config = {
            "name": "Medicaid - Feature Engineering",
            "tags": {
                "layer": "gold",
                "environment": "production"
            },
            "tasks": [
                {
                    "task_key": "engineer_features",
                    "description": "Create features for ML models",
                    "notebook_task": {
                        "notebook_path": "/Medicaid/Notebooks/03_gold_features",
                        "source": "WORKSPACE"
                    },
                    "new_cluster": cluster_config,
                    "timeout_seconds": 10800,
                    "max_retries": 2
                }
            ],
            "schedule": {
                "quartz_cron_expression": "0 0 3 ? * SUN",  # Weekly Sunday at 3 AM
                "timezone_id": "America/New_York",
                "pause_status": "UNPAUSED"
            },
            "max_concurrent_runs": 1,
            "email_notifications": {
                "on_failure": ["ml-team@example.com"]
            }
        }

        return self._create_job(job_config)

    def create_model_training_job(self, cluster_config):
        """Create monthly model retraining job."""
        job_config = {
            "name": "Medicaid - ML Model Training",
            "tags": {
                "component": "ml",
                "environment": "production"
            },
            "tasks": [
                {
                    "task_key": "train_models",
                    "description": "Train and evaluate ML risk models",
                    "notebook_task": {
                        "notebook_path": "/Medicaid/Notebooks/04_ml_risk_models",
                        "source": "WORKSPACE"
                    },
                    "new_cluster": {
                        **cluster_config,
                        "node_type_id": "i3.2xlarge",  # GPU instance for faster training
                        "num_workers": 4
                    },
                    "timeout_seconds": 14400,
                    "max_retries": 1
                }
            ],
            "schedule": {
                "quartz_cron_expression": "0 0 0 1 * ?",  # Monthly on 1st at midnight
                "timezone_id": "America/New_York",
                "pause_status": "UNPAUSED"
            },
            "max_concurrent_runs": 1,
            "email_notifications": {
                "on_failure": ["ml-team@example.com"],
                "on_success": ["ml-team@example.com"]
            }
        }

        return self._create_job(job_config)

    def create_risk_scoring_job(self, cluster_config):
        """Create daily risk scoring and immunization targeting job."""
        job_config = {
            "name": "Medicaid - Risk Scoring & Immunization Targeting",
            "tags": {
                "component": "analytics",
                "environment": "production"
            },
            "tasks": [
                {
                    "task_key": "score_risks",
                    "description": "Score member risks and generate immunization targeting lists",
                    "notebook_task": {
                        "notebook_path": "/Medicaid/Notebooks/05_immunization_targeting",
                        "source": "WORKSPACE"
                    },
                    "new_cluster": cluster_config,
                    "timeout_seconds": 7200,
                    "max_retries": 2
                }
            ],
            "schedule": {
                "quartz_cron_expression": "0 0 4 * * ?",  # Daily at 4 AM
                "timezone_id": "America/New_York",
                "pause_status": "UNPAUSED"
            },
            "max_concurrent_runs": 1,
            "email_notifications": {
                "on_failure": ["analytics-team@example.com"],
                "on_success": ["analytics-team@example.com"]
            }
        }

        return self._create_job(job_config)

    def create_end_to_end_pipeline(self, cluster_config):
        """Create end-to-end pipeline job with task dependencies."""
        job_config = {
            "name": "Medicaid - End-to-End Pipeline",
            "tags": {
                "pipeline": "full",
                "environment": "production"
            },
            "tasks": [
                {
                    "task_key": "bronze_ingestion",
                    "notebook_task": {
                        "notebook_path": "/Medicaid/Notebooks/01_bronze_ingestion",
                        "source": "WORKSPACE"
                    },
                    "new_cluster": cluster_config,
                    "timeout_seconds": 7200
                },
                {
                    "task_key": "silver_cleaning",
                    "depends_on": [{"task_key": "bronze_ingestion"}],
                    "notebook_task": {
                        "notebook_path": "/Medicaid/Notebooks/02_silver_cleaning",
                        "source": "WORKSPACE"
                    },
                    "new_cluster": cluster_config,
                    "timeout_seconds": 7200
                },
                {
                    "task_key": "gold_features",
                    "depends_on": [{"task_key": "silver_cleaning"}],
                    "notebook_task": {
                        "notebook_path": "/Medicaid/Notebooks/03_gold_features",
                        "source": "WORKSPACE"
                    },
                    "new_cluster": cluster_config,
                    "timeout_seconds": 10800
                },
                {
                    "task_key": "ml_models",
                    "depends_on": [{"task_key": "gold_features"}],
                    "notebook_task": {
                        "notebook_path": "/Medicaid/Notebooks/04_ml_risk_models",
                        "source": "WORKSPACE"
                    },
                    "new_cluster": cluster_config,
                    "timeout_seconds": 14400
                },
                {
                    "task_key": "immunization_targeting",
                    "depends_on": [{"task_key": "ml_models"}],
                    "notebook_task": {
                        "notebook_path": "/Medicaid/Notebooks/05_immunization_targeting",
                        "source": "WORKSPACE"
                    },
                    "new_cluster": cluster_config,
                    "timeout_seconds": 7200
                }
            ],
            "max_concurrent_runs": 1,
            "email_notifications": {
                "on_failure": ["data-team@example.com", "ml-team@example.com"],
                "on_success": ["data-team@example.com"]
            }
        }

        return self._create_job(job_config)

    def _create_job(self, job_config):
        """Helper to create job and handle errors."""
        try:
            job_id = self.jobs_api.create_job(job_config)
            logger.info(f"✓ Created job: {job_config['name']} (ID: {job_id})")
            return job_id
        except Exception as e:
            logger.error(f"✗ Failed to create job {job_config['name']}: {str(e)}")
            return None

    def create_all_jobs(self):
        """Create all standard jobs."""
        logger.info("=" * 60)
        logger.info("CREATING DATABRICKS JOBS")
        logger.info("=" * 60)

        # Standard cluster configuration
        cluster_config = {
            "spark_version": "13.3.x-scala2.12",
            "node_type_id": "i3.xlarge",
            "num_workers": 2,
            "spark_conf": {
                "spark.databricks.delta.preview.enabled": "true",
                "spark.databricks.delta.optimizeWrite.enabled": "true",
                "spark.databricks.delta.autoCompact.enabled": "true"
            },
            "aws_attributes": {
                "availability": "SPOT_WITH_FALLBACK",
                "zone_id": "us-east-1a",
                "spot_bid_price_percent": 100
            }
        }

        jobs = [
            ("Bronze Ingestion", self.create_bronze_ingestion_job),
            ("Silver Cleaning", self.create_silver_cleaning_job),
            ("Feature Engineering", self.create_feature_engineering_job),
            ("Model Training", self.create_model_training_job),
            ("Risk Scoring", self.create_risk_scoring_job),
            ("End-to-End Pipeline", self.create_end_to_end_pipeline)
        ]

        created_jobs = []
        for job_name, create_func in jobs:
            logger.info(f"\nCreating {job_name} job...")
            job_id = create_func(cluster_config)
            if job_id:
                created_jobs.append((job_name, job_id))

        logger.info("\n" + "=" * 60)
        logger.info(f"JOB CREATION COMPLETE: {len(created_jobs)}/{len(jobs)} jobs created")
        logger.info("=" * 60)

        for job_name, job_id in created_jobs:
            logger.info(f"  {job_name}: {job_id}")


def main():
    """Main entry point."""
    databricks_host = os.getenv("DATABRICKS_HOST")
    databricks_token = os.getenv("DATABRICKS_TOKEN")

    if not databricks_host or not databricks_token:
        logger.error("ERROR: DATABRICKS_HOST and DATABRICKS_TOKEN required")
        sys.exit(1)

    creator = JobCreator(databricks_host, databricks_token)
    creator.create_all_jobs()


if __name__ == "__main__":
    main()
