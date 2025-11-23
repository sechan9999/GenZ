"""
FHIR End-to-End Pipeline Orchestration

Orchestrates the complete FHIR data pipeline:
1. Bronze: Ingest FHIR streams from Azure Event Hubs or Blob Storage
2. Silver: Normalize FHIR resources to structured tables
3. Gold: Create clinical aggregations and ML features
4. ML: Train and deploy predictive models with Azure ML

Usage:
    # Full pipeline (batch mode)
    python fhir_pipeline_main.py --mode batch --source blob

    # Streaming ingestion only
    python fhir_pipeline_main.py --mode streaming --source eventhub

    # ML training only
    python fhir_pipeline_main.py --mode ml-training

Author: MLOps Healthcare Team
Date: 2025-11-23
"""

import argparse
import logging
import sys
from datetime import datetime
from pyspark.sql import SparkSession

# Import pipeline modules
from pipelines.bronze_fhir_ingestion import FHIRBronzeIngestion
from pipelines.silver_fhir_normalization import FHIRSilverNormalization
from pipelines.gold_clinical_aggregation import FHIRGoldAggregation
from models.azureml_training import FHIRAzureMLTraining

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class FHIRPipelineOrchestrator:
    """
    Orchestrate end-to-end FHIR pipeline execution.

    Handles:
    - Pipeline configuration
    - Dependency management (Bronze → Silver → Gold → ML)
    - Error handling and retries
    - Monitoring and logging
    """

    def __init__(
        self,
        bronze_path: str,
        silver_path: str,
        gold_path: str,
        checkpoint_path: str,
        phi_hash_salt: str = "default_salt_change_in_production",
    ):
        """
        Initialize pipeline orchestrator.

        Args:
            bronze_path: Bronze layer Delta Lake path
            silver_path: Silver layer Delta Lake path
            gold_path: Gold layer Delta Lake path
            checkpoint_path: Streaming checkpoint path
            phi_hash_salt: Salt for PHI hashing
        """
        # Initialize Spark session
        self.spark = self._create_spark_session()

        # Store paths
        self.bronze_path = bronze_path
        self.silver_path = silver_path
        self.gold_path = gold_path
        self.checkpoint_path = checkpoint_path
        self.phi_hash_salt = phi_hash_salt

        # Initialize pipeline components
        self.bronze_ingestion = FHIRBronzeIngestion(
            spark=self.spark,
            bronze_path=bronze_path,
            checkpoint_path=checkpoint_path,
        )

        self.silver_normalization = FHIRSilverNormalization(
            spark=self.spark,
            bronze_path=bronze_path,
            silver_path=silver_path,
            phi_hash_salt=phi_hash_salt,
        )

        self.gold_aggregation = FHIRGoldAggregation(
            spark=self.spark,
            silver_path=silver_path,
            gold_path=gold_path,
        )

        self.ml_training = FHIRAzureMLTraining(
            spark=self.spark,
            gold_path=gold_path,
            mlflow_tracking_uri="databricks",
        )

        logger.info("✓ Pipeline orchestrator initialized")

    def _create_spark_session(self) -> SparkSession:
        """Create Spark session with Delta Lake configuration."""
        spark = (
            SparkSession.builder.appName("FHIR_Pipeline_Orchestration")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
            .config("spark.databricks.delta.properties.defaults.enableChangeDataFeed", "true")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .getOrCreate()
        )
        return spark

    def run_batch_pipeline(
        self,
        source_type: str,
        source_config: dict,
        lookback_days: int = 1,
    ) -> None:
        """
        Run full batch pipeline (Bronze → Silver → Gold).

        Args:
            source_type: 'blob' or 'adls'
            source_config: Source configuration dictionary
            lookback_days: Process last N days
        """
        logger.info("=== Starting Batch Pipeline ===")
        start_time = datetime.now()

        try:
            # Step 1: Bronze Ingestion
            logger.info("STEP 1/3: Bronze Layer Ingestion")
            if source_type == "blob":
                self.bronze_ingestion.ingest_from_blob_storage(
                    storage_account=source_config["storage_account"],
                    container_name=source_config["container_name"],
                    folder_path=source_config["folder_path"],
                    file_format=source_config.get("file_format", "json"),
                    mode="append",
                )
            elif source_type == "adls":
                self.bronze_ingestion.ingest_from_adls_gen2(
                    storage_account=source_config["storage_account"],
                    file_system=source_config["file_system"],
                    directory_path=source_config["directory_path"],
                    mode="append",
                )
            else:
                raise ValueError(f"Unsupported source type: {source_type}")

            # Create resource-specific tables
            self.bronze_ingestion.create_resource_specific_tables()

            # Step 2: Silver Normalization
            logger.info("STEP 2/3: Silver Layer Normalization")
            self.silver_normalization.run_all_normalizations(lookback_days=lookback_days)

            # Step 3: Gold Aggregation
            logger.info("STEP 3/3: Gold Layer Aggregation")
            self.gold_aggregation.run_all_aggregations(lookback_days=lookback_days)

            # Pipeline complete
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"✓ Batch pipeline complete in {duration:.2f} seconds")

        except Exception as e:
            logger.error(f"❌ Batch pipeline failed: {e}", exc_info=True)
            raise

    def run_streaming_pipeline(
        self,
        eventhub_connection_string: str,
        eventhub_name: str,
    ) -> None:
        """
        Run streaming ingestion pipeline (Event Hub → Bronze).

        This starts a long-running streaming query.

        Args:
            eventhub_connection_string: Azure Event Hub connection string
            eventhub_name: Event Hub name
        """
        logger.info("=== Starting Streaming Pipeline ===")

        try:
            # Start Event Hub streaming ingestion
            query = self.bronze_ingestion.ingest_from_event_hub(
                connection_string=eventhub_connection_string,
                event_hub_name=eventhub_name,
                consumer_group="$Default",
                starting_position="latest",
            )

            logger.info("Streaming query started. Press Ctrl+C to stop.")
            logger.info("To process Silver/Gold layers, run batch pipeline separately.")

            # Keep query running
            query.awaitTermination()

        except KeyboardInterrupt:
            logger.info("Streaming pipeline stopped by user")
        except Exception as e:
            logger.error(f"❌ Streaming pipeline failed: {e}", exc_info=True)
            raise

    def run_ml_training(
        self,
        model_types: list = ["diabetes_risk", "readmission_risk"],
    ) -> None:
        """
        Run ML model training.

        Args:
            model_types: List of models to train
        """
        logger.info("=== Starting ML Training ===")

        results = {}

        try:
            for model_type in model_types:
                if model_type == "diabetes_risk":
                    logger.info("Training diabetes risk model...")
                    results["diabetes_risk"] = self.ml_training.train_diabetes_risk_model(
                        experiment_name="fhir_diabetes_risk",
                        model_type="random_forest",
                    )
                    logger.info(f"Diabetes model AUC: {results['diabetes_risk']['auc']:.4f}")

                elif model_type == "readmission_risk":
                    logger.info("Training readmission risk model...")
                    results["readmission_risk"] = (
                        self.ml_training.train_readmission_risk_model()
                    )
                    logger.info(
                        f"Readmission model AUC: {results['readmission_risk']['auc']:.4f}"
                    )

            logger.info("✓ ML training complete")
            return results

        except Exception as e:
            logger.error(f"❌ ML training failed: {e}", exc_info=True)
            raise

    def optimize_all_tables(self) -> None:
        """Optimize all Delta Lake tables (run daily/weekly)."""
        logger.info("=== Optimizing Delta Lake Tables ===")

        try:
            # Optimize Bronze
            logger.info("Optimizing Bronze layer...")
            self.bronze_ingestion.optimize_bronze_tables()

            # Optimize Silver
            logger.info("Optimizing Silver layer...")
            for table in [
                "observations_normalized",
                "medications_normalized",
                "patients_normalized",
                "encounters_normalized",
            ]:
                self.spark.sql(
                    f"OPTIMIZE delta.`{self.silver_path}/{table}`"
                )
                self.spark.sql(
                    f"VACUUM delta.`{self.silver_path}/{table}` RETAIN 168 HOURS"
                )

            # Optimize Gold
            logger.info("Optimizing Gold layer...")
            for table in [
                "patient_vital_trends",
                "lab_result_trends",
                "chronic_disease_features",
            ]:
                self.spark.sql(
                    f"OPTIMIZE delta.`{self.gold_path}/{table}`"
                )

            logger.info("✓ Table optimization complete")

        except Exception as e:
            logger.error(f"❌ Optimization failed: {e}", exc_info=True)


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="FHIR Pipeline Orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["batch", "streaming", "ml-training", "optimize"],
        required=True,
        help="Pipeline execution mode",
    )

    parser.add_argument(
        "--source",
        choices=["blob", "adls", "eventhub"],
        help="Data source type (required for batch/streaming modes)",
    )

    parser.add_argument(
        "--storage-account",
        help="Azure storage account name",
    )

    parser.add_argument(
        "--container-name",
        help="Blob container name (for blob source)",
    )

    parser.add_argument(
        "--folder-path",
        help="Folder path in storage (for blob source)",
    )

    parser.add_argument(
        "--file-system",
        help="ADLS Gen2 file system name (for adls source)",
    )

    parser.add_argument(
        "--directory-path",
        help="Directory path in ADLS (for adls source)",
    )

    parser.add_argument(
        "--eventhub-connection-string",
        help="Event Hub connection string (for streaming mode)",
    )

    parser.add_argument(
        "--eventhub-name",
        help="Event Hub name (for streaming mode)",
    )

    parser.add_argument(
        "--lookback-days",
        type=int,
        default=1,
        help="Number of days to process (batch mode)",
    )

    parser.add_argument(
        "--bronze-path",
        default="/mnt/delta/fhir/bronze",
        help="Bronze layer path",
    )

    parser.add_argument(
        "--silver-path",
        default="/mnt/delta/fhir/silver",
        help="Silver layer path",
    )

    parser.add_argument(
        "--gold-path",
        default="/mnt/delta/fhir/gold",
        help="Gold layer path",
    )

    parser.add_argument(
        "--checkpoint-path",
        default="/mnt/delta/fhir/checkpoints",
        help="Streaming checkpoint path",
    )

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = FHIRPipelineOrchestrator(
        bronze_path=args.bronze_path,
        silver_path=args.silver_path,
        gold_path=args.gold_path,
        checkpoint_path=args.checkpoint_path,
    )

    # Execute based on mode
    if args.mode == "batch":
        if not args.source:
            logger.error("--source is required for batch mode")
            sys.exit(1)

        if args.source == "blob":
            source_config = {
                "storage_account": args.storage_account,
                "container_name": args.container_name,
                "folder_path": args.folder_path,
            }
        elif args.source == "adls":
            source_config = {
                "storage_account": args.storage_account,
                "file_system": args.file_system,
                "directory_path": args.directory_path,
            }

        orchestrator.run_batch_pipeline(
            source_type=args.source,
            source_config=source_config,
            lookback_days=args.lookback_days,
        )

    elif args.mode == "streaming":
        if not args.eventhub_connection_string or not args.eventhub_name:
            logger.error(
                "--eventhub-connection-string and --eventhub-name required for streaming"
            )
            sys.exit(1)

        orchestrator.run_streaming_pipeline(
            eventhub_connection_string=args.eventhub_connection_string,
            eventhub_name=args.eventhub_name,
        )

    elif args.mode == "ml-training":
        orchestrator.run_ml_training()

    elif args.mode == "optimize":
        orchestrator.optimize_all_tables()


if __name__ == "__main__":
    main()
