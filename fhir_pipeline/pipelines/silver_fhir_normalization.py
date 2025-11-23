"""
FHIR Silver Layer Normalization Pipeline

Parses FHIR JSON from Bronze layer and creates normalized,
analytics-ready tables with structured columns.

FHIR Resources Normalized:
- Observation → Structured lab results and vital signs
- MedicationStatement → Medication history
- Patient → Demographics (PHI-protected)
- Encounter → Clinical visit information

Author: MLOps Healthcare Team
Date: 2025-11-23
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    TimestampType,
    IntegerType,
    BooleanType,
)
from delta import DeltaTable
import hashlib
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FHIRSilverNormalization:
    """
    Normalize FHIR resources from Bronze to Silver layer.

    Transformations:
    - Parse FHIR JSON to structured columns
    - Extract CodeableConcept values (LOINC, SNOMED, RxNorm)
    - Convert timestamps to standard format
    - Hash PHI fields (patient_id, MRN)
    - Validate data quality
    - Enrich with reference data
    """

    def __init__(
        self,
        spark: SparkSession,
        bronze_path: str,
        silver_path: str,
        phi_hash_salt: str = "default_salt_change_in_production",
    ):
        """
        Initialize FHIR Silver normalization pipeline.

        Args:
            spark: Active SparkSession
            bronze_path: Path to Bronze layer Delta tables
            silver_path: Path to Silver layer output
            phi_hash_salt: Salt for PHI hashing (load from secrets in production)
        """
        self.spark = spark
        self.bronze_path = bronze_path
        self.silver_path = silver_path
        self.phi_hash_salt = phi_hash_salt

    def normalize_observations(self, lookback_days: int = 1) -> None:
        """
        Normalize Observation resources (lab results, vital signs).

        FHIR Structure:
        {
            "resourceType": "Observation",
            "id": "obs-12345",
            "status": "final",
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "2345-7",
                    "display": "Glucose"
                }]
            },
            "subject": {"reference": "Patient/pt-456"},
            "effectiveDateTime": "2025-11-23T10:00:00Z",
            "valueQuantity": {
                "value": 95,
                "unit": "mg/dL"
            }
        }

        Args:
            lookback_days: Process last N days of data (incremental)
        """
        logger.info("Normalizing Observation resources...")

        # Read Bronze observations
        df_bronze = (
            self.spark.read.format("delta")
            .load(f"{self.bronze_path}/observation_raw")
            .filter(
                F.col("ingestion_date")
                >= F.date_sub(F.current_date(), lookback_days)
            )
        )

        # Parse FHIR JSON to structured columns
        df_silver = (
            df_bronze
            # Core identifiers
            .withColumn("observation_id", F.get_json_object("fhir_json", "$.id"))
            .withColumn("status", F.get_json_object("fhir_json", "$.status"))
            # LOINC code (lab test identifier)
            .withColumn(
                "loinc_code",
                F.get_json_object("fhir_json", "$.code.coding[0].code"),
            )
            .withColumn(
                "loinc_display",
                F.get_json_object("fhir_json", "$.code.coding[0].display"),
            )
            .withColumn(
                "loinc_system",
                F.get_json_object("fhir_json", "$.code.coding[0].system"),
            )
            # Patient reference (extract patient ID)
            .withColumn(
                "patient_reference",
                F.get_json_object("fhir_json", "$.subject.reference"),
            )
            .withColumn(
                "patient_id_raw",
                F.regexp_extract(F.col("patient_reference"), r"Patient/(.+)", 1),
            )
            # Timestamp
            .withColumn(
                "effective_datetime",
                F.to_timestamp(
                    F.get_json_object("fhir_json", "$.effectiveDateTime")
                ),
            )
            # Result value (numeric)
            .withColumn(
                "value_numeric",
                F.get_json_object("fhir_json", "$.valueQuantity.value").cast(
                    DoubleType()
                ),
            )
            .withColumn(
                "value_unit", F.get_json_object("fhir_json", "$.valueQuantity.unit")
            )
            # Result value (string for qualitative results)
            .withColumn(
                "value_string",
                F.get_json_object("fhir_json", "$.valueString"),
            )
            # Reference ranges
            .withColumn(
                "reference_range_low",
                F.get_json_object("fhir_json", "$.referenceRange[0].low.value").cast(
                    DoubleType()
                ),
            )
            .withColumn(
                "reference_range_high",
                F.get_json_object("fhir_json", "$.referenceRange[0].high.value").cast(
                    DoubleType()
                ),
            )
            # Performer (lab/device)
            .withColumn(
                "performer_reference",
                F.get_json_object("fhir_json", "$.performer[0].reference"),
            )
            .withColumn(
                "device_id",
                F.regexp_extract(F.col("performer_reference"), r"Device/(.+)", 1),
            )
        )

        # Hash patient_id for PHI protection
        df_silver = df_silver.withColumn(
            "patient_id_hashed", self._hash_pii_udf(F.col("patient_id_raw"))
        )

        # Add data quality flags
        df_silver = self._add_observation_quality_flags(df_silver)

        # Select final columns
        df_silver = df_silver.select(
            "observation_id",
            "patient_id_hashed",
            "loinc_code",
            "loinc_display",
            "status",
            "effective_datetime",
            "value_numeric",
            "value_unit",
            "value_string",
            "reference_range_low",
            "reference_range_high",
            "device_id",
            "is_valid",
            "is_abnormal",
            "abnormal_severity",
            "ingestion_timestamp",
            F.date_format("effective_datetime", "yyyy-MM-dd").alias("observation_date"),
        )

        # Write to Delta Lake
        record_count = df_silver.count()
        logger.info(f"Normalizing {record_count} Observation records")

        df_silver.write.format("delta").mode("overwrite").partitionBy(
            "observation_date"
        ).save(f"{self.silver_path}/observations_normalized")

        logger.info("✓ Observation normalization complete")

    def normalize_medication_statements(self, lookback_days: int = 1) -> None:
        """
        Normalize MedicationStatement resources (medication history).

        FHIR Structure:
        {
            "resourceType": "MedicationStatement",
            "id": "med-789",
            "status": "active",
            "medicationCodeableConcept": {
                "coding": [{
                    "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                    "code": "197361",
                    "display": "Metformin 500 MG"
                }]
            },
            "subject": {"reference": "Patient/pt-456"},
            "effectivePeriod": {
                "start": "2025-01-01T00:00:00Z"
            },
            "dosage": [{
                "text": "Take 1 tablet twice daily"
            }]
        }

        Args:
            lookback_days: Process last N days
        """
        logger.info("Normalizing MedicationStatement resources...")

        # Read Bronze medication statements
        df_bronze = (
            self.spark.read.format("delta")
            .load(f"{self.bronze_path}/medicationstatement_raw")
            .filter(
                F.col("ingestion_date")
                >= F.date_sub(F.current_date(), lookback_days)
            )
        )

        # Parse FHIR JSON
        df_silver = (
            df_bronze
            # Core identifiers
            .withColumn(
                "medication_statement_id", F.get_json_object("fhir_json", "$.id")
            )
            .withColumn("status", F.get_json_object("fhir_json", "$.status"))
            # RxNorm code (medication identifier)
            .withColumn(
                "rxnorm_code",
                F.get_json_object(
                    "fhir_json", "$.medicationCodeableConcept.coding[0].code"
                ),
            )
            .withColumn(
                "medication_name",
                F.get_json_object(
                    "fhir_json", "$.medicationCodeableConcept.coding[0].display"
                ),
            )
            # Patient
            .withColumn(
                "patient_reference",
                F.get_json_object("fhir_json", "$.subject.reference"),
            )
            .withColumn(
                "patient_id_raw",
                F.regexp_extract(F.col("patient_reference"), r"Patient/(.+)", 1),
            )
            # Effective period
            .withColumn(
                "effective_start",
                F.to_timestamp(
                    F.get_json_object("fhir_json", "$.effectivePeriod.start")
                ),
            )
            .withColumn(
                "effective_end",
                F.to_timestamp(
                    F.get_json_object("fhir_json", "$.effectivePeriod.end")
                ),
            )
            # Dosage
            .withColumn("dosage_text", F.get_json_object("fhir_json", "$.dosage[0].text"))
        )

        # Hash patient_id
        df_silver = df_silver.withColumn(
            "patient_id_hashed", self._hash_pii_udf(F.col("patient_id_raw"))
        )

        # Calculate medication duration
        df_silver = df_silver.withColumn(
            "duration_days",
            F.when(
                F.col("effective_end").isNotNull(),
                F.datediff(F.col("effective_end"), F.col("effective_start")),
            ).otherwise(None),
        )

        # Add is_active flag
        df_silver = df_silver.withColumn(
            "is_active",
            (F.col("status") == "active")
            & (
                F.col("effective_end").isNull()
                | (F.col("effective_end") >= F.current_timestamp())
            ),
        )

        # Select final columns
        df_silver = df_silver.select(
            "medication_statement_id",
            "patient_id_hashed",
            "rxnorm_code",
            "medication_name",
            "status",
            "is_active",
            "effective_start",
            "effective_end",
            "duration_days",
            "dosage_text",
            "ingestion_timestamp",
            F.date_format("effective_start", "yyyy-MM-dd").alias("start_date"),
        )

        # Write to Delta Lake
        record_count = df_silver.count()
        logger.info(f"Normalizing {record_count} MedicationStatement records")

        df_silver.write.format("delta").mode("overwrite").partitionBy("start_date").save(
            f"{self.silver_path}/medications_normalized"
        )

        logger.info("✓ MedicationStatement normalization complete")

    def normalize_patients(self, lookback_days: int = 30) -> None:
        """
        Normalize Patient resources (demographics, PHI-protected).

        FHIR Structure:
        {
            "resourceType": "Patient",
            "id": "pt-456",
            "identifier": [{
                "type": {"coding": [{"code": "MR"}]},
                "value": "MRN-12345"
            }],
            "name": [{
                "family": "Smith",
                "given": ["John"]
            }],
            "gender": "male",
            "birthDate": "1980-01-15",
            "address": [{
                "city": "Atlanta",
                "state": "GA",
                "postalCode": "30303"
            }]
        }

        Args:
            lookback_days: Process last N days
        """
        logger.info("Normalizing Patient resources...")

        # Read Bronze patients
        df_bronze = (
            self.spark.read.format("delta")
            .load(f"{self.bronze_path}/patient_raw")
            .filter(
                F.col("ingestion_date")
                >= F.date_sub(F.current_date(), lookback_days)
            )
        )

        # Parse FHIR JSON
        df_silver = (
            df_bronze
            # Core identifiers
            .withColumn("patient_id_raw", F.get_json_object("fhir_json", "$.id"))
            .withColumn(
                "mrn",
                F.get_json_object("fhir_json", "$.identifier[0].value"),
            )
            # Demographics
            .withColumn("gender", F.get_json_object("fhir_json", "$.gender"))
            .withColumn(
                "birth_date",
                F.to_date(F.get_json_object("fhir_json", "$.birthDate")),
            )
            # Address (de-identified to city/state only)
            .withColumn("city", F.get_json_object("fhir_json", "$.address[0].city"))
            .withColumn("state", F.get_json_object("fhir_json", "$.address[0].state"))
            .withColumn(
                "postal_code_prefix",
                F.substring(
                    F.get_json_object("fhir_json", "$.address[0].postalCode"), 1, 3
                ),
            )
        )

        # Hash all PHI
        df_silver = df_silver.withColumn(
            "patient_id_hashed", self._hash_pii_udf(F.col("patient_id_raw"))
        )
        df_silver = df_silver.withColumn(
            "mrn_hashed", self._hash_pii_udf(F.col("mrn"))
        )

        # Calculate age
        df_silver = df_silver.withColumn(
            "age_years",
            F.floor(F.months_between(F.current_date(), F.col("birth_date")) / 12),
        )

        # Age bucket (for de-identification)
        df_silver = df_silver.withColumn(
            "age_bucket",
            F.when(F.col("age_years") < 18, "0-17")
            .when(F.col("age_years") < 30, "18-29")
            .when(F.col("age_years") < 40, "30-39")
            .when(F.col("age_years") < 50, "40-49")
            .when(F.col("age_years") < 60, "50-59")
            .when(F.col("age_years") < 70, "60-69")
            .otherwise("70+"),
        )

        # Select final columns (NO raw PHI)
        df_silver = df_silver.select(
            "patient_id_hashed",
            "mrn_hashed",
            "gender",
            "age_years",
            "age_bucket",
            "city",
            "state",
            "postal_code_prefix",  # Only first 3 digits
            "ingestion_timestamp",
        )

        # Write to Delta Lake (NO partitioning for patient data)
        record_count = df_silver.count()
        logger.info(f"Normalizing {record_count} Patient records")

        df_silver.write.format("delta").mode("overwrite").save(
            f"{self.silver_path}/patients_normalized"
        )

        logger.info("✓ Patient normalization complete")

    def normalize_encounters(self, lookback_days: int = 7) -> None:
        """
        Normalize Encounter resources (clinical visits).

        Args:
            lookback_days: Process last N days
        """
        logger.info("Normalizing Encounter resources...")

        df_bronze = (
            self.spark.read.format("delta")
            .load(f"{self.bronze_path}/encounter_raw")
            .filter(
                F.col("ingestion_date")
                >= F.date_sub(F.current_date(), lookback_days)
            )
        )

        df_silver = (
            df_bronze.withColumn("encounter_id", F.get_json_object("fhir_json", "$.id"))
            .withColumn("status", F.get_json_object("fhir_json", "$.status"))
            .withColumn(
                "encounter_class",
                F.get_json_object("fhir_json", "$.class.code"),
            )
            .withColumn(
                "patient_id_raw",
                F.regexp_extract(
                    F.get_json_object("fhir_json", "$.subject.reference"),
                    r"Patient/(.+)",
                    1,
                ),
            )
            .withColumn(
                "period_start",
                F.to_timestamp(F.get_json_object("fhir_json", "$.period.start")),
            )
            .withColumn(
                "period_end",
                F.to_timestamp(F.get_json_object("fhir_json", "$.period.end")),
            )
        )

        df_silver = df_silver.withColumn(
            "patient_id_hashed", self._hash_pii_udf(F.col("patient_id_raw"))
        )

        df_silver = df_silver.withColumn(
            "duration_hours",
            F.round(
                (
                    F.unix_timestamp("period_end") - F.unix_timestamp("period_start")
                )
                / 3600,
                2,
            ),
        )

        df_silver = df_silver.select(
            "encounter_id",
            "patient_id_hashed",
            "status",
            "encounter_class",
            "period_start",
            "period_end",
            "duration_hours",
            "ingestion_timestamp",
            F.date_format("period_start", "yyyy-MM-dd").alias("encounter_date"),
        )

        record_count = df_silver.count()
        logger.info(f"Normalizing {record_count} Encounter records")

        df_silver.write.format("delta").mode("overwrite").partitionBy(
            "encounter_date"
        ).save(f"{self.silver_path}/encounters_normalized")

        logger.info("✓ Encounter normalization complete")

    def _hash_pii_udf(self, col):
        """Create UDF for SHA-256 hashing with salt."""

        @F.udf(returnType=StringType())
        def hash_value(value: str) -> str:
            if value is None:
                return None
            salted = f"{value}_{self.phi_hash_salt}"
            return hashlib.sha256(salted.encode()).hexdigest()

        return hash_value(col)

    def _add_observation_quality_flags(self, df: DataFrame) -> DataFrame:
        """Add data quality and clinical flags to observations."""

        df = df.withColumn(
            "is_valid",
            (F.col("observation_id").isNotNull())
            & (F.col("loinc_code").isNotNull())
            & (F.col("patient_id_raw").isNotNull())
            & (F.col("effective_datetime").isNotNull())
            & (
                F.col("value_numeric").isNotNull()
                | F.col("value_string").isNotNull()
            ),
        )

        # Flag abnormal results (outside reference range)
        df = df.withColumn(
            "is_abnormal",
            F.when(
                F.col("value_numeric").isNotNull()
                & F.col("reference_range_low").isNotNull()
                & F.col("reference_range_high").isNotNull(),
                (F.col("value_numeric") < F.col("reference_range_low"))
                | (F.col("value_numeric") > F.col("reference_range_high")),
            ).otherwise(False),
        )

        # Severity of abnormality
        df = df.withColumn(
            "abnormal_severity",
            F.when(
                ~F.col("is_abnormal"),
                "NORMAL",
            )
            .when(
                (F.col("value_numeric") < F.col("reference_range_low") * 0.5)
                | (F.col("value_numeric") > F.col("reference_range_high") * 1.5),
                "CRITICAL",
            )
            .otherwise("ABNORMAL"),
        )

        return df

    def run_all_normalizations(self, lookback_days: int = 1) -> None:
        """Run all normalization pipelines."""
        logger.info("=== Starting FHIR Silver Normalization ===")

        self.normalize_observations(lookback_days)
        self.normalize_medication_statements(lookback_days)
        self.normalize_patients(lookback_days=30)  # Full refresh monthly
        self.normalize_encounters(lookback_days=7)

        logger.info("=== FHIR Silver Normalization Complete ===")


if __name__ == "__main__":
    spark = (
        SparkSession.builder.appName("FHIR Silver Normalization")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .getOrCreate()
    )

    BRONZE_PATH = "/mnt/delta/fhir/bronze"
    SILVER_PATH = "/mnt/delta/fhir/silver"

    # Load PHI hash salt from secrets (production)
    # PHI_HASH_SALT = dbutils.secrets.get(scope="fhir-secrets", key="phi-hash-salt")
    PHI_HASH_SALT = "change_me_in_production_12345"

    normalization = FHIRSilverNormalization(
        spark=spark,
        bronze_path=BRONZE_PATH,
        silver_path=SILVER_PATH,
        phi_hash_salt=PHI_HASH_SALT,
    )

    # Run all normalizations (incremental - last 1 day)
    normalization.run_all_normalizations(lookback_days=1)

    logger.info("✓ FHIR Silver pipeline complete")
