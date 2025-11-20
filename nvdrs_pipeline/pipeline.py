"""
NVDRS NLP Pipeline - Core Implementation
Integrates Spark NLP and Hugging Face for narrative analysis
"""

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import time

# PySpark and Spark NLP imports
try:
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql.functions import col, udf, lit
    from pyspark.sql.types import StringType, ArrayType, FloatType, StructType, StructField

    from sparknlp.base import DocumentAssembler, Pipeline
    from sparknlp.annotator import (
        Tokenizer,
        SentenceDetector,
        BertForTokenClassification,
        NerConverter,
        Normalizer,
    )

    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    logging.warning("Spark NLP not available. Install with: pip install spark-nlp pyspark")

from nvdrs_pipeline.config import NVDRSConfig
from nvdrs_pipeline.models import (
    NVDRSRecord,
    NERResult,
    RiskFactors,
    SubstanceUse,
    MentalHealthIndicators,
    SocialStressors,
    PipelineResult,
)
from nvdrs_pipeline.pii_redaction import PIIRedactor, redact_spark_dataframe

logger = logging.getLogger(__name__)


class NVDRSPipeline:
    """
    Complete NVDRS NLP Pipeline

    Processes coroner/medical examiner and law enforcement narratives to extract:
    - Named entities (medications, conditions, etc.)
    - Risk factors (substance use, mental health, social stressors)
    - Intent classification (suicide vs accidental vs undetermined)

    Example:
        >>> pipeline = NVDRSPipeline()
        >>> results = pipeline.process_narratives(df)
        >>> pipeline.save_results(results, "output.parquet")
    """

    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        config: Optional[NVDRSConfig] = None,
        enable_pii_redaction: bool = True,
    ):
        """
        Initialize NVDRS Pipeline

        Args:
            spark: Existing SparkSession (creates new if None)
            config: Configuration object (uses default if None)
            enable_pii_redaction: Whether to redact PII
        """
        if not SPARK_AVAILABLE:
            raise ImportError(
                "Spark NLP is required. Install with:\n"
                "pip install spark-nlp pyspark"
            )

        self.config = config or NVDRSConfig()
        self.spark = spark or self._create_spark_session()
        self.enable_pii_redaction = enable_pii_redaction

        if self.enable_pii_redaction:
            self.pii_redactor = PIIRedactor(
                redact_names=self.config.REDACT_NAMES,
                redact_dates=self.config.REDACT_DATES,
                redact_ages=False,  # Keep ages for analysis
            )

        self.nlp_pipeline = None
        logger.info("NVDRS Pipeline initialized")

    def _create_spark_session(self) -> SparkSession:
        """Create SparkSession with appropriate configuration"""
        logger.info("Creating Spark session...")

        spark = (
            SparkSession.builder
            .appName(self.config.SPARK_APP_NAME)
            .master(self.config.SPARK_MASTER)
            .config("spark.driver.memory", self.config.SPARK_DRIVER_MEMORY)
            .config("spark.executor.memory", self.config.SPARK_EXECUTOR_MEMORY)
            .config("spark.kryoserializer.buffer.max", "2000M")
            .config("spark.driver.maxResultSize", "0")
            .getOrCreate()
        )

        logger.info(f"Spark session created: {spark.version}")
        return spark

    def build_nlp_pipeline(self) -> Pipeline:
        """
        Build Spark NLP pipeline with clinical NER

        Pipeline stages:
        1. Document Assembly
        2. Sentence Detection
        3. Tokenization
        4. Normalization
        5. Clinical NER (BioBERT/ClinicalBERT)
        6. NER Conversion

        Returns:
            Configured Spark NLP Pipeline
        """
        logger.info("Building NLP pipeline...")

        # Stage 1: Document Assembly
        document_assembler = (
            DocumentAssembler()
            .setInputCol("text")
            .setOutputCol("document")
        )

        # Stage 2: Sentence Detection
        sentence_detector = (
            SentenceDetector()
            .setInputCols(["document"])
            .setOutputCol("sentence")
        )

        # Stage 3: Tokenization
        tokenizer = (
            Tokenizer()
            .setInputCols(["sentence"])
            .setOutputCol("token")
        )

        # Stage 4: Normalization (clean text)
        normalizer = (
            Normalizer()
            .setInputCols(["token"])
            .setOutputCol("normalized")
            .setLowercase(not self.config.CASE_SENSITIVE)
        )

        # Stage 5: Clinical NER using pre-trained BERT model
        # This is the "brain" - loads clinical/biomedical NER model
        logger.info(f"Loading NER model: {self.config.NER_MODEL}")

        clinical_ner = (
            BertForTokenClassification.pretrained(
                self.config.NER_MODEL,
                self.config.NER_LANGUAGE,
                self.config.NER_SOURCE
            )
            .setInputCols(["sentence", "token"])
            .setOutputCol("ner")
            .setCaseSensitive(self.config.CASE_SENSITIVE)
            .setMaxSentenceLength(self.config.MAX_SENTENCE_LENGTH)
        )

        # Stage 6: Convert NER chunks to readable format
        ner_converter = (
            NerConverter()
            .setInputCols(["sentence", "token", "ner"])
            .setOutputCol("ner_chunk")
        )

        # Build complete pipeline
        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            tokenizer,
            normalizer,
            clinical_ner,
            ner_converter,
        ])

        logger.info("NLP pipeline built successfully")
        return pipeline

    def process_narratives(
        self,
        df: DataFrame,
        narrative_columns: List[str] = None,
    ) -> PipelineResult:
        """
        Process NVDRS narratives through NLP pipeline

        Args:
            df: Spark DataFrame with narrative columns
            narrative_columns: List of columns to process (default: ['CME_Narrative', 'LE_Narrative'])

        Returns:
            PipelineResult with extracted entities and risk factors
        """
        start_time = time.time()

        if narrative_columns is None:
            narrative_columns = ['CME_Narrative', 'LE_Narrative']

        logger.info(f"Processing {df.count()} records...")

        # Step 1: PII Redaction
        if self.enable_pii_redaction:
            logger.info("Applying PII redaction...")
            df = redact_spark_dataframe(df, narrative_columns, self.pii_redactor)
            # Use redacted columns for processing
            processing_columns = [f"{col}_redacted" for col in narrative_columns]
        else:
            processing_columns = narrative_columns

        # Step 2: Combine narratives into single text column
        # Concatenate CME and LE narratives with separator
        df = df.withColumn(
            "text",
            col(processing_columns[0]).cast(StringType())
        )

        if len(processing_columns) > 1:
            from pyspark.sql.functions import concat_ws
            df = df.withColumn(
                "text",
                concat_ws(" | ", *[col(c) for c in processing_columns])
            )

        # Step 3: Build and fit NLP pipeline
        if self.nlp_pipeline is None:
            self.nlp_pipeline = self.build_nlp_pipeline()

        logger.info("Fitting NLP model...")
        pipeline_model = self.nlp_pipeline.fit(df)

        logger.info("Transforming data...")
        result_df = pipeline_model.transform(df)

        # Step 4: Extract entities and risk factors
        logger.info("Extracting risk factors...")
        processed_df = self._extract_risk_factors(result_df)

        # Step 5: Collect results
        records = self._df_to_records(processed_df)

        processing_time = time.time() - start_time

        # Generate summary statistics
        summary_stats = self._generate_summary_statistics(records)

        result = PipelineResult(
            total_records=len(records),
            successful=len(records),
            records=records,
            processing_time_seconds=processing_time,
            summary_statistics=summary_stats,
        )

        logger.info(
            f"Processing complete: {result.successful}/{result.total_records} "
            f"records in {processing_time:.2f}s"
        )

        return result

    def _extract_risk_factors(self, df: DataFrame) -> DataFrame:
        """
        Extract risk factors from NER results

        Analyzes extracted entities and text to identify:
        - Substance use (opioids, alcohol, etc.)
        - Mental health indicators
        - Social stressors
        """

        # UDF to analyze entities and extract risk factors
        def extract_risk_factors_udf(text, entities):
            """Extract risk factors from text and entities"""
            if not text:
                return {}

            text_lower = text.lower()

            # Substance use detection
            opioid_mentioned = any(
                keyword in text_lower
                for keyword in self.config.OPIOID_KEYWORDS
            )
            opioid_types = [
                keyword for keyword in self.config.OPIOID_KEYWORDS
                if keyword in text_lower
            ]

            alcohol_mentioned = "alcohol" in text_lower
            overdose_indicated = "overdose" in text_lower or "od" in text_lower

            # Mental health detection
            depression_mentioned = any(
                keyword in text_lower
                for keyword in ["depression", "depressed"]
            )
            anxiety_mentioned = "anxiety" in text_lower
            suicide_history = any(
                keyword in text_lower
                for keyword in ["previous suicide", "prior suicide", "suicide attempt"]
            )

            # Social stressors
            financial_crisis = any(
                keyword in text_lower
                for keyword in self.config.FINANCIAL_KEYWORDS
            )
            relationship_problems = any(
                keyword in text_lower
                for keyword in self.config.RELATIONSHIP_KEYWORDS
            )
            legal_problems = any(
                keyword in text_lower
                for keyword in self.config.LEGAL_KEYWORDS
            )

            # Calculate simple risk score (0-1)
            risk_factors = [
                opioid_mentioned,
                alcohol_mentioned,
                overdose_indicated,
                depression_mentioned,
                suicide_history,
                financial_crisis,
                relationship_problems,
                legal_problems,
            ]
            risk_score = sum(risk_factors) / len(risk_factors)

            return {
                "opioid_mentioned": opioid_mentioned,
                "opioid_types": opioid_types,
                "alcohol_mentioned": alcohol_mentioned,
                "overdose_indicated": overdose_indicated,
                "depression_mentioned": depression_mentioned,
                "anxiety_mentioned": anxiety_mentioned,
                "suicide_history": suicide_history,
                "financial_crisis": financial_crisis,
                "relationship_problems": relationship_problems,
                "legal_problems": legal_problems,
                "risk_score": risk_score,
            }

        # Register UDF
        from pyspark.sql.types import MapType
        risk_factors_udf_func = udf(
            extract_risk_factors_udf,
            MapType(StringType(), StringType())
        )

        # Apply risk factor extraction
        df = df.withColumn(
            "risk_factors",
            risk_factors_udf_func(col("text"), col("ner_chunk"))
        )

        return df

    def _df_to_records(self, df: DataFrame) -> List[NVDRSRecord]:
        """Convert Spark DataFrame to list of NVDRSRecord objects"""
        records = []

        for row in df.collect():
            # Extract risk factors from the dictionary
            risk_dict = row.get("risk_factors", {})

            risk_factors = RiskFactors(
                substance_use=SubstanceUse(
                    opioid_mentioned=risk_dict.get("opioid_mentioned", False),
                    opioid_types=risk_dict.get("opioid_types", []),
                    alcohol_mentioned=risk_dict.get("alcohol_mentioned", False),
                    overdose_indicated=risk_dict.get("overdose_indicated", False),
                ),
                mental_health=MentalHealthIndicators(
                    depression_mentioned=risk_dict.get("depression_mentioned", False),
                    anxiety_mentioned=risk_dict.get("anxiety_mentioned", False),
                    suicide_history=risk_dict.get("suicide_history", False),
                ),
                social_stressors=SocialStressors(
                    financial_crisis=risk_dict.get("financial_crisis", False),
                    relationship_problems=risk_dict.get("relationship_problems", False),
                    legal_problems=risk_dict.get("legal_problems", False),
                ),
                risk_score=float(risk_dict.get("risk_score", 0.0)),
            )

            record = NVDRSRecord(
                record_id=row.get("record_id", f"NVDRS_{datetime.now().strftime('%Y%m%d%H%M%S')}"),
                cme_narrative=row.get("CME_Narrative_redacted") or row.get("CME_Narrative"),
                le_narrative=row.get("LE_Narrative_redacted") or row.get("LE_Narrative"),
                risk_factors=risk_factors,
            )

            records.append(record)

        return records

    def _generate_summary_statistics(self, records: List[NVDRSRecord]) -> Dict:
        """Generate summary statistics from processed records"""
        if not records:
            return {}

        total = len(records)

        stats = {
            "total_records": total,
            "substance_use": {
                "opioid_mentions": sum(1 for r in records if r.risk_factors.substance_use.opioid_mentioned),
                "alcohol_mentions": sum(1 for r in records if r.risk_factors.substance_use.alcohol_mentioned),
                "overdose_indicated": sum(1 for r in records if r.risk_factors.substance_use.overdose_indicated),
            },
            "mental_health": {
                "depression": sum(1 for r in records if r.risk_factors.mental_health.depression_mentioned),
                "anxiety": sum(1 for r in records if r.risk_factors.mental_health.anxiety_mentioned),
                "suicide_history": sum(1 for r in records if r.risk_factors.mental_health.suicide_history),
            },
            "social_stressors": {
                "financial_crisis": sum(1 for r in records if r.risk_factors.social_stressors.financial_crisis),
                "relationship_problems": sum(1 for r in records if r.risk_factors.social_stressors.relationship_problems),
                "legal_problems": sum(1 for r in records if r.risk_factors.social_stressors.legal_problems),
            },
            "risk_scores": {
                "mean": sum(r.risk_factors.risk_score for r in records) / total,
                "high_risk_count": sum(1 for r in records if r.risk_factors.risk_score > 0.6),
            }
        }

        # Calculate percentages
        for category in ["substance_use", "mental_health", "social_stressors"]:
            for key, count in stats[category].items():
                stats[category][f"{key}_percent"] = round(count / total * 100, 2)

        return stats

    def save_results(
        self,
        result: PipelineResult,
        output_path: str,
        format: str = "parquet"
    ):
        """
        Save pipeline results to file

        Args:
            result: PipelineResult to save
            output_path: Output file path
            format: Output format (parquet, json, csv)
        """
        logger.info(f"Saving results to {output_path} (format: {format})")

        # Convert to dictionaries
        data = [record.model_dump() for record in result.records]

        # Create DataFrame
        df = self.spark.createDataFrame(data)

        # Save in specified format
        if format == "parquet":
            df.write.mode("overwrite").parquet(output_path)
        elif format == "json":
            df.write.mode("overwrite").json(output_path)
        elif format == "csv":
            df.write.mode("overwrite").option("header", True).csv(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Results saved successfully")

    def stop(self):
        """Stop Spark session"""
        if self.spark:
            self.spark.stop()
            logger.info("Spark session stopped")


if __name__ == "__main__":
    # Example usage
    print("NVDRS NLP Pipeline Example")
    print("=" * 60)

    # This would normally be run with real NVDRS data
    print("\nTo use this pipeline:")
    print("1. Install dependencies: pip install spark-nlp pyspark")
    print("2. Load your NVDRS data into a Spark DataFrame")
    print("3. Initialize the pipeline: pipeline = NVDRSPipeline()")
    print("4. Process narratives: results = pipeline.process_narratives(df)")
    print("5. Save results: pipeline.save_results(results, 'output.parquet')")
    print("\nSee examples/nvdrs_example.py for complete example")
