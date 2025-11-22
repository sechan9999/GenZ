"""
PROGRAM: nvdrs_train_nlp_spark.py
PURPOSE: Spark NLP with BioBERT for train suicide case identification
AUTHOR: Public Health Research
DATE: 2025-11-22

APPROACH: Use Spark NLP + BioBERT for entity extraction and fuzzy matching
INTEGRATES WITH: nvdrs_train_nlp_case_definition.sas

REQUIREMENTS:
    pip install pyspark spark-nlp pandas numpy fuzzywuzzy python-Levenshtein
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, udf, when, concat_ws, array_contains,
    lower, regexp_replace, trim, explode, split
)
from pyspark.sql.types import StringType, IntegerType, ArrayType, StructType, StructField

from sparknlp.base import DocumentAssembler, Pipeline, Finisher
from sparknlp.annotator import (
    Tokenizer,
    BertForTokenClassification,
    NerConverter,
    Normalizer,
    StopWordsCleaner
)

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz, process
import re
from typing import List, Dict, Tuple

# =============================================================================
# SECTION 1: INITIALIZE SPARK SESSION WITH SPARK NLP
# =============================================================================

def create_spark_session():
    """Initialize Spark session with Spark NLP library."""
    spark = SparkSession.builder \
        .appName("NVDRS_Train_Suicide_NLP") \
        .config("spark.driver.memory", "16g") \
        .config("spark.executor.memory", "16g") \
        .config("spark.kryoserializer.buffer.max", "2000M") \
        .config("spark.driver.maxResultSize", "0") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.1.0") \
        .getOrCreate()

    return spark


# =============================================================================
# SECTION 2: DATA LOADING AND PREPROCESSING
# =============================================================================

def load_nvdrs_data(spark, input_path: str):
    """
    Load NVDRS restricted-use data from CSV or Parquet.

    Expected columns:
        - IncidentID, Year, State, Age, Sex, Race, Hispanic
        - CMENotes, LENarrative, Circumstances
        - ICD10Code, WeaponType1
    """
    # Load data (adjust format as needed)
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # Filter to suicides only
    df = df.filter(col("DeathManner") == "Suicide")

    # Combine narrative fields
    df = df.withColumn(
        "narrative_all",
        concat_ws(" ",
            col("CMENotes"),
            col("LENarrative"),
            col("Circumstances"),
            col("InjuryLocation"),
            col("WeaponDescription")
        )
    )

    # Clean text
    df = df.withColumn("narrative_clean", lower(col("narrative_all")))
    df = df.withColumn("narrative_clean", regexp_replace(col("narrative_clean"), r"[^\w\s]", " "))
    df = df.withColumn("narrative_clean", regexp_replace(col("narrative_clean"), r"\s+", " "))
    df = df.withColumn("narrative_clean", trim(col("narrative_clean")))

    return df


# =============================================================================
# SECTION 3: FUZZY MATCHING FUNCTIONS
# =============================================================================

# Define train-related target terms
TRAIN_TERMS = [
    "train", "railway", "railroad", "railcar", "locomotive",
    "metro", "subway", "transit", "rail", "amtrak", "freight"
]

# Common typos and variations
TRAIN_TYPOS = [
    "tran", "trian", "traing", "trainm", "trrain", "tarins",
    "track", "trace", "tracke", "tracks"
]

def fuzzy_match_terms(text: str, threshold: int = 80) -> Dict[str, any]:
    """
    Fuzzy match train-related terms using Levenshtein distance.

    Args:
        text: Input narrative text
        threshold: Similarity threshold (0-100)

    Returns:
        Dictionary with match results
    """
    if not text:
        return {
            "matched": False,
            "term": None,
            "score": 0,
            "method": None
        }

    words = text.split()
    best_match = {"matched": False, "term": None, "score": 0, "method": None}

    for word in words:
        # Skip very short words
        if len(word) < 3:
            continue

        # Check against target terms
        for target in TRAIN_TERMS:
            # Exact match
            if word == target:
                return {
                    "matched": True,
                    "term": target,
                    "score": 100,
                    "method": "exact"
                }

            # Fuzzy match
            score = fuzz.ratio(word, target)
            if score >= threshold and score > best_match["score"]:
                best_match = {
                    "matched": True,
                    "term": f"{word}->{target}",
                    "score": score,
                    "method": "fuzzy"
                }

    return best_match

# Register as UDF
fuzzy_match_udf = udf(lambda x: fuzzy_match_terms(x),
                       StructType([
                           StructField("matched", IntegerType()),
                           StructField("term", StringType()),
                           StructField("score", IntegerType()),
                           StructField("method", StringType())
                       ]))


# =============================================================================
# SECTION 4: REGEX PATTERN MATCHING
# =============================================================================

def regex_pattern_match(text: str) -> Dict[str, any]:
    """
    Match train-related patterns using regex.

    Patterns:
        1. Train variations with typos
        2. Context-aware track mentions
        3. Railway/railroad
        4. Location indicators
        5. Action phrases (struck by, hit by)
    """
    if not text:
        return {"matched": False, "patterns": []}

    patterns_found = []

    # Pattern 1: Train variations (handles typos)
    if re.search(r'\btr[ae]?[iam]{0,2}n[gs]?\b', text, re.IGNORECASE):
        patterns_found.append("TRAIN_TYPO")

    # Pattern 2: Track with positive context
    if re.search(r'\b(hit|struck|on|by|under|railroad|railway)\s+track', text, re.IGNORECASE):
        patterns_found.append("TRACK_CONTEXT")

    # Pattern 3: Railway/Railroad
    if re.search(r'\brail(way|road|car)', text, re.IGNORECASE):
        patterns_found.append("RAILWAY")

    # Pattern 4: Locomotive
    if re.search(r'\blocomotiv', text, re.IGNORECASE):
        patterns_found.append("LOCOMOTIVE")

    # Pattern 5: Transit systems
    if re.search(r'\b(metro|subway|transit|light rail)', text, re.IGNORECASE):
        patterns_found.append("TRANSIT")

    # Pattern 6: Action phrases
    if re.search(r'(struck|hit|killed|run over|ran over).{0,30}(train|locomotive)', text, re.IGNORECASE):
        patterns_found.append("STRUCK_TRAIN")

    # Pattern 7: Location indicators
    if re.search(r'\b(train station|rail yard|tracks?|crossing|platform)', text, re.IGNORECASE):
        patterns_found.append("LOCATION")

    # Pattern 8: Personnel mentions
    if re.search(r'\b(conductor|engineer|amtrak|freight|passenger)', text, re.IGNORECASE):
        patterns_found.append("PERSONNEL")

    return {
        "matched": len(patterns_found) > 0,
        "patterns": patterns_found,
        "count": len(patterns_found)
    }

# Register as UDF
regex_match_udf = udf(lambda x: regex_pattern_match(x),
                       StructType([
                           StructField("matched", IntegerType()),
                           StructField("patterns", ArrayType(StringType())),
                           StructField("count", IntegerType())
                       ]))


# =============================================================================
# SECTION 5: SPARK NLP PIPELINE WITH BIOBERT
# =============================================================================

def build_clinical_nlp_pipeline():
    """
    Build Spark NLP pipeline for entity extraction.
    Uses BioBERT for clinical/medical terminology recognition.
    """

    # 1. Document Assembler
    document_assembler = DocumentAssembler() \
        .setInputCol("narrative_clean") \
        .setOutputCol("document") \
        .setCleanupMode("shrink")

    # 2. Tokenizer
    tokenizer = Tokenizer() \
        .setInputCols(["document"]) \
        .setOutputCol("token")

    # 3. Normalizer (handles case, punctuation)
    normalizer = Normalizer() \
        .setInputCols(["token"]) \
        .setOutputCol("normalized") \
        .setLowercase(True)

    # 4. Stop words cleaner
    stopwords_cleaner = StopWordsCleaner() \
        .setInputCols(["normalized"]) \
        .setOutputCol("cleanTokens") \
        .setCaseSensitive(False)

    # 5. BioBERT for NER (Named Entity Recognition)
    # Note: Use clinical model for medical terminology
    try:
        ner_model = BertForTokenClassification.pretrained(
            "bert_token_classifier_ner_clinical",
            "en",
            "clinical/models"
        ) \
            .setInputCols(["document", "token"]) \
            .setOutputCol("ner") \
            .setCaseSensitive(False)
    except:
        # Fallback to general NER if clinical model unavailable
        print("Clinical NER model not available, using general NER")
        ner_model = BertForTokenClassification.pretrained(
            "bert_token_classifier_ner_base",
            "en"
        ) \
            .setInputCols(["document", "token"]) \
            .setOutputCol("ner") \
            .setCaseSensitive(False)

    # 6. Convert NER chunks
    ner_converter = NerConverter() \
        .setInputCols(["document", "token", "ner"]) \
        .setOutputCol("ner_chunk")

    # 7. Finisher (converts Spark NLP annotations to readable format)
    finisher = Finisher() \
        .setInputCols(["ner_chunk"]) \
        .setOutputCols(["entities"]) \
        .setOutputAsArray(True) \
        .setCleanAnnotations(False)

    # Build pipeline
    pipeline = Pipeline(stages=[
        document_assembler,
        tokenizer,
        normalizer,
        stopwords_cleaner,
        ner_model,
        ner_converter,
        finisher
    ])

    return pipeline


# =============================================================================
# SECTION 6: ENSEMBLE CLASSIFICATION
# =============================================================================

def apply_ensemble_logic(df):
    """
    Apply ensemble classification combining:
        1. Fuzzy matching
        2. Regex patterns
        3. NER entities
        4. ICD codes (gold standard)
    """

    # Apply fuzzy matching
    df = df.withColumn("fuzzy_result", fuzzy_match_udf(col("narrative_clean")))
    df = df.withColumn("fuzzy_match", col("fuzzy_result.matched"))
    df = df.withColumn("fuzzy_term", col("fuzzy_result.term"))
    df = df.withColumn("fuzzy_score", col("fuzzy_result.score"))

    # Apply regex patterns
    df = df.withColumn("regex_result", regex_match_udf(col("narrative_clean")))
    df = df.withColumn("regex_match", col("regex_result.matched"))
    df = df.withColumn("regex_patterns", col("regex_result.patterns"))
    df = df.withColumn("regex_count", col("regex_result.count"))

    # Check if entities contain train-related terms
    df = df.withColumn(
        "entity_match",
        when(
            array_contains(col("entities"), "train") |
            array_contains(col("entities"), "railway") |
            array_contains(col("entities"), "railroad") |
            array_contains(col("entities"), "locomotive"),
            1
        ).otherwise(0)
    )

    # ICD code validation
    df = df.withColumn(
        "icd_train",
        when(
            col("ICD10Code").rlike("X81\\.[0189]?"),
            1
        ).otherwise(0)
    )

    # Weapon type validation
    df = df.withColumn(
        "weapon_train",
        when(
            lower(col("WeaponType1")).isin(["train", "railway", "railroad"]),
            1
        ).otherwise(0)
    )

    # Calculate confidence level
    df = df.withColumn(
        "method_count",
        col("fuzzy_match") + col("regex_match") + col("entity_match")
    )

    df = df.withColumn(
        "confidence",
        when(col("method_count") >= 2, "HIGH")
        .when((col("fuzzy_score") == 100) | (col("regex_count") >= 3), "HIGH")
        .when(col("method_count") >= 1, "MEDIUM")
        .otherwise("LOW")
    )

    # Final classification
    df = df.withColumn(
        "train_suicide_nlp",
        when(
            (col("confidence").isin(["HIGH", "MEDIUM"])),
            1
        ).otherwise(0)
    )

    # Combine with coded data
    df = df.withColumn(
        "train_suicide_coded",
        when((col("icd_train") == 1) | (col("weapon_train") == 1), 1)
        .otherwise(0)
    )

    # Final indicator (use either method)
    df = df.withColumn(
        "train_suicide",
        when(
            (col("train_suicide_nlp") == 1) | (col("train_suicide_coded") == 1),
            1
        ).otherwise(0)
    )

    # Flag source of detection
    df = df.withColumn(
        "detection_source",
        when((col("train_suicide_nlp") == 1) & (col("train_suicide_coded") == 1), "BOTH")
        .when(col("train_suicide_nlp") == 1, "NLP_ONLY")
        .when(col("train_suicide_coded") == 1, "CODED_ONLY")
        .otherwise("NONE")
    )

    return df


# =============================================================================
# SECTION 7: VALIDATION AND METRICS
# =============================================================================

def calculate_performance_metrics(df):
    """
    Calculate sensitivity, specificity, PPV, NPV
    using ICD/weapon codes as gold standard.
    """

    # Convert to Pandas for easier calculation
    results = df.select(
        "train_suicide_nlp",
        "train_suicide_coded",
        "confidence"
    ).toPandas()

    # Confusion matrix
    tp = ((results['train_suicide_nlp'] == 1) & (results['train_suicide_coded'] == 1)).sum()
    tn = ((results['train_suicide_nlp'] == 0) & (results['train_suicide_coded'] == 0)).sum()
    fp = ((results['train_suicide_nlp'] == 1) & (results['train_suicide_coded'] == 0)).sum()
    fn = ((results['train_suicide_nlp'] == 0) & (results['train_suicide_coded'] == 1)).sum()

    # Metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    metrics = {
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV": ppv,
        "NPV": npv,
        "Accuracy": accuracy,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
    }

    return metrics


# =============================================================================
# SECTION 8: MAIN EXECUTION WORKFLOW
# =============================================================================

def main(input_path: str, output_path: str):
    """
    Main workflow for train suicide NLP classification.

    Args:
        input_path: Path to NVDRS CSV file
        output_path: Path for output files
    """

    print("=" * 80)
    print("NVDRS TRAIN SUICIDE NLP CLASSIFICATION")
    print("=" * 80)

    # 1. Initialize Spark
    print("\n[1/7] Initializing Spark session...")
    spark = create_spark_session()

    # 2. Load data
    print("[2/7] Loading NVDRS data...")
    df = load_nvdrs_data(spark, input_path)
    print(f"   Loaded {df.count()} suicide cases")

    # 3. Build and apply NLP pipeline
    print("[3/7] Building Spark NLP pipeline with BioBERT...")
    nlp_pipeline = build_clinical_nlp_pipeline()

    print("[4/7] Applying NLP pipeline (this may take several minutes)...")
    pipeline_model = nlp_pipeline.fit(df)
    df = pipeline_model.transform(df)

    # 4. Apply ensemble classification
    print("[5/7] Applying ensemble classification...")
    df = apply_ensemble_logic(df)

    # 5. Calculate metrics
    print("[6/7] Calculating performance metrics...")
    metrics = calculate_performance_metrics(df)

    print("\n" + "=" * 50)
    print("PERFORMANCE METRICS")
    print("=" * 50)
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric:.<30} {value:.3f}")
        else:
            print(f"{metric:.<30} {value}")

    # 6. Summary statistics
    print("\n" + "=" * 50)
    print("CASE CLASSIFICATION SUMMARY")
    print("=" * 50)

    summary = df.groupBy("detection_source").count().toPandas()
    print(summary.to_string(index=False))

    # 7. Save results
    print(f"\n[7/7] Saving results to {output_path}...")

    # Save full results as Parquet
    df.write.mode("overwrite").parquet(f"{output_path}/train_suicide_nlp_results.parquet")

    # Save summary as CSV
    df.select(
        "IncidentID", "Year", "State", "Age", "Sex",
        "train_suicide", "train_suicide_nlp", "train_suicide_coded",
        "detection_source", "confidence",
        "fuzzy_match", "regex_match", "entity_match",
        "narrative_clean"
    ).toPandas().to_csv(f"{output_path}/train_suicide_summary.csv", index=False)

    # Save cases for manual review
    review_sample = df.filter(
        (col("train_suicide_nlp") != col("train_suicide_coded"))
    ).limit(200).toPandas()

    review_sample.to_csv(f"{output_path}/manual_review_sample.csv", index=False)

    print("\n✓ Analysis complete!")
    print(f"   - Full results: {output_path}/train_suicide_nlp_results.parquet")
    print(f"   - Summary CSV: {output_path}/train_suicide_summary.csv")
    print(f"   - Manual review: {output_path}/manual_review_sample.csv")

    spark.stop()


# =============================================================================
# SECTION 9: EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == "__main__":
    """
    Example usage:

    python nvdrs_train_nlp_spark.py
    """

    # Configuration
    INPUT_PATH = "/path/to/nvdrs/restricted_use_data.csv"
    OUTPUT_PATH = "/path/to/output"

    # Run analysis
    main(INPUT_PATH, OUTPUT_PATH)


# =============================================================================
# SECTION 10: UTILITY FUNCTIONS FOR INTEGRATION WITH SAS
# =============================================================================

def export_for_sas_comparison(spark_results_path: str, sas_results_path: str, output_path: str):
    """
    Compare Spark NLP results with SAS NLP results.

    Args:
        spark_results_path: Path to Spark NLP results (parquet)
        sas_results_path: Path to SAS NLP results (CSV from PROC EXPORT)
        output_path: Path for comparison output
    """

    spark = create_spark_session()

    # Load results
    spark_results = spark.read.parquet(spark_results_path)
    sas_results = spark.read.csv(sas_results_path, header=True, inferSchema=True)

    # Join on IncidentID
    comparison = spark_results.join(
        sas_results.select("IncidentID", "train_suicide_nlp").withColumnRenamed("train_suicide_nlp", "sas_nlp"),
        on="IncidentID",
        how="inner"
    )

    # Calculate agreement
    comparison = comparison.withColumn(
        "agreement",
        when(col("train_suicide_nlp") == col("sas_nlp"), "AGREE")
        .otherwise("DISAGREE")
    )

    # Summary
    agreement_summary = comparison.groupBy("agreement").count().toPandas()
    print("\nSAS vs Spark NLP Agreement:")
    print(agreement_summary)

    # Save comparison
    comparison.toPandas().to_csv(f"{output_path}/sas_spark_comparison.csv", index=False)

    spark.stop()


# =============================================================================
# ADDITIONAL HELPER FUNCTIONS
# =============================================================================

def get_example_narratives():
    """Return example narratives with typos for testing."""
    return [
        "Victim was struck by a tran at the crossing",  # typo: tran
        "Found on railroad tracks, hit by trainm",      # typo: trainm
        "Jumped in front of Metro at station",
        "Lay down on tracke before locomotive arrived", # typo: tracke
        "Training course was completed yesterday",       # false positive
        "Tracked down suspect at residence",             # false positive
        "Died at rail yard, run over by freight train",
        "Struck by Amtrak passenger trian",              # typo: trian
    ]


def test_nlp_functions():
    """Test NLP functions with example data."""
    examples = get_example_narratives()

    print("\n" + "=" * 80)
    print("TESTING NLP FUNCTIONS")
    print("=" * 80)

    for i, text in enumerate(examples, 1):
        print(f"\n[{i}] {text}")

        # Test fuzzy matching
        fuzzy = fuzzy_match_terms(text.lower())
        print(f"   Fuzzy: {fuzzy}")

        # Test regex
        regex = regex_pattern_match(text.lower())
        print(f"   Regex: {regex}")


# Uncomment to run tests
# test_nlp_functions()
