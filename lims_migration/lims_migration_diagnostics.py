"""
LIMS Data Migration Diagnostics for Azure Databricks

This module helps diagnose and fix common LIMS data migration issues:
- Incomplete data migrations
- Slow migration performance
- Failed records
- Data quality issues

Author: GenZ Agent Project
Date: 2025-11-24
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, sum as spark_sum, avg, max as spark_max,
    min as spark_min, current_timestamp, datediff, lit, when
)
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, LongType, DoubleType
from delta.tables import DeltaTable
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LIMSMigrationDiagnostics:
    """
    Comprehensive diagnostics for LIMS data migration issues.

    Identifies:
    - Missing records
    - Slow migrations
    - Data quality issues
    - Performance bottlenecks
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def compare_source_target_counts(
        self,
        source_path: str,
        target_path: str,
        key_column: str = "sample_id"
    ) -> Dict[str, int]:
        """
        Compare record counts between source and target.

        Args:
            source_path: Path to source table (JDBC, Delta, etc.)
            target_path: Path to target Delta table
            key_column: Primary key column for comparison

        Returns:
            Dictionary with count comparison
        """
        logger.info("🔍 Comparing source and target counts...")

        # Read source and target
        source_df = self.spark.read.format("delta").load(source_path)
        target_df = self.spark.read.format("delta").load(target_path)

        source_count = source_df.count()
        target_count = target_df.count()
        missing_count = source_count - target_count

        results = {
            "source_count": source_count,
            "target_count": target_count,
            "missing_count": missing_count,
            "migration_percentage": (target_count / source_count * 100) if source_count > 0 else 0
        }

        logger.info(f"📊 Source: {source_count:,} | Target: {target_count:,} | Missing: {missing_count:,}")
        logger.info(f"✅ Migration Progress: {results['migration_percentage']:.2f}%")

        return results

    def find_missing_records(
        self,
        source_path: str,
        target_path: str,
        key_column: str = "sample_id"
    ):
        """
        Identify specific records that didn't migrate.

        Returns DataFrame of missing records.
        """
        logger.info("🔍 Finding missing records...")

        source_df = self.spark.read.format("delta").load(source_path)
        target_df = self.spark.read.format("delta").load(target_path)

        # Left anti join to find records in source but not in target
        missing_df = source_df.join(
            target_df.select(key_column),
            on=key_column,
            how="left_anti"
        )

        missing_count = missing_df.count()
        logger.info(f"❌ Found {missing_count:,} missing records")

        if missing_count > 0:
            # Analyze missing records
            logger.info("\n📋 Sample of missing records:")
            missing_df.select(key_column, "test_code", "collection_date").show(10, truncate=False)

            # Analyze patterns in missing data
            logger.info("\n🔬 Missing records by test type:")
            missing_df.groupBy("test_code").count().orderBy(col("count").desc()).show(10)

        return missing_df

    def analyze_migration_performance(
        self,
        target_path: str,
        migration_start_time: datetime,
        expected_throughput: int = 1000  # records per minute
    ) -> Dict[str, any]:
        """
        Analyze migration speed and identify bottlenecks.

        Args:
            target_path: Path to target table
            migration_start_time: When migration started
            expected_throughput: Expected records per minute

        Returns:
            Performance metrics
        """
        logger.info("📊 Analyzing migration performance...")

        target_df = self.spark.read.format("delta").load(target_path)
        current_count = target_df.count()

        # Calculate duration
        elapsed_time = datetime.now() - migration_start_time
        elapsed_minutes = elapsed_time.total_seconds() / 60

        # Calculate throughput
        actual_throughput = current_count / elapsed_minutes if elapsed_minutes > 0 else 0
        performance_ratio = (actual_throughput / expected_throughput * 100) if expected_throughput > 0 else 0

        # Estimate completion time
        if actual_throughput > 0:
            # Assuming we know total records from source
            # This would need to be passed in or calculated
            estimated_total_minutes = current_count / actual_throughput
            estimated_completion = migration_start_time + timedelta(minutes=estimated_total_minutes)
        else:
            estimated_completion = None

        results = {
            "records_migrated": current_count,
            "elapsed_time_minutes": elapsed_minutes,
            "actual_throughput_per_minute": actual_throughput,
            "expected_throughput_per_minute": expected_throughput,
            "performance_percentage": performance_ratio,
            "estimated_completion": estimated_completion
        }

        logger.info(f"⏱️  Elapsed: {elapsed_minutes:.1f} min")
        logger.info(f"📈 Throughput: {actual_throughput:.0f} records/min (expected: {expected_throughput})")
        logger.info(f"🎯 Performance: {performance_ratio:.1f}% of expected")

        if performance_ratio < 50:
            logger.warning("⚠️  SLOW MIGRATION DETECTED - Performance below 50% of expected")
            self._suggest_performance_fixes()

        return results

    def check_data_quality(
        self,
        target_path: str
    ) -> Dict[str, any]:
        """
        Check data quality issues that might slow migration.

        Checks:
        - Null values
        - Duplicate records
        - Invalid formats
        - Outliers
        """
        logger.info("🔬 Checking data quality...")

        df = self.spark.read.format("delta").load(target_path)
        total_count = df.count()

        results = {}

        # Check for nulls in critical columns
        critical_columns = ["sample_id", "test_code", "result_value"]
        for col_name in critical_columns:
            if col_name in df.columns:
                null_count = df.filter(col(col_name).isNull()).count()
                null_percentage = (null_count / total_count * 100) if total_count > 0 else 0
                results[f"{col_name}_nulls"] = {
                    "count": null_count,
                    "percentage": null_percentage
                }

                if null_percentage > 5:
                    logger.warning(f"⚠️  {col_name} has {null_percentage:.2f}% null values")

        # Check for duplicates
        duplicate_count = df.groupBy("sample_id").count().filter("count > 1").count()
        results["duplicate_samples"] = duplicate_count

        if duplicate_count > 0:
            logger.warning(f"⚠️  Found {duplicate_count:,} duplicate sample_ids")

        # Check for data skew (problematic for performance)
        if "test_code" in df.columns:
            test_distribution = df.groupBy("test_code").count().orderBy(col("count").desc()).limit(10)
            logger.info("\n📊 Top 10 test types by volume:")
            test_distribution.show()

            # Check if top test is > 50% of data (indicates skew)
            max_test_count = test_distribution.first()["count"]
            skew_percentage = (max_test_count / total_count * 100) if total_count > 0 else 0
            results["data_skew_percentage"] = skew_percentage

            if skew_percentage > 50:
                logger.warning(f"⚠️  DATA SKEW DETECTED: One test type represents {skew_percentage:.1f}% of data")
                logger.warning("    This can cause uneven partition sizes and slow performance")

        logger.info("\n✅ Data quality check complete")
        return results

    def identify_slow_partitions(
        self,
        target_path: str
    ):
        """
        Identify partitions that are taking longer to process.
        """
        logger.info("🔍 Analyzing partition performance...")

        df = self.spark.read.format("delta").load(target_path)

        # Get partition information
        if df.rdd.getNumPartitions() > 0:
            partition_counts = df.rdd.mapPartitions(
                lambda iterator: [sum(1 for _ in iterator)]
            ).collect()

            avg_records_per_partition = sum(partition_counts) / len(partition_counts)
            max_partition_size = max(partition_counts)
            min_partition_size = min(partition_counts)

            skew_ratio = max_partition_size / avg_records_per_partition if avg_records_per_partition > 0 else 0

            logger.info(f"📦 Total partitions: {len(partition_counts)}")
            logger.info(f"📊 Avg records/partition: {avg_records_per_partition:.0f}")
            logger.info(f"📈 Max partition size: {max_partition_size}")
            logger.info(f"📉 Min partition size: {min_partition_size}")
            logger.info(f"⚖️  Skew ratio: {skew_ratio:.2f}x")

            if skew_ratio > 3:
                logger.warning("⚠️  PARTITION SKEW DETECTED")
                logger.warning("    Some partitions are 3x larger than average")
                logger.warning("    Recommendation: Repartition data before migration")

    def _suggest_performance_fixes(self):
        """Suggest fixes for slow migrations."""
        logger.info("\n💡 PERFORMANCE OPTIMIZATION SUGGESTIONS:")
        logger.info("1. Increase cluster size (more workers)")
        logger.info("2. Repartition data: df.repartition(200)")
        logger.info("3. Increase parallelism: spark.conf.set('spark.sql.shuffle.partitions', '400')")
        logger.info("4. Use Z-ordering on Delta tables")
        logger.info("5. Enable adaptive query execution (AQE)")
        logger.info("6. Check for data skew and salt skewed keys")
        logger.info("7. Increase executor memory")
        logger.info("8. Use .cache() for repeatedly accessed DataFrames")

    def generate_migration_report(
        self,
        source_path: str,
        target_path: str,
        migration_start_time: datetime,
        report_path: str = "/tmp/migration_report"
    ):
        """
        Generate comprehensive migration report.
        """
        logger.info("=" * 70)
        logger.info("📋 LIMS MIGRATION DIAGNOSTIC REPORT")
        logger.info("=" * 70)
        logger.info(f"Report Generated: {datetime.now()}")
        logger.info(f"Migration Started: {migration_start_time}")
        logger.info("")

        # 1. Count comparison
        count_results = self.compare_source_target_counts(source_path, target_path)

        # 2. Find missing records
        missing_df = self.find_missing_records(source_path, target_path)

        # 3. Performance analysis
        perf_results = self.analyze_migration_performance(target_path, migration_start_time)

        # 4. Data quality
        quality_results = self.check_data_quality(target_path)

        # 5. Partition analysis
        self.identify_slow_partitions(target_path)

        # Save missing records to Delta for review
        if missing_df.count() > 0:
            missing_records_path = f"{report_path}/missing_records"
            missing_df.write.format("delta").mode("overwrite").save(missing_records_path)
            logger.info(f"\n💾 Missing records saved to: {missing_records_path}")

        logger.info("\n" + "=" * 70)
        logger.info("✅ DIAGNOSTIC REPORT COMPLETE")
        logger.info("=" * 70)

        return {
            "counts": count_results,
            "performance": perf_results,
            "quality": quality_results,
            "missing_records_path": f"{report_path}/missing_records" if missing_df.count() > 0 else None
        }


class MigrationHealthMonitor:
    """
    Real-time monitoring for ongoing migrations.

    Use this to track migration progress and alert on issues.
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.metrics_path = "/tmp/migration_metrics"
        self._initialize_metrics_table()

    def _initialize_metrics_table(self):
        """Create metrics tracking table."""
        schema = StructType([
            StructField("timestamp", TimestampType(), False),
            StructField("records_migrated", LongType(), False),
            StructField("throughput_per_minute", DoubleType(), True),
            StructField("error_count", LongType(), True),
            StructField("status", StringType(), True)
        ])

        try:
            DeltaTable.createIfNotExists(self.spark) \
                .location(self.metrics_path) \
                .addColumns(schema) \
                .execute()
            logger.info(f"📊 Metrics table initialized at {self.metrics_path}")
        except Exception as e:
            logger.warning(f"Metrics table may already exist: {e}")

    def record_checkpoint(
        self,
        target_path: str,
        error_count: int = 0,
        status: str = "running"
    ):
        """
        Record a checkpoint during migration.

        Call this periodically (e.g., every 5 minutes) to track progress.
        """
        current_count = self.spark.read.format("delta").load(target_path).count()

        # Calculate throughput
        metrics_df = self.spark.read.format("delta").load(self.metrics_path)

        if metrics_df.count() > 0:
            last_checkpoint = metrics_df.orderBy(col("timestamp").desc()).first()
            time_diff = (datetime.now() - last_checkpoint["timestamp"]).total_seconds() / 60
            records_diff = current_count - last_checkpoint["records_migrated"]
            throughput = records_diff / time_diff if time_diff > 0 else 0
        else:
            throughput = 0

        # Record checkpoint
        checkpoint_data = [(
            datetime.now(),
            current_count,
            throughput,
            error_count,
            status
        )]

        checkpoint_df = self.spark.createDataFrame(
            checkpoint_data,
            ["timestamp", "records_migrated", "throughput_per_minute", "error_count", "status"]
        )

        checkpoint_df.write.format("delta").mode("append").save(self.metrics_path)

        logger.info(f"✅ Checkpoint recorded: {current_count:,} records, {throughput:.0f}/min throughput")

    def get_migration_trend(self):
        """
        Get migration trend over time.
        """
        metrics_df = self.spark.read.format("delta").load(self.metrics_path)

        logger.info("\n📈 MIGRATION TREND:")
        metrics_df.orderBy("timestamp").select(
            "timestamp",
            "records_migrated",
            "throughput_per_minute",
            "error_count",
            "status"
        ).show(20, truncate=False)

        return metrics_df

    def check_health(self, alert_threshold_throughput: int = 500):
        """
        Check current migration health and alert on issues.
        """
        metrics_df = self.spark.read.format("delta").load(self.metrics_path)

        if metrics_df.count() < 2:
            logger.info("⏳ Not enough data points for health check")
            return

        # Get last 5 checkpoints
        recent_metrics = metrics_df.orderBy(col("timestamp").desc()).limit(5)

        avg_throughput = recent_metrics.agg(avg("throughput_per_minute")).first()[0]
        total_errors = recent_metrics.agg(spark_sum("error_count")).first()[0]

        logger.info("\n🏥 MIGRATION HEALTH CHECK:")
        logger.info(f"📊 Avg throughput (last 5 checkpoints): {avg_throughput:.0f} records/min")
        logger.info(f"❌ Total errors: {total_errors}")

        # Alerts
        if avg_throughput < alert_threshold_throughput:
            logger.warning(f"⚠️  ALERT: Throughput ({avg_throughput:.0f}) below threshold ({alert_threshold_throughput})")
            logger.warning("    Possible causes: cluster underprovisioned, data skew, network issues")

        if total_errors > 100:
            logger.error(f"🚨 ALERT: High error count ({total_errors})")
            logger.error("    Check error logs and consider using retry queue system")


# Example usage
if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("LIMSMigrationDiagnostics") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()

    # Example: Diagnose migration issues
    diagnostics = LIMSMigrationDiagnostics(spark)

    source_path = "/mnt/delta/lims_source"
    target_path = "/mnt/delta/lims_target"
    migration_start = datetime.now() - timedelta(hours=2)

    report = diagnostics.generate_migration_report(
        source_path=source_path,
        target_path=target_path,
        migration_start_time=migration_start
    )

    print("\n📊 Full Report:", report)
