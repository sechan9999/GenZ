"""
Optimized LIMS Data Migration Pipeline for Azure Databricks

This module provides a high-performance, fault-tolerant migration pipeline that:
- Handles large-scale LIMS data migrations
- Uses checkpointing for fault tolerance
- Optimizes for Azure Databricks performance
- Integrates with retry queue for failed records
- Handles data skew automatically

Author: GenZ Agent Project
Date: 2025-11-24
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, lit, current_timestamp, monotonically_increasing_id,
    hash, abs as spark_abs, md5, concat_ws, row_number
)
from pyspark.sql.window import Window
from delta.tables import DeltaTable
from typing import Dict, Optional, Callable
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MigrationConfig:
    """Configuration for LIMS migration pipeline."""

    def __init__(
        self,
        source_path: str,
        target_path: str,
        checkpoint_path: str = "/tmp/migration_checkpoint",
        failed_records_path: str = "/tmp/migration_failed",
        batch_size: int = 10000,
        num_partitions: int = 200,
        enable_adaptive_execution: bool = True,
        enable_z_ordering: bool = True,
        z_order_columns: list = None
    ):
        self.source_path = source_path
        self.target_path = target_path
        self.checkpoint_path = checkpoint_path
        self.failed_records_path = failed_records_path
        self.batch_size = batch_size
        self.num_partitions = num_partitions
        self.enable_adaptive_execution = enable_adaptive_execution
        self.enable_z_ordering = enable_z_ordering
        self.z_order_columns = z_order_columns or ["test_code", "collection_date"]


class OptimizedLIMSMigration:
    """
    High-performance LIMS data migration pipeline.

    Features:
    - Incremental processing with checkpointing
    - Automatic parallelization and optimization
    - Data skew handling
    - Failed record tracking
    - Progress monitoring
    - Integration with retry queue
    """

    def __init__(self, spark: SparkSession, config: MigrationConfig):
        self.spark = spark
        self.config = config
        self._configure_spark()
        self._initialize_tables()

    def _configure_spark(self):
        """Configure Spark for optimal migration performance."""
        logger.info("⚙️  Configuring Spark for migration...")

        # Adaptive Query Execution (AQE) - helps with skew
        if self.config.enable_adaptive_execution:
            self.spark.conf.set("spark.sql.adaptive.enabled", "true")
            self.spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
            self.spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
            logger.info("✅ Adaptive Query Execution (AQE) enabled")

        # Shuffle partitions
        self.spark.conf.set("spark.sql.shuffle.partitions", str(self.config.num_partitions))
        logger.info(f"✅ Shuffle partitions set to {self.config.num_partitions}")

        # Delta optimizations
        self.spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", "true")
        self.spark.conf.set("spark.databricks.delta.autoCompact.enabled", "true")
        logger.info("✅ Delta Lake optimizations enabled")

        # Broadcast joins for small dimension tables
        self.spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "100MB")

    def _initialize_tables(self):
        """Initialize target and checkpoint tables."""
        logger.info("📦 Initializing tables...")

        # Create target table if doesn't exist
        try:
            DeltaTable.forPath(self.spark, self.config.target_path)
            logger.info(f"✅ Target table exists: {self.config.target_path}")
        except Exception:
            logger.info(f"📝 Creating target table: {self.config.target_path}")
            # Table will be created on first write

        # Create checkpoint table
        checkpoint_schema = "batch_id STRING, batch_start STRING, batch_end STRING, " \
                          "records_processed LONG, status STRING, timestamp TIMESTAMP"

        self.spark.sql(f"""
            CREATE TABLE IF NOT EXISTS delta.`{self.config.checkpoint_path}` (
                {checkpoint_schema}
            ) USING DELTA
        """)
        logger.info(f"✅ Checkpoint table ready: {self.config.checkpoint_path}")

    def migrate_full_dataset(
        self,
        transformation_func: Optional[Callable[[DataFrame], DataFrame]] = None,
        validation_func: Optional[Callable[[DataFrame], DataFrame]] = None
    ):
        """
        Migrate entire LIMS dataset with optimizations.

        Args:
            transformation_func: Optional function to transform data during migration
            validation_func: Optional function to validate records before migration

        Returns:
            Migration statistics
        """
        logger.info("=" * 70)
        logger.info("🚀 STARTING OPTIMIZED LIMS MIGRATION")
        logger.info("=" * 70)

        start_time = datetime.now()

        # Step 1: Read source data
        logger.info(f"\n📖 Reading source data from: {self.config.source_path}")
        source_df = self._read_source_optimized()

        total_records = source_df.count()
        logger.info(f"📊 Total records to migrate: {total_records:,}")

        # Step 2: Handle data skew
        logger.info("\n⚖️  Checking for data skew...")
        source_df = self._handle_data_skew(source_df)

        # Step 3: Apply transformations
        if transformation_func:
            logger.info("\n🔄 Applying transformations...")
            source_df = transformation_func(source_df)

        # Step 4: Validate records
        if validation_func:
            logger.info("\n✅ Validating records...")
            source_df, failed_df = self._validate_records(source_df, validation_func)
            if failed_df.count() > 0:
                self._save_failed_records(failed_df, "validation_failed")

        # Step 5: Migrate in batches
        logger.info("\n📦 Starting batch migration...")
        stats = self._migrate_in_batches(source_df)

        # Step 6: Optimize target table
        logger.info("\n🔧 Optimizing target table...")
        self._optimize_target_table()

        # Step 7: Generate report
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60

        stats["total_records"] = total_records
        stats["duration_minutes"] = duration
        stats["throughput_per_minute"] = total_records / duration if duration > 0 else 0

        logger.info("\n" + "=" * 70)
        logger.info("✅ MIGRATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"📊 Total records: {total_records:,}")
        logger.info(f"⏱️  Duration: {duration:.2f} minutes")
        logger.info(f"📈 Throughput: {stats['throughput_per_minute']:.0f} records/minute")
        logger.info("=" * 70)

        return stats

    def migrate_incremental(
        self,
        watermark_column: str = "modified_date",
        last_watermark: Optional[str] = None
    ):
        """
        Migrate only new/updated records since last run.

        Args:
            watermark_column: Column to track incremental changes
            last_watermark: Last processed watermark value (if None, auto-detect)

        Returns:
            Migration statistics
        """
        logger.info("🔄 STARTING INCREMENTAL MIGRATION")

        # Get last watermark if not provided
        if not last_watermark:
            last_watermark = self._get_last_watermark(watermark_column)

        logger.info(f"📅 Last watermark: {last_watermark}")

        # Read only new records
        source_df = self._read_source_optimized()

        if last_watermark:
            incremental_df = source_df.filter(col(watermark_column) > lit(last_watermark))
        else:
            incremental_df = source_df

        new_records = incremental_df.count()
        logger.info(f"📊 New/updated records: {new_records:,}")

        if new_records == 0:
            logger.info("✅ No new records to migrate")
            return {"new_records": 0}

        # Migrate new records
        stats = self._migrate_in_batches(incremental_df)

        # Update watermark
        max_watermark = incremental_df.agg({watermark_column: "max"}).first()[0]
        self._save_watermark(watermark_column, max_watermark)

        logger.info(f"✅ Incremental migration complete: {new_records:,} records")
        return stats

    def _read_source_optimized(self) -> DataFrame:
        """
        Read source data with optimizations.

        Supports multiple source types:
        - Delta Lake
        - JDBC (SQL Server, Oracle, etc.)
        - Parquet
        - CSV
        """
        logger.info("📖 Reading source data...")

        # Detect source type from path
        if self.config.source_path.startswith("jdbc:"):
            # JDBC source (e.g., SQL Server LIMS database)
            return self._read_jdbc_optimized()
        elif self.config.source_path.endswith(".parquet"):
            return self.spark.read.parquet(self.config.source_path)
        elif self.config.source_path.endswith(".csv"):
            return self.spark.read.option("header", "true").csv(self.config.source_path)
        else:
            # Assume Delta Lake
            return self.spark.read.format("delta").load(self.config.source_path)

    def _read_jdbc_optimized(self) -> DataFrame:
        """
        Read from JDBC source with optimizations for large tables.

        This is critical for LIMS migrations from SQL Server.
        """
        logger.info("🔌 Reading from JDBC source...")

        # Parse JDBC connection details
        # Format: jdbc:sqlserver://server:port;database=db;user=user;password=pass;table=table
        jdbc_parts = self.config.source_path.split(";")
        jdbc_url = jdbc_parts[0]
        table_name = next(p.split("=")[1] for p in jdbc_parts if p.startswith("table="))

        # Use partitioning for parallel reads
        # This is KEY for performance with large LIMS tables
        df = self.spark.read \
            .format("jdbc") \
            .option("url", jdbc_url) \
            .option("dbtable", table_name) \
            .option("numPartitions", str(self.config.num_partitions)) \
            .option("partitionColumn", "sample_id") \
            .option("lowerBound", "1") \
            .option("upperBound", "10000000") \
            .option("fetchsize", "10000") \
            .load()

        logger.info(f"✅ JDBC read configured with {self.config.num_partitions} partitions")
        return df

    def _handle_data_skew(self, df: DataFrame) -> DataFrame:
        """
        Handle data skew by salting skewed keys.

        Data skew is a common cause of slow migrations.
        """
        # Check for skew in common columns
        skew_columns = ["test_code", "facility_id", "department_id"]

        for col_name in skew_columns:
            if col_name not in df.columns:
                continue

            # Check distribution
            dist = df.groupBy(col_name).count().orderBy(col("count").desc()).limit(1)
            max_count = dist.first()["count"] if dist.count() > 0 else 0
            total_count = df.count()
            skew_percentage = (max_count / total_count * 100) if total_count > 0 else 0

            if skew_percentage > 30:
                logger.warning(f"⚠️  Skew detected in {col_name}: {skew_percentage:.1f}%")
                logger.info(f"🧂 Applying salt to {col_name}...")

                # Add salt column for better distribution
                df = df.withColumn(
                    f"{col_name}_salted",
                    concat_ws("_", col(col_name), (spark_abs(hash(col("sample_id"))) % 10))
                )

        return df

    def _validate_records(
        self,
        df: DataFrame,
        validation_func: Callable[[DataFrame], DataFrame]
    ) -> tuple:
        """
        Validate records and separate failed records.

        Returns:
            (valid_df, failed_df)
        """
        # Add validation flag
        validated_df = validation_func(df)

        # Assume validation_func adds a 'is_valid' column
        valid_df = validated_df.filter(col("is_valid") == True).drop("is_valid")
        failed_df = validated_df.filter(col("is_valid") == False).drop("is_valid")

        valid_count = valid_df.count()
        failed_count = failed_df.count()

        logger.info(f"✅ Valid records: {valid_count:,}")
        logger.info(f"❌ Failed validation: {failed_count:,}")

        return valid_df, failed_df

    def _migrate_in_batches(self, df: DataFrame) -> Dict:
        """
        Migrate data in batches with checkpointing.

        This prevents "all-or-nothing" failures in large migrations.
        """
        total_records = df.count()
        num_batches = (total_records // self.config.batch_size) + 1

        logger.info(f"📦 Migrating in {num_batches} batches of {self.config.batch_size:,} records")

        # Add batch ID column
        df_with_batch = df.withColumn(
            "migration_batch_id",
            (row_number().over(Window.orderBy(monotonically_increasing_id())) / self.config.batch_size).cast("long")
        )

        successful_batches = 0
        failed_batches = 0
        total_migrated = 0

        for batch_id in range(num_batches):
            try:
                logger.info(f"\n📦 Processing batch {batch_id + 1}/{num_batches}...")

                # Get batch data
                batch_df = df_with_batch.filter(col("migration_batch_id") == batch_id) \
                    .drop("migration_batch_id")

                batch_count = batch_df.count()

                if batch_count == 0:
                    continue

                # Write batch to target
                batch_df.write \
                    .format("delta") \
                    .mode("append") \
                    .save(self.config.target_path)

                total_migrated += batch_count
                successful_batches += 1

                # Record checkpoint
                self._record_checkpoint(
                    batch_id=str(batch_id),
                    records_processed=batch_count,
                    status="success"
                )

                logger.info(f"✅ Batch {batch_id + 1} complete: {batch_count:,} records")
                logger.info(f"📊 Progress: {total_migrated:,}/{total_records:,} "
                          f"({total_migrated/total_records*100:.1f}%)")

            except Exception as e:
                logger.error(f"❌ Batch {batch_id + 1} failed: {e}")
                failed_batches += 1

                # Record failed checkpoint
                self._record_checkpoint(
                    batch_id=str(batch_id),
                    records_processed=0,
                    status=f"failed: {str(e)}"
                )

                # Save failed batch for retry
                try:
                    batch_df.write \
                        .format("delta") \
                        .mode("append") \
                        .save(f"{self.config.failed_records_path}/batch_{batch_id}")
                    logger.info(f"💾 Failed batch saved for retry")
                except Exception as save_error:
                    logger.error(f"Failed to save failed batch: {save_error}")

        return {
            "successful_batches": successful_batches,
            "failed_batches": failed_batches,
            "total_migrated": total_migrated
        }

    def _record_checkpoint(self, batch_id: str, records_processed: int, status: str):
        """Record migration checkpoint."""
        checkpoint_data = [(
            batch_id,
            datetime.now().isoformat(),
            datetime.now().isoformat(),
            records_processed,
            status,
            datetime.now()
        )]

        checkpoint_df = self.spark.createDataFrame(
            checkpoint_data,
            ["batch_id", "batch_start", "batch_end", "records_processed", "status", "timestamp"]
        )

        checkpoint_df.write \
            .format("delta") \
            .mode("append") \
            .save(self.config.checkpoint_path)

    def _optimize_target_table(self):
        """
        Optimize target Delta table.

        Runs:
        - OPTIMIZE (compaction)
        - Z-ORDER (for fast queries)
        - VACUUM (cleanup old files)
        """
        logger.info("🔧 Running Delta optimizations...")

        target_table = DeltaTable.forPath(self.spark, self.config.target_path)

        # 1. OPTIMIZE (compaction)
        logger.info("  ⚡ Running OPTIMIZE...")
        target_table.optimize().executeCompaction()

        # 2. Z-ORDER (if enabled)
        if self.config.enable_z_ordering and self.config.z_order_columns:
            logger.info(f"  📊 Running Z-ORDER on {self.config.z_order_columns}...")
            z_order_cols = ", ".join(self.config.z_order_columns)
            self.spark.sql(f"""
                OPTIMIZE delta.`{self.config.target_path}`
                ZORDER BY ({z_order_cols})
            """)

        # 3. VACUUM (cleanup - be careful with retention period)
        logger.info("  🧹 Running VACUUM (7 day retention)...")
        target_table.vacuum(168)  # 7 days in hours

        logger.info("✅ Optimizations complete")

    def _save_failed_records(self, failed_df: DataFrame, reason: str):
        """Save failed records for manual review or retry."""
        failed_path = f"{self.config.failed_records_path}/{reason}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        failed_df.withColumn("failure_reason", lit(reason)) \
            .withColumn("failed_at", current_timestamp()) \
            .write \
            .format("delta") \
            .mode("append") \
            .save(failed_path)

        logger.info(f"💾 Failed records saved to: {failed_path}")

    def _get_last_watermark(self, watermark_column: str) -> Optional[str]:
        """Get last processed watermark from target table."""
        try:
            target_df = self.spark.read.format("delta").load(self.config.target_path)
            max_value = target_df.agg({watermark_column: "max"}).first()[0]
            return str(max_value) if max_value else None
        except Exception:
            return None

    def _save_watermark(self, column: str, value):
        """Save watermark for next incremental run."""
        watermark_path = f"{self.config.checkpoint_path}_watermark"

        watermark_data = [(column, str(value), datetime.now())]
        watermark_df = self.spark.createDataFrame(
            watermark_data,
            ["column_name", "watermark_value", "timestamp"]
        )

        watermark_df.write \
            .format("delta") \
            .mode("overwrite") \
            .save(watermark_path)

    def resume_failed_batches(self):
        """
        Resume migration from failed batches.

        Reads checkpoint table to find failed batches and retries them.
        """
        logger.info("🔄 Checking for failed batches to resume...")

        checkpoint_df = self.spark.read.format("delta").load(self.config.checkpoint_path)
        failed_batches = checkpoint_df.filter(col("status").like("failed%"))

        failed_count = failed_batches.count()

        if failed_count == 0:
            logger.info("✅ No failed batches to resume")
            return

        logger.info(f"⚠️  Found {failed_count} failed batches")
        failed_batches.select("batch_id", "status", "timestamp").show()

        for row in failed_batches.collect():
            batch_id = row["batch_id"]
            failed_batch_path = f"{self.config.failed_records_path}/batch_{batch_id}"

            try:
                logger.info(f"🔄 Retrying batch {batch_id}...")

                # Read failed batch
                batch_df = self.spark.read.format("delta").load(failed_batch_path)

                # Retry migration
                batch_df.write \
                    .format("delta") \
                    .mode("append") \
                    .save(self.config.target_path)

                logger.info(f"✅ Batch {batch_id} successfully retried")

                # Update checkpoint
                self._record_checkpoint(
                    batch_id=f"{batch_id}_retry",
                    records_processed=batch_df.count(),
                    status="success_retry"
                )

            except Exception as e:
                logger.error(f"❌ Retry failed for batch {batch_id}: {e}")


# Example usage
if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("OptimizedLIMSMigration") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()

    # Configure migration
    config = MigrationConfig(
        source_path="jdbc:sqlserver://lims-server:1433;database=LIMS;table=lab_results",
        target_path="/mnt/delta/lims_migrated",
        checkpoint_path="/mnt/delta/lims_checkpoint",
        batch_size=50000,
        num_partitions=400
    )

    # Run migration
    migration = OptimizedLIMSMigration(spark, config)

    # Option 1: Full migration
    stats = migration.migrate_full_dataset()

    # Option 2: Incremental migration
    # stats = migration.migrate_incremental(watermark_column="modified_date")

    # Option 3: Resume failed batches
    # migration.resume_failed_batches()

    print("Migration complete:", stats)
