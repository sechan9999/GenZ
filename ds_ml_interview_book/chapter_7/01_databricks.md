# Chapter 7: Databricks & Spark

## PySpark Fundamentals

### Q1: Explain Spark's execution model (transformations vs actions)

**A:**

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count

spark = SparkSession.builder \
    .appName("Interview Demo") \
    .getOrCreate()

# Transformations (LAZY - build execution plan)
df = spark.read.parquet("s3://bucket/data.parquet")  # Transformation
df_filtered = df.filter(col("age") > 25)             # Transformation
df_grouped = df_filtered.groupBy("country").agg(avg("salary"))  # Transformation

# Nothing has executed yet! Spark built a DAG (Directed Acyclic Graph)

# Actions (EAGER - trigger execution)
result = df_grouped.collect()  # NOW Spark executes the entire plan
df_grouped.show()              # Action
count = df_filtered.count()    # Action
df_grouped.write.parquet("output/")  # Action
```

**Why lazy evaluation matters:**

| Transformation | Action |
|---|---|
| Returns DataFrame/RDD | Returns value/writes data |
| Lazy (builds plan) | Eager (executes plan) |
| Examples: `filter`, `select`, `groupBy` | Examples: `collect`, `count`, `show`, `write` |

**Key insight:** "Spark optimizes the entire pipeline before execution. If you filter before a join, Spark pushes the filter down to read less data — but only because it sees the whole plan before executing."

---

### Q2: How does partitioning work in Spark? When should you repartition?

**A:**

```python
# Check current partitions
print(f"Current partitions: {df.rdd.getNumPartitions()}")

# Repartition (shuffle, expensive)
df_repartitioned = df.repartition(100)  # Hash partition across 100 partitions

# Repartition by column (for join optimization)
df_repartitioned = df.repartition(100, "user_id")  # All same user_id in same partition

# Coalesce (reduce partitions without shuffle, cheap)
df_coalesced = df.coalesce(10)  # Only use for reducing partitions

# Write with partitioning (directory structure)
df.write.partitionBy("year", "month").parquet("output/")
# Creates: output/year=2024/month=01/part-00000.parquet
```

**When to repartition:**

```python
# Scenario 1: After filter (reduces data size)
df_large = spark.read.parquet("large.parquet")  # 1000 partitions
df_small = df_large.filter(col("country") == "US")  # Now only 5% of data
df_small = df_small.coalesce(50)  # Reduce partitions to match data size

# Scenario 2: Before expensive operation (join, groupBy)
df1 = spark.read.parquet("users.parquet")  # 100 partitions
df2 = spark.read.parquet("events.parquet")  # 10 partitions

# Repartition both to same count for balanced join
df1 = df1.repartition(200, "user_id")
df2 = df2.repartition(200, "user_id")

result = df1.join(df2, "user_id")  # Efficient join, no shuffle needed

# Scenario 3: Before write (control output file size)
df.repartition(100).write.parquet("output/")  # Creates 100 files
```

**Rule of thumb:**
- Partition size: 128MB - 1GB per partition
- Number of partitions: 2-4x number of cores
- Use `coalesce` to reduce, `repartition` to increase or redistribute

---

### Q3: Explain broadcast joins vs shuffle joins. When to use each?

**A:**

```python
from pyspark.sql.functions import broadcast

# Large table (1TB)
large_df = spark.read.parquet("large_events.parquet")  # 1 billion rows

# Small table (1MB)
small_df = spark.read.parquet("small_lookup.parquet")  # 1000 rows

## Shuffle join (DEFAULT - expensive for large tables)
# Both tables shuffled across network
result = large_df.join(small_df, "id")

## Broadcast join (OPTIMIZED - copy small table to all workers)
# Small table sent to every executor once, no shuffle
result = large_df.join(broadcast(small_df), "id")

# Spark auto-broadcasts if table < 10MB (configurable)
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 10 * 1024 * 1024)  # 10MB
```

**Broadcast vs Shuffle:**

| Join Type | Network I/O | When to use |
|---|---|---|
| Shuffle | O(n+m) both sides | Both tables large |
| Broadcast | O(m) small side | One table <1GB |

**Performance difference:**
```python
# Shuffle join: 10min (shuffles 1TB across network)
large.join(small, "id").count()

# Broadcast join: 2min (sends 1MB to all workers)
large.join(broadcast(small), "id").count()

# 5x speedup!
```

---

## Databricks ML Features

### Q4: How do you use Databricks Feature Store for ML pipelines?

**A:**

```python
from databricks import feature_store
from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

# Step 1: Create feature table
def compute_user_features(df):
    """Compute features from raw data"""
    return df.groupBy("user_id").agg(
        avg("transaction_amount").alias("avg_transaction"),
        count("*").alias("transaction_count"),
        max("transaction_date").alias("last_transaction_date")
    )

user_features_df = compute_user_features(raw_transactions_df)

# Step 2: Register feature table
fs.create_table(
    name="ml.user_features",
    primary_keys=["user_id"],
    df=user_features_df,
    description="User transaction features for churn prediction"
)

# Step 3: Training - automatically joins features
from databricks.feature_store import FeatureLookup

feature_lookups = [
    FeatureLookup(
        table_name="ml.user_features",
        lookup_key=["user_id"]
    )
]

training_set = fs.create_training_set(
    df=labels_df,  # Just user_id + label
    feature_lookups=feature_lookups,
    label="churn"
)

training_df = training_set.load_df()

# Step 4: Train model with feature metadata
from sklearn.ensemble import RandomForestClassifier

X_train = training_df.drop("churn")
y_train = training_df["churn"]

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 5: Log model WITH feature dependencies
fs.log_model(
    model=model,
    artifact_path="model",
    flavor=mlflow.sklearn,
    training_set=training_set
)

# Step 6: Inference - automatically fetches latest features
# No need to manually join!
predictions = fs.score_batch(
    model_uri=f"models:/churn_model/production",
    df=new_users_df  # Just user_id
)
```

**Benefits:**
- ✅ Feature reuse across models
- ✅ Point-in-time correctness (no data leakage)
- ✅ Auto-join at inference time
- ✅ Feature lineage tracking
- ✅ Monitoring & drift detection

---

### Q5: Explain Delta Lake's time travel and how it enables ML reproducibility

**A:**

```python
from delta.tables import DeltaTable

# Write data with Delta Lake
df.write.format("delta").save("/mnt/delta/ml_features")

# Append more data
new_df.write.format("delta").mode("append").save("/mnt/delta/ml_features")

## Time travel (access historical versions)

# Read latest version
df_latest = spark.read.format("delta").load("/mnt/delta/ml_features")

# Read version from 7 days ago
df_week_ago = spark.read.format("delta") \
    .option("timestampAsOf", "2024-03-25") \
    .load("/mnt/delta/ml_features")

# Read specific version number
df_v10 = spark.read.format("delta") \
    .option("versionAsOf", 10) \
    .load("/mnt/delta/ml_features")

## ML Reproducibility use case

# Training: Record the data version
training_version = 42
training_timestamp = "2024-03-25 10:00:00"

# Train model
df_train = spark.read.format("delta") \
    .option("versionAsOf", training_version) \
    .load("/mnt/delta/ml_features")

model.fit(df_train)

# Log metadata
mlflow.log_param("data_version", training_version)
mlflow.log_param("data_timestamp", training_timestamp)

# Later: Reproduce exact training data
df_reproduce = spark.read.format("delta") \
    .option("versionAsOf", training_version) \
    .load("/mnt/delta/ml_features")

## Rollback bad writes
delta_table = DeltaTable.forPath(spark, "/mnt/delta/ml_features")
delta_table.restoreToVersion(41)  # Rollback to version before bad write
```

**Delta Lake benefits for ML:**

| Feature | ML benefit |
|---|---|
| Time travel | Reproduce exact training data |
| ACID transactions | No partial writes during feature computation |
| Schema evolution | Add features without breaking pipelines |
| Audit history | Track all data changes |
| Merge/Upsert | Efficiently update features |

---

### Q6: How do you optimize Databricks notebooks for large-scale ML?

**A:**

```python
## Optimization 1: Cache intermediate results
df_features = spark.read.parquet("features.parquet")
df_features = df_features.filter(col("date") > "2024-01-01")

# Cache in memory (use for repeated access)
df_features.cache()  # or .persist(StorageLevel.MEMORY_AND_DISK)

# Use multiple times
train_df = df_features.filter(col("split") == "train")
val_df = df_features.filter(col("split") == "val")

# Don't forget to unpersist when done
df_features.unpersist()

## Optimization 2: Adaptive Query Execution (AQE)
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")

## Optimization 3: Dynamic partition pruning
# Automatically prunes partitions based on join conditions
spark.conf.set("spark.sql.optimizer.dynamicPartitionPruning.enabled", "true")

## Optimization 4: Predicate pushdown
# Push filters to data source (read less data)
df = spark.read.format("delta") \
    .load("/mnt/data") \
    .filter(col("date") >= "2024-01-01")  # Pushed to Delta, only reads relevant files

## Optimization 5: Column pruning
# Only read needed columns
df = spark.read.parquet("data.parquet").select("user_id", "feature1", "feature2")
# NOT: df = spark.read.parquet("data.parquet") then select later

## Optimization 6: Use Arrow for Pandas conversions
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

# 10-100x faster conversion
pandas_df = spark_df.toPandas()  # Uses Arrow
```

**Common pitfalls:**
```python
# ❌ BAD: Collect large DataFrame to driver
large_df.collect()  # OOM!

# ✅ GOOD: Sample first
large_df.sample(0.01).collect()

# ❌ BAD: UDF without optimization
from pyspark.sql.functions import udf
@udf("double")
def slow_udf(x):
    return x * 2

# ✅ GOOD: Use vectorized Pandas UDF
from pyspark.sql.functions import pandas_udf
@pandas_udf("double")
def fast_udf(x: pd.Series) -> pd.Series:
    return x * 2  # Operates on batches

# ❌ BAD: Loop over rows
for row in df.collect():  # Horrible for big data
    process(row)

# ✅ GOOD: Use DataFrame operations
df.mapInPandas(process_batch, schema=output_schema)
```

---

## Summary: Databricks Checklist

| Concept | Key Insight | Interview Signal |
|---|---|---|
| Transformations vs Actions | Lazy evaluation enables optimization | "Spark builds DAG, optimizes before execution" |
| Partitioning | Control parallelism and file size | "2-4x cores, 128MB-1GB per partition" |
| Broadcast joins | Copy small tables to avoid shuffle | "Use for tables <1GB to save network I/O" |
| Feature Store | Centralized feature management | "Point-in-time joins prevent leakage" |
| Delta Lake time travel | Data versioning for reproducibility | "Record data version in MLflow for reproducibility" |
| Optimization | Cache, AQE, pushdown, Arrow | "Predicate pushdown + column pruning minimize reads" |
