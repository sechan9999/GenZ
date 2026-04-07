# Chapter 1: Python for Data Science

## Core Python Concepts

### Q1: Explain Python's memory management and garbage collection

**A:** Python uses reference counting and a cyclic garbage collector:

```python
import sys

# Reference counting
x = [1, 2, 3]
print(sys.getrefcount(x))  # 2 (one from x, one from getrefcount)

y = x  # increases refcount
print(sys.getrefcount(x))  # 3

del y  # decreases refcount
print(sys.getrefcount(x))  # 2

# Garbage collection for cycles
import gc

class Node:
    def __init__(self):
        self.ref = None

a = Node()
b = Node()
a.ref = b
b.ref = a  # circular reference

del a, b  # refcount won't reach 0, but GC will collect them

gc.collect()  # manually trigger GC
print(gc.get_count())  # (threshold0, threshold1, threshold2)
```

**Key points:**
- **Reference counting**: Object deallocated when refcount hits 0
- **Cyclic GC**: Detects and collects reference cycles
- **Generations**: 0 (young), 1 (middle), 2 (old) — Gen 0 collected most frequently
- **`__del__`**: Don't rely on it for cleanup; use context managers instead

**Follow-up: When would you disable GC?**
> In tight loops where you know there are no cycles, disabling GC can give 10-20% speedup:
```python
import gc
gc.disable()
# ... tight loop ...
gc.enable()
```

---

### Q2: What's the difference between `__str__`, `__repr__`, and `__format__`?

**A:**
```python
class DataPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        """User-friendly string (informal)"""
        return f"Point at ({self.x}, {self.y})"
    
    def __repr__(self):
        """Developer-friendly string (unambiguous, ideally executable)"""
        return f"DataPoint(x={self.x}, y={self.y})"
    
    def __format__(self, format_spec):
        """Custom formatting with f-strings"""
        if format_spec == 'polar':
            import math
            r = math.sqrt(self.x**2 + self.y**2)
            theta = math.atan2(self.y, self.x)
            return f"(r={r:.2f}, θ={theta:.2f})"
        return str(self)

p = DataPoint(3, 4)
print(str(p))           # Point at (3, 4)
print(repr(p))          # DataPoint(x=3, y=4)
print(f"{p}")           # Point at (3, 4) (uses __str__)
print(f"{p!r}")         # DataPoint(x=3, y=4) (forces __repr__)
print(f"{p:polar}")     # (r=5.00, θ=0.93)
```

**Rule of thumb:**
- `__repr__`: eval(repr(obj)) should reconstruct the object
- `__str__`: human-readable for end users
- `__format__`: when you need multiple output formats

---

### Q3: Explain Python's GIL and its implications for data science

**A:** The Global Interpreter Lock (GIL) allows only one thread to execute Python bytecode at a time.

```python
import threading
import time

counter = 0

def increment(n):
    global counter
    for _ in range(n):
        counter += 1

# Multi-threaded (GIL limits this)
threads = []
for _ in range(4):
    t = threading.Thread(target=increment, args=(1_000_000,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print(counter)  # May not be 4,000,000 due to race conditions!
```

**Implications for DS:**
- **CPU-bound tasks (model training)**: GIL kills performance → use `multiprocessing`
- **I/O-bound tasks (data loading)**: Threading is fine
- **NumPy/Pandas**: Release GIL during C operations → threading works well
- **Dask, Ray, multiprocessing**: Bypass GIL with separate processes

**Example: Proper parallelization**
```python
from multiprocessing import Pool
import numpy as np

def train_model(data_chunk):
    # Simulate training
    return np.mean(data_chunk)

if __name__ == '__main__':
    data = np.random.randn(1000, 100)
    chunks = np.array_split(data, 4)
    
    with Pool(4) as pool:
        results = pool.map(train_model, chunks)
    
    print(results)
```

---

## NumPy & Pandas

### Q4: Explain broadcasting in NumPy and give a real-world DS example

**A:**
```python
import numpy as np

# Broadcasting rules:
# 1. If arrays differ in rank, prepend 1s to the smaller-rank array
# 2. Two dimensions are compatible if they're equal or one is 1
# 3. Arrays can be broadcast if they're compatible in all dimensions

# Example 1: Mean centering
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])  # shape (3, 3)

mean = X.mean(axis=0)  # shape (3,) → broadcasts to (1, 3)
X_centered = X - mean  # (3, 3) - (1, 3) → (3, 3)

print(X_centered)
# [[-3. -3. -3.]
#  [ 0.  0.  0.]
#  [ 3.  3.  3.]]

# Example 2: Standardization (real DS task)
X = np.random.randn(1000, 5)  # 1000 samples, 5 features

mean = X.mean(axis=0, keepdims=True)  # shape (1, 5)
std = X.std(axis=0, keepdims=True)    # shape (1, 5)

X_standardized = (X - mean) / std  # broadcasts correctly

# Example 3: Pairwise distance (vectorized)
points = np.array([[1, 2], [3, 4], [5, 6]])  # shape (3, 2)

# Expand dims for broadcasting
p1 = points[:, np.newaxis, :]  # shape (3, 1, 2)
p2 = points[np.newaxis, :, :]  # shape (1, 3, 2)

distances = np.sqrt(((p1 - p2) ** 2).sum(axis=2))  # shape (3, 3)
print(distances)
```

**Why it matters:**
- **10-100x faster** than loops
- **Memory efficient**: No copies made until actual operation
- **Readable**: Expresses matrix operations naturally

---

### Q5: When should you use `.loc` vs `.iloc` vs `.at` vs `.iat` in Pandas?

**A:**
```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
}, index=['row1', 'row2', 'row3'])

# .loc — label-based (rows by index name, columns by name)
print(df.loc['row1', 'A'])          # 1
print(df.loc['row1':'row2', 'A':'B'])  # slicing INCLUSIVE

# .iloc — integer position-based
print(df.iloc[0, 0])                # 1
print(df.iloc[0:2, 0:2])            # slicing EXCLUSIVE (like normal Python)

# .at — fast scalar access by label (single value only)
print(df.at['row1', 'A'])           # 1 (fastest)

# .iat — fast scalar access by position
print(df.iat[0, 0])                 # 1 (fastest)

# Performance comparison
import timeit

setup = '''
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randn(1000, 100))
'''

print(timeit.timeit('df.loc[0, 0]', setup=setup, number=10000))   # ~0.05s
print(timeit.timeit('df.iloc[0, 0]', setup=setup, number=10000))  # ~0.05s
print(timeit.timeit('df.at[0, 0]', setup=setup, number=10000))    # ~0.01s
print(timeit.timeit('df.iat[0, 0]', setup=setup, number=10000))   # ~0.01s
```

**Decision tree:**
```
Need to access...
├─ Single scalar?
│  ├─ By label? → .at
│  └─ By position? → .iat
└─ Multiple values / slice?
   ├─ By label? → .loc
   └─ By position? → .iloc
```

---

### Q6: How do you handle missing data in Pandas? Explain the trade-offs.

**A:**
```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'age': [25, np.nan, 35, 40, np.nan],
    'income': [50000, 60000, np.nan, 80000, 55000],
    'score': [85, 90, 88, np.nan, 92]
})

# Strategy 1: Drop rows with ANY missing values
df_dropped = df.dropna()  # Only 2 rows remain
# Pro: Simple, no bias from imputation
# Con: Loss of data, may introduce bias if missingness is not random

# Strategy 2: Drop columns with too many missing values
df_drop_cols = df.dropna(axis=1, thresh=4)  # keep cols with ≥4 non-null
# Pro: Removes unreliable features
# Con: May lose predictive features

# Strategy 3: Mean/median imputation
df_mean = df.fillna(df.mean(numeric_only=True))
df_median = df.fillna(df.median(numeric_only=True))
# Pro: Preserves sample size
# Con: Distorts distribution, underestimates variance

# Strategy 4: Forward/backward fill (time series)
df_sorted = df.sort_index()
df_ffill = df_sorted.fillna(method='ffill')  # forward fill
df_bfill = df_sorted.fillna(method='bfill')  # backward fill
# Pro: Preserves temporal patterns
# Con: Only valid for time-ordered data

# Strategy 5: Interpolation
df_interpolated = df.interpolate(method='linear')
# Pro: Smooth transitions
# Con: Assumes linear relationship

# Strategy 6: Model-based imputation (best for ML)
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=2)
df_knn = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns,
    index=df.index
)
# Pro: Captures feature relationships
# Con: Computationally expensive

# Strategy 7: Indicator variable for missingness
df_with_indicator = df.copy()
df_with_indicator['age_missing'] = df['age'].isna().astype(int)
df_with_indicator['age'] = df_with_indicator['age'].fillna(df['age'].median())
# Pro: Model can learn if missingness is informative (MAR/MNAR)
# Con: Doubles feature count
```

**Production best practice:**
```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Define which columns get which treatment
numeric_features = ['age', 'income']
categorical_features = ['category']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
```

---

## Python Best Practices for DS

### Q7: Explain Python's context managers and why they're important for data science

**A:**
```python
# Without context manager (bad)
file = open('data.csv')
data = file.read()
file.close()  # Might not run if exception occurs!

# With context manager (good)
with open('data.csv') as file:
    data = file.read()
# Automatically closes, even if exception occurs

# Custom context manager for timing
import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    start = time.time()
    print(f"Starting {name}...")
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"{name} took {elapsed:.2f}s")

# Usage in ML pipeline
with timer("Data loading"):
    df = pd.read_csv('large_dataset.csv')

with timer("Feature engineering"):
    df['new_feature'] = df['a'] * df['b']

with timer("Model training"):
    model.fit(X_train, y_train)

# Real-world DS example: Database connection
from contextlib import contextmanager
import psycopg2

@contextmanager
def get_db_connection(config):
    conn = psycopg2.connect(**config)
    try:
        yield conn
        conn.commit()
    except:
        conn.rollback()
        raise
    finally:
        conn.close()

# Usage
with get_db_connection(db_config) as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE active = true")
    data = cursor.fetchall()
# Connection automatically closed and committed/rolled back
```

---

### Q8: What are Python generators and when should you use them in data science?

**A:**
```python
# Without generator (loads all in memory — BAD for big data)
def load_all_data(file_path):
    data = []
    with open(file_path) as f:
        for line in f:
            data.append(process(line))
    return data  # Entire list in memory

# With generator (lazy evaluation — GOOD)
def load_data_generator(file_path):
    with open(file_path) as f:
        for line in f:
            yield process(line)  # One item at a time

# Real-world DS examples

# Example 1: Mini-batch generator for training
def batch_generator(X, y, batch_size=32):
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]

# Usage
for X_batch, y_batch in batch_generator(X_train, y_train, batch_size=64):
    model.partial_fit(X_batch, y_batch)

# Example 2: Streaming data processing
def process_large_csv(file_path, chunk_size=1000):
    """Process CSV in chunks without loading all into memory"""
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Process chunk
        chunk_processed = transform(chunk)
        yield chunk_processed

# Aggregate results
total = 0
for chunk in process_large_csv('huge_file.csv'):
    total += chunk['value'].sum()

# Example 3: Infinite data stream (online learning)
def data_stream():
    """Simulate infinite data stream"""
    while True:
        # Fetch new data from API, queue, etc.
        data = fetch_new_data()
        if data:
            yield data
        time.sleep(1)

# Online learning
for batch in data_stream():
    model.partial_fit(batch, batch['target'])
    if should_stop():
        break
```

**Memory comparison:**
```python
import sys

# List comprehension (eager)
numbers_list = [x**2 for x in range(1000000)]
print(sys.getsizeof(numbers_list))  # ~8 MB

# Generator expression (lazy)
numbers_gen = (x**2 for x in range(1000000))
print(sys.getsizeof(numbers_gen))   # ~128 bytes!
```

**When to use generators in DS:**
- Processing datasets larger than RAM
- Streaming data pipelines
- Mini-batch training
- Data augmentation (generating variations on-the-fly)
- ETL pipelines

---

## Common Pitfalls

### Q9: Explain mutable default arguments and how they can break ML pipelines

**A:**
```python
# WRONG (dangerous default)
def add_sample(sample, dataset=[]):
    dataset.append(sample)
    return dataset

# This breaks!
batch1 = add_sample({'id': 1})  # [{'id': 1}]
batch2 = add_sample({'id': 2})  # [{'id': 1}, {'id': 2}] — SURPRISE!
# Default list is shared across all calls!

# CORRECT
def add_sample(sample, dataset=None):
    if dataset is None:
        dataset = []
    dataset.append(sample)
    return dataset

# Real-world ML example that breaks
class DataProcessor:
    def __init__(self, transformations=[]):  # BUG!
        self.transformations = transformations
    
    def add_transformation(self, transform):
        self.transformations.append(transform)

# This breaks
processor1 = DataProcessor()
processor1.add_transformation(lambda x: x * 2)

processor2 = DataProcessor()  # Shares transformations with processor1!
print(len(processor2.transformations))  # 1 — WRONG!

# CORRECT
class DataProcessor:
    def __init__(self, transformations=None):
        self.transformations = transformations if transformations is not None else []
```

**Rule:** Never use mutable defaults. Use `None` and create a new instance inside the function.

---

### Q10: How do you profile Python code for ML workloads?

**A:**
```python
# Method 1: cProfile (built-in, comprehensive)
import cProfile
import pstats

def train_model():
    # ... training code ...
    pass

profiler = cProfile.Profile()
profiler.enable()
train_model()
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # top 10 functions

# Method 2: line_profiler (line-by-line)
# pip install line_profiler
# Add @profile decorator to functions
# Run: kernprof -l -v script.py

@profile
def feature_engineering(df):
    df['feature1'] = df['a'] * df['b']  # <-- will show time per line
    df['feature2'] = df['c'].apply(lambda x: x**2)
    return df

# Method 3: memory_profiler (memory usage)
# pip install memory_profiler
from memory_profiler import profile

@profile
def load_data():
    df = pd.read_csv('large.csv')  # <-- will show memory increase
    return df

# Method 4: Py-Spy (sampling profiler, no code changes)
# pip install py-spy
# py-spy record -o profile.svg -- python train.py

# Method 5: snakeviz (visualize cProfile output)
# pip install snakeviz
# python -m cProfile -o profile.stats train.py
# snakeviz profile.stats

# Quick timing for experiments
import time

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

@timeit
def preprocess_data(df):
    # ... processing ...
    return df
```

---

## Summary: Python Essentials Checklist

| Topic | Key Concept | Interview Signal |
|---|---|---|
| Memory | Reference counting + GC | "Objects deleted when refcount=0, GC handles cycles" |
| GIL | Limits threading for CPU | "Use multiprocessing for CPU-bound, threading for I/O" |
| Broadcasting | NumPy vectorization | "10-100x faster than loops, memory efficient" |
| Pandas indexing | .loc/.iloc/.at/.iat | ".at/.iat for scalars, .loc/.iloc for slices" |
| Missing data | Imputation strategies | "KNN/median imputation + missingness indicator" |
| Context managers | Resource management | "Always use `with` for files, connections, transactions" |
| Generators | Lazy evaluation | "Use for datasets larger than RAM, streaming" |
| Mutable defaults | Function arguments | "Never use `[]` or `{}` as default, use `None`" |
| Profiling | Performance optimization | "cProfile for time, memory_profiler for RAM" |
