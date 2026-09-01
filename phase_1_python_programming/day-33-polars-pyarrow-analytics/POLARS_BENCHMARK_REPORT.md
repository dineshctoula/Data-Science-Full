# 🚀 Day 33 Performance Report: Polars & PyArrow Analytics Engine

## 📌 Executive Summary
This report details performance benchmarks between **Pandas (Eager)**, **Polars (Eager & Lazy Mode)**, and **PyArrow Zero-Copy In-Memory Bridge** on a synthetic dataset of **2,000,000 financial transactions** (52.07 MB Parquet).

---

## 📊 Benchmark Results

### 1. Eager API Execution Comparison
| Benchmark Task | Pandas Execution (s) | Polars Execution (s) | Speedup Factor |
| :--- | :---: | :---: | :---: |
| **Parquet Load Time** | `0.3247s` | `0.1296s` | **`2.51x`** |
| **Filtered Multi-Column GroupBy** | `0.2032s` | `0.0575s` | **`3.53x`** |
| **Vectorized Multi-Column Expressions** | `0.0851s` | `0.0292s` | **`2.91x`** |

---

### 2. Lazy Engine & Query Plan Optimization
Polars Lazy API (`pl.scan_parquet`) optimizes execution via **Predicate Pushdown** (filtering at file-scan level) and **Projection Pushdown** (loading only requested columns).

- **Pandas Full Scan & Aggregation**: `0.2755 seconds`
- **Polars Optimized Lazy Query**: `0.0445 seconds`
- **Lazy Engine Speedup**: **`6.19x faster than Pandas`**

---

### 3. Window Expressions (`.over()`) & Streaming Engine
Polars provides high-speed window functions without expensive explicit `groupby().transform()` calls.

- **Pandas Windowing (`transform` / `cumsum`)**: `2.6006 seconds`
- **Polars Windowing (`.over()`)**: `0.8251 seconds`
- **Window Speedup**: **`3.15x`**
- **Polars Streaming Engine (`collect(streaming=True)`)**: `0.0864 seconds`

---

### 4. Apache Arrow Zero-Copy Interoperability
- **PyArrow Read Table**: `0.2823s`
- **PyArrow -> Polars**: `0.339996s` *(Zero-Copy Memory Pointer Sharing)*
- **Polars -> PyArrow**: `0.095723s` *(Zero-Copy Memory Pointer Sharing)*
- **Polars -> Pandas**: `0.1203s`

---

## 💡 Key Takeaways & Architecture Insights
1. **Rust Engine & SIMD Vectorization**: Polars compiles expressions directly into optimized Rust machine code with SIMD parallelization.
2. **Apache Arrow Layout**: Polars uses Apache Arrow contiguous column-oriented memory, eliminating copying costs.
3. **Query Optimization**: `pl.scan_parquet` inspects the query tree before execution, minimizing disk I/O and CPU memory allocation.
