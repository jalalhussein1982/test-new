# Data Ingestion & Preparation Pipeline

## Complete Technical Blueprint

**Version:** 1.0  
**Last Updated:** December 2025  
**Purpose:** End-to-end data preparation pipeline from raw ingestion to correlation-ready output

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Phase I: Ingestion & Schema Enforcement](#phase-i-ingestion--schema-enforcement)
3. [Phase I-A: Duplicate Detection & Resolution](#phase-i-a-duplicate-detection--resolution)
4. [Phase II: Scope Definition (Dimensionality Reduction)](#phase-ii-scope-definition-dimensionality-reduction)
5. [Phase III: The Sanitation Layer (Data Integrity)](#phase-iii-the-sanitation-layer-data-integrity)
6. [Phase IV: Distribution Analysis & Outlier Handling](#phase-iv-distribution-analysis--outlier-handling)
7. [Phase IV-A: Multicollinearity Pre-Screening](#phase-iv-a-multicollinearity-pre-screening)
8. [Phase V: The "Pre-Correlation" Bridge (Feature Engineering)](#phase-v-the-pre-correlation-bridge-feature-engineering)
9. [Distribution Inspector (Global Utility)](#distribution-inspector-global-utility)
10. [Data State Architecture](#data-state-architecture)
11. [Pipeline Configuration Object](#pipeline-configuration-object)
12. [Rollback Interface & State Management](#rollback-interface--state-management)
13. [Implementation Reference](#implementation-reference)
14. [GDPR Compliance Framework](#gdpr-compliance-framework)
15. [Deployment Guide (GitHub + Streamlit Cloud)](#deployment-guide-github--streamlit-cloud)
16. [Appendix: Quick Reference](#appendix-quick-reference)

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA PREPARATION PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │  Phase I │───▶│Phase I-A │───▶│ Phase II │───▶│Phase III │             │
│  │ Ingest & │    │Duplicate │    │  Scope   │    │ Sanitize │             │
│  │  Schema  │    │Detection │    │Definition│    │  & Clean │             │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘             │
│       │                                               │                    │
│       ▼                                               ▼                    │
│   df_schema                                       df_clean                 │
│                                                       │                    │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐        │                    │
│  │ Phase V  │◀───│Phase IV-A│◀───│ Phase IV │◀───────┘                    │
│  │ Feature  │    │Multicollin│   │ Outlier  │                             │
│  │   Eng.   │    │  Screen  │    │ Handling │                             │
│  └──────────┘    └──────────┘    └──────────┘                             │
│       │                                                                    │
│       ▼                                                                    │
│   df_final ──────▶ [CORRELATION ANALYSIS / EXPORT]                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase I: Ingestion & Schema Enforcement

**Objective:** Correctly interpret the raw binary data into a structured DataFrame with semantically correct data types.

### 1. Loader Interface

| Component | Specification |
|-----------|---------------|
| Accepted Formats | CSV, Excel (`.xlsx`), Parquet, JSON |
| System Action | Read file into memory, detect delimiter/encoding automatically |
| Error Handling | Surface parsing errors with line numbers; offer encoding override |

### 2. Schema Inference & Validation

**System Action:** Detect types (Int, Float, Object/String, Bool, DateTime)

**User Action:** Review and Cast

| Critical Check | Description |
|----------------|-------------|
| Categorical Integers | User must confirm that categorical integers (e.g., Zip Codes, IDs, Phone Numbers) are cast to Strings/Categories, not Integers |
| DateTime Parsing | Parse DateTime columns correctly; offer format string override if auto-detection fails |
| Boolean Detection | Confirm columns with values like `0/1`, `Y/N`, `True/False` are cast appropriately |

**UI Component:** Editable schema table

```
┌─────────────────────────────────────────────────────────────────┐
│  Column Name    │ Inferred Type │ Override Type ▼ │ Sample     │
├─────────────────┼───────────────┼─────────────────┼────────────┤
│  customer_id    │ int64         │ [category ▼]    │ 10042      │
│  zip_code       │ int64         │ [string ▼]      │ 90210      │
│  order_date     │ object        │ [datetime ▼]    │ 2024-01-15 │
│  amount         │ float64       │ [float64 ▼]     │ 149.99     │
│  is_active      │ int64         │ [boolean ▼]     │ 1          │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Initial Profiling

**Display:** Basic metadata dashboard

| Metric | Description |
|--------|-------------|
| Row Count | Total observations |
| Column Count | Total features |
| Memory Usage | DataFrame memory footprint |
| Sample Rows | `head()` preview (first 5-10 rows) |
| Type Distribution | Count of columns per data type |

**Output State:** `df_schema`

---

## Phase I-A: Duplicate Detection & Resolution

**Objective:** Eliminate redundant observations that inflate sample size and distort variance.

### 1. Exact Duplicate Scan

| Component | Specification |
|-----------|---------------|
| System Action | Identify rows where ALL column values are identical |
| Display | Count of exact duplicates, preview of duplicate groups |
| User Action | **Keep First**, **Keep Last**, or **Drop All Duplicates** |

### 2. Subset Duplicate Scan

| Component | Specification |
|-----------|---------------|
| System Action | Identify rows where KEY columns (user-defined) are identical |
| Use Case | Same transaction ID with different timestamps — likely data entry error |
| User Action | Select key columns, then resolve as above |

**UI Component:** Key column selector

```
┌─────────────────────────────────────────────────────────────────┐
│  Define Subset Keys for Duplicate Detection                     │
├─────────────────────────────────────────────────────────────────┤
│  ☑ transaction_id                                               │
│  ☑ customer_id                                                  │
│  ☐ order_date                                                   │
│  ☐ amount                                                       │
├─────────────────────────────────────────────────────────────────┤
│  Duplicates Found: 47 groups (183 rows)                         │
│  [Preview Duplicates]  [Keep First]  [Keep Last]  [Drop All]   │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Near-Duplicate Flagging (Optional)

| Component | Specification |
|-----------|---------------|
| System Action | Flag rows with high similarity scores |
| Algorithms | Levenshtein distance on string columns, threshold on numerical deviation |
| User Action | Manual review and resolution |
| Default State | Disabled (opt-in feature) |

**Output State:** `df_deduplicated`

---

## Phase II: Scope Definition (Dimensionality Reduction)

**Objective:** Remove irrelevant features to improve processing speed and noise reduction.

### 1. Column Selection

**User Action:** Select/Deselect columns to include in the analysis pipeline

**UI Component:** Column selector with metadata

```
┌─────────────────────────────────────────────────────────────────┐
│  Select Columns for Analysis                                    │
├─────────────────────────────────────────────────────────────────┤
│  ☑ customer_id      │ category │ 8,412 unique │                │
│  ☑ order_date       │ datetime │ 365 unique   │                │
│  ☑ amount           │ float64  │ 2,891 unique │                │
│  ☐ internal_notes   │ object   │ 9,847 unique │ ⚠ High Card.  │
│  ☐ created_by       │ object   │ 1 unique     │ ⚠ Zero Var.   │
│  ☑ region           │ category │ 5 unique     │                │
├─────────────────────────────────────────────────────────────────┤
│  Selected: 4 of 6 columns                                       │
│  [Select All]  [Deselect All]  [Apply Recommendations]          │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Automated Recommendations

**System Action:** Flag and suggest dropping:

| Flag Type | Detection Rule | Rationale |
|-----------|----------------|-----------|
| Zero Variance | Columns with only 1 unique value | No predictive/analytical value |
| High Cardinality/ID | `Unique Count == Row Count` (unless index) | Likely identifier, not feature |
| Near-Zero Variance | >95% of values are single value | Limited analytical value |

**Output State:** `df_scoped`

---

## Phase III: The Sanitation Layer (Data Integrity)

**Objective:** Ensure mathematical validity of the dataset.

### Step A: Logical Validity (Constraint Enforcement)

**Concept:** Invalid data is distinct from missing data.

**User Action:** Define domain constraints

| Constraint Type | Example | Syntax |
|-----------------|---------|--------|
| Range | Age must be positive | `Age > 0` |
| Bounded | Probability must be ≤ 1 | `Probability <= 1` |
| Conditional | End date after start date | `end_date >= start_date` |
| Categorical | Status in allowed values | `Status IN ('Active', 'Inactive')` |

**System Action:** Scan rows against user-defined constraints

**Display:** Table of violating rows with violated constraint indicated

```
┌─────────────────────────────────────────────────────────────────┐
│  Constraint Violations Found: 23 rows                           │
├─────────────────────────────────────────────────────────────────┤
│  Row  │ Column      │ Value │ Constraint      │ Action ▼       │
├───────┼─────────────┼───────┼─────────────────┼────────────────┤
│  142  │ age         │ -5    │ age > 0         │ [Convert NaN ▼]│
│  891  │ probability │ 1.3   │ probability <= 1│ [Drop Row ▼]   │
│  2041 │ end_date    │ NULL  │ end >= start    │ [Flag ▼]       │
├─────────────────────────────────────────────────────────────────┤
│  Bulk Actions: [Drop All] [Convert All to NaN] [Flag All]      │
└─────────────────────────────────────────────────────────────────┘
```

**Resolution Options:**

| Option | Behavior |
|--------|----------|
| **Drop Row** | Remove entirely from pipeline |
| **Convert to NaN** | Value becomes missing, enters imputation workflow |
| **Flag & Retain** | Mark row with `_constraint_violation` boolean column, keep original value |

### Step B: Missing Value Management (Imputation)

**Visual Aid:** Missingness Matrix/Heatmap

- Identify patterns (e.g., if Column A is null, is Column B also always null?)
- Detect MCAR (Missing Completely at Random) vs MAR (Missing at Random) vs MNAR (Missing Not at Random)

**System Action:** Calculate % missing per column

**Display:** Missing value summary

```
┌─────────────────────────────────────────────────────────────────┐
│  Missing Value Analysis                                         │
├─────────────────────────────────────────────────────────────────┤
│  Column        │ Missing │ % Missing │ Strategy ▼              │
├────────────────┼─────────┼───────────┼─────────────────────────┤
│  income        │ 342     │ 3.4%      │ [Median ▼]              │
│  region        │ 12      │ 0.1%      │ [Mode ▼]                │
│  satisfaction  │ 1,204   │ 12.0%     │ [Drop Rows ▼]           │
│  comments      │ 4,521   │ 45.2%     │ ["Unknown" ▼]           │
├─────────────────────────────────────────────────────────────────┤
│  [View Missingness Heatmap]  [Auto-Recommend Strategies]        │
└─────────────────────────────────────────────────────────────────┘
```

**Resolution Strategy (Per Column):**

| Strategy | Use Case | Implementation |
|----------|----------|----------------|
| **Drop Rows** | % missing is low (<5%) | Remove rows with NaN in this column |
| **Mean** | Numerical, normally distributed | Replace with column mean |
| **Median** | Numerical, skewed distribution | Replace with column median |
| **Constant** | Domain-specific default | Replace with user-defined value (e.g., 0) |
| **Mode** | Categorical | Replace with most frequent value |
| **"Unknown" Tag** | Categorical, missingness is informative | Replace with explicit "Unknown" category |

**Output State:** `df_clean`

---

## Phase IV: Distribution Analysis & Outlier Handling

**Objective:** Understand the shape of the data and mitigate extreme values that distort statistics.

### 1. Univariate Profiling

**Display:** Side-by-side "Before" and "After" statistics

| Metric | Pre-Outlier Treatment | Post-Outlier Treatment |
|--------|----------------------|------------------------|
| Mean | X | Y |
| Median | X | Y |
| Std Dev | X | Y |
| Skewness | X | Y |
| Kurtosis | X | Y |
| Min | X | Y |
| Max | X | Y |

**Visualizations:**

- Histogram with KDE overlay (distribution shape)
- Boxplot (spread and potential outliers)
- Overlaid before/after comparison

### 2. Outlier Detection Engine

| Component | Specification |
|-----------|---------------|
| Default Algorithm | **IQR Method** (Interquartile Range) |
| Rationale | Assumes less about underlying distribution than Z-Scores |
| Default Threshold | $1.5 \times IQR$ |
| User Configuration | Adjustable multiplier (1.0 – 3.0 slider) |

**IQR Method Formula:**

$$\text{Lower Bound} = Q_1 - (k \times IQR)$$
$$\text{Upper Bound} = Q_3 + (k \times IQR)$$
$$\text{where } IQR = Q_3 - Q_1 \text{ and } k = 1.5 \text{ (default)}$$

**Alternative Methods (User Selectable):**

| Method | Use Case |
|--------|----------|
| Z-Score | Normally distributed data; threshold typically ±3 |
| Modified Z-Score | Robust to non-normality; uses MAD |
| Percentile | Direct cap at user-defined percentiles |

### 3. Outlier Resolution Strategy

**Per-Column User Decision:**

| Strategy | Behavior | Use Case |
|----------|----------|----------|
| **Keep** | Treat as valid edge case | Domain knowledge confirms legitimacy |
| **Drop** | Remove entire row | Outlier likely represents data error |
| **Winsorize (Cap/Floor)** | Replace values > 99th percentile with 99th percentile value | Reduce leverage without data loss |

**UI Component:** Outlier resolution interface

```
┌─────────────────────────────────────────────────────────────────┐
│  Outlier Analysis: income                                       │
├─────────────────────────────────────────────────────────────────┤
│  Method: [IQR ▼]  Threshold: [1.5 ───●─── 3.0]                 │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  [Boxplot visualization with outliers highlighted]       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Outliers Detected: 47 values (0.5% of data)                   │
│  Range: [-$2,340 ... $892,000] outside bounds [$12K, $245K]    │
│                                                                 │
│  Resolution: ○ Keep  ○ Drop Rows  ● Winsorize                  │
│  Winsorize Percentiles: [1st ▼] to [99th ▼]                    │
│                                                                 │
│  [Preview Effect]  [Apply]                                      │
└─────────────────────────────────────────────────────────────────┘
```

**Output State:** `df_outlier_handled`

---

## Phase IV-A: Multicollinearity Pre-Screening

**Objective:** Identify redundant features that carry duplicate information.

### Step 0: Scope Configuration

**User Action:** Define which columns enter multicollinearity analysis.

**Options (Multi-select):**

| Option | Description |
|--------|-------------|
| **Continuous Numerical Only** | Float columns, native integers representing quantities |
| **Include Label-Encoded Ordinals** | Treat ordinal categoricals (Low/Medium/High → 1/2/3) as numerical |
| **Include Binary Flags** | Boolean columns and binary indicators |

**System Guidance:** Display tooltip warning:

> ⚠️ "Including label-encoded ordinals assumes linear relationships between ordinal levels. This may inflate VIF if ordinal spacing is non-uniform (e.g., 'Low' to 'Medium' ≠ 'Medium' to 'High' in real-world magnitude)."

**Implementation Note:** If user selects "Include Label-Encoded Ordinals," system applies **transient label encoding** for VIF/correlation calculation only. Original data state remains unmodified until Phase V.

### 1. Pairwise Correlation Scan

| Component | Specification |
|-----------|---------------|
| System Action | Compute Pearson correlation matrix for selected column scope |
| Default Threshold | Flag pairs where \|r\| > 0.90 |
| User Configuration | Adjustable via slider: 0.70 – 0.99 |
| Display | Heatmap with flagged pairs highlighted; filter toggle to show only flagged pairs |

### 2. Variance Inflation Factor (VIF)

| Component | Specification |
|-----------|---------------|
| System Action | Calculate VIF for each column within selected scope |
| Interpretation | VIF of 10 means 90% of variance is explained by other predictors |

**VIF Formula:**

$$VIF_i = \frac{1}{1 - R_i^2}$$

where $R_i^2$ is the R-squared from regressing feature $i$ on all other features.

**Threshold Options (User Selectable):**

| Level | Threshold | Interpretation |
|-------|-----------|----------------|
| Conservative | VIF > 5 | Strict multicollinearity control |
| Moderate | VIF > 10 | Standard threshold |
| Permissive | VIF > 20 | Relaxed, for exploratory analysis |

**Display:** Ranked table (highest VIF first) with conditional formatting

```
┌─────────────────────────────────────────────────────────────────┐
│  VIF Analysis                                                   │
├─────────────────────────────────────────────────────────────────┤
│  Threshold: ○ >5 (Conservative)  ● >10 (Moderate)  ○ >20       │
├─────────────────────────────────────────────────────────────────┤
│  Column          │ VIF    │ Status    │ Action ▼               │
├──────────────────┼────────┼───────────┼────────────────────────┤
│  total_spend     │ 45.2   │ 🔴 High   │ [Drop ▼]               │
│  avg_transaction │ 38.7   │ 🔴 High   │ [Keep & Flag ▼]        │
│  purchase_count  │ 12.3   │ 🟡 Mod    │ [Keep ▼]               │
│  account_age     │ 2.1    │ 🟢 OK     │ —                      │
│  region_encoded  │ 1.4    │ 🟢 OK     │ —                      │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Resolution Strategy

**Per Flagged Pair/Column:**

| Strategy | Behavior |
|----------|----------|
| **Drop Feature** | Remove from pipeline |
| **Combine Features** | PCA option or manual formula input (e.g., `ratio = A / B`) |
| **Keep & Acknowledge** | Adds `_multicollinear_flag` metadata; proceed with interpretive caution |

**Output State:** `df_collinear_resolved`

---

## Phase V: The "Pre-Correlation" Bridge (Feature Engineering)

**Objective:** Prepare data for mathematical comparison (Correlation/Analysis).

### 1. Categorical Encoding

**Convert String columns to numerical representations.**

| Encoding Type | Use Case | Example |
|---------------|----------|---------|
| **One-Hot Encoding** | Nominal data (no inherent order) | Gender, Color, Country |
| **Label Encoding** | Ordinal data (inherent order) | Low/Medium/High → 1/2/3 |

**UI Component:** Encoding assignment

```
┌─────────────────────────────────────────────────────────────────┐
│  Categorical Encoding                                           │
├─────────────────────────────────────────────────────────────────┤
│  Column        │ Unique Values │ Encoding ▼     │ Preview      │
├────────────────┼───────────────┼────────────────┼──────────────┤
│  gender        │ 3             │ [One-Hot ▼]    │ M→[1,0,0]    │
│  region        │ 5             │ [One-Hot ▼]    │ West→[0,0,1] │
│  priority      │ 3             │ [Label ▼]      │ Low→1, Med→2 │
│  satisfaction  │ 5             │ [Label ▼]      │ 1-5 scale    │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Normalization/Scaling (Optional)

| Method | Formula | Use Case |
|--------|---------|----------|
| **Min-Max Scaling** | $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$ | Bound features to [0, 1] range |
| **Standard Scaling (Z-Score)** | $x' = \frac{x - \mu}{\sigma}$ | Center at 0, unit variance |
| **Robust Scaling** | $x' = \frac{x - Q_2}{Q_3 - Q_1}$ | Robust to outliers |

**User Configuration:**

```
┌─────────────────────────────────────────────────────────────────┐
│  Scaling Options                                                │
├─────────────────────────────────────────────────────────────────┤
│  Apply Scaling: ☑ Yes                                          │
│  Method: ○ Min-Max [0,1]  ● Standard (Z-Score)  ○ Robust       │
│  Apply To: ○ All Numerical  ● Selected Columns Only            │
│                                                                 │
│  Selected: ☑ income  ☑ age  ☐ purchase_count  ☑ tenure        │
└─────────────────────────────────────────────────────────────────┘
```

**Output State:** `df_final`

---

## Distribution Inspector (Global Utility)

**Objective:** On-demand distribution visualization accessible at any pipeline state, enabling users to diagnose data shape without embedding mandatory plots at each phase.

**Access:** Persistent sidebar button or keyboard shortcut (e.g., `Ctrl+D`)

### Core Functionality

#### 1. Single Column View

| Visualization | Description |
|---------------|-------------|
| Histogram + KDE | Distribution shape with kernel density estimate overlay |
| Boxplot | Spread, quartiles, and potential outliers |
| Violin Plot | Combined density and boxplot (optional toggle) |
| Value Counts | Bar chart for categorical columns |

**Statistics Panel:**

| Metric | Description |
|--------|-------------|
| Mean | Arithmetic average |
| Median | 50th percentile |
| Std Dev | Standard deviation |
| Skewness | Distribution asymmetry |
| Kurtosis | Tail heaviness |
| Min / Max | Range bounds |
| Percentiles | 1st, 5th, 25th, 75th, 95th, 99th |
| Missing | Count and percentage |

#### 2. Cross-State Comparison

**Purpose:** Visualize how a column's distribution changes across pipeline states.

**Use Cases:**

| Comparison | Insight |
|------------|---------|
| `df_raw` → `df_schema` | Effect of type casting |
| `df_clean` (pre-imputation) → `df_clean` (post-imputation) | Imputation impact on distribution center |
| `df_clean` → `df_outlier_handled` | Outlier treatment effect on spread and skewness |
| `df_outlier_handled` → `df_final` | Scaling transformation validation |

**Display:**

- Overlaid histograms (semi-transparent, color-coded)
- Side-by-side boxplots
- Delta table with percentage change for each statistic

#### 3. Multi-Column Overview

| View | Description |
|------|-------------|
| Small Multiples Grid | Thumbnail histograms for all numerical columns in current state |
| Correlation Heatmap Preview | Quick correlation matrix (pre-Phase IV-A context) |
| Missing Value Heatmap | Missingness pattern across all columns |
| Type Distribution | Pie/bar chart of column types in current state |

### UI Component

```
┌─────────────────────────────────────────────────────────────────┐
│  📊 Distribution Inspector                            [✕ Close] │
├─────────────────────────────────────────────────────────────────┤
│  Mode: ○ Single Column  ● Compare States  ○ Multi-Column       │
├─────────────────────────────────────────────────────────────────┤
│  State A: [df_clean ▼]        State B: [df_outlier_handled ▼]  │
│  Column: [income ▼]                                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                         │   │
│  │         ░░░░░▒▒▒▒▓▓▓▓████▓▓▒▒░░                        │   │
│  │       ░░▒▒▓▓████████████████▓▓▒▒░░                     │   │
│  │     ░▒▓███████████████████████▓▒░                      │   │
│  │   ░▒▓█████████████████████████▓▒░   ← df_clean (blue)  │   │
│  │  ░▒███████████████████████████▒░                       │   │
│  │  ▒████████████████████████████▒     ← df_outlier (org) │   │
│  │  ▓███████████████████████████▓                         │   │
│  │  ████████████████████████████                          │   │
│  │  ▓██████████████████████████▓                          │   │
│  │ ─┴────┴────┴────┴────┴────┴────┴─                      │   │
│  │  0   20k   40k   60k   80k  100k  120k                 │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  [Histogram ▼]  [Show KDE ☑]  [Log Scale ☐]  [Bins: 30 ▼]      │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  Statistical Comparison                                         │
├─────────────────┬────────────┬──────────────────┬──────────────┤
│  Metric         │ df_clean   │ df_outlier_handl │ Δ Change    │
├─────────────────┼────────────┼──────────────────┼──────────────┤
│  Mean           │ 52,340     │ 51,890           │ -0.86%  ↓   │
│  Median         │ 48,200     │ 48,200           │ 0.00%   ─   │
│  Std Dev        │ 24,500     │ 18,200           │ -25.7%  ↓↓  │
│  Skewness       │ 1.82       │ 0.94             │ -48.4%  ↓↓  │
│  Kurtosis       │ 4.21       │ 2.85             │ -32.3%  ↓↓  │
│  Min            │ -2,340     │ 5,200            │ +322%   ↑↑  │
│  Max            │ 892,000    │ 145,000          │ -83.7%  ↓↓  │
├─────────────────┴────────────┴──────────────────┴──────────────┤
│  Interpretation: Winsorization significantly reduced spread    │
│  and right-skew. Mean shifted minimally; median unchanged.     │
├─────────────────────────────────────────────────────────────────┤
│  [Export PNG]  [Export Statistics CSV]  [Add to Report]        │
└─────────────────────────────────────────────────────────────────┘
```

### Multi-Column Grid View

```
┌─────────────────────────────────────────────────────────────────┐
│  📊 Distribution Inspector — Multi-Column Overview              │
├─────────────────────────────────────────────────────────────────┤
│  State: [df_final ▼]    Show: [Numerical Only ▼]               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ ▂▄▆█▆▄▂ │  │ █▇▅▃▂▁  │  │ ▁▂▄▆█▆▄ │  │ ▃▅▇█▇▅▃ │       │
│  │  income  │  │   age    │  │  tenure  │  │  score   │       │
│  │ μ=51.8k  │  │ μ=34.2   │  │ μ=4.8yr  │  │ μ=72.3   │       │
│  │ sk=0.94  │  │ sk=-0.21 │  │ sk=1.12  │  │ sk=-0.45 │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ █▆▄▂▁   │  │ ▁▃▅▇█▇▅ │  │ ▄▆█▆▄▂  │  │ ▂▄▆█▆▄▂ │       │
│  │ purchases│  │ recency  │  │ frequency│  │ monetary │       │
│  │ μ=12.4   │  │ μ=45.2d  │  │ μ=3.2    │  │ μ=284.50 │       │
│  │ sk=2.34  │  │ sk=0.67  │  │ sk=1.89  │  │ sk=1.45  │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│                                                                 │
│  Legend: μ = mean, sk = skewness                               │
│  🔴 High skew (|sk| > 2)  🟡 Moderate (1-2)  🟢 Low (< 1)      │
│                                                                 │
│  [Click any thumbnail to expand]  [Export All]  [Correlation]  │
└─────────────────────────────────────────────────────────────────┘
```

### Phase-Specific Prompts

While the inspector is globally accessible, the system should **prompt** users at critical junctures:

| Phase | Prompt Trigger | Suggested View |
|-------|----------------|----------------|
| Phase I (Schema) | After type casting numerical columns | Single column: verify range sanity |
| Phase III-B (Imputation) | After imputation applied | Compare states: pre vs post imputation |
| Phase IV (Outliers) | After outlier resolution | Compare states: before vs after treatment |
| Phase V (Scaling) | After scaling applied | Compare states: unscaled vs scaled |

**Prompt UI:**

```
┌─────────────────────────────────────────────────────────────────┐
│  💡 Recommended: Review Distribution Changes                    │
├─────────────────────────────────────────────────────────────────┤
│  You just applied median imputation to 3 columns.              │
│  This may have affected distribution shape.                     │
│                                                                 │
│  [Open Inspector]  [Skip]  [Don't show again for this phase]   │
└─────────────────────────────────────────────────────────────────┘
```

### Export Options

| Format | Contents |
|--------|----------|
| PNG/SVG | Current visualization |
| CSV | Statistics table for selected column(s) |
| PDF Report | Multi-page report with all columns, comparisons, and interpretations |
| JSON | Machine-readable statistics for programmatic use |

---

## Data State Architecture

### State Definitions

| State | Description | Checkpoint Phase | Rollback Enabled |
|-------|-------------|------------------|------------------|
| `df_raw` | File exactly as uploaded | Immutable Origin | No (origin point) |
| `df_schema` | Correct types, selected columns | Post Phase I | Yes |
| `df_deduplicated` | After duplicate resolution | Post Phase I-A | Yes |
| `df_scoped` | After column selection | Post Phase II | Yes |
| `df_clean` | After validity checks and imputation | Post Phase III | Yes |
| `df_outlier_handled` | After outlier treatment | Post Phase IV | Yes |
| `df_collinear_resolved` | After multicollinearity resolution | Post Phase IV-A | Yes |
| `df_final` | After encoding and scaling | Post Phase V | Yes |

### State Flow Diagram

```
df_raw (immutable)
    │
    ▼
df_schema ◄─────────────────────────────────────┐
    │                                            │
    ▼                                            │
df_deduplicated ◄───────────────────────────┐   │
    │                                        │   │
    ▼                                        │   │
df_scoped ◄─────────────────────────────┐   │   │
    │                                    │   │   │
    ▼                                    │   │   │
df_clean ◄──────────────────────────┐   │   │   │  ROLLBACK
    │                                │   │   │   │  PATHS
    ▼                                │   │   │   │
df_outlier_handled ◄────────────┐   │   │   │   │
    │                            │   │   │   │   │
    ▼                            │   │   │   │   │
df_collinear_resolved ◄─────┐   │   │   │   │   │
    │                        │   │   │   │   │   │
    ▼                        │   │   │   │   │   │
df_final ────────────────────┴───┴───┴───┴───┴───┘
    │
    ▼
[EXPORT / CORRELATION ANALYSIS]
```

---

## Pipeline Configuration Object

```python
pipeline_config = {
    "metadata": {
        "version": "1.0",
        "created_at": "2025-01-15T10:30:00Z",
        "source_file": "sales_data.csv",
        "source_hash": "a1b2c3d4..."
    },
    
    "schema": {
        "type_overrides": {
            "zip_code": "category",
            "customer_id": "category",
            "order_date": "datetime64[ns]"
        },
        "selected_columns": ["customer_id", "order_date", "amount", "region", "category"]
    },
    
    "duplicates": {
        "exact_resolution": "keep_first",  # keep_first | keep_last | drop_all
        "subset_keys": ["transaction_id", "customer_id"],
        "subset_resolution": "keep_first",
        "near_duplicate_enabled": false,
        "near_duplicate_threshold": 0.95
    },
    
    "scope": {
        "dropped_columns": ["internal_notes", "created_by"],
        "drop_zero_variance": true,
        "drop_high_cardinality": true
    },
    
    "constraints": {
        "rules": {
            "age": "> 0",
            "probability": "<= 1",
            "end_date": ">= start_date"
        },
        "violations_action": {
            "age": "convert_nan",
            "probability": "drop_row",
            "end_date": "flag_retain"
        }
    },
    
    "imputation": {
        "numerical_strategy": {
            "income": {"method": "median"},
            "age": {"method": "mean"},
            "score": {"method": "constant", "value": 0}
        },
        "categorical_strategy": {
            "region": {"method": "mode"},
            "category": {"method": "unknown_tag", "tag": "Unknown"}
        },
        "drop_threshold": 0.05  # Drop rows if column has < 5% missing
    },
    
    "outliers": {
        "method": "iqr",  # iqr | zscore | modified_zscore | percentile
        "threshold": 1.5,
        "resolution": {
            "income": {"action": "winsorize", "lower": 1, "upper": 99},
            "age": {"action": "keep"},
            "transaction_amount": {"action": "drop"}
        }
    },
    
    "multicollinearity": {
        "scope": ["continuous", "ordinals"],  # continuous | ordinals | binary
        "correlation_threshold": 0.90,
        "vif_threshold": 10,
        "resolved_actions": {
            "total_spend": "dropped",
            "avg_transaction|purchase_count": "kept_with_flag"
        }
    },
    
    "encoding": {
        "one_hot": ["gender", "region", "category"],
        "label": {
            "priority": {"mapping": {"Low": 1, "Medium": 2, "High": 3}},
            "satisfaction": {"mapping": "auto"}  # Auto-infer ordinal order
        }
    },
    
    "scaling": {
        "enabled": true,
        "method": "standard",  # standard | minmax | robust
        "columns": ["income", "age", "tenure"]  # null = all numerical
    }
}
```

---

## Rollback Interface & State Management

### Architecture: Branching State Model

**Core Principle:** Each checkpoint is immutable once committed. Rollback creates a new branch rather than overwriting history.

```
df_raw
  └── df_schema (v1)
        └── df_deduplicated (v1)
              └── df_scoped (v1)
                    └── df_clean (v1)
                          └── df_outlier_handled (v1)
                                └── df_collinear_resolved (v1)
                                      └── df_final (v1)
                          └── df_outlier_handled (v2)  ← Rolled back, changed strategy
                                └── df_collinear_resolved (v2)
                                      └── df_final (v2)
```

### Storage Strategy (MVP)

**Approach:** Full DataFrame Copy + LRU Cache Eviction

| Component | Implementation |
|-----------|----------------|
| In-Memory Storage | Store full DataFrame copy at each checkpoint |
| Cache Limit | Retain 5 most recently accessed states in memory |
| Eviction Policy | LRU (Least Recently Used) — evict oldest accessed state when limit reached |
| Disk Fallback | Evicted states serialize to temp Parquet files; reload on access |
| Session Cleanup | Purge temp files on session end or explicit user action |

### User Interface Components

#### 1. State Timeline (Sidebar Panel)

```
┌─────────────────────────────────────┐
│  📊 Pipeline States                 │
├─────────────────────────────────────┤
│  Branch: [main ▼] [+ New Branch]    │
├─────────────────────────────────────┤
│                                     │
│  ● df_raw                           │
│  │ 10,000 rows × 25 cols            │
│  │ 10:30:00                         │
│  │                                  │
│  ● df_schema                        │
│  │ Cast 3 columns                   │
│  │ 10:32:00                         │
│  │                                  │
│  ● df_deduplicated                  │
│  │ Dropped 142 duplicates           │
│  │ 10:35:00                         │
│  │                                  │
│  ● df_scoped                        │
│  │ Selected 18 of 25 columns        │
│  │ 10:38:00                         │
│  │                                  │
│  ◉ df_clean  ← ACTIVE               │
│  │ Imputed 3 cols, flagged 12 rows  │
│  │ 10:45:00                         │
│  │                                  │
│  ○ df_outlier_handled (pending)     │
│  ○ df_collinear_resolved (pending)  │
│  ○ df_final (pending)               │
│                                     │
├─────────────────────────────────────┤
│  [↩ Rollback] [📤 Export] [🔄 Comp] │
└─────────────────────────────────────┘
```

#### 2. Rollback Modal

```
┌─────────────────────────────────────────────┐
│  ↩ Rollback to State                        │
├─────────────────────────────────────────────┤
│                                             │
│  Target: df_schema (main)                   │
│  Timestamp: 2025-01-15 10:32:00             │
│                                             │
│  ⚠ This will create a new branch.           │
│  Current branch "main" remains unchanged.   │
│                                             │
│  New branch name:                           │
│  ┌─────────────────────────────────────┐    │
│  │ branch-1                            │    │
│  └─────────────────────────────────────┘    │
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌────────┐    │
│  │ Branch & │  │ Preview  │  │ Cancel │    │
│  │ Continue │  │ Only     │  │        │    │
│  └──────────┘  └──────────┘  └────────┘    │
│                                             │
└─────────────────────────────────────────────┘
```

#### 3. Branch Comparison View

```
┌────────────────────────────────────────────────────────────────┐
│  🔄 Compare Branches                                           │
├────────────────────────────────────────────────────────────────┤
│  Branch A: [main ▼]          Branch B: [branch-1 ▼]            │
│  State: df_final             State: df_final                   │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Summary Comparison:                                           │
│  ┌─────────────────┬──────────────┬──────────────┐            │
│  │ Metric          │ main         │ branch-1     │            │
│  ├─────────────────┼──────────────┼──────────────┤            │
│  │ Row Count       │ 9,412        │ 9,156        │            │
│  │ Column Count    │ 32           │ 30           │            │
│  │ Memory (MB)     │ 24.3         │ 22.1         │            │
│  └─────────────────┴──────────────┴──────────────┘            │
│                                                                │
│  Column Differences:                                           │
│  • branch-1 dropped: [col_x, col_y]                           │
│  • main retained: [col_x, col_y] (multicollinear, flagged)    │
│                                                                │
│  Distribution Comparison: [Select Column ▼]                    │
│  ┌────────────────────────────────────────────┐               │
│  │  [Overlaid histogram visualization]        │               │
│  │  Blue: main | Orange: branch-1             │               │
│  └────────────────────────────────────────────┘               │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Branch Management

| Action | Behavior |
|--------|----------|
| **Switch Branch** | Load terminal state of selected branch as active |
| **Rename Branch** | User-defined label for clarity (e.g., "IQR Outliers" vs "Z-Score Outliers") |
| **Compare Branches** | Side-by-side diff of `df_final` across two branches |
| **Delete Branch** | Remove branch and all associated states (confirmation required) |
| **Merge Branch** | Not supported — branches represent alternative pipelines |

### Edge Cases & Guardrails

| Scenario | System Behavior |
|----------|-----------------|
| User uploads new file | Prompt: "Clear all branches and start fresh?" (confirmation required) |
| Branch limit reached (10) | Prompt user to delete unused branches before creating new |
| Session timeout/crash | Auto-save `state_store` to localStorage every 60 seconds; recover on reload |
| Disk quota exceeded | Alert user, suggest exporting and deleting old branches |
| Concurrent tab edits | Lock pipeline to single active tab; show warning in secondary tabs |

---

## Implementation Reference

### State Manager Class

```python
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Optional
import pandas as pd
import hashlib
import json


@dataclass
class StateRecord:
    """Metadata for a single pipeline state."""
    id: str
    parent: Optional[str]
    timestamp: str
    data_hash: str
    row_count: int
    col_count: int
    config_snapshot: dict
    delta_summary: str


@dataclass
class Branch:
    """A branch in the pipeline state tree."""
    created_at: str
    forked_from: Optional[dict]
    states: list[StateRecord]
    
    def to_dict(self) -> dict:
        return {
            "created_at": self.created_at,
            "forked_from": self.forked_from,
            "states": [asdict(s) for s in self.states]
        }


class StateNotFoundError(Exception):
    """Raised when a requested state cannot be found."""
    pass


class StateManager:
    """
    Manages pipeline states with branching and rollback support.
    
    Uses full DataFrame copies with LRU eviction for MVP.
    Evicted states persist to disk as Parquet files.
    """
    
    def __init__(
        self, 
        cache_size: int = 5, 
        temp_dir: Path = Path("/tmp/pipeline_states")
    ):
        self.cache_size = cache_size
        self.temp_dir = temp_dir
        self.temp_dir.mkdir(exist_ok=True)
        
        self.branches: dict[str, Branch] = {
            "main": Branch(
                created_at=pd.Timestamp.now().isoformat(),
                forked_from=None,
                states=[]
            )
        }
        self.active_branch: str = "main"
        self._memory_cache: dict[str, pd.DataFrame] = {}
        self._access_order: list[str] = []
    
    def commit_state(
        self,
        state_id: str,
        df: pd.DataFrame,
        config_snapshot: dict,
        delta_summary: str
    ) -> None:
        """
        Commit a new state to the active branch.
        
        Args:
            state_id: Identifier for this state (e.g., 'df_clean')
            df: The DataFrame at this checkpoint
            config_snapshot: Configuration used to reach this state
            delta_summary: Human-readable description of changes
        """
        state_key = f"{self.active_branch}::{state_id}"
        
        # Store in memory cache
        self._add_to_cache(state_key, df.copy())
        
        # Record metadata
        branch = self.branches[self.active_branch]
        parent = branch.states[-1].id if branch.states else None
        
        branch.states.append(StateRecord(
            id=state_id,
            parent=parent,
            timestamp=pd.Timestamp.now().isoformat(),
            data_hash=self._compute_hash(df),
            row_count=len(df),
            col_count=len(df.columns),
            config_snapshot=config_snapshot,
            delta_summary=delta_summary
        ))
    
    def load_state(self, branch: str, state_id: str) -> pd.DataFrame:
        """
        Load a state from cache or disk.
        
        Args:
            branch: Branch name
            state_id: State identifier
            
        Returns:
            DataFrame at the requested state
            
        Raises:
            StateNotFoundError: If state doesn't exist
        """
        state_key = f"{branch}::{state_id}"
        
        # Check memory cache
        if state_key in self._memory_cache:
            self._touch(state_key)
            return self._memory_cache[state_key].copy()
        
        # Load from disk
        disk_path = self.temp_dir / f"{state_key.replace('::', '__')}.parquet"
        if disk_path.exists():
            df = pd.read_parquet(disk_path)
            self._add_to_cache(state_key, df)
            return df.copy()
        
        raise StateNotFoundError(f"State {state_key} not found")
    
    def create_branch(
        self, 
        new_branch_name: str, 
        fork_from_state: str
    ) -> None:
        """
        Create a new branch from a specific state.
        
        Args:
            new_branch_name: Name for the new branch
            fork_from_state: State ID to fork from
        """
        source_branch = self.active_branch
        source_df = self.load_state(source_branch, fork_from_state)
        source_config = self._get_config_at_state(source_branch, fork_from_state)
        
        # Create new branch
        self.branches[new_branch_name] = Branch(
            created_at=pd.Timestamp.now().isoformat(),
            forked_from={"branch": source_branch, "state_id": fork_from_state},
            states=[]
        )
        
        # Switch to new branch and commit fork point
        self.active_branch = new_branch_name
        self.commit_state(
            state_id=fork_from_state,
            df=source_df,
            config_snapshot=source_config,
            delta_summary=f"Forked from {source_branch}::{fork_from_state}"
        )
    
    def switch_branch(self, branch_name: str) -> pd.DataFrame:
        """
        Switch to a different branch.
        
        Args:
            branch_name: Branch to switch to
            
        Returns:
            DataFrame at the terminal state of that branch
        """
        if branch_name not in self.branches:
            raise ValueError(f"Branch '{branch_name}' does not exist")
        
        self.active_branch = branch_name
        terminal_state = self.branches[branch_name].states[-1].id
        return self.load_state(branch_name, terminal_state)
    
    def delete_branch(self, branch_name: str) -> None:
        """
        Delete a branch and its cached states.
        
        Args:
            branch_name: Branch to delete (cannot be 'main')
        """
        if branch_name == "main":
            raise ValueError("Cannot delete main branch")
        if branch_name == self.active_branch:
            raise ValueError("Cannot delete active branch")
        
        # Remove from cache
        keys_to_remove = [k for k in self._memory_cache if k.startswith(f"{branch_name}::")]
        for key in keys_to_remove:
            del self._memory_cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
        
        # Remove disk files
        for file in self.temp_dir.glob(f"{branch_name}__*.parquet"):
            file.unlink()
        
        # Remove branch
        del self.branches[branch_name]
    
    def get_branch_summary(self, branch_name: str) -> dict:
        """Get summary statistics for a branch."""
        branch = self.branches[branch_name]
        if not branch.states:
            return {"states": 0, "terminal_state": None}
        
        terminal = branch.states[-1]
        return {
            "states": len(branch.states),
            "terminal_state": terminal.id,
            "row_count": terminal.row_count,
            "col_count": terminal.col_count,
            "last_modified": terminal.timestamp
        }
    
    def compare_branches(
        self, 
        branch_a: str, 
        branch_b: str,
        state_id: str = "df_final"
    ) -> dict:
        """
        Compare terminal states of two branches.
        
        Returns:
            Dictionary with comparison metrics
        """
        df_a = self.load_state(branch_a, state_id)
        df_b = self.load_state(branch_b, state_id)
        
        cols_a = set(df_a.columns)
        cols_b = set(df_b.columns)
        
        return {
            "branch_a": {
                "rows": len(df_a),
                "cols": len(df_a.columns),
                "memory_mb": df_a.memory_usage(deep=True).sum() / 1e6
            },
            "branch_b": {
                "rows": len(df_b),
                "cols": len(df_b.columns),
                "memory_mb": df_b.memory_usage(deep=True).sum() / 1e6
            },
            "columns_only_in_a": list(cols_a - cols_b),
            "columns_only_in_b": list(cols_b - cols_a),
            "common_columns": list(cols_a & cols_b)
        }
    
    def export_pipeline(self, output_dir: Path) -> None:
        """
        Export complete pipeline for reproducibility.
        
        Creates:
            - df_raw.parquet: Original uploaded data
            - state_store.json: Complete branch/state metadata
            - pipeline_config.json: Terminal configuration
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export raw data
        raw_df = self.load_state("main", "df_raw")
        raw_df.to_parquet(output_dir / "df_raw.parquet")
        
        # Export state store
        store_export = {
            "branches": {
                name: branch.to_dict() 
                for name, branch in self.branches.items()
            },
            "active_branch": self.active_branch
        }
        with open(output_dir / "state_store.json", "w") as f:
            json.dump(store_export, f, indent=2)
        
        # Export terminal config
        terminal_state = self.branches[self.active_branch].states[-1]
        with open(output_dir / "pipeline_config.json", "w") as f:
            json.dump(terminal_state.config_snapshot, f, indent=2)
    
    def _add_to_cache(self, state_key: str, df: pd.DataFrame) -> None:
        """Add to cache with LRU eviction."""
        if state_key in self._memory_cache:
            self._touch(state_key)
            return
        
        # Evict if at capacity
        while len(self._memory_cache) >= self.cache_size:
            evict_key = self._access_order.pop(0)
            evicted_df = self._memory_cache.pop(evict_key)
            
            # Persist to disk
            safe_key = evict_key.replace("::", "__")
            disk_path = self.temp_dir / f"{safe_key}.parquet"
            evicted_df.to_parquet(disk_path)
        
        self._memory_cache[state_key] = df
        self._access_order.append(state_key)
    
    def _touch(self, state_key: str) -> None:
        """Mark state as recently accessed."""
        if state_key in self._access_order:
            self._access_order.remove(state_key)
        self._access_order.append(state_key)
    
    def _compute_hash(self, df: pd.DataFrame) -> str:
        """Compute hash for integrity verification."""
        return hashlib.md5(
            pd.util.hash_pandas_object(df).values.tobytes()
        ).hexdigest()
    
    def _get_config_at_state(self, branch: str, state_id: str) -> dict:
        """Retrieve config snapshot for a specific state."""
        for state in self.branches[branch].states:
            if state.id == state_id:
                return state.config_snapshot
        return {}
    
    def cleanup(self) -> None:
        """Purge all temp files."""
        for file in self.temp_dir.glob("*.parquet"):
            file.unlink()
```

### Future Optimization Path

| Phase | Trigger | Implementation |
|-------|---------|----------------|
| MVP (Current) | Datasets < 100MB | Full copy + LRU eviction |
| Scale Phase 1 | Datasets 100MB–1GB | Delta storage + lazy reconstruction |
| Scale Phase 2 | Datasets > 1GB | Server-side state management + streaming |

---

## GDPR Compliance Framework

**Objective:** Ensure the application meets EU General Data Protection Regulation requirements for processing user-uploaded data.

### Compliance Architecture

| GDPR Principle | Implementation |
|----------------|----------------|
| **Lawful Basis** | Explicit consent gate before file upload |
| **Transparency** | Privacy notice with full processing details |
| **Data Minimization** | Process only user-uploaded data; no telemetry |
| **Storage Limitation** | Session-scoped only; no persistence |
| **Integrity & Confidentiality** | HTTPS; no third-party transmission |
| **Accountability** | Documented policies; audit-ready cleanup logs |

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    GDPR-COMPLIANT DATA FLOW                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User's Machine          Streamlit Cloud           Third Parties│
│  ┌──────────┐           ┌──────────────┐          ┌───────────┐│
│  │ CSV/Excel│──upload──▶│  RAM (df_*)  │          │   None    ││
│  │   File   │           │              │          │ (no APIs) ││
│  └──────────┘           │  /tmp cache  │          └───────────┘│
│                         │  (Parquet)   │                       │
│       ▲                 │              │                       │
│       │                 │  Session     │                       │
│       │                 │  State       │                       │
│  ┌────┴─────┐           └──────────────┘                       │
│  │ Download │◀──export──       │                               │
│  │ df_final │                  │                               │
│  └──────────┘                  ▼                               │
│                         Session End:                           │
│                         Data Purged                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Required UI Components

#### 1. Consent Gate (Pre-Upload)

The application MUST display this before any file upload is enabled:

```
┌─────────────────────────────────────────────────────────────────┐
│  🔒 Data Privacy Notice                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Before uploading data, please review how it will be handled:  │
│                                                                 │
│  ✓ Processed in-memory only (not stored permanently)           │
│  ✓ Deleted automatically when you close this tab               │
│  ✓ Never shared with third parties                             │
│  ✓ You can delete all data anytime via the footer control      │
│                                                                 │
│  [▼ Read Full Privacy Policy]                                  │
│                                                                 │
│  ☐ I understand and consent to data processing                 │
│                                                                 │
│  [Continue to Application]  (disabled until checkbox selected) │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 2. Persistent Privacy Footer

Displayed on every page:

```
┌─────────────────────────────────────────────────────────────────┐
│  🔒 Data processed in-memory only    [🗑️ Delete All]  [📋 Policy]│
└─────────────────────────────────────────────────────────────────┘
```

#### 3. Privacy Policy Expander Content

```markdown
### How Your Data Is Handled

**Data Controller:** [Your Name/Organization]

**Processing Location:** Streamlit Community Cloud (servers may be in United States)

**What We Process:**
- The file(s) you upload for analysis
- Configuration choices you make during the pipeline

**What We Do NOT Do:**
- Store your data beyond your browser session
- Share your data with third parties
- Use your data for training or analytics
- Log or retain any uploaded file contents

**Data Retention:**
- All uploaded data exists only in temporary memory
- Data is automatically deleted when you close the browser tab
- No backups or copies are retained

**Your Rights (GDPR Article 15-22):**
- **Access:** Your data is visible to you throughout the session
- **Rectification:** You control all data modifications via the pipeline
- **Erasure:** Close the browser tab to immediately delete all data
- **Portability:** Export your processed data at any time

**Security Measures:**
- All connections use HTTPS encryption
- No data persistence beyond session memory
- No third-party analytics or tracking

**Legal Basis:** Consent (GDPR Article 6(1)(a))

**Contact:** [your-email@domain.com]
```

### Session Cleanup Behavior

| Trigger | Action |
|---------|--------|
| Browser tab closed | Streamlit session ends; all data purged |
| Session timeout (default: 15 min idle) | Automatic cleanup |
| User clicks "Delete All Data" | Immediate purge; session reset |
| Server restart/deploy | All sessions terminated; no persistence |

### Data Flow Constraints

**Prohibited:**

- ❌ Logging uploaded file contents
- ❌ Third-party analytics with data payload (e.g., Sentry with full context)
- ❌ External API calls with user data
- ❌ Persistent database storage
- ❌ Cookies beyond session ID
- ❌ Google Analytics or similar tracking

**Required:**

- ✅ HTTPS for all connections (Streamlit Cloud default)
- ✅ Session-scoped state only
- ✅ Temp file cleanup on session end
- ✅ User-accessible deletion control
- ✅ Privacy policy link accessible from every page
- ✅ Consent timestamp recorded in session state

### GDPR Consent Component Implementation

```python
# components/gdpr_consent.py

import streamlit as st
from datetime import datetime

def show_gdpr_consent() -> bool:
    """
    Display GDPR consent banner. Returns True if user consents.
    Must be shown before any file upload is enabled.
    """
    
    if st.session_state.get("gdpr_consent_given"):
        return True
    
    st.markdown("---")
    st.markdown("## 🔒 Data Privacy Notice")
    
    with st.expander("Read Full Privacy Policy", expanded=False):
        st.markdown("""
        ### How Your Data Is Handled
        
        **Data Controller:** [Your Name/Organization]
        
        **Processing Location:** Streamlit Community Cloud
        
        **What We Process:**
        - The file(s) you upload for analysis
        - Configuration choices you make during the pipeline
        
        **What We Do NOT Do:**
        - Store your data beyond your browser session
        - Share your data with third parties
        - Use your data for training or analytics
        - Log or retain any uploaded file contents
        
        **Data Retention:**
        - All uploaded data exists only in temporary memory
        - Data is automatically deleted when you close the browser tab
        - No backups or copies are retained
        
        **Your Rights (GDPR Article 15-22):**
        - **Access:** Your data is visible to you throughout the session
        - **Rectification:** You control all data modifications via the pipeline
        - **Erasure:** Close the browser tab to immediately delete all data
        - **Portability:** Export your processed data at any time
        
        **Security Measures:**
        - All connections use HTTPS encryption
        - No data persistence beyond session memory
        - No third-party analytics or tracking
        
        **Legal Basis:** Consent (GDPR Article 6(1)(a))
        
        **Contact:** [your-email@domain.com]
        """)
    
    st.markdown("""
    **Summary:** Your uploaded data is processed in-memory only and 
    automatically deleted when your session ends. No data is stored, 
    shared, or retained.
    """)
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        consent = st.checkbox("I understand and consent")
    
    with col2:
        if consent:
            if st.button("Continue to Application", type="primary"):
                st.session_state.gdpr_consent_given = True
                st.session_state.consent_timestamp = datetime.utcnow().isoformat()
                st.rerun()
    
    if not consent:
        st.info("☝️ Please review and accept the privacy notice to continue.")
    
    return False


def show_privacy_footer():
    """Persistent footer with privacy controls."""
    
    st.markdown("---")
    cols = st.columns([2, 1, 1])
    
    with cols[0]:
        st.caption("🔒 Your data is processed in-memory only and not stored.")
    
    with cols[1]:
        if st.button("🗑️ Delete All Data", key="footer_delete"):
            clear_all_session_data()
            st.success("All data cleared.")
            st.rerun()
    
    with cols[2]:
        if st.button("📋 Privacy Policy", key="footer_privacy"):
            st.session_state.show_privacy_modal = True


def clear_all_session_data():
    """Explicitly clear all user data from session."""
    
    # Clear all dataframe states
    keys_to_clear = [
        k for k in st.session_state.keys() 
        if k.startswith("df_") or k.startswith("pipeline_") or k.startswith("state_")
    ]
    
    for key in keys_to_clear:
        del st.session_state[key]
    
    # Clear temp files
    import shutil
    from pathlib import Path
    temp_dir = Path("/tmp/pipeline_states")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        temp_dir.mkdir()
```

### StateManager Cleanup Hooks

```python
# Add to pipeline/state_manager.py

import atexit
import signal

class StateManager:
    def __init__(self, ...):
        # ... existing init ...
        
        # Register cleanup on session end
        atexit.register(self.cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals."""
        self.cleanup()
        exit(0)
    
    def cleanup(self) -> None:
        """Purge all temp files — GDPR erasure compliance."""
        import shutil
        
        # Clear memory
        self._memory_cache.clear()
        self._access_order.clear()
        
        # Clear disk
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        
        # Log for compliance audit (no user data in log)
        print(f"[GDPR] Session data purged at {pd.Timestamp.now().isoformat()}")
```

### Audit Trail

The StateManager logs cleanup events without user data:

```
[GDPR] Session data purged at 2025-01-15T14:32:00Z
```

No personally identifiable information or file contents appear in logs.

---

## Deployment Guide (GitHub + Streamlit Cloud)

### Repository Structure

```
data-pipeline-app/
├── .streamlit/
│   └── config.toml
├── app.py                      # Main Streamlit entry point
├── pipeline/
│   ├── __init__.py
│   ├── state_manager.py        # StateManager class with rollback
│   ├── ingestion.py            # Phase I: Schema enforcement
│   ├── duplicates.py           # Phase I-A: Duplicate detection
│   ├── scope.py                # Phase II: Column selection
│   ├── cleaning.py             # Phase III: Constraints & imputation
│   ├── outliers.py             # Phase IV: Outlier handling
│   ├── multicollinearity.py    # Phase IV-A: VIF & correlation
│   └── encoding.py             # Phase V: Encoding & scaling
├── components/
│   ├── __init__.py
│   ├── gdpr_consent.py         # GDPR consent gate & footer
│   ├── distribution_inspector.py
│   ├── rollback_interface.py
│   └── ui_elements.py
├── utils/
│   ├── __init__.py
│   ├── visualization.py        # Plotting utilities
│   └── export.py               # Export functionality
├── requirements.txt
├── README.md
├── PRIVACY.md
├── LICENSE
└── .gitignore
```

### Streamlit Configuration

```toml
# .streamlit/config.toml

[server]
maxUploadSize = 100
enableXsrfProtection = true
enableCORS = false

[browser]
gatherUsageStats = false  # CRITICAL: Disable Streamlit telemetry for GDPR

[client]
showErrorDetails = false  # Prevent data leakage in error messages

[theme]
primaryColor = "#4F8BF9"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Requirements File

```txt
# requirements.txt

streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=14.0.0
scipy>=1.11.0
statsmodels>=0.14.0
plotly>=5.18.0
openpyxl>=3.1.0
xlsxwriter>=3.1.0
```

### Main Application Entry Point

```python
# app.py

import streamlit as st
from components.gdpr_consent import show_gdpr_consent, show_privacy_footer
from pipeline.state_manager import StateManager

st.set_page_config(
    page_title="Data Preparation Pipeline",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_session_state():
    """Initialize session state variables."""
    if "state_manager" not in st.session_state:
        st.session_state.state_manager = StateManager()
    if "current_phase" not in st.session_state:
        st.session_state.current_phase = "upload"

def main():
    st.title("📊 Data Preparation Pipeline")
    
    # GDPR gate — must consent before accessing app
    if not show_gdpr_consent():
        st.stop()
    
    # Initialize session
    init_session_state()
    
    # Sidebar: State Timeline & Rollback Interface
    with st.sidebar:
        st.header("Pipeline States")
        # ... rollback interface component ...
    
    # Main content: Current phase
    # ... phase-specific UI based on st.session_state.current_phase ...
    
    # Persistent privacy footer
    show_privacy_footer()


if __name__ == "__main__":
    main()
```

### Git Ignore File

```gitignore
# .gitignore

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/

# Streamlit
.streamlit/secrets.toml

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project-specific
*.parquet
*.csv
*.xlsx
/tmp/
pipeline_states/
```

### Privacy Policy File

```markdown
# PRIVACY.md

# Privacy Policy

Last Updated: [Date]

## Overview

This application processes data files entirely in-memory. No user data is stored, 
logged, or transmitted to third parties.

## Data Processing

| Data Type | Processing Location | Retention Period |
|-----------|---------------------|------------------|
| Uploaded files | Streamlit Cloud (RAM) | Session only |
| Pipeline configuration | Streamlit Cloud (RAM) | Session only |
| Temporary cache | Streamlit Cloud (/tmp) | Session only |

## GDPR Compliance

This application is designed to comply with the EU General Data Protection 
Regulation (GDPR):

- **Lawful Basis:** User consent prior to upload (Article 6(1)(a))
- **Data Minimization:** Only user-provided data is processed (Article 5(1)(c))
- **Storage Limitation:** No data retained beyond session (Article 5(1)(e))
- **Right to Erasure:** Automatic on session end; manual via "Delete All Data" (Article 17)
- **Data Portability:** Export available at any pipeline stage (Article 20)

## Sub-Processors

| Provider | Purpose | Location | DPA |
|----------|---------|----------|-----|
| Streamlit (Snowflake Inc.) | Application hosting | USA | [Snowflake DPA](https://www.snowflake.com/legal/dpa/) |

## Data Controller

[Your Name / Organization]
[Your Address]
[your-email@domain.com]

## User Rights

To exercise your GDPR rights:
1. Use the in-app "Delete All Data" button for immediate erasure
2. Close your browser tab to end the session and delete all data
3. Contact the Data Controller for any other requests

## Changes to This Policy

We may update this policy periodically. The "Last Updated" date will reflect 
the most recent revision.
```

### README File

```markdown
# README.md

# 📊 Data Preparation Pipeline

A comprehensive, GDPR-compliant data preparation tool for cleaning, validating, 
and transforming datasets before correlation analysis.

## Features

- **Multi-format ingestion:** CSV, Excel, Parquet, JSON
- **Schema enforcement:** Type casting with user validation
- **Duplicate detection:** Exact, subset, and near-duplicate identification
- **Data sanitation:** Constraint enforcement and missing value imputation
- **Outlier handling:** IQR, Z-score, and winsorization methods
- **Multicollinearity screening:** VIF and correlation matrix analysis
- **Feature engineering:** Encoding and scaling transformations
- **Distribution inspector:** On-demand visualization at any pipeline state
- **Branching rollback:** Non-destructive state management with comparison

## Privacy & GDPR

This application processes all data in-memory only. No data is stored, logged, 
or transmitted to third parties. See [PRIVACY.md](PRIVACY.md) for full details.

## Deployment

### Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Cloud

1. Fork/clone this repository
2. Connect to [share.streamlit.io](https://share.streamlit.io)
3. Select repository and `app.py` as entry point
4. Deploy

## License

[Your chosen license]
```

### Deployment Steps

**1. Create GitHub Repository**

```bash
# Initialize repository
git init
git add .
git commit -m "Initial commit: Data Pipeline MVP"

# Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/data-pipeline-app.git
git branch -M main
git push -u origin main
```

**2. Deploy to Streamlit Cloud**

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub account (if not already connected)
4. Select:
   - Repository: `YOUR_USERNAME/data-pipeline-app`
   - Branch: `main`
   - Main file path: `app.py`
5. Click "Deploy"

**3. Configure App Settings (Optional)**

In Streamlit Cloud dashboard:
- Set custom subdomain: `your-app-name.streamlit.app`
- Configure secrets (if needed in future): Settings → Secrets

**4. Verify Deployment**

- Test file upload functionality
- Verify GDPR consent gate appears
- Confirm "Delete All Data" clears session
- Check privacy footer displays correctly

### Streamlit Cloud Limitations

| Constraint | Limit | Workaround |
|------------|-------|------------|
| RAM | 1 GB | Limit upload size; use chunked processing |
| Upload size | Configurable (default 200MB) | Set `maxUploadSize` in config |
| Session timeout | ~15 min idle | User re-uploads if session expires |
| Public repos only (free tier) | N/A | Use private repos with paid tier |
| Sleep on inactivity | ~7 days | App wakes on next visit |

### Future Scaling Path

| Phase | Trigger | Migration Target |
|-------|---------|------------------|
| MVP | < 100 users/day | Streamlit Cloud (free) |
| Growth | 100-1000 users/day | Streamlit Cloud (paid) or Railway |
| Scale | > 1000 users/day | AWS ECS / GCP Cloud Run |

---

## Appendix: Quick Reference

### Phase Checklist

- [ ] **Phase I:** Schema enforced, types validated
- [ ] **Phase I-A:** Duplicates resolved
- [ ] **Phase II:** Columns scoped, irrelevant features dropped
- [ ] **Phase III-A:** Constraints enforced, violations handled
- [ ] **Phase III-B:** Missing values imputed
- [ ] **Phase IV:** Outliers detected and resolved
- [ ] **Phase IV-A:** Multicollinearity screened, redundant features addressed
- [ ] **Phase V:** Categoricals encoded, scaling applied
- [ ] **GDPR:** Consent obtained, privacy footer visible
- [ ] **Export:** `df_final` ready for correlation analysis

### State Progression Summary

```
RAW → SCHEMA → DEDUPLICATED → SCOPED → CLEAN → OUTLIER_HANDLED → COLLINEAR_RESOLVED → FINAL
```

### Key Configuration Decisions

| Decision Point | Options | Default |
|----------------|---------|---------|
| Duplicate resolution | keep_first / keep_last / drop_all | keep_first |
| Constraint violation | drop_row / convert_nan / flag_retain | convert_nan |
| Imputation (numerical) | mean / median / constant | median |
| Imputation (categorical) | mode / unknown_tag | mode |
| Outlier method | iqr / zscore / percentile | iqr |
| Outlier resolution | keep / drop / winsorize | winsorize |
| VIF threshold | 5 / 10 / 20 | 10 |
| Scaling method | standard / minmax / robust | standard |

### GDPR Compliance Checklist

- [ ] Consent gate displayed before file upload
- [ ] Privacy policy accessible from every page
- [ ] "Delete All Data" button in footer
- [ ] `gatherUsageStats = false` in config
- [ ] No third-party analytics integrated
- [ ] Session cleanup on termination
- [ ] PRIVACY.md in repository root

### Deployment Checklist

- [ ] Repository structure matches specification
- [ ] `.streamlit/config.toml` configured correctly
- [ ] `requirements.txt` includes all dependencies
- [ ] `README.md` and `PRIVACY.md` present
- [ ] `.gitignore` excludes data files and temp directories
- [ ] GitHub repository connected to Streamlit Cloud
- [ ] App deploys and GDPR gate functions correctly

### File Type Support Matrix

| Format | Extension | Library | Notes |
|--------|-----------|---------|-------|
| CSV | .csv | pandas | Auto-detect delimiter |
| Excel | .xlsx, .xls | openpyxl | Multiple sheets supported |
| Parquet | .parquet | pyarrow | Preserves types |
| JSON | .json | pandas | Records or columnar |

### Keyboard Shortcuts (Proposed)

| Shortcut | Action |
|----------|--------|
| `Ctrl+D` | Open Distribution Inspector |
| `Ctrl+Z` | Open Rollback Interface |
| `Ctrl+E` | Export Current State |
| `Ctrl+S` | Save Configuration |
| `Esc` | Close Modal/Inspector |

### Error Handling Reference

| Error Type | User Message | Resolution |
|------------|--------------|------------|
| File too large | "File exceeds 100MB limit" | Suggest chunking or sampling |
| Invalid file type | "Unsupported format" | List accepted formats |
| Parsing error | "Could not parse file" | Offer encoding override |
| Memory exceeded | "Dataset too large for processing" | Suggest column reduction |
| Session expired | "Session timed out" | Re-upload file |

---

## Document Metadata

| Field | Value |
|-------|-------|
| **Version** | 1.0 |
| **Last Updated** | December 2025 |
| **Status** | Complete Blueprint |
| **Target Platform** | Streamlit Cloud |
| **GDPR Status** | Compliant by Design |
| **Repository Template** | GitHub + Streamlit Cloud |

---

**End of Document**
