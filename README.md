# Statistical IDE for Regression Analysis

A Type-Checked Statistical Compiler that prevents invalid analysis states.

## Overview

This application is a Streamlit-based IDE designed for rigorous regression analysis. It treats statistical assumptions as type constraints, ensuring that models are only applied when their underlying assumptions (linearity, normality, homoscedasticity, etc.) are met.

## Key Features

- **Immutable Data Snapshots**: Data states are preserved as immutable snapshots, allowing for safe experimentation and backtracking.
- **Type-Checked Statistics**: Statistical tests act as type constraints, preventing the application of incompatible models.
- **Stateful Transformations**: Transformations are fit on training data and correctly applied to test data to prevent leakage.
- **Cached Diagnostics**: Diagnostic results are cached and automatically invalidated when the data state changes.
- **Impact Analysis**: Remediation strategies show collateral impact before execution.

## Architecture

The system is built on a layered architecture:
1.  **Core Data Structures**: Immutable snapshots and provenance tracking.
2.  **Model Registry & Type System**: Definitions of models and their required statistical assumptions.
3.  **Diagnostic Engine**: Automated testing of statistical assumptions.

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the Streamlit application:

```bash
streamlit run statistical_ide.py
```

## Dependencies

-   streamlit
-   pandas
-   numpy
-   scikit-learn
-   scipy
-   statsmodels
-   plotly
-   openpyxl

## License

MIT
