"""
Statistical IDE for Regression Analysis
=======================================

A Type-Checked Statistical Compiler that prevents invalid analysis states.

Architecture:
- Data states as immutable snapshots with computed type traits
- Statistical tests as type constraints on model compatibility
- Transformations are stateful (fit on train, apply to test)
- All diagnostics are cached and invalidated on state changes
- Remediation shows collateral impact before execution

Author: Statistical IDE Team
License: MIT
"""

from __future__ import annotations

import uuid
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import streamlit as st
from numpy.linalg import LinAlgError
from scipy.stats import anderson, shapiro
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


# =============================================================================
# LAYER 1: CORE DATA STRUCTURES
# =============================================================================


class AssumptionTrait(Enum):
    """
    Enumeration of statistical assumptions that can be tested.

    These traits represent the core assumptions underlying various
    regression models. Each trait has associated diagnostic tests
    and remediation strategies.
    """
    LINEARITY = auto()
    NORMALITY = auto()
    HOMOSCEDASTICITY = auto()
    NO_MULTICOLLINEARITY = auto()
    NO_OUTLIERS = auto()


@dataclass
class FittedTransformer:
    """
    A wrapper for sklearn transformers that tracks fitting provenance.

    This ensures proper train/test separation by recording which
    snapshot the transformer was fit on.

    Attributes:
        name: Human-readable name of the transformation.
        sklearn_instance: The fitted sklearn transformer object.
        fit_on: Identifier of the snapshot used for fitting.
        description: Detailed description of the transformation.
    """
    name: str
    sklearn_instance: Any
    fit_on: str
    description: str = ""

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted transformation to data.

        Args:
            X: DataFrame to transform.

        Returns:
            Transformed DataFrame with same column names.
        """
        transformed = self.sklearn_instance.transform(X)
        return pd.DataFrame(transformed, columns=X.columns, index=X.index)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse the transformation if possible.

        Args:
            X: DataFrame to inverse transform.

        Returns:
            Inverse transformed DataFrame.
        """
        if hasattr(self.sklearn_instance, 'inverse_transform'):
            inversed = self.sklearn_instance.inverse_transform(X)
            return pd.DataFrame(inversed, columns=X.columns, index=X.index)
        raise NotImplementedError(f"{self.name} does not support inverse transform")


@dataclass
class AssumptionResult:
    """
    Result of a statistical assumption test.

    Contains all information needed to display the test result,
    including statistical values, interpretation, and actionable message.

    Attributes:
        trait: The assumption being tested.
        status: Test outcome ('Pass', 'Fail', 'Warning', 'Error').
        test_name: Name of the statistical test used.
        statistic: Test statistic value.
        p_value: P-value if applicable.
        threshold: Decision threshold used.
        severity: Impact level if assumption is violated.
        snapshot_id: ID of the snapshot tested.
        message: Human-readable interpretation.
        details: Additional test-specific details.
    """
    trait: AssumptionTrait
    status: Literal['Pass', 'Fail', 'Warning', 'Error']
    test_name: str
    statistic: float
    p_value: Optional[float]
    threshold: float
    severity: Literal['Critical', 'Moderate', 'Low']
    snapshot_id: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DataSnapshot:
    """
    Immutable snapshot of dataset state at a point in the analysis pipeline.

    DataSnapshots form a directed acyclic graph (DAG) representing the
    transformation history. Each snapshot contains the complete state
    needed to reproduce the analysis.

    Attributes:
        X_train: Training features DataFrame.
        X_test: Test features DataFrame.
        y_train: Training target Series.
        y_test: Test target Series.
        transformation_chain: Ordered tuple of applied transformations.
        traits: Cached set of passing assumption traits.
        snapshot_id: Unique identifier for this snapshot.
        parent_id: ID of the parent snapshot (None for root).
        metadata: Additional information (description, timestamp, etc.).
        feature_names: Original feature names before transformations.
        target_name: Name of the target variable.
    """
    X_train: Any  # pd.DataFrame (frozen=True requires hashable)
    X_test: Any   # pd.DataFrame
    y_train: Any  # pd.Series
    y_test: Any   # pd.Series
    transformation_chain: Tuple[FittedTransformer, ...]
    traits: FrozenSet[AssumptionTrait]
    snapshot_id: str
    parent_id: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    feature_names: Tuple[str, ...] = field(default_factory=tuple)
    target_name: str = "target"

    def __hash__(self):
        return hash(self.snapshot_id)

    @classmethod
    def create_initial(
        cls,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        target_name: str = "target",
        description: str = "Initial data split"
    ) -> DataSnapshot:
        """
        Factory method to create the root snapshot from raw data.

        Args:
            X_train: Training features.
            X_test: Test features.
            y_train: Training target.
            y_test: Test target.
            target_name: Name of target variable.
            description: Description of this snapshot.

        Returns:
            New DataSnapshot representing the initial state.
        """
        snapshot_id = str(uuid.uuid4())[:8]
        return cls(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            transformation_chain=tuple(),
            traits=frozenset(),
            snapshot_id=snapshot_id,
            parent_id=None,
            metadata={"description": description, "created": pd.Timestamp.now()},
            feature_names=tuple(X_train.columns),
            target_name=target_name,
        )

    def derive(
        self,
        transformer: FittedTransformer,
        X_train_new: pd.DataFrame,
        X_test_new: pd.DataFrame,
        y_train_new: Optional[pd.Series] = None,
        y_test_new: Optional[pd.Series] = None,
        description: str = ""
    ) -> DataSnapshot:
        """
        Create a new snapshot by applying a transformation.

        Maintains the immutable transformation chain and clears
        cached traits (forcing recomputation on the new state).

        Args:
            transformer: The fitted transformer to add to chain.
            X_train_new: Transformed training features.
            X_test_new: Transformed test features.
            y_train_new: New training target (if transformed).
            y_test_new: New test target (if transformed).
            description: Description of the transformation.

        Returns:
            New DataSnapshot with updated state.
        """
        new_id = str(uuid.uuid4())[:8]
        return DataSnapshot(
            X_train=X_train_new,
            X_test=X_test_new,
            y_train=y_train_new if y_train_new is not None else self.y_train,
            y_test=y_test_new if y_test_new is not None else self.y_test,
            transformation_chain=self.transformation_chain + (transformer,),
            traits=frozenset(),  # Clear cached traits
            snapshot_id=new_id,
            parent_id=self.snapshot_id,
            metadata={"description": description, "created": pd.Timestamp.now()},
            feature_names=tuple(X_train_new.columns),
            target_name=self.target_name,
        )

    def get_feature_df(self) -> pd.DataFrame:
        """Get training features as DataFrame."""
        return pd.DataFrame(self.X_train)

    def get_n_samples(self) -> int:
        """Get number of training samples."""
        return len(self.X_train)

    def get_n_features(self) -> int:
        """Get number of features."""
        return self.X_train.shape[1]


# =============================================================================
# LAYER 2: MODEL REGISTRY & TYPE SYSTEM
# =============================================================================


MODEL_REGISTRY: Dict[str, Dict[str, set]] = {
    'OLS': {
        'required': {
            AssumptionTrait.LINEARITY,
            AssumptionTrait.HOMOSCEDASTICITY,
            AssumptionTrait.NO_MULTICOLLINEARITY
        },
        'recommended': {
            AssumptionTrait.NORMALITY,
            AssumptionTrait.NO_OUTLIERS
        },
        'description': (
            "Ordinary Least Squares regression. Requires linearity, "
            "homoscedasticity, and no multicollinearity. Normality is "
            "needed for valid inference."
        ),
        'sklearn_class': LinearRegression
    },
    'Ridge': {
        'required': {
            AssumptionTrait.LINEARITY,
            AssumptionTrait.HOMOSCEDASTICITY
        },
        'recommended': {
            AssumptionTrait.NORMALITY
        },
        'description': (
            "L2-regularized linear regression. More robust to "
            "multicollinearity than OLS due to shrinkage penalty."
        ),
        'sklearn_class': Ridge
    },
    'Lasso': {
        'required': {
            AssumptionTrait.LINEARITY,
            AssumptionTrait.HOMOSCEDASTICITY
        },
        'recommended': {
            AssumptionTrait.NORMALITY
        },
        'description': (
            "L1-regularized linear regression. Performs feature selection "
            "by shrinking coefficients to exactly zero."
        ),
        'sklearn_class': Lasso
    },
    'ElasticNet': {
        'required': {
            AssumptionTrait.LINEARITY
        },
        'recommended': {
            AssumptionTrait.HOMOSCEDASTICITY,
            AssumptionTrait.NORMALITY
        },
        'description': (
            "Combined L1 and L2 regularization. Balances feature selection "
            "with coefficient shrinkage."
        ),
        'sklearn_class': None  # Would need sklearn.linear_model.ElasticNet
    }
}


def get_model_compatibility(
    model_name: str,
    passing_traits: FrozenSet[AssumptionTrait]
) -> Dict[str, Any]:
    """
    Evaluate model compatibility given passing assumption traits.

    Args:
        model_name: Name of the model to check.
        passing_traits: Set of traits that pass diagnostics.

    Returns:
        Dictionary with compatibility status and details.
    """
    if model_name not in MODEL_REGISTRY:
        return {
            'compatible': False,
            'status': 'Unknown Model',
            'missing_required': set(),
            'missing_recommended': set()
        }

    model_spec = MODEL_REGISTRY[model_name]
    required = model_spec['required']
    recommended = model_spec['recommended']

    missing_required = required - passing_traits
    missing_recommended = recommended - passing_traits

    if not missing_required:
        status = 'Compatible' if not missing_recommended else 'Compatible (with warnings)'
    else:
        status = f'Incompatible ({len(missing_required)} Critical Errors)'

    return {
        'compatible': len(missing_required) == 0,
        'status': status,
        'missing_required': missing_required,
        'missing_recommended': missing_recommended,
        'n_critical': len(missing_required),
        'n_warnings': len(missing_recommended)
    }


# =============================================================================
# LAYER 3: DIAGNOSTIC ENGINE
# =============================================================================


class DiagnosticEngine:
    """
    Engine for running and caching statistical diagnostic tests.

    The DiagnosticEngine maintains a cache of test results keyed by
    (snapshot_id, trait) to avoid redundant computation. It provides
    individual trait tests and a full diagnostic suite.
    """

    _cache: Dict[Tuple[str, AssumptionTrait], AssumptionResult] = {}

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the entire diagnostic cache."""
        cls._cache.clear()

    @classmethod
    def clear_snapshot_cache(cls, snapshot_id: str) -> None:
        """Clear cache entries for a specific snapshot."""
        keys_to_remove = [k for k in cls._cache if k[0] == snapshot_id]
        for key in keys_to_remove:
            del cls._cache[key]

    @classmethod
    def check_trait(
        cls,
        snapshot: DataSnapshot,
        trait: AssumptionTrait
    ) -> AssumptionResult:
        """
        Check a specific assumption trait, using cache if available.

        Args:
            snapshot: The data snapshot to test.
            trait: The assumption trait to check.

        Returns:
            AssumptionResult with test outcome and details.
        """
        cache_key = (snapshot.snapshot_id, trait)

        if cache_key in cls._cache:
            return cls._cache[cache_key]

        # Dispatch to specific test
        test_methods = {
            AssumptionTrait.LINEARITY: cls._test_linearity,
            AssumptionTrait.NORMALITY: cls._test_normality,
            AssumptionTrait.HOMOSCEDASTICITY: cls._test_homoscedasticity,
            AssumptionTrait.NO_MULTICOLLINEARITY: cls._test_multicollinearity,
            AssumptionTrait.NO_OUTLIERS: cls._test_outliers,
        }

        test_method = test_methods.get(trait)
        if test_method is None:
            result = AssumptionResult(
                trait=trait,
                status='Error',
                test_name='Unknown',
                statistic=0.0,
                p_value=None,
                threshold=0.0,
                severity='Critical',
                snapshot_id=snapshot.snapshot_id,
                message=f"No test implemented for {trait.name}"
            )
        else:
            result = test_method(snapshot)

        cls._cache[cache_key] = result
        return result

    @classmethod
    def _test_linearity(cls, snapshot: DataSnapshot) -> AssumptionResult:
        """
        Test linearity assumption via Pearson correlation.

        Computes correlation between each feature and the target.
        Flags if any |r| < 0.1 (weak linear relationship).

        Args:
            snapshot: Data snapshot to test.

        Returns:
            AssumptionResult for linearity test.
        """
        try:
            X_train = pd.DataFrame(snapshot.X_train)
            y_train = pd.Series(snapshot.y_train)

            correlations = {}
            weak_correlations = []

            for col in X_train.columns:
                corr = np.corrcoef(X_train[col].values, y_train.values)[0, 1]
                correlations[col] = corr
                if abs(corr) < 0.1:
                    weak_correlations.append((col, corr))

            min_abs_corr = min(abs(c) for c in correlations.values())
            avg_abs_corr = np.mean([abs(c) for c in correlations.values()])

            if len(weak_correlations) == 0:
                status = 'Pass'
                message = f"All features have |r| ≥ 0.1 with target (avg |r|={avg_abs_corr:.3f})"
            elif len(weak_correlations) <= len(X_train.columns) * 0.2:
                status = 'Warning'
                weak_names = [w[0] for w in weak_correlations]
                message = f"Weak linear relationship for: {', '.join(weak_names)}"
            else:
                status = 'Fail'
                message = f"{len(weak_correlations)} features have weak correlation (|r| < 0.1)"

            return AssumptionResult(
                trait=AssumptionTrait.LINEARITY,
                status=status,
                test_name="Pearson Correlation Analysis",
                statistic=avg_abs_corr,
                p_value=None,
                threshold=0.1,
                severity='Moderate',
                snapshot_id=snapshot.snapshot_id,
                message=message,
                details={
                    'correlations': correlations,
                    'weak_correlations': weak_correlations,
                    'min_abs_correlation': min_abs_corr
                }
            )

        except Exception as e:
            return AssumptionResult(
                trait=AssumptionTrait.LINEARITY,
                status='Error',
                test_name="Pearson Correlation Analysis",
                statistic=0.0,
                p_value=None,
                threshold=0.1,
                severity='Moderate',
                snapshot_id=snapshot.snapshot_id,
                message=f"Test failed: {str(e)}"
            )

    @classmethod
    def _test_normality(cls, snapshot: DataSnapshot) -> AssumptionResult:
        """
        Test normality of residuals using Shapiro-Wilk or Anderson-Darling.

        Uses Anderson-Darling if n > 5000 (Shapiro-Wilk's limit),
        otherwise uses Shapiro-Wilk. Tests residuals from OLS fit,
        not raw features.

        Args:
            snapshot: Data snapshot to test.

        Returns:
            AssumptionResult for normality test.
        """
        try:
            X_train = pd.DataFrame(snapshot.X_train)
            y_train = pd.Series(snapshot.y_train)

            # Fit OLS to get residuals
            model = LinearRegression()
            model.fit(X_train, y_train)
            residuals = y_train - model.predict(X_train)

            n = len(residuals)

            if n > 5000:
                # Use Anderson-Darling for large samples
                result = anderson(residuals, dist='norm')
                # Use 5% significance level (index 2)
                statistic = result.statistic
                critical_value = result.critical_values[2]
                threshold = critical_value

                if statistic < critical_value:
                    status = 'Pass'
                    message = f"Anderson-Darling statistic ({statistic:.3f}) < critical value ({critical_value:.3f})"
                else:
                    status = 'Fail'
                    message = f"Residuals non-normal: A-D stat ({statistic:.3f}) > critical ({critical_value:.3f})"

                return AssumptionResult(
                    trait=AssumptionTrait.NORMALITY,
                    status=status,
                    test_name="Anderson-Darling Test",
                    statistic=statistic,
                    p_value=None,
                    threshold=threshold,
                    severity='Low',
                    snapshot_id=snapshot.snapshot_id,
                    message=message,
                    details={
                        'critical_values': dict(zip(
                            ['15%', '10%', '5%', '2.5%', '1%'],
                            result.critical_values
                        )),
                        'n_samples': n
                    }
                )
            else:
                # Use Shapiro-Wilk for smaller samples
                # Shapiro-Wilk requires n <= 5000
                sample_size = min(n, 5000)
                if sample_size < n:
                    residuals_sample = np.random.choice(residuals, sample_size, replace=False)
                else:
                    residuals_sample = residuals

                statistic, p_value = shapiro(residuals_sample)
                threshold = 0.05

                if p_value >= threshold:
                    status = 'Pass'
                    message = f"Residuals appear normal (Shapiro-Wilk p={p_value:.4f} ≥ 0.05)"
                else:
                    status = 'Fail'
                    message = f"Residuals non-normal (Shapiro-Wilk p={p_value:.4f} < 0.05)"

                return AssumptionResult(
                    trait=AssumptionTrait.NORMALITY,
                    status=status,
                    test_name="Shapiro-Wilk Test",
                    statistic=statistic,
                    p_value=p_value,
                    threshold=threshold,
                    severity='Low',
                    snapshot_id=snapshot.snapshot_id,
                    message=message,
                    details={'n_samples': sample_size}
                )

        except Exception as e:
            return AssumptionResult(
                trait=AssumptionTrait.NORMALITY,
                status='Error',
                test_name="Normality Test",
                statistic=0.0,
                p_value=None,
                threshold=0.05,
                severity='Low',
                snapshot_id=snapshot.snapshot_id,
                message=f"Test failed: {str(e)}"
            )

    @classmethod
    def _test_homoscedasticity(cls, snapshot: DataSnapshot) -> AssumptionResult:
        """
        Test homoscedasticity using Breusch-Pagan test.

        Fits OLS model and tests if residual variance is constant
        across fitted values. p < 0.05 indicates heteroscedasticity.

        Args:
            snapshot: Data snapshot to test.

        Returns:
            AssumptionResult for homoscedasticity test.
        """
        try:
            X_train = pd.DataFrame(snapshot.X_train)
            y_train = pd.Series(snapshot.y_train).values

            # Fit OLS to get residuals
            model = LinearRegression()
            model.fit(X_train, y_train)
            residuals = y_train - model.predict(X_train)

            # Add constant for Breusch-Pagan test
            X_with_const = np.column_stack([np.ones(len(X_train)), X_train.values])

            # Breusch-Pagan test
            bp_stat, bp_pvalue, f_stat, f_pvalue = het_breuschpagan(
                residuals, X_with_const
            )

            threshold = 0.05

            if bp_pvalue >= threshold:
                status = 'Pass'
                message = f"Homoscedastic: Breusch-Pagan p={bp_pvalue:.4f} ≥ 0.05"
            else:
                status = 'Fail'
                message = f"Heteroscedastic: Breusch-Pagan p={bp_pvalue:.4f} < 0.05"

            return AssumptionResult(
                trait=AssumptionTrait.HOMOSCEDASTICITY,
                status=status,
                test_name="Breusch-Pagan Test",
                statistic=bp_stat,
                p_value=bp_pvalue,
                threshold=threshold,
                severity='Critical',
                snapshot_id=snapshot.snapshot_id,
                message=message,
                details={
                    'f_statistic': f_stat,
                    'f_pvalue': f_pvalue
                }
            )

        except (LinAlgError, ValueError) as e:
            return AssumptionResult(
                trait=AssumptionTrait.HOMOSCEDASTICITY,
                status='Error',
                test_name="Breusch-Pagan Test",
                statistic=0.0,
                p_value=None,
                threshold=0.05,
                severity='Critical',
                snapshot_id=snapshot.snapshot_id,
                message=f"Test failed: {str(e)}"
            )

    @classmethod
    def _test_multicollinearity(cls, snapshot: DataSnapshot) -> AssumptionResult:
        """
        Test for multicollinearity using VIF and Condition Number.

        Computes Variance Inflation Factor for each feature.
        FAIL if any VIF > 10 OR Condition Number > 30.

        Args:
            snapshot: Data snapshot to test.

        Returns:
            AssumptionResult for multicollinearity test.
        """
        try:
            X_train = pd.DataFrame(snapshot.X_train)

            if X_train.shape[1] < 2:
                return AssumptionResult(
                    trait=AssumptionTrait.NO_MULTICOLLINEARITY,
                    status='Pass',
                    test_name="VIF Analysis",
                    statistic=1.0,
                    p_value=None,
                    threshold=10.0,
                    severity='Critical',
                    snapshot_id=snapshot.snapshot_id,
                    message="Single feature - multicollinearity not applicable",
                    details={'vif_scores': {}}
                )

            # Calculate VIF for each feature
            vif_data = {}
            X_values = X_train.values

            for i, col in enumerate(X_train.columns):
                try:
                    vif = variance_inflation_factor(X_values, i)
                    vif_data[col] = vif
                except (LinAlgError, ZeroDivisionError):
                    vif_data[col] = np.inf

            # Calculate condition number
            X_centered = X_train - X_train.mean()
            X_std = X_centered.std()
            X_std[X_std == 0] = 1  # Avoid division by zero
            X_standardized = X_centered / X_std

            try:
                eigenvalues = np.linalg.eigvals(X_standardized.T @ X_standardized)
                eigenvalues = np.real(eigenvalues)
                eigenvalues = eigenvalues[eigenvalues > 0]
                condition_number = np.sqrt(np.max(eigenvalues) / np.min(eigenvalues))
            except (LinAlgError, ValueError):
                condition_number = np.inf

            max_vif = max(vif_data.values()) if vif_data else 0
            high_vif_features = [(k, v) for k, v in vif_data.items() if v > 10]

            vif_threshold = 10.0
            cn_threshold = 30.0

            if max_vif <= vif_threshold and condition_number <= cn_threshold:
                status = 'Pass'
                message = f"No multicollinearity: max VIF={max_vif:.2f}, CN={condition_number:.2f}"
            elif max_vif > vif_threshold:
                status = 'Fail'
                worst_feature = max(vif_data.items(), key=lambda x: x[1])
                message = f"Multicollinearity detected: '{worst_feature[0]}' has VIF={worst_feature[1]:.2f} > 10"
            else:
                status = 'Fail'
                message = f"High condition number ({condition_number:.2f} > 30) indicates multicollinearity"

            return AssumptionResult(
                trait=AssumptionTrait.NO_MULTICOLLINEARITY,
                status=status,
                test_name="VIF & Condition Number Analysis",
                statistic=max_vif,
                p_value=None,
                threshold=vif_threshold,
                severity='Critical',
                snapshot_id=snapshot.snapshot_id,
                message=message,
                details={
                    'vif_scores': vif_data,
                    'condition_number': condition_number,
                    'high_vif_features': high_vif_features
                }
            )

        except Exception as e:
            return AssumptionResult(
                trait=AssumptionTrait.NO_MULTICOLLINEARITY,
                status='Error',
                test_name="VIF Analysis",
                statistic=0.0,
                p_value=None,
                threshold=10.0,
                severity='Critical',
                snapshot_id=snapshot.snapshot_id,
                message=f"Test failed: {str(e)}"
            )

    @classmethod
    def _test_outliers(cls, snapshot: DataSnapshot) -> AssumptionResult:
        """
        Test for influential outliers using Cook's Distance.

        FAIL if any point has Cook's D > 4/n.

        Args:
            snapshot: Data snapshot to test.

        Returns:
            AssumptionResult for outlier test.
        """
        try:
            X_train = pd.DataFrame(snapshot.X_train)
            y_train = pd.Series(snapshot.y_train).values

            n = len(y_train)
            p = X_train.shape[1]

            # Fit OLS
            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_train)
            residuals = y_train - predictions

            # Calculate leverage (hat values)
            X_with_const = np.column_stack([np.ones(n), X_train.values])
            try:
                H = X_with_const @ np.linalg.inv(X_with_const.T @ X_with_const) @ X_with_const.T
                leverage = np.diag(H)
            except LinAlgError:
                # Use pseudo-inverse if singular
                H = X_with_const @ np.linalg.pinv(X_with_const.T @ X_with_const) @ X_with_const.T
                leverage = np.diag(H)

            # Calculate MSE
            mse = np.sum(residuals ** 2) / (n - p - 1)

            # Calculate Cook's Distance
            cooks_d = (residuals ** 2 / ((p + 1) * mse)) * (leverage / (1 - leverage) ** 2)

            threshold = 4 / n
            influential_points = np.where(cooks_d > threshold)[0]
            max_cooks_d = np.max(cooks_d)

            if len(influential_points) == 0:
                status = 'Pass'
                message = f"No influential outliers (max Cook's D={max_cooks_d:.4f} < {threshold:.4f})"
            elif len(influential_points) <= n * 0.01:  # Less than 1%
                status = 'Warning'
                message = f"{len(influential_points)} potentially influential points (Cook's D > {threshold:.4f})"
            else:
                status = 'Fail'
                message = f"{len(influential_points)} influential outliers detected (Cook's D > {threshold:.4f})"

            return AssumptionResult(
                trait=AssumptionTrait.NO_OUTLIERS,
                status=status,
                test_name="Cook's Distance Analysis",
                statistic=max_cooks_d,
                p_value=None,
                threshold=threshold,
                severity='Moderate',
                snapshot_id=snapshot.snapshot_id,
                message=message,
                details={
                    'cooks_distance': cooks_d,
                    'influential_indices': influential_points.tolist(),
                    'n_influential': len(influential_points),
                    'leverage': leverage
                }
            )

        except Exception as e:
            return AssumptionResult(
                trait=AssumptionTrait.NO_OUTLIERS,
                status='Error',
                test_name="Cook's Distance Analysis",
                statistic=0.0,
                p_value=None,
                threshold=0.0,
                severity='Moderate',
                snapshot_id=snapshot.snapshot_id,
                message=f"Test failed: {str(e)}"
            )

    @classmethod
    def run_full_diagnostics(
        cls,
        snapshot: DataSnapshot,
        model_name: str
    ) -> List[AssumptionResult]:
        """
        Run complete diagnostic suite for a model.

        Checks all required and recommended traits for the specified model,
        using cached results when available.

        Args:
            snapshot: Data snapshot to diagnose.
            model_name: Name of the target model.

        Returns:
            List of AssumptionResult for all tested traits.
        """
        if model_name not in MODEL_REGISTRY:
            return []

        model_spec = MODEL_REGISTRY[model_name]
        traits_to_check = model_spec['required'] | model_spec['recommended']

        results = []
        for trait in traits_to_check:
            result = cls.check_trait(snapshot, trait)
            results.append(result)

        return results


# =============================================================================
# LAYER 4: REMEDIATION SYSTEM
# =============================================================================


@dataclass
class Remediation:
    """
    A remediation action that can fix assumption violations.

    Attributes:
        name: Human-readable name.
        description: Detailed description of the remediation.
        targets_trait: The assumption this remediation addresses.
        action: Function that applies the remediation to a snapshot.
        side_effects: List of potential negative consequences.
    """
    name: str
    description: str
    targets_trait: AssumptionTrait
    action: Callable[[DataSnapshot], DataSnapshot]
    side_effects: List[str]


def apply_log_transform_dv(snapshot: DataSnapshot) -> DataSnapshot:
    """
    Apply log transformation to the dependent variable.

    Handles negative and zero values by using log1p(x - min + 1).
    """
    y_train = pd.Series(snapshot.y_train)
    y_test = pd.Series(snapshot.y_test)

    # Handle negative/zero values
    min_val = min(y_train.min(), y_test.min())
    offset = 0 if min_val > 0 else abs(min_val) + 1

    y_train_log = np.log1p(y_train + offset)
    y_test_log = np.log1p(y_test + offset)

    # Create a pseudo-transformer for tracking
    @dataclass
    class LogTransformDV:
        offset: float
        def transform(self, y):
            return np.log1p(y + self.offset)
        def inverse_transform(self, y):
            return np.expm1(y) - self.offset

    transformer = FittedTransformer(
        name="Log Transform (DV)",
        sklearn_instance=LogTransformDV(offset=offset),
        fit_on=snapshot.snapshot_id,
        description=f"Applied log1p transformation to DV with offset={offset:.2f}"
    )

    return snapshot.derive(
        transformer=transformer,
        X_train_new=pd.DataFrame(snapshot.X_train),
        X_test_new=pd.DataFrame(snapshot.X_test),
        y_train_new=y_train_log,
        y_test_new=y_test_log,
        description="Log-transformed dependent variable"
    )


def apply_robust_scaling(snapshot: DataSnapshot) -> DataSnapshot:
    """Apply RobustScaler to features (resistant to outliers)."""
    X_train = pd.DataFrame(snapshot.X_train)
    X_test = pd.DataFrame(snapshot.X_test)

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    transformer = FittedTransformer(
        name="Robust Scaling",
        sklearn_instance=scaler,
        fit_on=snapshot.snapshot_id,
        description="Applied RobustScaler (median-based, IQR normalized)"
    )

    return snapshot.derive(
        transformer=transformer,
        X_train_new=pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index),
        X_test_new=pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index),
        description="Applied robust scaling to features"
    )


def apply_standard_scaling(snapshot: DataSnapshot) -> DataSnapshot:
    """Apply StandardScaler to features (z-score normalization)."""
    X_train = pd.DataFrame(snapshot.X_train)
    X_test = pd.DataFrame(snapshot.X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    transformer = FittedTransformer(
        name="Standard Scaling",
        sklearn_instance=scaler,
        fit_on=snapshot.snapshot_id,
        description="Applied StandardScaler (z-score normalization)"
    )

    return snapshot.derive(
        transformer=transformer,
        X_train_new=pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index),
        X_test_new=pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index),
        description="Applied standard scaling to features"
    )


def drop_max_vif_feature(snapshot: DataSnapshot) -> DataSnapshot:
    """Drop the feature with highest VIF."""
    X_train = pd.DataFrame(snapshot.X_train)
    X_test = pd.DataFrame(snapshot.X_test)

    if X_train.shape[1] <= 1:
        return snapshot

    # Find max VIF feature
    vif_scores = {}
    for i, col in enumerate(X_train.columns):
        try:
            vif = variance_inflation_factor(X_train.values, i)
            vif_scores[col] = vif
        except:
            vif_scores[col] = np.inf

    max_vif_col = max(vif_scores.items(), key=lambda x: x[1])[0]

    X_train_new = X_train.drop(columns=[max_vif_col])
    X_test_new = X_test.drop(columns=[max_vif_col])

    # Create a pseudo-transformer for tracking
    @dataclass
    class FeatureDropper:
        dropped_column: str
        def transform(self, X):
            return X.drop(columns=[self.dropped_column], errors='ignore')

    transformer = FittedTransformer(
        name=f"Drop Feature '{max_vif_col}'",
        sklearn_instance=FeatureDropper(dropped_column=max_vif_col),
        fit_on=snapshot.snapshot_id,
        description=f"Dropped feature '{max_vif_col}' (VIF={vif_scores[max_vif_col]:.2f})"
    )

    return snapshot.derive(
        transformer=transformer,
        X_train_new=X_train_new,
        X_test_new=X_test_new,
        description=f"Dropped high-VIF feature: {max_vif_col}"
    )


def apply_pca(snapshot: DataSnapshot, variance_threshold: float = 0.95) -> DataSnapshot:
    """Apply PCA to reduce multicollinearity."""
    X_train = pd.DataFrame(snapshot.X_train)
    X_test = pd.DataFrame(snapshot.X_test)

    # Standardize first
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    # Apply PCA
    pca = PCA(n_components=variance_threshold)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    n_components = X_train_pca.shape[1]
    component_names = [f"PC{i+1}" for i in range(n_components)]

    # Create combined transformer
    @dataclass
    class ScalerPCAPipeline:
        scaler: StandardScaler
        pca: PCA
        def transform(self, X):
            return self.pca.transform(self.scaler.transform(X))

    transformer = FittedTransformer(
        name=f"PCA ({n_components} components)",
        sklearn_instance=ScalerPCAPipeline(scaler=scaler, pca=pca),
        fit_on=snapshot.snapshot_id,
        description=f"Applied PCA retaining {variance_threshold*100:.0f}% variance ({n_components} components)"
    )

    return snapshot.derive(
        transformer=transformer,
        X_train_new=pd.DataFrame(X_train_pca, columns=component_names, index=X_train.index),
        X_test_new=pd.DataFrame(X_test_pca, columns=component_names, index=X_test.index),
        description=f"Applied PCA ({n_components} components, {variance_threshold*100:.0f}% variance)"
    )


def apply_power_transform(snapshot: DataSnapshot) -> DataSnapshot:
    """Apply Yeo-Johnson power transformation to features."""
    X_train = pd.DataFrame(snapshot.X_train)
    X_test = pd.DataFrame(snapshot.X_test)

    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    X_train_transformed = pt.fit_transform(X_train)
    X_test_transformed = pt.transform(X_test)

    transformer = FittedTransformer(
        name="Power Transform (Yeo-Johnson)",
        sklearn_instance=pt,
        fit_on=snapshot.snapshot_id,
        description="Applied Yeo-Johnson power transformation"
    )

    return snapshot.derive(
        transformer=transformer,
        X_train_new=pd.DataFrame(X_train_transformed, columns=X_train.columns, index=X_train.index),
        X_test_new=pd.DataFrame(X_test_transformed, columns=X_test.columns, index=X_test.index),
        description="Applied Yeo-Johnson power transformation"
    )


def remove_outliers(snapshot: DataSnapshot) -> DataSnapshot:
    """Remove influential outliers based on Cook's Distance."""
    X_train = pd.DataFrame(snapshot.X_train)
    y_train = pd.Series(snapshot.y_train)

    n = len(y_train)
    p = X_train.shape[1]

    # Fit OLS and calculate Cook's Distance
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_train)
    residuals = y_train - predictions

    # Calculate leverage
    X_with_const = np.column_stack([np.ones(n), X_train.values])
    try:
        H = X_with_const @ np.linalg.inv(X_with_const.T @ X_with_const) @ X_with_const.T
    except LinAlgError:
        H = X_with_const @ np.linalg.pinv(X_with_const.T @ X_with_const) @ X_with_const.T
    leverage = np.diag(H)

    # Calculate Cook's Distance
    mse = np.sum(residuals ** 2) / (n - p - 1)
    cooks_d = (residuals ** 2 / ((p + 1) * mse)) * (leverage / (1 - leverage) ** 2)

    threshold = 4 / n
    keep_mask = cooks_d <= threshold

    X_train_clean = X_train[keep_mask]
    y_train_clean = y_train[keep_mask]

    n_removed = n - len(X_train_clean)

    @dataclass
    class OutlierRemover:
        removed_indices: List[int]
        threshold: float

    transformer = FittedTransformer(
        name=f"Remove Outliers ({n_removed} points)",
        sklearn_instance=OutlierRemover(
            removed_indices=list(np.where(~keep_mask)[0]),
            threshold=threshold
        ),
        fit_on=snapshot.snapshot_id,
        description=f"Removed {n_removed} outliers (Cook's D > {threshold:.4f})"
    )

    return snapshot.derive(
        transformer=transformer,
        X_train_new=X_train_clean.reset_index(drop=True),
        X_test_new=pd.DataFrame(snapshot.X_test),
        y_train_new=y_train_clean.reset_index(drop=True),
        y_test_new=pd.Series(snapshot.y_test),
        description=f"Removed {n_removed} outliers based on Cook's Distance"
    )


def flag_for_wls(snapshot: DataSnapshot) -> DataSnapshot:
    """Flag snapshot for Weighted Least Squares (metadata update only)."""
    new_metadata = dict(snapshot.metadata)
    new_metadata['use_wls'] = True
    new_metadata['wls_reason'] = 'Heteroscedasticity detected'

    @dataclass
    class WLSFlag:
        pass

    transformer = FittedTransformer(
        name="Flag for WLS",
        sklearn_instance=WLSFlag(),
        fit_on=snapshot.snapshot_id,
        description="Flagged for Weighted Least Squares fitting"
    )

    return DataSnapshot(
        X_train=snapshot.X_train,
        X_test=snapshot.X_test,
        y_train=snapshot.y_train,
        y_test=snapshot.y_test,
        transformation_chain=snapshot.transformation_chain + (transformer,),
        traits=frozenset(),
        snapshot_id=str(uuid.uuid4())[:8],
        parent_id=snapshot.snapshot_id,
        metadata=new_metadata,
        feature_names=snapshot.feature_names,
        target_name=snapshot.target_name
    )


# Remediation mappings
REMEDIATION_MAP: Dict[AssumptionTrait, List[Remediation]] = {
    AssumptionTrait.HOMOSCEDASTICITY: [
        Remediation(
            name="Log Transform (DV)",
            description="Apply log transformation to the dependent variable to stabilize variance.",
            targets_trait=AssumptionTrait.HOMOSCEDASTICITY,
            action=apply_log_transform_dv,
            side_effects=["Changes coefficient interpretation to % change", "Requires positive DV"]
        ),
        Remediation(
            name="Weighted Least Squares",
            description="Flag the model to use WLS with variance weights.",
            targets_trait=AssumptionTrait.HOMOSCEDASTICITY,
            action=flag_for_wls,
            side_effects=["Requires residual variance model", "More complex estimation"]
        ),
        Remediation(
            name="Power Transform (Features)",
            description="Apply Yeo-Johnson transformation to normalize features.",
            targets_trait=AssumptionTrait.HOMOSCEDASTICITY,
            action=apply_power_transform,
            side_effects=["Changes feature distributions", "May affect interpretability"]
        )
    ],
    AssumptionTrait.NO_MULTICOLLINEARITY: [
        Remediation(
            name="Drop Highest VIF Feature",
            description="Remove the feature with the highest Variance Inflation Factor.",
            targets_trait=AssumptionTrait.NO_MULTICOLLINEARITY,
            action=drop_max_vif_feature,
            side_effects=["Removes predictor - changes hypothesis", "May need multiple applications"]
        ),
        Remediation(
            name="Principal Component Analysis",
            description="Replace features with principal components (95% variance retained).",
            targets_trait=AssumptionTrait.NO_MULTICOLLINEARITY,
            action=apply_pca,
            side_effects=["Destroys interpretability", "Creates orthogonal features"]
        )
    ],
    AssumptionTrait.NO_OUTLIERS: [
        Remediation(
            name="Remove Influential Points",
            description="Remove observations with Cook's Distance > 4/n.",
            targets_trait=AssumptionTrait.NO_OUTLIERS,
            action=remove_outliers,
            side_effects=["Reduces sample size", "May remove valid extreme values"]
        ),
        Remediation(
            name="Robust Scaling",
            description="Apply median-based scaling resistant to outliers.",
            targets_trait=AssumptionTrait.NO_OUTLIERS,
            action=apply_robust_scaling,
            side_effects=["Changes feature scale", "Outliers remain but less influential"]
        )
    ],
    AssumptionTrait.NORMALITY: [
        Remediation(
            name="Power Transform (DV)",
            description="Apply Yeo-Johnson transformation to dependent variable.",
            targets_trait=AssumptionTrait.NORMALITY,
            action=apply_log_transform_dv,
            side_effects=["Changes coefficient interpretation", "Non-linear transformation"]
        ),
        Remediation(
            name="Power Transform (Features)",
            description="Apply Yeo-Johnson transformation to all features.",
            targets_trait=AssumptionTrait.NORMALITY,
            action=apply_power_transform,
            side_effects=["Changes feature distributions", "May not fix residual normality"]
        )
    ],
    AssumptionTrait.LINEARITY: [
        Remediation(
            name="Power Transform (Features)",
            description="Apply non-linear transformation to features.",
            targets_trait=AssumptionTrait.LINEARITY,
            action=apply_power_transform,
            side_effects=["Changes relationships", "May introduce other issues"]
        )
    ]
}


class RemediationEngine:
    """
    Engine for applying and previewing remediation actions.

    Provides impact analysis before applying changes and
    tracks the effects on all assumption traits.
    """

    @staticmethod
    def get_remediations(trait: AssumptionTrait) -> List[Remediation]:
        """Get available remediations for a trait violation."""
        return REMEDIATION_MAP.get(trait, [])

    @staticmethod
    def preview_impact(
        snapshot: DataSnapshot,
        remediation: Remediation,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Preview the impact of applying a remediation.

        Simulates the remediation and compares diagnostics before/after.

        Args:
            snapshot: Current data snapshot.
            remediation: Remediation to preview.
            model_name: Target model name.

        Returns:
            Dictionary with impact analysis results.
        """
        # Get current diagnostics
        current_results = DiagnosticEngine.run_full_diagnostics(snapshot, model_name)
        current_passing = {r.trait for r in current_results if r.status == 'Pass'}

        # Apply remediation to get simulated snapshot
        try:
            sim_snapshot = remediation.action(snapshot)
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'recommendation': 'Rejected'
            }

        # Run diagnostics on simulated snapshot
        sim_results = DiagnosticEngine.run_full_diagnostics(sim_snapshot, model_name)
        sim_passing = {r.trait for r in sim_results if r.status == 'Pass'}

        # Calculate changes
        fixed = sim_passing - current_passing
        broken = current_passing - sim_passing

        # Calculate R² delta via cross-validation
        try:
            X_train_curr = pd.DataFrame(snapshot.X_train)
            y_train_curr = pd.Series(snapshot.y_train)
            X_train_sim = pd.DataFrame(sim_snapshot.X_train)
            y_train_sim = pd.Series(sim_snapshot.y_train)

            model_class = MODEL_REGISTRY[model_name].get('sklearn_class', LinearRegression)
            if model_class is None:
                model_class = LinearRegression

            curr_model = model_class()
            sim_model = model_class()

            curr_cv = cross_val_score(curr_model, X_train_curr, y_train_curr, cv=3, scoring='r2')
            sim_cv = cross_val_score(sim_model, X_train_sim, y_train_sim, cv=3, scoring='r2')

            r2_delta = np.mean(sim_cv) - np.mean(curr_cv)
        except Exception:
            r2_delta = 0.0

        # Determine recommendation
        if len(broken) > 0:
            recommendation = 'Risky'
        elif remediation.targets_trait in fixed:
            recommendation = 'Apply'
        elif r2_delta < -0.05:
            recommendation = 'Risky'
        else:
            recommendation = 'Apply'

        # Get new violations
        new_violations = [r for r in sim_results if r.status == 'Fail' and r.trait not in {
            res.trait for res in current_results if res.status == 'Fail'
        }]

        return {
            'success': True,
            'fixed': list(fixed),
            'broken': list(broken),
            'r2_delta': r2_delta,
            'r2_before': np.mean(curr_cv) if 'curr_cv' in dir() else None,
            'r2_after': np.mean(sim_cv) if 'sim_cv' in dir() else None,
            'new_violations': new_violations,
            'recommendation': recommendation,
            'sim_snapshot': sim_snapshot,
            'sim_results': sim_results
        }


# =============================================================================
# LAYER 5: EXPLANATION SYSTEM
# =============================================================================


EXPLANATIONS: Dict[AssumptionTrait, Dict[str, str]] = {
    AssumptionTrait.LINEARITY: {
        'title': 'Linearity Assumption',
        'definition': (
            "The relationship between independent variables (X) and the dependent "
            "variable (Y) should be linear. The model assumes Y = β₀ + β₁X₁ + ... + βₚXₚ + ε."
        ),
        'consequences': (
            "• Model predictions will be systematically biased\n"
            "• Coefficients won't accurately represent true relationships\n"
            "• R² may be misleadingly low despite real relationships existing"
        ),
        'references': (
            "• Wooldridge, J.M. (2016). Introductory Econometrics, Ch. 3\n"
            "• James, G. et al. (2013). An Introduction to Statistical Learning, Ch. 3"
        )
    },
    AssumptionTrait.NORMALITY: {
        'title': 'Normality of Residuals',
        'definition': (
            "The residuals (errors) should be normally distributed: ε ~ N(0, σ²). "
            "This assumption is needed for valid hypothesis tests and confidence intervals."
        ),
        'consequences': (
            "• Standard errors may be incorrect\n"
            "• t-tests and F-tests may not be valid\n"
            "• Confidence intervals may have wrong coverage\n"
            "• Note: For large samples (n > 30), CLT often provides approximate validity"
        ),
        'references': (
            "• Greene, W.H. (2018). Econometric Analysis, Ch. 4\n"
            "• Kutner et al. (2005). Applied Linear Statistical Models, Ch. 3"
        )
    },
    AssumptionTrait.HOMOSCEDASTICITY: {
        'title': 'Homoscedasticity (Constant Variance)',
        'definition': (
            "Var(ε|X) should be constant. The spread of residuals should not change "
            "with the level of fitted values or any predictor."
        ),
        'consequences': (
            "• Standard errors are biased (usually underestimated)\n"
            "• Hypothesis tests (t, F) produce invalid p-values\n"
            "• Confidence intervals have incorrect width\n"
            "• OLS estimators remain unbiased but inefficient"
        ),
        'references': (
            "• Wooldridge, J.M. (2016). Introductory Econometrics, Ch. 8\n"
            "• Greene, W.H. (2018). Econometric Analysis, Ch. 9"
        )
    },
    AssumptionTrait.NO_MULTICOLLINEARITY: {
        'title': 'No Perfect Multicollinearity',
        'definition': (
            "Independent variables should not be perfectly correlated with each other. "
            "High (but not perfect) correlation inflates variance of coefficient estimates."
        ),
        'consequences': (
            "• Coefficient estimates become unstable (high variance)\n"
            "• Standard errors are inflated\n"
            "• Difficult to isolate individual effects of predictors\n"
            "• Model may be sensitive to small data changes"
        ),
        'references': (
            "• Kennedy, P. (2008). A Guide to Econometrics, Ch. 11\n"
            "• Kutner et al. (2005). Applied Linear Statistical Models, Ch. 7"
        )
    },
    AssumptionTrait.NO_OUTLIERS: {
        'title': 'No Influential Outliers',
        'definition': (
            "No single observation should have disproportionate influence on the "
            "regression results. Measured by Cook's Distance combining leverage and residual."
        ),
        'consequences': (
            "• Regression line may be pulled toward outliers\n"
            "• Coefficient estimates may not represent the bulk of data\n"
            "• R² can be artificially inflated or deflated\n"
            "• Predictions may be unreliable for typical observations"
        ),
        'references': (
            "• Cook, R.D. (1977). Detection of Influential Observation in Linear Regression\n"
            "• Belsley, D.A. et al. (1980). Regression Diagnostics"
        )
    }
}


# =============================================================================
# LAYER 6: VISUALIZATION FUNCTIONS
# =============================================================================


def plot_residuals_vs_fitted(snapshot: DataSnapshot) -> go.Figure:
    """Create residuals vs fitted values plot."""
    X_train = pd.DataFrame(snapshot.X_train)
    y_train = pd.Series(snapshot.y_train)

    model = LinearRegression()
    model.fit(X_train, y_train)
    fitted = model.predict(X_train)
    residuals = y_train - fitted

    fig = px.scatter(
        x=fitted, y=residuals,
        labels={'x': 'Fitted Values', 'y': 'Residuals'},
        title='Residuals vs Fitted Values'
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(height=400)
    return fig


def plot_qq(snapshot: DataSnapshot) -> go.Figure:
    """Create Q-Q plot for residual normality."""
    X_train = pd.DataFrame(snapshot.X_train)
    y_train = pd.Series(snapshot.y_train)

    model = LinearRegression()
    model.fit(X_train, y_train)
    residuals = y_train - model.predict(X_train)

    residuals_sorted = np.sort(residuals)
    n = len(residuals_sorted)
    theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, n))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=theoretical, y=residuals_sorted,
        mode='markers', name='Residuals'
    ))

    # Add reference line
    min_val = min(theoretical.min(), residuals_sorted.min())
    max_val = max(theoretical.max(), residuals_sorted.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines', name='Normal Line',
        line=dict(color='red', dash='dash')
    ))

    fig.update_layout(
        title='Q-Q Plot (Residual Normality)',
        xaxis_title='Theoretical Quantiles',
        yaxis_title='Sample Quantiles',
        height=400
    )
    return fig


def plot_vif_chart(snapshot: DataSnapshot) -> go.Figure:
    """Create VIF bar chart."""
    X_train = pd.DataFrame(snapshot.X_train)

    if X_train.shape[1] < 2:
        fig = go.Figure()
        fig.add_annotation(text="Need at least 2 features for VIF", showarrow=False)
        return fig

    vif_data = []
    for i, col in enumerate(X_train.columns):
        try:
            vif = variance_inflation_factor(X_train.values, i)
            vif_data.append({'Feature': col, 'VIF': min(vif, 50)})  # Cap for display
        except:
            vif_data.append({'Feature': col, 'VIF': 0})

    df_vif = pd.DataFrame(vif_data)

    colors = ['red' if v > 10 else 'orange' if v > 5 else 'green' for v in df_vif['VIF']]

    fig = px.bar(
        df_vif, x='Feature', y='VIF',
        title='Variance Inflation Factors'
    )
    fig.update_traces(marker_color=colors)
    fig.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="VIF=10 threshold")
    fig.update_layout(height=400)
    return fig


def plot_cooks_distance(snapshot: DataSnapshot) -> go.Figure:
    """Create Cook's Distance plot."""
    X_train = pd.DataFrame(snapshot.X_train)
    y_train = pd.Series(snapshot.y_train).values

    n = len(y_train)
    p = X_train.shape[1]

    model = LinearRegression()
    model.fit(X_train, y_train)
    residuals = y_train - model.predict(X_train)

    X_with_const = np.column_stack([np.ones(n), X_train.values])
    try:
        H = X_with_const @ np.linalg.inv(X_with_const.T @ X_with_const) @ X_with_const.T
    except LinAlgError:
        H = X_with_const @ np.linalg.pinv(X_with_const.T @ X_with_const) @ X_with_const.T
    leverage = np.diag(H)

    mse = np.sum(residuals ** 2) / (n - p - 1)
    cooks_d = (residuals ** 2 / ((p + 1) * mse)) * (leverage / (1 - leverage) ** 2)

    threshold = 4 / n
    colors = ['red' if d > threshold else 'blue' for d in cooks_d]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(n)), y=cooks_d,
        mode='markers',
        marker=dict(color=colors),
        name="Cook's D"
    ))
    fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Threshold: 4/n = {threshold:.4f}")
    fig.update_layout(
        title="Cook's Distance",
        xaxis_title='Observation Index',
        yaxis_title="Cook's Distance",
        height=400
    )
    return fig


def plot_correlation_heatmap(snapshot: DataSnapshot) -> go.Figure:
    """Create correlation heatmap."""
    X_train = pd.DataFrame(snapshot.X_train)
    y_train = pd.Series(snapshot.y_train)

    # Combine X and y for correlation
    df_combined = X_train.copy()
    df_combined[snapshot.target_name] = y_train.values

    corr_matrix = df_combined.corr()

    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title='Correlation Matrix'
    )
    fig.update_layout(height=500)
    return fig


# =============================================================================
# LAYER 7: SAMPLE DATA LOADERS
# =============================================================================


def load_california_housing_data() -> Tuple[pd.DataFrame, pd.Series, str]:
    """Load California Housing dataset."""
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='MedHouseValue')
    return X, y, 'MedHouseValue'


def load_synthetic_regression_data(
    n_samples: int = 500,
    n_features: int = 5,
    noise: float = 10.0
) -> Tuple[pd.DataFrame, pd.Series, str]:
    """Generate synthetic regression data with known properties."""
    np.random.seed(42)

    # Generate correlated features
    mean = np.zeros(n_features)
    cov = np.eye(n_features)
    cov[0, 1] = cov[1, 0] = 0.7  # Introduce some collinearity

    X = np.random.multivariate_normal(mean, cov, n_samples)

    # Generate y with known coefficients
    true_coefs = np.array([3.0, -2.0, 1.5, -1.0, 0.5])[:n_features]
    y = X @ true_coefs + np.random.normal(0, noise, n_samples)

    # Add some heteroscedasticity
    y = y + X[:, 0] * np.random.normal(0, 2, n_samples)

    feature_names = [f'Feature_{i+1}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='Target')

    return X_df, y_series, 'Target'


# =============================================================================
# LAYER 8: REPORT GENERATION
# =============================================================================


def generate_analysis_report(
    snapshots: List[DataSnapshot],
    results_history: Dict[str, List[AssumptionResult]],
    model_name: str,
    final_metrics: Optional[Dict[str, float]] = None
) -> str:
    """
    Generate a comprehensive markdown analysis report.

    Args:
        snapshots: List of all data snapshots.
        results_history: Diagnostic results keyed by snapshot_id.
        model_name: Name of the target model.
        final_metrics: Final model performance metrics.

    Returns:
        Markdown formatted report string.
    """
    report = []
    report.append("# Statistical Regression Analysis Report")
    report.append(f"\n**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n**Target Model:** {model_name}")
    report.append("\n---\n")

    # Snapshot Timeline
    report.append("## Data Transformation Timeline\n")
    for i, snapshot in enumerate(snapshots):
        desc = snapshot.metadata.get('description', 'No description')
        created = snapshot.metadata.get('created', 'Unknown')
        report.append(f"### Snapshot {i+1}: `{snapshot.snapshot_id}`")
        report.append(f"- **Description:** {desc}")
        report.append(f"- **Created:** {created}")
        report.append(f"- **Features:** {snapshot.get_n_features()}")
        report.append(f"- **Training Samples:** {snapshot.get_n_samples()}")

        if snapshot.transformation_chain:
            report.append("\n**Transformations Applied:**")
            for t in snapshot.transformation_chain:
                report.append(f"  - {t.name}: {t.description}")
        report.append("")

    # Diagnostic Results
    report.append("\n## Diagnostic Results\n")

    for snapshot_id, results in results_history.items():
        report.append(f"### Snapshot `{snapshot_id}`\n")
        report.append("| Trait | Status | Test | Statistic | p-value | Message |")
        report.append("|-------|--------|------|-----------|---------|---------|")

        for r in results:
            status_icon = {'Pass': '✅', 'Fail': '❌', 'Warning': '⚠️', 'Error': '🔴'}.get(r.status, '❓')
            p_val = f"{r.p_value:.4f}" if r.p_value is not None else "N/A"
            report.append(f"| {r.trait.name} | {status_icon} {r.status} | {r.test_name} | {r.statistic:.4f} | {p_val} | {r.message[:50]}... |")
        report.append("")

    # Final Metrics
    if final_metrics:
        report.append("\n## Model Performance\n")
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        for metric, value in final_metrics.items():
            report.append(f"| {metric} | {value:.4f} |")

    # Reproducibility
    report.append("\n## Reproducibility\n")
    report.append("```python")
    report.append("# To reproduce this analysis:")
    report.append("import pandas as pd")
    report.append("from sklearn.model_selection import train_test_split")
    report.append("from sklearn.preprocessing import StandardScaler")
    report.append("")
    report.append("# Load your data")
    report.append("# X, y = load_data()")
    report.append("# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)")
    report.append("```")

    return "\n".join(report)


# =============================================================================
# LAYER 9: STREAMLIT UI
# =============================================================================


def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'snapshots' not in st.session_state:
        st.session_state.snapshots = []
    if 'current_snapshot_idx' not in st.session_state:
        st.session_state.current_snapshot_idx = -1
    if 'results_history' not in st.session_state:
        st.session_state.results_history = {}
    if 'model_name' not in st.session_state:
        st.session_state.model_name = 'OLS'
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    if 'show_explanation' not in st.session_state:
        st.session_state.show_explanation = None


def get_current_snapshot() -> Optional[DataSnapshot]:
    """Get the current active snapshot."""
    if st.session_state.current_snapshot_idx >= 0:
        return st.session_state.snapshots[st.session_state.current_snapshot_idx]
    return None


def add_snapshot(snapshot: DataSnapshot):
    """Add a new snapshot and make it current."""
    st.session_state.snapshots.append(snapshot)
    st.session_state.current_snapshot_idx = len(st.session_state.snapshots) - 1
    st.session_state.data_loaded = True


def render_sidebar():
    """Render the sidebar UI components."""
    st.sidebar.title("📊 Statistical IDE")
    st.sidebar.markdown("---")

    # Data Loading Section
    st.sidebar.header("1. Load Data")

    data_source = st.sidebar.radio(
        "Data Source",
        ["Upload CSV/Excel", "California Housing", "Synthetic Data"],
        key="data_source"
    )

    if data_source == "Upload CSV/Excel":
        uploaded_file = st.sidebar.file_uploader(
            "Choose file",
            type=['csv', 'xlsx'],
            key="file_uploader"
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                st.sidebar.success(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

                # Target selection
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                target_col = st.sidebar.selectbox(
                    "Select Target Variable (DV)",
                    numeric_cols,
                    key="target_select"
                )

                feature_cols = st.sidebar.multiselect(
                    "Select Features (IVs)",
                    [c for c in numeric_cols if c != target_col],
                    default=[c for c in numeric_cols if c != target_col][:5],
                    key="feature_select"
                )

                if st.sidebar.button("Create Initial Snapshot", key="create_snapshot"):
                    X = df[feature_cols].dropna()
                    y = df[target_col].loc[X.index]

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )

                    snapshot = DataSnapshot.create_initial(
                        X_train=X_train.reset_index(drop=True),
                        X_test=X_test.reset_index(drop=True),
                        y_train=y_train.reset_index(drop=True),
                        y_test=y_test.reset_index(drop=True),
                        target_name=target_col,
                        description=f"Initial split from {uploaded_file.name}"
                    )
                    add_snapshot(snapshot)
                    DiagnosticEngine.clear_cache()
                    st.rerun()

            except Exception as e:
                st.sidebar.error(f"Error loading file: {e}")

    elif data_source == "California Housing":
        if st.sidebar.button("Load California Housing", key="load_california"):
            X, y, target_name = load_california_housing_data()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            snapshot = DataSnapshot.create_initial(
                X_train=X_train.reset_index(drop=True),
                X_test=X_test.reset_index(drop=True),
                y_train=y_train.reset_index(drop=True),
                y_test=y_test.reset_index(drop=True),
                target_name=target_name,
                description="California Housing Dataset"
            )
            add_snapshot(snapshot)
            DiagnosticEngine.clear_cache()
            st.rerun()

    else:  # Synthetic Data
        n_samples = st.sidebar.slider("Samples", 100, 2000, 500, key="n_samples")
        n_features = st.sidebar.slider("Features", 2, 10, 5, key="n_features")
        noise = st.sidebar.slider("Noise Level", 1.0, 50.0, 10.0, key="noise")

        if st.sidebar.button("Generate Synthetic Data", key="gen_synthetic"):
            X, y, target_name = load_synthetic_regression_data(n_samples, n_features, noise)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            snapshot = DataSnapshot.create_initial(
                X_train=X_train.reset_index(drop=True),
                X_test=X_test.reset_index(drop=True),
                y_train=y_train.reset_index(drop=True),
                y_test=y_test.reset_index(drop=True),
                target_name=target_name,
                description=f"Synthetic data (n={n_samples}, p={n_features}, noise={noise})"
            )
            add_snapshot(snapshot)
            DiagnosticEngine.clear_cache()
            st.rerun()

    st.sidebar.markdown("---")

    # Model Selection
    st.sidebar.header("2. Select Model")
    st.session_state.model_name = st.sidebar.selectbox(
        "Regression Model",
        list(MODEL_REGISTRY.keys()),
        index=list(MODEL_REGISTRY.keys()).index(st.session_state.model_name),
        key="model_select"
    )

    model_spec = MODEL_REGISTRY[st.session_state.model_name]
    with st.sidebar.expander("Model Requirements"):
        st.markdown("**Required:**")
        for trait in model_spec['required']:
            st.markdown(f"- {trait.name}")
        st.markdown("**Recommended:**")
        for trait in model_spec['recommended']:
            st.markdown(f"- {trait.name}")

    st.sidebar.markdown("---")

    # Snapshot History
    st.sidebar.header("3. Snapshot History")

    if st.session_state.snapshots:
        for i, snapshot in enumerate(st.session_state.snapshots):
            is_current = i == st.session_state.current_snapshot_idx
            icon = "🔵" if is_current else "⚪"
            desc = snapshot.metadata.get('description', 'No description')[:25]

            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                st.markdown(f"{icon} **{snapshot.snapshot_id}**  \n{desc}...")
            with col2:
                if not is_current:
                    if st.button("↩️", key=f"revert_{i}"):
                        st.session_state.current_snapshot_idx = i
                        st.rerun()
    else:
        st.sidebar.info("No snapshots yet. Load data to begin.")


def render_compiler_status(snapshot: DataSnapshot, results: List[AssumptionResult]):
    """Render the compiler status header."""
    passing_traits = frozenset(r.trait for r in results if r.status == 'Pass')
    compatibility = get_model_compatibility(st.session_state.model_name, passing_traits)

    if compatibility['compatible']:
        if compatibility['n_warnings'] == 0:
            status_color = "🟢"
            status_text = "COMPATIBLE"
        else:
            status_color = "🟡"
            status_text = f"COMPATIBLE ({compatibility['n_warnings']} Warnings)"
    else:
        status_color = "🔴"
        status_text = f"INCOMPATIBLE ({compatibility['n_critical']} Critical Errors)"

    st.markdown(f"""
    <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="margin: 0; color: white;">
            {status_color} Compiler Status: {status_text}
        </h3>
        <p style="margin: 5px 0 0 0; color: #888;">
            Model: <strong>{st.session_state.model_name}</strong> |
            Snapshot: <strong>{snapshot.snapshot_id}</strong> |
            Features: {snapshot.get_n_features()} |
            Samples: {snapshot.get_n_samples()}
        </p>
    </div>
    """, unsafe_allow_html=True)

    return compatibility


def render_diagnostic_report(
    snapshot: DataSnapshot,
    results: List[AssumptionResult],
    compatibility: Dict
):
    """Render the diagnostic report panel."""
    st.subheader("📋 Diagnostic Report")

    model_spec = MODEL_REGISTRY[st.session_state.model_name]
    required_traits = model_spec['required']
    recommended_traits = model_spec['recommended']

    # Sort results: Critical failures first, then warnings, then passes
    def sort_key(r):
        if r.status == 'Fail' and r.trait in required_traits:
            return 0
        elif r.status == 'Fail':
            return 1
        elif r.status == 'Warning':
            return 2
        else:
            return 3

    sorted_results = sorted(results, key=sort_key)

    for result in sorted_results:
        is_required = result.trait in required_traits

        # Status icon
        if result.status == 'Pass':
            icon = "✅"
            color = "green"
        elif result.status == 'Fail':
            icon = "❌"
            color = "red"
        elif result.status == 'Warning':
            icon = "⚠️"
            color = "orange"
        else:
            icon = "🔴"
            color = "gray"

        requirement_badge = "**[REQUIRED]**" if is_required else "*[Recommended]*"

        with st.expander(f"{icon} {result.trait.name} - {result.status} {requirement_badge}", expanded=(result.status == 'Fail')):
            st.markdown(f"**Test:** {result.test_name}")
            st.markdown(f"**Message:** {result.message}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Statistic", f"{result.statistic:.4f}")
            with col2:
                if result.p_value is not None:
                    st.metric("p-value", f"{result.p_value:.4f}")
            with col3:
                st.metric("Threshold", f"{result.threshold:.4f}")

            # Buttons row
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("📖 Explain", key=f"explain_{result.trait.name}"):
                    st.session_state.show_explanation = result.trait

            # Remediation options (only for failures)
            if result.status in ['Fail', 'Warning']:
                with btn_col2:
                    st.markdown("**Fix Options:**")

                remediations = RemediationEngine.get_remediations(result.trait)
                for rem in remediations:
                    rem_col1, rem_col2 = st.columns([3, 1])
                    with rem_col1:
                        st.markdown(f"• **{rem.name}**")
                        st.caption(f"Side effects: {', '.join(rem.side_effects)}")
                    with rem_col2:
                        if st.button("Preview", key=f"preview_{result.trait.name}_{rem.name}"):
                            with st.spinner("Analyzing impact..."):
                                impact = RemediationEngine.preview_impact(
                                    snapshot, rem, st.session_state.model_name
                                )

                            if impact.get('success', False):
                                st.markdown("**Impact Preview:**")

                                if impact['fixed']:
                                    st.success(f"✅ Fixed: {[t.name for t in impact['fixed']]}")
                                if impact['broken']:
                                    st.error(f"❌ Broken: {[t.name for t in impact['broken']]}")

                                st.metric("R² Change", f"{impact['r2_delta']:+.4f}")

                                rec_color = {
                                    'Apply': 'green',
                                    'Risky': 'orange',
                                    'Rejected': 'red'
                                }.get(impact['recommendation'], 'gray')

                                st.markdown(f"**Recommendation:** :{rec_color}[{impact['recommendation']}]")

                                if impact['recommendation'] != 'Rejected':
                                    if st.button(f"Apply {rem.name}", key=f"apply_{result.trait.name}_{rem.name}"):
                                        new_snapshot = impact['sim_snapshot']
                                        add_snapshot(new_snapshot)
                                        st.rerun()
                            else:
                                st.error(f"Preview failed: {impact.get('error', 'Unknown error')}")


def render_explanation_modal():
    """Render explanation modal if triggered."""
    if st.session_state.show_explanation is not None:
        trait = st.session_state.show_explanation
        explanation = EXPLANATIONS.get(trait, {})

        st.markdown("---")
        st.subheader(f"📚 {explanation.get('title', trait.name)}")

        st.markdown("### Definition")
        st.info(explanation.get('definition', 'No definition available.'))

        st.markdown("### Consequences for Model")
        st.warning(explanation.get('consequences', 'No consequences documented.'))

        st.markdown("### References")
        st.caption(explanation.get('references', 'No references available.'))

        if st.button("Close Explanation"):
            st.session_state.show_explanation = None
            st.rerun()


def render_visualizations(snapshot: DataSnapshot):
    """Render visualization tabs."""
    st.subheader("📊 Visualizations")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Residuals vs Fitted",
        "Q-Q Plot",
        "VIF Chart",
        "Cook's Distance",
        "Correlation Matrix"
    ])

    with tab1:
        fig = plot_residuals_vs_fitted(snapshot)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = plot_qq(snapshot)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = plot_vif_chart(snapshot)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        fig = plot_cooks_distance(snapshot)
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        fig = plot_correlation_heatmap(snapshot)
        st.plotly_chart(fig, use_container_width=True)


def render_train_model(snapshot: DataSnapshot, compatibility: Dict):
    """Render model training section."""
    st.markdown("---")
    st.subheader("🔧 Model Training")

    if not compatibility['compatible']:
        st.warning("⚠️ Model training is disabled. Resolve critical assumption violations first.")
        st.button("🔧 Train Model", disabled=True, key="train_disabled")
    else:
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔧 Train Model", type="primary", key="train_model"):
                with st.spinner("Training model..."):
                    X_train = pd.DataFrame(snapshot.X_train)
                    y_train = pd.Series(snapshot.y_train)
                    X_test = pd.DataFrame(snapshot.X_test)
                    y_test = pd.Series(snapshot.y_test)

                    model_class = MODEL_REGISTRY[st.session_state.model_name].get(
                        'sklearn_class', LinearRegression
                    )
                    if model_class is None:
                        model_class = LinearRegression

                    model = model_class()
                    model.fit(X_train, y_train)

                    # Compute metrics
                    train_score = model.score(X_train, y_train)
                    test_score = model.score(X_test, y_test)

                    train_pred = model.predict(X_train)
                    test_pred = model.predict(X_test)

                    train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
                    test_rmse = np.sqrt(np.mean((y_test - test_pred) ** 2))

                    st.session_state.trained_model = {
                        'model': model,
                        'train_r2': train_score,
                        'test_r2': test_score,
                        'train_rmse': train_rmse,
                        'test_rmse': test_rmse,
                        'snapshot_id': snapshot.snapshot_id
                    }

        if st.session_state.trained_model is not None:
            with col2:
                m = st.session_state.trained_model
                st.markdown(f"**Trained on snapshot:** `{m['snapshot_id']}`")

            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                st.metric("Train R²", f"{m['train_r2']:.4f}")
            with mcol2:
                st.metric("Test R²", f"{m['test_r2']:.4f}")
            with mcol3:
                st.metric("Train RMSE", f"{m['train_rmse']:.4f}")
            with mcol4:
                st.metric("Test RMSE", f"{m['test_rmse']:.4f}")

            # Coefficients
            if hasattr(m['model'], 'coef_'):
                with st.expander("Model Coefficients"):
                    coef_df = pd.DataFrame({
                        'Feature': list(snapshot.feature_names),
                        'Coefficient': m['model'].coef_
                    })
                    if hasattr(m['model'], 'intercept_'):
                        intercept_row = pd.DataFrame({
                            'Feature': ['(Intercept)'],
                            'Coefficient': [m['model'].intercept_]
                        })
                        coef_df = pd.concat([intercept_row, coef_df], ignore_index=True)
                    st.dataframe(coef_df)


def render_export_section(snapshot: DataSnapshot, results: List[AssumptionResult]):
    """Render export functionality."""
    st.markdown("---")
    st.subheader("📤 Export Analysis")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📄 Generate Report", key="gen_report"):
            results_history = {}
            for s in st.session_state.snapshots:
                s_results = DiagnosticEngine.run_full_diagnostics(s, st.session_state.model_name)
                results_history[s.snapshot_id] = s_results

            final_metrics = None
            if st.session_state.trained_model:
                m = st.session_state.trained_model
                final_metrics = {
                    'Train R²': m['train_r2'],
                    'Test R²': m['test_r2'],
                    'Train RMSE': m['train_rmse'],
                    'Test RMSE': m['test_rmse']
                }

            report = generate_analysis_report(
                st.session_state.snapshots,
                results_history,
                st.session_state.model_name,
                final_metrics
            )

            st.download_button(
                label="⬇️ Download Report (Markdown)",
                data=report,
                file_name="regression_analysis_report.md",
                mime="text/markdown",
                key="download_report"
            )

    with col2:
        st.caption("Report includes: snapshot timeline, diagnostics, transformations, and model performance.")


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Statistical IDE for Regression",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
        .stExpander {
            background-color: #f0f2f6;
            border-radius: 10px;
        }
        .metric-container {
            background-color: #ffffff;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    init_session_state()

    # Render sidebar
    render_sidebar()

    # Main content
    st.title("📊 Statistical IDE for Regression Analysis")
    st.markdown("*A Type-Checked Statistical Compiler for Valid Regression Analysis*")

    snapshot = get_current_snapshot()

    if snapshot is None:
        st.info("👈 Load data from the sidebar to begin your analysis.")

        # Show welcome information
        st.markdown("""
        ## Welcome to the Statistical IDE

        This tool helps you build valid regression models by:

        1. **Type-Checking Assumptions** - Automatically diagnoses assumption violations
        2. **Preventing Invalid States** - Won't let you train models with critical violations
        3. **Smart Remediation** - Suggests fixes with impact analysis
        4. **Transformation Tracking** - Maintains immutable audit trail

        ### Getting Started

        1. Upload your data or use a sample dataset
        2. Select your target (dependent) variable
        3. Choose a regression model type
        4. Review and fix any assumption violations
        5. Train your model when all critical checks pass

        ### Supported Models

        - **OLS** - Ordinary Least Squares (strictest assumptions)
        - **Ridge** - L2 regularization (robust to multicollinearity)
        - **Lasso** - L1 regularization (feature selection)
        - **ElasticNet** - Combined L1/L2 regularization
        """)
        return

    # Run diagnostics
    results = DiagnosticEngine.run_full_diagnostics(snapshot, st.session_state.model_name)
    st.session_state.results_history[snapshot.snapshot_id] = results

    # Render main UI components
    compatibility = render_compiler_status(snapshot, results)

    # Two-column layout for report and visualizations
    col_left, col_right = st.columns([1, 1])

    with col_left:
        render_diagnostic_report(snapshot, results, compatibility)

    with col_right:
        render_visualizations(snapshot)

    # Explanation modal (if triggered)
    render_explanation_modal()

    # Training section
    render_train_model(snapshot, compatibility)

    # Export section
    render_export_section(snapshot, results)


if __name__ == "__main__":
    main()
