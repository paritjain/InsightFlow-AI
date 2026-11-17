"""
Universal Result Validation Engine.

Guarantees analytical integrity, filter satisfaction, sample size sufficiency,
and mathematical safety before generating conversational answers.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from engine.contracts.chat_contracts import (
    QueryPlan,
    QueryIntent,
    ExecutionResult,
    ResultValidationStatus,
    FilterOperator,
    ColumnFilter
)


class UniversalResultValidator:
    """
    Validates ExecutionResult output against the QueryPlan specification,
    filter criteria, sample sizes, and data distribution integrity.
    """

    def __init__(self, col_metadata: Optional[Dict[str, Dict[str, Any]]] = None):
        self.col_metadata = col_metadata or {}

    def validate(
        self,
        plan: QueryPlan,
        result: ExecutionResult,
        df: Optional[pd.DataFrame] = None
    ) -> ExecutionResult:
        """
        Runs comprehensive validation checks on ExecutionResult and enriches warnings/status.
        """
        # If execution already encountered an unhandled error or was blocked
        if result.status in [ResultValidationStatus.ERROR, ResultValidationStatus.BLOCKED, ResultValidationStatus.AMBIGUOUS]:
            return result

        warnings: List[str] = list(result.warnings)

        # 1. Row Count & Emptiness Verification
        if result.rows_returned == 0:
            if result.total_dataset_rows == 0:
                result.status = ResultValidationStatus.INSUFFICIENT_DATA
                result.error_message = "The dataset contains zero records."
                return result
            elif plan.intent == QueryIntent.OUTLIER_ANALYSIS:
                # 0 outliers found is a valid analytical outcome
                result.status = ResultValidationStatus.PASS
                return result
            elif plan.filters:
                result.status = ResultValidationStatus.NO_MATCH
                warnings.append("No records in the dataset matched the specified filter criteria.")
                result.warnings = warnings
                return result
            else:
                result.status = ResultValidationStatus.WARNING
                warnings.append("Query returned zero rows.")
                result.warnings = warnings
                return result

        # 2. Filter Predicate Satisfaction Check (Data Leakage Guard)
        if plan.filters and result.data_sample:
            self._verify_filter_satisfaction(plan.filters, result.data_sample, warnings)

        # 3. Statistical Power & Sample Size Validation
        if plan.intent == QueryIntent.CORRELATION_ANALYSIS:
            sample_size = result.records_evaluated or result.total_dataset_rows
            if result.data_sample and "sample_size" in result.data_sample[0]:
                sample_size = result.data_sample[0]["sample_size"]

            if sample_size < 3:
                result.status = ResultValidationStatus.WARNING
                warnings.append(
                    f"Sample size (N={sample_size}) is too small for statistical correlation analysis. "
                    "At least 3 complete observations are required."
                )

        elif plan.intent in [QueryIntent.GROUP_COMPARISON, QueryIntent.HYPOTHESIS_TEST]:
            if result.data_sample:
                small_groups = [
                    str(r.get(plan.dimension, "unknown"))
                    for r in result.data_sample
                    if r.get("group_count", r.get("count", 10)) < 3
                ]
                if small_groups:
                    warnings.append(
                        f"Groups with very small sample sizes (<3 observations) detected: {', '.join(small_groups[:3])}. "
                        "Averages may be statistically unstable."
                    )

        # 4. Zero Variance / Constant Column Check
        if plan.metric and df is not None and plan.metric in df.columns:
            s = df[plan.metric].dropna()
            if len(s) > 0 and s.nunique() <= 1:
                warnings.append(
                    f"Target measure '{plan.metric}' is constant (zero variance, value={s.iloc[0]}). "
                    "Variance-based statistics and correlations cannot be meaningfully interpreted."
                )

        # 5. Semantic Value Range Audits
        if plan.metric and self.col_metadata:
            meta = self.col_metadata.get(plan.metric, {})
            subtype = str(meta.get("semantic_subtype", meta.get("measure_type", "")))
            
            # Percentage Check
            if subtype == "Percentage" and result.scalar_value is not None:
                val = float(result.scalar_value)
                if val < 0.0 or val > 100.0:
                    warnings.append(f"Computed percentage value {val:.2f}% falls outside standard 0-100% bounds.")

            # Duration Check
            if subtype == "Duration" and result.scalar_value is not None:
                val = float(result.scalar_value)
                if val < 0.0:
                    warnings.append(f"Computed duration {val:.2f} is negative, which violates physical constraints.")

        # Update warnings in result
        result.warnings = warnings
        analytical_warnings = [
            w for w in warnings 
            if any(term in w.lower() for term in [
                'violates', 'zero variance', 'small sample', 'too small', 'unstable', 'outside', 'negative', 'error'
            ])
        ]
        if analytical_warnings and result.status == ResultValidationStatus.PASS:
            result.status = ResultValidationStatus.WARNING

        return result

    def _verify_filter_satisfaction(
        self,
        filters: List[ColumnFilter],
        data_sample: List[Dict[str, Any]],
        warnings: List[str]
    ):
        """Verifies that sample rows strictly conform to the query filters."""
        for f in filters:
            col = f.column
            op = f.operator
            val = f.value

            for i, row in enumerate(data_sample[:20]):
                if col not in row:
                    continue
                row_val = row[col]
                if pd.isna(row_val):
                    continue

                satisfied = True
                if op == FilterOperator.EQUALS:
                    satisfied = (str(row_val).lower() == str(val).lower())
                elif op == FilterOperator.NOT_EQUALS:
                    satisfied = (str(row_val).lower() != str(val).lower())
                elif op == FilterOperator.GREATER_THAN:
                    satisfied = (float(row_val) > float(val))
                elif op == FilterOperator.GREATER_EQUAL:
                    satisfied = (float(row_val) >= float(val))
                elif op == FilterOperator.LESS_THAN:
                    satisfied = (float(row_val) < float(val))
                elif op == FilterOperator.LESS_EQUAL:
                    satisfied = (float(row_val) <= float(val))

                if not satisfied:
                    warnings.append(
                        f"Data integrity warning: Sample record {i} has {col}={row_val}, "
                        f"which violates filter predicate {col} {op.value} {val}."
                    )
                    break
