"""
Deterministic DuckDB In-Memory Analytical Execution Engine.

Executes structured QueryPlan contracts with zero-copy DataFrame binding,
safe parameterized SQL generation, strict read-only guarantees,
and millisecond-level analytical performance.
"""

import time
import uuid
import re
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import duckdb

from engine.contracts.chat_contracts import (
    QueryPlan,
    QueryIntent,
    FilterOperator,
    ColumnFilter,
    ExecutionResult,
    ResultValidationStatus
)


class DuckDBAnalyticalEngine:
    """
    In-Memory Analytical Execution Engine powered by DuckDB.
    
    Provides deterministic, zero-hallucination computation for:
    - Global scalar aggregations
    - Group comparisons and rankings
    - Time-series trend groupings
    - Bivariate and matrix correlations
    - Distribution profiling and IQR outlier detection
    - Data quality audits and filtered record slicing
    """

    def __init__(self, df: Optional[pd.DataFrame] = None, table_name: str = "dataset"):
        self.table_name = table_name
        self.con = duckdb.connect(database=":memory:")
        self.df: Optional[pd.DataFrame] = None
        self.total_rows: int = 0
        if df is not None:
            self.register_dataframe(df)

    def register_dataframe(self, df: pd.DataFrame, table_name: Optional[str] = None):
        """Registers a pandas DataFrame into DuckDB in-memory session with zero-copy."""
        if table_name:
            self.table_name = table_name
        self.df = df
        self.total_rows = len(df)
        self.con.register(self.table_name, df)

    def execute_plan(
        self,
        plan: QueryPlan,
        df: Optional[pd.DataFrame] = None
    ) -> ExecutionResult:
        """
        Executes a typed QueryPlan contract deterministically and returns an ExecutionResult.
        """
        start_time = time.perf_counter()

        if df is not None:
            self.register_dataframe(df)

        if self.df is None or self.total_rows == 0:
            return ExecutionResult(
                plan_id=plan.plan_id,
                engine_used="DuckDB",
                status=ResultValidationStatus.INSUFFICIENT_DATA,
                error_message="Dataset is empty or not registered.",
                total_dataset_rows=0,
                rows_returned=0
            )

        if plan.intent == QueryIntent.AMBIGUOUS:
            return ExecutionResult(
                plan_id=plan.plan_id,
                engine_used="DuckDB",
                status=ResultValidationStatus.AMBIGUOUS,
                error_message=plan.validation_message or "Query is ambiguous.",
                warnings=[plan.clarification_prompt or "Clarification required."],
                total_dataset_rows=self.total_rows
            )

        # 1. Compile SQL Query and Parameters
        try:
            sql, params = self.compile_sql(plan)
        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000.0
            return ExecutionResult(
                plan_id=plan.plan_id,
                engine_used="DuckDB",
                status=ResultValidationStatus.ERROR,
                execution_time_ms=execution_time_ms,
                error_message=f"SQL compilation failed: {str(e)}",
                total_dataset_rows=self.total_rows
            )

        # 2. Strict Read-Only Security Guard
        normalized_sql = sql.strip().upper()
        forbidden_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE", "ATTACH", "COPY"]
        if any(re.search(rf'\b{kw}\b', normalized_sql) for kw in forbidden_keywords):
            execution_time_ms = (time.perf_counter() - start_time) * 1000.0
            return ExecutionResult(
                plan_id=plan.plan_id,
                engine_used="DuckDB",
                status=ResultValidationStatus.BLOCKED,
                execution_time_ms=execution_time_ms,
                error_message="Non-read-only SQL statements are strictly forbidden.",
                total_dataset_rows=self.total_rows
            )

        # 3. Execute in DuckDB
        try:
            # Handle special two-phase intents like OUTLIER_ANALYSIS
            if plan.intent == QueryIntent.OUTLIER_ANALYSIS:
                return self._execute_outlier_analysis(plan, start_time)

            if params:
                cursor = self.con.execute(sql, params)
            else:
                cursor = self.con.execute(sql)

            result_df = cursor.df()
            execution_time_ms = (time.perf_counter() - start_time) * 1000.0

            rows_returned = len(result_df)
            columns = list(result_df.columns)
            data_sample = result_df.head(100).to_dict(orient="records")

            # Determine scalar value if applicable
            scalar_val = None
            if rows_returned == 1 and len(columns) == 1:
                scalar_val = result_df.iloc[0, 0]
                if pd.isna(scalar_val):
                    scalar_val = None
            elif plan.intent == QueryIntent.GLOBAL_AGGREGATION and rows_returned == 1:
                # If single row of aggregations, pick primary aggregation value as scalar
                first_row = data_sample[0] if data_sample else {}
                for k, v in first_row.items():
                    if k != "total_count" and not pd.isna(v):
                        scalar_val = v
                        break
                if scalar_val is None and "total_count" in first_row:
                    scalar_val = first_row["total_count"]

            # Handle correlation specific return
            if plan.intent == QueryIntent.CORRELATION_ANALYSIS and rows_returned == 1:
                if "pearson_r" in result_df.columns:
                    val = result_df["pearson_r"].iloc[0]
                    scalar_val = float(val) if not pd.isna(val) else 0.0

            records_eval = self.total_rows
            if plan.filters:
                # Calculate coverage of filtered records
                cov_sql = f'SELECT COUNT(*) FROM "{self.table_name}"'
                where_clause, cov_params = self._build_where_clause(plan.filters)
                if where_clause:
                    cov_sql += " " + where_clause
                    records_eval = self.con.execute(cov_sql, cov_params).fetchone()[0]

            coverage_pct = round((records_eval / self.total_rows) * 100.0, 2) if self.total_rows > 0 else 0.0

            return ExecutionResult(
                plan_id=plan.plan_id,
                engine_used="DuckDB",
                status=ResultValidationStatus.PASS,
                execution_time_ms=execution_time_ms,
                rows_returned=rows_returned,
                records_evaluated=records_eval,
                total_dataset_rows=self.total_rows,
                coverage_pct=coverage_pct,
                data_sample=data_sample,
                scalar_value=scalar_val,
                tabular_columns=columns,
                compiled_query=sql,
                parameters_used={f"p{i}": v for i, v in enumerate(params)} if params else {}
            )

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000.0
            return ExecutionResult(
                plan_id=plan.plan_id,
                engine_used="DuckDB",
                status=ResultValidationStatus.ERROR,
                execution_time_ms=execution_time_ms,
                error_message=f"DuckDB execution error: {str(e)}",
                compiled_query=sql,
                total_dataset_rows=self.total_rows
            )

    def compile_sql(self, plan: QueryPlan) -> Tuple[str, List[Any]]:
        """
        Compiles a structured QueryPlan into a safe parameterized SQL string and parameter list.
        """
        table = f'"{self.table_name}"'
        where_clause, params = self._build_where_clause(plan.filters)

        intent = plan.intent

        # 1. Global Aggregation
        if intent == QueryIntent.GLOBAL_AGGREGATION:
            metric = plan.metric
            aggregations = plan.aggregations or ["MEAN"]
            
            select_items = ['COUNT(*) AS total_count']
            if metric:
                q_metric = self._quote(metric)
                for agg in aggregations:
                    agg_u = str(agg).upper()
                    if agg_u in ["SUM"]:
                        select_items.append(f'SUM({q_metric}) AS {self._quote(f"sum_{metric}")}')
                    elif agg_u in ["MEAN", "AVG"]:
                        select_items.append(f'AVG({q_metric}) AS {self._quote(f"avg_{metric}")}')
                    elif agg_u in ["MIN"]:
                        select_items.append(f'MIN({q_metric}) AS {self._quote(f"min_{metric}")}')
                    elif agg_u in ["MAX"]:
                        select_items.append(f'MAX({q_metric}) AS {self._quote(f"max_{metric}")}')
                    elif agg_u in ["MEDIAN"]:
                        select_items.append(f'MEDIAN({q_metric}) AS {self._quote(f"median_{metric}")}')
                    elif agg_u in ["STDDEV", "STD"]:
                        select_items.append(f'STDDEV({q_metric}) AS {self._quote(f"stddev_{metric}")}')
                    elif agg_u in ["COUNT_DISTINCT"]:
                        select_items.append(f'COUNT(DISTINCT {q_metric}) AS {self._quote(f"distinct_{metric}")}')
            sql = f"SELECT {', '.join(select_items)} FROM {table}"
            if where_clause:
                sql += f" {where_clause}"
            return sql, params

        # 2. Group Comparison
        elif intent == QueryIntent.GROUP_COMPARISON:
            dim = self._quote(plan.dimension or self._get_fallback_dimension())
            metric = plan.metric
            agg_func = "AVG" if plan.aggregation in ["MEAN", "AVG", None] else str(plan.aggregation).upper()
            
            if metric:
                q_metric = self._quote(metric)
                val_expr = f"{agg_func}({q_metric})"
                select_clause = f"{dim}, COUNT(*) AS group_count, {val_expr} AS metric_val"
                order_expr = "metric_val"
            else:
                select_clause = f"{dim}, COUNT(*) AS count"
                order_expr = "count"

            sort_dir = "DESC" if plan.sort in ["descending", "desc", None] else "ASC"
            sql = f"SELECT {select_clause} FROM {table}"
            if where_clause:
                sql += f" {where_clause}"
            sql += f" GROUP BY {dim} ORDER BY {order_expr} {sort_dir}"
            if plan.limit:
                sql += f" LIMIT {plan.limit}"
            else:
                sql += " LIMIT 50"
            return sql, params

        # 3. Ranking Top N
        elif intent == QueryIntent.RANKING_TOP_N:
            dim = self._quote(plan.dimension or self._get_fallback_dimension())
            metric = plan.metric
            agg_func = "SUM" if plan.aggregation in ["SUM"] else "AVG"
            
            if metric:
                q_metric = self._quote(metric)
                select_clause = f"{dim}, COUNT(*) AS count, {agg_func}({q_metric}) AS rank_val"
                order_col = "rank_val"
            else:
                select_clause = f"{dim}, COUNT(*) AS count"
                order_col = "count"

            sort_dir = "ASC" if plan.sort in ["ascending", "asc"] else "DESC"
            limit_val = plan.limit or 5

            sql = f"SELECT {select_clause} FROM {table}"
            if where_clause:
                sql += f" {where_clause}"
            sql += f" GROUP BY {dim} ORDER BY {order_col} {sort_dir} LIMIT {limit_val}"
            return sql, params

        # 4. Distinct Count
        elif intent == QueryIntent.DISTINCT_COUNT:
            target_col = self._quote(plan.dimension or plan.metric or list(self.df.columns)[0])
            sql = f"SELECT COUNT(DISTINCT {target_col}) AS distinct_count FROM {table}"
            if where_clause:
                sql += f" {where_clause}"
            return sql, params

        # 5. Row Lookup
        elif intent == QueryIntent.ROW_LOOKUP:
            sql = f"SELECT * FROM {table}"
            if where_clause:
                sql += f" {where_clause}"
            sql += " LIMIT 1"
            return sql, params

        # 6. Filtered Records
        elif intent == QueryIntent.FILTERED_RECORDS:
            sql = f"SELECT * FROM {table}"
            if where_clause:
                sql += f" {where_clause}"
            limit_val = plan.limit or 100
            sql += f" LIMIT {limit_val}"
            return sql, params

        # 7. Time Series Trend
        elif intent == QueryIntent.TIME_SERIES_TREND:
            time_col = self._quote(plan.time_column or self._get_fallback_time_column())
            grain = plan.time_grain or "month"
            metric = plan.metric
            
            period_expr = f"DATE_TRUNC('{grain}', {time_col})"
            if metric:
                q_metric = self._quote(metric)
                select_clause = f"{period_expr} AS period, COUNT(*) AS count, AVG({q_metric}) AS avg_val, SUM({q_metric}) AS sum_val"
            else:
                select_clause = f"{period_expr} AS period, COUNT(*) AS count"

            sql = f"SELECT {select_clause} FROM {table}"
            if where_clause:
                sql += f" {where_clause} AND {time_col} IS NOT NULL"
            else:
                sql += f" WHERE {time_col} IS NOT NULL"
            sql += " GROUP BY 1 ORDER BY 1 ASC"
            return sql, params

        # 8. Correlation Analysis
        elif intent == QueryIntent.CORRELATION_ANALYSIS:
            vars_to_use = plan.variables or plan.target_columns
            if len(vars_to_use) >= 2:
                v1 = self._quote(vars_to_use[0])
                v2 = self._quote(vars_to_use[1])
            else:
                # Fallback to first two numeric measures
                num_cols = [c for c in self.df.columns if pd.api.types.is_numeric_dtype(self.df[c])]
                v1 = self._quote(num_cols[0] if len(num_cols) > 0 else "col1")
                v2 = self._quote(num_cols[1] if len(num_cols) > 1 else "col2")

            sql = f"SELECT CORR({v1}, {v2}) AS pearson_r, COUNT(*) AS sample_size FROM {table}"
            corr_filter = f"{v1} IS NOT NULL AND {v2} IS NOT NULL"
            if where_clause:
                sql += f" {where_clause} AND {corr_filter}"
            else:
                sql += f" WHERE {corr_filter}"
            return sql, params

        # 9. Distribution Analysis
        elif intent == QueryIntent.DISTRIBUTION_ANALYSIS:
            metric = self._quote(plan.metric or self._get_fallback_measure())
            sql = (
                f"SELECT COUNT({metric}) AS count, "
                f"AVG({metric}) AS mean, "
                f"STDDEV({metric}) AS stddev, "
                f"MIN({metric}) AS min, "
                f"QUANTILE_CONT({metric}, 0.25) AS q25, "
                f"MEDIAN({metric}) AS median, "
                f"QUANTILE_CONT({metric}, 0.75) AS q75, "
                f"MAX({metric}) AS max, "
                f"SKEWNESS({metric}) AS skewness, "
                f"KURTOSIS({metric}) AS kurtosis "
                f"FROM {table}"
            )
            if where_clause:
                sql += f" {where_clause} AND {metric} IS NOT NULL"
            else:
                sql += f" WHERE {metric} IS NOT NULL"
            return sql, params

        # 10. Data Quality Audit
        elif intent == QueryIntent.DATA_QUALITY:
            # Audit missingness and uniqueness across columns
            select_items = ["COUNT(*) AS total_records"]
            for col in list(self.df.columns)[:15]:
                q_col = self._quote(col)
                c_clean = re.sub(r'[^a-zA-Z0-9_]', '_', col)
                select_items.append(f"COUNT(*) - COUNT({q_col}) AS missing_{c_clean}")
                select_items.append(f"COUNT(DISTINCT {q_col}) AS distinct_{c_clean}")
            sql = f"SELECT {', '.join(select_items)} FROM {table}"
            if where_clause:
                sql += f" {where_clause}"
            return sql, params

        # 11. Dataset Summary (Default)
        else:
            sql = f"SELECT COUNT(*) AS total_rows, {len(self.df.columns)} AS total_columns FROM {table}"
            if where_clause:
                sql += f" {where_clause}"
            return sql, params

    def _execute_outlier_analysis(self, plan: QueryPlan, start_time: float) -> ExecutionResult:
        """Computes continuous IQR percentiles and extracts outliers with DuckDB."""
        metric_raw = plan.metric or self._get_fallback_measure()
        metric = self._quote(metric_raw)
        table = f'"{self.table_name}"'
        where_clause, params = self._build_where_clause(plan.filters)

        # Phase 1: Compute Quartiles and Bounds
        quartile_sql = f"SELECT QUANTILE_CONT({metric}, 0.25) AS q1, QUANTILE_CONT({metric}, 0.75) AS q3, MEDIAN({metric}) AS median FROM {table}"
        if where_clause:
            quartile_sql += f" {where_clause} AND {metric} IS NOT NULL"
        else:
            quartile_sql += f" WHERE {metric} IS NOT NULL"

        q_df = self.con.execute(quartile_sql, params).df()
        q1 = float(q_df["q1"].iloc[0])
        q3 = float(q_df["q3"].iloc[0])
        median_val = float(q_df["median"].iloc[0])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Phase 2: Retrieve records violating bounds
        outlier_sql = (
            f"SELECT *, ({metric} - {median_val}) / NULLIF({iqr}, 0) AS iqr_distance "
            f"FROM {table} "
            f"WHERE ({metric} < {lower_bound} OR {metric} > {upper_bound}) "
        )
        if where_clause:
            outlier_sql += f"AND {where_clause.replace('WHERE ', '')} "
        outlier_sql += f"ORDER BY ABS({metric} - {median_val}) DESC LIMIT 50"

        outlier_df = self.con.execute(outlier_sql, params).df()
        execution_time_ms = (time.perf_counter() - start_time) * 1000.0

        rows_returned = len(outlier_df)
        data_sample = outlier_df.head(50).to_dict(orient="records")

        return ExecutionResult(
            plan_id=plan.plan_id,
            engine_used="DuckDB",
            status=ResultValidationStatus.PASS,
            execution_time_ms=execution_time_ms,
            rows_returned=rows_returned,
            records_evaluated=self.total_rows,
            total_dataset_rows=self.total_rows,
            coverage_pct=round((rows_returned / self.total_rows) * 100.0, 2) if self.total_rows > 0 else 0.0,
            data_sample=data_sample,
            scalar_value=rows_returned,
            tabular_columns=list(outlier_df.columns),
            compiled_query=outlier_sql,
            warnings=[f"IQR Bounds for {metric_raw}: [{lower_bound:.2f}, {upper_bound:.2f}]. Found {rows_returned} outliers."]
        )

    def _build_where_clause(self, filters: List[ColumnFilter]) -> Tuple[str, List[Any]]:
        """Constructs safe parameterized SQL WHERE conditions from ColumnFilter contracts."""
        if not filters:
            return "", []

        clauses: List[str] = []
        params: List[Any] = []

        for f in filters:
            q_col = self._quote(f.column)
            op = f.operator
            val = f.value

            # Parameter type safety: coerce parameter to match column dtype in DuckDB to avoid conversion exceptions
            is_col_numeric = True
            if self.df is not None and f.column in self.df.columns:
                is_col_numeric = pd.api.types.is_numeric_dtype(self.df[f.column])

            if not is_col_numeric:
                if op in [FilterOperator.EQUALS, FilterOperator.NOT_EQUALS, FilterOperator.LIKE, FilterOperator.ILIKE]:
                    val = str(val)
                elif op == FilterOperator.IN and isinstance(val, (list, tuple, set)):
                    val = [str(v) for v in val]
            else:
                if op in [FilterOperator.EQUALS, FilterOperator.NOT_EQUALS, FilterOperator.GREATER_THAN, FilterOperator.GREATER_EQUAL, FilterOperator.LESS_THAN, FilterOperator.LESS_EQUAL]:
                    try:
                        val = float(val) if '.' in str(val) else int(val)
                    except (ValueError, TypeError):
                        pass

            if op == FilterOperator.EQUALS:
                clauses.append(f"{q_col} = ?")
                params.append(val)
            elif op == FilterOperator.NOT_EQUALS:
                clauses.append(f"{q_col} != ?")
                params.append(val)
            elif op == FilterOperator.GREATER_THAN:
                clauses.append(f"{q_col} > ?")
                params.append(val)
            elif op == FilterOperator.GREATER_EQUAL:
                clauses.append(f"{q_col} >= ?")
                params.append(val)
            elif op == FilterOperator.LESS_THAN:
                clauses.append(f"{q_col} < ?")
                params.append(val)
            elif op == FilterOperator.LESS_EQUAL:
                clauses.append(f"{q_col} <= ?")
                params.append(val)
            elif op == FilterOperator.BETWEEN:
                clauses.append(f"{q_col} BETWEEN ? AND ?")
                params.append(val)
                params.append(f.secondary_value)
            elif op == FilterOperator.IS_NULL:
                clauses.append(f"{q_col} IS NULL")
            elif op == FilterOperator.IS_NOT_NULL:
                clauses.append(f"{q_col} IS NOT NULL")
            elif op == FilterOperator.LIKE:
                clauses.append(f"{q_col} LIKE ?")
                params.append(val)
            elif op == FilterOperator.ILIKE:
                clauses.append(f"{q_col} ILIKE ?")
                params.append(val)
            elif op == FilterOperator.IN:
                if isinstance(val, (list, tuple, set)) and val:
                    placeholders = ", ".join(["?"] * len(val))
                    clauses.append(f"{q_col} IN ({placeholders})")
                    params.extend(list(val))
                else:
                    clauses.append(f"{q_col} = ?")
                    params.append(val)

        if not clauses:
            return "", []
        return f"WHERE {' AND '.join(clauses)}", params

    def _quote(self, identifier: str) -> str:
        """Escapes and double-quotes SQL column identifiers for safety."""
        clean = identifier.replace('"', '""')
        return f'"{clean}"'

    def _get_fallback_dimension(self) -> str:
        """Selects the first categorical column or non-numeric column."""
        for c in self.df.columns:
            if not pd.api.types.is_numeric_dtype(self.df[c]) and not pd.api.types.is_datetime64_any_dtype(self.df[c]):
                return str(c)
        return str(self.df.columns[0])

    def _get_fallback_measure(self) -> str:
        """Selects the first numeric measure column."""
        for c in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[c]):
                return str(c)
        return str(self.df.columns[0])

    def _get_fallback_time_column(self) -> str:
        """Selects the first datetime column or string column matching date keywords."""
        for c in self.df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.df[c]):
                return str(c)
        for c in self.df.columns:
            if any(k in str(c).lower() for k in ['date', 'time', 'timestamp', 'day']):
                return str(c)
        return str(self.df.columns[0])
