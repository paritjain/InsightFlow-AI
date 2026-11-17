import re
from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd

from engine.contracts.enums import DerivationType, RelationshipType
from engine.contracts.lineage_contract import DerivationContract


class DerivationAndLineageEngine:
    """
    Universal Algebraic Derivation & Lineage Engine (Prompt Sections 15 & 16).
    Detects mathematical relationships:
    - Pairwise Linear: A = k * B, A = B / k, A = k * B + c, A = percentage transformation of B
    - Multiplicative: A = B * C
    - Division / Ratio: A = B / C, A = C / B
    - Additive: A = B + C
    - Subtractive: A = B - C, A = C - B
    - Percentage Ratio: A = (B / C) * 100

    Stores:
    - derivation_type: OBSERVED, DERIVED, DETERMINISTIC_TRANSFORMATION, AGGREGATED
    - derived_from: List of source columns
    - derivation_formula: String formula
    - transformation_type: MULTIPLICATIVE, RATIO, LINEAR_SCALING, PERCENTAGE_SCALING, etc.
    - relationship_type: STRUCTURAL_RELATIONSHIP / DETERMINISTIC_TRANSFORMATION
    """

    @classmethod
    def scan_dataset(
        cls,
        df: pd.DataFrame,
        candidate_cols: Optional[List[str]] = None,
        tolerance: float = 1e-3
    ) -> List[DerivationContract]:
        """
        Scans a DataFrame and returns verified DerivationContract objects.
        """
        if df is None or df.empty or len(df) < 3:
            return []

        if candidate_cols is None:
            candidate_cols = [
                c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c]) and not df[c].dropna().empty
            ]

        # Guard against combinatorial explosion on wide datasets (e.g. 60+ columns)
        if len(candidate_cols) > 25:
            candidate_cols = candidate_cols[:25]

        if len(candidate_cols) < 2:
            return []

        # Convert candidate columns to float matrix
        num_df = df[candidate_cols].apply(pd.to_numeric, errors='coerce')
        valid_cols = [c for c in candidate_cols if num_df[c].notna().sum() >= 3]
        if len(valid_cols) < 2:
            return []

        # Extract 2D numpy array once: shape (N, M)
        X = num_df[valid_cols].to_numpy()
        M = len(valid_cols)

        # Standard deviations for constant filtering
        stds = np.nanstd(X, axis=0)

        derivations: List[DerivationContract] = []
        seen_targets: Set[str] = set()
        seen_pairs: Set[Tuple[str, str]] = set()

        # Step 1: Pairwise Linear & Deterministic Scaling (A = k * B or A = k * B + c)
        for i, col_a in enumerate(valid_cols):
            if stds[i] <= 1e-9:
                continue
            s_a = X[:, i]

            for j, col_b in enumerate(valid_cols):
                if i == j:
                    continue
                if stds[j] <= 1e-9:
                    continue
                s_b = X[:, j]

                pair_key = tuple(sorted([col_a, col_b]))
                mask = np.isfinite(s_a) & np.isfinite(s_b)
                if np.sum(mask) < 5:
                    continue

                a_vals = s_a[mask]
                b_vals = s_b[mask]

                # Check exact identity A == B
                if np.all(np.abs(a_vals - b_vals) <= tolerance):
                    if col_a not in seen_targets:
                        seen_targets.add(col_a)
                        derivations.append(DerivationContract(
                            derivation_id=f"DERIV-{len(derivations)+1:04d}",
                            target_column=col_a,
                            derivation_type=DerivationType.DETERMINISTIC_TRANSFORMATION,
                            formula=f"{col_a} = {col_b}",
                            source_columns=[col_b],
                            transformation_type="IDENTITY",
                            relationship_type=RelationshipType.STRUCTURAL_RELATIONSHIP,
                            multiplier=1.0,
                            constant_term=0.0,
                            is_deterministic=True,
                            confidence=1.0,
                            description=f"'{col_a}' is identical to '{col_b}'."
                        ))
                    continue

                # Check proportional scaling: A = k * B (without intercept)
                non_zero_mask = np.abs(b_vals) > 1e-9
                if np.sum(non_zero_mask) >= 5:
                    ratios = a_vals[non_zero_mask] / b_vals[non_zero_mask]
                    k_est = float(np.median(ratios))
                    residuals = np.abs(a_vals - k_est * b_vals)
                    max_err = float(np.max(residuals))

                    if max_err <= tolerance or (float(np.mean(residuals)) <= tolerance and float(np.percentile(residuals, 95)) <= tolerance):
                        target_col = col_a
                        source_col = col_b
                        multiplier = k_est

                        a_is_derived_name = bool(re.search(r'(_pct|_percent|_index|_normalized|_scaled|_ratio)$', col_a.lower()))
                        b_is_derived_name = bool(re.search(r'(_pct|_percent|_index|_normalized|_scaled|_ratio)$', col_b.lower()))

                        if b_is_derived_name and not a_is_derived_name:
                            target_col = col_b
                            source_col = col_a
                            multiplier = 1.0 / k_est if abs(k_est) > 1e-9 else k_est

                        if target_col not in seen_targets:
                            seen_targets.add(target_col)
                            if abs(multiplier - round(multiplier)) < 1e-4:
                                mult_str = str(int(round(multiplier)))
                            else:
                                mult_str = f"{multiplier:.4g}"

                            t_type = "PERCENTAGE_SCALING" if ("pct" in target_col.lower() or "percent" in target_col.lower()) else "LINEAR_SCALING"
                            desc = f"'{target_col}' is a deterministic linear transformation of '{source_col}' ({target_col} = {source_col} * {mult_str})."

                            derivations.append(DerivationContract(
                                derivation_id=f"DERIV-{len(derivations)+1:04d}",
                                target_column=target_col,
                                derivation_type=DerivationType.DETERMINISTIC_TRANSFORMATION,
                                formula=f"{target_col} = {source_col} * {mult_str}",
                                source_columns=[source_col],
                                transformation_type=t_type,
                                relationship_type=RelationshipType.STRUCTURAL_RELATIONSHIP,
                                multiplier=multiplier,
                                constant_term=0.0,
                                is_deterministic=True,
                                confidence=1.0,
                                description=desc
                            ))
                            continue

                # Check affine linear: A = k * B + c (where R^2 == 1.0 within 1e-4)
                if pair_key not in seen_pairs:
                    corr_matrix = np.corrcoef(a_vals, b_vals)
                    corr = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
                    if abs(abs(corr) - 1.0) <= 1e-4:
                        slope, intercept = np.polyfit(b_vals, a_vals, 1)
                        res = np.abs(a_vals - (slope * b_vals + intercept))
                        if np.max(res) <= tolerance:
                            seen_pairs.add(pair_key)
                            target_col = col_a
                            source_col = col_b
                            mult = float(slope)
                            c_term = float(intercept)

                            a_is_derived = bool(re.search(r'(_pct|_percent|_index|_normalized|_scaled)$', col_a.lower()))
                            b_is_derived = bool(re.search(r'(_pct|_percent|_index|_normalized|_scaled)$', col_b.lower()))
                            if b_is_derived and not a_is_derived:
                                target_col = col_b
                                source_col = col_a
                                mult = 1.0 / slope if abs(slope) > 1e-9 else slope
                                c_term = -intercept / slope if abs(slope) > 1e-9 else intercept

                            if target_col not in seen_targets:
                                seen_targets.add(target_col)
                                derivations.append(DerivationContract(
                                    derivation_id=f"DERIV-{len(derivations)+1:04d}",
                                    target_column=target_col,
                                    derivation_type=DerivationType.DETERMINISTIC_TRANSFORMATION,
                                    formula=f"{target_col} = {mult:.4g} * {source_col} + {c_term:.4g}",
                                    source_columns=[source_col],
                                    transformation_type="AFFINE_LINEAR",
                                    relationship_type=RelationshipType.STRUCTURAL_RELATIONSHIP,
                                    multiplier=mult,
                                    constant_term=c_term,
                                    is_deterministic=True,
                                    confidence=0.99,
                                    description=f"'{target_col}' is a deterministic transformation of '{source_col}'."
                                ))

        # Step 2: Triplet Multi-Variable Derivations (A = B * C, A = B / C, A = B + C, A = B - C, A = (B / C) * 100)
        for i, col_a in enumerate(valid_cols):
            if col_a in seen_targets:
                continue
            s_a = X[:, i]

            for j, col_b in enumerate(valid_cols):
                if i == j:
                    continue
                s_b = X[:, j]

                for k, col_c in enumerate(valid_cols):
                    if k == i or k <= j:
                        continue
                    s_c = X[:, k]

                    mask = np.isfinite(s_a) & np.isfinite(s_b) & np.isfinite(s_c)
                    if np.sum(mask) < 5:
                        continue

                    a_vals = s_a[mask]
                    b_vals = s_b[mask]
                    c_vals = s_c[mask]

                    # Multiplicative: A = B * C (e.g. Risk_Score = Severity_Rating * Likelihood_Rating)
                    mult_err = np.max(np.abs(a_vals - (b_vals * c_vals)))
                    if mult_err <= tolerance:
                        seen_targets.add(col_a)
                        derivations.append(DerivationContract(
                            derivation_id=f"DERIV-{len(derivations)+1:04d}",
                            target_column=col_a,
                            derivation_type=DerivationType.DERIVED,
                            formula=f"{col_a} = {col_b} * {col_c}",
                            source_columns=[col_b, col_c],
                            transformation_type="MULTIPLICATION",
                            relationship_type=RelationshipType.STRUCTURAL_RELATIONSHIP,
                            is_deterministic=True,
                            confidence=0.99,
                            description=f"'{col_a}' is derived as the product of '{col_b}' and '{col_c}'."
                        ))
                        break

                    # Addition: A = B + C
                    add_err = np.max(np.abs(a_vals - (b_vals + c_vals)))
                    if add_err <= tolerance:
                        seen_targets.add(col_a)
                        derivations.append(DerivationContract(
                            derivation_id=f"DERIV-{len(derivations)+1:04d}",
                            target_column=col_a,
                            derivation_type=DerivationType.DERIVED,
                            formula=f"{col_a} = {col_b} + {col_c}",
                            source_columns=[col_b, col_c],
                            transformation_type="ADDITION",
                            relationship_type=RelationshipType.STRUCTURAL_RELATIONSHIP,
                            is_deterministic=True,
                            confidence=0.99,
                            description=f"'{col_a}' is derived as the sum of '{col_b}' and '{col_c}'."
                        ))
                        break

                    # Subtraction: A = B - C
                    sub_bc_err = np.max(np.abs(a_vals - (b_vals - c_vals)))
                    if sub_bc_err <= tolerance:
                        seen_targets.add(col_a)
                        derivations.append(DerivationContract(
                            derivation_id=f"DERIV-{len(derivations)+1:04d}",
                            target_column=col_a,
                            derivation_type=DerivationType.DERIVED,
                            formula=f"{col_a} = {col_b} - {col_c}",
                            source_columns=[col_b, col_c],
                            transformation_type="SUBTRACTION",
                            relationship_type=RelationshipType.STRUCTURAL_RELATIONSHIP,
                            is_deterministic=True,
                            confidence=0.99,
                            description=f"'{col_a}' is derived as '{col_b} - {col_c}'."
                        ))
                        break

                    # Subtraction: A = C - B
                    sub_cb_err = np.max(np.abs(a_vals - (c_vals - b_vals)))
                    if sub_cb_err <= tolerance:
                        seen_targets.add(col_a)
                        derivations.append(DerivationContract(
                            derivation_id=f"DERIV-{len(derivations)+1:04d}",
                            target_column=col_a,
                            derivation_type=DerivationType.DERIVED,
                            formula=f"{col_a} = {col_c} - {col_b}",
                            source_columns=[col_c, col_b],
                            transformation_type="SUBTRACTION",
                            relationship_type=RelationshipType.STRUCTURAL_RELATIONSHIP,
                            is_deterministic=True,
                            confidence=0.99,
                            description=f"'{col_a}' is derived as '{col_c} - {col_b}'."
                        ))
                        break

                    # Ratio: A = B / C (where C != 0)
                    if np.all(np.abs(c_vals) > 1e-9):
                        div_bc_err = np.max(np.abs(a_vals - (b_vals / c_vals)))
                        if div_bc_err <= tolerance:
                            seen_targets.add(col_a)
                            derivations.append(DerivationContract(
                                derivation_id=f"DERIV-{len(derivations)+1:04d}",
                                target_column=col_a,
                                derivation_type=DerivationType.DERIVED,
                                formula=f"{col_a} = {col_b} / {col_c}",
                                source_columns=[col_b, col_c],
                                transformation_type="RATIO",
                                relationship_type=RelationshipType.STRUCTURAL_RELATIONSHIP,
                                is_deterministic=True,
                                confidence=0.99,
                                description=f"'{col_a}' is derived as the ratio '{col_b} / {col_c}'."
                            ))
                            break

                    # Percentage Ratio: A = (B / C) * 100
                    if np.all(np.abs(c_vals) > 1e-9):
                        pct_bc_err = np.max(np.abs(a_vals - ((b_vals / c_vals) * 100.0)))
                        if pct_bc_err <= tolerance:
                            seen_targets.add(col_a)
                            derivations.append(DerivationContract(
                                derivation_id=f"DERIV-{len(derivations)+1:04d}",
                                target_column=col_a,
                                derivation_type=DerivationType.DERIVED,
                                formula=f"{col_a} = ({col_b} / {col_c}) * 100",
                                source_columns=[col_b, col_c],
                                transformation_type="PERCENTAGE_RATIO",
                                relationship_type=RelationshipType.STRUCTURAL_RELATIONSHIP,
                                is_deterministic=True,
                                confidence=0.99,
                                description=f"'{col_a}' is derived as the percentage '({col_b} / {col_c}) * 100'."
                            ))
                            break

        return derivations

    @classmethod
    def check_deterministic_relationship(
        cls,
        df: pd.DataFrame,
        col_x: str,
        col_y: str,
        derivations: Optional[List[DerivationContract]] = None,
        tolerance: float = 1e-3
    ) -> Dict[str, Any]:
        """
        Evaluates whether two variables have a deterministic or structural relationship.
        Returns:
        {
            "is_deterministic": bool,
            "relationship_type": str,
            "classification": str,  # STRUCTURAL_RELATIONSHIP or STATISTICAL_ASSOCIATION
            "formula": Optional[str],
            "description": str
        }
        """
        cx = str(col_x).strip()
        cy = str(col_y).strip()

        if cx == cy:
            return {
                "is_deterministic": True,
                "relationship_type": "IDENTICAL",
                "classification": "STRUCTURAL_RELATIONSHIP",
                "formula": f"{cx} == {cy}",
                "description": f"'{cx}' is identical to itself."
            }

        # Check existing derivations
        if derivations:
            for d in derivations:
                target = d.target_column
                sources = d.source_columns
                if (target == cy and cx in sources) or (target == cx and cy in sources):
                    return {
                        "is_deterministic": True,
                        "relationship_type": str(d.derivation_type),
                        "classification": "STRUCTURAL_RELATIONSHIP",
                        "formula": d.formula,
                        "description": f"'{target}' is a deterministic transformation of '{sources[0] if len(sources)==1 else sources}'."
                    }

        # Live pairwise check between col_x and col_y if numeric
        if cx in df.columns and cy in df.columns:
            if pd.api.types.is_numeric_dtype(df[cx]) and pd.api.types.is_numeric_dtype(df[cy]):
                pair_df = df[[cx, cy]].apply(pd.to_numeric, errors='coerce').dropna()
                if len(pair_df) >= 5 and pair_df[cx].std() > 1e-9 and pair_df[cy].std() > 1e-9:
                    x_vals = pair_df[cx].values
                    y_vals = pair_df[cy].values

                    # Identity
                    if np.all(np.abs(x_vals - y_vals) <= tolerance):
                        return {
                            "is_deterministic": True,
                            "relationship_type": "DETERMINISTIC_TRANSFORMATION",
                            "classification": "STRUCTURAL_RELATIONSHIP",
                            "formula": f"{cx} == {cy}",
                            "description": f"'{cx}' and '{cy}' have identical values."
                        }

                    # Ratio scaling Y = k * X
                    mask_x = np.abs(x_vals) > 1e-9
                    if np.sum(mask_x) >= 5:
                        ratios = y_vals[mask_x] / x_vals[mask_x]
                        k = float(np.median(ratios))
                        if np.max(np.abs(y_vals - k * x_vals)) <= tolerance:
                            k_str = f"{int(round(k))}" if abs(k - round(k)) < 1e-4 else f"{k:.4g}"
                            return {
                                "is_deterministic": True,
                                "relationship_type": "DETERMINISTIC_TRANSFORMATION",
                                "classification": "STRUCTURAL_RELATIONSHIP",
                                "formula": f"{cy} = {cx} * {k_str}",
                                "description": f"'{cy}' is a deterministic scaling of '{cx}'."
                            }

                    # Perfect linear correlation (|r| >= 0.9999) with exact fit
                    corr = float(pair_df[cx].corr(pair_df[cy]))
                    if abs(abs(corr) - 1.0) <= 1e-4:
                        slope, intercept = np.polyfit(x_vals, y_vals, 1)
                        if np.max(np.abs(y_vals - (slope * x_vals + intercept))) <= tolerance:
                            return {
                                "is_deterministic": True,
                                "relationship_type": "DETERMINISTIC_TRANSFORMATION",
                                "classification": "STRUCTURAL_RELATIONSHIP",
                                "formula": f"{cy} = {slope:.4g} * {cx} + {intercept:.4g}",
                                "description": f"'{cy}' is an exact deterministic transformation of '{cx}'."
                            }

        return {
            "is_deterministic": False,
            "relationship_type": "INDEPENDENT",
            "classification": "STATISTICAL_ASSOCIATION",
            "formula": None,
            "description": f"No deterministic structural relationship detected between '{cx}' and '{cy}'."
        }
