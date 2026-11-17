import re
import difflib
from typing import Dict, List, Optional, Tuple, Any, Set
import pandas as pd

from engine.contracts.chat_contracts import ResolvedEntity, ResolutionStatus, ColumnFilter, FilterOperator
from engine.contracts.dataset_contract import DatasetContract
from engine.matcher import IntelligentFieldMatcher, DynamicAliasDictionary


def filter_prefixed_candidates(cand_names: List[str], query_low: str) -> List[str]:
    """
    If candidate columns include both an unprefixed primary column ('Event ID')
    and joined prefixed column ('24-25_Event ID'), prefer the base column unless
    the query explicitly references the prefix.
    """
    if len(cand_names) <= 1:
        return cand_names
    unprefixed = []
    for c in cand_names:
        match = re.match(r'^[a-zA-Z0-9_\-]+\s*_(.+)$', c)
        if match:
            base = match.group(1).strip()
            if base in cand_names:
                prefix = c[:match.start(1)-1].lower()
                if prefix not in query_low:
                    continue
        unprefixed.append(c)
    return unprefixed if unprefixed else cand_names


FILTER_VALUE_STOPWORDS: Set[str] = {
    'count', 'counts', 'sum', 'sums', 'avg', 'average', 'mean', 'median',
    'min', 'minimum', 'max', 'maximum', 'std', 'stddev', 'variance',
    'total', 'totals', 'distribution', 'trend', 'trends', 'rate', 'ratio',
    'by', 'per', 'across', 'each', 'all', 'every', 'and', 'or', 'vs', 'versus',
    'where', 'with', 'in', 'on', 'at', 'to', 'from', 'for', 'of', 'over', 'under',
    'between', 'rows', 'row', 'records', 'record', 'data', 'dataset', 'table',
    'chart', 'plot', 'show', 'list', 'display', 'get', 'give', 'tell', 'find',
    'which', 'who', 'what', 'when', 'why', 'how', 'many', 'much', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
    'did', 'the', 'a', 'an', 'this', 'that', 'these', 'those', 'there',
    'column', 'columns', 'value', 'values', 'metric', 'dimension', 'group',
    'groups', 'grouped', 'breakdown', 'split', 'categorized', 'classified',
    'top', 'bottom', 'highest', 'lowest', 'rank', 'ranking', 'outlier', 'outliers',
    'difference', 'significant', 'relationship', 'correlation', 'overview',
    'summary', 'summarize', 'clean', 'null', 'nulls', 'missing', 'duplicate',
    'duplicates', 'unique', 'distinct', 'more', 'less', 'greater', 'than'
}


ANALYTICAL_INTENT_STOPWORDS: Set[str] = {
    'what', 'is', 'are', 'was', 'were', 'the', 'of', 'in', 'for', 'by', 'and', 'or', 'to', 'from',
    'with', 'show', 'give', 'tell', 'me', 'average', 'mean', 'sum', 'total',
    'count', 'max', 'maximum', 'min', 'minimum', 'top', 'bottom', 'how', 'many',
    'dataset', 'table', 'data', 'which', 'who', 'where', 'when', 'does', 'do',
    'calculate', 'find', 'get', 'list', 'distribution', 'correlation',
    'trend', 'relationship', 'outliers', 'highest', 'lowest', 'highest-risk',
    'rate', 'ratio', 'value', 'values', 'column', 'columns', 'cases', 'case',
    'records', 'record', 'rows', 'row', 'has', 'have', 'had', 'having',
    'between', 'versus', 'vs', 'per', 'across', 'wise', 'analysis', 'compare'
}


class UniversalSchemaAndSemanticResolver:
    """
    Universal Schema and Semantic Resolution Engine for Tabular Datasets.
    
    Binds directly to the authoritative DatasetContract, Data Dictionary, and active DataFrame.
    Dynamically maps natural language user terms, synonyms, abbreviations, and cell values
    into schema columns and filter candidates with strict ambiguity detection.
    """

    def __init__(
        self,
        contract: Optional[DatasetContract] = None,
        dictionary: Optional[List[Dict[str, Any]]] = None,
        df: Optional[pd.DataFrame] = None,
        sem_layer_dict: Optional[Dict[str, Any]] = None
    ):
        self.contract = contract
        self.dictionary = dictionary or []
        self.df = df
        self.sem_layer_dict = sem_layer_dict or {}

        # Core Index structures
        self.columns: List[str] = []
        self.column_map: Dict[str, str] = {}         # lower_clean -> actual_name
        self.norm_map: Dict[str, str] = {}           # alphanumeric -> actual_name
        self.col_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Categorized Column Sets
        self.measures: List[str] = []
        self.dimensions: List[str] = []
        self.time_cols: List[str] = []
        self.id_cols: List[str] = []
        self.primary_keys: List[str] = []
        self.foreign_keys: List[str] = []

        # Value-level Inverted Index for Categorical Entities (e.g., 'Germany' -> 'Country')
        self.value_index: Dict[str, Tuple[str, Any]] = {}

        self._build_indexes()

    def _build_indexes(self):
        """Constructs schema indexes from contract, dictionary, or dataframe."""
        # 1. Populate column names and metadata
        if self.contract and hasattr(self.contract, "columns") and self.contract.columns:
            for col_name, col_contract in self.contract.columns.items():
                c_str = str(col_name)
                self.columns.append(c_str)
                meta = col_contract.to_dict() if hasattr(col_contract, "to_dict") else dict(col_contract)
                self.col_metadata[c_str] = meta
        elif self.dictionary:
            for item in self.dictionary:
                col_name = str(item.get("column_name") or item.get("name") or "")
                if col_name:
                    self.columns.append(col_name)
                    self.col_metadata[col_name] = item
        elif self.sem_layer_dict:
            for col_name, meta in self.sem_layer_dict.items():
                c_str = str(col_name)
                self.columns.append(c_str)
                self.col_metadata[c_str] = meta
        elif self.df is not None:
            for c in self.df.columns:
                c_str = str(c)
                self.columns.append(c_str)
                self.col_metadata[c_str] = {
                    "physical_dtype": str(self.df[c].dtype),
                    "semantic_role": "Measure" if pd.api.types.is_numeric_dtype(self.df[c]) else "Dimension"
                }

        # Deduplicate columns preserving order and collapsing whitespace
        cleaned_columns = []
        seen_col_norms = {}
        for c in self.columns:
            c_str = re.sub(r'\s+', ' ', str(c)).strip()
            norm_key = re.sub(r'[^a-z0-9]+', '', c_str.lower())
            if norm_key not in seen_col_norms:
                seen_col_norms[norm_key] = c
                cleaned_columns.append(c)
            elif c in self.col_metadata and seen_col_norms[norm_key] not in self.col_metadata:
                self.col_metadata[seen_col_norms[norm_key]] = self.col_metadata[c]
        self.columns = cleaned_columns

        # 2. Build normalized mappings and role partitions
        for c in self.columns:
            c_clean = re.sub(r'\s+', ' ', c).strip()
            c_low = c_clean.lower()
            norm = re.sub(r'[^a-z0-9]+', '', c_low)
            self.column_map[c_low] = c
            self.column_map[c.lower().strip()] = c
            self.norm_map[norm] = c

            meta = self.col_metadata.get(c, {})
            role = str(meta.get("semantic_role", meta.get("role", "")))
            canonical_role = str(meta.get("canonical_role", ""))
            measure_type = str(meta.get("measure_type", meta.get("semantic_subtype", "")))
            key_role = str(meta.get("key_role", meta.get("relational_role", "")))

            is_pk = (
                meta.get("is_primary_key", False)
                or key_role in ["PRIMARY_KEY", "PRIMARY KEY"]
                or role == "Primary Key"
            )
            is_fk = (
                meta.get("is_foreign_key", False)
                or key_role in ["FOREIGN_KEY", "FOREIGN KEY"]
                or role == "Foreign Key"
            )
            is_id = (
                meta.get("is_identifier", False)
                or is_pk or is_fk
                or role in ["Identifier", "ID", "IDENTIFIER"]
                or canonical_role == "IDENTIFIER"
            )
            is_meas = (
                meta.get("is_measure", False)
                or role in ["Measure", "Target/Outcome", "NUMERIC MEASURE"]
                or canonical_role == "NUMERIC MEASURE"
            )
            is_time = (
                meta.get("is_time", False)
                or role in ["Date/Time", "Time", "TEMPORAL"]
                or canonical_role == "TEMPORAL"
            )
            is_dim = (
                meta.get("is_dimension", False)
                or role in ["Dimension", "Geographic", "Boolean", "CATEGORICAL DIMENSION", "Category"]
                or canonical_role == "CATEGORICAL DIMENSION"
            )

            # Fallback based on physical data type if no roles populated
            if not is_meas and not is_dim and not is_time and not is_id:
                dtype_str = str(meta.get("physical_dtype", meta.get("data_type", ""))).lower()
                if any(t in dtype_str for t in ["int", "float", "double", "decimal"]):
                    is_meas = True
                elif any(t in dtype_str for t in ["date", "time", "timestamp"]):
                    is_time = True
                else:
                    is_dim = True

            if is_pk:
                self.primary_keys.append(c)
            if is_fk:
                self.foreign_keys.append(c)
            if is_id:
                self.id_cols.append(c)
            if is_meas and not is_id:
                self.measures.append(c)
            if is_time and not is_id:
                self.time_cols.append(c)
            if is_dim and not is_id and not is_meas:
                self.dimensions.append(c)

        # 3. Build Categorical Value Inverted Index and Column Value Map
        self.col_value_map: Dict[str, Dict[str, Any]] = {}
        if self.df is not None and not self.df.empty:
            for c in self.df.columns:
                s = self.df[c].dropna()
                if len(s) > 0 and s.nunique() <= 1000:
                    self.col_value_map[c.lower()] = {str(v).strip().lower(): v for v in s.unique()}

            for dim_col in (self.dimensions + self.id_cols):
                if dim_col in self.df.columns:
                    s = self.df[dim_col].dropna()
                    if len(s) > 0 and s.nunique() <= 500:
                        unique_vals = s.unique()
                        for u_val in unique_vals:
                            u_str = str(u_val).strip()
                            if len(u_str) >= 2 and not u_str.isnumeric():
                                u_low = u_str.lower()
                                if u_low not in self.column_map:
                                    self.value_index[u_low] = (dim_col, u_val)

        # 4. Initialize Multi-Strategy Intelligent Field Matcher
        self.matcher = IntelligentFieldMatcher(
            columns=self.columns,
            dictionary=self.dictionary,
            col_metadata=self.col_metadata
        )

    def resolve_term(
        self,
        term: str,
        preferred_role: Optional[str] = None
    ) -> ResolvedEntity:
        """
        Resolves a single search term or natural language concept to a schema column.
        Enforces explicit ambiguity detection and confidence weighting.
        """
        if not term or not term.strip():
            return ResolvedEntity(requested_term=term, status=ResolutionStatus.NOT_FOUND)

        t_clean = re.sub(r'\s+', ' ', term).strip()
        t_low = t_clean.lower()
        t_norm = re.sub(r'[^a-z0-9]+', '', t_low)

        # Check if term is entirely composed of analytical/intent stopwords (e.g. 'what is the correlation', 'average', 'between')
        words_in_term = [w.lower() for w in re.findall(r'[a-zA-Z0-9]+', t_low)]
        if words_in_term and all(w in ANALYTICAL_INTENT_STOPWORDS for w in words_in_term):
            if t_low not in self.column_map and t_norm not in self.norm_map:
                return ResolvedEntity(requested_term=term, status=ResolutionStatus.NOT_FOUND)

        # 1. Exact Column Name Match
        if t_low in self.column_map:
            col = self.column_map[t_low]
            if self._matches_role(col, preferred_role):
                return self._create_resolved_entity(term, col, ResolutionStatus.EXACT_MATCH, 1.0, "EXACT_NAME")

        # 2. Normalized Column Match (handles spaces vs underscores)
        if t_norm in self.norm_map:
            col = self.norm_map[t_norm]
            if self._matches_role(col, preferred_role):
                return self._create_resolved_entity(term, col, ResolutionStatus.EXACT_MATCH, 0.98, "NORMALIZED_NAME")

        # 3. Substring & Word Boundary Match
        candidates: List[Tuple[str, float]] = []
        for c_low, col in self.column_map.items():
            if not self._matches_role(col, preferred_role):
                continue
            if t_low == c_low:
                continue
            words = c_low.replace('_', ' ').split()
            if t_low in words:
                candidates.append((col, 0.90))
            elif t_low in c_low and len(t_low) >= 3:
                candidates.append((col, 0.85))

        if len(candidates) == 1:
            col, conf = candidates[0]
            return self._create_resolved_entity(term, col, ResolutionStatus.SEMANTIC_MATCH, conf, "SUBSTRING_MATCH")
        elif len(candidates) > 1:
            # Ambiguity Check: Multiple matching columns (e.g. Risk_Score vs Risk_Index_Pct)
            cand_names = list(dict.fromkeys([c[0] for c in candidates]))

            # Disambiguation Check A: Do all candidates normalize to the exact same column name?
            cand_norms = {re.sub(r'[^a-z0-9]+', '', c.lower()): c for c in cand_names}
            if len(cand_norms) == 1:
                return self._create_resolved_entity(
                    term, cand_names[0], ResolutionStatus.SEMANTIC_MATCH, 0.92, "NORMALIZED_DEDUPLICATION"
                )

            # Disambiguation Check B: Filter out table-prefixed duplicates if base column present
            cand_filtered = filter_prefixed_candidates(cand_names, t_low)
            if len(cand_filtered) == 1:
                return self._create_resolved_entity(
                    term, cand_filtered[0], ResolutionStatus.SEMANTIC_MATCH, 0.92, "BASE_COLUMN_MATCH",
                    alternatives=[c for c in cand_names if c != cand_filtered[0]]
                )

            # Disambiguation Check C: Does one candidate match the term without table prefix (e.g. 'Description of Incident' vs '24-25_Description of Incident')?
            exact_unprefixed = [c for c in cand_names if re.sub(r'^[a-zA-Z0-9_\-]+_', '', c).lower() == t_low or re.sub(r'[^a-z0-9]+', '', re.sub(r'^[a-zA-Z0-9_\-]+_', '', c).lower()) == t_norm]
            if len(exact_unprefixed) == 1:
                return self._create_resolved_entity(
                    term, exact_unprefixed[0], ResolutionStatus.SEMANTIC_MATCH, 0.95, "UNPREFIXED_EXACT_MATCH",
                    alternatives=[c for c in cand_names if c != exact_unprefixed[0]]
                )

            # Disambiguate if exact word matches one column exactly
            exact_word = [c for c in cand_filtered if t_low in re.findall(r'[a-zA-Z0-9]+', c.lower())]
            if len(exact_word) == 1:
                return self._create_resolved_entity(
                    term, exact_word[0], ResolutionStatus.SEMANTIC_MATCH, 0.90, "WORD_DISAMBIGUATION",
                    alternatives=[c for c in cand_filtered if c != exact_word[0]]
                )
            return ResolvedEntity(
                requested_term=term,
                status=ResolutionStatus.AMBIGUOUS,
                confidence=0.5,
                candidate_alternatives=cand_filtered,
                resolution_method="AMBIGUOUS_SUBSTRING"
            )

        # 4. Domain & Cross-Industry Business Synonyms
        synonym_match = self._match_by_synonyms(t_low, preferred_role)
        if synonym_match:
            col, conf, method = synonym_match
            return self._create_resolved_entity(term, col, ResolutionStatus.SEMANTIC_MATCH, conf, method)

        # 5. Fuzzy String Similarity (SequenceMatcher)
        target_cols = self._get_candidate_cols_for_role(preferred_role)
        fuzzy_matches: List[Tuple[str, float]] = []
        for col in target_cols:
            c_low = col.lower()
            ratio = difflib.SequenceMatcher(None, t_low, c_low).ratio()
            # Partial word ratio
            words = c_low.replace('_', ' ').split()
            for w in words:
                w_ratio = difflib.SequenceMatcher(None, t_low, w).ratio()
                if w_ratio > ratio:
                    ratio = w_ratio
            # Prevent short words (<=4 chars, e.g. 'lost', 'cost') from false fuzzy collisions with schema columns (e.g. 'check list')
            min_thresh = 0.88 if len(t_low) <= 4 else 0.75
            if ratio >= min_thresh:
                fuzzy_matches.append((col, round(ratio * 0.9, 2)))

        if len(fuzzy_matches) == 1:
            col, conf = fuzzy_matches[0]
            return self._create_resolved_entity(term, col, ResolutionStatus.FUZZY_MATCH, conf, "FUZZY_SIMILARITY")
        elif len(fuzzy_matches) > 1:
            fuzzy_matches.sort(key=lambda x: x[1], reverse=True)
            cand_names = list(dict.fromkeys([f[0] for f in fuzzy_matches]))

            # Disambiguation Check A: Do all candidates normalize to the exact same column name?
            cand_norms = {re.sub(r'[^a-z0-9]+', '', c.lower()): c for c in cand_names}
            if len(cand_norms) == 1:
                return self._create_resolved_entity(
                    term, cand_names[0], ResolutionStatus.FUZZY_MATCH, fuzzy_matches[0][1], "NORMALIZED_DEDUPLICATION"
                )

            # Disambiguation Check B: Filter out table-prefixed duplicates if base column present
            cand_filtered = filter_prefixed_candidates(cand_names, t_low)
            if len(cand_filtered) == 1:
                return self._create_resolved_entity(
                    term, cand_filtered[0], ResolutionStatus.FUZZY_MATCH, fuzzy_matches[0][1], "BASE_COLUMN_MATCH",
                    alternatives=[c for c in cand_names if c != cand_filtered[0]]
                )

            # Disambiguation Check C: Does one candidate match the term without table prefix?
            exact_unprefixed = [c for c in cand_names if re.sub(r'^[a-zA-Z0-9_\-]+_', '', c).lower() == t_low or re.sub(r'[^a-z0-9]+', '', re.sub(r'^[a-zA-Z0-9_\-]+_', '', c).lower()) == t_norm]
            if len(exact_unprefixed) == 1:
                return self._create_resolved_entity(
                    term, exact_unprefixed[0], ResolutionStatus.FUZZY_MATCH, 0.92, "UNPREFIXED_FUZZY_MATCH",
                    alternatives=[c for c in cand_names if c != exact_unprefixed[0]]
                )

            # Disambiguate if exact word matches one column exactly
            exact_word = [c for c in cand_filtered if t_low in re.findall(r'[a-zA-Z0-9]+', c.lower())]
            if len(exact_word) == 1:
                return self._create_resolved_entity(
                    term, exact_word[0], ResolutionStatus.FUZZY_MATCH, 0.90, "WORD_DISAMBIGUATION",
                    alternatives=[c for c in cand_filtered if c != exact_word[0]]
                )

            top_col, top_conf = fuzzy_matches[0]
            second_conf = fuzzy_matches[1][1]
            if top_conf - second_conf >= 0.10:
                return self._create_resolved_entity(
                    term, top_col, ResolutionStatus.FUZZY_MATCH, top_conf, "FUZZY_BEST_MATCH",
                    alternatives=[f[0] for f in fuzzy_matches[1:]]
                )
            return ResolvedEntity(
                requested_term=term,
                status=ResolutionStatus.AMBIGUOUS,
                confidence=top_conf,
                candidate_alternatives=cand_filtered,
                resolution_method="FUZZY_AMBIGUOUS"
            )

        # 6. Data Dictionary Description Match
        stopwords = {'for', 'the', 'and', 'with', 'from', 'this', 'dataset', 'column', 'measure', 'each', 'all'}
        if t_low not in stopwords and len(t_low) >= 4:
            for col, meta in self.col_metadata.items():
                if not self._matches_role(col, preferred_role):
                    continue
                desc = str(meta.get("description", meta.get("business_meaning", ""))).lower()
                if t_low in desc:
                    return self._create_resolved_entity(term, col, ResolutionStatus.SEMANTIC_MATCH, 0.72, "DICTIONARY_DESCRIPTION")

        # 7. Intelligent Multi-Strategy Matcher (Dynamic Aliases, Abbreviations, RapidFuzz, TF-IDF n-grams)
        if hasattr(self, "matcher") and self.matcher:
            matcher_candidates = self.matcher.find_matches(term, preferred_role=preferred_role, top_k=4)
            if matcher_candidates:
                top_m = matcher_candidates[0]
                if top_m.confidence >= 0.90:
                    # Check ambiguity among top matcher candidates
                    if len(matcher_candidates) > 1 and (top_m.confidence - matcher_candidates[1].confidence) < 0.05 and top_m.confidence < 0.98:
                        cand_names = [m.column for m in matcher_candidates[:4]]
                        cand_filtered = filter_prefixed_candidates(cand_names, t_low)
                        if len(cand_filtered) > 1:
                            return ResolvedEntity(
                                requested_term=term,
                                status=ResolutionStatus.AMBIGUOUS,
                                confidence=top_m.confidence,
                                candidate_alternatives=cand_filtered,
                                resolution_method="MATCHER_AMBIGUOUS"
                            )
                    m_status = ResolutionStatus.EXACT_MATCH if top_m.confidence >= 0.98 else (
                        ResolutionStatus.SEMANTIC_MATCH if top_m.match_type in ["ALIAS", "SUBSTRING", "SUB_COMPONENT", "SEMANTIC"] else ResolutionStatus.FUZZY_MATCH
                    )
                    return self._create_resolved_entity(
                        term, top_m.column, m_status, top_m.confidence, f"MATCHER_{top_m.match_type}",
                        alternatives=[m.column for m in matcher_candidates[1:4]]
                    )
                elif top_m.confidence >= 0.70:
                    cand_names = [m.column for m in matcher_candidates[:4]]
                    return ResolvedEntity(
                        requested_term=term,
                        status=ResolutionStatus.AMBIGUOUS,
                        confidence=top_m.confidence,
                        candidate_alternatives=cand_names,
                        resolution_method="MATCHER_MEDIUM_CONFIDENCE"
                    )

        return ResolvedEntity(requested_term=term, status=ResolutionStatus.NOT_FOUND)

    def resolve_value_entity(self, term: str) -> Optional[Tuple[str, Any]]:
        """
        Checks if a user term corresponds to an exact categorical value in the dataset.
        Example: 'workshop' -> ('Plant_Zone', 'Workshop')
        """
        if not term:
            return None
        t_low = term.strip().lower()
        if t_low in self.value_index:
            return self.value_index[t_low]
        # Partial match on multi-word entities
        for val_low, binding in self.value_index.items():
            if val_low == t_low or (len(t_low) >= 4 and t_low in val_low.split()):
                return binding
        return None

    def resolve_column(self, term: str, role_filter: Optional[str] = None) -> Optional[Tuple[str, float]]:
        """
        Backward-compatible method matching existing SemanticLayer interface.
        Returns: (column_name, confidence_score) or None
        """
        res = self.resolve_term(term, preferred_role=role_filter)
        if res.status in [ResolutionStatus.EXACT_MATCH, ResolutionStatus.SEMANTIC_MATCH, ResolutionStatus.FUZZY_MATCH]:
            if res.resolved_column:
                return res.resolved_column, res.confidence
        return None

    def _create_resolved_entity(
        self,
        term: str,
        col: str,
        status: ResolutionStatus,
        confidence: float,
        method: str,
        alternatives: Optional[List[str]] = None
    ) -> ResolvedEntity:
        meta = self.col_metadata.get(col, {})
        role = str(meta.get("semantic_role", meta.get("role", "")))
        m_type = str(meta.get("measure_type", meta.get("semantic_subtype", "")))
        d_type = str(meta.get("physical_dtype", meta.get("data_type", "")))

        is_meas = col in self.measures or role == "Measure"
        is_dim = col in self.dimensions or role == "Dimension"
        is_time = col in self.time_cols or role in ["Date/Time", "Time"]
        is_id = col in self.id_cols or role == "Identifier"

        return ResolvedEntity(
            requested_term=term,
            resolved_column=col,
            canonical_identifier=meta.get("column_id", col),
            semantic_role=role,
            measure_type=m_type,
            data_type=d_type,
            status=status,
            confidence=confidence,
            candidate_alternatives=alternatives or [],
            resolution_method=method,
            is_measure=is_meas,
            is_dimension=is_dim,
            is_identifier=is_id,
            is_temporal=is_time
        )

    def _matches_role(self, col: str, preferred_role: Optional[str]) -> bool:
        if not preferred_role:
            return True
        r = preferred_role.lower()
        if r in ["measure", "metric"]:
            return col in self.measures and col not in self.id_cols
        elif r in ["dimension", "category"]:
            return col in self.dimensions or col in self.id_cols or col in self.time_cols
        elif r in ["time", "temporal", "date"]:
            return col in self.time_cols and col not in self.id_cols
        elif r in ["id", "identifier", "key"]:
            return col in self.id_cols or col in self.primary_keys or col in self.foreign_keys
        return True

    def _get_candidate_cols_for_role(self, preferred_role: Optional[str]) -> List[str]:
        if not preferred_role:
            return self.columns
        r = preferred_role.lower()
        if r in ["measure", "metric"]:
            return self.measures
        elif r in ["dimension", "category"]:
            return self.dimensions + self.id_cols
        elif r in ["time", "temporal"]:
            return self.time_cols
        elif r in ["id", "identifier"]:
            return self.id_cols
        return self.columns

    def _match_by_synonyms(
        self,
        term: str,
        preferred_role: Optional[str]
    ) -> Optional[Tuple[str, float, str]]:
        """Matches domain concepts across Finance, Operations, Safety, HR, Logistics."""
        concept_synonyms: Dict[str, List[str]] = {
            # Financial Measures
            "REVENUE": ['revenue', 'sales', 'turnover', 'income', 'gmv', 'billings', 'receipts', 'topline'],
            "PROFIT": ['profit', 'earnings', 'ebitda', 'net_income', 'margin_amount'],
            "COST": ['cost', 'expense', 'expenses', 'spend', 'spending', 'budget', 'operating_cost', 'cogs', 'freight'],
            "PRICE": ['price', 'unit_price', 'rate', 'tariff', 'fee', 'charge'],
            "DISCOUNT": ['discount', 'rebate', 'deduction', 'allowance', 'concession'],

            # Statistical / Percentage / Ratio Measures
            "PERCENTAGE": ['percentage', 'percent', 'pct', 'ratio', 'share', 'proportion', 'growth', 'yield', 'conversion', 'rate'],

            # Ratings, Performance & Risk
            "RATING": ['rating', 'score', 'satisfaction', 'nps', 'feedback', 'grade', 'evaluation', 'performance', 'mark', 'points'],
            "RISK": ['risk', 'hazard', 'severity', 'likelihood', 'incident_rate', 'risk_index'],

            # Temporal & Duration
            "DURATION": ['duration', 'tenure', 'hours', 'exposure_hours', 'days', 'time_taken', 'lead_time', 'delay', 'transit_time', 'cycle_time'],
            "TEMPORAL": ['date', 'time', 'timestamp', 'period', 'month', 'year', 'day', 'quarter', 'week', 'orderdate'],

            # Operational & Volume Measures
            "QUANTITY": ['qty', 'quantity', 'units', 'volume', 'count', 'items', 'headcount', 'batch_size', 'mileage', 'distance', 'weight', 'capacity'],

            # Entities / Dimensions
            "CUSTOMER": ['customer', 'client', 'buyer', 'consumer', 'account', 'user', 'patron'],
            "PRODUCT": ['product', 'item', 'sku', 'merchandise', 'article', 'good'],
            "EMPLOYEE": ['employee', 'worker', 'staff', 'personnel', 'agent', 'technician', 'operator'],
            "LOCATION": ['location', 'region', 'country', 'city', 'state', 'plant', 'zone', 'site', 'facility', 'warehouse', 'office', 'branch'],
            "CATEGORY": ['category', 'segment', 'department', 'channel', 'status', 'type', 'class', 'group', 'division', 'role', 'carrier', 'dept']
        }

        # 1. Identify which concepts match the user term
        matching_concepts: List[Tuple[str, List[str]]] = []
        for concept, kws in concept_synonyms.items():
            for kw in kws:
                if kw == term or re.search(rf'\b{re.escape(kw)}\b', term):
                    matching_concepts.append((concept, kws))
                    break

        if not matching_concepts:
            return None

        # 2. Score candidate columns across the matching concepts
        candidate_cols = self._get_candidate_cols_for_role(preferred_role)
        scored_candidates: List[Tuple[str, float, str]] = []

        for col in candidate_cols:
            if not self._matches_role(col, preferred_role):
                continue
            meta = self.col_metadata.get(col, {})
            c_low = col.lower()
            c_words = set(c_low.replace('_', ' ').split())
            subtype = str(meta.get("semantic_subtype", meta.get("measure_type", ""))).upper()

            for concept, kws in matching_concepts:
                # Direct word match between column name words and concept keywords
                matched_kws = c_words.intersection(set(kws))
                if matched_kws:
                    # Give higher confidence if the exact user term or primary keyword matched
                    conf = 0.90 if term in matched_kws else 0.84
                    scored_candidates.append((col, conf, f"SYNONYM_{concept}"))
                elif subtype == concept or (concept == "PERCENTAGE" and subtype == "PERCENTAGE"):
                    scored_candidates.append((col, 0.82, f"SUBTYPE_{concept}"))

        if not scored_candidates:
            return None

        # Deduplicate and sort by confidence
        unique_matches: Dict[str, Tuple[float, str]] = {}
        for col, conf, method in scored_candidates:
            if col not in unique_matches or conf > unique_matches[col][0]:
                unique_matches[col] = (conf, method)

        sorted_cand = sorted(unique_matches.items(), key=lambda x: x[1][0], reverse=True)
        top_col, (top_conf, top_method) = sorted_cand[0]

        if len(sorted_cand) > 1:
            second_conf = sorted_cand[1][1][0]
            if top_conf - second_conf < 0.05:
                # Ambiguous synonym match among multiple columns
                return None  # Will fall through or trigger ambiguity
        return top_col, top_conf, top_method

    def extract_filters_from_query(self, query: str) -> List[ColumnFilter]:
        """
        Parses filter conditions from a natural language query.
        Handles:
        1. Categorical cell value mentions (e.g. 'workshop' -> Plant_Zone == 'Workshop')
        2. Relational comparison expressions (e.g. 'risk score > 15', 'exposure hours <= 20')
        3. Between ranges (e.g. 'between 10 and 20')
        """
        if not query:
            return []

        filters: List[ColumnFilter] = []
        covered_spans: List[Tuple[int, int]] = []
        q_low = query.lower()

        # 1. Relational comparison patterns: <column/concept> (>=|<=|!=|>|<|==|=) <num>
        op_map = {
            ">=": FilterOperator.GREATER_EQUAL,
            "<=": FilterOperator.LESS_EQUAL,
            "!=": FilterOperator.NOT_EQUALS,
            "==": FilterOperator.EQUALS,
            "=": FilterOperator.EQUALS,
            ">": FilterOperator.GREATER_THAN,
            "<": FilterOperator.LESS_THAN,
        }
        comp_pattern = re.compile(
            r'(?:\b(?:where|and|or|records|record|filter|having|with|for|when)\b|\A|,|;)\s*([a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+){0,2})\s*(>=|<=|!=|==|>|<|=)\s*([0-9]+(?:\.[0-9]+)?)',
            re.IGNORECASE
        )
        for m in comp_pattern.finditer(query):
            col_expr = m.group(1).strip()
            while True:
                cleaned = re.sub(r'^(?:where|and|or|for|with|filter|records|record|if|having|whose|that|the)\s+', '', col_expr, flags=re.IGNORECASE)
                if cleaned == col_expr:
                    break
                col_expr = cleaned

            op_sym = m.group(2)
            val_str = m.group(3)
            res = self.resolve_term(col_expr)
            if res.status in [ResolutionStatus.EXACT_MATCH, ResolutionStatus.SEMANTIC_MATCH, ResolutionStatus.FUZZY_MATCH] and res.resolved_column:
                val = float(val_str) if '.' in val_str else int(val_str)
                filters.append(ColumnFilter(
                    column=res.resolved_column,
                    operator=op_map.get(op_sym, FilterOperator.EQUALS),
                    value=val,
                    raw_predicate_text=m.group(0).strip()
                ))
                covered_spans.append((m.span(1)[0], m.end()))

        # 2. BETWEEN expressions: <col> between <num1> and <num2>
        between_pattern = re.compile(
            r'([a-zA-Z0-9_\s]{2,30}?)\s+between\s+([0-9]+(?:\.[0-9]+)?)\s+and\s+([0-9]+(?:\.[0-9]+)?)',
            re.IGNORECASE
        )
        for m in between_pattern.finditer(query):
            col_expr = m.group(1).strip()
            v1_str, v2_str = m.group(2), m.group(3)
            res = self.resolve_term(col_expr)
            if res.status in [ResolutionStatus.EXACT_MATCH, ResolutionStatus.SEMANTIC_MATCH, ResolutionStatus.FUZZY_MATCH] and res.resolved_column:
                v1 = float(v1_str) if '.' in v1_str else int(v1_str)
                v2 = float(v2_str) if '.' in v2_str else int(v2_str)
                filters.append(ColumnFilter(
                    column=res.resolved_column,
                    operator=FilterOperator.BETWEEN,
                    value=v1,
                    secondary_value=v2,
                    raw_predicate_text=m.group(0)
                ))
                covered_spans.append((m.start(), m.end()))

        # 2.5 Column-Anchored Filter Recognition (<column> [op/is/named/called/of/:/=] <value>)
        # Directly resolves patterns like "category 3", "category is 3", "category: 3", "category '3'"
        known_cols = sorted(self.column_map.items(), key=lambda x: len(x[0]), reverse=True)
        for col_alias, canonical_col in known_cols:
            if len(col_alias) < 2:
                continue
            pattern = re.compile(
                rf'\b{re.escape(col_alias)}\b\s*'
                r'(?:(>=|<=|!=|==|>|<|=)|(?:is|named|called|equals|equal\s+to|of|:)?)\s*'
                r'(?:"([^"]+)"|\'([^\']+)\'|([a-zA-Z0-9_\-\.\/]+))',
                re.IGNORECASE
            )
            for m in pattern.finditer(query):
                start, end = m.start(), m.end()
                if any(cs <= start and end <= ce for cs, ce in covered_spans):
                    continue
                op_sym = m.group(1)
                val_raw = m.group(2) or m.group(3) or m.group(4)
                if not val_raw or val_raw.lower() in FILTER_VALUE_STOPWORDS:
                    continue
                if val_raw.lower() in self.column_map:
                    continue

                col_low = canonical_col.lower()
                matched_val = None
                if col_low in self.col_value_map and val_raw.lower() in self.col_value_map[col_low]:
                    matched_val = self.col_value_map[col_low][val_raw.lower()]
                elif self.df is not None and canonical_col in self.df.columns:
                    s = self.df[canonical_col].dropna()
                    if pd.api.types.is_numeric_dtype(s):
                        try:
                            num_v = float(val_raw) if '.' in val_raw else int(val_raw)
                            matched_val = num_v
                        except (ValueError, TypeError):
                            pass

                if matched_val is not None:
                    op = op_map.get(op_sym, FilterOperator.EQUALS)
                    if not any(f.column == canonical_col and f.value == matched_val for f in filters):
                        filters.append(ColumnFilter(
                            column=canonical_col,
                            operator=op,
                            value=matched_val,
                            raw_predicate_text=query[start:end].strip()
                        ))
                        covered_spans.append((start, end))

        # 3. Categorical cell values from value_index
        # Sort values by length descending so multi-word phrases match before single-word substrings
        sorted_vals = sorted(self.value_index.items(), key=lambda x: len(x[0]), reverse=True)
        for val_low, (col, orig_val) in sorted_vals:
            if len(val_low) < 2:
                continue
            # Support plural/singular flexibility (e.g. 'lost day case' matches 'lost day cases')
            base_val = val_low[:-1] if val_low.endswith('s') and not val_low.endswith('ss') and len(val_low) > 3 else val_low
            pattern = rf'\b{re.escape(base_val)}(?:s|es)?\b'
            for m in re.finditer(pattern, q_low):
                start, end = m.start(), m.end()
                if any(cs <= start and end <= ce for cs, ce in covered_spans):
                    continue
                if not any(f.column == col and f.value == orig_val for f in filters):
                    filters.append(ColumnFilter(
                        column=col,
                        operator=FilterOperator.EQUALS,
                        value=orig_val,
                        raw_predicate_text=query[start:end]
                    ))
                    covered_spans.append((start, end))

        self.last_covered_spans = covered_spans
        return filters

    def extract_entities_from_query(self, query: str, covered_spans: Optional[List[Tuple[int, int]]] = None) -> Dict[str, Any]:
        """
        Extracts and classifies schema entities from free-text user questions.
        Returns a dictionary partitioned into measures, dimensions, temporal, identifiers, and ambiguous entities.
        Filters out tokens that fall within already covered filter spans.
        """
        if not query:
            return {"measures": [], "dimensions": [], "temporal": [], "identifiers": [], "ambiguous": []}

        if covered_spans is None:
            covered_spans = getattr(self, "last_covered_spans", [])
        active_covered_spans = list(covered_spans)

        words = re.findall(r'[a-zA-Z0-9_]+', query)
        stopwords = {
            'what', 'is', 'the', 'of', 'in', 'for', 'by', 'and', 'or', 'to', 'from',
            'with', 'show', 'give', 'tell', 'me', 'average', 'mean', 'sum', 'total',
            'count', 'max', 'maximum', 'min', 'minimum', 'top', 'bottom', 'how', 'many',
            'dataset', 'table', 'data', 'which', 'who', 'where', 'when', 'does', 'do',
            'are', 'calculate', 'find', 'get', 'list', 'distribution', 'correlation',
            'trend', 'relationship', 'outliers', 'highest', 'lowest', 'highest-risk',
            'rate', 'ratio', 'value', 'values', 'column', 'columns', 'cases', 'case',
            'records', 'record', 'rows', 'row', 'has', 'have', 'had', 'having',
            'wise', 'analysis', 'compare', 'comparison'
        }
        edge_stopwords = {'by', 'for', 'in', 'as', 'across', 'per', 'on', 'with', 'from', 'to', 'at', 'about', 'and', 'of', 'has', 'have', 'had', 'having', 'which', 'between', 'wise', 'analysis', 'compare'}

        # Track character spans of each word token to respect covered_spans
        token_spans: List[Tuple[int, int]] = []
        cur_idx = 0
        q_low = query.lower()
        for w in words:
            w_low = w.lower()
            found_idx = q_low.find(w_low, cur_idx)
            if found_idx != -1:
                token_spans.append((found_idx, found_idx + len(w)))
                cur_idx = found_idx + len(w)
            else:
                token_spans.append((cur_idx, cur_idx + len(w)))

        resolved_measures: Dict[str, ResolvedEntity] = {}
        resolved_dims: Dict[str, ResolvedEntity] = {}
        resolved_time: Dict[str, ResolvedEntity] = {}
        resolved_ids: Dict[str, ResolvedEntity] = {}
        ambiguous_entities: List[ResolvedEntity] = []

        n = len(words)
        for size in range(min(4, n), 0, -1):
            for i in range(n - size + 1):
                ngram_tokens = words[i:i+size]
                term = " ".join(ngram_tokens).strip()
                t_low = term.lower()
                if t_low in stopwords or len(t_low) < 2:
                    continue

                # Skip multi-word ngrams that start or end with an edge stopword or contain conjunctions unless exact column match
                if size > 1 and t_low not in self.column_map:
                    if ngram_tokens[0].lower() in edge_stopwords or ngram_tokens[-1].lower() in edge_stopwords:
                        continue
                    conjunction_markers = [' and ', ' or ', ' vs ', ' versus ', ' between ']
                    if any(m in f" {t_low} " for m in conjunction_markers):
                        continue

                # Check if this ngram overlaps with an already covered span
                if active_covered_spans and i < len(token_spans) and (i + size - 1) < len(token_spans):
                    t_start = token_spans[i][0]
                    t_end = token_spans[i + size - 1][1]
                    if any(max(t_start, cs) < min(t_end, ce) for cs, ce in active_covered_spans):
                        continue

                entity = self.resolve_term(term)
                if entity.status == ResolutionStatus.AMBIGUOUS:
                    clean_cands = filter_prefixed_candidates(
                        list(dict.fromkeys([c.strip() for c in entity.candidate_alternatives])),
                        q_low
                    )
                    cand_norms = {re.sub(r'[^a-z0-9]+', '', c.lower()): c for c in clean_cands}
                    if len(cand_norms) <= 1:
                        continue
                    entity.candidate_alternatives = clean_cands
                    if not any(a.requested_term.lower() == t_low for a in ambiguous_entities):
                        ambiguous_entities.append(entity)
                elif entity.status in [ResolutionStatus.EXACT_MATCH, ResolutionStatus.SEMANTIC_MATCH, ResolutionStatus.FUZZY_MATCH]:
                    col = entity.resolved_column
                    if not col:
                        continue
                    # Cover span for resolved entities to avoid false sub-ngram ambiguities
                    if entity.confidence >= 0.70 and i < len(token_spans) and (i + size - 1) < len(token_spans):
                        t_start = token_spans[i][0]
                        t_end = token_spans[i + size - 1][1]
                        active_covered_spans.append((t_start, t_end))

                    if entity.is_identifier:
                        if col not in resolved_ids or entity.confidence > resolved_ids[col].confidence:
                            resolved_ids[col] = entity
                    elif entity.is_temporal:
                        if col not in resolved_time or entity.confidence > resolved_time[col].confidence:
                            resolved_time[col] = entity
                    elif entity.is_measure:
                        if col not in resolved_measures or entity.confidence > resolved_measures[col].confidence:
                            resolved_measures[col] = entity
                    elif entity.is_dimension:
                        if col not in resolved_dims or entity.confidence > resolved_dims[col].confidence:
                            resolved_dims[col] = entity

        # Prune ambiguous entities that overlap with already resolved columns
        all_resolved_cols = {
            e.resolved_column for e in (
                list(resolved_measures.values()) + list(resolved_dims.values()) +
                list(resolved_time.values()) + list(resolved_ids.values())
            ) if e.resolved_column
        }
        filtered_ambiguous = []
        for ambig in ambiguous_entities:
            cands = set(ambig.candidate_alternatives)
            if cands.intersection(all_resolved_cols):
                continue
            ambig_t = ambig.requested_term.lower()
            if any(ambig_t in col.lower() for col in all_resolved_cols):
                continue
            filtered_ambiguous.append(ambig)
        ambiguous_entities = filtered_ambiguous

        # Supplementary Pass: If no measures, dimensions, or ambiguous candidates were found by sliding n-grams, query the IntelligentFieldMatcher on the whole query text
        if not resolved_measures and not resolved_dims and not ambiguous_entities and hasattr(self, "matcher") and self.matcher:
            full_matches = self.matcher.find_matches(query, top_k=3)
            for m in full_matches:
                if m.confidence >= 0.88:
                    col = m.column
                    entity = self._create_resolved_entity(
                        query, col,
                        ResolutionStatus.EXACT_MATCH if m.confidence >= 0.98 else ResolutionStatus.SEMANTIC_MATCH,
                        m.confidence, f"FULL_QUERY_{m.match_type}"
                    )
                    if entity.is_measure and col not in resolved_measures:
                        resolved_measures[col] = entity
                    elif (entity.is_dimension or entity.is_identifier) and col not in resolved_dims:
                        resolved_dims[col] = entity
                    elif entity.is_temporal and col not in resolved_time:
                        resolved_time[col] = entity

        return {
            "measures": sorted(list(resolved_measures.values()), key=lambda x: x.confidence, reverse=True),
            "dimensions": sorted(list(resolved_dims.values()), key=lambda x: x.confidence, reverse=True),
            "temporal": sorted(list(resolved_time.values()), key=lambda x: x.confidence, reverse=True),
            "identifiers": sorted(list(resolved_ids.values()), key=lambda x: x.confidence, reverse=True),
            "ambiguous": ambiguous_entities
        }

