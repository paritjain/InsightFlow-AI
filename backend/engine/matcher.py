"""
Intelligent Semantic Matching & Fuzzy Query Resolution Engine.

Implements robust, multi-strategy column and field resolution using:
- Exact & case-insensitive matching
- Partial and token subset matching
- Fuzzy edit distance & token similarity via RapidFuzz
- Semantic similarity using TF-IDF sublinear n-gram embeddings & cosine similarity
- Dynamic, dataset-aware alias and acronym generation
- Business domain synonyms & data dictionary description integration
- 3-tier confidence-based decision engine (HIGH, MEDIUM, LOW)
"""

import re
import math
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    import difflib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Standard business abbreviations & bi-directional synonyms
COMMON_ABBREVIATIONS: Dict[str, List[str]] = {
    "dept": ["department"],
    "department": ["dept", "division", "team", "unit", "function", "sector"],
    "dev": ["deviation"],
    "deviation": ["dev", "variance", "discrepancy", "non-conformance", "nonconformance", "irregularity", "gap", "anomaly"],
    "cat": ["category"],
    "category": ["cat", "classification", "type", "class", "group", "segment", "kind"],
    "qty": ["quantity"],
    "quantity": ["qty", "units", "volume", "count", "amount"],
    "amt": ["amount"],
    "amount": ["amt", "sum", "total", "value"],
    "pct": ["percentage", "percent"],
    "percentage": ["pct", "percent", "ratio", "proportion", "share", "rate"],
    "percent": ["pct", "percentage", "ratio"],
    "rev": ["revenue"],
    "revenue": ["rev", "sales", "turnover", "income", "billings", "topline"],
    "cust": ["customer"],
    "customer": ["cust", "client", "buyer", "consumer", "account"],
    "prod": ["product"],
    "product": ["prod", "item", "sku", "merchandise", "good", "part"],
    "emp": ["employee"],
    "employee": ["emp", "worker", "staff", "personnel", "operator", "technician"],
    "obs": ["observation"],
    "observation": ["obs", "finding", "audit", "record", "note"],
    "mo": ["month"],
    "month": ["mo", "monthly"],
    "monthly": ["month", "mo", "by month", "per month"],
    "yr": ["year"],
    "year": ["yr", "annual", "yearly"],
    "yearly": ["year", "yr", "annual"],
    "dt": ["date"],
    "date": ["dt", "timestamp", "time", "day"],
    "perf": ["performance"],
    "performance": ["perf", "score", "rating", "kpi", "achievement", "evaluation"],
    "near miss": ["near miss", "close call", "incident", "hazard", "unsafe act", "unsafe condition", "safety alert", "whistleblowing", "audit finding"],
    "nm": ["near miss"],
    "sev": ["severity"],
    "severity": ["sev", "risk", "criticality", "priority", "impact", "danger"],
    "num": ["number"],
    "number": ["num", "count", "id"],
    "desc": ["description"],
    "description": ["desc", "details", "narrative", "comment"],
    "loc": ["location"],
    "location": ["loc", "site", "plant", "facility", "zone", "warehouse", "area", "region"],
    "avg": ["average"],
    "average": ["avg", "mean"],
    "tot": ["total"],
    "total": ["tot", "sum"],
    "min": ["minimum"],
    "max": ["maximum"],
    "diff": ["difference"],
    "profit": ["margin", "ebitda", "net income", "earnings", "operating profit"],
    "cost": ["expense", "spend", "spending", "cogs", "expenditure"]
}

# General query stopwords to strip when matching field names
QUERY_STRIP_STOPWORDS = {
    'show', 'me', 'give', 'tell', 'display', 'list', 'find', 'get', 'calculate',
    'what', 'is', 'the', 'of', 'in', 'for', 'by', 'and', 'or', 'to', 'from',
    'with', 'a', 'an', 'this', 'that', 'these', 'those', 'there', 'across', 'each',
    'all', 'every', 'per', 'please', 'can', 'you', 'how', 'many', 'much', 'wise',
    'analysis', 'breakdown', 'split', 'compare', 'comparison', 'distribution',
    'trend', 'highest', 'lowest', 'top', 'bottom', 'overview', 'summary',
    'metric', 'metrics', 'field', 'fields', 'column', 'columns', 'value', 'values'
}


def normalize_term(text: str) -> str:
    """Normalizes a string by lowercasing, replacing punctuation with spaces, and trimming."""
    if not text:
        return ""
    cleaned = re.sub(r'[^a-zA-Z0-9]+', ' ', str(text).lower())
    return re.sub(r'\s+', ' ', cleaned).strip()


def generate_acronym(text: str) -> str:
    """Generates an acronym from the initial letters of words in a title/column name."""
    words = re.findall(r'[a-zA-Z0-9]+', text)
    if len(words) >= 2:
        return "".join([w[0].lower() for w in words])
    return ""


class DynamicAliasDictionary:
    """
    Dataset-aware dynamic alias dictionary.
    
    Generates rich alias mappings per column dynamically using:
    - Original column name
    - Simplified / normalized column name
    - Slash / hyphen / underscore segmented components (e.g. 'Near Miss/Audit/Whistleblowing')
    - Acronyms (e.g. 'Deviation Category' -> 'dc', 'Cost of Goods Sold' -> 'cogs')
    - Business abbreviations and bi-directional expansions
    - Cross-domain synonyms (Safety, Quality, Operations, Finance, HR)
    - Semantic descriptions, business meanings, and suggested uses from Data Dictionary
    """

    def __init__(self, columns: List[str], dictionary: Optional[List[Dict[str, Any]]] = None):
        self.columns = columns
        self.dictionary = dictionary or []
        self.col_to_aliases: Dict[str, Set[str]] = {c: set() for c in columns}
        self.alias_to_col: Dict[str, List[Tuple[str, float, str]]] = {}  # alias -> list of (col, weight, reason)
        self._build_dynamic_aliases()

    def _build_dynamic_aliases(self):
        # 1. Map data dictionary entries for fast lookup
        dict_meta: Dict[str, Dict[str, Any]] = {}
        for item in self.dictionary:
            col_name = str(item.get("column_name") or item.get("name") or "")
            if col_name:
                dict_meta[col_name] = item

        for col in self.columns:
            c_str = str(col)
            c_norm = normalize_term(c_str)
            self._add_alias(col, c_str.lower(), weight=1.0, reason="exact_column_lower")
            self._add_alias(col, c_norm, weight=0.98, reason="normalized_column")

            # 2. Segmented sub-components: "Near Miss/Audit/Whistleblowing" -> "near miss", "audit", "whistleblowing"
            parts = re.split(r'[/_\-\|\(\)]+', c_str)
            if len(parts) > 1:
                for p in parts:
                    p_norm = normalize_term(p)
                    if len(p_norm) >= 2 and p_norm not in QUERY_STRIP_STOPWORDS:
                        self._add_alias(col, p_norm, weight=0.92, reason=f"sub_component_{p_norm}")

            # 3. Acronyms (e.g. Deviation Category -> dc, Near Miss -> nm)
            acronym = generate_acronym(c_str)
            if len(acronym) >= 2:
                self._add_alias(col, acronym, weight=0.88, reason="acronym")

            # 4. Token-level abbreviations & synonyms
            tokens = c_norm.split()
            for token in tokens:
                if token in COMMON_ABBREVIATIONS:
                    for syn in COMMON_ABBREVIATIONS[token]:
                        # Token replacement in full phrase
                        expanded_phrase = c_norm.replace(token, syn)
                        self._add_alias(col, expanded_phrase, weight=0.94, reason=f"abbreviation_{token}_to_{syn}")
                        # Also register individual synonym if distinct
                        if len(syn) >= 3 and syn not in QUERY_STRIP_STOPWORDS and len(tokens) == 1:
                            self._add_alias(col, syn, weight=0.90, reason=f"synonym_{syn}")

            # Check if entire column normalized phrase has standard abbreviations
            if c_norm in COMMON_ABBREVIATIONS:
                for syn in COMMON_ABBREVIATIONS[c_norm]:
                    self._add_alias(col, syn, weight=0.95, reason="direct_synonym")

            # Check combinations e.g. "dept analysis" -> dept matches department
            for abbr, expansions in COMMON_ABBREVIATIONS.items():
                if abbr in tokens:
                    for exp in expansions:
                        new_phrase = " ".join([exp if t == abbr else t for t in tokens])
                        self._add_alias(col, new_phrase, weight=0.94, reason=f"phrase_expansion_{abbr}")

            # 5. Extract domain phrases from Data Dictionary descriptions & business meanings
            meta = dict_meta.get(col, {})
            desc = str(meta.get("business_meaning", meta.get("description", ""))).lower()
            role = str(meta.get("semantic_role", meta.get("role", ""))).lower()
            subtype = str(meta.get("semantic_subtype", meta.get("semantic_type", ""))).lower()

            if desc and len(desc) > 3:
                # Extract key phrases from description
                desc_words = re.findall(r'[a-zA-Z0-9_\-]+', desc)
                cleaned_desc_words = [w for w in desc_words if len(w) >= 3 and w not in QUERY_STRIP_STOPWORDS]
                for w in cleaned_desc_words:
                    self._add_alias(col, w, weight=0.75, reason="dictionary_keyword")
                # 2-word n-grams from description
                for i in range(len(cleaned_desc_words) - 1):
                    bigram = f"{cleaned_desc_words[i]} {cleaned_desc_words[i+1]}"
                    self._add_alias(col, bigram, weight=0.82, reason="dictionary_phrase")

            if subtype and subtype not in ["general", "dimension", "measure", "none"]:
                self._add_alias(col, subtype, weight=0.85, reason=f"subtype_{subtype}")

    def _add_alias(self, col: str, alias: str, weight: float, reason: str):
        alias = alias.strip().lower()
        if not alias or len(alias) < 2:
            return
        self.col_to_aliases[col].add(alias)
        if alias not in self.alias_to_col:
            self.alias_to_col[alias] = []
        if not any(c == col for c, _, _ in self.alias_to_col[alias]):
            self.alias_to_col[alias].append((col, weight, reason))

    def get_aliases_for_column(self, col: str) -> List[str]:
        return sorted(list(self.col_to_aliases.get(col, set())))

    def get_column_matches_for_alias(self, alias: str) -> List[Tuple[str, float, str]]:
        norm_a = normalize_term(alias)
        return self.alias_to_col.get(norm_a, [])


class MatchSuggestion:
    """Represents a scored matching candidate field with explanation."""
    def __init__(
        self,
        column: str,
        confidence: float,
        confidence_pct: int,
        match_type: str,
        explanation: str,
        semantic_role: Optional[str] = None,
        data_type: Optional[str] = None,
        is_primary: bool = False
    ):
        self.column = column
        self.confidence = round(confidence, 4)
        self.confidence_pct = confidence_pct
        self.match_type = match_type
        self.explanation = explanation
        self.semantic_role = semantic_role or "Dimension"
        self.data_type = data_type or "VARCHAR"
        self.is_primary = is_primary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "confidence": self.confidence,
            "confidence_pct": self.confidence_pct,
            "match_type": self.match_type,
            "explanation": self.explanation,
            "semantic_role": self.semantic_role,
            "data_type": self.data_type,
            "is_primary": self.is_primary
        }


class IntelligentFieldMatcher:
    """
    Intelligent Column and Field Matching Engine.
    
    Provides high-accuracy query understanding and field identification using:
    - Exact, case-insensitive, and normalized matching
    - Dynamic alias, abbreviation, and synonym expansion
    - Substring and token overlap analysis
    - Edit distance & fuzzy token similarity (RapidFuzz / Levenshtein)
    - Semantic similarity using TF-IDF n-gram vectorization and cosine similarity
    - 3-tier confidence decision system:
        HIGH (>= 90%): Auto-resolved
        MEDIUM (70-89%): Confirmation required with interactive options
        LOW (< 70%): Clarification required
    """

    def __init__(
        self,
        columns: List[str],
        dictionary: Optional[List[Dict[str, Any]]] = None,
        col_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        high_threshold: float = 0.90,
        medium_threshold: float = 0.70
    ):
        self.columns = [str(c) for c in columns]
        self.dictionary = dictionary or []
        self.col_metadata = col_metadata or {}
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

        # Initialize Dynamic Alias Dictionary
        self.alias_dict = DynamicAliasDictionary(self.columns, self.dictionary)

        # Build TF-IDF semantic vector space over column names, aliases, and descriptions
        self._build_semantic_index()

    def _build_semantic_index(self):
        """Constructs a TF-IDF character and word n-gram model over column documents."""
        self.corpus_texts: List[str] = []
        self.corpus_cols: List[str] = []

        for col in self.columns:
            aliases = self.alias_dict.get_aliases_for_column(col)
            meta = self.col_metadata.get(col, {})
            desc = str(meta.get("business_meaning", meta.get("description", "")))
            role = str(meta.get("semantic_role", meta.get("role", "")))
            subtype = str(meta.get("semantic_subtype", meta.get("semantic_type", "")))

            # Document combining column name, aliases, description, and roles
            doc_parts = [col, col.replace('_', ' '), " ".join(aliases), desc, role, subtype]
            doc = " ".join([p for p in doc_parts if p]).strip()
            self.corpus_texts.append(doc)
            self.corpus_cols.append(col)

        if self.corpus_texts:
            try:
                # Sublinear TF scaling with char_wb 3-5 n-grams and word 1-2 n-grams
                self.vectorizer = TfidfVectorizer(
                    analyzer='char_wb',
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                    lowercase=True
                )
                self.corpus_matrix = self.vectorizer.fit_transform(self.corpus_texts)
            except Exception:
                self.vectorizer = None
                self.corpus_matrix = None
        else:
            self.vectorizer = None
            self.corpus_matrix = None

    def find_matches(
        self,
        query_or_term: str,
        preferred_role: Optional[str] = None,
        top_k: int = 5
    ) -> List[MatchSuggestion]:
        """
        Evaluates a user query or term against all dataset columns using all matching layers.
        Returns a sorted list of MatchSuggestion objects in descending order of confidence.
        """
        if not query_or_term or not query_or_term.strip():
            return []

        raw_query = query_or_term.strip()
        q_norm = normalize_term(raw_query)
        if not q_norm:
            return []

        # Remove generic query words if phrase is longer than 1 word (e.g. "department wise" -> "department")
        q_words = q_norm.split()
        filtered_words = [w for w in q_words if w not in QUERY_STRIP_STOPWORDS]
        q_clean = " ".join(filtered_words) if filtered_words else q_norm

        # Expand abbreviations in cleaned query (e.g. "dept" -> "department")
        expanded_tokens = []
        for w in q_clean.split():
            if w in COMMON_ABBREVIATIONS:
                expanded_tokens.append(COMMON_ABBREVIATIONS[w][0])
            else:
                expanded_tokens.append(w)
        q_expanded = " ".join(expanded_tokens)

        scores: Dict[str, Tuple[float, str, str]] = {}  # col -> (confidence, match_type, explanation)

        # 1. Exact, Case-Insensitive, and Normalized Match
        for col in self.columns:
            c_norm = normalize_term(col)
            if raw_query.lower() == col.lower() or q_norm == c_norm:
                scores[col] = (1.0, "EXACT", f"Exact match with column '{col}'")
            elif q_clean == c_norm or q_expanded == c_norm:
                scores[col] = (0.98, "EXACT", f"Exact match with normalized column '{col}'")

        # 2. Dynamic Alias Dictionary Match
        for check_term in [raw_query.lower(), q_norm, q_clean, q_expanded]:
            alias_matches = self.alias_dict.get_column_matches_for_alias(check_term)
            for col, weight, reason in alias_matches:
                if col not in scores or weight > scores[col][0]:
                    scores[col] = (weight, "ALIAS", f"Matched dynamic alias '{check_term}' ({reason})")

        # 3. Substring & Token Subset Match
        for col in self.columns:
            if col in scores and scores[col][0] >= 0.98:
                continue
            c_norm = normalize_term(col)
            c_words = set(c_norm.split())

            # Whole term is a substring of column name (e.g. "deviation" in "deviation category")
            if (len(q_clean) >= 4 and q_clean in c_norm) or (len(q_norm) >= 4 and q_norm in c_norm):
                conf = 0.92 if q_clean == c_norm else 0.88
                if col not in scores or conf > scores[col][0]:
                    scores[col] = (conf, "SUBSTRING", f"Query term is contained in column '{col}'")

            # Column name is contained in user query (e.g. "show deviation category wise analysis" contains "deviation category")
            if len(c_norm) >= 4 and (c_norm in q_norm or c_norm in raw_query.lower()):
                conf = 0.95
                if col not in scores or conf > scores[col][0]:
                    scores[col] = (conf, "SUBSTRING", f"Column '{col}' is explicitly mentioned in query")

            # Sub-component match (e.g. "near miss" in "near miss audit whistleblowing")
            parts = [normalize_term(p) for p in re.split(r'[/_\-\|\(\)]+', col)]
            for p in parts:
                if p and len(p) >= 3:
                    if q_clean == p or q_norm == p or q_expanded == p:
                        conf = 0.95
                        if col not in scores or conf > scores[col][0]:
                            scores[col] = (conf, "SUB_COMPONENT", f"Exact match with column component '{p}' in '{col}'")
                    elif p in q_clean or p in q_norm or q_clean in p:
                        conf = 0.88
                        if col not in scores or conf > scores[col][0]:
                            scores[col] = (conf, "SUB_COMPONENT", f"Matched column component '{p}' in '{col}'")

        # 4. RapidFuzz Edit Distance & Token Similarity
        for col in self.columns:
            if col in scores and scores[col][0] >= 0.95:
                continue
            c_norm = normalize_term(col)

            best_fuzz_ratio = 0.0
            best_fuzz_type = ""

            # Evaluate against column name and all its aliases
            targets = [c_norm] + list(self.alias_dict.get_aliases_for_column(col))
            for target in targets:
                if not target:
                    continue
                for q_target in [q_clean, q_expanded, q_norm]:
                    if RAPIDFUZZ_AVAILABLE:
                        # Rapidfuzz token_set_ratio & ratio
                        r1 = fuzz.ratio(q_target, target) / 100.0
                        r2 = fuzz.token_set_ratio(q_target, target) / 100.0
                        r3 = fuzz.partial_ratio(q_target, target) / 100.0
                        # Guard against spurious partial matches where main ratios are low
                        if r3 > 0.75 and r1 < 0.40 and r2 < 0.50:
                            r3 = 0.0
                        r_comp = max(r1, r2 * 0.94, r3 * 0.85)
                    else:
                        # Difflib fallback
                        r_comp = difflib.SequenceMatcher(None, q_target, target).ratio()

                    # Guard against short words creating false positives (e.g. 'cat' vs 'date')
                    if len(q_target) <= 3 and r_comp < 0.95:
                        continue

                    if r_comp > best_fuzz_ratio:
                        best_fuzz_ratio = r_comp
                        best_fuzz_type = f"fuzzy_match_on_{target[:20]}"

            # Calibrate fuzzy score (must have substantial similarity)
            if best_fuzz_ratio >= 0.75:
                calibrated_conf = round(best_fuzz_ratio * 0.95, 4)
                if col not in scores or calibrated_conf > scores[col][0]:
                    scores[col] = (calibrated_conf, "FUZZY", f"Fuzzy similarity ({round(best_fuzz_ratio*100)}%) with '{col}'")

        # 5. TF-IDF Character/Word N-Gram Semantic Vector Cosine Similarity
        if self.vectorizer is not None and self.corpus_matrix is not None:
            try:
                q_vec = self.vectorizer.transform([q_clean or q_norm])
                sims = cosine_similarity(q_vec, self.corpus_matrix)[0]
                for idx, col in enumerate(self.corpus_cols):
                    sim = float(sims[idx])
                    if sim >= 0.55:
                        # Non-linear boost for semantic similarity
                        sem_conf = round(math.sqrt(sim) * 0.86, 4)
                        if col not in scores or sem_conf > scores[col][0]:
                            scores[col] = (sem_conf, "SEMANTIC", f"Semantic n-gram similarity ({round(sim*100)}%) with '{col}'")
            except Exception:
                pass

        # 6. Apply preferred role filter or penalty if specified
        final_candidates: List[MatchSuggestion] = []
        for col, (conf, match_type, explanation) in scores.items():
            meta = self.col_metadata.get(col, {})
            role = str(meta.get("semantic_role", meta.get("role", "Dimension")))
            dtype = str(meta.get("physical_dtype", meta.get("data_type", "VARCHAR")))
            is_id = meta.get("is_identifier", False) or role.lower() in ["identifier", "id", "primary key", "foreign key"] or meta.get("is_primary_key", False)

            # Check strict role compatibility
            if preferred_role:
                r_low = preferred_role.lower()
                role_low = role.lower()
                if r_low in ["measure", "metric"]:
                    if is_id or role_low not in ["measure", "numeric measure", "target/outcome"]:
                        continue
                elif r_low in ["id", "identifier", "key"]:
                    if not is_id:
                        continue
                elif r_low in ["temporal", "time", "date"]:
                    if role_low not in ["temporal", "date/time", "time", "date"]:
                        continue
                elif r_low in ["dimension", "category"]:
                    if role_low in ["measure", "numeric measure", "target/outcome"] and not is_id:
                        continue

            conf_pct = min(100, max(1, int(round(conf * 100))))
            final_candidates.append(MatchSuggestion(
                column=col,
                confidence=conf,
                confidence_pct=conf_pct,
                match_type=match_type,
                explanation=explanation,
                semantic_role=role,
                data_type=dtype,
                is_primary=False
            ))

        # Sort descending by confidence
        final_candidates.sort(key=lambda s: s.confidence, reverse=True)

        if final_candidates:
            final_candidates[0].is_primary = True

        return final_candidates[:top_k]

    def evaluate_decision(
        self,
        query_or_term: str,
        preferred_role: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes confidence-based decision engine:
        - HIGH (90-100%): Unambiguous match, auto-resolved.
        - MEDIUM (70-89%): Confirmation required, returns top suggestions and Did-You-Mean prompt.
        - LOW (< 70%): Clarification required, returns available dataset columns.
        """
        suggestions = self.find_matches(query_or_term, preferred_role=preferred_role, top_k=5)

        if not suggestions:
            # Low confidence / No match found
            return {
                "decision": "LOW_CONFIDENCE",
                "suggestion_mode": "clarification_required",
                "resolved_field": None,
                "confidence": 0.0,
                "confidence_pct": 0,
                "suggestions": [],
                "clarification_prompt": f"I couldn't identify a matching field for '{query_or_term}'. Here are the available columns in your dataset:",
                "available_columns": self.columns,
                "interactive_options": ["Search again", "View Data Dictionary"]
            }

        top = suggestions[0]

        # Check for ambiguity: Is there a second candidate with virtually identical confidence?
        is_ambiguous = False
        if len(suggestions) > 1:
            second = suggestions[1]
            if (top.confidence - second.confidence) < 0.04 and top.confidence < 0.98:
                is_ambiguous = True

        # HIGH CONFIDENCE: 90% - 100% and not ambiguous
        if top.confidence >= self.high_threshold and not is_ambiguous:
            return {
                "decision": "HIGH_CONFIDENCE",
                "suggestion_mode": "auto_resolved",
                "resolved_field": top.column,
                "confidence": top.confidence,
                "confidence_pct": top.confidence_pct,
                "suggestions": [s.to_dict() for s in suggestions],
                "indication": f"Resolved field: {top.column} ({top.confidence_pct}% match)",
                "interactive_options": []
            }

        # MEDIUM CONFIDENCE: 70% - 89% (or ambiguous high confidence)
        elif top.confidence >= self.medium_threshold or (top.confidence >= self.high_threshold and is_ambiguous):
            top_field = top.column
            prompt_lines = [
                f"I found some potentially matching fields for '{query_or_term}':",
                ""
            ]
            for idx, s in enumerate(suggestions[:3], 1):
                prompt_lines.append(f"{idx}. {s.column} – {s.confidence_pct}% match ({s.explanation})")
            prompt_lines.append("")
            prompt_lines.append(f"Would you like me to use **{top_field}**?")

            return {
                "decision": "MEDIUM_CONFIDENCE",
                "suggestion_mode": "confirmation_required",
                "resolved_field": top_field,
                "confidence": top.confidence,
                "confidence_pct": top.confidence_pct,
                "suggestions": [s.to_dict() for s in suggestions],
                "clarification_prompt": "\n".join(prompt_lines),
                "interactive_options": [
                    f"Yes, use {top_field}",
                    "Show other matches",
                    "Search again"
                ]
            }

        # LOW CONFIDENCE: Below 70%
        else:
            role_hint = f" for {preferred_role}" if preferred_role else ""
            return {
                "decision": "LOW_CONFIDENCE",
                "suggestion_mode": "clarification_required",
                "resolved_field": None,
                "confidence": top.confidence,
                "confidence_pct": top.confidence_pct,
                "suggestions": [s.to_dict() for s in suggestions],
                "clarification_prompt": f"I couldn't confidently determine which dataset field corresponds to '{query_or_term}' (closest was '{top.column}' at {top.confidence_pct}%). Please choose from the available columns{role_hint}:",
                "available_columns": self.columns,
                "interactive_options": ["Search again", "View Data Dictionary"]
            }
