"""
excel_model_improved
====================

This module contains a re‑implementation of the core logic for normalising a
collection of product lists from multiple suppliers.  It is designed to be
independent of any existing repository structure and can be used as a drop‑in
module for experimentation or migration.  The implementation is heavily
inspired by the analysis of the original ``excel‑model`` project, but it
incorporates a number of improvements proposed in that analysis:

* A unified scoring scheme for selecting the **codigo**, **descripcion** and
  **precio** columns.  Each numeric column is evaluated on multiple criteria
  (header hints, numeric ratios, decimals, uniqueness, currency symbols,
  typical quantity patterns, etc.).  Rather than committing to a candidate
  column as soon as a heuristic matches, all columns are scored and the
  highest scoring candidate is chosen.  This prevents hard‑coded early exits
  that might ignore rules declared in YAML profiles.

* Automatic handling of relationships between price and quantity columns.
  When multiple numeric columns are present, the algorithm checks for a
  multiplicative relationship (e.g. ``price_per_pack ≈ price_per_unit × units_per_pack``).
  Columns that appear to be aggregate prices (such as ``USD x pack``) are
  penalised relative to their unit counterparts.  This generalises the
  previous hard‑coded rule that always preferred ``usd x mt`` over ``usd x rollo``.

* Consistent application of ``bad_price_headers`` defined in a YAML profile.
  Regardless of which heuristic yields a price candidate, the result is
  rejected if its header matches any entry in the profile's ``bad_price_headers``.

* Support for per‑supplier configuration via optional profile dictionaries.
  Profiles may specify lists of preferred header fragments for each field
  (``prefer_price_headers``, ``prefer_code_headers``, ``prefer_name_headers``),
  lists of forbidden headers (``bad_price_headers``) and other tuning
  parameters (weights, regular expressions, etc.).  These are merged with
  sensible defaults.

The public API exposes two functions:

``extract_tables_from_excel(path)``
    Reads an Excel file (``.xls`` or ``.xlsx``) and yields a sequence of
    DataFrames, one for each logical table found.  Blank rows are used to
    separate tables within a sheet.  Header rows are detected automatically.

``normalise_table(df, profile=None)``
    Given a DataFrame representing a single table, returns a new DataFrame
    containing only the normalised columns: ``codigo``, ``nombre`` and
    ``precio``.  If no suitable columns can be found the result will be
    empty.  The optional ``profile`` argument is a dictionary matching the
    structure of the YAML profiles used in the original project; unknown keys
    are ignored.

The implementation relies only on ``pandas`` and the Python standard library.
It is deliberately verbose and documented to aid future maintainers.  All
heuristic weights and threshold values are defined as module constants at
the top of the file; tuning them for a particular data set can be done
without modifying the core logic.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Configuration constants
#
# These constants control the behaviour of the scoring functions.  They can be
# overridden on a per‑profile basis by passing an appropriate ``profile``
# dictionary to ``normalise_table``.

DEFAULT_PRICE_HINTS: List[str] = [
    "precio", "price", "pvp", "p.u.", "usd", "ars", "neto",
    "lista", "venta", "unitario", "valor",
]
"""Fragments of header text that suggest a column contains a price.

The more of these tokens that appear in a header, the higher the base score
for a potential price column.
"""

DEFAULT_CODE_HINTS: List[str] = ["codigo", "cod", "sku", "id", "product id"]
"""Fragments of header text that suggest a column contains a product code."""

DEFAULT_NAME_HINTS: List[str] = [
    "descripcion", "descripcion del producto", "descripcion producto",
    "descripcion art", "producto", "nombre", "detalle", "articulo",
]
"""Fragments of header text that suggest a column contains a product name.

We purposefully include several variants of "descripcion" and "producto" to
match common Spanish headers.
"""

DEFAULT_QUANTITY_HINTS: List[str] = [
    "cantidad", "cant", "unidades", "und", "pack", "caja", "bulto",
    "bultos", "unidad", "unit", "pieces", "qty", "x pack", "x caja",
]
"""Fragments of header text that suggest a column contains quantities or
packaging information.  Price detection will penalise columns containing
these tokens.
"""

DEFAULT_BAD_PRICE_HEADERS: List[str] = [
    "und x pack", "und x caja", "unidades x pack", "unidades x caja",
    "cantidad", "cant", "unidades", "cantidad por pack", "cantidad x pack",
    "unidades x bulto", "unidades  x caja", "unidades x bulto",
]
"""Headers that should never be considered as price columns.  These entries
may be extended by a profile (via ``bad_price_headers``) to capture supplier
specific exceptions.  Matching is case insensitive and compares against the
entire header string after normalisation (spaces collapsed and leading/trailing
whitespace removed).
"""

DEFAULT_PREFER_PRICE_HEADERS: List[str] = []
"""Specific headers that are strongly preferred as price columns when they
appear.  For example, a profile might set this to ``["usd lista"]`` to
prioritise that column over all others.  The default is empty.
"""

DEFAULT_PREFER_CODE_HEADERS: List[str] = []
"""Specific headers that are strongly preferred as code columns when they
appear.  The default is empty.
"""

DEFAULT_PREFER_NAME_HEADERS: List[str] = []
"""Specific headers that are strongly preferred as name/description columns.
The default is empty.
"""

DEFAULT_WEIGHTS: Dict[str, float] = {
    # Price scoring weights
    "price_header_hint": 2.0,
    "price_currency_symbol": 1.5,
    "price_numeric_ratio": 1.0,
    "price_dec_ratio": 2.0,
    "price_typical_qty_penalty": -4.0,
    "price_bad_header_penalty": -10.0,
    "price_prefer_header_bonus": 8.0,
    "price_pack_agg_penalty": -3.0,
    # Code scoring weights
    "code_header_hint": 2.0,
    "code_unique_ratio": 2.0,
    "code_alnum_ratio": 1.5,
    # Penalise decimal heavy numeric columns when selecting codes.  Codes seldom
    # have fractional parts, so this weight applies to ``dec_ratio`` (not
    # inverted).
    "code_numeric_ratio_penalty": -2.0,
    # Penalise columns that are almost entirely numeric for the code field.  If
    # ``numeric_ratio`` exceeds 0.9 the extra above 0.9 is multiplied by this
    # factor.  Negative values discourage numeric columns without letters.
    "code_high_numeric_penalty": -1.0,
    "code_price_hint_penalty": -3.0,
    # Name scoring weights
    "name_header_hint": 2.0,
    "name_avg_length": 1.0,
    "name_numeric_ratio_penalty": -2.0,
    "name_unique_ratio": 1.0,
}
"""Default weighting factors used when computing column scores.

These values were chosen empirically based on common patterns in supplier
spreadsheets.  Profiles may override individual entries by specifying a
``weights`` dictionary.
"""

# -----------------------------------------------------------------------------
# Helper classes and functions

def _normalise_header(header: str) -> str:
    """Normalise a header string for comparison.

    Lowercases the header, strips leading/trailing whitespace, collapses
    multiple spaces to a single space and removes diacritics.  The resulting
    string is used for case‑insensitive comparisons against hint lists.
    """
    import unicodedata
    if not isinstance(header, str):
        return ""
    # normalise unicode (decompose accents)
    header = unicodedata.normalize('NFKD', header).encode('ASCII', 'ignore').decode('ASCII')
    header = header.lower().strip()
    # collapse spaces
    header = re.sub(r"\s+", " ", header)
    return header


def _contains_any(text: str, tokens: Iterable[str]) -> int:
    """Return the number of tokens that appear in ``text``.

    Both ``text`` and tokens are expected to be pre‑normalised with
    ``_normalise_header``.  This helper counts the number of token substrings
    found in ``text`` and returns the count.  It does not account for
    overlapping occurrences.
    """
    count = 0
    for tok in tokens:
        if tok and tok in text:
            count += 1
    return count


def _parse_numeric(value: object) -> Optional[float]:
    """Attempt to convert a cell value into a floating point number.

    Strings containing currency symbols, thousand separators and commas are
    tolerated.  Returns ``None`` if the value cannot be parsed.
    """
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        # Already a numeric type
        return float(value)
    # Convert to string and clean
    s = str(value)
    if not s or s.strip() == "":
        return None
    # Remove currency symbols and whitespace
    s = s.replace("$", "").replace("€", "").replace("USD", "").replace("ARS", "")
    s = s.replace("EUR", "").replace("\xa0", " ")
    s = s.strip()
    # Replace comma decimal separator by dot if there are more commas than dots
    # and no thousand separator that uses dot.
    # Also remove thousand separators (dot or comma) heuristically.
    # e.g. "1.234,56" -> "1234.56"; "1,234.56" -> "1234.56"; "1 234,56" -> "1234.56"
    s = s.replace(" ", "")
    # Count separators
    comma_count = s.count(',')
    dot_count = s.count('.')
    if comma_count > dot_count:
        # Assume comma is decimal separator; remove dots (thousand separators)
        s = s.replace('.', '')
        s = s.replace(',', '.')
    else:
        # Assume dot is decimal separator; remove commas (thousand separators)
        s = s.replace(',', '')
    # Now parse
    try:
        return float(s)
    except ValueError:
        return None


def _compute_column_stats(series: pd.Series) -> Tuple[float, float, float, float, float]:
    """Compute basic statistics for a series of values.

    Returns a tuple ``(numeric_ratio, dec_ratio, typical_qty_ratio, unique_ratio, alnum_ratio)``:

    * **numeric_ratio:** fraction of non‑blank values that can be parsed to
      numbers.
    * **dec_ratio:** among numeric values, fraction that have a fractional
      component (i.e. ``x % 1 != 0``).  A high ``dec_ratio`` suggests a price
      column (as opposed to codes or quantities).
    * **typical_qty_ratio:** among numeric values, fraction that match
      commonly seen quantity numbers (1, 2, 3, 4, 5, 6, 10, 12, 24, 50, 100).
      A high value indicates a quantity or packaging column.
    * **unique_ratio:** ratio of unique non‑blank values to total non‑blank
      values.  High uniqueness is a strong indicator for code columns.
    * **alnum_ratio:** among non‑numeric string values, fraction that contain
      both letters and digits.  Alphanumeric strings often indicate codes.
    """
    values = series.dropna().astype(object).tolist()
    if not values:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    num_count = 0
    dec_count = 0
    qty_count = 0
    alnum_count = 0
    typical_qty_values = {1, 2, 3, 4, 5, 6, 10, 12, 24, 50, 100}
    string_values = 0
    for v in values:
        num = _parse_numeric(v)
        if num is not None:
            num_count += 1
            if not math.isclose(num % 1.0, 0.0):
                dec_count += 1
            if num in typical_qty_values:
                qty_count += 1
        else:
            string_values += 1
            s = str(v)
            # detect alphanumeric patterns: must have at least one letter and one digit
            if re.search(r"[a-zA-Z]", s) and re.search(r"\d", s):
                alnum_count += 1
    total = len(values)
    numeric_ratio = num_count / total
    dec_ratio = dec_count / (num_count or 1)
    typical_qty_ratio = qty_count / (num_count or 1)
    # Unique ratio among non‑blank entries
    unique_values = len(set(values))
    unique_ratio = unique_values / total
    # Alphanumeric ratio among non‑numeric strings
    alnum_ratio = alnum_count / (string_values or 1)
    return numeric_ratio, dec_ratio, typical_qty_ratio, unique_ratio, alnum_ratio


def _contains_currency_symbol(series: pd.Series) -> float:
    """Return the fraction of values that include a currency symbol (e.g. $, USD, ARS, €)."""
    values = series.dropna().astype(str)
    if values.empty:
        return 0.0
    count = 0
    for v in values:
        s = v.lower()
        if '$' in s or 'usd' in s or 'ars' in s or '€' in s or 'eur' in s:
            count += 1
    return count / len(values)


@dataclass
class Profile:
    """Configuration profile for a supplier.

    This class wraps the raw profile dictionary (which may have been loaded
    from YAML) and exposes pre‑computed normalised fields.  Unknown keys are
    ignored.  All lists of header hints are normalised to lower case and
    spaces collapsed for consistent matching.
    """
    prefer_price_headers: List[str] = field(default_factory=lambda: DEFAULT_PREFER_PRICE_HEADERS.copy())
    prefer_code_headers: List[str] = field(default_factory=lambda: DEFAULT_PREFER_CODE_HEADERS.copy())
    prefer_name_headers: List[str] = field(default_factory=lambda: DEFAULT_PREFER_NAME_HEADERS.copy())
    bad_price_headers: List[str] = field(default_factory=lambda: DEFAULT_BAD_PRICE_HEADERS.copy())
    price_hints: List[str] = field(default_factory=lambda: DEFAULT_PRICE_HINTS.copy())
    code_hints: List[str] = field(default_factory=lambda: DEFAULT_CODE_HINTS.copy())
    name_hints: List[str] = field(default_factory=lambda: DEFAULT_NAME_HINTS.copy())
    quantity_hints: List[str] = field(default_factory=lambda: DEFAULT_QUANTITY_HINTS.copy())
    weights: Dict[str, float] = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())

    @staticmethod
    def from_dict(d: Optional[Dict[str, object]]) -> 'Profile':
        """Construct a profile from a dictionary loaded from YAML.

        Missing fields fall back to sensible defaults.  Lists are normalised.
        The ``weights`` field, if present, overrides the corresponding entries
        in the default weights dictionary on a key by key basis.
        """
        profile = Profile()
        if not d:
            return profile
        # Helper to normalise and extend lists
        def update_list(attr: str, values: Iterable[str]):
            current = getattr(profile, attr)
            for v in values:
                norm = _normalise_header(str(v))
                if norm and norm not in current:
                    current.append(norm)
        # Specific header preferences
        if 'prefer_price_headers' in d:
            update_list('prefer_price_headers', d['prefer_price_headers'])
        if 'prefer_code_headers' in d:
            update_list('prefer_code_headers', d['prefer_code_headers'])
        if 'prefer_name_headers' in d:
            update_list('prefer_name_headers', d['prefer_name_headers'])
        # Bad headers
        if 'bad_price_headers' in d:
            update_list('bad_price_headers', d['bad_price_headers'])
        # Hints (extend defaults)
        if 'price_hints' in d:
            update_list('price_hints', d['price_hints'])
        if 'code_hints' in d:
            update_list('code_hints', d['code_hints'])
        if 'name_hints' in d:
            update_list('name_hints', d['name_hints'])
        if 'quantity_hints' in d:
            update_list('quantity_hints', d['quantity_hints'])
        # Weights override
        if 'weights' in d and isinstance(d['weights'], dict):
            profile.weights.update(d['weights'])
        return profile


def extract_tables_from_excel(path: str) -> Iterable[pd.DataFrame]:
    """Yield DataFrames representing logical tables from the given Excel file.

    The Excel file may contain multiple sheets and within each sheet multiple
    tables separated by one or more completely blank rows.  Header rows are
    detected automatically by scanning the first 20 rows of each segment for
    rows containing likely column names.  If no header can be identified, the
    first non‑blank row of the segment is treated as the header.
    """
    xl = pd.ExcelFile(path)
    for sheet_name in xl.sheet_names:
        raw_df = xl.parse(sheet_name, header=None, dtype=object)
        if raw_df.empty:
            continue
        # Identify segments separated by blank rows (all NaN)
        is_blank = raw_df.apply(lambda row: row.isna().all(), axis=1)
        start = 0
        for idx, blank in enumerate(is_blank):
            if blank:
                if idx > start:
                    segment = raw_df.iloc[start:idx].reset_index(drop=True)
                    if not segment.dropna(how='all').empty:
                        table = _extract_table_from_segment(segment)
                        if table is not None:
                            yield table
                start = idx + 1
        # Last segment
        if start < len(raw_df):
            segment = raw_df.iloc[start:].reset_index(drop=True)
            if not segment.dropna(how='all').empty:
                table = _extract_table_from_segment(segment)
                if table is not None:
                    yield table


def _extract_table_from_segment(segment: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Given a contiguous segment of rows from an Excel sheet, extract a table.

    Header row detection heuristically scans the first 20 rows for the row
    containing the most textual content and at least one recognised hint.
    Returns a new DataFrame with columns labelled by the detected header.  If
    no suitable header can be found the function returns ``None``.
    """
    # Search for a candidate header row.  The header must contain at least one
    # recognised hint token (price/code/name) to be considered.  Without a
    # convincing header we fall back to generic column names (col_0, col_1, ...).
    max_header_score = -1
    header_idx: Optional[int] = None
    for i in range(min(20, len(segment))):
        row = segment.iloc[i]
        # Count non‑blank entries
        non_blank = row.notna().sum()
        if non_blank < 2:
            continue
        score = 0
        hint_count = 0
        text_cell_count = 0
        for cell in row:
            if isinstance(cell, str) and cell.strip() != "":
                text_cell_count += 1
                norm = _normalise_header(cell)
                if _contains_any(norm, DEFAULT_PRICE_HINTS + DEFAULT_CODE_HINTS + DEFAULT_NAME_HINTS):
                    hint_count += 1
        # Skip rows without any hint tokens entirely
        if hint_count == 0:
            continue
        # The header score is proportional to the number of textual cells and hints
        score = text_cell_count + hint_count * 2
        if score > max_header_score:
            max_header_score = score
            header_idx = i
    # If no header row with hints was found, treat the segment as headerless
    if header_idx is None:
        # Assign generic column names and return the data as is
        data = segment.reset_index(drop=True).copy()
        data.columns = [f"col_{i}" for i in range(data.shape[1])]
        data = data.dropna(axis=1, how='all')
        return data
    # Otherwise build DataFrame with detected header row
    header_row = segment.iloc[header_idx].tolist()
    header_row_norm = [str(h) if h is not None else "" for h in header_row]
    data = segment.iloc[header_idx + 1:].reset_index(drop=True).copy()
    data.columns = header_row_norm
    data = data.dropna(axis=1, how='all')
    return data


def normalise_table(df: pd.DataFrame, profile_dict: Optional[Dict[str, object]] = None) -> pd.DataFrame:
    """Normalise a DataFrame to the canonical columns ``codigo``, ``nombre`` and ``precio``.

    The returned DataFrame will contain only these three columns.  Columns are
    chosen using a weighted scoring scheme applied to all candidate columns.
    If any of the three cannot be identified, the corresponding column in the
    returned DataFrame will be empty.  The optional ``profile_dict`` allows
    callers to specify supplier specific overrides (e.g. ``bad_price_headers``).
    """
    profile = Profile.from_dict(profile_dict)
    # Work on a copy to avoid mutating the original
    data = df.copy()
    # Normalise header names for scoring and create a mapping
    headers_norm = {col: _normalise_header(col) for col in data.columns}
    # Compute stats for each column once
    column_stats: Dict[str, Tuple[float, float, float, float, float]] = {}
    currency_fractions: Dict[str, float] = {}
    for col in data.columns:
        series = data[col]
        column_stats[col] = _compute_column_stats(series)
        currency_fractions[col] = _contains_currency_symbol(series)
    # Build candidate scores
    price_scores: Dict[str, float] = {}
    code_scores: Dict[str, float] = {}
    name_scores: Dict[str, float] = {}
    # Precompute quantity candidate columns for price relationship analysis
    quantity_candidates = []
    for col in data.columns:
        norm_h = headers_norm[col]
        # consider as quantity candidate if header includes quantity hints or the typical qty ratio is high
        numeric_ratio, dec_ratio, qty_ratio, unique_ratio, alnum_ratio = column_stats[col]
        if qty_ratio > 0.5 or _contains_any(norm_h, profile.quantity_hints) > 0:
            quantity_candidates.append(col)
    # Precompute ratio relationships between numeric columns and quantity candidates
    # We'll use a simple method: for each candidate price column and quantity column, compute
    # median(quantity-adjusted price) and compare to the median of the column.  If the adjusted
    # values are nearly constant (low variance relative to raw), we treat the candidate as
    # aggregated and penalise it.
    def compute_pack_penalty(col: str) -> float:
        # Only applies if there are at least 2 numeric columns
        # and at least one quantity candidate.  Higher positive value means high penalty (bad).
        # We start at 0 and subtract if relationship suggests an aggregate price.
        norm_h = headers_norm[col]
        # Skip if clearly a quantity column
        if _contains_any(norm_h, profile.quantity_hints) > 0:
            return 0.0
        # Extract numeric values for this column
        series = data[col]
        values = series.dropna().apply(_parse_numeric)
        numeric_values = values.dropna()
        if len(numeric_values) < 5:
            return 0.0
        raw_median = float(np.median(list(numeric_values))) if len(numeric_values) > 0 else 0.0
        # Compare with each quantity column
        penalties = []
        for qty_col in quantity_candidates:
            if qty_col == col:
                continue
            qty_series = data[qty_col].dropna().apply(_parse_numeric).dropna()
            if len(qty_series) < 5:
                continue
            # Align indices
            joined = pd.concat([numeric_values, qty_series], axis=1, join='inner')
            if joined.empty:
                continue
            price_vals = joined.iloc[:, 0]
            qty_vals = joined.iloc[:, 1]
            # Avoid division by zero
            qty_nonzero = qty_vals.replace(0, np.nan).dropna()
            price_nonzero = price_vals.loc[qty_nonzero.index]
            if len(price_nonzero) < 5:
                continue
            adjusted = price_nonzero / qty_nonzero
            # Compute coefficient of variation (std / mean) of adjusted values
            # Lower CV indicates the column is aggregated (price ≈ unit_price * quantity).
            adj_vals = adjusted.dropna().values
            if len(adj_vals) < 5:
                continue
            mean_adj = adj_vals.mean()
            std_adj = adj_vals.std()
            if mean_adj <= 0:
                continue
            cv_adj = std_adj / mean_adj
            # Similarly compute CV of raw values
            raw_vals = price_nonzero.values
            mean_raw = raw_vals.mean()
            std_raw = raw_vals.std()
            cv_raw = std_raw / mean_raw if mean_raw > 0 else 0.0
            # If the adjusted variation is significantly lower (e.g. < half) than raw
            # variation, we treat this as an aggregate price.  Penalty is proportional
            # to the reduction in variation.
            if cv_adj < 0.6 * cv_raw and cv_raw > 0.0:
                penalties.append(cv_raw - cv_adj)
        if not penalties:
            return 0.0
        # Return total penalty (higher means more aggregated, thus bad for price)
        return sum(penalties)
    pack_penalties: Dict[str, float] = {col: compute_pack_penalty(col) for col in data.columns}
    # Score each column for price, code and name
    for col in data.columns:
        norm_h = headers_norm[col]
        num_ratio, dec_ratio, qty_ratio, uniq_ratio, alnum_ratio = column_stats[col]
        currency_frac = currency_fractions[col]
        # Price score
        p_score = 0.0
        # Strong preference if header in prefer_price_headers
        if norm_h in profile.prefer_price_headers:
            p_score += profile.weights.get('price_prefer_header_bonus', 8.0)
        # Reject if header in bad price list
        if norm_h in profile.bad_price_headers:
            # Apply a heavy negative penalty so this column is unlikely to be chosen
            p_score += profile.weights.get('price_bad_header_penalty', -10.0)
        # Add per token hints
        hint_count = _contains_any(norm_h, profile.price_hints)
        p_score += hint_count * profile.weights.get('price_header_hint', 2.0)
        # Penalise quantity hints
        qty_hint_count = _contains_any(norm_h, profile.quantity_hints)
        p_score += qty_hint_count * profile.weights.get('price_typical_qty_penalty', -4.0)
        # Statistical features
        p_score += num_ratio * profile.weights.get('price_numeric_ratio', 1.0)
        p_score += dec_ratio * profile.weights.get('price_dec_ratio', 2.0)
        p_score += currency_frac * profile.weights.get('price_currency_symbol', 1.5)
        p_score += (1.0 - qty_ratio) * profile.weights.get('price_typical_qty_penalty', -4.0)
        # Penalty if strongly aggregated with quantity columns
        p_score += pack_penalties[col] * profile.weights.get('price_pack_agg_penalty', -3.0)
        price_scores[col] = p_score
        # Code score
        c_score = 0.0
        if norm_h in profile.prefer_code_headers:
            c_score += profile.weights.get('price_prefer_header_bonus', 8.0)
        hint_c = _contains_any(norm_h, profile.code_hints)
        c_score += hint_c * profile.weights.get('code_header_hint', 2.0)
        # Penalise if header contains price hints (codes usually shouldn't contain price)
        price_hint_c = _contains_any(norm_h, profile.price_hints)
        c_score += price_hint_c * profile.weights.get('code_price_hint_penalty', -3.0)
        # Statistical features
        c_score += uniq_ratio * profile.weights.get('code_unique_ratio', 2.0)
        c_score += alnum_ratio * profile.weights.get('code_alnum_ratio', 1.5)
        # Penalise decimal content directly.  Codes should rarely have fractional values.
        c_score += dec_ratio * profile.weights.get('code_numeric_ratio_penalty', -2.0)
        # Additional penalty if the column is almost entirely numeric
        if num_ratio > 0.9:
            excess = num_ratio - 0.9
            c_score += excess * profile.weights.get('code_high_numeric_penalty', -1.0)
        code_scores[col] = c_score
        # Name/description score
        n_score = 0.0
        if norm_h in profile.prefer_name_headers:
            n_score += profile.weights.get('price_prefer_header_bonus', 8.0)
        hint_n = _contains_any(norm_h, profile.name_hints)
        n_score += hint_n * profile.weights.get('name_header_hint', 2.0)
        # Penalise if header contains price or quantity hints
        price_hint_n = _contains_any(norm_h, profile.price_hints)
        qty_hint_n = _contains_any(norm_h, profile.quantity_hints)
        n_score += price_hint_n * profile.weights.get('name_numeric_ratio_penalty', -2.0)
        n_score += qty_hint_n * profile.weights.get('name_numeric_ratio_penalty', -2.0)
        # Statistical features: longer text columns likely contain descriptions
        # Estimate average length of string values
        strings = data[col].dropna().astype(str)
        if not strings.empty:
            lengths = strings.str.len()
            avg_len = lengths.mean() / 50.0  # normalise by an arbitrary factor
            n_score += avg_len * profile.weights.get('name_avg_length', 1.0)
        n_score += uniq_ratio * profile.weights.get('name_unique_ratio', 1.0)
        # Reward columns with low numeric ratio: names are predominantly textual.
        # Use the absolute value of the configured penalty weight to convert it into a bonus.
        numeric_penalty = profile.weights.get('name_numeric_ratio_penalty', -2.0)
        n_score += (1.0 - num_ratio) * abs(numeric_penalty)
        # Penalise columns with a high decimal ratio (names rarely contain decimals)
        n_score += dec_ratio * numeric_penalty
        name_scores[col] = n_score
    # Select highest scoring columns for price, code and name
    price_col = max(price_scores, key=lambda c: price_scores[c]) if price_scores else None
    code_col = max(code_scores, key=lambda c: code_scores[c]) if code_scores else None
    name_col = max(name_scores, key=lambda c: name_scores[c]) if name_scores else None
    # If multiple potential price columns exist, apply a tie‑breaker based on
    # keyword preference.  This helps choose "Precio Neto" over
    # "Precio de Venta" and "Precio de Lista" when scores are similar.  The
    # order reflects typical desirability of pricing columns: net price,
    # sale price, list price, PVP, etc.  Only override the score based
    # selection if a preferred keyword appears in a different column and
    # the original score difference is within a small margin (0.5 points).
    if price_col is not None:
        # Determine second best score for comparison
        sorted_prices = sorted(price_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_col, top_score = sorted_prices[0]
        second_score = sorted_prices[1][1] if len(sorted_prices) > 1 else None
        # Compute margin between top two
        margin = abs(top_score - second_score) if second_score is not None else None
        # Build a list of candidate columns that contain price hints
        price_like_cols = [c for c in data.columns if _contains_any(headers_norm[c], profile.price_hints) > 0 and headers_norm[c] not in profile.bad_price_headers]
        preferred_keywords = [
            "neto",  # net price
            "unitario", "x unidad",  # unit price
            "venta",  # sale price
            "lista",  # list price
            "pvp premium", "pvp",
        ]
        # Only apply the override if margin is small (≤0.5) or if the top column is not a price‑like column
        override_allowed = (margin is not None and margin <= 0.5) or price_col not in price_like_cols
        if override_allowed:
            for kw in preferred_keywords:
                for col in price_like_cols:
                    if kw in headers_norm[col]:
                        price_col = col
                        # Exit both loops
                        break
                else:
                    continue
                break

    # Construct result
    result = pd.DataFrame()
    # Populate codigo
    if code_col is not None:
        result['codigo'] = data[code_col].astype(str).str.strip()
    else:
        result['codigo'] = ""
    # Populate nombre
    if name_col is not None:
        result['nombre'] = data[name_col].astype(str).str.strip()
    else:
        result['nombre'] = ""
    # Populate precio as numeric
    if price_col is not None:
        price_series = data[price_col].apply(_parse_numeric)
        result['precio'] = price_series
    else:
        result['precio'] = np.nan
    # Drop rows without name or precio
    result = result[(result['nombre'] != "") & (~result['precio'].isna())]
    # Remove duplicates
    result = result.drop_duplicates(subset=['codigo', 'nombre', 'precio'])
    return result.reset_index(drop=True)


__all__ = [
    'extract_tables_from_excel',
    'normalise_table',
    'Profile',
]