"""text_prep.py

Reusable preprocessing utilities for the NY311 topic modeling pipeline.

Conception phase requirements implemented here:
- Build one input text field by concatenating: 'Complaint Type', 'Descriptor',
  and (when available) 'Resolution Description'.
- Preprocessing plan (cleaning text):
  (1) Drop rows with missing/empty text and delete duplicates.
  (2) Apply normalization: convert text to lowercase and normalize whitespace.
  (3) Noise removal: remove URLs, email addresses, and phone-number-like strings;
      strip non-informative punctuation/symbols by replacing them with spaces.

Note: This module intentionally keeps cleaning *lightweight* (regex-based) and
matches what a CountVectorizer/TfidfVectorizer pipeline expects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


# -------------------------
# Regex patterns for noise
# -------------------------
_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+\b", flags=re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.[a-z]{2,}\b", flags=re.IGNORECASE)

# Phone-like patterns (fairly permissive; goal is to remove obvious phone strings)
_PHONE_RE = re.compile(
    r"\b(?:\+?\d{1,3}[\s\-.]?)?"          # optional country code
    r"(?:\(?\d{2,4}\)?[\s\-.]?)?"        # optional area code
    r"\d{3,4}[\s\-.]?\d{3,4}\b"           # local number
)

# Replace non-informative punctuation/symbols with spaces.
# Keep only letters, digits, and whitespace.
_NONINFO_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    """Clean a single string according to the conception phase preprocessing plan."""
    if s is None:
        return ""

    s = str(s)

    # (2) normalization: lowercase
    s = s.lower()

    # (3) noise removal
    s = _URL_RE.sub(" ", s)
    s = _EMAIL_RE.sub(" ", s)
    s = _PHONE_RE.sub(" ", s)

    # (3) strip punctuation/symbols by replacing with spaces
    s = _NONINFO_RE.sub(" ", s)

    # (2) normalize whitespace
    s = _WS_RE.sub(" ", s).strip()

    return s


def _pick_text_columns(df: pd.DataFrame) -> List[str]:
    """Select text columns per the conception phase (Resolution Description when available)."""
    preferred = ["Complaint Type", "Descriptor", "Resolution Description"]
    cols = [c for c in preferred if c in df.columns]
    if not cols:
        raise ValueError(
            "Expected at least one of these columns in the input CSV: "
            "'Complaint Type', 'Descriptor', 'Resolution Description'."
        )
    return cols


@dataclass
class PreprocessReport:
    used_columns: List[str]
    rows_input: int
    rows_after_empty_drop: int
    rows_after_row_dedup: int
    rows_after_text_dedup: int

    def to_lines(self) -> List[str]:
        return [
            f"Text columns used: {', '.join(self.used_columns)}",
            f"Rows input: {self.rows_input}",
            f"Rows after dropping empty/missing text: {self.rows_after_empty_drop}",
            f"Rows after dropping duplicate rows: {self.rows_after_row_dedup}",
            f"Rows after dropping duplicate cleaned texts: {self.rows_after_text_dedup}",
        ]


def build_and_clean_text(df: pd.DataFrame) -> Tuple[pd.Series, List[str]]:
    """Construct one input text field and return its cleaned version (aligned to df index)."""
    cols = _pick_text_columns(df)

    raw = (
        df[cols]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.strip()
    )

    cleaned = raw.map(clean_text)
    return cleaned, cols


def apply_conception_preprocessing(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, PreprocessReport]:
    """Apply the conception phase preprocessing plan.

    Returns:
      - df_clean: filtered dataframe (rows align with cleaned_text)
      - cleaned_text: cleaned text series (index aligns with df_clean)
      - report: counts and metadata for logging
    """
    rows_input = len(df)

    # Build + clean text
    cleaned_text, used_cols = build_and_clean_text(df)

    # (1) Drop rows with missing/empty text
    non_empty_mask = cleaned_text.astype(str).str.strip().ne("")
    df1 = df.loc[non_empty_mask].copy()
    text1 = cleaned_text.loc[non_empty_mask]
    rows_after_empty_drop = len(df1)

    # (1) Delete duplicate rows (exact duplicates)
    df2 = df1.drop_duplicates()
    text2 = text1.loc[df2.index]
    rows_after_row_dedup = len(df2)

    # (1) Delete duplicates based on cleaned text content
    # This matches the typical intent of "duplicates will be deleted" for text analytics.
    text3 = text2.drop_duplicates(keep="first")
    df3 = df2.loc[text3.index].copy()
    rows_after_text_dedup = len(df3)

    report = PreprocessReport(
        used_columns=used_cols,
        rows_input=rows_input,
        rows_after_empty_drop=rows_after_empty_drop,
        rows_after_row_dedup=rows_after_row_dedup,
        rows_after_text_dedup=rows_after_text_dedup,
    )

    return df3, text3, report
