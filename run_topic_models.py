"""
NYC 311 Topic Modeling (CountVectorizer+LDA and TF-IDF+NMF)

Usage:
  python run_topic_models.py --input "ny311_ready_900.csv" --k 7

Outputs (created under ./outputs):
  - vectorization_summary.txt
  - lda_topics.csv
  - nmf_topics.csv
  - representative_examples.csv
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def clean_text(s: str) -> str:
    """Basic cleaning: lowercase, remove URLs/emails/phone-like strings, replace symbols with space, normalize whitespace."""
    if s is None:
        return ""
    s = str(s).lower()

    # URLs
    s = re.sub(r"(https?://\S+|www\.\S+)", " ", s)

    # emails
    s = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", s)

    # phone-like patterns (simple heuristic)
    s = re.sub(r"\b(\+?\d[\d\-\(\)\s]{7,}\d)\b", " ", s)

    # non-informative symbols/punctuation -> space (keep letters/numbers)
    s = re.sub(r"[^a-z0-9\s]", " ", s)

    # whitespace normalization
    s = re.sub(r"\s+", " ", s).strip()
    return s


def top_words(feature_names, components, n_top=12):
    topics = []
    for topic_idx, comp in enumerate(components):
        top_idx = np.argsort(comp)[::-1][:n_top]
        topics.append([feature_names[i] for i in top_idx])
    return topics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to ny311 CSV (must include Complaint Type and Descriptor columns)")
    ap.add_argument("--k", type=int, default=7, help="Number of topics (K)")
    ap.add_argument("--include_resolution", action="store_true", help="Include Resolution Description in the text field (optional)")
    args = ap.parse_args()

    inp = Path(args.input)
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp, dtype=str, low_memory=False)

    # Build analysis text field (core complaint description)
    parts = [df.get("Complaint Type", ""), df.get("Descriptor", "")]
    if args.include_resolution and "Resolution Description" in df.columns:
        parts.append(df["Resolution Description"])

    text = (" ".join if False else None)

    # concatenate safely row-wise
    df["text_core"] = (
        pd.concat([p.fillna("") if hasattr(p, "fillna") else pd.Series(p) for p in parts], axis=1)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # drop empty
    df = df[df["text_core"].fillna("").str.strip().ne("")]
    df["text_clean"] = df["text_core"].map(clean_text)

    # vectorizers (use common English stopwords; 1-2 grams to preserve phrases)
    count_vec = CountVectorizer(stop_words="english", max_df=0.95, min_df=2, ngram_range=(1, 2))
    tfidf_vec = TfidfVectorizer(stop_words="english", max_df=0.95, min_df=2, ngram_range=(1, 2))

    X_counts = count_vec.fit_transform(df["text_clean"])
    X_tfidf = tfidf_vec.fit_transform(df["text_clean"])

    # small comparison summary
    vocab_size = X_counts.shape[1]
    top_count_idx = np.asarray(X_counts.sum(axis=0)).ravel().argsort()[::-1][:15]
    top_count_terms = [(count_vec.get_feature_names_out()[i], int(np.asarray(X_counts.sum(axis=0)).ravel()[i])) for i in top_count_idx]

    top_tfidf_idx = np.asarray(X_tfidf.mean(axis=0)).ravel().argsort()[::-1][:15]
    top_tfidf_terms = [(tfidf_vec.get_feature_names_out()[i], float(np.asarray(X_tfidf.mean(axis=0)).ravel()[i])) for i in top_tfidf_idx]

    with (out_dir / "vectorization_summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"Rows used: {len(df)}\n")
        f.write(f"Vocabulary size: {vocab_size}\n\n")
        f.write("Top terms by total COUNT:\n")
        for term, c in top_count_terms:
            f.write(f"  {term}: {c}\n")
        f.write("\nTop terms by mean TF-IDF:\n")
        for term, v in top_tfidf_terms:
            f.write(f"  {term}: {v:.4f}\n")

    # Topic models
    lda = LatentDirichletAllocation(n_components=args.k, random_state=42, learning_method="batch", max_iter=60)
    doc_topic_lda = lda.fit_transform(X_counts)
    lda_topics = top_words(count_vec.get_feature_names_out(), lda.components_, n_top=12)

    nmf = NMF(n_components=args.k, random_state=42, init="nndsvda", max_iter=600)
    W = nmf.fit_transform(X_tfidf)
    nmf_topics = top_words(tfidf_vec.get_feature_names_out(), nmf.components_, n_top=12)

    # Save topics
    lda_df = pd.DataFrame({"topic": list(range(args.k)), "top_words": ["; ".join(t) for t in lda_topics]})
    nmf_df = pd.DataFrame({"topic": list(range(args.k)), "top_words": ["; ".join(t) for t in nmf_topics]})
    lda_df.to_csv(out_dir / "lda_topics.csv", index=False, encoding="utf-8")
    nmf_df.to_csv(out_dir / "nmf_topics.csv", index=False, encoding="utf-8")

    # Representative examples: pick top 3 texts per topic (unique)
    def top_unique_examples(matrix, topic_idx, n=3):
        idx = np.argsort(matrix[:, topic_idx])[::-1]
        seen = set()
        out = []
        for i in idx:
            ex = df.iloc[i]["text_core"]
            if ex not in seen and str(ex).strip():
                out.append(ex)
                seen.add(ex)
            if len(out) >= n:
                break
        return out

    rows = []
    for t in range(args.k):
        rows.append({
            "topic": t,
            "model": "LDA",
            "examples": " | ".join(top_unique_examples(doc_topic_lda, t, n=3))
        })
    for t in range(args.k):
        rows.append({
            "topic": t,
            "model": "NMF",
            "examples": " | ".join(top_unique_examples(W, t, n=3))
        })
    pd.DataFrame(rows).to_csv(out_dir / "representative_examples.csv", index=False, encoding="utf-8")

    print("DONE. Outputs in:", out_dir)


if __name__ == "__main__":
    main()
