"""
NYC 311 Topic Modeling (CountVectorizer+LDA and TF-IDF+NMF)
+ Coherence scoring to help select an optimal number of topics (K).

Notes:
- If you run this file in an IDE by clicking "Run" with no arguments, it will try
  to use the default input file: ./ny311_ready_900.csv (if present).
- For auto-selecting K, the default coherence metric is "u_mass" (no gensim needed).
  You can optionally use gensim's "c_v" if gensim+scipy are compatible on your system.

Basic usage (fixed K):
  python run_topic_models_with_coherence.py --input "ny311_ready_900.csv" --k 7

Auto-select K via coherence (tests a range and chooses best):
  python run_topic_models_with_coherence.py --input "ny311_ready_900.csv" --select_k --k_min 3 --k_max 12 --k_step 1

Outputs (created under ./outputs):
  - vectorization_summary.txt
  - lda_topics.csv
  - nmf_topics.csv
  - representative_examples.csv
  - coherence_scores.csv           (only when --select_k is used)
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

    # phone-ish patterns (very rough)
    s = re.sub(r"\b(\+?\d[\d\-\s\(\)]{7,}\d)\b", " ", s)

    # Keep letters/numbers and a few separators; replace the rest with space
    s = re.sub(r"[^a-z0-9\s\-_/]", " ", s)

    # normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_text_field(df: pd.DataFrame, include_resolution: bool) -> pd.Series:
    """
    Build the analysis text from the NY311 columns.
    Required (if present): Complaint Type, Descriptor
    Optional: Resolution Description (often missing)
    """
    cols = [c for c in ["Complaint Type", "Descriptor"] if c in df.columns]
    if include_resolution and "Resolution Description" in df.columns:
        cols.append("Resolution Description")
    if not cols:
        raise ValueError("Expected at least 'Complaint Type' or 'Descriptor' column in the input CSV.")

    text = (
        df[cols]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .map(clean_text)
    )
    text = text[text.str.len() > 0]
    return text


def vectorize_texts(texts: pd.Series):
    """
    Vectorize with:
      - CountVectorizer (for LDA)
      - TfidfVectorizer (for NMF)
    """
    count_vec = CountVectorizer(
        stop_words="english",
        min_df=2,
        max_df=0.9
    )
    tfidf_vec = TfidfVectorizer(
        stop_words="english",
        min_df=2,
        max_df=0.9
    )

    X_counts = count_vec.fit_transform(texts)
    X_tfidf = tfidf_vec.fit_transform(texts)

    return (X_counts, count_vec), (X_tfidf, tfidf_vec)


def top_words_from_components(components: np.ndarray, feature_names, top_n: int = 10):
    """Return list of topics as lists of top_n words."""
    topics = []
    for topic_idx in range(components.shape[0]):
        top_idx = np.argsort(components[topic_idx])[::-1][:top_n]
        topics.append([feature_names[i] for i in top_idx])
    return topics


def fit_models(X_counts, count_vec, X_tfidf, tfidf_vec, k: int, random_state: int = 42):
    lda = LatentDirichletAllocation(
        n_components=k,
        random_state=random_state,
        learning_method="batch"
    )
    nmf = NMF(
        n_components=k,
        random_state=random_state,
        init="nndsvda",
        max_iter=400
    )

    lda.fit(X_counts)
    nmf.fit(X_tfidf)

    lda_topics = top_words_from_components(lda.components_, count_vec.get_feature_names_out(), top_n=10)
    nmf_topics = top_words_from_components(nmf.components_, tfidf_vec.get_feature_names_out(), top_n=10)

    return lda, nmf, lda_topics, nmf_topics


def coherence_u_mass(X_bin, feature_names, topics, top_n: int = 10) -> float:
    """
    Simple UMass coherence (no gensim):
      C = avg_topic avg_{i>j} log((D(w_i,w_j)+1) / D(w_j))
    where D(·) are document frequencies using a binary doc-term matrix.
    """
    # doc frequency per term
    df = np.asarray(X_bin.sum(axis=0)).ravel().astype(float)

    # map term -> index
    idx = {t: i for i, t in enumerate(feature_names)}

    topic_scores = []
    for words in topics:
        words = [w for w in words[:top_n] if w in idx]
        if len(words) < 2:
            continue

        pairs = []
        for i in range(1, len(words)):
            wi = idx[words[i]]
            for j in range(i):
                wj = idx[words[j]]
                # docs containing both
                co = float(X_bin[:, wi].multiply(X_bin[:, wj]).sum())
                denom = df[wj] if df[wj] > 0 else 1.0
                pairs.append(np.log((co + 1.0) / denom))
        if pairs:
            topic_scores.append(float(np.mean(pairs)))

    return float(np.mean(topic_scores)) if topic_scores else float("nan")


def coherence_score(
    kind: str,
    *,
    texts_for_gensim=None,
    topics=None,
    X_bin=None,
    feature_names=None
) -> float:
    """
    Compute a coherence score.
    - u_mass: uses X_bin and feature_names (no gensim)
    - c_v: uses gensim and tokenized texts; if gensim fails, falls back to u_mass when possible
    """
    kind = (kind or "u_mass").lower()

    if kind == "u_mass":
        if X_bin is None or feature_names is None or topics is None:
            return float("nan")
        return coherence_u_mass(X_bin, feature_names, topics)

    if kind == "c_v":
        try:
            from gensim.corpora import Dictionary
            from gensim.models.coherencemodel import CoherenceModel
        except Exception:
            # fallback
            if X_bin is not None and feature_names is not None and topics is not None:
                return coherence_u_mass(X_bin, feature_names, topics)
            return float("nan")

        if texts_for_gensim is None or topics is None:
            return float("nan")

        dictionary = Dictionary(texts_for_gensim)
        # convert topics list of words -> list of lists
        cm = CoherenceModel(topics=topics, texts=texts_for_gensim, dictionary=dictionary, coherence="c_v")
        return float(cm.get_coherence())

    return float("nan")


def tokenize_for_gensim(texts: pd.Series):
    # Very lightweight tokenization, consistent with Count/Tfidf vectorizers' goal.
    return [t.split() for t in texts.tolist()]


def save_topics_csv(path: Path, topics, model_name: str):
    rows = []
    for i, words in enumerate(topics, start=1):
        rows.append({"model": model_name, "topic_id": i, "top_words": ", ".join(words)})
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def save_representative_examples(out_path: Path, raw_df: pd.DataFrame, texts: pd.Series, lda, count_vec, k: int):
    """
    Save a small set of representative rows for each topic (LDA-based),
    using the highest topic probability per document.
    """
    X_counts = count_vec.transform(texts)
    doc_topic = lda.transform(X_counts)

    # keep only rows aligned with 'texts' index (after dropping empty)
    aligned = raw_df.loc[texts.index].copy()
    aligned["clean_text"] = texts

    aligned["best_topic"] = np.argmax(doc_topic, axis=1) + 1
    aligned["best_topic_prob"] = np.max(doc_topic, axis=1)

    reps = (
        aligned.sort_values(["best_topic", "best_topic_prob"], ascending=[True, False])
        .groupby("best_topic", as_index=False)
        .head(3)
        .loc[:, ["best_topic", "best_topic_prob", "Complaint Type", "Descriptor", "Resolution Description", "clean_text"]]
    )
    reps.to_csv(out_path, index=False, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="ny311_ready_900.csv",
        help="Path to input CSV (default: ny311_ready_900.csv in the repo root)."
    )
    ap.add_argument("--k", type=int, default=7, help="Number of topics (K) when not using --select_k.")
    ap.add_argument("--include_resolution", action="store_true", help="Include 'Resolution Description' in the text (if present).")

    ap.add_argument("--select_k", action="store_true", help="Auto-select K by testing a range and scoring coherence.")
    ap.add_argument("--k_min", type=int, default=3, help="Minimum K to test when using --select_k.")
    ap.add_argument("--k_max", type=int, default=12, help="Maximum K to test when using --select_k.")
    ap.add_argument("--k_step", type=int, default=1, help="Step for K range when using --select_k.")
    ap.add_argument("--coherence", default="u_mass", choices=["u_mass", "c_v"], help="Coherence metric to use for K selection.")
    ap.add_argument("--choose_by", default="avg", choices=["lda", "nmf", "avg"], help="Which model's score to choose K by.")

    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {in_path}. "
            f"Either put the CSV in the repo root as 'ny311_ready_900.csv' "
            f"or run with: --input path/to/your.csv"
        )

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    df = pd.read_csv(in_path, low_memory=False, dtype=str)
    texts = build_text_field(df, include_resolution=args.include_resolution)

    # Vectorize once (K does not affect vectorization)
    (X_counts, count_vec), (X_tfidf, tfidf_vec) = vectorize_texts(texts)
    X_counts_bin = (X_counts > 0).astype(np.int8)
    X_tfidf_bin = (X_tfidf > 0).astype(np.int8)

    # Optional tokenized texts for gensim c_v
    tokenized_texts = tokenize_for_gensim(texts) if args.coherence == "c_v" else None

    # If selecting K, compute coherence over a range
    chosen_k = args.k
    scores_rows = []

    if args.select_k:
        k_values = list(range(args.k_min, args.k_max + 1, args.k_step))
        best_score = -np.inf
        best_k = k_values[0]

        for k in k_values:
            lda, nmf, lda_topics, nmf_topics = fit_models(X_counts, count_vec, X_tfidf, tfidf_vec, k=k)

            lda_score = coherence_score(
                args.coherence,
                texts_for_gensim=tokenized_texts,
                topics=lda_topics,
                X_bin=X_counts_bin,
                feature_names=count_vec.get_feature_names_out(),
            )
            nmf_score = coherence_score(
                args.coherence,
                texts_for_gensim=tokenized_texts,
                topics=nmf_topics,
                X_bin=X_tfidf_bin,
                feature_names=tfidf_vec.get_feature_names_out(),
            )

            avg_score = float(np.nanmean([lda_score, nmf_score]))

            scores_rows.append(
                {"k": k, "coherence": args.coherence, "lda_score": lda_score, "nmf_score": nmf_score, "avg_score": avg_score}
            )

            if args.choose_by == "lda":
                cur = lda_score
            elif args.choose_by == "nmf":
                cur = nmf_score
            else:
                cur = avg_score

            if np.isfinite(cur) and cur > best_score:
                best_score = cur
                best_k = k

        chosen_k = best_k
        pd.DataFrame(scores_rows).to_csv(out_dir / "coherence_scores.csv", index=False, encoding="utf-8")

    # Fit final models at chosen K
    lda, nmf, lda_topics, nmf_topics = fit_models(X_counts, count_vec, X_tfidf, tfidf_vec, k=chosen_k)

    # Save topics + examples
    save_topics_csv(out_dir / "lda_topics.csv", lda_topics, "LDA")
    save_topics_csv(out_dir / "nmf_topics.csv", nmf_topics, "NMF")
    save_representative_examples(out_dir / "representative_examples.csv", df, texts, lda, count_vec, chosen_k)

    # Summary text
    summary = [
        f"Input file: {in_path}",
        f"Rows used after text build/clean: {len(texts)}",
        f"Include resolution: {args.include_resolution}",
        f"Chosen K: {chosen_k} {'(selected by coherence)' if args.select_k else ''}",
        f"Vectorizer Count features: {X_counts.shape[1]}",
        f"Vectorizer TF-IDF features: {X_tfidf.shape[1]}",
    ]
    (out_dir / "vectorization_summary.txt").write_text("\n".join(summary), encoding="utf-8")

    print("DONE")
    print("\n".join(summary))
    print(f"Outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
