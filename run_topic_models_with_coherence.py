"""run_topic_models_with_coherence_v3.py

NYC 311 Topic Modeling — Conception Phase (implementation)

Goals implemented (matching the provided conception phase plan):
- Sample data source: NY 311 Service Requests CSV (e.g., 900 samples)
- Complaint information used for analysis:
    * 'Complaint Type'
    * 'Descriptor'
    * and when available, 'Resolution Description'
  These columns are concatenated into one input text field.
- Preprocessing plan (cleaning text):
    (1) Drop rows with missing/empty text and delete duplicates.
    (2) Normalize: lowercase + normalize whitespace.
    (3) Noise removal: remove URLs, email addresses, phone-like strings; strip
        non-informative punctuation/symbols by replacing with spaces.
        Domain terms like "hot water" and "illegal parking" are preserved as words;
        bigrams are enabled by default so such phrases can be retained as features.
- Vectorization:
    * CountVectorizer (bag-of-words counts) for LDA
    * TfidfVectorizer (TF-IDF) for NMF
- Topic extraction:
    * LDA on count vectors
    * NMF on TF-IDF vectors
  A small K range can be tested and the best K chosen using a coherence score.

Outputs (written to ./outputs by default):
- vectorization_summary.txt
- preprocessing_summary.txt
- lda_topics.csv
- nmf_topics.csv
- representative_examples.csv
- coherence_scores.csv (only when --select_k is used)

Usage examples:
  python run_topic_models_with_coherence_v3.py --input ny311_ready_900.csv --k 7
  python run_topic_models_with_coherence_v3.py --input ny311_ready_900.csv --select_k --k_min 3 --k_max 12
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from text_prep import apply_conception_preprocessing


def vectorize_texts(
    texts: pd.Series,
    *,
    stop_words: Optional[str],
    min_df: int,
    max_df: float,
    ngram_max: int,
) -> Tuple[Tuple[np.ndarray, CountVectorizer], Tuple[np.ndarray, TfidfVectorizer]]:
    """Vectorize texts using CountVectorizer (for LDA) and TfidfVectorizer (for NMF).

    To make coherence computation consistent across models, the TF-IDF vectorizer
    is forced to reuse the CountVectorizer vocabulary.
    """
    ngram_range = (1, max(1, int(ngram_max)))

    count_vec = CountVectorizer(
        stop_words=stop_words,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
    )
    X_counts = count_vec.fit_transform(texts)

    tfidf_vec = TfidfVectorizer(
        vocabulary=count_vec.vocabulary_,
        stop_words=stop_words,
        ngram_range=ngram_range,
    )
    X_tfidf = tfidf_vec.fit_transform(texts)

    return (X_counts, count_vec), (X_tfidf, tfidf_vec)


def top_words_from_components(components: np.ndarray, feature_names: np.ndarray, top_n: int = 10) -> List[List[str]]:
    """Return topics as lists of top_n words."""
    topics: List[List[str]] = []
    for topic_idx in range(components.shape[0]):
        top_idx = np.argsort(components[topic_idx])[::-1][:top_n]
        topics.append([feature_names[i] for i in top_idx])
    return topics


def fit_models(X_counts, X_tfidf, feature_names: np.ndarray, k: int, random_state: int = 42):
    """Fit LDA (counts) and NMF (tf-idf) and return models + top-word topics."""
    lda = LatentDirichletAllocation(
        n_components=k,
        random_state=random_state,
        learning_method="batch",
        max_iter=25,
    )
    nmf = NMF(
        n_components=k,
        random_state=random_state,
        init="nndsvda",
        max_iter=600,
    )

    lda.fit(X_counts)
    nmf.fit(X_tfidf)

    lda_topics = top_words_from_components(lda.components_, feature_names, top_n=10)
    nmf_topics = top_words_from_components(nmf.components_, feature_names, top_n=10)

    return lda, nmf, lda_topics, nmf_topics


def coherence_u_mass(X_bin, feature_names: np.ndarray, topics: List[List[str]], top_n: int = 10) -> float:
    """UMass coherence using a binary doc-term matrix (no gensim required)."""
    df = np.asarray(X_bin.sum(axis=0)).ravel().astype(float)
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
                co = float(X_bin[:, wi].multiply(X_bin[:, wj]).sum())
                denom = df[wj] if df[wj] > 0 else 1.0
                pairs.append(np.log((co + 1.0) / denom))

        if pairs:
            topic_scores.append(float(np.mean(pairs)))

    return float(np.mean(topic_scores)) if topic_scores else float("nan")


def tokenize_for_gensim(texts: pd.Series, *, ngram_max: int, stop_words: Optional[str]) -> List[List[str]]:
    """Create tokens for gensim coherence.

    Matches the pipeline intent:
    - Basic whitespace tokenization
    - Optional stopword filtering
    - If ngram_max >= 2, include bigrams as single tokens (joined with a space)
      so tokens can match sklearn's bigram feature names.
    """
    sw = set(ENGLISH_STOP_WORDS) if stop_words == "english" else set()

    tokenized: List[List[str]] = []
    for t in texts.tolist():
        toks = [w for w in t.split() if w and (w not in sw)]

        if ngram_max >= 2:
            # add bigrams as tokens with space (e.g., "illegal parking")
            bigrams = [f"{toks[i]} {toks[i+1]}" for i in range(len(toks) - 1)]
            toks = toks + bigrams

        tokenized.append(toks)

    return tokenized


def coherence_c_v(texts_tokenized: List[List[str]], topics: List[List[str]]) -> float:
    """Compute c_v coherence using gensim if available."""
    try:
        from gensim.corpora import Dictionary
        from gensim.models.coherencemodel import CoherenceModel
    except Exception:
        return float("nan")

    dictionary = Dictionary(texts_tokenized)
    cm = CoherenceModel(topics=topics, texts=texts_tokenized, dictionary=dictionary, coherence="c_v")
    return float(cm.get_coherence())


def save_topics_csv(path: Path, topics: List[List[str]], model_name: str):
    rows = []
    for i, words in enumerate(topics, start=1):
        rows.append({"model": model_name, "topic_id": i, "top_words": ", ".join(words)})
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def save_representative_examples(out_path: Path, df_clean: pd.DataFrame, texts: pd.Series, lda, count_vec):
    """Save representative rows for each topic (LDA-based) using highest topic probability."""
    X_counts = count_vec.transform(texts)
    doc_topic = lda.transform(X_counts)

    aligned = df_clean.loc[texts.index].copy()
    aligned["clean_text"] = texts
    aligned["best_topic"] = np.argmax(doc_topic, axis=1) + 1
    aligned["best_topic_prob"] = np.max(doc_topic, axis=1)

    # Ensure columns exist before selecting
    cols = ["best_topic", "best_topic_prob"]
    for c in ["Complaint Type", "Descriptor", "Resolution Description", "clean_text"]:
        if c in aligned.columns:
            cols.append(c)

    reps = (
        aligned.sort_values(["best_topic", "best_topic_prob"], ascending=[True, False])
        .groupby("best_topic", as_index=False)
        .head(3)
        .loc[:, cols]
    )

    reps.to_csv(out_path, index=False, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        default="ny311_ready_900.csv",
        help="Path to input CSV (default: ny311_ready_900.csv in the repo root).",
    )

    # K selection
    ap.add_argument("--k", type=int, default=7, help="Number of topics (K) when not using --select_k.")
    ap.add_argument("--select_k", action="store_true", help="Auto-select K using coherence over a K range.")
    ap.add_argument("--k_min", type=int, default=3, help="Minimum K to test when using --select_k.")
    ap.add_argument("--k_max", type=int, default=12, help="Maximum K to test when using --select_k.")
    ap.add_argument("--k_step", type=int, default=1, help="Step for K range when using --select_k.")
    ap.add_argument("--coherence", default="u_mass", choices=["u_mass", "c_v"], help="Coherence metric.")
    ap.add_argument("--choose_by", default="avg", choices=["lda", "nmf", "avg"], help="Choose K by score source.")

    # Vectorization knobs (kept simple; defaults are sensible for 900 samples)
    ap.add_argument("--stop_words", default="english", choices=["english", "none"], help="Stopword removal.")
    ap.add_argument("--min_df", type=int, default=2, help="Count/Tf-idf min_df.")
    ap.add_argument("--max_df", type=float, default=0.9, help="Count/Tf-idf max_df.")
    ap.add_argument("--ngram_max", type=int, default=2, help="Use 1..N grams; default 2 keeps phrases like 'hot water'.")

    # Output
    ap.add_argument("--outputs", default="outputs", help="Output directory (default: ./outputs)")

    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {in_path}. "
            f"Either put the CSV in the repo root as 'ny311_ready_900.csv' "
            f"or run with: --input path/to/your.csv"
        )

    out_dir = Path(args.outputs)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path, low_memory=False, dtype=str)

    # ------------------------------
    # Preprocessing (conception phase)
    # ------------------------------
    df_clean, texts, prep_report = apply_conception_preprocessing(df)

    (out_dir / "preprocessing_summary.txt").write_text(
        "\n".join(prep_report.to_lines()),
        encoding="utf-8",
    )

    # ------------------------------
    # Vectorization
    # ------------------------------
    stop_words = None if args.stop_words == "none" else "english"

    (X_counts, count_vec), (X_tfidf, tfidf_vec) = vectorize_texts(
        texts,
        stop_words=stop_words,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_max=args.ngram_max,
    )

    feature_names = count_vec.get_feature_names_out()
    X_bin = (X_counts > 0).astype(np.int8)

    # Optional tokenized texts for gensim coherence (if requested)
    tokenized_texts = None
    if args.coherence == "c_v":
        tokenized_texts = tokenize_for_gensim(texts, ngram_max=args.ngram_max, stop_words=stop_words)

    # ------------------------------
    # K selection by coherence
    # ------------------------------
    chosen_k = args.k
    scores_rows = []

    if args.select_k:
        k_values = list(range(args.k_min, args.k_max + 1, args.k_step))
        best_score = -np.inf
        best_k = k_values[0]

        for k in k_values:
            lda, nmf, lda_topics, nmf_topics = fit_models(X_counts, X_tfidf, feature_names, k=k)

            if args.coherence == "u_mass":
                lda_score = coherence_u_mass(X_bin, feature_names, lda_topics)
                nmf_score = coherence_u_mass(X_bin, feature_names, nmf_topics)
            else:
                # c_v requires gensim; if unavailable, scores may be NaN
                lda_score = coherence_c_v(tokenized_texts, lda_topics) if tokenized_texts else float("nan")
                nmf_score = coherence_c_v(tokenized_texts, nmf_topics) if tokenized_texts else float("nan")

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

    # ------------------------------
    # Fit final models and export
    # ------------------------------
    lda, nmf, lda_topics, nmf_topics = fit_models(X_counts, X_tfidf, feature_names, k=chosen_k)

    save_topics_csv(out_dir / "lda_topics.csv", lda_topics, "LDA")
    save_topics_csv(out_dir / "nmf_topics.csv", nmf_topics, "NMF")
    save_representative_examples(out_dir / "representative_examples.csv", df_clean, texts, lda, count_vec)

    # Summary text
    summary = [
        f"Input file: {in_path}",
        f"Rows used after preprocessing: {len(texts)}",
        f"Chosen K: {chosen_k} {'(selected by coherence)' if args.select_k else ''}",
        f"Coherence metric (if selecting K): {args.coherence}",
        f"Vectorizer stop_words: {stop_words}",
        f"Vectorizer ngram_range: (1,{args.ngram_max})",
        f"Vectorizer min_df: {args.min_df}",
        f"Vectorizer max_df: {args.max_df}",
        f"Count features: {X_counts.shape[1]}",
        f"TF-IDF features: {X_tfidf.shape[1]}",
    ]

    (out_dir / "vectorization_summary.txt").write_text("\n".join(summary), encoding="utf-8")

    print("DONE")
    print("\n".join(summary))
    print(f"Outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
