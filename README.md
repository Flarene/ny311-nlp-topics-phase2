# NY311 NLP Topic Modeling (Conception Phase) — User Guide

This repository implements the **conception-phase plan** for topic modeling on a reduced sample (900 rows) of the NYC 311 service requests dataset.

It performs:

- Build one text field from: **Complaint Type + Descriptor + (when available) Resolution Description**
- Preprocess the text (drop empty, remove duplicates, normalize & clean noise)
- Vectorize the text using:
  - **CountVectorizer** (bag-of-words counts)
  - **TfidfVectorizer** (TF-IDF)
- Train topic models:
  - **LDA** on count vectors
  - **NMF** on TF-IDF vectors
- Evaluate candidate topic counts **K** using a **coherence score** and select the best K
- Write all outputs into an `outputs/` folder (kept in the repo for reviewers)

---

## 1) Requirements

- Python **3.9+** recommended (3.10 / 3.11 also fine)
- Install dependencies:

```bash
pip install -r requirements.txt
```

> If you use a virtual environment (recommended), create & activate it first.

---

## 2) Files you should have

- `run_topic_models_with_coherence_v2.py` (or v3 if you kept that name)
- `text_prep.py` (shared preprocessing functions)
- `ny311_ready_900.csv` (the dataset)
- `requirements.txt`

---

## 3) Dataset location (Windows)

You can run with an **absolute path**:

```bash
python run_topic_models_with_coherence_v2.py --input "C:\Users\skenr\Desktop\PythonProject2\ny311_ready_900.csv" --select_k --k_min 3 --k_max 12 --k_step 1 --choose_by avg
```

Or from the repo root (recommended), use a relative path:

```bash
python run_topic_models_with_coherence_v2.py --input ny311_ready_900.csv --select_k --k_min 3 --k_max 12 --k_step 1 --choose_by avg
```

---

## 4) What the program does (pipeline)

### Step A — Build “analysis text”
Creates one field by concatenating:
- Complaint Type
- Descriptor
- Resolution Description (only if that column exists)

### Step B — Preprocess (cleaning)
The preprocessing follows the conception plan:

1. **Drop missing/empty text**
2. **Delete duplicates**
   - duplicates in the dataset
   - duplicate cleaned text entries
3. **Normalize**
   - lowercase
   - normalize whitespace (multiple spaces/tabs/newlines -> single space)
4. **Noise removal**
   - remove URLs, email addresses, phone-number-like strings
   - replace punctuation/symbols with spaces

A summary is saved as: `outputs/preprocessing_summary.txt`

### Step C — Vectorize
- Count vectors: `CountVectorizer`
- TF-IDF vectors: `TfidfVectorizer`

A summary is saved as: `outputs/vectorization_summary.txt`

### Step D — Topic modeling
- LDA on counts
- NMF on TF-IDF

Topics are exported to:
- `outputs/lda_topics.csv`
- `outputs/nmf_topics.csv`

### Step E — Select the best K using coherence (when enabled)
When you pass `--select_k`, the script tests a small range of K values and writes:

- `outputs/coherence_scores.csv`

Then it picks the best K based on `--choose_by`:
- `lda` (LDA coherence only)
- `nmf` (NMF coherence only)
- `avg` (average of LDA & NMF)

### Step F — Representative examples
Creates a file that maps each topic to a few representative rows from the dataset:

- `outputs/representative_examples.csv`

---

## 5) Output folder (reviewer-friendly)

All artifacts are written to:

```
outputs/
  preprocessing_summary.txt
  vectorization_summary.txt
  coherence_scores.csv
  lda_topics.csv
  nmf_topics.csv
  representative_examples.csv
```

This repo intentionally **keeps** `outputs/` in version control so reviewers can inspect results without running code.

---

## 6) Common issues & fixes

### “FileNotFoundError”
- Ensure you run from the repo root OR pass an absolute `--input` path.

### “Module not found”
- Run `pip install -r requirements.txt`
- Confirm you're using the same Python environment you installed packages into.

### Coherence note
If your environment has gensim installed, coherence computations may use additional metrics; otherwise it falls back to the built-in UMass coherence.

---

## 7) Suggested reviewer workflow (no execution required)

1. Open `outputs/coherence_scores.csv` to see the tested K values.
2. Open `outputs/lda_topics.csv` and `outputs/nmf_topics.csv` to inspect topic keywords.
3. Open `outputs/representative_examples.csv` to see example complaints per topic.
4. Read `outputs/preprocessing_summary.txt` and `outputs/vectorization_summary.txt` to confirm the pipeline matches the conception phase.
