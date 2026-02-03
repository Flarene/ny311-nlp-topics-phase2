# NYC 311 Topic Modeling (NLP) — Phase 2

This repo runs a small, end-to-end topic modeling workflow on a **900-row sample** of the NYC 311 service requests dataset.
The goal is to surface the **most common complaint themes** using two vectorizers and two topic-modeling techniques.

## What’s inside

Pipeline (in order):

1. Build one analysis text field from **Complaint Type + Descriptor**  
   (optionally **+ Resolution Description** when it exists and you enable it)
2. Clean the text (lowercase, collapse whitespace, remove URLs/emails/phone-like strings, strip noisy punctuation)
3. Vectorize the text in two ways:
   - **CountVectorizer** (bag-of-words counts)
   - **TfidfVectorizer** (TF-IDF weights)
4. Extract topics in two ways:
   - **LDA** (trained on counts)
   - **NMF** (trained on TF-IDF)
5. Select the number of topics **K** by testing a small range and scoring coherence (**u_mass**)

Repo link (code + outputs): https://github.com/Flarene/ny311-nlp-topics-phase2

---

## Dataset

- Original source: NYC 311 “Service Requests” (via Kaggle)
- This repo includes a small prepared sample: **`ny311_ready_900.csv`**
- The full raw Kaggle file is **not** stored here because it is too large for GitHub.

---

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Run

From the repo root:

```bash
python run_topic_models_with_coherence.py --input ny311_ready_900.csv --select_k --k_min 3 --k_max 12 --k_step 1 --choose_by avg
```

To also use the “Resolution Description” text when available:

```bash
python run_topic_models_with_coherence.py --input ny311_ready_900.csv --include_resolution --select_k --k_min 3 --k_max 12 --k_step 1 --choose_by avg
```

Tip (PyCharm): set **Working directory** to the repo root so relative paths like `ny311_ready_900.csv` work.

---

## Outputs (committed for easy review)

All results are written to `outputs/`:

- `outputs/coherence_scores.csv` — coherence score per K (used to pick the final K)
- `outputs/lda_topics.csv` — top keywords per LDA topic
- `outputs/nmf_topics.csv` — top keywords per NMF topic
- `outputs/representative_examples.csv` — example complaints per topic (helps interpret topics)
- `outputs/preprocessing_summary.txt` — quick summary of how the text was cleaned
- `outputs/vectorization_summary.txt` — vectorizer settings + feature counts
- `outputs/vectorization_comparison.txt` — a small, evidence-based comparison (vocab size, sparsity, top terms)

---

## Common issues

- **FileNotFoundError**: run from the repo root, or pass a full path:
  - Windows: `--input "C:\Users\...\PythonProject2\ny311_ready_900.csv"`
- **Interpreter / missing packages**: make sure PyCharm is using the project `.venv`, then run `pip install -r requirements.txt`.

