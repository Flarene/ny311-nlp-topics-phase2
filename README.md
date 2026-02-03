# NYC 311 Topic Modeling (NLP)

This repo runs a small, end-to-end NLP workflow on a **900-row sample** of the NYC 311 service requests dataset to surface the **most common complaint topics**.

What it does, in order:
- Build one analysis text field from **Complaint Type** + **Descriptor** (+ **Resolution Description** when you choose to include it)
- Clean the text (basic normalization + removing obvious noise)
- Vectorize the text in two ways: **Bag-of-Words (CountVectorizer)** and **TF-IDF (TfidfVectorizer)**
- Extract topics in two ways: **LDA** (on counts) and **NMF** (on TF-IDF)
- **Compute coherence over a range of topic counts (K) and select the best K** 

---

## Dataset

- Source: NYC 311 “NY 311 Service Requests” (via Kaggle)
- This repo includes only a small sample: **`ny311_ready_900.csv`**
- The full raw dataset is intentionally **not** stored here (it is too large for GitHub).

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Run (required: coherence-based K selection)

From the repo root, run coherence scoring to choose K:

```bash
python run_topic_models_with_coherence.py --input ny311_ready_900.csv --select_k --k_min 3 --k_max 12 --k_step 1 --choose_by avg
```

This will:
- try multiple values of **K**
- write coherence results to `outputs/coherence_scores.csv`
- pick a final K (printed in the console output)
- export LDA + NMF topics and example rows into `outputs/`

### Include “Resolution Description” (optional)

If you want the model to also use the resolution text (when present), add:

```bash
--include_resolution
```

> Tip (PyCharm/IDE): if you click **Run** without arguments, you’ll see an error saying `--input` is required.  
> Add the parameters in **Run/Debug Configurations → Parameters**.

---

## Outputs

Results are written into `outputs/` (and this folder is kept in the repo so reviewers can open files directly):

- `outputs/coherence_scores.csv` – coherence per K (used to select the final K)
- `outputs/lda_topics.csv` – LDA topic keywords
- `outputs/nmf_topics.csv` – NMF topic keywords
- `outputs/representative_examples.csv` – example rows per topic
- `outputs/preprocessing_summary.txt` – quick summary of cleaning steps
- `outputs/vectorization_summary.txt` – vectorizer settings + feature counts

---

## Common issues

- **FileNotFoundError**: run from the repo root, or pass a full path like:
  - Windows: `--input "C:\\path\\to\\repo\\ny311_ready_900.csv"`
  - macOS/Linux: `--input "/path/to/repo/ny311_ready_900.csv"`
- **Missing packages**: make sure you installed into the same Python environment you’re running.
