# NYC 311 Topic Modeling (NLP)

This repo runs a small, end-to-end NLP workflow on a **900-row sample** of the NYC 311 service requests dataset to surface the **most common complaint topics**.

What the pipeline does:

- Builds one analysis text field from **Complaint Type** + **Descriptor** + **Resolution Description** (when the column is available)
- Cleans the text (basic normalization + removing obvious noise)
- Vectorizes the text two ways: **Bag-of-Words (CountVectorizer)** and **TF-IDF (TfidfVectorizer)**
- Extracts topics two ways: **LDA** (on counts) and **NMF** (on TF-IDF)
- Evaluates a small range of topic counts (**K**) using a **coherence score** and selects the best K

---

## Dataset

- **Name:** *NY 311 Service Requests* (City of New York 311 cases)
- **Source:** Kaggle — `new-york-city/ny-311-service-requests`
  - Link: `https://www.kaggle.com/datasets/new-york-city/ny-311-service-requests`
- This repo includes only a prepared sample: **`ny311_ready_900.csv`** (900 rows) to keep runtime reasonable and the repository lightweight.
- The full raw dataset is **not committed** (it is too large for standard GitHub pushes). If you want the full file, download it from Kaggle and keep it locally.

---

## Setup

From the repo root:

```bash
pip install -r requirements.txt
```

---

## Run

### 1) Coherence-based K selection

This runs the full pipeline and tests a K range (default 3–12) to choose a good number of topics:

```bash
python run_topic_models_with_coherence.py --input ny311_ready_900.csv --select_k --k_min 3 --k_max 12 --k_step 1 --choose_by avg
```

Notes:
- `--input` is optional (defaults to `ny311_ready_900.csv` in the repo root), but it’s safer to pass it explicitly if you run from an IDE with a different working directory.
- The selected K is printed in the console output.

### 2) Fixed K (quick run)

If you already know K:

```bash
python run_topic_models_with_coherence.py --input ny311_ready_900.csv --k 7
```

---

## Outputs

Results are written into `outputs/` (and this folder is kept in the repo so reviewers can open files directly):

- `outputs/coherence_scores.csv` – coherence per K (used to select the final K)
- `outputs/lda_topics.csv` – LDA topic keywords
- `outputs/nmf_topics.csv` – NMF topic keywords
- `outputs/representative_examples.csv` – example rows per topic
- `outputs/preprocessing_summary.txt` – quick summary of cleaning steps
- `outputs/vectorization_summary.txt` – vectorizer settings + feature counts
- `outputs/vectorization_comparison.txt` – simple quantitative comparison of count vs TF-IDF representations



