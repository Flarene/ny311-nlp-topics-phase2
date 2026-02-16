# NY 311 topic modeling

This repository contains a small, end-to-end NLP workflow that extracts common complaint themes from NYC 311 service requests using **two topic models**:

- **LDA** on **CountVectorizer** (bag-of-words)
- **NMF** on **TfidfVectorizer** (TF‑IDF)


## Data

- Dataset name: **NY 311 Service Requests** (City of New York 311 cases)
- Kaggle source: `https://www.kaggle.com/datasets/new-york-city/ny-311-service-requests`

To keep runtime reasonable, this repo includes a prepared sample: **`ny311_ready_900.csv`** (900 rows).  
The full raw Kaggle dataset is **not committed** (it is too large for standard GitHub pushes).

## The purpose of this project

The goal is to take short 311 complaint descriptions and automatically surface the **main topics** people report (e.g., parking issues, noise, street/sidewalk conditions). The output is designed to be easy for a reviewer to open and inspect (CSV + TXT files inside `outputs/`).

## How to run in compilers like PyCharm, VS Code, etc.

### 0) Open a terminal in the project folder
Make sure your terminal working directory is the repo root (the folder that contains `requirements.txt` and `run_topic_models_with_coherence.py`).

- **PyCharm:** open the built-in Terminal tab (it starts in the project root by default).
- **VS Code:** Terminal → New Terminal, then `cd` into the repo folder if needed.

### 1) Create/activate a virtual environment (recommended)

Windows (PowerShell):
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Run the pipeline (default: select K automatically)
From the repo root:
```bash
python run_topic_models_with_coherence.py --input ny311_ready_900.csv
```

By default, the script:
- builds the analysis text from `Complaint Type + Descriptor (+ Resolution Description when present)`
- cleans/normalizes the text
- vectorizes using **Count** and **TF‑IDF**
- tests a small K range (default **3–12**), computes **u_mass** coherence, and selects the best K
- trains **LDA (count)** and **NMF (TF‑IDF)** using the selected K
- writes outputs into `outputs/`

### Optional: customize the K search range
```bash
python run_topic_models_with_coherence.py --input ny311_ready_900.csv --k_min 3 --k_max 12 --k_step 1
```

### Optional: force a fixed K (skip K selection)
If you want to run with a specific K (e.g., for a quick test):
```bash
python run_topic_models_with_coherence.py --input ny311_ready_900.csv --no-select_k --k 7
```

## Outputs

All results are written into `outputs/` **and this folder is intentionally version-controlled** so a reviewer can open the artifacts directly.

- `outputs/coherence_scores.csv` – coherence per K (used to select K)
- `outputs/lda_topics.csv` – LDA topic keywords
- `outputs/nmf_topics.csv` – NMF topic keywords
- `outputs/representative_examples.csv` – example complaints per topic
- `outputs/preprocessing_summary.txt` – what cleaning steps were applied + row counts
- `outputs/vectorization_summary.txt` – vectorizer settings + feature counts
- `outputs/vectorization_comparison.txt` – brief comparison of Count vs TF‑IDF representations

## Notes on K selection

This project uses **u_mass coherence** to help pick K. For u_mass, values closer to **0** (less negative) indicate better coherence. In the final run artifacts provided with this repo, the best average coherence occurred at **K=10**.

---

If something fails to run, the first thing to check is the **working directory**: most “file not found” issues happen when the script is executed from a different folder than the repo root.
