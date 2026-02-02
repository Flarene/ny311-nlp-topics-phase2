# NY311 NLP Topic Modeling (Phase 2)

This project extracts the most prevalent complaint topics from NYC 311 service requests using two topic-modeling pipelines:

- **CountVectorizer + LDA**
- **TF-IDF + NMF**

It also supports **coherence scoring** to help choose an appropriate number of topics (**K**), and an optional flag to include the **Resolution Description** field when available.

## 1) Install dependencies
```bash
pip install -r requirements.txt
```

## 2) Data input
The default expected input file is:
- `./ny311_ready_900.csv`

You can also pass any CSV path via `--input` (relative or absolute).

## 3) Run the analysis

### A) Run with a fixed number of topics (K)
```bash
python run_topic_models_with_coherence_v2.py --input ny311_ready_900.csv --k 7
```

### B) Auto-select K using coherence (recommended)
This tests a K range and selects the best K by the chosen score.
```bash
python run_topic_models_with_coherence_v2.py --input ny311_ready_900.csv --select_k --k_min 3 --k_max 12 --k_step 1 --choose_by avg
```

### C) Include Resolution Description (optional)
If you want the model input text to also include `Resolution Description` **when it exists**, add:
- `--include_resolution`

Example:
```bash
python run_topic_models_with_coherence_v2.py --input ny311_ready_900.csv --select_k --k_min 3 --k_max 12 --include_resolution
```

## 4) Outputs
All files are written under `./outputs/`:

- `vectorization_summary.txt`
- `lda_topics.csv`
- `nmf_topics.csv`
- `representative_examples.csv`
- `coherence_scores.csv` (only when `--select_k` is used)

## PyCharm (Run button)
If you run by clicking **Run**, you must set **Script parameters** (otherwise `--input` may be missing).

Example Script parameters:
```
--input ny311_ready_900.csv --select_k --k_min 3 --k_max 12 --choose_by avg
```

If you get `FileNotFoundError`, check that:
- `ny311_ready_900.csv` is in the **project root**, or pass the full path in `--input`
- the **Working directory** in the Run Configuration points to the project root
