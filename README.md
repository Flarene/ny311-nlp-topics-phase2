## Run (fixed K)
```bash
pip install -r requirements.txt
python run_topic_models_with_coherence.py --input ny311_ready_900.csv --k 7
```

## Auto-select K with coherence
```bash
python run_topic_models_with_coherence.py --input ny311_ready_900.csv --select_k --k_min 3 --k_max 12 --k_step 1 --choose_by avg
```

### PyCharm (Run button)
If you run by clicking **Run**, add Script parameters:
`--input ny311_ready_900.csv --select_k --k_min 3 --k_max 12`
