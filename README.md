# NYC 311 Topic Modeling (NYC 311 Service Requests)


## Purpose
This program analyzes the **NYC 311 complaint dataset** using **NLP topic modeling**. It cleans and vectorizes the text, evaluates topic quality (coherence) across different topic counts, and generates a compact set of **interpretable topics** plus ready-to-use report artifacts.


# How to run this project in (PyCharm, VS Code, etc.)

## 1) Open a terminal
Open a terminal **in the folder where you want the project to live**.

- **VS Code:** Terminal → *New Terminal*
- **PyCharm:** Open the built-in *Terminal* tab


## 2) Clone the repository
Run:

```bash
git clone https://github.com/Flarene/ny311-nlp-topics-phase2.git
cd ny311-nlp-topics-phase2
```

## 3) Create + activate a virtual environment
**Windows (PowerShell) terminal in (PyCharm, VS Code, etc.) :**
```bash
python -m venv .venv
```

**macOS / Linux:**
```bash
python3 -m venv .venv
```

## 4) Install dependencies
```bash
pip install -r requirements.txt
```

## 5) Run topic modelling with coherence-based K selection:
```bash
python run_topic_models_with_coherence.py --input ny311_ready_900.csv
```

Outputs are written to `outputs/`.
