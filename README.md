# ML Batch Prediction Pipeline with AI Analysis

Built an end-to-end batch ML pipeline using Python, scikit-learn, Docker, and GitHub Actions, extended with automated evaluation, real pytest-based validation, and AI-powered post-run analysis using Ollama and ChromaDB RAG.

A portfolio project that combines a production-style batch machine learning pipeline with post-run AI analysis.

This project trains a churn prediction model, performs batch inference on new input data, evaluates model performance, runs automated tests, and generates an AI-written analysis report using Ollama and ChromaDB-based RAG.

---

## Why this project matters

Most beginner ML projects stop at model training.

This project goes further and shows the surrounding engineering work that companies actually care about:

- batch ML pipeline design
- model persistence and reuse
- Docker-based execution
- GitHub Actions CI/CD automation
- structured evaluation artifacts
- real pytest-based validation
- AI-assisted post-run analysis
- local RAG pipeline using Ollama + ChromaDB

---

## What it does

### P1 capabilities
- trains a churn prediction model from historical labeled CSV data
- saves the trained model as a reusable artifact
- loads new batch input data
- generates predictions in batch mode
- saves prediction outputs and logs

### P3 capabilities
- evaluates model performance on the held-out test split
- creates structured metrics artifacts
- runs real pytest tests and exports machine-readable test results
- builds a local vector store from project knowledge files
- retrieves relevant RAG context for analysis
- uses Ollama to generate a human-readable summary of the run
- packages all outputs into final analysis artifacts

---

## Data Stack

### Files

#### 1. `training_data.csv`
- 1500 labeled rows
- Use this to train and test a churn prediction model
- Target column: `churn`
  - `1` = churned
  - `0` = retained

#### 2. `input_data.csv`
- 300 unlabeled rows
- Use this as new incoming batch data for inference
- This file does not contain the `churn` column because the model should predict it

---

### Columns

- `customer_id`: unique customer identifier
- `age`: customer age
- `monthly_spend`: current recurring spend
- `tenure_months`: months as a customer
- `support_tickets`: number of support tickets raised
- `last_login_days`: days since last login
- `contract_type`: Monthly, Quarterly, or Annual
- `payment_method`: Card, Bank Transfer, UPI, or Wallet
- `region`: customer region
- `num_products`: number of subscribed products
- `discount_used`: `1` if discount applied, else `0`
- `avg_session_minutes`: average session length

---

### Suggested local folders

- `data/training_data.csv`
- `input/input_data.csv`


## Tech stack

- Python
- pandas
- scikit-learn
- joblib
- pytest
- pytest-json-report
- Docker
- GitHub Actions
- Ollama
- ChromaDB

---

## End-to-end flow

```text
training_data.csv
    ↓
train.py
    ↓
model.pkl
    ↓
predict.py + input_data.csv
    ↓
predictions output
    ↓
evaluate.py
    ↓
metrics.json + predictions_summary.json
    ↓
pytest + p3_analyze.py
    ↓
test_results.json + llm_summary.md + llm_summary.json
    ↓
build_run_payload.py
    ↓
run_payload.json

```

## Key outputs

After a successful run, the pipeline produces:

- `models/model.pkl`  
  Trained model artifact

- `artifacts/metrics.json`  
  Structured evaluation metrics

- `artifacts/predictions_summary.json`  
  Summary of the latest prediction batch

- `artifacts/test_results.json`  
  Real pytest JSON report

- `artifacts/llm_summary.md`  
  Human-readable AI analysis

- `artifacts/llm_summary.json`  
  Machine-readable LLM output with context and prompt details

- `artifacts/run_payload.json`  
  Final combined payload for downstream review

## Example engineering value

This project demonstrates how to move from “I trained a model” to “I built an ML workflow.”

It shows:

- repeatable ML execution
- containerized pipeline stages
- branch-based safe development
- CI/CD integration
- metrics consistency checks
- real test-report generation
- AI-generated operational summaries
- artifact-driven reporting

## What I built and debugged

During development, the project required fixing real pipeline issues, including:

- inconsistent evaluation logic between training and evaluation
- missing files in GitHub Actions branches
- broken dependency installation in Docker
- placeholder test result artifacts replaced with real pytest output
- missing LLM summary generation in CI
- incorrect prediction summary column detection
- RAG retrieval integration issues
- metric interpretation errors in generated summaries


## GitHub Actions workflow

The GitHub Actions pipeline includes three jobs:

### Train
- builds the Docker image
- trains the model
- uploads the trained model artifact

### Predict
- downloads the trained model
- runs batch prediction
- uploads prediction outputs and logs

### Analyze
- installs Python dependencies
- installs and starts Ollama
- evaluates the model
- runs pytest and exports the test report
- ingests the RAG knowledge base
- generates the AI summary
- uploads analysis artifacts

---

## Current status

This project is now working end to end for both local runs and CI runs.

It is a strong portfolio project because it combines:

- machine learning
- automation
- testing
- CI/CD
- LLM integration
- RAG integration
- debugging and iteration

---

## Current limitations

This is still a portfolio project, not a production platform.

Known limitations include:

- model quality is moderate
- test coverage is minimal
- no experiment tracking system yet
- no cloud deployment yet
- no monitoring dashboard yet
- no drift detection yet
- no scheduled retraining yet