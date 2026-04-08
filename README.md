# Project 1: Batch Prediction Pipeline

A production-style batch machine learning pipeline built locally with Python, Pandas, scikit-learn, Docker, and GitHub Actions.

This project simulates how companies run ML in batch mode.

Instead of serving live predictions through an API, the system:

- trains a model on historical labeled data
- waits for a new input CSV file
- processes the file in batch
- generates predictions
- saves results for downstream use
- archives processed input files

---

## What this project does

This pipeline solves a simple business-style ML problem using batch prediction.

### Example flow

1. Historical labeled data is used to train a model
2. The trained model is saved as `model.pkl`
3. A new CSV file is dropped into the input folder
4. The batch job reads that file
5. The model generates predictions
6. Predictions are written to an output CSV
7. The input file is moved to archive

This is close to how many real ML batch systems work in companies.

---

## Why this project matters

This project helps demonstrate practical ML engineering skills, not just notebook-based model training.

It shows:

- training vs inference separation
- model artifact handling
- folder-based batch ingestion
- repeatable pipeline runs
- Docker-based execution
- CI/CD with GitHub Actions
- production-style project structure

---

## End-to-end pipeline flow

### Training flow

- Read `training_data.csv`
- Clean and preprocess data
- Split data into train and test sets
- Train a machine learning model
- Evaluate the model
- Save trained artifact as `models/model.pkl`

### Prediction flow

- Watch for a new input CSV in the raw/input folder
- Load `models/model.pkl`
- Apply the same preprocessing logic
- Predict outcomes for new rows
- Save output into predictions folder
- Move processed input file into archive

## Main components

### 1. `train.py`

Responsible for:

- loading historical labeled data
- preprocessing features
- training the ML model
- evaluating the model
- saving the trained model artifact

**Output:**

- `models/model.pkl`

### 2. `predict.py`

Responsible for:

- loading new batch input data
- loading the saved model
- generating predictions
- saving prediction output
- archiving processed input files

**Outputs:**

- `data/processed/predictions.csv`
- archived input files in `archive/`

### 3. `Dockerfile`

Used to package the pipeline into a container so it runs the same way everywhere.

This is useful because:

- local machine behavior becomes consistent
- GitHub Actions can run the same image
- dependency issues are reduced

### 4. GitHub Actions workflow

Used to automate the batch pipeline in CI/CD.

It can:

- build the Docker image
- run training
- run prediction
- upload artifacts like trained model and reports

---

## Tech stack

- **Python**
- **Pandas**
- **scikit-learn**
- **joblib / pickle**
- **Docker**
- **GitHub Actions**

---

## Optional future upgrades

- SQLite or Postgres
- Prefect or Airflow
- AWS S3
- AWS Lambda trigger
- MLflow
- monitoring and alerts

## High-level architecture

### Architecture diagram

flowchart TD
    A[training_data.csv<br/>Historical labeled data] --> B[train.py]
    B --> C[Preprocessing]
    C --> D[Model Training]
    D --> E[model.pkl]

    F[input_data.csv<br/>New unseen batch file] --> G[predict.py]
    E --> G
    G --> H[Preprocessing for inference]
    H --> I[Generate predictions]
    I --> J[predictions.csv]
    G --> K[Move processed input to archive]

    J --> L[reports / outputs]
    K --> M[archive folder]

    ## Key concepts demonstrated

### 1. Training vs inference

These are separate steps.

- training builds the model
- inference uses the saved model on new data

This is a core production ML concept.

### 2. Model artifact management

The trained model is saved and reused later.

This simulates how real ML systems store models for deployment.

### 3. Batch processing

The system works on files in chunks or batches, not one request at a time.

This is common in enterprise ML.

### 4. Reproducibility

Docker makes the pipeline consistent across machines and CI/CD environments.

### 5. Automation

GitHub Actions simulates a real CI/CD pipeline for ML workflows.

---

## Limitations of current version

This is a learning project, so it is intentionally simple.

### Current limitations

- no model registry
- no feature store
- no drift monitoring
- no experiment tracking
- no scheduler built into the project
- no cloud storage integration yet
- no alerting

---

## Future improvements

Good next upgrades:

- add logging
- add input validation
- add model evaluation report
- add unit tests
- add Makefile
- add scheduler
- replace local folders with S3
- trigger prediction automatically when a new file arrives
- store outputs in SQLite or Postgres
- add MLflow for experiment tracking
- add monitoring for failed batch jobs