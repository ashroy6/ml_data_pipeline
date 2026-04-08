# Batch Prediction Pipeline

This project demonstrates a simple production-style ML batch pipeline.

## Pipelines

1. CI
   - Runs smoke tests
   - Builds Docker image

2. Train
   - Trains model from `data/raw/training_data.csv`
   - Saves `models/model.pkl`
   - Uploads model as artifact

3. Predict
   - Loads trained model
   - Reads `input/input_data.csv`
   - Writes prediction output to `output/`
   - Archives processed input to `archive/`
   - Writes logs to `logs/`

## Local Docker commands

### Build
```bash
docker build -t batch-ml-project1 .