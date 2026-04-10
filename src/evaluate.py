import json
from pathlib import Path

import pandas as pd
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "training_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
OUTPUT_DIR = PROJECT_ROOT / "output"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

COMMON_TARGET_NAMES = [
    "target",
    "label",
    "y",
    "class",
    "outcome",
    "survived",
    "churn",
    "purchased",
    "default",
]


def infer_target_column(df: pd.DataFrame) -> str:
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in COMMON_TARGET_NAMES:
        if candidate in lower_map:
            return lower_map[candidate]
    return df.columns[-1]


def find_latest_prediction_file() -> Path | None:
    files = sorted(OUTPUT_DIR.glob("predictions_*.csv"))
    return files[-1] if files else None


def summarize_predictions(latest_prediction_file: Path | None) -> dict:
    summary = {
        "latest_prediction_file": str(latest_prediction_file) if latest_prediction_file else None,
        "total_predictions": 0,
        "prediction_column": None,
        "prediction_distribution": {},
        "sample_rows": [],
    }

    if latest_prediction_file is None or not latest_prediction_file.exists():
        return summary

    df = pd.read_csv(latest_prediction_file)
    if df.empty:
        return summary

    prediction_col = None
    for col in ["prediction", "predicted", "predictions", "y_pred"]:
        if col in df.columns:
            prediction_col = col
            break

    if prediction_col is None:
        prediction_col = df.columns[-1]

    distribution = df[prediction_col].astype(str).value_counts(dropna=False).to_dict()

    summary.update(
        {
            "prediction_column": prediction_col,
            "total_predictions": int(len(df)),
            "prediction_distribution": distribution,
            "sample_rows": df.head(5).to_dict(orient="records"),
        }
    )
    return summary


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {TRAINING_DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    df = pd.read_csv(TRAINING_DATA_PATH)
    if df.empty:
        raise ValueError("Training data is empty.")

    target_col = infer_target_column(df)
    drop_columns = ["customer_id"]
    existing_drop_columns = [col for col in drop_columns if col in df.columns]

    X = df.drop(columns=[target_col] + existing_drop_columns)
    y = df[target_col]

    # Must match train.py exactly
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = load(MODEL_PATH)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    prediction_summary = summarize_predictions(find_latest_prediction_file())

    metrics = {
        "target_column": target_col,
        "dropped_columns": existing_drop_columns,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "confusion_matrix": cm.tolist(),
        "class_labels": sorted([str(v) for v in pd.Series(y).dropna().unique().tolist()]),
        "predictions_summary": prediction_summary,
    }

    metrics_path = ARTIFACTS_DIR / "metrics.json"
    predictions_summary_path = ARTIFACTS_DIR / "predictions_summary.json"

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with predictions_summary_path.open("w", encoding="utf-8") as f:
        json.dump(prediction_summary, f, indent=2)

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved prediction summary to {predictions_summary_path}")


if __name__ == "__main__":
    main()
