from pathlib import Path
from datetime import datetime
import logging
import shutil
import sqlite3

import joblib
import pandas as pd


def get_risk_band(prob: float) -> str:
    if prob >= 0.80:
        return "High"
    elif prob >= 0.60:
        return "Medium"
    else:
        return "Low"


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    input_dir = project_root / "input"
    archive_dir = project_root / "archive"
    failed_dir = project_root / "failed"
    output_dir = project_root / "output"
    logs_dir = project_root / "logs"

    model_file = project_root / "models" / "model.pkl"
    input_file = input_dir / "input_data.csv"
    log_file = logs_dir / "predict.log"
    db_file = project_root / "predictions.db"

    archive_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prediction_timestamp = datetime.now().isoformat(timespec="seconds")
    model_version = "v1"

    output_file = output_dir / f"predictions_{timestamp}.csv"

    logging.info("Prediction job started")

    try:
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        df = pd.read_csv(input_file)

        if df.empty:
            raise ValueError("Input file is empty")

        if "customer_id" not in df.columns:
            raise ValueError("Missing required column: customer_id")

        model = joblib.load(model_file)

        if not hasattr(model, "feature_names_in_"):
            raise ValueError(
                "Loaded model does not expose feature_names_in_. Retrain with current train.py."
            )

        required_feature_columns = list(model.feature_names_in_)

        default_values = {
            "contract_type": "Unknown",
            "region": "Unknown",
            "payment_method": "Unknown",
            "age": 0,
            "monthly_spend": 0.0,
            "tenure_months": 0,
            "support_tickets": 0,
            "last_login_days": 0,
            "num_logins_30d": 0,
            "avg_session_mins": 0.0,
            "avg_session_minutes": 0.0,
            "discount_used": 0,
            "num_products": 0,
        }

        for col in required_feature_columns:
            if col not in df.columns:
                df[col] = default_values.get(col, 0)
                logging.warning(f"Missing column '{col}' added with default value")

        customer_ids = df["customer_id"]
        X = df[required_feature_columns]

        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        risk_bands = [get_risk_band(prob) for prob in probabilities]

        results = pd.DataFrame(
            {
                "customer_id": customer_ids,
                "predicted_churn": predictions,
                "prediction_probability": probabilities.round(4),
                "risk_band": risk_bands,
                "model_version": model_version,
                "prediction_timestamp": prediction_timestamp,
            }
        )

        results.to_csv(output_file, index=False)

        with sqlite3.connect(db_file) as conn:
            results.to_sql("predictions", conn, if_exists="append", index=False)

        archived_input_file = archive_dir / f"input_data_{timestamp}.csv"
        shutil.move(str(input_file), str(archived_input_file))

        logging.info(f"Predictions generated successfully for {len(results)} records")
        logging.info(f"Output saved to: {output_file}")
        logging.info(f"Predictions loaded into database: {db_file}")
        logging.info(f"Input file archived to: {archived_input_file}")

        print(f"Predictions saved to: {output_file}")
        print(f"Predictions loaded into database: {db_file}")
        print(f"Input file archived to: {archived_input_file}")
        print(f"Log saved to: {log_file}")

    except Exception as e:
        logging.exception(f"Prediction job failed: {e}")

        if input_file.exists():
            failed_input_file = failed_dir / f"input_data_{timestamp}.csv"
            shutil.move(str(input_file), str(failed_input_file))
            logging.info(f"Bad input file moved to: {failed_input_file}")
            print(f"Bad input file moved to: {failed_input_file}")

        raise


if __name__ == "__main__":
    main()