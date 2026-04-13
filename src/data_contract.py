from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAINING_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "training_data.csv"
PREDICT_INPUT_PATH = PROJECT_ROOT / "input" / "input_data.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

REQUIRED_FEATURE_COLUMNS = [
    "customer_id",
    "age",
    "monthly_spend",
    "tenure_months",
    "support_tickets",
    "last_login_days",
    "contract_type",
    "payment_method",
    "region",
    "num_products",
    "discount_used",
    "avg_session_minutes",
]

TARGET_COLUMN = "churn"

PREDICT_REQUIRED_COLUMNS = REQUIRED_FEATURE_COLUMNS.copy()
TRAIN_REQUIRED_COLUMNS = REQUIRED_FEATURE_COLUMNS + [TARGET_COLUMN]

NUMERIC_COLUMNS = [
    "age",
    "monthly_spend",
    "tenure_months",
    "support_tickets",
    "last_login_days",
    "num_products",
    "discount_used",
    "avg_session_minutes",
]

MANDATORY_NON_NULL_COLUMNS = [
    "customer_id",
    "age",
    "monthly_spend",
    "tenure_months",
    "support_tickets",
    "last_login_days",
    "contract_type",
    "payment_method",
    "region",
    "num_products",
    "discount_used",
    "avg_session_minutes",
]

ALLOWED_TARGET_VALUES = {0, 1}

MAX_NULL_RATIO_WARNING = 0.10
MAX_DUPLICATE_RATIO_WARNING = 0.05
MIN_TRAIN_ROWS = 2
MIN_PREDICT_ROWS = 1
