from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAINING_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "training_data.csv"
PREDICT_INPUT_PATH = PROJECT_ROOT / "input" / "input_data.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Adjust these to match your real schema if needed.
REQUIRED_FEATURE_COLUMNS = [
    "customer_id",
    "age",
    "salary",
    "tenure",
]

TARGET_COLUMN = "target"

# Prediction input should not contain target.
PREDICT_REQUIRED_COLUMNS = REQUIRED_FEATURE_COLUMNS.copy()

# Training input must contain target too.
TRAIN_REQUIRED_COLUMNS = REQUIRED_FEATURE_COLUMNS + [TARGET_COLUMN]

NUMERIC_COLUMNS = [
    "age",
    "salary",
    "tenure",
]

MANDATORY_NON_NULL_COLUMNS = [
    "customer_id",
    "age",
    "salary",
    "tenure",
]

ALLOWED_TARGET_VALUES = {0, 1}

# Thresholds
MAX_NULL_RATIO_WARNING = 0.10
MAX_DUPLICATE_RATIO_WARNING = 0.05
MIN_TRAIN_ROWS = 20
MIN_PREDICT_ROWS = 1
