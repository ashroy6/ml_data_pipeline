import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data_contract import (
    ALLOWED_TARGET_VALUES,
    ARTIFACTS_DIR,
    MANDATORY_NON_NULL_COLUMNS,
    MAX_DUPLICATE_RATIO_WARNING,
    MAX_NULL_RATIO_WARNING,
    MIN_PREDICT_ROWS,
    MIN_TRAIN_ROWS,
    NUMERIC_COLUMNS,
    PREDICT_INPUT_PATH,
    PREDICT_REQUIRED_COLUMNS,
    TARGET_COLUMN,
    TRAINING_DATA_PATH,
    TRAIN_REQUIRED_COLUMNS,
)


def add_check(report: dict, level: str, check: str, status: str, details: str) -> None:
    report["checks"].append(
        {
            "level": level,
            "check": check,
            "status": status,
            "details": details,
        }
    )
    if status == "fail":
        report["summary"]["failed_checks"] += 1
    elif status == "warn":
        report["summary"]["warning_checks"] += 1
    else:
        report["summary"]["passed_checks"] += 1


def save_reports(report: dict, mode: str) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = ARTIFACTS_DIR / f"data_validation_{mode}.json"
    md_path = ARTIFACTS_DIR / f"data_validation_{mode}.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    lines = []
    lines.append(f"# Data Validation Report ({mode})")
    lines.append("")
    lines.append(f"- Generated at UTC: {report['generated_at_utc']}")
    lines.append(f"- Mode: {report['mode']}")
    lines.append(f"- Input file: {report['input_file']}")
    lines.append(f"- Rows: {report['row_count']}")
    lines.append(f"- Columns: {report['column_count']}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Passed checks: {report['summary']['passed_checks']}")
    lines.append(f"- Warning checks: {report['summary']['warning_checks']}")
    lines.append(f"- Failed checks: {report['summary']['failed_checks']}")
    lines.append("")
    lines.append("## Checks")

    for item in report["checks"]:
        lines.append(
            f"- [{item['status'].upper()}] {item['check']} ({item['level']}): {item['details']}"
        )

    lines.append("")
    lines.append("## Null counts by column")
    for col, value in report["null_counts"].items():
        lines.append(f"- {col}: {value}")

    lines.append("")
    lines.append("## Dtypes")
    for col, value in report["dtypes"].items():
        lines.append(f"- {col}: {value}")

    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved validation JSON to {json_path}")
    print(f"Saved validation Markdown to {md_path}")


def resolve_input_path(mode: str, input_path: str | None) -> Path:
    if input_path:
        return Path(input_path)

    if mode == "train":
        return TRAINING_DATA_PATH

    return PREDICT_INPUT_PATH


def validate_file_presence(report: dict, path: Path) -> bool:
    if not path.exists():
        add_check(
            report,
            level="critical",
            check="file_exists",
            status="fail",
            details=f"Input file not found: {path}",
        )
        return False

    if path.suffix.lower() != ".csv":
        add_check(
            report,
            level="critical",
            check="csv_extension",
            status="fail",
            details=f"Expected a .csv file, got: {path.name}",
        )
        return False

    if path.stat().st_size == 0:
        add_check(
            report,
            level="critical",
            check="file_not_empty",
            status="fail",
            details=f"CSV file is empty: {path}",
        )
        return False

    add_check(
        report,
        level="critical",
        check="file_presence_basic",
        status="pass",
        details=f"CSV file exists and is non-empty: {path}",
    )
    return True


def load_csv(report: dict, path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
        add_check(
            report,
            level="critical",
            check="csv_parse",
            status="pass",
            details="CSV parsed successfully",
        )
        return df
    except Exception as exc:
        add_check(
            report,
            level="critical",
            check="csv_parse",
            status="fail",
            details=f"Failed to parse CSV: {exc}",
        )
        return None


def validate_schema(report: dict, df: pd.DataFrame, mode: str) -> None:
    required = TRAIN_REQUIRED_COLUMNS if mode == "train" else PREDICT_REQUIRED_COLUMNS
    missing = [c for c in required if c not in df.columns]

    if missing:
        add_check(
            report,
            level="critical",
            check="required_columns",
            status="fail",
            details=f"Missing required columns: {missing}",
        )
    else:
        add_check(
            report,
            level="critical",
            check="required_columns",
            status="pass",
            details=f"All required columns present: {required}",
        )

    unexpected = [c for c in df.columns if c not in required]
    if unexpected:
        add_check(
            report,
            level="warning",
            check="unexpected_columns",
            status="warn",
            details=f"Unexpected extra columns found: {unexpected}",
        )
    else:
        add_check(
            report,
            level="warning",
            check="unexpected_columns",
            status="pass",
            details="No unexpected columns found",
        )


def validate_row_count(report: dict, df: pd.DataFrame, mode: str) -> None:
    min_rows = MIN_TRAIN_ROWS if mode == "train" else MIN_PREDICT_ROWS
    count = len(df)

    if count < min_rows:
        add_check(
            report,
            level="critical",
            check="row_count",
            status="fail",
            details=f"Row count {count} is below minimum required {min_rows}",
        )
    else:
        add_check(
            report,
            level="critical",
            check="row_count",
            status="pass",
            details=f"Row count is {count}",
        )


def validate_nulls(report: dict, df: pd.DataFrame) -> None:
    for col in MANDATORY_NON_NULL_COLUMNS:
        if col not in df.columns:
            continue

        null_count = int(df[col].isna().sum())
        null_ratio = float(df[col].isna().mean())

        if null_count > 0 and col in MANDATORY_NON_NULL_COLUMNS:
            add_check(
                report,
                level="critical",
                check=f"mandatory_non_null::{col}",
                status="fail",
                details=f"Column {col} has {null_count} nulls",
            )
        else:
            add_check(
                report,
                level="critical",
                check=f"mandatory_non_null::{col}",
                status="pass",
                details=f"Column {col} has no nulls",
            )

        if null_ratio > MAX_NULL_RATIO_WARNING:
            add_check(
                report,
                level="warning",
                check=f"null_ratio::{col}",
                status="warn",
                details=f"Column {col} null ratio {null_ratio:.2%} exceeds warning threshold {MAX_NULL_RATIO_WARNING:.2%}",
            )
        else:
            add_check(
                report,
                level="warning",
                check=f"null_ratio::{col}",
                status="pass",
                details=f"Column {col} null ratio {null_ratio:.2%}",
            )


def validate_numeric_columns(report: dict, df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue

        coerced = pd.to_numeric(df[col], errors="coerce")
        bad_count = int(coerced.isna().sum() - df[col].isna().sum())

        if bad_count > 0:
            add_check(
                report,
                level="critical",
                check=f"numeric_type::{col}",
                status="fail",
                details=f"Column {col} has {bad_count} non-numeric values",
            )
        else:
            add_check(
                report,
                level="critical",
                check=f"numeric_type::{col}",
                status="pass",
                details=f"Column {col} is numeric-compatible",
            )

        df[col] = coerced

    return df


def validate_duplicates(report: dict, df: pd.DataFrame) -> None:
    duplicate_count = int(df.duplicated().sum())
    duplicate_ratio = duplicate_count / len(df) if len(df) else 0.0

    if duplicate_ratio > MAX_DUPLICATE_RATIO_WARNING:
        add_check(
            report,
            level="warning",
            check="duplicate_rows",
            status="warn",
            details=f"Duplicate rows: {duplicate_count} ({duplicate_ratio:.2%})",
        )
    else:
        add_check(
            report,
            level="warning",
            check="duplicate_rows",
            status="pass",
            details=f"Duplicate rows: {duplicate_count} ({duplicate_ratio:.2%})",
        )

    if "customer_id" in df.columns:
        dup_keys = int(df["customer_id"].duplicated().sum())
        if dup_keys > 0:
            add_check(
                report,
                level="warning",
                check="duplicate_customer_id",
                status="warn",
                details=f"Duplicate customer_id values: {dup_keys}",
            )
        else:
            add_check(
                report,
                level="warning",
                check="duplicate_customer_id",
                status="pass",
                details="No duplicate customer_id values",
            )


def validate_target(report: dict, df: pd.DataFrame, mode: str) -> None:
    if mode != "train":
        return

    if TARGET_COLUMN not in df.columns:
        add_check(
            report,
            level="critical",
            check="target_exists",
            status="fail",
            details=f"Missing target column: {TARGET_COLUMN}",
        )
        return

    values = set(pd.to_numeric(df[TARGET_COLUMN], errors="coerce").dropna().astype(int).tolist())
    invalid = sorted([v for v in values if v not in ALLOWED_TARGET_VALUES])

    if invalid:
        add_check(
            report,
            level="critical",
            check="target_allowed_values",
            status="fail",
            details=f"Invalid target values found: {invalid}. Allowed: {sorted(ALLOWED_TARGET_VALUES)}",
        )
    else:
        add_check(
            report,
            level="critical",
            check="target_allowed_values",
            status="pass",
            details=f"Target values are within allowed set: {sorted(ALLOWED_TARGET_VALUES)}",
        )

    class_counts = df[TARGET_COLUMN].value_counts(dropna=False).to_dict()
    add_check(
        report,
        level="info",
        check="target_distribution",
        status="pass",
        details=f"Target distribution: {class_counts}",
    )


def build_report(mode: str, input_file: Path) -> dict:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "input_file": str(input_file),
        "row_count": 0,
        "column_count": 0,
        "columns": [],
        "dtypes": {},
        "null_counts": {},
        "checks": [],
        "summary": {
            "passed_checks": 0,
            "warning_checks": 0,
            "failed_checks": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CSV input data before train or predict.")
    parser.add_argument("--mode", choices=["train", "predict"], required=True)
    parser.add_argument("--input-path", default=None)
    args = parser.parse_args()

    input_path = resolve_input_path(args.mode, args.input_path)
    report = build_report(args.mode, input_path)

    if not validate_file_presence(report, input_path):
        save_reports(report, args.mode)
        sys.exit(1)

    df = load_csv(report, input_path)
    if df is None:
        save_reports(report, args.mode)
        sys.exit(1)

    report["row_count"] = int(len(df))
    report["column_count"] = int(len(df.columns))
    report["columns"] = list(df.columns)
    report["dtypes"] = {k: str(v) for k, v in df.dtypes.to_dict().items()}
    report["null_counts"] = {k: int(v) for k, v in df.isna().sum().to_dict().items()}

    validate_schema(report, df, args.mode)
    validate_row_count(report, df, args.mode)
    df = validate_numeric_columns(report, df)
    validate_nulls(report, df)
    validate_duplicates(report, df)
    validate_target(report, df, args.mode)

    save_reports(report, args.mode)

    if report["summary"]["failed_checks"] > 0:
        print("Data validation failed.")
        sys.exit(1)

    print("Data validation passed.")


if __name__ == "__main__":
    main()
