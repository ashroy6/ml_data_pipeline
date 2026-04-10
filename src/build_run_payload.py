import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOG_FILE = PROJECT_ROOT / "logs" / "predict.log"
TEST_RESULTS_FILE = ARTIFACTS_DIR / "test_results.json"


def read_json(path: Path, default):
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return default


def read_text_tail(path: Path, max_lines: int = 50) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-max_lines:])


def build_missing_test_results() -> dict:
    return {
        "status": "missing",
        "tests_run": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "note": "Real pytest report not found. Run pytest with --json-report to generate artifacts/test_results.json.",
    }


def normalize_test_results(test_results: dict) -> dict:
    if not isinstance(test_results, dict) or not test_results:
        return build_missing_test_results()

    # Handle pytest-json-report structure
    summary = test_results.get("summary", {})
    if isinstance(summary, dict):
        total = int(summary.get("total", 0) or 0)
        passed = int(summary.get("passed", 0) or 0)
        failed = int(summary.get("failed", 0) or 0)
        errors = int(summary.get("error", 0) or 0)
        skipped = int(summary.get("skipped", 0) or 0)

        if total > 0 or passed > 0 or failed > 0 or errors > 0 or skipped > 0:
            status = "passed"
            if failed > 0 or errors > 0:
                status = "failed"

            normalized = {
                "status": status,
                "tests_run": total,
                "tests_passed": passed,
                "tests_failed": failed + errors,
                "tests_skipped": skipped,
                "pytest_summary": summary,
            }

            if "created" in test_results:
                normalized["created"] = test_results["created"]

            if "duration" in test_results:
                normalized["duration"] = test_results["duration"]

            return normalized

    # If already in normalized custom format, keep it
    if {"status", "tests_run", "tests_passed", "tests_failed"}.issubset(test_results.keys()):
        return test_results

    return build_missing_test_results()


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = read_json(ARTIFACTS_DIR / "metrics.json", {})
    predictions_summary = read_json(ARTIFACTS_DIR / "predictions_summary.json", {})

    raw_test_results = read_json(TEST_RESULTS_FILE, {})
    test_results = normalize_test_results(raw_test_results)

    payload = {
        "project": "project1_sample_data",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "test_results": test_results,
        "predictions_summary": predictions_summary,
        "log_tail": read_text_tail(LOG_FILE, max_lines=40),
    }

    out_file = ARTIFACTS_DIR / "run_payload.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved payload to {out_file}")


if __name__ == "__main__":
    main()
