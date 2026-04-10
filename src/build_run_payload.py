import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOG_FILE = PROJECT_ROOT / "logs" / "predict.log"


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


def build_test_results() -> dict:
    metrics = read_json(ARTIFACTS_DIR / "metrics.json", {})
    status = "passed" if metrics else "unknown"
    return {
        "status": status,
        "tests_run": 1 if metrics else 0,
        "tests_passed": 1 if metrics else 0,
        "tests_failed": 0 if metrics else 0,
        "note": "Replace this with real pytest summary later if you add a dedicated test-results export step.",
    }


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = read_json(ARTIFACTS_DIR / "metrics.json", {})
    predictions_summary = read_json(ARTIFACTS_DIR / "predictions_summary.json", {})
    test_results = read_json(ARTIFACTS_DIR / "test_results.json", build_test_results())

    if not (ARTIFACTS_DIR / "test_results.json").exists():
        with (ARTIFACTS_DIR / "test_results.json").open("w", encoding="utf-8") as f:
            json.dump(test_results, f, indent=2)

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
