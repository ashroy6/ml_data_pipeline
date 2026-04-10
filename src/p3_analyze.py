import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TEST_RESULTS_PATH = ARTIFACTS_DIR / "test_results.json"
RUN_PAYLOAD_PATH = ARTIFACTS_DIR / "run_payload.json"


def run_pytest() -> dict:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "--json-report",
        f"--json-report-file={TEST_RESULTS_PATH}",
    ]

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    return {
        "command": " ".join(cmd),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "report_file": str(TEST_RESULTS_PATH),
    }


def build_run_payload(pytest_result: dict) -> dict:
    payload = {
        "project_root": str(PROJECT_ROOT),
        "artifacts_dir": str(ARTIFACTS_DIR),
        "pytest": pytest_result,
        "files_present": {
            "metrics_json": (ARTIFACTS_DIR / "metrics.json").exists(),
            "predictions_summary_json": (ARTIFACTS_DIR / "predictions_summary.json").exists(),
            "test_results_json": TEST_RESULTS_PATH.exists(),
            "llm_summary_json": (ARTIFACTS_DIR / "llm_summary.json").exists(),
            "llm_summary_md": (ARTIFACTS_DIR / "llm_summary.md").exists(),
        },
    }
    return payload


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    pytest_result = run_pytest()

    payload = build_run_payload(pytest_result)
    with RUN_PAYLOAD_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved run payload to {RUN_PAYLOAD_PATH}")
    print(f"Pytest report path: {TEST_RESULTS_PATH}")

    if pytest_result["stdout"]:
        print("\nPytest stdout:\n")
        print(pytest_result["stdout"])

    if pytest_result["stderr"]:
        print("\nPytest stderr:\n")
        print(pytest_result["stderr"])

    if pytest_result["returncode"] != 0:
        print("\nPytest reported failures.")
        sys.exit(pytest_result["returncode"])


if __name__ == "__main__":
    main()
