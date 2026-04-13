import json
import os
import subprocess
import sys
from pathlib import Path

from ollama import Client


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TEST_RESULTS_PATH = ARTIFACTS_DIR / "test_results.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
PREDICTIONS_SUMMARY_PATH = ARTIFACTS_DIR / "predictions_summary.json"
LLM_SUMMARY_JSON_PATH = ARTIFACTS_DIR / "llm_summary.json"
LLM_SUMMARY_MD_PATH = ARTIFACTS_DIR / "llm_summary.md"

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")


def read_json(path: Path, default):
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return default


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


def extract_test_summary(test_results: dict) -> dict:
    summary = test_results.get("summary", {}) if isinstance(test_results, dict) else {}
    return {
        "passed": int(summary.get("passed", 0) or 0),
        "failed": int(summary.get("failed", 0) or 0),
        "total": int(summary.get("total", 0) or 0),
        "collected": int(summary.get("collected", 0) or 0),
    }


def retrieve_context(query: str, n_results: int = 4) -> list[dict]:
    try:
        from retrieve_context import retrieve_relevant_context
        return retrieve_relevant_context(query=query, n_results=n_results)
    except Exception as e:
        return [{"source": "system", "text": f"Context retrieval failed: {e}"}]


def build_prompt(metrics: dict, predictions_summary: dict, test_summary: dict, contexts: list[dict]) -> str:
    context_text = "\n\n".join(
        [
            f"Source: {item.get('source', 'unknown')}\nContent: {item.get('text', '')}"
            for item in contexts
        ]
    )

    return f"""
You are analyzing an ML batch prediction pipeline run.

Use only the provided facts.
Do not invent values.
Do not skip numeric values.
Do not replace numeric values with vague phrases like "moderate" or "decent" unless you ALSO include the exact number.
Do not claim overfitting or underfitting unless there is explicit train-vs-test evidence.

Your response must be valid markdown and contain these exact sections in this exact order:

## Metric snapshot
## What looks good
## What looks weak
## Confusion matrix interpretation
## Business impact
## Likely causes
## Recommended next actions

Mandatory output rules:
- In "Metric snapshot", print the exact numeric values for:
  - target_column
  - train_rows
  - test_rows
  - accuracy
  - precision
  - recall
  - f1_score
  - confusion_matrix
  - tests passed / total
- Explicitly restate accuracy as both decimal and percentage where possible. Example: 0.66 (66%).
- In "Confusion matrix interpretation", explicitly refer to TN, FP, FN, TP using the actual matrix values.
- Explain what the false positives and false negatives mean in plain language.
- State clearly whether precision being lower than recall suggests too many false positives.
- Mention that accuracy alone is not enough for an imbalanced classification problem if class balance looks uneven.
- If there is no evidence for confidence scores, do not discuss confidence.
- Mention pipeline strengths too, not just model weaknesses.
- Use bullet points under every section except "Metric snapshot", where you may use bullets or short lines.
- Keep the wording concrete and evidence-based.

RUN METRICS:
{json.dumps(metrics, indent=2)}

PREDICTION SUMMARY:
{json.dumps(predictions_summary, indent=2)}

TEST SUMMARY:
{json.dumps(test_summary, indent=2)}

RAG CONTEXT:
{context_text}
""".strip()


def call_llm(prompt: str) -> str:
    client = Client(host=OLLAMA_HOST)
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    return response["message"]["content"]


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    pytest_result = run_pytest()

    if pytest_result["stdout"]:
        print("\nPytest stdout:\n")
        print(pytest_result["stdout"])

    if pytest_result["stderr"]:
        print("\nPytest stderr:\n")
        print(pytest_result["stderr"])

    test_results = read_json(TEST_RESULTS_PATH, {})
    test_summary = extract_test_summary(test_results)
    metrics = read_json(METRICS_PATH, {})
    predictions_summary = read_json(PREDICTIONS_SUMMARY_PATH, {})

    query = "Summarize the ML pipeline quality, metrics, confusion matrix, and next actions using exact numeric values."
    contexts = retrieve_context(query=query, n_results=4)

    prompt = build_prompt(
        metrics=metrics,
        predictions_summary=predictions_summary,
        test_summary=test_summary,
        contexts=contexts,
    )

    llm_text = call_llm(prompt)

    llm_summary = {
        "model": OLLAMA_MODEL,
        "ollama_host": OLLAMA_HOST,
        "prompt": prompt,
        "summary_markdown": llm_text,
        "metrics_snapshot": metrics,
        "predictions_summary_snapshot": predictions_summary,
        "test_summary_snapshot": test_summary,
        "retrieved_context": contexts,
        "pytest": pytest_result,
    }

    with LLM_SUMMARY_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(llm_summary, f, indent=2)

    with LLM_SUMMARY_MD_PATH.open("w", encoding="utf-8") as f:
        f.write(llm_text)

    print(f"Saved LLM summary JSON to {LLM_SUMMARY_JSON_PATH}")
    print(f"Saved LLM summary Markdown to {LLM_SUMMARY_MD_PATH}")

    if pytest_result["returncode"] != 0:
        print("\nPytest reported failures.")
        sys.exit(pytest_result["returncode"])


if __name__ == "__main__":
    main()
