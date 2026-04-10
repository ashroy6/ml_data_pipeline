import json
import os
from pathlib import Path

import ollama

from retrieve_context import retrieve_context


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")


def build_query(payload: dict) -> str:
    metrics = payload.get("metrics", {})
    test_results = payload.get("test_results", {})
    predictions_summary = payload.get("predictions_summary", {})

    return (
        "Analyze this ML batch pipeline run. "
        f"Accuracy={metrics.get('accuracy')}, "
        f"Precision={metrics.get('precision')}, "
        f"Recall={metrics.get('recall')}, "
        f"F1={metrics.get('f1_score')}, "
        f"Tests status={test_results.get('status')}, "
        f"Total predictions={predictions_summary.get('total_predictions')}. "
        "Use project notes and troubleshooting guidance if relevant."
    )


def build_prompt(payload: dict, rag_items: list[dict]) -> str:
    context_text = "\n\n".join(
        [
            f"[Source: {item['source']} | Chunk: {item['chunk_index']}]\n{item['content']}"
            for item in rag_items
        ]
    )

    return f"""
You are analyzing a batch ML pipeline run.

Rules:
1. Base your answer on the run payload and retrieved project context.
2. Be specific and practical.
3. Do not invent metrics that are not present.
4. Give:
   - concise summary
   - what looks good
   - what looks weak
   - likely causes
   - 3 practical recommendations
5. Keep the answer readable for a hiring manager and an engineer.

RUN PAYLOAD:
{json.dumps(payload, indent=2)}

RETRIEVED PROJECT CONTEXT:
{context_text if context_text else "No additional context retrieved."}
""".strip()


def main() -> None:
    payload_file = ARTIFACTS_DIR / "run_payload.json"
    if not payload_file.exists():
        raise FileNotFoundError(f"Missing run payload: {payload_file}")

    payload = json.loads(payload_file.read_text(encoding="utf-8"))
    rag_items = retrieve_context(build_query(payload), top_k=4)
    prompt = build_prompt(payload, rag_items)

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    analysis_text = response["message"]["content"].strip()

    md_file = ARTIFACTS_DIR / "llm_summary.md"
    json_file = ARTIFACTS_DIR / "llm_summary.json"

    md_file.write_text(analysis_text + "\n", encoding="utf-8")

    output = {
        "model": OLLAMA_MODEL,
        "analysis": analysis_text,
        "sources_used": rag_items,
    }
    json_file.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Saved {md_file}")
    print(f"Saved {json_file}")


if __name__ == "__main__":
    main()
