from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_results(paths: list[Path]) -> dict:
    rows = []
    seen = set()
    for path in paths:
        for row in read_jsonl(path):
            task_id = str(row.get("task_id", ""))
            if task_id and task_id not in seen:
                seen.add(task_id)
                rows.append(row)
    passed = sum(1 for row in rows if row.get("ok"))
    total = len(rows)
    return {"total": total, "passed": passed, "pass_at_1": passed / total if total else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/qwen3_8b_repair_ab_eval_parallel", type=Path)
    args = parser.parse_args()

    summaries = {}
    for model in ["base", "repair_lora"]:
        model_dir = args.root / model
        repair_losses = []
        for path in sorted(model_dir.glob("summary_shard_*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            repair_losses.append(float(data.get("repair", {}).get("val_loss", 0)))
        summaries[model] = {
            "repair": {
                "val_loss_avg": sum(repair_losses) / len(repair_losses) if repair_losses else None,
                "shards": len(repair_losses),
            },
            "humaneval": summarize_results(sorted(model_dir.glob("humaneval_results_shard_*.jsonl"))),
            "mbpp": summarize_results(sorted(model_dir.glob("mbpp_results_shard_*.jsonl"))),
        }

    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "ab_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
