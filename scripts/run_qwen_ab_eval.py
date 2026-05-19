from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_repair_texts(path: Path) -> list[str]:
    return [str(row.get("text", "")).strip() for row in read_jsonl(path) if str(row.get("text", "")).strip()]


def repair_val_split(texts: list[str], seed: int, val_ratio: float) -> list[str]:
    shuffled = list(texts)
    random.Random(seed).shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_ratio))
    return shuffled[:val_size]


def load_model(model_path: str, adapter_path: str | None, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.to(device)
    model.eval()
    return model, tokenizer


def repair_loss(model, tokenizer, texts: list[str], device: torch.device, max_length: int) -> float:
    losses = []
    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(
                text + tokenizer.eos_token,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
                return_tensors="pt",
            )
            if encoded["input_ids"].shape[1] < 32:
                continue
            batch = {key: value.to(device) for key, value in encoded.items()}
            batch["labels"] = batch["input_ids"].clone()
            loss = model(**batch).loss
            losses.append(float(loss.detach().cpu()))
    return sum(losses) / len(losses) if losses else math.nan


def generate_completion(model, tokenizer, prompt: str, device: torch.device, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded[len(prompt) :] if decoded.startswith(prompt) else decoded


def extract_code(text: str) -> str:
    if "```" in text:
        parts = text.split("```")
        for part in parts[1:]:
            if part.lstrip().startswith("python"):
                return textwrap.dedent(part.lstrip()[len("python") :]).strip()
        if len(parts) > 1:
            return textwrap.dedent(parts[1]).strip()

    code = text.replace("\r\n", "\n")
    definition = re.search(r"(?m)^\s*(def|class)\s+\w+\s*[\(:]", code)
    if definition:
        code = code[definition.start() :]
    else:
        code = "\n".join(line for line in code.splitlines() if not line.lstrip().startswith("#"))

    stop_markers = ["\n# Task:", "\n# Tests:", "\nif __name__"]
    for marker in stop_markers:
        index = code.find(marker)
        if index > 0:
            code = code[:index]
    return textwrap.dedent(code).strip()


def run_python_check(code: str, timeout: int, tmp_dir: Path, name: str) -> tuple[bool, str]:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"{safe_name(name)}.py"
    path.write_text(code, encoding="utf-8")
    try:
        completed = subprocess.run(
            ["python3", str(path)],
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        output = (completed.stdout + "\n" + completed.stderr).strip()
        return completed.returncode == 0, output[-2000:]
    except subprocess.TimeoutExpired:
        return False, "timeout"


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)[:120] or "task"


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for row in read_jsonl(path):
        ids.add(str(row.get("task_id", "")))
    return ids


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def eval_humaneval(model, tokenizer, rows: list[dict[str, Any]], device: torch.device, output: Path, tmp_dir: Path, limit: int | None, max_new_tokens: int, timeout: int) -> dict[str, Any]:
    done = completed_ids(output)
    selected = rows[:limit] if limit else rows
    for index, row in enumerate(selected, 1):
        task_id = str(row["task_id"])
        if task_id in done:
            continue
        prompt = str(row["prompt"])
        completion = generate_completion(model, tokenizer, prompt, device, max_new_tokens)
        program = prompt + completion + "\n" + str(row["test"]) + f"\ncheck({row['entry_point']})\n"
        ok, error = run_python_check(program, timeout, tmp_dir, f"humaneval_{task_id}")
        append_jsonl(output, {"task_id": task_id, "ok": ok, "error": error, "completion": completion[:4000]})
        print({"bench": "humaneval", "index": index, "task_id": task_id, "ok": ok}, flush=True)
    records = read_jsonl(output)
    total = len([row for row in records if row.get("task_id")])
    passed = sum(1 for row in records if row.get("ok"))
    return {"total": total, "passed": passed, "pass_at_1": passed / total if total else 0.0}


def mbpp_prompt(row: dict[str, Any]) -> str:
    tests = "\n".join(str(item) for item in row.get("test_list", [])[:3])
    return (
        "Write a complete Python solution for this task. Return only executable Python code, no explanation.\n\n"
        f"Task:\n{row.get('text', '')}\n\n"
        f"Tests:\n{tests}\n"
    )


def mbpp_chat_prompt(tokenizer, row: dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": "You are an expert Python programmer. Output code only."},
        {"role": "user", "content": mbpp_prompt(row)},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def eval_mbpp(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    device: torch.device,
    output: Path,
    tmp_dir: Path,
    limit: int | None,
    max_new_tokens: int,
    timeout: int,
    prompt_style: str,
) -> dict[str, Any]:
    done = completed_ids(output)
    selected = rows[:limit] if limit else rows
    for index, row in enumerate(selected, 1):
        task_id = str(row["task_id"])
        if task_id in done:
            continue
        prompt = mbpp_chat_prompt(tokenizer, row) if prompt_style == "chat" else mbpp_prompt(row)
        completion = generate_completion(model, tokenizer, prompt, device, max_new_tokens)
        code = extract_code(completion)
        tests = "\n".join(str(item) for item in row.get("test_list", []))
        setup = str(row.get("test_setup_code", ""))
        program = f"{setup}\n{code}\n\n{tests}\n"
        ok, error = run_python_check(program, timeout, tmp_dir, f"mbpp_{task_id}")
        append_jsonl(output, {"task_id": task_id, "ok": ok, "error": error, "completion": completion[:4000], "code": code[:4000]})
        print({"bench": "mbpp", "index": index, "task_id": task_id, "ok": ok}, flush=True)
    records = read_jsonl(output)
    total = len([row for row in records if row.get("task_id")])
    passed = sum(1 for row in records if row.get("ok"))
    return {"total": total, "passed": passed, "pass_at_1": passed / total if total else 0.0}


def eval_one_model(name: str, model_path: str, adapter_path: str | None, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    started = time.time()
    model, tokenizer = load_model(model_path, adapter_path, device)
    model_dir = args.output / name
    model_dir.mkdir(parents=True, exist_ok=True)

    repair_texts = repair_val_split(load_repair_texts(args.repair_corpus), args.seed, args.val_ratio)
    repair = {"val_loss": repair_loss(model, tokenizer, repair_texts, device, args.max_length), "examples": len(repair_texts)}
    print({"model": name, "repair": repair}, flush=True)

    humaneval_rows = select_shard(read_jsonl(args.humaneval), args.shard_index, args.num_shards)
    mbpp_rows = select_shard(read_jsonl(args.mbpp), args.shard_index, args.num_shards)
    suffix = f"_shard_{args.shard_index}_of_{args.num_shards}" if args.num_shards > 1 else ""

    if args.skip_humaneval:
        humaneval = {"skipped": True}
    else:
        humaneval = eval_humaneval(
            model,
            tokenizer,
            humaneval_rows,
            device,
            model_dir / f"humaneval_results{suffix}.jsonl",
            model_dir / f"tmp{suffix}",
            args.humaneval_limit,
            args.max_new_tokens,
            args.timeout,
        )
    if args.skip_mbpp:
        mbpp = {"skipped": True}
    else:
        mbpp = eval_mbpp(
            model,
            tokenizer,
            mbpp_rows,
            device,
            model_dir / f"mbpp_results{suffix}.jsonl",
            model_dir / f"tmp{suffix}",
            args.mbpp_limit,
            args.max_new_tokens,
            args.timeout,
            args.mbpp_prompt_style,
        )

    summary = {
        "model": name,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "repair": repair,
        "humaneval": humaneval,
        "mbpp": mbpp,
        "elapsed_sec": round(time.time() - started, 2),
    }
    (model_dir / f"summary{suffix}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    del model
    torch.cuda.empty_cache()
    return summary


def select_shard(rows: list[dict[str, Any]], shard_index: int, num_shards: int) -> list[dict[str, Any]]:
    if num_shards <= 1:
        return rows
    return [row for index, row in enumerate(rows) if index % num_shards == shard_index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--repair-corpus", required=True, type=Path)
    parser.add_argument("--humaneval", required=True, type=Path)
    parser.add_argument("--mbpp", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--humaneval-limit", type=int)
    parser.add_argument("--mbpp-limit", type=int)
    parser.add_argument("--only-model", choices=["base", "repair_lora"])
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--mbpp-prompt-style", choices=["raw", "chat"], default="raw")
    parser.add_argument("--skip-humaneval", action="store_true")
    parser.add_argument("--skip-mbpp", action="store_true")
    args = parser.parse_args()
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.only_model == "base":
        summaries = [eval_one_model("base", args.model, None, args, device)]
    elif args.only_model == "repair_lora":
        summaries = [eval_one_model("repair_lora", args.model, args.adapter, args, device)]
    else:
        summaries = [
            eval_one_model("base", args.model, None, args, device),
            eval_one_model("repair_lora", args.model, args.adapter, args, device),
        ]
    summary_name = "ab_summary.json" if args.num_shards == 1 and not args.only_model else f"ab_summary_{args.only_model or 'both'}_shard_{args.shard_index}_of_{args.num_shards}.json"
    (args.output / summary_name).write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
