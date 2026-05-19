from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


class TextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int) -> None:
        self.examples = []
        for text in texts:
            encoded = tokenizer(
                text + tokenizer.eos_token,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )
            if len(encoded["input_ids"]) >= 32:
                self.examples.append(encoded)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


def collate_batch(batch: list[dict[str, list[int]]], tokenizer) -> dict[str, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    attention_mask = []
    for item in batch:
        ids = item["input_ids"]
        pad_len = max_len - len(ids)
        input_ids.append(ids + [tokenizer.pad_token_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
    input_tensor = torch.tensor(input_ids, dtype=torch.long)
    mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
    labels = input_tensor.clone()
    labels[mask_tensor == 0] = -100
    return {"input_ids": input_tensor, "attention_mask": mask_tensor, "labels": labels}


def load_texts(path: Path) -> list[str]:
    texts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text = str(row.get("text", "")).strip()
        if text:
            texts.append(text)
    return texts


def evaluate(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / len(losses) if losses else math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    texts = load_texts(args.data)
    random.shuffle(texts)
    val_size = max(1, int(len(texts) * args.val_ratio))
    val_texts = texts[:val_size]
    train_texts = texts[val_size:]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = TextDataset(train_texts, tokenizer, args.max_length)
    val_ds = TextDataset(val_texts, tokenizer, args.max_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, tokenizer),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_update_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    warmup_steps = max(1, int(total_update_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_update_steps)

    log_path = args.output / "train_log.jsonl"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(
            json.dumps(
                {
                    "event": "started",
                    "train_examples": len(train_ds),
                    "val_examples": len(val_ds),
                    "total_update_steps": total_update_steps,
                    "warmup_steps": warmup_steps,
                    "args": vars(args) | {"data": str(args.data), "output": str(args.output)},
                },
                ensure_ascii=False,
            )
            + "\n"
        )

        global_step = 0
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(1, args.epochs + 1):
            running = 0.0
            for step, batch in enumerate(train_loader, 1):
                batch = {key: value.to(device) for key, value in batch.items()}
                loss = model(**batch).loss / args.grad_accum
                loss.backward()
                running += float(loss.detach().cpu()) * args.grad_accum
                if step % args.grad_accum == 0 or step == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if global_step % 5 == 0:
                        avg_loss = running / step
                        record = {"event": "train", "epoch": epoch, "global_step": global_step, "loss": avg_loss}
                        print(record, flush=True)
                        log.write(json.dumps(record, ensure_ascii=False) + "\n")
                        log.flush()
            val_loss = evaluate(model, val_loader, device)
            record = {"event": "eval", "epoch": epoch, "global_step": global_step, "val_loss": val_loss}
            print(record, flush=True)
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()

    model.save_pretrained(args.output / "adapter")
    tokenizer.save_pretrained(args.output / "adapter")
    (args.output / "done.json").write_text(
        json.dumps({"status": "done", "global_step": global_step}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
