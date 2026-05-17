from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def write_event(trace_path: Path, event: str, payload: dict[str, Any]) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": round(time.time(), 3),
        "event": event,
        "payload": payload,
    }
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue
