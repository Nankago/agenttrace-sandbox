from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from agenttrace_sandbox.config import AgentConfig


class ChatModel(Protocol):
    def complete(self, system: str, user: str) -> str:
        ...


@dataclass
class OpenAICompatibleChat:
    config: AgentConfig

    def complete(self, system: str, user: str) -> str:
        if not self.config.api_key:
            raise RuntimeError("OPENAI_API_KEY is required unless --mock is used.")
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        request = urllib.request.Request(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed: {exc.code} {detail}") from exc
        return data["choices"][0]["message"].get("content", "")


class MockCodingModel:
    """Deterministic model for local demos and tests.

    It intentionally mimics JSON tool-calling behavior without using any remote API.
    """

    def complete(self, system: str, user: str) -> str:
        if "Create a concise plan" in user:
            return "Inspect the relevant Python file, make the smallest fix, run tests, then summarize the diff."
        history_tools = re.findall(r'"tool":\s*"([^"]+)"', user)
        task = user.lower()
        if "subtract" in task:
            if "list_files" not in history_tools:
                return _json("list_files", {"path": "."}, "Find project files.")
            if "read_file" not in history_tools:
                return _json("read_file", {"path": "calculator.py"}, "Inspect the calculator implementation.")
            if "replace_in_file" not in history_tools:
                return _json(
                    "replace_in_file",
                    {"path": "calculator.py", "old": "return a + b  # BUG", "new": "return a - b"},
                    "Fix subtract to subtract the second operand.",
                )
            if "run_tests" not in history_tools:
                return _json("run_tests", {}, "Validate the fix.")
            if "git_diff" not in history_tools:
                return _json("git_diff", {}, "Capture final diff.")
            return _json("finish", {"summary": "Fixed subtract and validated with tests."}, "Task is complete.")
        if "list_files" not in history_tools:
            return _json("list_files", {"path": "."}, "Start by inspecting repository files.")
        return _json("finish", {"summary": "Mock model has no task-specific strategy."}, "Stop safely.")


def _json(tool: str, arguments: dict, reason: str) -> str:
    return json.dumps({"tool": tool, "arguments": arguments, "reason": reason})


def extract_json_object(text: str) -> dict:
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("no JSON object found")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("JSON value is not an object")
    return value
