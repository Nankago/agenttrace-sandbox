from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agenttrace_sandbox.sandbox import IGNORED_DIRS, Sandbox


TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".toml", ".yaml", ".yml", ".js", ".ts", ".tsx", ".jsx", ".html", ".css"}


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: str
    blocked: bool = False
    error_type: str = ""


class ToolRegistry:
    def __init__(self, sandbox: Sandbox, timeout: int):
        self.sandbox = sandbox
        self.timeout = timeout

    def descriptions(self) -> str:
        return """Available tools:
- list_files {"path": "."}
- read_file {"path": "relative/path.py"}
- grep {"pattern": "regex", "path": "."}
- replace_in_file {"path": "relative/path.py", "old": "exact text", "new": "replacement"}
- write_file {"path": "relative/path.py", "content": "full file content"}
- run_tests {"command": "pytest -q"}
- git_diff {}
- finish {"summary": "what changed and validation result"}"""

    def run(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        try:
            if name == "list_files":
                return self._list_files(arguments)
            if name == "read_file":
                return self._read_file(arguments)
            if name == "grep":
                return self._grep(arguments)
            if name == "replace_in_file":
                return self._replace_in_file(arguments)
            if name == "write_file":
                return self._write_file(arguments)
            if name == "run_tests":
                return self._run_tests(arguments)
            if name == "git_diff":
                return self._git_diff()
            if name == "finish":
                return ToolResult(True, str(arguments.get("summary", "finished")))
            return ToolResult(False, f"unknown tool: {name}", error_type="unknown_tool")
        except ValueError as exc:
            return ToolResult(False, str(exc), blocked=True, error_type="policy_block")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(False, f"{type(exc).__name__}: {exc}", error_type="tool_error")

    def _list_files(self, arguments: dict[str, Any]) -> ToolResult:
        base = self.sandbox.resolve(arguments.get("path", "."))
        files: list[str] = []
        targets = sorted(base.rglob("*")) if base.is_dir() else [base]
        for path in targets:
            if any(part in IGNORED_DIRS for part in path.parts):
                continue
            if path.is_file():
                files.append(path.relative_to(self.sandbox.workspace).as_posix())
            if len(files) >= 150:
                files.append("... truncated")
                break
        return ToolResult(True, "\n".join(files) or "no files")

    def _read_file(self, arguments: dict[str, Any]) -> ToolResult:
        path = self.sandbox.resolve(arguments["path"])
        if not path.exists() or not path.is_file():
            return ToolResult(False, f"file not found: {arguments['path']}", error_type="missing_file")
        text = path.read_text(encoding="utf-8", errors="replace")
        limit = int(arguments.get("limit", 12000))
        return ToolResult(True, text[:limit] + ("\n... truncated" if len(text) > limit else ""))

    def _grep(self, arguments: dict[str, Any]) -> ToolResult:
        pattern = str(arguments.get("pattern", ""))
        if not pattern:
            return ToolResult(False, "grep requires pattern", error_type="bad_args")
        regex = re.compile(pattern)
        base = self.sandbox.resolve(arguments.get("path", "."))
        targets = sorted(base.rglob("*")) if base.is_dir() else [base]
        matches: list[str] = []
        for path in targets:
            if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for line_no, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    rel = path.relative_to(self.sandbox.workspace).as_posix()
                    matches.append(f"{rel}:{line_no}: {line}")
                    if len(matches) >= 80:
                        return ToolResult(True, "\n".join(matches) + "\n... truncated")
        return ToolResult(True, "\n".join(matches) or "no matches")

    def _replace_in_file(self, arguments: dict[str, Any]) -> ToolResult:
        path = self.sandbox.validate_write(arguments["path"])
        old = arguments.get("old")
        new = arguments.get("new")
        if not isinstance(old, str) or not isinstance(new, str):
            return ToolResult(False, "old and new must be strings", error_type="bad_args")
        before = path.read_text(encoding="utf-8", errors="replace")
        count = before.count(old)
        if count != 1:
            return ToolResult(False, f"old text occurrence count is {count}; expected 1", error_type="edit_miss")
        after = before.replace(old, new, 1)
        path.write_text(after, encoding="utf-8")
        return ToolResult(True, unified_diff(path, before, after, self.sandbox.workspace))

    def _write_file(self, arguments: dict[str, Any]) -> ToolResult:
        path = self.sandbox.validate_write(arguments["path"])
        content = arguments.get("content")
        if not isinstance(content, str):
            return ToolResult(False, "content must be a string", error_type="bad_args")
        before = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ToolResult(True, unified_diff(path, before, content, self.sandbox.workspace))

    def _run_tests(self, arguments: dict[str, Any]) -> ToolResult:
        command = str(arguments.get("command") or "pytest -q")
        ok, output = self.sandbox.run_command(command, timeout=self.timeout)
        return ToolResult(ok, output, error_type="" if ok else "test_failed")

    def _git_diff(self) -> ToolResult:
        chunks: list[str] = []
        for path in sorted(self.sandbox.workspace.rglob("*")):
            if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
                continue
            rel = path.relative_to(self.sandbox.workspace)
            source = self.sandbox.source_repo / rel
            if not source.exists():
                before = ""
            else:
                before = source.read_text(encoding="utf-8", errors="replace")
            after = path.read_text(encoding="utf-8", errors="replace")
            if before != after:
                chunks.append(unified_diff(path, before, after, self.sandbox.workspace))
        return ToolResult(True, "\n".join(chunks) or "no diff")


def unified_diff(path: Path, before: str, after: str, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    return "\n".join(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
            lineterm="",
        )
    )
