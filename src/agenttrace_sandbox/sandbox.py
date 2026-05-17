from __future__ import annotations

import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path


IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules", ".venv", "venv"}
PROTECTED_NAMES = {".env", ".env.local", "id_rsa", "id_ed25519"}
DANGEROUS_PARTS = ("rm -rf", "sudo", "curl ", "wget ", "| sh", "| bash", "mkfs", "dd ", "chmod -R 777")
ALLOWED_COMMAND_PREFIXES = (
    "pytest",
    "python -m pytest",
    "python3 -m pytest",
    "python -m unittest",
    "python3 -m unittest",
    "npm test",
    "npm run test",
    "pnpm test",
    "yarn test",
)


@dataclass(frozen=True)
class Sandbox:
    run_id: str
    source_repo: Path
    workspace: Path
    trace_path: Path

    @classmethod
    def create(cls, source_repo: Path, runs_dir: Path) -> "Sandbox":
        run_id = uuid.uuid4().hex[:12]
        root = runs_dir / run_id
        workspace = root / "workspace"
        trace_path = root / "trace.jsonl"
        copy_repo(source_repo.resolve(), workspace)
        return cls(run_id=run_id, source_repo=source_repo.resolve(), workspace=workspace.resolve(), trace_path=trace_path.resolve())

    def resolve(self, candidate: str | Path) -> Path:
        path = (self.workspace / candidate).resolve() if not Path(candidate).is_absolute() else Path(candidate).resolve()
        if path != self.workspace and self.workspace not in path.parents:
            raise ValueError(f"path escapes sandbox workspace: {candidate}")
        return path

    def validate_write(self, candidate: str | Path) -> Path:
        path = self.resolve(candidate)
        if path.name in PROTECTED_NAMES:
            raise ValueError(f"refusing to edit protected file: {path.name}")
        if any(part in IGNORED_DIRS for part in path.parts):
            raise ValueError(f"refusing to edit generated/internal path: {candidate}")
        return path

    def validate_command(self, command: str) -> None:
        normalized = " ".join(command.split()) if command.strip() else "pytest -q"
        lowered = normalized.lower()
        if any(part in lowered for part in DANGEROUS_PARTS):
            raise ValueError(f"dangerous command blocked: {command}")
        if not normalized.startswith(ALLOWED_COMMAND_PREFIXES):
            raise ValueError(f"command is not allowlisted: {command}")

    def run_command(self, command: str, timeout: int) -> tuple[bool, str]:
        self.validate_command(command)
        completed = subprocess.run(
            command.split(),
            cwd=self.workspace,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        output = (completed.stdout + "\n" + completed.stderr).strip()
        return completed.returncode == 0, output or f"command exited with {completed.returncode}"


def copy_repo(source: Path, destination: Path) -> None:
    if not source.exists() or not source.is_dir():
        raise ValueError(f"source repo does not exist: {source}")
    if destination.exists():
        shutil.rmtree(destination)

    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {name for name in names if name in IGNORED_DIRS}

    shutil.copytree(source, destination, ignore=ignore)
