from __future__ import annotations

import shutil
import shlex
import subprocess
import uuid
import os
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
    backend: str = "local"
    docker_image: str = "python:3.11-slim"
    docker_network: str = "none"
    docker_memory: str = "1g"
    docker_cpus: str = "1"

    @classmethod
    def create(
        cls,
        source_repo: Path,
        runs_dir: Path,
        backend: str = "local",
        docker_image: str = "python:3.11-slim",
        docker_network: str = "none",
        docker_memory: str = "1g",
        docker_cpus: str = "1",
    ) -> "Sandbox":
        run_id = uuid.uuid4().hex[:12]
        root = runs_dir / run_id
        workspace = root / "workspace"
        trace_path = root / "trace.jsonl"
        copy_repo(source_repo.resolve(), workspace)
        if backend not in {"local", "docker"}:
            raise ValueError(f"unsupported sandbox backend: {backend}")
        return cls(
            run_id=run_id,
            source_repo=source_repo.resolve(),
            workspace=workspace.resolve(),
            trace_path=trace_path.resolve(),
            backend=backend,
            docker_image=docker_image,
            docker_network=docker_network,
            docker_memory=docker_memory,
            docker_cpus=docker_cpus,
        )

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
        if self.backend == "docker":
            return self.run_docker_command(command, timeout)
        completed = subprocess.run(
            shlex.split(command),
            cwd=self.workspace,
            env=pythonpath_env(self.workspace),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        output = (completed.stdout + "\n" + completed.stderr).strip()
        return completed.returncode == 0, output or f"command exited with {completed.returncode}"

    def run_docker_command(self, command: str, timeout: int) -> tuple[bool, str]:
        docker_cmd = build_docker_command(
            workspace=self.workspace,
            command=command,
            image=self.docker_image,
            network=self.docker_network,
            memory=self.docker_memory,
            cpus=self.docker_cpus,
        )
        try:
            completed = subprocess.run(
                docker_cmd,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError:
            return False, "docker executable not found; install Docker or use --sandbox local"
        output = (completed.stdout + "\n" + completed.stderr).strip()
        return completed.returncode == 0, output or f"docker command exited with {completed.returncode}"


def build_docker_command(workspace: Path, command: str, image: str, network: str, memory: str, cpus: str) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--network",
        network,
        "--memory",
        memory,
        "--cpus",
        cpus,
        "--pids-limit",
        "256",
        "-v",
        f"{workspace}:/workspace",
        "-w",
        "/workspace",
        image,
        "sh",
        "-lc",
        command,
    ]


def copy_repo(source: Path, destination: Path) -> None:
    if not source.exists() or not source.is_dir():
        raise ValueError(f"source repo does not exist: {source}")
    if destination.exists():
        shutil.rmtree(destination)

    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {name for name in names if name in IGNORED_DIRS}

    shutil.copytree(source, destination, ignore=ignore)


def pythonpath_env(workspace: Path) -> dict[str, str]:
    env = os.environ.copy()
    src_path = workspace / "src"
    if src_path.exists():
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(src_path) + ((os.pathsep + existing) if existing else "")
    return env
