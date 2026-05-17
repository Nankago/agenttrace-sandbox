from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AgentConfig:
    provider: str = "openai"
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_steps: int = 8
    json_retries: int = 2
    command_timeout: int = 30
    sandbox_backend: str = "local"
    docker_image: str = "python:3.11-slim"
    docker_network: str = "none"
    docker_memory: str = "1g"
    docker_cpus: str = "1"
    runs_dir: Path = Path("runs")

    @classmethod
    def from_env(cls) -> "AgentConfig":
        return cls(
            provider=os.getenv("AGENTTRACE_PROVIDER", "openai"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("AGENTTRACE_TEMPERATURE", "0.1")),
            max_steps=int(os.getenv("AGENTTRACE_MAX_STEPS", "8")),
            json_retries=int(os.getenv("AGENTTRACE_JSON_RETRIES", "2")),
            command_timeout=int(os.getenv("AGENTTRACE_COMMAND_TIMEOUT", "30")),
            sandbox_backend=os.getenv("AGENTTRACE_SANDBOX", "local"),
            docker_image=os.getenv("AGENTTRACE_DOCKER_IMAGE", "python:3.11-slim"),
            docker_network=os.getenv("AGENTTRACE_DOCKER_NETWORK", "none"),
            docker_memory=os.getenv("AGENTTRACE_DOCKER_MEMORY", "1g"),
            docker_cpus=os.getenv("AGENTTRACE_DOCKER_CPUS", "1"),
            runs_dir=Path(os.getenv("AGENTTRACE_RUNS_DIR", "runs")),
        )
