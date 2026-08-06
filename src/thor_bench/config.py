"""Configuration dataclasses and TOML loading."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

VALID_STACKS: tuple[str, ...] = ("vllm", "edgellm", "llamacpp", "sglang")


@dataclass
class Server_Config:
    """Serving endpoint under measurement."""
    url: str
    model: str
    stack: str

    def __post_init__(self) -> None:
        if self.stack not in VALID_STACKS:
            raise ValueError(f"stack {self.stack!r} not in {VALID_STACKS}")


@dataclass
class Run_Config:
    """Request loop parameters."""
    maxTokens: int = 128
    warmup: int = 3
    requests: int = 10
    timeoutS: float = 600.0
    runsDir: str = "runs"


@dataclass
class Bench_Config:
    server: Server_Config
    run: Run_Config = field(default_factory=Run_Config)


def loadConfig(path: str | Path) -> Bench_Config:
    """Load a benchmark config from TOML."""
    data: dict = tomllib.loads(Path(path).read_text())
    server = Server_Config(**data["server"])
    run = Run_Config(**data.get("run", {}))
    return Bench_Config(server=server, run=run)
