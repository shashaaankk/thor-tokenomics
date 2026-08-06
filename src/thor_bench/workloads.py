"""Workload definitions loaded from prompt TOML files."""

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Workload:
    """One prompt: optional system message plus user message."""
    name: str
    user: str
    system: str | None = None

    def messages(self) -> list[dict]:
        msgs: list[dict] = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        msgs.append({"role": "user", "content": self.user})
        return msgs


def loadWorkload(path: str | Path) -> Workload:
    """Load a workload from TOML with keys name, user, optional system."""
    data: dict = tomllib.loads(Path(path).read_text())
    return Workload(name=data["name"], user=data["user"], system=data.get("system"))
