"""Measurement loop: warmup, timed requests, archived JSONL plus metadata."""

import json
import subprocess
import time
from pathlib import Path

from thor_bench.config import Bench_Config
from thor_bench.client import streamRequest
from thor_bench.workloads import Workload


def _sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception as e:
        return f"unavailable: {e}"


def runBench(cfg: Bench_Config, workload: Workload) -> Path:
    """Execute one measurement cell. Returns the run directory."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    runDir = Path(cfg.run.runsDir) / f"{ts}-{cfg.server.stack}-{workload.name}"
    runDir.mkdir(parents=True, exist_ok=True)

    meta = {"server": vars(cfg.server), "run": vars(cfg.run),
            "workload": workload.name, "t_start": time.time(),
            "nvpmodel": _sh("nvpmodel -q"),
            "tegra_release": _sh("cat /etc/nv_tegra_release"),
            "jetson_clocks": _sh("jetson_clocks --show | head -5")}

    for i in range(cfg.run.warmup):
        streamRequest(cfg.server, workload, cfg.run.maxTokens, cfg.run.timeoutS)
        print(f"warmup {i + 1}/{cfg.run.warmup}")

    with (runDir / "requests.jsonl").open("a") as f:
        for i in range(cfg.run.requests):
            rec = streamRequest(cfg.server, workload, cfg.run.maxTokens,
                                cfg.run.timeoutS)
            f.write(json.dumps(rec.toDict()) + "\n")
            print(f"req {i + 1}/{cfg.run.requests}: ttft={rec.ttftMs:.1f}ms "
                  f"e2e={rec.e2eMs:.1f}ms chunks={rec.chunks} "
                  f"tokens={rec.completionTokens}")

    meta["t_end"] = time.time()
    (runDir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"run archived: {runDir}")
    return runDir
