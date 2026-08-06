"""Reduce a run directory to the latency breakdown."""

import json
from pathlib import Path

import numpy as np


def bootCi(x: list[float], iters: int = 2000, seed: int = 1) -> tuple[float, float, float]:
    """Median with bootstrap 95 percent CI."""
    arr = np.asarray(x, float)
    if len(arr) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(arr, (iters, len(arr)), replace=True), axis=1)
    return (float(np.median(arr)), float(np.percentile(meds, 2.5)),
            float(np.percentile(meds, 97.5)))


def analyzeRun(runDir: str | Path) -> dict:
    """Summarize one run directory."""
    runDir = Path(runDir)
    recs = [json.loads(l) for l in
            (runDir / "requests.jsonl").read_text().splitlines() if l.strip()]
    itl = [v for r in recs for v in r["itl_ms"]]
    ttft = [r["ttft_ms"] for r in recs]
    e2e = [r["e2e_ms"] for r in recs]
    n = recs[0].get("completion_tokens") or recs[0]["chunks"]

    ttftM, ttftLo, ttftHi = bootCi(ttft)
    tpotM, tpotLo, tpotHi = bootCi(itl)
    e2eM = float(np.median(e2e))
    summary = {
        "run": runDir.name, "n_requests": len(recs), "completion_tokens": n,
        "ttft_ms_p50": round(ttftM, 1), "ttft_ci": [round(ttftLo, 1), round(ttftHi, 1)],
        "ttft_ms_p95": round(float(np.percentile(ttft, 95)), 1),
        "tpot_ms_p50": round(tpotM, 3), "tpot_ci": [round(tpotLo, 3), round(tpotHi, 3)],
        "decode_tok_s": round(1000.0 / tpotM, 1) if tpotM > 0 else None,
        "e2e_ms_p50": round(e2eM, 1),
        "decomposition_ms": round(ttftM + (n - 1) * tpotM, 1),
        "decomposition_gap_pct": round(100 * (ttftM + (n - 1) * tpotM - e2eM) / e2eM, 2),
        "chunks_per_tokens": round(recs[0]["chunks"] / n, 3) if n else None,
    }
    (runDir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def printSummary(s: dict) -> None:
    print(f"\n===== {s['run']} =====")
    print(f"requests            : {s['n_requests']}")
    print(f"completion tokens   : {s['completion_tokens']}")
    print(f"chunks per token    : {s['chunks_per_tokens']}")
    print(f"TTFT p50            : {s['ttft_ms_p50']} ms  CI {s['ttft_ci']}")
    print(f"TTFT p95            : {s['ttft_ms_p95']} ms")
    print(f"TPOT p50            : {s['tpot_ms_p50']} ms  CI {s['tpot_ci']}")
    print(f"decode speed        : {s['decode_tok_s']} tok/s")
    print(f"E2E p50             : {s['e2e_ms_p50']} ms")
    print(f"TTFT + (n-1) x TPOT : {s['decomposition_ms']} ms "
          f"(gap {s['decomposition_gap_pct']} %)")
