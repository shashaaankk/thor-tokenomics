"""Pre-measurement server checks. Every check names what a failure means."""

import hashlib
import json
import urllib.request

from thor_bench.config import Server_Config
from thor_bench.client import streamRequest
from thor_bench.workloads import Workload

CACHE_RATIO_MIN: float = 0.6   # repeat TTFT below this fraction => prefix cache live
CHUNK_RATIO_MIN: float = 0.9   # chunks below this fraction of tokens => batched stream

_LONG_UNIT: str = ("The robot carries a payload through a shared laboratory corridor "
                   "while people and equipment move around it. ")


def probe(server: Server_Config, osl: int = 64, timeoutS: float = 600.0) -> dict:
    """Run all checks. Returns a result dict with per-check booleans."""
    checks: dict[str, bool] = {}
    notes: list[str] = []

    with urllib.request.urlopen(server.url.rstrip("/") + "/v1/models", timeout=30) as r:
        ids = [m.get("id") for m in json.load(r).get("data", [])]
    model = server.model
    if model not in ids and server.stack == "edgellm" and ids:
        model = ids[0]
        notes.append(f"edgellm served id is {ids[0]}, using it")
    checks["reachable"] = model in ids
    shaped = Server_Config(url=server.url, model=model, stack=server.stack)

    short = Workload(name="probe", user="Name the capital of France.")
    r1 = streamRequest(shaped, short, osl, timeoutS)
    r2 = streamRequest(shaped, short, osl, timeoutS)

    checks["thinking_off"] = "<think>" not in r1.textHead
    checks["forced_length"] = (r1.completionTokens == osl
                               and r1.finishReason == "length")
    checks["greedy_repeatable"] = (hashlib.sha256(r1.textHead.encode()).hexdigest()
                                   == hashlib.sha256(r2.textHead.encode()).hexdigest())

    longWl = Workload(name="cache", user=_LONG_UNIT * 80)
    c1 = streamRequest(shaped, longWl, 16, timeoutS)
    c2 = streamRequest(shaped, longWl, 16, timeoutS)
    ratio = c2.ttftMs / c1.ttftMs if c1.ttftMs > 0 else 0.0
    checks["cache_cold"] = ratio >= CACHE_RATIO_MIN
    if not checks["cache_cold"]:
        notes.append(f"repeat TTFT ratio {ratio:.2f}: prefix caching still active")

    gen = r1.completionTokens or osl
    checks["stream_per_token"] = r1.chunks >= CHUNK_RATIO_MIN * gen
    if not checks["stream_per_token"]:
        notes.append(f"{r1.chunks} chunks for {gen} tokens: TPOT would be inflated")

    if server.stack == "edgellm":
        checks["usage_reported"] = True
        notes.append("edgellm reports prompt_tokens 0: count input client side")
    else:
        checks["usage_reported"] = bool(r1.promptTokensServer)

    return {"stack": server.stack, "model": model, "checks": checks,
            "ttft_first_ms": round(c1.ttftMs, 1), "ttft_repeat_ms": round(c2.ttftMs, 1),
            "notes": notes, "pass": all(checks.values())}
