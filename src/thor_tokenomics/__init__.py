"""Token latency measurement against an OpenAI-compatible server.

One measurement cell = one config TOML (server) + one prompt TOML (workload).
Streams every request, times TTFT, inter-token intervals and E2E client side,
archives raw records, prints the breakdown.

  thor-tokenomics run configs/edgellm-3b-selfawq.toml prompts/scene4.toml
  thor-tokenomics report runs/<run-dir>

Validity checks run during warmup and land in the summary as warnings:
repeat TTFT ratio (prefix cache must be off) and chunks per token (the
stream must not batch tokens). Stack-specific request shaping is keyed by
the config's stack field.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
import tomllib
import urllib.request
from pathlib import Path

STACKS = ("vllm", "edgellm", "llamacpp", "sglang")
CACHE_RATIO_MIN = 0.6   # warmup repeat TTFT below this fraction => cache live
CHUNK_RATIO_MIN = 0.9   # chunks below this fraction of tokens => batched stream


def body_for(cfg: dict, prompt: dict, stream: bool = True) -> dict:
    """Greedy request body shaped for the target stack."""
    messages = []
    if prompt.get("system"):
        messages.append({"role": "system", "content": prompt["system"]})
    messages.append({"role": "user", "content": prompt["user"]})
    b = {"model": cfg["model"], "messages": messages, "max_tokens": cfg["max_tokens"]}
    if stream:
        b["stream"] = True
        b["stream_options"] = {"include_usage": True}
    if cfg["stack"] == "edgellm":
        # greedy via top_k; forced length is the server-side EDGELLM_IGNORE_EOS
        b["top_k"] = 1
        return b
    b["temperature"] = 0
    b["ignore_eos"] = True
    b["chat_template_kwargs"] = {"enable_thinking": False}
    if cfg["stack"] == "llamacpp":
        # GGUF metadata can override sampling defaults; pin every stochastic knob
        b.update({"top_k": 1, "cache_prompt": False, "dynatemp_range": 0,
                  "xtc_probability": 0, "mirostat": 0, "repeat_penalty": 1.0})
    return b


def request(cfg: dict, prompt: dict) -> dict:
    """One streamed request. Returns the raw timing record."""
    req = urllib.request.Request(
        cfg["url"].rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body_for(cfg, prompt)).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = t_prev = None
    itl, usage, text = [], {}, []
    finish = None
    with urllib.request.urlopen(req, timeout=cfg["timeout_s"]) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if not ch:
                continue
            finish = ch[0].get("finish_reason") or finish
            delta = ch[0].get("delta", {}).get("content")
            if delta:
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                else:
                    itl.append((now - t_prev) * 1000.0)
                t_prev = now
                text.append(delta)
    t_end = time.perf_counter()
    if t_first is None:
        raise RuntimeError("no content chunks streamed")
    return {"ttft_ms": round((t_first - t0) * 1000.0, 3),
            "e2e_ms": round((t_end - t0) * 1000.0, 3),
            "itl_ms": [round(x, 3) for x in itl],
            "chunks": len(itl) + 1,
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "finish_reason": finish,
            "text_head": "".join(text)[:120],
            "t_wall": time.time()}


def p(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def summarize(recs: list[dict], warnings: list[str]) -> dict:
    itl = [v for r in recs for v in r["itl_ms"]]
    ttft = [r["ttft_ms"] for r in recs]
    e2e = [r["e2e_ms"] for r in recs]
    n = recs[0]["completion_tokens"] or recs[0]["chunks"]
    ttft_med = statistics.median(ttft)
    tpot_med = statistics.median(itl)
    e2e_med = statistics.median(e2e)
    decomp = ttft_med + (n - 1) * tpot_med
    return {"n_requests": len(recs), "completion_tokens": n,
            "ttft_ms_p50": round(ttft_med, 1), "ttft_ms_p95": round(p(ttft, 0.95), 1),
            "tpot_ms_p50": round(tpot_med, 3), "tpot_ms_p99": round(p(itl, 0.99), 3),
            "decode_tok_s": round(1000.0 / tpot_med, 1),
            "e2e_ms_p50": round(e2e_med, 1),
            "decomposition_gap_pct": round(100 * (decomp - e2e_med) / e2e_med, 2),
            "chunks_per_token": round(recs[0]["chunks"] / n, 3),
            "warnings": warnings}


def show(s: dict) -> None:
    print(f"\nrequests            : {s['n_requests']}")
    print(f"completion tokens   : {s['completion_tokens']}")
    print(f"TTFT p50 / p95      : {s['ttft_ms_p50']} / {s['ttft_ms_p95']} ms")
    print(f"TPOT p50 / p99      : {s['tpot_ms_p50']} / {s['tpot_ms_p99']} ms")
    print(f"decode speed        : {s['decode_tok_s']} tok/s")
    print(f"E2E p50             : {s['e2e_ms_p50']} ms")
    print(f"decomposition gap   : {s['decomposition_gap_pct']} %")
    print(f"chunks per token    : {s['chunks_per_token']}")
    for w in s["warnings"]:
        print(f"WARNING: {w}")


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception as e:
        return f"unavailable: {e}"


def run(config_path: str, prompt_path: str) -> int:
    cfg = tomllib.loads(Path(config_path).read_text())
    prompt = tomllib.loads(Path(prompt_path).read_text())
    if cfg["stack"] not in STACKS:
        raise SystemExit(f"stack {cfg['stack']!r} not in {STACKS}")

    run_dir = Path(cfg.get("runs_dir", "runs")) / (
        time.strftime("%Y%m%d-%H%M%S") + f"-{cfg['stack']}-{prompt['name']}")
    run_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    warm = []
    for i in range(cfg["warmup"]):
        warm.append(request(cfg, prompt))
        print(f"warmup {i + 1}/{cfg['warmup']}")
    if len(warm) >= 2:
        ratio = warm[1]["ttft_ms"] / warm[0]["ttft_ms"]
        if ratio < CACHE_RATIO_MIN:
            warnings.append(f"repeat TTFT ratio {ratio:.2f}: prefix cache looks active, TTFT is a cache hit")
    gen = warm[-1]["completion_tokens"] or warm[-1]["chunks"]
    if warm[-1]["chunks"] < CHUNK_RATIO_MIN * gen:
        warnings.append(f"{warm[-1]['chunks']} chunks for {gen} tokens: stream batches tokens, TPOT inflated")
    if warm[-1]["finish_reason"] != "length":
        warnings.append(f"finish_reason {warm[-1]['finish_reason']!r}: output length not forced")

    recs = []
    with (run_dir / "requests.jsonl").open("a") as f:
        for i in range(cfg["requests"]):
            rec = request(cfg, prompt)
            recs.append(rec)
            f.write(json.dumps(rec) + "\n")
            print(f"req {i + 1}/{cfg['requests']}: ttft={rec['ttft_ms']:.1f}ms "
                  f"e2e={rec['e2e_ms']:.1f}ms tokens={rec['completion_tokens']}")

    s = summarize(recs, warnings)
    (run_dir / "summary.json").write_text(json.dumps(s, indent=2))
    (run_dir / "meta.json").write_text(json.dumps(
        {"config": cfg, "prompt_name": prompt["name"], "t": time.time(),
         "nvpmodel": sh("nvpmodel -q"),
         "tegra_release": sh("cat /etc/nv_tegra_release")}, indent=2))
    print(f"\narchived: {run_dir}")
    show(s)
    return 0


def report(run_dir: str) -> int:
    s = json.loads((Path(run_dir) / "summary.json").read_text())
    show(s)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="thor-tokenomics")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="measure one cell")
    r.add_argument("config")
    r.add_argument("prompt")
    p_ = sub.add_parser("report", help="re-print a past run")
    p_.add_argument("run_dir")
    a = ap.parse_args()
    if a.cmd == "run":
        return run(a.config, a.prompt)
    return report(a.run_dir)


if __name__ == "__main__":
    sys.exit(main())
