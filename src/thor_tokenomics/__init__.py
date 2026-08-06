"""Token latency measurement against an OpenAI-compatible server.

The client does not care what the server is. It asks /v1/models for the
model id, fingerprints the stack from the server's own endpoints, and
measures whatever answers.

  thor-tokenomics run prompts/scene4.toml
  thor-tokenomics run prompts/scene4.toml --url http://127.0.0.1:8080 --max-tokens 128
  thor-tokenomics report runs/<run-dir>

Validity checks run during warmup and land in the summary as warnings:
repeat TTFT ratio (prefix cache must be off), chunks per token (the stream
must not batch tokens), finish_reason (output length must be forced).
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

CACHE_RATIO_MIN = 0.6   # warmup repeat TTFT below this fraction => cache live
CHUNK_RATIO_MIN = 0.9   # chunks below this fraction of tokens => batched stream


def _get(url: str, path: str, timeout: float = 10) -> dict | None:
    try:
        with urllib.request.urlopen(url.rstrip("/") + path, timeout=timeout) as r:
            return json.load(r)
    except Exception:
        return None


def detect(url: str) -> dict:
    """Identify the server: model id from /v1/models, stack from fingerprints."""
    models = _get(url, "/v1/models")
    if not models or not models.get("data"):
        raise SystemExit(f"no model listed at {url}/v1/models, is the server up?")
    entry = models["data"][0]
    model = entry["id"]
    if entry.get("owned_by") == "tensorrt-edgellm":
        stack = "edgellm"
    elif _get(url, "/props") is not None:            # llama.cpp only
        stack = "llamacpp"
    elif _get(url, "/model_info") is not None:       # sglang only
        stack = "sglang"
    else:
        stack = "vllm"                               # vllm-shaped default
    return {"url": url, "model": model, "stack": stack}


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
    server_n = recs[0]["completion_tokens"]
    n = server_n or recs[0]["chunks"]
    ttft_med = statistics.median(ttft)
    tpot_med = statistics.median(itl)
    e2e_med = statistics.median(e2e)
    decomp = ttft_med + (n - 1) * tpot_med
    return {"n_requests": len(recs), "completion_tokens": n,
            "tokens_from": "server" if server_n else "chunk count",
            "ttft_ms_p50": round(ttft_med, 1), "ttft_ms_p95": round(p(ttft, 0.95), 1),
            "tpot_ms_p50": round(tpot_med, 3), "tpot_ms_p99": round(p(itl, 0.99), 3),
            "decode_tok_s": round(1000.0 / tpot_med, 1),
            "e2e_ms_p50": round(e2e_med, 1),
            "throughput_tok_s": round(n / (e2e_med / 1000.0), 1),
            "decomposition_gap_pct": round(100 * (decomp - e2e_med) / e2e_med, 2),
            "chunks_per_token": (round(recs[0]["chunks"] / server_n, 3)
                                 if server_n else None),
            "warnings": warnings}


def show(s: dict) -> None:
    print(f"\nrequests            : {s['n_requests']}")
    print(f"completion tokens   : {s['completion_tokens']} ({s['tokens_from']})")
    print(f"TTFT p50 / p95      : {s['ttft_ms_p50']} / {s['ttft_ms_p95']} ms")
    print(f"TPOT p50 / p99      : {s['tpot_ms_p50']} / {s['tpot_ms_p99']} ms")
    print(f"decode speed        : {s['decode_tok_s']} tok/s   (1000 / TPOT, decode phase only)")
    print(f"throughput          : {s['throughput_tok_s']} tok/s   (tokens / E2E, whole request)")
    print(f"E2E p50             : {s['e2e_ms_p50']} ms")
    print(f"decomposition gap   : {s['decomposition_gap_pct']} %")
    if s["chunks_per_token"] is not None:
        print(f"chunks per token    : {s['chunks_per_token']}")
    for w in s["warnings"]:
        print(f"WARNING: {w}")


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception as e:
        return f"unavailable: {e}"


def run(prompt_path: str, url: str, max_tokens: int, warmup: int,
        requests_n: int, timeout_s: float, runs_dir: str) -> int:
    prompt = tomllib.loads(Path(prompt_path).read_text())
    cfg = detect(url)
    cfg.update({"max_tokens": max_tokens, "warmup": warmup,
                "requests": requests_n, "timeout_s": timeout_s})
    print(f"server: {cfg['stack']} serving {cfg['model']} at {url}")

    run_dir = Path(runs_dir) / (
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
                  f"e2e={rec['e2e_ms']:.1f}ms "
                  f"tokens={rec['completion_tokens'] or rec['chunks']}")

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
    """Recompute the summary from raw records, so old runs gain new fields."""
    d = Path(run_dir)
    recs = [json.loads(l) for l in
            (d / "requests.jsonl").read_text().splitlines() if l.strip()]
    old = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
    s = summarize(recs, old.get("warnings", []))
    (d / "summary.json").write_text(json.dumps(s, indent=2))
    show(s)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="thor-tokenomics")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="measure whatever serves at --url")
    r.add_argument("prompt")
    r.add_argument("--url", default="http://127.0.0.1:8080")
    r.add_argument("--max-tokens", type=int, default=128)
    r.add_argument("--warmup", type=int, default=3)
    r.add_argument("--requests", type=int, default=10)
    r.add_argument("--timeout", type=float, default=600.0)
    r.add_argument("--runs-dir", default="runs")
    p_ = sub.add_parser("report", help="re-print a past run")
    p_.add_argument("run_dir")
    a = ap.parse_args()
    if a.cmd == "run":
        return run(a.prompt, a.url, a.max_tokens, a.warmup,
                   a.requests, a.timeout, a.runs_dir)
    return report(a.run_dir)


if __name__ == "__main__":
    sys.exit(main())
