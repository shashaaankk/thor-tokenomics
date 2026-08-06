# thor-llm-bench

Latency measurement client for LLM serving stacks on the Jetson AGX Thor.
Stack-neutral: talks to any OpenAI-compatible server and times the stream
client side. TTFT, per-token intervals, E2E, with the decomposition check
E2E vs TTFT + (n-1) x TPOT.

## Install

```bash
uv venv && source .venv/bin/activate && uv pip install -e .
```

## Use

```bash
thor-bench probe -c configs/edgellm-3b-selfawq.toml
thor-bench run   -c configs/edgellm-3b-selfawq.toml -w prompts/scene4.toml
thor-bench analyze runs/<run-dir>
```

A new stack or model is a new TOML in `configs/`. A new prompt is a new TOML
in `prompts/`. Stack request shaping (greedy, cache off, forced length) is
keyed by the `stack` field and lives in `src/thor_bench/client.py`.

Stack notes:

- edgellm: forced length is process wide, start the server with
  `EDGELLM_IGNORE_EOS=1`. `usage.prompt_tokens` is always 0.
- llamacpp: `cache_prompt false` is sent per request; the server should
  still run with `--no-cache-prompt --cache-ram 0 --parallel 1`.
- vllm / sglang: prefix caching must be disabled server side; the probe's
  cache_cold check verifies it.

Each run archives `requests.jsonl`, `meta.json` (power mode, L4T release)
and `summary.json` under `runs/`.
