# thor-tokenomics

Token economics measurement for LLM serving on the Jetson AGX Thor: what one
request costs in time, broken into its two phases, measured client side
against any OpenAI-compatible server.

## The model behind the numbers

A request has two phases. Prefill reads the whole prompt at once and ends
when the first token arrives. Decode then produces one token at a time.

- TTFT, time to first token: send to first streamed token. Contains prefill.
- TPOT, time per output token: the steady interval between streamed tokens
  during decode. Measured as the median of all inter-token intervals.
- E2E: send to last token.

The decomposition that ties them together:

    E2E(ISL, OSL) = TTFT(ISL) + (OSL - 1) x TPOT

ISL is prompt tokens, OSL is generated tokens. The tool measures E2E
directly and also computes the right-hand side; `decomposition gap` in the
summary is the difference in percent. A small gap means the three numbers
are consistent and TPOT is trustworthy. Decode speed is `1000 / TPOT_ms`
tokens per second.

Every request is greedy and the output length is forced, so all requests do
identical work and the medians mean something.

## Run

Terminal 1, start a backend (inside tmux):

```bash
./serve.sh edgellm                 # self-quantized Qwen2.5-3B int4_awq engine
./serve.sh vllm Qwen/Qwen2.5-3B-Instruct-AWQ
./serve.sh llamacpp ~/models/Qwen3-8B-Q4_K_M.gguf
```

Terminal 2, measure:

```bash
source .venv/bin/activate
thor-tokenomics run configs/edgellm-3b-selfawq.toml prompts/scene4.toml
thor-tokenomics report runs/<run-dir>
```

## Layout

- `src/thor_tokenomics/__init__.py` : all code, one module
- `configs/*.toml` : one file per measurement cell (server, stack, counts)
- `prompts/*.toml` : one file per workload (system + user text)
- `runs/` : per run, `requests.jsonl`, `summary.json`, `meta.json`

## Validity warnings

Checks run during warmup and print as WARNING lines:

- repeat TTFT collapses on an identical prompt: prefix caching is on and
  TTFT is a cache hit, fix the server flags
- streamed chunks fall below tokens generated: the stream batches tokens
  and TPOT is inflated
- `finish_reason` is not `length`: output length is not forced (for
  Edge-LLM the server must run with `EDGELLM_IGNORE_EOS=1`)

## Stack notes

- edgellm: `usage.prompt_tokens` is always 0; greedy is `top_k=1`
- llamacpp: `cache_prompt=false` sent per request, sampling knobs pinned
  because GGUF metadata can override defaults
- vllm / sglang: prefix caching must be off server side, `serve.sh` does it
