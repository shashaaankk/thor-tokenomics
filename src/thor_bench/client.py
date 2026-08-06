"""Streaming OpenAI-compatible client with per-token timing.

Stack-specific request shaping lives here and nowhere else:
  edgellm   greedy via top_k=1; forced length is the server-side
            EDGELLM_IGNORE_EOS env var; usage.prompt_tokens is always 0
  llamacpp  cache_prompt=false per request; GGUF metadata can override
            sampling defaults, so all stochastic knobs are pinned
  vllm      ignore_eos and enable_thinking=false per request
  sglang    same fields as vllm
"""

import json
import time
import urllib.request
from dataclasses import dataclass, field

from thor_bench.config import Server_Config
from thor_bench.workloads import Workload


@dataclass
class RequestRecord:
    """Timing and usage for one streamed request."""
    ttftMs: float
    e2eMs: float
    itlMs: list[float]
    chunks: int
    completionTokens: int | None
    promptTokensServer: int | None
    finishReason: str | None
    textHead: str
    tWall: float = field(default_factory=time.time)

    def toDict(self) -> dict:
        return {"ttft_ms": round(self.ttftMs, 3), "e2e_ms": round(self.e2eMs, 3),
                "itl_ms": [round(x, 3) for x in self.itlMs], "chunks": self.chunks,
                "completion_tokens": self.completionTokens,
                "prompt_tokens_server": self.promptTokensServer,
                "finish_reason": self.finishReason,
                "text_head": self.textHead, "t_wall": self.tWall}


def buildBody(server: Server_Config, workload: Workload, maxTokens: int,
              stream: bool = True) -> dict:
    """Greedy request body shaped for the target stack."""
    body: dict = {"model": server.model, "messages": workload.messages(),
                  "max_tokens": maxTokens}
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    if server.stack == "edgellm":
        body["top_k"] = 1
        return body
    body["temperature"] = 0
    body["ignore_eos"] = True
    body["chat_template_kwargs"] = {"enable_thinking": False}
    if server.stack == "llamacpp":
        body.update({"top_k": 1, "cache_prompt": False, "dynatemp_range": 0,
                     "xtc_probability": 0, "mirostat": 0, "repeat_penalty": 1.0})
    return body


def streamRequest(server: Server_Config, workload: Workload, maxTokens: int,
                  timeoutS: float) -> RequestRecord:
    """Send one streaming request and time every chunk."""
    body = buildBody(server, workload, maxTokens)
    req = urllib.request.Request(
        server.url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    tFirst = tPrev = None
    itl: list[float] = []
    usage: dict = {}
    finish: str | None = None
    text: list[str] = []
    with urllib.request.urlopen(req, timeout=timeoutS) as resp:
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
            choices = d.get("choices") or []
            if not choices:
                continue
            finish = choices[0].get("finish_reason") or finish
            delta = choices[0].get("delta", {}).get("content")
            if delta:
                now = time.perf_counter()
                if tFirst is None:
                    tFirst = now
                else:
                    itl.append((now - tPrev) * 1000.0)
                tPrev = now
                text.append(delta)
    tEnd = time.perf_counter()
    if tFirst is None:
        raise RuntimeError("no content chunks streamed")
    return RequestRecord(
        ttftMs=(tFirst - t0) * 1000.0,
        e2eMs=(tEnd - t0) * 1000.0,
        itlMs=itl,
        chunks=len(itl) + 1,
        completionTokens=usage.get("completion_tokens"),
        promptTokensServer=usage.get("prompt_tokens"),
        finishReason=finish,
        textHead="".join(text)[:120])
