#!/usr/bin/env bash
# serve.sh — start one serving backend on port 8080, foreground.
# Run inside tmux; Ctrl+C stops the server.
#
#   ./serve.sh edgellm [ENGINE_DIR]     # default: self-quantized Qwen2.5-3B int4_awq
#   ./serve.sh vllm MODEL_OR_PATH       # e.g. Qwen/Qwen2.5-3B-Instruct-AWQ
#   ./serve.sh llamacpp GGUF_PATH
#
# Machine paths, override via env:
#   THOR_BASE     default ~/thor-llm-throughput-check
#   EDGELLM_DIR   default $THOR_BASE/TensorRT-Edge-LLM
#   EDGELLM_VENV  default $THOR_BASE/.edgellm
#   LLAMA_BIN     default $THOR_BASE/llama.cpp/build/bin/llama-server
#   PORT          default 8080

set -euo pipefail

THOR_BASE="${THOR_BASE:-${HOME}/thor-llm-throughput-check}"
EDGELLM_DIR="${EDGELLM_DIR:-${THOR_BASE}/TensorRT-Edge-LLM}"
EDGELLM_VENV="${EDGELLM_VENV:-${THOR_BASE}/.edgellm}"
LLAMA_BIN="${LLAMA_BIN:-${THOR_BASE}/llama.cpp/build/bin/llama-server}"
PORT="${PORT:-8080}"

case "${1:-}" in
edgellm)
    ENGINE="${2:-${THOR_BASE}/models/qwen2.5-3b-instruct-selfawq-engine}"
    [ -f "${ENGINE}/llm.engine" ] || { echo "no llm.engine in ${ENGINE}"; exit 1; }
    PLUGIN="$(find "${EDGELLM_DIR}/build" -name 'libNvInfer_edgellm_plugin.so' | head -1)"
    echo "engine : ${ENGINE}"
    echo "port   : ${PORT}   forced length: on (EDGELLM_IGNORE_EOS=1)"
    # shellcheck disable=SC1091
    source "${EDGELLM_VENV}/bin/activate"
    cd "${EDGELLM_DIR}"
    EDGELLM_PLUGIN_PATH="${PLUGIN}" EDGELLM_IGNORE_EOS=1 \
    PYTHONPATH="${EDGELLM_DIR}:${EDGELLM_DIR}/build/pybind" \
    python - <<PY
from experimental.server import LLM
LLM(engine_dir="${ENGINE}").serve(host="0.0.0.0", port=${PORT})
PY
    ;;
vllm)
    MODEL="${2:?usage: ./serve.sh vllm MODEL_OR_PATH}"
    # shellcheck disable=SC1091
    source "${THOR_BASE}/vllm_env.sh"
    sudo sysctl -w vm.drop_caches=3
    vllm serve "${MODEL}" \
        --host 0.0.0.0 --port "${PORT}" \
        --max-model-len 16384 \
        --max-num-seqs 1 \
        --no-enable-prefix-caching \
        --gpu-memory-utilization 0.8 \
        --dtype float16
    ;;
llamacpp)
    GGUF="${2:?usage: ./serve.sh llamacpp GGUF_PATH}"
    exec "${LLAMA_BIN}" \
        --model "${GGUF}" --alias "$(basename "${GGUF}" .gguf)" \
        --host 0.0.0.0 --port "${PORT}" \
        --ctx-size 16384 --n-gpu-layers all \
        --parallel 1 --no-cont-batching \
        --no-cache-prompt --cache-ram 0 \
        --jinja --reasoning-budget 0 --metrics
    ;;
*)
    echo "usage: $0 {edgellm|vllm|llamacpp} [args]"
    exit 1
    ;;
esac
