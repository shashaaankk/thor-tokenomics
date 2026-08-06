#!/usr/bin/env bash
# serve.sh — start the serving backend described by a cell config, foreground.
# Run inside tmux; Ctrl+C stops the server.
#
#   ./serve.sh configs/edgellm-3b-selfawq.toml
#
# Reads stack, artifact and url (for the port) from the TOML. The same file
# drives the measurement client, so one config describes the whole cell.
#
# Machine paths, override via env:
#   THOR_BASE     default ~/thor-llm-throughput-check
#   EDGELLM_DIR   default $THOR_BASE/TensorRT-Edge-LLM
#   EDGELLM_VENV  default $THOR_BASE/.edgellm
#   LLAMA_BIN     default $THOR_BASE/llama.cpp/build/bin/llama-server

set -euo pipefail

CONFIG="${1:?usage: $0 configs/<cell>.toml}"
[ -f "${CONFIG}" ] || { echo "no such config: ${CONFIG}"; exit 1; }

THOR_BASE="${THOR_BASE:-${HOME}/thor-llm-throughput-check}"
EDGELLM_DIR="${EDGELLM_DIR:-${THOR_BASE}/TensorRT-Edge-LLM}"
EDGELLM_VENV="${EDGELLM_VENV:-${THOR_BASE}/.edgellm}"
LLAMA_BIN="${LLAMA_BIN:-${THOR_BASE}/llama.cpp/build/bin/llama-server}"

read -r STACK ARTIFACT PORT <<< "$(python3 - "${CONFIG}" <<'PY'
import sys, tomllib, pathlib, urllib.parse
c = tomllib.loads(pathlib.Path(sys.argv[1]).read_text())
art = str(pathlib.Path(c.get("artifact", "")).expanduser())
port = urllib.parse.urlparse(c["url"]).port or 8080
print(c["stack"], art, port)
PY
)"

echo "config  : ${CONFIG}"
echo "stack   : ${STACK}"
echo "artifact: ${ARTIFACT}"
echo "port    : ${PORT}"

case "${STACK}" in
edgellm)
    [ -f "${ARTIFACT}/llm.engine" ] || { echo "no llm.engine in ${ARTIFACT}"; exit 1; }
    PLUGIN="$(find "${EDGELLM_DIR}/build" -name 'libNvInfer_edgellm_plugin.so' | head -1)"
    echo "forced length: on (EDGELLM_IGNORE_EOS=1)"
    # shellcheck disable=SC1091
    source "${EDGELLM_VENV}/bin/activate"
    cd "${EDGELLM_DIR}"
    EDGELLM_PLUGIN_PATH="${PLUGIN}" EDGELLM_IGNORE_EOS=1 \
    PYTHONPATH="${EDGELLM_DIR}:${EDGELLM_DIR}/build/pybind" \
    python - <<PY
from experimental.server import LLM
LLM(engine_dir="${ARTIFACT}").serve(host="0.0.0.0", port=${PORT})
PY
    ;;
vllm)
    # shellcheck disable=SC1091
    source "${THOR_BASE}/vllm_env.sh"
    sudo sysctl -w vm.drop_caches=3
    vllm serve "${ARTIFACT}" \
        --host 0.0.0.0 --port "${PORT}" \
        --max-model-len 16384 \
        --max-num-seqs 1 \
        --no-enable-prefix-caching \
        --gpu-memory-utilization 0.8 \
        --dtype float16
    ;;
llamacpp)
    exec "${LLAMA_BIN}" \
        --model "${ARTIFACT}" --alias "$(basename "${ARTIFACT}" .gguf)" \
        --host 0.0.0.0 --port "${PORT}" \
        --ctx-size 16384 --n-gpu-layers all \
        --parallel 1 --no-cont-batching \
        --no-cache-prompt --cache-ram 0 \
        --jinja --reasoning-budget 0 --metrics
    ;;
*)
    echo "unknown stack: ${STACK}"
    exit 1
    ;;
esac
