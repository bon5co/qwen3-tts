#!/bin/bash
# Container entrypoint: fetch the selected model on first boot, then start the HTTP server.
set -e

MODEL="${MODEL:-small}"
MODELS_ROOT="${MODELS_ROOT:-/models}"
MODEL_DIR="${MODELS_ROOT}/${MODEL}"

if [ ! -f "${MODEL_DIR}/model.safetensors" ]; then
    echo "Model '${MODEL}' not found — downloading to ${MODEL_DIR} (one-time)..."
    bash /app/download_model.sh --model "${MODEL}" --dir "${MODEL_DIR}"
fi

set -- /app/qwen_tts -d "${MODEL_DIR}" --serve "${PORT:-8080}" --workers "${WORKERS:-1}"

if [ "${BATCH_SIZE:-1}" -ge 2 ]; then
    set -- "$@" --batch-size "${BATCH_SIZE}"
fi

case "${QUANT:-int8}" in
    int8) set -- "$@" --int8 ;;
    int4) set -- "$@" --int4 ;;
    bf16|none|"") ;;
    *) echo "Unknown QUANT='${QUANT}' (use int8|int4|bf16)" >&2; exit 1 ;;
esac

echo "Starting: $*"
exec "$@"
