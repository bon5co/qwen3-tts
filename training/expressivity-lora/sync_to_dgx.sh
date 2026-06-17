#!/usr/bin/env bash
# Sync the NEW CSP-FT + SER-judge scripts to the DGX, into the exact dirs dgx_csp_italian.sh expects.
# Run this from the Mac repo root TONIGHT once the DGX is up. Idempotent (just scp overwrites).
#
#   bash training/expressivity-lora/sync_to_dgx.sh            # uses host alias `dgx`, ~/qwen-ft
#   DGX=myhost ROOT=/data/qwen-ft bash training/expressivity-lora/sync_to_dgx.sh
#
# WHERE each script must land (matches the runner's host-stage vs docker-stage cwd):
#   ~/qwen-ft/                          host-run: the runner, prepare_emozionalmente.py, concat_manifests.py
#   ~/qwen-ft/tests/                    the SER judge (only needed if TRAIN_JUDGE=1)
#   ~/qwen-ft/Qwen3-TTS/finetuning/     docker-run: csp_probe.py, dgx_sft_expr_csp.py, dgx_dataset_expr_lang.py
set -euo pipefail
DGX="${DGX:-dgx}"
ROOT="${ROOT:-qwen-ft}"                 # path RELATIVE to the remote home (scp/ssh resolve it; $HOME isn't expanded by scp)
HERE="$(cd "$(dirname "$0")" && pwd)"   # training/expressivity-lora
REPO="$(cd "$HERE/../.." && pwd)"

echo "[sync] target: $DGX:$ROOT"
ssh "$DGX" "mkdir -p $ROOT/tests $ROOT/Qwen3-TTS/finetuning"

echo "[sync] host-run scripts -> $ROOT/"
scp "$HERE/dgx_csp_italian.sh" "$HERE/prepare_emozionalmente.py" "$HERE/concat_manifests.py" \
    "$HERE/dgx_emovo_prep.py" "$DGX:$ROOT/"

echo "[sync] SER judge -> $ROOT/tests/"
scp "$REPO/tests/train_ser_judge.py" "$REPO/tests/emo_judge.py" "$DGX:$ROOT/tests/"

echo "[sync] docker-run scripts -> $ROOT/Qwen3-TTS/finetuning/"
scp "$HERE/csp_probe.py" "$HERE/dgx_sft_expr_csp.py" "$HERE/dgx_dataset_expr_lang.py" \
    "$HERE/dgx_sft_expr_lang.py" "$HERE/prepare_data.py" "$DGX:$ROOT/Qwen3-TTS/finetuning/"

echo "[sync] DONE. Tonight, on the DGX:  cd ~/qwen-ft && nohup bash dgx_csp_italian.sh >/dev/null 2>&1 &"
echo "        (quality run: MINAGREE=3 ... ;  +SER judge: TRAIN_JUDGE=1 MINAGREE=3 ...)"
