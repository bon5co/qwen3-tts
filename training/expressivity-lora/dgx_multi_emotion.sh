#!/usr/bin/env bash
# Phase-2 MANY-SPEAKER emotion fine-tune orchestrator (PLAN Phase 2, docs/emotion-research.md step 1).
#
# Goal: make emotion GENERALIZE to cloned / novel x-vectors by training the dense L16-26 expressivity
# FT on emotion spoken by MANY identities -- EMOVO (6 IT actors) + ESD (10 EN speakers, rich text) +
# CREMA-D (91 EN actors, huge identity diversity) -- instead of EMOVO-only (the voice-specific result).
#
# This runner ONLY ORCHESTRATES. It does NOT modify any original script:
#   - prepare_esd.py / prepare_cremad.py / concat_manifests.py  = NEW dedicated aug steps (this epic)
#   - prepare_data.py (encode) / dgx_sft_expr.py (dense FT) / dgx_emovo_prep.py = ORIGINAL, untouched,
#     driven only via their existing CLI flags (--train_jsonl, --layers, ...).
#
# Idempotent + resumable: each stage is guarded by a <stage>.DONE marker (skip if present) and logs to
# ONE timestamped master log. Re-run safely after a failure; delete a marker to force a stage to re-run.
#
# Usage (on the DGX, from ~/qwen-ft):
#   nohup bash dgx_multi_emotion.sh > /dev/null 2>&1 &   # detaches; watch multi_emotion.log
#   tail -f ~/qwen-ft/multi_emotion.log
set -uo pipefail

ROOT="$HOME/qwen-ft"
cd "$ROOT"
LOG="$ROOT/multi_emotion.log"
FT="$ROOT/Qwen3-TTS/finetuning"
NVCR="nvcr.io/nvidia/pytorch:25.09-py3"          # encode (qwen-tts tokenizer on GPU)
TRAIN_IMG="qwen-ft:latest"                       # dense FT deps
TOK="/root/qwen-ft/models/tokenizer-12hz"
ESD_SPEAKERS="0001-0010"                         # English speakers only (cross-lingual transfer)
OUT_CKPT="$ROOT/out_multi_l1626"

ts()  { date "+%Y-%m-%d %H:%M:%S"; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }
done_marker() { echo "$ROOT/$1.DONE"; }
is_done() { [ -f "$(done_marker "$1")" ]; }
mark()    { touch "$(done_marker "$1")"; say "<<< stage '$1' DONE"; }
stage()   { say ">>> stage '$1' START"; }
# Gate: FAIL LOUD (exit) if an expected output file is missing/empty. Stage commands run docker with
# inner pipes that can mask a crashed python exit code, so we verify the actual artifact, not just $?.
need_file() { [ -s "$2" ] || { say "FAIL stage '$1': missing/empty output $2"; exit 1; }; }

# Mount the host qwen-ft BOTH at /root/qwen-ft (the EMOVO convention, paths baked by the in-docker
# EMOVO prep) AND at its real host path $ROOT (the convention the host-run ESD/CREMA-D preps baked into
# their jsonl `audio` fields) -> every audio path resolves inside the container, no jsonl rewrite needed.
run_nvcr()  { docker run --rm --gpus all --ipc=host -e PYTHONUNBUFFERED=1 \
                -v "$ROOT:/root/qwen-ft" -v "$ROOT:$ROOT" "$NVCR" bash -lc "$1"; }
run_train() { docker run --rm --gpus all --ipc=host -e PYTHONUNBUFFERED=1 \
                -v "$ROOT:/root/qwen-ft" -v "$ROOT:$ROOT" "$TRAIN_IMG" bash -c "$1"; }

say "================ MULTI-SPEAKER EMOTION FT START ================"
say "ESD speakers=$ESD_SPEAKERS  out_ckpt=$OUT_CKPT  log=$LOG"

# ---------------------------------------------------------------- STAGE 1: ESD download + prep (host)
if is_done esd_prep; then say "skip esd_prep (marker present)"; else
  stage esd_prep
  python3 -c 'import pyarrow' 2>/dev/null || pip install --break-system-packages -q pyarrow 2>&1 | tail -1
  if [ ! -f "$ROOT/esd/raw/.unzipped" ]; then
    say "downloading duanyu027/ESD (ESD.zip, ~2.4 GB) ..."
    ZIP=$(python3 -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('duanyu027/ESD','ESD.zip',repo_type='dataset'))")
    say "unzipping $ZIP -> $ROOT/esd/raw ..."
    mkdir -p "$ROOT/esd/raw"
    unzip -q -o "$ZIP" -d "$ROOT/esd/raw" && touch "$ROOT/esd/raw/.unzipped"
  fi
  # locate the dir that contains 0001/0001.txt (zip root varies)
  ESD_ROOT=$(dirname "$(find "$ROOT/esd/raw" -maxdepth 3 -name '0001.txt' | head -1)")
  ESD_ROOT=$(dirname "$ESD_ROOT")
  say "ESD root detected: $ESD_ROOT"
  python3 prepare_esd.py --esd-root "$ESD_ROOT" --out "$ROOT/esd/train_raw.jsonl" \
      --speakers "$ESD_SPEAKERS" 2>&1 | tee -a "$LOG"
  mark esd_prep
fi

# ------------------------------------------------------------ STAGE 2: CREMA-D download + prep (host)
if is_done cremad_prep; then say "skip cremad_prep (marker present)"; else
  stage cremad_prep
  python3 -c 'import pyarrow' 2>/dev/null || pip install --break-system-packages -q pyarrow 2>&1 | tail -1
  python3 prepare_cremad.py --out "$ROOT/cremad/train_raw.jsonl" 2>&1 | tee -a "$LOG"
  mark cremad_prep
fi

# ------------------------------------------------------- STAGE 3: codec-encode ESD + CREMA-D (docker)
# NOTE: run in qwen-ft:latest (its qwen_tts stack imports cleanly). The nvcr pytorch image has a broken
# torchaudio (_torchaudio.abi3.so won't load) -> the tokenizer crashes there.
if is_done encode; then say "skip encode (marker present)"; else
  stage encode
  run_train "
    cd /root/qwen-ft/Qwen3-TTS/finetuning
    for SET in esd cremad; do
      echo \"[encode] \$SET START \$(date)\"
      python3 -u prepare_data.py --device cuda:0 --tokenizer_model_path $TOK \
        --input_jsonl  /root/qwen-ft/\$SET/train_raw.jsonl \
        --output_jsonl /root/qwen-ft/\$SET/train_with_codes.jsonl
      echo \"[encode] \$SET -> \$(wc -l < /root/qwen-ft/\$SET/train_with_codes.jsonl 2>/dev/null) lines\"
    done
  " 2>&1 | tee -a "$LOG"
  need_file encode "$ROOT/esd/train_with_codes.jsonl"
  need_file encode "$ROOT/cremad/train_with_codes.jsonl"
  mark encode
fi

# --------------------------------------------------------------------- STAGE 4: concat manifests (host)
if is_done concat; then say "skip concat (marker present)"; else
  stage concat
  python3 concat_manifests.py --out "$ROOT/multi_emotion/train_with_codes.jsonl" \
      "$ROOT/emovo/train_with_codes.jsonl" \
      "$ROOT/esd/train_with_codes.jsonl" \
      "$ROOT/cremad/train_with_codes.jsonl" 2>&1 | tee -a "$LOG"
  need_file concat "$ROOT/multi_emotion/train_with_codes.jsonl"
  mark concat
fi

# ---------------------------------------------------------- STAGE 5: dense FT L16-26 (docker, original)
if is_done train; then say "skip train (marker present)"; else
  stage train
  run_train "
    cd /root/qwen-ft/Qwen3-TTS/finetuning &&
    python3 -u dgx_sft_expr.py \
      --train_jsonl /root/qwen-ft/multi_emotion/train_with_codes.jsonl \
      --output_model_path /root/qwen-ft/out_multi_l1626 \
      --layers 16-26 --num_epochs 5 &&
    chmod -R a+rX /root/qwen-ft/out_multi_l1626
  " 2>&1 | tee -a "$LOG"
  need_file train "$OUT_CKPT/checkpoint-final/model.safetensors"
  mark train
fi

say "checkpoint: $OUT_CKPT/checkpoint-final/  (full CV dense ckpt -> extract .expr locally next)"
say "================ MULTI-SPEAKER EMOTION FT ALL DONE ================"
touch "$ROOT/multi_emotion.ALLDONE"
