#!/usr/bin/env bash
# CSP-FT ITALIAN orchestrator (plan_emo_v2.md) — single-language Italian emotion fine-tune the PRO way:
#   1) build a bigger Italian emotion set: EMOVO (0.5h, 6 actors) + Emozionalmente (~6h, 431 actors)
#   2) PROBE which Talker layers carry emotion (csp_probe.py)  -> csp_layers_italian.json
#   3) CSP-FT: train ONLY those layers, FREEZE the rest incl. pronunciation (dgx_sft_expr_csp.py)
#
# WHY (research, plan_emo_v2.md): mixing languages in a small FT = negative transfer. Pros use
# Characteristic-Specific Partial FT (arXiv 2501.14273): probe+freeze keeps speech clean (CER 1.2% vs
# full-FT 3.9%) and REMOVES the need for the τ disentangle subtraction. Single language, more data, freeze.
#
# ISOLATION (lesson learned): EVERYTHING this run produces lives under RUN_DIR (default runs/csp_italian/)
# — data, .DONE markers, outputs, log. NOTHING is written into the shared $ROOT except via the read-only
# SOURCES it reads (models/, the EMOVO raw audio). So this run can never collide with / dirty other runs.
#
# Idempotent: each stage guarded by a <stage>.DONE marker IN RUN_DIR; delete a marker to force re-run.
#
# Usage (on the DGX, from ~/qwen-ft):
#   SMART=1 TRAIN_JUDGE=1 nohup bash dgx_csp_italian.sh >/dev/null 2>&1 &
#   tail -f ~/qwen-ft/runs/csp_italian/csp_italian.log
set -uo pipefail

ROOT="$HOME/qwen-ft"                              # shared mount base (read-only sources: models/, emovo audio)
RUN_DIR="${RUN_DIR:-$ROOT/runs/csp_italian}"      # ISOLATED run dir: all data/markers/outputs/log here
cd "$ROOT"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/csp_italian.log"
TRAIN_IMG="qwen-ft:latest"
TOK="/root/qwen-ft/models/tokenizer-12hz"         # shared, read-only (container path via mount)
MODEL="/root/qwen-ft/models/1.7B-CustomVoice"     # shared, read-only
PROBE_JSON="$RUN_DIR/csp_layers_italian.json"
OUT_CKPT="$RUN_DIR/out_csp_italian"
EMOZ_DIR="$RUN_DIR/emozionalmente_zenodo"
EPOCHS_FT="${EPOCHS_FT:-10}"
TOPK="${TOPK:-2}"
MINAGREE="${MINAGREE:-0}"

ts()  { date "+%Y-%m-%d %H:%M:%S"; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }
done_marker() { echo "$RUN_DIR/$1.DONE"; }         # markers live IN the isolated run dir -> no cross-run collision
is_done() { [ -f "$(done_marker "$1")" ]; }
mark()    { touch "$(done_marker "$1")"; say "<<< stage '$1' DONE"; }
stage()   { say ">>> stage '$1' START"; }
need_file() { [ -s "$2" ] || { say "FAIL stage '$1': missing/empty output $2"; exit 1; }; }

# Mount $ROOT at BOTH /root/qwen-ft (container convention) and its real host path, so RUN_DIR (a subdir of
# $ROOT) is reachable in-container at its absolute host path. In-docker commands below use $RUN_DIR / $ROOT
# host-absolute paths (valid via the second mount).
run_train() { docker run --rm --gpus all --ipc=host -e PYTHONUNBUFFERED=1 \
                -v "$ROOT:/root/qwen-ft" -v "$ROOT:$ROOT" "$TRAIN_IMG" bash -c "$1"; }

say "================ CSP-FT ITALIAN START ================"
say "run_dir=$RUN_DIR  model=$MODEL  out=$OUT_CKPT  epochs_ft=$EPOCHS_FT  top_k=$TOPK"

# ----------------------------------------------------------- STAGE 1: EMOVO prep (host) -> RUN_DIR/emovo
# EMOVO raw audio is a shared read-only SOURCE ($ROOT/emovo/wav24k, abs paths inside the jsonl). We only
# copy its manifest into RUN_DIR (or build it) so the encoded output stays isolated in RUN_DIR.
if is_done emovo_prep; then say "skip emovo_prep (marker present)"; else
  stage emovo_prep
  mkdir -p "$RUN_DIR/emovo"
  if [ -s "$RUN_DIR/emovo/train_raw.jsonl" ]; then
    say "RUN_DIR/emovo/train_raw.jsonl already present, reusing"
  elif [ -s "$ROOT/emovo/train_raw.jsonl" ]; then
    say "copying shared emovo/train_raw.jsonl into RUN_DIR (audio paths are absolute -> still resolve)"
    cp "$ROOT/emovo/train_raw.jsonl" "$RUN_DIR/emovo/train_raw.jsonl"
  else
    python3 dgx_emovo_prep.py --out "$RUN_DIR/emovo/train_raw.jsonl" 2>&1 | tee -a "$LOG"
  fi
  need_file emovo_prep "$RUN_DIR/emovo/train_raw.jsonl"
  mark emovo_prep
fi

# -------------------------------------------------- STAGE 2: Emozionalmente download + prep (host) -> RUN_DIR
# Zenodo package (records/12616095): same audio as CAMEO PLUS the human-validation CSVs -> SMART/MINAGREE.
if is_done emoz_prep; then say "skip emoz_prep (marker present)"; else
  stage emoz_prep
  python3 -c 'import soundfile' 2>/dev/null || pip install --break-system-packages -q soundfile 2>&1 | tail -2
  if [ ! -f "$EMOZ_DIR/emozionalmente/metadata/samples.csv" ]; then
    mkdir -p "$EMOZ_DIR"
    say "downloading emozionalmente.zip from Zenodo (~559 MB) ..."
    curl -L -s -o "$EMOZ_DIR/emozionalmente.zip" \
      "https://zenodo.org/records/12616095/files/emozionalmente.zip?download=1"
    say "unzipping ..."
    unzip -q -o "$EMOZ_DIR/emozionalmente.zip" -d "$EMOZ_DIR" -x "__MACOSX/*"
  fi
  # SMART=1 -> per-emotion bar (>=3/5 strong, ALL disgust+fear); else MINAGREE>0 -> uniform >=N; else all.
  AGREE_FLAG=""
  if [ "${SMART:-0}" = "1" ]; then AGREE_FLAG="--smart-agreement"
  elif [ "$MINAGREE" -gt 0 ]; then AGREE_FLAG="--min-agreement $MINAGREE"; fi
  say "emoz prep: --loudnorm $AGREE_FLAG (SMART=${SMART:-0} MINAGREE=$MINAGREE)"
  python3 prepare_emozionalmente.py --local-dir "$EMOZ_DIR" \
      --out "$RUN_DIR/emozionalmente/train_raw.jsonl" --loudnorm $AGREE_FLAG 2>&1 | tee -a "$LOG"
  need_file emoz_prep "$RUN_DIR/emozionalmente/train_raw.jsonl"
  mark emoz_prep
fi

# ------------------------------------- STAGE 2b: train the SER JUDGE (optional, TRAIN_JUDGE=1) ---------
if [ "${TRAIN_JUDGE:-0}" = "1" ]; then
  if is_done judge; then say "skip judge (marker present)"; else
    stage judge
    if [ ! -f "$ROOT/tests/train_ser_judge.py" ]; then
      say "WARN judge: tests/train_ser_judge.py not on DGX — skipping judge, continuing to CSP-FT"
    else
      run_train "
        cd /root/qwen-ft &&
        python3 -c 'import transformers,datasets,librosa,soundfile' 2>/dev/null ||
          pip install --break-system-packages -q transformers datasets librosa soundfile 2>&1 | tail -2
        python3 -u tests/train_ser_judge.py --data-dir $EMOZ_DIR --out $RUN_DIR/ser_judge_it --epochs ${JUDGE_EPOCHS:-8}
      " 2>&1 | tee -a "$LOG"
      # NON-FATAL: optional eval tool; if it fails, warn and proceed to the CSP-FT anyway.
      if [ -s "$RUN_DIR/ser_judge_it/test_report.json" ]; then mark judge
      else say "WARN judge training failed (see log) — continuing to CSP-FT WITHOUT the judge"; fi
    fi
  fi
fi

# ----------------------------------------------- STAGE 3: codec-encode EMOVO + Emozionalmente (docker)
if is_done encode; then say "skip encode (marker present)"; else
  stage encode
  run_train "
    cd /root/qwen-ft/Qwen3-TTS/finetuning
    for SET in emovo emozionalmente; do
      echo \"[encode] \$SET START \$(date)\"
      python3 -u prepare_data.py --device cuda:0 --tokenizer_model_path $TOK \
        --input_jsonl  $RUN_DIR/\$SET/train_raw.jsonl \
        --output_jsonl $RUN_DIR/\$SET/train_with_codes.jsonl
      echo \"[encode] \$SET -> \$(wc -l < $RUN_DIR/\$SET/train_with_codes.jsonl 2>/dev/null) lines\"
    done
  " 2>&1 | tee -a "$LOG"
  need_file encode "$RUN_DIR/emovo/train_with_codes.jsonl"
  need_file encode "$RUN_DIR/emozionalmente/train_with_codes.jsonl"
  mark encode
fi

# ------------------------------------------------------ STAGE 4: concat -> single Italian set (host)
if is_done concat; then say "skip concat (marker present)"; else
  stage concat
  python3 concat_manifests.py --out "$RUN_DIR/italian_emotion/train_with_codes.jsonl" \
      --langs Italian,Italian \
      "$RUN_DIR/emovo/train_with_codes.jsonl" \
      "$RUN_DIR/emozionalmente/train_with_codes.jsonl" 2>&1 | tee -a "$LOG"
  need_file concat "$RUN_DIR/italian_emotion/train_with_codes.jsonl"
  mark concat
fi

# ------------------------------------------------------ STAGE 5: CSP probe -> emotion layers (docker)
if is_done probe; then say "skip probe (marker present)"; else
  stage probe
  run_train "
    cd /root/qwen-ft/Qwen3-TTS/finetuning &&
    python3 -u csp_probe.py \
      --train_jsonl $RUN_DIR/italian_emotion/train_with_codes.jsonl \
      --init_model_path $MODEL --out_json $PROBE_JSON --epochs 3 --top_k $TOPK
  " 2>&1 | tee -a "$LOG"
  need_file probe "$PROBE_JSON"
  mark probe
fi

CSP_LAYERS=$(python3 -c "import json;print(','.join(map(str,json.load(open('$PROBE_JSON'))['selected']['top_k'])))")
say "probe selected CSP blocks: $CSP_LAYERS"
[ -n "$CSP_LAYERS" ] || { say "FAIL: empty CSP layer selection from $PROBE_JSON"; exit 1; }

# ------------------------------------------------------------ STAGE 6: CSP-FT (docker, NEW trainer)
if is_done train; then say "skip train (marker present)"; else
  stage train
  run_train "
    cd /root/qwen-ft/Qwen3-TTS/finetuning &&
    python3 -u dgx_sft_expr_csp.py \
      --train_jsonl $RUN_DIR/italian_emotion/train_with_codes.jsonl \
      --output_model_path $OUT_CKPT \
      --csp-layers '$CSP_LAYERS' --scope full --num_epochs $EPOCHS_FT &&
    chmod -R a+rX $OUT_CKPT
  " 2>&1 | tee -a "$LOG"
  need_file train "$OUT_CKPT/checkpoint-final/model.safetensors"
  mark train
fi

say "checkpoint: $OUT_CKPT/checkpoint-final/  (full CV ckpt -> pull + export_expr.py vs base -> italian_csp.expr)"
say "probe layers: $PROBE_JSON  (CSP blocks: $CSP_LAYERS)"
say "================ CSP-FT ITALIAN ALL DONE ================"
touch "$RUN_DIR/csp_italian.ALLDONE"
