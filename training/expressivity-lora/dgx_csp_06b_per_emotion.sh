#!/usr/bin/env bash
# ============================================================================================
# PER-EMOTION CSP-FT for the 0.6B — the fix for "the multi-emotion FT doesn't differentiate".
# A SEPARATE, ISOLATED run (runs/csp_06b_peremo), never touches the 1.7B or the multi-emotion run.
#
# WHY per-emotion (ear verdict 2026-06-30 on the multi-emotion italian_csp_06b.expr: "non emoziona,
# non distingue un'emozione dall'altra, come neutro a temperature diverse"):
#   The 0.6B IGNORES --instruct, so a single multi-emotion FT has NO way to SELECT the emotion at
#   inference -> it renders the AVERAGE. The probe hitting 0.91 emotion-classify acc proves the model
#   CAN represent emotion; it just isn't told which to produce. Fix: one .expr PER EMOTION (trained on
#   that emotion's data only). At inference the SELECTOR = which .expr you load (no instruct needed) —
#   exactly how the 1.7B uses the instruct to select, but file-selected for the instruct-less 0.6B.
#
# Reuses the already-encoded Italian set + the L15-20 band the multi-emotion probe found (no re-probe).
# Trains the FIXED 0.6B trainer (dgx_sft_expr_csp_06b.py, text_projection bridge). Exports one .expr/emo.
#
# Usage (DGX, ~/qwen-ft):
#   nohup bash dgx_csp_06b_per_emotion.sh >/dev/null 2>&1 &      # default: sad/anger/joy (validation)
#   EMOS="sadness anger joy fear disgust surprise" nohup bash dgx_csp_06b_per_emotion.sh ... &   # all
#   tail -f ~/qwen-ft/runs/csp_06b_peremo/peremo.log
# ============================================================================================
set -uo pipefail
ROOT="$HOME/qwen-ft"
RUN_DIR="${RUN_DIR:-$ROOT/runs/csp_06b_peremo}"
cd "$ROOT"; mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/peremo.log"
TRAIN_IMG="qwen-ft:latest"
MODEL="${MODEL:-/root/qwen-ft/models/0.6B-CustomVoice}"
MODEL_HOST="${MODEL/\/root\/qwen-ft/$ROOT}"
SRC_CODES="${SRC_CODES:-$ROOT/runs/csp_italian/italian_emotion/train_with_codes.jsonl}"  # the already-encoded set
CSP_LAYERS="${CSP_LAYERS:-15,16,17,18,19,20}"     # the 0.6B-native band the multi-emotion probe found
EPOCHS_FT="${EPOCHS_FT:-10}"
EMOS="${EMOS:-sadness anger joy}"                 # validation set; expand via env for all 6
# map dataset label -> short .expr name
shortname(){ case "$1" in sadness)echo sad;; *)echo "$1";; esac; }

ts(){ date "+%Y-%m-%d %H:%M:%S"; }
say(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
run_train(){ docker run --rm --gpus all --ipc=host -e PYTHONUNBUFFERED=1 \
               -v "$ROOT:/root/qwen-ft" -v "$ROOT:$ROOT" "$TRAIN_IMG" bash -c "$1"; }

say "================ PER-EMOTION 0.6B CSP-FT START ================"
say "run_dir=$RUN_DIR  model=$MODEL  band=$CSP_LAYERS  epochs=$EPOCHS_FT  emos=[$EMOS]  src=$SRC_CODES"
[ -s "$SRC_CODES" ] || { say "FAIL: encoded set not found: $SRC_CODES"; exit 1; }
[ -f "$MODEL_HOST/model.safetensors" ] || { say "FAIL: 0.6B model not found at $MODEL_HOST"; exit 1; }

# ---- STAGE A: split the encoded set into per-emotion jsonl (host) ----
if [ -f "$RUN_DIR/split.DONE" ]; then say "skip split"; else
  say ">>> split by emotion"
  mkdir -p "$RUN_DIR/data"
  python3 - "$SRC_CODES" "$RUN_DIR/data" $EMOS <<'PY' 2>&1 | tee -a "$LOG"
import sys,json,collections
src, outdir = sys.argv[1], sys.argv[2]; emos=set(sys.argv[3:])
files={e:open(f"{outdir}/{e}.jsonl","w") for e in emos}
c=collections.Counter()
for ln in open(src):
    try: r=json.loads(ln)
    except: continue
    e=r.get("emotion") or r.get("label")
    if e in files: files[e].write(ln); c[e]+=1
for f in files.values(): f.close()
print("per-emotion rows:", dict(c))
PY
  touch "$RUN_DIR/split.DONE"
fi

# ---- STAGE B: per-emotion CSP-FT + export ----
for E in $EMOS; do
  SN=$(shortname "$E")
  DATA="$RUN_DIR/data/$E.jsonl"
  CKPT="$RUN_DIR/out_$SN"
  EXPR_OUT="$RUN_DIR/italian_${SN}_06b.expr"
  [ -s "$DATA" ] || { say "skip $E (no data)"; continue; }
  if [ -f "$RUN_DIR/$SN.DONE" ]; then say "skip $SN (done)"; continue; fi
  say ">>> [$SN] CSP-FT on $(wc -l < "$DATA") rows, band $CSP_LAYERS"
  run_train "
    cd /root/qwen-ft/Qwen3-TTS/finetuning &&
    python3 -u dgx_sft_expr_csp_06b.py --train_jsonl $DATA \
      --init_model_path $MODEL --output_model_path $CKPT \
      --csp-layers '$CSP_LAYERS' --scope full --num_epochs $EPOCHS_FT &&
    chmod -R a+rX $CKPT
  " 2>&1 | tee -a "$LOG"
  [ -s "$CKPT/checkpoint-final/model.safetensors" ] || { say "FAIL [$SN]: no checkpoint"; continue; }
  say ">>> [$SN] export -> $EXPR_OUT"
  python3 -c 'import numpy' 2>/dev/null || pip install --break-system-packages -q numpy 2>&1 | tail -1
  python3 "$ROOT/tests/expr_extract.py" "$MODEL_HOST" "$CKPT/checkpoint-final" "$EXPR_OUT" \
      --lang Italian --hidden 1024 2>&1 | tee -a "$LOG"
  [ -s "$EXPR_OUT" ] && touch "$RUN_DIR/$SN.DONE" && say "<<< [$SN] DONE -> $EXPR_OUT"
done

say "================ PER-EMOTION 0.6B ALL DONE ================"
say "pull:  scp 'dgx:$RUN_DIR/italian_*_06b.expr' presets/expr/"
say "use :  ./qwen_tts -d qwen3-tts-0.6b -s ryan -l Italian -T 1.1 --expr presets/expr/italian_sad_06b.expr --expr-weight 1.0 --text '...'"
touch "$RUN_DIR/peremo.ALLDONE"
