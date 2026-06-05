#!/usr/bin/env bash
# Build a palette of Code-Predictor emotion/delivery steering vectors and a demo
# clip per tone. Each vector = mean cp_x(instruct) - mean cp_x(neutral), captured
# on a multi-sentence text so content averages out and only the delivery remains.
#
# Reusable .vec presets land in OUTDIR/<name>.vec; apply with:
#   ./qwen_tts -d <model> --text "..." --steer-vector OUTDIR/<name>.vec --steer-weight 0.7 ...
#
# Usage: bash tests/steer_palette.sh [model_dir] [outdir] [weight]
set -euo pipefail

MODEL="${1:-qwen3-tts-1.7b}"        # instruct is 1.7B-only → capture needs 1.7B
OUTDIR="${2:-voices/emotions}"
WEIGHT="${3:-0.7}"
SPK=ryan; LANG=English; SEED=42
BIN=./qwen_tts

# Rich, content-varied text → a clean (content-averaged) emotion direction.
CAP_TEXT="The meeting is scheduled for tomorrow. Please review the documents before then. I will send you the final report by email tonight."
# Neutral demo line that can plausibly carry any tone.
DEMO_TEXT="Well, here is the news everyone has been waiting for."

mkdir -p "$OUTDIR"

# preset -> instruct (delivery/tone framing for podcast/politics/audiobook/radio)
NAMES=(happy excited sad gloomy eccentric calm news dramatic)
INSTRUCTS=(
  "Speak in a happy, cheerful, upbeat tone, smiling"
  "Speak with high energy and excitement, enthusiastic and fast"
  "Speak in a sad, sorrowful, downcast tone"
  "Speak in a dark, gloomy, somber tone"
  "Speak in an eccentric, quirky, playful, theatrical way"
  "Speak in a calm, soft, soothing, relaxed tone"
  "Speak like a professional news anchor, clear and authoritative"
  "Speak in a dramatic, suspenseful, storytelling tone"
)

echo "== neutral baseline capture =="
QWEN_STEER_CAPTURE="$OUTDIR/_neutral.vec" "$BIN" -d "$MODEL" --text "$CAP_TEXT" \
  --seed $SEED -s $SPK -l $LANG -o /tmp/_pal_neutral.wav --silent

for i in "${!NAMES[@]}"; do
  n="${NAMES[$i]}"; ins="${INSTRUCTS[$i]}"
  echo "== capture: $n =="
  QWEN_STEER_CAPTURE="$OUTDIR/_$n.vec" "$BIN" -d "$MODEL" --text "$CAP_TEXT" \
    -I "$ins" --seed $SEED -s $SPK -l $LANG -o /tmp/_pal_$n.wav --silent
  python3 tests/steer_make.py "$OUTDIR/_$n.vec" "$OUTDIR/_neutral.vec" "$OUTDIR/$n.vec"
  echo "== demo: $n (weight $WEIGHT) =="
  "$BIN" -d "$MODEL" --text "$DEMO_TEXT" --steer-vector "$OUTDIR/$n.vec" \
    --steer-weight "$WEIGHT" --seed $SEED -s $SPK -l $LANG \
    -o "/tmp/demo_$n.wav" --silent
done

# Plain neutral demo for A/B reference.
"$BIN" -d "$MODEL" --text "$DEMO_TEXT" --seed $SEED -s $SPK -l $LANG -o /tmp/demo_neutral.wav --silent

rm -f "$OUTDIR"/_*.vec   # keep only the finished diff vectors
echo
echo "Palette written to $OUTDIR/  (weight $WEIGHT)"
echo "Listen:  afplay /tmp/demo_neutral.wav ; for n in ${NAMES[*]}; do echo \$n; afplay /tmp/demo_\$n.wav; done"
