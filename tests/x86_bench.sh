#!/usr/bin/env bash
# ============================================================================
# x86_bench.sh — clean RTF A/B on an x86 box (no fragile copy-paste pipes).
#
# Answers two questions:
#   [A] Does our optimized stack actually help AT THE SAME CORE COUNT? (-j1)
#       scalar-bf16 (~ the original code) vs VNNI-int8 (ours). Isolates the
#       kernel work from threading (which a VM can't schedule well).
#   [B] Full RTF matrix at -j4 across scalar / AVX2 / VNNI / int8 / int4 / bf16.
#
# Usage (on the box, inside the repo):
#     bash tests/x86_bench.sh                 # uses qwen3-tts-0.6b
#     bash tests/x86_bench.sh <model-dir>
#
# Builds the binaries it needs (qwen_tts_scalar / _avx2 / _avx512vnni) once and
# reuses them. Prints a single clean table. Paste the table back.
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
MODEL="${1:-qwen3-tts-0.6b}"
TXT="The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs."

if [ ! -d "$MODEL" ]; then
    echo "Model dir '$MODEL' not found. Pass one: bash tests/x86_bench.sh <model-dir>"
    exit 1
fi

# Build a SIMD level into a named binary if not already present.
build() { # $1=SIMD  $2=outname
    if [ -x "$2" ]; then echo ">> reuse $2"; return 0; fi
    echo ">> building $2 (SIMD=$1) ..."
    make clean >/dev/null 2>&1
    if make blas SIMD="$1" >/tmp/build_$1.log 2>&1; then
        cp -f qwen_tts "$2"
        echo "   ok ($(du -h "$2" | cut -f1))"
    else
        echo "   BUILD FAILED (SIMD=$1) — tail:"; tail -8 /tmp/build_$1.log
        return 1
    fi
}

build scalar     qwen_tts_scalar
build avx2       qwen_tts_avx2
build avx512vnni qwen_tts_avx512vnni

# Run one config and print "label  RTF x.xx  CP yy.y ms/f".
# Pass the FULL command (incl. optional 'env VAR=1' prefix); fixed args appended.
run() { # $1=label  then the command + its flags
    local label="$1"; shift
    local out rtf cp
    out=$("$@" -d "$MODEL" --text "$TXT" --seed 42 -s ryan -l English -o /tmp/bench.wav 2>&1)
    rtf=$(printf '%s\n' "$out" | grep -oE 'RTF [0-9.]+' | head -1 | awk '{print $2}')
    cp=$(printf '%s\n'  "$out" | grep -oE '[0-9.]+ ms/f' | tail -1)   # 2nd ms/f line = Code Predictor
    printf "  %-32s RTF %-7s CP %s\n" "$label" "${rtf:-ERR}" "${cp:-?}"
}

echo "================================================================"
echo " x86 RTF bench — $MODEL"
echo " CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
echo " vCPUs: $(nproc) | L3-share(cpu0): $(cat /sys/devices/system/cpu/cpu0/cache/index3/shared_cpu_list 2>/dev/null)"
echo "================================================================"
echo "[A] Same core count (-j1): does the kernel work help?"
run "scalar bf16 -j1 (~original)" ./qwen_tts_scalar          -j1
run "VNNI   int8 -j1 (ours)"      ./qwen_tts_avx512vnni --int8 -j1
echo
echo "[B] Full matrix (-j4):"
run "scalar bf16 -j4"             ./qwen_tts_scalar              -j4
run "avx2   int8 -j4"             ./qwen_tts_avx2        --int8  -j4
run "VNNI   int8 -j4"             ./qwen_tts_avx512vnni  --int8  -j4
run "VNNI   int8 -j4 (vnni off)"  env QWEN_NO_VNNI=1 ./qwen_tts_avx512vnni --int8 -j4
run "VNNI   int4 -j4"             ./qwen_tts_avx512vnni  --int4  -j4
run "VNNI   bf16 -j4"             ./qwen_tts_avx512vnni          -j4
echo "================================================================"
echo "Read [A]: if 'scalar bf16 -j1' RTF is much HIGHER (worse) than"
echo "'VNNI int8 -j1', our kernel stack works — the weak -j4 scaling is"
echo "the VM, not the code. Paste this whole table back."
