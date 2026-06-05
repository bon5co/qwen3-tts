#!/usr/bin/env python3
"""Build a Code-Predictor steering vector from two QWEN_STEER_CAPTURE dumps.

The emotion/prosody control vector is  vec = scale * (A - B), where A and B are
mean-cp_x .vec files captured with `QWEN_STEER_CAPTURE=path` — typically
A = an instruct run ("speak angrily"), B = a neutral run. Applying `vec` at
inference (`--steer-vector vec.vec --steer-weight N`) amplifies the (architecturally
weak) instruct signal in a controllable, dosable way.

.vec format: 'QSTV' magic (uint32 LE) + int32 dim + dim float32.

Usage:
  tests/steer_make.py ANGRY.vec NEUTRAL.vec OUT.vec [--scale 1.0] [--unit]
"""
import struct
import sys
import argparse

MAGIC = 0x56545351  # 'QSTV' little-endian


def read_vec(path):
    with open(path, "rb") as f:
        magic, dim = struct.unpack("<Ii", f.read(8))
        if magic != MAGIC:
            sys.exit(f"error: {path}: bad magic 0x{magic:08x} (not a .vec)")
        data = f.read(dim * 4)
        if len(data) != dim * 4:
            sys.exit(f"error: {path}: truncated (expected {dim} floats)")
        return dim, list(struct.unpack(f"<{dim}f", data))


def write_vec(path, vec):
    with open(path, "wb") as f:
        f.write(struct.pack("<Ii", MAGIC, len(vec)))
        f.write(struct.pack(f"<{len(vec)}f", *vec))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="positive capture (e.g. angry)")
    ap.add_argument("b", help="baseline capture (e.g. neutral)")
    ap.add_argument("out", help="output steering vector")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply the (a-b) diff (default 1.0; --steer-weight also scales)")
    ap.add_argument("--unit", action="store_true",
                    help="L2-normalize the diff before scaling (so --steer-weight is in 'norm units')")
    args = ap.parse_args()

    da, va = read_vec(args.a)
    db, vb = read_vec(args.b)
    if da != db:
        sys.exit(f"error: dim mismatch {da} vs {db}")

    diff = [x - y for x, y in zip(va, vb)]
    norm = sum(d * d for d in diff) ** 0.5
    if args.unit and norm > 0:
        diff = [d / norm for d in diff]
    diff = [d * args.scale for d in diff]

    write_vec(args.out, diff)
    out_norm = sum(d * d for d in diff) ** 0.5
    print(f"wrote {args.out}: dim={da}, raw_diff_L2={norm:.4f}, out_L2={out_norm:.4f}"
          f"{' (unit)' if args.unit else ''}, scale={args.scale}")


if __name__ == "__main__":
    main()
