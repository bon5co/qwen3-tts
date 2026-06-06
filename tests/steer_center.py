#!/usr/bin/env python3
"""Decorrelate an emotion palette by removing the shared common-mode component.

Each shipped preset is mean(cp_x|instruct) - mean(cp_x|neutral). Empirically the
presets are highly collinear (cos-to-mean 0.8-0.9): a large shared "I am being
instructed" direction dominates, and the emotion-specific part is only the residual.
That makes the tones sound alike. This rebuilds the palette as:

    vec_new = beta * mean_all  +  gamma * (vec - mean_all)

  beta  = how much shared "move away from neutral" energy to KEEP (0 = full centering)
  gamma = how much to AMPLIFY the distinctive residual
  --renorm: after the blend, rescale each vector back to its ORIGINAL norm
            (preserves the per-emotion baked strength the author tuned by ear)

Usage:
  python3 tests/steer_center.py <in_dir> <out_dir> [--beta 0.3] [--gamma 2.0] [--renorm]
"""
import sys, os, struct, glob, argparse
import numpy as np

NAMES = ["happy","excited","eager","proud","sad","gloomy","news","dramatic","calm"]

def load(p):
    b = open(p,'rb').read()
    assert b[:4] == b'QSTV', f"bad magic in {p}"
    dim = struct.unpack('<i', b[4:8])[0]
    v = np.frombuffer(b[8:8+4*dim], dtype='<f4').astype(np.float64)
    return v, dim

def save(p, v):
    with open(p,'wb') as f:
        f.write(b'QSTV')
        f.write(struct.pack('<i', len(v)))
        f.write(v.astype('<f4').tobytes())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('in_dir'); ap.add_argument('out_dir')
    ap.add_argument('--beta', type=float, default=0.3)
    ap.add_argument('--gamma', type=float, default=2.0)
    ap.add_argument('--renorm', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    names = [n for n in NAMES if os.path.exists(f"{a.in_dir}/{n}.vec")]
    V = {}; dim=None
    for n in names:
        V[n], dim = load(f"{a.in_dir}/{n}.vec")
    M = np.mean([V[n] for n in names], axis=0)
    print(f"common-mode |mean|={np.linalg.norm(M):.2f}  beta={a.beta} gamma={a.gamma} renorm={a.renorm}")
    for n in names:
        resid = V[n] - M
        out = a.beta * M + a.gamma * resid
        if a.renorm:
            on = np.linalg.norm(V[n]); cn = np.linalg.norm(out)
            if cn > 1e-9: out = out * (on / cn)
        save(f"{a.out_dir}/{n}.vec", out)
        print(f"  {n:9} |new|={np.linalg.norm(out):7.2f}")
    # report new contrast
    print("\nnew mean pairwise cosine (lower = more distinct):")
    W = {n: load(f"{a.out_dir}/{n}.vec")[0] for n in names}
    cs=[]
    for i,x in enumerate(names):
        for y in names[i+1:]:
            cs.append(np.dot(W[x],W[y])/(np.linalg.norm(W[x])*np.linalg.norm(W[y])+1e-9))
    print(f"  mean off-diagonal cosine = {np.mean(cs):+.3f}  (was +0.57 on shipped IT palette)")

if __name__ == '__main__':
    main()
