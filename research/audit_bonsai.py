"""Is every matrix in Bonsai actually one bit?

Our whole-network binarisation collapses while Bonsai keeps a working model.
Before assuming its sign-selection is smarter, check the simpler explanation:
that some layers were never binarised at all. Per group of 128 weights, a true
binary layer has exactly one distinct magnitude.
"""

import collections

import mlx.core as mx
import numpy as np
from mlx_lm import load

GROUP = 128
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")


def to_np(a):
    return np.array(a.astype(mx.float32), copy=True)


def magnitudes_per_group(w: np.ndarray, rows: int = 64) -> float:
    """Mean count of distinct magnitudes per group — 1.0 means binary."""
    sub = w[:rows]
    g = np.abs(sub.reshape(rows, -1, GROUP))
    return float(np.mean([len(np.unique(row)) for row in g.reshape(-1, GROUP)]))


def main():
    ours, _ = load("Qwen/Qwen3-1.7B")
    bonsai, _ = load("prism-ml/Bonsai-1.7B-unpacked")

    print(f"{'layer':24} {'|vals|/grp':>10} {'zeros':>7} {'signs':>7} {'|b|/|w|':>8}")
    tally = collections.Counter()
    for idx, (bb, ob) in enumerate(zip(bonsai.model.layers, ours.model.layers)):
        for parent, names in (("self_attn", ATTN), ("mlp", MLP)):
            for name in names:
                b = to_np(getattr(getattr(bb, parent), name).weight)
                w = to_np(getattr(getattr(ob, parent), name).weight)
                vals = magnitudes_per_group(b)
                zero = float((b == 0).mean())
                nz = b != 0
                agree = float((np.sign(b[nz]) == np.sign(w[nz])).mean())
                ratio = float(np.linalg.norm(b) / np.linalg.norm(w))
                tally["binary" if vals <= 1.01 else
                      "ternary" if vals <= 2.01 else "wider"] += 1
                if idx in (0, 13, 27):
                    print(f"L{idx:<2} {parent[:4]}.{name:12} {vals:10.2f} "
                          f"{zero:7.3f} {agree:7.3f} {ratio:8.3f}", flush=True)

    print("\nacross all 196 matrices:", dict(tally))


if __name__ == "__main__":
    main()
