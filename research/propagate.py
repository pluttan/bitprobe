"""Does a layer's quantization error stay local, or does it compound?

Layer-wise reconstruction wins decisively on its own metric, yet the published
model is worse by that same metric while scoring 90% on benchmarks. The only
way both can hold is if the published weights are chosen so that errors cancel
downstream rather than being small where they are made.

This runs the full network three times — once untouched, once with a single
matrix replaced by each candidate — and tracks how far the hidden state has
drifted after every block.
"""

import sys
import pathlib

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from bitprobe import metrics  # noqa: E402

GROUP = 128
SEQ = 2048
TARGET = "self_attn.q_proj"


def run(model, ids: mx.array) -> list[np.ndarray]:
    """Forward pass that keeps the hidden state after each block."""
    h = model.model.embed_tokens(ids)
    mask = create_attention_mask(h, None)
    states = []
    for layer in model.model.layers:
        h = layer(h, mask, None)
        mx.eval(h)
        states.append(np.array(h.astype(mx.float32), copy=True))
    return states


def drift(ref: list[np.ndarray], other: list[np.ndarray]) -> list[float]:
    return [
        float(np.linalg.norm(a - b) / np.linalg.norm(a))
        for a, b in zip(ref, other)
    ]


def main() -> None:
    model, _ = load("Qwen/Qwen3-1.7B")
    tokens = np.load("/tmp/calib_tokens.npy")[:SEQ]
    ids = mx.array(tokens[None, :])

    target = model.model.layers[0].self_attn.q_proj
    original = np.array(target.weight.astype(mx.float32), copy=True)

    print(f"reference forward, {len(model.model.layers)} layers", flush=True)
    ref = run(model, ids)

    approx = np.load("/tmp/recon_q_proj.npy")
    bonsai = np.load("/tmp/bonsai_q_proj.npy")
    naive = metrics.naive_binary(original.astype(np.float32), GROUP)

    curves = {}
    for name, matrix in (("reconstructed", approx), ("bonsai", bonsai), ("naive", naive)):
        target.weight = mx.array(matrix.astype(np.float32)).astype(mx.bfloat16)
        mx.eval(target.weight)
        curves[name] = drift(ref, run(model, ids))
        print(f"{name} done", flush=True)
    target.weight = mx.array(original).astype(mx.bfloat16)

    print("\n=== hidden-state drift after block N (relative) ===")
    print(f"{'block':>6} {'reconstructed':>14} {'bonsai':>10} {'naive':>10}")
    for i in (0, 1, 2, 4, 8, 12, 16, 20, 24, 27):
        print(f"{i:6} {curves['reconstructed'][i]:14.4f} "
              f"{curves['bonsai'][i]:10.4f} {curves['naive'][i]:10.4f}")

    print("\n=== growth from block 0 to last ===")
    for name, c in curves.items():
        print(f"  {name:14} {c[0]:.4f} -> {c[-1]:.4f}   (x{c[-1] / max(c[0], 1e-9):.2f})")


if __name__ == "__main__":
    main()
