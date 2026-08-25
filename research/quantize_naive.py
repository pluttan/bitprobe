"""Naive per-group binarisation of the whole network, as a floor to beat.

No calibration, no activations: sign of the weight, one scale per group of 128
chosen as the mean magnitude. If the Hessian-aware solver scores worse than
this, it is overfitting the calibration set rather than approximating the layer.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load

GROUP = 128
SEQ = 1024

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")


def naive(w: np.ndarray) -> np.ndarray:
    out, width = w.shape
    g = w.reshape(out, width // GROUP, GROUP)
    b = np.sign(g)
    b[b == 0] = 1.0
    return (b * np.abs(g).mean(axis=2, keepdims=True)).reshape(out, width)


def perplexity(model, tokens, limit=8):
    total, count = 0.0, 0
    for i in range(limit):
        chunk = tokens[i * SEQ:(i + 1) * SEQ + 1]
        if len(chunk) < SEQ + 1:
            break
        logits = model(mx.array(chunk[None, :-1])).astype(mx.float32)
        loss = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                       mx.array(chunk[None, 1:]).reshape(-1),
                                       reduction="mean")
        mx.eval(loss)
        total += float(loss) * SEQ
        count += SEQ
    return float(np.exp(total / count))


def main():
    tokens = np.load("/tmp/calib_tokens.npy")[40000:]
    model, _ = load("Qwen/Qwen3-1.7B")
    print(f"base           ppl = {perplexity(model, tokens):10.3f}", flush=True)

    for block in model.model.layers:
        for parent, names in ((block.self_attn, ATTN), (block.mlp, MLP)):
            for name in names:
                mod = getattr(parent, name)
                w = np.array(mod.weight.astype(mx.float32), copy=True)
                mod.weight = mx.array(naive(w)).astype(mx.bfloat16)
        mx.eval(block.parameters())

    print(f"naive 1-bit    ppl = {perplexity(model, tokens):10.3f}", flush=True)


if __name__ == "__main__":
    main()
