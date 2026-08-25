"""Calibration error vs held-out error for the first block.

A layer error of 0.13 that destroys perplexity means the error was measured on
the very activations the solver optimised against. This compares both, and
reports how far the reconstruction strayed in norm and how degenerate the
Hessian was.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

SEQ = 1024
BATCHES = 8

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")


def to_np(a):
    return np.array(a.astype(mx.float32), copy=True)


def collect(model, tokens, offset):
    """Every linear layer's input in block 0, under the unmodified network."""
    batches = [mx.array(tokens[offset + i * SEQ:offset + (i + 1) * SEQ][None, :])
               for i in range(BATCHES)]
    hidden = [model.model.embed_tokens(b) for b in batches]
    mask = create_attention_mask(hidden[0], None)
    block = model.model.layers[0]
    sa = block.self_attn

    attn_in, mlp_in, down_in, o_in = [], [], [], []
    for h in hidden:
        x = block.input_layernorm(h)
        attn_in.append(to_np(x))

        B, L, _ = x.shape
        q = sa.q_norm(sa.q_proj(x).reshape(B, L, sa.n_heads, -1))
        k = sa.k_norm(sa.k_proj(x).reshape(B, L, sa.n_kv_heads, -1))
        v = sa.v_proj(x).reshape(B, L, sa.n_kv_heads, -1).transpose(0, 2, 1, 3)
        q = sa.rope(q.transpose(0, 2, 1, 3))
        k = sa.rope(k.transpose(0, 2, 1, 3))
        att = mx.fast.scaled_dot_product_attention(q, k, v, scale=sa.scale, mask=mask)
        o_in.append(to_np(att.transpose(0, 2, 1, 3).reshape(B, L, -1)))

        h2 = h + block.self_attn(x, mask, None)
        m = block.post_attention_layernorm(h2)
        mlp_in.append(to_np(m))
        down_in.append(to_np(nn.silu(block.mlp.gate_proj(m)) * block.mlp.up_proj(m)))

    def flat(chunks):
        a = np.concatenate(chunks)
        return a.reshape(-1, a.shape[-1])

    a, m, d, o = flat(attn_in), flat(mlp_in), flat(down_in), flat(o_in)
    return {"q_proj": a, "k_proj": a, "v_proj": a, "o_proj": o,
            "gate_proj": m, "up_proj": m, "down_proj": d}


def main():
    tokens = np.load("/tmp/calib_tokens.npy")
    saved = mx.load("/tmp/quantized_model.safetensors")
    model, _ = load("Qwen/Qwen3-1.7B")
    block = model.model.layers[0]

    calib = collect(model, tokens, 0)
    heldout = collect(model, tokens, 40000)

    print(f"{'layer':10} {'calib':>8} {'heldout':>9} {'|q|/|w|':>8} {'cond':>10}")
    for name in ATTN + MLP:
        parent = "self_attn" if name in ATTN else "mlp"
        mod = getattr(getattr(block, parent), name)
        w = to_np(mod.weight)
        q = to_np(saved[f"model.layers.0.{parent}.{name}.weight"])

        errs = []
        for acts in (calib, heldout):
            x = acts[name]
            ref = x @ w.T
            errs.append(float(np.linalg.norm(x @ q.T - ref) / np.linalg.norm(ref)))

        x = calib[name]
        h = (x.T @ x) / len(x)
        ev = np.linalg.eigvalsh(h.astype(np.float64))
        print(f"{name:10} {errs[0]:8.4f} {errs[1]:9.4f} "
              f"{np.linalg.norm(q) / np.linalg.norm(w):8.3f} "
              f"{ev[-1] / max(ev[0], 1e-12):10.2e}", flush=True)


if __name__ == "__main__":
    main()
