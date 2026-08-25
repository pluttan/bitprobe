"""Quantize every linear layer of the network, sequentially.

Comparing a single replaced matrix is not a fair test of a fully quantized
model: the published weights were chosen together, each expecting the distorted
inputs the previous quantized layer produces. So the whole network has to be
converted the same way — layer by layer, always calibrating on activations that
already carry the error of everything upstream.
"""

import sys
import pathlib
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from research.reconstruct import GROUP, exact_group_scales, expand, objective  # noqa: E402

SEQ = 1024
BATCHES = 8
SWEEPS = 2
DAMP = 1e-2


def to_np(a: mx.array) -> np.ndarray:
    return np.array(a.astype(mx.float32), copy=True)


# ==============================
# ===  Quantization core     ===
# ==============================

def descend(w: np.ndarray, hessian: np.ndarray, sweeps: int = SWEEPS) -> np.ndarray:
    """Coordinate descent over signs with exact per-group scales."""
    out, width = w.shape
    b = np.sign(w)
    b[b == 0] = 1.0
    scales = exact_group_scales(w, b, hessian)

    diag = np.diag(hessian).copy()
    grad = (expand(scales, width) * b - w) @ hessian

    for _ in range(sweeps):
        s_full = expand(scales, width)
        for i in range(width):
            s = s_full[:, i]
            delta = -4.0 * s * b[:, i] * grad[:, i] + 4.0 * (s ** 2) * diag[i]
            take = delta < 0
            if not take.any():
                continue
            step = np.where(take, -2.0 * s * b[:, i], 0.0)
            grad += np.outer(step, hessian[i])
            b[take, i] *= -1.0

        candidate = exact_group_scales(w, b, hessian)
        cur = objective(expand(scales, width) * b - w, hessian)
        if objective(expand(candidate, width) * b - w, hessian) < cur:
            scales = candidate
        grad = (expand(scales, width) * b - w) @ hessian

    return expand(scales, width) * b


def quantize_linear(layer: nn.Module, acts: np.ndarray) -> float:
    """Replace one linear layer's weight with its binary reconstruction."""
    w = to_np(layer.weight)
    flat = acts.reshape(-1, acts.shape[-1])
    hessian = (flat.T @ flat) / len(flat)
    hessian += np.eye(hessian.shape[0], dtype=np.float32) * (
        DAMP * np.trace(hessian) / hessian.shape[0])

    approx = descend(w, hessian)
    layer.weight = mx.array(approx).astype(mx.bfloat16)
    mx.eval(layer.weight)
    ref = flat @ w.T
    return float(np.linalg.norm(flat @ approx.T - ref) / np.linalg.norm(ref))


# ==============================
# ===  Sequential sweep      ===
# ==============================

def block_inputs(block, h: mx.array, mask):
    """Reproduce a block's internals, exposing every linear layer's input."""
    attn_in = block.input_layernorm(h)
    r = block.self_attn(attn_in, mask, None)
    h2 = h + r
    mlp_in = block.post_attention_layernorm(h2)
    gate = block.mlp.gate_proj(mlp_in)
    up = block.mlp.up_proj(mlp_in)
    down_in = nn.silu(gate) * up
    return attn_in, mlp_in, down_in, h2


def main() -> None:
    model, _ = load("Qwen/Qwen3-1.7B")
    tokens = np.load("/tmp/calib_tokens.npy")
    batches = [mx.array(tokens[i * SEQ:(i + 1) * SEQ][None, :]) for i in range(BATCHES)]

    started = time.time()
    hidden = [model.model.embed_tokens(b) for b in batches]
    mask = create_attention_mask(hidden[0], None)

    for idx, block in enumerate(model.model.layers):
        # Collect this block's inputs under the already-quantized prefix.
        attn_ins, mlp_ins, down_ins = [], [], []
        for h in hidden:
            a, m, d, _ = block_inputs(block, h, mask)
            attn_ins.append(to_np(a)); mlp_ins.append(to_np(m)); down_ins.append(to_np(d))
        attn_in = np.concatenate(attn_ins); mlp_in = np.concatenate(mlp_ins)
        down_in = np.concatenate(down_ins)

        errs = []
        for name in ("q_proj", "k_proj", "v_proj"):
            errs.append(quantize_linear(getattr(block.self_attn, name), attn_in))

        # o_proj sees the attention output, which changed once q/k/v were quantized.
        o_ins = []
        for h in hidden:
            a = block.input_layernorm(h)
            B, L, _ = a.shape
            q = block.self_attn.q_norm(block.self_attn.q_proj(a).reshape(B, L, block.self_attn.n_heads, -1))
            k = block.self_attn.k_norm(block.self_attn.k_proj(a).reshape(B, L, block.self_attn.n_kv_heads, -1))
            v = block.self_attn.v_proj(a).reshape(B, L, block.self_attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
            q = block.self_attn.rope(q.transpose(0, 2, 1, 3))
            k = block.self_attn.rope(k.transpose(0, 2, 1, 3))
            att = mx.fast.scaled_dot_product_attention(q, k, v, scale=block.self_attn.scale, mask=mask)
            o_ins.append(to_np(att.transpose(0, 2, 1, 3).reshape(B, L, -1)))
        errs.append(quantize_linear(block.self_attn.o_proj, np.concatenate(o_ins)))

        for name in ("gate_proj", "up_proj"):
            errs.append(quantize_linear(getattr(block.mlp, name), mlp_in))
        errs.append(quantize_linear(block.mlp.down_proj, down_in))

        # Advance the hidden state through the now-quantized block.
        hidden = [block(h, mask, None) for h in hidden]
        for h in hidden:
            mx.eval(h)

        print(f"block {idx:2}  mean layer error {np.mean(errs):.4f}  "
              f"({time.time() - started:.0f}s)", flush=True)

    weights = {k: v for k, v in model.parameters().items()}
    mx.save_safetensors("/tmp/quantized_model.safetensors",
                        dict(tree_flatten_weights(weights)))
    print("saved quantized weights", flush=True)


def tree_flatten_weights(tree, prefix=""):
    items = []
    if isinstance(tree, dict):
        for k, v in tree.items():
            items.extend(tree_flatten_weights(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            items.extend(tree_flatten_weights(v, f"{prefix}.{i}"))
    else:
        items.append((prefix, tree))
    return items


if __name__ == "__main__":
    main()
