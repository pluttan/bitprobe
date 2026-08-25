"""Train one block's binary signs instead of solving for them.

The solver minimises the distance to the original matrix and collapses the
network; Bonsai's matrices are far from the original yet the network survives.
That only happens if the signs were chosen by gradient against the block's
output. This trains a single block that way and measures the same held-out
error the solver was scored on.
"""

import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

GROUP = 128
SEQ = 1024
STEPS = 400
LR = 3e-4
BLOCK = 0

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")


def quantize(w: mx.array) -> mx.array:
    """Binary weights with a per-group scale, transparent to the gradient."""
    g = w.reshape(w.shape[0], -1, GROUP)
    s = mx.abs(g).mean(axis=2, keepdims=True)
    b = mx.where(g >= 0, 1.0, -1.0)
    q = (b * s).reshape(w.shape)
    return w + mx.stop_gradient(q - w)


def as_tree(master: dict) -> dict:
    """Nest the flat master weights the way the block expects them."""
    tree = {"self_attn": {}, "mlp": {}}
    for name, w in master.items():
        parent = "self_attn" if name in ATTN else "mlp"
        tree[parent][name] = {"weight": quantize(w).astype(mx.bfloat16)}
    return tree


def hidden_for(model, tokens, offset, count):
    batches = [mx.array(tokens[offset + i * SEQ:offset + (i + 1) * SEQ][None, :])
               for i in range(count)]
    return [model.model.embed_tokens(b) for b in batches]


def main():
    tokens = np.load("/tmp/calib_tokens.npy")
    teacher_model, _ = load("Qwen/Qwen3-1.7B")
    student_model, _ = load("Qwen/Qwen3-1.7B")
    teacher = teacher_model.model.layers[BLOCK]
    student = student_model.model.layers[BLOCK]

    train_h = hidden_for(teacher_model, tokens, 0, 32)
    test_h = hidden_for(teacher_model, tokens, 40000, 8)
    mask = create_attention_mask(train_h[0], None)

    train_y = [mx.stop_gradient(teacher(h, mask, None)) for h in train_h]
    test_y = [mx.stop_gradient(teacher(h, mask, None)) for h in test_h]
    mx.eval(train_y, test_y)

    master = {}
    for parent, names in ((student.self_attn, ATTN), (student.mlp, MLP)):
        for name in names:
            master[name] = getattr(parent, name).weight.astype(mx.float32)

    def loss_fn(master, h, y):
        student.update(as_tree(master))
        return mx.mean(mx.square(student(h, mask, None).astype(mx.float32) -
                                 y.astype(mx.float32)))

    def held_out():
        student.update(as_tree(master))
        num = den = 0.0
        for h, y in zip(test_h, test_y):
            d = student(h, mask, None).astype(mx.float32) - y.astype(mx.float32)
            num += float(mx.sum(mx.square(d)))
            den += float(mx.sum(mx.square(y.astype(mx.float32))))
        return float(np.sqrt(num / den))

    print(f"held-out block error before training : {held_out():.4f}", flush=True)

    grad_fn = mx.value_and_grad(loss_fn)
    opt = optim.Adam(learning_rate=LR)
    rng = np.random.default_rng(0)
    started = time.time()

    for step in range(STEPS):
        i = int(rng.integers(len(train_h)))
        loss, grads = grad_fn(master, train_h[i], train_y[i])
        master = opt.apply_gradients(grads, master)
        mx.eval(master, loss)
        if (step + 1) % 50 == 0:
            print(f"step {step + 1:4}  loss {float(loss):.5f}  "
                  f"held-out {held_out():.4f}  ({time.time() - started:.0f}s)",
                  flush=True)

    # Keep the trained block so perplexity can be measured against it.
    student.update(as_tree(master))
    mx.eval(student.parameters())
    flat = {f"model.layers.{BLOCK}."
            f"{'self_attn' if n in ATTN else 'mlp'}.{n}.weight":
            quantize(w).astype(mx.bfloat16) for n, w in master.items()}
    mx.save_safetensors("/tmp/trained_block0.safetensors", flat)
    print("saved trained block", flush=True)


if __name__ == "__main__":
    main()
