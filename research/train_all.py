"""Binarise the whole network by training the signs, block by block.

The solver lost to gradient descent on a single block by a factor of sixteen,
so the same treatment goes to all 28. Each block is trained on the hidden state
the already-binarised prefix actually produces, but against the target the
full-precision teacher produces from its own clean state — so a block learns to
undo the drift it inherits instead of merely copying its teacher.

Three guards, each earned: the loss is divided by the target's own energy, or a
single learning rate cannot serve blocks whose activations grow with depth; the
gradient norm is capped; and a block is only accepted if it beat the plain
binarisation it started from, measured on held-out windows.
"""

import time

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

GROUP = 128
SEQ = 1024
WINDOWS = 39
STEPS = 1200
LR = 3e-4
CLIP = 1.0
CHECK = 100

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")


def quantize(w: mx.array) -> mx.array:
    """Binary weights with a per-group scale, transparent to the gradient."""
    g = w.reshape(w.shape[0], -1, GROUP)
    s = mx.abs(g).mean(axis=2, keepdims=True)
    b = mx.where(g >= 0, 1.0, -1.0)
    return w + mx.stop_gradient((b * s).reshape(w.shape) - w)


def as_tree(master: dict) -> dict:
    tree = {"self_attn": {}, "mlp": {}}
    for name, w in master.items():
        parent = "self_attn" if name in ATTN else "mlp"
        tree[parent][name] = {"weight": quantize(w).astype(mx.bfloat16)}
    return tree


def clip_grads(grads: dict, limit: float = CLIP) -> dict:
    total = mx.sqrt(sum(mx.sum(mx.square(g)) for g in grads.values()))
    factor = mx.minimum(1.0, limit / (total + 1e-6))
    return {k: g * factor for k, g in grads.items()}


def main():
    tokens = np.load("/tmp/calib_tokens.npy")
    teacher_model, _ = load("Qwen/Qwen3-1.7B")
    student_model, _ = load("Qwen/Qwen3-1.7B")

    embed = teacher_model.model.embed_tokens
    train = [embed(mx.array(tokens[i * SEQ:(i + 1) * SEQ][None, :]))
             for i in range(WINDOWS)]
    test = [embed(mx.array(tokens[40000 + i * SEQ:40000 + (i + 1) * SEQ][None, :]))
            for i in range(6)]
    mask = create_attention_mask(train[0], None)

    # Teacher and student diverge from here on: the student's chain carries the
    # error of every block already binarised.
    t_train, t_test = list(train), list(test)
    s_train, s_test = list(train), list(test)
    started = time.time()
    saved = {}

    for idx in range(28):
        teacher = teacher_model.model.layers[idx]
        student = student_model.model.layers[idx]

        y_train = [mx.stop_gradient(teacher(h, mask, None)) for h in t_train]
        y_test = [mx.stop_gradient(teacher(h, mask, None)) for h in t_test]
        mx.eval(y_train, y_test)

        master = {}
        for parent, names in ((student.self_attn, ATTN), (student.mlp, MLP)):
            for name in names:
                master[name] = getattr(parent, name).weight.astype(mx.float32)

        def loss_fn(master, h, y):
            student.update(as_tree(master))
            out = student(h, mask, None).astype(mx.float32)
            y = y.astype(mx.float32)
            return mx.mean(mx.square(out - y)) / mx.mean(mx.square(y))

        def held_out(state):
            student.update(as_tree(state))
            num = den = 0.0
            for h, y in zip(s_test, y_test):
                d = student(h, mask, None).astype(mx.float32) - y.astype(mx.float32)
                num += float(mx.sum(mx.square(d)))
                den += float(mx.sum(mx.square(y.astype(mx.float32))))
            return float(np.sqrt(num / den))

        before = held_out(master)
        best, best_state = before, {k: v for k, v in master.items()}

        grad_fn = mx.value_and_grad(loss_fn)
        opt = optim.Adam(learning_rate=optim.cosine_decay(LR, STEPS))
        rng = np.random.default_rng(idx)

        for step in range(STEPS):
            i = int(rng.integers(len(s_train)))
            _, grads = grad_fn(master, s_train[i], y_train[i])
            master = opt.apply_gradients(clip_grads(grads), master)
            mx.eval(master)
            if (step + 1) % CHECK == 0:
                score = held_out(master)
                if score < best:
                    best, best_state = score, {k: v for k, v in master.items()}

        student.update(as_tree(best_state))
        mx.eval(student.parameters())

        for name, w in best_state.items():
            parent = "self_attn" if name in ATTN else "mlp"
            saved[f"model.layers.{idx}.{parent}.{name}.weight"] = \
                quantize(w).astype(mx.bfloat16)

        t_train = [teacher(h, mask, None) for h in t_train]
        t_test = [teacher(h, mask, None) for h in t_test]
        s_train = [student(h, mask, None) for h in s_train]
        s_test = [student(h, mask, None) for h in s_test]
        mx.eval(t_train, t_test, s_train, s_test)

        print(f"block {idx:2}  held-out {before:.4f} -> {best:.4f}  "
              f"({time.time() - started:.0f}s)", flush=True)

    mx.save_safetensors("/tmp/trained_model.safetensors", saved)
    print("saved trained weights", flush=True)


if __name__ == "__main__":
    main()
