"""Which lever matters: learnable scales, or more data?

Stage one trained signs only, with the group scale pinned to the mean magnitude
of the weights it quantises. Bonsai's scales run about twice that, so the scale
is evidently not a by-product there. And forty thousand tokens seen twelve
hundred times is a memorised set, not a corpus. Both are tested here on one
block before spending two hours on all twenty-eight.
"""

import sys
import time

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

GROUP = 128
SEQ = 1024
STEPS = 800
LR_W = 3e-4
LR_S = 1e-2
CLIP = 1.0
BLOCK = 0

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")


def binarise(w: mx.array, log_s: mx.array | None) -> mx.array:
    """Signs from the master weight, scale either fixed or learned."""
    g = w.reshape(w.shape[0], -1, GROUP)
    b = mx.where(g >= 0, 1.0, -1.0)
    if log_s is None:
        s = mx.stop_gradient(mx.abs(g).mean(axis=2, keepdims=True))
        q = (b * s).reshape(w.shape)
        return w + mx.stop_gradient(q - w)
    q = (b * mx.exp(log_s)[..., None]).reshape(w.shape)
    # Straight-through for the sign, a real gradient for the scale.
    return q + (w - mx.stop_gradient(w))


def as_tree(master: dict, scales: dict | None) -> dict:
    tree = {"self_attn": {}, "mlp": {}}
    for name, w in master.items():
        parent = "self_attn" if name in ATTN else "mlp"
        log_s = None if scales is None else scales[name]
        tree[parent][name] = {"weight": binarise(w, log_s).astype(mx.bfloat16)}
    return tree


def clip_grads(grads: dict, limit: float = CLIP) -> dict:
    flat = [g for v in grads.values()
            for g in (v.values() if isinstance(v, dict) else [v])]
    total = mx.sqrt(sum(mx.sum(mx.square(g)) for g in flat))
    factor = mx.minimum(1.0, limit / (total + 1e-6))

    def scale(v):
        return {k: g * factor for k, g in v.items()} if isinstance(v, dict) else v * factor

    return {k: scale(v) for k, v in grads.items()}


def windows(model, tokens, offset, count):
    return [model.model.embed_tokens(
        mx.array(tokens[offset + i * SEQ:offset + (i + 1) * SEQ][None, :].astype(np.int32)))
        for i in range(count)]


def run(label, teacher_model, student_model, train_tok, test_tok,
        learn_scales, train_windows, freeze_signs=False):
    teacher = teacher_model.model.layers[BLOCK]
    student = student_model.model.layers[BLOCK]

    # Each variant starts from the untouched weights: the student block is a
    # live object, and without this every run would resume the previous one.
    student.update({parent: {name: {"weight": getattr(
        getattr(teacher, parent), name).weight}
        for name in names}
        for parent, names in (("self_attn", ATTN), ("mlp", MLP))})
    mx.eval(student.parameters())

    train = windows(teacher_model, train_tok, 0, train_windows)
    test = windows(teacher_model, test_tok, 0, 6)
    mask = create_attention_mask(train[0], None)
    y_train = [mx.stop_gradient(teacher(h, mask, None)) for h in train]
    y_test = [mx.stop_gradient(teacher(h, mask, None)) for h in test]
    mx.eval(y_train, y_test)

    master, scales = {}, {}
    for parent, names in ((student.self_attn, ATTN), (student.mlp, MLP)):
        for name in names:
            w = getattr(parent, name).weight.astype(mx.float32)
            master[name] = w
            g = w.reshape(w.shape[0], -1, GROUP)
            scales[name] = mx.log(mx.abs(g).mean(axis=2) + 1e-8)
    state = {"w": master, "s": scales} if learn_scales else {"w": master}

    def trees(state):
        return as_tree(state["w"], state.get("s"))

    def loss_fn(state, h, y):
        student.update(trees(state))
        out = student(h, mask, None).astype(mx.float32)
        y = y.astype(mx.float32)
        return mx.mean(mx.square(out - y)) / mx.mean(mx.square(y))

    def held_out(state):
        student.update(trees(state))
        num = den = 0.0
        for h, y in zip(test, y_test):
            d = student(h, mask, None).astype(mx.float32) - y.astype(mx.float32)
            num += float(mx.sum(mx.square(d)))
            den += float(mx.sum(mx.square(y.astype(mx.float32))))
        return float(np.sqrt(num / den))

    before = held_out(state)
    grad_fn = mx.value_and_grad(loss_fn)
    opts = {"w": optim.Adam(learning_rate=optim.cosine_decay(LR_W, STEPS))}
    if learn_scales:
        opts["s"] = optim.Adam(learning_rate=optim.cosine_decay(LR_S, STEPS))

    rng = np.random.default_rng(0)
    best = before
    started = time.time()
    for step in range(STEPS):
        i = int(rng.integers(len(train)))
        _, grads = grad_fn(state, train[i], y_train[i])
        grads = clip_grads(grads)
        if freeze_signs:
            grads["w"] = {k: mx.zeros_like(v) for k, v in grads["w"].items()}
        state = {k: opts[k].apply_gradients(grads[k], state[k]) for k in state}
        mx.eval(state)
        if (step + 1) % 100 == 0:
            best = min(best, held_out(state))

    print(f"{label:28} {before:8.4f} -> {best:8.4f}  ({time.time() - started:.0f}s)",
          flush=True)


def main():
    small = np.load("/tmp/calib_tokens.npy")
    big = np.load("/tmp/corpus_tokens.npy")
    teacher_model, _ = load("Qwen/Qwen3-1.7B")
    student_model, _ = load("Qwen/Qwen3-1.7B")

    print(f"{'variant':28} {'before':>8}    {'after':>8}")
    held = big[-20 * SEQ:]
    run("signs only, 40k tokens", teacher_model, student_model,
        small, held, False, 39)
    run("signs only, big corpus", teacher_model, student_model,
        big, held, False, 512)
    run("signs + scales, big corpus", teacher_model, student_model,
        big, held, True, 512)
    run("scales only, big corpus", teacher_model, student_model,
        big, held, True, 512, freeze_signs=True)


if __name__ == "__main__":
    main()
