"""Second pass over the whole network, with what the ablation showed matters.

Three changes from the first pass. The corpus is forty million tokens instead
of forty thousand, and every block draws its own fresh windows, so no block is
fitted to text another block already memorised. The group scale is learned
rather than pinned to the mean magnitude — Bonsai's scales run about twice the
naive value, which is not something a formula produces. And the master weights
are kept, so a later end-to-end stage can resume from here rather than from
weights already collapsed onto the quantisation grid.
"""

import time

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

GROUP = 128
SEQ = 512
TRAIN_WINDOWS = 192
TEST_WINDOWS = 24
STEPS = 1500
LR_W = 3e-4
LR_S = 1e-2
CLIP = 1.0
CHECK = 150

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")


def binarise(w: mx.array, log_s: mx.array) -> mx.array:
    """Signs from the master weight, scale learned; straight-through for signs."""
    g = w.reshape(w.shape[0], -1, GROUP)
    b = mx.where(g >= 0, 1.0, -1.0)
    q = (b * mx.exp(log_s)[..., None]).reshape(w.shape)
    return q + (w - mx.stop_gradient(w))


def as_tree(state: dict) -> dict:
    tree = {"self_attn": {}, "mlp": {}}
    for name, w in state["w"].items():
        parent = "self_attn" if name in ATTN else "mlp"
        tree[parent][name] = {"weight": binarise(w, state["s"][name]).astype(mx.bfloat16)}
    return tree


def clip_grads(grads: dict, limit: float = CLIP) -> dict:
    flat = [g for v in grads.values() for g in v.values()]
    total = mx.sqrt(sum(mx.sum(mx.square(g)) for g in flat))
    factor = mx.minimum(1.0, limit / (total + 1e-6))
    return {k: {n: g * factor for n, g in v.items()} for k, v in grads.items()}


def main() -> None:
    corpus = np.load("/tmp/corpus_tokens.npy")
    teacher_model, _ = load("Qwen/Qwen3-1.7B")
    student_model, _ = load("Qwen/Qwen3-1.7B")
    embed = teacher_model.model.embed_tokens

    rng = np.random.default_rng(0)
    held_start = len(corpus) - (TEST_WINDOWS + 1) * SEQ

    def windows(starts):
        return [embed(mx.array(corpus[s:s + SEQ][None, :].astype(np.int32)))
                for s in starts]

    test_starts = [held_start + i * SEQ for i in range(TEST_WINDOWS)]
    mask = create_attention_mask(windows(test_starts[:1])[0], None)

    started = time.time()
    saved, masters = {}, {}

    for idx in range(28):
        teacher = teacher_model.model.layers[idx]
        student = student_model.model.layers[idx]

        # Fresh text for this block, pushed through everything already binarised.
        starts = rng.integers(0, held_start - SEQ, TRAIN_WINDOWS)
        t_train, s_train = windows(starts), windows(starts)
        t_test, s_test = windows(test_starts), windows(test_starts)
        for prev in range(idx):
            t_prev = teacher_model.model.layers[prev]
            s_prev = student_model.model.layers[prev]
            t_train = [t_prev(h, mask, None) for h in t_train]
            t_test = [t_prev(h, mask, None) for h in t_test]
            s_train = [s_prev(h, mask, None) for h in s_train]
            s_test = [s_prev(h, mask, None) for h in s_test]
            mx.eval(t_train, t_test, s_train, s_test)

        y_train = [mx.stop_gradient(teacher(h, mask, None)) for h in t_train]
        y_test = [mx.stop_gradient(teacher(h, mask, None)) for h in t_test]
        mx.eval(y_train, y_test)
        del t_train, t_test

        state = {"w": {}, "s": {}}
        for parent, names in ((student.self_attn, ATTN), (student.mlp, MLP)):
            for name in names:
                w = getattr(parent, name).weight.astype(mx.float32)
                state["w"][name] = w
                g = w.reshape(w.shape[0], -1, GROUP)
                state["s"][name] = mx.log(mx.abs(g).mean(axis=2) + 1e-8)

        def loss_fn(state, h, y):
            student.update(as_tree(state))
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

        before = held_out(state)
        best, best_state = before, {k: dict(v) for k, v in state.items()}

        grad_fn = mx.value_and_grad(loss_fn)
        opts = {"w": optim.Adam(learning_rate=optim.cosine_decay(LR_W, STEPS)),
                "s": optim.Adam(learning_rate=optim.cosine_decay(LR_S, STEPS))}
        pick = np.random.default_rng(idx + 1)

        for step in range(STEPS):
            i = int(pick.integers(len(s_train)))
            _, grads = grad_fn(state, s_train[i], y_train[i])
            grads = clip_grads(grads)
            state = {k: opts[k].apply_gradients(grads[k], state[k]) for k in state}
            mx.eval(state)
            if (step + 1) % CHECK == 0:
                score = held_out(state)
                if score < best:
                    best, best_state = score, {k: dict(v) for k, v in state.items()}

        student.update(as_tree(best_state))
        mx.eval(student.parameters())

        for name, w in best_state["w"].items():
            parent = "self_attn" if name in ATTN else "mlp"
            key = f"model.layers.{idx}.{parent}.{name}.weight"
            saved[key] = binarise(w, best_state["s"][name]).astype(mx.bfloat16)
            masters[key] = w.astype(mx.bfloat16)
            masters[key + ".log_scale"] = best_state["s"][name].astype(mx.float32)

        del s_train, s_test, y_train, y_test, state, best_state
        print(f"block {idx:2}  held-out {before:.4f} -> {best:.4f}  "
              f"({time.time() - started:.0f}s)", flush=True)

    mx.save_safetensors("/tmp/trained2_model.safetensors", saved)
    mx.save_safetensors("/tmp/trained2_master.safetensors", masters)
    print("saved trained weights", flush=True)


if __name__ == "__main__":
    main()
