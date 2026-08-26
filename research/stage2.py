"""End-to-end distillation, which per-block training cannot do.

Fitting each block against its own teacher leaves the error to accumulate: the
held-out block error still triples between depth 3 and depth 23 however well
each block is fitted alone. A loss on the logits fixes that, because the
gradient reaching a block already knows what the rest of the network will do
with its output.

Both models will not fit in sixteen gigabytes at once, so the teacher's
predictions are cached first — top-K per position, which is most of the signal
at a fraction of the size — and only the student is resident while training.
Blocks are trained in groups for the same reason: the optimiser state for all
196 matrices would not fit either.
"""

import pathlib
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load

GROUP = 128
SEQ = 512
WINDOWS = 2048
TOPK = 64
GROUP_SIZE = 7
PASSES = 2
STEPS = 900
LR_W = 1e-4
LR_S = 3e-3
CLIP = 1.0
CACHE = "/tmp/teacher_topk.npz"

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")


def binarise(w: mx.array, log_s: mx.array) -> mx.array:
    g = w.reshape(w.shape[0], -1, GROUP)
    b = mx.where(g >= 0, 1.0, -1.0)
    q = (b * mx.exp(log_s)[..., None]).reshape(w.shape)
    return q + (w - mx.stop_gradient(w))


def clip_grads(grads, limit=CLIP):
    flat = [g for _, g in tree_flatten(grads)]
    total = mx.sqrt(sum(mx.sum(mx.square(g)) for g in flat))
    factor = mx.minimum(1.0, limit / (total + 1e-6))
    return tree_map(lambda g: g * factor, grads)


def cache_teacher(corpus, starts):
    """Top-K teacher predictions, so the teacher need not stay resident."""
    if pathlib.Path(CACHE).exists():
        d = np.load(CACHE)
        return d["idx"], d["logp"]

    model, _ = load("Qwen/Qwen3-1.7B")
    idx = np.zeros((len(starts), SEQ, TOPK), dtype=np.uint32)
    logp = np.zeros((len(starts), SEQ, TOPK), dtype=np.float16)
    started = time.time()
    for i, s in enumerate(starts):
        ids = mx.array(corpus[s:s + SEQ][None, :].astype(np.int32))
        logits = model(ids).astype(mx.float32)[0]
        top = mx.argpartition(-logits, TOPK, axis=-1)[:, :TOPK]
        vals = mx.take_along_axis(logits, top, axis=-1)
        vals = vals - mx.logsumexp(logits, axis=-1, keepdims=True)
        mx.eval(top, vals)
        idx[i] = np.array(top).astype(np.uint32)
        logp[i] = np.array(vals).astype(np.float16)
        if (i + 1) % 256 == 0:
            print(f"  cached {i + 1}/{len(starts)} "
                  f"({time.time() - started:.0f}s)", flush=True)
    np.savez(CACHE, idx=idx, logp=logp)
    del model
    return idx, logp


def perplexity(model, tokens, limit=8):
    total = count = 0.0
    for i in range(limit):
        chunk = tokens[i * 1024:(i + 1) * 1024 + 1]
        logits = model(mx.array(chunk[None, :-1].astype(np.int32))).astype(mx.float32)
        loss = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                       mx.array(chunk[None, 1:].astype(np.int32)).reshape(-1),
                                       reduction="mean")
        mx.eval(loss)
        total += float(loss) * 1024
        count += 1024
    return float(np.exp(total / count))


def main() -> None:
    corpus = np.load("/tmp/corpus_tokens.npy")
    rng = np.random.default_rng(7)
    held_start = len(corpus) - 40 * 1024
    starts = rng.integers(0, held_start - SEQ, WINDOWS)

    print("caching teacher predictions", flush=True)
    t_idx, t_logp = cache_teacher(corpus, starts)
    held = corpus[held_start:]

    student, _ = load("Qwen/Qwen3-1.7B")
    student.load_weights("/tmp/trained2_model.safetensors", strict=False)
    master = mx.load("/tmp/trained2_master.safetensors")
    mx.eval(student.parameters())
    print(f"starting perplexity {perplexity(student, held):.3f}", flush=True)

    def state_for(block_ids):
        """Weights and scales as two trees, so each gets its own optimiser."""
        state = {"w": {}, "s": {}}
        for b in block_ids:
            tag = str(b)
            state["w"][tag] = {"self_attn": {}, "mlp": {}}
            state["s"][tag] = {"self_attn": {}, "mlp": {}}
            for parent, names in (("self_attn", ATTN), ("mlp", MLP)):
                for name in names:
                    key = f"model.layers.{b}.{parent}.{name}.weight"
                    state["w"][tag][parent][name] = master[key].astype(mx.float32)
                    state["s"][tag][parent][name] = master[key + ".log_scale"]
        return state

    def apply(state):
        for tag, blk in state["w"].items():
            tree = {p: {n: {"weight": binarise(
                w, state["s"][tag][p][n]).astype(mx.bfloat16)}
                for n, w in part.items()} for p, part in blk.items()}
            student.model.layers[int(tag)].update(tree)

    groups = [list(range(i, min(i + GROUP_SIZE, 28)))
              for i in range(0, 28, GROUP_SIZE)]
    started = time.time()

    for p in range(PASSES):
        for group in groups:
            state = state_for(group)

            def loss_fn(state, i):
                apply(state)
                ids = mx.array(corpus[starts[i]:starts[i] + SEQ][None, :].astype(np.int32))
                logits = student(ids).astype(mx.float32)[0]
                sel = mx.take_along_axis(
                    logits, mx.array(t_idx[i].astype(np.int32)), axis=-1)
                logq = sel - mx.logsumexp(logits, axis=-1, keepdims=True)
                tp = mx.array(t_logp[i].astype(np.float32))
                weight = mx.softmax(tp, axis=-1)
                return mx.sum(weight * (mx.log(weight + 1e-9) - logq), axis=-1).mean()

            grad_fn = mx.value_and_grad(loss_fn)
            opts = {"w": optim.Adam(learning_rate=optim.cosine_decay(LR_W, STEPS)),
                    "s": optim.Adam(learning_rate=optim.cosine_decay(LR_S, STEPS))}
            pick = np.random.default_rng(p * 100 + group[0])

            for step in range(STEPS):
                i = int(pick.integers(WINDOWS))
                loss, grads = grad_fn(state, i)
                grads = clip_grads(grads)
                state = {k: opts[k].apply_gradients(grads[k], state[k])
                         for k in state}
                mx.eval(state, loss)

            apply(state)
            mx.eval(student.parameters())
            for tag, blk in state["w"].items():
                for pn, part in blk.items():
                    for n, w in part.items():
                        key = f"model.layers.{tag}.{pn}.{n}.weight"
                        master[key] = w.astype(mx.bfloat16)
                        master[key + ".log_scale"] = state["s"][tag][pn][n]
            del state

            print(f"pass {p} blocks {group[0]}-{group[-1]}  "
                  f"ppl {perplexity(student, held):.3f}  "
                  f"({time.time() - started:.0f}s)", flush=True)

    flat = {}
    for b in range(28):
        for parent, names in (("self_attn", ATTN), ("mlp", MLP)):
            for name in names:
                key = f"model.layers.{b}.{parent}.{name}.weight"
                flat[key] = binarise(master[key].astype(mx.float32),
                                     master[key + ".log_scale"]).astype(mx.bfloat16)
    mx.save_safetensors("/tmp/stage2_model.safetensors", flat)
    mx.save_safetensors("/tmp/stage2_master.safetensors", master)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
