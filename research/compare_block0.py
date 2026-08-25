"""Three ways to make block 0 binary, scored on the whole model.

Naive quantisation, the Hessian solver, and gradient-trained signs — each
inserted into an otherwise full-precision network. Block-level error says which
one reproduces the block; perplexity says which one leaves a usable model.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

GROUP = 128
SEQ = 1024
BLOCK = 0

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")


def naive(w: np.ndarray) -> np.ndarray:
    g = w.reshape(w.shape[0], -1, GROUP)
    b = np.sign(g)
    b[b == 0] = 1.0
    return (b * np.abs(g).mean(axis=2, keepdims=True)).reshape(w.shape)


def perplexity(model, tokens, limit=8):
    total, count = 0.0, 0
    for i in range(limit):
        chunk = tokens[i * SEQ:(i + 1) * SEQ + 1]
        logits = model(mx.array(chunk[None, :-1])).astype(mx.float32)
        loss = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                       mx.array(chunk[None, 1:]).reshape(-1),
                                       reduction="mean")
        mx.eval(loss)
        total += float(loss) * SEQ
        count += SEQ
    return float(np.exp(total / count))


def block_error(model, teacher, tokens):
    hs = [model.model.embed_tokens(mx.array(tokens[i * SEQ:(i + 1) * SEQ][None, :]))
          for i in range(8)]
    mask = create_attention_mask(hs[0], None)
    num = den = 0.0
    for h in hs:
        y = teacher(h, mask, None).astype(mx.float32)
        d = model.model.layers[BLOCK](h, mask, None).astype(mx.float32) - y
        num += float(mx.sum(mx.square(d)))
        den += float(mx.sum(mx.square(y)))
    return float(np.sqrt(num / den))


def main():
    held = np.load("/tmp/calib_tokens.npy")[40000:]
    ref, _ = load("Qwen/Qwen3-1.7B")
    teacher = ref.model.layers[BLOCK]

    base = dict(tree_flatten(ref.parameters()))
    solver = mx.load("/tmp/quantized_model.safetensors")
    trained = mx.load("/tmp/trained_block0.safetensors")
    prefix = f"model.layers.{BLOCK}."

    variants = {
        # Binarise the original weights — quantising the solver's output
        # would just reproduce it, since it is already binary.
        "naive": {k: mx.array(naive(np.array(base[k].astype(mx.float32))))
                  for k in solver
                  if k.startswith(prefix) and k.endswith("proj.weight")},
        "solver": {k: v for k, v in solver.items()
                   if k.startswith(prefix) and k.endswith("proj.weight")},
        "trained": dict(trained),
    }

    print(f"{'variant':10} {'block err':>10} {'ppl':>12}")
    print(f"{'fp16':10} {0.0:10.4f} {perplexity(ref, held):12.3f}", flush=True)

    for label, weights in variants.items():
        model, _ = load("Qwen/Qwen3-1.7B")
        model.load_weights([(k, v.astype(mx.bfloat16)) for k, v in weights.items()],
                           strict=False)
        mx.eval(model.parameters())
        print(f"{label:10} {block_error(model, teacher, held):10.4f} "
              f"{perplexity(model, held):12.3f}", flush=True)
        del model


if __name__ == "__main__":
    main()
