"""Were Bonsai's embeddings binarised, or retrained?

Their per-row cosine against Qwen sits at 0.798 — what plain sign quantisation
of a roughly Gaussian matrix gives. If the signs also match exactly, the
embedding table was quantised trivially and the base model is Qwen3-1.7B, which
in turn dates every flipped sign in the transformer blocks to training.
"""

import mlx.core as mx
import numpy as np
from mlx_lm import load

GROUP = 128


def main():
    ours, _ = load("Qwen/Qwen3-1.7B")
    bonsai, _ = load("prism-ml/Bonsai-1.7B-unpacked")

    w = np.array(ours.model.embed_tokens.weight.astype(mx.float32))
    b = np.array(bonsai.model.embed_tokens.weight.astype(mx.float32))
    shared = min(len(w), len(b))
    w, b = w[:shared], b[:shared]

    print(f"sign agreement          : {(np.sign(w) == np.sign(b)).mean():.4f}")

    scale = np.abs(b.reshape(shared, -1, GROUP)).mean(axis=2)
    naive = np.abs(w.reshape(shared, -1, GROUP)).mean(axis=2)
    print(f"scale  bonsai / mean|w| : {np.median(scale / naive):.4f}")

    err = np.linalg.norm(b - np.sign(w) * np.repeat(scale, GROUP, axis=1))
    print(f"residual vs sign(w)*s   : {err / np.linalg.norm(b):.6f}")

    # Same question for the blocks, split by weight magnitude.
    wq = np.array(ours.model.layers[0].self_attn.q_proj.weight.astype(mx.float32))
    bq = np.array(bonsai.model.layers[0].self_attn.q_proj.weight.astype(mx.float32))
    order = np.argsort(np.abs(wq).ravel())
    flips = (np.sign(wq) != np.sign(bq)).ravel()[order]
    tenth = len(flips) // 10
    print("\nL0 q_proj sign flips by weight magnitude (smallest to largest decile):")
    print("  " + "  ".join(f"{flips[i * tenth:(i + 1) * tenth].mean():.3f}"
                           for i in range(10)))


if __name__ == "__main__":
    main()
