"""How closely does Bonsai track Qwen's predictions?

Its hidden states diverge almost completely, so nothing inside was matched to
the original. What remains to check is the output: a model distilled from Qwen
reproduces its next-token distribution even while computing it differently. A
model merely trained on similar text does not.
"""

import mlx.core as mx
import numpy as np
from mlx_lm import load

SEQ = 1024
WINDOWS = 4


def softmax_np(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def main():
    tokens = np.load("/tmp/calib_tokens.npy")[40000:]
    ours, _ = load("Qwen/Qwen3-1.7B")
    bonsai, _ = load("prism-ml/Bonsai-1.7B-unpacked")

    e_ref = np.array(ours.model.embed_tokens.weight.astype(mx.float32))
    e_bon = np.array(bonsai.model.embed_tokens.weight.astype(mx.float32))
    # Bonsai trimmed the vocabulary, so only the shared prefix is comparable.
    shared = min(len(e_ref), len(e_bon))
    print(f"vocab  qwen {len(e_ref)}  bonsai {len(e_bon)}")
    rows = np.random.default_rng(0).choice(shared, 4096, replace=False)
    per_row = (e_ref[rows] * e_bon[rows]).sum(1) / (
        np.linalg.norm(e_ref[rows], axis=1) * np.linalg.norm(e_bon[rows], axis=1))
    print(f"embed_tokens  per-row cos  mean {per_row.mean():.4f}  "
          f"min {per_row.min():.4f}  max {per_row.max():.4f}")
    print(f"embed_tokens  distinct magnitudes in first group: "
          f"{len(np.unique(np.abs(e_bon[0, :128])))}")

    top1, kl, n = 0.0, 0.0, 0
    for i in range(WINDOWS):
        chunk = tokens[i * SEQ:(i + 1) * SEQ]
        ids = mx.array(chunk[None, :])
        lr = np.array(ours(ids).astype(mx.float32))[0][:, :shared]
        lb = np.array(bonsai(ids).astype(mx.float32))[0][:, :shared]
        pr, pb = softmax_np(lr), softmax_np(lb)
        top1 += float((lr.argmax(-1) == lb.argmax(-1)).sum())
        kl += float((pr * (np.log(pr + 1e-12) - np.log(pb + 1e-12))).sum())
        n += len(chunk)

    print(f"\ntop-1 agreement with Qwen : {top1 / n:.4f}")
    print(f"mean KL(Qwen || Bonsai)   : {kl / n:.4f}")


if __name__ == "__main__":
    main()
