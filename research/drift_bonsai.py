"""What does Bonsai preserve, if not the weights?

Its matrices are poor approximations of Qwen's — a third of the signs differ —
yet the model works. If the binary network was trained to imitate the original,
the agreement should show up in the hidden states rather than in the weights.
Anything that stays flat with depth was matched on purpose; anything that grows
is error the training merely tolerated.
"""

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

SEQ = 512


def to_np(a):
    return np.array(a.astype(mx.float32), copy=True)


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def cos(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    tokens = np.load("/tmp/calib_tokens.npy")[40000:40000 + SEQ]
    ids = mx.array(tokens[None, :])

    ours, _ = load("Qwen/Qwen3-1.7B")
    bonsai, _ = load("prism-ml/Bonsai-1.7B-unpacked")

    e_ref = ours.model.embed_tokens(ids)
    e_bon = bonsai.model.embed_tokens(ids)
    print(f"embeddings   rel {rel(to_np(e_bon), to_np(e_ref)):.4f}  "
          f"cos {cos(to_np(e_bon), to_np(e_ref)):.4f}")

    mask = create_attention_mask(e_ref, None)
    h_ref, h_bon = e_ref, e_bon
    print(f"\n{'block':>5} {'rel':>8} {'cos':>8} {'|h|ratio':>9}")
    for idx, (rb, bb) in enumerate(zip(ours.model.layers, bonsai.model.layers)):
        h_ref = rb(h_ref, mask, None)
        h_bon = bb(h_bon, mask, None)
        mx.eval(h_ref, h_bon)
        a, b = to_np(h_bon), to_np(h_ref)
        if idx in (0, 1, 2, 4, 8, 12, 16, 20, 24, 26, 27):
            print(f"{idx:>5} {rel(a, b):8.4f} {cos(a, b):8.4f} "
                  f"{np.linalg.norm(a) / np.linalg.norm(b):9.4f}", flush=True)


if __name__ == "__main__":
    main()
