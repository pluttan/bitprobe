"""Do trained signs move the way Bonsai's did?

Bonsai flips a third of the signs, and almost exclusively the small weights.
If short training reproduces that shape, the difference from Bonsai is training
budget rather than method.
"""

import mlx.core as mx
import numpy as np
from mlx_lm import load


def deciles(w, q):
    order = np.argsort(np.abs(w).ravel())
    flips = (np.sign(w) != np.sign(q)).ravel()[order]
    tenth = len(flips) // 10
    return [float(flips[i * tenth:(i + 1) * tenth].mean()) for i in range(10)]


def main():
    ref, _ = load("Qwen/Qwen3-1.7B")
    bonsai, _ = load("prism-ml/Bonsai-1.7B-unpacked")
    trained = mx.load("/tmp/trained_block0.safetensors")
    solver = mx.load("/tmp/quantized_model.safetensors")

    name, parent = "q_proj", "self_attn"
    key = f"model.layers.0.{parent}.{name}.weight"
    w = np.array(getattr(ref.model.layers[0], parent).__getattr__(name)
                 .weight.astype(mx.float32))
    variants = {
        "bonsai": np.array(getattr(bonsai.model.layers[0], parent)
                           .__getattr__(name).weight.astype(mx.float32)),
        "trained": np.array(trained[key].astype(mx.float32)),
        "solver": np.array(solver[key].astype(mx.float32)),
    }

    print("L0 q_proj — flipped signs per weight-magnitude decile (small to large)")
    for label, q in variants.items():
        d = deciles(w, q)
        print(f"{label:8} total {np.mean(d):.3f}  " +
              " ".join(f"{v:.3f}" for v in d), flush=True)


if __name__ == "__main__":
    main()
