"""Build the layer-0 cache (Hessian + weights) on a machine with a stable link."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from bitprobe import remote  # noqa: E402
from reconstruct import layer0_input  # noqa: E402

BASE = "Qwen/Qwen3-1.7B"
TENSOR = "model.layers.0.self_attn.q_proj.weight"

tokens = np.load("/tmp/calib_tokens.npy")[:32768]
print(f"tokens {len(tokens)}", flush=True)

x = layer0_input(BASE, tokens)
print(f"activations {x.shape}", flush=True)

hessian = (x.T @ x) / len(x)
hessian += np.eye(hessian.shape[0], dtype=np.float32) * (
    1e-2 * np.trace(hessian) / hessian.shape[0])

w = remote.load(BASE, TENSOR)
np.savez("/tmp/layer0_cache.npz", hessian=hessian, x_ref=x[:4096], w=w)
print(f"cached: hessian {hessian.shape}, w {w.shape}", flush=True)
