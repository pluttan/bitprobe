"""Perplexity on held-out text — the metric that actually decides the question.

Layer-wise error says who approximates a matrix better. Perplexity says who
still has a working language model. They do not have to agree.
"""

import sys
import pathlib

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load

SEQ = 1024
STRIDE = 1024


def perplexity(model, tokens: np.ndarray, limit: int = 16) -> float:
    """Token-level perplexity over non-overlapping windows."""
    total, count = 0.0, 0
    for i in range(limit):
        chunk = tokens[i * STRIDE:i * STRIDE + SEQ + 1]
        if len(chunk) < SEQ + 1:
            break
        ids = mx.array(chunk[None, :-1])
        targets = mx.array(chunk[None, 1:])
        logits = model(ids).astype(mx.float32)
        loss = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                       targets.reshape(-1), reduction="mean")
        mx.eval(loss)
        total += float(loss) * (SEQ)
        count += SEQ
    return float(np.exp(total / count))


def main() -> None:
    # Held out: the calibration set used the head of this file.
    tokens = np.load("/tmp/calib_tokens.npy")[40000:]
    print(f"held-out tokens: {len(tokens)}", flush=True)

    targets = [
        ("Qwen3-1.7B (fp16)", "Qwen/Qwen3-1.7B", None),
        ("Bonsai 1-bit", "prism-ml/Bonsai-1.7B-unpacked", None),
        ("ours (1-bit)", "Qwen/Qwen3-1.7B", "/tmp/quantized_model.safetensors"),
    ]

    for label, repo, override in targets:
        path = pathlib.Path(override) if override else None
        if override and not path.exists():
            print(f"{label:22} skipped (no {override})", flush=True)
            continue
        model, _ = load(repo)
        if override:
            model.load_weights(str(path), strict=False)
            mx.eval(model.parameters())
        ppl = perplexity(model, tokens)
        print(f"{label:22} perplexity = {ppl:8.3f}", flush=True)
        del model


if __name__ == "__main__":
    main()
