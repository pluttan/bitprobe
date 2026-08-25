"""The four-way comparison the whole exercise was aimed at."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load

SEQ = 1024


def perplexity(model, tokens, limit=16):
    total, count = 0.0, 0
    for i in range(limit):
        chunk = tokens[i * SEQ:(i + 1) * SEQ + 1]
        if len(chunk) < SEQ + 1:
            break
        logits = model(mx.array(chunk[None, :-1])).astype(mx.float32)
        loss = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                       mx.array(chunk[None, 1:]).reshape(-1),
                                       reduction="mean")
        mx.eval(loss)
        total += float(loss) * SEQ
        count += SEQ
    return float(np.exp(total / count))


def main():
    held = np.load("/tmp/calib_tokens.npy")[40000:]

    model, _ = load("Qwen/Qwen3-1.7B")
    print(f"{'Qwen3-1.7B (fp16)':24} {perplexity(model, held):12.3f}", flush=True)
    del model

    model, _ = load("prism-ml/Bonsai-1.7B-unpacked")
    print(f"{'Bonsai 1-bit':24} {perplexity(model, held):12.3f}", flush=True)
    del model

    for label, path in (("ours: solver", "/tmp/quantized_model.safetensors"),
                        ("ours: trained", "/tmp/trained_model.safetensors"),
                        ("ours: trained v2", "/tmp/trained2_model.safetensors")):
        model, _ = load("Qwen/Qwen3-1.7B")
        model.load_weights(path, strict=False)
        mx.eval(model.parameters())
        print(f"{label:24} {perplexity(model, held):12.3f}", flush=True)
        del model


if __name__ == "__main__":
    main()
