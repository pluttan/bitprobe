"""Where does the fully quantized model fall apart?

Layer-wise errors stayed at 0.09-0.15, yet perplexity exploded. Applying the
quantized weights to a growing prefix of blocks separates a gradual build-up
of error from a single block that destroys the network.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load

SEQ = 1024


def perplexity(model, tokens, limit=8):
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
    tokens = np.load("/tmp/calib_tokens.npy")[40000:]
    saved = mx.load("/tmp/quantized_model.safetensors")
    model, _ = load("Qwen/Qwen3-1.7B")

    print(f"{0:>3} blocks  ppl = {perplexity(model, tokens):10.3f}", flush=True)

    for idx in range(28):
        prefix = f"model.layers.{idx}."
        model.load_weights([(k, v) for k, v in saved.items()
                            if k.startswith(prefix)], strict=False)
        mx.eval(model.parameters())
        if idx + 1 in (1, 2, 4, 8, 14, 20, 24, 26, 27, 28):
            print(f"{idx + 1:>3} blocks  ppl = {perplexity(model, tokens):10.3f}",
                  flush=True)


if __name__ == "__main__":
    main()
