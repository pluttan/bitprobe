<div align="center">

# bitprobe

**See what a 1-bit model actually did to its weights**

</div>

Command-line tool that compares a low-bit quantized model against the full-precision model it was derived from. It reads tensors straight out of remote safetensors repositories over HTTP range requests, so neither model is ever downloaded in full — only the slices under study cross the wire. Alongside the raw comparison it computes what a naive sign quantizer would have produced, because the interesting question is not how far a quantized matrix sits from the original, but how that distance compares to the obvious baseline.

## ■ Features

- ❖ **Remote tensor access** — reads the safetensors header (a few kilobytes) and then fetches only the requested tensor or row slice by byte range; a 27B model can be probed from a laptop
- ❖ **Baseline comparison** — every measurement is reported next to naive per-group sign and ternary quantization, so a method that beats plain rounding is distinguishable from one that does not
- ❖ **Group detection** — infers the scaling group size from repeated magnitudes in a row, without reading any metadata
- ❖ **Sign-flip profile** — sign flip rate per magnitude decile of the base weights, showing whether small weights were traded away for accuracy elsewhere
- ❖ **Depth profile** — sweeps a tensor pattern across the layer stack and reports deviation at each depth
- ❖ **Correct BF16 handling** — base models ship BF16 while quantized ones ship F16; reading one as the other yields plausible-looking garbage, so both are decoded explicitly

## ■ Stack

<div align="center">

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| Numerics | NumPy 2 |
| Transport | urllib HTTP range requests |
| Model format | safetensors (HuggingFace) |
| Palette | Catppuccin Mocha |

</div>

## ■ How It Works

```
1. Fetch the safetensors header of both repositories — kilobytes, not gigabytes
2. Resolve the byte range of the requested tensor, or of its first N rows
3. Decode BF16 or F16 into float32
4. Quantize the base tensor naively to build a reference point
5. Report cosine similarity, sign agreement, relative error, scale ratio,
   zero fraction and the flip distribution across magnitude deciles
```

## ■ Usage

```bash
# Install
make install

# Storage structure of a quantized model: group size, dtype, sparsity
venv/bin/python3.12 main.py info prism-ml/Bonsai-1.7B-unpacked

# Compare one tensor against its full-precision base
venv/bin/python3.12 main.py compare \
    prism-ml/Bonsai-1.7B-unpacked Qwen/Qwen3-1.7B \
    model.layers.0.mlp.gate_proj.weight

# Sign flips by weight magnitude
venv/bin/python3.12 main.py flips \
    prism-ml/Bonsai-1.7B-unpacked Qwen/Qwen3-1.7B \
    model.layers.0.mlp.gate_proj.weight

# Deviation across the layer stack
venv/bin/python3.12 main.py profile \
    prism-ml/Bonsai-1.7B-unpacked Qwen/Qwen3-1.7B --limit 8

# All of the above
make demo
```

Sample output of `compare`:

```
model.layers.0.mlp.gate_proj.weight  [256, 2048]
  cos(model, base)        +0.4886
  cos(naive binary, base) +0.7909
  cos(naive ternary, base)+0.8926
  sign agreement          71.0%
  relative error          1.354
  naive relative error    0.612
  scale vs mean|w|        1.990
  zero fraction           0.0%
```

`--rows` controls how many rows are sampled per tensor (default 256, `0` for the whole
tensor) and `--group` sets the scaling group size (default 128).

## ■ License

MIT © [pluttan](https://github.com/pluttan)
