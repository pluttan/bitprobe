# What Bonsai actually is

Measured against `prism-ml/Bonsai-1.7B-unpacked` and `Qwen/Qwen3-1.7B`,
on 16 held-out windows of 1024 tokens.

## Perplexity

| model | perplexity |
|---|---|
| Qwen3-1.7B, fp16 | 12.32 |
| Bonsai, 1 bit | 16.53 |
| ours, Hessian solver | 923707.44 |
| ours, trained signs | 7044.33 |

## The solver was solving the wrong problem

Layer-wise reconstruction wins on its own metric and destroys the model. On
block 0 the solver reaches an output error of 0.12 where naive binarisation
reaches 0.86 — but held out, the same weights score 0.31, and the numbers only
get worse deeper in the matrix:

| layer | calibration | held out | Hessian condition |
|---|---|---|---|
| q_proj | 0.1207 | 0.3142 | 4.7e12 |
| v_proj | 0.2429 | 0.5843 | 4.7e12 |
| up_proj | 0.2653 | 0.5726 | 1.6e13 |
| down_proj | 0.2144 | 0.6957 | 4.5e07 |

Eight thousand calibration tokens cannot pin down a 2048- or 6144-dimensional
Hessian. The solver moves the weights into its near-null space, where the
objective charges nothing and real text charges everything. Damping at 1% of the
mean eigenvalue does not help, because a handful of outlier features carry the
trace.

## Bonsai is trained, not quantised

Every one of its 196 matrices is genuinely one bit — a single magnitude per
group of 128, no zeros, nothing left in higher precision. The embedding table is
binary too. But:

- Its embeddings are the *trivial* quantisation of Qwen's: signs agree 99.93% of
  the time and the scale is exactly `mean|w|`. So the base model is Qwen3-1.7B,
  and everything that moved, moved on purpose.
- In the transformer blocks 21–36% of the signs disagree with Qwen's, and the
  disagreement is graded by weight magnitude: 48% of the smallest decile, 5.6%
  of the largest. That is what gradient descent from a sign initialisation
  leaves behind — small weights cross zero easily, large ones do not.
- Hidden states diverge almost completely (cosine 0.10 at block 2) while the
  output still agrees: top-1 next token matches Qwen 70% of the time, mean KL
  0.55. Nothing inside was matched to the original; only the predictions were.
- Its config carries YaRN scaling (8192 to 32768) and a vocabulary trimmed to
  151669 tokens. Neither exists in Qwen3-1.7B.

No post-training method reproduces this. The matrices are not approximations of
Qwen's matrices — they are a different network computing the same function.

## Training the signs instead of solving for them

Straight-through gradients on the same binary parameterisation, one block at a
time, each block trained on the hidden state the already-binarised prefix
produces but against the teacher's clean output:

| block 0 | block error | model perplexity |
|---|---|---|
| naive | 0.8609 | 484621.33 |
| solver | 0.7159 | 5524.88 |
| trained, 46 seconds | 0.5570 | 345.27 |

The flip profile matches Bonsai's shape — trained signs move the small weights
(0.44 of the smallest decile) and leave the large ones alone (0.002), where the
solver flipped large weights nearly as often as small ones.

Across all 28 blocks the held-out block error falls from 0.86 to 0.46 at the
front, but the inherited drift dominates from block 3 onward and grows to 0.64
by the end. Perplexity lands at 7044 — 131x better than the solver, still 427x
short of Bonsai.

## Second pass: corpus and learned scales

An ablation on block 0, each variant starting from the untouched weights and
scored on the same held-out text:

| variant | block error |
|---|---|
| naive binarisation | 0.8690 |
| signs, 40k tokens | 0.5042 |
| signs, 40M tokens | 0.3883 |
| signs and scales, 40M tokens | 0.3652 |
| scales only, 40M tokens | 0.5403 |

Data is the larger lever, but the scale is not free either — which fits Bonsai's
scales running at twice the naive value. Retraining all 28 blocks with both, and
with fresh windows drawn per block, roughly halves the block error at every
depth (0.2007 at block 3 against 0.4316; 0.4606 at block 27 against 0.6379) and
brings perplexity to 1679.58.

| model | perplexity |
|---|---|
| Qwen3-1.7B, fp16 | 12.32 |
| Bonsai, 1 bit | 16.53 |
| ours, solver | 923707.44 |
| ours, trained signs | 7044.33 |
| ours, corpus and learned scales | 1679.58 |

## What closes the rest of the gap

Per-block distillation cannot: each block only compensates the drift it
inherits, and the block error still climbs from 0.20 at depth 3 to 0.52 at depth
23 no matter how well each block is fitted in isolation.

What Bonsai did, and what remains to do, is end-to-end distillation against the
teacher's logits over a real corpus. For a 1.7B model that is roughly
8 FLOPs per parameter per token — around 1.4e20 FLOPs for 10B tokens, or about a
day on twelve A100s at 40% utilisation, and a few days for the 30–50B tokens the
last of the gap would need.
