"""Prototype: reconstruct a binary weight matrix by minimising output error.

The question this answers: does plain layer-wise output reconstruction under a
binary constraint reproduce the fingerprint measured on Bonsai — roughly 70%
sign agreement, cosine near 0.5 against the base weights, and sign flips
concentrated on the smallest weights?

If it does, the published method needs nothing beyond the known OBS/GPTQ line.
If it does not, something else is going on.

The objective for one output channel is

    min ‖Xᵀ(s·b − w)‖²  =  (s·b − w)ᵀ H (s·b − w),   H = X Xᵀ

which is a binary quadratic problem, solved here by coordinate descent. All
output channels share the same H, so one coordinate can be swept across every
channel at once — that is what makes this tractable in numpy.
"""

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from bitprobe import metrics, remote  # noqa: E402

GROUP = 128
EPS = 1e-6


# ==============================
# ===  Calibration inputs    ===
# ==============================

def rms_norm(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """RMSNorm as used by Qwen, applied over the hidden dimension."""
    scale = np.sqrt((x.astype(np.float32) ** 2).mean(axis=-1, keepdims=True) + EPS)
    return (x / scale) * weight


def layer0_input(base: str, token_ids: np.ndarray) -> np.ndarray:
    """Activations entering the first attention block.

    The first block sees nothing but normalised embeddings, so its input is
    exact without running any transformer layer — which keeps this experiment
    free of a full forward implementation.
    """
    table = remote.load(base, "model.embed_tokens.weight")
    emb = table[token_ids]
    del table
    norm = remote.load(base, "model.layers.0.input_layernorm.weight")
    return rms_norm(emb, norm)


# ==============================
# ===  Binary reconstruction ===
# ==============================

def exact_group_scales(w: np.ndarray, b: np.ndarray, hessian: np.ndarray,
                       damping: float = 1e-3) -> np.ndarray:
    """Exact per-group scales for fixed signs.

    With signs held, the objective is quadratic in the scales, so the optimum
    solves A s = c with A[g,h] = B_g' H B_h and c[g] = B_g' H w, where B_g is
    the sign vector restricted to group g. Dropping the off-diagonal blocks of
    A — as the cheap approximation does — throws away the correlation between
    groups and produces scales that make the solution worse, not better.

    Sixteen groups per row means a 16x16 solve per output channel, which is
    negligible; the cost is building A from the Hessian blocks.
    """
    out, width = w.shape
    groups = width // GROUP

    bg = b.reshape(out, groups, GROUP)
    hblk = hessian.reshape(groups, GROUP, groups, GROUP).transpose(0, 2, 1, 3)

    # A[o,g,h] = bg[o,g] . hblk[g,h] . bg[o,h]
    partial = np.einsum("ogi,ghij->oghj", bg, hblk, optimize=True)
    a = np.einsum("oghj,ohj->ogh", partial, bg, optimize=True)

    hw = (w @ hessian).reshape(out, groups, GROUP)
    c = (bg * hw).sum(axis=2)

    # Damping keeps the solve stable when groups are nearly collinear.
    trace = np.einsum("ogg->o", a) / groups
    a = a + np.eye(groups, dtype=a.dtype) * (damping * trace)[:, None, None]

    scales = np.linalg.solve(a, c[:, :, None])[:, :, 0]
    return scales


def group_scales(w: np.ndarray, b: np.ndarray, hessian: np.ndarray) -> np.ndarray:
    """Scale per group that minimises output error, not weight error.

    Fitting the scale to the weights (mean of w*b) is wrong once signs have
    been flipped: flipped entries contribute negatively and drag the scale
    down, which then fights the sign search. The scale has to minimise the
    same quadratic the signs do, so it is solved against the Hessian —
    per group, ignoring cross-group terms, which keeps it vectorisable.
    """
    out, width = w.shape
    groups = width // GROUP
    hw = w @ hessian                                  # H is symmetric

    bg = b.reshape(out, groups, GROUP)
    num = (bg * hw.reshape(out, groups, GROUP)).sum(axis=2)

    den = np.empty((out, groups), dtype=np.float32)
    for g in range(groups):
        block = hessian[g * GROUP:(g + 1) * GROUP, g * GROUP:(g + 1) * GROUP]
        vg = bg[:, g, :]
        den[:, g] = np.einsum("oi,ij,oj->o", vg, block, vg, optimize=True)

    return np.maximum(num / np.maximum(den, 1e-8), 1e-8)


def expand(scales: np.ndarray, width: int) -> np.ndarray:
    return np.repeat(scales, GROUP, axis=1)[:, :width]


def objective(residual: np.ndarray, hessian: np.ndarray) -> float:
    """Total output error the descent is supposed to be driving down."""
    return float(np.einsum("oi,ij,oj->", residual, hessian, residual, optimize=True))


def reconstruct(w: np.ndarray, hessian: np.ndarray, sweeps: int = 3,
                update_scales: bool = True) -> np.ndarray:
    """Coordinate descent over signs, minimising output error under H."""
    out, width = w.shape
    b = np.sign(w)
    b[b == 0] = 1.0
    scales = np.abs(w.reshape(out, -1, GROUP)).mean(axis=2)
    if update_scales:
        scales = exact_group_scales(w, b, hessian)

    diag = np.diag(hessian).copy()
    residual = expand(scales, width) * b - w
    grad = residual @ hessian                      # (out, width)

    for sweep in range(sweeps):
        flipped = 0
        # Materialised once per sweep: rebuilding it per coordinate would cost
        # an (out x width) allocation on every one of `width` iterations.
        s_full = expand(scales, width)
        for i in range(width):
            s = s_full[:, i]
            delta = -4.0 * s * b[:, i] * grad[:, i] + 4.0 * (s ** 2) * diag[i]
            take = delta < 0
            if not take.any():
                continue
            step = np.where(take, -2.0 * s * b[:, i], 0.0)
            grad += np.outer(step, hessian[i])
            b[take, i] *= -1.0
            flipped += int(take.sum())

        if update_scales:
            candidate = exact_group_scales(w, b, hessian)
            trial = expand(candidate, width) * b - w
            # Accept only if the exact solve actually lowers the objective;
            # a damped solve can overshoot on ill-conditioned rows.
            if objective(trial, hessian) < objective(expand(scales, width) * b - w, hessian):
                scales = candidate
        residual = expand(scales, width) * b - w
        grad = residual @ hessian
        print(f"  sweep {sweep + 1}: {flipped:>9} flips  objective={objective(residual, hessian):.4e}")

    return expand(scales, width) * b


# ==============================
# ===  Entry point           ===
# ==============================

def main() -> None:
    base = "Qwen/Qwen3-1.7B"
    tensor = "model.layers.0.self_attn.q_proj.weight"
    # A few tens of thousands of tokens already give a well-conditioned
    # Hessian for a 2048-wide layer; more just costs memory.
    tokens = np.load("/tmp/calib_tokens.npy")[:32768]

    print(f"calibration tokens: {len(tokens)}")

    # The Hessian is 16 MB while the embedding table it comes from is 600 MB,
    # so it is worth keeping between runs.
    cache = pathlib.Path("/tmp/layer0_cache.npz")
    if cache.exists():
        blob = np.load(cache)
        hessian, x_ref, w = blob["hessian"], blob["x_ref"], blob["w"]
        print(f"loaded cached Hessian {hessian.shape} and weights {w.shape}")
    else:
        x = layer0_input(base, tokens)
        print(f"activations: {x.shape}")
        hessian = (x.T @ x) / len(x)
        hessian += np.eye(hessian.shape[0], dtype=np.float32) * (
            1e-2 * np.trace(hessian) / hessian.shape[0])
        w = remote.load(base, tensor)
        x_ref = x[:4096]          # kept for the output-error check
        np.savez(cache, hessian=hessian, x_ref=x_ref, w=w)
        del x
        print("cached Hessian for later runs")
    x = x_ref
    fixed = "--fixed-scale" in sys.argv
    print(f"weights: {w.shape}\nrunning coordinate descent "
          f"({'fixed' if fixed else 'fitted'} scales)...")
    sweeps = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--sweeps=")), 3))
    approx = reconstruct(w, hessian, sweeps=sweeps, update_scales=not fixed)

    np.save("/tmp/recon_q_proj.npy", approx)
    naive = metrics.naive_binary(w, GROUP)
    print("\n=== result vs base weights ===")
    for label, m in (("reconstructed", approx), ("naive binary", naive)):
        print(f"  {label:14} cos={metrics.cosine(m, w):+.4f}  "
              f"signs={metrics.sign_agreement(m, w) * 100:5.1f}%  "
              f"scale={metrics.scale_ratio(m, w, GROUP):.3f}")

    # The published model, measured on the same activations — without this the
    # comparison is only against a straw baseline.
    published = pathlib.Path("/tmp/bonsai_q_proj.npy")
    rows = [("reconstructed", approx), ("naive binary", naive)]
    if published.exists():
        bonsai = np.load(published)
        rows.append(("bonsai", bonsai))
        print(f"  {'bonsai':14} cos={metrics.cosine(bonsai, w):+.4f}  "
              f"signs={metrics.sign_agreement(bonsai, w) * 100:5.1f}%  "
              f"scale={metrics.scale_ratio(bonsai, w, GROUP):.3f}")

    print("\n=== output error on calibration activations ===")
    ref = x @ w.T
    for label, m in rows:
        err = np.linalg.norm(x @ m.T - ref) / np.linalg.norm(ref)
        print(f"  {label:14} relative output error = {err:.4f}")

    # Bonsai rescaled the norms and quantized the embedding table, so its matrix
    # is meant to consume its own activations. Feeding it Qwen's inputs
    # understates it; this block gives each matrix the inputs it was built for.
    bx = pathlib.Path("/tmp/bonsai_x.npy")
    if published.exists() and bx.exists():
        x_b = np.load(bx)
        n = min(len(x), len(x_b))
        ref = x[:n] @ w.T
        print("\n=== honest comparison: each matrix on its own activations ===")
        pairs = [
            ("reconstructed", x[:n], approx),
            ("naive binary", x[:n], naive),
            ("bonsai (own acts)", x_b[:n], np.load(published)),
            ("bonsai (qwen acts)", x[:n], np.load(published)),
        ]
        for label, inp, mat in pairs:
            err = np.linalg.norm(inp @ mat.T - ref) / np.linalg.norm(ref)
            print(f"  {label:20} relative output error = {err:.4f}")

    print("\n=== flips by magnitude (reconstructed) ===")
    for i, (lo, hi, rate) in enumerate(metrics.flip_by_magnitude(approx, w), 1):
        if i in (1, 5, 10):
            print(f"  decile {i:2}  |w| {lo:.4f}-{hi:.4f}  {rate * 100:5.1f}%")


if __name__ == "__main__":
    main()
