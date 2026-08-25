"""Comparison metrics for low-bit weight matrices.

Every metric here answers one question: how does a quantized matrix relate to
the full-precision matrix it came from? Distance alone is not the interesting
part — a method that preserves network behaviour may land far from the
original weights, so the baselines matter as much as the measurements.
"""

import numpy as np


# ==============================
# ===  Similarity            ===
# ==============================

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two matrices flattened as vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float((a * b).sum() / (na * nb))


def sign_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of entries whose signs match."""
    return float((np.sign(a) == np.sign(b)).mean())


def relative_error(approx: np.ndarray, target: np.ndarray) -> float:
    """Frobenius error of an approximation, relative to the target norm."""
    return float(np.linalg.norm(approx - target) / np.linalg.norm(target))


# ==============================
# ===  Baselines             ===
# ==============================

def naive_binary(w: np.ndarray, group: int = 128) -> np.ndarray:
    """Sign quantization with a per-group mean-absolute scale.

    This is what a straightforward quantizer produces, and it is the honest
    baseline: any method claiming to do better should be compared against it
    rather than against the unquantized matrix.
    """
    g = w.reshape(w.shape[0], -1, group)
    scale = np.abs(g).mean(axis=2, keepdims=True)
    return (np.sign(g) * scale).reshape(w.shape)


def naive_ternary(w: np.ndarray, group: int = 128, zeros: float = 0.4) -> np.ndarray:
    """Ternary quantization that zeroes the smallest weights in each group."""
    g = w.reshape(w.shape[0], -1, group)
    scale = np.abs(g).mean(axis=2, keepdims=True)
    rank = np.argsort(np.argsort(np.abs(g), axis=2), axis=2)
    keep = rank >= int(group * zeros)
    return (np.sign(g) * keep * scale).reshape(w.shape)


# ==============================
# ===  Structure probes      ===
# ==============================

def detect_group(w: np.ndarray, limit: int = 4096) -> int | None:
    """Infer the scaling group size from repeated magnitudes in a row.

    A quantized row holds one magnitude per group, so the distance between
    magnitude changes reveals the group size without reading any metadata.
    """
    row = np.abs(w[0][:limit])
    changes = np.flatnonzero(np.diff(row) != 0) + 1
    if len(changes) < 2:
        return None
    gaps = np.diff(np.r_[0, changes])
    first = int(gaps[0])
    return first if np.all(gaps[:-1] == first) else None


def zero_fraction(w: np.ndarray) -> float:
    """Fraction of exactly-zero entries, i.e. the ternary sparsity."""
    return float((w == 0).mean())


def flip_by_magnitude(
    quant: np.ndarray, base: np.ndarray, bins: int = 10
) -> list[tuple[float, float, float]]:
    """Sign-flip rate per magnitude decile of the base weights.

    A method that merely rounds flips nothing. A method that trades small
    weights for accuracy elsewhere flips the small ones and spares the large,
    which shows up here as a steep gradient.
    """
    mag = np.abs(base).ravel()
    flipped = (np.sign(quant) != np.sign(base)).ravel()
    edges = np.quantile(mag, np.linspace(0, 1, bins + 1))

    out = []
    for i in range(bins):
        mask = (mag >= edges[i]) & (mag <= edges[i + 1])
        rate = float(flipped[mask].mean()) if mask.any() else float("nan")
        out.append((float(edges[i]), float(edges[i + 1]), rate))
    return out


def scale_ratio(quant: np.ndarray, base: np.ndarray, group: int = 128) -> float:
    """Stored scale divided by the mean-absolute scale of the base weights.

    A ratio near one means the scale follows the obvious formula; anything
    else means it was fitted to something other than the weights themselves.
    """
    q = np.abs(quant.reshape(quant.shape[0], -1, group)).mean(axis=2)
    b = np.abs(base.reshape(base.shape[0], -1, group)).mean(axis=2)
    return float((q / b).mean())
