"""Remote access to safetensors repositories.

Low-bit models are compared against their full-precision base, and both sides
run into tens of gigabytes. Downloading them in full to look at a handful of
matrices is wasteful, so every tensor is served over HTTP range requests and
only the bytes under study cross the wire.
"""

import json
import time
import urllib.request

import numpy as np

HF = "https://huggingface.co"

_HEADERS: dict[str, dict] = {}


# ==============================
# ===  Raw byte transport    ===
# ==============================

def _get(url: str, start: int, length: int, tries: int = 4) -> bytes:
    """Fetch a byte range, retrying on transient network failures.

    Range reads against the CDN fail sporadically under load; a plain retry
    with backoff recovers far more often than it gives up.
    """
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={start}-{start + length - 1}"}
            )
            return urllib.request.urlopen(req, timeout=90).read()
        except Exception as exc:  # noqa: BLE001 - retried below, re-raised at end
            last = exc
            if attempt < tries - 1:
                time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"range read failed for {url}: {last}")


def _shards(repo: str) -> list[str]:
    """List the safetensors files of a repository."""
    with urllib.request.urlopen(f"{HF}/api/models/{repo}", timeout=60) as resp:
        meta = json.load(resp)
    names = [
        f["rfilename"]
        for f in meta.get("siblings", [])
        if f["rfilename"].endswith(".safetensors")
    ]
    if not names:
        raise RuntimeError(f"no safetensors in {repo}")
    return sorted(names)


# ==============================
# ===  Tensor index          ===
# ==============================

def header(repo: str) -> dict[str, dict]:
    """Build a tensor index for a repository.

    The safetensors header is a JSON blob at the start of each shard, so the
    whole index costs a few kilobytes regardless of model size.
    """
    if repo in _HEADERS:
        return _HEADERS[repo]

    index: dict[str, dict] = {}
    for shard in _shards(repo):
        url = f"{HF}/{repo}/resolve/main/{shard}"
        size = int.from_bytes(_get(url, 0, 8), "little")
        meta = json.loads(_get(url, 8, size))
        meta.pop("__metadata__", None)
        for name, entry in meta.items():
            index[name] = {**entry, "url": url, "base": 8 + size}

    _HEADERS[repo] = index
    return index


# ==============================
# ===  Tensor loading        ===
# ==============================

def _decode(raw: bytes, dtype: str) -> np.ndarray:
    """Decode raw bytes into float32.

    BF16 shares its exponent layout with float32 rather than float16, so it is
    widened by shifting into the high half. Reading it as float16 silently
    yields plausible-looking garbage, which is worse than an error.
    """
    if dtype == "BF16":
        half = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
        return (half << 16).view(np.float32)
    if dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).copy()
    raise ValueError(f"unsupported dtype {dtype}")


def load(repo: str, name: str, rows: int | None = None) -> np.ndarray:
    """Load a tensor, or its first `rows` rows, as float32.

    Row slicing keeps statistics affordable on large matrices: safetensors is
    row-major, so a prefix of rows is a contiguous byte range.
    """
    entry = header(repo)[name]
    start, end = entry["data_offsets"]
    shape = entry["shape"]

    if rows is None or len(shape) < 2:
        raw = _get(entry["url"], entry["base"] + start, end - start)
        return _decode(raw, entry["dtype"]).reshape(shape)

    rows = min(rows, shape[0])
    width = int(np.prod(shape[1:]))
    itemsize = 2 if entry["dtype"] in ("F16", "BF16") else 4
    raw = _get(entry["url"], entry["base"] + start, rows * width * itemsize)
    return _decode(raw, entry["dtype"]).reshape([rows] + list(shape[1:]))


def common_tensors(a: str, b: str) -> list[str]:
    """Tensor names present in both repositories with identical shapes."""
    ha, hb = header(a), header(b)
    return sorted(
        n for n in set(ha) & set(hb) if ha[n]["shape"] == hb[n]["shape"]
    )
