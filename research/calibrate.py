"""Build a calibration token set for the reconstruction experiment.

Quantization that fits the real activation distribution needs text that looks
like what the model actually sees, so the sample mixes prose, technical
writing and code rather than a single register.
"""

import json
import urllib.parse
import urllib.request

import numpy as np
from tokenizers import Tokenizer

ARTICLES = [
    "Quantization (signal processing)", "Neural network (machine learning)",
    "Linear algebra", "History of the Soviet Union", "Photosynthesis",
    "Jazz", "Byzantine Empire", "Semiconductor", "Cooking", "Chess",
]

CODE = '''
def coordinate_descent(w, hessian, sweeps=3):
    b = np.sign(w)
    for _ in range(sweeps):
        for i in range(w.shape[1]):
            delta = -4 * s * b[:, i] * grad[:, i] + 4 * s ** 2 * diag[i]
            b[delta < 0, i] *= -1
    return b

class Server:
    def __init__(self, host: str, port: int = 8080) -> None:
        self.host, self.port = host, port
'''


def wiki(title: str) -> str:
    url = ("https://en.wikipedia.org/w/api.php?action=query&prop=extracts"
           "&explaintext=1&format=json&titles=" + urllib.parse.quote(title))
    # Wikipedia rejects requests without a descriptive User-Agent.
    req = urllib.request.Request(url, headers={"User-Agent": "bitprobe-calibration/0.1"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        pages = json.load(resp)["query"]["pages"]
    return next(iter(pages.values())).get("extract", "")


def main() -> None:
    chunks = []
    for title in ARTICLES:
        try:
            text = wiki(title)
            chunks.append(text)
            print(f"  {title[:40]:42} {len(text):7} chars")
        except Exception as exc:  # noqa: BLE001 - a missing article is not fatal
            print(f"  {title[:40]:42} failed: {exc}")
    chunks.append(CODE * 4)

    tok = Tokenizer.from_file("/tmp/qwen_tokenizer.json")
    ids = []
    for chunk in chunks:
        ids.extend(tok.encode(chunk[:60000]).ids)

    ids = np.array(ids, dtype=np.int64)
    np.save("/tmp/calib_tokens.npy", ids)
    print(f"\ntotal tokens: {len(ids)}  unique: {len(np.unique(ids))}")


if __name__ == "__main__":
    main()
