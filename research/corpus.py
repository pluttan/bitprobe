"""Build a training corpus large enough that the signs generalise.

The first attempt trained on forty thousand tokens seen twelve hundred times
over, which fits the calibration set rather than the language. FineWeb-Edu's
sample shards are plain parquet, so no dataset library is needed — just the
text column, tokenised and concatenated.
"""

import sys

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

REPO = "HuggingFaceFW/fineweb-edu"
SHARD = "sample/10BT/{:03d}_00000.parquet"
OUT = "/tmp/corpus_tokens.npy"


def main() -> None:
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 40_000_000
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    eos = tok.eos_token_id

    chunks, total, shard = [], 0, 0
    while total < target:
        path = hf_hub_download(REPO, SHARD.format(shard), repo_type="dataset")
        table = pq.read_table(path, columns=["text"])
        print(f"shard {shard}: {table.num_rows} documents", flush=True)

        texts = table.column("text").to_pylist()
        for i in range(0, len(texts), 512):
            batch = tok(texts[i:i + 512], add_special_tokens=False)["input_ids"]
            for ids in batch:
                chunks.append(np.array(ids + [eos], dtype=np.uint32))
                total += len(ids) + 1
            if total >= target:
                break
            if (i // 512) % 20 == 0:
                print(f"  {total / 1e6:.1f}M tokens", flush=True)
        shard += 1

    tokens = np.concatenate(chunks)[:target]
    np.save(OUT, tokens)
    print(f"saved {len(tokens) / 1e6:.2f}M tokens to {OUT}", flush=True)


if __name__ == "__main__":
    main()
