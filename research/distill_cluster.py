"""End-to-end binary distillation for a multi-GPU box.

Per-block training on a laptop got the model to a perplexity of 1680 and
stopped improving for a structural reason: a block fitted against its own
teacher cannot see what the rest of the network will do with its output, so the
error accumulates down the stack. Only a loss on the logits removes that, and
that needs the whole network in the graph at once — which is what this script
does and what a 16 GB machine cannot.

Nothing here is novel. It is the training recipe the measurements pointed at:
signs initialised from the base model, kept binary through a straight-through
estimator, with a learned scale per group of 128, trained against the
full-precision teacher's distribution.

Launch:
    torchrun --nproc_per_node=12 distill_cluster.py --tokens 10e9

Untested — this machine has no CUDA device. Read it before trusting it.
"""

import argparse
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

GROUP = 128


# ==============================
# ===  Binary linear layer   ===
# ==============================

class BinaryLinear(torch.nn.Module):
    """A linear layer whose weights are one bit per element at every forward.

    The master weight stays in floating point and receives gradients; only its
    sign reaches the matmul. The scale is a real parameter rather than the mean
    magnitude, because Bonsai's scales sit at roughly twice the naive value —
    not something a formula produces.
    """

    def __init__(self, linear: torch.nn.Linear):
        super().__init__()
        w = linear.weight.data.float()
        out, width = w.shape
        assert width % GROUP == 0, f"width {width} not divisible by {GROUP}"
        self.master = torch.nn.Parameter(w)
        groups = w.view(out, width // GROUP, GROUP)
        self.log_scale = torch.nn.Parameter(
            groups.abs().mean(dim=2).clamp_min(1e-8).log())
        self.bias = linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, width = self.master.shape
        g = self.master.view(out, -1, GROUP)
        signs = torch.where(g >= 0, 1.0, -1.0)
        q = (signs * self.log_scale.exp().unsqueeze(-1)).view(out, width)
        # Straight-through: the sign is a step function, so the gradient is
        # taken from the master weight instead.
        q = q + (self.master - self.master.detach())
        return F.linear(x, q.to(x.dtype), self.bias)


def binarise_model(model, skip_embedding: bool = True) -> int:
    """Replace every linear layer inside the transformer blocks."""
    count = 0
    for block in model.model.layers:
        for parent in (block.self_attn, block.mlp):
            for name, child in list(parent.named_children()):
                if isinstance(child, torch.nn.Linear):
                    setattr(parent, name, BinaryLinear(child))
                    count += 1
    if not skip_embedding:
        raise NotImplementedError(
            "quantising the embedding table needs a custom lookup; Bonsai does "
            "it, and it is worth doing, but it is not what closes this gap")
    return count


# ==============================
# ===  Data                  ===
# ==============================

class TokenStream(torch.utils.data.IterableDataset):
    """Fixed-length windows over a flat token file, sharded across ranks."""

    def __init__(self, path: str, seq: int, rank: int, world: int, seed: int = 0):
        self.path, self.seq = path, seq
        self.rank, self.world, self.seed = rank, world, seed

    def __iter__(self):
        import numpy as np
        tokens = np.load(self.path, mmap_mode="r")
        limit = len(tokens) - self.seq - 1
        rng = np.random.default_rng(self.seed + self.rank)
        while True:
            start = int(rng.integers(0, limit))
            chunk = torch.from_numpy(
                tokens[start:start + self.seq + 1].astype("int64"))
            yield chunk[:-1], chunk[1:]


# ==============================
# ===  Training              ===
# ==============================

def distillation_loss(student_logits, teacher_logits, targets, alpha: float):
    """Match the teacher's distribution, anchored by the real next token."""
    s = F.log_softmax(student_logits.float(), dim=-1)
    t = F.log_softmax(teacher_logits.float(), dim=-1)
    kl = F.kl_div(s, t, reduction="batchmean", log_target=True) / s.shape[1]
    ce = F.cross_entropy(student_logits.float().flatten(0, 1), targets.flatten())
    return alpha * kl + (1 - alpha) * ce, kl.item(), ce.item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--tokens", type=float, default=10e9)
    ap.add_argument("--corpus", default="/data/fineweb_tokens.npy")
    ap.add_argument("--init", default=None,
                    help="master weights from the per-block stage, if any")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--scale-lr", type=float, default=5e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--out", default="/data/bonsai_replica")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda")

    from transformers import AutoModelForCausalLM

    teacher = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float32)
    replaced = binarise_model(student)
    if args.init:
        from safetensors.torch import load_file
        state = load_file(args.init)
        missing = student.load_state_dict(
            {k: v for k, v in state.items()}, strict=False)
        if rank == 0:
            print(f"init: {len(state)} tensors, missing {len(missing.missing_keys)}")
    student = student.to(device)
    student.gradient_checkpointing_enable()
    student = DDP(student, device_ids=[rank % torch.cuda.device_count()])

    scales = [p for n, p in student.named_parameters() if n.endswith("log_scale")]
    weights = [p for n, p in student.named_parameters() if not n.endswith("log_scale")]
    opt = torch.optim.AdamW([
        {"params": weights, "lr": args.lr},
        {"params": scales, "lr": args.scale_lr},
    ], betas=(0.9, 0.95), weight_decay=0.0)

    per_step = args.micro_batch * args.accum * args.seq * world
    total_steps = int(args.tokens / per_step)
    if rank == 0:
        print(f"{replaced} binary layers, {world} ranks, "
              f"{per_step / 1e3:.0f}k tokens/step, {total_steps} steps")

    def lr_at(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        p = (step - args.warmup) / max(1, total_steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    stream = TokenStream(args.corpus, args.seq, rank, world)
    loader = torch.utils.data.DataLoader(
        stream, batch_size=args.micro_batch, num_workers=2, pin_memory=True)
    it = iter(loader)

    started = time.time()
    for step in range(total_steps):
        factor = lr_at(step)
        opt.param_groups[0]["lr"] = args.lr * factor
        opt.param_groups[1]["lr"] = args.scale_lr * factor

        opt.zero_grad(set_to_none=True)
        kl_sum = ce_sum = 0.0
        for micro in range(args.accum):
            ids, targets = next(it)
            ids, targets = ids.to(device, non_blocking=True), targets.to(device)
            with torch.no_grad():
                ref = teacher(ids).logits
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = student(ids).logits
            loss, kl, ce = distillation_loss(out, ref, targets, args.alpha)
            (loss / args.accum).backward()
            kl_sum += kl / args.accum
            ce_sum += ce / args.accum

        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        if rank == 0 and step % 50 == 0:
            seen = (step + 1) * per_step
            print(f"step {step:6}  kl {kl_sum:.4f}  ce {ce_sum:.4f}  "
                  f"ppl {math.exp(min(20, ce_sum)):8.2f}  "
                  f"{seen / 1e9:.2f}B tokens  ({time.time() - started:.0f}s)",
                  flush=True)

        if rank == 0 and step and step % 2000 == 0:
            os.makedirs(args.out, exist_ok=True)
            torch.save(student.module.state_dict(),
                       f"{args.out}/step{step}.pt")

    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        torch.save(student.module.state_dict(), f"{args.out}/final.pt")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
