"""Command line interface."""

import argparse
import re

import numpy as np

from . import metrics, remote, ui

DEFAULT_GROUP = 128


# ==============================
# ===  Commands              ===
# ==============================

def cmd_info(args) -> None:
    """Report the storage structure of a quantized model."""
    index = remote.header(args.model)
    ui.heading(f"{args.model}  —  {len(index)} tensors")
    ui.rule()

    sample = args.tensor or next(
        n for n, e in index.items() if len(e["shape"]) == 2
    )
    w = remote.load(args.model, sample, rows=8)
    group = metrics.detect_group(w)

    print(f"  sample tensor : {sample}")
    print(f"  dtype         : {index[sample]['dtype']}")
    print(f"  shape         : {index[sample]['shape']}")
    print(f"  group size    : {group if group else 'not detected'}")
    print(f"  distinct |w|  : {len(np.unique(np.abs(w[0])))} in first row")
    print(f"  zero fraction : {metrics.zero_fraction(w):.3f}")


def cmd_compare(args) -> None:
    """Compare one tensor between a quantized model and its base."""
    q = remote.load(args.model, args.tensor, rows=args.rows)
    b = remote.load(args.base, args.tensor, rows=args.rows)

    nb = metrics.naive_binary(b, args.group)
    nt = metrics.naive_ternary(b, args.group, metrics.zero_fraction(q) or 0.4)

    ui.heading(f"{args.tensor}  {list(q.shape)}")
    ui.rule()
    print(f"  cos(model, base)        {ui.grade(metrics.cosine(q, b), 1.0, 0.0)}")
    print(f"  cos(naive binary, base) {ui.grade(metrics.cosine(nb, b), 1.0, 0.0)}")
    print(f"  cos(naive ternary, base){ui.grade(metrics.cosine(nt, b), 1.0, 0.0)}")
    print(f"  sign agreement          {metrics.sign_agreement(q, b) * 100:.1f}%")
    print(f"  relative error          {metrics.relative_error(q, b):.3f}")
    print(f"  naive relative error    {metrics.relative_error(nb, b):.3f}")
    print(f"  scale vs mean|w|        {metrics.scale_ratio(q, b, args.group):.3f}")
    print(f"  zero fraction           {metrics.zero_fraction(q) * 100:.1f}%")


def cmd_flips(args) -> None:
    """Show how sign flips are distributed across weight magnitudes."""
    q = remote.load(args.model, args.tensor, rows=args.rows)
    b = remote.load(args.base, args.tensor, rows=args.rows)

    ui.heading(f"sign flips by magnitude  —  {args.tensor}")
    ui.rule()
    for i, (lo, hi, rate) in enumerate(metrics.flip_by_magnitude(q, b), 1):
        bar = "█" * int(rate * 40)
        colour = "red" if rate > 0.35 else "yellow" if rate > 0.15 else "green"
        print(
            f"  decile {i:2}  |w| {lo:.4f}–{hi:.4f}  "
            f"{rate * 100:5.1f}%  {ui.paint(bar, colour)}"
        )


def _layer_order(name: str) -> tuple[int, str]:
    """Sort key that puts layer 2 before layer 11 instead of after it."""
    match = re.search(r"\.layers\.(\d+)\.", name)
    return (int(match.group(1)) if match else -1, name)


def cmd_profile(args) -> None:
    """Walk the layer stack and report deviation at each depth."""
    names = sorted(
        (
            n for n in remote.common_tensors(args.model, args.base)
            if args.pattern in n and len(remote.header(args.model)[n]["shape"]) == 2
        ),
        key=_layer_order,
    )
    if args.limit:
        step = max(1, len(names) // args.limit)
        names = names[::step]

    ui.heading(f"depth profile  —  {len(names)} tensors matching '{args.pattern}'")
    ui.rule(72)
    print(f"  {'tensor':44} {'cos':>8} {'naive':>8} {'sign%':>7}")
    for name in names:
        try:
            q = remote.load(args.model, name, rows=args.rows)
            b = remote.load(args.base, name, rows=args.rows)
        except Exception as exc:  # noqa: BLE001 - one bad tensor must not stop the sweep
            print(f"  {name:44} {ui.paint('failed: ' + str(exc)[:20], 'red')}")
            continue
        nb = metrics.naive_binary(b, args.group)
        short = name.replace("model.layers.", "L").replace(".weight", "")
        print(
            f"  {short:44} {ui.grade(metrics.cosine(q, b), 1.0, 0.0)} "
            f"{ui.grade(metrics.cosine(nb, b), 1.0, 0.0)} "
            f"{metrics.sign_agreement(q, b) * 100:6.1f}%"
        )


# ==============================
# ===  Entry point           ===
# ==============================

def build_parser() -> argparse.ArgumentParser:
    # Shared options live on a parent parser so they are accepted after the
    # subcommand, where anyone would naturally type them.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--group", type=int, default=DEFAULT_GROUP,
                        help="scaling group size (default: %(default)s)")
    common.add_argument("--rows", type=int, default=256,
                        help="rows sampled per tensor, 0 for all (default: %(default)s)")

    p = argparse.ArgumentParser(
        prog="bitprobe",
        parents=[common],
        description="Inspect how a low-bit model relates to its full-precision base",
    )
    sub = p.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", parents=[common],
                          help="report storage structure of a model")
    info.add_argument("model")
    info.add_argument("--tensor")
    info.set_defaults(func=cmd_info)

    for name, fn, helptext in (
        ("compare", cmd_compare, "compare one tensor against the base model"),
        ("flips", cmd_flips, "sign flips by weight magnitude"),
    ):
        sp = sub.add_parser(name, parents=[common], help=helptext)
        sp.add_argument("model")
        sp.add_argument("base")
        sp.add_argument("tensor")
        sp.set_defaults(func=fn)

    prof = sub.add_parser("profile", parents=[common],
                          help="deviation across the layer stack")
    prof.add_argument("model")
    prof.add_argument("base")
    prof.add_argument("--pattern", default="mlp.down_proj")
    prof.add_argument("--limit", type=int, default=12)
    prof.set_defaults(func=cmd_profile)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.rows == 0:
        args.rows = None
    try:
        args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - surfaced as a message, not a traceback
        print(ui.paint(f"error: {exc}", "red"))
        return 1
    return 0
