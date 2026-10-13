"""Figures for the Habr article (docs/habr-bonsai.md), drawn from measurements.

Every number below is copied from a file that recorded it, named next to it:
research/FINDINGS.md for the Bonsai and solver work, bindistill for the
distillation runs. The article and the figures read the same numbers, so when a
measurement changes it changes here and in the text, nowhere else.

Each figure is a self-contained card. Habr follows the reader's system theme,
so a picture lands on a white page for some readers and a near-black one for
others: the card carries its own background (Habr's dark --background-secondary,
#262626) and the corners outside it are transparent, so it sits cleanly on both.
Text colours are Habr's own (--text-main, --text-secondary). The three series
colours are Habr's label hues darkened until they pass the colour-blind and
contrast checks on #262626: Bonsai orchid, the layer-wise solver orange, my
trained models cyan. Baselines (naive rounding) are grey, the 16-bit reference
is the text colour - neither is a series.

Run:  venv/bin/python docs/figures.py   ->  docs/img/*.png

Signed: pluttan
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter, NullFormatter

OUT = Path(__file__).resolve().parent / "img"

# ==============================
# ===  Measurements          ===
# ==============================

# FINDINGS.md, perplexity table (Qwen3-1.7B family, same held-out text).
PERPLEXITY = (
    ("Qwen3-1.7B, 16 бит", 12.32, "ref"),
    ("Bonsai, 1 бит", 16.53, "bonsai"),
    ("моё обучение по блокам", 1679.58, "mine"),
    ("мой послойный решатель", 923707.44, "solver"),
)

# FINDINGS.md, solver on block 0: output error on calibration vs held-out text.
OVERFIT = (
    ("q_proj", 0.12, 0.31),
    ("v_proj", 0.24, 0.58),
    ("up_proj", 0.27, 0.57),
    ("down_proj", 0.21, 0.70),
)

# Sign flips by decile of |w|, smallest first, in percent.
# Bonsai: `main.py flips prism-ml/Bonsai-1.7B-unpacked Qwen/Qwen3-1.7B
#          model.layers.0.mlp.gate_proj.weight`, measured 13.10.2026.
# Student: bindistill/docs/signature-qwen3-0.6b.json, all 196 matrices.
FLIPS = {
    "bonsai": (45.2, 43.7, 40.9, 37.0, 33.3, 28.8, 24.1, 19.0, 13.0, 5.3),
    "mine": (45.754, 37.833, 30.147, 22.658, 15.329, 8.489, 3.313, 0.842,
             0.152, 0.023),
}

# FINDINGS.md, ablation on block 0, held-out block error.
ABLATION = (
    ("наивное округление", 0.8690, "naive"),
    ("только масштабы, 40 млн токенов", 0.5403, "mine"),
    ("знаки, 40 тыс. токенов", 0.5042, "mine"),
    ("знаки, 40 млн токенов", 0.3883, "mine"),
    ("знаки и масштабы, 40 млн токенов", 0.3652, "mine"),
)

# bindistill/distill/figures.py CURVE: run 1, five held-out evaluations,
# the last one at the end of the 300M-token budget.
CURVE = {"mine": (81.9, 42.7, 34.6, 30.2, 27.85), "teacher": 25.02}

# bindistill/distill/figures.py AGREEMENT: run 2, held-out olmo-mix, percent.
AGREEMENT = (
    ("первый токен совпал", 72.2, 0.0),
    ("выбор учителя в пятёрке студента", 93.5, 0.1),
    ("совпал там, где учитель уверен", 96.4, 0.0),
)

# bindistill memory figure: Qwen3-0.6B, MB. Blocks vs the 16-bit vocabulary table.
MEMORY = (
    ("16 бит везде", 840.0, 296.9),
    ("1 бит в блоках", 59.1, 296.8),
)

# ==============================
# ===  Look                  ===
# ==============================

C = {
    "card": "#262626",     # Habr dark --background-secondary
    "ink": "#dedede",      # --text-main
    "muted": "#ababab",    # --text-secondary
    "track": "#333333",    # empty part of a bar
    "grid": "#383838",
    "bonsai": "#c434f4",   # --label-dark-orchid, darkened to pass the checks
    "solver": "#e07306",   # --label-sorbus, darkened
    "mine": "#05a4bd",     # --label-pelorous, darkened
    "ref": "#dedede",
    "naive": "#6e6e6e",
}
W = 8.0
DPI = 200
PILL = 13            # bar thickness in points


def style():
    have = {f.name for f in font_manager.fontManager.ttflist}
    sans = next((n for n in ("Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans")
                 if n in have), "sans-serif")
    rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [sans], "font.size": 11,
        "text.color": C["ink"], "axes.labelcolor": C["muted"],
        "xtick.color": C["muted"], "ytick.color": C["muted"],
        "axes.edgecolor": C["grid"], "figure.dpi": DPI, "savefig.dpi": DPI,
    })


def ru(v: float, digits: int = 2, strip: bool = True) -> str:
    """Comma decimal, spaced thousands. The space is a no-break space:
    Helvetica has no thin-space glyph and silently drops it."""
    text = f"{v:,.{digits}f}".replace(",", " ").replace(".", ",")
    if strip and "," in text:
        text = text.rstrip("0").rstrip(",")
    return text


def card(height: float, title: str, subtitle: str, legend=()):
    """A transparent figure with one rounded card on it and the header drawn."""
    fig = plt.figure(figsize=(W, height), facecolor="none")
    fig.patches.append(FancyBboxPatch(
        (0.004, 0.006), 0.992, 0.988, transform=fig.transFigure, zorder=-10,
        boxstyle=f"round,pad=0,rounding_size={0.22 / height:.3f}",
        facecolor=C["card"], edgecolor="none", mutation_aspect=W / height))
    top = 1 - 0.42 / height
    fig.text(0.055, top, title, fontsize=17, fontweight="bold", va="top")
    fig.text(0.055, top - 0.36 / height, subtitle, fontsize=10.5,
             color=C["muted"], va="top")
    x = 0.055
    for label, key in legend:
        fig.text(x, top - 0.68 / height, "●", fontsize=12, color=C[key], va="top")
        fig.text(x + 0.022, top - 0.685 / height, label, fontsize=10.5,
                 color=C["muted"], va="top")
        x += 0.045 + 0.0105 * len(label)
    return fig


def axes(fig, height: float, header: float):
    """Plot area under a header of `header` inches."""
    ax = fig.add_axes([0.055, 0.35 / height, 0.89, 1 - (header + 0.5) / height])
    ax.set_facecolor("none")
    return ax


def line_axes(fig, height: float, header: float):
    """Plot area for a line chart: room on the left for tick labels, below for the x label."""
    ax = fig.add_axes([0.11, 0.85 / height, 0.82, 1 - (header + 1.0) / height])
    ax.set_facecolor("none")
    return ax


def pill(ax, y, x0, x1, color, **kw):
    ax.plot([x0, x1], [y, y], color=color, linewidth=PILL,
            solid_capstyle="round", **kw)


def save(fig, name: str):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, transparent=True)
    plt.close(fig)
    print(f"  {name}")


def bare(ax):
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(which="both", length=0)


# ==============================
# ===  The figures           ===
# ==============================

def fig_perplexity():
    """The whole story: log scale, because the ends differ 75 000 times."""
    h = 4.4
    fig = card(h, "Перплексия: от миллиона до Bonsai",
               "Qwen3-1.7B и однобитные версии. Меньше лучше, шкала логарифмическая")
    ax = axes(fig, h, 1.25)
    lo, hi = math.log10(5), math.log10(3e7)
    pos = lambda v: (math.log10(v) - lo) / (hi - lo)
    for k in range(1, 8):
        ax.plot([pos(10 ** k)] * 2, [-0.4, 3.25], color=C["grid"], lw=0.8, zorder=1)
        ax.text(pos(10 ** k), -0.68, ru(10 ** k, 0), fontsize=8.5,
                color=C["muted"], ha="center")
    for i, (name, v, key) in enumerate(reversed(PERPLEXITY)):
        pill(ax, i, 0.01, 0.99, C["track"], zorder=2)
        pill(ax, i, 0.01, max(pos(v), 0.011), C[key], zorder=3)
        ax.text(0, i + 0.27, name, fontsize=11.5, va="bottom")
        ax.text(0.99, i + 0.27, ru(v, 2 if v < 100 else 0), fontsize=13,
                fontweight="bold", color=C[key], ha="right", va="bottom")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.8, 3.7)
    ax.axis("off")
    save(fig, "perplexity.png")


def fig_overfit():
    """The solver memorises the calibration text: error doubles on new text."""
    h = 5.4
    fig = card(h, "Решатель запомнил калибровку",
               "Ошибка на выходе слоя, блок 0. Справа во сколько раз она выросла",
               legend=(("на калибровочном тексте", "naive"),
                       ("на новом тексте", "solver")))
    ax = axes(fig, h, 1.5)
    top = 0.8
    for i, (name, calib, held) in enumerate(reversed(OVERFIT)):
        y = i * 1.5
        ax.text(0, y + 0.3, name, fontsize=11.5, va="bottom")
        for dy, v, key in ((0.0, held, "solver"), (-0.42, calib, "naive")):
            pill(ax, y + dy, 0.01, 0.88, C["track"], zorder=2)
            pill(ax, y + dy, 0.01, 0.01 + 0.87 * v / top, C[key], zorder=3)
            ax.text(0.02 + 0.87 * v / top, y + dy, "  " + ru(v, 2, strip=False),
                    fontsize=9.5, color=C["muted"], va="center")
        ax.text(0.99, y - 0.21, f"×{ru(held / calib, 1)}", fontsize=15,
                fontweight="bold", color=C["solver"], ha="right", va="center")
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.75, 5.0)
    ax.axis("off")
    save(fig, "overfit.png")


def fig_flips():
    """The training signature: small weights flip, large ones stay put."""
    h = 4.8
    fig = card(h, "Как по весам узнать обучение",
               "Доля перевёрнутых знаков по децилям модуля исходного веса, шкала логарифмическая")
    ax = line_axes(fig, h, 1.25)
    x = range(1, 11)
    ends = {"bonsai": "Bonsai-1.7B\ngate_proj, блок 0", "mine": "мой студент 0,6B\nвсе матрицы"}
    for key in ("bonsai", "mine"):
        ys = FLIPS[key]
        ax.plot(x, ys, color=C[key], lw=2.6, zorder=3, solid_capstyle="round")
        ax.scatter(x, ys, s=46, color=C[key], edgecolor=C["card"], linewidth=1.6, zorder=4)
        ax.text(10.35, ys[-1], ends[key], color=C[key], fontsize=10, va="center",
                fontweight="bold", linespacing=1.25)
    ax.axhline(50, color=C["muted"], lw=1, ls=(0, (4, 4)), zorder=2)
    ax.text(1, 58, "50 %: знак случаен", fontsize=9.5, color=C["muted"])
    ax.text(1, 0.011, "у наивного округления 0 % во всех децилях", fontsize=9.5,
            color=C["muted"])
    ax.set_yscale("log")
    ax.set_ylim(0.008, 100)
    ax.set_xlim(0.6, 12.6)
    ax.set_xticks(list(x))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{ru(v, 2) if v < 1 else ru(v, 0)} %"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="y", color=C["grid"], lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("от самых мелких весов к самым крупным", fontsize=10)
    bare(ax)
    save(fig, "flips.png")


def fig_ablation():
    """Block 0: what each ingredient buys. Data is the big lever."""
    h = 4.6
    fig = card(h, "Что даёт обучение одного блока",
               "Ошибка блока 0 на новом тексте. Меньше лучше")
    ax = axes(fig, h, 1.25)
    for i, (name, v, key) in enumerate(reversed(ABLATION)):
        pill(ax, i, 0.01, 0.99, C["track"], zorder=2)
        pill(ax, i, 0.01, 0.01 + 0.98 * v, C[key], zorder=3)
        ax.text(0, i + 0.27, name, fontsize=11.5, va="bottom")
        ax.text(0.99, i + 0.27, ru(v, 2, strip=False), fontsize=13,
                fontweight="bold", color=C[key] if key != "naive" else C["muted"],
                ha="right", va="bottom")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.5, 4.75)
    ax.axis("off")
    save(fig, "ablation.png")


def fig_curve():
    """Run 1: the student still improves when the token budget ends."""
    h = 4.4
    fig = card(h, "Студент догоняет учителя",
               "Перплексия на отложенном тексте по ходу обучения, первый запуск")
    ax = line_axes(fig, h, 1.25)
    x = range(1, 6)
    ys = CURVE["mine"]
    ax.fill_between(x, ys, CURVE["teacher"], color=C["mine"], alpha=0.12, lw=0, zorder=2)
    ax.plot(x, ys, color=C["mine"], lw=2.6, zorder=3)
    ax.scatter(x, ys, s=46, color=C["mine"], edgecolor=C["card"], linewidth=1.6, zorder=4)
    ax.axhline(CURVE["teacher"], color=C["ink"], lw=1.2, ls=(0, (5, 4)), zorder=2)
    ax.text(1.05, CURVE["teacher"] - 6.5, f"учитель Qwen3-0.6B, {ru(CURVE['teacher'])}",
            fontsize=10, color=C["ink"])
    ax.text(5, ys[-1] + 4, f"{ru(ys[-1])}", fontsize=14, fontweight="bold",
            color=C["mine"], ha="center")
    ax.text(5, ys[-1] + 11.5, "бюджет 300 млн\nтокенов исчерпан", fontsize=9.5,
            color=C["muted"], ha="center", linespacing=1.2)
    ax.text(1.12, ys[0], ru(ys[0], 1), fontsize=11, color=C["mine"], va="center")
    ax.set_xticks(list(x), [f"замер {i}" for i in x])
    ax.set_ylim(17, 90)
    ax.set_xlim(0.7, 5.5)
    ax.grid(axis="y", color=C["grid"], lw=0.8)
    ax.set_axisbelow(True)
    bare(ax)
    save(fig, "curve.png")


def fig_agreement():
    """Run 2: how often the one-bit student picks what the teacher picks."""
    h = 4.6
    fig = card(h, "Однобитный студент выбирает как учитель",
               "Второй запуск, отложенный текст. Наивное округление почти никогда не угадывает",
               )
    ax = axes(fig, h, 1.25)
    for i, (name, s, n) in enumerate(reversed(AGREEMENT)):
        y = i * 1.45
        ax.text(0, y + 0.27, name, fontsize=11.5, va="bottom")
        ax.text(0.99, y + 0.27, f"{ru(s, 1)} %", fontsize=14, fontweight="bold",
                color=C["mine"], ha="right", va="bottom")
        pill(ax, y, 0.01, 0.99, C["track"], zorder=2)
        pill(ax, y, 0.01, 0.01 + 0.98 * s / 100, C["mine"], zorder=3)
        ax.text(0.0, y - 0.36, f"наивное округление: {ru(n, 1)} %", fontsize=9.5,
                color=C["muted"], ha="left", va="center")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.65, 3.6)
    ax.axis("off")
    save(fig, "agreement.png")


def fig_memory():
    """Qwen3-0.6B on disk: blocks shrink 14x, the 16-bit vocabulary does not."""
    h = 3.6
    fig = card(h, "Куда уходит место",
               "Qwen3-0.6B на диске, МБ. Блоки сжимаются в 14 раз, словарь остаётся 16-битным",
               legend=(("матрицы блоков", "mine"), ("таблица словаря", "naive")))
    ax = axes(fig, h, 1.55)
    total = max(b + v for _, b, v in MEMORY)
    scale = 0.86 / total
    for i, (name, b, v) in enumerate(reversed(MEMORY)):
        y = i * 1.0
        ax.text(0, y + 0.27, name, fontsize=11.5, va="bottom")
        ax.text(0.99, y + 0.27, f"{ru(b + v, 0)} МБ", fontsize=14,
                fontweight="bold", ha="right", va="bottom")
        end = 0.01 + (b + v) * scale
        cut = 0.01 + b * scale
        pill(ax, y, 0.01, end, C["naive"], zorder=2)
        pill(ax, y, 0.01, max(cut, 0.011), C["mine"], zorder=3)
        if b > 150:
            ax.text((0.01 + cut) / 2, y, ru(b, 0), fontsize=9.5, color="white",
                    ha="center", va="center", fontweight="bold", zorder=5)
        ax.text((cut + end) / 2 + 0.01, y, ru(v, 0), fontsize=9.5, color="white",
                ha="center", va="center", fontweight="bold", zorder=5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.5, 1.75)
    ax.axis("off")
    save(fig, "memory.png")


def main():
    style()
    print(f"рисую в {OUT}")
    for fig in (fig_perplexity, fig_overfit, fig_flips, fig_ablation,
                fig_curve, fig_agreement, fig_memory):
        fig()


if __name__ == "__main__":
    main()
