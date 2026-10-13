"""Figures for the Habr article (docs/habr-bonsai.md), drawn from measurements.

Every number below is copied from a file that recorded it, named next to it:
research/FINDINGS.md for the Bonsai and solver work, bindistill for the
distillation runs. The article and the figures read the same numbers, so when a
measurement changes it changes here and in the text, nowhere else.

Colours are Catppuccin Latte, checked for colour-blind separation on a white
page (Habr is white): Bonsai mauve, the layer-wise solver peach, my trained
models sapphire. Reference models are ink, naive rounding is grey - neither is
a "series", both are baselines.

Run:  venv/bin/python docs/figures.py   ->  docs/img/*.png

Signed: pluttan
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

OUT = Path(__file__).resolve().parent / "img"

# ==============================
# ===  Measurements          ===
# ==============================

# FINDINGS.md, perplexity table (WikiText-style held-out text, Qwen3-1.7B family).
PERPLEXITY = (
    ("Qwen3-1.7B, 16 бит", 12.32, "ink"),
    ("Bonsai, 1 бит", 16.53, "bonsai"),
    ("моё обучение по блокам", 1679.58, "mine"),
    ("мой послойный решатель", 923707.44, "solver"),
)

# FINDINGS.md, solver on block 0: output error on calibration vs held-out text.
OVERFIT = {
    "layers": ("q_proj", "v_proj", "up_proj", "down_proj"),
    "calibration": (0.12, 0.24, 0.27, 0.21),
    "heldout": (0.31, 0.58, 0.57, 0.70),
}

# Sign flips by decile of |w|, smallest first, in percent.
# Bonsai: `main.py flips prism-ml/Bonsai-1.7B-unpacked Qwen/Qwen3-1.7B
#          model.layers.0.mlp.gate_proj.weight`, measured 13.10.2026.
# Student: bindistill/docs/signature-qwen3-0.6b.json, all 196 matrices.
FLIPS = {
    "bonsai": (45.2, 43.7, 40.9, 37.0, 33.3, 28.8, 24.1, 19.0, 13.0, 5.3),
    "student": (45.754, 37.833, 30.147, 22.658, 15.329, 8.489, 3.313, 0.842,
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
CURVE = {"student": (81.9, 42.7, 34.6, 30.2, 27.85), "teacher": 25.02}

# bindistill/distill/figures.py AGREEMENT: run 2, held-out olmo-mix.
AGREEMENT = {
    "labels": ("первый токен\nсовпал", "выбор учителя\nв пятёрке студента",
               "совпал, когда\nучитель уверен"),
    "student": (72.2, 93.5, 96.4),
    "naive": (0.0, 0.1, 0.0),
}

# bindistill memory figure: Qwen3-0.6B, MB. Blocks vs the 16-bit vocabulary table.
MEMORY = {
    "rows": ("16 бит везде", "1 бит в блоках"),
    "blocks": (840.0, 59.1),
    "vocab": (296.9, 296.8),
}

# ==============================
# ===  Look                  ===
# ==============================

LATTE = {
    "bonsai": "#8839ef",   # mauve
    "solver": "#fe640b",   # peach
    "mine": "#209fb5",     # sapphire
    "ink": "#4c4f69",      # text
    "naive": "#9ca0b0",    # overlay0
    "grid": "#dce0e8",     # crust
    "muted": "#6c6f85",    # subtext0
}
WIDTH = 7.0
DPI = 200


def style():
    have = {f.name for f in font_manager.fontManager.ttflist}
    sans = next((n for n in ("Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans")
                 if n in have), "sans-serif")
    rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [sans], "font.size": 11,
        "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "axes.edgecolor": LATTE["ink"],
        "axes.labelcolor": LATTE["ink"], "text.color": LATTE["ink"],
        "xtick.color": LATTE["ink"], "ytick.color": LATTE["ink"],
        "axes.linewidth": 0.8, "figure.dpi": DPI, "savefig.dpi": DPI,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
        "legend.frameon": False, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "white",
    })


def ru(v: float, digits: int = 2, strip: bool = True) -> str:
    """Comma decimal, spaced thousands, as the article writes numbers.

    The space is a no-break space: Helvetica has no thin-space glyph and drops it.
    """
    text = f"{v:,.{digits}f}".replace(",", "\u00a0").replace(".", ",")
    if strip and "," in text:
        text = text.rstrip("0").rstrip(",")
    return text


def tick_ru(v, _):
    return ru(v, 2) if v < 1 else ru(v, 0) if v >= 1000 else ru(v, 1)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=LATTE["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name: str):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"  {name}")


# ==============================
# ===  The figures           ===
# ==============================

def fig_perplexity():
    """The whole story in one bar each: log scale, because the ends differ 75 000x."""
    rows = sorted(PERPLEXITY, key=lambda r: r[1], reverse=True)
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    y = range(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[LATTE[r[2]] for r in rows],
            height=0.62, edgecolor="white", linewidth=2, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(5, 2e7)
    ax.set_yticks(list(y), [r[0] for r in rows])
    for i, r in enumerate(rows):
        ax.annotate(ru(r[1], 2 if r[1] < 100 else 0), xy=(r[1], i), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=10.5,
                    fontweight="bold" if r[2] == "bonsai" else "normal")
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(FuncFormatter(tick_ru))
    ax.set_xlabel("перплексия, логарифмическая шкала (меньше лучше)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    grid(ax, "x")
    save(fig, "perplexity.png")


def fig_overfit():
    """Calibration error against held-out error: the solver memorises the calibration."""
    import numpy as np
    x = np.arange(len(OVERFIT["layers"]))
    w = 0.36
    fig, ax = plt.subplots(figsize=(WIDTH, 3.0))
    ax.bar(x - w / 2 - 0.01, OVERFIT["calibration"], w, color=LATTE["naive"],
           label="на калибровочном тексте", zorder=3)
    ax.bar(x + w / 2 + 0.01, OVERFIT["heldout"], w, color=LATTE["solver"],
           label="на новом тексте", zorder=3)
    for i, (a, b) in enumerate(zip(OVERFIT["calibration"], OVERFIT["heldout"])):
        ax.annotate(f"×{ru(b / a, 1)}", xy=(i + w / 2, b), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=10,
                    color=LATTE["ink"])
    ax.set_xticks(x, OVERFIT["layers"])
    ax.set_ylabel("ошибка на выходе слоя")
    ax.set_ylim(0, 0.82)
    ax.yaxis.set_major_formatter(FuncFormatter(tick_ru))
    ax.legend(loc="upper left", ncols=2)
    grid(ax)
    save(fig, "overfit.png")


def fig_flips():
    """The training signature: small weights flip, large ones stay."""
    x = range(1, 11)
    fig, ax = plt.subplots(figsize=(WIDTH, 3.3))
    for key, label in (("bonsai", "Bonsai-1.7B (gate_proj, блок 0)"),
                       ("student", "мой однобитный студент (0,6B, все матрицы)")):
        ax.plot(x, FLIPS[key], color=LATTE[key if key == "bonsai" else "mine"],
                linewidth=2.2, marker="o", markersize=7,
                markeredgecolor="white", markeredgewidth=1.5, label=label, zorder=3)
    ax.axhline(50, color=LATTE["naive"], linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("50 %: знак случаен", xy=(10.4, 50), xytext=(0, 4),
                textcoords="offset points", ha="right", fontsize=9.5,
                color=LATTE["muted"])
    ax.annotate("при наивном округлении знаки не меняются вовсе",
                xy=(1, 0.012), fontsize=9.5, color=LATTE["muted"])
    ax.set_yscale("log")
    ax.set_ylim(0.008, 90)
    ax.yaxis.set_major_formatter(FuncFormatter(tick_ru))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(list(x))
    ax.set_xlim(0.6, 10.4)
    ax.set_xlabel("дециль модуля исходного веса: от самых мелких к самым крупным")
    ax.set_ylabel("знаков перевёрнуто, %")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 0.08))
    grid(ax)
    save(fig, "flips.png")


def fig_ablation():
    """Block 0: what each ingredient buys. Data is the big lever."""
    rows = ABLATION
    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    y = range(len(rows))[::-1]
    ax.barh(list(y), [r[1] for r in rows], color=[LATTE[r[2]] for r in rows],
            height=0.62, edgecolor="white", linewidth=2, zorder=3)
    ax.set_yticks(list(y), [r[0] for r in rows])
    for yy, r in zip(y, rows):
        ax.annotate(ru(r[1], 2, strip=False), xy=(r[1], yy), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=10.5)
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(FuncFormatter(tick_ru))
    ax.set_xlabel("ошибка блока 0 на новом тексте (меньше лучше)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    grid(ax, "x")
    save(fig, "ablation.png")


def fig_curve():
    """Run 1: the student is still improving when the token budget ends."""
    x = range(1, 6)
    fig, ax = plt.subplots(figsize=(WIDTH, 3.0))
    ax.plot(x, CURVE["student"], color=LATTE["mine"], linewidth=2.2, marker="o",
            markersize=7, markeredgecolor="white", markeredgewidth=1.5,
            label="однобитный студент", zorder=3)
    ax.axhline(CURVE["teacher"], color=LATTE["ink"], linewidth=1.2,
               linestyle=(0, (5, 3)), label=f"учитель Qwen3-0.6B, {ru(CURVE['teacher'])}")
    ax.annotate(f"{ru(CURVE['student'][-1])}\nбюджет 300 млн токенов исчерпан",
                xy=(5, CURVE["student"][-1]), xytext=(0, 26),
                textcoords="offset points", ha="right", va="bottom", fontsize=10)
    ax.set_xticks(list(x))
    ax.set_xlabel("замер на отложенном тексте по ходу обучения")
    ax.set_ylabel("перплексия")
    ax.set_ylim(20, 88)
    ax.yaxis.set_major_formatter(FuncFormatter(tick_ru))
    ax.legend(loc="upper right")
    grid(ax)
    save(fig, "curve.png")


def fig_agreement():
    """Run 2: how often the one-bit student picks what the teacher picks."""
    import numpy as np
    x = np.arange(3)
    w = 0.36
    fig, ax = plt.subplots(figsize=(WIDTH, 3.0))
    ax.bar(x - w / 2 - 0.01, AGREEMENT["student"], w, color=LATTE["mine"],
           label="однобитный студент", zorder=3)
    ax.bar(x + w / 2 + 0.01, [max(v, 0.6) for v in AGREEMENT["naive"]], w,
           color=LATTE["naive"], label="наивное округление", zorder=3)
    for i, (s, n) in enumerate(zip(AGREEMENT["student"], AGREEMENT["naive"])):
        ax.annotate(f"{ru(s, 1)} %", xy=(i - w / 2, s), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=10.5,
                    fontweight="bold")
        ax.annotate(f"{ru(n, 1)} %", xy=(i + w / 2, 0.6), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=10,
                    color=LATTE["muted"])
    ax.set_xticks(x, AGREEMENT["labels"])
    ax.set_ylabel("доля токенов, %")
    ax.set_ylim(0, 106)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncols=2)
    grid(ax)
    save(fig, "agreement.png")


def fig_memory():
    """Qwen3-0.6B on disk: blocks shrink 14x, the 16-bit vocabulary does not."""
    fig, ax = plt.subplots(figsize=(WIDTH, 2.2))
    y = (1, 0)
    ax.barh(y, MEMORY["blocks"], color=LATTE["mine"], height=0.56,
            edgecolor="white", linewidth=2, label="матрицы блоков", zorder=3)
    ax.barh(y, MEMORY["vocab"], left=MEMORY["blocks"], color=LATTE["naive"],
            height=0.56, edgecolor="white", linewidth=2,
            label="таблица словаря, 16 бит", zorder=3)
    for yy, b, v in zip(y, MEMORY["blocks"], MEMORY["vocab"]):
        for left, width in ((0, b), (b, v)):
            if width > 150:
                ax.annotate(ru(width, 0), xy=(left + width / 2, yy), ha="center",
                            va="center", fontsize=10, color="white",
                            fontweight="bold")
        ax.annotate(f"{ru(b + v, 0)} МБ", xy=(b + v, yy), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=10.5,
                    fontweight="bold")
    ax.set_yticks(y, MEMORY["rows"])
    ax.set_xlim(0, 1350)
    ax.set_xlabel("размер модели, МБ")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncols=2)
    grid(ax, "x")
    save(fig, "memory.png")


def main():
    style()
    print(f"рисую в {OUT}")
    for fig in (fig_perplexity, fig_overfit, fig_flips, fig_ablation,
                fig_curve, fig_agreement, fig_memory):
        fig()


if __name__ == "__main__":
    main()
