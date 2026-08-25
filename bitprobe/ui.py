"""Terminal styling.

Colours are the Catppuccin Mocha palette, applied as 24-bit ANSI so the output
looks the same regardless of the terminal's own theme. Styling is dropped
automatically when stdout is not a terminal, keeping piped output clean.
"""

import sys

MOCHA = {
    "text": (205, 214, 244),
    "subtext": (166, 173, 200),
    "green": (166, 227, 161),
    "red": (243, 139, 168),
    "yellow": (249, 226, 175),
    "blue": (137, 180, 250),
    "mauve": (203, 166, 247),
    "teal": (148, 226, 213),
    "surface": (88, 91, 112),
}


def _tty() -> bool:
    return sys.stdout.isatty()


def paint(text: str, colour: str, bold: bool = False) -> str:
    """Wrap text in a palette colour."""
    if not _tty() or colour not in MOCHA:
        return text
    r, g, b = MOCHA[colour]
    prefix = "\033[1m" if bold else ""
    return f"{prefix}\033[38;2;{r};{g};{b}m{text}\033[0m"


def heading(text: str) -> None:
    print(paint(text, "mauve", bold=True))


def rule(width: int = 64) -> None:
    print(paint("─" * width, "surface"))


def grade(value: float, good: float, bad: float) -> str:
    """Colour a number by where it falls between a good and a bad threshold."""
    span = good - bad
    pos = (value - bad) / span if span else 0.0
    colour = "green" if pos > 0.66 else "yellow" if pos > 0.33 else "red"
    return paint(f"{value:+.4f}", colour)
