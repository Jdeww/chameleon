"""Text measurement and font-size fitting.

Uses the renderer's own font lookup (brand-kit fonts via
`renderer/converter/brand_fonts.py`) and greedy wrap
(`renderer/converter/wrap.py`), so a size that fits here is the size that
fits in the rendered PNG. Sizes are whole pixels because cr-renderer
truncates `fontSize` to an int when it draws.

If cr-renderer/skia isn't importable, falls back to a rough per-character
estimate so the engine still runs (with less accurate fitting).
"""

import math
import textwrap
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional, Tuple

FONTS_DIR = Path(__file__).resolve().parent.parent / "brand_kit" / "fonts"

# Fallback metrics, as a fraction of font size (roughly Poppins/Lora).
_APPROX_CHAR_WIDTH = 0.55
_APPROX_ASCENT = 1.05
_APPROX_DESCENT = 0.35


@dataclass(frozen=True)
class TextBlock:
    size: int
    lines: int
    width: float  # widest wrapped line
    height: float  # top of the first line to the bottom of the last
    overflow: bool  # a word is wider than the box, or max_lines exceeded


@lru_cache(maxsize=1)
def _font_manager():
    try:
        from renderer.converter.brand_fonts import load_brand_font_manager

        return load_brand_font_manager(str(FONTS_DIR))
    except Exception:  # cr_renderer/skia missing, or no font files
        return None


def _font_key(el: dict[str, Any]) -> Tuple[str, str, str]:
    """(family, weight, style) the same way convert.py derives them."""
    bold = el.get("font_weight", "regular") == "bold"
    italic = el.get("font_style", "normal") == "italic"
    if bold and italic:
        style = "bolditalic"
    elif bold:
        style = "bold"
    elif italic:
        style = "italic"
    else:
        style = "regular"
    return el.get("font_family", ""), "bold" if bold else "regular", style


@lru_cache(maxsize=20000)
def _char_widths(family: str, weight: str, style: str, size: int, text: str) -> Tuple[float, ...]:
    fm = _font_manager()
    if fm is None:
        return tuple(size * _APPROX_CHAR_WIDTH for _ in text)
    from renderer.converter.wrap import char_widths

    return tuple(char_widths(fm, family, weight, style, float(size), text))


@lru_cache(maxsize=2000)
def _vertical_metrics(family: str, weight: str, style: str, size: int) -> Tuple[float, float]:
    """(ascent, descent) in px, both positive."""
    fm = _font_manager()
    if fm is None:
        return size * _APPROX_ASCENT, size * _APPROX_DESCENT
    import skia  # type: ignore

    font = skia.Font(skia.Typeface.MakeFromData(fm.lookup(family, weight, style)), size)
    metrics = font.getMetrics()
    return -metrics.fAscent, metrics.fDescent


def _wrap(text: str, widths: Tuple[float, ...], max_width: float, size: int) -> List[List[int]]:
    """Wrap and return each line as a list of character indices."""
    if _font_manager() is not None:
        from renderer.converter.wrap import wrap_text

        wrapped = wrap_text(text, list(widths), max_width)
    else:
        per_line = max(1, int(max_width / (size * _APPROX_CHAR_WIDTH)))
        wrapped = "\n".join(textwrap.wrap(text, per_line, break_long_words=False)) or text
    lines, current = [], []
    for i, c in enumerate(wrapped):
        if c == "\n":
            lines.append(current)
            current = []
        else:
            current.append(i)
    lines.append(current)
    return lines


def measure(el: dict[str, Any], size: int, width: float) -> TextBlock:
    """Lay out `el`'s content at `size` px wrapped to `width`."""
    size = max(1, int(size))
    content = str(el.get("content", ""))
    family, weight, style = _font_key(el)
    widths = _char_widths(family, weight, style, size, content)
    lines = _wrap(content, widths, width, size)

    if _font_manager() is not None:
        widest = max(sum(widths[i] for i in line if i < len(widths)) for line in lines)
    else:
        widest = max(len(line) for line in lines) * size * _APPROX_CHAR_WIDTH
    ascent, descent = _vertical_metrics(family, weight, style, size)
    line_height = float(el.get("line_height", 1.2) or 1.2)
    height = (len(lines) - 1) * size * line_height + ascent + descent

    max_lines = el.get("max_lines")
    overflow = widest > width + 0.5 or (bool(max_lines) and len(lines) > max_lines)
    return TextBlock(size, len(lines), widest, height, overflow)


def fit_font_size(
    el: dict[str, Any], width: float, height: float, max_size: int, min_size: int = 1
) -> TextBlock:
    """Largest whole-px size in [min_size, max_size] whose wrapped block
    fits `width` x `height`. If even `min_size` doesn't fit, returns the
    `min_size` block (with `overflow` set when it spills)."""
    max_size = max(1, int(max_size))
    min_size = max(1, min(int(min_size), max_size))

    def fits(block: TextBlock) -> bool:
        return not block.overflow and block.height <= height + 0.5

    best: Optional[TextBlock] = None
    lo, hi = min_size, max_size
    while lo <= hi:
        mid = (lo + hi) // 2
        block = measure(el, mid, width)
        if fits(block):
            best, lo = block, mid + 1
        else:
            hi = mid - 1
    if best is not None:
        return best
    block = measure(el, min_size, width)
    if block.height > height + 0.5 and not block.overflow:
        block = TextBlock(block.size, block.lines, block.width, block.height, True)
    return block


def fit_to_width(el: dict[str, Any], width: float, size: int, min_size: int = 1) -> TextBlock:
    """Largest size <= `size` at which no word overflows `width` (height is
    unconstrained)."""
    return fit_font_size(el, width, math.inf, size, min_size)
