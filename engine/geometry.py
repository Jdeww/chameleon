"""Small geometry helpers shared by the engine and the self-check.

Boxes are (x0, y0, x1, y1) tuples in canvas pixels.
"""

import math
import re
from typing import Any, Iterable, Optional, Tuple

Box = Tuple[float, float, float, float]

_ALPHA_RE = re.compile(r"rgba\(\s*[\d.]+\s*,\s*[\d.]+\s*,\s*[\d.]+\s*,\s*([\d.]+)\s*\)")


def aabb(el: dict[str, Any]) -> Box:
    """Axis-aligned bounding box of an element, accounting for `rotation`
    (degrees, about the element's center - same as the renderer)."""
    x, y, w, h = el["x"], el["y"], el["width"], el["height"]
    theta = math.radians(el.get("rotation", 0) or 0)
    if theta == 0:
        return (x, y, x + w, y + h)
    c, s = abs(math.cos(theta)), abs(math.sin(theta))
    bw, bh = w * c + h * s, w * s + h * c
    cx, cy = x + w / 2, y + h / 2
    return (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)


def union(boxes: Iterable[Box]) -> Box:
    boxes = list(boxes)
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a: Box, b: Box) -> Optional[Box]:
    box = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    return box if box[2] > box[0] and box[3] > box[1] else None


def fill_alpha(color: Optional[str]) -> float:
    """Alpha of an `rgba(r,g,b,a)` string; 1.0 for anything else."""
    match = _ALPHA_RE.match((color or "").strip())
    return float(match.group(1)) if match else 1.0
