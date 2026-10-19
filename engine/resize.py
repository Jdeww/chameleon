"""
THIS IS THE FILE YOU IMPLEMENT.

Everything else in this repo (input loading, output writing, rendering,
the Dockerfile) is plumbing so you can focus on exactly one function:
`resize()`. See RULES.md and the design schema for the full contract.
"""

from typing import Any

from engine.analyze import analyze
from engine.geometry import aabb
from engine.layouts import candidates
from engine.score import score


def resize(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    """
    Args:
        source: {"canvas": {"width": int, "height": int}, "elements": [...]}
                 — the design at its original size. See
                 chameleon_design_schema_v0.9.json for the full element spec
                 (text / image / shape, position, size, rotation, opacity,
                 z_index, group_id, and type-specific fields).
        target_canvas: {"width": int, "height": int} — the shape you're
                 resizing into.

    Returns:
        A design dict with the SAME shape as `source`:
        {"canvas": target_canvas, "elements": [...]}

        Hard requirements (see RULES.md for the full list):
        - Every element `id` in `source` must appear exactly once in your
          output. Do not add, remove, or invent elements.
        - Do not regenerate or redraw any asset's pixel content — you are
          repositioning and resizing the elements you were given, nothing
          more. This is a disqualifying rule, not a style preference.
        - Non-background image elements should be kept in proportion
          (matching the source aspect ratio within ~5%) unless they are a
          full-bleed / `fit: "cover"` element, which is expected to crop.
        - `font_size` may change to help text fit. `font_family` and
          `content` should not, for this stage of scoring (see RULES.md,
          "Adaptive typography and copy").
        - Nothing should end up outside `target_canvas`'s bounds.

    Approach (layout retargeting, no ML):
      1. analyze.py  - split elements into background / decor / units
                       (a group_id is one rigid unit) and guess roles.
      2. layouts.py  - build candidate layouts: proportional re-map,
                       vertical stacks, and column arrangements for wide
                       targets. Text is re-fitted with real glyph metrics.
      3. score.py    - pick the candidate with the most legible text,
                       best-sized imagery and no content overlaps.
      4. below       - assemble every source id exactly once and clamp
                       anything still poking outside the canvas.
    """
    an = analyze(source)
    W, H = float(target_canvas["width"]), float(target_canvas["height"])
    best = max(candidates(an, target_canvas), key=lambda pl: score(pl, an, W, H))

    out_elements = []
    for el in source["elements"]:
        out = dict(best.elements.get(el["id"], el))  # never drop an id
        _clamp_into(out, W, H)
        for key in ("x", "y", "width", "height"):
            out[key] = round(out[key], 2)
        out_elements.append(out)

    return {
        "canvas": {"width": target_canvas["width"], "height": target_canvas["height"]},
        "elements": out_elements,
    }


def _clamp_into(el: dict[str, Any], W: float, H: float) -> None:
    """Shrink (uniformly, about the center) and shift `el` so its rotated
    bounding box lies inside the canvas."""
    x0, y0, x1, y1 = aabb(el)
    bw, bh = x1 - x0, y1 - y0
    f = min(1.0, W / bw if bw else 1.0, H / bh if bh else 1.0)
    if f < 1.0:
        cx, cy = el["x"] + el["width"] / 2, el["y"] + el["height"] / 2
        el["width"] *= f
        el["height"] *= f
        el["x"], el["y"] = cx - el["width"] / 2, cy - el["height"] / 2
        if el["type"] == "text":
            el["font_size"] = max(1, int(el["font_size"] * f))
        x0, y0, x1, y1 = aabb(el)
    el["x"] += max(0.0, -x0) - max(0.0, x1 - W)
    el["y"] += max(0.0, -y0) - max(0.0, y1 - H)
