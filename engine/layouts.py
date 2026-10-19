"""Candidate layouts for a target canvas.

Every candidate is built from two primitives:

- `_stack`: lays rows of units top-to-bottom inside a box. A row is a set of
  units that sat side by side in the source (e.g. three badges); it scales
  rigidly. A row holding one lone text element instead re-wraps to the box
  width. One global scale `k` is binary-searched so the stack fills the
  box height.
- `_columns`: splits the canvas into side-by-side columns (for wide
  targets), sizes rigid columns from their content, gives the rest to text
  columns, then runs `_stack` inside each.

`candidates()` yields a proportional re-map, a full-canvas stack, and -
for landscape targets - several column arrangements. `score.py` picks one.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from engine.analyze import ROLE_MIN_FONT, ROLE_WEIGHT, Analysis, Unit
from engine.geometry import Box, aabb, union
from engine.textfit import fit_to_width, measure

# Smallest px a label inside a rigid group (button, badge, ribbon) should
# shrink to while there's still room.
LABEL_MIN_FONT = 6

# How fast each role gives up space when a stack must shrink: its scale is
# k ** exponent for k < 1, so < 1 protects a role and > 1 sacrifices it first.
SHRINK_EXPONENT = {
    "headline": 0.6,
    "cta": 0.75,
    "body": 0.85,
    "logo": 1.0,
    "badge": 1.0,
    "legal": 1.0,
    "ribbon": 1.1,
    "graphic": 1.2,
    "hero": 1.25,
    "image": 1.4,
}


# Hero/logo short side as a fraction of the canvas short side, kept while
# there's room (the hero's box includes the asset's own transparent padding).
IMAGE_MIN_SIDE = {"hero": 0.3, "logo": 0.1}


def _shrink(k: float, units: List[Unit]) -> float:
    if k >= 1:
        return k
    return k ** min(SHRINK_EXPONENT.get(u.role, 1.0) for u in units)


def _label_floor(units: List[Unit]) -> float:
    """Smallest scale that keeps every label in these rigid units legible
    (never above source size)."""
    sizes = [m["font_size"] for u in units for m in u.members if m["type"] == "text"]
    # +1e-6 so int() truncation of font_size * f still lands on the floor.
    return min(1.0, max((LABEL_MIN_FONT + 1e-6) / s for s in sizes)) if sizes else 0.0


def _image_floor(ctx: "Ctx", units: List[Unit]) -> float:
    """Smallest scale that keeps hero/logo images visibly sized: their short
    side at least IMAGE_MIN_SIDE of the canvas's short side."""
    floors = [
        IMAGE_MIN_SIDE[u.role] * min(ctx.W, ctx.H) / max(min(u.w, u.h), 1e-6)
        for u in units
        if u.role in IMAGE_MIN_SIDE
    ]
    return max(floors, default=0.0)


def _text_demand(units: List[Unit]) -> float:
    """Priority-weighted area (px^2) a column's lone texts need at their floors -
    used to share width between text columns."""
    demand = 0.0
    for u in units:
        if u.is_text:
            el = u.members[0]
            size = min(el["font_size"], ROLE_MIN_FONT.get(u.role, LABEL_MIN_FONT))
            demand += ROLE_WEIGHT.get(u.role, 0.5) * len(str(el.get("content", ""))) * size * size
    return demand


@dataclass
class Placement:
    name: str
    elements: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # id -> output element
    unit_boxes: List[Tuple[Unit, Box]] = field(default_factory=list)
    overflow: int = 0  # text elements that don't fit their box


@dataclass
class Ctx:
    an: Analysis
    W: float
    H: float

    @property
    def sx(self) -> float:
        return self.W / self.an.width

    @property
    def sy(self) -> float:
        return self.H / self.an.height

    @property
    def s_min(self) -> float:
        return min(self.sx, self.sy)

    @property
    def rigid_cap(self) -> float:
        """Max scale for images/groups - modest growth to use spare room,
        more only if the whole canvas grew."""
        return 1.5 * max(1.0, self.s_min)

    @property
    def text_cap(self) -> float:
        return self.rigid_cap

    @property
    def margin(self) -> float:
        return max(4.0, 0.05 * min(self.W, self.H))


@dataclass
class Row:
    units: List[Unit]
    box: Box

    @property
    def w(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def h(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def flexible(self) -> bool:
        return len(self.units) == 1 and self.units[0].is_text


# ---------------------------------------------------------------------------
# Placing units
# ---------------------------------------------------------------------------


def _place_rigid(pl: Placement, units: List[Unit], origin: Tuple[float, float], tx: float, ty: float, f: float) -> None:
    """Scale `units` by `f` about source point `origin`, landing it at (tx, ty)."""
    ox, oy = origin
    for u in units:
        for el in u.members:
            out = dict(el)
            w, h = el["width"] * f, el["height"] * f
            cx, cy = el["x"] + el["width"] / 2, el["y"] + el["height"] / 2
            out["x"] = tx + (cx - ox) * f - w / 2
            out["y"] = ty + (cy - oy) * f - h / 2
            out["width"], out["height"] = w, h
            if el["type"] == "text":
                out["font_size"] = max(1, int(el["font_size"] * f))
                if el.get("letter_spacing"):
                    out["letter_spacing"] = el["letter_spacing"] * f
            elif el["type"] == "shape":
                for key in ("corner_radius", "stroke_width"):
                    if el.get(key):
                        out[key] = el[key] * f
            pl.elements[el["id"]] = out
        pl.unit_boxes.append((u, union(aabb(pl.elements[m["id"]]) for m in u.members)))


def _place_text(pl: Placement, u: Unit, x: float, y: float, width: float, size: int) -> float:
    """Place a lone text unit with its box wrapped to `width`; returns height."""
    el = u.members[0]
    block = measure(el, size, width)
    out = dict(el)
    out.update(x=x, y=y, width=width, height=block.height, font_size=block.size)
    pl.elements[el["id"]] = out
    pl.unit_boxes.append((u, (x, y, x + width, y + block.height)))
    if block.overflow:
        pl.overflow += 1
    return block.height


# ---------------------------------------------------------------------------
# Stack
# ---------------------------------------------------------------------------


def _rows(units: List[Unit], split: bool) -> List[Row]:
    """Cluster units into rows by vertical overlap in the source."""
    rows: List[Row] = []
    for u in sorted(units, key=lambda u: (u.y, u.x)):
        last = rows[-1] if rows else None
        if not split and last is not None and not last.flexible and not u.is_text:
            overlap = min(last.box[3], u.box[3]) - max(last.box[1], u.box[1])
            if overlap >= 0.5 * min(last.h, u.h):
                last.units.append(u)
                last.box = union([last.box, u.box])
                continue
        rows.append(Row([u], u.box))
    return rows


def _wrap_row(row: Row, width: float) -> List[Row]:
    """Wrap a side-by-side row onto several lines when it's too wide to keep
    its labels legible at `width` (e.g. three badges -> two + one)."""
    floor = _label_floor(row.units)
    if len(row.units) == 1 or row.w * floor <= width:
        return [row]
    lines: List[Row] = []
    for u in sorted(row.units, key=lambda u: u.x):
        if lines:
            merged = union([lines[-1].box, u.box])
            if (merged[2] - merged[0]) * floor <= width:
                lines[-1].units.append(u)
                lines[-1].box = merged
                continue
        lines.append(Row([u], u.box))
    return lines


def _text_size(ctx: Ctx, u: Unit, k: float, floor_scale: float) -> int:
    el = u.members[0]
    src = el["font_size"]
    floor = min(src, ROLE_MIN_FONT.get(u.role, 1) * floor_scale)
    return max(1, int(max(floor, src * _shrink(k, [u]))))


def _stack(ctx: Ctx, pl: Placement, units: List[Unit], box: Box, split: bool = False) -> None:
    bx, by, bw, bh = box[0], box[1], box[2] - box[0], box[3] - box[1]
    rows = _rows(units, split)
    if not rows:
        return
    rows = [r for row in rows for r in _wrap_row(row, bw)]

    def gap(i: int, k: float) -> float:
        prev, nxt = rows[i - 1], rows[i]
        src_gap = max(0.0, nxt.box[1] - prev.box[3])
        return min(src_gap * k, 0.1 * bh) + 0.012 * bh

    def layout(k: float, floor_scale: float) -> Tuple[float, List[Tuple[float, Any]]]:
        sizes, total = [], 0.0
        for i, row in enumerate(rows):
            if row.flexible:
                u = row.units[0]
                block = fit_to_width(u.members[0], bw, _text_size(ctx, u, k, floor_scale))
                sizes.append((block.height, block.size))
                total += block.height
            else:
                fit_w = bw / max(row.w, 1e-6)
                f = min(_shrink(k, row.units), ctx.rigid_cap, fit_w)
                floor = max(_label_floor(row.units), _image_floor(ctx, row.units))
                f = max(f, min(floor * floor_scale, fit_w, ctx.rigid_cap))
                sizes.append((row.h * f, f))
                total += row.h * f
            if i:
                total += gap(i, k)
        return total, sizes

    k, floor_scale = 1e-3, 0.0
    for fs in (1.0, 0.85, 0.7, 0.6, 0.45, 0.0):
        if layout(1e-3, fs)[0] <= bh:
            floor_scale = fs
            lo, hi = 1e-3, ctx.text_cap
            if layout(hi, fs)[0] <= bh:
                lo = hi
            else:
                for _ in range(28):
                    mid = (lo + hi) / 2
                    if layout(mid, fs)[0] <= bh:
                        lo = mid
                    else:
                        hi = mid
            k = lo
            break

    total, sizes = layout(k, floor_scale)
    free = max(0.0, bh - total)
    extra = min(free / (len(rows) + 1), 0.12 * bh)
    y = by + (free - extra * (len(rows) - 1)) / 2

    for i, (row, (h, val)) in enumerate(zip(rows, sizes)):
        if i:
            y += gap(i, k) + extra
        if row.flexible:
            _place_text(pl, row.units[0], bx, y, bw, val)
        else:
            rw = row.w * val
            hpos = row.box[0] / max(1.0, ctx.an.width - row.w)
            x = bx + (bw - rw) * min(1.0, max(0.0, hpos))
            _place_rigid(pl, row.units, (row.box[0], row.box[1]), x, y, val)
        y += h


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------


def _columns(ctx: Ctx, name: str, cols: List[List[Unit]], split: bool, alpha: float = 1.0) -> Optional[Placement]:
    """`alpha` scales rigid (image/button) columns' natural width - smaller
    hands more room to text - but never below what keeps labels legible."""
    cols = [c for c in cols if c]
    if len(cols) < 2:
        return None
    pl = Placement(name)
    m = ctx.margin
    avail_w, avail_h = ctx.W - 2 * m, ctx.H - 2 * m
    gutter = max(m, 0.015 * ctx.W)
    inner_w = avail_w - gutter * (len(cols) - 1)
    if inner_w <= 0:
        return None

    widths: List[Optional[float]] = []
    legible: List[float] = []
    for col in cols:
        if any(u.is_text for u in col):
            widths.append(None)
            legible.append(0.0)
            continue
        rows = _rows(col, split)
        stack_h = sum(r.h for r in rows) * (1 + 0.04 * (len(rows) - 1))
        k = min(ctx.rigid_cap, avail_h / max(stack_h, 1e-6))
        legible_w = min(max(u.w for u in col) * _label_floor(col), 0.35 * inner_w)
        widths.append(min(max(max(r.w for r in rows) * k * alpha, legible_w), 0.35 * inner_w))
        legible.append(legible_w)

    n_flex = widths.count(None)
    fixed = sum(w for w in widths if w is not None)
    min_flex = 0.22 * inner_w * n_flex
    over = fixed - (inner_w - min_flex)
    if over > 0 and fixed > 0:
        # Take width back from what's above each column's legible minimum
        # first; only scale below it if that's still not enough.
        excess = sum(w - lw for w, lw in zip(widths, legible) if w is not None)
        if over <= excess:
            cut = over / excess if excess else 0.0
            widths = [None if w is None else w - (w - lw) * cut for w, lw in zip(widths, legible)]
        else:
            shrink = max(0.0, inner_w - min_flex) / max(sum(legible), 1e-6)
            widths = [None if w is None else lw * shrink for w, lw in zip(widths, legible)]
        fixed = sum(w for w in widths if w is not None)
    # Text columns share what's left in proportion to how much text they hold.
    demands = [_text_demand(col) if w is None else 0.0 for col, w in zip(cols, widths)]
    flex_total = inner_w - fixed
    spare = 0.0 if n_flex else flex_total
    gap = gutter + spare / (len(cols) + 1)

    x = m + (spare / (len(cols) + 1) if not n_flex else 0.0)
    for col, w, demand in zip(cols, widths, demands):
        if w is None:
            share = demand / sum(demands) if sum(demands) else 1.0 / n_flex
            cw = flex_total * (0.7 * share + 0.3 / n_flex)  # damped toward an even split
        else:
            cw = w
        _stack(ctx, pl, col, (x, m, x + cw, m + avail_h), split)
        x += cw + gap
    return pl


# ---------------------------------------------------------------------------
# Proportional re-map (good when the aspect ratio barely changes)
# ---------------------------------------------------------------------------


def _proportional(ctx: Ctx) -> Placement:
    pl = Placement("proportional")
    s = ctx.s_min
    for u in ctx.an.units:
        cx = (u.x + u.w / 2) * ctx.sx
        cy = (u.y + u.h / 2) * ctx.sy
        w, h = u.w * s, u.h * s
        x = min(max(cx - w / 2, 0.0), max(0.0, ctx.W - w))
        y = min(max(cy - h / 2, 0.0), max(0.0, ctx.H - h))
        if u.is_text:
            el = u.members[0]
            _place_text(pl, u, x, y, w, max(1, int(el["font_size"] * s)))
        else:
            _place_rigid(pl, [u], (u.x, u.y), x, y, s)
    return pl


# ---------------------------------------------------------------------------
# Background + decor (shared by every candidate)
# ---------------------------------------------------------------------------


def _finish(ctx: Ctx, pl: Placement) -> Placement:
    for el in ctx.an.background:
        out = dict(el)
        out.update(x=0.0, y=0.0, width=ctx.W, height=ctx.H)
        pl.elements[el["id"]] = out

    for el in ctx.an.decor:
        out = dict(el)
        f = min(ctx.s_min, 0.9 * min(ctx.W, ctx.H) / max(el["width"], el["height"], 1e-6))
        w, h = el["width"] * f, el["height"] * f
        cx = (el["x"] + el["width"] / 2) * ctx.sx
        cy = (el["y"] + el["height"] / 2) * ctx.sy
        out["width"], out["height"] = w, h
        out["x"] = min(max(cx - w / 2, 0.0), ctx.W - w)
        out["y"] = min(max(cy - h / 2, 0.0), ctx.H - h)
        if el.get("stroke_width"):
            out["stroke_width"] = el["stroke_width"] * f
        pl.elements[el["id"]] = out
    return pl


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def candidates(an: Analysis, target: Dict[str, Any]) -> Iterator[Placement]:
    ctx = Ctx(an, float(target["width"]), float(target["height"]))
    m = ctx.margin
    full = (m, m, ctx.W - m, ctx.H - m)

    yield _finish(ctx, _proportional(ctx))
    for split in (False, True):
        pl = Placement(f"stack{'-split' if split else ''}")
        _stack(ctx, pl, an.units, full, split)
        yield _finish(ctx, pl)

    # Columns only when the target is landscape AND noticeably wider than
    # the source - a same-shape target should keep the source arrangement.
    if ctx.W / ctx.H < 1.0 or math.log((ctx.W / ctx.H) / (an.width / an.height)) < 0.15:
        return

    def pick(*roles: str) -> List[Unit]:
        return [u for u in an.units if u.role in roles]

    hero = pick("hero")
    brand = pick("logo")
    text = pick("headline", "body")
    legal = pick("legal")
    action = pick("cta")
    extras = pick("badge", "ribbon", "image", "graphic")
    others = [u for u in an.units if u not in hero]

    def ordered(*groups: List[Unit]) -> List[Unit]:
        chosen = {id(u) for g in groups for u in g}
        return [u for u in an.units if id(u) in chosen]

    layouts = {
        "cols:hero|rest": [hero, others],
        "cols:hero|text|side": [hero, ordered(text, legal), ordered(brand, extras, action)],
        "cols:hero|brand+text|side": [hero, ordered(brand, text, legal), ordered(extras, action)],
        "cols:side|hero|text|cta": [ordered(brand, extras), hero, ordered(text, legal), action],
    }
    if ctx.W / ctx.H >= 4:
        layouts["cols:brand|hero|text|extras|cta"] = [brand, hero, text, ordered(extras, legal), action]
        layouts["cols:brand|hero|head|body|extras|cta"] = [
            brand, hero, pick("headline"), ordered(pick("body"), legal), extras, action,
        ]
        layouts["cols:brand+ribbon|hero|head|body|badges|cta+img"] = [
            ordered(brand, pick("ribbon")), hero, pick("headline"), ordered(pick("body"), legal),
            pick("badge"), ordered(action, pick("image", "graphic")),
        ]

    seen = set()
    for name, cols in layouts.items():
        sig = tuple(tuple(id(u) for u in c) for c in cols if c)
        if sig in seen:
            continue
        seen.add(sig)
        for split in (False, True):
            for alpha in (1.0, 0.65, 0.4):
                label = f"{name}{'-split' if split else ''}@{alpha}"
                pl = _columns(ctx, label, cols, split, alpha)
                if pl is not None:
                    yield _finish(ctx, pl)
