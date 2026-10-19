"""Source-design analysis.

The schema carries no semantic labels, so roles are guessed from geometry,
type, z-order and font size alone. Elements are split into:

- background: full-bleed elements, which simply re-cover the target canvas
- decor: faint, ungrouped shapes (and shapes bleeding off the canvas),
  which sit behind content and may overlap it
- units: everything else. All members of a `group_id` form one unit that
  is moved and scaled rigidly, so a label stays on its button.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from engine.geometry import Box, aabb, area, fill_alpha, intersection, union

BACKGROUND_COVERAGE = 0.85
DECOR_MAX_ALPHA = 0.5

# How much each role matters when scoring a layout (higher = protect more).
ROLE_WEIGHT = {
    "headline": 1.0,
    "cta": 0.9,
    "hero": 0.8,
    "logo": 0.7,
    "body": 0.6,
    "badge": 0.4,
    "ribbon": 0.4,
    "image": 0.3,
    "legal": 0.25,
    "graphic": 0.2,
}

# Legibility floors (px) used while sizing; relaxed if nothing else fits.
ROLE_MIN_FONT = {"headline": 14, "body": 9, "legal": 6}


@dataclass
class Unit:
    key: str
    members: List[Dict[str, Any]]
    box: Box  # source AABB of all members
    role: str = "graphic"

    @property
    def x(self) -> float:
        return self.box[0]

    @property
    def y(self) -> float:
        return self.box[1]

    @property
    def w(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def h(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def is_text(self) -> bool:
        """A lone, unrotated text element - its box can re-wrap freely."""
        el = self.members[0]
        return len(self.members) == 1 and el["type"] == "text" and not el.get("rotation")


@dataclass
class Analysis:
    width: float
    height: float
    background: List[Dict[str, Any]] = field(default_factory=list)
    decor: List[Dict[str, Any]] = field(default_factory=list)
    units: List[Unit] = field(default_factory=list)  # reading order

    def by_role(self, *roles: str) -> List[Unit]:
        return [u for u in self.units if u.role in roles]


def analyze(source: Dict[str, Any]) -> Analysis:
    W, H = float(source["canvas"]["width"]), float(source["canvas"]["height"])
    canvas: Box = (0.0, 0.0, W, H)
    an = Analysis(W, H)

    groups: Dict[str, List[Dict[str, Any]]] = {}
    singles: List[Dict[str, Any]] = []
    for el in source["elements"]:
        box = aabb(el)
        overlap = intersection(box, canvas)
        coverage = area(overlap) / (W * H) if overlap else 0.0
        if el["type"] != "text" and coverage >= BACKGROUND_COVERAGE:
            an.background.append(el)
        elif el.get("group_id"):
            groups.setdefault(el["group_id"], []).append(el)
        elif el["type"] == "shape" and _is_decor(el, box, W, H):
            an.decor.append(el)
        else:
            singles.append(el)

    units = [Unit(gid, members, union(aabb(m) for m in members)) for gid, members in groups.items()]
    units += [Unit(el["id"], [el], aabb(el)) for el in singles]
    _assign_roles(units, W, H)
    an.units = sorted(units, key=lambda u: (u.y, u.x))
    return an


def _is_decor(el: Dict[str, Any], box: Box, W: float, H: float) -> bool:
    alpha = float(el.get("opacity", 1) or 0) * fill_alpha(el.get("fill_color"))
    bleeds = box[0] < 0 or box[1] < 0 or box[2] > W or box[3] > H
    return alpha < DECOR_MAX_ALPHA or bleeds


def _assign_roles(units: List[Unit], W: float, H: float) -> None:
    canvas_area = W * H

    # Groups: rotated -> ribbon; shape + text -> button (largest is the CTA).
    buttons = []
    for u in units:
        if len(u.members) == 1:
            continue
        types = {m["type"] for m in u.members}
        if any(m.get("rotation") for m in u.members):
            u.role = "ribbon"
        elif "text" in types and "shape" in types:
            u.role = "badge"
            buttons.append(u)
        else:
            u.role = "graphic"
    if buttons:
        max(buttons, key=lambda u: (area(u.box), u.y)).role = "cta"

    # Images: small one in the top band -> logo; largest remaining -> hero.
    images = [u for u in units if len(u.members) == 1 and u.members[0]["type"] == "image"]
    for u in images:
        top_band = (u.y + u.h / 2) < 0.22 * H
        u.role = "logo" if top_band and area(u.box) < 0.12 * canvas_area else "image"
    rest = [u for u in images if u.role == "image"]
    if rest:
        hero = max(rest, key=lambda u: area(u.box))
        if area(hero.box) >= 0.04 * canvas_area:
            hero.role = "hero"

    # Lone texts: biggest font -> headline; small text near the bottom -> legal.
    texts = [u for u in units if len(u.members) == 1 and u.members[0]["type"] == "text"]
    if texts:
        top = max(texts, key=lambda u: (u.members[0]["font_size"], area(u.box)))
        top_size = top.members[0]["font_size"]
        for u in texts:
            size = u.members[0]["font_size"]
            if u is top:
                u.role = "headline"
            elif size <= 0.35 * top_size and (u.y + u.h / 2 > 0.88 * H or size <= 13):
                u.role = "legal"
            else:
                u.role = "body"
