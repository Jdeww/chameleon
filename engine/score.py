"""Scores a candidate layout; `resize()` keeps the highest-scoring one.

Rewards legible text and well-sized imagery (relative to an
area-preserving scale of the source), and canvas coverage. Penalises
content units overlapping each other and text that doesn't fit its box.
Decor and background are excluded - they're meant to sit behind content.
"""

import math

from engine.analyze import ROLE_WEIGHT, Analysis
from engine.geometry import area, intersection, union
from engine.layouts import Placement

OVERLAP_PENALTY = 4.0
OVERFLOW_PENALTY = 2.0
ILLEGIBLE_PENALTY = 3.0
# Below these px a role reads as illegible (capped at the source size).
LEGIBLE_PX = {"headline": 12, "legal": 4}
DEFAULT_LEGIBLE_PX = 6
# Below this short side (fraction of the canvas short side) an image reads
# as missing.
VISIBLE_SIDE = {"hero": 0.2, "logo": 0.06}
FIDELITY_WEIGHT = 3.0
FIDELITY_FADE = 0.5  # |log aspect change| at which faithfulness stops counting


def score(pl: Placement, an: Analysis, W: float, H: float) -> float:
    s_ref = math.sqrt((W * H) / (an.width * an.height))
    total = 0.0

    for u, box in pl.unit_boxes:
        weight = ROLE_WEIGHT.get(u.role, 0.2)
        for el in u.members:
            out = pl.elements[el["id"]]
            if el["type"] == "text":
                ratio = out["font_size"] / (el["font_size"] * s_ref)
                total += weight * min(ratio, 1.3)
                floor = min(LEGIBLE_PX.get(u.role, DEFAULT_LEGIBLE_PX), el["font_size"])
                if out["font_size"] < floor:
                    # A fixed step (any illegible text is a visible defect)
                    # plus a part that grows with how far below it is.
                    total -= max(weight, 0.4) * (1.0 + ILLEGIBLE_PENALTY * (1 - out["font_size"] / floor))
        if u.role in ("hero", "logo", "cta"):
            ratio = math.sqrt(area(box) / max(u.w * u.h, 1e-6)) / s_ref
            total += weight * min(ratio, 1.3)
        if u.role in VISIBLE_SIDE:
            side = min(box[2] - box[0], box[3] - box[1])
            need = VISIBLE_SIDE[u.role] * min(W, H)
            if side < need:
                total -= weight * (1.0 + ILLEGIBLE_PENALTY * (1 - side / need))

    boxes = [b for _, b in pl.unit_boxes]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            inter = intersection(boxes[i], boxes[j])
            if inter:
                smaller = max(min(area(boxes[i]), area(boxes[j])), 1e-6)
                total -= OVERLAP_PENALTY * area(inter) / smaller

    total -= OVERFLOW_PENALTY * pl.overflow

    if boxes:
        coverage = area(union(boxes)) / (W * H)
        total += min(coverage / 0.75, 1.0)

    # Faithfulness: when the aspect ratio barely changes, the original
    # arrangement is the designer's intent - reward keeping units near their
    # relative positions. Fades out as the target shape diverges.
    r = abs(math.log((W / H) / (an.width / an.height)))
    closeness = max(0.0, 1.0 - r / FIDELITY_FADE)
    if closeness and pl.unit_boxes:
        disp = sum(
            math.hypot(
                (b[0] + b[2]) / (2 * W) - (u.x + u.w / 2) / an.width,
                (b[1] + b[3]) / (2 * H) - (u.y + u.h / 2) / an.height,
            )
            for u, b in pl.unit_boxes
        ) / len(pl.unit_boxes)
        total += FIDELITY_WEIGHT * closeness * (1.0 - min(1.0, disp / 0.25))
    return total
