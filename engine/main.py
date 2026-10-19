"""
Orchestrator — you shouldn't need to edit this file. It:
  1. Reads every *.json task file from /app/input (source + target_canvas)
  2. Calls your resize() from resize.py
  3. Runs a quick self-check (structural only — NOT the official scoring;
     see RULES.md for what the judge actually checks)
  4. Writes <task>_output.json and <task>_output.png to /app/output

Mirrors the same input/ -> output/ convention as the earlier India-program
challenges: drop task files in, get one JSON + one PNG out per task, no
manual steps in between.
"""

import json
import sys
import time
from pathlib import Path

from engine.geometry import aabb
from engine.resize import resize

# Relative to the current working directory, which is /app both inside the
# Docker image (WORKDIR /app) and when run locally from this repo's root -
# so this works identically in both "python -m engine.main" (no Docker) and
# the containerized entrypoint.
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
ASSETS_DIR = Path("brand_kit/assets")

REQUIRED_COMMON = ["id", "type", "x", "y", "width", "height"]
REQUIRED_BY_TYPE = {
    "text": ["content", "font_family", "font_size", "color"],
    "image": ["asset_id"],
    "shape": ["shape_type", "fill_color"],
}


def self_check(source: dict, output: dict, target_canvas: dict | None = None) -> list[str]:
    """Quick, non-authoritative sanity check so obvious mistakes are caught
    locally before you submit. This is NOT the official scorer — see
    RULES.md for what actually gates and grades your submission."""
    problems = []

    if "canvas" not in output or "elements" not in output:
        return ["output is missing top-level 'canvas' or 'elements'"]

    src_ids = [el["id"] for el in source["elements"]]
    out_ids = [el.get("id") for el in output["elements"]]
    if set(src_ids) != set(out_ids):
        problems.append(
            f"element id mismatch — missing {sorted(set(src_ids) - set(out_ids))}, "
            f"unexpected {sorted(set(out_ids) - set(src_ids), key=str)}"
        )
    dupes = sorted({i for i in out_ids if out_ids.count(i) > 1}, key=str)
    if dupes:
        problems.append(f"duplicate element ids: {dupes}")

    cw, ch = output["canvas"].get("width"), output["canvas"].get("height")
    if target_canvas and (cw, ch) != (target_canvas["width"], target_canvas["height"]):
        problems.append(f"canvas {cw}x{ch} != target {target_canvas['width']}x{target_canvas['height']}")

    src = {el["id"]: el for el in source["elements"]}
    for el in output["elements"]:
        missing = [f for f in REQUIRED_COMMON if f not in el]
        missing += [f for f in REQUIRED_BY_TYPE.get(el.get("type"), []) if f not in el]
        if missing:
            problems.append(f"element '{el.get('id')}' (type={el.get('type')}): missing {missing}")
            continue
        if cw and ch:
            x0, y0, x1, y1 = aabb(el)  # rotated bounds, not just the unrotated box
            if x0 < -0.5 or y0 < -0.5 or x1 > cw + 0.5 or y1 > ch + 0.5:
                problems.append(f"element '{el['id']}' is off-canvas ({cw}x{ch})")
        s = src.get(el["id"])
        if s is None:
            continue
        if el["type"] == "text":
            if el["content"] != s["content"] or el["font_family"] != s["font_family"]:
                problems.append(f"element '{el['id']}': content or font_family changed")
            if el["font_size"] < 1:
                problems.append(f"element '{el['id']}': font_size < 1")
        if el["type"] == "image" and s.get("fit") != "cover" and min(s["height"], el["height"]) > 0:
            drift = abs((el["width"] / el["height"]) / (s["width"] / s["height"]) - 1)
            if drift > 0.05:
                problems.append(f"element '{el['id']}': aspect ratio off by {drift:.0%}")

    return problems


def render_to_png(design: dict, assets_dir: Path, png_path: Path) -> list[str]:
    """Renders `design` to a PNG and returns the renderer's warnings (e.g.
    missing assets). Prefers the real renderer (transformer-convertor,
    wrapping cr-renderer) in ./renderer; falls back to a plain
    box-and-label preview if it can't be imported — and says so, so a
    broken install isn't mistaken for a real preview."""
    try:
        from renderer.converter.convert import render_design_to_png  # type: ignore
    except ImportError as exc:
        _fallback_render(design, png_path)
        return [f"real renderer unavailable ({exc}) - wrote placeholder boxes instead"]
    return render_design_to_png(design, assets_dir, str(png_path))


def _fallback_render(design: dict, png_path: Path) -> None:
    """Placeholder only — labeled boxes, not real text/asset rendering.
    Confirms the pipeline runs end-to-end; do not use this to judge your
    actual layout quality. Use transformer-convertor (RULES.md / FAQ Q3)
    for a real preview."""
    from PIL import Image, ImageDraw

    w, h = design["canvas"]["width"], design["canvas"]["height"]
    img = Image.new("RGB", (w, h), "#EDE7DD")
    draw = ImageDraw.Draw(img)
    palette = {"text": "#2B6CB0", "image": "#B7791F", "shape": "#2F855A"}
    for el in sorted(design["elements"], key=lambda e: e.get("z_index", 0)):
        x0, y0 = el["x"], el["y"]
        x1, y1 = x0 + el["width"], y0 + el["height"]
        color = palette.get(el["type"], "#555555")
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.text((x0 + 4, y0 + 4), el["id"], fill=color)
    img.save(png_path)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    task_files = sorted(INPUT_DIR.glob("*.json"))
    if not task_files:
        print(f"No task files found in {INPUT_DIR}", file=sys.stderr)
        return 1

    exit_code = 0
    for task_path in task_files:
        # One bad task must not stop the rest from being produced.
        try:
            task = json.loads(task_path.read_text(encoding="utf-8"))
            source, target_canvas = task["source"], task["target_canvas"]

            t0 = time.time()
            output = resize(source, target_canvas)
            elapsed = time.time() - t0

            problems = self_check(source, output, target_canvas)
            stem = task_path.stem
            (OUTPUT_DIR / f"{stem}_output.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
            render_warnings = render_to_png(output, ASSETS_DIR, OUTPUT_DIR / f"{stem}_output.png")
        except Exception as exc:
            exit_code = 1
            print(f"[{task_path.name}] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        status = "OK" if not problems else f"{len(problems)} self-check warning(s)"
        print(f"[{task_path.name}] {status} in {elapsed:.2f}s "
              f"-> {stem}_output.json, {stem}_output.png")
        for p in problems:
            print(f"    - {p}")
        for w in render_warnings:
            print(f"    (renderer) {w}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
