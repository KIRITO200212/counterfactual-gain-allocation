"""Pixel-space counterfactual operators for the Phase-1 revision.

Implements section 4 of ``results/phase1/phase1_plan.md``.  All operators
work on raw PIL pixels so that interventions re-enter the frozen model
through the complete ``processor -> ViT -> LLM`` canonical path, the same
route validated by the global-hflip positive control.

Frozen canonical rules (v1, recorded in the run contract):

- integer boxes: floor/ceil rounding of GT float boxes, clamped to the image;
- eligibility requires the INTEGER boxes to be disjoint, not only the float
  boxes (plan7 section 11 governance fix): crop/paste operate on the
  floor/ceil-expanded integer boxes, so a sub-pixel float gap can still
  share pixels; such pairs are rejected with ``integer_box_overlap``;
- intermediate canonical grid: each crop is resized to its own box size
  rounded to the nearest 28-px multiple (the processor token edge), with a
  28-px floor;
- every edit branch performs exactly two bicubic resizes through that grid,
  so the resampling null matches the main branch in resize count and
  intermediate raster;
- interpolation: ``PIL.Image.Resampling.BICUBIC``;
- paste writes the resized raster into the integer target box origin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from PIL import Image

# Frozen processor identity (models/SpatialLadder-3B/preprocessor_config.json).
PATCH_SIZE = 14
MERGE_SIZE = 2
TOKEN_EDGE = PATCH_SIZE * MERGE_SIZE
MIN_PIXELS = 12544
MAX_PIXELS = 100352

# Frozen eligibility thresholds from the read-only support audit
# (reports/support_audit_pixel.md).  They were fixed once from offline
# geometry distributions before any model output was consulted.
UNION_AREA_RATIO_MAX = 0.35
BOX_SIZE_RATIO_MAX = 4.0
RESIZED_MIN_TOKEN_SUPPORT_MIN = 1.0
COMPACT_ENVELOPE_AREA_RATIO_MAX = 0.35

# Frozen genuine C/D matching criteria (same values as the support audit).
CD_AREA_FLOOR_OF_SMALLER = 0.05
CD_AREA_RATIO_MIN = 0.1
CD_DISTANCE_BAND = (0.25, 4.0)

INTERPOLATION = Image.Resampling.BICUBIC


@dataclass(frozen=True)
class PixelOperatorResult:
    """One deterministic pixel-space edit attempt with full audit metadata."""

    accepted: bool
    reject_reason: str
    image: Image.Image | None
    metadata: dict[str, Any] = field(default_factory=dict)


def processor_resized_size(height: int, width: int) -> tuple[int, int]:
    """Replicate the frozen Qwen processor resolution rounding (read-only audit)."""

    factor = TOKEN_EDGE
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > MAX_PIXELS:
        beta = math.sqrt((height * width) / MAX_PIXELS)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    if h_bar * w_bar < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def integer_box(box: Sequence[float], image: Image.Image) -> tuple[int, int, int, int]:
    """Round a float GT box to deterministic integer pixel coordinates."""

    x0 = max(0, math.floor(float(box[0])))
    y0 = max(0, math.floor(float(box[1])))
    x1 = min(image.width, math.ceil(float(box[2])))
    y1 = min(image.height, math.ceil(float(box[3])))
    return x0, y0, x1, y1


def box_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1])
    )


def intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    x0, y0 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x1, y1 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def box_center(box: Sequence[float]) -> tuple[float, float]:
    return (float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0


def canonical_grid(box: Sequence[float]) -> tuple[int, int]:
    """Frozen intermediate raster size for one crop (28-px quantized)."""

    width = max(TOKEN_EDGE, round((float(box[2]) - float(box[0])) / TOKEN_EDGE) * TOKEN_EDGE)
    height = max(TOKEN_EDGE, round((float(box[3]) - float(box[1])) / TOKEN_EDGE) * TOKEN_EDGE)
    return width, height


def resized_min_token_support(
    box_a: Sequence[float], box_b: Sequence[float], image: Image.Image
) -> float:
    """Smaller expected token footprint at the frozen processor resolution."""

    resized_h, resized_w = processor_resized_size(image.height, image.width)
    scale_x = resized_w / image.width
    scale_y = resized_h / image.height

    def tokens(box: Sequence[float]) -> float:
        w = max(0.0, (float(box[2]) - float(box[0])) * scale_x)
        h = max(0.0, (float(box[3]) - float(box[1])) * scale_y)
        return (w / TOKEN_EDGE) * (h / TOKEN_EDGE)

    return min(tokens(box_a), tokens(box_b))


def base_eligibility(
    image: Image.Image, box_a: Sequence[float], box_b: Sequence[float]
) -> tuple[str | None, dict[str, Any]]:
    """Shared v1 eligibility gate; returns (reject_reason or None, stats)."""

    stats: dict[str, Any] = {}
    area_a, area_b = box_area(box_a), box_area(box_b)
    image_area = float(image.width * image.height)
    overlap = intersection_area(box_a, box_b)
    union = area_a + area_b - overlap
    stats["area_a"] = area_a
    stats["area_b"] = area_b
    stats["overlap_pixels"] = overlap
    stats["union_area_ratio"] = union / image_area
    if area_a <= 0.0 or area_b <= 0.0:
        return "empty_box", stats
    ix0, iy0, ix1, iy1 = integer_box(box_a, image)
    jx0, jy0, jx1, jy1 = integer_box(box_b, image)
    if ix1 <= ix0 or iy1 <= iy0 or jx1 <= jx0 or jy1 <= jy0:
        return "degenerate_integer_box", stats
    stats["box_size_ratio"] = max(area_a, area_b) / min(area_a, area_b)
    stats["resized_min_token_support"] = resized_min_token_support(box_a, box_b, image)
    if overlap > 0:
        return "overlapping_boxes", stats
    # crop/paste consume the floor/ceil-expanded integer boxes; float-disjoint
    # boxes with a sub-pixel gap can still share pixels after rounding.
    integer_overlap = float(
        max(0, min(ix1, jx1) - max(ix0, jx0)) * max(0, min(iy1, jy1) - max(iy0, jy0))
    )
    stats["integer_overlap_pixels"] = integer_overlap
    if integer_overlap > 0:
        return "integer_box_overlap", stats
    if stats["union_area_ratio"] > UNION_AREA_RATIO_MAX:
        return "union_area_over_cap", stats
    if stats["box_size_ratio"] > BOX_SIZE_RATIO_MAX:
        return "box_size_ratio_over_cap", stats
    if stats["resized_min_token_support"] < RESIZED_MIN_TOKEN_SUPPORT_MIN:
        return "insufficient_resized_token_support", stats
    return None, stats


def _two_stage_raster(
    image: Image.Image, box: Sequence[float], target_box: Sequence[float]
) -> Image.Image:
    """crop -> canonical grid -> target box size, exactly two resizes."""

    x0, y0, x1, y1 = integer_box(box, image)
    grid_w, grid_h = canonical_grid(box)
    tx0, ty0, tx1, ty1 = integer_box(target_box, image)
    crop = image.crop((x0, y0, x1, y1))
    intermediate = crop.resize((grid_w, grid_h), INTERPOLATION)
    final = intermediate.resize((tx1 - tx0, ty1 - ty0), INTERPOLATION)
    crop.close()
    intermediate.close()
    return final


def _paste(
    base: Image.Image, raster: Image.Image, target_box: Sequence[float]
) -> Image.Image:
    out = base.copy()
    tx0, ty0, _, _ = integer_box(target_box, base)
    out.paste(raster, (tx0, ty0))
    raster.close()
    return out


def pixel_pair_slot_swap(
    image: Image.Image, box_a: Sequence[float], box_b: Sequence[float]
) -> PixelOperatorResult:
    """Main operator: write A's raster into B's box and B's into A's."""

    reject, stats = base_eligibility(image, box_a, box_b)
    if reject is not None:
        return PixelOperatorResult(False, reject, None, stats)
    raster_a = _two_stage_raster(image, box_a, box_b)
    raster_b = _two_stage_raster(image, box_b, box_a)
    edited = _paste(image, raster_a, box_b)
    bx0, by0, _, _ = integer_box(box_a, image)
    edited.paste(raster_b, (bx0, by0))
    raster_b.close()
    stats["operator"] = "pixel_pair_slot_swap"
    stats["grid_a"] = canonical_grid(box_a)
    stats["grid_b"] = canonical_grid(box_b)
    return PixelOperatorResult(True, "accepted", edited, stats)


def canonical_resampling_return(
    image: Image.Image, box_a: Sequence[float], box_b: Sequence[float]
) -> PixelOperatorResult:
    """Null 1: identical crop/grid/resize/paste pipeline, written back in place."""

    reject, stats = base_eligibility(image, box_a, box_b)
    if reject is not None:
        return PixelOperatorResult(False, reject, None, stats)
    raster_a = _two_stage_raster(image, box_a, box_a)
    raster_b = _two_stage_raster(image, box_b, box_b)
    edited = _paste(image, raster_a, box_a)
    bx0, by0, _, _ = integer_box(box_b, image)
    edited.paste(raster_b, (bx0, by0))
    raster_b.close()
    stats["operator"] = "canonical_resampling_return"
    return PixelOperatorResult(True, "accepted", edited, stats)


def select_irrelevant_pair(
    box_a: Sequence[float],
    box_b: Sequence[float],
    inventory: Sequence[Mapping[str, Any]],
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]] | None:
    """Frozen deterministic C/D selection from a same-image GT inventory.

    Criteria mirror the read-only support audit: disjoint from A/B and from
    each other, area floors, and A/B-comparable center distance.  Among valid
    pairs, choose the one whose center distance is closest to the A/B
    distance; break ties by total-area closeness, then lexicographic boxes.
    """

    candidates: list[tuple[float, float, float, float]] = []
    for obj in inventory:
        box = tuple(float(value) for value in obj["box"])
        if box_area(box) <= 0:
            continue
        if intersection_area(box, box_a) > 0 or intersection_area(box, box_b) > 0:
            continue
        candidates.append(box)  # type: ignore[arg-type]
    smaller_area = min(box_area(box_a), box_area(box_b))
    cxa, cya = box_center(box_a)
    cxb, cyb = box_center(box_b)
    dist_ab = math.hypot(cxa - cxb, cya - cyb)
    best = None
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            c_box, d_box = candidates[i], candidates[j]
            if intersection_area(c_box, d_box) > 0:
                continue
            area_c, area_d = box_area(c_box), box_area(d_box)
            if area_c <= CD_AREA_FLOOR_OF_SMALLER * smaller_area:
                continue
            if area_d <= CD_AREA_FLOOR_OF_SMALLER * smaller_area:
                continue
            if min(area_c, area_d) / max(area_c, area_d) < CD_AREA_RATIO_MIN:
                continue
            cxc, cyc = box_center(c_box)
            cxd, cyd = box_center(d_box)
            dist_cd = math.hypot(cxc - cxd, cyc - cyd)
            if dist_ab > 0 and not (
                CD_DISTANCE_BAND[0] * dist_ab <= dist_cd <= CD_DISTANCE_BAND[1] * dist_ab
            ):
                continue
            key = (
                abs(dist_cd - dist_ab),
                abs(area_c + area_d - box_area(box_a) - box_area(box_b)),
                c_box,
                d_box,
            )
            if best is None or key < best[0]:
                best = (key, (c_box, d_box))
    return best[1] if best is not None else None


def irrelevant_pixel_swap(
    image: Image.Image,
    box_a: Sequence[float],
    box_b: Sequence[float],
    inventory: Sequence[Mapping[str, Any]],
) -> PixelOperatorResult:
    """Null 2: genuine C/D pixel swap that provably leaves A/B pixels intact."""

    selection = select_irrelevant_pair(box_a, box_b, inventory)
    if selection is None:
        return PixelOperatorResult(
            False, "cd_not_available", None, {"operator": "irrelevant_pixel_swap"}
        )
    box_c, box_d = selection
    reject, stats = base_eligibility(image, box_c, box_d)
    if reject is not None:
        return PixelOperatorResult(
            False, f"cd_{reject}", None, {"operator": "irrelevant_pixel_swap", **stats}
        )
    raster_c = _two_stage_raster(image, box_c, box_d)
    raster_d = _two_stage_raster(image, box_d, box_c)
    edited = _paste(image, raster_c, box_d)
    dx0, dy0, _, _ = integer_box(box_c, image)
    edited.paste(raster_d, (dx0, dy0))
    raster_d.close()
    stats["operator"] = "irrelevant_pixel_swap"
    stats["box_c"] = list(box_c)
    stats["box_d"] = list(box_d)
    stats["ab_pixels_preserved"] = True
    return PixelOperatorResult(True, "accepted", edited, stats)


def _envelope(
    box_a: Sequence[float], box_b: Sequence[float]
) -> tuple[float, float, float, float]:
    return (
        min(float(box_a[0]), float(box_b[0])),
        min(float(box_a[1]), float(box_b[1])),
        max(float(box_a[2]), float(box_b[2])),
        max(float(box_a[3]), float(box_b[3])),
    )


def compact_pair_reflection(
    image: Image.Image,
    box_a: Sequence[float],
    box_b: Sequence[float],
    *,
    direction: str,
) -> PixelOperatorResult:
    """Pair-ROI H/V reflection restricted to compact envelopes.

    Demoted per plan section 4.4 to a compact-pair diagnostic upper bound and
    intermediate-strength positive control; never the main training operator.
    """

    if direction not in {"horizontal", "vertical"}:
        raise ValueError("direction must be horizontal or vertical")
    envelope = _envelope(box_a, box_b)
    image_area = float(image.width * image.height)
    env_area = box_area(envelope)
    stats = {
        "operator": f"compact_pair_{'h' if direction == 'horizontal' else 'v'}flip",
        "envelope_area_ratio": env_area / image_area,
        "support": "same_touched_area_support",
    }
    if intersection_area(box_a, box_b) > 0:
        return PixelOperatorResult(False, "overlapping_boxes", None, stats)
    if env_area / image_area > COMPACT_ENVELOPE_AREA_RATIO_MAX:
        return PixelOperatorResult(False, "envelope_not_compact", None, stats)
    x0, y0, x1, y1 = integer_box(envelope, image)
    if x1 <= x0 or y1 <= y0:
        return PixelOperatorResult(False, "degenerate_integer_box", None, stats)
    region = image.crop((x0, y0, x1, y1))
    transposed = region.transpose(
        Image.Transpose.FLIP_LEFT_RIGHT
        if direction == "horizontal"
        else Image.Transpose.FLIP_TOP_BOTTOM
    )
    region.close()
    edited = image.copy()
    edited.paste(transposed, (x0, y0))
    transposed.close()
    return PixelOperatorResult(True, "accepted", edited, stats)


def canonical_noop(image: Image.Image) -> PixelOperatorResult:
    """Identity control: exact pixel copy, no transformation."""

    return PixelOperatorResult(
        True, "accepted", image.copy(), {"operator": "canonical_noop"}
    )
