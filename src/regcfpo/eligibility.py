"""Single shared pixel-eligibility function for all data-layer consumers.

Hardening per amendment ``analysis_integrity_and_diagonal_disambiguation
_20260820``: support audits, capacity audits, reserve reallocation, and any
future training manifest must call :func:`evaluate_pixel_pair_eligibility`
so the frozen thresholds can never drift between consumers.  The shared
function mirrors ``pixel_ops.base_eligibility`` decision-for-decision,
including the integer-box (floor/ceil) disjointness check added by the
plan7 section 11 governance fix; equivalence is proven by unit tests over
the full RelationPair-v3 candidate set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

# Frozen processor identity (models/SpatialLadder-3B/preprocessor_config.json).
TOKEN_EDGE = 28
MIN_PIXELS = 12544
MAX_PIXELS = 100352

# Frozen eligibility thresholds from the read-only support audit.
UNION_AREA_RATIO_MAX = 0.35
BOX_SIZE_RATIO_MAX = 4.0
RESIZED_MIN_TOKEN_SUPPORT_MIN = 1.0


@dataclass(frozen=True)
class EligibilityResult:
    """Deterministic eligibility decision with full audit statistics."""

    eligible: bool
    reject_reason: str
    union_area_ratio: float
    box_size_ratio: float
    resized_min_token_support: float
    overlap_pixels: float


def processor_resized_size(height: float, width: float) -> tuple[int, int]:
    """Replicate the frozen Qwen processor resolution rounding."""

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


def _area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1])
    )


def _intersection(a: Sequence[float], b: Sequence[float]) -> float:
    return max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0]))) * max(
        0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1]))
    )


def evaluate_pixel_pair_eligibility(
    box_a: Sequence[float],
    box_b: Sequence[float],
    image_size_hw: tuple[float, float],
) -> EligibilityResult:
    """Apply the frozen v1 eligibility gate to one candidate pair.

    Decision order and reasons mirror ``pixel_ops.base_eligibility`` exactly
    (empty box, degenerate integer box, float overlap, integer-box overlap,
    union area, size ratio, resized token support).  ``image_size_hw`` is
    ``(height, width)``.
    """

    height, width = float(image_size_hw[0]), float(image_size_hw[1])
    image_area = height * width
    area_a, area_b = _area(box_a), _area(box_b)
    overlap = _intersection(box_a, box_b)
    union_ratio = (area_a + area_b - overlap) / image_area

    def reject(reason: str, ratio: float = float("nan"), support: float = float("nan")):
        return EligibilityResult(
            eligible=False,
            reject_reason=reason,
            union_area_ratio=union_ratio,
            box_size_ratio=ratio,
            resized_min_token_support=support,
            overlap_pixels=overlap,
        )

    if area_a <= 0.0 or area_b <= 0.0:
        return reject("empty_box")
    boxes = (box_a, box_b)
    int_boxes = []
    for box in boxes:
        x0, y0 = math.floor(float(box[0])), math.floor(float(box[1]))
        x1 = min(int(width), math.ceil(float(box[2])))
        y1 = min(int(height), math.ceil(float(box[3])))
        x0, y0 = max(0, x0), max(0, y0)
        if x1 <= x0 or y1 <= y0:
            return reject("degenerate_integer_box")
        int_boxes.append((x0, y0, x1, y1))
    size_ratio = max(area_a, area_b) / min(area_a, area_b)
    resized_h, resized_w = processor_resized_size(height, width)
    scale_x, scale_y = resized_w / width, resized_h / height
    support = min(
        ((float(b[2]) - float(b[0])) * scale_x / TOKEN_EDGE)
        * ((float(b[3]) - float(b[1])) * scale_y / TOKEN_EDGE)
        for b in boxes
    )
    if overlap > 0:
        return reject("overlapping_boxes", size_ratio, support)
    (ix0, iy0, ix1, iy1), (jx0, jy0, jx1, jy1) = int_boxes
    if max(0, min(ix1, jx1) - max(ix0, jx0)) * max(0, min(iy1, jy1) - max(iy0, jy0)) > 0:
        return reject("integer_box_overlap", size_ratio, support)
    if union_ratio > UNION_AREA_RATIO_MAX:
        return reject("union_area_over_cap", size_ratio, support)
    if size_ratio > BOX_SIZE_RATIO_MAX:
        return reject("box_size_ratio_over_cap", size_ratio, support)
    if support < RESIZED_MIN_TOKEN_SUPPORT_MIN:
        return reject("insufficient_resized_token_support", size_ratio, support)
    return EligibilityResult(
        eligible=True,
        reject_reason="accepted",
        union_area_ratio=union_ratio,
        box_size_ratio=size_ratio,
        resized_min_token_support=support,
        overlap_pixels=overlap,
    )


def assert_scene_lineage_disjoint(
    splits: Mapping[str, set[str]],
    viewed_scenes: set[str],
) -> None:
    """Fail-closed lineage assertions for the final v3 freeze.

    Requires train/replay/holdout (or any named split) to be disjoint from
    the viewed lineage and from each other, verified on explicit scene-id
    sets rather than per-row lineage fields.
    """

    names = sorted(splits)
    for name, scenes in splits.items():
        leaked = scenes & viewed_scenes
        if leaked:
            raise ValueError(
                f"split {name!r} contains viewed scenes: {sorted(leaked)[:5]}"
            )
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            overlap = splits[first] & splits[second]
            if overlap:
                raise ValueError(
                    f"splits {first!r} and {second!r} overlap: {sorted(overlap)[:5]}"
                )
