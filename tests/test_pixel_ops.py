"""Unit tests for the pixel-space operator family (plan section 4)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from regcfpo.operators.pixel_ops import (
    BOX_SIZE_RATIO_MAX,
    UNION_AREA_RATIO_MAX,
    base_eligibility,
    canonical_grid,
    canonical_noop,
    canonical_resampling_return,
    compact_pair_reflection,
    integer_box,
    irrelevant_pixel_swap,
    pixel_pair_slot_swap,
    processor_resized_size,
    select_irrelevant_pair,
)


def make_image(width: int = 640, height: int = 480) -> Image.Image:
    rng = np.random.default_rng(20270820)
    array = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


@pytest.fixture()
def image() -> Image.Image:
    return make_image()


BOX_A = [40.0, 60.0, 140.0, 160.0]
BOX_B = [420.0, 80.0, 520.0, 180.0]


def mask(image: Image.Image, box_a, box_b) -> np.ndarray:
    mask_array = np.zeros((image.height, image.width), dtype=bool)
    ax0, ay0, ax1, ay1 = integer_box(box_a, image)
    bx0, by0, bx1, by1 = integer_box(box_b, image)
    mask_array[ay0:ay1, ax0:ax1] = True
    mask_array[by0:by1, bx0:bx1] = True
    return mask_array


def test_integer_box_rounds_and_clamps(image):
    assert integer_box([0.2, 0.9, 639.1, 479.2], image) == (0, 0, 640, 480)
    assert integer_box([-5.0, -5.0, 700.0, 500.0], image) == (0, 0, 640, 480)


def test_processor_resized_size_matches_frozen_identity():
    assert processor_resized_size(968, 1296) == (252, 364)
    # 4:3 images collapse to the same max-pixel-bounded grid
    assert processor_resized_size(480, 640) == (252, 364)


def test_canonical_grid_quantizes_to_token_edge():
    assert canonical_grid(BOX_A) == (112, 112)
    assert canonical_grid([0.0, 0.0, 10.0, 10.0]) == (28, 28)


def test_base_eligibility_accepts_clean_pair(image):
    reject, stats = base_eligibility(image, BOX_A, BOX_B)
    assert reject is None
    assert stats["overlap_pixels"] == 0.0
    assert stats["union_area_ratio"] <= UNION_AREA_RATIO_MAX
    assert stats["box_size_ratio"] <= BOX_SIZE_RATIO_MAX


def test_base_eligibility_rejects_overlap(image):
    reject, _ = base_eligibility(image, BOX_A, [100.0, 100.0, 200.0, 200.0])
    assert reject == "overlapping_boxes"


def test_base_eligibility_rejects_integer_box_overlap(image):
    # Float gap is 0.4px, but floor/ceil expansion makes the integer boxes
    # share one pixel column: ceil(110.5)=111 > floor(110.9)=110.  crop/paste
    # operate on the integer boxes, so the pair must be rejected.
    reject, stats = base_eligibility(
        image, [10.5, 15.0, 110.5, 115.0], [110.9, 15.0, 211.4, 115.0]
    )
    assert reject == "integer_box_overlap"
    assert stats["overlap_pixels"] == 0.0
    assert stats["integer_overlap_pixels"] > 0.0


def test_base_eligibility_accepts_three_px_gap(image):
    # Float gap 3.0px: the integer boxes stay disjoint, so the pair remains
    # eligible (integer overlap can only occur for sub-1px float gaps).
    reject, stats = base_eligibility(
        image, [10.5, 15.0, 110.5, 115.0], [113.5, 15.0, 214.0, 115.0]
    )
    assert reject is None
    assert stats["integer_overlap_pixels"] == 0.0


def test_base_eligibility_rejects_size_ratio(image):
    reject, _ = base_eligibility(image, BOX_A, [420.0, 80.0, 640.0, 480.0])
    assert reject in {"box_size_ratio_over_cap", "union_area_over_cap"}


def test_swap_only_touches_boxes_and_is_deterministic(image):
    result = pixel_pair_slot_swap(image, BOX_A, BOX_B)
    assert result.accepted
    assert result.image is not None
    original = np.asarray(image)
    edited = np.asarray(result.image)
    touched = mask(image, BOX_A, BOX_B)
    assert bool((original[~touched] == edited[~touched]).all())
    assert bool((original[touched] != edited[touched]).any())
    again = pixel_pair_slot_swap(image, BOX_A, BOX_B)
    assert np.asarray(again.image).tolist() == edited.tolist()


def test_swap_exchanges_box_contents(image):
    result = pixel_pair_slot_swap(image, BOX_A, BOX_B)
    assert result.image is not None
    ax0, ay0, ax1, ay1 = integer_box(BOX_A, image)
    bx0, by0, bx1, by1 = integer_box(BOX_B, image)
    slot_a = np.asarray(result.image)[ay0:ay1, ax0:ax1]
    slot_b = np.asarray(result.image)[by0:by1, bx0:bx1]
    # slot A now holds B's raster resized to A's integer box; sizes match slots
    assert slot_a.shape == (ay1 - ay0, ax1 - ax0, 3)
    assert slot_b.shape == (by1 - by0, bx1 - bx0, 3)
    # content check: resampling B twice through the canonical grid reproduces slot A
    from regcfpo.operators.pixel_ops import INTERPOLATION

    crop_b = image.crop(integer_box(BOX_B, image))
    grid_w, grid_h = canonical_grid(BOX_B)
    raster = crop_b.resize((grid_w, grid_h), INTERPOLATION).resize(
        (ax1 - ax0, ay1 - ay0), INTERPOLATION
    )
    assert (np.asarray(raster) == slot_a).all()


def test_resampling_null_preserves_box_centers(image):
    result = canonical_resampling_return(image, BOX_A, BOX_B)
    assert result.accepted
    original = np.asarray(image)
    edited = np.asarray(result.image)
    touched = mask(image, BOX_A, BOX_B)
    assert bool((original[~touched] == edited[~touched]).all())


def test_irrelevant_swap_requires_inventory(image):
    result = irrelevant_pixel_swap(image, BOX_A, BOX_B, [])
    assert not result.accepted
    assert result.reject_reason == "cd_not_available"


def test_irrelevant_swap_preserves_a_b_exactly(image):
    inventory = [
        {"box": [200.0, 300.0, 300.0, 400.0], "labels": ["chair"]},
        {"box": [500.0, 300.0, 600.0, 400.0], "labels": ["table"]},
    ]
    result = irrelevant_pixel_swap(image, BOX_A, BOX_B, inventory)
    assert result.accepted, result.reject_reason
    original = np.asarray(image)
    edited = np.asarray(result.image)
    ab_mask = mask(image, BOX_A, BOX_B)
    assert bool((original[ab_mask] == edited[ab_mask]).all())
    cd_mask = np.zeros_like(ab_mask)
    cd_mask[300:400, 200:300] = True
    cd_mask[300:400, 500:600] = True
    assert bool((original[cd_mask] != edited[cd_mask]).any())


def test_select_irrelevant_pair_is_deterministic():
    inventory = [
        {"box": [200.0, 300.0, 300.0, 400.0], "labels": ["chair"]},
        {"box": [500.0, 300.0, 600.0, 400.0], "labels": ["table"]},
        {"box": [60.0, 70.0, 130.0, 150.0], "labels": ["lamp"]},
    ]
    first = select_irrelevant_pair(BOX_A, BOX_B, inventory)
    second = select_irrelevant_pair(BOX_A, BOX_B, list(reversed(inventory)))
    assert first is not None
    assert set(map(tuple, first)) == set(map(tuple, second))
    # the A/B-overlapping lamp can never be selected
    assert [60.0, 70.0, 130.0, 150.0] not in [list(box) for box in first]


def test_compact_reflection_rejects_large_envelope(image):
    wide_a = [10.0, 10.0, 200.0, 400.0]
    wide_b = [430.0, 10.0, 630.0, 400.0]
    result = compact_pair_reflection(image, wide_a, wide_b, direction="horizontal")
    assert not result.accepted
    assert result.reject_reason == "envelope_not_compact"
    assert result.metadata["envelope_area_ratio"] > 0.35


def test_compact_reflection_flips_envelope_only(image):
    close_a = [40.0, 60.0, 140.0, 160.0]
    close_b = [180.0, 80.0, 260.0, 150.0]
    result = compact_pair_reflection(image, close_a, close_b, direction="horizontal")
    assert result.accepted, result.reject_reason
    original = np.asarray(image)
    edited = np.asarray(result.image)
    ex0, ey0, ex1, ey1 = 40, 60, 260, 160
    outside = np.ones((image.height, image.width), dtype=bool)
    outside[ey0:ey1, ex0:ex1] = False
    assert bool((original[outside] == edited[outside]).all())
    region = original[ey0:ey1, ex0:ex1]
    flipped = region[:, ::-1]
    assert (np.asarray(result.image)[ey0:ey1, ex0:ex1] == flipped).all()


def test_canonical_noop_is_exact_copy(image):
    result = canonical_noop(image)
    assert result.accepted
    assert np.asarray(result.image).tolist() == np.asarray(image).tolist()
