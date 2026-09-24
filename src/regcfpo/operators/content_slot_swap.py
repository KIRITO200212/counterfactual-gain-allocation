"""Local content-slot counterfactual for visual-token tensors."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .geometry import (
    BatchValue,
    OperatorResult,
    RejectReason,
    finalize_result,
    pair_transport_displacement,
    resample_token_grid,
    validate_pair_rows,
    validate_rectangular_pair_grids,
)


def content_slot_swap(
    embeddings: Tensor,
    position_ids: Tensor,
    mask_a: Tensor,
    mask_b: Tensor,
    *,
    require_equal_token_count: bool = False,
    displacement_budget: Optional[BatchValue] = None,
    touched_token_budget: Optional[BatchValue] = None,
    active_visual_token_mask: Optional[Tensor] = None,
    touched_token_ratio_budget: Optional[BatchValue] = None,
) -> OperatorResult:
    """Exchange A/B visual content while retaining their spatial slots.

    Each group must be a complete contiguous ``H x W`` rectangle on one Qwen
    temporal plane.  Payloads are ordered by their explicit height/width
    coordinates and bilinearly resized in two dimensions to the destination
    grid.  Equal ``H x W`` shapes use an exact clone, making a double swap
    bit-for-bit exact.  ``require_equal_token_count`` retains its literal count
    gate; equal-count grids with different shapes are still resized in 2D.
    """

    if not embeddings.is_floating_point():
        raise ValueError("content-slot operators require floating point embeddings")

    accepted, reasons = validate_pair_rows(embeddings, position_ids, mask_a, mask_b)
    grids_a, grids_b = validate_rectangular_pair_grids(
        position_ids, mask_a, mask_b, accepted, reasons
    )
    count_a = mask_a.sum(dim=1)
    count_b = mask_b.sum(dim=1)
    if require_equal_token_count:
        unequal = count_a != count_b
        rows = (unequal & accepted).nonzero(as_tuple=False).flatten().tolist()
        for row in rows:
            reasons[row] = RejectReason.INVALID_ARGUMENT.value
            accepted[row] = False

    candidate_embeddings = embeddings.clone()
    attempted_touched = mask_a | mask_b
    grid_shape_a = torch.zeros(
        (embeddings.shape[0], 2), dtype=torch.long, device=embeddings.device
    )
    grid_shape_b = torch.zeros_like(grid_shape_a)
    geometry_valid = torch.zeros(
        embeddings.shape[0], dtype=torch.bool, device=embeddings.device
    )
    for row, (grid_a, grid_b) in enumerate(zip(grids_a, grids_b)):
        if grid_a is None or grid_b is None:
            continue
        grid_shape_a[row] = torch.tensor(grid_a.shape, device=embeddings.device)
        grid_shape_b[row] = torch.tensor(grid_b.shape, device=embeddings.device)
        geometry_valid[row] = True

    for row in accepted.nonzero(as_tuple=False).flatten().tolist():
        grid_a = grids_a[row]
        grid_b = grids_b[row]
        assert grid_a is not None and grid_b is not None
        source_a = embeddings[row, grid_a.token_indices].clone()
        source_b = embeddings[row, grid_b.token_indices].clone()
        candidate_embeddings[row, grid_a.token_indices] = resample_token_grid(
            source_b, grid_b.shape, grid_a.shape
        )
        candidate_embeddings[row, grid_b.token_indices] = resample_token_grid(
            source_a, grid_a.shape, grid_b.shape
        )

    attempted_displacement = pair_transport_displacement(
        position_ids, mask_a, mask_b
    )
    same_grid_shape = (grid_shape_a == grid_shape_b).all(dim=1) & geometry_valid
    return finalize_result(
        input_embeddings=embeddings,
        input_position_ids=position_ids,
        output_embeddings=candidate_embeddings,
        output_position_ids=position_ids.clone(),
        attempted_touched_mask=attempted_touched,
        accepted=accepted,
        reasons=reasons,
        attempted_displacement=attempted_displacement,
        displacement_budget=displacement_budget,
        touched_token_budget=touched_token_budget,
        active_visual_token_mask=active_visual_token_mask,
        touched_token_ratio_budget=touched_token_ratio_budget,
        metadata={
            "operator": "content_slot_swap",
            "token_count_a": count_a,
            "token_count_b": count_b,
            "grid_shape_a": grid_shape_a,
            "grid_shape_b": grid_shape_b,
            "resampled": geometry_valid & ~same_grid_shape,
            "strictly_reversible": same_grid_shape,
        },
    )


swap_content_slots = content_slot_swap


__all__ = ["content_slot_swap", "swap_content_slots"]
