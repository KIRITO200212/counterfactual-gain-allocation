"""Relation-preserving resampling control for content-slot interventions."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .geometry import (
    BatchValue,
    OperatorResult,
    finalize_result,
    resample_token_grid,
    validate_pair_rows,
    validate_rectangular_pair_grids,
)


def resampling_null(
    embeddings: Tensor,
    position_ids: Tensor,
    mask_a: Tensor,
    mask_b: Tensor,
    *,
    bridge_token_counts: bool = True,
    displacement_budget: Optional[BatchValue] = None,
    touched_token_budget: Optional[BatchValue] = None,
    active_visual_token_mask: Optional[Tensor] = None,
    touched_token_ratio_budget: Optional[BatchValue] = None,
) -> OperatorResult:
    """Resample A and B but write each payload back to its factual slot.

    With ``bridge_token_counts=True``, A follows ``(Ha,Wa) -> (Hb,Wb) ->
    (Ha,Wa)`` and B follows the matched reverse path, using the same 2D
    bilinear primitive as :func:`content_slot_swap`.  This exposes
    interpolation and token-norm effects without changing object identity or
    the A/B relation.  Equal grid shapes remain exact identities.
    """

    if not embeddings.is_floating_point():
        raise ValueError("resampling operators require floating point embeddings")
    accepted, reasons = validate_pair_rows(embeddings, position_ids, mask_a, mask_b)
    grids_a, grids_b = validate_rectangular_pair_grids(
        position_ids, mask_a, mask_b, accepted, reasons
    )
    candidate_embeddings = embeddings.clone()
    attempted_touched = mask_a | mask_b
    count_a = mask_a.sum(dim=1)
    count_b = mask_b.sum(dim=1)
    grid_shape_a = torch.zeros(
        (embeddings.shape[0], 2), dtype=torch.long, device=embeddings.device
    )
    grid_shape_b = torch.zeros_like(grid_shape_a)
    for row, (grid_a, grid_b) in enumerate(zip(grids_a, grids_b)):
        if grid_a is None or grid_b is None:
            continue
        grid_shape_a[row] = torch.tensor(grid_a.shape, device=embeddings.device)
        grid_shape_b[row] = torch.tensor(grid_b.shape, device=embeddings.device)

    for row in accepted.nonzero(as_tuple=False).flatten().tolist():
        grid_a = grids_a[row]
        grid_b = grids_b[row]
        assert grid_a is not None and grid_b is not None
        source_a = embeddings[row, grid_a.token_indices].clone()
        source_b = embeddings[row, grid_b.token_indices].clone()
        if bridge_token_counts:
            processed_a = resample_token_grid(
                resample_token_grid(source_a, grid_a.shape, grid_b.shape),
                grid_b.shape,
                grid_a.shape,
            )
            processed_b = resample_token_grid(
                resample_token_grid(source_b, grid_b.shape, grid_a.shape),
                grid_a.shape,
                grid_b.shape,
            )
        else:
            # Still execute the same primitive; the equal-shape fast path is
            # intentionally an exact clone for a zero-distortion control.
            processed_a = resample_token_grid(source_a, grid_a.shape, grid_a.shape)
            processed_b = resample_token_grid(source_b, grid_b.shape, grid_b.shape)
        candidate_embeddings[row, grid_a.token_indices] = processed_a
        candidate_embeddings[row, grid_b.token_indices] = processed_b

    zero_displacement = torch.zeros(
        embeddings.shape[0], device=embeddings.device, dtype=torch.float32
    )
    return finalize_result(
        input_embeddings=embeddings,
        input_position_ids=position_ids,
        output_embeddings=candidate_embeddings,
        output_position_ids=position_ids.clone(),
        attempted_touched_mask=attempted_touched,
        accepted=accepted,
        reasons=reasons,
        attempted_displacement=zero_displacement,
        displacement_budget=displacement_budget,
        touched_token_budget=touched_token_budget,
        active_visual_token_mask=active_visual_token_mask,
        touched_token_ratio_budget=touched_token_ratio_budget,
        metadata={
            "operator": "resampling_null",
            "token_count_a": count_a,
            "token_count_b": count_b,
            "grid_shape_a": grid_shape_a,
            "grid_shape_b": grid_shape_b,
            "bridge_token_counts": bridge_token_counts,
            "relation_preserved_by_construction": True,
        },
    )


resample_in_place_null = resampling_null


__all__ = ["resampling_null", "resample_in_place_null"]
