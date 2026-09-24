"""Local position-slot counterfactual for visual-token tensors."""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor

from .geometry import (
    BatchValue,
    CoordinateBounds,
    OperatorResult,
    RejectReason,
    coordinates_in_bounds,
    finalize_result,
    first_rejection,
    relation_sign,
    validate_axis,
    validate_pair_rows,
)


def position_slot_swap(
    embeddings: Tensor,
    position_ids: Tensor,
    mask_a: Tensor,
    mask_b: Tensor,
    *,
    axis: int = 2,
    anchor: Literal["min", "mean"] = "min",
    coordinate_bounds: Optional[CoordinateBounds] = None,
    require_relation_flip: bool = True,
    displacement_budget: Optional[BatchValue] = None,
    touched_token_budget: Optional[BatchValue] = None,
    active_visual_token_mask: Optional[Tensor] = None,
    touched_token_ratio_budget: Optional[BatchValue] = None,
) -> OperatorResult:
    """Swap the spatial slots of token groups A and B.

    Each group is translated by the difference between its anchor and the
    other group's anchor.  Unlike interpolating two unequal token blocks, this
    is an involution: applying the same operation twice restores the original
    position ids exactly.  ``anchor="min"`` is the integer-safe default;
    ``anchor="mean"`` is useful for floating point position ids and rejects an
    integer row when the required translation is fractional.

    Only ``position_ids[..., axis]`` is changed.  Embeddings, sequence order,
    token count, and the other two position axes remain untouched.
    """

    axis = validate_axis(axis)
    if anchor not in ("min", "mean"):
        raise ValueError(f"anchor must be 'min' or 'mean', got {anchor!r}")

    accepted, reasons = validate_pair_rows(embeddings, position_ids, mask_a, mask_b)
    batch = embeddings.shape[0]
    candidate_positions = position_ids.clone()
    attempted_touched = mask_a | mask_b

    sign_before = relation_sign(position_ids, mask_a, mask_b, axis)
    first_rejection(
        reasons,
        accepted,
        sign_before == 0,
        RejectReason.RELATION_UNDEFINED,
    )

    delta_a = torch.zeros(batch, device=position_ids.device, dtype=torch.float64)
    delta_b = torch.zeros_like(delta_a)
    coordinate = position_ids[..., axis]

    for row in accepted.nonzero(as_tuple=False).flatten().tolist():
        values_a = coordinate[row, mask_a[row]]
        values_b = coordinate[row, mask_b[row]]
        if anchor == "min":
            anchor_a = values_a.min()
            anchor_b = values_b.min()
        else:
            anchor_a = values_a.to(torch.float64).mean()
            anchor_b = values_b.to(torch.float64).mean()
        move_a = anchor_b.to(torch.float64) - anchor_a.to(torch.float64)
        move_b = -move_a

        if not position_ids.is_floating_point():
            integral = torch.equal(move_a, move_a.round()) and torch.equal(
                move_b, move_b.round()
            )
            if not integral:
                reasons[row] = RejectReason.INVALID_ARGUMENT.value
                accepted[row] = False
                continue
            move_a = move_a.round().to(position_ids.dtype)
            move_b = move_b.round().to(position_ids.dtype)
        else:
            move_a = move_a.to(position_ids.dtype)
            move_b = move_b.to(position_ids.dtype)

        candidate_positions[row, mask_a[row], axis] = values_a + move_a
        candidate_positions[row, mask_b[row], axis] = values_b + move_b
        delta_a[row] = move_a.to(torch.float64)
        delta_b[row] = move_b.to(torch.float64)

    sign_after = relation_sign(candidate_positions, mask_a, mask_b, axis)
    if require_relation_flip:
        first_rejection(
            reasons,
            accepted,
            sign_after != -sign_before,
            RejectReason.RELATION_NOT_FLIPPED,
        )

    in_bounds = coordinates_in_bounds(
        candidate_positions, attempted_touched, axis, coordinate_bounds
    )
    first_rejection(reasons, accepted, ~in_bounds, RejectReason.OUT_OF_BOUNDS)

    count_a = mask_a.sum(dim=1).to(torch.float64)
    count_b = mask_b.sum(dim=1).to(torch.float64)
    attempted_displacement = (
        delta_a.abs() * count_a + delta_b.abs() * count_b
    ).to(torch.float32)

    return finalize_result(
        input_embeddings=embeddings,
        input_position_ids=position_ids,
        output_embeddings=embeddings.clone(),
        output_position_ids=candidate_positions,
        attempted_touched_mask=attempted_touched,
        accepted=accepted,
        reasons=reasons,
        attempted_displacement=attempted_displacement,
        displacement_budget=displacement_budget,
        touched_token_budget=touched_token_budget,
        active_visual_token_mask=active_visual_token_mask,
        touched_token_ratio_budget=touched_token_ratio_budget,
        metadata={
            "operator": "position_slot_swap",
            "position_axis": axis,
            "anchor": anchor,
            "relation_sign_before": sign_before,
            "relation_sign_after_attempt": sign_after,
            "delta_a": delta_a.to(torch.float32),
            "delta_b": delta_b.to(torch.float32),
        },
    )


swap_position_slots = position_slot_swap


__all__ = ["position_slot_swap", "swap_position_slots"]
