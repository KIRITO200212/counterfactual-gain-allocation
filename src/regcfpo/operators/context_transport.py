"""Context-matched, relation-preserving position transport control."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .geometry import (
    BatchValue,
    CoordinateBounds,
    OperatorResult,
    RejectReason,
    as_batch_tensor,
    finalize_result,
    first_rejection,
    normalize_coordinate_bounds,
    relation_sign,
    validate_axis,
    validate_pair_rows,
)


def context_transport(
    embeddings: Tensor,
    position_ids: Tensor,
    mask_a: Tensor,
    mask_b: Tensor,
    *,
    relation_axis: int = 2,
    transport_axis: int = 1,
    shift: Optional[BatchValue] = None,
    target_displacement: Optional[BatchValue] = None,
    preferred_direction: int = 1,
    coordinate_bounds: Optional[CoordinateBounds] = None,
    displacement_match_rtol: float = 0.05,
    displacement_match_atol: float = 0.0,
    displacement_budget: Optional[BatchValue] = None,
    touched_token_budget: Optional[BatchValue] = None,
    active_visual_token_mask: Optional[Tensor] = None,
    touched_token_ratio_budget: Optional[BatchValue] = None,
) -> OperatorResult:
    """Translate A and B together while preserving their left/right sign.

    Supply either a signed ``shift`` or a ``target_displacement``.  The latter
    chooses the closest representable common shift and checks the requested
    budget with ``torch.isclose``.  When bounded coordinates make the preferred
    direction infeasible, the opposite direction is tried automatically.

    The default moves both objects vertically (axis 1), leaving the horizontal
    relation (axis 2) unchanged even before the explicit sign check.
    """

    relation_axis = validate_axis(relation_axis)
    transport_axis = validate_axis(transport_axis)
    if shift is not None and target_displacement is not None:
        raise ValueError("provide shift or target_displacement, not both")
    if preferred_direction not in (-1, 1):
        raise ValueError("preferred_direction must be -1 or 1")
    if displacement_match_rtol < 0 or displacement_match_atol < 0:
        raise ValueError("displacement matching tolerances must be non-negative")
    accepted, reasons = validate_pair_rows(embeddings, position_ids, mask_a, mask_b)
    batch = embeddings.shape[0]
    device = embeddings.device
    normalized_bounds = normalize_coordinate_bounds(
        coordinate_bounds, batch, device=device
    )
    attempted_touched = mask_a | mask_b
    touched_count = attempted_touched.sum(dim=1).to(torch.float32)

    sign_before = relation_sign(position_ids, mask_a, mask_b, relation_axis)
    first_rejection(
        reasons, accepted, sign_before == 0, RejectReason.RELATION_UNDEFINED
    )

    requested_target = as_batch_tensor(
        target_displacement,
        batch,
        device=device,
        dtype=torch.float32,
        name="target_displacement",
    )
    explicit_shift = as_batch_tensor(
        shift,
        batch,
        device=device,
        dtype=torch.float64,
        name="shift",
    )
    if requested_target is not None and bool((requested_target < 0).any().item()):
        raise ValueError("target_displacement must be non-negative")

    if explicit_shift is None:
        if requested_target is None:
            requested_shift = torch.full(
                (batch,), float(preferred_direction), device=device, dtype=torch.float64
            )
        else:
            magnitude = requested_target.to(torch.float64) / touched_count.clamp_min(1)
            if not position_ids.is_floating_point():
                magnitude = magnitude.round()
                magnitude = torch.where(
                    (requested_target > 0) & (magnitude == 0),
                    torch.ones_like(magnitude),
                    magnitude,
                )
            requested_shift = magnitude * preferred_direction
    else:
        requested_shift = explicit_shift

    candidate_positions = position_ids.clone()
    realized_shift = torch.zeros(batch, device=device, dtype=torch.float64)
    if normalized_bounds is None:
        normalized_bounds = torch.tensor(
            [-float("inf"), float("inf")], device=device, dtype=torch.float64
        ).expand(batch, 2)

    for row in accepted.nonzero(as_tuple=False).flatten().tolist():
        row_shift = requested_shift[row]
        if not bool(torch.isfinite(row_shift).item()):
            reasons[row] = RejectReason.INVALID_ARGUMENT.value
            accepted[row] = False
            continue
        if not position_ids.is_floating_point() and not torch.equal(
            row_shift, row_shift.round()
        ):
            reasons[row] = RejectReason.INVALID_ARGUMENT.value
            accepted[row] = False
            continue

        current = position_ids[row, attempted_touched[row], transport_axis].to(torch.float64)
        candidates = (row_shift, -row_shift) if row_shift != 0 else (row_shift,)
        chosen = None
        lower, upper = normalized_bounds[row]
        for candidate in candidates:
            moved = current + candidate
            if bool(((moved >= lower) & (moved <= upper)).all().item()):
                chosen = candidate
                break
        if chosen is None:
            reasons[row] = RejectReason.NO_FEASIBLE_TRANSPORT.value
            accepted[row] = False
            continue

        typed_shift = chosen.to(position_ids.dtype)
        candidate_positions[row, attempted_touched[row], transport_axis] = (
            position_ids[row, attempted_touched[row], transport_axis] + typed_shift
        )
        realized_shift[row] = chosen

    attempted_displacement = (
        realized_shift.abs().to(torch.float32) * touched_count
    )
    if requested_target is not None:
        matched = torch.isclose(
            attempted_displacement,
            requested_target,
            rtol=displacement_match_rtol,
            atol=displacement_match_atol,
        )
        first_rejection(
            reasons,
            accepted,
            ~matched,
            RejectReason.DISPLACEMENT_BUDGET_UNMATCHED,
        )

    sign_after = relation_sign(candidate_positions, mask_a, mask_b, relation_axis)
    first_rejection(
        reasons,
        accepted,
        sign_after != sign_before,
        RejectReason.RELATION_SIGN_CHANGED,
    )

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
            "operator": "context_transport",
            "relation_axis": relation_axis,
            "transport_axis": transport_axis,
            "relation_sign_before": sign_before,
            "relation_sign_after_attempt": sign_after,
            "requested_shift": requested_shift.to(torch.float32),
            "realized_shift": realized_shift.to(torch.float32),
            "target_displacement": requested_target,
        },
    )


context_transport_null = context_transport


__all__ = ["context_transport", "context_transport_null"]
