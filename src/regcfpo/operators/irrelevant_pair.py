"""Irrelevant-pair control that transactionally protects task operands A/B."""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor

from .content_slot_swap import content_slot_swap
from .geometry import (
    BatchValue,
    CoordinateBounds,
    OperatorResult,
    RejectReason,
    as_batch_tensor,
    finalize_result,
    first_rejection,
    validate_pair_rows,
    validate_tensor_primitives,
)
from .position_slot_swap import position_slot_swap


def irrelevant_pair_edit(
    embeddings: Tensor,
    position_ids: Tensor,
    protected_a_mask: Tensor,
    protected_b_mask: Tensor,
    irrelevant_c_mask: Tensor,
    irrelevant_d_mask: Tensor,
    *,
    operator: Literal["content", "position"] = "content",
    position_axis: int = 2,
    coordinate_bounds: Optional[CoordinateBounds] = None,
    require_relation_flip: bool = True,
    target_touched_token_count: Optional[BatchValue] = None,
    touched_match_atol: float = 0.0,
    target_displacement: Optional[BatchValue] = None,
    displacement_match_rtol: float = 0.05,
    displacement_match_atol: float = 0.0,
    displacement_budget: Optional[BatchValue] = None,
    touched_token_budget: Optional[BatchValue] = None,
    active_visual_token_mask: Optional[Tensor] = None,
    touched_token_ratio_budget: Optional[BatchValue] = None,
) -> OperatorResult:
    """Edit C/D with the chosen swap while guaranteeing A/B stay untouched.

    Optional targets enforce the matching criteria used by the audit manifest:
    touched-token count and total displacement.  A failed match is rejected and
    rolled back, with the attempted metrics retained in ``metadata``.
    """

    validate_tensor_primitives(
        embeddings,
        position_ids,
        protected_a_mask,
        protected_b_mask,
        irrelevant_c_mask,
        irrelevant_d_mask,
    )
    if operator not in ("content", "position"):
        raise ValueError("operator must be 'content' or 'position'")
    if touched_match_atol < 0 or displacement_match_rtol < 0 or displacement_match_atol < 0:
        raise ValueError("matching tolerances must be non-negative")

    accepted, reasons = validate_pair_rows(
        embeddings, position_ids, irrelevant_c_mask, irrelevant_d_mask
    )
    protected_valid, protected_reasons = validate_pair_rows(
        embeddings, position_ids, protected_a_mask, protected_b_mask
    )
    for row in (~protected_valid & accepted).nonzero(as_tuple=False).flatten().tolist():
        reasons[row] = protected_reasons[row]
        accepted[row] = False

    protected_mask = protected_a_mask | protected_b_mask
    attempted_touched = irrelevant_c_mask | irrelevant_d_mask
    protected_overlap = (protected_mask & attempted_touched).any(dim=1)
    first_rejection(
        reasons,
        accepted,
        protected_overlap,
        RejectReason.PROTECTED_TOKEN_TOUCHED,
    )

    if operator == "content":
        inner = content_slot_swap(
            embeddings,
            position_ids,
            irrelevant_c_mask,
            irrelevant_d_mask,
            active_visual_token_mask=active_visual_token_mask,
            touched_token_ratio_budget=touched_token_ratio_budget,
        )
    else:
        inner = position_slot_swap(
            embeddings,
            position_ids,
            irrelevant_c_mask,
            irrelevant_d_mask,
            axis=position_axis,
            coordinate_bounds=coordinate_bounds,
            require_relation_flip=require_relation_flip,
            active_visual_token_mask=active_visual_token_mask,
            touched_token_ratio_budget=touched_token_ratio_budget,
        )

    for row in (accepted & ~inner.accepted).nonzero(as_tuple=False).flatten().tolist():
        reasons[row] = inner.reject_reason[row]
        accepted[row] = False

    attempted_count = attempted_touched.sum(dim=1).to(torch.float32)
    expected_count = as_batch_tensor(
        target_touched_token_count,
        embeddings.shape[0],
        device=embeddings.device,
        dtype=torch.float32,
        name="target_touched_token_count",
    )
    if expected_count is not None:
        if bool((expected_count < 0).any().item()):
            raise ValueError("target_touched_token_count must be non-negative")
        touched_matched = (attempted_count - expected_count).abs() <= touched_match_atol
        first_rejection(
            reasons,
            accepted,
            ~touched_matched,
            RejectReason.TOUCHED_TOKEN_BUDGET_UNMATCHED,
        )

    attempted_displacement = inner.metadata["attempted_total_displacement"]
    expected_displacement = as_batch_tensor(
        target_displacement,
        embeddings.shape[0],
        device=embeddings.device,
        dtype=torch.float32,
        name="target_displacement",
    )
    if expected_displacement is not None:
        if bool((expected_displacement < 0).any().item()):
            raise ValueError("target_displacement must be non-negative")
        displacement_matched = torch.isclose(
            attempted_displacement,
            expected_displacement,
            rtol=displacement_match_rtol,
            atol=displacement_match_atol,
        )
        first_rejection(
            reasons,
            accepted,
            ~displacement_matched,
            RejectReason.DISPLACEMENT_BUDGET_UNMATCHED,
        )

    # Belt-and-suspenders invariant: even a future inner implementation cannot
    # silently write protected tokens.
    actually_changed = (
        (inner.embeddings != embeddings).any(dim=-1)
        | (inner.position_ids != position_ids).any(dim=-1)
    )
    protected_changed = (actually_changed & protected_mask).any(dim=1)
    first_rejection(
        reasons,
        accepted,
        protected_changed,
        RejectReason.PROTECTED_TOKEN_TOUCHED,
    )

    metadata = dict(inner.metadata)
    metadata.update(
        {
            "operator": "irrelevant_pair_edit",
            "inner_operator": operator,
            "target_touched_token_count": expected_count,
            "target_displacement": expected_displacement,
            "protected_overlap_attempt": protected_overlap,
        }
    )
    return finalize_result(
        input_embeddings=embeddings,
        input_position_ids=position_ids,
        output_embeddings=inner.embeddings,
        output_position_ids=inner.position_ids,
        attempted_touched_mask=attempted_touched,
        accepted=accepted,
        reasons=reasons,
        attempted_displacement=attempted_displacement,
        displacement_budget=displacement_budget,
        touched_token_budget=touched_token_budget,
        active_visual_token_mask=active_visual_token_mask,
        touched_token_ratio_budget=touched_token_ratio_budget,
        metadata=metadata,
    )


irrelevant_pair_control = irrelevant_pair_edit


__all__ = ["irrelevant_pair_control", "irrelevant_pair_edit"]
