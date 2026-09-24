"""Model-independent visual-token intervention budget decisions."""

from __future__ import annotations

from typing import Any

import torch


def audit_mask_pair(
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
    active_visual_mask: torch.Tensor,
    *,
    max_modified_ratio: float,
) -> dict[str, Any]:
    """Return the frozen visual-token budget decision for one batch row."""

    if not isinstance(mask_a, torch.Tensor) or mask_a.ndim != 2 or mask_a.shape[0] != 1:
        raise ValueError("mask_a must have shape [1,S]")
    expected_shape = mask_a.shape
    for name, mask in {
        "mask_a": mask_a,
        "mask_b": mask_b,
        "active_visual_mask": active_visual_mask,
    }.items():
        if not isinstance(mask, torch.Tensor) or mask.shape != expected_shape:
            raise ValueError(f"{name} must have shape [1,S]")
        if mask.dtype != torch.bool or mask.device != mask_a.device:
            raise ValueError(f"{name} must be a boolean tensor on the same device")
    if not 0.0 <= max_modified_ratio <= 1.0:
        raise ValueError("max_modified_ratio must lie in [0,1]")
    if bool(((mask_a | mask_b) & ~active_visual_mask).any().item()):
        raise ValueError("object masks may only select active visual tokens")

    active_count = int(active_visual_mask.sum().item())
    if active_count == 0:
        raise ValueError("active visual-token mask is empty")
    count_a = int(mask_a.sum().item())
    count_b = int(mask_b.sum().item())
    overlap_count = int((mask_a & mask_b).sum().item())
    union_count = int((mask_a | mask_b).sum().item())
    modified_ratio = float(union_count / active_count)
    if overlap_count:
        reason = "masks_overlap"
    elif modified_ratio > max_modified_ratio:
        reason = "modified_visual_token_ratio_exceeded"
    else:
        reason = "accepted"
    return {
        "active_visual_token_count": active_count,
        "mask_a_token_count": count_a,
        "mask_b_token_count": count_b,
        "union_visual_token_count": union_count,
        "overlap_visual_token_count": overlap_count,
        "modified_visual_token_ratio": modified_ratio,
        "operator_valid": reason == "accepted",
        "reject_reason": reason,
    }


__all__ = ["audit_mask_pair"]
