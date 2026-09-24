"""Tensor geometry shared by the counterfactual operators.

The operators in this package deliberately stop at the Qwen boundary.  They
consume batched visual-token embeddings, M-RoPE-like position ids, and boolean
token masks.  Model-specific code is responsible for locating the visual span
and constructing the masks.

All public operators use a transactional convention: malformed *tensor
shapes* raise immediately, while an invalid item inside an otherwise valid
batch is rejected and restored to its factual input.  Consequently, callers
can safely log and collate every attempted intervention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
import torch.nn.functional as F


BatchValue = Union[int, float, Tensor]
CoordinateBounds = Union[
    Tensor,
    Sequence[float],
    Sequence[Sequence[float]],
]


class RejectReason(str, Enum):
    """Stable reason codes used in experiment logs."""

    NONE = "accepted"
    EMPTY_MASK_A = "empty_mask_a"
    EMPTY_MASK_B = "empty_mask_b"
    MASKS_OVERLAP = "masks_overlap"
    NON_FINITE_INPUT = "non_finite_input"
    NON_FINITE_OUTPUT = "non_finite_output"
    RELATION_UNDEFINED = "relation_undefined"
    RELATION_NOT_FLIPPED = "relation_not_flipped"
    RELATION_SIGN_CHANGED = "relation_sign_changed"
    OUT_OF_BOUNDS = "out_of_bounds"
    NO_FEASIBLE_TRANSPORT = "no_feasible_transport"
    DISPLACEMENT_BUDGET_EXCEEDED = "displacement_budget_exceeded"
    DISPLACEMENT_BUDGET_UNMATCHED = "displacement_budget_unmatched"
    TOUCHED_TOKEN_BUDGET_EXCEEDED = "touched_token_budget_exceeded"
    TOUCHED_TOKEN_BUDGET_UNMATCHED = "touched_token_budget_unmatched"
    TOUCHED_TOKEN_RATIO_BUDGET_EXCEEDED = "touched_token_ratio_budget_exceeded"
    EMPTY_ACTIVE_VISUAL_MASK = "empty_active_visual_mask"
    NON_VISUAL_TOKEN_TOUCHED = "non_visual_token_touched"
    PROTECTED_TOKEN_TOUCHED = "protected_token_touched"
    MULTIPLE_TEMPORAL_PLANES = "multiple_temporal_planes"
    NON_RECTANGULAR_ROI = "non_rectangular_roi"
    INVALID_ARGUMENT = "invalid_argument"


@dataclass(frozen=True)
class OperatorResult:
    """Result of one batched tensor intervention.

    ``accepted`` and every metric are per-example tensors.  ``reject_reason``
    has exactly one stable string code per batch item.  Rejected rows have
    factual tensors, an empty touched mask, and zero realized displacement.
    ``attempted_*`` metrics in ``metadata`` retain the pre-rejection audit data.
    """

    embeddings: Tensor
    position_ids: Tensor
    touched_token_mask: Tensor
    accepted: Tensor
    reject_reason: Tuple[str, ...]
    touched_token_count: Tensor
    touched_token_ratio: Tensor
    total_displacement: Tensor
    displacement_budget: Optional[Tensor] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    touched_token_ratio_budget: Optional[Tensor] = None

    @property
    def reject_reasons(self) -> Tuple[str, ...]:
        """Plural alias convenient for logging code."""

        return self.reject_reason

    @property
    def all_accepted(self) -> bool:
        return bool(self.accepted.all().item())

    @property
    def modified_token_ratio(self) -> Tensor:
        """Alias matching the name used by the operator audit report."""

        return self.touched_token_ratio

    def assert_finite(self) -> None:
        """Raise when an accepted output contains NaN or Inf."""

        finite = finite_rows(self.embeddings) & finite_rows(self.position_ids)
        bad = self.accepted & ~finite
        if bool(bad.any().item()):
            rows = bad.nonzero(as_tuple=False).flatten().tolist()
            raise FloatingPointError(f"non-finite accepted operator rows: {rows}")


@dataclass(frozen=True)
class RectangularTokenGrid:
    """One ROI represented as row-major token indices and its ``(H, W)`` shape."""

    token_indices: Tensor
    height: int
    width: int

    @property
    def shape(self) -> Tuple[int, int]:
        return self.height, self.width


def validate_tensor_primitives(
    embeddings: Tensor,
    position_ids: Tensor,
    *masks: Tensor,
) -> Tuple[int, int]:
    """Validate the model-independent ``[B,N,D]``/``[B,N,3]`` contract."""

    if not isinstance(embeddings, Tensor) or embeddings.ndim != 3:
        raise ValueError("embeddings must be a torch.Tensor with shape [B,N,D]")
    if not isinstance(position_ids, Tensor) or position_ids.ndim != 3:
        raise ValueError("position_ids must be a torch.Tensor with shape [B,N,3]")
    batch, tokens, _ = embeddings.shape
    if position_ids.shape != (batch, tokens, 3):
        raise ValueError(
            "position_ids must have shape [B,N,3] matching embeddings; "
            f"got {tuple(position_ids.shape)} for {tuple(embeddings.shape)}"
        )
    if embeddings.device != position_ids.device:
        raise ValueError("embeddings and position_ids must be on the same device")
    for index, mask in enumerate(masks):
        if not isinstance(mask, Tensor) or mask.shape != (batch, tokens):
            raise ValueError(
                f"mask {index} must be a torch.Tensor with shape [B,N]; "
                f"got {getattr(mask, 'shape', None)}"
            )
        if mask.device != embeddings.device:
            raise ValueError(f"mask {index} must be on the same device as embeddings")
        if mask.dtype != torch.bool:
            raise ValueError(f"mask {index} must have dtype torch.bool")
    return batch, tokens


def validate_axis(axis: int) -> int:
    if axis not in (0, 1, 2):
        raise ValueError(f"position axis must be 0, 1, or 2; got {axis}")
    return axis


def finite_rows(value: Tensor) -> Tensor:
    """Return a ``[B]`` mask indicating finite tensor rows."""

    if value.ndim < 1:
        raise ValueError("finite_rows expects a batched tensor")
    if value.is_floating_point() or value.is_complex():
        return torch.isfinite(value).reshape(value.shape[0], -1).all(dim=1)
    return torch.ones(value.shape[0], dtype=torch.bool, device=value.device)


def as_batch_tensor(
    value: Optional[BatchValue],
    batch: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    name: str = "value",
) -> Optional[Tensor]:
    """Normalize a scalar or ``[B]`` value without moving it off device."""

    if value is None:
        return None
    result = torch.as_tensor(value, device=device, dtype=dtype)
    if result.ndim == 0:
        result = result.expand(batch).clone()
    elif result.shape != (batch,):
        raise ValueError(f"{name} must be scalar or shape [B], got {tuple(result.shape)}")
    return result


def first_rejection(
    reasons: list[str],
    accepted: Tensor,
    rows: Tensor,
    reason: Union[RejectReason, str],
) -> None:
    """Reject selected rows while preserving the first causal reason."""

    reason_value = reason.value if isinstance(reason, RejectReason) else str(reason)
    selected = (rows & accepted).nonzero(as_tuple=False).flatten().tolist()
    for row in selected:
        reasons[row] = reason_value
    accepted &= ~rows


def validate_pair_rows(
    embeddings: Tensor,
    position_ids: Tensor,
    mask_a: Tensor,
    mask_b: Tensor,
) -> Tuple[Tensor, list[str]]:
    """Return initial per-row validity and stable first-reason codes."""

    batch, _ = validate_tensor_primitives(embeddings, position_ids, mask_a, mask_b)
    accepted = torch.ones(batch, dtype=torch.bool, device=embeddings.device)
    reasons = [RejectReason.NONE.value for _ in range(batch)]
    first_rejection(reasons, accepted, ~finite_rows(embeddings), RejectReason.NON_FINITE_INPUT)
    first_rejection(reasons, accepted, ~finite_rows(position_ids), RejectReason.NON_FINITE_INPUT)
    first_rejection(reasons, accepted, ~mask_a.any(dim=1), RejectReason.EMPTY_MASK_A)
    first_rejection(reasons, accepted, ~mask_b.any(dim=1), RejectReason.EMPTY_MASK_B)
    first_rejection(reasons, accepted, (mask_a & mask_b).any(dim=1), RejectReason.MASKS_OVERLAP)
    return accepted, reasons


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Per-row masked mean for ``[B,N]`` or ``[B,N,K]`` values."""

    if values.ndim not in (2, 3) or mask.shape != values.shape[:2]:
        raise ValueError("masked_mean expects values [B,N] or [B,N,K] and mask [B,N]")
    weights = mask.to(values.dtype)
    denominator = weights.sum(dim=1).clamp_min(1)
    if values.ndim == 3:
        weights = weights.unsqueeze(-1)
        denominator = denominator.unsqueeze(-1)
    return (values * weights).sum(dim=1) / denominator


def masked_min(values: Tensor, mask: Tensor) -> Tensor:
    """Per-row masked minimum for a ``[B,N]`` coordinate tensor."""

    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError("masked_min expects values and mask with shape [B,N]")
    if values.dtype == torch.bool:
        fill = True
    elif values.is_floating_point():
        fill = torch.tensor(float("inf"), device=values.device, dtype=values.dtype)
    else:
        fill = torch.iinfo(values.dtype).max
    return values.masked_fill(~mask, fill).min(dim=1).values


def relation_sign(position_ids: Tensor, mask_a: Tensor, mask_b: Tensor, axis: int = 2) -> Tensor:
    """Sign of ``center(A)-center(B)`` along one position axis."""

    validate_axis(axis)
    if position_ids.ndim != 3 or position_ids.shape[-1] != 3:
        raise ValueError("position_ids must have shape [B,N,3]")
    if mask_a.shape != position_ids.shape[:2] or mask_b.shape != position_ids.shape[:2]:
        raise ValueError("mask_a and mask_b must have shape [B,N]")
    if mask_a.dtype != torch.bool or mask_b.dtype != torch.bool:
        raise ValueError("mask_a and mask_b must have dtype torch.bool")
    if mask_a.device != position_ids.device or mask_b.device != position_ids.device:
        raise ValueError("position_ids and masks must be on the same device")
    coordinate = position_ids[..., axis].to(torch.float64)
    center_a = masked_mean(coordinate, mask_a)
    center_b = masked_mean(coordinate, mask_b)
    return torch.sign(center_a - center_b).to(torch.int8)


def coordinates_in_bounds(
    position_ids: Tensor,
    touched_mask: Tensor,
    axis: int,
    bounds: Optional[CoordinateBounds],
) -> Tensor:
    """Check edited coordinates against global or per-row bounds."""

    batch = position_ids.shape[0]
    normalized_bounds = normalize_coordinate_bounds(
        bounds, batch, device=position_ids.device
    )
    if normalized_bounds is None:
        return torch.ones(batch, dtype=torch.bool, device=position_ids.device)
    lower = normalized_bounds[:, 0].unsqueeze(1)
    upper = normalized_bounds[:, 1].unsqueeze(1)
    coordinate = position_ids[..., validate_axis(axis)].to(torch.float64)
    bad = touched_mask & ((coordinate < lower) | (coordinate > upper))
    return ~bad.any(dim=1)


def normalize_coordinate_bounds(
    bounds: Optional[CoordinateBounds],
    batch: int,
    *,
    device: torch.device,
) -> Optional[Tensor]:
    """Normalize ``[2]`` global or ``[B,2]`` per-row coordinate bounds."""

    if bounds is None:
        return None
    result = torch.as_tensor(bounds, device=device, dtype=torch.float64)
    if result.shape == (2,):
        result = result.expand(batch, 2).clone()
    elif result.shape != (batch, 2):
        raise ValueError(
            "coordinate_bounds must have shape [2] or [B,2]; "
            f"got {tuple(result.shape)} for batch size {batch}"
        )
    if bool(torch.isnan(result).any().item()):
        raise ValueError("coordinate_bounds must not contain NaN")
    if bool((result[:, 0] > result[:, 1]).any().item()):
        raise ValueError("coordinate_bounds must satisfy lower <= upper in every row")
    return result


def l1_position_displacement(before: Tensor, after: Tensor, touched_mask: Tensor) -> Tensor:
    """Sum of absolute coordinate motion over all touched tokens and axes."""

    if before.shape != after.shape or before.ndim != 3:
        raise ValueError("before and after position tensors must share shape [B,N,3]")
    if touched_mask.shape != before.shape[:2]:
        raise ValueError("touched_mask must have shape [B,N]")
    delta = (after.to(torch.float64) - before.to(torch.float64)).abs().sum(dim=-1)
    return (delta * touched_mask.to(delta.dtype)).sum(dim=1).to(torch.float32)


def pair_transport_displacement(position_ids: Tensor, mask_a: Tensor, mask_b: Tensor) -> Tensor:
    """Audit displacement for swapping content between two token groups.

    Content does not alter ``position_ids``.  We therefore measure how far the
    two groups' payloads travel: Euclidean center distance times the number of
    destination tokens.
    """

    centers_a = masked_mean(position_ids.to(torch.float64), mask_a)
    centers_b = masked_mean(position_ids.to(torch.float64), mask_b)
    center_distance = torch.linalg.vector_norm(centers_a - centers_b, ord=2, dim=-1)
    count = (mask_a | mask_b).sum(dim=1).to(center_distance.dtype)
    return (center_distance * count).to(torch.float32)


def rectangular_token_grid(
    position_ids_row: Tensor,
    mask_row: Tensor,
) -> Tuple[Optional[RectangularTokenGrid], Optional[RejectReason]]:
    """Parse one masked Qwen ROI as a complete row-major ``H x W`` grid.

    A valid ROI contains exactly one token for every coordinate in a
    contiguous height/width Cartesian product and lies on one temporal plane.
    The returned indices are ordered by height first and width second,
    independent of their original sequence order.
    """

    if position_ids_row.ndim != 2 or position_ids_row.shape[-1] != 3:
        raise ValueError("position_ids_row must have shape [N,3]")
    if mask_row.shape != position_ids_row.shape[:1] or mask_row.dtype != torch.bool:
        raise ValueError("mask_row must be boolean with shape [N]")
    if mask_row.device != position_ids_row.device:
        raise ValueError("position_ids_row and mask_row must be on the same device")

    token_indices = mask_row.nonzero(as_tuple=False).flatten()
    if token_indices.numel() == 0:
        raise ValueError("rectangular_token_grid requires a non-empty mask")
    coordinates = position_ids_row[token_indices]
    if torch.unique(coordinates[:, 0]).numel() != 1:
        return None, RejectReason.MULTIPLE_TEMPORAL_PLANES

    heights = torch.unique(coordinates[:, 1], sorted=True)
    widths = torch.unique(coordinates[:, 2], sorted=True)
    contiguous_height = heights.numel() == 1 or bool(
        (torch.diff(heights) == 1).all().item()
    )
    contiguous_width = widths.numel() == 1 or bool(
        (torch.diff(widths) == 1).all().item()
    )
    height = int(heights.numel())
    width = int(widths.numel())
    if (
        not contiguous_height
        or not contiguous_width
        or height * width != int(token_indices.numel())
    ):
        return None, RejectReason.NON_RECTANGULAR_ROI

    height_rank = torch.searchsorted(heights, coordinates[:, 1].contiguous())
    width_rank = torch.searchsorted(widths, coordinates[:, 2].contiguous())
    linear_rank = height_rank * width + width_rank
    sorted_rank, order = torch.sort(linear_rank)
    expected_rank = torch.arange(
        height * width, device=linear_rank.device, dtype=linear_rank.dtype
    )
    if not torch.equal(sorted_rank, expected_rank):
        # This also rejects duplicate spatial coordinates, even when the token
        # count happens to match H*W.
        return None, RejectReason.NON_RECTANGULAR_ROI
    return (
        RectangularTokenGrid(
            token_indices=token_indices[order],
            height=height,
            width=width,
        ),
        None,
    )


def validate_rectangular_pair_grids(
    position_ids: Tensor,
    mask_a: Tensor,
    mask_b: Tensor,
    accepted: Tensor,
    reasons: list[str],
) -> Tuple[list[Optional[RectangularTokenGrid]], list[Optional[RectangularTokenGrid]]]:
    """Validate and parse both ROIs for every provisionally accepted row."""

    batch = position_ids.shape[0]
    grids_a: list[Optional[RectangularTokenGrid]] = [None] * batch
    grids_b: list[Optional[RectangularTokenGrid]] = [None] * batch
    for row in accepted.nonzero(as_tuple=False).flatten().tolist():
        grid_a, reason_a = rectangular_token_grid(position_ids[row], mask_a[row])
        if reason_a is not None:
            reasons[row] = reason_a.value
            accepted[row] = False
            continue
        grid_b, reason_b = rectangular_token_grid(position_ids[row], mask_b[row])
        if reason_b is not None:
            reasons[row] = reason_b.value
            accepted[row] = False
            continue
        grids_a[row] = grid_a
        grids_b[row] = grid_b
    return grids_a, grids_b


def resample_token_grid(
    tokens: Tensor,
    source_shape: Tuple[int, int],
    target_shape: Tuple[int, int],
) -> Tensor:
    """Bilinearly resize row-major ``[H*W,D]`` tokens on their true 2D grid."""

    if tokens.ndim != 2:
        raise ValueError("tokens must have shape [H*W,D]")
    source_height, source_width = source_shape
    target_height, target_width = target_shape
    if min(source_height, source_width, target_height, target_width) < 1:
        raise ValueError("source and target grid dimensions must be >= 1")
    if tokens.shape[0] != source_height * source_width:
        raise ValueError("token count must equal source_height * source_width")
    if source_shape == target_shape:
        return tokens.clone()

    # interpolate consumes NCHW.  A float32 work buffer keeps CPU half/bfloat16
    # supported without detaching autograd; the result returns to model dtype.
    source_dtype = tokens.dtype
    work = tokens.reshape(source_height, source_width, -1).permute(2, 0, 1).unsqueeze(0)
    if work.dtype in (torch.float16, torch.bfloat16):
        work = work.float()
    result = F.interpolate(
        work,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )
    return (
        result.squeeze(0)
        .permute(1, 2, 0)
        .reshape(target_height * target_width, -1)
        .to(source_dtype)
    )


def resample_token_sequence(tokens: Tensor, target_length: int) -> Tensor:
    """Differentiably resample a ``[M,D]`` token sequence to ``[L,D]``."""

    if tokens.ndim != 2 or tokens.shape[0] < 1:
        raise ValueError("tokens must have shape [M,D] with M >= 1")
    if target_length < 1:
        raise ValueError("target_length must be >= 1")
    if target_length == tokens.shape[0]:
        return tokens.clone()
    # interpolate supports the floating embedding dtypes used by the model.  A
    # float32 work buffer avoids unsupported CPU half interpolation and is cast
    # back without detaching the graph.
    source_dtype = tokens.dtype
    work = tokens.transpose(0, 1).unsqueeze(0)
    if work.dtype in (torch.float16, torch.bfloat16):
        work = work.float()
    result = F.interpolate(work, size=target_length, mode="linear", align_corners=False)
    return result.squeeze(0).transpose(0, 1).to(source_dtype)


def finalize_result(
    *,
    input_embeddings: Tensor,
    input_position_ids: Tensor,
    output_embeddings: Tensor,
    output_position_ids: Tensor,
    attempted_touched_mask: Tensor,
    accepted: Tensor,
    reasons: list[str],
    attempted_displacement: Optional[Tensor] = None,
    displacement_budget: Optional[BatchValue] = None,
    touched_token_budget: Optional[BatchValue] = None,
    active_visual_token_mask: Optional[Tensor] = None,
    touched_token_ratio_budget: Optional[BatchValue] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> OperatorResult:
    """Apply budgets/finite checks and transactionally roll back rejects.

    When ``active_visual_token_mask`` is supplied, touched-token ratios use its
    per-row population as the denominator.  Attempts that touch a token outside
    that mask are rejected.  Omitting it preserves the visual-only tensor
    contract, where the full sequence length ``N`` is the denominator.
    """

    masks = (attempted_touched_mask,)
    if active_visual_token_mask is not None:
        masks += (active_visual_token_mask,)
    batch, tokens = validate_tensor_primitives(input_embeddings, input_position_ids, *masks)
    if output_embeddings.shape != input_embeddings.shape:
        raise ValueError("operator changed the embeddings tensor shape")
    if output_position_ids.shape != input_position_ids.shape:
        raise ValueError("operator changed the position_ids tensor shape")
    if accepted.shape != (batch,) or accepted.dtype != torch.bool:
        raise ValueError("accepted must be a boolean tensor with shape [B]")
    if len(reasons) != batch:
        raise ValueError("reasons must contain one code per batch item")

    attempted_touched = attempted_touched_mask.sum(dim=1).to(torch.long)
    if active_visual_token_mask is None:
        if tokens == 0:
            raise ValueError("token dimension N must be positive")
        ratio_denominator = torch.full(
            (batch,), tokens, device=input_embeddings.device, dtype=torch.long
        )
        attempted_outside_active = torch.zeros(
            batch, device=input_embeddings.device, dtype=torch.long
        )
    else:
        ratio_denominator = active_visual_token_mask.sum(dim=1).to(torch.long)
        first_rejection(
            reasons,
            accepted,
            ratio_denominator == 0,
            RejectReason.EMPTY_ACTIVE_VISUAL_MASK,
        )
        attempted_outside_active = (
            attempted_touched_mask & ~active_visual_token_mask
        ).sum(dim=1)
        first_rejection(
            reasons,
            accepted,
            attempted_outside_active > 0,
            RejectReason.NON_VISUAL_TOKEN_TOUCHED,
        )
    attempted_ratio = attempted_touched.to(torch.float32) / ratio_denominator.clamp_min(1).to(
        torch.float32
    )
    if attempted_displacement is None:
        attempted_displacement = l1_position_displacement(
            input_position_ids, output_position_ids, attempted_touched_mask
        )
    else:
        attempted_displacement = as_batch_tensor(
            attempted_displacement,
            batch,
            device=input_embeddings.device,
            dtype=torch.float32,
            name="attempted_displacement",
        )
        assert attempted_displacement is not None

    finite_output = finite_rows(output_embeddings) & finite_rows(output_position_ids)
    first_rejection(reasons, accepted, ~finite_output, RejectReason.NON_FINITE_OUTPUT)

    displacement_limit = as_batch_tensor(
        displacement_budget,
        batch,
        device=input_embeddings.device,
        dtype=torch.float32,
        name="displacement_budget",
    )
    if displacement_limit is not None:
        if bool((displacement_limit < 0).any().item()):
            raise ValueError("displacement_budget must be non-negative")
        exceeded = attempted_displacement > displacement_limit
        first_rejection(
            reasons, accepted, exceeded, RejectReason.DISPLACEMENT_BUDGET_EXCEEDED
        )

    touched_limit = as_batch_tensor(
        touched_token_budget,
        batch,
        device=input_embeddings.device,
        dtype=torch.float32,
        name="touched_token_budget",
    )
    if touched_limit is not None:
        if bool((touched_limit < 0).any().item()):
            raise ValueError("touched_token_budget must be non-negative")
        exceeded = attempted_touched.to(torch.float32) > touched_limit
        first_rejection(
            reasons, accepted, exceeded, RejectReason.TOUCHED_TOKEN_BUDGET_EXCEEDED
        )

    touched_ratio_limit = as_batch_tensor(
        touched_token_ratio_budget,
        batch,
        device=input_embeddings.device,
        dtype=torch.float32,
        name="touched_token_ratio_budget",
    )
    if touched_ratio_limit is not None:
        invalid_ratio_limit = (
            ~torch.isfinite(touched_ratio_limit)
            | (touched_ratio_limit < 0)
            | (touched_ratio_limit > 1)
        )
        if bool(invalid_ratio_limit.any().item()):
            raise ValueError("touched_token_ratio_budget must be finite and within [0, 1]")
        exceeded = attempted_ratio > touched_ratio_limit
        first_rejection(
            reasons,
            accepted,
            exceeded,
            RejectReason.TOUCHED_TOKEN_RATIO_BUDGET_EXCEEDED,
        )

    # A rejected attempt never leaks a partial intervention downstream.
    row_selector = accepted.view(batch, 1, 1)
    final_embeddings = torch.where(row_selector, output_embeddings, input_embeddings.clone())
    final_positions = torch.where(row_selector, output_position_ids, input_position_ids.clone())
    final_touched_mask = attempted_touched_mask & accepted.unsqueeze(1)
    final_touched = final_touched_mask.sum(dim=1).to(torch.long)
    final_ratio = final_touched.to(torch.float32) / ratio_denominator.clamp_min(1).to(
        torch.float32
    )
    final_displacement = torch.where(
        accepted, attempted_displacement, torch.zeros_like(attempted_displacement)
    )

    audit_metadata = dict(metadata or {})
    audit_metadata.setdefault("attempted_touched_token_count", attempted_touched)
    audit_metadata.setdefault("attempted_touched_token_ratio", attempted_ratio)
    audit_metadata.setdefault("touched_token_ratio_denominator", ratio_denominator)
    audit_metadata.setdefault(
        "attempted_non_visual_token_count", attempted_outside_active
    )
    audit_metadata.setdefault(
        "active_visual_token_mask_provided", active_visual_token_mask is not None
    )
    audit_metadata.setdefault("attempted_total_displacement", attempted_displacement)

    result = OperatorResult(
        embeddings=final_embeddings,
        position_ids=final_positions,
        touched_token_mask=final_touched_mask,
        accepted=accepted.clone(),
        reject_reason=tuple(reasons),
        touched_token_count=final_touched,
        touched_token_ratio=final_ratio,
        total_displacement=final_displacement,
        displacement_budget=displacement_limit,
        touched_token_ratio_budget=touched_ratio_limit,
        metadata=audit_metadata,
    )
    result.assert_finite()
    return result


def reasons_from_result(result: OperatorResult) -> list[str]:
    """Mutable copy for wrappers that add further validation."""

    return list(result.reject_reason)


__all__ = [
    "BatchValue",
    "CoordinateBounds",
    "OperatorResult",
    "RectangularTokenGrid",
    "RejectReason",
    "as_batch_tensor",
    "coordinates_in_bounds",
    "finalize_result",
    "finite_rows",
    "first_rejection",
    "l1_position_displacement",
    "masked_mean",
    "masked_min",
    "normalize_coordinate_bounds",
    "pair_transport_displacement",
    "relation_sign",
    "rectangular_token_grid",
    "resample_token_grid",
    "resample_token_sequence",
    "validate_axis",
    "validate_pair_rows",
    "validate_rectangular_pair_grids",
    "validate_tensor_primitives",
]
