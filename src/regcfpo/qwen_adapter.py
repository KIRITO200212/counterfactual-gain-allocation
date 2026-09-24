"""Small, testable contracts at the transformers Qwen2.5-VL boundary.

The counterfactual operators intentionally use model-independent tensors:
``inputs_embeds`` have shape ``[B, S, D]`` and M-RoPE positions have shape
``[B, S, 3]``.  Transformers 4.49's Qwen2.5-VL implementation instead expects
positions in ``[3, B, S]`` order.  This module is the only place that needs to
know that detail.

No function here loads a model.  In particular, candidate scoring is a pure
tensor primitive and the teacher-forcing contract explicitly rejects cached or
generation calls.  This keeps adapter behavior unit-testable before an A40 is
used for an integration smoke test.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Optional, Sequence, Union

import torch
from torch import Tensor
import torch.nn.functional as F


TensorLike = Union[Tensor, Sequence[Sequence[float]]]


def _is_integer_dtype(dtype: torch.dtype) -> bool:
    return dtype in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }


def _require_integer_tensor(value: Any, name: str, ndim: int) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != ndim:
        raise ValueError(f"{name} must be a torch.Tensor with {ndim} dimensions")
    if not _is_integer_dtype(value.dtype):
        raise ValueError(f"{name} must have an integer dtype, got {value.dtype}")
    return value


def _metadata_tensor(
    value: TensorLike,
    *,
    name: str,
    shape: tuple[int, int],
    device: torch.device,
) -> Tensor:
    """Normalize numeric CPU metadata onto the model-input device."""

    try:
        result = torch.as_tensor(value, dtype=torch.float64, device=device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{name} must be numeric and convertible to a tensor") from error
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(result.shape)}")
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return result


def boxes_to_visual_token_masks(
    input_ids: Tensor,
    image_grid_thw: Tensor,
    boxes_xyxy: TensorLike,
    original_image_sizes_hw: TensorLike,
    *,
    image_token_id: int,
    spatial_merge_size: int,
    attention_mask: Optional[Tensor] = None,
) -> Tensor:
    """Map one original-image ``xyxy`` box per sample to Qwen LLM tokens.

    This Phase-1 adapter deliberately supports exactly one image per batch
    sample.  Accordingly, ``image_grid_thw`` is ``[B, 3]`` rather than Qwen's
    more general flattened ``[num_images, 3]`` representation, and every row
    must contain one contiguous image-token block.  Rejecting multi-block
    prompts is safer than silently pairing a box with the wrong image.

    ``original_image_sizes_hw`` contains ``(height, width)``.  Boxes use
    original-image pixel coordinates and are clipped to the image boundary.
    A decoder visual cell is selected when its center is in the clipped box,
    using half-open bounds ``x1 <= x < x2`` and ``y1 <= y < y2``.  The same
    spatial selection is repeated for every temporal grid index.  This is
    equivalent to scaling the box to Qwen's resized image, but does not require
    a hard-coded patch size.

    Transformers 4.49 restores vision outputs to temporal/height/width
    row-major order before scattering them into ``image_token_id`` slots.  The
    local mask follows that exact ``t -> h -> w`` order.  For each sample the
    active image-token count is strictly checked against
    ``t * (h / merge) * (w / merge)``.

    Args:
        input_ids: Full padded token ids with shape ``[B, S]``.
        image_grid_thw: One Qwen pre-merge grid ``(t, h, w)`` per sample.
        boxes_xyxy: One ``(x1, y1, x2, y2)`` box per sample.
        original_image_sizes_hw: One actual ``(height, width)`` per sample.
        image_token_id: ``model.config.image_token_id``.
        spatial_merge_size: ``model.config.vision_config.spatial_merge_size``.
        attention_mask: Optional binary mask with shape ``[B, S]``.  Padding is
            never selected.

    Returns:
        A boolean ``[B, S]`` mask on the same device as ``input_ids``.

    Raises:
        ValueError: For malformed shapes/dtypes, non-divisible spatial grids,
            degenerate/empty mapped boxes, discontiguous/multiple visual
            blocks, or any visual-token count mismatch.
    """

    input_ids = _require_integer_tensor(input_ids, "input_ids", 2)
    batch, sequence_length = input_ids.shape
    if isinstance(image_token_id, bool) or not isinstance(image_token_id, Integral):
        raise ValueError("image_token_id must be an integer")
    if (
        isinstance(spatial_merge_size, bool)
        or not isinstance(spatial_merge_size, Integral)
        or spatial_merge_size <= 0
    ):
        raise ValueError("spatial_merge_size must be a positive integer")
    spatial_merge_size = int(spatial_merge_size)

    image_grid_thw = _require_integer_tensor(image_grid_thw, "image_grid_thw", 2)
    if image_grid_thw.shape != (batch, 3):
        raise ValueError(
            "single-image adapter requires image_grid_thw with shape "
            f"[B,3] = {(batch, 3)}, got {tuple(image_grid_thw.shape)}"
        )
    if image_grid_thw.device != input_ids.device:
        raise ValueError("image_grid_thw and input_ids must be on the same device")

    boxes = _metadata_tensor(
        boxes_xyxy,
        name="boxes_xyxy",
        shape=(batch, 4),
        device=input_ids.device,
    )
    image_sizes = _metadata_tensor(
        original_image_sizes_hw,
        name="original_image_sizes_hw",
        shape=(batch, 2),
        device=input_ids.device,
    )
    if bool((image_sizes <= 0).any().item()):
        raise ValueError("original image height and width must be positive")

    if attention_mask is None:
        active = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        if not isinstance(attention_mask, Tensor):
            raise ValueError("attention_mask must be a torch.Tensor with 2 dimensions")
        if attention_mask.dtype != torch.bool:
            attention_mask = _require_integer_tensor(attention_mask, "attention_mask", 2)
        if attention_mask.shape != (batch, sequence_length):
            raise ValueError(
                "attention_mask must match input_ids shape; "
                f"got {tuple(attention_mask.shape)} and {tuple(input_ids.shape)}"
            )
        if attention_mask.device != input_ids.device:
            raise ValueError("attention_mask and input_ids must be on the same device")
        if not bool(((attention_mask == 0) | (attention_mask == 1)).all().item()):
            raise ValueError("attention_mask must contain only 0/1 values")
        active = attention_mask.to(torch.bool)

    if bool((image_grid_thw <= 0).any().item()):
        raise ValueError("all image_grid_thw entries must be positive")

    result = torch.zeros((batch, sequence_length), dtype=torch.bool, device=input_ids.device)
    for row in range(batch):
        temporal, grid_height, grid_width = (
            int(value) for value in image_grid_thw[row].tolist()
        )
        if grid_height % spatial_merge_size or grid_width % spatial_merge_size:
            raise ValueError(
                f"sample {row} spatial grid ({grid_height}, {grid_width}) is not divisible "
                f"by spatial_merge_size={spatial_merge_size}"
            )
        llm_height = grid_height // spatial_merge_size
        llm_width = grid_width // spatial_merge_size
        expected_tokens = temporal * llm_height * llm_width

        token_positions = torch.nonzero(
            (input_ids[row] == int(image_token_id)) & active[row], as_tuple=False
        ).flatten()
        actual_tokens = int(token_positions.numel())
        if actual_tokens != expected_tokens:
            raise ValueError(
                f"sample {row} image token count mismatch: expected {expected_tokens} from "
                f"grid {temporal}x{grid_height}x{grid_width} and merge "
                f"{spatial_merge_size}, found {actual_tokens}"
            )
        if actual_tokens > 1 and not bool(
            (token_positions[1:] == token_positions[:-1] + 1).all().item()
        ):
            raise ValueError(
                f"sample {row} image tokens form multiple/discontiguous blocks; "
                "the single-image adapter requires one contiguous block"
            )

        image_height, image_width = image_sizes[row]
        box = boxes[row]
        x1 = box[0].clamp_min(0).minimum(image_width)
        y1 = box[1].clamp_min(0).minimum(image_height)
        x2 = box[2].clamp_min(0).minimum(image_width)
        y2 = box[3].clamp_min(0).minimum(image_height)
        if not bool(((x1 < x2) & (y1 < y2)).item()):
            raise ValueError(f"sample {row} box is empty after clipping to the image boundary")

        local_indices = torch.arange(expected_tokens, device=input_ids.device)
        spatial_indices = local_indices.remainder(llm_height * llm_width)
        cell_y = torch.div(spatial_indices, llm_width, rounding_mode="floor")
        cell_x = spatial_indices.remainder(llm_width)
        center_x = (cell_x.to(torch.float64) + 0.5) * image_width / llm_width
        center_y = (cell_y.to(torch.float64) + 0.5) * image_height / llm_height
        local_mask = (
            (center_x >= x1)
            & (center_x < x2)
            & (center_y >= y1)
            & (center_y < y2)
        )
        if not bool(local_mask.any().item()):
            raise ValueError(
                f"sample {row} box maps to no visual token centers at LLM grid "
                f"{temporal}x{llm_height}x{llm_width}"
            )
        result[row, token_positions] = local_mask

    return result


def qwen_to_operator_position_ids(position_ids: Tensor) -> Tensor:
    """Convert Qwen ``[3, B, S]`` M-RoPE ids to operator ``[B, S, 3]``."""

    position_ids = _require_integer_tensor(position_ids, "position_ids", 3)
    if position_ids.shape[0] != 3:
        raise ValueError(
            "Qwen position_ids must have shape [3,B,S], "
            f"got {tuple(position_ids.shape)}"
        )
    return position_ids.permute(1, 2, 0).contiguous()


def operator_to_qwen_position_ids(position_ids: Tensor) -> Tensor:
    """Convert operator ``[B, S, 3]`` M-RoPE ids to Qwen ``[3, B, S]``."""

    position_ids = _require_integer_tensor(position_ids, "position_ids", 3)
    if position_ids.shape[-1] != 3:
        raise ValueError(
            "operator position_ids must have shape [B,S,3], "
            f"got {tuple(position_ids.shape)}"
        )
    return position_ids.permute(2, 0, 1).contiguous()


def validate_teacher_forced_call(
    *,
    use_cache: Any,
    past_key_values: Any = None,
    cache_position: Any = None,
    for_generation: bool = False,
) -> None:
    """Reject unsafe call modes for a full-sequence intervention.

    Custom visual M-RoPE positions are only supported here for one complete,
    teacher-forced forward pass.  A cached continuation would need a separately
    proven RoPE-delta policy, and ``generate`` may create or reuse that cache
    internally.  Callers must therefore state ``use_cache=False`` explicitly.
    """

    if for_generation:
        raise ValueError("intervention inputs are scoring-only and cannot be used with generate()")
    if use_cache is not False:
        raise ValueError("teacher-forced intervention requires explicit use_cache=False")
    if past_key_values is not None:
        raise ValueError("past_key_values are forbidden for teacher-forced intervention")
    if cache_position is not None:
        raise ValueError("cache_position is forbidden for teacher-forced intervention")


@dataclass(frozen=True)
class TeacherForcedInterventionInputs:
    """Validated full-sequence inputs for Qwen position/content interventions.

    ``inputs_embeds`` must already contain both token embeddings and the Qwen
    visual embeddings scattered into their full-sequence slots.  It is the
    single content authority: raw ``input_ids``/``pixel_values`` are purposely
    absent from :meth:`as_model_kwargs`, avoiding an accidental second visual
    scatter.  ``position_ids`` is stored in Qwen's native ``[3, B, S]`` layout.
    The embeddings must come from the exact teacher-forced token sequence
    represented by ``labels``.  In particular, for a multi-token candidate,
    the embedding of each earlier candidate token must be present before a
    later candidate token is scored; changing labels alone is not sufficient.

    ``candidate_mask`` marks label coordinates, not logit coordinates.  Thus a
    candidate token at sequence index ``j`` is scored by logits at ``j - 1``.
    Index zero cannot be selected.
    """

    inputs_embeds: Tensor
    position_ids: Tensor
    attention_mask: Tensor
    labels: Tensor
    candidate_mask: Tensor

    def __post_init__(self) -> None:
        _validate_teacher_forced_tensors(
            self.inputs_embeds,
            self.position_ids,
            self.attention_mask,
            self.labels,
            self.candidate_mask,
        )

    def as_model_kwargs(self) -> dict[str, Any]:
        """Return kwargs for ``model.forward`` (never for ``model.generate``).

        Labels stay on this contract for the span scorer but are not passed to
        Qwen.  Transformers 4.49 otherwise upcasts the entire ``[B,S,V]``
        logits tensor and computes an unrelated full-sequence cross entropy.
        """

        validate_teacher_forced_call(use_cache=False)
        return {
            "inputs_embeds": self.inputs_embeds,
            "position_ids": self.position_ids,
            "attention_mask": self.attention_mask,
            "use_cache": False,
            "return_dict": True,
        }

    def validate_call(
        self,
        *,
        use_cache: Any,
        past_key_values: Any = None,
        cache_position: Any = None,
        for_generation: bool = False,
    ) -> None:
        """Validate extra call state before dispatching a forward pass."""

        validate_teacher_forced_call(
            use_cache=use_cache,
            past_key_values=past_key_values,
            cache_position=cache_position,
            for_generation=for_generation,
        )


def _validate_teacher_forced_tensors(
    inputs_embeds: Tensor,
    qwen_position_ids: Tensor,
    attention_mask: Tensor,
    labels: Tensor,
    candidate_mask: Tensor,
) -> None:
    if not isinstance(inputs_embeds, Tensor) or inputs_embeds.ndim != 3:
        raise ValueError("inputs_embeds must be a torch.Tensor with shape [B,S,D]")
    if not inputs_embeds.is_floating_point():
        raise ValueError("inputs_embeds must have a floating point dtype")
    batch, sequence_length, _ = inputs_embeds.shape

    qwen_position_ids = _require_integer_tensor(qwen_position_ids, "position_ids", 3)
    if qwen_position_ids.shape != (3, batch, sequence_length):
        raise ValueError(
            "position_ids must have Qwen shape [3,B,S] matching inputs_embeds; "
            f"got {tuple(qwen_position_ids.shape)} for {tuple(inputs_embeds.shape)}"
        )
    labels = _require_integer_tensor(labels, "labels", 2)
    if labels.shape != (batch, sequence_length):
        raise ValueError("labels must have shape [B,S] matching inputs_embeds")
    if not isinstance(attention_mask, Tensor) or attention_mask.ndim != 2:
        raise ValueError("attention_mask must be a tensor with shape [B,S]")
    if attention_mask.shape != (batch, sequence_length):
        raise ValueError("attention_mask must have shape [B,S] matching inputs_embeds")
    if attention_mask.dtype != torch.bool and not _is_integer_dtype(attention_mask.dtype):
        raise ValueError("attention_mask must have a boolean or integer dtype")
    if not bool(((attention_mask == 0) | (attention_mask == 1)).all().item()):
        raise ValueError("attention_mask must contain only 0/1 values")
    if not isinstance(candidate_mask, Tensor) or candidate_mask.shape != (batch, sequence_length):
        raise ValueError("candidate_mask must be a tensor with shape [B,S]")
    if candidate_mask.dtype != torch.bool:
        raise ValueError("candidate_mask must have dtype torch.bool")

    tensors = {
        "position_ids": qwen_position_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "candidate_mask": candidate_mask,
    }
    for name, value in tensors.items():
        if value.device != inputs_embeds.device:
            raise ValueError(f"{name} and inputs_embeds must be on the same device")

    active = attention_mask.to(torch.bool)
    if bool((candidate_mask & ~active).any().item()):
        raise ValueError("candidate_mask cannot select padding")
    if bool(candidate_mask[:, 0].any().item()):
        raise ValueError("candidate_mask cannot select index 0, which has no causal predictor")
    if bool((candidate_mask[:, 1:] & ~active[:, :-1]).any().item()):
        raise ValueError("every candidate token must have an active preceding predictor token")
    _validate_candidate_spans(labels, candidate_mask, vocab_size=None)


def build_teacher_forced_intervention_inputs(
    inputs_embeds: Tensor,
    operator_position_ids: Tensor,
    attention_mask: Tensor,
    labels: Tensor,
    candidate_mask: Tensor,
) -> TeacherForcedInterventionInputs:
    """Build the safe Qwen forward contract from operator-layout positions."""

    return TeacherForcedInterventionInputs(
        inputs_embeds=inputs_embeds,
        position_ids=operator_to_qwen_position_ids(operator_position_ids),
        attention_mask=attention_mask,
        labels=labels,
        candidate_mask=candidate_mask,
    )


def _validate_candidate_spans(
    labels: Tensor,
    candidate_mask: Tensor,
    *,
    vocab_size: Optional[int],
) -> None:
    batch = labels.shape[0]
    for row in range(batch):
        positions = torch.nonzero(candidate_mask[row], as_tuple=False).flatten()
        if positions.numel() == 0:
            raise ValueError(f"candidate_mask row {row} is empty")
        if positions.numel() > 1 and not bool(
            (positions[1:] == positions[:-1] + 1).all().item()
        ):
            raise ValueError(f"candidate_mask row {row} must select one contiguous span")
        selected_labels = labels[row, positions]
        if bool((selected_labels < 0).any().item()):
            raise ValueError(f"candidate_mask row {row} selects a negative/ignored label")
        if vocab_size is not None and bool((selected_labels >= vocab_size).any().item()):
            raise ValueError(
                f"candidate_mask row {row} selects a label outside vocab size {vocab_size}"
            )


def candidate_span_log_probs(logits: Tensor, labels: Tensor, candidate_mask: Tensor) -> Tensor:
    """Sum causal candidate-span log probabilities in FP32 for each batch row.

    ``candidate_mask`` is aligned with ``labels``.  As in Qwen's causal-LM
    loss, ``logits[:, j - 1]`` predicts ``labels[:, j]``; index zero is
    therefore invalid.  Log-softmax and aggregation are always performed in
    FP32 even when model logits are FP16/BF16.  Gradients are preserved.  This
    low-level primitive has no attention mask; use
    :class:`TeacherForcedInterventionInputs` first (or independently verify)
    that each selected token and its preceding predictor are both active.
    """

    if not isinstance(logits, Tensor) or logits.ndim != 3:
        raise ValueError("logits must be a torch.Tensor with shape [B,S,V]")
    if not logits.is_floating_point():
        raise ValueError("logits must have a floating point dtype")
    batch, sequence_length, vocab_size = logits.shape
    if vocab_size <= 0:
        raise ValueError("logits vocabulary dimension must be positive")
    labels = _require_integer_tensor(labels, "labels", 2)
    if labels.shape != (batch, sequence_length):
        raise ValueError("labels must have shape [B,S] matching logits")
    if not isinstance(candidate_mask, Tensor) or candidate_mask.shape != (batch, sequence_length):
        raise ValueError("candidate_mask must be a tensor with shape [B,S]")
    if candidate_mask.dtype != torch.bool:
        raise ValueError("candidate_mask must have dtype torch.bool")
    if labels.device != logits.device or candidate_mask.device != logits.device:
        raise ValueError("logits, labels, and candidate_mask must be on the same device")
    if bool(candidate_mask[:, 0].any().item()):
        raise ValueError("candidate_mask cannot select index 0, which has no causal predictor")
    _validate_candidate_spans(labels, candidate_mask, vocab_size=vocab_size)

    locations = torch.nonzero(candidate_mask, as_tuple=False)
    row_indices = locations[:, 0]
    label_indices = locations[:, 1]
    selected_logits = logits[row_indices, label_indices - 1].float()
    if not bool(torch.isfinite(selected_logits).all().item()):
        raise FloatingPointError("candidate predictor logits contain NaN or Inf")
    targets = labels[row_indices, label_indices].to(torch.long)
    token_log_probs = F.log_softmax(selected_logits, dim=-1).gather(
        dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)
    result = torch.zeros(batch, dtype=torch.float32, device=logits.device)
    return result.scatter_add(0, row_indices, token_log_probs)


def candidate_span_mean_log_probs(
    logits: Tensor, labels: Tensor, candidate_mask: Tensor
) -> Tensor:
    """Mean causal log probability per candidate token for each batch row.

    This is the length-normalized score used by the frozen full-template
    training proxy.  Validation, FP32 accumulation, and gradient behavior are
    inherited from :func:`candidate_span_log_probs`; every candidate span is
    guaranteed to contain at least one token before the division is applied.
    """

    totals = candidate_span_log_probs(logits, labels, candidate_mask)
    token_counts = candidate_mask.sum(dim=1).to(dtype=totals.dtype)
    return totals / token_counts


aggregate_candidate_log_probs = candidate_span_log_probs
box_to_visual_token_mask = boxes_to_visual_token_masks


__all__ = [
    "TeacherForcedInterventionInputs",
    "aggregate_candidate_log_probs",
    "box_to_visual_token_mask",
    "boxes_to_visual_token_masks",
    "build_teacher_forced_intervention_inputs",
    "candidate_span_mean_log_probs",
    "candidate_span_log_probs",
    "operator_to_qwen_position_ids",
    "qwen_to_operator_position_ids",
    "validate_teacher_forced_call",
]
