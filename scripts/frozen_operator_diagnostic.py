#!/usr/bin/env python3
"""Run the frozen Qwen no-op and five-branch operator diagnostic.

This is experiment code.  It writes resumable, sample-keyed JSONL plus a
derived JSON summary.  Command, stdout, stderr, environment, and exit-status
capture remain the responsibility of ``infra/recording/run_logged.py``.

The diagnostic is deliberately batch-one and teacher-forced.  Original and
mapped candidates are separately tokenized and embedded, so a multi-token
candidate contains its own autoregressive prefix embeddings instead of merely
changing labels.  Custom M-RoPE positions are never passed to ``generate`` or
to a cached forward call.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from regcfpo.data import RelationPairV2
from regcfpo.operators import (
    content_slot_swap,
    context_transport,
    position_slot_swap,
    resampling_null,
)
from regcfpo.operators.geometry import OperatorResult
from regcfpo.qwen_adapter import (
    boxes_to_visual_token_masks,
    build_teacher_forced_intervention_inputs,
    candidate_span_log_probs,
    qwen_to_operator_position_ids,
)
from regcfpo.qwen_compat import (
    load_qwen_config_with_compat,
    validate_qwen_weight_tying,
)
from regcfpo.run_contract import ensure_run_contract
from regcfpo.stats import scene_level_paired_bootstrap


PROMPT_POLICY = "chat_template_answer_letter_only_v1"
CONDITIONS = (
    "factual",
    "global_hflip",
    "position_slot",
    "content_slot",
    "resampling_null",
    "context_transport_null",
)
LATENT_CONDITIONS = tuple(
    condition for condition in CONDITIONS if condition != "global_hflip"
)
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20270818


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_code_hashes() -> dict[str, str]:
    project_root = Path(__file__).resolve().parents[1]
    relative_paths = (
        "scripts/frozen_operator_diagnostic.py",
        "src/regcfpo/qwen_adapter.py",
        "src/regcfpo/qwen_compat.py",
        "src/regcfpo/stats.py",
        "src/regcfpo/operators/geometry.py",
        "src/regcfpo/operators/position_slot_swap.py",
        "src/regcfpo/operators/content_slot_swap.py",
        "src/regcfpo/operators/resampling_null.py",
        "src/regcfpo/operators/context_transport.py",
    )
    return {
        relative: sha256_file(project_root / relative) for relative in relative_paths
    }


@dataclass
class CandidateInputs:
    """One exact candidate sequence at the Qwen/operator boundary."""

    name: str
    text: str
    token_ids: list[int]
    input_ids: Tensor
    attention_mask: Tensor
    pixel_values: Tensor
    image_grid_thw: Tensor
    inputs_embeds: Tensor
    position_ids: Tensor
    labels: Tensor
    candidate_mask: Tensor
    active_visual_mask: Tensor
    mask_a: Tensor
    mask_b: Tensor


@dataclass
class CanonicalCandidateInputs:
    """One separately tokenized candidate for a raw multimodal forward."""

    name: str
    text: str
    token_ids: list[int]
    input_ids: Tensor
    attention_mask: Tensor
    pixel_values: Tensor
    image_grid_thw: Tensor
    labels: Tensor
    candidate_mask: Tensor


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{path}: every row must contain a non-empty sample_id")
        if sample_id in completed:
            raise ValueError(f"{path}: duplicate completed sample_id {sample_id!r}")
        completed[sample_id] = row
    return completed


def validate_completed_rows(
    completed: Mapping[str, Mapping[str, Any]],
    selected: list[dict[str, Any]],
) -> None:
    selected_by_id = {str(row["sample_id"]): row for row in selected}
    unexpected = sorted(set(completed) - set(selected_by_id))
    if unexpected:
        raise ValueError(f"completed output contains unselected sample IDs: {unexpected[:5]}")
    for sample_id, row in completed.items():
        expected = selected_by_id[sample_id]
        for field in ("sample_id", "scene_id"):
            if row.get(field) != expected[field]:
                raise ValueError(
                    f"completed row {sample_id!r} has mismatched {field}: "
                    f"{row.get(field)!r} != {expected[field]!r}"
                )
        if row.get("schema_version") != 1 or not isinstance(row.get("conditions"), dict):
            raise ValueError(f"completed row {sample_id!r} has an invalid diagnostic schema")


def diagnostic_prompt(record: Mapping[str, Any]) -> str:
    """Return the frozen answer-letter-only user text.

    This smoke prompt is intentionally distinct from the full CoT rollout
    prompt.  It tests the intervention/scoring path; it is not evidence that
    the answer-only proxy agrees with a generated CoT distribution.
    """

    options = "\n".join(str(option) for option in record["options"])
    return (
        f"Question: {record['question']}\nOptions:\n{options}\n"
        "Answer with only the option letter."
    )


def candidate_suffix_mask(base_input_ids: Tensor, full_input_ids: Tensor) -> Tensor:
    """Mark the non-empty candidate suffix after an exact tokenized prompt.

    Requiring exact prefix preservation avoids silently scoring a BPE token
    whose embedding was built from a different prompt boundary.  The returned
    span may contain any positive number of tokens.
    """

    if (
        not isinstance(base_input_ids, Tensor)
        or not isinstance(full_input_ids, Tensor)
        or base_input_ids.ndim != 2
        or full_input_ids.ndim != 2
        or base_input_ids.shape[0] != 1
        or full_input_ids.shape[0] != 1
    ):
        raise ValueError("base_input_ids and full_input_ids must have shape [1,S]")
    if base_input_ids.device != full_input_ids.device:
        raise ValueError("base_input_ids and full_input_ids must share a device")
    prompt_length = base_input_ids.shape[1]
    if full_input_ids.shape[1] <= prompt_length:
        raise ValueError("candidate must contribute at least one token")
    if not torch.equal(full_input_ids[:, :prompt_length], base_input_ids):
        raise ValueError(
            "tokenized prompt is not an exact prefix of the candidate sequence; "
            "use an explicit boundary that does not retokenize the prompt"
        )
    mask = torch.zeros_like(full_input_ids, dtype=torch.bool)
    mask[:, prompt_length:] = True
    return mask


def active_coordinate_bounds(
    position_ids: Tensor, active_visual_mask: Tensor, *, axis: int
) -> Tensor:
    """Derive per-row inclusive bounds from factual active visual positions."""

    if not isinstance(position_ids, Tensor) or position_ids.ndim != 3:
        raise ValueError("position_ids must have shape [B,S,3]")
    if position_ids.shape[-1] != 3:
        raise ValueError("position_ids must have shape [B,S,3]")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    if (
        not isinstance(active_visual_mask, Tensor)
        or active_visual_mask.shape != position_ids.shape[:2]
        or active_visual_mask.dtype != torch.bool
        or active_visual_mask.device != position_ids.device
    ):
        raise ValueError("active_visual_mask must be boolean [B,S] on the same device")
    bounds: list[Tensor] = []
    for row in range(position_ids.shape[0]):
        values = position_ids[row, active_visual_mask[row], axis]
        if values.numel() == 0:
            raise ValueError(f"active visual mask row {row} is empty")
        bounds.append(torch.stack((values.min(), values.max())))
    return torch.stack(bounds).to(torch.float64)


def _selected_predictor_logits(logits: Tensor, candidate_mask: Tensor) -> Tensor:
    locations = torch.nonzero(candidate_mask, as_tuple=False)
    return logits[locations[:, 0], locations[:, 1] - 1].float()


def noop_comparison(
    canonical_logits: Tensor,
    manual_logits: Tensor,
    labels: Tensor,
    candidate_mask: Tensor,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Compare only the causal predictors used by the candidate scorer."""

    if threshold <= 0:
        raise ValueError("threshold must be positive")
    canonical_selected = _selected_predictor_logits(canonical_logits, candidate_mask)
    manual_selected = _selected_predictor_logits(manual_logits, candidate_mask)
    if canonical_selected.shape != manual_selected.shape:
        raise ValueError("canonical and manual candidate predictor shapes differ")
    absolute = (canonical_selected - manual_selected).abs()
    canonical_score = candidate_span_log_probs(
        canonical_logits, labels, candidate_mask
    ).item()
    manual_score = candidate_span_log_probs(manual_logits, labels, candidate_mask).item()
    maximum = float(absolute.max().item())
    return {
        "candidate_predictor_count": int(candidate_mask.sum().item()),
        "compared_logit_count": int(absolute.numel()),
        "max_abs_predictor_logit_error": maximum,
        "mean_abs_predictor_logit_error": float(absolute.mean().item()),
        "canonical_candidate_logprob": float(canonical_score),
        "manual_candidate_logprob": float(manual_score),
        "candidate_logprob_abs_error": float(abs(canonical_score - manual_score)),
        "threshold": float(threshold),
        "passed": maximum < threshold,
    }


def _tensor_scalar(value: Any, *, row: int = 0) -> Any:
    if value is None:
        return None
    if isinstance(value, Tensor):
        selected = value[row] if value.ndim else value
        if selected.numel() == 1:
            return selected.item()
        return selected.detach().cpu().tolist()
    return value


def operator_audit(result: OperatorResult) -> dict[str, Any]:
    """Serialize the realized and pre-rollback metrics for one batch-one result."""

    audit = {
        "accepted": bool(result.accepted[0].item()),
        "reject_reason": result.reject_reason[0],
        "active_visual_token_count": int(
            _tensor_scalar(result.metadata["touched_token_ratio_denominator"])
        ),
        "touched_visual_token_count": int(result.touched_token_count[0].item()),
        "touched_visual_token_ratio": float(result.touched_token_ratio[0].item()),
        "total_displacement": float(result.total_displacement[0].item()),
        "attempted_touched_visual_token_count": int(
            _tensor_scalar(result.metadata["attempted_touched_token_count"])
        ),
        "attempted_touched_visual_token_ratio": float(
            _tensor_scalar(result.metadata["attempted_touched_token_ratio"])
        ),
        "attempted_total_displacement": float(
            _tensor_scalar(result.metadata["attempted_total_displacement"])
        ),
    }
    for name in ("target_displacement", "requested_shift", "realized_shift"):
        if name in result.metadata:
            audit[name] = _tensor_scalar(result.metadata[name])
    return audit


def _manual_scatter(
    model: Qwen2_5_VLForConditionalGeneration,
    input_ids: Tensor,
    pixel_values: Tensor,
    image_grid_thw: Tensor,
) -> Tensor:
    token_embeds = model.model.embed_tokens(input_ids)
    image_embeds = model.visual(
        pixel_values.to(dtype=model.visual.dtype), grid_thw=image_grid_thw
    ).to(device=token_embeds.device, dtype=token_embeds.dtype)
    active = input_ids == model.config.image_token_id
    if int(active.sum().item()) != image_embeds.shape[0]:
        raise ValueError(
            "manual visual scatter mismatch: "
            f"tokens={int(active.sum().item())}, features={image_embeds.shape[0]}"
        )
    return token_embeds.masked_scatter(active.unsqueeze(-1).expand_as(token_embeds), image_embeds)


def _processor_inputs(
    processor: AutoProcessor,
    *,
    text: str,
    image: Image.Image,
    device: str,
) -> dict[str, Tensor]:
    values = processor(text=[text], images=[image], padding=False, return_tensors="pt")
    required = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"processor output is missing required fields: {missing}")
    return {name: values[name].to(device) for name in required}


def prepare_candidate(
    *,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    prompt: str,
    candidate_name: str,
    candidate_text: str,
    image: Image.Image,
    record: Mapping[str, Any],
    device: str,
) -> CandidateInputs:
    """Tokenize, visually scatter, and locate one exact candidate sequence."""

    base = _processor_inputs(processor, text=prompt, image=image, device=device)
    full = _processor_inputs(
        processor, text=prompt + candidate_text, image=image, device=device
    )
    candidate_mask = candidate_suffix_mask(base["input_ids"], full["input_ids"])
    if not torch.equal(base["image_grid_thw"], full["image_grid_thw"]):
        raise ValueError("candidate text unexpectedly changed image_grid_thw")
    with torch.inference_mode():
        inputs_embeds = _manual_scatter(
            model,
            full["input_ids"],
            full["pixel_values"],
            full["image_grid_thw"],
        )
        qwen_positions, _ = model.get_rope_index(
            full["input_ids"],
            full["image_grid_thw"],
            None,
            None,
            full["attention_mask"],
        )
    position_ids = qwen_to_operator_position_ids(qwen_positions)
    active_visual = (
        (full["input_ids"] == model.config.image_token_id)
        & full["attention_mask"].to(torch.bool)
    )
    size_hw = (image.height, image.width)
    merge_size = model.config.vision_config.spatial_merge_size
    mask_a = boxes_to_visual_token_masks(
        full["input_ids"],
        full["image_grid_thw"],
        [record["gt_box_a"]],
        [size_hw],
        image_token_id=model.config.image_token_id,
        spatial_merge_size=merge_size,
        attention_mask=full["attention_mask"],
    )
    mask_b = boxes_to_visual_token_masks(
        full["input_ids"],
        full["image_grid_thw"],
        [record["gt_box_b"]],
        [size_hw],
        image_token_id=model.config.image_token_id,
        spatial_merge_size=merge_size,
        attention_mask=full["attention_mask"],
    )
    if bool(((mask_a | mask_b) & ~active_visual).any().item()):
        raise ValueError("object mask selected a non-visual token")
    return CandidateInputs(
        name=candidate_name,
        text=candidate_text,
        token_ids=full["input_ids"][candidate_mask].detach().cpu().tolist(),
        input_ids=full["input_ids"],
        attention_mask=full["attention_mask"],
        pixel_values=full["pixel_values"],
        image_grid_thw=full["image_grid_thw"],
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        labels=full["input_ids"].clone(),
        candidate_mask=candidate_mask,
        active_visual_mask=active_visual,
        mask_a=mask_a,
        mask_b=mask_b,
    )


def prepare_canonical_candidate(
    *,
    processor: AutoProcessor,
    prompt: str,
    candidate_name: str,
    candidate_text: str,
    image: Image.Image,
    device: str,
) -> CanonicalCandidateInputs:
    """Build an exact raw Qwen sequence without manual visual scattering."""

    base = _processor_inputs(processor, text=prompt, image=image, device=device)
    full = _processor_inputs(
        processor, text=prompt + candidate_text, image=image, device=device
    )
    candidate_mask = candidate_suffix_mask(base["input_ids"], full["input_ids"])
    if not torch.equal(base["image_grid_thw"], full["image_grid_thw"]):
        raise ValueError("candidate text unexpectedly changed image_grid_thw")
    return CanonicalCandidateInputs(
        name=candidate_name,
        text=candidate_text,
        token_ids=full["input_ids"][candidate_mask].detach().cpu().tolist(),
        input_ids=full["input_ids"],
        attention_mask=full["attention_mask"],
        pixel_values=full["pixel_values"],
        image_grid_thw=full["image_grid_thw"],
        labels=full["input_ids"].clone(),
        candidate_mask=candidate_mask,
    )


def score_canonical_candidate(
    model: Qwen2_5_VLForConditionalGeneration,
    candidate: CanonicalCandidateInputs,
) -> float:
    """Score one raw multimodal candidate with canonical automatic M-RoPE.

    ``position_ids``, ``past_key_values``, and ``cache_position`` are omitted
    rather than copied from a factual/custom branch.  With ``use_cache=False``
    and no past, Qwen recomputes the complete positions for this exact flipped
    sequence and never creates or consumes a KV cache.
    """

    with torch.inference_mode():
        logits = model(
            input_ids=candidate.input_ids,
            attention_mask=candidate.attention_mask,
            pixel_values=candidate.pixel_values,
            image_grid_thw=candidate.image_grid_thw,
            use_cache=False,
            return_dict=True,
        ).logits
        score = candidate_span_log_probs(
            logits, candidate.labels, candidate.candidate_mask
        )[0].item()
    del logits
    return float(score)


def horizontal_flip_image(image: Image.Image) -> Image.Image:
    """Return a new canonical left-right pixel flip without mutating input."""

    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


def run_global_hflip(
    *,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    prompt: str,
    record: Mapping[str, Any],
    image: Image.Image,
    device: str,
) -> dict[str, Any]:
    """Score the canonical global horizontal-image-flip positive control."""

    _cuda_sync(device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))
        baseline_allocated = torch.cuda.memory_allocated(torch.device(device))
        baseline_reserved = torch.cuda.memory_reserved(torch.device(device))
    else:
        baseline_allocated = baseline_reserved = 0
    started = time.perf_counter()
    flipped = horizontal_flip_image(image)
    try:
        candidates = {
            "original": prepare_canonical_candidate(
                processor=processor,
                prompt=prompt,
                candidate_name="original",
                candidate_text=str(record["answer_letter"]),
                image=flipped,
                device=device,
            ),
            "mapped": prepare_canonical_candidate(
                processor=processor,
                prompt=prompt,
                candidate_name="mapped",
                candidate_text=str(record["mapped_answer_letter"]),
                image=flipped,
                device=device,
            ),
        }
        scores = {
            name: score_canonical_candidate(model, candidate)
            for name, candidate in candidates.items()
        }
    finally:
        flipped.close()
    _cuda_sync(device)
    elapsed = time.perf_counter() - started
    if device.startswith("cuda"):
        peak_allocated = torch.cuda.max_memory_allocated(torch.device(device))
        peak_reserved = torch.cuda.max_memory_reserved(torch.device(device))
    else:
        peak_allocated = peak_reserved = 0

    active_counts = {
        name: int(
            (
                (candidate.input_ids == model.config.image_token_id)
                & candidate.attention_mask.to(torch.bool)
            ).sum().item()
        )
        for name, candidate in candidates.items()
    }
    if len(set(active_counts.values())) != 1 or next(iter(active_counts.values())) < 1:
        raise ValueError(
            f"global hflip candidate visual-token counts disagree: {active_counts}"
        )
    active_count = next(iter(active_counts.values()))
    return {
        "accepted": True,
        "reject_reason": "accepted",
        "positive_control": True,
        "intervention": "canonical_global_horizontal_image_flip",
        "canonical_raw_multimodal_forward": True,
        "custom_position_ids_supplied": False,
        "past_key_values_supplied": False,
        "cache_position_supplied": False,
        "use_cache": False,
        "candidate_construction": "separate full sequence tokenization and embeddings",
        "candidate_token_ids": {
            name: candidate.token_ids for name, candidate in candidates.items()
        },
        "active_visual_token_count": active_count,
        "touched_visual_token_count": active_count,
        "touched_visual_token_ratio": 1.0,
        "total_displacement": None,
        "attempted_touched_visual_token_count": active_count,
        "attempted_touched_visual_token_ratio": 1.0,
        "attempted_total_displacement": None,
        "s_orig": scores["original"],
        "s_mapped": scores["mapped"],
        "directional_gap": scores["mapped"] - scores["original"],
        "wall_seconds": elapsed,
        "gpu_baseline_allocated_bytes": baseline_allocated,
        "gpu_baseline_reserved_bytes": baseline_reserved,
        "gpu_peak_allocated_bytes": peak_allocated,
        "gpu_peak_reserved_bytes": peak_reserved,
        "gpu_peak_incremental_allocated_bytes": max(
            0, peak_allocated - baseline_allocated
        ),
        "gpu_peak_incremental_reserved_bytes": max(0, peak_reserved - baseline_reserved),
    }


def run_noop_candidate(
    model: Qwen2_5_VLForConditionalGeneration,
    candidate: CandidateInputs,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Compare canonical raw multimodal forward with the explicit no-op path."""

    with torch.inference_mode():
        canonical = model(
            input_ids=candidate.input_ids,
            attention_mask=candidate.attention_mask,
            pixel_values=candidate.pixel_values,
            image_grid_thw=candidate.image_grid_thw,
            use_cache=False,
            return_dict=True,
        ).logits
        contract = build_teacher_forced_intervention_inputs(
            candidate.inputs_embeds,
            candidate.position_ids,
            candidate.attention_mask,
            candidate.labels,
            candidate.candidate_mask,
        )
        manual = model(**contract.as_model_kwargs()).logits
        comparison = noop_comparison(
            canonical,
            manual,
            candidate.labels,
            candidate.candidate_mask,
            threshold=threshold,
        )
    del canonical, manual
    return comparison


def apply_condition(
    condition: str,
    candidate: CandidateInputs,
    *,
    max_modified_ratio: float,
    matched_displacement: Tensor | None = None,
) -> OperatorResult | None:
    if condition == "factual":
        return None
    common = {
        "active_visual_token_mask": candidate.active_visual_mask,
        "touched_token_ratio_budget": max_modified_ratio,
    }
    if condition == "position_slot":
        return position_slot_swap(
            candidate.inputs_embeds,
            candidate.position_ids,
            candidate.mask_a,
            candidate.mask_b,
            axis=2,
            coordinate_bounds=active_coordinate_bounds(
                candidate.position_ids, candidate.active_visual_mask, axis=2
            ),
            **common,
        )
    if condition == "content_slot":
        return content_slot_swap(
            candidate.inputs_embeds,
            candidate.position_ids,
            candidate.mask_a,
            candidate.mask_b,
            **common,
        )
    if condition == "resampling_null":
        return resampling_null(
            candidate.inputs_embeds,
            candidate.position_ids,
            candidate.mask_a,
            candidate.mask_b,
            bridge_token_counts=True,
            **common,
        )
    if condition == "context_transport_null":
        if matched_displacement is None:
            raise ValueError("context transport requires position-slot matched displacement")
        return context_transport(
            candidate.inputs_embeds,
            candidate.position_ids,
            candidate.mask_a,
            candidate.mask_b,
            relation_axis=2,
            transport_axis=1,
            target_displacement=matched_displacement,
            coordinate_bounds=active_coordinate_bounds(
                candidate.position_ids, candidate.active_visual_mask, axis=1
            ),
            **common,
        )
    raise ValueError(f"unknown diagnostic condition {condition!r}")


def _score_operator_output(
    model: Qwen2_5_VLForConditionalGeneration,
    candidate: CandidateInputs,
    result: OperatorResult | None,
) -> float:
    embeddings = candidate.inputs_embeds if result is None else result.embeddings
    positions = candidate.position_ids if result is None else result.position_ids
    contract = build_teacher_forced_intervention_inputs(
        embeddings,
        positions,
        candidate.attention_mask,
        candidate.labels,
        candidate.candidate_mask,
    )
    with torch.inference_mode():
        logits = model(**contract.as_model_kwargs()).logits
        score = candidate_span_log_probs(
            logits, candidate.labels, candidate.candidate_mask
        )[0].item()
    del logits
    return float(score)


def _cuda_sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def run_condition(
    *,
    model: Qwen2_5_VLForConditionalGeneration,
    candidates: Mapping[str, CandidateInputs],
    condition: str,
    max_modified_ratio: float,
    device: str,
) -> dict[str, Any]:
    """Apply and score one branch serially for original and mapped candidates."""

    _cuda_sync(device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))
        baseline_allocated = torch.cuda.memory_allocated(torch.device(device))
        baseline_reserved = torch.cuda.memory_reserved(torch.device(device))
    else:
        baseline_allocated = baseline_reserved = 0
    started = time.perf_counter()

    results: dict[str, OperatorResult | None] = {}
    scores: dict[str, float] = {}
    coordinate_bounds: dict[str, list[list[float]]] = {}
    matched_targets: dict[str, float] = {}
    for name in ("original", "mapped"):
        candidate = candidates[name]
        matched: Tensor | None = None
        if condition == "position_slot":
            coordinate_bounds[name] = active_coordinate_bounds(
                candidate.position_ids, candidate.active_visual_mask, axis=2
            ).detach().cpu().tolist()
        if condition == "context_transport_null":
            position_reference = apply_condition(
                "position_slot",
                candidate,
                max_modified_ratio=max_modified_ratio,
            )
            assert position_reference is not None
            matched = position_reference.metadata["attempted_total_displacement"]
            matched_targets[name] = float(matched[0].item())
            coordinate_bounds[name] = active_coordinate_bounds(
                candidate.position_ids, candidate.active_visual_mask, axis=1
            ).detach().cpu().tolist()
        result = apply_condition(
            condition,
            candidate,
            max_modified_ratio=max_modified_ratio,
            matched_displacement=matched,
        )
        results[name] = result
        scores[name] = _score_operator_output(model, candidate, result)

    _cuda_sync(device)
    elapsed = time.perf_counter() - started
    if device.startswith("cuda"):
        peak_allocated = torch.cuda.max_memory_allocated(torch.device(device))
        peak_reserved = torch.cuda.max_memory_reserved(torch.device(device))
    else:
        peak_allocated = peak_reserved = 0

    if condition == "factual":
        active_count = int(candidates["original"].active_visual_mask.sum().item())
        audit: dict[str, Any] = {
            "accepted": True,
            "reject_reason": "factual_no_intervention",
            "active_visual_token_count": active_count,
            "touched_visual_token_count": 0,
            "touched_visual_token_ratio": 0.0,
            "total_displacement": 0.0,
            "attempted_touched_visual_token_count": 0,
            "attempted_touched_visual_token_ratio": 0.0,
            "attempted_total_displacement": 0.0,
        }
        per_candidate_audit = {name: audit for name in candidates}
    else:
        per_candidate_audit = {
            name: operator_audit(result)
            for name, result in results.items()
            if result is not None
        }
        audit = per_candidate_audit["original"]

    record = {
        **audit,
        "candidate_operator_audits": per_candidate_audit,
        "s_orig": scores["original"],
        "s_mapped": scores["mapped"],
        "directional_gap": scores["mapped"] - scores["original"],
        "wall_seconds": elapsed,
        "gpu_baseline_allocated_bytes": baseline_allocated,
        "gpu_baseline_reserved_bytes": baseline_reserved,
        "gpu_peak_allocated_bytes": peak_allocated,
        "gpu_peak_reserved_bytes": peak_reserved,
        "gpu_peak_incremental_allocated_bytes": max(
            0, peak_allocated - baseline_allocated
        ),
        "gpu_peak_incremental_reserved_bytes": max(0, peak_reserved - baseline_reserved),
        "rejected_rows_are_scored_after_transactional_factual_rollback": True,
    }
    if condition == "position_slot":
        record["coordinate_axis"] = 2
        record["factual_active_visual_coordinate_bounds"] = coordinate_bounds
    if condition == "context_transport_null":
        record["coordinate_axis"] = 1
        record["factual_active_visual_coordinate_bounds"] = coordinate_bounds
        record["matched_position_slot_target_displacement"] = matched_targets
    return record


def run_sample(
    *,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    record: Mapping[str, Any],
    images_root: Path,
    device: str,
    max_modified_ratio: float,
    noop_threshold: float,
) -> dict[str, Any]:
    image_path = images_root / str(record["image"])
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    prompt_text = diagnostic_prompt(record)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        candidates = {
            "original": prepare_candidate(
                model=model,
                processor=processor,
                prompt=prompt,
                candidate_name="original",
                candidate_text=str(record["answer_letter"]),
                image=image,
                record=record,
                device=device,
            ),
            "mapped": prepare_candidate(
                model=model,
                processor=processor,
                prompt=prompt,
                candidate_name="mapped",
                candidate_text=str(record["mapped_answer_letter"]),
                image=image,
                record=record,
                device=device,
            ),
        }
        global_hflip = run_global_hflip(
            model=model,
            processor=processor,
            prompt=prompt,
            record=record,
            image=image,
            device=device,
        )
        image_size_hw = [image.height, image.width]

    noop_started = time.perf_counter()
    noop = {
        name: run_noop_candidate(model, candidate, threshold=noop_threshold)
        for name, candidate in candidates.items()
    }
    _cuda_sync(device)
    noop_wall = time.perf_counter() - noop_started
    noop_max_error = max(
        value["max_abs_predictor_logit_error"] for value in noop.values()
    )
    if noop_max_error >= noop_threshold:
        raise RuntimeError(
            f"sample {record['sample_id']} no-op predictor logit error "
            f"{noop_max_error} is not below {noop_threshold}"
        )

    latent_conditions = {
        condition: run_condition(
            model=model,
            candidates=candidates,
            condition=condition,
            max_modified_ratio=max_modified_ratio,
            device=device,
        )
        for condition in LATENT_CONDITIONS
    }
    conditions = {
        condition: (
            global_hflip
            if condition == "global_hflip"
            else latent_conditions[condition]
        )
        for condition in CONDITIONS
    }
    original = candidates["original"]
    temporal, grid_h, grid_w = (
        int(value) for value in original.image_grid_thw[0].tolist()
    )
    merge = int(model.config.vision_config.spatial_merge_size)
    return {
        "schema_version": 1,
        "diagnostic": "frozen_qwen_operator_phase1a",
        "sample_id": record["sample_id"],
        "scene_id": record["scene_id"],
        "image": record["image"],
        "image_size_hw": image_size_hw,
        "image_grid_thw": [temporal, grid_h, grid_w],
        "llm_visual_grid_thw": [temporal, grid_h // merge, grid_w // merge],
        "prompt_policy": PROMPT_POLICY,
        "candidate_construction": "separate full sequence tokenization and embeddings",
        "candidates": {
            name: {
                "text": candidate.text,
                "token_ids": candidate.token_ids,
                "token_count": len(candidate.token_ids),
            }
            for name, candidate in candidates.items()
        },
        "noop": {
            "threshold": noop_threshold,
            "max_abs_predictor_logit_error": noop_max_error,
            "passed": all(value["passed"] for value in noop.values()),
            "wall_seconds": noop_wall,
            "candidates": noop,
        },
        "conditions": conditions,
    }


def descriptive_operator_factual_effect(
    rows: list[dict[str, Any]], operator: str
) -> dict[str, Any]:
    """Describe accepted ``operator - factual`` directional-gap differences."""

    valid = [row for row in rows if row["conditions"][operator]["accepted"] is True]
    differences = np.asarray(
        [
            row["conditions"][operator]["directional_gap"]
            - row["conditions"]["factual"]["directional_gap"]
            for row in valid
        ],
        dtype=np.float32,
    )
    result: dict[str, Any] = {
        "comparison": f"{operator}-factual",
        "accepted_pairs_only": True,
        "total_samples": len(rows),
        "valid_pairs": len(valid),
        "valid_scenes": len({row["scene_id"] for row in valid}),
        "excluded_operator_rejects": len(rows) - len(valid),
    }
    if differences.size == 0:
        result.update(
            {
                "status": "NOT_ESTIMABLE",
                "reason": "no accepted operator rows",
                "mean_directional_gap_difference": None,
            }
        )
    else:
        result.update(
            {
                "status": "DESCRIPTIVE_ONLY",
                "mean_directional_gap_difference": float(
                    differences.mean(dtype=np.float32)
                ),
                "min_directional_gap_difference": float(differences.min()),
                "max_directional_gap_difference": float(differences.max()),
            }
        )
    return result


def paired_control_effect(
    rows: list[dict[str, Any]],
    *,
    treatment: str,
    control: str,
    num_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap accepted treatment/control gap differences by scene.

    Rejected rows are excluded before any values are read.  In particular, a
    transactional factual rollback from a rejected null is never treated as a
    valid null observation.
    """

    treatment_rejects = [
        row for row in rows if row["conditions"][treatment]["accepted"] is not True
    ]
    control_rejects = [
        row for row in rows if row["conditions"][control]["accepted"] is not True
    ]
    valid = [
        row
        for row in rows
        if row["conditions"][treatment]["accepted"] is True
        and row["conditions"][control]["accepted"] is True
    ]
    scenes = [str(row["scene_id"]) for row in valid]
    unique_scenes = set(scenes)
    result: dict[str, Any] = {
        "comparison": f"{treatment}-{control}",
        "estimand": "scene-mean paired directional-gap difference",
        "accepted_treatment_and_control_only": True,
        "rejected_transactional_rollbacks_excluded": True,
        "total_samples": len(rows),
        "valid_pairs": len(valid),
        "valid_scenes": len(unique_scenes),
        "excluded_treatment_rejects": len(treatment_rejects),
        "excluded_control_rejects": len(control_rejects),
        "excluded_union_rejects": len(rows) - len(valid),
        "num_resamples": num_resamples,
        "seed": seed,
    }
    if len(unique_scenes) < 2:
        result.update(
            {
                "status": "NOT_ESTIMABLE",
                "reason": "scene-level paired bootstrap requires at least two valid scenes",
                "estimate": None,
                "ci_low": None,
                "ci_high": None,
                "excludes_zero": None,
            }
        )
        return result

    treatment_values = np.asarray(
        [row["conditions"][treatment]["directional_gap"] for row in valid],
        dtype=np.float32,
    )
    control_values = np.asarray(
        [row["conditions"][control]["directional_gap"] for row in valid],
        dtype=np.float32,
    )
    bootstrap = scene_level_paired_bootstrap(
        treatment_values,
        control_values,
        scenes,
        num_resamples=num_resamples,
        seed=seed,
    )
    result.update({"status": "ESTIMATED", **bootstrap.to_dict()})
    return result


def make_summary(
    rows: list[dict[str, Any]],
    *,
    model_path: Path,
    manifest_path: Path,
    output_path: Path,
    device: str,
    attention: str,
    min_pixels: int,
    max_pixels: int,
    max_modified_ratio: float,
    noop_threshold: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    config_compatibility: Mapping[str, Any],
    model_load_audit: Mapping[str, Any],
    run_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize zero completed diagnostic rows")
    condition_summary: dict[str, Any] = {}
    for condition in CONDITIONS:
        values = [row["conditions"][condition] for row in rows]
        accepted_values = [value for value in values if value["accepted"] is True]
        accepted_count = len(accepted_values)
        condition_summary[condition] = {
            "accepted_samples": accepted_count,
            "rejected_samples": len(values) - accepted_count,
            "acceptance_rate": accepted_count / len(values),
            "reject_reason_counts": {
                reason: sum(value["reject_reason"] == reason for value in values)
                for reason in sorted({value["reject_reason"] for value in values})
            },
            "mean_directional_gap_all_rows_transactional_output": sum(
                value["directional_gap"] for value in values
            )
            / len(values),
            "mean_directional_gap_accepted_only": (
                sum(value["directional_gap"] for value in accepted_values)
                / accepted_count
                if accepted_count
                else None
            ),
            "mean_wall_seconds": sum(value["wall_seconds"] for value in values)
            / len(values),
            "max_gpu_peak_allocated_bytes": max(
                value["gpu_peak_allocated_bytes"] for value in values
            ),
            "max_gpu_peak_incremental_allocated_bytes": max(
                value["gpu_peak_incremental_allocated_bytes"] for value in values
            ),
        }
    paired_effects = {
        "content_minus_resampling": paired_control_effect(
            rows,
            treatment="content_slot",
            control="resampling_null",
            num_resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        ),
        "position_minus_context": paired_control_effect(
            rows,
            treatment="position_slot",
            control="context_transport_null",
            num_resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        ),
    }
    descriptive_effects = {
        condition: descriptive_operator_factual_effect(rows, condition)
        for condition in CONDITIONS
        if condition != "factual"
    }
    return {
        "schema_version": 1,
        "diagnostic": "frozen_qwen_operator_phase1a",
        "engineering_smoke_not_a_statistical_gate": True,
        "model": str(model_path),
        "manifest": str(manifest_path),
        "output_jsonl": str(output_path),
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "attention": attention,
        "batch_size": 1,
        "prompt_policy": PROMPT_POLICY,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "max_modified_visual_token_ratio": max_modified_ratio,
        "noop_threshold_strict_less_than": noop_threshold,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "config_compatibility": dict(config_compatibility),
        "model_load_audit": dict(model_load_audit),
        "run_contract": dict(run_contract),
        "completed_samples": len(rows),
        "completed_scene_count": len({row["scene_id"] for row in rows}),
        "all_noop_passed": all(row["noop"]["passed"] for row in rows),
        "max_noop_predictor_logit_error": max(
            row["noop"]["max_abs_predictor_logit_error"] for row in rows
        ),
        "mean_noop_wall_seconds": sum(row["noop"]["wall_seconds"] for row in rows)
        / len(rows),
        "mean_recorded_noop_plus_condition_wall_seconds_per_sample": sum(
            row["noop"]["wall_seconds"]
            + sum(value["wall_seconds"] for value in row["conditions"].values())
            for row in rows
        )
        / len(rows),
        "max_gpu_peak_allocated_bytes_across_conditions": max(
            value["gpu_peak_allocated_bytes"]
            for row in rows
            for value in row["conditions"].values()
        ),
        "max_gpu_peak_incremental_allocated_bytes_across_conditions": max(
            value["gpu_peak_incremental_allocated_bytes"]
            for row in rows
            for value in row["conditions"].values()
        ),
        "condition_summary": condition_summary,
        "paired_control_effects": paired_effects,
        "accepted_operator_minus_factual_descriptive_effects": descriptive_effects,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--model-manifest", type=Path, default=Path("manifests/model_manifest.json")
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--images-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--attention", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument("--min-pixels", type=int, default=16 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=128 * 28 * 28)
    parser.add_argument("--max-modified-ratio", type=float, default=0.35)
    parser.add_argument("--noop-threshold", type=float, default=1e-5)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.min_pixels < 1 or args.max_pixels < args.min_pixels:
        parser.error("pixel bounds must be positive and ordered")
    if not 0.0 <= args.max_modified_ratio <= 1.0:
        parser.error("--max-modified-ratio must lie in [0,1]")
    if args.noop_threshold <= 0:
        parser.error("--noop-threshold must be positive")
    if args.bootstrap_resamples < 2:
        parser.error("--bootstrap-resamples must be at least two")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error(f"CUDA device requested but CUDA is unavailable: {args.device}")

    selected = [
        RelationPairV2.from_dict(row).to_dict()
        for row in list(read_jsonl(args.manifest))[: args.limit]
    ]
    if not selected:
        parser.error("manifest selection is empty")
    code_hashes = frozen_code_hashes()
    run_contract = ensure_run_contract(
        args.output.parent / f"{args.output.stem}.run_config.json",
        {
            "schema_version": 3,
            "diagnostic": "frozen_qwen_operator_phase1a",
            "prompt_policy": PROMPT_POLICY,
            "conditions": list(CONDITIONS),
            "global_hflip": {
                "enabled": True,
                "role": "positive_control",
                "transform": "PIL.Image.Transpose.FLIP_LEFT_RIGHT",
                "candidate_construction": "separate full sequence tokenization",
                "forward": "canonical_input_ids_plus_pixel_values_auto_mrope",
                "custom_position_ids_supplied": False,
                "past_key_values_supplied": False,
                "cache_position_supplied": False,
                "use_cache": False,
                "implementation_file": "scripts/frozen_operator_diagnostic.py",
                "implementation_sha256": code_hashes[
                    "scripts/frozen_operator_diagnostic.py"
                ],
            },
            "model_path": str(args.model),
            "model_config_sha256": sha256_file(args.model / "config.json"),
            "model_manifest_path": str(args.model_manifest),
            "model_manifest_sha256": sha256_file(args.model_manifest),
            "manifest_path": str(args.manifest),
            "manifest_sha256": sha256_file(args.manifest),
            "ordered_sample_ids": [row["sample_id"] for row in selected],
            "ordered_scene_ids": [row["scene_id"] for row in selected],
            "limit": args.limit,
            "device": args.device,
            "dtype": "bfloat16" if args.device.startswith("cuda") else "float32",
            "attention": args.attention,
            "batch_size": 1,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "max_modified_visual_token_ratio": args.max_modified_ratio,
            "noop_threshold": args.noop_threshold,
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.bootstrap_seed,
            "teacher_forced": True,
            "use_cache": False,
            "resume_policy": "deterministic sample-keyed rows under exact contract",
            "torch_version": torch.__version__,
            "code_sha256": code_hashes,
        },
        existing_artifacts=(args.output, args.summary),
    )
    completed = load_completed(args.output)
    validate_completed_rows(completed, selected)
    config_result = load_qwen_config_with_compat(args.model, local_files_only=True)
    config_compatibility = config_result.to_record()
    print(
        "config_compatibility="
        + json.dumps(config_compatibility, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.image_processor.min_pixels = args.min_pixels
    processor.image_processor.max_pixels = args.max_pixels
    model_dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        config=config_result.config,
        local_files_only=True,
        torch_dtype=model_dtype,
        attn_implementation=args.attention,
        low_cpu_mem_usage=True,
        device_map={"": args.device},
    ).eval()
    model_load_audit = validate_qwen_weight_tying(model)
    print(
        "model_load_audit="
        + json.dumps(model_load_audit, ensure_ascii=False, sort_keys=True),
        flush=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as sink:
        for index, record in enumerate(selected, start=1):
            sample_id = record.get("sample_id")
            if sample_id in completed:
                print(f"sample={sample_id} already complete", flush=True)
                continue
            row = run_sample(
                model=model,
                processor=processor,
                record=record,
                images_root=args.images_root,
                device=args.device,
                max_modified_ratio=args.max_modified_ratio,
                noop_threshold=args.noop_threshold,
            )
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            completed[str(sample_id)] = row
            print(
                f"sample={sample_id} completed={index}/{len(selected)} "
                f"noop_error={row['noop']['max_abs_predictor_logit_error']}",
                flush=True,
            )

    selected_ids = {str(row["sample_id"]) for row in selected}
    summary_rows = [
        completed[str(record["sample_id"])]
        for record in selected
        if str(record["sample_id"]) in completed
        and str(record["sample_id"]) in selected_ids
    ]
    summary = make_summary(
        summary_rows,
        model_path=args.model,
        manifest_path=args.manifest,
        output_path=args.output,
        device=args.device,
        attention=args.attention,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_modified_ratio=args.max_modified_ratio,
        noop_threshold=args.noop_threshold,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        config_compatibility=config_compatibility,
        model_load_audit=model_load_audit,
        run_contract=run_contract,
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
