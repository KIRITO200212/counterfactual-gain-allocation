"""On-policy answer-prefix credit for the Plan9 PairAug objective.

The credit is read at the single causal logit position immediately before
the generated answer candidate token.  Consequently it is conditioned on
the generated visual reasoning trajectory and the answer-tag prefix, but not
on the sampled answer itself.  The mapped-vs-original logit margin is turned
into a bounded continuous reward with ``sigmoid(margin / temperature)`` and
group-standardized exactly like the vendored GRPO trainer (unbiased standard
deviation and ``1e-4`` denominator epsilon).

This module contains no vendored-trainer imports, so the token-boundary and
reward math remain CPU-unit-testable outside the training environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


OPTION_LETTERS = ("A", "B", "C", "D")
GRPO_REWARD_STD_EPSILON = 1e-4


@dataclass(frozen=True)
class AnswerTokenStyle:
    """One tokenizer spelling of an answer tag and its four option tokens.

    ``prefix_ids`` are the completion tokens immediately preceding the
    option-bearing token.  ``candidate_token_ids`` are ordered A/B/C/D.
    Qwen's no-space spelling, for example, normally has a fused ``">A"``
    token; the prefix therefore ends at ``"answer"`` rather than assuming
    that ``">"`` is a standalone token.
    """

    name: str
    prefix_ids: tuple[int, ...]
    candidate_token_ids: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("answer token style requires a non-empty name")
        if not self.prefix_ids:
            raise ValueError(f"answer token style {self.name!r} has an empty prefix")
        if len(self.candidate_token_ids) != len(OPTION_LETTERS):
            raise ValueError("candidate_token_ids must contain A/B/C/D")
        if len(set(self.candidate_token_ids)) != len(OPTION_LETTERS):
            raise ValueError("A/B/C/D candidate token ids must be distinct")
        values = (*self.prefix_ids, *self.candidate_token_ids)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise ValueError("answer token ids must be non-negative integers")


@dataclass(frozen=True)
class AnswerCandidateLocations:
    """Per-row completion-relative option-token locations.

    Invalid or ambiguous rows are fail-closed: ``valid`` is false and their
    position/style placeholders must not contribute credit.
    """

    positions: Tensor
    style_indices: Tensor
    valid: Tensor


def _tokenize_without_special_tokens(tokenizer: Any, text: str) -> tuple[int, ...]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if isinstance(ids, Tensor):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        if len(ids) != 1:
            raise ValueError("tokenizer unexpectedly returned multiple rows")
        ids = ids[0]
    try:
        result = tuple(int(value) for value in ids)
    except (TypeError, ValueError):
        raise ValueError(f"tokenizer returned invalid ids for {text!r}") from None
    if not result:
        raise ValueError(f"tokenizer returned no ids for {text!r}")
    return result


def _derive_style(tokenizer: Any, *, name: str, template: str) -> AnswerTokenStyle:
    encodings = [
        _tokenize_without_special_tokens(tokenizer, template.format(letter=letter))
        for letter in OPTION_LETTERS
    ]
    lengths = {len(ids) for ids in encodings}
    if len(lengths) != 1:
        raise ValueError(
            f"answer style {name!r} changes token length across A/B/C/D: "
            f"{[len(ids) for ids in encodings]}"
        )
    differing = [
        index
        for index in range(len(encodings[0]))
        if len({ids[index] for ids in encodings}) > 1
    ]
    if len(differing) != 1:
        raise ValueError(
            f"answer style {name!r} must have exactly one option-bearing token; "
            f"found differing positions {differing}"
        )
    candidate_index = differing[0]
    if candidate_index == 0:
        raise ValueError(f"answer style {name!r} has no causal answer-tag prefix")
    prefix = encodings[0][:candidate_index]
    if any(ids[:candidate_index] != prefix for ids in encodings[1:]):
        raise ValueError(f"answer style {name!r} has inconsistent prefixes")
    return AnswerTokenStyle(
        name=name,
        prefix_ids=prefix,
        candidate_token_ids=tuple(ids[candidate_index] for ids in encodings),
    )


def build_answer_token_styles(tokenizer: Any) -> tuple[AnswerTokenStyle, ...]:
    """Resolve the two accepted answer spellings from the active tokenizer.

    Full closing tags are included while deriving the unique candidate token
    so byte-pair fusion is observed under the same right context as a valid
    completion.  Location matching uses only the causal prefix and candidate
    token; no future/suffix token enters the credit.
    """

    styles = (
        _derive_style(
            tokenizer,
            name="compact",
            template="<answer>{letter}</answer>",
        ),
        _derive_style(
            tokenizer,
            name="spaced",
            template="<answer> {letter} </answer>",
        ),
    )
    if len({(style.prefix_ids, style.candidate_token_ids) for style in styles}) != len(styles):
        raise ValueError("compact and spaced answer styles collapsed to the same token pattern")
    return styles


def locate_answer_candidate_tokens(
    completion_ids: Tensor,
    completion_mask: Tensor,
    styles: Sequence[AnswerTokenStyle],
) -> AnswerCandidateLocations:
    """Find exactly one tagged A/B/C/D candidate token in each completion.

    The search is restricted to tokens enabled by ``completion_mask``.
    Missing and multiply matched answers are invalid instead of being guessed;
    they receive zero prefix credit at the reward layer.
    """

    if not isinstance(completion_ids, Tensor) or completion_ids.ndim != 2:
        raise ValueError("completion_ids must have shape [B,C]")
    if not isinstance(completion_mask, Tensor) or completion_mask.shape != completion_ids.shape:
        raise ValueError("completion_mask must share completion_ids shape")
    if not styles:
        raise ValueError("at least one answer token style is required")
    for style in styles:
        if not isinstance(style, AnswerTokenStyle):
            raise TypeError("styles must contain AnswerTokenStyle instances")
    if not bool(((completion_mask == 0) | (completion_mask == 1)).all()):
        raise ValueError("completion_mask must be binary")

    batch_size = completion_ids.shape[0]
    positions = torch.zeros(batch_size, dtype=torch.long, device=completion_ids.device)
    style_indices = torch.zeros(batch_size, dtype=torch.long, device=completion_ids.device)
    valid = torch.zeros(batch_size, dtype=torch.bool, device=completion_ids.device)

    # The batch is intentionally tiny (G=4).  Converting each active row to a
    # Python list makes the fail-closed multiple-match logic explicit and is
    # negligible next to VLM generation/forward cost.
    for row_index in range(batch_size):
        active_length = int(completion_mask[row_index].sum().item())
        row = [int(value) for value in completion_ids[row_index, :active_length].tolist()]
        matches: list[tuple[int, int]] = []
        for style_index, style in enumerate(styles):
            prefix = list(style.prefix_ids)
            candidate_set = set(style.candidate_token_ids)
            prefix_length = len(prefix)
            for position in range(prefix_length, active_length):
                if row[position] not in candidate_set:
                    continue
                if row[position - prefix_length : position] == prefix:
                    matches.append((position, style_index))
        if len(matches) == 1:
            position, style_index = matches[0]
            positions[row_index] = position
            style_indices[row_index] = style_index
            valid[row_index] = True
    return AnswerCandidateLocations(
        positions=positions,
        style_indices=style_indices,
        valid=valid,
    )


def answer_prefix_margins(
    logits: Tensor,
    *,
    prompt_length: int,
    locations: AnswerCandidateLocations,
    original_indices: Tensor,
    mapped_indices: Tensor,
    styles: Sequence[AnswerTokenStyle],
) -> Tensor:
    """Return mapped-minus-original logits at the pre-answer position.

    For completion position ``p``, the only logit row read is
    ``prompt_length + p - 1``.  This is the causal prediction *before* the
    option-bearing token at ``prompt_length + p`` and is the mechanical
    no-answer-leak contract.
    """

    option_logits = answer_prefix_option_logits(
        logits,
        prompt_length=prompt_length,
        locations=locations,
        styles=styles,
    )
    batch_size = option_logits.shape[0]
    for name, tensor in (
        ("original_indices", original_indices),
        ("mapped_indices", mapped_indices),
    ):
        if not isinstance(tensor, Tensor) or tensor.ndim != 1 or tensor.shape[0] != batch_size:
            raise ValueError(f"{name} must have shape [{batch_size}]")
    if bool(((original_indices < 0) | (original_indices >= len(OPTION_LETTERS))).any()):
        raise ValueError("original_indices must lie in [0,3]")
    if bool(((mapped_indices < 0) | (mapped_indices >= len(OPTION_LETTERS))).any()):
        raise ValueError("mapped_indices must lie in [0,3]")
    if bool((original_indices == mapped_indices).any()):
        raise ValueError("original and mapped option indices must differ")
    row_indices = torch.arange(batch_size, device=option_logits.device)
    original_indices = original_indices.to(device=logits.device, dtype=torch.long)
    mapped_indices = mapped_indices.to(device=logits.device, dtype=torch.long)
    valid = locations.valid.to(device=logits.device)
    mapped_logits = option_logits[row_indices, mapped_indices]
    original_logits = option_logits[row_indices, original_indices]
    raw = mapped_logits - original_logits
    return torch.where(valid, raw, torch.zeros_like(raw))


def answer_prefix_option_logits(
    logits: Tensor,
    *,
    prompt_length: int,
    locations: AnswerCandidateLocations,
    styles: Sequence[AnswerTokenStyle],
) -> Tensor:
    """Return the causal A-D option logits immediately before the answer.

    The result has shape ``[B,4]`` in A/B/C/D order.  Invalid or ambiguous
    completions are fail-closed to exact zeros; callers must still use
    ``locations.valid`` before constructing a loss.
    """

    if not isinstance(logits, Tensor) or logits.ndim != 3:
        raise ValueError("logits must have shape [B,L,V]")
    batch_size, sequence_length, vocab_size = logits.shape
    if isinstance(prompt_length, bool) or not isinstance(prompt_length, int) or prompt_length < 1:
        raise ValueError("prompt_length must be a positive integer")
    for name, tensor in (
        ("positions", locations.positions),
        ("style_indices", locations.style_indices),
        ("valid", locations.valid),
    ):
        if not isinstance(tensor, Tensor) or tensor.ndim != 1 or tensor.shape[0] != batch_size:
            raise ValueError(f"{name} must have shape [{batch_size}]")
    if locations.valid.dtype != torch.bool:
        raise ValueError("locations.valid must be boolean")
    if not styles:
        raise ValueError("at least one answer token style is required")
    if bool(((locations.style_indices < 0) | (locations.style_indices >= len(styles))).any()):
        raise ValueError("style index outside the resolved style table")

    candidate_table = torch.tensor(
        [style.candidate_token_ids for style in styles],
        dtype=torch.long,
        device=logits.device,
    )
    if bool((candidate_table >= vocab_size).any()):
        raise ValueError("answer candidate token id exceeds model vocabulary")
    style_indices = locations.style_indices.to(device=logits.device, dtype=torch.long)
    positions = locations.positions.to(device=logits.device, dtype=torch.long)
    valid = locations.valid.to(device=logits.device)
    prediction_indices = prompt_length + positions - 1
    if bool((valid & ((prediction_indices < 0) | (prediction_indices >= sequence_length))).any()):
        raise ValueError("valid answer position has no in-range causal prediction logit")
    prediction_indices = prediction_indices.clamp(min=0, max=sequence_length - 1)
    row_indices = torch.arange(batch_size, device=logits.device)
    option_token_ids = candidate_table[style_indices]
    selected = logits[row_indices, prediction_indices].gather(1, option_token_ids).float()
    return torch.where(valid.unsqueeze(1), selected, torch.zeros_like(selected))


def answer_prefix_reasoning_mask(
    completion_mask: Tensor,
    locations: AnswerCandidateLocations,
    styles: Sequence[AnswerTokenStyle],
) -> Tensor:
    """Mask only generated tokens strictly before the ``<answer>`` tag.

    The answer tag, sampled option, closing tag, and EOS/padding therefore
    receive no prefix-credit policy gradient.  Invalid or ambiguous answer
    locations produce an all-zero row instead of guessing a boundary.
    """

    if not isinstance(completion_mask, Tensor) or completion_mask.ndim != 2:
        raise ValueError("completion_mask must have shape [B,C]")
    batch_size, completion_length = completion_mask.shape
    if not bool(((completion_mask == 0) | (completion_mask == 1)).all()):
        raise ValueError("completion_mask must be binary")
    for name, tensor in (
        ("positions", locations.positions),
        ("style_indices", locations.style_indices),
        ("valid", locations.valid),
    ):
        if not isinstance(tensor, Tensor) or tensor.ndim != 1 or tensor.shape[0] != batch_size:
            raise ValueError(f"{name} must have shape [{batch_size}]")
    if locations.valid.dtype != torch.bool:
        raise ValueError("locations.valid must be boolean")
    if not styles:
        raise ValueError("at least one answer token style is required")
    if bool(((locations.style_indices < 0) | (locations.style_indices >= len(styles))).any()):
        raise ValueError("style index outside the resolved style table")

    prefix_lengths = torch.tensor(
        [len(style.prefix_ids) for style in styles],
        dtype=torch.long,
        device=completion_mask.device,
    )
    style_indices = locations.style_indices.to(device=completion_mask.device, dtype=torch.long)
    positions = locations.positions.to(device=completion_mask.device, dtype=torch.long)
    valid = locations.valid.to(device=completion_mask.device)
    answer_starts = positions - prefix_lengths[style_indices]
    if bool((valid & ((answer_starts < 0) | (answer_starts >= completion_length))).any()):
        raise ValueError("valid answer style has an out-of-range tag start")
    columns = torch.arange(completion_length, device=completion_mask.device).unsqueeze(0)
    reasoning = columns < answer_starts.clamp(min=0).unsqueeze(1)
    return reasoning & completion_mask.bool() & valid.unsqueeze(1)


def bounded_prefix_credit(margins: Tensor, valid: Tensor, *, temperature: float) -> Tensor:
    """Map finite margins to ``[0,1]`` credit; invalid rows get exact zero."""

    if not isinstance(margins, Tensor) or margins.ndim != 1:
        raise ValueError("margins must be 1-D")
    if not isinstance(valid, Tensor) or valid.shape != margins.shape or valid.dtype != torch.bool:
        raise ValueError("valid must be a boolean tensor matching margins")
    if not bool(torch.isfinite(margins).all()):
        raise FloatingPointError("prefix margins contain NaN or Inf")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be numeric")
    if not torch.isfinite(torch.tensor(float(temperature))) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    credit = torch.sigmoid(margins.float() / float(temperature))
    return torch.where(valid, credit, torch.zeros_like(credit))


def group_standardized_advantages(
    rewards: Tensor,
    num_generations: int,
    *,
    epsilon: float = GRPO_REWARD_STD_EPSILON,
) -> tuple[Tensor, Tensor]:
    """Exact vendored-GRPO group normalization for a flat global reward vector.

    Returns ``(advantages, repeated_group_std)``.  ``Tensor.std`` deliberately
    keeps its default unbiased correction, matching the pinned vendor code.
    """

    if not isinstance(rewards, Tensor) or rewards.ndim != 1:
        raise ValueError("rewards must be 1-D")
    if not bool(torch.isfinite(rewards).all()):
        raise FloatingPointError("rewards contain NaN or Inf")
    if isinstance(num_generations, bool) or not isinstance(num_generations, int) or num_generations < 2:
        raise ValueError("num_generations must be an integer >= 2")
    if rewards.numel() == 0 or rewards.numel() % num_generations != 0:
        raise ValueError("reward count must be a non-zero multiple of num_generations")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or float(epsilon) <= 0.0:
        raise ValueError("epsilon must be positive")
    grouped = rewards.view(-1, num_generations)
    means = grouped.mean(dim=1).repeat_interleave(num_generations)
    stds = grouped.std(dim=1).repeat_interleave(num_generations)
    return (rewards - means) / (stds + float(epsilon)), stds


def all_wrong_prefix_advantages(
    accuracy_rewards: Tensor,
    credits: Tensor,
    valid: Tensor,
    num_generations: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Route standardized prefix credit only to fully valid all-wrong groups.

    Returns ``(advantages, active_rows, repeated_credit_std)``.  Accuracy is
    kept separate from format/outcome rewards: the ordinary vendored GRPO
    loss remains responsible for those rewards on the full completion.  A
    group with any invalid/ambiguous answer boundary is disabled as a whole
    so a fail-closed zero cannot become an artificial low-credit comparison.
    """

    tensors = {
        "accuracy_rewards": accuracy_rewards,
        "credits": credits,
        "valid": valid,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, Tensor) or tensor.ndim != 1:
            raise ValueError(f"{name} must be 1-D")
    if not (accuracy_rewards.shape == credits.shape == valid.shape):
        raise ValueError("accuracy, credit, and valid tensors must share shape")
    if valid.dtype != torch.bool:
        raise ValueError("valid must be boolean")
    if not bool(torch.isfinite(accuracy_rewards).all()):
        raise FloatingPointError("accuracy rewards contain NaN or Inf")
    if not bool(torch.isfinite(credits).all()):
        raise FloatingPointError("credits contain NaN or Inf")
    if bool(((accuracy_rewards < 0.0) | (accuracy_rewards > 1.0)).any()):
        raise ValueError("accuracy rewards must lie in [0,1]")
    if isinstance(num_generations, bool) or not isinstance(num_generations, int):
        raise ValueError("num_generations must be an integer")
    if num_generations < 2 or credits.numel() == 0 or credits.numel() % num_generations != 0:
        raise ValueError("credit count must be a non-zero multiple of num_generations >= 2")

    grouped_accuracy = accuracy_rewards.view(-1, num_generations)
    grouped_valid = valid.view(-1, num_generations)
    active_groups = (grouped_accuracy == 0.0).all(dim=1) & grouped_valid.all(dim=1)
    active_rows = active_groups.repeat_interleave(num_generations)
    advantages, repeated_stds = group_standardized_advantages(
        credits,
        num_generations,
    )
    return advantages * active_rows.float(), active_rows, repeated_stds


def credit_softmax_bridge_weights(
    credits: Tensor,
    active_rows: Tensor,
    num_generations: int,
    *,
    temperature: float = 1.0,
) -> Tensor:
    """Detached within-group bridge weights with mean one on active groups."""

    if not isinstance(credits, Tensor) or credits.ndim != 1:
        raise ValueError("credits must be 1-D")
    if not isinstance(active_rows, Tensor) or active_rows.shape != credits.shape:
        raise ValueError("active_rows must match credits")
    if active_rows.dtype != torch.bool:
        raise ValueError("active_rows must be boolean")
    if not bool(torch.isfinite(credits).all()):
        raise FloatingPointError("credits contain NaN or Inf")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be numeric")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if isinstance(num_generations, bool) or not isinstance(num_generations, int):
        raise ValueError("num_generations must be an integer")
    if num_generations < 2 or credits.numel() == 0 or credits.numel() % num_generations != 0:
        raise ValueError("credit count must be a non-zero multiple of num_generations >= 2")
    grouped_active = active_rows.view(-1, num_generations)
    if not bool((grouped_active == grouped_active[:, :1]).all()):
        raise ValueError("bridge routing must activate or disable complete groups")
    grouped_weights = torch.softmax(
        credits.detach().float().view(-1, num_generations) / float(temperature),
        dim=1,
    ) * float(num_generations)
    grouped_weights = grouped_weights * grouped_active[:, :1].float()
    return grouped_weights.flatten()


def four_way_answer_bridge_loss(
    option_logits: Tensor,
    mapped_indices: Tensor,
    weights: Tensor,
    valid: Tensor,
) -> tuple[Tensor, Tensor]:
    """Weighted A-D cross-entropy at the pre-answer causal logit position.

    Returns ``(loss, per_row_cross_entropy)``.  The scalar is normalized by
    the sum of active weights.  With no active row it is an exact
    graph-connected zero, allowing every rank to execute the same code path.
    """

    if not isinstance(option_logits, Tensor) or option_logits.ndim != 2:
        raise ValueError("option_logits must have shape [B,4]")
    batch_size, option_count = option_logits.shape
    if option_count != len(OPTION_LETTERS):
        raise ValueError("option_logits must contain A-D columns")
    for name, tensor in (
        ("mapped_indices", mapped_indices),
        ("weights", weights),
        ("valid", valid),
    ):
        if not isinstance(tensor, Tensor) or tensor.ndim != 1 or tensor.shape[0] != batch_size:
            raise ValueError(f"{name} must have shape [{batch_size}]")
    if valid.dtype != torch.bool:
        raise ValueError("valid must be boolean")
    if bool(((mapped_indices < 0) | (mapped_indices >= option_count)).any()):
        raise ValueError("mapped_indices must lie in [0,3]")
    if not bool(torch.isfinite(option_logits).all()):
        raise FloatingPointError("option logits contain NaN or Inf")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0.0).any()):
        raise ValueError("bridge weights must be finite and non-negative")
    if bool(((weights > 0.0) & ~valid).any()):
        raise ValueError("invalid answer rows cannot carry bridge weight")

    per_row = F.cross_entropy(
        option_logits.float(),
        mapped_indices.to(device=option_logits.device, dtype=torch.long),
        reduction="none",
    )
    weights = weights.to(device=option_logits.device, dtype=per_row.dtype).detach()
    denominator = weights.sum()
    if float(denominator.detach().item()) == 0.0:
        return option_logits.sum() * 0.0, per_row
    return (weights * per_row).sum() / denominator, per_row


__all__ = [
    "AnswerCandidateLocations",
    "AnswerTokenStyle",
    "GRPO_REWARD_STD_EPSILON",
    "OPTION_LETTERS",
    "all_wrong_prefix_advantages",
    "answer_prefix_margins",
    "answer_prefix_option_logits",
    "answer_prefix_reasoning_mask",
    "bounded_prefix_credit",
    "build_answer_token_styles",
    "credit_softmax_bridge_weights",
    "four_way_answer_bridge_loss",
    "group_standardized_advantages",
    "locate_answer_candidate_tokens",
]
