from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from regcfpo.training.auxiliary import PairAugReferenceTrustConfig  # noqa: E402
from regcfpo.training.prefix_credit import (  # noqa: E402
    GRPO_REWARD_STD_EPSILON,
    all_wrong_prefix_advantages,
    answer_prefix_margins,
    answer_prefix_option_logits,
    answer_prefix_reasoning_mask,
    bounded_prefix_credit,
    build_answer_token_styles,
    credit_softmax_bridge_weights,
    four_way_answer_bridge_loss,
    group_standardized_advantages,
    locate_answer_candidate_tokens,
)


class _FakeQwenTokenizer:
    compact = {"A": 23465, "B": 36721, "C": 25630, "D": 36495}
    spaced = {"A": 362, "B": 425, "C": 356, "D": 422}

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert add_special_tokens is False
        for letter in "ABCD":
            if text == f"<answer>{letter}</answer>":
                return {"input_ids": [27, 9217, self.compact[letter], 522, 9217, 29]}
            if text == f"<answer> {letter} </answer>":
                return {"input_ids": [27, 9217, 29, self.spaced[letter], 694, 9217, 29]}
        raise AssertionError(f"unexpected tokenization request: {text}")


def _styles():
    return build_answer_token_styles(_FakeQwenTokenizer())


def test_token_styles_observe_qwen_fused_compact_answer_token() -> None:
    compact, spaced = _styles()
    assert compact.prefix_ids == (27, 9217)
    assert compact.candidate_token_ids == (23465, 36721, 25630, 36495)
    assert spaced.prefix_ids == (27, 9217, 29)
    assert spaced.candidate_token_ids == (362, 425, 356, 422)


def test_locator_accepts_compact_and_spaced_but_fails_closed_on_ambiguity() -> None:
    completion_ids = torch.tensor(
        [
            [10, 11, 27, 9217, 36721, 522, 9217, 29, 0, 0],
            [12, 27, 9217, 29, 356, 694, 9217, 29, 0, 0],
            [27, 9217, 23465, 27, 9217, 36721, 0, 0, 0, 0],
            [27, 9217, 999, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    completion_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    locations = locate_answer_candidate_tokens(
        completion_ids,
        completion_mask,
        _styles(),
    )
    assert locations.valid.tolist() == [True, True, False, False]
    assert locations.positions[:2].tolist() == [4, 4]
    assert locations.style_indices[:2].tolist() == [0, 1]


def test_margin_reads_only_logit_immediately_before_sampled_answer() -> None:
    styles = _styles()
    completion_ids = torch.tensor(
        [
            [5, 27, 9217, 23465, 0],
            [6, 27, 9217, 29, 362],
        ]
    )
    completion_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    locations = locate_answer_candidate_tokens(completion_ids, completion_mask, styles)
    prompt_length = 3
    logits = torch.zeros(2, prompt_length + completion_ids.shape[1], 40000)

    # Row 0 predicts the compact answer at absolute token P+3 from logit P+2.
    compact_pred = prompt_length + 3 - 1
    logits[0, compact_pred, styles[0].candidate_token_ids[1]] = 4.0  # mapped B
    logits[0, compact_pred, styles[0].candidate_token_ids[0]] = 1.0  # original A
    # Values after the sampled answer must be irrelevant (future-token leak guard).
    logits[0, compact_pred + 1, styles[0].candidate_token_ids[1]] = -100.0
    logits[0, compact_pred + 1, styles[0].candidate_token_ids[0]] = 100.0

    spaced_pred = prompt_length + 4 - 1
    logits[1, spaced_pred, styles[1].candidate_token_ids[2]] = 2.5  # mapped C
    logits[1, spaced_pred, styles[1].candidate_token_ids[0]] = -0.5  # original A
    margins = answer_prefix_margins(
        logits,
        prompt_length=prompt_length,
        locations=locations,
        original_indices=torch.tensor([0, 0]),
        mapped_indices=torch.tensor([1, 2]),
        styles=styles,
    )
    assert margins.tolist() == pytest.approx([3.0, 3.0])


def test_reasoning_mask_stops_before_answer_tag_for_both_styles() -> None:
    completion_ids = torch.tensor(
        [
            [10, 11, 27, 9217, 36721, 522, 9217, 29, 0, 0],
            [12, 27, 9217, 29, 356, 694, 9217, 29, 0, 0],
            [27, 9217, 23465, 27, 9217, 36721, 0, 0, 0, 0],
        ]
    )
    completion_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        ]
    )
    styles = _styles()
    locations = locate_answer_candidate_tokens(completion_ids, completion_mask, styles)
    reasoning = answer_prefix_reasoning_mask(completion_mask, locations, styles)
    assert reasoning[0].tolist() == [True, True, False, False, False, False, False, False, False, False]
    assert reasoning[1].tolist() == [True, False, False, False, False, False, False, False, False, False]
    assert not bool(reasoning[2].any())

    token_scores = torch.ones_like(completion_mask, dtype=torch.float32, requires_grad=True)
    loss = (token_scores * reasoning).sum()
    loss.backward()
    assert token_scores.grad is not None
    assert token_scores.grad[0, :2].tolist() == [1.0, 1.0]
    assert token_scores.grad[0, 2:].abs().sum().item() == 0.0


def test_four_way_bridge_uses_pre_answer_logits_and_correct_gradient_direction() -> None:
    styles = _styles()
    completion_ids = torch.tensor([[5, 27, 9217, 23465, 0]])
    completion_mask = torch.tensor([[1, 1, 1, 1, 0]])
    locations = locate_answer_candidate_tokens(completion_ids, completion_mask, styles)
    prompt_length = 3
    logits = torch.zeros(
        1,
        prompt_length + completion_ids.shape[1],
        40000,
        requires_grad=True,
    )
    prediction = prompt_length + 3 - 1
    with torch.no_grad():
        logits[0, prediction, styles[0].candidate_token_ids[0]] = 2.0
        logits[0, prediction, styles[0].candidate_token_ids[1]] = 0.0
        logits[0, prediction + 1, styles[0].candidate_token_ids[1]] = 100.0
    option_logits = answer_prefix_option_logits(
        logits,
        prompt_length=prompt_length,
        locations=locations,
        styles=styles,
    )
    loss, _ = four_way_answer_bridge_loss(
        option_logits,
        mapped_indices=torch.tensor([1]),
        weights=torch.ones(1),
        valid=locations.valid,
    )
    loss.backward()
    assert logits.grad is not None
    mapped_token = styles[0].candidate_token_ids[1]
    original_token = styles[0].candidate_token_ids[0]
    assert logits.grad[0, prediction, mapped_token].item() < 0.0
    assert logits.grad[0, prediction, original_token].item() > 0.0
    assert logits.grad[0, prediction + 1].abs().sum().item() == 0.0


def test_continuous_credit_breaks_all_wrong_base_reward_saturation() -> None:
    margins = torch.tensor([-2.0, -0.5, 0.5, 2.0])
    valid = torch.ones(4, dtype=torch.bool)
    credit = bounded_prefix_credit(margins, valid, temperature=1.0)
    assert bool(((credit > 0.0) & (credit < 1.0)).all())
    base_rewards = torch.zeros(4)
    advantages, stds = group_standardized_advantages(base_rewards + credit, 4)
    assert stds[0].item() > 0.0
    assert stds[0].item() > GRPO_REWARD_STD_EPSILON
    assert advantages.abs().mean().item() > 0.0
    assert advantages.mean().item() == pytest.approx(0.0, abs=1e-6)


def test_plan10_routes_prefix_only_to_fully_valid_all_wrong_groups() -> None:
    accuracy = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    credit = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.5])
    valid = torch.ones(8, dtype=torch.bool)
    advantages, active, stds = all_wrong_prefix_advantages(
        accuracy, credit, valid, 4
    )
    assert active.tolist() == [True] * 4 + [False] * 4
    assert advantages[:4].abs().sum().item() > 0.0
    assert advantages[:4].mean().item() == pytest.approx(0.0, abs=1e-6)
    assert advantages[4:].abs().sum().item() == 0.0
    assert stds[0].item() > GRPO_REWARD_STD_EPSILON

    bridge_weights = credit_softmax_bridge_weights(credit, active, 4)
    assert bridge_weights[:4].sum().item() == pytest.approx(4.0)
    assert bridge_weights[4:].sum().item() == 0.0

    invalid = valid.clone()
    invalid[2] = False
    invalid_advantages, invalid_active, _ = all_wrong_prefix_advantages(
        accuracy, credit, invalid, 4
    )
    assert not bool(invalid_active[:4].any())
    assert invalid_advantages[:4].abs().sum().item() == 0.0


def test_invalid_answer_gets_exactly_zero_credit() -> None:
    credit = bounded_prefix_credit(
        torch.tensor([100.0, -100.0]),
        torch.tensor([False, True]),
        temperature=1.0,
    )
    assert credit[0].item() == 0.0
    assert 0.0 <= credit[1].item() < 1e-20


def test_joint_config_allows_prefix_instead_of_static_gain() -> None:
    config = PairAugReferenceTrustConfig(
        gain_weight=0.0,
        trust_weight=0.1,
        prefix_credit_weight=1.0,
        prefix_credit_temperature=1.0,
    )
    assert config.gain_weight == 0.0
    plan10 = PairAugReferenceTrustConfig(
        gain_weight=0.0,
        trust_weight=0.1,
        prefix_credit_weight=1.0,
        prefix_credit_scope="reasoning_all_wrong",
        answer_bridge_weight=0.25,
    )
    assert plan10.answer_bridge_weight == pytest.approx(0.25)
    with pytest.raises(ValueError, match="answer bridge requires"):
        PairAugReferenceTrustConfig(
            gain_weight=0.0,
            trust_weight=0.1,
            prefix_credit_weight=1.0,
            prefix_credit_scope="full_completion",
            answer_bridge_weight=0.25,
        )
    with pytest.raises(ValueError, match="at least one positive"):
        PairAugReferenceTrustConfig(
            gain_weight=0.0,
            trust_weight=0.0,
            prefix_credit_weight=0.0,
        )
