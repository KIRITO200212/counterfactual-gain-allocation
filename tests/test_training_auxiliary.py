"""Unit tests for the ReG-CFPO auxiliary loss scaffold."""

from __future__ import annotations

import math

import pytest
import torch

from regcfpo.training.auxiliary import (
    GateConfig,
    MONOTONIC_GATE_CREDIT_MODE,
    closed_form_success_credit,
    directional_loss,
    normalize_gate_weights,
    monotonic_success_credit,
    null_anchor_loss,
    regcfpo_auxiliary_loss,
    saturation_gate_flags,
)


def test_saturation_gate_strata():
    rewards = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
    flags = saturation_gate_flags(rewards)
    assert flags.tolist() == [1.0, 1.0, 1.0, 1.0, 0.0]


def test_saturation_gate_rejects_out_of_range():
    with pytest.raises(ValueError):
        saturation_gate_flags(torch.tensor([1.5]))
    with pytest.raises(ValueError):
        saturation_gate_flags(torch.tensor([-0.1]))


def test_saturation_gate_nan_fails_fast():
    with pytest.raises(FloatingPointError):
        saturation_gate_flags(torch.tensor([float("nan")]))
    with pytest.raises(FloatingPointError):
        saturation_gate_flags(torch.tensor([float("inf")]))


def test_closed_form_credit_g4_exact_values():
    """plan7 section 1.1 R1 frozen table: k=1 -> 1.000, k=2 -> 0.577, k=3 -> 0.333."""

    means = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])  # k = 0..4 at G=4
    credit = closed_form_success_credit(means, 4)
    assert credit[0].item() == 0.0  # ALL_WRONG
    assert credit[1].item() == pytest.approx(1.0, abs=1e-6)
    assert credit[2].item() == pytest.approx(1.0 / math.sqrt(3), abs=1e-6)
    assert credit[3].item() == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert credit[4].item() == 1.0  # SAT


def test_closed_form_credit_matches_population_std_bruteforce_g8():
    """G=8: the closed form equals max_j[A_ij]^+ / sqrt(G-1) with ddof=0."""

    num_generations = 8
    for k in range(num_generations + 1):
        rewards = torch.tensor([1.0] * k + [0.0] * (num_generations - k))
        credit = closed_form_success_credit(rewards.mean().unsqueeze(0), num_generations)
        if k == 0:
            expected = 0.0
        elif k == num_generations:
            expected = 1.0
        else:
            mean = rewards.mean()
            population_std = rewards.std(correction=0)
            max_positive = ((rewards - mean) / population_std).clamp_min(0.0).max()
            expected = (max_positive / math.sqrt(num_generations - 1)).item()
        assert credit.item() == pytest.approx(expected, abs=1e-6), f"k={k}"


def test_closed_form_credit_is_batch_independent():
    """A MIX group's credit must not change with the batch companions."""

    alone = closed_form_success_credit(torch.tensor([0.5]), 4)
    batched = closed_form_success_credit(torch.tensor([1.0, 0.5, 0.0, 0.25]), 4)
    assert alone.item() == pytest.approx(batched[1].item())
    # the old batch-mean normalization made a lone MIX group's credit exactly
    # 1; the closed form keeps it at the k/G value regardless of companions
    assert alone.item() == pytest.approx(1.0 / math.sqrt(3), abs=1e-6)


def test_closed_form_credit_rejects_invalid_inputs():
    with pytest.raises(FloatingPointError):
        closed_form_success_credit(torch.tensor([float("nan")]), 4)
    with pytest.raises(FloatingPointError):
        closed_form_success_credit(torch.tensor([float("inf")]), 4)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        closed_form_success_credit(torch.tensor([1.5]), 4)
    with pytest.raises(ValueError, match=">= 2"):
        closed_form_success_credit(torch.tensor([0.5]), 1)
    with pytest.raises(ValueError, match="integer"):
        closed_form_success_credit(torch.tensor([0.5]), 4.0)


def test_monotonic_credit_g4_exact_values_and_stop_gradient():
    means = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], requires_grad=True)
    credit = monotonic_success_credit(means, 4)
    assert credit.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert not credit.requires_grad


def test_monotonic_credit_rejects_off_grid_mean():
    with pytest.raises(ValueError, match="k/G grid"):
        monotonic_success_credit(torch.tensor([0.3]), 4)


def test_gate_config_rejects_unknown_credit_mode():
    with pytest.raises(ValueError, match="credit_mode"):
        GateConfig(credit_mode="silent_new_semantics")
    assert GateConfig(credit_mode=MONOTONIC_GATE_CREDIT_MODE).credit_mode == MONOTONIC_GATE_CREDIT_MODE


def test_normalize_gate_weights_keeps_inactive_exactly_zero():
    raw = torch.tensor([0.0, 1.0, 1.0, 3.0])
    weights = normalize_gate_weights(raw)
    assert weights[0].item() == 0.0
    assert weights[1].item() > 0.0
    assert float(weights.max()) <= 4.0


def test_normalize_gate_weights_clips():
    raw = torch.tensor([1.0, 100.0])
    weights = normalize_gate_weights(raw, clip=1.0)
    assert float(weights.max()) == pytest.approx(1.0)


def test_normalize_gate_weights_is_stop_gradient():
    raw = torch.tensor([1.0, 2.0], requires_grad=True)
    weights = normalize_gate_weights(raw)
    # stop-gradient end to end: normalized credits must be graph-detached
    assert not weights.requires_grad


def test_normalize_gate_weights_nan_fails_fast():
    with pytest.raises(FloatingPointError):
        normalize_gate_weights(torch.tensor([float("nan")]))


def test_directional_loss_margin_behavior():
    weights = torch.tensor([1.0, 1.0])
    g_train = torch.tensor([2.0, 0.0])  # above margin / below margin
    loss = directional_loss(weights, g_train, margin=1.0)
    assert loss.item() == pytest.approx(0.5)  # only second row violates


def test_directional_loss_keeps_policy_gradient():
    weights = torch.tensor([1.0])
    g_train = torch.tensor([0.0], requires_grad=True)
    loss = directional_loss(weights, g_train, margin=1.0)
    loss.backward()
    assert g_train.grad is not None and float(g_train.grad) == -1.0


def test_directional_loss_nan_fails_fast():
    with pytest.raises(FloatingPointError):
        directional_loss(torch.tensor([1.0]), torch.tensor([float("nan")]))


def test_null_anchor_loss_violation():
    weights = torch.tensor([1.0, 1.0])
    s_null = torch.tensor([-2.0, 0.5])  # anchored / violating
    loss = null_anchor_loss(weights, s_null, margin_null=1.0)
    assert loss.item() == pytest.approx(0.75)  # (0 + 1.5) / 2


def test_auxiliary_loss_applies_internal_normalization():
    weights = torch.tensor([0.0, 1.0, 1.0])  # mean 2/3 -> normalized [0, 1.5, 1.5]
    g_train = torch.tensor([0.0, 0.0, 0.0])
    s_null = torch.tensor([-10.0, -10.0, -10.0])  # anchored, L_null = 0
    loss = regcfpo_auxiliary_loss(
        weights=weights,
        g_train=g_train,
        s_null=s_null,
        gamma=1.0,
        lambda_null=1.0,
        config=GateConfig(margin_dir=1.0),
    )
    # mean over 3 of [0*1, 1.5*1, 1.5*1] = 1.0 (float32 eps tolerance)
    assert loss.item() == pytest.approx(1.0, abs=1e-4)


def test_auxiliary_loss_normalization_can_be_disabled_for_global_callers():
    weights = torch.tensor([1.0, 1.0])
    g_train = torch.tensor([0.0, 0.0])
    s_null = torch.tensor([-10.0, -10.0])
    loss = regcfpo_auxiliary_loss(
        weights=weights,
        g_train=g_train,
        s_null=s_null,
        gamma=1.0,
        lambda_null=0.0,
        normalize_weights=False,
    )
    assert loss.item() == pytest.approx(1.0)


def test_gamma_zero_identity_exact():
    weights = torch.tensor([1.0, 1.0])
    g_train = torch.tensor([-5.0, -5.0], requires_grad=True)
    s_null = torch.tensor([5.0, 5.0], requires_grad=True)
    loss = regcfpo_auxiliary_loss(
        weights=weights, g_train=g_train, s_null=s_null, gamma=0.0, lambda_null=1.0
    )
    assert loss.item() == 0.0
    loss.backward()
    assert float(g_train.grad.abs().sum()) == 0.0
    assert float(s_null.grad.abs().sum()) == 0.0


def test_gamma_zero_nan_still_fails_fast():
    with pytest.raises(FloatingPointError):
        regcfpo_auxiliary_loss(
            weights=torch.tensor([1.0]),
            g_train=torch.tensor([float("nan")]),
            s_null=torch.tensor([0.0]),
            gamma=0.0,
            lambda_null=1.0,
        )


def test_inactive_samples_still_carry_zero_weight_not_missing():
    # ZeRO-3 rule: inactive samples stay in the batch with weight zero,
    # so the loss is defined over the full batch tensor.
    weights = torch.tensor([0.0, 0.0, 1.0])
    g_train = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    s_null = torch.tensor([-10.0, -10.0, -10.0])
    loss = regcfpo_auxiliary_loss(
        weights=weights,
        g_train=g_train,
        s_null=s_null,
        gamma=1.0,
        lambda_null=0.0,
        config=GateConfig(margin_dir=1.0),
        normalize_weights=False,
    )
    assert loss.item() == pytest.approx(1.0 / 3.0)
