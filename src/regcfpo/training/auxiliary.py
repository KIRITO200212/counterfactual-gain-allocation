"""Trainer scaffold for ReG-CFPO Stage-4 post-training (loss-level core).

Scaffold per ``results/phase1/phase1_plan4.md`` sections 9-11 and the
saturation-aware gate design in plan.md section 1.1.  Shared by all
objectives (``grpo``, ``vanilla_cfpo``, ``global_duality``, ``regcfpo``):

- saturation-aware success credit ``u_i`` (continuous; plan.md section 1.1
  with the plan7 section 1.1 R1 revision): SAT groups (all-correct, zero
  advantage) carry credit 1; MIX groups carry the batch-independent closed
  form ``(1-p) / (sqrt(p(1-p)) * sqrt(G-1))`` with ``p = k/G``
  (``closed_form_success_credit``); ALL_WRONG groups carry 0;
- gate-weight normalization with clipping, ``w_tilde = clip(w / (mean+eps))``,
  stop-gradient end to end (``w = sg[...]`` semantics);
- directional margin loss ``L_dir = mean w [m - G_train]_+``;
- null-anchor loss ``L_null = mean w [m0 + s(null)]_+``;
- inactive samples carry weight zero (never skipped forwards), keeping
  ZeRO-3 collectives consistent across ranks.

Input contract (fail-fast, plan4 section 9.1):

- all tensor inputs must be finite; NaN/Inf raises ``FloatingPointError``
  immediately (the smoke hard gate "no NaN" must be observable, matching the
  ``qwen_adapter.py`` discipline);
- ``group_rewards`` must be the ACCURACY-ONLY component mean per group, not
  the vendored trainer's multi-component reward sum, and must be computed
  after the global reward gather; the final weight normalization/clipping
  (``normalize_gate_weights``) must be applied on the global batch (all
  ranks);
- ``c_loc`` (locator confidence) and ``c_op`` (operator reliability) are
  preregistered multipliers that the CALLER pre-multiplies into the raw
  weights before calling this module (for the GT-box Stage-4 configuration
  both are 1.0; they stay caller-side so the module never needs locator
  state).

Hyperparameter provenance and calibration plan: ``m = m0 = 1.0`` are
margins in log-odds units (one nat of likelihood ratio); ``clip = 4.0`` and
``eps = 1e-6`` follow plan.md section 1.1.  ``gamma`` and ``lambda_null``
have no validated values yet: engineering smoke must log L_dir, L_null, the
GRPO policy loss, w_tilde mean and clip-hit fraction, and the 50-step
scientific smoke freezes gamma/lambda only via the preregistered
loss-scale rule (auxiliary mass at initialization targeted at 10-30% of
the per-token GRPO objective), never by dev-accuracy tuning.

The AReG-CFPO block (``AReGConfig``, ``areg_swap_loss``,
``areg_preservation_loss``, ``areg_cfpo_auxiliary_loss``) implements the
replacement objective of amendment
``asymmetric_directional_credit_revision_20260826`` (plan7 section 4 R1).
UNIT CONTRACT: unlike the regcfpo margins above (mean-per-token over the
candidate span, the frozen historical convention), every AReG score, margin,
and reference value is in TOTAL nats over the 7-token ``full_template_span``
(``s_tilde = 7 * s_mean``), matching the offline theta0 reference columns in
the train manifest.  The killed regcfpo functions are unchanged.

The vendored VLMGRPOTrainer integration, dataset feature preservation, and
trainer-level equivalence tests (gamma=0 identity, null-only
no-directional-gradient, gate behavior, rank-consistent auxiliary forward
counts) are separate work items gated by TRAINING_SMOKE_AUTHORIZED.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

EPSILON = 1e-6

# V1 is retained byte-for-byte in behavior via the default mode.  Plan9 V2
# must opt into the monotonic raw k/G credit explicitly.
LEGACY_GATE_CREDIT_MODE = "legacy_closed_form_normalized"
MONOTONIC_GATE_CREDIT_MODE = "monotonic_k_over_g_raw"
GATE_CREDIT_MODES = (LEGACY_GATE_CREDIT_MODE, MONOTONIC_GATE_CREDIT_MODE)

# Plan9 joint PairAug/reference-gain objective.  The trust region is defined
# on the normalized distribution over the four frozen answer candidates.
JOINT_OPTION_COUNT = 4
JOINT_REFERENCE_FIELDS = (
    "joint_s_swap_0",
    "joint_source_logps_0",
    "joint_null_logps_0",
)
LEGACY_PREFIX_CREDIT_SCOPE = "full_completion"
PLAN10_PREFIX_CREDIT_SCOPE = "reasoning_all_wrong"
PREFIX_CREDIT_SCOPES = (
    LEGACY_PREFIX_CREDIT_SCOPE,
    PLAN10_PREFIX_CREDIT_SCOPE,
)


def _require_finite_1d(name: str, tensor: Tensor) -> Tensor:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"{name} contains NaN or Inf; refusing to continue")
    return tensor


@dataclass(frozen=True)
class GateConfig:
    """Gate hyperparameters with an explicit V1/V2 credit policy.

    ``legacy_closed_form_normalized`` is the frozen V1 behavior.  The plan9
    V2 mode ``monotonic_k_over_g_raw`` uses the factual-success fraction
    directly and deliberately bypasses batch normalization/clipping, so a
    singleton MIX group cannot be silently promoted to weight one.
    """

    margin_dir: float = 1.0
    margin_null: float = 1.0
    weight_clip: float = 4.0
    epsilon: float = EPSILON
    credit_mode: str = LEGACY_GATE_CREDIT_MODE

    def __post_init__(self) -> None:
        if self.credit_mode not in GATE_CREDIT_MODES:
            raise ValueError(
                f"unknown gate credit_mode {self.credit_mode!r}; "
                f"expected one of {list(GATE_CREDIT_MODES)}"
            )


@dataclass(frozen=True)
class PairAugReferenceTrustConfig:
    """Hyperparameters for the true Plan9 joint objective.

    ``gain_delta`` is the requested per-example increase in the swapped-image
    mapped-vs-original margin relative to frozen theta0.  ``gain_weight`` and
    ``trust_weight`` are separate coefficients inside the auxiliary bracket;
    ``gamma`` remains the bracket's overall scale in the trainer.  The optional
    ``prefix_credit_weight`` adds bounded mapped-vs-original credit at
    ``prefix_credit_temperature``.  The legacy ``full_completion`` scope
    reproduces Plan9 V4.  The Plan10 ``reasoning_all_wrong`` scope keeps the
    ordinary outcome loss on the full completion, routes prefix credit only
    to pre-``<answer>`` reasoning tokens in fully valid all-wrong groups, and
    may add a four-way on-policy answer bridge.  Source and null preservation
    are one-sided trust-region constraints: they contribute gradient only
    after KL(theta0 || theta) exceeds their slack.
    """

    gain_delta: float = 1.0
    gain_weight: float = 1.0
    trust_weight: float = 0.1
    source_kl_slack: float = 0.02
    null_kl_slack: float = 0.02
    prefix_credit_weight: float = 0.0
    prefix_credit_temperature: float = 1.0
    prefix_credit_scope: str = LEGACY_PREFIX_CREDIT_SCOPE
    answer_bridge_weight: float = 0.0
    answer_bridge_temperature: float = 1.0

    def __post_init__(self) -> None:
        values = {
            "gain_delta": self.gain_delta,
            "gain_weight": self.gain_weight,
            "trust_weight": self.trust_weight,
            "source_kl_slack": self.source_kl_slack,
            "null_kl_slack": self.null_kl_slack,
            "prefix_credit_weight": self.prefix_credit_weight,
            "prefix_credit_temperature": self.prefix_credit_temperature,
            "answer_bridge_weight": self.answer_bridge_weight,
            "answer_bridge_temperature": self.answer_bridge_temperature,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric, got {value!r}")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.gain_delta < 0.0:
            raise ValueError("gain_delta must be non-negative")
        if self.gain_weight < 0.0:
            raise ValueError("gain_weight must be non-negative")
        if self.trust_weight < 0.0:
            raise ValueError("trust_weight must be non-negative")
        if self.source_kl_slack < 0.0 or self.null_kl_slack < 0.0:
            raise ValueError("source/null KL slack must be non-negative")
        if self.prefix_credit_weight < 0.0:
            raise ValueError("prefix_credit_weight must be non-negative")
        if self.prefix_credit_temperature <= 0.0:
            raise ValueError("prefix_credit_temperature must be positive")
        if self.prefix_credit_scope not in PREFIX_CREDIT_SCOPES:
            raise ValueError(
                f"unknown prefix_credit_scope {self.prefix_credit_scope!r}; "
                f"expected one of {list(PREFIX_CREDIT_SCOPES)}"
            )
        if self.answer_bridge_weight < 0.0:
            raise ValueError("answer_bridge_weight must be non-negative")
        if self.answer_bridge_temperature <= 0.0:
            raise ValueError("answer_bridge_temperature must be positive")
        if self.answer_bridge_weight > 0.0 and (
            self.prefix_credit_scope != PLAN10_PREFIX_CREDIT_SCOPE
            or self.prefix_credit_weight <= 0.0
        ):
            raise ValueError(
                "answer bridge requires positive prefix credit in "
                f"{PLAN10_PREFIX_CREDIT_SCOPE!r} scope"
            )
        if (
            self.gain_weight == 0.0
            and self.trust_weight == 0.0
            and self.prefix_credit_weight == 0.0
            and self.answer_bridge_weight == 0.0
        ):
            raise ValueError("joint objective requires at least one positive loss/reward weight")


def saturation_gate_flags(group_rewards: Tensor) -> Tensor:
    """Return the stratum gate in {0, 1} from accuracy-only group means.

    SAT (mean == 1) and MIX (0 < mean < 1) keep the credit enabled;
    ALL_WRONG (mean == 0) disables it.  NaN/Inf raises immediately.
    """

    means = _require_finite_1d("group_rewards", group_rewards).float()
    if bool(((means < 0.0) | (means > 1.0)).any()):
        raise ValueError("group reward means must lie in [0, 1]")
    return (means > 0.0).float()


def closed_form_success_credit(
    group_rewards: Tensor, num_generations: int, *, epsilon: float = EPSILON
) -> Tensor:
    """Batch-independent closed-form success credit ``u_i`` (plan7 section 1.1 R1).

    For binary accuracy rewards, group size ``G`` and ``k`` correct rollouts
    (``p = k / G``):

    - SAT (``k == G``): credit 1 — the outcome reward is saturated but the
      factual answers are reliable;
    - ALL_WRONG (``k == 0``): credit 0;
    - MIX (``0 < k < G``): ``(1 - p) / (sqrt(p(1-p)) * sqrt(G-1))``, exactly
      ``max_j[A_ij]^+ / sqrt(G-1)`` under the POPULATION std (ddof=0)
      reading of the group-relative advantage.

    The credit is computed directly from ``k/G`` and never passes through
    the empirical advantage of ``group_accuracy_rewards`` (whose ddof=1
    deliberately mirrors the vendored trainer and serves the policy side
    only), so a MIX group's credit does not depend on the other groups in
    the batch — the previous batch-mean normalization degenerated to exactly
    1 under the one-prompt-group-per-step configuration and the continuous
    credit never actually took effect.  For G=4 the MIX values are
    k=1 -> 1.000, k=2 -> 0.577 (1/sqrt(3)), k=3 -> 0.333 (1/3).

    Informed choice (plan7 section 1.1, option 1): the credit is
    NON-MONOTONIC in ``k`` and DISCONTINUOUS at ``k == G`` (a U-shape: the
    k=G-1 stratum — the closest-to-saturation one — gets the LOWEST MIX
    credit, while SAT jumps back to 1).  MIX credit derives from
    outcome-advantage strength while SAT credit derives from "factual
    answers reliable but outcome signal exhausted"; the two semantics are
    deliberately not on the same scale.
    """

    means = _require_finite_1d("group_rewards", group_rewards).float()
    if not isinstance(num_generations, int) or isinstance(num_generations, bool):
        raise ValueError("num_generations must be an integer")
    if num_generations < 2:
        raise ValueError("num_generations must be >= 2 for group-relative credit")
    if bool(((means < 0.0) | (means > 1.0)).any()):
        raise ValueError("group reward means must lie in [0, 1]")
    is_sat = means >= 1.0 - epsilon
    is_all_wrong = means <= 0.0
    # clamp keeps the formula finite on non-binary edge means; for binary
    # rewards (the contract) MIX means are k/G and never reach the clamps.
    safe_p = means.clamp(min=epsilon, max=1.0 - epsilon)
    mix_credit = (1.0 - safe_p) / (
        torch.sqrt(safe_p * (1.0 - safe_p)) * math.sqrt(num_generations - 1)
    )
    return torch.where(
        is_sat,
        torch.ones_like(means),
        torch.where(is_all_wrong, torch.zeros_like(means), mix_credit),
    )


def monotonic_success_credit(
    group_rewards: Tensor, num_generations: int, *, epsilon: float = EPSILON
) -> Tensor:
    """Plan9 V2 factual-reliability credit ``w = k/G``.

    The input is the per-group mean of binary accuracy rewards.  Values must
    lie on the discrete ``k/G`` grid; accepting arbitrary fractions would
    hide a reward-routing bug.  Returned credits are detached and monotonic:
    for G=4, ``[0, .25, .5, .75, 1]``.
    """

    means = _require_finite_1d("group_rewards", group_rewards).float()
    if not isinstance(num_generations, int) or isinstance(num_generations, bool):
        raise ValueError("num_generations must be an integer")
    if num_generations < 2:
        raise ValueError("num_generations must be >= 2 for group-relative credit")
    if bool(((means < 0.0) | (means > 1.0)).any()):
        raise ValueError("group reward means must lie in [0, 1]")
    successes = torch.round(means * num_generations)
    reconstructed = successes / float(num_generations)
    if not bool(torch.isclose(means, reconstructed, atol=epsilon, rtol=0.0).all()):
        raise ValueError(
            "group reward means must lie on the binary-accuracy k/G grid for "
            "monotonic_k_over_g_raw credit"
        )
    return reconstructed.detach()


def normalize_gate_weights(
    raw_weights: Tensor, *, clip: float = 4.0, epsilon: float = EPSILON
) -> Tensor:
    """Batch-normalize raw gate weights: ``clip(w / (mean(w) + eps))``.

    Stop-gradient end to end (``w = sg[u * c_loc * c_op]``): the returned
    weights never propagate gradients.  Inactive samples (weight zero) stay
    exactly zero.  Callers must pass global-batch weights when running
    distributed (see module input contract).
    """

    weights = _require_finite_1d("raw_weights", raw_weights).float()
    if bool((weights < 0.0).any()):
        raise ValueError("gate weights must be non-negative")
    weights = weights.detach()
    mean = weights.mean()
    normalized = weights / (mean + epsilon)
    normalized = torch.clamp(normalized, max=clip)
    return torch.where(weights > 0, normalized, torch.zeros_like(normalized))


def directional_loss(
    weights: Tensor, g_train: Tensor, *, margin: float = 1.0
) -> Tensor:
    """``L_dir = mean_i w_i [m - G_train_i]_+``.

    ``g_train = s(swap) - s(null)`` keeps gradients through the policy;
    weights are stop-gradient credit assignments.
    """

    weights = _require_finite_1d("weights", weights).detach()
    g_train = _require_finite_1d("g_train", g_train)
    if weights.shape != g_train.shape:
        raise ValueError("weights and g_train must share shape")
    deficit = torch.clamp(margin - g_train, min=0.0)
    return (weights * deficit).mean()


def null_anchor_loss(
    weights: Tensor, s_null: Tensor, *, margin_null: float = 1.0
) -> Tensor:
    """``L_null = mean_i w_i [m0 + s(null_i)]_+`` keeping the original answer
    supported under the relation-preserving null edit."""

    weights = _require_finite_1d("weights", weights).detach()
    s_null = _require_finite_1d("s_null", s_null)
    if weights.shape != s_null.shape:
        raise ValueError("weights and s_null must share shape")
    violation = torch.clamp(margin_null + s_null, min=0.0)
    return (weights * violation).mean()


def regcfpo_auxiliary_loss(
    *,
    weights: Tensor,
    g_train: Tensor,
    s_null: Tensor,
    gamma: float,
    lambda_null: float,
    config: GateConfig | None = None,
    normalize_weights: bool = True,
) -> Tensor:
    """``gamma * (L_dir + lambda_null * L_null)`` with gamma=0 identity.

    Raw gate credits are normalized and clipped internally
    (``normalize_gate_weights`` with ``config.weight_clip``/``epsilon``)
    unless ``normalize_weights`` is false for callers that already applied
    the global-batch normalization.  At ``gamma == 0`` the result is exactly
    zero (continued-GRPO equivalence) with both auxiliary score tensors kept
    graph-connected so gradients exist and are exactly zero; finite inputs
    are enforced so NaN can never be silently absorbed.
    """

    cfg = config or GateConfig()
    _require_finite_1d("weights", weights)
    _require_finite_1d("g_train", g_train)
    _require_finite_1d("s_null", s_null)
    if gamma == 0.0:
        return g_train.sum() * 0.0 + s_null.sum() * 0.0
    effective = (
        normalize_gate_weights(weights, clip=cfg.weight_clip, epsilon=cfg.epsilon)
        if normalize_weights
        else weights.detach()
    )
    l_dir = directional_loss(effective, g_train, margin=cfg.margin_dir)
    l_null = null_anchor_loss(effective, s_null, margin_null=cfg.margin_null)
    return gamma * (l_dir + lambda_null * l_null)


# --------------------------------------------------------------------- #
# AReG-CFPO (amendment asymmetric_directional_credit_revision_20260826)   #
# --------------------------------------------------------------------- #

#: SmoothL1 transition point of the AReG preservation anchors, in total nats:
#: deviations below ``beta`` are quadratic, above it linear (torch convention).
AREG_SMOOTH_L1_BETA = 1.0


@dataclass(frozen=True)
class AReGConfig:
    """Frozen hyperparameters of the AReG-CFPO auxiliary loss (plan7 sec. 4 R1).

    UNIT CONTRACT: every score and margin here is in TOTAL nats over the
    7-token ``full_template_span`` (``s_tilde = 7 * s_mean``; amendment
    ``score_unit: total_nats``).  The killed regcfpo margins
    (``GateConfig.margin_dir``/``margin_null``) are mean-per-token and must
    never be mixed into this config.

    - ``margin_swap`` (``m_s``): absolute arm of the swap target, frozen 1.0.
    - ``margin_did`` (``m_d``): additive offset on the detached null score,
      frozen 6.5 (``m_s - median_theta0[s_null]``, quantile-calibrated and
      binding-prechecked on the frozen 174-row cohort).
    - ``lambda_pres``: weight of the preservation anchor inside the auxiliary
      bracket of ``L = L_GRPO + beta * KL + gamma * (L_swap + lambda_pres *
      L_pres)``.
    - ``eta``: relative weight of the factual-anchor (difference) term.
    - ``eta_abs``: relative weight of the absolute anchor term
      ``SmoothL1(logp(y|I_fact) - logp_theta0(y|I_fact))``; initial value
      equals ``eta`` per the amendment, then calibrated by the gradient-ratio
      rule on 20-step engineering runs only.
    - ``weight_clip``/``epsilon``: gate-weight normalization parameters for
      the standalone (``normalize_weights=True``) path; the trainer always
      normalizes on the global batch at generation time and calls with
      ``normalize_weights=False``.
    """

    margin_swap: float = 1.0
    margin_did: float = 6.5
    lambda_pres: float = 1.0
    eta: float = 1.0
    eta_abs: float = 1.0
    smooth_l1_beta: float = AREG_SMOOTH_L1_BETA
    weight_clip: float = 4.0
    epsilon: float = EPSILON


def areg_swap_target(s_null: Tensor, *, margin_swap: float, margin_did: float) -> Tensor:
    """``t_i = max(m_s, sg[s_null_i] + m_d)`` (total nats).

    ``sg`` is a hard stop-gradient: the null-branch score sets the per-sample
    calibration bar but receives NO directional gradient through this target
    (plan7 section 4.2 — the optimizer must not satisfy the hinge by pushing
    the null score down).  The null score DOES still receive the preservation
    anchor gradient via :func:`areg_preservation_loss`; only the directional
    path is detached here.
    """

    detached = _require_finite_1d("s_null", s_null).detach()
    return torch.clamp_min(detached + margin_did, margin_swap)


def areg_did_arm_mask(s_null: Tensor, *, margin_swap: float, margin_did: float) -> Tensor:
    """Boolean mask: True where the DiD arm strictly dominates ``t_i``.

    ``sg[s_null] + m_d > m_s`` (a tie counts as the absolute arm).  First-class
    monitor: the amendment's margin binding precheck predicts 40-60% DiD-arm
    dominance on the frozen cohort.
    """

    detached = _require_finite_1d("s_null", s_null).detach()
    return detached + margin_did > margin_swap


def areg_swap_loss(
    weights: Tensor,
    s_swap: Tensor,
    s_null: Tensor,
    *,
    margin_swap: float,
    margin_did: float,
) -> Tensor:
    """``L_swap = mean_i w_i [t_i - s_swap_i]_+`` (total nats).

    The only directional gradient path is through ``s_swap``; ``t_i`` is
    stop-gradient by construction (:func:`areg_swap_target`) and ``weights``
    are stop-gradient credit assignments.
    """

    weights = _require_finite_1d("weights", weights).detach()
    s_swap = _require_finite_1d("s_swap", s_swap)
    target = areg_swap_target(s_null, margin_swap=margin_swap, margin_did=margin_did)
    if weights.shape != s_swap.shape or weights.shape != target.shape:
        raise ValueError("weights, s_swap, and s_null must share shape")
    deficit = torch.clamp(target - s_swap, min=0.0)
    return (weights * deficit).mean()


def areg_preservation_loss(
    weights: Tensor,
    s_null: Tensor,
    s_fact: Tensor,
    logp_fact_orig: Tensor,
    *,
    s_null_0: Tensor,
    s_fact_0: Tensor,
    logp_fact_orig_0: Tensor,
    eta: float,
    eta_abs: float,
    beta: float = AREG_SMOOTH_L1_BETA,
) -> Tensor:
    """Theta0-anchored preservation loss (plan7 section 4.3 R1), total nats.

    ``L_pres = mean_i w_i [ SmoothL1(s_null_i - s_null_i_0)
    + eta * SmoothL1(s_fact_i - s_fact_i_0)
    + eta_abs * SmoothL1(logp(y_i|I_fact) - logp_theta0(y_i|I_fact)) ]``.

    The first two terms pin the mapped-vs-original DIFFERENCES under the null
    and factual branches; the third pins the ABSOLUTE original-answer
    log-prob on the unedited image, closing the "all six absolute log-probs
    sink together" failure mode that a difference-only anchor cannot see.
    Reference values (``*_0``) are theta0 scores precomputed offline into the
    train manifest (zero extra forwards); they are detached constants here.
    Gradients flow through all three current-policy scores (this is the one
    place the null branch receives gradient).
    """

    weights = _require_finite_1d("weights", weights).detach()
    s_null = _require_finite_1d("s_null", s_null)
    s_fact = _require_finite_1d("s_fact", s_fact)
    logp_fact_orig = _require_finite_1d("logp_fact_orig", logp_fact_orig)
    s_null_0 = _require_finite_1d("s_null_0", s_null_0).detach()
    s_fact_0 = _require_finite_1d("s_fact_0", s_fact_0).detach()
    logp_fact_orig_0 = _require_finite_1d("logp_fact_orig_0", logp_fact_orig_0).detach()
    shape = weights.shape
    for name, tensor in (
        ("s_null", s_null),
        ("s_fact", s_fact),
        ("logp_fact_orig", logp_fact_orig),
        ("s_null_0", s_null_0),
        ("s_fact_0", s_fact_0),
        ("logp_fact_orig_0", logp_fact_orig_0),
    ):
        if tensor.shape != shape:
            raise ValueError(f"{name} must share the weights shape; AReG losses are per-group")
    anchor = (
        F.smooth_l1_loss(s_null, s_null_0, reduction="none", beta=beta)
        + eta * F.smooth_l1_loss(s_fact, s_fact_0, reduction="none", beta=beta)
        + eta_abs
        * F.smooth_l1_loss(logp_fact_orig, logp_fact_orig_0, reduction="none", beta=beta)
    )
    return (weights * anchor).mean()


#: Manifest columns carrying the offline theta0 references (total nats).
AREG_REFERENCE_FIELDS = ("areg_s_null_0", "areg_s_fact_0", "areg_logp_fact_orig_0")


def build_areg_reference_tensors(
    groups: Sequence[Mapping[str, Any]], *, device: torch.device | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    """Read the theta0 reference columns from per-group manifest metadata.

    Returns ``(s_null_0, s_fact_0, logp_fact_orig_0)`` (float32, total nats)
    aligned with ``groups``.  Fail-closed: a DIRECTIONAL group with a
    missing/non-finite reference is a manifest contract violation and raises.
    Replay groups (aux weight exactly zero, forwards never individually
    skipped) are permitted null references and are zero-filled — the zero
    gate weight masks their contribution exactly.
    """

    columns: dict[str, list[float]] = {field: [] for field in AREG_REFERENCE_FIELDS}
    for meta in groups:
        row_kind = str(meta["row_kind"])
        for field in AREG_REFERENCE_FIELDS:
            value = meta.get(field)
            if value is None or not math.isfinite(float(value)):
                if row_kind == "directional":
                    raise ValueError(
                        f"directional group {meta.get('sample_id')!r} carries a "
                        f"missing/non-finite {field} reference column; the merged "
                        "manifest contract requires finite theta0 references on "
                        "every directional row"
                    )
                value = 0.0  # replay row: masked by the zero gate weight
            columns[field].append(float(value))
    return tuple(
        torch.tensor(columns[field], dtype=torch.float32, device=device)
        for field in AREG_REFERENCE_FIELDS
    )


def areg_cfpo_auxiliary_loss(
    *,
    weights: Tensor,
    s_swap: Tensor,
    s_null: Tensor,
    s_fact: Tensor,
    logp_fact_orig: Tensor,
    s_null_0: Tensor | None = None,
    s_fact_0: Tensor | None = None,
    logp_fact_orig_0: Tensor | None = None,
    gamma: float,
    config: AReGConfig | None = None,
    normalize_weights: bool = True,
) -> Tensor:
    """``gamma * (L_swap + lambda_pres * L_pres)`` with a gamma=0 identity.

    Mirrors :func:`regcfpo_auxiliary_loss`: raw gate credits are normalized
    and clipped internally unless ``normalize_weights`` is false (the trainer
    normalizes on the GLOBAL batch at generation time).  At ``gamma == 0``
    the result is exactly zero (continued-GRPO equivalence, E0.2 under the
    6-forward configuration) with all four score tensors kept graph-connected
    so gradients exist and are exactly zero; the theta0 reference tensors are
    not consumed in that case and may be omitted.
    """

    cfg = config or AReGConfig()
    _require_finite_1d("weights", weights)
    _require_finite_1d("s_swap", s_swap)
    _require_finite_1d("s_null", s_null)
    _require_finite_1d("s_fact", s_fact)
    _require_finite_1d("logp_fact_orig", logp_fact_orig)
    if gamma == 0.0:
        return (s_swap.sum() + s_null.sum() + s_fact.sum() + logp_fact_orig.sum()) * 0.0
    if s_null_0 is None or s_fact_0 is None or logp_fact_orig_0 is None:
        raise ValueError(
            "areg_cfpo_auxiliary_loss requires the theta0 reference tensors "
            "(s_null_0, s_fact_0, logp_fact_orig_0) when gamma > 0"
        )
    effective = (
        normalize_gate_weights(weights, clip=cfg.weight_clip, epsilon=cfg.epsilon)
        if normalize_weights
        else weights.detach()
    )
    l_swap = areg_swap_loss(
        effective,
        s_swap,
        s_null,
        margin_swap=cfg.margin_swap,
        margin_did=cfg.margin_did,
    )
    l_pres = areg_preservation_loss(
        effective,
        s_null,
        s_fact,
        logp_fact_orig,
        s_null_0=s_null_0,
        s_fact_0=s_fact_0,
        logp_fact_orig_0=logp_fact_orig_0,
        eta=cfg.eta,
        eta_abs=cfg.eta_abs,
        beta=cfg.smooth_l1_beta,
    )
    return gamma * (l_swap + cfg.lambda_pres * l_pres)


# --------------------------------------------------------------------- #
# Plan9 true joint PairAug + reference-gain + trust-region objective     #
# --------------------------------------------------------------------- #


def _require_finite_option_matrix(name: str, tensor: Tensor) -> Tensor:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 2 or tensor.shape[1] != JOINT_OPTION_COUNT:
        raise ValueError(
            f"{name} must have shape [N, {JOINT_OPTION_COUNT}], "
            f"got {tuple(tensor.shape)}"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"{name} contains NaN or Inf; refusing to continue")
    return tensor


def candidate_distribution_forward_kl(
    current_logps: Tensor,
    reference_logps: Tensor,
) -> Tensor:
    """Per-row ``KL(p_theta0 || p_theta)`` over the four answer candidates.

    Inputs are arbitrary finite candidate log scores in total nats.  Each row
    is normalized with log-softmax, so common answer-template probability
    drift does not trigger preservation by itself.  The frozen reference is
    detached; gradients flow only through ``current_logps``.
    """

    current = _require_finite_option_matrix("current_logps", current_logps).float()
    reference = _require_finite_option_matrix(
        "reference_logps", reference_logps
    ).detach().float()
    if current.shape != reference.shape:
        raise ValueError("current_logps and reference_logps must share shape")
    current_log_probs = torch.log_softmax(current, dim=-1)
    reference_log_probs = torch.log_softmax(reference, dim=-1)
    reference_probs = reference_log_probs.exp()
    kl = (reference_probs * (reference_log_probs - current_log_probs)).sum(dim=-1)
    # Roundoff can produce tiny negative values at exact identity.  Clamping
    # preserves the mathematical non-negativity and an exact zero baseline.
    return torch.clamp_min(kl, 0.0)


def reference_gain_softplus_loss(
    weights: Tensor,
    s_swap: Tensor,
    s_swap_0: Tensor,
    *,
    gain_delta: float,
) -> Tensor:
    """Dense frozen-reference gain loss.

    ``softplus(delta - (s_swap - s_swap_0))`` stays informative even when all
    current PairAug rollouts are wrong and GRPO has zero within-group reward
    variance.  It is therefore intentionally not gated by rollout success;
    callers pass a dense one on every swapped-input group.
    """

    weights = _require_finite_1d("weights", weights).detach()
    s_swap = _require_finite_1d("s_swap", s_swap)
    s_swap_0 = _require_finite_1d("s_swap_0", s_swap_0).detach()
    if not (weights.shape == s_swap.shape == s_swap_0.shape):
        raise ValueError("weights, s_swap, and s_swap_0 must share shape")
    if not math.isfinite(float(gain_delta)) or gain_delta < 0.0:
        raise ValueError("gain_delta must be finite and non-negative")
    gain = s_swap - s_swap_0
    return (weights * F.softplus(float(gain_delta) - gain)).mean()


def candidate_trust_region_loss(
    weights: Tensor,
    current_logps: Tensor,
    reference_logps: Tensor,
    *,
    slack: float,
) -> tuple[Tensor, Tensor]:
    """One-sided candidate-distribution trust region.

    Returns ``(mean weighted violation, per-row KL)``.  The loss is exactly
    zero, with zero gradient, while ``KL(theta0 || theta) <= slack``.
    """

    weights = _require_finite_1d("weights", weights).detach()
    if not math.isfinite(float(slack)) or slack < 0.0:
        raise ValueError("trust-region slack must be finite and non-negative")
    kl = candidate_distribution_forward_kl(current_logps, reference_logps)
    if kl.shape != weights.shape:
        raise ValueError("candidate KL rows must align with weights")
    violation = torch.relu(kl - float(slack))
    return (weights * violation).mean(), kl


def build_joint_reference_tensors(
    groups: Sequence[Mapping[str, Any]], *, device: torch.device | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    """Read frozen theta0 references for active Plan9 joint swap groups."""

    swap_values: list[float] = []
    source_rows: list[list[float]] = []
    null_rows: list[list[float]] = []
    for meta in groups:
        sample_id = str(meta.get("sample_id", ""))
        swap_value = meta.get("joint_s_swap_0")
        source_value = meta.get("joint_source_logps_0")
        null_value = meta.get("joint_null_logps_0")
        if swap_value is None or isinstance(swap_value, bool):
            raise ValueError(f"joint group {sample_id!r} lacks finite joint_s_swap_0")
        try:
            swap_float = float(swap_value)
        except (TypeError, ValueError):
            raise ValueError(
                f"joint group {sample_id!r} lacks finite joint_s_swap_0"
            ) from None
        if not math.isfinite(swap_float):
            raise ValueError(f"joint group {sample_id!r} lacks finite joint_s_swap_0")

        parsed_vectors: list[list[float]] = []
        for field, value in (
            ("joint_source_logps_0", source_value),
            ("joint_null_logps_0", null_value),
        ):
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise ValueError(
                    f"joint group {sample_id!r} requires four-value {field}"
                )
            try:
                vector = [float(item) for item in value]
            except (TypeError, ValueError):
                raise ValueError(
                    f"joint group {sample_id!r} requires finite four-value {field}"
                ) from None
            if len(vector) != JOINT_OPTION_COUNT or not all(
                math.isfinite(item) for item in vector
            ):
                raise ValueError(
                    f"joint group {sample_id!r} requires finite four-value {field}"
                )
            parsed_vectors.append(vector)
        swap_values.append(swap_float)
        source_rows.append(parsed_vectors[0])
        null_rows.append(parsed_vectors[1])
    return (
        torch.tensor(swap_values, dtype=torch.float32, device=device),
        torch.tensor(source_rows, dtype=torch.float32, device=device),
        torch.tensor(null_rows, dtype=torch.float32, device=device),
    )


def pairaug_reference_trust_auxiliary_loss(
    *,
    weights: Tensor,
    s_swap: Tensor,
    source_logps: Tensor,
    null_logps: Tensor,
    s_swap_0: Tensor,
    source_logps_0: Tensor,
    null_logps_0: Tensor,
    gamma: float,
    config: PairAugReferenceTrustConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Joint dense reference-gain and slack trust-region auxiliary bracket."""

    if not math.isfinite(float(gamma)) or gamma < 0.0:
        raise ValueError("gamma must be finite and non-negative")
    gain = reference_gain_softplus_loss(
        weights,
        s_swap,
        s_swap_0,
        gain_delta=config.gain_delta,
    )
    source_trust, source_kl = candidate_trust_region_loss(
        weights,
        source_logps,
        source_logps_0,
        slack=config.source_kl_slack,
    )
    null_trust, null_kl = candidate_trust_region_loss(
        weights,
        null_logps,
        null_logps_0,
        slack=config.null_kl_slack,
    )
    trust = source_trust + null_trust
    total = float(gamma) * (
        config.gain_weight * gain + config.trust_weight * trust
    )
    return total, {
        "gain": gain,
        "source_trust": source_trust,
        "null_trust": null_trust,
        "source_kl": source_kl,
        "null_kl": null_kl,
        "swap_gain": s_swap - s_swap_0.detach(),
    }


__all__ = [
    "AREG_REFERENCE_FIELDS",
    "AREG_SMOOTH_L1_BETA",
    "EPSILON",
    "AReGConfig",
    "GateConfig",
    "JOINT_OPTION_COUNT",
    "JOINT_REFERENCE_FIELDS",
    "LEGACY_PREFIX_CREDIT_SCOPE",
    "PLAN10_PREFIX_CREDIT_SCOPE",
    "PREFIX_CREDIT_SCOPES",
    "PairAugReferenceTrustConfig",
    "areg_cfpo_auxiliary_loss",
    "areg_did_arm_mask",
    "areg_preservation_loss",
    "areg_swap_loss",
    "areg_swap_target",
    "build_areg_reference_tensors",
    "build_joint_reference_tensors",
    "candidate_distribution_forward_kl",
    "candidate_trust_region_loss",
    "closed_form_success_credit",
    "GATE_CREDIT_MODES",
    "LEGACY_GATE_CREDIT_MODE",
    "MONOTONIC_GATE_CREDIT_MODE",
    "monotonic_success_credit",
    "saturation_gate_flags",
    "normalize_gate_weights",
    "pairaug_reference_trust_auxiliary_loss",
    "reference_gain_softplus_loss",
    "directional_loss",
    "null_anchor_loss",
    "regcfpo_auxiliary_loss",
]
