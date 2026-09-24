"""Fail-closed objective registry for Stage-4 GRPO post-training.

Implements the registry from ``results/phase1/phase1_plan5.md`` section 5.2,
extended by amendment ``asymmetric_directional_credit_revision_20260826``
(``configs/phase1_prereg.yaml``) with the AReG-CFPO replacement objective.
Four objectives are registered:

- ``continued_grpo``: vendored ``VLMGRPOTrainer`` behavior, unchanged.  The
  auxiliary branch is disabled and both auxiliary hyperparameters must be
  exactly zero (a non-zero value would silently do nothing, which is worse
  than failing loudly).
- ``regcfpo``: the original relation-grounded CFPO objective.  Per unique
  prompt the auxiliary branch scores the factual and mapped answer candidates
  under the ``pixel_pair_slot_swap`` edit and the ``canonical_resampling_return``
  null edit, builds the mapped-vs-factual log-likelihood margins, and adds
  ``gamma * (L_dir + lambda_null * L_null)`` (directional resampling DiD plus
  null anchor) with saturation-gated weights.  KILLED by preregistration
  amendment ``asymmetric_directional_credit_revision_20260826`` (symmetric DiD
  + null-pushing learned the null shortcut): the spec is retained UNCHANGED
  solely to reproduce historical runs and must not be trained again.
- ``areg_cfpo``: AReG-CFPO, the replacement objective frozen by the same
  amendment (plan7 section 4 R1).  Branches ``(factual, swap, null)`` x
  candidates ``(factual_answer, mapped_answer)`` give 6 auxiliary forwards
  per group; all auxiliary scores/margins are TOTAL nats over the 7-token
  ``full_template_span`` (``s_tilde = 7 * s_mean``).  The directional hinge is
  ``L_swap = w [t_i - s_swap]+`` with the null-detached per-sample target
  ``t_i = max(m_s, sg[s_null] + m_d)`` (sg = stop-gradient: the null branch
  calibrates the bar but receives no directional gradient), and the
  preservation anchor is
  ``L_pres = w [SmoothL1(s_null - s_null_0) + eta * SmoothL1(s_fact - s_fact_0)
  + eta_abs * SmoothL1(logp(y|I_fact) - logp_theta0(y|I_fact))]`` with the
  theta0 reference values read from the train manifest (precomputed offline,
  zero extra forwards).  Total loss
  ``L = L_GRPO + beta * KL + gamma * (L_swap + lambda_pres * L_pres)``.
- ``local_nondirectional_cfpo``: the "CFPO-style local non-directional
  counterfactual baseline" of plan5 section 5.2.  It uses the *same* pixel
  pair-slot swap edit but only reinforces the factual/swap divergence of the
  factual answer score (original image vs swapped image).  It never uses the
  mapped-answer direction and never uses the resampling null, so it cannot be
  claimed as a line-by-line reproduction of the original attention-CFPO.

``global_duality`` is deliberately NOT registered.  Its definition must be
pinned by a preregistration amendment before the formal training matrix; the
smoke matrix (plan5 section 7, E3) does not need it.  Until that amendment
lands, requesting it fails closed like any other unknown objective.

This module is pure Python (no torch/transformers/datasets imports) so the
registry and its validation rules are unit-testable in any environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

#: Auxiliary edit branches the trainer knows how to construct.
AUXILIARY_BRANCHES = ("factual", "swap", "null")

#: Answer candidates the trainer knows how to score.  The option_* names are
#: used only by the Plan9 candidate-distribution trust region.
AUXILIARY_CANDIDATES = (
    "factual_answer",
    "mapped_answer",
    "option_a",
    "option_b",
    "option_c",
    "option_d",
)


@dataclass(frozen=True)
class ObjectiveSpec:
    """Frozen structural configuration of one Stage-4 objective.

    ``branches`` lists the pixel-edit branches executed per unique prompt and
    ``candidates`` the answer candidates scored per branch; the Cartesian
    product fixes the auxiliary forward count per group, which must be
    identical on every rank (ZeRO-3 collective consistency).
    """

    name: str
    auxiliary_enabled: bool
    branches: tuple[str, ...]
    candidates: tuple[str, ...]
    uses_mapped_direction: bool
    uses_resampling_null: bool
    description: str

    def __post_init__(self) -> None:
        unknown_branches = set(self.branches) - set(AUXILIARY_BRANCHES)
        if unknown_branches:
            raise ValueError(f"unknown auxiliary branches: {sorted(unknown_branches)}")
        unknown_candidates = set(self.candidates) - set(AUXILIARY_CANDIDATES)
        if unknown_candidates:
            raise ValueError(f"unknown auxiliary candidates: {sorted(unknown_candidates)}")
        if self.auxiliary_enabled and not self.branches:
            raise ValueError("auxiliary-enabled objectives must declare branches")
        if not self.auxiliary_enabled and (self.branches or self.candidates):
            raise ValueError("auxiliary-disabled objectives cannot declare branches/candidates")

    @property
    def forwards_per_group(self) -> int:
        """Auxiliary model forwards per unique prompt per ``compute_loss``."""

        return len(self.branches) * len(self.candidates)

    def validate_hyperparameters(
        self,
        *,
        gamma: float,
        lambda_null: float,
        allow_gamma_zero: bool = False,
        margin_swap: float | None = None,
        margin_did: float | None = None,
        lambda_pres: float | None = None,
        eta: float | None = None,
        eta_abs: float | None = None,
    ) -> None:
        """Fail-closed hyperparameter policy for this objective.

        ``allow_gamma_zero`` exists solely for the gamma=0 strict-equivalence
        check (``scripts/trainer_equivalence_check.py``), which deliberately
        constructs an auxiliary-enabled objective with gamma=0 to prove the
        ReG trainer reduces to continued GRPO.  Real training entry points
        must keep the default ``allow_gamma_zero=False``.

        AReG-CFPO parameters (``margin_swap`` ``m_s``, ``margin_did`` ``m_d``,
        ``lambda_pres``, ``eta``, ``eta_abs``) are mandatory for
        ``areg_cfpo`` and forbidden for every other objective (a silently
        ignored value is worse than a loud failure).  ``m_s``/``m_d`` are
        margins in TOTAL nats over the 7-token ``full_template_span``
        (amendment ``asymmetric_directional_credit_revision_20260826``,
        score_unit: total_nats) — NOT the mean-per-token units the killed
        regcfpo margins ``margin_dir``/``margin_null`` use.  All five must be
        finite and non-negative; ``m_d`` is additive on top of the detached
        null score and ``lambda_pres``/``eta``/``eta_abs`` are loss weights.
        ``areg_cfpo`` has no null-pushing term, so ``lambda_null`` must be
        exactly 0.0 there.
        """

        numeric: list[tuple[str, float]] = [("gamma", gamma), ("lambda_null", lambda_null)]
        areg_values = {
            "margin_swap": margin_swap,
            "margin_did": margin_did,
            "lambda_pres": lambda_pres,
            "eta": eta,
            "eta_abs": eta_abs,
        }
        if self.name == "areg_cfpo":
            missing = [name for name, value in areg_values.items() if value is None]
            if missing:
                raise ValueError(
                    f"objective {self.name!r} requires explicit AReG hyperparameters "
                    f"{missing} (fail-closed: no silent defaults)"
                )
        else:
            unexpected = [name for name, value in areg_values.items() if value is not None]
            if unexpected:
                raise ValueError(
                    f"objective {self.name!r} has no AReG terms; non-null "
                    f"{unexpected} would silently do nothing"
                )
        numeric.extend(
            (name, value) for name, value in areg_values.items() if value is not None
        )
        for name, value in numeric:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric, got {value!r}")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}")
        gamma = float(gamma)
        lambda_null = float(lambda_null)
        if gamma < 0.0 or lambda_null < 0.0:
            raise ValueError("gamma and lambda_null must be non-negative")
        negatives = [name for name, value in numeric[2:] if float(value) < 0.0]
        if negatives:
            raise ValueError(f"AReG hyperparameters must be non-negative: {negatives}")
        if not self.auxiliary_enabled:
            if gamma != 0.0 or lambda_null != 0.0:
                raise ValueError(
                    f"objective {self.name!r} has no auxiliary term; "
                    "non-zero gamma/lambda_null would silently do nothing"
                )
            return
        if gamma == 0.0 and not allow_gamma_zero:
            raise ValueError(
                f"objective {self.name!r} requires gamma > 0 for training; "
                "gamma=0 is reserved for the strict-equivalence check"
            )
        if self.name == "local_nondirectional_cfpo" and lambda_null != 0.0:
            raise ValueError(
                "local_nondirectional_cfpo has no null-anchor term; "
                "lambda_null must be exactly 0.0"
            )
        if self.name == "areg_cfpo" and lambda_null != 0.0:
            raise ValueError(
                "areg_cfpo has no null-pushing term (the null branch enters only "
                "through the detached target and the preservation anchor); "
                "lambda_null must be exactly 0.0"
            )
        if self.name == "pairaug_refgain_trust" and lambda_null != 0.0:
            raise ValueError(
                "pairaug_refgain_trust uses an explicit candidate-distribution "
                "trust region; lambda_null must be exactly 0.0"
            )


_CONTINUED_GRPO = ObjectiveSpec(
    name="continued_grpo",
    auxiliary_enabled=False,
    branches=(),
    candidates=(),
    uses_mapped_direction=False,
    uses_resampling_null=False,
    description=(
        "Vendored VLMGRPOTrainer GRPO on the same Stage-4 data; "
        "no counterfactual auxiliary term."
    ),
)

_REGCFPO = ObjectiveSpec(
    name="regcfpo",
    auxiliary_enabled=True,
    branches=("swap", "null"),
    candidates=("factual_answer", "mapped_answer"),
    uses_mapped_direction=True,
    uses_resampling_null=True,
    description=(
        "ReG-CFPO: gamma * (L_dir + lambda_null * L_null) with directional "
        "resampling DiD margins and the saturation-aware gate.  KILLED by "
        "amendment asymmetric_directional_credit_revision_20260826; retained "
        "unchanged for historical-run reproduction only."
    ),
)

_AREG_CFPO = ObjectiveSpec(
    name="areg_cfpo",
    auxiliary_enabled=True,
    branches=("factual", "swap", "null"),
    candidates=("factual_answer", "mapped_answer"),
    uses_mapped_direction=True,
    uses_resampling_null=True,
    description=(
        "AReG-CFPO (amendment asymmetric_directional_credit_revision_20260826): "
        "gamma * (L_swap + lambda_pres * L_pres) with the null-detached target "
        "t_i = max(m_s, sg[s_null] + m_d), theta0-anchored preservation, and "
        "total-nats scores; 6 auxiliary forwards per group."
    ),
)

_PAIRAUG_REFGAIN_TRUST = ObjectiveSpec(
    name="pairaug_refgain_trust",
    auxiliary_enabled=True,
    branches=("factual", "swap", "null"),
    candidates=("option_a", "option_b", "option_c", "option_d"),
    uses_mapped_direction=True,
    uses_resampling_null=True,
    description=(
        "Plan9 true joint objective: current-input paired source/swap GRPO, "
        "dense frozen-reference swap gain, and slack source/null candidate-"
        "distribution trust regions in the same optimizer update."
    ),
)

_LOCAL_NONDIRECTIONAL = ObjectiveSpec(
    name="local_nondirectional_cfpo",
    auxiliary_enabled=True,
    branches=("factual", "swap"),
    candidates=("factual_answer",),
    uses_mapped_direction=False,
    uses_resampling_null=False,
    description=(
        "CFPO-style local non-directional baseline (plan5 section 5.2): same "
        "pixel pair-slot swap edit, but only the factual/swap divergence of "
        "the factual answer score is reinforced; no mapped direction, no "
        "resampling DiD."
    ),
)

#: The registry is the single authority on runnable objectives.  Anything not
#: listed here (including ``global_duality``, pending an amendment) is
#: rejected at parse time.
OBJECTIVE_REGISTRY: Mapping[str, ObjectiveSpec] = {
    spec.name: spec
    for spec in (
        _CONTINUED_GRPO,
        _REGCFPO,
        _LOCAL_NONDIRECTIONAL,
        _AREG_CFPO,
        _PAIRAUG_REFGAIN_TRUST,
    )
}


def registered_objectives() -> tuple[str, ...]:
    """Return the registered objective names in stable order."""

    return tuple(OBJECTIVE_REGISTRY)


def get_objective_spec(name: str) -> ObjectiveSpec:
    """Return the spec for ``name``; fail closed on anything unregistered."""

    if not isinstance(name, str):
        raise ValueError(f"objective name must be a string, got {name!r}")
    try:
        return OBJECTIVE_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unregistered objective {name!r}; registered objectives: "
            f"{list(OBJECTIVE_REGISTRY)}. `global_duality` is intentionally "
            "absent: its definition awaits a preregistration amendment before "
            "the formal matrix (smoke E3 does not use it)."
        ) from None


__all__ = [
    "AUXILIARY_BRANCHES",
    "AUXILIARY_CANDIDATES",
    "OBJECTIVE_REGISTRY",
    "ObjectiveSpec",
    "get_objective_spec",
    "registered_objectives",
]
