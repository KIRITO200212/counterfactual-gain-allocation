"""Training scaffold for ReG-CFPO Stage-4 post-training.

Loss-level core, objective registry, and global-gate glue are pure and
unit-testable on CPU.  ``regcfpo.training.trainer`` (the vendored-trainer
subclass) is intentionally NOT re-exported here: it imports the vendored
``open_r1`` package, which only exists in the training environment
(``.conda/regcfpo-train``).  Smoke runs remain gated by
``TRAINING_SMOKE_AUTHORIZED`` (currently false).
"""

from regcfpo.training.auxiliary import (
    AREG_SMOOTH_L1_BETA,
    EPSILON,
    GATE_CREDIT_MODES,
    LEGACY_GATE_CREDIT_MODE,
    MONOTONIC_GATE_CREDIT_MODE,
    AReGConfig,
    GateConfig,
    areg_cfpo_auxiliary_loss,
    areg_did_arm_mask,
    areg_preservation_loss,
    areg_swap_loss,
    areg_swap_target,
    build_areg_reference_tensors,
    closed_form_success_credit,
    directional_loss,
    normalize_gate_weights,
    monotonic_success_credit,
    null_anchor_loss,
    regcfpo_auxiliary_loss,
    saturation_gate_flags,
)
from regcfpo.training.gate import (
    AuxiliaryExecutionPlan,
    GateStrataMetrics,
    GlobalGatePayload,
    collapse_generation_groups,
    compute_global_gate_payload,
    group_accuracy_rewards,
    local_group_slice,
    plan_auxiliary_execution,
)
from regcfpo.training.objectives import (
    OBJECTIVE_REGISTRY,
    ObjectiveSpec,
    get_objective_spec,
    registered_objectives,
)
from regcfpo.training.rewards import official_stage3_na_reward
from regcfpo.training.sampling import CycleOrderSampler

__all__ = [
    "AREG_SMOOTH_L1_BETA",
    "AReGConfig",
    "AuxiliaryExecutionPlan",
    "CycleOrderSampler",
    "EPSILON",
    "GATE_CREDIT_MODES",
    "GateConfig",
    "GateStrataMetrics",
    "GlobalGatePayload",
    "LEGACY_GATE_CREDIT_MODE",
    "MONOTONIC_GATE_CREDIT_MODE",
    "OBJECTIVE_REGISTRY",
    "ObjectiveSpec",
    "areg_cfpo_auxiliary_loss",
    "areg_did_arm_mask",
    "areg_preservation_loss",
    "areg_swap_loss",
    "areg_swap_target",
    "build_areg_reference_tensors",
    "closed_form_success_credit",
    "collapse_generation_groups",
    "compute_global_gate_payload",
    "directional_loss",
    "get_objective_spec",
    "group_accuracy_rewards",
    "local_group_slice",
    "normalize_gate_weights",
    "monotonic_success_credit",
    "null_anchor_loss",
    "official_stage3_na_reward",
    "plan_auxiliary_execution",
    "regcfpo_auxiliary_loss",
    "registered_objectives",
    "saturation_gate_flags",
]
