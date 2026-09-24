"""ReG-CFPO trainer subclass over the vendored ``VLMGRPOTrainer``.

IMPORTANT: this module is importable only in the training environment
(``.conda/regcfpo-train``), because the vendored ``open_r1`` package is
deliberately absent from the read-only inference environment.  Pure gate
logic lives in ``regcfpo.training.gate`` and the objective registry in
``regcfpo.training.objectives`` so both stay unit-testable on CPU anywhere.

Integration contract (plan5 sections 5.2-5.3, auxiliary.py input contract):

- The gate is computed in ``_generate_and_score_completions`` AFTER the
  vendor's global reward gather (grpo_trainer.py:750), from the
  ACCURACY-ONLY reward component — never from the multi-component reward sum
  at grpo_trainer.py:753.  The accuracy column is captured by wrapping the
  accuracy reward function; one extra ``accelerator.gather`` (same collective
  on every rank, issued right after ``super()`` returns) reconstructs the
  gathered accuracy vector in exactly the vendor's rank-major order.
- Credit normalization/clipping run on the GLOBAL batch
  (``compute_global_gate_payload``).  ``compute_loss`` therefore calls
  ``regcfpo_auxiliary_loss`` with ``normalize_weights=False``: re-normalizing
  the already globally normalized weights would double-count the batch
  statistics.  This is the documented global-batch contract of the trainer.
- Forward-count consistency: the vendored generation groups prompts across
  the WHOLE global batch, and this trainer scores EVERY global group on
  every rank (redundant-but-identical forwards; DDP/ZeRO gradient averaging
  leaves the result unchanged because all ranks compute the same auxiliary
  loss).  The current runs are single-process — per-device batch 4 with
  G=4 (one prompt group per optimizer step), no DeepSpeed
  (``--no-deepspeed``) — so "every rank" is trivially one process; with
  multiple ranks a group can straddle ranks (per-rank batch smaller than
  G), which is exactly why the redundant scoring is kept rather than
  slicing groups across ranks.  The
  only skip decision is ``global_active == 0``, computed from the gathered
  global batch and hence identical on every rank — all ranks skip together.
  Locally zero-weight groups are never skipped; their zero weights mask them.
  ``aux/forward_count`` is logged per ``compute_loss`` call so the
  equivalence check can verify counts match across ranks.  The redundant
  scoring costs a factor ``num_processes`` in auxiliary compute; a sharded
  variant needs a differentiable cross-rank score gather and is deliberately
  out of scope (``gate.local_group_slice`` is reserved for it).
- Auxiliary scores carry gradients and are therefore NEVER written into
  ``self._buffered_inputs``: the buffer only carries detached gate weights
  and JSON-able group metadata (properties of the generated rollouts).  When
  ``num_iterations > 1`` reuses the buffer, ``compute_loss`` recomputes the
  auxiliary scores against the CURRENT policy parameters on every inner
  iteration — required for correctness of the PPO-style inner loop.
- The auxiliary branch calls the processor directly and passes the edited
  PIL images through the full ``processor -> ViT -> LLM`` path, mirroring
  grpo_trainer.py:639-645.  It does NOT use
  ``vlm_module.prepare_model_inputs``: that helper is dead code in the
  vendored rollout path and contains the ``imgaes=images`` typo
  (vlm_modules/qwen_module.py:53) which would silently drop the images.  The
  vendor typo is deliberately left unpatched (no vendor modifications).
- Candidate scoring reuses ``qwen_adapter.candidate_span_mean_log_probs`` with
  gradients enabled, ``use_cache=False`` (enforced via
  ``validate_teacher_forced_call``), one forward per (branch, candidate,
  group), executed strictly serially to bound peak memory.  Pixel edits are
  non-differentiable CPU PIL ops; gradients flow from the edited image
  through the frozen forward path into the policy parameters.

AReG-CFPO deltas (amendment ``asymmetric_directional_credit_revision_20260826``):

- SCORE UNITS: the areg_cfpo objective scores candidates in TOTAL nats
  (``candidate_span_log_probs`` sums over the 7-token ``full_template_span``,
  plan7 section 4.1); regcfpo/local_nondirectional_cfpo keep the frozen
  MEAN-per-token convention above.  The unit switch lives entirely inside
  the areg assembly/collection path.
- Replay groups (merged manifest ``row_kind == "replay"``) carry aux weight
  exactly zero via the gate mask (``compute_global_gate_payload``
  ``group_aux_mask``).  Under the registered one-prompt-group-per-step
  configuration a replay step has ``global_active == False`` and ALL ranks
  skip the auxiliary forwards together — the only collective-consistent
  skip, identical to the historical all-inactive case; the logged schema is
  unchanged (``aux/forward_count == 0``, ``aux_gate/*`` still emitted).  If a
  future configuration packs multiple prompt groups per optimizer step,
  replay groups MUST still be forwarded with weight zero (never skipped
  individually): per-rank forward counts derive only from the group count,
  so weight-0-never-skip keeps every collective aligned across ranks.
- First-class monitors (amendment ``first_class_monitors``): the areg
  assembly logs the six absolute answer log-probs (orig/mapped x
  factual/swap/null), the ``t_i`` DiD-arm dominance fraction, and
  p(<think>) read at the prompt-final logit position of the factual-branch
  forward; format reward and completion_length are already vendored logs
  (``rewards/format_reward``, ``completion_length``).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence, Union

import torch
from torch import Tensor

from accelerate.utils import gather_object
from open_r1.trainer.grpo_trainer import VLMGRPOTrainer
from PIL import Image

from regcfpo.operators.pixel_ops import (
    canonical_noop,
    canonical_resampling_return,
    pixel_pair_slot_swap,
)
from regcfpo.qwen_adapter import (
    candidate_span_log_probs,
    candidate_span_mean_log_probs,
    validate_teacher_forced_call,
)
from regcfpo.training.auxiliary import (
    AReGConfig,
    GateConfig,
    LEGACY_PREFIX_CREDIT_SCOPE,
    PLAN10_PREFIX_CREDIT_SCOPE,
    PairAugReferenceTrustConfig,
    areg_cfpo_auxiliary_loss,
    areg_did_arm_mask,
    areg_preservation_loss,
    areg_swap_loss,
    build_areg_reference_tensors,
    build_joint_reference_tensors,
    directional_loss,
    null_anchor_loss,
    pairaug_reference_trust_auxiliary_loss,
    regcfpo_auxiliary_loss,
)
from regcfpo.training.conversation import sanitize_conversation_for_template
from regcfpo.training.gate import (
    collapse_generation_groups,
    compute_global_gate_payload,
    plan_auxiliary_execution,
)
from regcfpo.training.objectives import ObjectiveSpec, get_objective_spec
from regcfpo.training.prefix_credit import (
    AnswerTokenStyle,
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
from regcfpo.training.sampling import CycleOrderSampler

#: Row fields copied into the per-group auxiliary metadata payload.  These
#: must match the preserved dataset columns of ``scripts/train_stage4.py``.
#: The last five implement the AReG merged-manifest contract (amendment
#: asymmetric_directional_credit_revision_20260826): ``row_kind`` /
#: ``cycle_position`` on every row; the three theta0 reference floats (total
#: nats) on directional rows, NaN on replay rows.
AUX_GROUP_METADATA_FIELDS = (
    "sample_id",
    "question_id",
    "scene_id",
    "relation_class",
    "relation_family",
    "answer_letter",
    "mapped_answer_letter",
    "gt_box_a",
    "gt_box_b",
    "operator_main",
    "operator_null",
    "image_path",
    "prompt",
    "row_kind",
    "cycle_position",
    "areg_s_fact_0",
    "areg_s_null_0",
    "areg_logp_fact_orig_0",
    "pairaug_branch",
    "pairaug_parent_sample_id",
    "pairaug_source_image_path",
    "pairaug_original_letter",
    "pairaug_mapped_letter",
    "joint_s_swap_0",
    "joint_source_logps_0",
    "joint_null_logps_0",
)

#: Candidate text convention of the frozen full-template teacher-forced arm
#: (answer_proxy_gate.py arm 3).
CANDIDATE_TEXT_TEMPLATE = "<answer> {letter} </answer>"
OPTION_CANDIDATE_LETTERS = {
    "option_a": "A",
    "option_b": "B",
    "option_c": "C",
    "option_d": "D",
}


class AuxiliaryOperatorError(RuntimeError):
    """Raised when a frozen pixel operator rejects an active training row."""


def _load_rgb_image(path: str) -> Image.Image:
    """Default image loader: one RGB PIL image per path (canonical chain)."""

    with Image.open(path) as image:
        return image.convert("RGB")


def _candidate_suffix_mask(base_input_ids: Tensor, full_input_ids: Tensor) -> Tensor:
    """Mark the non-empty candidate suffix after an exact tokenized prompt.

    Mirrors ``scripts/frozen_operator_diagnostic.py:candidate_suffix_mask``;
    kept local because ``src`` modules must not import from ``scripts``.
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


class _RewardCapture:
    """Wrap a scalar reward function to capture its per-row local outputs.

    The vendored trainer gathers ``rewards_per_func`` internally but never
    exposes the gathered accuracy-only column, which the gate needs.  The
    wrapper stashes the local outputs so the trainer can gather exactly the
    same values with one extra collective.  ``__name__`` is forwarded because
    the vendored metric logging uses it (grpo_trainer.py:781).
    """

    def __init__(self, func: Callable[..., Sequence[float]]) -> None:
        self._func = func
        self.__name__ = getattr(func, "__name__", "captured_reward")
        self.captured: list[float] | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> Sequence[float]:
        rewards = self._func(*args, **kwargs)
        self.captured = [float(value) for value in rewards]
        return rewards


class ReGCFPOTrainer(VLMGRPOTrainer):
    """``VLMGRPOTrainer`` plus the Stage-4 counterfactual auxiliary objective.

    Args beyond the vendored trainer:

    - ``objective_spec``: an :class:`ObjectiveSpec` obtained from
      ``regcfpo.training.objectives.get_objective_spec`` (fail-closed:
      hand-rolled or unregistered specs are rejected).
    - ``gamma`` / ``lambda_null``: auxiliary loss scale.  ``gamma=0`` is
      accepted here (exact continued-GRPO identity, used by the equivalence
      check); entry points enforce ``gamma > 0`` for real auxiliary runs.
    - ``gate_config``: frozen gate hyperparameters (margins, clip, epsilon).
    - ``areg_config``: frozen AReG-CFPO hyperparameters (``AReGConfig``,
      total-nats units).  Mandatory for ``objective_spec.name ==
      "areg_cfpo"`` and rejected for every other objective.
    - ``accuracy_reward_index``: index of the accuracy reward inside
      ``reward_funcs``; that function is wrapped for gate capture.
    - ``format_reward_index``: index of the format reward.  It is captured
      only when on-policy prefix credit is enabled, because the optimized
      reward must reproduce both vendored base-reward components exactly.
    - ``image_loader``: path -> RGB PIL image (injectable for tests).
    """

    def __init__(
        self,
        model: Union[str, Any],
        reward_funcs: Any,
        *args: Any,
        objective_spec: ObjectiveSpec,
        gamma: float = 0.0,
        lambda_null: float = 0.0,
        gate_config: GateConfig | None = None,
        areg_config: AReGConfig | None = None,
        joint_config: PairAugReferenceTrustConfig | None = None,
        accuracy_reward_index: int = 0,
        format_reward_index: int = 1,
        image_loader: Callable[[str], Image.Image] | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(objective_spec, ObjectiveSpec):
            raise ValueError("objective_spec must be an objectives.ObjectiveSpec")
        registered = get_objective_spec(objective_spec.name)  # fail-closed
        if registered != objective_spec:
            raise ValueError(
                "objective_spec fields differ from the registry entry for "
                f"{objective_spec.name!r}; construct it via get_objective_spec"
            )
        if objective_spec.name == "areg_cfpo":
            if not isinstance(areg_config, AReGConfig):
                raise ValueError("areg_cfpo requires an auxiliary.AReGConfig instance")
            if joint_config is not None:
                raise ValueError("areg_cfpo does not accept a Plan9 joint config")
            objective_spec.validate_hyperparameters(
                gamma=gamma,
                lambda_null=lambda_null,
                allow_gamma_zero=True,
                margin_swap=areg_config.margin_swap,
                margin_did=areg_config.margin_did,
                lambda_pres=areg_config.lambda_pres,
                eta=areg_config.eta,
                eta_abs=areg_config.eta_abs,
            )
        elif objective_spec.name == "pairaug_refgain_trust":
            if areg_config is not None:
                raise ValueError("pairaug_refgain_trust does not accept an AReG config")
            if not isinstance(joint_config, PairAugReferenceTrustConfig):
                raise ValueError(
                    "pairaug_refgain_trust requires a PairAugReferenceTrustConfig"
                )
            objective_spec.validate_hyperparameters(
                gamma=gamma, lambda_null=lambda_null, allow_gamma_zero=True
            )
        else:
            if areg_config is not None:
                raise ValueError(
                    f"objective {objective_spec.name!r} has no AReG terms; "
                    "areg_config would silently do nothing"
                )
            if joint_config is not None:
                raise ValueError(
                    f"objective {objective_spec.name!r} has no Plan9 joint terms; "
                    "joint_config would silently do nothing"
                )
            objective_spec.validate_hyperparameters(
                gamma=gamma, lambda_null=lambda_null, allow_gamma_zero=True
            )
        self.objective_spec = objective_spec
        self.gamma = float(gamma)
        self.lambda_null = float(lambda_null)
        self.gate_config = gate_config or GateConfig()
        self.areg_config = areg_config
        self.joint_config = joint_config
        self._image_loader = image_loader or _load_rgb_image
        self._think_first_token_id: int | None = None
        self._prefix_answer_styles: tuple[AnswerTokenStyle, ...] | None = None
        self._prefix_credit_enabled = bool(
            joint_config is not None and joint_config.prefix_credit_weight > 0.0
        )

        reward_list = list(reward_funcs) if isinstance(reward_funcs, list) else [reward_funcs]
        if not 0 <= accuracy_reward_index < len(reward_list):
            raise ValueError(
                f"accuracy_reward_index {accuracy_reward_index} out of range for "
                f"{len(reward_list)} reward functions"
            )
        self._accuracy_capture = _RewardCapture(reward_list[accuracy_reward_index])
        reward_list[accuracy_reward_index] = self._accuracy_capture
        self._format_capture: _RewardCapture | None = None
        if self._prefix_credit_enabled:
            if not 0 <= format_reward_index < len(reward_list):
                raise ValueError(
                    f"format_reward_index {format_reward_index} out of range for "
                    f"{len(reward_list)} reward functions"
                )
            if format_reward_index == accuracy_reward_index:
                raise ValueError("accuracy and format reward indices must differ")
            self._format_capture = _RewardCapture(reward_list[format_reward_index])
            reward_list[format_reward_index] = self._format_capture

        super().__init__(model, reward_list, *args, **kwargs)

        if self._prefix_credit_enabled and self.num_iterations != 1:
            raise ValueError(
                "on-policy prefix credit requires num_iterations=1; recomputing "
                "current-policy credit for stale multi-iteration rollouts is invalid"
            )

        # Per-generation-cycle caches of deterministic prompt renderings and
        # pixel edits, keyed by sample_id.  Cleared on every fresh generation;
        # they hold only the current global groups' data.
        self._aux_edit_cache: dict[str, dict[str, Image.Image]] = {}
        self._aux_prompt_cache: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Generation-time gate (after the vendor's global reward gather)       #
    # ------------------------------------------------------------------ #

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]], model: Any
    ) -> dict[str, Union[torch.Tensor, Any]]:
        spec = self.objective_spec
        if spec.auxiliary_enabled:
            self._accuracy_capture.captured = None
            if self._format_capture is not None:
                self._format_capture.captured = None
            self._aux_edit_cache.clear()
            self._aux_prompt_cache.clear()
        result = super()._generate_and_score_completions(inputs, model)
        if not spec.auxiliary_enabled:
            return result

        captured = self._accuracy_capture.captured
        if captured is None:
            raise RuntimeError(
                "the accuracy reward was not invoked during generation; "
                "the gate cannot be built"
            )
        local_accuracy = torch.as_tensor(
            captured, dtype=torch.float32, device=self.accelerator.device
        )
        if local_accuracy.shape[0] != len(inputs):
            raise RuntimeError(
                f"accuracy reward count {local_accuracy.shape[0]} does not match "
                f"the batch row count {len(inputs)}"
            )
        if not bool(torch.isfinite(local_accuracy).all()):
            raise FloatingPointError("accuracy reward contains NaN or Inf")
        # Same collective on every rank, right after the vendor's gathers.
        # Prefix credit needs both fixed base-reward columns; gather them in
        # one matrix so the collective count stays identical across ranks.
        format_global: Tensor | None = None
        if self._format_capture is not None:
            captured_format = self._format_capture.captured
            if captured_format is None:
                raise RuntimeError(
                    "the format reward was not invoked during generation; "
                    "prefix-credit base rewards cannot be reconstructed"
                )
            local_format = torch.as_tensor(
                captured_format,
                dtype=torch.float32,
                device=self.accelerator.device,
            )
            if local_format.shape != local_accuracy.shape:
                raise RuntimeError(
                    f"format reward count {local_format.shape[0]} does not match "
                    f"the batch row count {len(inputs)}"
                )
            if not bool(torch.isfinite(local_format).all()):
                raise FloatingPointError("format reward contains NaN or Inf")
            gathered_base = self.accelerator.gather(
                torch.stack((local_accuracy, local_format), dim=-1)
            )
            accuracy_global = gathered_base[:, 0]
            format_global = gathered_base[:, 1]
        else:
            accuracy_global = self.accelerator.gather(local_accuracy)

        # Gather slim per-row metadata (rank-major order, matching the
        # gathered rewards) and collapse to one row per unique GLOBAL prompt.
        slim_rows = [self._slim_row(row) for row in inputs]
        global_rows = gather_object(slim_rows)
        group_rows = collapse_generation_groups(global_rows, self.num_generations)

        # AReG merged-manifest contract (amendment
        # asymmetric_directional_credit_revision_20260826): replay groups
        # carry aux weight exactly zero.  The mask derives from the gathered
        # global rows and is therefore identical on every rank; it scales the
        # credits BEFORE normalization inside compute_global_gate_payload.
        # Old manifests default to row_kind="directional" in
        # train_stage4.make_conversation, so this is a no-op for them.
        row_kinds = [str(row["row_kind"]) for row in group_rows]
        unknown_kinds = sorted({kind for kind in row_kinds if kind not in ("directional", "replay")})
        if unknown_kinds:
            raise ValueError(f"unknown row_kind values {unknown_kinds}; the gate cannot route them")
        if spec.name == "pairaug_refgain_trust":
            branches = [str(row.get("pairaug_branch", "")) for row in group_rows]
            unknown_branches = sorted(set(branches) - {"source", "swap"})
            if unknown_branches:
                raise ValueError(
                    "pairaug_refgain_trust requires pairaug_branch in "
                    f"{{'source','swap'}}, got {unknown_branches}"
                )
            # The dense reference/trust auxiliary executes only on the swap
            # microbatch.  Source and swap rows are adjacent and accumulated
            # into the same optimizer update by the manifest/GA=2 contract.
            aux_mask_values = [1.0 if branch == "swap" else 0.0 for branch in branches]
        else:
            aux_mask_values = [1.0 if kind == "directional" else 0.0 for kind in row_kinds]
        aux_mask = torch.tensor(
            aux_mask_values,
            dtype=torch.float32,
            device=self.accelerator.device,
        )
        payload = compute_global_gate_payload(
            accuracy_global,
            self.num_generations,
            self.gate_config,
            group_aux_mask=aux_mask,
        )
        if len(group_rows) != payload.num_groups:
            raise RuntimeError(
                f"collapsed {len(group_rows)} unique groups but the gathered "
                f"rewards imply {payload.num_groups}; generation layout violated"
            )

        if spec.name == "pairaug_refgain_trust":
            # Dense gain is the escape hatch for PairAug groups whose rollout
            # accuracy has zero within-group variance.  Keep the observed k/G
            # credit in gate metrics, but do not suppress the auxiliary when
            # k=0: every swap group receives unit auxiliary weight.
            execution_weights = aux_mask.detach()
            execution_active = bool((execution_weights > 0.0).any().item())
            self._metrics["aux_gate/joint_dense_weight_mean"].append(
                execution_weights.mean().item()
            )
        else:
            execution_weights = payload.weights
            execution_active = payload.global_active
        result["regcfpo_gate_weights"] = execution_weights  # detached, global
        result["regcfpo_aux_groups"] = [dict(row) for row in group_rows]
        result["regcfpo_global_active"] = execution_active
        if self._prefix_credit_enabled:
            if format_global is None:  # pragma: no cover - capture enforced above
                raise RuntimeError("prefix credit lacks gathered format rewards")
            original_indices: list[int] = []
            mapped_indices: list[int] = []
            for row in slim_rows:
                original = str(row["pairaug_original_letter"]).upper()
                mapped = str(row["pairaug_mapped_letter"]).upper()
                if original not in "ABCD" or mapped not in "ABCD" or original == mapped:
                    raise ValueError(
                        "prefix credit requires distinct A-D original/mapped letters; "
                        f"got {original!r}/{mapped!r} for {row.get('sample_id')!r}"
                    )
                original_indices.append(ord(original) - ord("A"))
                mapped_indices.append(ord(mapped) - ord("A"))
            global_swap_mask = torch.tensor(
                [str(row.get("pairaug_branch", "")) == "swap" for row in global_rows],
                dtype=torch.bool,
                device=self.accelerator.device,
            )
            grouped_swap_mask = global_swap_mask.view(-1, self.num_generations)
            if not bool((grouped_swap_mask == grouped_swap_mask[:, :1]).all()):
                raise ValueError(
                    "prefix-credit reward routing requires each generation group "
                    "to contain only source rows or only swap rows"
                )
            result["regcfpo_prefix_accuracy_global"] = accuracy_global.detach()
            result["regcfpo_prefix_format_global"] = format_global.detach()
            result["regcfpo_prefix_swap_mask_global"] = global_swap_mask
            result["regcfpo_prefix_original_indices"] = torch.tensor(
                original_indices, dtype=torch.long, device=self.accelerator.device
            )
            result["regcfpo_prefix_mapped_indices"] = torch.tensor(
                mapped_indices, dtype=torch.long, device=self.accelerator.device
            )

        self._metrics["aux_gate/sat_fraction"].append(payload.strata.sat_fraction)
        self._metrics["aux_gate/mix_fraction"].append(payload.strata.mix_fraction)
        self._metrics["aux_gate/all_wrong_fraction"].append(payload.strata.all_wrong_fraction)
        self._metrics["aux_gate/w_tilde_mean"].append(payload.w_tilde_mean)
        self._metrics["aux_gate/clip_hit_fraction"].append(payload.clip_hit_fraction)
        self._metrics["aux_gate/active_groups"].append(float(payload.active_count))
        self._metrics["aux_gate/replay_groups"].append(
            float(sum(kind == "replay" for kind in row_kinds))
        )
        return result

    @staticmethod
    def _slim_row(row: Mapping[str, Any]) -> dict[str, Any]:
        missing = [field for field in AUX_GROUP_METADATA_FIELDS if field not in row]
        if missing:
            raise ValueError(
                f"batch row is missing preserved metadata columns: {missing}; "
                "the entry point must keep them through dataset.map"
            )
        return {field: row[field] for field in AUX_GROUP_METADATA_FIELDS}

    # ------------------------------------------------------------------ #
    # Loss-time auxiliary objective                                        #
    # ------------------------------------------------------------------ #

    def compute_loss(
        self,
        model: Any,
        inputs: Any,
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Tensor:
        """GRPO loss plus the objective's auxiliary term.

        The auxiliary scores are recomputed against the current policy on
        EVERY call (including ``num_iterations > 1`` inner iterations that
        reuse buffered rollouts); only detached gate weights and metadata are
        buffered.  Gradient-carrying auxiliary tensors never enter
        ``_buffered_inputs``.
        """

        if self._prefix_credit_enabled:
            loss = self._compute_policy_loss_with_prefix_credit(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        else:
            loss = super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        spec = self.objective_spec
        if not spec.auxiliary_enabled:
            return loss
        buffered = self._buffered_inputs[(self._step - 1) % self.args.gradient_accumulation_steps]
        auxiliary = self._compute_auxiliary_objective(model, buffered)
        if auxiliary is None:
            # All ranks skip together (global active count is zero).
            return loss
        return loss + auxiliary

    def _compute_policy_loss_with_prefix_credit(
        self,
        model: Any,
        inputs: Any,
        *,
        return_outputs: bool,
        num_items_in_batch: Any,
    ) -> Tensor:
        """Vendored GRPO policy loss with swap rewards replaced in-place.

        This intentionally mirrors the pinned vendor's ``compute_loss``.  The
        only scientific change is the swap group's reward/advantage: after a
        single current-policy forward, the causal mapped-vs-original margin
        at the answer boundary is detached, converted to continuous credit,
        added to accuracy+format, globally gathered, and group-standardized.
        The same forward supplies sampled-token log probabilities, so there is
        no second policy graph and no answer retokenization.
        """

        del num_items_in_batch  # vendored loss ignores this argument
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        if self.state.global_step % self.num_iterations == 0:
            buffered = self._generate_and_score_completions(inputs, model)
            self._buffered_inputs[
                self._step % self.args.gradient_accumulation_steps
            ] = buffered
        else:  # pragma: no cover - prefix mode rejects num_iterations != 1
            buffered = self._buffered_inputs[
                self._step % self.args.gradient_accumulation_steps
            ]
        self._step += 1

        prompt_ids = buffered["prompt_ids"]
        prompt_mask = buffered["prompt_mask"]
        completion_ids = buffered["completion_ids"]
        completion_mask = buffered["completion_mask"]
        multimodal_inputs = buffered["multimodal_inputs"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        cfg = self.joint_config
        if cfg is None:  # pragma: no cover - init contract
            raise RuntimeError("prefix-credit loss requires a joint config")
        prefix_advantages: Tensor | None = None
        reasoning_mask: Tensor | None = None
        bridge_loss: Tensor | None = None
        bridge_weights: Tensor | None = None
        local_active_rows: Tensor | None = None
        swap_mask_global = buffered["regcfpo_prefix_swap_mask_global"].bool()
        if bool(swap_mask_global.any()):
            (
                per_token_logps,
                option_logits,
                local_margins,
                local_valid,
                reasoning_mask,
            ) = (
                self._get_per_token_logps_and_prefix_margins(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    completion_ids=completion_ids,
                    completion_mask=completion_mask,
                    prompt_length=prompt_ids.size(1),
                    original_indices=buffered["regcfpo_prefix_original_indices"],
                    mapped_indices=buffered["regcfpo_prefix_mapped_indices"],
                    multimodal_inputs=multimodal_inputs,
                )
            )
            global_margins = self.accelerator.gather(local_margins.detach().float())
            global_valid = self.accelerator.gather(local_valid.detach().float()).bool()
            if cfg.prefix_credit_scope == LEGACY_PREFIX_CREDIT_SCOPE:
                advantages = self._prefix_credit_advantages(
                    buffered,
                    global_margins=global_margins,
                    global_valid=global_valid,
                )
            elif cfg.prefix_credit_scope == PLAN10_PREFIX_CREDIT_SCOPE:
                advantages = buffered["advantages"]
                (
                    prefix_advantages,
                    local_active_rows,
                    bridge_weights,
                ) = self._plan10_prefix_payload(
                    buffered,
                    global_margins=global_margins,
                    global_valid=global_valid,
                )
                bridge_loss, bridge_per_row = four_way_answer_bridge_loss(
                    option_logits,
                    buffered["regcfpo_prefix_mapped_indices"],
                    bridge_weights,
                    local_valid,
                )
                active = bridge_weights > 0.0
                option_probs = option_logits.detach().float().softmax(dim=-1)
                rows = torch.arange(option_logits.shape[0], device=option_logits.device)
                mapped_indices = buffered["regcfpo_prefix_mapped_indices"].to(
                    device=option_logits.device, dtype=torch.long
                )
                original_indices = buffered["regcfpo_prefix_original_indices"].to(
                    device=option_logits.device, dtype=torch.long
                )
                if bool(active.any()):
                    self._metrics["bridge/loss"].append(bridge_loss.detach().item())
                    self._metrics["bridge/cross_entropy_mean"].append(
                        bridge_per_row.detach()[active].mean().item()
                    )
                    self._metrics["bridge/mapped_probability_mean"].append(
                        option_probs[rows, mapped_indices][active].mean().item()
                    )
                    self._metrics["bridge/original_probability_mean"].append(
                        option_probs[rows, original_indices][active].mean().item()
                    )
                    self._metrics["bridge/mapped_top1_fraction"].append(
                        (option_probs.argmax(dim=-1)[active] == mapped_indices[active])
                        .float()
                        .mean()
                        .item()
                    )
                else:
                    for name in (
                        "bridge/loss",
                        "bridge/cross_entropy_mean",
                        "bridge/mapped_probability_mean",
                        "bridge/original_probability_mean",
                        "bridge/mapped_top1_fraction",
                    ):
                        self._metrics[name].append(0.0)
            else:  # pragma: no cover - config validates this
                raise RuntimeError(f"unsupported prefix scope {cfg.prefix_credit_scope!r}")
        else:
            # Source PairAug rows keep the byte-for-byte vendored reward and
            # advantage.  They still share the optimizer update with the next
            # swap microbatch through the frozen GA=2 cycle contract.
            per_token_logps = self._get_per_token_logps(
                model,
                input_ids,
                attention_mask,
                **multimodal_inputs,
            )
            advantages = buffered["advantages"]

        # Drop prompt positions (-1 for the causal shift), as in the vendor.
        per_token_logps = per_token_logps[:, prompt_ids.size(1) - 1 :]
        old_per_token_logps = per_token_logps.detach()

        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(
            coef_1,
            1 - self.epsilon_low,
            1 + self.epsilon_high,
        )
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if self.beta > 0:
            ref_per_token_logps = buffered["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )
            per_token_loss = per_token_loss + self.beta * per_token_kl
            mean_kl = (
                (per_token_kl * completion_mask).sum(dim=1)
                / completion_mask.sum(dim=1)
            ).mean()
            self._metrics["kl"].append(
                self.accelerator.gather_for_metrics(mean_kl).mean().item()
            )

        loss = (
            (per_token_loss * completion_mask).sum(dim=1)
            / completion_mask.sum(dim=1)
        ).mean()
        if prefix_advantages is not None:
            if reasoning_mask is None or local_active_rows is None or bridge_loss is None:
                raise RuntimeError("Plan10 prefix tensors were not constructed")
            active_reasoning_mask = (
                reasoning_mask.bool() & local_active_rows.bool().unsqueeze(1)
            )
            prefix_loss1 = coef_1 * prefix_advantages.unsqueeze(1)
            prefix_loss2 = coef_2 * prefix_advantages.unsqueeze(1)
            prefix_per_token_loss = -torch.min(prefix_loss1, prefix_loss2)
            reasoning_counts = active_reasoning_mask.sum(dim=1)
            prefix_per_row = (
                (prefix_per_token_loss * active_reasoning_mask).sum(dim=1)
                / reasoning_counts.clamp_min(1)
            )
            prefix_loss = prefix_per_row.mean()
            loss = (
                loss
                + float(cfg.prefix_credit_weight) * prefix_loss
                + float(cfg.answer_bridge_weight) * bridge_loss
            )
            prefix_clip = (prefix_loss1 < prefix_loss2).float()
            prefix_denominator = active_reasoning_mask.sum().clamp_min(1)
            self._metrics["prefix/reasoning_policy_loss"].append(
                prefix_loss.detach().item()
            )
            self._metrics["prefix/reasoning_clip_ratio"].append(
                (
                    (prefix_clip * active_reasoning_mask).sum()
                    / prefix_denominator
                ).detach().item()
            )
            self._metrics["prefix/reasoning_token_fraction"].append(
                (
                    active_reasoning_mask.sum()
                    / completion_mask.sum().clamp_min(1)
                ).detach().item()
            )
            self._metrics["bridge/weighted_loss"].append(
                (float(cfg.answer_bridge_weight) * bridge_loss).detach().item()
            )
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics["clip_ratio"].append(
            self.accelerator.gather_for_metrics(clip_ratio).mean().item()
        )
        return loss

    def _get_per_token_logps_and_prefix_margins(
        self,
        model: Any,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        completion_ids: Tensor,
        completion_mask: Tensor,
        prompt_length: int,
        original_indices: Tensor,
        mapped_indices: Tensor,
        multimodal_inputs: Mapping[str, Any],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """One forward for sampled log-probs and Plan10 boundary tensors.

        Returns per-token sampled log-probs, causal A-D option logits,
        mapped-minus-original margins, answer-location validity, and the
        reasoning-only mask that ends strictly before ``<answer>``.
        """

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **multimodal_inputs,
        )
        logits = outputs.logits
        locations = locate_answer_candidate_tokens(
            completion_ids,
            completion_mask,
            self._resolve_prefix_answer_styles(),
        )
        option_logits = answer_prefix_option_logits(
            logits,
            prompt_length=prompt_length,
            locations=locations,
            styles=self._resolve_prefix_answer_styles(),
        )
        margins = answer_prefix_margins(
            logits,
            prompt_length=prompt_length,
            locations=locations,
            original_indices=original_indices,
            mapped_indices=mapped_indices,
            styles=self._resolve_prefix_answer_styles(),
        )
        reasoning_mask = answer_prefix_reasoning_mask(
            completion_mask,
            locations,
            self._resolve_prefix_answer_styles(),
        )

        shifted_logits = logits[:, :-1, :]
        shifted_input_ids = input_ids[:, 1:]
        per_token_logps: list[Tensor] = []
        # Same row-wise log-softmax loop as the pinned vendor to cap memory.
        for logits_row, input_ids_row in zip(shifted_logits, shifted_input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            per_token_logps.append(
                torch.gather(
                    log_probs,
                    dim=1,
                    index=input_ids_row.unsqueeze(1),
                ).squeeze(1)
            )
        return (
            torch.stack(per_token_logps),
            option_logits,
            margins,
            locations.valid,
            reasoning_mask,
        )

    def _prefix_credit_advantages(
        self,
        buffered: Mapping[str, Any],
        *,
        global_margins: Tensor,
        global_valid: Tensor,
    ) -> Tensor:
        """Build global optimized rewards, then return this rank's slice."""

        cfg = self.joint_config
        if cfg is None or cfg.prefix_credit_weight <= 0.0:  # pragma: no cover
            raise RuntimeError("prefix-credit advantages requested while disabled")
        accuracy = buffered["regcfpo_prefix_accuracy_global"].float()
        format_reward = buffered["regcfpo_prefix_format_global"].float()
        swap_mask = buffered["regcfpo_prefix_swap_mask_global"].bool()
        expected_shape = accuracy.shape
        for name, tensor in (
            ("format", format_reward),
            ("swap mask", swap_mask),
            ("margins", global_margins),
            ("valid mask", global_valid),
        ):
            if tensor.shape != expected_shape:
                raise RuntimeError(
                    f"global prefix {name} shape {tuple(tensor.shape)} does not "
                    f"match rewards {tuple(expected_shape)}"
                )
        credit = bounded_prefix_credit(
            global_margins,
            global_valid,
            temperature=cfg.prefix_credit_temperature,
        )
        base_rewards = accuracy + format_reward
        optimized_rewards = base_rewards + (
            float(cfg.prefix_credit_weight) * credit * swap_mask.float()
        )
        advantages_global, optimized_stds = group_standardized_advantages(
            optimized_rewards,
            self.num_generations,
        )
        _, base_stds = group_standardized_advantages(
            base_rewards,
            self.num_generations,
        )

        expected_local = buffered["completion_ids"].shape[0]
        process_slice = slice(
            self.accelerator.process_index * expected_local,
            (self.accelerator.process_index + 1) * expected_local,
        )
        local_advantages = advantages_global[process_slice]
        if local_advantages.shape[0] != expected_local:
            raise RuntimeError(
                f"prefix advantage slice has {local_advantages.shape[0]} rows; "
                f"expected local batch {expected_local}"
            )

        swap_valid = global_valid & swap_mask
        valid_margins = global_margins[swap_valid]
        swap_credit = credit[swap_mask]
        swap_advantages = advantages_global[swap_mask]
        grouped_swap = swap_mask.view(-1, self.num_generations)[:, 0]
        grouped_opt_std = optimized_stds.view(-1, self.num_generations)[:, 0]
        grouped_base_std = base_stds.view(-1, self.num_generations)[:, 0]
        self._metrics["prefix/valid_fraction"].append(
            global_valid[swap_mask].float().mean().item()
        )
        self._metrics["prefix/margin_mean"].append(
            valid_margins.mean().item() if valid_margins.numel() else 0.0
        )
        self._metrics["prefix/margin_std"].append(
            valid_margins.std(unbiased=False).item() if valid_margins.numel() else 0.0
        )
        self._metrics["prefix/credit_mean"].append(swap_credit.mean().item())
        self._metrics["prefix/credit_std"].append(
            swap_credit.std(unbiased=False).item()
        )
        self._metrics["prefix/optimized_reward_mean"].append(
            optimized_rewards[swap_mask].mean().item()
        )
        self._metrics["prefix/optimized_group_std_mean"].append(
            grouped_opt_std[grouped_swap].mean().item()
        )
        self._metrics["prefix/base_group_std_mean"].append(
            grouped_base_std[grouped_swap].mean().item()
        )
        self._metrics["prefix/nonzero_group_std_fraction"].append(
            (grouped_opt_std[grouped_swap] > 1e-6).float().mean().item()
        )
        self._metrics["prefix/non_epsilon_dominated_group_fraction"].append(
            (
                grouped_opt_std[grouped_swap] > GRPO_REWARD_STD_EPSILON
            ).float().mean().item()
        )
        self._metrics["prefix/advantage_abs_mean"].append(
            swap_advantages.abs().mean().item()
        )
        return local_advantages.detach()

    def _plan10_prefix_payload(
        self,
        buffered: Mapping[str, Any],
        *,
        global_margins: Tensor,
        global_valid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return local reasoning advantages, active rows, and bridge weights.

        Prefix credit is standardized independently of the ordinary outcome
        reward and is routed only to fully valid all-wrong swap groups.  The
        full-completion vendored advantages remain in ``buffered`` and are
        applied separately by the caller.
        """

        cfg = self.joint_config
        if (
            cfg is None
            or cfg.prefix_credit_scope != PLAN10_PREFIX_CREDIT_SCOPE
            or cfg.prefix_credit_weight <= 0.0
        ):  # pragma: no cover - caller/config enforce this
            raise RuntimeError("Plan10 prefix payload requested while disabled")
        accuracy = buffered["regcfpo_prefix_accuracy_global"].float()
        format_reward = buffered["regcfpo_prefix_format_global"].float()
        swap_mask = buffered["regcfpo_prefix_swap_mask_global"].bool()
        expected_shape = accuracy.shape
        for name, tensor in (
            ("format", format_reward),
            ("swap mask", swap_mask),
            ("margins", global_margins),
            ("valid mask", global_valid),
        ):
            if tensor.shape != expected_shape:
                raise RuntimeError(
                    f"global Plan10 {name} shape {tuple(tensor.shape)} does not "
                    f"match rewards {tuple(expected_shape)}"
                )
        credit = bounded_prefix_credit(
            global_margins,
            global_valid,
            temperature=cfg.prefix_credit_temperature,
        )
        routed_valid = global_valid & swap_mask
        prefix_advantages, active_rows, credit_stds = all_wrong_prefix_advantages(
            accuracy,
            credit,
            routed_valid,
            self.num_generations,
        )
        bridge_weights_global = credit_softmax_bridge_weights(
            credit,
            active_rows,
            self.num_generations,
            temperature=cfg.answer_bridge_temperature,
        )

        expected_local = buffered["completion_ids"].shape[0]
        process_slice = slice(
            self.accelerator.process_index * expected_local,
            (self.accelerator.process_index + 1) * expected_local,
        )
        local_advantages = prefix_advantages[process_slice]
        local_active_rows = active_rows[process_slice]
        local_bridge_weights = bridge_weights_global[process_slice]
        for name, tensor in (
            ("advantages", local_advantages),
            ("active rows", local_active_rows),
            ("bridge weights", local_bridge_weights),
        ):
            if tensor.shape[0] != expected_local:
                raise RuntimeError(
                    f"local Plan10 {name} has {tensor.shape[0]} rows; "
                    f"expected {expected_local}"
                )

        grouped_swap = swap_mask.view(-1, self.num_generations)[:, 0]
        grouped_active = active_rows.view(-1, self.num_generations)[:, 0]
        grouped_valid = routed_valid.view(-1, self.num_generations).all(dim=1)
        grouped_credit_std = credit_stds.view(-1, self.num_generations)[:, 0]
        base_rewards = accuracy + format_reward
        _, base_stds = group_standardized_advantages(
            base_rewards,
            self.num_generations,
        )
        grouped_base_std = base_stds.view(-1, self.num_generations)[:, 0]
        valid_margins = global_margins[routed_valid]
        swap_credit = credit[swap_mask]
        active_advantages = prefix_advantages[active_rows]

        def selected_mean(values: Tensor, selected: Tensor) -> float:
            chosen = values[selected]
            return chosen.mean().item() if chosen.numel() else 0.0

        self._metrics["prefix/valid_fraction"].append(
            global_valid[swap_mask].float().mean().item()
        )
        self._metrics["prefix/all_valid_group_fraction"].append(
            selected_mean(grouped_valid.float(), grouped_swap)
        )
        self._metrics["prefix/active_all_wrong_group_fraction"].append(
            selected_mean(grouped_active.float(), grouped_swap)
        )
        self._metrics["prefix/margin_mean"].append(
            valid_margins.mean().item() if valid_margins.numel() else 0.0
        )
        self._metrics["prefix/margin_std"].append(
            valid_margins.std(unbiased=False).item() if valid_margins.numel() else 0.0
        )
        self._metrics["prefix/credit_mean"].append(
            swap_credit.mean().item() if swap_credit.numel() else 0.0
        )
        self._metrics["prefix/credit_std"].append(
            swap_credit.std(unbiased=False).item() if swap_credit.numel() else 0.0
        )
        self._metrics["prefix/optimized_reward_mean"].append(
            selected_mean(credit, swap_mask)
        )
        self._metrics["prefix/optimized_group_std_mean"].append(
            selected_mean(grouped_credit_std, grouped_active)
        )
        self._metrics["prefix/base_group_std_mean"].append(
            selected_mean(grouped_base_std, grouped_swap)
        )
        self._metrics["prefix/nonzero_group_std_fraction"].append(
            selected_mean((grouped_credit_std > 1e-6).float(), grouped_active)
        )
        self._metrics["prefix/non_epsilon_dominated_group_fraction"].append(
            selected_mean(
                (grouped_credit_std > GRPO_REWARD_STD_EPSILON).float(),
                grouped_active,
            )
        )
        self._metrics["prefix/advantage_abs_mean"].append(
            active_advantages.abs().mean().item() if active_advantages.numel() else 0.0
        )
        self._metrics["bridge/active_weight_mean"].append(
            bridge_weights_global[active_rows].mean().item()
            if bool(active_rows.any())
            else 0.0
        )
        self._metrics["bridge/active_weight_max"].append(
            bridge_weights_global[active_rows].max().item()
            if bool(active_rows.any())
            else 0.0
        )
        return (
            local_advantages.detach(),
            local_active_rows.detach(),
            local_bridge_weights.detach(),
        )

    def _resolve_prefix_answer_styles(self) -> tuple[AnswerTokenStyle, ...]:
        """Resolve tokenizer-specific answer token boundaries exactly once."""

        if self._prefix_answer_styles is None:
            tokenizer = getattr(
                self.processing_class,
                "tokenizer",
                self.processing_class,
            )
            self._prefix_answer_styles = build_answer_token_styles(tokenizer)
        return self._prefix_answer_styles

    def _compute_auxiliary_objective(
        self, model: Any, buffered: Mapping[str, Any]
    ) -> Tensor | None:
        spec = self.objective_spec
        weights = buffered["regcfpo_gate_weights"]
        groups: list[Mapping[str, Any]] = buffered["regcfpo_aux_groups"]
        global_active = bool(buffered["regcfpo_global_active"])

        # plan_auxiliary_execution encodes the rank-consistency rule; here
        # the executed set is the full global batch on every rank (see the
        # module docstring), so weights/groups are already global.  A replay
        # group (AReG merged manifest) forces its gate weight to zero, so a
        # replay-only step yields global_active False and every rank skips
        # together; the log schema below is unchanged (count 0 is logged).
        plan = plan_auxiliary_execution(
            global_active=global_active,
            executed_weights=weights,
            forwards_per_group=spec.forwards_per_group,
        )
        self._metrics["aux/forward_count"].append(float(plan.forward_count))
        if not plan.run_forwards:
            return None

        scores, monitors = self._collect_auxiliary_scores(model, groups, spec)
        executed = spec.forwards_per_group * len(groups)
        if executed != plan.forward_count:
            raise RuntimeError(
                f"auxiliary forward count drift: planned {plan.forward_count}, "
                f"executed {executed}"
            )
        weights = weights.to(self.accelerator.device)
        return self._assemble_auxiliary_loss(spec, scores, weights, groups, monitors)

    def _collect_auxiliary_scores(
        self,
        model: Any,
        groups: Sequence[Mapping[str, Any]],
        spec: ObjectiveSpec,
    ) -> tuple[dict[tuple[str, str], Tensor], dict[str, Tensor | None]]:
        """Teacher-forced scores for every (branch, candidate, group).

        One forward per triple, strictly serial (swap/factual/null branches in
        registry order) to bound peak memory; every forward count is identical
        across ranks by construction.  Scores keep gradients.

        UNIT CONTRACT: ``areg_cfpo`` scores are TOTAL nats over the candidate
        span (``aggregation="sum"``, amendment score_unit total_nats); every
        other objective keeps the frozen mean-per-token convention
        (``aggregation="mean"``).  For areg_cfpo the factual-branch
        factual-answer forward additionally reads p(<think>) at the
        prompt-final logit position (first-class monitor, plan7 4.4b) at zero
        extra forward cost.

        Returns ``(scores, monitors)`` where ``monitors["p_think_first_token"]``
        is a detached per-group tensor (areg only, else None).
        """

        aggregation = (
            "sum"
            if spec.name in ("areg_cfpo", "pairaug_refgain_trust")
            else "mean"
        )
        think_token_id = self._resolve_think_first_token_id() if spec.name == "areg_cfpo" else None
        think_probs: list[float] = []
        scores: dict[tuple[str, str], Tensor] = {}
        for branch in spec.branches:
            for candidate in spec.candidates:
                per_group: list[Tensor] = []
                for meta in groups:
                    prompt_text = self._aux_prompt_text(meta)
                    image = self._aux_image(meta, branch)
                    if candidate == "factual_answer":
                        letter = meta["answer_letter"]
                    elif candidate == "mapped_answer":
                        letter = meta["mapped_answer_letter"]
                    else:
                        try:
                            letter = OPTION_CANDIDATE_LETTERS[candidate]
                        except KeyError:
                            raise ValueError(
                                f"unknown auxiliary candidate {candidate!r}"
                            ) from None
                    monitor_think = (
                        think_token_id is not None
                        and branch == "factual"
                        and candidate == "factual_answer"
                    )
                    result = self._score_candidate_on_image(
                        model,
                        image=image,
                        prompt_text=prompt_text,
                        candidate_text=CANDIDATE_TEXT_TEMPLATE.format(letter=letter),
                        aggregation=aggregation,
                        think_first_token_id=think_token_id if monitor_think else None,
                    )
                    if monitor_think:
                        score, p_think = result
                        think_probs.append(p_think)
                    else:
                        score = result
                    per_group.append(score)
                scores[(branch, candidate)] = torch.stack(per_group)
        monitors: dict[str, Tensor | None] = {
            "p_think_first_token": (
                torch.tensor(think_probs, dtype=torch.float32) if think_probs else None
            )
        }
        return scores, monitors

    def _assemble_auxiliary_loss(
        self,
        spec: ObjectiveSpec,
        scores: Mapping[tuple[str, str], Tensor],
        weights: Tensor,
        groups: Sequence[Mapping[str, Any]] | None = None,
        monitors: Mapping[str, Tensor | None] | None = None,
    ) -> Tensor:
        """Map collected scores to the objective's auxiliary loss + metrics."""
        if spec.name == "regcfpo":
            s_swap = scores[("swap", "mapped_answer")] - scores[("swap", "factual_answer")]
            s_null = scores[("null", "mapped_answer")] - scores[("null", "factual_answer")]
            g_train = s_swap - s_null
            # normalize_weights=False: the gate weights were normalized and
            # clipped on the GLOBAL batch at generation time (module
            # docstring); re-normalizing here would double-count.
            auxiliary = regcfpo_auxiliary_loss(
                weights=weights,
                g_train=g_train,
                s_null=s_null,
                gamma=self.gamma,
                lambda_null=self.lambda_null,
                config=self.gate_config,
                normalize_weights=False,
            )
            l_dir = directional_loss(weights, g_train, margin=self.gate_config.margin_dir)
            l_null = null_anchor_loss(weights, s_null, margin_null=self.gate_config.margin_null)
            self._metrics["aux/l_dir"].append(l_dir.detach().item())
            self._metrics["aux/l_null"].append(l_null.detach().item())
            self._metrics["aux/g_train_mean"].append(g_train.detach().mean().item())
            return auxiliary
        if spec.name == "areg_cfpo":
            return self._assemble_areg_loss(scores, weights, groups, monitors)
        if spec.name == "pairaug_refgain_trust":
            return self._assemble_pairaug_reference_trust_loss(
                scores, weights, groups
            )
        if spec.name == "local_nondirectional_cfpo":
            # plan5 section 5.2: only the factual/swap divergence of the
            # factual answer score; no mapped direction, no resampling null.
            divergence = scores[("factual", "factual_answer")] - scores[("swap", "factual_answer")]
            l_local = directional_loss(
                weights, divergence, margin=self.gate_config.margin_dir
            )
            self._metrics["aux/l_local_divergence"].append(l_local.detach().item())
            self._metrics["aux/divergence_mean"].append(divergence.detach().mean().item())
            return self.gamma * l_local
        raise RuntimeError(
            f"no auxiliary loss assembly for registered objective {spec.name!r}; "
            "update ReGCFPOTrainer when extending the registry"
        )

    def _assemble_pairaug_reference_trust_loss(
        self,
        scores: Mapping[tuple[str, str], Tensor],
        weights: Tensor,
        groups: Sequence[Mapping[str, Any]] | None,
    ) -> Tensor:
        """Assemble dense swap gain plus source/null slack trust regions."""

        cfg = self.joint_config
        if cfg is None:  # pragma: no cover - __init__ enforces this
            raise RuntimeError("joint loss assembly without a joint config")
        if groups is None:
            raise RuntimeError("joint loss assembly requires group metadata")
        candidates = ("option_a", "option_b", "option_c", "option_d")
        source_logps = torch.stack(
            [scores[("factual", candidate)] for candidate in candidates], dim=-1
        )
        swap_logps = torch.stack(
            [scores[("swap", candidate)] for candidate in candidates], dim=-1
        )
        null_logps = torch.stack(
            [scores[("null", candidate)] for candidate in candidates], dim=-1
        )
        original_indices: list[int] = []
        mapped_indices: list[int] = []
        for meta in groups:
            original = str(meta["pairaug_original_letter"]).upper()
            mapped = str(meta["pairaug_mapped_letter"]).upper()
            if original not in "ABCD" or mapped not in "ABCD" or original == mapped:
                raise ValueError(
                    f"invalid joint original/mapped letters for {meta.get('sample_id')!r}: "
                    f"{original!r}/{mapped!r}"
                )
            original_indices.append(ord(original) - ord("A"))
            mapped_indices.append(ord(mapped) - ord("A"))
        row_index = torch.arange(swap_logps.shape[0], device=swap_logps.device)
        original_index = torch.tensor(
            original_indices, dtype=torch.long, device=swap_logps.device
        )
        mapped_index = torch.tensor(
            mapped_indices, dtype=torch.long, device=swap_logps.device
        )
        s_swap = (
            swap_logps[row_index, mapped_index]
            - swap_logps[row_index, original_index]
        )
        s_swap_0, source_logps_0, null_logps_0 = build_joint_reference_tensors(
            groups, device=swap_logps.device
        )
        auxiliary, parts = pairaug_reference_trust_auxiliary_loss(
            weights=weights,
            s_swap=s_swap,
            source_logps=source_logps,
            null_logps=null_logps,
            s_swap_0=s_swap_0,
            source_logps_0=source_logps_0,
            null_logps_0=null_logps_0,
            gamma=self.gamma,
            config=cfg,
        )
        self._metrics["aux_joint/l_gain"].append(parts["gain"].detach().item())
        self._metrics["aux_joint/l_source_trust"].append(
            parts["source_trust"].detach().item()
        )
        self._metrics["aux_joint/l_null_trust"].append(
            parts["null_trust"].detach().item()
        )
        self._metrics["aux_joint/swap_gain_mean"].append(
            parts["swap_gain"].detach().mean().item()
        )
        self._metrics["aux_joint/source_kl_mean"].append(
            parts["source_kl"].detach().mean().item()
        )
        self._metrics["aux_joint/null_kl_mean"].append(
            parts["null_kl"].detach().mean().item()
        )
        self._metrics["aux_joint/source_trust_active_fraction"].append(
            (parts["source_kl"].detach() > cfg.source_kl_slack).float().mean().item()
        )
        self._metrics["aux_joint/null_trust_active_fraction"].append(
            (parts["null_kl"].detach() > cfg.null_kl_slack).float().mean().item()
        )
        return auxiliary

    def _assemble_areg_loss(
        self,
        scores: Mapping[tuple[str, str], Tensor],
        weights: Tensor,
        groups: Sequence[Mapping[str, Any]] | None,
        monitors: Mapping[str, Tensor | None] | None,
    ) -> Tensor:
        """AReG-CFPO assembly: ``gamma * (L_swap + lambda_pres * L_pres)``.

        All scores here are TOTAL nats (the collection used
        ``aggregation="sum"``), matching the theta0 reference columns and the
        frozen margins ``m_s``/``m_d``.  At ``gamma == 0`` (the E0.2
        strict-equivalence configuration) the reference columns are never
        consumed — the manifest need not carry them — and the result is
        exactly zero with graph-connected scores.

        First-class monitors (amendment ``first_class_monitors``) are logged
        on every executed step: the six absolute answer log-probs
        (orig/mapped x factual/swap/null), the ``t_i`` DiD-arm dominance
        fraction, and p(<think>).
        """

        cfg = self.areg_config
        if cfg is None:  # pragma: no cover - __init__ enforces this
            raise RuntimeError("areg_cfpo assembly without an AReGConfig")
        s_swap = scores[("swap", "mapped_answer")] - scores[("swap", "factual_answer")]
        s_null = scores[("null", "mapped_answer")] - scores[("null", "factual_answer")]
        s_fact = scores[("factual", "mapped_answer")] - scores[("factual", "factual_answer")]
        logp_fact_orig = scores[("factual", "factual_answer")]

        if self.gamma == 0.0:
            auxiliary = areg_cfpo_auxiliary_loss(
                weights=weights,
                s_swap=s_swap,
                s_null=s_null,
                s_fact=s_fact,
                logp_fact_orig=logp_fact_orig,
                gamma=0.0,
                config=cfg,
                normalize_weights=False,
            )
        else:
            if groups is None:
                raise RuntimeError("areg_cfpo assembly requires the group metadata rows")
            s_null_0, s_fact_0, logp_fact_orig_0 = build_areg_reference_tensors(
                groups, device=s_swap.device
            )
            auxiliary = areg_cfpo_auxiliary_loss(
                weights=weights,
                s_swap=s_swap,
                s_null=s_null,
                s_fact=s_fact,
                logp_fact_orig=logp_fact_orig,
                s_null_0=s_null_0,
                s_fact_0=s_fact_0,
                logp_fact_orig_0=logp_fact_orig_0,
                gamma=self.gamma,
                config=cfg,
                normalize_weights=False,
            )
            l_swap = areg_swap_loss(
                weights, s_swap, s_null,
                margin_swap=cfg.margin_swap, margin_did=cfg.margin_did,
            )
            l_pres = areg_preservation_loss(
                weights, s_null, s_fact, logp_fact_orig,
                s_null_0=s_null_0, s_fact_0=s_fact_0, logp_fact_orig_0=logp_fact_orig_0,
                eta=cfg.eta, eta_abs=cfg.eta_abs, beta=cfg.smooth_l1_beta,
            )
            self._metrics["aux/l_swap"].append(l_swap.detach().item())
            self._metrics["aux/l_pres"].append(l_pres.detach().item())

        # First-class monitors (also under gamma=0: the E0.2 records double
        # as the pre-training baseline for these quantities).
        self._metrics["aux_monitor/logp_factual_orig"].append(
            scores[("factual", "factual_answer")].detach().mean().item()
        )
        self._metrics["aux_monitor/logp_factual_mapped"].append(
            scores[("factual", "mapped_answer")].detach().mean().item()
        )
        self._metrics["aux_monitor/logp_swap_orig"].append(
            scores[("swap", "factual_answer")].detach().mean().item()
        )
        self._metrics["aux_monitor/logp_swap_mapped"].append(
            scores[("swap", "mapped_answer")].detach().mean().item()
        )
        self._metrics["aux_monitor/logp_null_orig"].append(
            scores[("null", "factual_answer")].detach().mean().item()
        )
        self._metrics["aux_monitor/logp_null_mapped"].append(
            scores[("null", "mapped_answer")].detach().mean().item()
        )
        did_mask = areg_did_arm_mask(
            s_null, margin_swap=cfg.margin_swap, margin_did=cfg.margin_did
        )
        self._metrics["aux_monitor/t_did_arm_fraction"].append(
            did_mask.float().mean().item()
        )
        p_think = (monitors or {}).get("p_think_first_token")
        if p_think is not None:
            self._metrics["aux_monitor/p_think_first_token"].append(p_think.mean().item())
        return auxiliary

    def _resolve_think_first_token_id(self) -> int:
        """Token id behind the p(<think>) first-class monitor (plan7 4.4b).

        ``<think>`` is a multi-token string in the Qwen2.5 vocab
        (``<th``, ``ink``, ``>``); the monitor reads the probability of the
        canonical FIRST token at the prompt-final logit position — the event
        "the rollout opens a think block" (the remaining tokens are
        near-deterministic continuations of the first).  Resolved once from
        the processing class's tokenizer and cached.
        """

        if self._think_first_token_id is None:
            tokenizer = getattr(self.processing_class, "tokenizer", self.processing_class)
            ids = tokenizer("<think>", add_special_tokens=False)["input_ids"]
            if not ids:
                raise ValueError("the tokenizer produced no token ids for '<think>'")
            self._think_first_token_id = int(ids[0])
        return self._think_first_token_id

    def _aux_prompt_text(self, meta: Mapping[str, Any]) -> str:
        """Render the exact rollout prompt text for one group (cached).

        The stored conversation rows carry None-valued content keys from the
        datasets round-trip; they are sanitized before rendering so the chat
        template emits exactly one image placeholder, byte-identical to the
        vendored rollout path (grpo_trainer.py content-item cleaning).
        """

        sample_id = str(meta["sample_id"])
        if sample_id not in self._aux_prompt_cache:
            self._aux_prompt_cache[sample_id] = self.processing_class.apply_chat_template(
                sanitize_conversation_for_template(meta["prompt"]),
                tokenize=False,
                add_generation_prompt=True,
            )
        return self._aux_prompt_cache[sample_id]

    def _aux_image(self, meta: Mapping[str, Any], branch: str) -> Image.Image:
        """Deterministically rebuild one branch image for one group.

        Pixel edits are pure CPU PIL ops, so caching per generation cycle is
        exact.  An operator rejection on an executed row is a data-contract
        violation (the entry point filters ``operator_valid`` rows) and is
        fail-fast on every rank.
        """

        sample_id = str(meta["sample_id"])
        edits = self._aux_edit_cache.get(sample_id)
        if edits is None:
            edits = {}
            self._aux_edit_cache[sample_id] = edits
        if branch not in edits:
            image_path = (
                meta["pairaug_source_image_path"]
                if self.objective_spec.name == "pairaug_refgain_trust"
                else meta["image_path"]
            )
            if isinstance(image_path, (list, tuple)):
                if len(image_path) != 1:
                    raise AuxiliaryOperatorError(
                        f"sample {sample_id}: the auxiliary branch supports exactly "
                        f"one image per row, got {len(image_path)}"
                    )
                image_path = image_path[0]
            source = self._image_loader(str(image_path))
            try:
                if branch == "factual":
                    edits[branch] = canonical_noop(source).image
                elif branch in ("swap", "null"):
                    operator = (
                        pixel_pair_slot_swap if branch == "swap" else canonical_resampling_return
                    )
                    result = operator(source, meta["gt_box_a"], meta["gt_box_b"])
                    if not result.accepted or result.image is None:
                        raise AuxiliaryOperatorError(
                            f"sample {sample_id}: frozen operator {result.metadata.get('operator', branch)} "
                            f"rejected an active training row: {result.reject_reason}"
                        )
                    edits[branch] = result.image
                else:
                    raise AuxiliaryOperatorError(f"unknown auxiliary branch {branch!r}")
            finally:
                source.close()
        return edits[branch]

    def _score_candidate_on_image(
        self,
        model: Any,
        *,
        image: Image.Image,
        prompt_text: str,
        candidate_text: str,
        aggregation: str = "mean",
        think_first_token_id: int | None = None,
    ) -> Tensor | tuple[Tensor, float]:
        """Teacher-forced candidate score with gradients through the policy.

        Direct processor + model call (grpo_trainer.py:639-645 pattern);
        ``vlm_module.prepare_model_inputs`` is intentionally NOT used (dead
        code with the ``imgaes=images`` typo at qwen_module.py:53).  No
        ``inference_mode``: the score must stay graph-connected.  The
        two-branch/serial discipline is enforced by the caller's loop order.

        ``aggregation``: ``"mean"`` (default) is the frozen historical
        length-normalized convention (``candidate_span_mean_log_probs``)
        used by regcfpo/local_nondirectional_cfpo; ``"sum"`` returns TOTAL
        nats over the 7-token span (``candidate_span_log_probs``) and is used
        ONLY by the areg_cfpo path (amendment score_unit: total_nats).

        ``think_first_token_id``: when set (the areg factual-branch
        factual-answer forward only), the prompt-final logit row is also
        read and ``(score, p_think)`` is returned, where ``p_think`` is the
        FP32 softmax probability of that token id — the p(<think>)
        first-class monitor at zero extra forward cost.
        """

        processor = self.processing_class
        base = processor(text=[prompt_text], images=[image], padding=False, return_tensors="pt")
        full = processor(
            text=[prompt_text + candidate_text],
            images=[image],
            padding=False,
            return_tensors="pt",
        )
        for name in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
            if name not in full:
                raise ValueError(f"processor output is missing required field {name!r}")
        candidate_mask = _candidate_suffix_mask(base["input_ids"], full["input_ids"])
        if not torch.equal(base["image_grid_thw"], full["image_grid_thw"]):
            raise ValueError("candidate text unexpectedly changed image_grid_thw")

        device = self.accelerator.device
        input_ids = full["input_ids"].to(device)
        attention_mask = full["attention_mask"].to(device)
        pixel_values = full["pixel_values"].to(device)
        image_grid_thw = full["image_grid_thw"].to(device)
        candidate_mask = candidate_mask.to(device)

        validate_teacher_forced_call(use_cache=False)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=False,
            return_dict=True,
        )
        if aggregation == "mean":
            score = candidate_span_mean_log_probs(
                outputs.logits, input_ids, candidate_mask
            )[0]
        elif aggregation == "sum":
            score = candidate_span_log_probs(outputs.logits, input_ids, candidate_mask)[0]
        else:
            raise ValueError(f"unknown score aggregation {aggregation!r}")
        p_think: float | None = None
        if think_first_token_id is not None:
            prompt_length = base["input_ids"].shape[1]
            prompt_end_logits = outputs.logits[0, prompt_length - 1].float()
            if not 0 <= think_first_token_id < prompt_end_logits.shape[0]:
                raise ValueError(
                    f"think_first_token_id {think_first_token_id} outside the "
                    f"vocab dimension {prompt_end_logits.shape[0]}"
                )
            p_think = float(
                torch.softmax(prompt_end_logits, dim=-1)[think_first_token_id].item()
            )
        del outputs
        if think_first_token_id is not None:
            return score, p_think
        return score


class CycleOrderedReGCFPOTrainer(ReGCFPOTrainer):
    """ReGCFPOTrainer with the deterministic cycle-exposure sampler.

    Amendment ``asymmetric_directional_credit_revision_20260826``
    exposure_rule: every manifest row is visited exactly once per cycle in
    ``cycle_position`` order; the vendored ``RepeatRandomSampler`` draws a
    fresh randperm per epoch (with-replacement across epochs) and cannot
    provide this.  ``scripts/train_stage4.py`` sorts the merged manifest by
    ``cycle_position`` before building the dataset, so the sequential layout
    of ``CycleOrderSampler`` IS the cycle order (348 rows at one prompt group
    per step = one 348-step cycle).  Installed through the ``trainer_class``
    instrumentation seam for every objective run on a cycle-position
    manifest — including the continued-GRPO control leg, which must see the
    identical exposure schedule (single-scientific-variable rule).
    """

    def _get_train_sampler(self) -> CycleOrderSampler:
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        return CycleOrderSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
        )


__all__ = [
    "AUX_GROUP_METADATA_FIELDS",
    "CANDIDATE_TEXT_TEMPLATE",
    "AuxiliaryOperatorError",
    "CycleOrderedReGCFPOTrainer",
    "ReGCFPOTrainer",
]
