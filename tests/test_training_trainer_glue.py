"""CPU unit tests for the Stage-4 trainer glue (no model, no open_r1).

Covers: dataset Features/passthrough completeness, objective registry
fail-closed behavior, entry-point validation (hyperparameters, resume
policy, atomic contract), the global-gather gate path, and the unique-prompt
dedup of the auxiliary branch.  The trainer subclass itself imports the
vendored ``open_r1`` stack and is exercised only in the training
environment; here its metadata contract is checked statically via AST.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import train_stage4  # noqa: E402
from regcfpo.run_contract import RunContractError  # noqa: E402
from regcfpo.training.auxiliary import GateConfig  # noqa: E402
from regcfpo.training.gate import (  # noqa: E402
    collapse_generation_groups,
    compute_global_gate_payload,
    group_accuracy_rewards,
    local_group_slice,
    plan_auxiliary_execution,
)
from regcfpo.training.objectives import (  # noqa: E402
    OBJECTIVE_REGISTRY,
    get_objective_spec,
    registered_objectives,
)


def _v3_example() -> dict:
    return {
        "schema_version": "RelationPair-v3",
        "sample_id": "spld-00001",
        "scene_id": "scene0007",
        "image": "scene0007_00/42.jpg",
        "question": "is the chair to the desk's left or right?",
        "options": ["A. left", "B. right", "C. left-front", "D. right-back"],
        "answer_relation": "left",
        "mapped_relation": "right",
        "answer_letter": "A",
        "mapped_answer_letter": "B",
        "entity_a": "chair",
        "entity_b": "desk",
        "gt_box_a": [1.0, 2.0, 30.0, 40.0],
        "gt_box_b": [50.0, 60.0, 90.0, 100.0],
        "relation_class": "axis",
        "operator_valid": True,
    }


# --------------------------------------------------------------------- #
# Features / conversation passthrough                                    #
# --------------------------------------------------------------------- #


def test_features_spec_covers_task_contract_fields():
    spec_names = {name for name, _ in train_stage4.STAGE4_FEATURES_SPEC}
    for required in (
        "sample_id",
        "scene_id",
        "question_id",
        "relation_class",
        "relation_family",
        "answer_letter",
        "mapped_answer_letter",
        "gt_box_a",
        "gt_box_b",
        "operator_main",
        "operator_null",
        "prompt",
        "image_path",
        # AReG merged-manifest contract (amendment 20260826)
        "row_kind",
        "cycle_position",
        "areg_s_fact_0",
        "areg_s_null_0",
        "areg_logp_fact_orig_0",
        "answer",
    ):
        assert required in spec_names


def test_make_conversation_output_matches_spec_exactly():
    row = train_stage4.make_conversation(_v3_example(), image_root=Path("/img"))
    assert set(row) == train_stage4.REQUIRED_STAGE4_COLUMNS
    # passthrough integrity
    example = _v3_example()
    for field in (
        "sample_id",
        "scene_id",
        "answer_letter",
        "mapped_answer_letter",
        "relation_class",
        "gt_box_a",
        "gt_box_b",
    ):
        assert row[field] == example[field]
    assert row["question_id"] == example["sample_id"]
    assert row["relation_family"] == example["relation_class"]
    assert row["operator_main"] == "pixel_pair_slot_swap"
    assert row["operator_null"] == "canonical_resampling_return"
    assert row["data_type"] == "single_image"
    # AReG columns default on legacy manifests (never consumed there)
    assert row["row_kind"] == "directional"
    assert row["cycle_position"] == -1
    assert row["answer"] == ""
    assert math.isnan(row["areg_s_fact_0"])
    assert math.isnan(row["areg_s_null_0"])
    assert math.isnan(row["areg_logp_fact_orig_0"])
    # vendored-layout conversation: one image item then one text item
    content = row["prompt"][0]["content"]
    assert [item["type"] for item in content] == ["image", "text"]
    assert content[0]["image"].endswith("scene0007_00/42.jpg")
    assert row["image_path"] == [content[0]["image"]]


def test_validate_stage4_columns_fail_closed():
    train_stage4.validate_stage4_columns(sorted(train_stage4.REQUIRED_STAGE4_COLUMNS))
    with pytest.raises(ValueError, match="missing"):
        train_stage4.validate_stage4_columns(["sample_id", "prompt"])
    with pytest.raises(ValueError, match="extra"):
        train_stage4.validate_stage4_columns(
            sorted(train_stage4.REQUIRED_STAGE4_COLUMNS | {"surprise"})
        )


def test_trainer_aux_metadata_fields_are_preserved_by_features_spec():
    """Static AST check: the trainer's required row fields stay in the spec."""

    source = (PROJECT_ROOT / "src/regcfpo/training/trainer.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    fields = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "AUX_GROUP_METADATA_FIELDS"
            for target in node.targets
        ):
            fields = ast.literal_eval(node.value)
    assert fields is not None, "AUX_GROUP_METADATA_FIELDS not found in trainer.py"
    spec_names = {name for name, _ in train_stage4.STAGE4_FEATURES_SPEC}
    assert set(fields) <= spec_names


def test_trainer_candidate_score_units_per_objective():
    """Score-unit contract: mean is the frozen default; sum is areg-only.

    The regcfpo/local paths must keep the frozen V3-proxy length-normalized
    score (``candidate_span_mean_log_probs``); the areg_cfpo path uses TOTAL
    nats (``candidate_span_log_probs``) via the explicit ``aggregation``
    parameter, whose default must stay ``"mean"`` (amendment
    asymmetric_directional_credit_revision_20260826, score_unit total_nats).
    """

    source = (PROJECT_ROOT / "src/regcfpo/training/trainer.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    scorer = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_score_candidate_on_image"
    )
    calls = {
        node.func.id
        for node in ast.walk(scorer)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "candidate_span_mean_log_probs" in calls
    assert "candidate_span_log_probs" in calls  # areg total-nats path only
    kwarg = next(
        (arg, default)
        for arg, default in zip(scorer.args.kwonlyargs, scorer.args.kw_defaults)
        if arg.arg == "aggregation"
    )
    assert isinstance(kwarg[1], ast.Constant) and kwarg[1].value == "mean"


def test_official_prompt_parity_with_saturation_audit():
    script = PROJECT_ROOT / "scripts" / "reward_saturation_audit.py"
    spec = importlib.util.spec_from_file_location("reward_saturation_audit", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    record = _v3_example()
    assert train_stage4.official_prompt_text(record["question"], record["options"]) == (
        module.official_prompt(record)
    )


# --------------------------------------------------------------------- #
# Reward functions                                                       #
# --------------------------------------------------------------------- #


def _conversation_completion(text: str):
    return [[{"role": "assistant", "content": text}]]


def test_accuracy_reward_binary_letter_match():
    completions = _conversation_completion("<think>x</think><answer>A</answer>") + _conversation_completion(
        "<think>x</think><answer>B</answer>"
    ) + _conversation_completion("no tags")
    rewards = train_stage4.accuracy_reward(
        completions, answer_letter=["A", "A", "C"], prompts=[None] * 3
    )
    assert rewards == [1.0, 0.0, 0.0]


def test_parse_answer_letter_aligned_with_eval_side():
    """Train-side parsing mirrors ``regcfpo.audit.clean_stage3_mca_text``:
    last block wins, one trailing period tolerated, case-folded."""

    assert train_stage4.parse_answer_letter("<think>x</think><answer>A.</answer>") == "A"
    assert train_stage4.parse_answer_letter("<answer>b</answer>") == "B"
    assert (
        train_stage4.parse_answer_letter("<answer>B</answer> then <answer>A</answer>")
        == "A"
    )
    # multi-letter / empty / missing blocks never reduce to one letter
    assert train_stage4.parse_answer_letter("<answer>AB</answer>") is None
    assert train_stage4.parse_answer_letter("<answer> </answer>") is None
    assert train_stage4.parse_answer_letter("<answer></answer>") is None
    assert train_stage4.parse_answer_letter("no tags at all") is None


def test_accuracy_reward_tolerates_trailing_period_like_eval():
    """``<answer>A.</answer>`` must score 1 in training exactly as in eval."""

    completions = _conversation_completion("<think>x</think><answer>A.</answer>")
    rewards = train_stage4.accuracy_reward(completions, answer_letter=["A"])
    assert rewards == [1.0]


def test_format_reward_vendor_semantics():
    good = _conversation_completion("<think>reasoning</think>\n<answer>A</answer>")
    nested = _conversation_completion("<think>a<think>b</think></think><answer>A</answer>")
    double = _conversation_completion("<think>a</think><think>b</think><answer>A</answer>")
    bare = _conversation_completion("answer is A")
    rewards = train_stage4.format_reward(good + nested + double + bare)
    assert rewards == [1.0, 0.0, 0.0, 0.0]


# --------------------------------------------------------------------- #
# Two-tier training authorization (plan7 section 11)                     #
# --------------------------------------------------------------------- #


def test_authorization_smoke_tier_unchanged(monkeypatch):
    monkeypatch.delenv("TRAINING_SMOKE_AUTHORIZED", raising=False)
    with pytest.raises(SystemExit, match="TRAINING_SMOKE_AUTHORIZED"):
        train_stage4.require_training_authorization(formal=False, max_steps=5)
    monkeypatch.setenv("TRAINING_SMOKE_AUTHORIZED", "true")
    monkeypatch.delenv("FORMAL_TRAINING_AUTHORIZED", raising=False)
    # smoke path: max_steps <= 50 without --formal needs no formal variable
    train_stage4.require_training_authorization(formal=False, max_steps=50)
    train_stage4.require_training_authorization(formal=False, max_steps=-1)


def test_authorization_diagnostic_tier_fail_closed(monkeypatch):
    monkeypatch.setenv("TRAINING_SMOKE_AUTHORIZED", "true")
    monkeypatch.delenv("OBJECTIVE_V2_DIAGNOSTIC_AUTHORIZED", raising=False)
    monkeypatch.setenv("FORMAL_TRAINING_AUTHORIZED", "false")
    with pytest.raises(SystemExit, match="OBJECTIVE_V2_DIAGNOSTIC_AUTHORIZED"):
        train_stage4.require_training_authorization(formal=False, max_steps=51)
    with pytest.raises(SystemExit, match="OBJECTIVE_V2_DIAGNOSTIC_AUTHORIZED"):
        train_stage4.require_training_authorization(formal=False, max_steps=348)


def test_authorization_diagnostic_tier_passes_when_authorized(monkeypatch):
    monkeypatch.setenv("TRAINING_SMOKE_AUTHORIZED", "true")
    monkeypatch.setenv("OBJECTIVE_V2_DIAGNOSTIC_AUTHORIZED", "true")
    monkeypatch.setenv("FORMAL_TRAINING_AUTHORIZED", "false")
    train_stage4.require_training_authorization(formal=False, max_steps=51)
    train_stage4.require_training_authorization(formal=False, max_steps=348)


def test_authorization_formal_tier_fail_closed(monkeypatch):
    monkeypatch.setenv("TRAINING_SMOKE_AUTHORIZED", "true")
    monkeypatch.setenv("OBJECTIVE_V2_DIAGNOSTIC_AUTHORIZED", "true")
    monkeypatch.setenv("FORMAL_TRAINING_AUTHORIZED", "false")
    with pytest.raises(SystemExit, match="FORMAL_TRAINING_AUTHORIZED"):
        train_stage4.require_training_authorization(formal=True, max_steps=5)
    with pytest.raises(SystemExit, match="FORMAL_TRAINING_AUTHORIZED"):
        train_stage4.require_training_authorization(formal=False, max_steps=349)


def test_authorization_refusal_prints_both_variables(monkeypatch, capsys):
    monkeypatch.setenv("TRAINING_SMOKE_AUTHORIZED", "true")
    monkeypatch.delenv("FORMAL_TRAINING_AUTHORIZED", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        train_stage4.require_training_authorization(formal=True, max_steps=348)
    message = str(excinfo.value)
    assert "TRAINING_SMOKE_AUTHORIZED='true'" in message
    assert "FORMAL_TRAINING_AUTHORIZED=None" in message
    monkeypatch.delenv("OBJECTIVE_V2_DIAGNOSTIC_AUTHORIZED", raising=False)
    with pytest.raises(SystemExit) as excinfo_diag:
        train_stage4.require_training_authorization(formal=False, max_steps=348)
    message_diag = str(excinfo_diag.value)
    assert "TRAINING_SMOKE_AUTHORIZED='true'" in message_diag
    assert "OBJECTIVE_V2_DIAGNOSTIC_AUTHORIZED=None" in message_diag


def test_authorization_formal_passes_when_authorized(monkeypatch):
    monkeypatch.setenv("TRAINING_SMOKE_AUTHORIZED", "true")
    monkeypatch.setenv("FORMAL_TRAINING_AUTHORIZED", "true")
    train_stage4.require_training_authorization(formal=True, max_steps=348)
    train_stage4.require_training_authorization(formal=False, max_steps=400)


# --------------------------------------------------------------------- #
# Objective registry                                                     #
# --------------------------------------------------------------------- #


def test_registry_registers_expected_objectives():
    assert registered_objectives() == (
        "continued_grpo",
        "regcfpo",
        "local_nondirectional_cfpo",
        "areg_cfpo",
        "pairaug_refgain_trust",
    )
    # the KILLED objective is retained untouched for historical reproduction
    assert OBJECTIVE_REGISTRY["regcfpo"].branches == ("swap", "null")
    assert OBJECTIVE_REGISTRY["regcfpo"].candidates == ("factual_answer", "mapped_answer")
    assert OBJECTIVE_REGISTRY["regcfpo"].forwards_per_group == 4
    local = OBJECTIVE_REGISTRY["local_nondirectional_cfpo"]
    assert local.branches == ("factual", "swap")
    assert local.candidates == ("factual_answer",)
    assert local.forwards_per_group == 2
    assert not local.uses_mapped_direction and not local.uses_resampling_null
    assert not OBJECTIVE_REGISTRY["continued_grpo"].auxiliary_enabled
    # AReG-CFPO (amendment asymmetric_directional_credit_revision_20260826)
    areg = OBJECTIVE_REGISTRY["areg_cfpo"]
    assert areg.branches == ("factual", "swap", "null")
    assert areg.candidates == ("factual_answer", "mapped_answer")
    assert areg.forwards_per_group == 6
    assert areg.auxiliary_enabled


def test_registry_fail_closed_on_unregistered():
    with pytest.raises(ValueError, match="global_duality"):
        get_objective_spec("global_duality")
    with pytest.raises(ValueError, match="unregistered objective"):
        get_objective_spec("vanilla_cfpo")
    with pytest.raises(ValueError):
        get_objective_spec("")


def test_hyperparameter_policy_per_objective():
    spec, gamma, lambda_null, areg = train_stage4.resolve_hyperparameters("regcfpo", 1.0, None)
    assert (spec.name, gamma, lambda_null) == ("regcfpo", 1.0, 0.5)
    assert areg is None
    with pytest.raises(ValueError, match="gamma > 0"):
        train_stage4.resolve_hyperparameters("regcfpo", None, None)
    _, gamma, lambda_null, _ = train_stage4.resolve_hyperparameters(
        "local_nondirectional_cfpo", 0.5, None
    )
    assert (gamma, lambda_null) == (0.5, 0.0)
    with pytest.raises(ValueError, match="lambda_null"):
        train_stage4.resolve_hyperparameters("local_nondirectional_cfpo", 0.5, 0.5)
    _, gamma, lambda_null, _ = train_stage4.resolve_hyperparameters("continued_grpo", None, None)
    assert (gamma, lambda_null) == (0.0, 0.0)
    with pytest.raises(ValueError, match="silently do nothing"):
        train_stage4.resolve_hyperparameters("continued_grpo", 0.5, None)
    with pytest.raises(ValueError, match="non-negative"):
        train_stage4.resolve_hyperparameters("regcfpo", -1.0, None)
    with pytest.raises(ValueError, match="finite"):
        train_stage4.resolve_hyperparameters("regcfpo", float("nan"), None)


def test_hyperparameter_policy_areg_defaults_are_frozen_values():
    spec, gamma, lambda_null, areg = train_stage4.resolve_hyperparameters("areg_cfpo", 0.5, None)
    assert spec.name == "areg_cfpo" and gamma == 0.5 and lambda_null == 0.0
    # amendment-frozen margins (total nats); eta_abs defaults to eta
    assert (areg.margin_swap, areg.margin_did) == (1.0, 6.5)
    assert (areg.lambda_pres, areg.eta, areg.eta_abs) == (1.0, 1.0, 1.0)
    _, _, _, areg = train_stage4.resolve_hyperparameters(
        "areg_cfpo", 0.5, None, eta=0.3
    )
    assert areg.eta == 0.3 and areg.eta_abs == 0.3  # eta_abs tracks eta
    _, _, _, areg = train_stage4.resolve_hyperparameters(
        "areg_cfpo", 0.5, None, margin_did=7.0, lambda_pres=2.0, eta_abs=0.1
    )
    assert (areg.margin_did, areg.lambda_pres, areg.eta_abs) == (7.0, 2.0, 0.1)
    # areg parameters on a non-areg objective fail closed
    with pytest.raises(ValueError, match="no AReG terms"):
        train_stage4.resolve_hyperparameters("regcfpo", 1.0, None, margin_did=6.5)
    with pytest.raises(ValueError, match="gamma > 0"):
        train_stage4.resolve_hyperparameters("areg_cfpo", None, None)


# --------------------------------------------------------------------- #
# Gate on a simulated global gather                                      #
# --------------------------------------------------------------------- #


def test_group_accuracy_rewards_fold():
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    means, max_pos = group_accuracy_rewards(rewards, 4)
    assert means.tolist() == [1.0, 0.5, 0.0]
    assert max_pos[0].item() == 0.0  # SAT: zero advantage
    assert max_pos[1].item() > 0.0  # MIX: positive advantage present
    assert max_pos[2].item() == 0.0  # ALL_WRONG
    with pytest.raises(ValueError, match="multiple"):
        group_accuracy_rewards(torch.ones(3), 4)
    with pytest.raises(ValueError, match=">= 2"):
        group_accuracy_rewards(torch.ones(4), 1)
    with pytest.raises(FloatingPointError):
        group_accuracy_rewards(torch.tensor([1.0, float("inf"), 0.0, 1.0]), 4)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        group_accuracy_rewards(torch.tensor([1.0, 2.0, 0.0, 1.0]), 4)


def test_global_gate_payload_strata_and_weights():
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    payload = compute_global_gate_payload(rewards, 4)
    assert payload.num_groups == 3
    assert payload.strata.sat_fraction == pytest.approx(1 / 3)
    assert payload.strata.mix_fraction == pytest.approx(1 / 3)
    assert payload.strata.all_wrong_fraction == pytest.approx(1 / 3)
    assert payload.global_active and payload.active_count == 2
    weights = payload.weights.tolist()
    assert weights[0] > 0.0 and weights[1] > 0.0 and weights[2] == 0.0
    assert not payload.weights.requires_grad  # stop-gradient end to end
    # plan7 section 1.1 R1 closed form: SAT credit exactly 1; the k=2/G=4
    # MIX credit is exactly 1/sqrt(3) independent of the batch companions.
    credits = payload.credits.tolist()
    assert credits[0] == pytest.approx(1.0)
    assert credits[1] == pytest.approx(1.0 / math.sqrt(3), abs=1e-6)
    assert credits[2] == 0.0
    # the vendored-trainer advantage mirror is still surfaced (diagnostic)
    assert payload.group_max_positive_advantage[0].item() == 0.0
    assert payload.group_max_positive_advantage[1].item() > 0.0


def test_global_gate_payload_single_group_closed_form_credit():
    """The one-prompt-group-per-step configuration no longer self-normalizes.

    Previously a lone MIX group's credit was batch-normalized to exactly 1;
    now the k/G closed form applies (k=2, G=4 -> 1/sqrt(3)).  The final
    normalize_gate_weights batch normalization is unchanged, so a lone
    active group still carries weight ~1 after normalization.
    """

    payload = compute_global_gate_payload(torch.tensor([1.0, 0.0, 1.0, 0.0]), 4)
    assert payload.credits.tolist() == pytest.approx([1.0 / math.sqrt(3)], abs=1e-6)
    assert payload.weights.tolist() == pytest.approx([1.0], abs=1e-4)
    sat = compute_global_gate_payload(torch.ones(4), 4)
    assert sat.credits.tolist() == [1.0]
    all_wrong = compute_global_gate_payload(torch.zeros(4), 4)
    assert all_wrong.credits.tolist() == [0.0]
    assert not all_wrong.global_active and all_wrong.active_count == 0


def test_global_gate_payload_plan9_monotonic_singleton_is_not_normalized():
    cfg = GateConfig(credit_mode="monotonic_k_over_g_raw")
    for successes, expected in ((0, 0.0), (1, 0.25), (2, 0.5), (3, 0.75), (4, 1.0)):
        rewards = torch.tensor([1.0] * successes + [0.0] * (4 - successes))
        payload = compute_global_gate_payload(rewards, 4, cfg)
        assert payload.credits.tolist() == pytest.approx([expected])
        assert payload.weights.tolist() == pytest.approx([expected])
        assert payload.credit_mode == "monotonic_k_over_g_raw"
        assert not payload.normalization_applied
        assert payload.clip_hit_fraction == 0.0


def test_global_gate_payload_credit_is_batch_independent():
    mix_group = [1.0, 1.0, 0.0, 0.0]  # k=2, G=4
    alone = compute_global_gate_payload(torch.tensor(mix_group), 4)
    batched = compute_global_gate_payload(
        torch.tensor([1.0] * 4 + mix_group + [0.0] * 4), 4
    )
    assert batched.credits[1].item() == pytest.approx(alone.credits[0].item())


def test_plan_auxiliary_execution_rank_consistency():
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    payload = compute_global_gate_payload(rewards, 4)
    plan = plan_auxiliary_execution(
        global_active=payload.global_active,
        executed_weights=payload.weights,
        forwards_per_group=4,
    )
    assert plan.run_forwards and plan.forward_count == 8
    # every-rank-skips-together case
    empty = compute_global_gate_payload(torch.zeros(8), 4)
    plan = plan_auxiliary_execution(
        global_active=empty.global_active,
        executed_weights=empty.weights,
        forwards_per_group=4,
    )
    assert not plan.run_forwards and plan.forward_count == 0
    with pytest.raises(FloatingPointError):
        plan_auxiliary_execution(
            global_active=True,
            executed_weights=torch.tensor([float("nan")]),
            forwards_per_group=4,
        )


def test_local_group_slice_divisibility():
    assert local_group_slice(4, 0, 2) == slice(0, 2)
    assert local_group_slice(4, 1, 2) == slice(2, 4)
    with pytest.raises(ValueError, match="divisible"):
        local_group_slice(1, 0, 2)
    with pytest.raises(ValueError, match="out of range"):
        local_group_slice(4, 2, 2)


def test_collapse_generation_groups_dedup_and_layout():
    def row(sample_id, question_id="q"):
        return {"sample_id": sample_id, "question_id": question_id, "payload": sample_id}

    rows = [row("a")] * 4 + [row("b")] * 4
    groups = collapse_generation_groups(rows, 4)
    assert [group["sample_id"] for group in groups] == ["a", "b"]
    with pytest.raises(ValueError, match="layout violated"):
        collapse_generation_groups([row("a"), row("b"), row("a"), row("a")], 4)
    with pytest.raises(ValueError, match="twice"):
        collapse_generation_groups([row("a")] * 4 + [row("a")] * 4, 4)
    with pytest.raises(ValueError, match="multiple"):
        collapse_generation_groups([row("a")] * 3, 4)
    with pytest.raises(ValueError, match="missing"):
        collapse_generation_groups([{"sample_id": "a"}] * 4, 4)


# --------------------------------------------------------------------- #
# Entry validation: resume policy, atomic contract, dry-run              #
# --------------------------------------------------------------------- #


def _fake_checkpoint(
    run_dir: Path, name="checkpoint-25", complete=True, global_step: int | None = None
) -> Path:
    checkpoint = run_dir / name
    checkpoint.mkdir(parents=True)
    trainer_state = {} if global_step is None else {"global_step": global_step}
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(trainer_state), encoding="utf-8"
    )
    if complete:
        (checkpoint / "rng_state_0.pth").write_bytes(b"\x00")
        (checkpoint / "optimizer.pt").write_bytes(b"\x00")
    return checkpoint


def test_resume_is_explicit_only(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint = _fake_checkpoint(run_dir)
    with pytest.raises(SystemExit, match="never implicit"):
        train_stage4.resolve_resume_checkpoint(run_dir, None)
    assert train_stage4.resolve_resume_checkpoint(run_dir, checkpoint) == checkpoint.resolve()
    with pytest.raises(SystemExit, match="inside"):
        train_stage4.resolve_resume_checkpoint(run_dir, tmp_path / "checkpoint-9")
    broken = _fake_checkpoint(tmp_path / "other", name="checkpoint-3", complete=False)
    (tmp_path / "other").mkdir(exist_ok=True)
    with pytest.raises(SystemExit, match="rng_state"):
        train_stage4.resolve_resume_checkpoint(tmp_path / "other", broken)
    fresh = tmp_path / "fresh"
    assert train_stage4.resolve_resume_checkpoint(fresh, None) is None


def test_completed_checkpoint_cannot_resume(tmp_path):
    run_dir = tmp_path / "completed"
    checkpoint = _fake_checkpoint(run_dir, name="checkpoint-5", global_step=5)
    with pytest.raises(SystemExit, match="run is complete and must not resume"):
        train_stage4.resolve_resume_checkpoint(run_dir, checkpoint, max_steps=5)
    assert (
        train_stage4.resolve_resume_checkpoint(run_dir, checkpoint, max_steps=6)
        == checkpoint.resolve()
    )


def test_contract_atomic_write_and_verification(tmp_path):
    path = tmp_path / "run_contract.json"
    contract = {"schema_version": 1, "seed": 1234, "objective": "regcfpo"}
    train_stage4.ensure_run_contract_atomic(path, contract)
    assert json.loads(path.read_text(encoding="utf-8")) == contract
    # idempotent re-entry with the identical contract
    train_stage4.ensure_run_contract_atomic(path, contract)
    with pytest.raises(RunContractError, match="objective"):
        train_stage4.ensure_run_contract_atomic(path, {**contract, "objective": "continued_grpo"})
    stale = tmp_path / "run2_contract.json.tmp"
    stale.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="partial contract"):
        train_stage4.ensure_run_contract_atomic(tmp_path / "run2_contract.json", contract)
    # pre-existing artifacts without a contract are refused
    artifact = tmp_path / "checkpoint-1"
    artifact.mkdir()
    with pytest.raises(RunContractError, match="pre-existing outputs"):
        train_stage4.ensure_run_contract_atomic(
            tmp_path / "run3_contract.json", contract, existing_artifacts=[artifact]
        )


def test_load_data_rows_fail_closed(tmp_path):
    manifest = tmp_path / "rows.jsonl"
    manifest.write_text(json.dumps(_v3_example()) + "\n", encoding="utf-8")
    rows = train_stage4.load_data_rows(manifest)
    assert len(rows) == 1 and rows[0]["sample_id"] == "spld-00001"
    bad = _v3_example()
    bad["operator_valid"] = False
    manifest.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="operator_valid"):
        train_stage4.load_data_rows(manifest)
    manifest.write_text(json.dumps({"sample_id": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields"):
        train_stage4.load_data_rows(manifest)
    manifest.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        train_stage4.load_data_rows(manifest)


def _merged_directional(position: int, **overrides) -> dict:
    row = _v3_example()
    row["row_kind"] = "directional"
    row["cycle_position"] = position
    row["areg_s_fact_0"] = -5.4
    row["areg_s_null_0"] = -5.5
    row["areg_logp_fact_orig_0"] = -36.0
    row.update(overrides)
    return row


def _merged_replay(position: int, **overrides) -> dict:
    row = {
        "row_kind": "replay",
        "cycle_position": position,
        "question_id": 6474,
        "scene_id": "scene0104",
        "image": "scene0104_00/7.jpg",
        "question": "How many chairs are there?",
        "options": None,
        "answer": "41",
    }
    row.update(overrides)
    return row


def test_load_merged_manifest_validates_and_sorts_by_cycle_position(tmp_path):
    manifest = tmp_path / "merged.jsonl"
    rows_in = [
        _merged_directional(2),
        _merged_replay(0),
        _merged_replay(1),
        _merged_directional(3, sample_id="spld-00002"),
    ]
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows_in), encoding="utf-8"
    )
    rows = train_stage4.load_data_rows(manifest)
    assert [row["cycle_position"] for row in rows] == [0, 1, 2, 3]
    assert [row["row_kind"] for row in rows] == [
        "replay", "replay", "directional", "directional",
    ]
    assert train_stage4.manifest_uses_cycle_order(rows)
    # replay conversation uses the official NA prompt and keeps the answer
    conv = train_stage4.make_conversation(rows[0], image_root=Path("/img"))
    assert conv["answer"] == "41"
    assert conv["sample_id"] == "6474"  # question_id fallback
    assert conv["prompt"][0]["content"][1]["text"].endswith(
        "a numerical value (e.g., 42 or 3.1) within the <answer> </answer> tags."
    )
    assert conv["operator_valid"] is False and conv["gt_box_a"] == []
    assert math.isnan(conv["areg_s_null_0"])
    dconv = train_stage4.make_conversation(rows[2], image_root=Path("/img"))
    assert dconv["areg_s_fact_0"] == -5.4


def _merged_replay_mc(position: int, **overrides) -> dict:
    row = _merged_replay(
        position,
        question="Which object is closer to the camera?",
        options=["A. door", "B. window"],
        answer="B",
    )
    row.update(overrides)
    return row


def test_load_merged_manifest_mc_replay_row(tmp_path):
    """MC replay rows (letter answer + options) load and get the MCA prompt."""

    import e3_rollout_eval as e3

    manifest = tmp_path / "merged.jsonl"
    manifest.write_text(json.dumps(_merged_replay_mc(0)) + "\n", encoding="utf-8")
    rows = train_stage4.load_data_rows(manifest)
    assert rows[0]["answer"] == "B"
    conv = train_stage4.make_conversation(rows[0], image_root=Path("/img"))
    assert conv["answer_letter"] == "B"
    assert conv["options"] == ["A. door", "B. window"]
    # e3_rollout_eval mode=="mca" parity: the exact directional MCA prompt
    record = {"question": rows[0]["question"], "options": rows[0]["options"]}
    text = conv["prompt"][0]["content"][1]["text"]
    assert text == e3.build_prompt(record, "mca")
    assert "option's letter" in text


def test_load_merged_manifest_numeric_replay_row_options_string_none(tmp_path):
    """Numeric replay rows carry options as the literal string "None" in the
    real pool; the conversation must normalize it to an empty list (never
    iterate the string into characters) and keep the NA prompt."""

    manifest = tmp_path / "merged.jsonl"
    manifest.write_text(
        json.dumps(_merged_replay(0, options="None")) + "\n", encoding="utf-8"
    )
    rows = train_stage4.load_data_rows(manifest)
    conv = train_stage4.make_conversation(rows[0], image_root=Path("/img"))
    assert conv["options"] == []
    assert conv["answer_letter"] == ""
    text = conv["prompt"][0]["content"][1]["text"]
    assert "Options:" not in text
    assert text.endswith("a numerical value (e.g., 42 or 3.1) within the <answer> </answer> tags.")


def test_load_merged_manifest_replay_dual_form_fail_closed(tmp_path):
    manifest = tmp_path / "merged.jsonl"

    def write(row):
        manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

    # letter answer without options: error carries line number + sample id
    write(_merged_replay(0, answer="C", options=None))
    with pytest.raises(ValueError, match=r":1:.*6474.*MC letter"):
        train_stage4.load_data_rows(manifest)
    write(_merged_replay(0, answer="C", options="None"))
    with pytest.raises(ValueError, match=r":1:.*6474.*MC letter"):
        train_stage4.load_data_rows(manifest)
    # letter out of range of the options list
    write(_merged_replay(0, answer="D", options=["A. door", "B. window"]))
    with pytest.raises(ValueError, match="out of range"):
        train_stage4.load_data_rows(manifest)
    # non-letter non-float answer
    write(_merged_replay(0, answer="many"))
    with pytest.raises(ValueError, match="neither"):
        train_stage4.load_data_rows(manifest)
    # non-finite numeric answer
    write(_merged_replay(0, answer="nan"))
    with pytest.raises(ValueError, match="finite"):
        train_stage4.load_data_rows(manifest)


def test_load_merged_manifest_fail_closed(tmp_path):
    manifest = tmp_path / "merged.jsonl"

    def write(rows):
        manifest.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    # unknown row_kind
    write([_merged_replay(0, row_kind="mystery")])
    with pytest.raises(ValueError, match="row_kind"):
        train_stage4.load_data_rows(manifest)
    # replay rows must carry the numeric answer + identity
    write([_merged_replay(0, answer="not-a-number")])
    with pytest.raises(ValueError, match="float-parseable"):
        train_stage4.load_data_rows(manifest)
    replay_no_id = _merged_replay(0)
    del replay_no_id["question_id"]
    write([replay_no_id])
    with pytest.raises(ValueError, match="sample_id or question_id"):
        train_stage4.load_data_rows(manifest)
    # directional rows of a merged manifest need finite theta0 references
    write([_merged_directional(0, areg_s_null_0=None)])
    with pytest.raises(ValueError, match="areg_s_null_0"):
        train_stage4.load_data_rows(manifest)
    # cycle_position must be the exact permutation 0..N-1
    write([_merged_replay(0), _merged_directional(0)])
    with pytest.raises(ValueError, match="permutation"):
        train_stage4.load_data_rows(manifest)
    write([_merged_replay(1), _merged_directional(0)])
    assert len(train_stage4.load_data_rows(manifest)) == 2
    # mixed legacy/merged manifests are refused
    write([_v3_example(), _merged_replay(1)])
    with pytest.raises(ValueError, match="every row or none"):
        train_stage4.load_data_rows(manifest)


def test_areg_objective_rejects_legacy_manifest(tmp_path):
    data_file = PROJECT_ROOT / "manifests/v3_final/v3_directional_train.jsonl"
    model_path = PROJECT_ROOT / "models/SpatialLadder-3B"
    if not data_file.is_file() or not model_path.is_dir():
        pytest.skip("real manifest/model checkpoint not available")
    with pytest.raises(SystemExit, match="merged directional\\+replay manifest"):
        train_stage4.main(
            [
                "--data-file", str(data_file),
                "--image-root", str(PROJECT_ROOT / "data/raw/spatialladder26k/images"),
                "--model-path", str(model_path),
                "--run-dir", str(tmp_path / "run"),
                "--objective", "areg_cfpo",
                "--gamma", "0.5",
                "--beta", "0.01",
                "--dry-run",
            ]
        )


def test_areg_dry_run_on_merged_manifest(tmp_path, capsys):
    model_path = PROJECT_ROOT / "models/SpatialLadder-3B"
    if not model_path.is_dir():
        pytest.skip("model checkpoint not available")
    manifest = tmp_path / "merged.jsonl"
    rows = [_merged_replay(0)] + [
        _merged_directional(i + 1, sample_id=f"spld-{i:05d}") for i in range(3)
    ]
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    argv = [
        "--data-file", str(manifest),
        "--image-root", str(PROJECT_ROOT / "data/raw/spatialladder26k/images"),
        "--model-path", str(model_path),
        "--run-dir", str(tmp_path / "run"),
        "--objective", "areg_cfpo",
        "--gamma", "0.5",
        "--beta", "0.01",
        "--dry-run",
    ]
    assert train_stage4.main(argv) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["objective"] == "areg_cfpo"
    assert out["cycle_ordered"] is True
    assert out["areg"]["margin_swap"] == 1.0 and out["areg"]["margin_did"] == 6.5
    assert out["contract"]["grpo"]["beta"] == 0.01
    hard_stop = out["contract"]["auxiliary"]["format_hard_stop"]
    assert set(hard_stop) == {"baseline", "max_drop", "window"}


def test_areg_dry_run_on_real_merged_manifest(tmp_path, capsys):
    """The real W6a merged manifest (174 directional + 174 replay rows, the
    replay pool shipping BOTH answer forms: 169 numeric + 5 MC letter rows)
    must pass the dry-run contract validation."""

    data_file = PROJECT_ROOT / "manifests/v3_final/v3_areg_train_348.jsonl"
    model_path = PROJECT_ROOT / "models/SpatialLadder-3B"
    if not data_file.is_file() or not model_path.is_dir():
        pytest.skip("real merged manifest/model checkpoint not available")
    argv = [
        "--data-file", str(data_file),
        "--image-root", str(PROJECT_ROOT / "data/raw/spatialladder26k/images"),
        "--model-path", str(model_path),
        "--run-dir", str(tmp_path / "run"),
        "--objective", "areg_cfpo",
        "--beta", "0.01",
        "--gamma", "0.02",
        "--max-steps", "348",
        "--dry-run",
    ]
    assert train_stage4.main(argv) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["objective"] == "areg_cfpo"
    assert out["rows"] == 348
    assert out["cycle_ordered"] is True
    assert out["areg"]["margin_swap"] == 1.0 and out["areg"]["margin_did"] == 6.5


def test_official_na_prompt_parity_with_eval_side():
    """The replay prompt must match e3_rollout_eval.build_prompt(..., 'na')."""

    import e3_rollout_eval as e3

    record = {
        "sample_id": "r1",
        "question": "How tall is the chair?",
        "options": None,
        "question_type": "object size",
        "answer": "41",
    }
    assert train_stage4.official_na_prompt_text(record["question"], None) == (
        e3.build_prompt(record, "na")
    )
    record_with_options = {**record, "options": ["A. 1", "B. 2"]}
    assert train_stage4.official_na_prompt_text(
        record_with_options["question"], record_with_options["options"]
    ) == e3.build_prompt(record_with_options, "na")
    image_tagged = {**record, "question": "<image>How tall is the chair?"}
    assert "<image>" not in train_stage4.official_na_prompt_text(image_tagged["question"])


def test_dry_run_prints_contract_without_training(tmp_path, capsys):
    data_file = PROJECT_ROOT / "manifests/v3_final/v3_directional_train.jsonl"
    model_path = PROJECT_ROOT / "models/SpatialLadder-3B"
    if not data_file.is_file() or not model_path.is_dir():
        pytest.skip("real manifest/model checkpoint not available")
    argv = [
        "--data-file", str(data_file),
        "--image-root", str(PROJECT_ROOT / "data/raw/spatialladder26k/images"),
        "--model-path", str(model_path),
        "--run-dir", str(tmp_path / "run"),
        "--objective", "regcfpo",
        "--gamma", "1.0",
        "--dry-run",
    ]
    assert train_stage4.main(argv) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["objective"] == "regcfpo"
    assert out["gamma"] == 1.0 and out["lambda_null"] == 0.5
    assert out["rows"] == 174
    assert out["training_authorized"] is False
    contract = out["contract"]
    assert contract["model"]["config_compat"]["action"] == (
        "flattened_raw_text_config_for_transformers_4_49"
    )
    assert contract["runtime"]["attn_implementation"] == "sdpa"
    assert contract["runtime"]["save_total_limit"] == 2
    assert contract["grpo"]["num_generations"] == 4
    assert not (tmp_path / "run").exists()  # dry-run writes nothing


def test_dry_run_rejects_unregistered_objective(tmp_path):
    with pytest.raises(ValueError, match="unregistered objective"):
        train_stage4.main(
            [
                "--data-file", "x.jsonl",
                "--image-root", ".",
                "--model-path", ".",
                "--run-dir", str(tmp_path),
                "--objective", "global_duality",
                "--dry-run",
                "--no-deepspeed",
            ]
        )


def test_features_cast_roundtrip_with_real_datasets():
    """Dataset.map must accept build_stage4_features() on a real conversation row.

    Regression for the datasets>=5 Sequence(raw-dict) behavior that produced a
    struct-of-lists for the prompt column and broke the map cast at trainer
    build time (found by the first gamma-zero probe run).  Skipped where the
    training-only ``datasets`` package is unavailable.
    """

    datasets = pytest.importorskip("datasets")
    row = {
        "schema_version": 3,
        "sample_id": "spld-test",
        "scene_id": "scene_test",
        "question_id": "q_test",
        "relation_class": "diagonal",
        "relation_family": "LF_RB",
        "question": "q?",
        "options": ["A. x", "B. y", "C. z", "D. w"],
        "image": "img.png",
        "answer_relation": "left-back",
        "mapped_relation": "right-front",
        "answer_letter": "A",
        "mapped_answer_letter": "B",
        "entity_a": "a",
        "entity_b": "b",
        "gt_box_a": [0.0, 0.0, 1.0, 1.0],
        "gt_box_b": [1.0, 1.0, 2.0, 2.0],
        "operator_valid": True,
    }
    conv = train_stage4.make_conversation(row, image_root=Path("/tmp"))
    features = train_stage4.build_stage4_features()
    prompt_feature = features["prompt"]
    # list-of-message-structs, not a struct-of-lists
    assert not isinstance(prompt_feature, dict)
    dataset = datasets.Dataset.from_list([conv])
    mapped = dataset.map(lambda x: x, features=features)
    assert set(train_stage4.REQUIRED_STAGE4_COLUMNS) <= set(mapped.column_names)
    assert mapped[0]["prompt"][0]["role"] == "user"


def test_build_stage4_dataset_drops_manifest_only_columns_on_merged_rows():
    """Regression for the GPU-pipeline ``KeyError: 'question_type'``.

    Merged-manifest rows carry replay-provenance columns (``question_type``,
    ``replay_bucket``, ...) beyond the contract columns.  ``Dataset.map``
    with the explicit contract features must DROP them (``remove_columns``);
    otherwise the surviving extra columns hit the arrow_writer at finalize
    and raise ``KeyError`` (trainer_equivalence_check._build_trainer on
    ``v3_areg_train_348.jsonl``).  Skipped where ``datasets`` is unavailable.
    """

    pytest.importorskip("datasets")
    rows = [
        _merged_replay(0, question_type="object count", replay_bucket="counting"),
        _merged_replay_mc(1, question_type="relative distance", replay_bucket="other"),
        _merged_directional(2, question_type="relative direction", replay_bucket=None),
    ]
    dataset = train_stage4.build_stage4_dataset(rows, image_root=Path("/img"))
    assert set(dataset.column_names) == set(train_stage4.REQUIRED_STAGE4_COLUMNS)
    assert [dataset[i]["row_kind"] for i in range(3)] == ["replay", "replay", "directional"]
    assert dataset[0]["options"] == []  # numeric replay: options normalized away
    assert dataset[1]["answer_letter"] == "B"  # MC replay letter mirrored
    assert dataset[2]["areg_s_fact_0"] == pytest.approx(-5.4)


def test_sanitize_conversation_drops_none_keys():
    """The chat template treats key PRESENCE ('image' in content) as an image
    item; sanitized rows must carry exactly one image item and no None values."""

    from regcfpo.training.conversation import sanitize_conversation_for_template

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "text": None, "image": "x.png"},
                {"type": "text", "text": "hello", "image": None},
            ],
        }
    ]
    cleaned = sanitize_conversation_for_template(messages)
    assert cleaned == [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "x.png"},
                {"type": "text", "text": "hello"},
            ],
        }
    ]
    image_like = [
        item
        for item in cleaned[0]["content"]
        if item.get("type") == "image" or "image" in item
    ]
    assert len(image_like) == 1
