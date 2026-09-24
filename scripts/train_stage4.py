#!/usr/bin/env python3
"""Stage-4 GRPO post-training entry for ReG-CFPO on SpatialLadder-3B.

Static entry point for ``results/phase1/phase1_plan5.md`` section 5.  It is
import-safe and ``--help``/``--dry-run`` work without the training
environment; the vendored ``open_r1`` stack, torch, and datasets are imported
lazily only on the real training path (``.conda/regcfpo-train``).

Hard rules implemented here (plan5 section 5.3 and the environment recipe):

- ``TRAINING_SMOKE_AUTHORIZED=true`` is required to start training; this
  entry point itself grants no authorization.  Diagnostic-scope runs
  (``--max-steps`` in ``(50, 348]`` without ``--formal``) additionally
  require ``OBJECTIVE_V2_DIAGNOSTIC_AUTHORIZED=true``; formal-scope runs
  (``--formal`` or ``--max-steps > 348``) require
  ``FORMAL_TRAINING_AUTHORIZED=true`` (fail-closed, plan7 section 11).
  ``--dry-run`` only validates and prints the resolved run configuration.
- Dataset Features are explicit and complete (``STAGE4_FEATURES_SPEC``), so
  ``Dataset.map`` can never silently drop sample_id/scene_id/question_id/
  relation_class/relation_family/answer_letter/mapped_answer_letter/gt_box_a/
  gt_box_b/operator metadata (contrast the vendored stage3 entry, whose
  ``map`` relies on implicit column survival; grpo_spld_stage3.py:347-421).
- Resume is explicit-only: ``checkpoint-*`` discovery never implies resume
  (the vendored ``grpo_spld_stage3.py:453-456`` auto-resume is banned).
  ``--resume-from-checkpoint`` must point inside the run directory, contain
  trainer/rng state, and the immutable run contract must match exactly. A
  checkpoint that has already reached ``max_steps`` is rejected instead of
  silently executing an extra zero-learning-rate step.
- Checkpoint retention is bounded by ``--save-total-limit`` (default 2), so a
  long single-GPU run cannot fill the rented server's 200GB data disk while
  still keeping one mid-run checkpoint for crash recovery on the formal
  200-348 step runs.
- The model config is repaired through
  ``regcfpo.qwen_compat.load_qwen_config_with_compat`` BEFORE any weights are
  loaded; when the 4.53->4.49 repair applies, a run-local symlink view with
  the rewritten ``config.json`` is created and used as the model path.
  ``validate_qwen_weight_tying`` audits the loaded model before training.
- ``--attn-implementation sdpa`` is the default and ``flash_attention_2`` is
  rejected (flash-attn is not installed by preregistered deviation).
- The vendored ``monkey_patch_qwen2_5vl_*`` helpers are deliberately NOT
  applied: the flash-attn patch is unimportable without flash-attn (SDPA
  deviation), and the ZeRO-3 forward patch targets mixed image/text batches
  while Stage-4 data is uniformly single-image.

AReG-CFPO (amendment ``asymmetric_directional_credit_revision_20260826``):

- ``--objective areg_cfpo`` requires the merged directional+replay manifest
  (``row_kind``/``cycle_position``/``areg_*`` theta0 reference columns, total
  nats) and fails closed on the legacy directional-only format.  Replay rows
  are scored by the ported official ``na_reward``
  (``regcfpo.training.rewards``) and carry aux weight zero.
- Merged manifests are iterated in deterministic ``cycle_position`` order via
  ``CycleOrderedReGCFPOTrainer`` (trainer_class seam) for EVERY objective,
  including the continued-GRPO control leg: one cycle = every row exactly
  once = 348 steps at one prompt group per step.
- ``--beta 0.01`` is the registered diagnostic value (the project-local
  default 0.0 was an undiscussed deviation from the vendored run script).
- ``--format-reward-baseline`` is mandatory for areg_cfpo runs and drives the
  format hard stop (rolling-window drop > ``--format-stop-drop``).

Run contract: ``regcfpo.run_contract.ensure_run_contract`` binds model/data/
code/seed/optimizer/scheduler/objective identity to an immutable JSON in the
run directory, written atomically (tmp file + os.replace).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from regcfpo.audit import stage3_format_valid  # noqa: E402
from regcfpo.exposure_audit import compute_exposure_audit  # noqa: E402
from regcfpo.run_contract import ensure_run_contract  # noqa: E402
from regcfpo.training.auxiliary import (  # noqa: E402
    GATE_CREDIT_MODES,
    PREFIX_CREDIT_SCOPES,
    AReGConfig,
    PairAugReferenceTrustConfig,
)
from regcfpo.training.objectives import (  # noqa: E402
    ObjectiveSpec,
    get_objective_spec,
    registered_objectives,
)
from regcfpo.training.rewards import official_stage3_na_reward  # noqa: E402

DEFAULT_DEEPSPEED_CONFIG = (
    PROJECT_ROOT
    / "vendor/SpatialLadder/VLM-R1/src/open-r1-multimodal/local_scripts/zero3.json"
)
DEFAULT_MAX_PIXELS = 100352  # frozen processor identity (pixel_ops.MAX_PIXELS)
DEFAULT_MIN_PIXELS = 12544  # frozen processor identity (pixel_ops.MIN_PIXELS)

OPERATOR_MAIN = "pixel_pair_slot_swap"
OPERATOR_NULL = "canonical_resampling_return"

#: Explicit dataset Features contract.  ``question_id`` and
#: ``relation_family`` are derived columns for the v3 schema (question_id ==
#: sample_id; relation_family == relation_class) so the Stage-4 training
#: contract keeps the field names the analysis chain standardizes on.
#: ``conversation`` is materialized by ``build_stage4_features`` as the
#: nested chat-message structure the vendored rollout path consumes.
#:
#: AReG merged-manifest columns (amendment
#: asymmetric_directional_credit_revision_20260826): ``row_kind``
#: ("directional"/"replay"), ``cycle_position`` (unique int 0..N-1 fixing the
#: deterministic exposure order), the three theta0 reference floats in TOTAL
#: nats on directional rows (NaN on replay rows), and ``answer`` (the numeric
#: replay reference; empty string on directional rows).  Legacy manifests
#: without the new fields are still accepted: ``make_conversation`` fills the
#: documented defaults (row_kind="directional", cycle_position=-1).
STAGE4_FEATURES_SPEC: tuple[tuple[str, str], ...] = (
    ("schema_version", "string"),
    ("sample_id", "string"),
    ("scene_id", "string"),
    ("question_id", "string"),
    ("image", "string"),
    ("image_path", "sequence_string"),
    ("question", "string"),
    ("options", "sequence_string"),
    ("answer", "string"),
    ("answer_relation", "string"),
    ("mapped_relation", "string"),
    ("relation_class", "string"),
    ("relation_family", "string"),
    ("answer_letter", "string"),
    ("mapped_answer_letter", "string"),
    ("entity_a", "string"),
    ("entity_b", "string"),
    ("gt_box_a", "sequence_float64"),
    ("gt_box_b", "sequence_float64"),
    ("operator_valid", "bool"),
    ("operator_main", "string"),
    ("operator_null", "string"),
    ("data_type", "string"),
    ("row_kind", "string"),
    ("cycle_position", "int64"),
    ("areg_s_fact_0", "float64"),
    ("areg_s_null_0", "float64"),
    ("areg_logp_fact_orig_0", "float64"),
    ("pairaug_branch", "string"),
    ("pairaug_parent_sample_id", "string"),
    ("pairaug_source_image_path", "string"),
    ("pairaug_original_letter", "string"),
    ("pairaug_mapped_letter", "string"),
    ("joint_s_swap_0", "float64"),
    ("joint_source_logps_0", "sequence_float64"),
    ("joint_null_logps_0", "sequence_float64"),
    ("prompt", "conversation"),
)

REQUIRED_STAGE4_COLUMNS = frozenset(name for name, _ in STAGE4_FEATURES_SPEC)

#: Fields every DIRECTIONAL input JSONL row must provide (v3 schema; also the
#: full required set for legacy manifests without an explicit ``row_kind``).
REQUIRED_INPUT_FIELDS = frozenset(
    {
        "schema_version",
        "sample_id",
        "scene_id",
        "image",
        "question",
        "options",
        "answer_relation",
        "mapped_relation",
        "answer_letter",
        "mapped_answer_letter",
        "entity_a",
        "entity_b",
        "gt_box_a",
        "gt_box_b",
        "relation_class",
        "operator_valid",
    }
)

#: Fields every REPLAY input JSONL row must provide (merged-manifest rows
#: with ``row_kind == "replay"``).  ``sample_id`` or ``question_id`` identifies
#: the row; ``answer`` follows the dual-form contract
#: (``classify_replay_answer_form``): numeric rows are scored by the ported
#: official ``na_reward``, MC rows (letter answer + non-empty options) by the
#: MCA letter match.  Replay rows are relation-operator-free, so the
#: directional fields (boxes, letters, operator_valid) are intentionally
#: absent.
REQUIRED_INPUT_FIELDS_REPLAY = frozenset(
    {
        "scene_id",
        "image",
        "question",
        "answer",
    }
)

#: Theta0 reference columns (TOTAL nats) required finite on every directional
#: row of a merged manifest (precomputed offline from the Proxy V3 artifact;
#: null on replay rows).
AREG_REFERENCE_FIELDS = ("areg_s_fact_0", "areg_s_null_0", "areg_logp_fact_orig_0")

ROW_KINDS = ("directional", "replay")

#: Answer-block extraction mirroring the evaluation-side normalizer
#: (``regcfpo.audit.clean_stage3_mca_text``): last ``<answer>`` block wins.
_ANSWER_BLOCK_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

#: Source files whose sha256 binds the training code identity in the
#: contract (the project is not a git checkout).
CONTRACT_CODE_FILES = (
    "src/regcfpo/training/auxiliary.py",
    "src/regcfpo/training/gate.py",
    "src/regcfpo/training/objectives.py",
    "src/regcfpo/training/prefix_credit.py",
    "src/regcfpo/training/rewards.py",
    "src/regcfpo/training/sampling.py",
    "src/regcfpo/training/trainer.py",
    "src/regcfpo/qwen_adapter.py",
    "src/regcfpo/qwen_compat.py",
    "src/regcfpo/run_contract.py",
    "src/regcfpo/operators/pixel_ops.py",
    "scripts/train_stage4.py",
    "vendor/SpatialLadder/VLM-R1/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py",
    "vendor/SpatialLadder/VLM-R1/src/open-r1-multimodal/src/open_r1/trainer/grpo_config.py",
)


def official_prompt_text(question: str, options: Sequence[str]) -> str:
    """The frozen official Stage-3 CoT prompt, byte-identical to
    ``scripts/reward_saturation_audit.py:official_prompt`` (parity is pinned
    by ``tests/test_training_trainer_glue.py``)."""

    question_with_options = question + "\nOptions:\n" + "\n".join(options)
    return (
        f"Question: {question_with_options} \n"
        "Please Think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', "
        "'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought "
        "expressions It's encouraged to include self-reflection or verification in the "
        "reasoning process. \n"
        "Please provide your detailed reasoning between the <think> </think> tags, and then "
        "answer the question with the option's letter from the given choices (e.g., A, B, "
        "etc.) within the <answer> </answer> tags."
    )


#: Official stage3 numeric-answer (NA) prompt templates, byte-identical to
#: the vendored ``grpo_spld_stage3.py:get_question_prompt`` NA branch
#: (grpo_spld_stage3.py:362-378) and mirrored by the evaluation side
#: (``scripts/e3_rollout_eval.py`` STAGE3_PRE_PROMPT / NA_POST_PROMPT; parity
#: pinned by ``tests/test_training_trainer_glue.py``).  Used for the merged
#: manifest's ``row_kind == "replay"`` rows, whose answers are numeric.
STAGE3_PRE_PROMPT = (
    "Question: {question} \n"
    "Please Think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', "
    "'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought "
    "expressions It's encouraged to include self-reflection or verification in the "
    "reasoning process. \n"
)
NA_POST_PROMPT = (
    "Please provide your detailed reasoning between the <think> </think> tags, "
    "and then answer the question with a numerical value (e.g., 42 or 3.1) "
    "within the <answer> </answer> tags."
)


def official_na_prompt_text(question: str, options: Sequence[str] | None = None) -> str:
    """The official Stage-3 numeric-answer prompt (replay rows).

    Mirrors ``e3_rollout_eval.build_prompt(record, "na")``: the ``<image>``
    placeholder is stripped from the question, a non-empty options list is
    appended, then the shared pre-prompt and the NA post-prompt.
    """

    text = str(question).replace("<image>", "")
    if options:
        text += "\nOptions:\n" + "\n".join(str(option) for option in options)
    return STAGE3_PRE_PROMPT.format(question=text) + NA_POST_PROMPT


def parse_answer_letter(text: str) -> str | None:
    """Extract the answer letter, aligned with the evaluation-side parser.

    The parsing contract mirrors ``regcfpo.audit.clean_stage3_mca_text`` as
    used by the evaluation scorers (e.g.
    ``scripts/e3_rollout_eval.py:score_completion``: last ``<answer>``
    block wins, surrounding whitespace stripped, trailing periods tolerated
    (``.rstrip(".")``), comparison case-folded (returned uppercased here).
    Anything that does not reduce to exactly one letter returns ``None``.
    ``<answer>A.</answer>`` therefore scores identically in training and
    evaluation (the previous strict regex scored it 0 in training and 1 in
    evaluation); ``AB`` and empty blocks stay wrong on both sides.
    """

    matches = _ANSWER_BLOCK_PATTERN.findall(text)
    if not matches:
        return None
    value = matches[-1].strip().rstrip(".")
    if len(value) != 1 or not value.isalpha():
        return None
    return value.upper()


def _completion_contents(completions: Sequence[Any]) -> list[str]:
    contents = []
    for completion in completions:
        if (
            isinstance(completion, list)
            and len(completion) == 1
            and isinstance(completion[0], dict)
            and isinstance(completion[0].get("content"), str)
        ):
            contents.append(completion[0]["content"])
        elif isinstance(completion, str):
            contents.append(completion)
        else:
            raise ValueError(f"unsupported completion structure: {type(completion)!r}")
    return contents


def accuracy_reward(
    completions: Sequence[Any],
    answer_letter: Sequence[str] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Binary exact-match reward of the parsed answer letter (0/1).

    This is the accuracy-only component the saturation gate consumes; it must
    stay index 0 of the reward list (the trainer records
    ``accuracy_reward_index`` in the run contract).

    Row-kind routing (AReG merged manifest): rows with ``row_kind ==
    "replay"`` route by answer form via ``classify_replay_answer_form`` on
    the ``answer``/``options`` columns — NUMERIC rows are scored by the
    ported official stage3 ``na_reward``
    (``regcfpo.training.rewards.official_stage3_na_reward``) against the
    ``answer`` column; MC rows (letter answer + options) are scored by the
    MCA letter match against the ``answer`` letter, exactly like directional
    rows.  A letter row can therefore never enter ``na_reward`` and a
    numeric row never enters the letter match; malformed rows raise instead
    of silently misrouting.  Every non-replay row keeps the MCA letter
    match.  Calls without a ``row_kind`` column (legacy manifests, direct
    unit-test calls) behave exactly as before.
    """

    contents = _completion_contents(completions)
    kinds = list(kwargs.get("row_kind") or ["directional"] * len(contents))
    answers = list(kwargs.get("answer") or [None] * len(contents))
    options_col = list(kwargs.get("options") or [None] * len(contents))
    letters = list(answer_letter) if answer_letter is not None else [None] * len(contents)
    if not (len(contents) == len(kinds) == len(answers) == len(letters) == len(options_col)):
        raise ValueError(
            "completions, row_kind, answer, options, and answer_letter must have "
            "equal length"
        )
    rewards = []
    for content, kind, answer, options, expected in zip(
        contents, kinds, answers, options_col, letters
    ):
        if kind == "replay":
            form = classify_replay_answer_form(answer, options)
            if form == "numeric":
                rewards.append(official_stage3_na_reward(content, str(answer)))
            else:
                letter = parse_answer_letter(content)
                rewards.append(
                    1.0 if letter is not None and letter == str(answer).strip().upper() else 0.0
                )
        else:
            letter = parse_answer_letter(content)
            rewards.append(
                1.0 if letter is not None and letter == str(expected).upper() else 0.0
            )
    return rewards


def format_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Vendored stage3 format reward (single think+answer, no nested think).

    Single source of truth: ``regcfpo.audit.stage3_format_valid`` (verified
    logically equivalent to the previous local implementation: same
    fullmatch regex, same nested-think rejection, same single-tag counts).
    """

    return [
        1.0 if stage3_format_valid(content) else 0.0
        for content in _completion_contents(completions)
    ]


def build_stage4_features() -> Any:
    """Materialize ``STAGE4_FEATURES_SPEC`` as explicit ``datasets.Features``."""

    from datasets import Features, Sequence as DatasetsSequence, Value
    from datasets import List as DatasetsList

    columns: dict[str, Any] = {}
    for name, kind in STAGE4_FEATURES_SPEC:
        if kind == "string":
            columns[name] = Value("string")
        elif kind == "bool":
            columns[name] = Value("bool")
        elif kind == "int64":
            columns[name] = Value("int64")
        elif kind == "float64":
            columns[name] = Value("float64")
        elif kind == "sequence_string":
            columns[name] = DatasetsSequence(Value("string"))
        elif kind == "sequence_float64":
            columns[name] = DatasetsSequence(Value("float64"))
        elif kind == "conversation":
            # datasets>=5: ``Sequence(raw_dict)`` wraps each struct FIELD in a
            # list (struct-of-lists) instead of producing a list-of-structs;
            # use ``List`` explicitly so the conversation column stays
            # list<message struct>, matching ``make_conversation``.
            columns[name] = DatasetsList(
                {
                    "role": Value("string"),
                    "content": DatasetsList(
                        {
                            "type": Value("string"),
                            "text": Value("string"),
                            "image": Value("string"),
                        }
                    ),
                }
            )
        else:
            raise ValueError(f"unknown features kind {kind!r} for column {name!r}")
    return Features(columns)


def _optional_float(value: Any) -> float:
    """Float passthrough for nullable manifest columns (None -> NaN)."""

    if value is None:
        return float("nan")
    return float(value)


def _normalize_replay_options(options: Any) -> list[str] | None:
    """Official option normalization for replay rows (e3_rollout_eval parity)."""

    if options is None:
        return None
    if isinstance(options, str):
        if options.strip() in ("", "None"):
            return None
        raise ValueError(f"options must be a list or null, got string: {options!r}")
    cleaned = [str(option) for option in options]
    return cleaned or None


#: Replay answer forms of the merged manifest (dual-form contract).
REPLAY_ANSWER_FORMS = ("numeric", "mc")

_MC_ANSWER_LETTERS = frozenset("ABCD")


def classify_replay_answer_form(answer: Any, options: Any) -> str:
    """Classify one replay row's answer form (merged-manifest dual-form contract).

    The v3 replay pool (``manifests/v3_final/v3_replay.jsonl``, 174 rows)
    ships TWO answer shapes:

    - NUMERIC (169 rows: counting / object size / absolute distance across
      multi_view / single_image / video): ``answer`` parses as a finite
      float and ``options`` is absent (null / empty / the literal string
      ``"None"``).  Scored by the ported official stage3 ``na_reward``.
    - MC (5 rows: letter answers on single_image/video MC questions):
      ``answer`` is a single letter A-D (case-insensitive, surrounding
      whitespace tolerated) and ``options`` is a non-empty list COVERING
      that letter.  Scored by the MCA letter match, exactly like
      directional rows (e3_rollout_eval ``mode == "mca"`` parity).

    Returns ``"numeric"`` or ``"mc"``.  Anything else — letter without
    options, letter out of range of the options list, non-letter non-float,
    empty answer — raises ``ValueError``: the replay contract is fail-closed
    so a malformed pool row can never silently route into the wrong reward
    path.  This single pure function is THE routing rule shared by
    ``load_data_rows`` (manifest validation), ``make_conversation`` (prompt
    selection) and ``accuracy_reward`` (scorer selection).
    """

    normalized = _normalize_replay_options(options)
    text = "" if answer is None else str(answer).strip()
    letter = text.upper()
    if len(text) == 1 and letter in _MC_ANSWER_LETTERS:
        if not normalized:
            raise ValueError(
                f"replay answer {answer!r} is an MC letter but options are absent/empty"
            )
        if ord(letter) - ord("A") >= len(normalized):
            raise ValueError(
                f"replay answer {answer!r} is out of range for {len(normalized)} options"
            )
        return "mc"
    try:
        value = float(text)
    except ValueError:
        raise ValueError(
            f"replay answer {answer!r} is neither a single A-D MC letter with "
            "options nor float-parseable (the official na_reward target)"
        ) from None
    if not math.isfinite(value):
        raise ValueError(f"replay answer {answer!r} is not a finite number")
    return "numeric"


def make_conversation(example: Mapping[str, Any], *, image_root: Path) -> dict[str, Any]:
    """Build the vendored-layout conversation row, preserving ALL contract
    columns explicitly (fail-closed against silent ``map`` column drops).

    The prompt content structure mirrors the vendored stage3
    ``make_conversation_from_jsonl`` (grpo_spld_stage3.py:386-418): one image
    item then one text item, single-image ``data_type``.

    Row kinds (AReG merged manifest): ``directional`` rows use the MCA prompt
    (``official_prompt_text``) and carry the operator/box/reference columns;
    ``replay`` rows route by answer form (``classify_replay_answer_form``):
    NUMERIC rows use the official numeric-answer prompt
    (``official_na_prompt_text``) and carry the numeric ``answer``; MC rows
    (letter answer + options) use the directional MCA prompt and mirror the
    letter into ``answer_letter`` so the MCA accuracy path scores them
    exactly like directional rows.  Both replay forms emit empty directional
    placeholders (never consumed: replay groups get aux weight zero).
    Legacy manifests without ``row_kind`` are treated as directional with
    ``cycle_position = -1`` (no cycle schedule).
    """

    row_kind = str(example.get("row_kind") or "directional")
    if row_kind not in ROW_KINDS:
        raise ValueError(f"unknown row_kind {row_kind!r}; expected one of {ROW_KINDS}")
    image_path = str(Path(image_root) / str(example["image"]))
    if row_kind == "replay":
        sample_id = str(example.get("sample_id") or example.get("question_id"))
        options = _normalize_replay_options(example.get("options"))
        form = classify_replay_answer_form(example.get("answer"), example.get("options"))
        options_out = list(options) if options else []
        if form == "mc":
            # e3_rollout_eval mode=="mca" parity: same frozen MCA prompt as
            # directional rows (no ``<image>`` stripping), letter mirrored
            # into answer_letter for the MCA accuracy path.
            prompt_text = official_prompt_text(str(example["question"]), options)
            answer_letter = str(example["answer"]).strip().upper()
        else:
            prompt_text = official_na_prompt_text(str(example["question"]), options)
            answer_letter = str(example.get("answer_letter") or "")
    else:
        sample_id = str(example["sample_id"])
        prompt_text = official_prompt_text(str(example["question"]), example["options"])
        answer_letter = str(example.get("answer_letter") or "")
        options_out = [str(option) for option in example["options"]]
    cycle_position = example.get("cycle_position", -1)
    if cycle_position is None:
        cycle_position = -1
    pairaug_source_image = str(example.get("pairaug_source_image") or "")
    pairaug_source_image_path = (
        str(Path(image_root) / pairaug_source_image) if pairaug_source_image else ""
    )
    return {
        "schema_version": str(example.get("schema_version") or "RelationPair-v3"),
        "sample_id": sample_id,
        "scene_id": str(example["scene_id"]),
        "question_id": sample_id,
        "image": str(example["image"]),
        "image_path": [image_path],
        "question": str(example["question"]),
        "options": options_out,
        "answer": str(example.get("answer") or ""),
        "answer_relation": str(example.get("answer_relation") or ""),
        "mapped_relation": str(example.get("mapped_relation") or ""),
        "relation_class": str(example.get("relation_class") or row_kind),
        "relation_family": str(example.get("relation_class") or row_kind),
        "answer_letter": answer_letter,
        "mapped_answer_letter": str(example.get("mapped_answer_letter") or ""),
        "entity_a": str(example.get("entity_a") or ""),
        "entity_b": str(example.get("entity_b") or ""),
        "gt_box_a": [float(value) for value in (example.get("gt_box_a") or [])],
        "gt_box_b": [float(value) for value in (example.get("gt_box_b") or [])],
        "operator_valid": bool(example.get("operator_valid", False)),
        "operator_main": str(example.get("operator_main") or (OPERATOR_MAIN if row_kind == "directional" else "")),
        "operator_null": str(example.get("operator_null") or (OPERATOR_NULL if row_kind == "directional" else "")),
        "data_type": "single_image",
        "row_kind": row_kind,
        "cycle_position": int(cycle_position),
        "areg_s_fact_0": _optional_float(example.get("areg_s_fact_0")),
        "areg_s_null_0": _optional_float(example.get("areg_s_null_0")),
        "areg_logp_fact_orig_0": _optional_float(example.get("areg_logp_fact_orig_0")),
        "pairaug_branch": str(example.get("pairaug_branch") or ""),
        "pairaug_parent_sample_id": str(
            example.get("pairaug_parent_sample_id") or ""
        ),
        "pairaug_source_image_path": pairaug_source_image_path,
        "pairaug_original_letter": str(
            example.get("pairaug_original_letter") or ""
        ),
        "pairaug_mapped_letter": str(
            example.get("pairaug_mapped_letter") or ""
        ),
        "joint_s_swap_0": _optional_float(example.get("joint_s_swap_0")),
        "joint_source_logps_0": [
            float(value) for value in (example.get("joint_source_logps_0") or [])
        ],
        "joint_null_logps_0": [
            float(value) for value in (example.get("joint_null_logps_0") or [])
        ],
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "text": None, "image": image_path},
                    {"type": "text", "text": prompt_text, "image": None},
                ],
            }
        ],
    }


def validate_stage4_columns(columns: Sequence[str]) -> None:
    """Fail fast unless the mapped dataset carries every contract column."""

    missing = REQUIRED_STAGE4_COLUMNS - set(columns)
    extra = set(columns) - REQUIRED_STAGE4_COLUMNS
    if missing or extra:
        raise ValueError(
            f"stage4 dataset columns violated: missing={sorted(missing)} extra={sorted(extra)}"
        )


def build_stage4_dataset(data_rows: Sequence[Mapping[str, Any]], *, image_root: Path) -> Any:
    """Materialize the contract-column training dataset from validated rows.

    Single source of the ``from_list`` + ``make_conversation`` map + explicit
    ``STAGE4_FEATURES_SPEC`` cast + column-contract validation pipeline,
    shared by ``_run_training`` here and by
    ``scripts/trainer_equivalence_check.py`` (no second features/map copy to
    drift).  Manifest-only columns (the merged manifest's replay provenance
    such as ``question_type``/``replay_bucket``) are DROPPED at map time via
    ``remove_columns``: ``make_conversation`` returns exactly the contract
    columns, and without the drop the surviving extra columns hit the
    explicit-features cast and raise ``KeyError`` in the datasets
    arrow_writer at finalize (first observed on the GPU pipeline with
    ``question_type``).
    """

    import functools

    from datasets import Dataset

    # datasets ``from_list`` infers the arrow schema from the FIRST row only:
    # keys absent from that row are silently DROPPED from later rows (row
    # order after the cycle_position sort would otherwise decide whether
    # e.g. directional-only fields survive).  Pad every row to the union of
    # keys with None (pyarrow unifies null + typed columns fine), making the
    # dataset independent of row order and heterogeneous key sets.
    rows = [dict(row) for row in data_rows]
    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row)
    for row in rows:
        for key in all_keys:
            row.setdefault(key, None)

    dataset = Dataset.from_list(rows)
    dataset = dataset.map(
        functools.partial(make_conversation, image_root=image_root),
        features=build_stage4_features(),
        remove_columns=dataset.column_names,
        desc="stage4 make_conversation",
    )
    validate_stage4_columns(dataset.column_names)
    return dataset


def load_data_rows(path: Path) -> list[dict[str, Any]]:
    """Read the Stage-4 JSONL manifest with fail-closed field validation.

    Two formats are accepted:

    - LEGACY (v3 directional): no ``row_kind`` column; every row must carry
      ``REQUIRED_INPUT_FIELDS`` and ``operator_valid is True``.  Row order is
      preserved exactly (historical reproduction of the pre-AReG runs).
    - MERGED (AReG, amendment asymmetric_directional_credit_revision_20260826):
      every row carries ``row_kind`` in {"directional", "replay"} and a unique
      integer ``cycle_position`` forming the exact permutation ``0..N-1``.
      Directional rows additionally carry the three finite theta0 reference
      columns (``AREG_REFERENCE_FIELDS``, total nats); replay rows carry
      ``REQUIRED_INPUT_FIELDS_REPLAY`` and an ``answer`` satisfying the
      dual-form contract (``classify_replay_answer_form``: finite-float
      numeric, or single A-D letter with covering non-empty options).  The
      returned rows are SORTED BY ``cycle_position`` so the dataset index
      order IS the registered exposure order (consumed by
      ``CycleOrderSampler`` via the trainer_class seam).
    """

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no rows")

    explicit_kinds = ["row_kind" in row for row in rows]
    if any(explicit_kinds) and not all(explicit_kinds):
        raise ValueError(
            f"{path}: row_kind must be present on every row or none "
            "(mixed legacy/merged manifests are refused)"
        )
    if not all(explicit_kinds):
        for line_number, row in enumerate(rows, start=1):
            _validate_directional_row(path, line_number, row)
        return rows

    positions: list[int] = []
    for line_number, row in enumerate(rows, start=1):
        kind = row["row_kind"]
        if kind not in ROW_KINDS:
            raise ValueError(
                f"{path}:{line_number}: row_kind must be one of {ROW_KINDS}, got {kind!r}"
            )
        if kind == "replay":
            missing = REQUIRED_INPUT_FIELDS_REPLAY - set(row)
            if missing:
                raise ValueError(f"{path}:{line_number}: missing fields {sorted(missing)}")
            if not ("sample_id" in row or "question_id" in row):
                raise ValueError(
                    f"{path}:{line_number}: replay rows need sample_id or question_id"
                )
            sample_ref = row.get("sample_id", row.get("question_id"))
            try:
                classify_replay_answer_form(row.get("answer"), row.get("options"))
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: replay row {sample_ref!r}: {exc}"
                ) from exc
        else:
            _validate_directional_row(path, line_number, row)
            for field in AREG_REFERENCE_FIELDS:
                value = row.get(field)
                if value is None or isinstance(value, bool) or not math.isfinite(float(value)):
                    raise ValueError(
                        f"{path}:{line_number}: directional rows of a merged manifest "
                        f"must carry a finite {field} (theta0 reference, total nats), "
                        f"got {value!r}"
                    )
        position = row.get("cycle_position")
        if isinstance(position, bool) or not isinstance(position, int):
            raise ValueError(
                f"{path}:{line_number}: cycle_position must be an int, got {position!r}"
            )
        positions.append(position)
    if sorted(positions) != list(range(len(rows))):
        raise ValueError(
            f"{path}: cycle_position values must form the exact permutation "
            f"0..{len(rows) - 1} (every row visited exactly once per cycle)"
        )
    return [row for _, row in sorted(zip(positions, rows), key=lambda pair: pair[0])]


def _validate_directional_row(path: Path, line_number: int, row: Mapping[str, Any]) -> None:
    """Fail-closed per-row validation shared by legacy and merged manifests."""

    missing = REQUIRED_INPUT_FIELDS - set(row)
    if missing:
        raise ValueError(f"{path}:{line_number}: missing fields {sorted(missing)}")
    if row.get("operator_valid") is not True:
        raise ValueError(
            f"{path}:{line_number}: operator_valid is not true for sample "
            f"{row.get('sample_id')!r}; pre-filter the manifest (the auxiliary "
            "branch requires eligible rows)"
        )


def manifest_uses_cycle_order(rows: Sequence[Mapping[str, Any]]) -> bool:
    """True when the loaded rows carry the merged-manifest cycle schedule."""

    return bool(rows) and all("row_kind" in row for row in rows)


def validate_plan9_joint_contract(
    rows: Sequence[Mapping[str, Any]], args: argparse.Namespace
) -> None:
    """Fail closed unless source/swap GRPO and dense auxiliaries truly co-train.

    Consecutive source/swap rows are accumulated with GA=2 into one optimizer
    update.  The manifest may encode several independently shuffled cycles,
    but every encoded row is consumed exactly once (``max_steps = N/2``).
    """

    if not rows or len(rows) % 2:
        raise ValueError("Plan9 joint manifest must contain an even non-zero row count")
    if args.gradient_accumulation_steps != 2:
        raise ValueError(
            "pairaug_refgain_trust requires gradient_accumulation_steps=2 so "
            "the paired source/swap GRPO losses and swap auxiliary share one update"
        )
    if args.per_device_train_batch_size != args.num_generations:
        raise ValueError(
            "pairaug_refgain_trust requires per-device batch size == num_generations "
            "(one prompt group per microbatch)"
        )
    if args.num_iterations != 1:
        raise ValueError("pairaug_refgain_trust currently requires num_iterations=1")
    if args.lr_scheduler_type != "constant":
        raise ValueError("pairaug_refgain_trust requires a constant learning rate")
    if args.gate_credit_mode != "monotonic_k_over_g_raw":
        raise ValueError(
            "pairaug_refgain_trust requires monotonic_k_over_g_raw for truthful "
            "rollout-credit monitoring"
        )
    if args.max_steps != len(rows) // 2:
        raise ValueError(
            f"pairaug_refgain_trust requires max_steps={len(rows) // 2} to consume "
            "every encoded pair exactly once"
        )

    seen_ids: set[str] = set()
    required = {
        "pairaug_branch",
        "pairaug_parent_sample_id",
        "pairaug_source_image",
        "pairaug_original_letter",
        "pairaug_mapped_letter",
        "pairaug_cycle",
        "joint_s_swap_0",
        "joint_source_logps_0",
        "joint_null_logps_0",
        "gt_box_a",
        "gt_box_b",
        "operator_valid",
    }
    for pair_start in range(0, len(rows), 2):
        pair = rows[pair_start : pair_start + 2]
        parents = {str(row.get("pairaug_parent_sample_id", "")) for row in pair}
        cycles = {row.get("pairaug_cycle") for row in pair}
        branches = {str(row.get("pairaug_branch", "")) for row in pair}
        if len(parents) != 1 or len(cycles) != 1 or branches != {"source", "swap"}:
            raise ValueError(
                f"joint rows {pair_start}:{pair_start + 2} are not one adjacent "
                f"source/swap parent pair: parents={parents}, cycles={cycles}, "
                f"branches={branches}"
            )
        for row in pair:
            missing = required - set(row)
            if missing:
                raise ValueError(
                    f"joint row {row.get('sample_id')!r} missing {sorted(missing)}"
                )
            sample_id = str(row.get("sample_id", ""))
            if not sample_id or sample_id in seen_ids:
                raise ValueError(f"missing/duplicate joint sample_id {sample_id!r}")
            seen_ids.add(sample_id)
            if row.get("row_kind") != "replay":
                raise ValueError(f"joint row {sample_id!r} must use replay reward routing")
            if row.get("operator_valid") is not True:
                raise ValueError(f"joint row {sample_id!r} has invalid pixel operator")
            options = row.get("options")
            if not isinstance(options, list) or len(options) != 4:
                raise ValueError(f"joint row {sample_id!r} must carry four options")
            original = str(row.get("pairaug_original_letter", "")).upper()
            mapped = str(row.get("pairaug_mapped_letter", "")).upper()
            branch = str(row["pairaug_branch"])
            if original not in "ABCD" or mapped not in "ABCD" or original == mapped:
                raise ValueError(
                    f"joint row {sample_id!r} has invalid original/mapped letters"
                )
            expected_answer = original if branch == "source" else mapped
            if str(row.get("answer", "")).upper() != expected_answer:
                raise ValueError(
                    f"joint row {sample_id!r} current-input answer does not match branch"
                )
            scalar = row.get("joint_s_swap_0")
            if isinstance(scalar, bool) or scalar is None or not math.isfinite(float(scalar)):
                raise ValueError(f"joint row {sample_id!r} has invalid joint_s_swap_0")
            for field in ("joint_source_logps_0", "joint_null_logps_0"):
                values = row.get(field)
                if (
                    not isinstance(values, list)
                    or len(values) != 4
                    or not all(math.isfinite(float(value)) for value in values)
                ):
                    raise ValueError(
                        f"joint row {sample_id!r} requires four finite {field} values"
                    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_identity() -> dict[str, str]:
    return {rel: sha256_file(PROJECT_ROOT / rel) for rel in CONTRACT_CODE_FILES}


def resolve_hyperparameters(
    objective: str,
    gamma: float | None,
    lambda_null: float | None,
    *,
    margin_swap: float | None = None,
    margin_did: float | None = None,
    lambda_pres: float | None = None,
    eta: float | None = None,
    eta_abs: float | None = None,
) -> tuple[ObjectiveSpec, float, float, AReGConfig | None]:
    """Registry lookup plus per-objective hyperparameter defaults/policy.

    Returns ``(spec, gamma, lambda_null, areg_config)``; ``areg_config`` is
    None for non-AReG objectives.  AReG defaults are the amendment-frozen
    margins (``m_s = 1.0``, ``m_d = 6.5`` total nats) and the initial anchor
    weights (``lambda_pres = eta = 1.0``, ``eta_abs = eta`` per the
    amendment); gamma/lambda_pres/eta/eta_abs remain subject to the
    preregistered gradient-ratio calibration rule on 20-step engineering runs
    and are never tuned against dev accuracy.
    """

    spec = get_objective_spec(objective)  # fail-closed on unregistered names
    resolved_gamma = 0.0 if gamma is None else float(gamma)
    if lambda_null is None:
        # plan5 section 5: lambda_null initial value 0.5 for regcfpo only.
        resolved_lambda_null = 0.5 if spec.name == "regcfpo" else 0.0
    else:
        resolved_lambda_null = float(lambda_null)
    areg_config: AReGConfig | None = None
    if spec.name == "areg_cfpo":
        resolved_eta = 1.0 if eta is None else float(eta)
        areg_config = AReGConfig(
            margin_swap=1.0 if margin_swap is None else float(margin_swap),
            margin_did=6.5 if margin_did is None else float(margin_did),
            lambda_pres=1.0 if lambda_pres is None else float(lambda_pres),
            eta=resolved_eta,
            eta_abs=resolved_eta if eta_abs is None else float(eta_abs),
        )
        spec.validate_hyperparameters(
            gamma=resolved_gamma,
            lambda_null=resolved_lambda_null,
            margin_swap=areg_config.margin_swap,
            margin_did=areg_config.margin_did,
            lambda_pres=areg_config.lambda_pres,
            eta=areg_config.eta,
            eta_abs=areg_config.eta_abs,
        )
    else:
        # Non-AReG objectives take no AReG parameters; forward them raw so
        # the registry policy fails closed on any that were supplied.
        spec.validate_hyperparameters(
            gamma=resolved_gamma,
            lambda_null=resolved_lambda_null,
            margin_swap=margin_swap,
            margin_did=margin_did,
            lambda_pres=lambda_pres,
            eta=eta,
            eta_abs=eta_abs,
        )
    return spec, resolved_gamma, resolved_lambda_null, areg_config


def resolve_joint_config(
    spec: ObjectiveSpec, args: argparse.Namespace
) -> PairAugReferenceTrustConfig | None:
    """Resolve Plan9 joint-only settings and reject silent spillover."""

    names = (
        "joint_gain_delta",
        "joint_gain_weight",
        "joint_trust_weight",
        "joint_source_kl_slack",
        "joint_null_kl_slack",
        "joint_prefix_credit_weight",
        "joint_prefix_credit_temperature",
        "joint_prefix_credit_scope",
        "joint_answer_bridge_weight",
        "joint_answer_bridge_temperature",
    )
    supplied = [name for name in names if getattr(args, name, None) is not None]
    if spec.name != "pairaug_refgain_trust":
        if supplied:
            raise ValueError(
                f"objective {spec.name!r} has no Plan9 joint terms; supplied {supplied}"
            )
        return None
    return PairAugReferenceTrustConfig(
        gain_delta=(
            1.0
            if getattr(args, "joint_gain_delta", None) is None
            else args.joint_gain_delta
        ),
        gain_weight=(
            1.0
            if getattr(args, "joint_gain_weight", None) is None
            else args.joint_gain_weight
        ),
        trust_weight=(
            0.1
            if getattr(args, "joint_trust_weight", None) is None
            else args.joint_trust_weight
        ),
        source_kl_slack=(
            0.02
            if getattr(args, "joint_source_kl_slack", None) is None
            else args.joint_source_kl_slack
        ),
        null_kl_slack=(
            0.02
            if getattr(args, "joint_null_kl_slack", None) is None
            else args.joint_null_kl_slack
        ),
        prefix_credit_weight=(
            0.0
            if getattr(args, "joint_prefix_credit_weight", None) is None
            else args.joint_prefix_credit_weight
        ),
        prefix_credit_temperature=(
            1.0
            if getattr(args, "joint_prefix_credit_temperature", None) is None
            else args.joint_prefix_credit_temperature
        ),
        prefix_credit_scope=(
            "full_completion"
            if getattr(args, "joint_prefix_credit_scope", None) is None
            else args.joint_prefix_credit_scope
        ),
        answer_bridge_weight=(
            0.0
            if getattr(args, "joint_answer_bridge_weight", None) is None
            else args.joint_answer_bridge_weight
        ),
        answer_bridge_temperature=(
            1.0
            if getattr(args, "joint_answer_bridge_temperature", None) is None
            else args.joint_answer_bridge_temperature
        ),
    )


def find_checkpoints(run_dir: Path) -> list[Path]:
    if not run_dir.is_dir():
        return []
    return sorted(
        path for path in run_dir.glob("checkpoint-*") if path.is_dir()
    )


def resolve_resume_checkpoint(
    run_dir: Path, explicit: Path | None, *, max_steps: int | None = None
) -> Path | None:
    """Explicit-only resume policy with a completed-run guard."""

    discovered = find_checkpoints(run_dir)
    if explicit is None:
        if discovered:
            names = [path.name for path in discovered[:5]]
            raise SystemExit(
                f"run directory {run_dir} already contains checkpoints {names}; "
                "resume is never implicit. Pass --resume-from-checkpoint explicitly "
                "(the run contract must match) or choose a fresh --run-dir."
            )
        return None
    checkpoint = explicit.resolve()
    run_dir_resolved = run_dir.resolve()
    if run_dir_resolved not in checkpoint.parents:
        raise SystemExit(f"resume checkpoint {checkpoint} must live inside {run_dir_resolved}")
    if not checkpoint.is_dir() or not checkpoint.name.startswith("checkpoint-"):
        raise SystemExit(f"invalid resume checkpoint: {checkpoint}")
    required = ["trainer_state.json"]
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise SystemExit(f"resume checkpoint {checkpoint} is missing {missing}")
    try:
        trainer_state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid trainer state in {checkpoint}: {exc}") from exc
    global_step = trainer_state.get("global_step")
    if max_steps is not None and max_steps > 0 and global_step is not None:
        try:
            completed_step = int(global_step)
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"invalid global_step {global_step!r} in {checkpoint / 'trainer_state.json'}"
            ) from exc
        if completed_step >= max_steps:
            raise SystemExit(
                f"checkpoint {checkpoint} already reached global_step={completed_step} "
                f">= max_steps={max_steps}; the run is complete and must not resume"
            )
    if not list(checkpoint.glob("rng_state*.pth")):
        raise SystemExit(
            f"resume checkpoint {checkpoint} has no rng_state*.pth; "
            "RNG-state-bound resume is mandatory"
        )
    optimizer_markers = ["optimizer.pt", "scheduler.pt"] + [
        "mp_rank_00_model_states.pt",
    ]
    if not any((checkpoint / name).is_file() for name in optimizer_markers) and not list(
        checkpoint.glob("*model_states*.pt")
    ):
        raise SystemExit(
            f"resume checkpoint {checkpoint} carries no optimizer/scheduler state; "
            "refusing an optimizer-state-free resume"
        )
    return checkpoint


def ensure_run_contract_atomic(
    path: Path, contract: Mapping[str, Any], *, existing_artifacts: Sequence[Path] = ()
) -> dict[str, Any]:
    """Atomic create-then-verify wrapper over ``ensure_run_contract``."""

    if path.exists():
        return ensure_run_contract(path, contract)
    tmp_path = path.with_name(path.name + ".tmp")
    if tmp_path.exists():
        raise SystemExit(f"partial contract {tmp_path} exists; refusing to continue")
    normalized = ensure_run_contract(tmp_path, contract, existing_artifacts=existing_artifacts)
    os.replace(tmp_path, path)
    return ensure_run_contract(path, normalized)


def build_run_contract(
    *,
    args: argparse.Namespace,
    spec: ObjectiveSpec,
    gamma: float,
    lambda_null: float,
    data_rows: Sequence[Mapping[str, Any]],
    config_audit: Mapping[str, Any],
    areg_config: AReGConfig | None = None,
    joint_config: PairAugReferenceTrustConfig | None = None,
) -> dict[str, Any]:
    """Immutable run identity: model/data/code/seed/optimizer/scheduler/RNG."""

    deepspeed_path = Path(args.deepspeed).resolve() if args.deepspeed else None
    return {
        "schema_version": 1,
        "stage": "stage4_grpo_post_training",
        "objective": spec.name,
        "model": {
            "path": str(Path(args.model_path).resolve()),
            "config_sha256": sha256_file(Path(args.model_path) / "config.json"),
            "config_compat": dict(config_audit),
        },
        "data": {
            "file": str(Path(args.data_file).resolve()),
            "sha256": sha256_file(args.data_file),
            "num_rows": len(data_rows),
            "features": [list(item) for item in STAGE4_FEATURES_SPEC],
            "cycle_ordered": manifest_uses_cycle_order(data_rows),
        },
        "code": code_identity(),
        "seed": args.seed,
        "rng": {"seed": args.seed, "data_seed": args.seed, "resume_requires_rng_state": True},
        "optimizer": {
            "name": args.optim,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "adam_beta1": args.adam_beta1,
            "adam_beta2": args.adam_beta2,
            "max_grad_norm": args.max_grad_norm,
        },
        "scheduler": {
            "lr_scheduler_type": args.lr_scheduler_type,
            "warmup_steps": args.warmup_steps,
        },
        "grpo": {
            "num_generations": args.num_generations,
            "max_completion_length": args.max_completion_length,
            "beta": args.beta,
            "num_iterations": args.num_iterations,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_steps": args.max_steps,
            "num_train_epochs": args.num_train_epochs,
        },
        "auxiliary": {
            "gamma": gamma,
            "lambda_null": lambda_null,
            "margin_dir": args.margin_dir,
            "margin_null": args.margin_null,
            "weight_clip": args.weight_clip,
            "gate_credit_mode": args.gate_credit_mode,
            "accuracy_reward_index": 0,
            "reward_funcs": ["accuracy", "format"],
            "areg": (
                {
                    "margin_swap": areg_config.margin_swap,
                    "margin_did": areg_config.margin_did,
                    "lambda_pres": areg_config.lambda_pres,
                    "eta": areg_config.eta,
                    "eta_abs": areg_config.eta_abs,
                    "smooth_l1_beta": areg_config.smooth_l1_beta,
                    "score_unit": "total_nats",
                }
                if areg_config is not None
                else None
            ),
            "plan9_joint": (
                {
                    "gain_delta": joint_config.gain_delta,
                    "gain_weight": joint_config.gain_weight,
                    "trust_weight": joint_config.trust_weight,
                    "source_kl_slack": joint_config.source_kl_slack,
                    "null_kl_slack": joint_config.null_kl_slack,
                    "prefix_credit_weight": joint_config.prefix_credit_weight,
                    "prefix_credit_temperature": joint_config.prefix_credit_temperature,
                    "prefix_credit_read_position": "causal_logit_before_option_token",
                    "prefix_credit_transform": "sigmoid(mapped_minus_original/temperature)",
                    "prefix_credit_scope": joint_config.prefix_credit_scope,
                    "answer_bridge_weight": joint_config.answer_bridge_weight,
                    "answer_bridge_temperature": joint_config.answer_bridge_temperature,
                    "answer_bridge_target": "four_way_mapped_option_cross_entropy",
                    "answer_bridge_weighting": "detached_within_group_credit_softmax",
                    "score_unit": "total_nats",
                    "trust_distribution": "four_candidate_normalized",
                    "trust_direction": "KL(theta0||theta)",
                    "dense_gain_ignores_rollout_saturation": True,
                }
                if joint_config is not None
                else None
            ),
            "format_hard_stop": (
                {
                    "baseline": args.format_reward_baseline,
                    "max_drop": args.format_stop_drop,
                    "window": args.format_stop_window,
                }
                if spec.name == "areg_cfpo"
                else None
            ),
        },
        "runtime": {
            "attn_implementation": args.attn_implementation,
            "torch_dtype": "bfloat16",
            "deepspeed": str(deepspeed_path) if deepspeed_path else None,
            "deepspeed_sha256": sha256_file(deepspeed_path) if deepspeed_path else None,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "freeze_vision_modules": args.freeze_vision_modules,
            "gradient_checkpointing": args.gradient_checkpointing,
            "save_steps": args.save_steps,
            "save_total_limit": args.save_total_limit,
            "checkpoint_strategy": (
                "none_final_model_only"
                if getattr(args, "no_training_checkpoints", False)
                else "steps_with_optimizer_state_plus_final_model"
            ),
        },
        "exposure_audit": (
            compute_exposure_audit(data_rows)
            if manifest_uses_cycle_order(data_rows)
            else None
        ),
    }


def prepare_model_config_view(model_path: Path, run_dir: Path) -> tuple[str, dict[str, Any]]:
    """Apply the 4.53->4.49 config repair BEFORE model loading.

    Returns ``(model_id_for_trainer, audit_record)``.  When the compatibility
    path applies, a run-local symlink view is created: every checkpoint file
    is symlinked except ``config.json``, which is rewritten with the repaired
    config.  The original checkpoint is never modified.
    """

    from regcfpo.qwen_compat import load_qwen_config_with_compat

    result = load_qwen_config_with_compat(model_path)
    record = result.to_record()
    if not result.compatibility_applied:
        return str(model_path), record
    view = run_dir / "model_config_view"
    if view.exists():
        existing = json.loads((view / "config.json").read_text(encoding="utf-8"))
        repaired = json.loads(result.config.to_json_string())
        if existing != repaired:
            raise SystemExit(f"existing model config view {view} does not match the repair")
        return str(view), record
    view.mkdir(parents=True)
    for item in sorted(model_path.iterdir()):
        if item.name == "config.json":
            continue
        os.symlink(item.resolve(), view / item.name)
    result.config.to_json_file(view / "config.json")
    return str(view), record


class FormatHardStopMonitor:
    """Rolling-window format-reward hard stop (amendment first_class_monitors).

    Rule: within the 348-step leg, if the rolling mean of the per-step format
    reward drops more than ``max_drop`` below the frozen theta0 ``baseline``,
    the leg stops immediately.  The window (default 20 steps) makes the stop
    robust to single-group noise (one group = 4 completions, so one malformed
    completion moves a single step by 25 points); the trade-off is that the
    stop can trigger no earlier than step ``window``.  Pure logic lives here
    for CPU testability; the TrainerCallback adapter is built lazily by
    ``build_format_hard_stop_callback``.
    """

    def __init__(self, *, baseline: float, max_drop: float = 0.01, window: int = 20) -> None:
        if not math.isfinite(baseline) or not 0.0 <= baseline <= 1.0:
            raise ValueError(f"format baseline must lie in [0, 1], got {baseline!r}")
        if not math.isfinite(max_drop) or max_drop <= 0.0:
            raise ValueError(f"max_drop must be positive and finite, got {max_drop!r}")
        if not isinstance(window, int) or isinstance(window, bool) or window < 1:
            raise ValueError(f"window must be a positive integer, got {window!r}")
        self.baseline = float(baseline)
        self.max_drop = float(max_drop)
        self.window = window
        self.values: list[float] = []

    def observe(self, format_reward: float) -> bool:
        """Record one step's mean format reward; True means STOP now."""

        if not math.isfinite(format_reward):
            raise FloatingPointError(
                f"format reward is non-finite ({format_reward!r}); refusing to continue"
            )
        self.values.append(float(format_reward))
        self.values = self.values[-self.window :]
        if len(self.values) < self.window:
            return False
        rolling_mean = sum(self.values) / len(self.values)
        return self.baseline - rolling_mean > self.max_drop


def build_format_hard_stop_callback(monitor: FormatHardStopMonitor) -> Any:
    """TrainerCallback adapter over FormatHardStopMonitor (lazy transformers).

    Reads the vendored per-step ``rewards/format_reward`` log entry; on a
    stop decision it sets ``control.should_training_stop`` and prints an
    auditable line to stdout.
    """

    from transformers import TrainerCallback

    class _FormatHardStopCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "rewards/format_reward" not in logs:
                return
            if monitor.observe(float(logs["rewards/format_reward"])):
                recent = sum(monitor.values) / len(monitor.values)
                print(
                    "FORMAT HARD STOP: rolling "
                    f"{monitor.window}-step format reward {recent:.5f} dropped "
                    f"more than {monitor.max_drop} below the theta0 baseline "
                    f"{monitor.baseline} (amendment first_class_monitors hard_stop)",
                    flush=True,
                )
                control.should_training_stop = True

    return _FormatHardStopCallback()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-file", type=Path, required=True, help="Stage-4 JSONL manifest")
    parser.add_argument("--image-root", type=Path, required=True, help="image root for relative paths")
    parser.add_argument("--model-path", type=Path, required=True, help="local SpatialLadder-3B checkpoint")
    parser.add_argument("--run-dir", type=Path, required=True, help="dedicated run directory")
    parser.add_argument("--objective", required=True, help=f"one of {list(registered_objectives())}")
    parser.add_argument("--gamma", type=float, default=None, help="auxiliary loss scale")
    parser.add_argument("--lambda-null", type=float, default=None, help="null-anchor weight")
    parser.add_argument("--margin-dir", type=float, default=1.0)
    parser.add_argument("--margin-null", type=float, default=1.0)
    parser.add_argument("--weight-clip", type=float, default=4.0)
    parser.add_argument(
        "--gate-credit-mode",
        choices=GATE_CREDIT_MODES,
        default="legacy_closed_form_normalized",
        help=(
            "V1 keeps closed-form credit plus batch normalization; plan9 V2 "
            "must explicitly select monotonic_k_over_g_raw"
        ),
    )
    parser.add_argument(
        "--margin-swap",
        type=float,
        default=None,
        help="AReG m_s (total nats); default 1.0 (amendment-frozen)",
    )
    parser.add_argument(
        "--margin-did",
        type=float,
        default=None,
        help="AReG m_d (total nats); default 6.5 (amendment-frozen, quantile-calibrated)",
    )
    parser.add_argument(
        "--lambda-pres",
        type=float,
        default=None,
        help="AReG preservation anchor weight; default 1.0 (calibrated by the "
        "gradient-ratio rule on 20-step engineering runs only)",
    )
    parser.add_argument("--eta", type=float, default=None,
                        help="AReG factual-anchor weight; default 1.0")
    parser.add_argument("--eta-abs", type=float, default=None,
                        help="AReG absolute-anchor weight; default = eta (amendment)")
    parser.add_argument(
        "--joint-gain-delta", type=float, default=None,
        help="Plan9 joint target increase in swapped mapped-vs-original margin",
    )
    parser.add_argument(
        "--joint-gain-weight", type=float, default=None,
        help="Plan9 joint reference-gain coefficient inside the auxiliary bracket",
    )
    parser.add_argument(
        "--joint-trust-weight", type=float, default=None,
        help="Plan9 joint source+null trust coefficient inside the auxiliary bracket",
    )
    parser.add_argument(
        "--joint-source-kl-slack", type=float, default=None,
        help="Plan9 source KL(theta0||theta) trust-region slack",
    )
    parser.add_argument(
        "--joint-null-kl-slack", type=float, default=None,
        help="Plan9 null KL(theta0||theta) trust-region slack",
    )
    parser.add_argument(
        "--joint-prefix-credit-weight", type=float, default=None,
        help=(
            "Plan9 on-policy swap reward coefficient for the continuous "
            "reasoning-prefix mapped-vs-original credit; default 0 (disabled)"
        ),
    )
    parser.add_argument(
        "--joint-prefix-credit-temperature", type=float, default=None,
        help="temperature for sigmoid(mapped-minus-original prefix logit margin); default 1",
    )
    parser.add_argument(
        "--joint-prefix-credit-scope",
        choices=PREFIX_CREDIT_SCOPES,
        default=None,
        help=(
            "full_completion reproduces Plan9 V4; reasoning_all_wrong routes "
            "prefix advantage only before <answer> in valid all-wrong groups"
        ),
    )
    parser.add_argument(
        "--joint-answer-bridge-weight",
        type=float,
        default=None,
        help="Plan10 four-way mapped-answer bridge coefficient; default 0",
    )
    parser.add_argument(
        "--joint-answer-bridge-temperature",
        type=float,
        default=None,
        help="temperature for detached within-group credit-softmax bridge weights; default 1",
    )
    parser.add_argument(
        "--format-reward-baseline",
        type=float,
        default=None,
        help="theta0 format-reward baseline for the AReG hard stop (required "
        "for areg_cfpo runs; the amendment hard-stops the leg when the "
        "rolling format reward drops > --format-stop-drop below it)",
    )
    parser.add_argument("--format-stop-drop", type=float, default=0.01,
                        help="AReG format hard-stop drop threshold (1 point = 0.01)")
    parser.add_argument("--format-stop-window", type=int, default=20,
                        help="AReG format hard-stop rolling window in steps")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-iterations", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--beta", type=float, default=0.0, help="GRPO KL coefficient")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=float, default=1)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="maximum rolling checkpoints retained in a run directory",
    )
    parser.add_argument(
        "--no-training-checkpoints",
        action="store_true",
        help=(
            "disable Trainer optimizer/scheduler checkpoints while still saving "
            "run-dir/final after successful training; the run cannot be resumed"
        ),
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=("sdpa", "eager"),
        help="flash_attention_2 is rejected: not installed (preregistered SDPA deviation)",
    )
    parser.add_argument("--deepspeed", type=Path, default=DEFAULT_DEEPSPEED_CONFIG)
    parser.add_argument("--no-deepspeed", action="store_true", help="single-process debug path")
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--freeze-vision-modules", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None,
                        help="explicit resume; discovery alone never resumes")
    parser.add_argument("--formal", action="store_true",
                        help="mark this as a formal-scope run; requires "
                        "FORMAL_TRAINING_AUTHORIZED=true (also forced when "
                        "--max-steps > 348; the (50, 348] diagnostic scope "
                        "requires OBJECTIVE_V2_DIAGNOSTIC_AUTHORIZED=true)")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate and print the resolved run configuration only")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


def require_training_authorization(*, formal: bool, max_steps: int) -> None:
    """Fail-closed three-tier training authorization (plan7 section 11).

    Every training start requires ``TRAINING_SMOKE_AUTHORIZED=true``.
    Diagnostic-scope runs (``--max-steps`` in ``(50, 348]`` without
    ``--formal``) additionally require ``OBJECTIVE_V2_DIAGNOSTIC_AUTHORIZED
    =true`` (amendment ``asymmetric_directional_credit_revision_20260826``;
    348 = one full balanced cycle of the frozen diagnostic matrix).
    Formal-scope runs (``--formal`` or ``--max-steps > 348``) require
    ``FORMAL_TRAINING_AUTHORIZED=true``.  A refusal prints the current
    values of every consulted variable so the denial is auditable.
    """

    smoke = os.environ.get("TRAINING_SMOKE_AUTHORIZED")
    if smoke != "true":
        raise SystemExit(
            "TRAINING_SMOKE_AUTHORIZED is not 'true'; this entry point grants no "
            "training authorization. Use --dry-run to inspect the configuration."
        )
    if formal or max_steps > 348:
        formal_authorized = os.environ.get("FORMAL_TRAINING_AUTHORIZED")
        if formal_authorized != "true":
            raise SystemExit(
                "formal-scope training requested (--formal or --max-steps > 348: "
                f"formal={formal}, max_steps={max_steps}) but "
                "FORMAL_TRAINING_AUTHORIZED is not 'true'; refusing to start. "
                f"TRAINING_SMOKE_AUTHORIZED={smoke!r}, "
                f"FORMAL_TRAINING_AUTHORIZED={formal_authorized!r}"
            )
    elif max_steps > 50:
        diagnostic_authorized = os.environ.get("OBJECTIVE_V2_DIAGNOSTIC_AUTHORIZED")
        if diagnostic_authorized != "true":
            raise SystemExit(
                "diagnostic-scope training requested (--max-steps in (50, 348] "
                f"without --formal: max_steps={max_steps}) but "
                "OBJECTIVE_V2_DIAGNOSTIC_AUTHORIZED is not 'true'; refusing to "
                f"start. TRAINING_SMOKE_AUTHORIZED={smoke!r}, "
                f"OBJECTIVE_V2_DIAGNOSTIC_AUTHORIZED={diagnostic_authorized!r}"
            )


def run(
    args: argparse.Namespace,
    *,
    extra_callbacks: Sequence[Any] = (),
    trainer_class: type | None = None,
):
    """Resolve, validate, contract, and train; returns the trainer.

    ``extra_callbacks`` and ``trainer_class`` are instrumentation seams for
    the E1 engineering-smoke driver (timing/memory/step recording); the
    production CLI path (``main``) passes neither.  The authorization gate
    below applies to every caller.
    """

    if args.no_deepspeed:
        args.deepspeed = None
    if args.deepspeed is not None and not Path(args.deepspeed).is_file():
        raise SystemExit(f"deepspeed config not found: {args.deepspeed}")
    if args.num_generations < 2:
        raise SystemExit("--num-generations must be >= 2")
    if args.save_steps < 1:
        raise SystemExit("--save-steps must be >= 1")
    if args.save_total_limit < 1:
        raise SystemExit("--save-total-limit must be >= 1")
    if args.no_training_checkpoints and args.resume_from_checkpoint is not None:
        raise SystemExit(
            "--no-training-checkpoints cannot be combined with "
            "--resume-from-checkpoint"
        )
    spec, gamma, lambda_null, areg_config = resolve_hyperparameters(
        args.objective,
        args.gamma,
        args.lambda_null,
        margin_swap=args.margin_swap,
        margin_did=args.margin_did,
        lambda_pres=args.lambda_pres,
        eta=args.eta,
        eta_abs=args.eta_abs,
    )
    joint_config = resolve_joint_config(spec, args)
    if (
        joint_config is not None
        and joint_config.prefix_credit_weight > 0.0
        and args.num_iterations != 1
    ):
        raise SystemExit(
            "Plan9 on-policy prefix credit requires --num-iterations 1; "
            "multi-iteration PPO would recompute current-policy credit on stale rollouts"
        )

    data_rows = load_data_rows(args.data_file)
    cycle_ordered = manifest_uses_cycle_order(data_rows)
    if spec.name == "areg_cfpo" and not cycle_ordered:
        raise SystemExit(
            "objective areg_cfpo requires the merged directional+replay manifest "
            "with row_kind/cycle_position and the areg_* theta0 reference columns "
            "(amendment asymmetric_directional_credit_revision_20260826); the "
            "given manifest is in the legacy directional-only format"
        )
    if spec.name == "areg_cfpo" and not args.dry_run:
        if args.format_reward_baseline is None:
            raise SystemExit(
                "objective areg_cfpo requires --format-reward-baseline (theta0 "
                "format reward) for the amendment's format hard stop"
            )
    if spec.name == "pairaug_refgain_trust":
        if not cycle_ordered:
            raise SystemExit(
                "pairaug_refgain_trust requires a cycle-position joint manifest"
            )
        validate_plan9_joint_contract(data_rows, args)
    if not Path(args.image_root).is_dir():
        raise SystemExit(f"image root not found: {args.image_root}")
    config_audit = _audit_model_config(args.model_path)

    resume = resolve_resume_checkpoint(
        args.run_dir, args.resume_from_checkpoint, max_steps=args.max_steps
    )
    contract = build_run_contract(
        args=args,
        spec=spec,
        gamma=gamma,
        lambda_null=lambda_null,
        data_rows=data_rows,
        config_audit=config_audit,
        areg_config=areg_config,
        joint_config=joint_config,
    )

    if args.dry_run:
        summary = {
            "objective": spec.name,
            "gamma": gamma,
            "lambda_null": lambda_null,
            "areg": (
                {
                    "margin_swap": areg_config.margin_swap,
                    "margin_did": areg_config.margin_did,
                    "lambda_pres": areg_config.lambda_pres,
                    "eta": areg_config.eta,
                    "eta_abs": areg_config.eta_abs,
                }
                if areg_config is not None
                else None
            ),
            "plan9_joint": (
                {
                    "gain_delta": joint_config.gain_delta,
                    "gain_weight": joint_config.gain_weight,
                    "trust_weight": joint_config.trust_weight,
                    "source_kl_slack": joint_config.source_kl_slack,
                    "null_kl_slack": joint_config.null_kl_slack,
                    "prefix_credit_weight": joint_config.prefix_credit_weight,
                    "prefix_credit_temperature": joint_config.prefix_credit_temperature,
                    "prefix_credit_scope": joint_config.prefix_credit_scope,
                    "answer_bridge_weight": joint_config.answer_bridge_weight,
                    "answer_bridge_temperature": joint_config.answer_bridge_temperature,
                }
                if joint_config is not None
                else None
            ),
            "rows": len(data_rows),
            "cycle_ordered": cycle_ordered,
            "run_dir": str(args.run_dir),
            "resume_from_checkpoint": str(resume) if resume else None,
            "training_authorized": os.environ.get("TRAINING_SMOKE_AUTHORIZED") == "true",
            "contract": contract,
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
        return None

    require_training_authorization(formal=args.formal, max_steps=args.max_steps)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    existing_artifacts = [
        path for path in sorted(args.run_dir.iterdir()) if path.name != "run_contract.json"
    ]
    ensure_run_contract_atomic(
        args.run_dir / "run_contract.json", contract, existing_artifacts=existing_artifacts
    )
    model_id, _audit = prepare_model_config_view(args.model_path, args.run_dir)
    return _run_training(
        args, spec, gamma, lambda_null, data_rows, model_id, resume,
        areg_config=areg_config, joint_config=joint_config,
        cycle_ordered=cycle_ordered,
        extra_callbacks=extra_callbacks, trainer_class=trainer_class,
    )


def _audit_model_config(model_path: Path) -> dict[str, Any]:
    """Run the Qwen config compatibility audit (CPU-only, no weights)."""

    from regcfpo.qwen_compat import load_qwen_config_with_compat

    return load_qwen_config_with_compat(model_path).to_record()


def _run_training(
    args: argparse.Namespace,
    spec: ObjectiveSpec,
    gamma: float,
    lambda_null: float,
    data_rows: list[dict[str, Any]],
    model_id: str,
    resume: Path | None,
    *,
    areg_config: AReGConfig | None = None,
    joint_config: PairAugReferenceTrustConfig | None = None,
    cycle_ordered: bool = False,
    extra_callbacks: Sequence[Any] = (),
    trainer_class: type | None = None,
):
    """Heavy path: vendored trainer construction and ``train()``.

    All training-environment imports live here so the module stays importable
    in the inference environment for tests and ``--dry-run``.  Returns the
    trainer so instrumented callers (the E1 engineering driver) can inspect
    final state.

    ``trainer_class`` defaults to ``ReGCFPOTrainer``; on a cycle-position
    (merged) manifest it defaults to ``CycleOrderedReGCFPOTrainer`` instead —
    the deterministic cycle-exposure schedule required by the amendment,
    installed through this seam for EVERY objective (including the
    continued-GRPO control leg, single-scientific-variable rule).  An
    explicitly passed ``trainer_class`` always wins.
    """

    from open_r1.vlm_modules.qwen_module import Qwen2VLModule
    from open_r1.trainer.grpo_config import GRPOConfig

    from regcfpo.qwen_compat import validate_qwen_weight_tying
    from regcfpo.training.auxiliary import GateConfig
    from regcfpo.training.trainer import CycleOrderedReGCFPOTrainer, ReGCFPOTrainer

    if trainer_class is not None:
        resolved_trainer_class = trainer_class
    elif cycle_ordered:
        resolved_trainer_class = CycleOrderedReGCFPOTrainer
    else:
        resolved_trainer_class = ReGCFPOTrainer

    dataset = build_stage4_dataset(data_rows, image_root=args.image_root)

    training_args = GRPOConfig(
        output_dir=str(args.run_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        num_iterations=args.num_iterations,
        beta=args.beta,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        max_grad_norm=args.max_grad_norm,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        seed=args.seed,
        data_seed=args.seed,
        bf16=True,
        deepspeed=str(args.deepspeed) if args.deepspeed else None,
        logging_steps=args.logging_steps,
        save_strategy=("no" if args.no_training_checkpoints else "steps"),
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=False,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=0,
    )

    trainer = resolved_trainer_class(
        model=model_id,
        reward_funcs=[accuracy_reward, format_reward],
        args=training_args,
        vlm_module=Qwen2VLModule(),
        train_dataset=dataset,
        objective_spec=spec,
        gamma=gamma,
        lambda_null=lambda_null,
        gate_config=GateConfig(
            margin_dir=args.margin_dir,
            margin_null=args.margin_null,
            weight_clip=args.weight_clip,
            credit_mode=args.gate_credit_mode,
        ),
        areg_config=areg_config,
        joint_config=joint_config,
        accuracy_reward_index=0,
        format_reward_index=1,
        freeze_vision_modules=args.freeze_vision_modules,
        attn_implementation=args.attn_implementation,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
    )
    for callback in extra_callbacks:
        trainer.add_callback(callback)
    if spec.name == "areg_cfpo":
        trainer.add_callback(
            build_format_hard_stop_callback(
                FormatHardStopMonitor(
                    baseline=args.format_reward_baseline,
                    max_drop=args.format_stop_drop,
                    window=args.format_stop_window,
                )
            )
        )

    tying = validate_qwen_weight_tying(trainer.model)
    tying_path = args.run_dir / "weight_tying_audit.json"
    tmp_path = tying_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(tying, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, tying_path)

    trainer.train(resume_from_checkpoint=str(resume) if resume is not None else None)
    trainer.save_model(str(args.run_dir / "final"))
    return trainer


if __name__ == "__main__":
    raise SystemExit(main())
