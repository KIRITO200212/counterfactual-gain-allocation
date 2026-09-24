"""Equivalence tests for the train-side official na_reward port.

``regcfpo.training.rewards.official_stage3_na_reward`` must be logically
identical to the evaluation-side port
``scripts/e3_rollout_eval.py:official_stage3_na_reward`` (itself a
bug-for-bug port of the vendored stage3 ``na_reward``).  The numeric replay
rows of the AReG merged manifest are scored with it, so any drift would
desync training rewards from evaluation scoring.  The module also pins the
replay dual-form contract (``classify_replay_answer_form``): letter-answer
rows with options route to the MCA letter match, never to ``na_reward``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import e3_rollout_eval as e3  # noqa: E402
import train_stage4  # noqa: E402
from regcfpo.training.rewards import official_stage3_na_reward  # noqa: E402

#: Fixture grid covering exact match, unparseable sides, the linspace quirk
#: regime, the zero-target documented deviation, multi-block, whitespace and
#: trailing-period normalization, and negative/fractional values.
FIXTURE_GRID = [
    ("<answer>41</answer>", "41"),
    ("<answer>many</answer>", "41"),
    ("<answer>41</answer>", "many"),
    ("no tags at all", "41"),
    ("<answer>14</answer>", "10"),
    ("<answer>10.4</answer>", "10"),
    ("<answer>0</answer>", "0"),
    ("<answer>1</answer>", "0"),
    ("<answer>0.0</answer>", "0"),
    ("<answer>5</answer> noise <answer>7</answer>", "7"),
    ("<answer>5</answer> noise <answer>7</answer>", "5"),
    ("<answer> 42. </answer>", "42"),
    ("<answer>-3</answer>", "-3"),
    ("<answer>-3</answer>", "3"),
    ("<answer>2.5</answer>", "2.55"),
    ("<think>x</think>\n<answer>8</answer>", "8"),
    ("<answer>100</answer>", "-100"),
    ("<answer></answer>", "41"),
]


@pytest.mark.parametrize("completion,answer", FIXTURE_GRID)
def test_na_reward_matches_eval_side_port(completion, answer):
    assert official_stage3_na_reward(completion, answer) == pytest.approx(
        e3.official_stage3_na_reward(completion, answer), abs=1e-12
    )


def test_na_reward_hardcoded_reference_values():
    # Mirrors tests/test_e3_rollout_eval.py::TestNaReward expectations.
    assert official_stage3_na_reward("<answer>41</answer>", "41") == 1.0
    assert official_stage3_na_reward("<answer>many</answer>", "41") == 0.0
    assert official_stage3_na_reward("<answer>14</answer>", "10") == pytest.approx(0.3)
    assert official_stage3_na_reward("<answer>10.4</answer>", "10") == pytest.approx(1.0)
    assert official_stage3_na_reward("<answer>0</answer>", "0") == 1.0
    assert official_stage3_na_reward("<answer>1</answer>", "0") == 0.0


def _conversation_completion(text: str):
    return [[{"role": "assistant", "content": text}]]


def test_accuracy_reward_routes_replay_rows_to_na_reward():
    completions = _conversation_completion("<think>x</think><answer>14</answer>")
    rewards = train_stage4.accuracy_reward(
        completions,
        answer_letter=[""],
        row_kind=["replay"],
        answer=["10"],
    )
    assert rewards == [pytest.approx(0.3)]


def test_accuracy_reward_replay_requires_answer():
    completions = _conversation_completion("<think>x</think><answer>14</answer>")
    with pytest.raises(ValueError, match="answer"):
        train_stage4.accuracy_reward(completions, row_kind=["replay"], answer=[""])


def test_accuracy_reward_mixed_batch_routing():
    completions = (
        _conversation_completion("<think>x</think><answer>A</answer>")
        + _conversation_completion("<think>x</think><answer>10.4</answer>")
        + _conversation_completion("<think>x</think><answer>B</answer>")
    )
    rewards = train_stage4.accuracy_reward(
        completions,
        answer_letter=["A", "", "A"],
        row_kind=["directional", "replay", "directional"],
        answer=["", "10", ""],
    )
    assert rewards == [1.0, pytest.approx(1.0), 0.0]


def test_accuracy_reward_legacy_calls_unchanged_without_row_kind():
    completions = _conversation_completion("<think>x</think><answer>A.</answer>")
    assert train_stage4.accuracy_reward(completions, answer_letter=["A"]) == [1.0]


# ---------------------------------------------------------------------------
# Replay dual-form contract (169 numeric + 5 MC rows in the real v3 pool)
# ---------------------------------------------------------------------------


def test_classify_replay_answer_form_numeric():
    # numeric rows: finite float answer, options absent in all its spellings
    assert train_stage4.classify_replay_answer_form("41", None) == "numeric"
    assert train_stage4.classify_replay_answer_form("0.3", []) == "numeric"
    assert train_stage4.classify_replay_answer_form("-3.5", "None") == "numeric"
    assert train_stage4.classify_replay_answer_form(90, "") == "numeric"
    # a numeric answer keeps the numeric form even if options are present
    assert train_stage4.classify_replay_answer_form("2", ["A. 1", "B. 2"]) == "numeric"


def test_classify_replay_answer_form_mc():
    options = ["A. door", "B. window"]
    assert train_stage4.classify_replay_answer_form("B", options) == "mc"
    # case-insensitive, whitespace tolerated
    assert train_stage4.classify_replay_answer_form(" b ", options) == "mc"
    assert train_stage4.classify_replay_answer_form(
        "D", ["A. x", "B. x", "C. x", "D. x"]
    ) == "mc"


def test_classify_replay_answer_form_fail_closed():
    # letter without options
    with pytest.raises(ValueError, match="MC letter"):
        train_stage4.classify_replay_answer_form("C", None)
    with pytest.raises(ValueError, match="MC letter"):
        train_stage4.classify_replay_answer_form("C", "None")
    # letter out of range of the options list
    with pytest.raises(ValueError, match="out of range"):
        train_stage4.classify_replay_answer_form("D", ["A. door", "B. window"])
    # non-letter non-float
    with pytest.raises(ValueError, match="neither"):
        train_stage4.classify_replay_answer_form("many", None)
    with pytest.raises(ValueError, match="neither"):
        train_stage4.classify_replay_answer_form("", None)
    with pytest.raises(ValueError, match="neither"):
        train_stage4.classify_replay_answer_form(None, None)
    # non-finite numerics are refused
    with pytest.raises(ValueError, match="finite"):
        train_stage4.classify_replay_answer_form("nan", None)
    with pytest.raises(ValueError, match="finite"):
        train_stage4.classify_replay_answer_form("inf", None)


def test_accuracy_reward_routes_mc_replay_rows_to_letter_match():
    options = ["A. whiteboard\n", "B. podium\n", "C. cabinet\n", "D. chair\n"]
    # correct letter scores 1.0 — only possible via the MCA path (na_reward
    # would fail to float-parse the "D" completion and return 0.0)
    rewards = train_stage4.accuracy_reward(
        _conversation_completion("<think>x</think><answer>D</answer>"),
        answer_letter=[""],
        row_kind=["replay"],
        answer=["D"],
        options=[options],
    )
    assert rewards == [1.0]
    # wrong letter scores 0.0
    rewards = train_stage4.accuracy_reward(
        _conversation_completion("<think>x</think><answer>A.</answer>"),
        answer_letter=[""],
        row_kind=["replay"],
        answer=["D"],
        options=[options],
    )
    assert rewards == [0.0]


def test_accuracy_reward_mixed_batch_dual_form_routing():
    completions = (
        _conversation_completion("<think>x</think><answer>10.4</answer>")
        + _conversation_completion("<think>x</think><answer>B</answer>")
        + _conversation_completion("<think>x</think><answer>B</answer>")
    )
    rewards = train_stage4.accuracy_reward(
        completions,
        answer_letter=["", "", "B"],
        row_kind=["replay", "replay", "directional"],
        answer=["10", "B", ""],
        options=[None, ["A. door", "B. window"], None],
    )
    # numeric replay -> na_reward tolerance; MC replay + directional -> letters
    assert rewards == [pytest.approx(1.0), 1.0, 1.0]


def test_accuracy_reward_malformed_replay_rows_fail_closed():
    completions = _conversation_completion("<think>x</think><answer>B</answer>")
    # letter answer but options missing
    with pytest.raises(ValueError, match="MC letter"):
        train_stage4.accuracy_reward(
            completions, row_kind=["replay"], answer=["B"], options=[None]
        )
    # non-letter non-float answer
    with pytest.raises(ValueError, match="neither"):
        train_stage4.accuracy_reward(
            completions, row_kind=["replay"], answer=["many"], options=[None]
        )
