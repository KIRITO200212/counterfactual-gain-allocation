#!/usr/bin/env python3
"""Loss-aligned answer-proxy Gate V3 readout (pure CPU, no model code).

Implements the frozen ``LOSS_ALIGNED_PROXY_GATE_V3`` from amendment
``loss_aligned_proxy_amendment_20260822`` (configs/phase1_prereg.yaml;
authority results/phase1/phase1_plan5.md).  The frozen definitions are:

- Delta_proxy   = logp(y_cf) - logp(y)     (teacher-forced, per arm)
- Delta_rollout = log((n(y_cf) + 0.5) / (n(y) + 0.5))

so a larger mapped-answer count maps to a *positive* Delta_rollout (the
amendment corrects the inverted ratio in plan5 section 3.3).  A tie row is
exactly ``n(y_cf) == n(y)`` (equivalently Delta_rollout == 0.0); tie rows are
excluded from the conditional sign agreement but always reported separately.

Gates (evaluated per arm on the overall stratum):
  A. answered_fraction >= 0.95 per newly generated branch (swap/null);
  B. factual top-1 agreement >= 0.90 (rollout argmax ties form a set; a proxy
     top-1 inside the set counts as a hit; tie rows are reported);
  C. conditional sign agreement on non-tie rows: swap >= 0.90, null >= 0.95;
  D. swap scene-aggregated Spearman >= 0.80 with scene-bootstrap percentile
     LCB95 > 0.70 (row-level Spearman is descriptive; null is not gated on
     correlation);
  E. four-candidate Spearman (raw / empirical ceiling / ceiling-normalized /
     top-1-in-rollout-argmax-set) is descriptive only.

Arm selection is the frozen rule: full_template_span passes -> primary;
else answer_only passes -> primary; else STOP.

Input validation is fail-closed: run rows must match the manifest exactly
(unique sample_ids, each input file ordered consistently with the manifest,
union equal to the manifest cohort).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:  # scipy is preferred for the midrank Spearman; the pure fallback is unit-tested
    from scipy.stats import spearmanr as _scipy_spearmanr
except ImportError:  # pragma: no cover - environment dependent
    _scipy_spearmanr = None

ARMS = ("answer_only", "full_template_span")
BRANCH_FACTUAL = "factual"
BRANCH_SWAP = "pixel_pair_slot_swap"
BRANCH_NULL = "canonical_resampling_return"
BRANCHES = (BRANCH_FACTUAL, BRANCH_SWAP, BRANCH_NULL)
NEWLY_GENERATED_BRANCHES = (BRANCH_SWAP, BRANCH_NULL)
STRATA = ("overall", "axis", "diagonal", "LF_RB", "LB_RF")

SMOOTHING = 0.5
THRESHOLDS = {
    "answered_fraction_min": 0.95,
    "factual_top1_agreement_min": 0.90,
    "swap_conditional_sign_min": 0.90,
    "null_conditional_sign_min": 0.95,
    "swap_scene_spearman_min": 0.80,
    "swap_scene_spearman_lcb95_min": 0.70,
}
BOOTSTRAP_NUM_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260822
BOOTSTRAP_ALPHA = 0.05

DEFAULT_MANIFEST = Path("manifests/v3_final/v3_directional_train.jsonl")


class ReadoutError(ValueError):
    """Raised when the V3 run output is unsafe to interpret."""


# ---------------------------------------------------------------------------
# small io helpers (kept local so this module stays importable without torch)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ReadoutError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ReadoutError(f"{path}:{line_number}: expected a JSON object")
            yield value


# ---------------------------------------------------------------------------
# frozen row semantics


def relation_family(answer_relation: str, mapped_relation: str) -> str:
    """Map a (answer_relation, mapped_relation) pair to its frozen family."""

    pair = frozenset((answer_relation, mapped_relation))
    if pair == frozenset(("left", "right")):
        return "axis"
    if pair == frozenset(("left-front", "right-back")):
        return "LF_RB"
    if pair == frozenset(("left-back", "right-front")):
        return "LB_RF"
    raise ReadoutError(
        f"unrecognized relation pair: {answer_relation!r} -> {mapped_relation!r}"
    )


def strata_of(row: Mapping[str, Any]) -> list[str]:
    strata = ["overall"]
    relation_class = row["relation_class"]
    if relation_class == "axis":
        strata.append("axis")
    elif relation_class == "diagonal":
        strata.append("diagonal")
        strata.append(str(row["relation_family"]))
    else:
        raise ReadoutError(f"unknown relation_class {relation_class!r}")
    return strata


def smoothed_log_odds(n_mapped: int, n_original: int) -> float:
    """Frozen rollout reference: log((n(y_cf) + 0.5) / (n(y) + 0.5))."""

    return math.log((n_mapped + SMOOTHING) / (n_original + SMOOTHING))


def rollout_delta(counts: Mapping[str, int], mapped: str, original: str) -> float:
    return smoothed_log_odds(counts[mapped], counts[original])


def is_tie(counts: Mapping[str, int], mapped: str, original: str) -> bool:
    """Exact tie predicate: equal integer counts (iff Delta_rollout == 0)."""

    return counts[mapped] == counts[original]


def sign_of(value: float) -> int:
    return int(value > 0) - int(value < 0)


# ---------------------------------------------------------------------------
# midrank Spearman (scipy preferred, pure fallback pinned by unit tests)


def midrank(values: Sequence[float]) -> np.ndarray:
    """Average ranks with ties (1-based midranks), stable for float input."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ReadoutError("midrank requires a non-empty one-dimensional input")
    order = np.argsort(array, kind="stable")
    sorted_values = array[order]
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        stop = start + 1
        while stop < array.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size != y.size or x.size == 0:
        raise ReadoutError("pearson requires two equally sized non-empty vectors")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = math.sqrt(float(x_centered @ x_centered) * float(y_centered @ y_centered))
    if denominator == 0.0:
        return float("nan")
    return float(x_centered @ y_centered / denominator)


def spearman_midrank_pure(x: Sequence[float], y: Sequence[float]) -> float:
    """Midrank Spearman without scipy; constant inputs return NaN."""

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size != y_arr.size or x_arr.size == 0:
        raise ReadoutError("spearman requires two equally sized non-empty vectors")
    if x_arr.std() == 0.0 or y_arr.std() == 0.0:
        return float("nan")
    return _pearson(midrank(x_arr), midrank(y_arr))


def spearman_midrank(x: Sequence[float], y: Sequence[float]) -> float:
    """Midrank Spearman; constant inputs return NaN (excluded by callers)."""

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size != y_arr.size or x_arr.size == 0:
        raise ReadoutError("spearman requires two equally sized non-empty vectors")
    if x_arr.std() == 0.0 or y_arr.std() == 0.0:
        return float("nan")
    if _scipy_spearmanr is not None:
        value = _scipy_spearmanr(x_arr, y_arr).statistic
        return float(value) if value is not None and not np.isnan(value) else float("nan")
    return spearman_midrank_pure(x_arr, y_arr)


def empirical_spearman_ceiling(counts: Mapping[str, int]) -> float:
    """Max Spearman any teacher-forced score vector can reach vs this reference.

    The reference midranks are fixed; by the rearrangement inequality the
    maximum assigns the distinct score ranks 1..n to positions in ascending
    reference-rank order (any assignment within tied positions gives the same
    value).  A constant reference has no defined correlation and returns NaN.
    """

    reference = np.asarray([counts[letter] for letter in sorted(counts)], dtype=np.float64)
    if reference.size == 0:
        raise ReadoutError("empirical ceiling requires a non-empty reference")
    if reference.std() == 0.0:
        return float("nan")
    reference_ranks = midrank(reference)
    n = reference.size
    order = np.argsort(reference_ranks, kind="stable")
    best_score_ranks = np.empty(n, dtype=np.float64)
    best_score_ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    return _pearson(best_score_ranks, reference_ranks)


# ---------------------------------------------------------------------------
# per-row extraction


def _accepted_branch(row: Mapping[str, Any], branch: str) -> Mapping[str, Any] | None:
    data = row.get("branches", {}).get(branch)
    if not isinstance(data, Mapping) or not data.get("accepted"):
        return None
    return data


def extract_row(row: Mapping[str, Any], branch: str, arm: str) -> dict[str, Any] | None:
    """Pull every gate-relevant quantity for one (row, branch, arm)."""

    data = _accepted_branch(row, branch)
    if data is None:
        return None
    rollout = data["rollout"]
    counts = {letter: int(value) for letter, value in rollout["counts"].items()}
    arms = data["arms"]
    letters = sorted(arms)
    scores = {
        letter: float(arms[letter][arm]["length_normalized"]) for letter in letters
    }
    original = row["answer_letter"]
    mapped = row["mapped_answer_letter"]
    per_rollout = rollout.get("per_rollout") or []
    truncation = [bool(entry.get("hit_max_new_tokens")) for entry in per_rollout]
    return {
        "sample_id": row["sample_id"],
        "scene_id": row["scene_id"],
        "letters": letters,
        "counts": counts,
        "scores": scores,
        "delta_proxy": scores[mapped] - scores[original],
        "delta_rollout": rollout_delta(counts, mapped, original),
        "tie": is_tie(counts, mapped, original),
        "answered_fraction": float(rollout["answered_fraction"]),
        "format_valid_fraction": float(rollout["format_valid_fraction"]),
        "truncation_fraction": (
            float(np.mean(truncation)) if truncation else None
        ),
        "num_generations": int(rollout["num_generations"]),
    }


# ---------------------------------------------------------------------------
# gate metric blocks


def completion_gate_metrics(extracted: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Gate A: completion/format/truncation for one newly generated branch."""

    if not extracted:
        return {
            "n_rows": 0,
            "mean_answered_fraction": None,
            "min_answered_fraction": None,
            "rows_below_threshold": 0,
            "mean_format_valid_fraction": None,
            "mean_truncation_fraction": None,
            "answered_fraction_pass": False,
        }
    answered = np.asarray([row["answered_fraction"] for row in extracted])
    formatted = np.asarray([row["format_valid_fraction"] for row in extracted])
    truncated = [row["truncation_fraction"] for row in extracted]
    truncated_values = [value for value in truncated if value is not None]
    mean_answered = float(answered.mean())
    return {
        "n_rows": len(extracted),
        "mean_answered_fraction": mean_answered,
        "min_answered_fraction": float(answered.min()),
        "rows_below_threshold": int(
            (answered < THRESHOLDS["answered_fraction_min"]).sum()
        ),
        "mean_format_valid_fraction": float(formatted.mean()),
        "mean_truncation_fraction": (
            float(np.mean(truncated_values)) if truncated_values else None
        ),
        "answered_fraction_pass": bool(
            mean_answered >= THRESHOLDS["answered_fraction_min"]
        ),
    }


def _top1_argmax_set(counts: Mapping[str, int]) -> set[str]:
    best = max(counts.values())
    return {letter for letter, value in counts.items() if value == best}


def _proxy_top1(scores: Mapping[str, float]) -> str:
    best = max(scores.values())
    return sorted(letter for letter, value in scores.items() if value == best)[0]


def fidelity_metrics(extracted: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Gate B: factual top-1 agreement with rollout argmax-set tie handling."""

    hits = 0
    tie_rows = 0
    for row in extracted:
        argmax_set = _top1_argmax_set(row["counts"])
        if len(argmax_set) > 1:
            tie_rows += 1
        if _proxy_top1(row["scores"]) in argmax_set:
            hits += 1
    n = len(extracted)
    return {
        "n_rows": n,
        "top1_agreement": (hits / n) if n else None,
        "rollout_argmax_tie_rows": tie_rows,
        "rollout_argmax_tie_fraction": (tie_rows / n) if n else None,
        "top1_pass": bool(n) and (hits / n) >= THRESHOLDS["factual_top1_agreement_min"],
    }


def directional_metrics(extracted: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Gate C: tie-aware conditional sign agreement plus tie-row reporting."""

    ties = [row for row in extracted if row["tie"]]
    non_ties = [row for row in extracted if not row["tie"]]
    hits = sum(
        1
        for row in non_ties
        if sign_of(row["delta_proxy"]) == sign_of(row["delta_rollout"])
    )
    n = len(extracted)
    mean_abs_tie = (
        float(np.mean([abs(row["delta_proxy"]) for row in ties])) if ties else None
    )
    mean_abs_non_tie = (
        float(np.mean([abs(row["delta_proxy"]) for row in non_ties]))
        if non_ties
        else None
    )
    return {
        "n_rows": n,
        "n_tie": len(ties),
        "tie_rate": (len(ties) / n) if n else None,
        "n_non_tie": len(non_ties),
        "conditional_sign_agreement": (hits / len(non_ties)) if non_ties else None,
        "mean_abs_proxy_delta_tie_rows": mean_abs_tie,
        "mean_abs_proxy_delta_non_tie_rows": mean_abs_non_tie,
    }


def aggregate_by_scene(
    extracted: list[Mapping[str, Any]],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Average Delta_proxy and Delta_rollout within each scene.

    Scenes are sorted and each scene contributes exactly one point, so a
    multi-question scene carries no extra weight.
    """

    by_scene: dict[str, list[tuple[float, float]]] = {}
    for row in extracted:
        by_scene.setdefault(str(row["scene_id"]), []).append(
            (float(row["delta_proxy"]), float(row["delta_rollout"]))
        )
    scene_ids = sorted(by_scene)
    proxy_means = np.asarray(
        [np.mean([pair[0] for pair in by_scene[scene]]) for scene in scene_ids]
    )
    rollout_means = np.asarray(
        [np.mean([pair[1] for pair in by_scene[scene]]) for scene in scene_ids]
    )
    return scene_ids, proxy_means, rollout_means


def scene_spearman_bootstrap(
    scene_proxy: Sequence[float],
    scene_rollout: Sequence[float],
    *,
    num_resamples: int = BOOTSTRAP_NUM_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = BOOTSTRAP_ALPHA,
) -> dict[str, Any]:
    """Percentile bootstrap over scene-level Spearman points.

    Scenes are resampled with replacement and the Spearman is recomputed on
    the resampled scene points.  Deterministic given ``seed``.  Fewer than two
    scenes are NOT_ESTIMABLE.  Resamples whose resampled vectors are constant
    yield NaN and are excluded from the percentile (counted in the report).
    """

    x = np.asarray(scene_proxy, dtype=np.float64)
    y = np.asarray(scene_rollout, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ReadoutError("scene bootstrap requires equally sized 1-D inputs")
    num_scenes = int(x.size)
    if num_scenes < 2:
        return {
            "estimable": False,
            "reason": "NOT_ESTIMABLE_FEWER_THAN_TWO_SCENES",
            "num_scenes": num_scenes,
            "estimate": None,
            "lcb95": None,
            "num_resamples": int(num_resamples),
            "seed": int(seed),
            "alpha": float(alpha),
            "nan_resamples_excluded": 0,
        }
    estimate = spearman_midrank(x, y)
    rng = np.random.default_rng(seed)
    estimates = np.empty(int(num_resamples), dtype=np.float64)
    for resample in range(int(num_resamples)):
        indices = rng.integers(0, num_scenes, size=num_scenes, endpoint=False)
        estimates[resample] = spearman_midrank(x[indices], y[indices])
    finite = estimates[np.isfinite(estimates)]
    lcb95 = float(np.quantile(finite, alpha, method="linear")) if finite.size else None
    return {
        "estimable": True,
        "reason": None,
        "num_scenes": num_scenes,
        "estimate": estimate if np.isfinite(estimate) else None,
        "lcb95": lcb95,
        "num_resamples": int(num_resamples),
        "seed": int(seed),
        "alpha": float(alpha),
        "nan_resamples_excluded": int(num_resamples - finite.size),
    }


def correlation_metrics(
    extracted: list[Mapping[str, Any]],
    *,
    num_resamples: int = BOOTSTRAP_NUM_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Gate D block: scene-aggregated Spearman plus descriptive row-level."""

    scene_ids, proxy_means, rollout_means = aggregate_by_scene(extracted)
    row_level = (
        spearman_midrank(
            [row["delta_proxy"] for row in extracted],
            [row["delta_rollout"] for row in extracted],
        )
        if extracted
        else float("nan")
    )
    bootstrap = scene_spearman_bootstrap(
        proxy_means, rollout_means, num_resamples=num_resamples, seed=seed
    )
    return {
        "n_rows": len(extracted),
        "num_scenes": len(scene_ids),
        "scene_level": bootstrap,
        "row_level_spearman_descriptive": (
            row_level if np.isfinite(row_level) else None
        ),
        "scene_spearman_pass": bool(
            bootstrap["estimable"]
            and bootstrap["estimate"] is not None
            and bootstrap["estimate"] >= THRESHOLDS["swap_scene_spearman_min"]
        ),
        "scene_lcb95_pass": bool(
            bootstrap["estimable"]
            and bootstrap["lcb95"] is not None
            and bootstrap["lcb95"] > THRESHOLDS["swap_scene_spearman_lcb95_min"]
        ),
    }


def four_candidate_metrics(extracted: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Gate E (descriptive): raw / ceiling / normalized Spearman, top-1 set."""

    raw_rhos: list[float] = []
    ceilings: list[float] = []
    normalized: list[float] = []
    top1_hits = 0
    constant_rows = 0
    for row in extracted:
        letters = row["letters"]
        scores = [row["scores"][letter] for letter in letters]
        reference = [row["counts"][letter] for letter in letters]
        rho = spearman_midrank(scores, reference)
        ceiling = empirical_spearman_ceiling(row["counts"])
        if np.isnan(rho):
            constant_rows += 1
        else:
            raw_rhos.append(rho)
            if not np.isnan(ceiling):
                ceilings.append(ceiling)
                if ceiling > 0:
                    normalized.append(rho / ceiling)
        if _proxy_top1(row["scores"]) in _top1_argmax_set(row["counts"]):
            top1_hits += 1
    n = len(extracted)
    return {
        "n_rows": n,
        "mean_raw_spearman": float(np.mean(raw_rhos)) if raw_rhos else None,
        "rows_excluded_constant": constant_rows,
        "mean_empirical_ceiling": float(np.mean(ceilings)) if ceilings else None,
        "mean_ceiling_normalized_spearman": (
            float(np.mean(normalized)) if normalized else None
        ),
        "top1_in_rollout_argmax_set_fraction": (top1_hits / n) if n else None,
    }


# ---------------------------------------------------------------------------
# strata assembly and gate evaluation


def _extract_all(
    rows: list[Mapping[str, Any]], branch: str, arm: str
) -> tuple[list[dict[str, Any]], int]:
    extracted: list[dict[str, Any]] = []
    rejected = 0
    for row in rows:
        value = extract_row(row, branch, arm)
        if value is None:
            rejected += 1
        else:
            extracted.append(value)
    return extracted, rejected


def compute_stratum_report(
    rows: list[Mapping[str, Any]],
    arm: str,
    *,
    num_resamples: int = BOOTSTRAP_NUM_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    branches: dict[str, Any] = {}
    factual, factual_rejected = _extract_all(rows, BRANCH_FACTUAL, arm)
    branches[BRANCH_FACTUAL] = {
        "rejected_rows": factual_rejected,
        "fidelity": fidelity_metrics(factual),
        "four_candidate_descriptive": four_candidate_metrics(factual),
    }
    for branch in NEWLY_GENERATED_BRANCHES:
        extracted, rejected = _extract_all(rows, branch, arm)
        directional = directional_metrics(extracted)
        if branch == BRANCH_NULL:
            # Descriptive only (amendment gate D): the null branch is judged
            # by gate C plus factual-answer retention, never by correlation.
            directional["original_answer_retention_mean_frequency"] = _null_retention(
                rows, extracted
            )
        branches[branch] = {
            "rejected_rows": rejected,
            "completion": completion_gate_metrics(extracted),
            "directional": directional,
            "correlation": correlation_metrics(
                extracted, num_resamples=num_resamples, seed=seed
            ),
            "four_candidate_descriptive": four_candidate_metrics(extracted),
        }
    return {"n_rows": len(rows), "branches": branches}


def _null_retention(
    rows: list[Mapping[str, Any]], extracted: list[Mapping[str, Any]]
) -> float | None:
    """Descriptive: mean rollout frequency of the original answer on null."""

    answer_by_id = {str(row["sample_id"]): str(row["answer_letter"]) for row in rows}
    values = [
        entry["counts"][answer_by_id[entry["sample_id"]]] / entry["num_generations"]
        for entry in extracted
        if entry["sample_id"] in answer_by_id
    ]
    return float(np.mean(values)) if values else None


def evaluate_arm_gates(overall: Mapping[str, Any]) -> dict[str, Any]:
    """Fail-closed gate evaluation on the overall stratum for one arm."""

    branches = overall["branches"]
    completion = {
        branch: branches[branch]["completion"] for branch in NEWLY_GENERATED_BRANCHES
    }
    gate_a_pass = all(
        block["answered_fraction_pass"] for block in completion.values()
    )
    fidelity = branches[BRANCH_FACTUAL]["fidelity"]
    gate_b_pass = fidelity["top1_pass"]
    sign_swap = branches[BRANCH_SWAP]["directional"]["conditional_sign_agreement"]
    sign_null = branches[BRANCH_NULL]["directional"]["conditional_sign_agreement"]
    gate_c_pass = bool(
        sign_swap is not None
        and sign_swap >= THRESHOLDS["swap_conditional_sign_min"]
        and sign_null is not None
        and sign_null >= THRESHOLDS["null_conditional_sign_min"]
    )
    correlation = branches[BRANCH_SWAP]["correlation"]
    gate_d_pass = bool(
        correlation["scene_spearman_pass"] and correlation["scene_lcb95_pass"]
    )
    return {
        "A_completion": {"branches": completion, "pass": gate_a_pass},
        "B_factual_fidelity": {**fidelity, "pass": gate_b_pass},
        "C_tie_aware_directional": {
            "swap_conditional_sign_agreement": sign_swap,
            "null_conditional_sign_agreement": sign_null,
            "swap_min": THRESHOLDS["swap_conditional_sign_min"],
            "null_min": THRESHOLDS["null_conditional_sign_min"],
            "pass": gate_c_pass,
        },
        "D_loss_aligned_correlation": {
            "swap_scene_level": correlation["scene_level"],
            "swap_scene_spearman_min": THRESHOLDS["swap_scene_spearman_min"],
            "swap_scene_spearman_lcb95_min": THRESHOLDS["swap_scene_spearman_lcb95_min"],
            "row_level_spearman_descriptive": correlation[
                "row_level_spearman_descriptive"
            ],
            "pass": gate_d_pass,
        },
        "arm_pass": bool(gate_a_pass and gate_b_pass and gate_c_pass and gate_d_pass),
    }


def select_arm(arm_pass: Mapping[str, bool]) -> dict[str, Any]:
    """Frozen arm-selection rule (may not be overridden by observed numbers)."""

    rule = (
        "if full_template_span passes V3: primary=full_template_span, "
        "answer_only=efficiency ablation; elif only answer_only passes: "
        "primary=answer_only; else STOP"
    )
    if arm_pass.get("full_template_span"):
        return {
            "decision": "full_template_span",
            "primary_proxy": "full_template_span",
            "efficiency_ablation": "answer_only",
            "rule": rule,
        }
    if arm_pass.get("answer_only"):
        return {
            "decision": "answer_only",
            "primary_proxy": "answer_only",
            "efficiency_ablation": None,
            "rule": rule,
        }
    return {
        "decision": "STOP",
        "primary_proxy": None,
        "efficiency_ablation": None,
        "rule": rule,
    }


def compute_arm_report(
    rows: list[Mapping[str, Any]],
    arm: str,
    *,
    num_resamples: int = BOOTSTRAP_NUM_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    strata: dict[str, Any] = {}
    for stratum in STRATA:
        subset = [row for row in rows if stratum in strata_of(row)]
        strata[stratum] = compute_stratum_report(
            subset, arm, num_resamples=num_resamples, seed=seed
        )
    for stratum in STRATA:
        correlation = strata[stratum]["branches"][BRANCH_SWAP]["correlation"]
        estimate = correlation["scene_level"]["estimate"]
        strata[stratum]["direction_reversed_descriptive"] = bool(
            estimate is not None and estimate < 0.0
        )
        correlation["scene_level"]["lcb95_required_for_gate"] = stratum == "overall"
    gate = evaluate_arm_gates(strata["overall"])
    return {"strata": strata, "gate": gate, "arm_pass": gate["arm_pass"]}


# ---------------------------------------------------------------------------
# fail-closed input validation


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path))
    if not rows:
        raise ReadoutError(f"manifest {path} is empty")
    seen: set[str] = set()
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ReadoutError(f"manifest {path}: every row needs a non-empty sample_id")
        if sample_id in seen:
            raise ReadoutError(f"manifest {path}: duplicate sample_id {sample_id!r}")
        seen.add(sample_id)
        expected_family = relation_family(row["answer_relation"], row["mapped_relation"])
        if expected_family not in ("axis", "LF_RB", "LB_RF"):
            raise ReadoutError(f"manifest {path}: bad relation family for {sample_id!r}")
    return rows


def load_and_validate_run(
    run_paths: Sequence[Path], manifest_rows: list[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Merge shard outputs fail-closed and return rows in manifest order."""

    expected_order = [str(row["sample_id"]) for row in manifest_rows]
    position = {sample_id: index for index, sample_id in enumerate(expected_order)}
    manifest_by_id = {str(row["sample_id"]): row for row in manifest_rows}
    collected: dict[str, dict[str, Any]] = {}
    for path in run_paths:
        last_position = -1
        for row in read_jsonl(path):
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or sample_id not in position:
                raise ReadoutError(
                    f"{path}: unknown or invalid sample_id {sample_id!r}"
                )
            if sample_id in collected:
                raise ReadoutError(f"{path}: duplicate sample_id {sample_id!r}")
            if position[sample_id] <= last_position:
                raise ReadoutError(
                    f"{path}: rows out of manifest order at {sample_id!r}"
                )
            last_position = position[sample_id]
            expected = manifest_by_id[sample_id]
            for field in (
                "scene_id",
                "relation_class",
                "answer_letter",
                "mapped_answer_letter",
            ):
                if row.get(field) != expected[field]:
                    raise ReadoutError(
                        f"{path}: {field} mismatch for {sample_id!r}: "
                        f"{row.get(field)!r} != {expected[field]!r}"
                    )
            expected_family = relation_family(
                expected["answer_relation"], expected["mapped_relation"]
            )
            if row.get("relation_family") != expected_family:
                raise ReadoutError(
                    f"{path}: relation_family mismatch for {sample_id!r}: "
                    f"{row.get('relation_family')!r} != {expected_family!r}"
                )
            collected[sample_id] = row
    missing = [sample_id for sample_id in expected_order if sample_id not in collected]
    if missing:
        raise ReadoutError(
            f"run output is incomplete: {len(collected)}/{len(expected_order)} rows; "
            f"missing={missing[:5]}"
        )
    return [collected[sample_id] for sample_id in expected_order]


# ---------------------------------------------------------------------------
# main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-jsonl", required=True, type=Path, nargs="+")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_NUM_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.bootstrap_resamples < 100:
        parser.error("--bootstrap-resamples must be at least 100")

    manifest_rows = load_manifest(args.manifest)
    rows = load_and_validate_run(args.run_jsonl, manifest_rows)

    report: dict[str, Any] = {
        "schema_version": 3,
        "gate": "LOSS_ALIGNED_PROXY_GATE_V3",
        "amendment": "loss_aligned_proxy_amendment_20260822",
        "analyzer_sha256": sha256_file(Path(__file__).resolve()),
        "input_run_jsonl_sha256": {
            str(path): sha256_file(path) for path in args.run_jsonl
        },
        "manifest_path": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "rows": len(rows),
        "scenes": len({row["scene_id"] for row in rows}),
        "strata_sizes": {
            stratum: sum(1 for row in rows if stratum in strata_of(row))
            for stratum in STRATA
        },
        "thresholds": dict(THRESHOLDS),
        "smoothing": SMOOTHING,
        "bootstrap": {
            "num_resamples": int(args.bootstrap_resamples),
            "seed": int(args.bootstrap_seed),
            "method": "scene resample with replacement, percentile LCB",
            "alpha": BOOTSTRAP_ALPHA,
        },
        "tie_policy": (
            "tie rows have exactly equal n(y_cf) and n(y); they are excluded "
            "from conditional sign agreement but always reported (tie rate and "
            "proxy |delta| on tie vs non-tie rows)"
        ),
        "delta_definitions": {
            "delta_proxy": "logp(y_cf) - logp(y) on length-normalized arm scores",
            "delta_rollout": "log((n(y_cf) + 0.5) / (n(y) + 0.5))",
        },
        "arms": {},
    }
    arm_pass: dict[str, bool] = {}
    for arm in ARMS:
        arm_report = compute_arm_report(
            rows,
            arm,
            num_resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
        )
        report["arms"][arm] = arm_report
        arm_pass[arm] = bool(arm_report["arm_pass"])
    selection = select_arm(arm_pass)
    report["arm_selection"] = selection
    report["gate_result"] = "PASS" if selection["decision"] != "STOP" else "FAIL"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    tmp.replace(args.output)

    summary = {
        arm: {
            "A_completion": report["arms"][arm]["gate"]["A_completion"]["pass"],
            "B_factual_top1": report["arms"][arm]["gate"]["B_factual_fidelity"][
                "top1_agreement"
            ],
            "C_swap_sign": report["arms"][arm]["gate"]["C_tie_aware_directional"][
                "swap_conditional_sign_agreement"
            ],
            "C_null_sign": report["arms"][arm]["gate"]["C_tie_aware_directional"][
                "null_conditional_sign_agreement"
            ],
            "D_scene_rho": report["arms"][arm]["gate"]["D_loss_aligned_correlation"][
                "swap_scene_level"
            ]["estimate"],
            "D_scene_lcb95": report["arms"][arm]["gate"]["D_loss_aligned_correlation"][
                "swap_scene_level"
            ]["lcb95"],
            "arm_pass": arm_pass[arm],
        }
        for arm in ARMS
    }
    summary["arm_selection"] = selection["decision"]
    summary["gate_result"] = report["gate_result"]
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
