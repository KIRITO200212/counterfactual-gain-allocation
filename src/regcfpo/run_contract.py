"""Immutable machine-readable contracts for resumable experiment outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


class RunContractError(ValueError):
    """Raised when existing artifacts do not match the requested run."""


def ensure_run_contract(
    path: Path,
    contract: Mapping[str, Any],
    *,
    existing_artifacts: Iterable[Path] = (),
) -> dict[str, Any]:
    """Create an immutable JSON contract or verify an exact existing match."""

    if not isinstance(contract, Mapping) or not contract:
        raise RunContractError("run contract must be a non-empty mapping")
    try:
        normalized = json.loads(json.dumps(dict(contract), ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise RunContractError("run contract must be JSON serializable") from exc
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RunContractError(f"invalid existing run contract {path}: {exc}") from exc
        if existing != normalized:
            keys = sorted(
                key
                for key in set(existing) | set(normalized)
                if existing.get(key) != normalized.get(key)
            )
            raise RunContractError(
                f"requested run does not match {path}; differing keys: {keys}"
            )
        return normalized

    artifacts = [artifact for artifact in existing_artifacts if artifact.exists()]
    if artifacts:
        names = [str(artifact) for artifact in artifacts[:5]]
        raise RunContractError(
            "refusing to adopt pre-existing outputs without a run contract: "
            f"{names}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return normalized


__all__ = ["RunContractError", "ensure_run_contract"]
