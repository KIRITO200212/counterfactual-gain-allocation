"""Conversation sanitization for template rendering (pure, import-safe).

The SpatialLadder-3B checkpoint's chat template treats the PRESENCE of the
``image`` key as an image item (``'image' in content``), so
datasets-round-tripped conversation rows — whose content items carry both
``text`` and ``image`` keys with None values — would render a spurious
second image placeholder.  Dropping None-valued keys reproduces the
vendored rollout cleaning (grpo_trainer.py content-item loop,
grpo_trainer.py:583-627) and the official saturation/proxy message
construction (reward_saturation_audit.py) byte-for-byte.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def sanitize_conversation_for_template(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return messages with None-valued content-item keys removed."""

    cleaned: list[dict[str, Any]] = []
    for message in messages:
        contents = [
            {key: value for key, value in item.items() if value is not None}
            for item in message["content"]
        ]
        cleaned.append({"role": message["role"], "content": contents})
    return cleaned


__all__ = ["sanitize_conversation_for_template"]
