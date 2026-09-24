"""RelationPair-v3 primitives: six swap-flippable relations.

Preregistered in ``configs/phase1_prereg.yaml`` under ``relation_pair_v3``
(amendment ``relation_pair_v3_protocol_20260820``).  The pixel pair-slot swap
moves A to B's box and B to A's box, which is a point reflection of the
center offsets.  The counterfactual mapping therefore flips both the
horizontal and the depth component:

    left <-> right                     (validated by the 447-row gate)
    left-front <-> right-back          (hypothesis; unseen-scene validation)
    left-back  <-> right-front         (hypothesis; unseen-scene validation)

Pure front/back relations have no horizontal component and depth cannot be
manipulated by a 2D box swap, so they stay excluded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

SUPPORTED_RELATIONS_V3 = frozenset(
    {"left", "right", "left-front", "right-front", "left-back", "right-back"}
)

SWAP_MAPPING_V3: Mapping[str, str] = {
    "left": "right",
    "right": "left",
    "left-front": "right-back",
    "right-back": "left-front",
    "left-back": "right-front",
    "right-front": "left-back",
}

AXIS_RELATIONS = frozenset({"left", "right"})
DIAGONAL_RELATIONS = frozenset(
    {"left-front", "right-front", "left-back", "right-back"}
)

_OPTION_PREFIX_RE = re.compile(
    r"^\s*(?:\(?[A-Z]\)?\s*[.):\-]|[A-Z]\s+)\s*", re.IGNORECASE
)


class OptionRelationV3Error(ValueError):
    """Raised when an option set cannot support a unique v3 mapping."""


def normalize_relation_v3(relation: str) -> str:
    """Canonicalize a relation string to one of the six supported forms."""

    if not isinstance(relation, str):
        raise ValueError("relation must be a string")
    canonical = relation.strip().lower().replace("_", " ").replace("-", " ")
    canonical = "-".join(canonical.split())
    if canonical not in SUPPORTED_RELATIONS_V3:
        raise ValueError(
            f"unsupported v3 relation {relation!r}; expected one of "
            f"{sorted(SUPPORTED_RELATIONS_V3)}"
        )
    return canonical


def swap_mapped_relation_v3(relation: str) -> str:
    """Return the point-reflection counterfactual relation."""

    return SWAP_MAPPING_V3[normalize_relation_v3(relation)]


def horizontal_component(relation: str) -> str:
    """Return the left/right component of a supported relation."""

    canonical = normalize_relation_v3(relation)
    return "left" if canonical.startswith("left") else "right"


def flip_horizontal_component(relation: str) -> str:
    """Flip only the horizontal component: r10 = (-h, d)."""

    canonical = normalize_relation_v3(relation)
    if canonical in AXIS_RELATIONS:
        raise ValueError("axis relations have no depth component to hold fixed")
    horizontal, depth = canonical.split("-")
    flipped = "left" if horizontal == "right" else "right"
    return f"{flipped}-{depth}"


def flip_depth_component(relation: str) -> str:
    """Flip only the depth component: r01 = (h, -d)."""

    canonical = normalize_relation_v3(relation)
    if canonical in AXIS_RELATIONS:
        raise ValueError("axis relations have no depth component to flip")
    horizontal, depth = canonical.split("-")
    flipped = "front" if depth == "back" else "back"
    return f"{horizontal}-{flipped}"


def dualize_question(question: str, entity_a: str, entity_b: str) -> str:
    """Exchange the two entity mentions to form the B-relative-to-A question.

    Uses the same non-overlapping unique-span rule as the v2/v3 builders:
    each entity label must occur exactly once as a boundary-aware match, and
    the spans must not overlap.  The question text is otherwise untouched, so
    option letters and relation vocabulary stay identical.
    """

    spans: dict[str, tuple[int, int]] = {}
    for entity in (entity_a, entity_b):
        matches = list(
            re.finditer(
                rf"(?<!\w){re.escape(entity.strip())}(?!\w)",
                question,
                flags=re.IGNORECASE,
            )
        )
        if len(matches) != 1:
            raise ValueError(
                f"entity {entity!r} must occur exactly once in the question"
            )
        spans[entity] = matches[0].span()
    (start_a, end_a), (start_b, end_b) = spans[entity_a], spans[entity_b]
    if start_a < end_b and start_b < end_a:
        raise ValueError("entity spans overlap; dual question is unsafe")
    ordered = sorted(
        ((start_a, end_a, entity_b), (start_b, end_b, entity_a)), key=lambda t: t[0]
    )
    pieces: list[str] = []
    cursor = 0
    for start, end, replacement in ordered:
        pieces.append(question[cursor:start])
        pieces.append(replacement)
        cursor = end
    pieces.append(question[cursor:])
    return "".join(pieces)


def parse_option_relation_v3(option: str) -> str | None:
    """Parse one option into a supported relation or ``None`` if unrelated.

    Options mentioning both left and right are ambiguous and rejected; pure
    front/back options are unrelated for v3 and return ``None``.
    """

    if not isinstance(option, str) or not option.strip():
        raise OptionRelationV3Error("each answer option must be a non-empty string")
    text = _OPTION_PREFIX_RE.sub("", option.strip()).replace("_", " ").lower()
    text = "-".join(text.split())
    if text in SUPPORTED_RELATIONS_V3:
        return text
    if "front" in text or "back" in text or "left" in text or "right" in text:
        return None
    return None


def resolve_option_relations_v3(options: Sequence[str]) -> dict[str, str]:
    """Return ``option letter -> relation`` for the recognized v3 options.

    Duplicate semantics (two options meaning the same relation) are rejected
    because they destroy the uniqueness of the counterfactual answer letter.
    """

    if isinstance(options, (str, bytes)) or not isinstance(options, Sequence):
        raise OptionRelationV3Error("options must be a sequence of strings")
    if not 2 <= len(options) <= 26:
        raise OptionRelationV3Error("options must contain between 2 and 26 choices")
    by_letter: dict[str, str] = {}
    by_relation: dict[str, str] = {}
    for index, option in enumerate(options):
        relation = parse_option_relation_v3(option)
        if relation is None:
            continue
        letter = chr(ord("A") + index)
        if relation in by_relation and by_relation[relation] != letter:
            raise OptionRelationV3Error(
                f"options {by_relation[relation]} and {letter} both mean {relation!r}"
            )
        by_letter[letter] = relation
        by_relation[relation] = letter
    return by_letter


@dataclass(frozen=True)
class RelationPairV3:
    """One strict v3 counterfactual row with its point-reflection mapping.

    Hardened per amendment ``analysis_integrity_and_diagonal_disambiguation
    _20260820``: every instance is fail-closed validated on construction.
    """

    sample_id: str
    scene_id: str
    image: str
    question: str
    options: tuple[str, ...]
    answer_relation: str
    mapped_relation: str
    answer_letter: str
    mapped_answer_letter: str
    entity_a: str
    entity_b: str
    gt_box_a: tuple[float, float, float, float]
    gt_box_b: tuple[float, float, float, float]
    relation_class: str
    operator_valid: bool

    def __post_init__(self) -> None:
        if not self.sample_id or not isinstance(self.sample_id, str):
            raise ValueError("sample_id must be a non-empty string")
        if not self.scene_id or not isinstance(self.scene_id, str):
            raise ValueError("scene_id must be a non-empty string")
        if not self.image or "/" not in self.image:
            raise ValueError("image must be a non-empty '<scene_dir>/<frame>' path")
        if not self.question.strip():
            raise ValueError("question must be non-empty")
        if not 2 <= len(self.options) <= 26:
            raise ValueError("options must contain between 2 and 26 choices")
        if self.answer_relation not in SUPPORTED_RELATIONS_V3:
            raise ValueError(f"unsupported answer relation {self.answer_relation!r}")
        if self.mapped_relation != swap_mapped_relation_v3(self.answer_relation):
            raise ValueError(
                "mapped_relation inconsistent with the point-reflection mapping: "
                f"{self.answer_relation!r} -> {self.mapped_relation!r}"
            )
        expected_class = (
            "axis" if self.answer_relation in AXIS_RELATIONS else "diagonal"
        )
        if self.relation_class != expected_class:
            raise ValueError(
                f"relation_class {self.relation_class!r} inconsistent with "
                f"relation {self.answer_relation!r}"
            )
        for letter in (self.answer_letter, self.mapped_answer_letter):
            if not (isinstance(letter, str) and len(letter) == 1 and letter.isupper()):
                raise ValueError(f"invalid option letter {letter!r}")
        if self.answer_letter == self.mapped_answer_letter:
            raise ValueError("answer and mapped letters must differ")
        letters = resolve_option_relations_v3(self.options)
        if letters.get(self.answer_letter) != self.answer_relation:
            raise ValueError(
                f"answer letter {self.answer_letter} does not parse to "
                f"{self.answer_relation!r} under the option parser"
            )
        if letters.get(self.mapped_answer_letter) != self.mapped_relation:
            raise ValueError(
                f"mapped letter {self.mapped_answer_letter} does not parse to "
                f"{self.mapped_relation!r} under the option parser"
            )
        if not self.entity_a.strip() or not self.entity_b.strip():
            raise ValueError("entity labels must be non-empty")
        for name, box in (("gt_box_a", self.gt_box_a), ("gt_box_b", self.gt_box_b)):
            if len(box) != 4 or any(not isinstance(v, (int, float)) for v in box):
                raise ValueError(f"{name} must be four numeric coordinates")
            x0, y0, x1, y1 = (float(v) for v in box)
            if x1 <= x0 or y1 <= y0 or x0 < 0 or y0 < 0:
                raise ValueError(f"{name} is not a legal box: {box!r}")
        center_a = (float(self.gt_box_a[0]) + float(self.gt_box_a[2])) / 2.0
        center_b = (float(self.gt_box_b[0]) + float(self.gt_box_b[2])) / 2.0
        if center_a == center_b:
            raise ValueError("equal x-centers: relation undefined, row must be rejected")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "RelationPair-v3",
            "sample_id": self.sample_id,
            "scene_id": self.scene_id,
            "image": self.image,
            "question": self.question,
            "options": list(self.options),
            "answer_relation": self.answer_relation,
            "mapped_relation": self.mapped_relation,
            "answer_letter": self.answer_letter,
            "mapped_answer_letter": self.mapped_answer_letter,
            "entity_a": self.entity_a,
            "entity_b": self.entity_b,
            "gt_box_a": list(self.gt_box_a),
            "gt_box_b": list(self.gt_box_b),
            "relation_class": self.relation_class,
            "operator_valid": self.operator_valid,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> "RelationPairV3":
        allowed = {
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
        unknown = set(row) - allowed
        if unknown:
            raise ValueError(f"unknown RelationPair-v3 fields: {sorted(unknown)}")
        return cls(
            sample_id=str(row["sample_id"]),
            scene_id=str(row["scene_id"]),
            image=str(row["image"]),
            question=str(row["question"]),
            options=tuple(str(option) for option in row["options"]),  # type: ignore[arg-type]
            answer_relation=normalize_relation_v3(str(row["answer_relation"])),
            mapped_relation=normalize_relation_v3(str(row["mapped_relation"])),
            answer_letter=str(row["answer_letter"]),
            mapped_answer_letter=str(row["mapped_answer_letter"]),
            entity_a=str(row["entity_a"]),
            entity_b=str(row["entity_b"]),
            gt_box_a=tuple(float(v) for v in row["gt_box_a"]),  # type: ignore[arg-type]
            gt_box_b=tuple(float(v) for v in row["gt_box_b"]),  # type: ignore[arg-type]
            relation_class=str(row["relation_class"]),
            operator_valid=bool(row["operator_valid"]),
        )
