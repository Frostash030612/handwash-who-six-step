"""Canonical action classes, label spaces and dataset alias mappings."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from typing import Final

from handwash.errors import ConfigError

__all__ = [
    "CANONICAL_STEPS",
    "LABEL_SPACES",
    "NON_WASH_STEPS",
    "STEP_EN",
    "STEP_ORDER",
    "STEP_ORDER_INDEX",
    "STEP_ZH",
    "WASH_STEPS",
    "LabelSpace",
    "Step",
    "StepFamily",
    "canonicalize",
    "canonicalize_sequence",
    "get_label_space",
    "is_wash_step",
    "num_classes",
    "to_space",
]


class StepFamily(str, Enum):
    WASH = "wash"
    NON_WASH = "non_wash"


class Step(str, Enum):
    STEP_1 = "step_1_palm_to_palm"
    STEP_2 = "step_2_palm_over_dorsum"
    STEP_3 = "step_3_fingers_interlaced"
    STEP_4 = "step_4_backs_of_fingers"
    STEP_5 = "step_5_rotational_thumbs"
    STEP_6 = "step_6_rotational_fingertips"
    FAUCET_ON = "faucet_on"
    FAUCET_OFF = "faucet_off"
    WASHING_HANDS = "washing_hands"
    OTHER = "other"
    UNKNOWN = "unknown"

    @property
    def family(self) -> StepFamily:

        return StepFamily.WASH if self in WASH_STEP_SET else StepFamily.NON_WASH

    @property
    def order_index(self) -> int:

        return STEP_ORDER_INDEX.get(self, 0)

    @property
    def is_who_step(self) -> bool:
        return self in STEP_ORDER_SET


CANONICAL_STEPS: Final[tuple[Step, ...]] = (
    Step.STEP_1,
    Step.STEP_2,
    Step.STEP_3,
    Step.STEP_4,
    Step.STEP_5,
    Step.STEP_6,
)

STEP_ORDER: Final[tuple[Step, ...]] = CANONICAL_STEPS
STEP_ORDER_SET: Final[frozenset[Step]] = frozenset(CANONICAL_STEPS)
STEP_ORDER_INDEX: Final[Mapping[Step, int]] = {s: i + 1 for i, s in enumerate(CANONICAL_STEPS)}


WASH_STEPS: Final[tuple[Step, ...]] = CANONICAL_STEPS
WASH_STEP_SET: Final[frozenset[Step]] = STEP_ORDER_SET


NON_WASH_STEPS: Final[tuple[Step, ...]] = (
    Step.FAUCET_ON,
    Step.FAUCET_OFF,
    Step.WASHING_HANDS,
    Step.OTHER,
    Step.UNKNOWN,
)

# Retain the legacy exported name for API compatibility; this distribution uses English text.
STEP_ZH: Final[Mapping[Step, str]] = {
    Step.STEP_1: "Palm-to-palm rubbing",
    Step.STEP_2: "Palm over dorsum",
    Step.STEP_3: "Palm to palm with fingers interlaced",
    Step.STEP_4: "Backs of fingers against opposing palms",
    Step.STEP_5: "Rotational thumb rubbing",
    Step.STEP_6: "Rotational fingertip rubbing in palm",
    Step.FAUCET_ON: "Turn on faucet",
    Step.FAUCET_OFF: "Turn off faucet",
    Step.WASHING_HANDS: "Transitional washing action",
    Step.OTHER: "Other action",
    Step.UNKNOWN: "Unknown / unrecognized",
}

STEP_EN: Final[Mapping[Step, str]] = {
    Step.STEP_1: "Palm to palm",
    Step.STEP_2: "Palm over dorsum",
    Step.STEP_3: "Fingers interlaced",
    Step.STEP_4: "Backs of fingers to opposing palms",
    Step.STEP_5: "Rotational rubbing of thumbs",
    Step.STEP_6: "Rotational rubbing of fingertips",
    Step.FAUCET_ON: "Faucet on",
    Step.FAUCET_OFF: "Faucet off",
    Step.WASHING_HANDS: "Washing hands (transitional)",
    Step.OTHER: "Other",
    Step.UNKNOWN: "Unknown",
}


def num_classes(*, include_non_wash: bool = True) -> int:

    return len(Step) if include_non_wash else len(CANONICAL_STEPS)


def is_wash_step(label: Step | str) -> bool:

    try:
        return canonicalize(label).family is StepFamily.WASH
    except ConfigError:
        return False


_DATASET_ALIASES: Final[Mapping[str, Mapping[str, Step]]] = {
    "pskuss": {
        "movement_0": Step.OTHER,
        "movement_1": Step.STEP_1,
        "movement_2": Step.STEP_2,
        "movement_3": Step.STEP_3,
        "movement_4": Step.STEP_4,
        "movement_5": Step.STEP_5,
        "movement_6": Step.STEP_6,
        "movement_7": Step.FAUCET_OFF,
        "0": Step.OTHER,
        "1": Step.STEP_1,
        "2": Step.STEP_2,
        "3": Step.STEP_3,
        "4": Step.STEP_4,
        "5": Step.STEP_5,
        "6": Step.STEP_6,
        "7": Step.FAUCET_OFF,
        "palm to palm": Step.STEP_1,
        "palm over dorsum, fingers interlaced": Step.STEP_2,
        "palm to palm, fingers interlaced": Step.STEP_3,
        "backs of fingers to opposing palm, fingers interlocked": Step.STEP_4,
        "rotational rubbing of the thumb": Step.STEP_5,
        "fingertips to palm": Step.STEP_6,
        "turning off the faucet with a paper towel": Step.FAUCET_OFF,
        "other movement": Step.OTHER,
        "step 1": Step.STEP_1,
        "step 2": Step.STEP_2,
        "step 3": Step.STEP_3,
        "step 4": Step.STEP_4,
        "step 5": Step.STEP_5,
        "step 6": Step.STEP_6,
        "faucet on": Step.FAUCET_ON,
        "faucet off": Step.FAUCET_OFF,
        "washing hands": Step.WASHING_HANDS,
        "other": Step.OTHER,
        "unknown": Step.UNKNOWN,
    },
    "metc": {
        # Zenodo METC subset: published movement codes are 0=other, 1..6=WHO steps.
        "0": Step.OTHER,
        "1": Step.STEP_1,
        "2": Step.STEP_2,
        "3": Step.STEP_3,
        "4": Step.STEP_4,
        "5": Step.STEP_5,
        "6": Step.STEP_6,
        "movement_0": Step.OTHER,
        "movement_1": Step.STEP_1,
        "movement_2": Step.STEP_2,
        "movement_3": Step.STEP_3,
        "movement_4": Step.STEP_4,
        "movement_5": Step.STEP_5,
        "movement_6": Step.STEP_6,
        "step_1": Step.STEP_1,
        "step_2": Step.STEP_2,
        "step_3": Step.STEP_3,
        "step_4": Step.STEP_4,
        "step_5": Step.STEP_5,
        "step_6": Step.STEP_6,
        "faucet_on": Step.FAUCET_ON,
        "faucet_off": Step.FAUCET_OFF,
        "other": Step.OTHER,
    },
    "metc_public": {
        "0": Step.OTHER,
        "1": Step.STEP_1,
        "2": Step.STEP_2,
        "3": Step.STEP_3,
        "4": Step.STEP_4,
        "5": Step.STEP_5,
        "6": Step.STEP_6,
        "movement_0": Step.OTHER,
        "movement_1": Step.STEP_1,
        "movement_2": Step.STEP_2,
        "movement_3": Step.STEP_3,
        "movement_4": Step.STEP_4,
        "movement_5": Step.STEP_5,
        "movement_6": Step.STEP_6,
        "step_1": Step.STEP_1,
        "step_2": Step.STEP_2,
        "step_3": Step.STEP_3,
        "step_4": Step.STEP_4,
        "step_5": Step.STEP_5,
        "step_6": Step.STEP_6,
        "other": Step.OTHER,
    },
    "kaggle": {
        "step1_water": Step.STEP_1,
        "step2_water": Step.STEP_2,
        "step3_water": Step.STEP_3,
        "step4_water": Step.STEP_4,
        "step5_water": Step.STEP_5,
        "step6_water": Step.STEP_6,
        "step7_water": Step.OTHER,
        "step1": Step.STEP_1,
        "step2": Step.STEP_2,
        "step3": Step.STEP_3,
        "step4": Step.STEP_4,
        "step5": Step.STEP_5,
        "step6": Step.STEP_6,
        "step7": Step.OTHER,
        "not_washing": Step.OTHER,
        "nowashing": Step.OTHER,
    },
    "jurmala": {
        "step 1": Step.STEP_1,
        "step 2": Step.STEP_2,
        "step 3": Step.STEP_3,
        "step 4": Step.STEP_4,
        "step 5": Step.STEP_5,
        "step 6": Step.STEP_6,
        "faucet on": Step.FAUCET_ON,
        "faucet off": Step.FAUCET_OFF,
        "other": Step.OTHER,
    },
    "selfrecorded": {
        "step_1": Step.STEP_1,
        "step_2": Step.STEP_2,
        "step_3": Step.STEP_3,
        "step_4": Step.STEP_4,
        "step_5": Step.STEP_5,
        "step_6": Step.STEP_6,
        "faucet_on": Step.FAUCET_ON,
        "faucet_off": Step.FAUCET_OFF,
        "other": Step.OTHER,
    },
    "synthetic": {
        "step_1": Step.STEP_1,
        "step_2": Step.STEP_2,
        "step_3": Step.STEP_3,
        "step_4": Step.STEP_4,
        "step_5": Step.STEP_5,
        "step_6": Step.STEP_6,
    },
}


_CANONICAL_ALIASES: Final[dict[str, Step]] = {
    **{s.value: s for s in Step},
    **{s.value.replace("_", " "): s for s in Step},
    **{f"step{i}": s for i, s in enumerate(CANONICAL_STEPS, start=1)},
    **{f"step {i}": s for i, s in enumerate(CANONICAL_STEPS, start=1)},
    **{f"index{i}step": s for i, s in enumerate(CANONICAL_STEPS, start=1)},
    "palm to palm": Step.STEP_1,
    "palm over dorsum": Step.STEP_2,
    "fingers interlaced": Step.STEP_3,
    "backs of fingers": Step.STEP_4,
    "rotational thumbs": Step.STEP_5,
    "rotational fingertips": Step.STEP_6,
    "Faucet on": Step.FAUCET_ON,
    "Faucet off": Step.FAUCET_OFF,
    "Other": Step.OTHER,
    "Unknown": Step.UNKNOWN,
}


def _normalize(text: str) -> str:

    return re.sub(r"[\s_\-]+", " ", str(text).strip().lower()).strip()


class LabelSpace:
    __slots__ = ("_aliases", "_index", "_labels", "name")

    def __init__(self, name: str, labels: Sequence[Step], aliases: Mapping[str, Step] | None = None) -> None:
        if not labels:
            raise ConfigError(f"Label space `{name}` is empty")
        if len(set(labels)) != len(labels):
            dup = [s.value for s in labels if list(labels).count(s) > 1]
            raise ConfigError(f"Label space `{name}` has duplicate classes: {sorted(set(dup))}")
        self.name = name
        self._labels: tuple[Step, ...] = tuple(labels)
        self._index: Mapping[Step, int] = {s: i for i, s in enumerate(self._labels)}
        self._aliases: Mapping[str, Step] = dict(aliases or {})

    @property
    def labels(self) -> tuple[Step, ...]:

        return self._labels

    def __len__(self) -> int:
        return len(self._labels)

    def __contains__(self, item: object) -> bool:
        try:
            return self.canonicalize(item) in self._index  # type: ignore[arg-type]
        except ConfigError:
            return False

    def __iter__(self):
        return iter(self._labels)

    def __repr__(self) -> str:
        return f"LabelSpace(name={self.name!r}, size={len(self._labels)})"

    def canonicalize(self, raw: object) -> Step:

        if isinstance(raw, Step):
            return raw
        key = _normalize(str(raw))

        candidates = (key, key.replace(" ", ""), key.replace(" ", "_"))
        for table in (self._aliases, _DATASET_ALIASES.get(self.name, {}), _CANONICAL_ALIASES):
            for candidate in candidates:
                if candidate in table:
                    return table[candidate]
        raise ConfigError(
            f"Label space `{self.name}` unrecognized label: {raw!r}",
            hint="Add label aliases to _DATASET_ALIASES in core/labels.py.",
        )

    def to_index(self, label: object) -> int:

        step = self.canonicalize(label)
        try:
            return self._index[step]
        except KeyError as exc:
            raise ConfigError(
                f"Label `{step.value}` is not in label space `{self.name}` within",
                hint=f"Classes in this namespace: {[s.value for s in self._labels]}",
            ) from exc

    def to_label(self, index: int) -> Step:

        if not 0 <= index < len(self._labels):
            raise ConfigError(
                f"Class index is out of range: {index}(label space `{self.name}` total {len(self._labels)} classes)"
            )
        return self._labels[index]

    def to_space(self, label: object, target: LabelSpace) -> int:

        step = self.canonicalize(label)
        if step not in target._index:
            raise ConfigError(
                f"Label `{step.value}` cannot be mapped to label space `{target.name}`",
                hint="For cross-dataset evaluation, use macro metrics and mark absent target classes as N/A.",
            )
        return target._index[step]

    def indices_of(self, steps: Iterable[Step]) -> list[int]:

        return [self._index[s] for s in steps if s in self._index]


LABEL_SPACES: Final[Mapping[str, LabelSpace]] = {
    "pskuss": LabelSpace(
        "pskuss",
        (
            Step.STEP_1,
            Step.STEP_2,
            Step.STEP_3,
            Step.STEP_4,
            Step.STEP_5,
            Step.STEP_6,
            Step.FAUCET_ON,
            Step.FAUCET_OFF,
            Step.WASHING_HANDS,
            Step.OTHER,
        ),
    ),
    "metc": LabelSpace(
        "metc",
        (
            Step.STEP_1,
            Step.STEP_2,
            Step.STEP_3,
            Step.STEP_4,
            Step.STEP_5,
            Step.STEP_6,
            Step.FAUCET_ON,
            Step.FAUCET_OFF,
            Step.OTHER,
        ),
    ),
    "kaggle": LabelSpace("kaggle", CANONICAL_STEPS),
    "jurmala": LabelSpace("jurmala", CANONICAL_STEPS),
    "synthetic": LabelSpace("synthetic", CANONICAL_STEPS),
    "selfrecorded": LabelSpace(
        "selfrecorded",
        (
            Step.STEP_1,
            Step.STEP_2,
            Step.STEP_3,
            Step.STEP_4,
            Step.STEP_5,
            Step.STEP_6,
            Step.FAUCET_ON,
            Step.FAUCET_OFF,
            Step.OTHER,
        ),
    ),
    # Exact published METC label set: six WHO movements plus other (no faucet events).
    # Appended so existing label-space indices/checkpoints retain their meaning.
    "metc_public": LabelSpace("metc_public", (*CANONICAL_STEPS, Step.OTHER)),
    # Full seven-class export matches the Ultralytics directory order: 0=other.
    "who7": LabelSpace("who7", (Step.OTHER, *CANONICAL_STEPS)),
}


def get_label_space(name: str) -> LabelSpace:

    normalized = _normalize(name)
    candidates = (normalized.replace(" ", ""), normalized.replace(" ", "_"))
    for key in candidates:
        if key in LABEL_SPACES:
            return LABEL_SPACES[key]
    raise ConfigError(
        f"Undefined label space: {name!r}",
        hint=f"Available: {sorted(LABEL_SPACES)}; add new labels in core/labels.py and update the configuration schema_version.",
    )


def canonicalize(raw: object, *, dataset: str = "selfrecorded") -> Step:

    return get_label_space(dataset).canonicalize(raw)


def canonicalize_sequence(raws: Iterable[object], *, dataset: str = "selfrecorded") -> tuple[Step, ...]:

    space = get_label_space(dataset)
    return tuple(space.canonicalize(r) for r in raws)


def to_space(label: object, space: str, *, dataset: str = "selfrecorded") -> int:

    return get_label_space(dataset).to_space(label, get_label_space(space))
