"""Deterministic, dependency-free natural-language task parser.

Handles the instruction styles this project targets:

* "Sort all objects into their matching bins."
* "Move the milk first, then the cereal."
* "Pick up the can and place it into its designated container."
* "Put the bread in the far left bin, then the milk in its bin."
* "Sort everything except the can."
* "Move the milk after the cereal."

Algorithm
---------
1. Normalise text and split it into *clauses* at sequencing words ("then", "after that", ";").
2. In each clause, find object mentions (longest alias first, fuzzy fallback for typos) and the
   target phrase ("into/in/to <container>"), resolving "its/their matching bin" to each
   object's designated container.
3. Detect nouns used as manipulation objects that are not in the catalogue and reject them,
   as well as target phrases that do not name a known container.
4. Apply ordering modifiers ("first", "after", "before", "last") and remove duplicates.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from robotics.language.schema import ActionType, PlanningError, TaskPlan, TaskStep
from robotics.simulation.scene import CONTAINERS, OBJECT_CATALOG, OBJECT_NAMES, designated_container

_ALL_WORDS = (
    r"(?:all(?: the)? (?:objects|items|things)|everything|all of them|every object|each object)"
)
_SEQUENCE_SPLIT = re.compile(
    r"\s*(?:;|\.\s|,?\s*\b(?:and then|then|after that|afterwards|next|finally)\b)\s*"
)
_EXCEPT = re.compile(r"\b(?:except(?: for)?|but not|other than|apart from)\b(.*)$")
_TARGET_PREP = re.compile(
    r"\b(?:into|in|inside|to|onto|on)\s+(?:the\s+|its\s+|their\s+|his\s+|her\s+)?(.+)$"
)
_DESIGNATED = re.compile(
    r"\b(?:its|their|the)\s+(?:own\s+)?(?:matching|designated|corresponding|correct|right|proper|respective|assigned)?\s*"
    r"(?:bins?|containers?|compartments?|places?|spots?|boxes)\b"
    r"|\b(?:matching|designated|corresponding|correct|respective|assigned)\s+(?:bins?|containers?|compartments?|places?)\b"
)
_MANIP_VERBS = r"(?:pick up|pick|grab|move|put|place|take|bring|sort|get|grasp|lift|drop|transfer|carry|deliver)"
_UNKNOWN_OBJ = re.compile(
    rf"\b{_MANIP_VERBS}\s+(?:up\s+)?(?:(?:the|a|an|my|that|this)\s+)?([a-z][a-z\-]+)"
)
_STOP_NOUNS = {
    "all", "everything", "objects", "items", "things", "them", "it", "each", "every", "up", "and",
    "first", "then", "object", "item", "both", "these", "those", "rest", "remaining", "other", "others",
}  # fmt: skip
_CONTAINER_WORDS = {"bin", "bins", "container", "containers", "compartment", "compartments", "box"}

# alias -> canonical object name, longest aliases first so "cereal box" wins over "box"
_ALIASES: list[tuple[str, str]] = sorted(
    ((alias, spec.name) for spec in OBJECT_CATALOG for alias in (spec.name, *spec.aliases)),
    key=lambda t: -len(t[0]),
)
_CONTAINER_ALIASES: list[tuple[str, str]] = sorted(
    (
        (alias, name)
        for name, c in CONTAINERS.items()
        for alias in (name.replace("_", " "), *c.aliases)
    ),
    key=lambda t: -len(t[0]),
)


@dataclass
class _Mention:
    obj: str
    pos: int
    fuzzy: bool = False


def _normalise(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[!?]", ".", text)
    text = re.sub(r"[^a-z0-9\s\.,;\-']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text if text.endswith(".") else text + "."


def _find_objects(clause: str) -> list[_Mention]:
    mentions: list[_Mention] = []
    taken = [False] * len(clause)
    for alias, name in _ALIASES:
        for m in re.finditer(rf"\b{re.escape(alias)}s?\b", clause):
            if any(taken[m.start() : m.end()]):
                continue
            # "milk bin" names a container, not the object
            tail = clause[m.end() : m.end() + 14]
            if re.match(r"\s+(?:bin|container|compartment)", tail):
                continue
            for i in range(m.start(), m.end()):
                taken[i] = True
            mentions.append(_Mention(name, m.start()))
    if not mentions:
        # Fuzzy fallback for typos ("cerael", "milc")
        for m in re.finditer(r"[a-z]{3,}", clause):
            word = m.group(0)
            if word in _STOP_NOUNS or word in _CONTAINER_WORDS:
                continue
            close = difflib.get_close_matches(word, OBJECT_NAMES, n=1, cutoff=0.75)
            if close:
                mentions.append(_Mention(close[0], m.start(), fuzzy=True))
    mentions.sort(key=lambda x: x.pos)
    return mentions


def _resolve_container(phrase: str) -> str | None:
    for alias, name in _CONTAINER_ALIASES:
        if re.search(rf"\b{re.escape(alias)}\b", phrase):
            return name
    return None


def _unknown_objects(clause: str) -> list[str]:
    unknown = []
    for m in _UNKNOWN_OBJ.finditer(clause):
        word = m.group(1)
        if word in _STOP_NOUNS or word in _CONTAINER_WORDS:
            continue
        if any(re.fullmatch(rf"{re.escape(a.split()[0])}s?", word) for a, _ in _ALIASES):
            continue
        if difflib.get_close_matches(word, OBJECT_NAMES, n=1, cutoff=0.75):
            continue
        unknown.append(word)
    return unknown


class RuleBasedTaskParser:
    """Instruction → :class:`TaskPlan` without any external service."""

    name = "rule_based"

    def parse(self, instruction: str) -> TaskPlan:
        if not instruction or not instruction.strip():
            raise PlanningError("empty_instruction", "The instruction is empty.")
        text = _normalise(instruction)
        notes: list[str] = []

        # "except ..." exclusions apply to "all objects" requests
        excluded: set[str] = set()
        m = _EXCEPT.search(text)
        if m:
            excluded = {mm.obj for mm in _find_objects(m.group(1))}
            text = text[: m.start()].strip() + "."

        clauses = [c.strip(" ,.") for c in _SEQUENCE_SPLIT.split(text) if c and c.strip(" ,.")]
        # (object, target, confidence, explicit_target)
        entries: list[tuple[str, str, float]] = []
        unknown: list[str] = []
        invalid_targets: list[str] = []

        for clause in clauses:
            # Several "X into A and Y into B" pairs inside one clause
            for sub in self._split_pairs(clause):
                unknown += _unknown_objects(sub)
                mentions = _find_objects(sub)
                if re.search(rf"\b{_ALL_WORDS}\b", sub) or re.search(
                    r"\bsort\b(?!.*\b(?:the|a)\b\s+\w)", sub
                ):
                    present = {mm.obj for mm in mentions}
                    mentions += [
                        _Mention(n, 10_000 + i)
                        for i, n in enumerate(OBJECT_NAMES)
                        if n not in present
                    ]
                if not mentions:
                    continue
                target_name: str | None = None
                target_conf = 1.0
                tm = _TARGET_PREP.search(sub)
                if tm:
                    phrase = tm.group(1)
                    if _DESIGNATED.search(sub):
                        target_name = None  # designated per object
                    else:
                        target_name = _resolve_container(phrase)
                        if target_name is None:
                            obj_words = {a for a, _ in _ALIASES}
                            words = phrase.strip(" .").split()
                            # "put the milk in the cereal box" style phrases reaching an alias
                            if not any(w in obj_words for w in words):
                                invalid_targets.append(phrase.strip(" ."))
                                continue
                else:
                    target_conf = 0.85  # target not stated: assume designated container
                for mm in mentions:
                    tgt = target_name or designated_container(mm.obj).name
                    conf = target_conf * (0.8 if mm.fuzzy else 1.0)
                    if mm.fuzzy:
                        notes.append(f"interpreted a misspelled word as '{mm.obj}'")
                    entries.append((mm.obj, tgt, conf))

        if unknown:
            raise PlanningError(
                "unknown_object",
                f"I don't know the object(s) {sorted(set(unknown))}. Objects in this scene: {list(OBJECT_NAMES)}.",
                suggestions=list(OBJECT_NAMES),
            )
        if invalid_targets:
            raise PlanningError(
                "invalid_target",
                f"Unknown target location(s) {sorted(set(invalid_targets))}. "
                f"Valid containers: {sorted(CONTAINERS)}.",
                suggestions=sorted(CONTAINERS),
            )
        if not entries:
            raise PlanningError(
                "no_objects",
                f"No known object found in '{instruction}'. Try naming one of {list(OBJECT_NAMES)}.",
                suggestions=list(OBJECT_NAMES),
            )

        # De-duplicate, keeping the first occurrence
        seen: dict[str, tuple[str, float]] = {}
        order: list[str] = []
        for obj, tgt, conf in entries:
            if obj in excluded:
                continue
            if obj in seen:
                notes.append(f"duplicate request for '{obj}' ignored")
                continue
            seen[obj] = (tgt, conf)
            order.append(obj)
        if excluded:
            notes.append(f"excluded: {sorted(excluded)}")
        if not order:
            raise PlanningError("no_objects", "Every mentioned object was excluded; nothing to do.")

        order = self._apply_order_modifiers(text, order)
        steps = [
            TaskStep(
                object_name=obj,
                target_location=seen[obj][0],
                execution_order=i + 1,
                action_type=ActionType.PICK_AND_PLACE,
                confidence=round(seen[obj][1], 3),
            )
            for i, obj in enumerate(order)
        ]
        return TaskPlan(
            instruction=instruction,
            steps=steps,
            planner=self.name,
            confidence=round(min(s.confidence for s in steps), 3),
            notes=notes,
        )

    @staticmethod
    def _split_pairs(clause: str) -> list[str]:
        """Split "X into A and Y into B" into two parts, but keep "X and Y into A" together.

        A part is closed only when it has both an object and its own target phrase; parts
        without objects ("... and place it into its bin") are merged into the previous part.
        """
        raw = re.split(r"\s*,?\s*\band\b\s+", clause)
        parts: list[str] = []
        buf = ""
        for piece in raw:
            buf = f"{buf} and {piece}" if buf else piece
            if _find_objects(buf) and _TARGET_PREP.search(buf):
                parts.append(buf)
                buf = ""
        if buf:
            if parts and not _find_objects(buf):
                parts[-1] = f"{parts[-1]} and {buf}"
            else:
                parts.append(buf)
        return parts

    @staticmethod
    def _apply_order_modifiers(text: str, order: list[str]) -> list[str]:
        order = list(order)

        def obj_in(fragment: str) -> str | None:
            ms = _find_objects(fragment)
            return ms[0].obj if ms else None

        # "X after Y"  → Y before X ;  "X before Y" → X before Y
        for m in re.finditer(r"([a-z ]+?)\s+\b(after|before)\b\s+(?:the\s+)?([a-z ]+)", text):
            a, rel, b = obj_in(m.group(1).split(",")[-1]), m.group(2), obj_in(m.group(3))
            if a and b and a in order and b in order and a != b:
                first, second = (b, a) if rel == "after" else (a, b)
                order.remove(first)
                order.insert(order.index(second), first)
        # "... X last"
        m = re.search(r"(?:the\s+)?([a-z ]+?)\s+last\b", text)
        if m:
            obj = obj_in(m.group(1).split(",")[-1])
            if obj in order:
                order.remove(obj)
                order.append(obj)
        # "X first" (wins over everything else)
        m = re.search(r"(?:the\s+)?([a-z ]+?)\s+first\b|\bfirst\b,?\s+(?:the\s+)?([a-z ]+)", text)
        if m:
            obj = obj_in((m.group(1) or "").split(",")[-1]) or obj_in(m.group(2) or "")
            if obj in order:
                order.remove(obj)
                order.insert(0, obj)
        return order
