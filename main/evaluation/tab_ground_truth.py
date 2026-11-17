"""Strict, inference-independent evaluation against six-line ASCII guitar TAB.

String indices follow GuitarSet (0 = low E, 5 = high E). Column spacing and
bar numbers are presentation: this evaluator compares ordered onset groups,
including the simultaneous notes in each group. It does not claim to evaluate
tempo, durations, or rests, which ASCII spacing alone does not specify.

Run ``python -m main.evaluation.tab_ground_truth --truth reference.txt
--prediction generated.txt --output comparison.json``. JSON predictions use
``{"events": [{"notes": [{"string_idx": 5, "fret_idx": 7}]}]}``;
an optional note ``technique`` is one of /, \\, h, p. No reference is passed
to any inference function in this module.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable, Sequence


LABELS = ("e", "B", "G", "D", "A", "E")
TECHNIQUES = frozenset(("", "/", "\\", "h", "p"))


@dataclass(frozen=True, order=True)
class TabPosition:
    string_idx: int
    fret_idx: int
    technique: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.string_idx, int) or not 0 <= self.string_idx <= 5:
            raise ValueError("string_idx must be an integer from 0 to 5")
        if not isinstance(self.fret_idx, int) or not 0 <= self.fret_idx <= 24:
            raise ValueError("fret_idx must be an integer from 0 to 24")
        if self.technique not in TECHNIQUES:
            raise ValueError(f"unsupported technique: {self.technique!r}")


@dataclass(frozen=True)
class TabEvent:
    notes: tuple[TabPosition, ...]
    measure: int | None = None
    column: int | None = None
    onset_sec: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "notes", tuple(sorted(self.notes)))
        if not self.notes:
            raise ValueError("an onset event must contain at least one note")
        strings = [n.string_idx for n in self.notes]
        if len(strings) != len(set(strings)):
            raise ValueError("two notes cannot occupy one string in the same event")


def parse_ascii_tab(text: str) -> list[TabEvent]:
    """Parse aligned six-row systems without silently dropping unknown syntax.

    Number-only measure headings and comment lines starting with # are allowed.
    Frets align at their first digit, including two-digit frets. Technique marks
    attach to the immediately following fret; their presence is evaluated
    separately from string/fret accuracy.
    """
    systems: list[list[str]] = []
    rows: list[str] = []
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#") or re.fullmatch(r"[\d\s]+", line):
            if rows:
                raise ValueError(f"incomplete tab system before line {line_number}")
            continue
        if not re.match(r"^[eBGDAE]\|", line):
            raise ValueError(f"unrecognized TAB syntax on line {line_number}: {line!r}")
        expected_label = LABELS[len(rows)]
        if line[0] != expected_label:
            raise ValueError(f"expected string {expected_label} on line {line_number}")
        rows.append(line)
        if len(rows) == 6:
            systems.append(rows)
            rows = []
    if rows:
        raise ValueError("incomplete tab system: expected all six strings")
    if not systems:
        raise ValueError("no six-line TAB system found")

    events: list[TabEvent] = []
    measure_number = 0
    for system in systems:
        if all(row[1:] == "|" for row in system):
            continue  # Empty output from guitar_tab.render_ascii_tab.
        if any(not row.endswith("|") for row in system):
            raise ValueError("every tab row must end with a bar delimiter")
        measures = [row.split("|")[1:-1] for row in system]
        if len({len(row) for row in measures}) != 1:
            raise ValueError("all six strings must have the same bar count")
        for measure_rows in zip(*measures):
            measure_number += 1
            if len({len(row) for row in measure_rows}) != 1:
                raise ValueError(f"unaligned string widths in measure {measure_number}")
            columns: dict[int, list[TabPosition]] = {}
            for row_number, row in enumerate(measure_rows):
                residual = list(row)
                for match in re.finditer(r"([/\\hp]?)(\d+)", row):
                    technique, fret_text = match.groups()
                    column = match.end() - len(fret_text)
                    columns.setdefault(column, []).append(
                        TabPosition(5 - row_number, int(fret_text), technique)
                    )
                    residual[match.start():match.end()] = "-" * len(match.group())
                if any(char not in "- " for char in residual):
                    raise ValueError(f"unsupported notation in measure {measure_number}: {row!r}")
            events.extend(
                TabEvent(tuple(columns[column]), measure_number, column)
                for column in sorted(columns)
            )
    return events


def events_from_tab_notes(notes: Iterable, *, tolerance: float = 0.05) -> list[TabEvent]:
    """Group TabNote-compatible objects by their first onset, not chain distance."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite nonnegative number")
    ordered = sorted(notes, key=lambda n: (n.onset_sec, n.string_idx, n.fret_idx))
    groups: list[list] = []
    for note in ordered:
        if not math.isfinite(note.onset_sec):
            raise ValueError("onsets must be finite")
        if not groups or note.onset_sec - groups[-1][0].onset_sec > tolerance + 1e-9:
            groups.append([])
        groups[-1].append(note)
    return [TabEvent(tuple(TabPosition(n.string_idx, n.fret_idx,
                                      getattr(n, "technique", "")) for n in group),
                     onset_sec=group[0].onset_sec) for group in groups]


def events_from_json(data: dict) -> list[TabEvent]:
    """Read the documented event JSON format; reject duplicate physical strings."""
    return [TabEvent(tuple(TabPosition(**note) for note in event["notes"]),
                     event.get("measure"), event.get("column"), event.get("onset_sec"))
            for event in data["events"]]


def _key(event: TabEvent, include_techniques: bool = False) -> tuple:
    return tuple((n.string_idx, n.fret_idx, n.technique) if include_techniques
                 else (n.string_idx, n.fret_idx) for n in event.notes)


def evaluate_events(expected: Sequence[TabEvent], actual: Sequence[TabEvent]) -> dict:
    """Return exact sequence checks plus deterministic Levenshtein diagnostics.

    Unit edit costs apply to whole onset groups. Ties maximize shared notes.
    Note precision/recall are computed within that alignment; substitutions can
    retain shared positions.
    A missing, extra, regrouped, repeated, or wrong-string note prevents an exact
    position match, even when its MIDI pitch occurs elsewhere in the sequence.
    """
    a, b = [_key(e) for e in expected], [_key(e) for e in actual]
    rows, cols = len(a), len(b)
    distance = [[0] * (cols + 1) for _ in range(rows + 1)]
    shared = [[0] * (cols + 1) for _ in range(rows + 1)]
    steps = [[""] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows + 1):
        distance[i][0] = i
        steps[i][0] = "delete"
    for j in range(cols + 1):
        distance[0][j] = j
        steps[0][j] = "insert"
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            overlap = sum((Counter(a[i - 1]) & Counter(b[j - 1])).values())
            choices = [
                (distance[i - 1][j - 1] + (a[i - 1] != b[j - 1]),
                 -(shared[i - 1][j - 1] + overlap), 0, "diagonal"),
                (distance[i - 1][j] + 1, -shared[i - 1][j], 1, "delete"),
                (distance[i][j - 1] + 1, -shared[i][j - 1], 2, "insert"),
            ]
            cost, negative_shared, _, step = min(choices)
            distance[i][j], shared[i][j], steps[i][j] = cost, -negative_shared, step
    aligned = []
    i, j = rows, cols
    while i or j:
        if steps[i][j] == "diagonal":
            aligned.append(("match" if a[i - 1] == b[j - 1] else "substitute", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif steps[i][j] == "delete":
            aligned.append(("delete", i - 1, None))
            i -= 1
        else:
            aligned.append(("insert", None, j - 1))
            j -= 1
    aligned.reverse()
    diagnostics = []
    matched_notes = 0
    for operation, ei, ai in aligned:
        wanted, got = Counter(a[ei] if ei is not None else ()), Counter(b[ai] if ai is not None else ())
        matched_notes += sum((wanted & got).values())
        technique_match = (ei is not None and ai is not None
                           and _key(expected[ei], True) == _key(actual[ai], True))
        diagnostics.append({
            "operation": operation, "expected_index": ei, "actual_index": ai,
            "expected": asdict(expected[ei]) if ei is not None else None,
            "actual": asdict(actual[ai]) if ai is not None else None,
            "missing_positions": list((wanted - got).elements()),
            "extra_positions": list((got - wanted).elements()),
            "techniques_match": technique_match,
        })
    expected_count = sum(map(len, a))
    actual_count = sum(map(len, b))
    precision = matched_notes / actual_count if actual_count else float(not expected_count)
    recall = matched_notes / expected_count if expected_count else float(not actual_count)
    positions_exact = a == b
    techniques_exact = [_key(e, True) for e in expected] == [_key(e, True) for e in actual]
    return {
        "comparison_scope": "ordered simultaneous string/fret events and technique marks; excludes timing, rests, bar layout, and whitespace",
        "exact_match": positions_exact and techniques_exact,
        "positions_exact": positions_exact, "techniques_exact": techniques_exact,
        "expected_events": rows, "actual_events": cols,
        "expected_notes": expected_count, "actual_notes": actual_count,
        "matched_events": sum(op == "match" for op, _, _ in aligned),
        "matched_notes": matched_notes,
        "missing_notes": expected_count - matched_notes,
        "extra_notes": actual_count - matched_notes,
        "event_edit_distance": distance[-1][-1],
        "note_precision": precision, "note_recall": recall,
        "note_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "errors": [d for d in diagnostics if d["operation"] != "match" or not d["techniques_match"]],
        "alignment": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    truth_text = args.truth.read_text()
    prediction_text = args.prediction.read_text()
    predicted = (events_from_json(json.loads(prediction_text))
                 if args.prediction.suffix.lower() == ".json" else parse_ascii_tab(prediction_text))
    report = evaluate_events(parse_ascii_tab(truth_text), predicted)
    report["text_exact"] = truth_text == prediction_text if args.prediction.suffix.lower() != ".json" else None
    report["truth_path"] = str(args.truth)
    report["prediction_path"] = str(args.prediction)
    report["truth_sha256"] = hashlib.sha256(args.truth.read_bytes()).hexdigest()
    report["prediction_sha256"] = hashlib.sha256(args.prediction.read_bytes()).hexdigest()
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end="")
    raise SystemExit(0 if report["exact_match"] else 1)


if __name__ == "__main__":
    main()
