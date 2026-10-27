#!/usr/bin/env python3
"""Evaluate framewise active-pitch and generated ASCII-tab metrics on GuitarSet."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pretty_midi
from mir_eval import multipitch

from pipeline.guitar_tab import (
    MAX_FRET,
    ONSET_GROUP_TOLERANCE_SEC,
    STANDARD_TUNING_MIDI,
    TAB_STRING_LABELS,
    TabNote,
    midi_to_tab_notes,
    render_ascii_tab,
)
from pipeline.video_to_score import extract_audio, read_model_audio, transcribe_audio


DATASET = "jhartquist/guitarset"
ROWS_API = "https://datasets-server.huggingface.co/rows"
ROW_RANGES = ((0, 2), (16, 2), (60, 2), (76, 2))
FRAME_RATE = 100.0
PITCH_WINDOW_SEMITONES = 0.5
TAB_ONSET_TOLERANCE_SEC = ONSET_GROUP_TOLERANCE_SEC
CACHE_DIR = Path(tempfile.gettempdir()) / "guitar-cv-amt-mt3-benchmark"
REPORT_MD = Path("research/mt3_guitarset_framewise_f1.md")
COMPARISON_REPORT_MD = Path("research/mt3_guitarset_tab_comparisons.md")
REPORT_JSON = Path("research/mt3_guitarset_framewise_f1.json")


def fetch_rows(offset: int, length: int) -> tuple[list[dict[str, Any]], str | None]:
    query = urlencode(
        {
            "dataset": DATASET,
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": length,
        }
    )
    request = Request(
        f"{ROWS_API}?{query}",
        headers={"User-Agent": "guitar-cv-amt-mt3-framewise/1.0"},
    )
    with urlopen(request, timeout=90) as response:
        data = json.load(response)
        revision = response.headers.get("x-revision")
    rows = [item["row"] for item in data["rows"]]
    if len(rows) != length:
        raise RuntimeError(f"Expected {length} rows at offset {offset}; received {len(rows)}.")
    return rows, revision


def fetch_subset(limit: int) -> tuple[list[dict[str, Any]], str | None]:
    selected = []
    revisions = set()
    for offset, length in ROW_RANGES[: (limit + 1) // 2]:
        rows, revision = fetch_rows(offset, length)
        selected.extend(rows)
        if revision:
            revisions.add(revision)
    if len(revisions) > 1:
        raise RuntimeError(f"Dataset changed during download: {sorted(revisions)}")
    return selected[:limit], next(iter(revisions), None)


def audio_source(row: dict[str, Any]) -> str:
    field = row["audio_mono_mic"]
    if isinstance(field, list):
        field = field[0]
    return field["src"]


def fetch_audio(row: dict[str, Any], directory: Path) -> Path:
    target = directory / f"{row['track_id']}.wav"
    if target.is_file() and target.stat().st_size > 44:
        return target
    request = Request(audio_source(row), headers={"User-Agent": "guitar-cv-amt-framewise/1.0"})
    partial = target.with_suffix(".wav.part")
    with urlopen(request, timeout=180) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output)
    partial.replace(target)
    return target


def reference_notes(row: dict[str, Any]) -> list[tuple[float, float, float]]:
    result = []
    for note in row["notes"]:
        onset, offset, pitch = (
            float(note["onset_s"]),
            float(note["offset_s"]),
            float(note["midi"]),
        )
        if offset > onset and np.isfinite([onset, offset, pitch]).all():
            result.append((onset, offset, pitch))
    return result


def _parse_string_index(value: Any) -> int | None:
    """Return GuitarSet's low-to-high string index, or ``None`` for a bad label."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        index = int(value)
    elif isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            return None
        index = int(value)
    elif isinstance(value, str):
        raw_label = value.strip()
        if raw_label == "E":
            return 0
        if raw_label == "e":
            return 5
        label = raw_label.lower()
        try:
            numeric = float(label)
        except ValueError:
            numeric = None
        if numeric is not None:
            if not np.isfinite(numeric) or not numeric.is_integer():
                return None
            index = int(numeric)
        else:
            normalized = "".join(character for character in label if character.isalnum())
            labels = {
                "lowe": 0,
                "e2": 0,
                "a": 1,
                "a2": 1,
                "d": 2,
                "d3": 2,
                "g": 3,
                "g3": 3,
                "b": 4,
                "b3": 4,
                "highe": 5,
                "e4": 5,
            }
            index = labels.get(normalized, -1)
    else:
        return None
    return index if 0 <= index < len(STANDARD_TUNING_MIDI) else None


def note_to_tab_note(note: Mapping[str, Any]) -> TabNote | None:
    """Convert one GuitarSet note row to a playable, immutable tab note."""
    try:
        onset = float(note["onset_s"])
        offset = float(note["offset_s"])
        pitch = float(note["midi"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite([onset, offset, pitch]).all() or offset <= onset:
        return None
    string_idx = _parse_string_index(note.get("string"))
    if string_idx is None:
        return None
    pitch_midi = int(round(pitch))
    fret_idx = pitch_midi - int(STANDARD_TUNING_MIDI[string_idx])
    if not 0 <= fret_idx <= MAX_FRET:
        return None
    return TabNote(
        onset_sec=onset,
        offset_sec=offset,
        pitch_midi=pitch_midi,
        string_idx=string_idx,
        fret_idx=fret_idx,
    )


def reference_tab_notes(row: Mapping[str, Any]) -> list[TabNote]:
    """Build deterministic playable reference tabs from a GuitarSet row."""
    converted = [
        converted_note
        for raw_note in row["notes"]
        if (converted_note := note_to_tab_note(raw_note)) is not None
    ]
    return sorted(
        converted,
        key=lambda note: (
            note.onset_sec,
            note.offset_sec,
            note.pitch_midi,
            note.string_idx,
            note.fret_idx,
        ),
    )


def _is_playable_tab_note(note: TabNote) -> bool:
    try:
        return (
            np.isfinite([note.onset_sec, note.offset_sec, note.pitch_midi]).all()
            and note.offset_sec > note.onset_sec
            and 0 <= int(note.string_idx) < len(STANDARD_TUNING_MIDI)
            and 0 <= int(note.fret_idx) <= MAX_FRET
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _match_indices(
    reference: list[TabNote],
    predicted: list[TabNote],
    *,
    key,
    onset_tolerance: float,
) -> list[tuple[int, int]]:
    """Match events by key and onset with deterministic one-to-one greedy sweeps."""
    if onset_tolerance < 0:
        raise ValueError("onset_tolerance must be non-negative")
    comparison_tolerance = onset_tolerance + (
        1e-12 if onset_tolerance > 0 else 0.0
    )
    reference_groups: dict[Any, list[int]] = {}
    predicted_groups: dict[Any, list[int]] = {}
    for index, note in enumerate(reference):
        if _is_playable_tab_note(note):
            reference_groups.setdefault(key(note), []).append(index)
    for index, note in enumerate(predicted):
        if _is_playable_tab_note(note):
            predicted_groups.setdefault(key(note), []).append(index)

    pairs: list[tuple[int, int]] = []
    for event_key in sorted(
        reference_groups.keys() & predicted_groups.keys(), key=repr
    ):
        reference_indices = sorted(
            reference_groups[event_key],
            key=lambda index: (
                float(reference[index].onset_sec),
                float(reference[index].offset_sec),
                index,
            ),
        )
        predicted_indices = sorted(
            predicted_groups[event_key],
            key=lambda index: (
                float(predicted[index].onset_sec),
                float(predicted[index].offset_sec),
                index,
            ),
        )
        reference_position = 0
        predicted_position = 0
        while (
            reference_position < len(reference_indices)
            and predicted_position < len(predicted_indices)
        ):
            reference_index = reference_indices[reference_position]
            predicted_index = predicted_indices[predicted_position]
            reference_onset = float(reference[reference_index].onset_sec)
            predicted_onset = float(predicted[predicted_index].onset_sec)
            if predicted_onset < reference_onset - comparison_tolerance:
                predicted_position += 1
            elif reference_onset < predicted_onset - comparison_tolerance:
                reference_position += 1
            else:
                pairs.append((reference_index, predicted_index))
                reference_position += 1
                predicted_position += 1
    return sorted(pairs)


def match_tab_events(
    reference: list[TabNote],
    predicted: list[TabNote],
    *,
    onset_tolerance: float = TAB_ONSET_TOLERANCE_SEC,
) -> list[tuple[TabNote, TabNote]]:
    """Return one-to-one exact string/fret event matches."""
    pairs = _match_indices(
        reference,
        predicted,
        key=lambda note: (int(note.string_idx), int(note.fret_idx)),
        onset_tolerance=onset_tolerance,
    )
    return [
        (reference[reference_index], predicted[predicted_index])
        for reference_index, predicted_index in pairs
    ]


def _pitch_match_indices(
    reference: list[TabNote],
    predicted: list[TabNote],
    *,
    onset_tolerance: float,
) -> list[tuple[int, int]]:
    return _match_indices(
        reference,
        predicted,
        key=lambda note: int(round(note.pitch_midi)),
        onset_tolerance=onset_tolerance,
    )


def evaluate_tab_notes(
    reference: list[TabNote],
    predicted: list[TabNote],
    *,
    predicted_midi_count: int | None = None,
    reference_note_count: int | None = None,
    onset_tolerance: float = TAB_ONSET_TOLERANCE_SEC,
) -> dict[str, Any]:
    """Calculate exact-tab and pitch-matched assignment metrics.

    Exact-tab precision uses every predicted MIDI event as its denominator,
    including events omitted by ``midi_to_tab_notes`` because they could not be
    assigned to a playable string/fret.
    """
    playable_reference = [note for note in reference if _is_playable_tab_note(note)]
    playable_predicted = [note for note in predicted if _is_playable_tab_note(note)]
    if predicted_midi_count is None:
        predicted_midi_count = len(predicted)
    if predicted_midi_count < len(playable_predicted):
        raise ValueError("predicted_midi_count cannot be smaller than rendered tab notes")
    if reference_note_count is None:
        reference_note_count = len(reference)

    exact_pairs = _match_indices(
        playable_reference,
        playable_predicted,
        key=lambda note: (int(note.string_idx), int(note.fret_idx)),
        onset_tolerance=onset_tolerance,
    )
    pitch_pairs = _pitch_match_indices(
        playable_reference,
        playable_predicted,
        onset_tolerance=onset_tolerance,
    )
    assignment_correct = sum(
        playable_reference[reference_index].string_idx
        == playable_predicted[predicted_index].string_idx
        and playable_reference[reference_index].fret_idx
        == playable_predicted[predicted_index].fret_idx
        for reference_index, predicted_index in pitch_pairs
    )
    true_positives = len(exact_pairs)
    reference_count = len(playable_reference)
    false_positives = predicted_midi_count - true_positives
    false_negatives = reference_count - true_positives
    precision = true_positives / predicted_midi_count if predicted_midi_count else 0.0
    recall = true_positives / reference_count if reference_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    invalid_reference_count = max(0, int(reference_note_count) - reference_count)
    return {
        "score": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "assignment_accuracy": float(
                assignment_correct / len(pitch_pairs) if pitch_pairs else 0.0
            ),
        },
        "counts": {
            "reference_note_count": int(reference_note_count),
            "reference_tab_note_count": reference_count,
            "unplayable_or_invalid_reference_note_count": invalid_reference_count,
            "predicted_midi_note_count": int(predicted_midi_count),
            "rendered_predicted_tab_note_count": len(playable_predicted),
            "unplayable_or_unassigned_predicted_note_count": (
                predicted_midi_count - len(playable_predicted)
            ),
            "exact_tab_true_positive_count": true_positives,
            "exact_tab_false_positive_count": false_positives,
            "exact_tab_false_negative_count": false_negatives,
            "pitch_matched_playable_note_count": len(pitch_pairs),
            "assignment_correct_count": assignment_correct,
        },
    }


def aggregate_tab_evaluations(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool per-clip tab results without matching events across clip boundaries."""
    count_keys = (
        "reference_note_count",
        "reference_tab_note_count",
        "unplayable_or_invalid_reference_note_count",
        "predicted_midi_note_count",
        "rendered_predicted_tab_note_count",
        "unplayable_or_unassigned_predicted_note_count",
        "exact_tab_true_positive_count",
        "exact_tab_false_positive_count",
        "exact_tab_false_negative_count",
        "pitch_matched_playable_note_count",
        "assignment_correct_count",
    )
    counts = {
        key: sum(int(evaluation["counts"].get(key, 0)) for evaluation in evaluations)
        for key in count_keys
    }
    true_positives = counts["exact_tab_true_positive_count"]
    reference_count = counts["reference_tab_note_count"]
    predicted_count = counts["predicted_midi_note_count"]
    precision = true_positives / predicted_count if predicted_count else 0.0
    recall = true_positives / reference_count if reference_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    pitch_matches = counts["pitch_matched_playable_note_count"]
    return {
        "score": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "assignment_accuracy": float(
                counts["assignment_correct_count"] / pitch_matches
                if pitch_matches
                else 0.0
            ),
        },
        "counts": counts,
    }


def write_tab_artifacts(
    reference: list[TabNote],
    predicted: list[TabNote],
    *,
    reference_path: Path,
    predicted_path: Path,
) -> dict[str, str]:
    """Render and write one clip's reference/predicted tab artifacts."""
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    predicted_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_text(render_ascii_tab(reference), encoding="utf-8")
    predicted_path.write_text(render_ascii_tab(predicted), encoding="utf-8")
    return {
        "reference": str(reference_path),
        "predicted": str(predicted_path),
    }


def _comparison_groups(
    notes: Iterable[TabNote], *, onset_tolerance: float
) -> list[tuple[TabNote, ...]]:
    """Group valid tab notes using the renderer's inclusive onset tolerance."""
    if onset_tolerance < 0:
        raise ValueError("onset_tolerance must be non-negative")
    valid_notes: list[TabNote] = []
    for note in notes:
        try:
            if (
                0 <= int(note.string_idx) < len(STANDARD_TUNING_MIDI)
                and 0 <= int(note.fret_idx) <= MAX_FRET
                and math.isfinite(float(note.onset_sec))
            ):
                valid_notes.append(note)
        except (AttributeError, TypeError, ValueError):
            continue
    ordered = sorted(
        valid_notes,
        key=lambda note: (
            float(note.onset_sec),
            int(note.pitch_midi),
            float(note.offset_sec),
            int(note.string_idx),
            int(note.fret_idx),
        ),
    )
    groups: list[list[TabNote]] = []
    group_start: float | None = None
    for note in ordered:
        onset = float(note.onset_sec)
        if (
            group_start is None
            or onset - group_start > onset_tolerance + 1e-9
        ):
            groups.append([])
            group_start = onset
        groups[-1].append(note)
    return [tuple(group) for group in groups]


def align_tab_event_groups(
    reference: Iterable[TabNote],
    predicted: Iterable[TabNote],
    *,
    onset_tolerance: float = TAB_ONSET_TOLERANCE_SEC,
) -> list[tuple[tuple[TabNote, ...] | None, tuple[TabNote, ...] | None]]:
    """Align independently grouped events by onset, never by sequence position."""
    reference_groups = _comparison_groups(
        reference, onset_tolerance=onset_tolerance
    )
    predicted_groups = _comparison_groups(
        predicted, onset_tolerance=onset_tolerance
    )
    comparison_tolerance = onset_tolerance + (
        1e-12 if onset_tolerance > 0 else 0.0
    )
    aligned: list[
        tuple[tuple[TabNote, ...] | None, tuple[TabNote, ...] | None]
    ] = []
    reference_position = 0
    predicted_position = 0
    while (
        reference_position < len(reference_groups)
        and predicted_position < len(predicted_groups)
    ):
        reference_group = reference_groups[reference_position]
        predicted_group = predicted_groups[predicted_position]
        reference_onset = float(reference_group[0].onset_sec)
        predicted_onset = float(predicted_group[0].onset_sec)
        if predicted_onset < reference_onset - comparison_tolerance:
            aligned.append((None, predicted_group))
            predicted_position += 1
        elif reference_onset < predicted_onset - comparison_tolerance:
            aligned.append((reference_group, None))
            reference_position += 1
        else:
            aligned.append((reference_group, predicted_group))
            reference_position += 1
            predicted_position += 1
    aligned.extend(
        (group, None) for group in reference_groups[reference_position:]
    )
    aligned.extend(
        (None, group) for group in predicted_groups[predicted_position:]
    )
    return aligned


def _comparison_cells(
    event: tuple[TabNote, ...] | None,
) -> dict[int, int]:
    if event is None:
        return {}
    cells: dict[int, int] = {}
    for note in event:
        cells.setdefault(int(note.string_idx), int(note.fret_idx))
    return cells


def _comparison_side_lines(
    events: list[tuple[TabNote, ...] | None],
    widths: list[int],
) -> list[str]:
    lines: list[str] = []
    for row_index, label in enumerate(TAB_STRING_LABELS):
        string_idx = len(STANDARD_TUNING_MIDI) - 1 - row_index
        line = [f"{label}|"]
        for event, width in zip(events, widths):
            fret = _comparison_cells(event).get(string_idx)
            line.append("-" * width if fret is None else str(fret).ljust(width, "-"))
        if widths:
            line.append("|")
        lines.append("".join(line))
    return lines


def render_tab_comparison_systems(
    reference: Iterable[TabNote],
    predicted: Iterable[TabNote],
    *,
    events_per_system: int = 16,
    onset_tolerance: float = TAB_ONSET_TOLERANCE_SEC,
) -> list[str]:
    """Render aligned Ground truth/Generated systems with shared cell widths."""
    if events_per_system < 1:
        raise ValueError("events_per_system must be at least 1")
    aligned = align_tab_event_groups(
        reference, predicted, onset_tolerance=onset_tolerance
    )
    systems: list[str] = []
    chunks = (
        [aligned[start : start + events_per_system]
         for start in range(0, len(aligned), events_per_system)]
        if aligned
        else [[]]
    )
    for chunk in chunks:
        widths = []
        for reference_event, predicted_event in chunk:
            reference_cells = _comparison_cells(reference_event)
            predicted_cells = _comparison_cells(predicted_event)
            max_digits = max(
                (len(str(fret))
                 for fret in (*reference_cells.values(), *predicted_cells.values())),
                default=1,
            )
            widths.append(max(2, max_digits + 1))
        reference_lines = _comparison_side_lines(
            [event[0] for event in chunk], widths
        )
        predicted_lines = _comparison_side_lines(
            [event[1] for event in chunk], widths
        )
        side_width = max(
            len("Ground truth"),
            len("Generated"),
            2 + sum(widths) + (1 if widths else 0),
        )
        separator = "  "
        lines = [
            f"{'Ground truth':<{side_width}}"
            f"{separator}{'Generated':<{side_width}}"
        ]
        lines.extend(
            f"{reference_line:<{side_width}}{separator}{predicted_line:<{side_width}}"
            for reference_line, predicted_line in zip(
                reference_lines, predicted_lines
            )
        )
        systems.append("\n".join(lines))
    return systems


def render_tab_comparison(
    reference: Iterable[TabNote],
    predicted: Iterable[TabNote],
    *,
    events_per_system: int = 16,
    onset_tolerance: float = TAB_ONSET_TOLERANCE_SEC,
) -> str:
    """Render all aligned comparison systems as one deterministic text block."""
    return "\n\n".join(
        render_tab_comparison_systems(
            reference,
            predicted,
            events_per_system=events_per_system,
            onset_tolerance=onset_tolerance,
        )
    )


def predicted_notes(path: Path) -> list[tuple[float, float, float]]:
    midi = pretty_midi.PrettyMIDI(str(path))
    return [
        (float(note.start), float(note.end), float(note.pitch))
        for instrument in midi.instruments
        if not bool(getattr(instrument, "is_drum", False))
        for note in instrument.notes
        if note.end > note.start
    ]


def pitch_frames(
    notes: list[tuple[float, float, float]], duration_sec: float
) -> list[np.ndarray]:
    frame_count = max(1, int(math.ceil(duration_sec * FRAME_RATE)))
    centers = (np.arange(frame_count, dtype=float) + 0.5) / FRAME_RATE
    return [
        np.asarray(
            sorted(
                {
                    pitch
                    for onset, offset, pitch in notes
                    if onset <= time_sec < offset
                }
            ),
            dtype=float,
        )
        for time_sec in centers
    ]


def framewise_f1(
    reference_frames: list[np.ndarray], estimate_frames: list[np.ndarray]
) -> dict[str, float]:
    true_positives = multipitch.compute_num_true_positives(
        reference_frames,
        estimate_frames,
        window=PITCH_WINDOW_SEMITONES,
    )
    reference_counts = multipitch.compute_num_freqs(reference_frames)
    estimate_counts = multipitch.compute_num_freqs(estimate_frames)
    precision, recall, _ = multipitch.compute_accuracy(
        true_positives, reference_counts, estimate_counts
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def render_comparison_report(
    report: Mapping[str, Any],
    comparison_tabs: Mapping[str, tuple[list[TabNote], list[TabNote]]],
    output_path: Path,
) -> None:
    """Write one aligned, human-readable comparison report for every clip."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# GuitarSet ground-truth versus generated tab comparisons",
        "",
        "This report is generated local evaluation output; upstream GuitarSet rows "
        "are not modified.",
        "",
        "Reference and generated notes are first grouped independently by onset. "
        "Groups share one chronological event column only when their onset "
        "difference is within the inclusive 50 ms tolerance. Unmatched reference "
        "or generated events keep their own column, and dashes on the missing "
        "side represent those unmatched events. Each event column has one width "
        "shared by both sides, and systems wrap deterministically at 16 columns.",
        "",
    ]
    for clip in report["clips"]:
        track_id = str(clip["track_id"])
        try:
            reference, predicted = comparison_tabs[track_id]
        except KeyError as error:
            raise ValueError(f"Missing tab comparison data for {track_id}") from error
        tab_score = clip.get("tab", {}).get("score", {})
        lines.extend(
            [
                f"## {track_id}",
                "",
                f"Player: {clip['player']}; style: {clip['style']}; "
                f"duration: {float(clip['duration_seconds']):.3f} seconds.",
                "",
                f"Exact-tab precision / recall / F1: "
                f"**{100 * float(tab_score.get('precision', 0.0)):.6f}% / "
                f"{100 * float(tab_score.get('recall', 0.0)):.6f}% / "
                f"{100 * float(tab_score.get('f1', 0.0)):.6f}%**.",
                f"String/fret assignment accuracy: "
                f"**{100 * float(tab_score.get('assignment_accuracy', 0.0)):.6f}%**.",
                "",
            ]
        )
        systems = render_tab_comparison_systems(reference, predicted)
        for system_index, system in enumerate(systems, start=1):
            lines.extend(
                [
                    f"### Comparison system {system_index}",
                    "",
                    "```text",
                    system,
                    "```",
                    "",
                ]
            )
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def render_report(report: dict[str, Any], markdown: Path, json_path: Path) -> None:
    markdown.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    score = report["score"]
    tab_report = report.get("tab", {})
    tab_score = dict(tab_report.get("score", {}))
    tab_score.setdefault("precision", 0.0)
    tab_score.setdefault("recall", 0.0)
    tab_score.setdefault("f1", 0.0)
    tab_score.setdefault("assignment_accuracy", 0.0)
    tab_counts = tab_report.get("counts", {})
    lines = [
        "# GuitarSet framewise active-pitch result",
        "",
        f"Run date: {report['run_date']}",
        "",
        f"MR-MT3 scored **{100 * score['f1']:.1f}% framewise active-pitch F1** on this fixed {report['clip_count']}-clip GuitarSet subset.",
        "",
        f"Precision: {100 * score['precision']:.1f}%. Recall: {100 * score['recall']:.1f}%.",
        "",
        "The score compares active MIDI pitches at 10 ms frame centers. A pitch matches within 0.5 semitones; F1 is the harmonic mean of framewise precision and recall. This measures pitch activity overlap rather than note-by-note onset or offset accuracy.",
        "",
        f"The subset contains {report['clip_count']} acoustic microphone recordings from {report['player_count']} players ({report['style_counts']['comp']} comp and {report['style_counts']['solo']} solo), totaling {report['audio_seconds']:.1f} seconds. The audio was converted by this repository's FFmpeg extraction path and transcribed by its MR-MT3 path.",
        f"Fixed Hugging Face row ranges: {tuple(tuple(row_range) for row_range in ROW_RANGES)}; the default limit evaluates all 8 selected clips.",
        "",
        "| Track | Player | Style | Framewise active-pitch F1 | Exact tab P/R/F1 | Assignment accuracy | Ref notes | Pred MIDI | Rendered tab | Unplayable/unassigned |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for clip in report["clips"]:
        clip_tab = clip.get("tab", {})
        clip_tab_score = dict(clip_tab.get("score", {}))
        clip_tab_score.setdefault("precision", 0.0)
        clip_tab_score.setdefault("recall", 0.0)
        clip_tab_score.setdefault("f1", 0.0)
        clip_tab_score.setdefault("assignment_accuracy", 0.0)
        clip_tab_counts = clip_tab.get("counts", {})
        lines.append(
            f"| `{clip['track_id']}` | {clip['player']} | {clip['style']} | "
            f"{100 * clip['score']['f1']:.1f}% | "
            f"{100 * clip_tab_score['precision']:.1f}% / "
            f"{100 * clip_tab_score['recall']:.1f}% / "
            f"{100 * clip_tab_score['f1']:.1f}% | "
            f"{100 * clip_tab_score['assignment_accuracy']:.1f}% | "
            f"{clip_tab_counts.get('reference_note_count', 0)} | "
            f"{clip_tab_counts.get('predicted_midi_note_count', 0)} | "
            f"{clip_tab_counts.get('rendered_predicted_tab_note_count', 0)} | "
            f"{clip_tab_counts.get('unplayable_or_unassigned_predicted_note_count', 0)} |"
        )
    comparison_report_path = report.get("comparison_report")
    if comparison_report_path:
        comparison_display = str(comparison_report_path)
        comparison_link = Path(
            os.path.relpath(comparison_display, start=markdown.parent)
        ).as_posix()
        lines.extend(
            [
                "",
                f"Open the [ground-truth versus generated tab comparison report]"
                f"({comparison_link}); output path: `{comparison_display}`.",
            ]
        )
    lines.extend(["", "Per-clip tab artifacts:"])
    for clip in report["clips"]:
        artifacts = clip.get("tab", {}).get("artifacts", {})
        lines.append(
            f"- `{clip['track_id']}` reference: `{artifacts.get('reference', '')}`; "
            f"predicted: `{artifacts.get('predicted', '')}`"
        )
        clip_counts = clip.get("tab", {}).get("counts", {})
        lines.append(
            f"  counts: reference tab {clip_counts.get('reference_tab_note_count', 0)}, "
            f"predicted MIDI {clip_counts.get('predicted_midi_note_count', 0)}, "
            f"rendered {clip_counts.get('rendered_predicted_tab_note_count', 0)}, "
            f"unplayable/unassigned {clip_counts.get('unplayable_or_unassigned_predicted_note_count', 0)}; "
            f"exact TP/FP/FN {clip_counts.get('exact_tab_true_positive_count', 0)}/"
            f"{clip_counts.get('exact_tab_false_positive_count', 0)}/"
            f"{clip_counts.get('exact_tab_false_negative_count', 0)}; "
            f"pitch-matched/assignment-correct "
            f"{clip_counts.get('pitch_matched_playable_note_count', 0)}/"
            f"{clip_counts.get('assignment_correct_count', 0)}"
        )
    lines.extend(
        [
            "",
            "## Generated guitar-tab augmentation",
            "",
            f"Pooled exact-tab precision/recall/F1: **{100 * tab_score['precision']:.1f}% / {100 * tab_score['recall']:.1f}% / {100 * tab_score['f1']:.1f}%**.",
            f"Pitch-matched string/fret assignment accuracy: **{100 * tab_score['assignment_accuracy']:.1f}%**.",
            "",
            "Tab events are matched one-to-one within a 50 ms onset tolerance and must have identical string and fret. Assignment accuracy instead matches playable events by onset and rounded pitch, then checks their string/fret assignment.",
            "",
            f"Counts: {tab_counts.get('reference_note_count', 0)} reference notes ({tab_counts.get('reference_tab_note_count', 0)} playable, {tab_counts.get('unplayable_or_invalid_reference_note_count', 0)} invalid/unplayable reference), {tab_counts.get('predicted_midi_note_count', 0)} predicted MIDI notes, {tab_counts.get('rendered_predicted_tab_note_count', 0)} rendered predicted tab notes, {tab_counts.get('unplayable_or_unassigned_predicted_note_count', 0)} unplayable/unassigned predictions; exact-tab TP/FP/FN = {tab_counts.get('exact_tab_true_positive_count', 0)}/{tab_counts.get('exact_tab_false_positive_count', 0)}/{tab_counts.get('exact_tab_false_negative_count', 0)}; pitch-matched/assignment-correct = {tab_counts.get('pitch_matched_playable_note_count', 0)}/{tab_counts.get('assignment_correct_count', 0)}.",
            "",
            "Exact-tab precision uses all counted predicted MIDI notes as its denominator, so non-drum MIDI predictions that cannot be assigned to a playable string/fret remain disclosed and penalize precision rather than disappearing. Reference notes with invalid string/fret labels are excluded from playable-tab recall and retained in the reference-note count.",
            "Predicted MIDI counts include every nonzero-duration non-drum MIDI note; drum tracks are excluded to match guitar-tab assignment, while unplayable/unassigned non-drum predictions remain counted and penalize precision.",
            "",
            f"Reference and predicted ASCII-tab artifacts are cached under `{tab_report.get('artifact_root', 'the benchmark cache/tabs')}`. They are a generated local benchmark augmentation; upstream GuitarSet rows are not mutated.",
            "",
            "This is a small acoustic-guitar benchmark and should be described with its metric and subset. GuitarSet reference note pitch and offset labels include automatic estimates. The [GuitarSet dataset](https://doi.org/10.5281/zenodo.1422265) is accessed through its [Hugging Face mirror](https://huggingface.co/datasets/jhartquist/guitarset). The framewise pitch metric follows `mir_eval`'s multipitch matching procedure.",
            "",
            "Reproduce with:",
            "",
            "```bash",
            "uv run --python 3.10 --extra transcription -- python -m scripts.evaluate_mt3_guitarset_framewise",
            "```",
            "",
            "The selected audio, MIDI files, and generated tab artifacts are cached in the system temporary directory and are not stored in the repository.",
            "",
        ]
    )
    markdown.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, choices=(2, 4, 6, 8), default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reuse-midi", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--markdown", type=Path, default=REPORT_MD)
    parser.add_argument("--json", type=Path, default=REPORT_JSON)
    parser.add_argument("--comparisons", type=Path, default=COMPARISON_REPORT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audio_dir = args.cache_dir / "audio"
    midi_dir = args.cache_dir / "midi"
    wav_dir = args.cache_dir / "wav_16k"
    tab_dir = args.cache_dir / "tabs"
    reference_tab_dir = tab_dir / "reference"
    predicted_tab_dir = tab_dir / "predicted"
    for directory in (
        audio_dir,
        midi_dir,
        wav_dir,
        reference_tab_dir,
        predicted_tab_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    print("Fetching the fixed GuitarSet subset...")
    rows, dataset_revision = fetch_subset(args.limit)
    clips = []
    pooled_reference: list[np.ndarray] = []
    pooled_estimate: list[np.ndarray] = []
    tab_evaluations: list[dict[str, Any]] = []
    comparison_tabs: dict[str, tuple[list[TabNote], list[TabNote]]] = {}
    for index, row in enumerate(rows, start=1):
        track_id = row["track_id"]
        midi_path = midi_dir / f"{track_id}.mid"
        action = "reusing cached MIDI" if args.reuse_midi else "transcribing"
        print(f"[{index}/{len(rows)}] {track_id}: {action}")
        if args.reuse_midi:
            if not midi_path.is_file():
                raise SystemExit(f"Cached MIDI is missing: {midi_path}")
        else:
            input_audio = fetch_audio(row, audio_dir)
            model_audio = wav_dir / f"{track_id}.wav"
            extract_audio(input_audio, model_audio, overwrite=True)
            transcribe_audio(
                read_model_audio(model_audio),
                midi_path,
                model_name="mr_mt3",
                device=args.device,
            )

        duration = float(row["duration_s"])
        source_reference_notes = reference_notes(row)
        reference_tabs = reference_tab_notes(row)
        predicted_midi_notes = predicted_notes(midi_path)
        predicted_tabs = midi_to_tab_notes(midi_path)
        comparison_tabs[track_id] = (reference_tabs, predicted_tabs)
        reference_tab_path = reference_tab_dir / f"{track_id}.txt"
        predicted_tab_path = predicted_tab_dir / f"{track_id}.txt"
        tab_artifacts = write_tab_artifacts(
            reference_tabs,
            predicted_tabs,
            reference_path=reference_tab_path,
            predicted_path=predicted_tab_path,
        )

        ref_frames = pitch_frames(source_reference_notes, duration)
        est_frames = pitch_frames(predicted_midi_notes, duration)
        tab_evaluation = evaluate_tab_notes(
            reference_tabs,
            predicted_tabs,
            predicted_midi_count=len(predicted_midi_notes),
            reference_note_count=len(source_reference_notes),
        )
        clips.append(
            {
                "track_id": track_id,
                "player": int(row["player"]),
                "style": row["style"],
                "duration_seconds": duration,
                "score": framewise_f1(ref_frames, est_frames),
                "tab": {
                    "artifacts": tab_artifacts,
                    "score": tab_evaluation["score"],
                    "counts": tab_evaluation["counts"],
                },
            }
        )
        pooled_reference.extend(ref_frames)
        pooled_estimate.extend(est_frames)
        tab_evaluations.append(tab_evaluation)

    pooled_tab_evaluation = aggregate_tab_evaluations(tab_evaluations)
    report = {
        "run_date": datetime.now(timezone.utc).date().isoformat(),
        "dataset": DATASET,
        "dataset_revision": dataset_revision,
        "row_ranges": [list(row_range) for row_range in ROW_RANGES],
        "model": "mr_mt3",
        "metric": "framewise active-pitch F1",
        "frame_rate": FRAME_RATE,
        "pitch_window_semitones": PITCH_WINDOW_SEMITONES,
        "tab_metric": "one-to-one exact string/fret event matching",
        "tab_onset_tolerance_seconds": TAB_ONSET_TOLERANCE_SEC,
        "comparison_report": str(args.comparisons),
        "clip_count": len(clips),
        "player_count": len({clip["player"] for clip in clips}),
        "style_counts": {
            style: sum(clip["style"] == style for clip in clips)
            for style in ("comp", "solo")
        },
        "audio_seconds": sum(clip["duration_seconds"] for clip in clips),
        "score": framewise_f1(pooled_reference, pooled_estimate),
        "tab": {
            "metric": "one-to-one exact string/fret event matching",
            "onset_tolerance_seconds": TAB_ONSET_TOLERANCE_SEC,
            "provenance": "generated local augmentation; upstream GuitarSet rows are not mutated",
            "predicted_midi_scope": "all nonzero-duration non-drum MIDI notes",
            "artifact_root": str(tab_dir),
            "artifact_directories": {
                "reference": str(reference_tab_dir),
                "predicted": str(predicted_tab_dir),
            },
            "score": pooled_tab_evaluation["score"],
            "counts": pooled_tab_evaluation["counts"],
        },
        "clips": clips,
    }
    render_comparison_report(report, comparison_tabs, args.comparisons)
    render_report(report, args.markdown, args.json)
    print(f"Framewise active-pitch F1: {100 * report['score']['f1']:.1f}%")
    print(f"Exact tab F1: {100 * report['tab']['score']['f1']:.1f}%")
    print(f"Wrote {args.markdown}, {args.json}, and {args.comparisons}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
