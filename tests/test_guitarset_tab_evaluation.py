"""Deterministic tests for the local GuitarSet tab-evaluation augmentation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import scripts.evaluate_mt3_guitarset_framewise as evaluator
from pipeline.guitar_tab import TabNote
from scripts.evaluate_mt3_guitarset_framewise import (
    ROW_RANGES,
    align_tab_event_groups,
    evaluate_tab_notes,
    match_tab_events,
    note_to_tab_note,
    reference_tab_notes,
    render_report,
    render_tab_comparison,
    render_tab_comparison_systems,
    write_tab_artifacts,
)


def _row_note(onset: float, offset: float, midi: float, string) -> dict:
    return {
        "onset_s": onset,
        "offset_s": offset,
        "midi": midi,
        "string": string,
    }


def _tab(onset: float, pitch: int, string: int, fret: int) -> TabNote:
    return TabNote(
        onset_sec=onset,
        offset_sec=onset + 0.2,
        pitch_midi=pitch,
        string_idx=string,
        fret_idx=fret,
    )


def test_reference_string_to_fret_conversion_drops_invalid_labels_and_frets():
    row = {
        "notes": [
            _row_note(0.0, 0.2, 40.4, 0),
            _row_note(0.3, 0.5, 64.6, "high-E"),
            _row_note(0.6, 0.8, 40, "not-a-string"),
            _row_note(0.9, 1.1, 39, 0),
            _row_note(1.2, 1.4, 90, 5),
        ]
    }

    notes = reference_tab_notes(row)

    assert [(note.string_idx, note.fret_idx, note.pitch_midi) for note in notes] == [
        (0, 0, 40),
        (5, 1, 65),
    ]
    assert note_to_tab_note(_row_note(0.0, 0.1, 40, "E")).string_idx == 0
    assert note_to_tab_note(_row_note(0.0, 0.1, 64, "e")).string_idx == 5
    assert note_to_tab_note(_row_note(0.0, 0.1, 40, 6)) is None


def test_tab_matching_is_one_to_one_and_includes_tolerance_boundary():
    reference = [_tab(1.0, 40, 0, 0), _tab(1.1, 40, 0, 0)]
    predicted = [_tab(1.05, 40, 0, 0), _tab(1.15, 40, 0, 0)]

    matches = match_tab_events(reference, predicted)

    assert len(matches) == 2
    assert [pair[0].onset_sec for pair in matches] == [1.0, 1.1]
    assert len(match_tab_events(reference, predicted[:1])) == 1

def test_comparison_grid_pairs_onset_groups_and_keeps_unmatched_columns():
    reference = [
        _tab(0.0, 64, 5, 0),
        _tab(1.0, 65, 5, 1),
        _tab(2.0, 66, 5, 2),
    ]
    predicted = [
        _tab(0.05, 64, 5, 0),
        _tab(0.2, 67, 5, 3),
        _tab(1.05, 65, 5, 1),
    ]

    aligned = align_tab_event_groups(reference, predicted)

    assert [
        (
            None if reference_group is None else reference_group[0].onset_sec,
            None if predicted_group is None else predicted_group[0].onset_sec,
        )
        for reference_group, predicted_group in aligned
    ] == [(0.0, 0.05), (None, 0.2), (1.0, 1.05), (2.0, None)]
    system = render_tab_comparison(reference, predicted)
    assert "e|0---1-2-|" in system
    assert "e|0-3-1---|" in system


def test_comparison_renderer_shares_multidigit_width_and_has_six_lines_per_side():
    systems = render_tab_comparison_systems(
        [_tab(0.0, 64, 5, 12)],
        [_tab(0.02, 64, 5, 3)],
    )

    lines = systems[0].splitlines()
    assert len(lines) == 7
    assert len({len(line) for line in lines}) == 1
    assert "12-" in lines[1]
    assert "3--" in lines[1]
    assert all(
        lines[index + 1].count(f"{label}|") == 2
        for index, label in enumerate(("e", "B", "G", "D", "A", "E"))
    )


def test_comparison_renderer_wraps_deterministically_at_sixteen_columns():
    reference = [_tab(index * 0.1, 40 + index, 0, index) for index in range(17)]
    predicted = [
        _tab(index * 0.1 + 0.01, 40 + index, 0, index)
        for index in range(17)
    ]

    systems = render_tab_comparison_systems(reference, predicted)

    assert len(systems) == 2
    assert all(len(system.splitlines()) == 7 for system in systems)
    assert all(
        all(
            system.splitlines()[index + 1].count(f"{label}|") == 2
            for index, label in enumerate(("e", "B", "G", "D", "A", "E"))
        )
        for system in systems
    )


def test_exact_tab_metrics_penalize_omitted_unplayable_predictions():
    reference = [_tab(0.0, 40, 0, 0), _tab(1.0, 45, 1, 0)]
    predicted = [_tab(0.04, 40, 0, 0), _tab(1.04, 45, 0, 5)]

    evaluation = evaluate_tab_notes(
        reference,
        predicted,
        predicted_midi_count=3,
    )

    assert evaluation["counts"] == {
        "reference_note_count": 2,
        "reference_tab_note_count": 2,
        "unplayable_or_invalid_reference_note_count": 0,
        "predicted_midi_note_count": 3,
        "rendered_predicted_tab_note_count": 2,
        "unplayable_or_unassigned_predicted_note_count": 1,
        "exact_tab_true_positive_count": 1,
        "exact_tab_false_positive_count": 2,
        "exact_tab_false_negative_count": 1,
        "pitch_matched_playable_note_count": 2,
        "assignment_correct_count": 1,
    }
    assert evaluation["score"]["precision"] == 1 / 3
    assert evaluation["score"]["recall"] == 1 / 2
    assert evaluation["score"]["f1"] == 0.4
    assert evaluation["score"]["assignment_accuracy"] == 1 / 2


def test_tab_artifacts_and_report_fields_are_created_without_network(tmp_path: Path):
    reference = [_tab(0.0, 40, 0, 0)]
    predicted = [_tab(0.02, 40, 0, 0)]
    artifact_root = tmp_path / "tabs"
    artifacts = write_tab_artifacts(
        reference,
        predicted,
        reference_path=artifact_root / "reference" / "clip.txt",
        predicted_path=artifact_root / "predicted" / "clip.txt",
    )
    evaluation = evaluate_tab_notes(reference, predicted)
    report = {
        "run_date": "2026-01-01",
        "dataset": "synthetic",
        "dataset_revision": None,
        "model": "synthetic",
        "metric": "framewise active-pitch F1",
        "frame_rate": 100.0,
        "pitch_window_semitones": 0.5,
        "tab_metric": "one-to-one exact string/fret event matching",
        "tab_onset_tolerance_seconds": 0.05,
        "clip_count": 1,
        "player_count": 1,
        "style_counts": {"comp": 0, "solo": 1},
        "audio_seconds": 1.0,
        "score": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
        "tab": {
            "artifact_root": str(artifact_root),
            "score": evaluation["score"],
            "counts": evaluation["counts"],
        },
        "clips": [
            {
                "track_id": "clip",
                "player": 0,
                "style": "solo",
                "duration_seconds": 1.0,
                "score": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
                "tab": {
                    "artifacts": artifacts,
                    "score": evaluation["score"],
                    "counts": evaluation["counts"],
                },
            }
        ],
    }
    markdown = tmp_path / "report.md"
    report_json = tmp_path / "report.json"
    render_report(report, markdown, report_json)

    assert Path(artifacts["reference"]).is_file()
    assert Path(artifacts["predicted"]).is_file()
    assert json.loads(report_json.read_text(encoding="utf-8"))["tab"]["counts"] == evaluation[
        "counts"
    ]
    markdown_text = markdown.read_text(encoding="utf-8")
    assert "Exact tab P/R/F1" in markdown_text
    assert "generated local benchmark augmentation" in markdown_text
    assert artifacts["reference"] in markdown_text
    assert artifacts["predicted"] in markdown_text


def test_predicted_notes_excludes_drums_before_tab_denominator(tmp_path: Path, monkeypatch):
    class FakeNote:
        def __init__(self, start: float, end: float, pitch: float):
            self.start = start
            self.end = end
            self.pitch = pitch

    class FakeInstrument:
        def __init__(self, is_drum: bool, notes):
            self.is_drum = is_drum
            self.notes = notes

    class FakeMidi:
        instruments = [
            FakeInstrument(True, [FakeNote(0.0, 0.2, 40.0)]),
            FakeInstrument(False, [FakeNote(0.1, 0.3, 41.0)]),
            FakeInstrument(False, [FakeNote(0.4, 0.4, 42.0)]),
        ]

    monkeypatch.setattr(evaluator.pretty_midi, "PrettyMIDI", lambda path: FakeMidi())

    predicted_notes = evaluator.predicted_notes(tmp_path / "synthetic.mid")
    assert predicted_notes == [(0.1, 0.3, 41.0)]
    tab_evaluation = evaluate_tab_notes(
        [_tab(0.1, 41, 0, 1)],
        [_tab(0.1, 41, 0, 1)],
        predicted_midi_count=len(predicted_notes) + 1,
    )
    assert tab_evaluation["counts"]["rendered_predicted_tab_note_count"] == 1
    assert tab_evaluation["counts"]["unplayable_or_unassigned_predicted_note_count"] == 1


def test_fixed_eight_clip_subset_remains_unchanged():
    assert ROW_RANGES == ((0, 2), (16, 2), (60, 2), (76, 2))


def test_main_writes_tab_artifacts_and_report_with_mocked_midi(tmp_path: Path, monkeypatch):
    cache_dir = tmp_path / "cache"
    midi_path = cache_dir / "midi" / "clip.mid"
    midi_path.parent.mkdir(parents=True)
    midi_path.write_bytes(b"synthetic MIDI placeholder")
    predicted = _tab(0.02, 40, 0, 0)
    row = {
        "track_id": "clip",
        "player": 0,
        "style": "solo",
        "duration_s": 1.0,
        "notes": [_row_note(0.0, 0.2, 40, 0)],
    }
    markdown = tmp_path / "result.md"
    report_json = tmp_path / "result.json"
    comparisons = tmp_path / "comparisons.md"
    monkeypatch.setattr(
        evaluator,
        "fetch_subset",
        lambda limit: ([row], "synthetic-revision"),
    )
    monkeypatch.setattr(
        evaluator,
        "predicted_notes",
        lambda path: [(0.02, 0.2, 40.0), (0.4, 0.5, 100.0)],
    )
    monkeypatch.setattr(evaluator, "midi_to_tab_notes", lambda path: [predicted])
    monkeypatch.setattr(evaluator, "pitch_frames", lambda *args: [])
    monkeypatch.setattr(
        evaluator,
        "framewise_f1",
        lambda *args: {"precision": 1.0, "recall": 1.0, "f1": 1.0},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_mt3_guitarset_framewise",
            "--limit",
            "2",
            "--reuse-midi",
            "--cache-dir",
            str(cache_dir),
            "--markdown",
            str(markdown),
            "--json",
            str(report_json),
            "--comparisons",
            str(comparisons),
        ],
    )

    assert evaluator.main() == 0

    report = json.loads(report_json.read_text(encoding="utf-8"))
    clip_report = report["clips"][0]
    assert report["row_ranges"] == [[0, 2], [16, 2], [60, 2], [76, 2]]
    assert report["tab"]["counts"]["predicted_midi_note_count"] == 2
    assert report["tab"]["counts"]["unplayable_or_unassigned_predicted_note_count"] == 1
    assert report["tab"]["artifact_directories"] == {
        "reference": str(cache_dir / "tabs" / "reference"),
        "predicted": str(cache_dir / "tabs" / "predicted"),
    }
    assert Path(clip_report["tab"]["artifacts"]["reference"]).is_file()
    assert Path(clip_report["tab"]["artifacts"]["predicted"]).is_file()
    assert report["comparison_report"] == str(comparisons)
    assert comparisons.is_file()
    assert "generated local benchmark augmentation" in markdown.read_text(
        encoding="utf-8"
    )


def test_main_writes_all_clip_comparison_report_with_mocked_midi(
    tmp_path: Path, monkeypatch
):
    cache_dir = tmp_path / "cache"
    midi_dir = cache_dir / "midi"
    midi_dir.mkdir(parents=True)
    rows = []
    for index in range(8):
        track_id = f"clip-{index}"
        (midi_dir / f"{track_id}.mid").write_bytes(b"synthetic MIDI placeholder")
        rows.append(
            {
                "track_id": track_id,
                "player": index % 2,
                "style": "comp" if index % 2 == 0 else "solo",
                "duration_s": 1.0,
                "notes": [_row_note(0.0, 0.2, 64, "e")],
            }
        )
    predicted = _tab(0.02, 64, 5, 0)
    markdown = tmp_path / "result.md"
    report_json = tmp_path / "result.json"
    comparisons = tmp_path / "comparisons.md"
    monkeypatch.setattr(
        evaluator,
        "fetch_subset",
        lambda limit: (rows[:limit], "synthetic-revision"),
    )
    monkeypatch.setattr(
        evaluator,
        "predicted_notes",
        lambda path: [(0.02, 0.2, 64.0)],
    )
    monkeypatch.setattr(evaluator, "midi_to_tab_notes", lambda path: [predicted])
    monkeypatch.setattr(evaluator, "pitch_frames", lambda *args: [])
    monkeypatch.setattr(
        evaluator,
        "framewise_f1",
        lambda *args: {"precision": 1.0, "recall": 1.0, "f1": 1.0},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_mt3_guitarset_framewise",
            "--limit",
            "8",
            "--reuse-midi",
            "--cache-dir",
            str(cache_dir),
            "--markdown",
            str(markdown),
            "--json",
            str(report_json),
            "--comparisons",
            str(comparisons),
        ],
    )

    assert evaluator.main() == 0

    report = json.loads(report_json.read_text(encoding="utf-8"))
    comparison_text = comparisons.read_text(encoding="utf-8")
    assert report["clip_count"] == 8
    assert report["comparison_report"] == str(comparisons)
    assert comparison_text.count("\n## ") == 8
    assert comparison_text.count("Exact-tab precision / recall / F1") == 8
    assert comparison_text.count("String/fret assignment accuracy") == 8
    blocks = comparison_text.split("```text\n")[1:]
    assert len(blocks) == 8
    for index, block in enumerate(blocks):
        assert f"## clip-{index}" in comparison_text
        body = block.split("\n```", 1)[0].splitlines()
        assert len(body) == 7
        assert body[0].startswith("Ground truth")
        assert all(
            body[index + 1].count(f"{label}|") == 2
            for index, label in enumerate(("e", "B", "G", "D", "A", "E"))
        )
