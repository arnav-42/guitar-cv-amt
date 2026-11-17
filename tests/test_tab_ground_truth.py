import json
import subprocess
import sys

import pytest

from main.audio.guitar_tab import TabNote, render_ascii_tab
from main.evaluation.tab_ground_truth import (
    TabEvent, TabPosition, evaluate_events, events_from_json,
    events_from_tab_notes, parse_ascii_tab,
)


def event(*positions):
    return TabEvent(tuple(TabPosition(*p) for p in positions))


def test_parser_preserves_aligned_chords_and_slide():
    source = "\n".join((
        "   3", "e|-2--------|", "B|-3--------|", "G|----2-----|",
        "D|-4--------|", "A|------/8-7|", "E|----------|",
    ))
    events = parse_ascii_tab(source)
    assert len(events) == 4
    assert sum(len(e.notes) for e in events) == 6
    # Preserve exactly the written simultaneity, even if another score might
    # put B3 one slot after the D4/e2 chord.
    assert events[0].notes == event((2, 4), (4, 3), (5, 2)).notes
    assert events[-2].notes == (TabPosition(1, 8, "/"),)


def test_multidigit_frets_and_multiple_systems_roundtrip():
    notes = [TabNote(0, .4, 52, 0, 12), TabNote(0, .4, 76, 5, 12),
             TabNote(1, 1.4, 59, 4, 0), TabNote(2, 2.4, 66, 5, 2)]
    parsed = parse_ascii_tab(render_ascii_tab(notes, events_per_system=2))
    assert evaluate_events(events_from_tab_notes(notes), parsed)["exact_match"]


@pytest.mark.parametrize("text", [
    "e|0|\nB|-|",  # Missing strings.
    "e|x|\nB|-|\nG|-|\nD|-|\nA|-|\nE|-|",  # Unsupported muted note.
    "e|0-|\nB|-|\nG|-|\nD|-|\nA|-|\nE|-|",  # Unaligned measure.
    "e|25|\nB|--|\nG|--|\nD|--|\nA|--|\nE|--|",  # Outside supported frets.
    "e|--|\nB|--|\nG|--|\nD|--|\nA|--|\nE|/--|",  # Dangling technique.
])
def test_malformed_reference_is_not_silently_accepted(text):
    with pytest.raises(ValueError):
        parse_ascii_tab(text)


def test_wrong_string_same_pitch_is_an_error():
    report = evaluate_events([event((5, 0))], [event((4, 5))])
    assert not report["exact_match"]
    assert report["event_edit_distance"] == 1
    assert report["matched_notes"] == 0
    assert report["missing_notes"] == report["extra_notes"] == 1


def test_missing_event_does_not_shift_all_subsequent_comparisons():
    expected = [event((0, 0)), event((1, 2)), event((2, 4)), event((3, 6))]
    report = evaluate_events(expected, [expected[0], *expected[2:]])
    assert report["event_edit_distance"] == 1
    assert report["matched_events"] == 3
    assert report["missing_notes"] == 1
    assert report["extra_notes"] == 0
    assert report["errors"][0]["operation"] == "delete"


def test_duplicate_and_extra_notes_prevent_perfect_score():
    expected = [event((0, 0)), event((1, 2))]
    report = evaluate_events(expected, [*expected, expected[-1]])
    assert not report["exact_match"]
    assert report["note_recall"] == 1
    assert report["note_precision"] == pytest.approx(2 / 3)
    assert report["extra_notes"] == 1


def test_chord_grouping_and_slide_are_evaluated():
    chord = event((1, 2), (3, 0))
    assert not evaluate_events([chord], [event((1, 2)), event((3, 0))])["positions_exact"]
    report = evaluate_events([event((1, 8, "/"))], [event((1, 8))])
    assert report["positions_exact"]
    assert not report["techniques_exact"]
    assert not report["exact_match"]
    assert report["event_edit_distance"] == 0


def test_equal_edit_alignments_retain_the_most_shared_notes():
    reference = [event((2, 4), (4, 3), (5, 2))]
    predicted = [event((2, 4), (5, 2)), event((4, 3))]
    report = evaluate_events(reference, predicted)
    assert report["event_edit_distance"] == 2
    assert report["matched_notes"] == 2
    assert report["missing_notes"] == report["extra_notes"] == 1


def test_chord_grouping_uses_first_onset_not_chaining():
    notes = [TabNote(0, 1, 40, 0, 0), TabNote(.04, 1, 45, 1, 0),
             TabNote(.08, 1, 50, 2, 0)]
    assert len(events_from_tab_notes(notes, tolerance=.05)) == 2


def test_duplicate_string_fails_json_validation():
    with pytest.raises(ValueError, match="one string"):
        events_from_json({"events": [{"notes": [
            {"string_idx": 1, "fret_idx": 2}, {"string_idx": 1, "fret_idx": 3},
        ]}]})


def test_empty_tab_is_an_evaluable_empty_prediction():
    empty = parse_ascii_tab(render_ascii_tab([]))
    assert empty == []
    assert evaluate_events([], empty)["exact_match"]
    assert evaluate_events([event((0, 0))], empty)["missing_notes"] == 1


def test_cli_writes_json_and_fails_for_an_incorrect_prediction(tmp_path):
    truth = tmp_path / "truth.txt"
    predicted = tmp_path / "predicted.json"
    report_path = tmp_path / "report.json"
    truth.write_text("e|0|\nB|-|\nG|-|\nD|-|\nA|-|\nE|-|\n")
    predicted.write_text(json.dumps({"events": [{"notes": [{"string_idx": 4, "fret_idx": 5}]}]}))
    result = subprocess.run([sys.executable, "-m", "main.evaluation.tab_ground_truth",
                             "--truth", str(truth), "--prediction", str(predicted),
                             "--output", str(report_path)], capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    report = json.loads(report_path.read_text())
    assert report["missing_notes"] == report["extra_notes"] == 1
