"""The audiovisual manifest must describe the actual, verified final TAB."""

import hashlib
import json

import pytest

from main.audio.guitar_tab import TabNote, read_midi_notes, render_ascii_tab
from main.audio.video_to_score import PipelineError
from scripts.transcribe_av import export_transcription


@pytest.fixture
def saved_run(tmp_path):
    pretty_midi = pytest.importorskip("pretty_midi")
    source = tmp_path / "performance.mp4"
    calibration = tmp_path / "calibration.json"
    source.write_bytes(b"synthetic video for export test")
    calibration.write_text("{}")
    midi = pretty_midi.PrettyMIDI()
    guitar = pretty_midi.Instrument(program=24)
    guitar.notes = [
        pretty_midi.Note(100, 64, 0.0, 0.4),
        pretty_midi.Note(100, 45, 1.0, 1.4),
    ]
    midi.instruments.append(guitar)
    midi.write(str(tmp_path / "transcription.mid"))
    score = read_midi_notes(tmp_path / "transcription.mid")
    # Video resolves E4 to B5, while the later open A remains audio-derived.
    assignments = [(4, 5), (1, 0)]
    notes = [TabNote(n.onset_sec, n.offset_sec, n.pitch_midi, s, f, n.identity)
             for n, (s, f) in zip(score, assignments)]
    (tmp_path / "tablature.txt").write_text(render_ascii_tab(notes) + "\n")
    diagnostics = {
        "accepted": True,
        "counts": {"notes_with_visual_evidence": 1},
        "per_note": [
            {"source_idx": n.source_idx, "source": "visual" if f else "audio",
             "chosen": {"string_idx": s, "fret_idx": f}}
            for n, (s, f) in zip(score, assignments)
        ],
    }
    (tmp_path / "visual_diagnostics.json").write_text(json.dumps(diagnostics))
    return source, calibration, tmp_path, diagnostics


def _export(run):
    source, calibration, output, diagnostics = run
    (output / "visual_diagnostics.json").write_text(json.dumps(diagnostics))
    return export_transcription(source, calibration, output, "yourmt3")


def test_export_preserves_supported_assignment_and_audio_fallback_per_note(saved_run):
    result = _export(saved_run)
    assert [(n["pitch_midi"], n["string_idx"], n["fret_idx"])
            for n in result["notes"]] == [(64, 4, 5), (45, 1, 0)]
    assert [e["visual_supported"] for e in result["evidence"]] == [True, False]
    assert result["notes_changed_by_video"] == 1
    assert result["events"][0]["notes"] == [{"string_idx": 4, "fret_idx": 5}]
    assert result["source"]["sha256"] == hashlib.sha256(saved_run[0].read_bytes()).hexdigest()
    assert json.loads((saved_run[2] / "transcription.json").read_text()) == result


@pytest.mark.parametrize("accepted,source,error", [
    (False, "visual", "No accepted visual evidence"),
    (True, "audio", "Zero final positions have visual support"),
])
def test_export_rejects_audio_only_run(saved_run, accepted, source, error):
    saved_run[3]["accepted"] = accepted
    saved_run[3]["per_note"][0]["source"] = source
    with pytest.raises(PipelineError, match=error):
        _export(saved_run)
    assert not (saved_run[2] / "transcription.json").exists()


def test_export_rejects_position_that_contradicts_midi_pitch(saved_run):
    saved_run[3]["per_note"][0]["chosen"]["fret_idx"] = 6
    with pytest.raises(PipelineError, match="contradicts the audio pitch"):
        _export(saved_run)


def test_export_rejects_missing_assignment(saved_run):
    saved_run[3]["per_note"].pop()
    with pytest.raises(PipelineError, match="Missing final assignment"):
        _export(saved_run)


def test_export_rejects_valid_pitch_position_that_differs_from_saved_tab(saved_run):
    # E4 on the open first string is pitch-correct but contradicts the saved B5.
    saved_run[3]["per_note"][0]["chosen"] = {"string_idx": 5, "fret_idx": 0}
    with pytest.raises(PipelineError, match="Saved TAB and final visual assignments disagree"):
        _export(saved_run)
