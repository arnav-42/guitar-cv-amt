from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline import video_to_score
from pipeline.guitar_tab import ScoreNoteId
from pipeline.visual_fingering import VisualAlignment, VisualInferenceResult


def _write_single_note_midi(path: Path) -> None:
    pretty_midi = pytest.importorskip("pretty_midi")
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=24)
    instrument.notes.append(pretty_midi.Note(100, 64, 0.0, 0.5))
    midi.instruments.append(instrument)
    midi.write(str(path))


def _write_duplicate_chord_midi(path: Path) -> None:
    pretty_midi = pytest.importorskip("pretty_midi")
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=24)
    instrument.notes.extend(
        [
            pretty_midi.Note(100, 64, 0.0, 0.5),
            pretty_midi.Note(100, 64, 0.0, 0.5),
        ]
    )
    midi.instruments.append(instrument)
    midi.write(str(path))

def _patch_audio_pipeline(monkeypatch: pytest.MonkeyPatch):
    def fake_extract(_media: Path, wav_path: Path, *, overwrite: bool):
        wav_path.write_bytes(b"audio")

    def fake_audio(_wav: Path):
        return object()

    def fake_transcribe(_audio, midi_path: Path, **_kwargs):
        _write_single_note_midi(midi_path)

    monkeypatch.setattr(video_to_score, "extract_audio", fake_extract)
    monkeypatch.setattr(video_to_score, "read_model_audio", fake_audio)
    monkeypatch.setattr(video_to_score, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(video_to_score, "write_musicxml", lambda *_args: None)


def test_visual_orchestration_preserves_audio_baseline_and_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "performance.mp4"
    source.write_bytes(b"video")
    _patch_audio_pipeline(monkeypatch)

    def accepted_analyzer(_video, notes, **_kwargs):
        return VisualInferenceResult(
            True,
            "accepted test evidence",
            VisualAlignment((0, 1, 2, 3, 4, 5), 1.0, 0.1, 1, 1.0, 0.9),
            {notes[0].identity: {(4, 5): 1.0}},
            events_seen=1,
            events_supported=1,
        )

    output_dir = tmp_path / "accepted"
    video_to_score.run(
        str(source),
        output_dir,
        model_name="mr_mt3",
        device="cpu",
        overwrite=False,
        visual=True,
        visual_analyzer=accepted_analyzer,
    )
    baseline = (output_dir / "audio_tablature.txt").read_bytes()
    final = (output_dir / "tablature.txt").read_bytes()
    assert baseline != final
    assert b"5" in final.splitlines()[1]
    diagnostics = json.loads((output_dir / "visual_diagnostics.json").read_text())
    assert diagnostics["mode"] == "string_row"
    assert diagnostics["alignment"]["mode"] == "string_row"
    assert diagnostics["alignment"]["string_order"] == [0, 1, 2, 3, 4, 5]
    assert "fret_direction" not in diagnostics["alignment"]
    assert "fret_offset" not in diagnostics["alignment"]
    assert diagnostics["per_note"][0]["source"] == "visual"
    assert diagnostics["per_note"][0]["chosen"]["fret_idx"] == 5

    with pytest.raises(video_to_score.PipelineError, match="tablature.txt"):
        video_to_score.run(
            str(source),
            output_dir,
            model_name="mr_mt3",
            device="cpu",
            overwrite=False,
            visual=True,
            visual_analyzer=accepted_analyzer,
        )



def test_accepted_weak_visual_evidence_falls_back_byte_for_byte_and_require_visual_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "performance.mp4"
    source.write_bytes(b"video")
    _patch_audio_pipeline(monkeypatch)

    def weak_analyzer(_video, notes, **_kwargs):
        return VisualInferenceResult(
            True,
            "accepted weak evidence",
            VisualAlignment((0, 1, 2, 3, 4, 5), 1.0, 0.1, 1, 1.0, 0.9),
            {notes[0].identity: {(0, 24): 0.01}},
            events_seen=1,
            events_supported=1,
        )

    output_dir = tmp_path / "weak-fallback"
    video_to_score.run(
        str(source),
        output_dir,
        model_name="mr_mt3",
        device="cpu",
        overwrite=False,
        visual=True,
        visual_analyzer=weak_analyzer,
    )
    assert (output_dir / "tablature.txt").read_bytes() == (
        output_dir / "audio_tablature.txt"
    ).read_bytes()
    diagnostics = json.loads((output_dir / "visual_diagnostics.json").read_text())
    assert diagnostics["accepted"] is False
    assert diagnostics["status"] == "fallback"
    assert diagnostics["reason"] == (
        "accepted visual evidence selected zero evidence-supported final positions"
    )

    required_dir = tmp_path / "weak-required"
    with pytest.raises(video_to_score.PipelineError, match="require-visual"):
        video_to_score.run(
            str(source),
            required_dir,
            model_name="mr_mt3",
            device="cpu",
            overwrite=False,
            require_visual=True,
            visual_analyzer=weak_analyzer,
        )
    assert (required_dir / "tablature.txt").read_bytes() == (
        required_dir / "audio_tablature.txt"
    ).read_bytes()


def test_visual_failure_falls_back_byte_for_byte_and_require_visual_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "performance.mp4"
    source.write_bytes(b"video")
    _patch_audio_pipeline(monkeypatch)

    def failed_analyzer(_video, _notes, **_kwargs):
        return VisualInferenceResult(False, "no usable grid", None, {}, events_seen=1)

    output_dir = tmp_path / "fallback"
    video_to_score.run(
        str(source),
        output_dir,
        model_name="mr_mt3",
        device="cpu",
        overwrite=False,
        visual=True,
        visual_analyzer=failed_analyzer,
    )
    assert (output_dir / "audio_tablature.txt").read_bytes() == (output_dir / "tablature.txt").read_bytes()
    diagnostics = json.loads((output_dir / "visual_diagnostics.json").read_text())
    assert diagnostics["status"] == "fallback"
    assert diagnostics["reason"] == "no usable grid"

    required_dir = tmp_path / "required"
    with pytest.raises(video_to_score.PipelineError, match="require-visual"):
        video_to_score.run(
            str(source),
            required_dir,
            model_name="mr_mt3",
            device="cpu",
            overwrite=False,
            require_visual=True,
            visual_analyzer=failed_analyzer,
        )

def test_visual_diagnostics_use_final_chord_assignment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "performance.mp4"
    source.write_bytes(b"video")
    _patch_audio_pipeline(monkeypatch)

    def fake_transcribe(_audio, midi_path: Path, **_kwargs):
        _write_duplicate_chord_midi(midi_path)

    monkeypatch.setattr(video_to_score, "transcribe_audio", fake_transcribe)

    def accepted_analyzer(_video, notes, **_kwargs):
        evidence = {
            notes[0].identity: {(4, 5): 1.0},
            notes[1].identity: {(4, 5): 1.0},
        }
        return VisualInferenceResult(
            True,
            "chord evidence",
            VisualAlignment((0, 1, 2, 3, 4, 5), 1.0, 0.1, 1, 1.0, 0.9),
            evidence,
            events_seen=1,
            events_supported=1,
        )

    output_dir = tmp_path / "chord"
    video_to_score.run(
        str(source),
        output_dir,
        model_name="mr_mt3",
        device="cpu",
        overwrite=False,
        visual=True,
        visual_analyzer=accepted_analyzer,
    )

    diagnostics = json.loads((output_dir / "visual_diagnostics.json").read_text())
    per_note = {
        item["source_idx"]: item for item in diagnostics["per_note"]
    }
    assert sorted(item["source"] for item in per_note.values()) == ["audio", "visual"]
    visual_item = next(item for item in per_note.values() if item["source"] == "visual")
    audio_item = next(item for item in per_note.values() if item["source"] == "audio")
    assert visual_item["chosen"] == {
        "string_idx": 4,
        "fret_idx": 5,
        "likelihood": 1.0,
    }
    assert audio_item["chosen"] != visual_item["chosen"]


def test_unexpected_visual_analyzer_error_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "performance.mp4"
    source.write_bytes(b"video")
    _patch_audio_pipeline(monkeypatch)

    def broken_analyzer(_video, _notes, **_kwargs):
        raise AssertionError("programmer error")

    output_dir = tmp_path / "unexpected"
    with pytest.raises(AssertionError, match="programmer error"):
        video_to_score.run(
            str(source),
            output_dir,
            model_name="mr_mt3",
            device="cpu",
            overwrite=False,
            visual=True,
            visual_analyzer=broken_analyzer,
        )
    assert not (output_dir / "visual_diagnostics.json").exists()


def test_visual_youtube_rejection_precedes_outputs_download_and_transcription(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = []
    monkeypatch.setattr(video_to_score, "download_youtube_source", lambda *_args: calls.append("download"))
    monkeypatch.setattr(video_to_score, "extract_audio", lambda *_args, **_kwargs: calls.append("extract"))
    monkeypatch.setattr(video_to_score, "transcribe_audio", lambda *_args, **_kwargs: calls.append("transcribe"))
    output_dir = tmp_path / "youtube"

    with pytest.raises(video_to_score.PipelineError, match="local video"):
        video_to_score.run(
            "https://www.youtube.com/watch?v=abc123",
            output_dir,
            model_name="mr_mt3",
            device="cpu",
            overwrite=False,
            visual=True,
        )

    assert calls == []
    assert not output_dir.exists()

def test_visual_analyzer_keyword_contract_does_not_retry_on_type_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "performance.mp4"
    source.write_bytes(b"video")
    _patch_audio_pipeline(monkeypatch)

    def positional_only(_video, _notes):
        return VisualInferenceResult(False, "unused", None, {})

    output_dir = tmp_path / "type-error"
    with pytest.raises(TypeError):
        video_to_score.run(
            str(source),
            output_dir,
            model_name="mr_mt3",
            device="cpu",
            overwrite=False,
            visual=True,
            visual_analyzer=positional_only,
        )
