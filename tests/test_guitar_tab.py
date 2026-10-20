from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.guitar_tab import (
    STANDARD_TUNING_MIDI,
    TabNote,
    midi_to_tab_notes,
    render_ascii_tab,
)
from pipeline import video_to_score


def _write_midi(tmp_path: Path, notes: list[tuple[float, float, float]], *, drums=()):
    pretty_midi = pytest.importorskip("pretty_midi")
    midi = pretty_midi.PrettyMIDI()
    guitar = pretty_midi.Instrument(program=24)
    guitar.notes.extend(
        pretty_midi.Note(velocity=100, pitch=int(round(pitch)), start=start, end=end)
        for start, end, pitch in notes
    )
    midi.instruments.append(guitar)
    if drums:
        drum = pretty_midi.Instrument(program=0, is_drum=True)
        drum.notes.extend(
            pretty_midi.Note(velocity=100, pitch=int(round(pitch)), start=start, end=end)
            for start, end, pitch in drums
        )
        midi.instruments.append(drum)
    path = tmp_path / "input.mid"
    midi.write(str(path))
    return path


def test_standard_tuning_open_strings_and_drum_filter(tmp_path: Path):
    path = _write_midi(
        tmp_path,
        [(float(index), float(index) + 0.4, pitch) for index, pitch in enumerate(STANDARD_TUNING_MIDI)],
        drums=[(0.0, 0.5, 36.0)],
    )

    notes = midi_to_tab_notes(path)

    assert [(note.pitch_midi, note.string_idx, note.fret_idx) for note in notes] == [
        (pitch, index, 0) for index, pitch in enumerate(STANDARD_TUNING_MIDI)
    ]


def test_ambiguous_pitches_keep_a_coherent_position(tmp_path: Path):
    path = _write_midi(
        tmp_path,
        [(0.0, 0.4, 64.0), (1.0, 1.4, 65.0), (2.0, 2.4, 64.0)],
    )

    notes = midi_to_tab_notes(path)

    assert [(note.pitch_midi, note.string_idx, note.fret_idx) for note in notes] == [
        (64, 5, 0),
        (65, 5, 1),
        (64, 5, 0),
    ]


def test_simultaneous_chord_uses_unique_strings_and_aligned_columns(tmp_path: Path):
    path = _write_midi(
        tmp_path,
        [(0.0, 0.5, pitch) for pitch in (60.0, 64.0, 67.0)],
    )

    notes = midi_to_tab_notes(path)
    rendered = render_ascii_tab(notes)
    lines = rendered.splitlines()

    assert len(lines) == 6
    assert [line[0] for line in lines] == ["e", "B", "G", "D", "A", "E"]
    assert len({note.string_idx for note in notes}) == len(notes)
    assert len({len(line) for line in lines}) == 1


def test_multidigit_frets_and_system_wrapping_are_deterministic():
    notes = [
        TabNote(0.0, 0.4, 64, 5, 0),
        TabNote(0.0, 0.4, 40, 0, 0),
        TabNote(1.0, 1.4, 52, 0, 12),
        TabNote(2.0, 2.4, 64, 5, 0),
    ]

    rendered = render_ascii_tab(notes, events_per_system=2)
    systems = rendered.split("\n\n")
    assert rendered == render_ascii_tab(notes, events_per_system=2)

    assert len(systems) == 2
    assert all(len(system.splitlines()) == 6 for system in systems)
    assert all(len({len(line) for line in system.splitlines()}) == 1 for system in systems)
    assert "12-" in systems[0].splitlines()[-1]


def test_midi_pitch_is_rounded_and_unplayable_notes_are_omitted(tmp_path: Path):
    path = _write_midi(
        tmp_path,
        [(0.0, 0.4, 40.4), (1.0, 1.4, 90.0)],
    )

    notes = midi_to_tab_notes(path)

    assert len(notes) == 1
    assert notes[0].pitch_midi == 40
    assert notes[0].fret_idx == 0


def test_midi_to_tab_omits_nonpositive_duration_notes(tmp_path: Path, monkeypatch):
    pretty_midi = pytest.importorskip("pretty_midi")

    class FakeNote:
        def __init__(self, start: float, end: float, pitch: float):
            self.start = start
            self.end = end
            self.pitch = pitch

    class FakeInstrument:
        def __init__(self, notes):
            self.is_drum = False
            self.notes = notes

    class FakeMidi:
        instruments = [
            FakeInstrument(
                [
                    FakeNote(0.0, 0.5, 40.0),
                    FakeNote(1.0, 1.0, 45.0),
                    FakeNote(2.0, 1.0, 50.0),
                ]
            )
        ]

    monkeypatch.setattr(pretty_midi, "PrettyMIDI", lambda _path: FakeMidi())

    notes = midi_to_tab_notes(tmp_path / "mock.mid")

    assert [(note.onset_sec, note.offset_sec, note.pitch_midi) for note in notes] == [
        (0.0, 0.5, 40)
    ]
    assert [(note.string_idx, note.fret_idx) for note in notes] == [(0, 0)]
    assert render_ascii_tab(notes) == "\n".join(
        ["e|--|", "B|--|", "G|--|", "D|--|", "A|--|", "E|0-|"]
    )


def test_video_pipeline_includes_tablature_output_without_model(tmp_path: Path, monkeypatch):
    source = tmp_path / "performance.mp4"
    source.write_bytes(b"placeholder")
    output_dir = tmp_path / "out"

    monkeypatch.setattr(video_to_score, "extract_audio", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(video_to_score, "read_model_audio", lambda _path: object())

    def fake_transcribe(_audio, midi_path, **_kwargs):
        midi_path.write_bytes(b"midi")

    def fake_tab(midi_path, tab_path):
        assert midi_path.name == "transcription.mid"
        tab_path.write_text("e|--|\nB|--|\nG|--|\nD|--|\nA|--|\nE|--|\n", encoding="utf-8")

    monkeypatch.setattr(video_to_score, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(video_to_score, "write_tablature", fake_tab)
    monkeypatch.setattr(video_to_score, "write_musicxml", lambda *_args: None)

    video_to_score.run(
        str(source),
        output_dir,
        model_name="mr_mt3",
        device="cpu",
        overwrite=False,
    )

    assert (output_dir / "tablature.txt").is_file()

    with pytest.raises(video_to_score.PipelineError, match="tablature.txt"):
        video_to_score.run(
            str(source),
            output_dir,
            model_name="mr_mt3",
            device="cpu",
            overwrite=False,
        )




def test_video_parser_documents_tablature_output():
    help_text = video_to_score.build_parser().format_help()
    assert "tablature.txt" in help_text


def test_tab_note_is_immutable():
    note = TabNote(0.0, 1.0, 40, 0, 0)
    with pytest.raises(AttributeError):
        note.fret_idx = 1  # type: ignore[misc]
