from __future__ import annotations

import numpy as np
import pretty_midi

from main.audio.audio_fusion import AudioOnset
from main.audio.video_to_score import recover_missing_onsets


def test_recovery_adds_isolated_attack_and_missing_chord_bass(tmp_path, monkeypatch):
    midi_path = tmp_path / "transcription.mid"
    midi = pretty_midi.PrettyMIDI()
    guitar = pretty_midi.Instrument(program=24)
    guitar.notes.extend([
        pretty_midi.Note(90, 59, 3.30, 3.65),
        pretty_midi.Note(90, 55, 3.31, 3.65),
        pretty_midi.Note(90, 60, 4.10, 4.45),
    ])
    midi.instruments.append(guitar)
    midi.write(str(midi_path))
    onsets = [
        AudioOnset(3.28, None, 47, 0.55, 0.9),  # low fundamental of chord
        AudioOnset(3.30, None, 59, 0.8, 1.0),   # existing note
        AudioOnset(3.31, None, 67, 0.8, 1.0),   # higher harmonic, not bass
        AudioOnset(3.72, None, 64, 0.59, 0.65), # isolated attack
        AudioOnset(4.55, None, 45, 0.52, 0.15), # weak transient
    ]
    monkeypatch.setattr("main.audio.audio_fusion.detect_onsets", lambda *_args, **_kwargs: onsets)

    recovered = recover_missing_onsets(np.zeros(16000, dtype=np.float32), midi_path)
    loaded = pretty_midi.PrettyMIDI(str(midi_path))

    assert [(item["pitch_midi"], round(item["onset_sec"], 2)) for item in recovered] == [
        (47, 3.28), (64, 3.72),
    ]
    assert sorted(note.pitch for note in loaded.instruments[0].notes) == [47, 55, 59, 60, 64]
