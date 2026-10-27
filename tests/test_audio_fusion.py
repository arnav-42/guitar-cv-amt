import numpy as np

from pipeline.audio_fusion import (
    AudioOnset,
    VisualCandidate,
    VisualFrame,
    _mono_float,
    detect_onsets,
    fuse_onsets,
    segment_distance,
)


def _synthetic_pluck(frequency=220.0, start_sec=0.2, duration_sec=1.0):
    sample_rate = 22050
    times = np.arange(int(sample_rate * duration_sec)) / sample_rate
    samples = np.zeros_like(times)
    active = times >= start_sec
    samples[active] = (
        0.4 * np.sin(2 * np.pi * frequency * times[active])
        * np.exp(-(times[active] - start_sec) * 2)
    )
    return samples, sample_rate


def test_spectral_flux_detects_pluck_and_estimates_pitch():
    samples, sample_rate = _synthetic_pluck()

    onsets = detect_onsets(samples, sample_rate)

    assert onsets
    assert abs(onsets[0].time_sec - 0.2) < 0.05
    assert onsets[0].pitch_midi == 57
    assert onsets[0].pitch_confidence > 0.5


def test_pitch_and_contact_likelihood_rank_fingering_candidate():
    onset = AudioOnset(0.2, 220.0, 57, 1.0, 1.0)
    frame = VisualFrame(0.21, (
        VisualCandidate(2, 7, 2.0, "index", 3.0, "middle"),
        VisualCandidate(0, 17, 30.0, "ring", 30.0, "pinky"),
    ))

    event, = fuse_onsets([onset], [frame])

    assert (event.string_idx, event.fret_idx) == (2, 7)
    assert (event.press_finger, event.pluck_finger) == ("index", "middle")
    assert event.confidence > 0.99


def test_score_pitch_hint_replaces_uncertain_audio_pitch():
    onset = AudioOnset(0.2, 440.0, 69, 1.0, 1.0)
    frame = VisualFrame(0.2, (
        VisualCandidate(2, 7, 2.0, "index"),
        VisualCandidate(5, 5, 2.0, "ring"),
    ))

    event, = fuse_onsets([onset], [frame], score_pitch_hints={0: 57})

    assert (event.string_idx, event.fret_idx) == (2, 7)
    assert event.pitch_confidence == 1.0


def test_segment_distance_supports_2d_and_3d_segments():
    assert segment_distance([0, 0], [1, 0], [0, 2], [1, 2]) == 2.0
    assert segment_distance([0, 0, 0], [1, 0, 0], [0, 0, 3], [1, 0, 3]) == 3.0


def test_multichannel_integer_audio_is_normalized_before_analysis():
    samples = np.array([[32767, -32768], [0, 16384]], dtype=np.int16)

    mono = _mono_float(samples)

    assert mono.shape == (2,)
    assert np.max(np.abs(mono)) <= 1.0
    assert mono[1] > 0.0