"""Audio-led note events with probabilistic visual fingering evidence.

The visual side is an input contract: callers provide distances from tracked
finger segments to candidate string/fret segments. Distances may come from
image coordinates today or calibrated 3D motion capture later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks, stft


OPEN_STRING_MIDI = (40, 45, 50, 55, 59, 64)


@dataclass(frozen=True)
class AudioOnset:
    time_sec: float
    pitch_hz: float | None
    pitch_midi: int | None
    pitch_confidence: float
    strength: float


@dataclass(frozen=True)
class VisualCandidate:
    string_idx: int
    fret_idx: int
    press_distance: float | None
    press_finger: str | None
    pluck_distance: float | None = None
    pluck_finger: str | None = None
    confidence: float = 1.0


@dataclass(frozen=True)
class VisualFrame:
    time_sec: float
    candidates: tuple[VisualCandidate, ...]


@dataclass(frozen=True)
class FusedNoteEvent:
    onset_sec: float
    pitch_midi: int | None
    string_idx: int | None
    fret_idx: int | None
    press_finger: str | None
    pluck_finger: str | None
    confidence: float
    pitch_confidence: float


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read PCM or floating-point WAV audio as normalized mono float samples."""
    sample_rate, samples = wavfile.read(path)
    return _mono_float(samples), int(sample_rate)


def _mono_float(samples: np.ndarray) -> np.ndarray:
    source = np.asarray(samples)
    source_dtype = source.dtype
    if source.ndim == 2:
        source = source.astype(np.float64).mean(axis=1)
    elif source.ndim != 1:
        raise ValueError("audio must be a mono or multichannel sample array")

    if np.issubdtype(source_dtype, np.integer):
        info = np.iinfo(source_dtype)
        if np.issubdtype(source_dtype, np.unsignedinteger):
            midpoint = (info.max + 1) / 2
            source = (source - midpoint) / midpoint
        else:
            source = source / max(abs(info.min), info.max)
    else:
        source = source.astype(np.float64)
    if not np.isfinite(source).all():
        raise ValueError("audio samples must be finite")
    return np.asarray(source, dtype=np.float64)


def detect_onsets(
    samples: np.ndarray,
    sample_rate: int,
    *,
    frame_length: int = 2048,
    hop_length: int = 256,
    min_interval_sec: float = 0.08,
    threshold_mad: float = 2.5,
) -> list[AudioOnset]:
    """Detect plucks using positive spectral flux, then estimate each pitch.

    This detects acoustic transients rather than a sustained visual state.
    """
    mono = _mono_float(samples)
    if sample_rate <= 0 or frame_length < 32 or hop_length < 1:
        raise ValueError("sample rate, frame length, and hop length must be positive")
    if min_interval_sec < 0 or threshold_mad < 0:
        raise ValueError("onset interval and threshold must be nonnegative")
    if len(mono) < 32:
        return []

    nperseg = min(frame_length, len(mono))
    hop = min(hop_length, nperseg)
    _, times, spectrum = stft(
        mono,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg - hop,
        boundary="zeros",
        padded=True,
    )
    magnitude = np.abs(spectrum)
    positive_flux = np.maximum(np.diff(magnitude, axis=1), 0.0).sum(axis=0)
    prior_energy = magnitude[:, :-1].sum(axis=0)
    flux = positive_flux / np.maximum(prior_energy, 1e-10)
    flux_times = times[1:]
    if len(flux) < 3 or not np.isfinite(flux).all():
        return []

    median = float(np.median(flux))
    mad = float(np.median(np.abs(flux - median)))
    threshold = median + threshold_mad * max(1.4826 * mad, 1e-4)
    distance = max(1, int(round(min_interval_sec * sample_rate / hop)))
    peak_indices, properties = find_peaks(
        flux, height=threshold, distance=distance, prominence=max(1e-4, mad)
    )

    onsets = []
    for index, strength in zip(peak_indices, properties["peak_heights"]):
        onset_sec = float(flux_times[index])
        pitch_hz, pitch_midi, pitch_confidence = estimate_pitch(
            mono, sample_rate, onset_sec
        )
        onsets.append(AudioOnset(
            time_sec=onset_sec,
            pitch_hz=pitch_hz,
            pitch_midi=pitch_midi,
            pitch_confidence=pitch_confidence,
            strength=float(strength),
        ))
    return onsets


def estimate_pitch(
    samples: np.ndarray,
    sample_rate: int,
    onset_sec: float,
    *,
    min_midi: int = 36,
    max_midi: int = 84,
    window_sec: float = 0.16,
) -> tuple[float | None, int | None, float]:
    """Estimate a monophonic guitar pitch by harmonic salience near an onset."""
    mono = _mono_float(samples)
    if sample_rate <= 0 or min_midi > max_midi or window_sec <= 0:
        raise ValueError("invalid pitch-estimation range or audio parameters")
    start = max(0, int(round((onset_sec + 0.025) * sample_rate)))
    end = min(len(mono), start + int(round(window_sec * sample_rate)))
    if end - start < 64:
        start = max(0, int(round(onset_sec * sample_rate)))
        end = min(len(mono), start + int(round(window_sec * sample_rate)))
    segment = mono[start:end]
    if len(segment) < 64 or float(np.max(np.abs(segment), initial=0.0)) < 1e-8:
        return None, None, 0.0

    segment = segment - np.mean(segment)
    n_fft = max(8192, 1 << (len(segment) - 1).bit_length())
    magnitude = np.abs(np.fft.rfft(segment * np.hanning(len(segment)), n=n_fft))
    bin_hz = sample_rate / n_fft
    midi_values = np.arange(min_midi, max_midi + 1)
    frequencies = 440.0 * np.power(2.0, (midi_values - 69) / 12.0)
    scores = np.zeros(len(midi_values), dtype=np.float64)
    nyquist = sample_rate / 2.0
    for harmonic in range(1, 9):
        target = frequencies * harmonic
        valid = target < nyquist
        bins = np.rint(target[valid] / bin_hz).astype(int)
        bins = np.clip(bins, 1, len(magnitude) - 2)
        local = np.maximum.reduce((magnitude[bins - 1], magnitude[bins], magnitude[bins + 1]))
        scores[valid] += local / np.sqrt(harmonic)

    order = np.argsort(scores)
    best_index = int(order[-1])
    best_score = float(scores[best_index])
    runner_up = float(scores[order[-2]]) if len(order) > 1 else 0.0
    confidence = best_score / max(best_score + runner_up, 1e-12)
    return float(frequencies[best_index]), int(midi_values[best_index]), float(confidence)


def segment_distance(
    first_start: Sequence[float],
    first_end: Sequence[float],
    second_start: Sequence[float],
    second_end: Sequence[float],
) -> float:
    """Return the shortest Euclidean distance between two 2D/3D segments."""
    p0, p1, q0, q1 = (np.asarray(point, dtype=np.float64) for point in (
        first_start, first_end, second_start, second_end
    ))
    if not (p0.ndim == p1.ndim == q0.ndim == q1.ndim == 1):
        raise ValueError("segment endpoints must be coordinate vectors")
    if not (p0.shape == p1.shape == q0.shape == q1.shape) or p0.size not in (2, 3):
        raise ValueError("segments must use matching 2D or 3D coordinates")
    if not all(np.isfinite(point).all() for point in (p0, p1, q0, q1)):
        raise ValueError("segment coordinates must be finite")

    u, v, w = p1 - p0, q1 - q0, p0 - q0
    a, b, c = float(u @ u), float(u @ v), float(v @ v)
    d, e = float(u @ w), float(v @ w)
    denominator = a * c - b * b
    if a <= 1e-12 and c <= 1e-12:
        return float(np.linalg.norm(p0 - q0))
    if a <= 1e-12:
        s, t = 0.0, float(np.clip(e / c, 0.0, 1.0))
    elif c <= 1e-12:
        s, t = float(np.clip(-d / a, 0.0, 1.0)), 0.0
    else:
        s = float(np.clip((b * e - c * d) / denominator, 0.0, 1.0)) if denominator > 1e-12 else 0.0
        t = float(np.clip((b * s + e) / c, 0.0, 1.0))
        s = float(np.clip((b * t - d) / a, 0.0, 1.0))
    return float(np.linalg.norm((p0 + s * u) - (q0 + t * v)))


def fuse_onsets(
    onsets: Sequence[AudioOnset],
    frames: Sequence[VisualFrame],
    *,
    match_tolerance_sec: float = 0.06,
    score_pitch_hints: Mapping[int, int] | None = None,
    contact_distance: float = 12.0,
    distance_scale: float = 4.0,
) -> list[FusedNoteEvent]:
    """Rank visual candidates at each audio onset using pitch and geometry.

    ``score_pitch_hints`` maps onset index to MIDI pitch, allowing an aligned
    score to replace uncertain audio pitch as in the reference methodology.
    """
    if match_tolerance_sec < 0 or contact_distance < 0 or distance_scale <= 0:
        raise ValueError("invalid matching tolerance or geometric likelihood scale")
    ordered_frames = sorted(frames, key=lambda frame: frame.time_sec)
    output = []
    for onset_index, onset in enumerate(onsets):
        effective_pitch = onset.pitch_midi
        pitch_confidence = onset.pitch_confidence
        if score_pitch_hints and onset_index in score_pitch_hints:
            effective_pitch = int(score_pitch_hints[onset_index])
            pitch_confidence = 1.0

        nearby = [frame for frame in ordered_frames
                  if abs(frame.time_sec - onset.time_sec) <= match_tolerance_sec]
        frame = min(nearby, key=lambda item: abs(item.time_sec - onset.time_sec)) if nearby else None
        ranked = []
        if frame is not None:
            for candidate in frame.candidates:
                probability = _candidate_probability(
                    candidate,
                    effective_pitch,
                    pitch_confidence,
                    contact_distance,
                    distance_scale,
                )
                ranked.append((probability, candidate))

        if not ranked:
            output.append(FusedNoteEvent(
                onset_sec=onset.time_sec,
                pitch_midi=effective_pitch,
                string_idx=None,
                fret_idx=None,
                press_finger=None,
                pluck_finger=None,
                confidence=0.0,
                pitch_confidence=pitch_confidence,
            ))
            continue

        total_probability = sum(probability for probability, _ in ranked)
        probability, candidate = max(
            ranked,
            key=lambda item: (item[0], -item[1].string_idx, -item[1].fret_idx),
        )
        output.append(FusedNoteEvent(
            onset_sec=onset.time_sec,
            pitch_midi=effective_pitch,
            string_idx=candidate.string_idx,
            fret_idx=candidate.fret_idx,
            press_finger=candidate.press_finger,
            pluck_finger=candidate.pluck_finger,
            confidence=(probability / total_probability if total_probability else 0.0),
            pitch_confidence=pitch_confidence,
        ))
    return output


def _candidate_probability(
    candidate: VisualCandidate,
    pitch_midi: int | None,
    pitch_confidence: float,
    contact_distance: float,
    distance_scale: float,
) -> float:
    if not 0 <= candidate.string_idx < len(OPEN_STRING_MIDI):
        return 0.0
    if not 0 <= candidate.fret_idx <= 24 or not 0.0 <= candidate.confidence <= 1.0:
        return 0.0
    expected_pitch = OPEN_STRING_MIDI[candidate.string_idx] + candidate.fret_idx
    if pitch_midi is None:
        pitch_probability = 0.5
    else:
        cents = 100.0 * abs(expected_pitch - pitch_midi)
        pitch_probability = float(np.exp(-0.5 * (cents / 70.0) ** 2))
        pitch_probability = (1.0 - pitch_confidence) * 0.5 + pitch_confidence * pitch_probability

    press_probability = _contact_probability(
        candidate.press_distance, contact_distance, distance_scale
    )
    pluck_probability = _contact_probability(
        candidate.pluck_distance, contact_distance, distance_scale
    )
    return float(np.clip(
        pitch_probability * press_probability * pluck_probability * candidate.confidence,
        0.0,
        1.0,
    ))


def _contact_probability(distance: float | None, midpoint: float, scale: float) -> float:
    if distance is None:
        return 1.0
    if not np.isfinite(distance) or distance < 0:
        return 0.0
    z = float(np.clip((distance - midpoint) / scale, -60.0, 60.0))
    return 1.0 / (1.0 + np.exp(z))