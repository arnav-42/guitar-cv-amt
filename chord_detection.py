"""
FFT-based chord detection functions extracted from fft_chord_candidates.ipynb
"""
import numpy as np
from itertools import product

SR = 44100
TUNING = [82.41, 110.00, 146.83, 196.00, 246.94, 329.63]  # Standard guitar tuning (E2, A2, D3, G3, B3, E4)

def freq_from_fret(open_freq: float, fret: int) -> float:
    """Calculate frequency from open string frequency and fret number"""
    return open_freq * (2 ** (fret / 12))

def karplus_strong(freq: float, dur: float = 2.0, sr: int = SR) -> np.ndarray:
    """Synthesize audio using Karplus-Strong algorithm"""
    N = int(sr * dur)
    if freq <= 0:
        return np.zeros(N, dtype=np.float32)

    L = max(2, int(sr / freq))
    buf = np.random.uniform(-1, 1, L).astype(np.float32)
    out = np.zeros(N, dtype=np.float32)

    for i in range(N):
        out[i] = buf[i % L]
        buf[i % L] = 0.996 * 0.5 * (buf[i % L] + buf[(i + 1) % L])

    return out

def synth_chord_array(frets, dur: float = 2.0, sr: int = SR) -> np.ndarray:
    """Synthesize a chord from fret positions"""
    layers = []
    for i, f in enumerate(frets):
        if f == "X":
            continue
        freq = freq_from_fret(TUNING[i], int(f))
        layers.append(karplus_strong(freq, dur=dur, sr=sr))

    if not layers:
        return np.zeros(int(sr * dur), dtype=np.float32)

    mix = np.sum(layers, axis=0).astype(np.float32)
    mix /= (np.max(np.abs(mix)) + 1e-9)
    return mix

def enumerate_possible_chords(
    overlaps,
    allow_open: bool = True,
    allow_mute: bool = True,
    allow_press: bool = True
):
    """Generate all possible chord candidates from CV-detected overlaps"""
    assert len(overlaps) == 6, "Provide exactly 6 strings (low E to high E)."

    per_string_options = []
    for val in overlaps:
        opts = []
        if val is None:
            if allow_open: opts.append(0)
            if allow_mute: opts.append("X")
        else:
            if allow_press: opts.append(int(val))
            if allow_open:  opts.append(0)
            if allow_mute:  opts.append("X")
        per_string_options.append(opts)

    seen = set()
    chords = []
    for combo in product(*per_string_options):
        if all(c == "X" for c in combo):  # skip all-muted
            continue
        tup = tuple(combo)
        if tup not in seen:
            seen.add(tup)
            chords.append(list(combo))
    return chords

def _frame_audio(audio: np.ndarray, sr: int = SR, win_s: float = 0.046, hop_s: float = 0.023):
    """Frame audio and compute FFT spectrum"""
    win = int(win_s * sr)
    hop = int(hop_s * sr)
    if win < 32:
        win = 2048
    if hop < 16:
        hop = win // 2

    nfft = 1
    while nfft < win:
        nfft *= 2

    w = np.hanning(win).astype(np.float32)
    frames = []
    for start in range(0, max(1, len(audio) - win), hop):
        seg = audio[start:start + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        X = np.fft.rfft(seg * w, n=nfft)
        frames.append(np.abs(X))

    if not frames:
        X = np.fft.rfft(np.pad(audio, (0, win - len(audio))), n=nfft)
        frames = [np.abs(X)]

    S = np.stack(frames, axis=0)
    freqs = np.fft.rfftfreq(nfft, d=1 / sr)
    spec = np.median(S, axis=0)
    spec = spec / (np.max(spec) + 1e-9)
    return spec, freqs

def _hz_to_idx(freqs: np.ndarray, hz: float) -> int:
    """Convert frequency in Hz to index in frequency array"""
    if hz <= freqs[0]:
        return 0
    if hz >= freqs[-1]:
        return len(freqs) - 1
    return int(np.argmin(np.abs(freqs - hz)))

def _candidate_harmonics_for_string(open_hz: float, fret: int, max_hz: float, max_harm: int = 6):
    """Get harmonic frequencies for a string at a given fret"""
    f0 = freq_from_fret(open_hz, fret)
    hs = []
    for h in range(1, max_harm + 1):
        fh = f0 * h
        if fh > max_hz:
            break
        hs.append(fh)
    return hs

def _collect_expected_harmonics(frets, freqs: np.ndarray, max_harm: int = 6):
    """Collect all expected harmonic frequencies for a chord"""
    expected = []
    for s_idx, f in enumerate(frets):
        if f == "X":
            continue
        if isinstance(f, (int, np.integer)):
            expected.extend(
                _candidate_harmonics_for_string(TUNING[s_idx], int(f), freqs[-1], max_harm=max_harm)
            )
    return expected

def _cents_window(hz: float, cents: float = 35):
    """Calculate frequency window in cents"""
    r = 2 ** (cents / 1200.0)
    return hz / r, hz * r

def score_chord_against_audio(
    frets,
    test_spec: np.ndarray,
    test_freqs: np.ndarray,
    cents_tol: float = 35,
    max_harm: int = 6
) -> float:
    """Score how well a chord matches the audio spectrum"""
    expected = _collect_expected_harmonics(frets, test_freqs, max_harm=max_harm)
    if not expected:
        return 0.0

    cumsum = np.cumsum(test_spec)
    tot = cumsum[-1] + 1e-9
    score = 0.0

    for fh in expected:
        lo, hi = _cents_window(fh, cents=cents_tol)
        i0 = _hz_to_idx(test_freqs, lo)
        i1 = _hz_to_idx(test_freqs, hi)
        if i1 <= i0:
            i1 = min(i0 + 1, len(test_spec) - 1)
        band_energy = cumsum[i1] - (cumsum[i0 - 1] if i0 > 0 else 0.0)
        score += band_energy

    return float(score / tot)

def rank_chords_by_audio(
    possible_chords,
    test_audio: np.ndarray | None = None,
    test_wav: str | None = None,
    sr: int = SR,
    cents_tol: float = 35,
    max_harm: int = 6,
    top_k: int = 10
):
    """Rank candidate chords by how well they match the audio"""
    assert (test_audio is not None) or (test_wav is not None), "Provide test_audio or test_wav."

    if test_wav is not None:
        import wave
        with wave.open(test_wav, 'rb') as wf:
            assert wf.getnchannels() == 1, "Use mono WAV."
            assert wf.getsampwidth() == 2, "Use 16-bit PCM WAV."
            assert wf.getframerate() == sr, f"WAV must be {sr} Hz."
            frames = wf.readframes(wf.getnframes())
            test_audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        test_audio = np.asarray(test_audio, dtype=np.float32)

    test_audio = test_audio - np.mean(test_audio)
    peak = np.max(np.abs(test_audio)) + 1e-9
    test_audio = test_audio / peak

    test_spec, test_freqs = _frame_audio(test_audio, sr=sr)
    scored = []
    for ch in possible_chords:
        s = score_chord_against_audio(ch, test_spec, test_freqs, cents_tol=cents_tol, max_harm=max_harm)
        scored.append((ch, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

