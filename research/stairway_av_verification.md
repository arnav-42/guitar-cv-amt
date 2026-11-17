# Calibrated audio/video transcription verification

Verified locally on 2026-11-16 against `videos/stairway-intro-short.mp4`.
Reproduce using the `scripts.transcribe_av` and `main.evaluation.tab_ground_truth`
commands in the README. Both transcription/visual extras and FFmpeg are required.

## Pipeline and actual evidence

The pipeline composes inventory methods: YourMT3 audio transcription,
spectral-flux and harmonic-salience onset recovery, YOLO/EFA image geometry,
MediaPipe hand landmarks, measured fret-wire calibration tracked with LK/ORB,
and audio-constrained whole-chord dynamic programming with unique strings.
For this clip tracking used LK; ORB is the fallback implementation.
Calibration is manual geometry from the video, not note labels.

The fresh full inference run produced 36 notes in 26 attacks. One pitch-64
attack at 12.032 seconds was recovered acoustically. Of 179 inspected frames,
64 had an EFA grid; tracked calibration supplies the fret coordinates when
occlusion prevents an EFA grid. There were 25 supported events and 30 final
notes with exact visual position evidence. Open-string notes need not have
a visible fingertip contact.

Video changes the opening MIDI-57 note from the audio-only choice G-string
fret 2 to the performed D-string fret 7. All pitches and attack times come
from audio. This is a measurable contribution from video, not simply a
video-processing stage that leaves the result unchanged.

Saved outputs in `output/stairway-verified/`:

- `tablature.txt`: generated string/fret sequence.
- `audio_tablature.txt`: audio-only assignment ablation.
- `transcription.json`: timed notes, grouped events, input SHA-256 hashes,
  method inventory, and per-note visual contribution.
- `visual_diagnostics.json`: raw EFA, contact, and tracking details.
- `visual_overlay.mp4`: presentation overlay with stable tracked calibration,
  every sampled right-hand thumb/fingertip, note attacks/assignments, and the printed
  source tab covered by an opaque footer.
- `evaluation.json` and `audio_evaluation.json`: strict comparisons.
- `audio.wav`, `transcription.mid`, and `sheet.musicxml`: intermediate/export files.

## Reference scope and corrections

The original supplied text included final A-string `0 /8 7` notes outside this
15.018-second recording. They were removed at the user's explicit request.
The reference is never passed to transcription or fingering inference.

The corrected text has 36 notes but currently groups them into 25 attacks:
in numbered measure 3 it places B-string fret 3 in the same column as
D-string fret 4 and high-E fret 2. The recorded performance has three attacks:

| Time from the fresh MIDI | Played strings/frets | MIDI pitches |
| --- | --- | --- |
| 8.320 / 8.350 s | high E2 + D4 | 66 + 54 |
| 8.760 s | B3 | 62 |
| 9.150 s | G2 | 57 |

Independent spectral analysis finds the latter two attacks at 8.752 and
9.152 seconds. The B3 fundamental rises about 94 times at its separate attack.
Evidence is saved in `output/stairway-audio-audit/measure3_audit.json`.

With the current text, strict evaluation correctly fails: 35/36 positions
match within event alignment, with B3 grouped differently. Its reported one
missing and one extra position refer to that different onset grouping, not
to an omitted or invented pitch. All per-string note sequences agree. Against
the older time-annotated `research/av_stairway_ground_truth.json`, the fresh
run scores 36/36 exact positions, no missing pitches, and no extra pitches.
The older reference is a secondary check, not a replacement for the user's file.

Exact evaluation means the ordered simultaneous string/fret events agree.
It does not mean identical whitespace, bar layout, rests, durations, or tempo.
This single calibrated clip is not evidence of perfect accuracy on new videos.

## Regression checks

69 tests passed across the new parser/exporter and existing visual orchestration,
visual fingering, onset recovery, guitar TAB, and calibration tracking tests.
The new evaluator checks missing/extra/repeated notes, wrong strings, simultaneous
grouping, technique marks, and malformed input. The exporter rejects missing
visual evidence and disagreements between MIDI, diagnostics, and rendered TAB.
