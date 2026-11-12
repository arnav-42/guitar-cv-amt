# Stairway AV pipeline restart log

## Goal

Generate string/fret tablature from `videos/stairway-intro-short.mp4` that matches the tab printed in the video, using audio for pitches/onsets and video for played positions. Only after the short clip is solved, consider `videos/stairway-intro.webm`.

## Baseline on 2026-11-10

- Existing command in `README.md`: `python -m main.audio.video_to_score videos/stairway-intro-short.mp4 --model yourmt3 --visual --output-dir output/stairway-intro-short --overwrite`.
- Existing output is already present in `output/stairway-intro-short/`. It has 35 MIDI notes in 25 onset groups. The visual path sampled 175 frames, found a usable grid in 64, supported 8 events, and labeled 6 final positions as visually supported.
- The current output **does not match** the tab in the video. The audio baseline and AV output diverge especially after 6.6 seconds; the AV system accepts fret positions 7–14 on strings that appear inconsistent with the performance. The code currently uses string-row alignment without calibration for this output.
- Ground truth is now in `research/av_stairway_ground_truth.json`: 26 attacks and 36 string/fret notes. `python -m scripts.evaluate_stairway_short` reports **23/36** exact positions for the saved AV output, versus **33/36** for the audio-only baseline. The baseline has two wrong positions (first D7 at 3.3 s, high E open at 9.97 s) and one missing E4 onset at 12.03 s. The saved AV output adds many wrong high-fret positions.
- `main/audio/audio_fusion.detect_onsets` detects the missing 12.032 s attack with pitch 64, confidence 0.59, and spectral-flux strength 0.65. Other unmatched acoustic peaks in this clip are weaker or too close to existing MIDI attacks. This supports an audio onset recovery step.
- Inspecting the saved visual overlay at 7 s showed EFA/YOLO grid only on a small nut-side neck fragment, even though the full guitar neck is visible. Current contact extraction was gated on that grid. `visual_fingering.py` was changed to use a tracked calibration directly for row and fret whenever supplied, including if EFA has no grid. Initial calibration still had a wrong equal-temperament longitudinal mapping (visible fret 5 became estimated fret 7), so calibrated alignment is currently rejected at 3/23 supported events. The fretboard agent is adding measured wire anchors.
- `research/stairway-short-calibration.json` contains 12 manually measured fret wire centers from video frame 105 and outer-string corners. `BoardCalibration` uses the measured wires for absolute fret interpolation and tracks them with the board. The full CLI run with this calibration matches the printed TAB.
- The YOLO mask audit found two legitimate neck fragments separated by the hand. `_clean_mask` kept only the largest. The fretboard agent fixed it and ran 10 focused tests successfully; EFA still may be coarse because the string lines are hard to certify in occluded frames. The calibrated contact path is independent of EFA in those frames.
- Important source files: `main/audio/video_to_score.py` (CLI/orchestration), `main/video/hands/visual_fingering.py` (visual extraction, semantic alignment), `main/audio/guitar_tab.py` (assignment), `main/video/fretboard/audio_visual_fusion.py` (calibrated board geometry), `main/video/fretboard/fretboard_rectify.py` and `fret_detect_yolo.py` (board detection).
- A calibration example exists in `pilot-fusion/clip0_calibration.json`; its applicability to this video has not yet been confirmed.

## Checklist

- [x] Extract exact ground truth from the bottom tab in the short video, with time-aligned string/fret events.
- [x] Determine whether audio transcription pitches/onsets match that truth.
- [x] Measure video board/hand/fret geometry and fix broken detection/tracking.
- [x] Make AV assignments reproduce the short-video ground truth; validate programmatically, not just by looking at ASCII spacing.
- [x] Run focused regression tests and a full short-video pipeline pass.
- [x] Document the large-video result and final limitations.

## Current work ownership

- Root agent: pipeline integration, audio analysis, comparison, regression.
- `ground_truth` subagent: bottom-overlay ground-truth extraction, writes `research/av_stairway_ground_truth.md`.
- `fret_audit` subagent: inspect/fix focused fretboard modules and their tests.

## Verified short-clip result

Run from the repository root with the installed Python 3.10 environment:

```bash
.venv/bin/python -m main.audio.video_to_score videos/stairway-intro-short.mp4 \
  --model yourmt3 --visual \
  --visual-calibration research/stairway-short-calibration.json \
  --output-dir output/stairway-intro-short --overwrite
.venv/bin/python -m scripts.evaluate_stairway_short
```

The complete CLI run succeeded. The evaluator reports **36/36 exact string/fret positions, 0 missing pitches, 0 extra pitches**. Diagnostics: 26 events seen, 25 supported, 179 frames inspected, 64 with an EFA grid, 30 final notes with exact visual evidence. Audio onset recovery added one pitch-64 note at 12.032 s. The ground-truth overlay is read only by the evaluation script, never by transcription or fusion.

The open-string ergonomic reward and the recovered onset bring audio-only assignment to **35/36** positions. Calibrated video evidence changes the first note from G-string fret 2 to the played D-string fret 7, reaching 36/36. The earlier uncalibrated visual path changed many correct positions incorrectly; for this video, supply the included calibration.

Focused regression command:

```bash
UV_CACHE_DIR=/private/tmp/uv-fret uv run --offline --python 3.10 \
  --with pytest --extra visual -- pytest -q \
  tests/test_visual_pipeline.py tests/test_visual_fingering.py \
  tests/test_fret_detect_yolo.py tests/test_calibration_tracking.py \
  tests/test_board_geometry.py tests/test_fretboard_rectify.py
```

The focused regression suite, including audio onset recovery, reports **71 passed**. The run used the shared cached pytest environment and emitted no warnings.

## Larger video

`videos/stairway-intro.webm` is 28.021 s, 1920×1080 at 30 fps. Frame 105 differs from the short clip by only mean absolute pixel value 2.16, so the short calibration is applicable. A full calibrated CLI run was saved to `output/stairway-intro/`. Evaluate it against `research/av_stairway_full_ground_truth.json` with `python -m scripts.evaluate_stairway_short --output-dir output/stairway-intro --truth research/av_stairway_full_ground_truth.json`.

The first full-video CLI run completed. `research/av_stairway_full_ground_truth.json` now records 55 played attacks and 75 notes through 28 s. Against that reference, the full-video output scores **60/75 exact positions, 8 missing pitches, and 22 extra pitches**. Its first 15 s score 32/36 because the 28 s YourMT3 transcription is substantially noisier than transcription of the standalone 15 s clip: duplicate onsets around 4–6 s, false notes around 8 s, missing chord bass notes at 13–14 s, and a spurious sustained multi-pitch chord at 16.66 s. Video alignment is accepted (47/63 events, 57 exact visual positions), but it cannot repair absent or duplicate audio pitches.

The audio recovery function has since been extended to insert a strong low fundamental at an existing chord onset. On a copy of the completed full-video MIDI, it adds missing bass pitches 47 at 13.28 s, 45 at 13.728/14.128 s, and 45 at 27.488 s; on the completed short-clip MIDI it adds nothing. A short-clip full CLI rerun after this change still scores 36/36. A windowed-transcription experiment produced similarly noisy results, so the full 28 s clip remains less accurate than the solved short clip. The evaluator, full-video ground truth, and diagnostic output make the remaining audio-model failures reproducible.

## Next steps if resumed

1. Read this file and the two Stairway ground-truth markdown files.
2. For further full-video work, reduce duplicate and hallucinated YourMT3 notes before changing visual fusion; the current evidence shows audio omissions dominate.
3. Run `python -m scripts.evaluate_stairway_short` after any short-video code or calibration change.
