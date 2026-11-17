# Repository Method Inventory

This repository contains several parallel implementations for finding a guitar
fretboard, estimating frets and strings, tracking hands, and combining visual
evidence with audio. This document groups the meaningful alternatives and marks
their approximate status.

## Status labels

- **Active/default**: used by the documented current workflow.
- **Active alternative**: implemented and usable, but not the default path.
- **Benchmarked baseline**: retained for comparison in the fretboard evaluator.
- **Exploratory/legacy**: notebooks, demos, or older implementations.
- **Deprecated**: under `main/deprecated/`; retained for reference only.

## 1. Fretboard segmentation and board proposals

### Model families

There are two relevant fretboard segmentation model families:

1. **YOLOv8 segmentation** — the documented single-image and visual-fingering
   default. It produces one or more native-resolution fretboard masks.
2. **Mask R-CNN with a ResNet-50 FPN backbone** — an alternative proposal model
   used by the evaluator, the atlas workflow, and older demos.

The main model artifacts are:

- `main/models/fretboard/yolo_weights_best.pt` — YOLOv8 segmentation checkpoint.
- `main/models/fretboard/model_weights.pt` — original Mask R-CNN checkpoint.
- `main/models/fretboard/model_weights_new.pt` — newer Mask R-CNN checkpoint.
- `main/deprecated/fretboard/string_detection/model_weights.pt` — separate
  deprecated string-detection experiment checkpoint.

Relevant code:

- `main/training/fretboard/train_yolo.py`
- `main/training/fretboard/train_mask.py`
- `main/video/fretboard/fret_detect_yolo.py`
- `main/video/fretboard/rectify_atlas.py`
- `main/evaluation/fretboard/evaluate_all.py`

The atlas workflow can also accept a user-supplied binary mask and can score
multiple YOLO and Mask R-CNN proposals instead of blindly selecting the largest
component.

### Other guitar localization

`main/video/other/fret_tracking_demo.py` uses a general YOLOv8 OpenImages model
to detect a **guitar bounding box**, then derives a heuristic neck ROI. This is
not the same as fretboard segmentation and is mainly a webcam/strum-pick demo.

## 2. Fret-line and fret-position estimation

The historical evaluator contains six explicitly benchmarked methods. Each
takes an image and a fretboard mask and returns estimated fret lines.

| Method | Approach | Location | Status |
|---|---|---|---|
| A: Arnav Canonicalization | Minimum-area rectangle, perspective warp, Rule of 18 | `evaluate_all.py::method_a` | Benchmarked baseline |
| B: PCA + Warp | PCA/tapered mask geometry, perspective warp, Rule of 18 | `fret_detect_yolo.py::detect_frets` and `evaluate_all.py::method_b` | Active/default historical path |
| C: PCA without warp | PCA longitudinal axis and interpolated board width, no rectification | `evaluate_all.py::method_c` | Benchmarked baseline |
| D: Paleari & Huet | Hough orientation, rotation, Sobel projection, skew search, peak finding | `evaluate_all.py::method_d` | Benchmarked baseline |
| E: Paleari refined | Mask-aware version of D with a minimum-area-rectangle fallback | `evaluate_all.py::method_e` | Benchmarked baseline |
| F: Hybrid | PCA warp followed by Sobel peak detection and spacing validation | `evaluate_all.py::method_f` | Benchmarked baseline |

The active single-image detector uses Method B. Its fret positions are inferred
from the board mask and the Rule of 18; it does not independently observe every
physical fret wire.

### Evidence-Constrained Fretboard Atlas

The newer atlas path is a separate approach rather than another Rule-of-18
variant:

1. Use YOLO and/or Mask R-CNN to propose board masks.
2. Estimate robust board corners from the mask.
3. Detect actual fret and string line evidence with LSD and ridge/profile
   methods.
4. Refine fret centers and fit a relative equal-temperament fret lattice.
5. Intersect fret and string lines into a checked mesh.
6. Warp through a piecewise-bilinear grid rather than a single rectangular
   crop.

Relevant files:

- `main/video/fretboard/rectify_atlas.py`
- `main/video/fretboard/fretboard_rectify.py`
- `main/video/fretboard/board_geometry.py`
- `main/video/fretboard/grid_warp.py`
- `research/fretboard_atlas.md`

The atlas intentionally does not hallucinate 21 observed frets. It can return a
`coarse` preview when the internal grid evidence is insufficient.

### Other fret experiments

Additional fret-line implementations exist in:

- `main/video/fretboard/video_demo_hough.py` — Canny plus `HoughLinesP` on a
  Mask R-CNN-warped board.
- `notebooks/hough_transformation_fret_detection.ipynb` — Mask R-CNN,
  canonicalization, and Hough-based fret detection.
- `notebooks/template.ipynb` — multiscale template matching to locate a board
  ROI.
- `notebooks/template_hough.ipynb` — template ROI, thresholded mask, fitted
  quadrilateral, perspective warp, and Hough fret lines.
- `notebooks/canonical_fretboard.ipynb` — older Mask R-CNN plus
  minimum-area-rectangle canonicalization.

## 3. String detection

### Current named detector families

`main/video/fretboard/string_detection.py` exposes two main detectors and a
fusion step:

1. **`comb_band_slope_strip_fan`**
   - Thin-line ridge response.
   - Shared tilt/shear search.
   - Six-string equal-spacing comb.
   - Fret-crossing band constraint.
   - Per-string strip refinement.
   - Linear slope variation to model perspective fan-out.

2. **`lsd_strip_cascade`**
   - LSD line-segment proposals.
   - String-line consensus.
   - Falls back to the strip-projection detector when LSD evidence is
     incomplete.

3. **`fuse_arbitrate`**
   - Runs both families.
   - Averages them when they agree within a spacing tolerance.
   - Otherwise selects the result with stronger image evidence.

The documented notebook also preserves a simpler baseline:

- **Projection auto-tune** — preprocess the ROI, sum rows, and keep the six
  strongest row peaks.

### Related and older string paths

- `main/video/fretboard/string_evidence.py` implements the evidence-supported
  shared-slope projection plus independent longitudinal-strip checks. It is
  reused by the active string cascade and the atlas rectifier.
- `main/video/fretboard/fretboard_rectify.py` first uses LSD line candidates and
  string consensus, then falls back to `string_evidence.detect_strings`.
- `main/video/fretboard/video_demo_hough.py` uses Canny plus `HoughLinesP` to
  find horizontal string lines in a warped image.
- `main/deprecated/fretboard/string_detection/string_detection.py` uses PCA on
  the mask and draws six evenly spaced synthetic string lines. It does not
  directly observe string pixels.
- `notebooks/string_detection.ipynb` documents the projection baseline and the
  `comb_band_slope_strip_fan` method, including their held-out-frame results.

String detectors generally return screen-order rows. They do not, by
themselves, determine whether the top row is high E or low E.

## 4. Board geometry and rectification

There are several ways to get a board coordinate system:

- **Minimum-area rectangle** around the largest mask contour.
- **PCA plus tapered edge fitting** in `fret_detect_yolo.py`.
- **Robust board-corner estimation** using significant mask components,
  projected envelopes, convex geometry, and robust line fitting in
  `board_geometry.py`.
- **Evidence-constrained mesh rectification** using observed fret/string
  intersections in `fretboard_rectify.py` and `grid_warp.py`.
- **Manual four-corner calibration** through `BoardCalibration` in
  `audio_visual_fusion.py` and `video_fusion.py`.

The first four are visual geometry approaches. Manual calibration is an
explicit anchor used when absolute fret numbering is required.

## 5. Fret numbering and position mapping

The repository uses multiple notions of a fret coordinate, depending on the
workflow:

- **Rule of 18** — synthetic expected fret positions used by the PCA baseline.
- **Equal-tempered formula** — normalized position
  `X_n = 1 - 2^(-n/12)`.
- **Observed fret-wire interpolation** — measured wire points in a calibration
  file, interpolated and extrapolated by `BoardCalibration`.
- **Relative fret lattice** — atlas fitting of observed wires, which can support
  uniform relative cells but cannot identify the nut or absolute fret numbers
  without an anchor.
- **Audio pitch difference** — final fret candidate derived as MIDI pitch minus
  the open-string MIDI pitch.
- **Continuous calibrated fret coordinate** — used by `event_fusion.py` to map
  fingertip/finger-chain points to a numbered fret.

These are not interchangeable: some estimate image geometry, while others
resolve the musical string/fret identity.

## 6. Hand and fingertip detection

### Main repository workflow

The active local-video workflow uses **MediaPipe HandLandmarker** with the
bundled `main/models/hands/hand_landmarker.task` model. It extracts hand
landmarks and fingertip positions.

### GuiFiR benchmark alternatives

The optional GuiFiR scripts support several external hand-model routes:

- MediaPipe
- RTMPose hand crop
- RTMW-x
- ViTPose-B
- ViTPose-L
- HaMeR
- WiLoR

The relevant scripts are `scripts/guifir/predict_tips.py` and
`scripts/guifir/predict_mesh.py`. These models and their weights are generally
not vendored in this checkout.

## 7. Board tracking across video

`audio_visual_fusion.py` contains two tracking methods for moving from a
calibrated reference frame to another frame:

1. **Lucas–Kanade optical flow** with forward/backward consistency checks for
   small motions.
2. **ORB feature matching** with a Hamming matcher and RANSAC homography for
   larger jumps or when optical flow fails.

The visual pipeline can also bridge short tracking failures using nearby
accepted frames.

## 8. Audio transcription and audio analysis

### MT3 transcription choices

The current audio-to-score pipeline supports three MT3-Infer model choices:

- `mr_mt3`
- `mt3_pytorch`
- `yourmt3`

They all produce MIDI, which is then converted to guitar tablature.

### Non-MT3 audio paths

- `main/audio/audio_fusion.py` contains a lightweight spectral-flux onset
  detector and harmonic-salience pitch estimator.
- `main/deprecated/audio/chord_detection.py` and the FFT notebook synthesize
  candidate chords with Karplus–Strong, compute FFT spectra, and rank chord
  candidates by harmonic energy.
- `main/video/fretboard/yourmt3_bridge.py` is a separate bridge for running a
  local YourMT3 checkout in its own environment.

## 9. Visual/audio fingering and string assignment

There are multiple layers of position inference:

### Visual evidence modes

- **Uncalibrated string-row mode** — fingertip rows are aligned against two
  possible top-to-bottom string orders.
- **Calibrated fret-region mode** — a tracked board plus nut/numbered-wire
  calibration assigns fingertips to absolute fret regions.

### Audio-constrained assignment

- MIDI pitch enumerates all playable string/fret candidates under standard
  tuning.
- Visual contacts contribute likelihoods or tie-break scores.
- A whole-chord assignment enforces unique physical strings and is optimized
  with dynamic programming in `main/audio/guitar_tab.py`.

### Older event-fusion variants

`main/video/fretboard/event_fusion.py` retains several ablation modes:

- `tip_single`
- `single`
- `vote`
- `fusion`
- `hybrid`

It also contains context-based inference for unresolved notes using nearby
visually confirmed events.

## 10. Where the duplication is concentrated

If the goal is to consolidate the repository, the main overlapping clusters
are:

1. **Fretboard proposal**: YOLO versus Mask R-CNN.
2. **Mask-to-corners geometry**: minimum-area rectangle, PCA/tapered rails,
   robust board geometry, and manual calibration.
3. **Fret estimation**: six evaluator methods, Hough/template notebook paths,
   and the newer EFA wire/lattice method.
4. **String estimation**: comb/ridge detector, LSD cascade, strip-projection
   fallback, Hough detector, and deprecated PCA-generated rows.
5. **Hand modeling**: MediaPipe in the main workflow versus the external GuiFiR
   model set.
6. **Audio interpretation**: three MT3 models, lightweight spectral analysis,
   and the older FFT chord-candidate experiment.

The current documented default is roughly:

```text
YOLO fretboard mask
    -> PCA + perspective warp for the simple image path
    -> EFA observed grid for evidence-constrained rectification/fingering
    -> MediaPipe hand landmarks
    -> MT3 MIDI transcription
    -> audio-constrained string/fret assignment
```

The old evaluator, notebooks, demos, and `main/deprecated/` code remain useful
for comparison and research history, but they are not all part of one unified
production pipeline.

## 11. Reproducible calibrated audio/video pipeline

`scripts/transcribe_av.py` composes the existing implementations into a local
video workflow with mandatory visual evidence:

```text
video audio -> YourMT3 -> spectral-flux/harmonic onset recovery -> timed pitches
video frames -> YOLO/EFA + MediaPipe + calibrated LK/ORB tracking -> fret contacts
timed pitches + fret contacts -> whole-chord unique-string DP -> TAB + event JSON
```

Use measured image-only fret calibration to anchor absolute fret numbers. EFA
can remain coarse on occluded frames; accepted tracked calibration supplies
the absolute geometry in those frames. The command requires accepted visual
support and saves the audio-only baseline, diagnostic video, timed assignments,
input hashes, and visual contribution counts.

`main/evaluation/tab_ground_truth.py` separately compares saved outputs to an
ASCII reference, including simultaneous grouping and repeated attacks. It
does not supply notes or fingerings to inference. See the README command and
`research/stairway_av_verification.md` for the measured short-clip result.

`scripts/render_av_demo.py` renders a presentation overlay from the same
transcription JSON and video-only board calibration. It tracks the measured
board through the video, marks final assignments at audio attacks, and shows
tracking failures without drawing a grid. It intentionally omits unstable raw
per-frame atlas proposals; these remain available in `visual_diagnostics.json`.
