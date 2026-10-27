# Evidence-constrained fretboard atlas

Implemented and evaluated on September 29, 2026.

**Recommendation:** use a segmentation ensemble to propose the neck, recover its
internal fret/string lattice from image evidence, and rectify that lattice with
a checked, invertible mesh. A rectangular crop alone is an insufficient target.
The implementation is in `pipeline/fretboard_rectify.py`; the end-to-end entry
point is `python -m pipeline.rectify_atlas`.

This is a stronger candidate for this repository's standardized-grid objective,
with a measured improvement on the synthetic perspective case and useful output
on the bundled playing frame. It is **not established as universally superior**
to every prior method. Some baseline subpixel errors are smaller, and neither a
held-out real dataset nor the latest slide-only models are available here.

## What the slides and code actually establish

Source: `slides/Computer Vision Transcription.pdf`, 196 pages. Text was inspected
throughout; the relevant visual progression includes pages 20-21, 34-35, 58,
66-68, 77, 88-93, 100-108, 110-112, 121-125, 131-141, 147-156, 160-161,
171-176, 183-190 and 194-195.

| Existing approach | Evidence | Limitation for this task |
| --- | --- | --- |
| Segmentation + minimum-area rectangle | Canonicalization notebooks; slide 93 | Removes rotation, but a rectangle around a trapezoid does not recover its actual end caps or internal transverse direction. The visibly slanted frets remain in the crop. |
| PCA + tapered rails + homography (B) | `pipeline/fret_detect_yolo.py`; slides 124-125, 137 | Best available stored aggregate fret F1: YOLO 0.8067 at 2%, 0.6333 at 1%. Corners and end direction come from mask geometry; no real fret-wire measurements constrain the warp. |
| PCA without warp (C) | Evaluator; slide 137 | Strong strict-threshold score, but it does not produce the requested rectified texture. |
| Rotation/shear + Hough/projection (D/E) | Evaluator; slides 88-91 | Background lines and hand edges can dominate; one angle/shear cannot repair arbitrary local errors. |
| Sobel peaks in the warped mask (F) | Evaluator; slides 122-125 | Avoids unconditional phantom frets but loses recall and inherits the original corner error. |
| Template ROI + threshold mask + Hough | `notebooks/template*.ipynb`; slides 131-135, 147-154 | Recorded MPE 0.716%, with lower recall than B. Image-scale template matching is a proposal mechanism; it does not solve perspective correspondence or hand occlusion. Notebook templates are not distributed as standalone inputs. |
| String projection / K-means / auto-tuning | Slides 139, 156, 173-176, 187-190 | Hand edges corrupt clusters; horizontal projections assume a good warp. The slides explicitly say the reported 96% PASS rate checks spacing, not true string positions. |
| EfficientNet-B0 fret regression (H) | Slides 183-184 | Reported F1@2% 0.835, above B. Its implementation, weights and evaluation records are absent from this checkout. It must be included in a future real-data comparison. |
| ViViT | Slides 192-195 | A downstream temporal model; the described pipeline still relies on a canonical warp. No demonstrated rectification advantage is supplied. |

The active evaluator reduces predicted fret lines to midpoint x, matches them
to merged edges of annotation bounding boxes, and allows vertical overlap.
MPE excludes unmatched predictions and matches beyond 5% image width. Those
metrics cannot establish corner accuracy, string accuracy, or a rectangular
internal grid. A method can improve those scores and still produce a bad crop.

The current YOLO helper also directly resizes masks if dimensions differ. Those
dimensions can include inference letterboxing. The new adapter requests native
resolution masks and validates the dimensions; it keeps instances separate.
Existing baseline code is preserved for comparisons.

## The design

Call the method **Evidence-constrained Fretboard Atlas (EFA)**. Its useful change
is where the geometric constraints enter: observed internal wires control the
output coordinate system. The mask supplies a region hypothesis, not the final
answer. These are established building blocks arranged for this repository's
failure modes, not a claim of a newly discovered homography or line detector.

1. **Propose multiple neck regions.** Run the existing YOLO checkpoint and the
   original Mask R-CNN board checkpoint. Retain separate instance proposals.
   Do not union unrelated guitars. Rank usable results by an accepted internal
   grid, then accepted fret lattice, then observed wire count. This score is a
   heuristic, not calibrated confidence.
2. **Recover a robust coarse quadrilateral.** Prefer a well-supported convex
   four-sided silhouette. Otherwise fit binned rail and end envelopes with
   robust regression, retaining nearby significant fragments. Holes and local
   notches need not become corners. Reject degenerate geometry. Orient along
   a deterministic image axis; never infer the nut merely from apparent width.
3. **Measure transverse and longitudinal structure.** Extract subpixel line
   segments in the coarse view, merge both metal edges, and measure their
   union support. Estimate direction from segment directions rather than
   regressing all edge endpoints: uneven visibility of opposite wire edges
   otherwise creates a false tilt. Refine fret centers with narrow-band ridge
   measurements in independent horizontal strips. Frets must agree on a
   common line pencil.
4. **Recover faint strings with independent evidence.** If segment evidence
   is incomplete, search a shared shear using vertical bandpass ridge
   projections. Propose a roughly spaced string comb, then verify each string
   independently in longitudinal strips. A faint global peak can be proposed
   from neighboring strings, but it is accepted only with local image support.
   No forced six-cluster K-means and no evidence-free final string lines.
5. **Fit a relative physical fret sequence.** For standard equal-tempered
   straight-fret instruments, the normalized physical positions are
   `X_n = 1 - 2^(-n/12)`. Along a perspective image line they take the form
   `x_n = (a + b*q^n)/(1 + c*q^n)`, with `q = 2^(+/-1/12)`.
   Fit both directions robustly with bounded poles. A local spacing estimate
   permits small internal skips; inferred wires are explicitly marked. Only
   a sufficiently small residual enables equal-width relative fret intervals.
   The fit does not identify the nut, absolute fret numbers, or musical string
   names. Repeated missing frets can still cause correspondence ambiguity.
6. **Intersect the two line families.** Add mask rails and ROI end caps as
   boundaries, retaining partial end cells and board margins. Use each
   fret/string crossing as a mesh node. Six output strings occupy fixed rows
   `0.08 ... 0.92` of image height. Accepted relative fret intervals have equal
   width; end cells may be partial. Without an accepted lattice, preserve
   observed fret-center spacing.
7. **Resample original pixels through a checked inverse mesh.** Each target
   rectangle maps bilinearly to its source quadrilateral. Check all corner
   Jacobian determinants for nonzero, consistent orientation; reject folds.
   Sample the original RGB frame once for the final output, rather than
   resampling the coarse image again. Supply forward and inverse point
   mappings for fingertip coordinates. The coarse 3x3 homography alone is
   explicitly not the final mapping.
8. **Expose failure and missing information.** An empty/degenerate mask yields
   no result. Inadequate internal evidence yields a `coarse` preview with
   warnings. A validity mask records in-frame source samples. A provided hand
   mask produces conservative visibility using all bilinear contributors.
   Without that mask, visibility is unknown, not automatically unobstructed.

The mathematical rectangularity guarantee applies to **the fitted mesh**:
accepted crossing nodes land on the prescribed rows/columns, and corresponding
cell edges align with the rectangular grid. It does not prove that every fitted
line is the correct physical wire. Nor can any single-frame method recover
texture behind an opaque hand or outside the frame without inventing pixels.

Why a mesh? A good homography can align two consistent line pencils, and is an
excellent initializer. But equal physical fret indices are not equally spaced
in the image, and a general equal-cell atlas is not a single homography. A mesh
also provides an explicit path toward locally curved wires and lens correction.
This version fits straight line families; curved and multiscale/fanned frets
are not yet modeled.

## What was measured

Run `python scripts/benchmark_rectification.py --output-dir output/rectification/benchmark`.
The deterministic fixture renders a physically tapered board, fanning strings,
equal-temperament frets, wood texture and dots, projects it into a camera frame,
and retains exact floating-point crossing coordinates. It includes perspective,
rotation, hand occlusion, damaged segmentation, a partial crop and an empty
negative frame. Masks are supplied, so this isolates **rectification**, not
segmentation-model accuracy. The synthetic hand mask is supplied to EFA;
the historical baseline has no hand-mask interface.

Metrics below use the **same visible ground-truth points in the intersection
of both output domains**. Per-method coverage is also saved, so cropping out
difficult points cannot masquerade as better accuracy. The truth points are
independent of the fitted/drawn grid. Lower is better.

| Case | Fret x RMS, B (px) | EFA | String y RMS, B (px) | EFA | Fret-spacing CV, B | EFA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frontal | 0.001 | 0.006 | 0.035 | 0.252 | 47.13% | 0.56% |
| Perspective | 8.637 | 0.015 | 0.079 | 0.217 | 47.04% | 0.29% |
| Steep rotation | 0.394 | 0.011 | 0.026 | 0.239 | 47.01% | 0.33% |
| Hand occlusion | 0.001 | 0.006 | 0.037 | 0.230 | 47.13% | 0.70% |
| Mask notch/spur | 0.007 | 0.006 | 0.036 | 0.258 | 47.13% | 0.56% |
| Partial crop | 0.344 | 0.021 | 0.082 | 0.172 | 43.97% | 3.53% |

All six positive cases returned an evidence grid; both methods rejected the
empty mask. Fret-spacing CV measures standardization into **equal index cells**;
the old method was not designed to equalize cells. It should not be presented
as a failure to reproduce physical fret spacing. The new method substantially
improves the perspective fret orientation and equal-cell objective, while the
old baseline retains smaller string-straightness errors on these ideal planar
fixtures. This is a real tradeoff, not an across-the-board win. Runtime is
recorded separately and EFA costs more than the mask-only warp.

A separate stress run (`--fret-count 21 --seed 91`) also produced grids on all
six positive scenes and rejected the empty mask. It observed 18-20 interior
frets depending on crop/occlusion, without forcing a count of 21. Its paired
fret-spacing CV ranged from 0.89% to 4.11%, worse than the twelve-fret fixtures.
This exposes the loss of accuracy when the frame contains more tightly packed
frets; those results are retained under `output/rectification/stress_21/`.

The bundled 1920 x 1080 playing frame also runs through the real models. YOLO
supports 10 observed fret wires in its shortened region; the original Mask
R-CNN proposal supports 18 and is selected. The string-profile fallback obtains
six supported strings. This is qualitative evidence, not an annotated accuracy
measurement. `model_weights.pt` and the demo's identically named checkpoint
have identical SHA-256 hashes. The newer `model_weights_new.pt` produced several
shorter proposals on this frame, so the original board checkpoint is the default
second model. No claim is made that it wins on other guitars.

Default output is at `output/rectification/example/rectified.png`; the grid
overlay is a separate file. Occluding fingers remain visible in the output.
Outputs under `output/` and inspection files under `tmp/` are ignored by Git.

## Usage and integration

```bash
# End-to-end ensemble using the two bundled model weights.
python -m pipeline.rectify_atlas --image frame.png --output-dir output/atlas

# Existing segmentation and optional hand mask; no model load.
python -m pipeline.rectify_atlas --image frame.png --mask neck.png --occlusion-mask hand.png --output-dir output/atlas

# Lower-cost YOLO-only proposal path, or preserve observed x spacing.
python -m pipeline.rectify_atlas --image frame.png --yolo-only --spacing observed --output-dir output/atlas
```

```python
from pipeline.fretboard_rectify import rectify_fretboard

result = rectify_fretboard(rgb_uint8, neck_mask, output_size=(1024, 256))
if result is not None and result.quality["status"] == "grid":
    canonical = result.image
    canonical_fingertips = result.image_to_grid(fingertips_xy)
    original_points = result.grid_to_image(canonical_fingertips)
    # Points outside the mesh return NaN. Check validity before note mapping.
```

`geometry.json` records the mesh, target axes, selected proposal, observed and
inferred counts, support, lattice residuals and warnings. `mapping.npz` stores
the arrays without JSON rounding. These are the correct mappings to use after
rectification; downstream code must not apply only `coarse_transform`.

Automated verification covers physical perspective improvement, uniform
intervals, fixed string rows, occlusion footprints, mask damage/rotation,
degenerate inputs, no-evidence abstention, inverse-map round trips, JSON,
the model-free CLI, and preservation of existing baseline tests. The final
suite reports **24 passed**. This is software and synthetic-geometry validation,
not a real-world accuracy percentage.

## The next method improvements and the proof required

The implemented ensemble is a deployable research baseline, not the end of
the research program. My preferred full model retains this geometry and adds:

* **Learned amodal evidence with uncertainty:** train separate fret/rail/string
  heatmaps and a hand-occlusion map, including local orientation and visibility.
  Use the existing masks as weak supervision, synthetic physical renders for
  dense correspondences, and genuinely annotated real crossings for final
  calibration. A generic four-corner regressor gives too few constraints and
  is fragile when fingers hide endpoints.
* **Joint probabilistic correspondence:** replace the current local skip rule
  with monotone dynamic programming over fret detections and discrete gap
  counts, followed by robust joint line/mesh fitting. Retain competing lattice
  hypotheses when a repeated pattern cannot identify a unique index sequence.
* **Temporal anchoring:** detect the nut or an identifiable fret in a clear key
  frame, propagate geometry with visibility-aware feature tracking, and
  periodically redetect. Fuse *geometry* over time; an optional clean texture
  atlas must distinguish observed-at-another-time pixels from current pixels.
* **Small regularized curve corrections:** after camera calibration, fit curved
  string/fret traces only when residuals demand them. Penalize curvature, strain
  and Jacobian collapse so flexible warps do not make incorrect correspondences
  look artificially perfect.

These extensions are not represented as implemented or trained. A real test
set must split by performer, guitar and video, not random neighboring frames.
Annotate board corners, fret/string crossings, fret identities where knowable,
and visibility; include left-handed guitars, unusual string counts, dark
strings, blur, severe angle, partial necks, occlusion and clutter. Evaluate
endpoint/crossing errors relative to local string/fret spacing, 90th-percentile
error, coverage, wrong fret identities, calibration/abstention and temporal
jitter. Compare A-F, template/Hough, slide-only H if recovered, and ablations
of each new stage under identical masks. Predefine success criteria on
validation data and test once on held-out videos. That is the route to an
evidence-backed superiority claim.

## Primary references and prior art

The line frontend uses OpenCV's implementation of
[LSD (Grompone von Gioi et al., 2012)](https://www.ipol.im/pub/art/2012/gjmr-lsd/),
a published subpixel line-segment detector. Robust parameter fits use SciPy's
[least-squares optimizer](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).
Final inverse texture sampling follows the ordinary
[OpenCV remap convention](https://docs.opencv.org/4.x/d1/da0/tutorial_remap.html).

[TapToTab (Ghaleb et al., 2024)](https://arxiv.org/abs/2409.08618) combines YOLO
vision and Fourier audio analysis; its headline transcription results do not
constitute a benchmark for this atlas. Prior guitar-neck work already used
[feature matching and modified RANSAC homographies (Wang and Ohya)](https://waseda.elsevierpure.com/en/publications/an-accurate-and-robust-algorithm-for-tracking-guitar-neck-in-3d-b/).
The novelty here is a proposed evidence/geometry combination and its explicit
output contract for this repository; publishable novelty requires a fuller
prior-art study and independent evaluation.
