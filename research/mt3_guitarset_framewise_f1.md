# GuitarSet framewise active-pitch result

Run date: 2026-10-20

MR-MT3 scored **69.7% framewise active-pitch F1** on this fixed 8-clip GuitarSet subset.

Precision: 70.9%. Recall: 68.5%.

The score compares active MIDI pitches at 10 ms frame centers. A pitch matches within 0.5 semitones; F1 is the harmonic mean of framewise precision and recall. This measures pitch activity overlap rather than note-by-note onset or offset accuracy.

The subset contains 8 acoustic microphone recordings from 2 players (4 comp and 4 solo), totaling 231.5 seconds. The audio was converted by this repository's FFmpeg extraction path and transcribed by its MR-MT3 path.
Fixed Hugging Face row ranges: ((0, 2), (16, 2), (60, 2), (76, 2)); the default limit evaluates all 8 selected clips.

| Track | Player | Style | Framewise active-pitch F1 | Exact tab P/R/F1 | Assignment accuracy | Ref notes | Pred MIDI | Rendered tab | Unplayable/unassigned |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `00_BN1-129-Eb_comp` | 0 | comp | 77.9% | 9.8% / 9.8% / 9.8% | 38.2% | 133 | 133 | 133 | 0 |
| `00_BN1-129-Eb_solo` | 0 | solo | 86.8% | 19.7% / 19.7% / 19.7% | 42.4% | 71 | 71 | 71 | 0 |
| `00_Funk2-108-Eb_comp` | 0 | comp | 69.0% | 27.1% / 26.6% / 26.8% | 75.1% | 523 | 513 | 513 | 0 |
| `00_Funk2-108-Eb_solo` | 0 | solo | 72.4% | 29.4% / 28.9% / 29.2% | 72.9% | 121 | 119 | 119 | 0 |
| `01_BN1-129-Eb_comp` | 1 | comp | 83.8% | 13.5% / 13.3% / 13.4% | 48.3% | 105 | 104 | 104 | 0 |
| `01_BN1-129-Eb_solo` | 1 | solo | 74.6% | 0.0% / 0.0% / 0.0% | 0.0% | 61 | 61 | 61 | 0 |
| `01_Funk2-108-Eb_comp` | 1 | comp | 45.1% | 32.2% / 32.1% / 32.1% | 100.0% | 293 | 292 | 292 | 0 |
| `01_Funk2-108-Eb_solo` | 1 | solo | 48.5% | 8.0% / 7.9% / 7.9% | 22.7% | 191 | 188 | 188 | 0 |

Open the [ground-truth versus generated tab comparison report](mt3_guitarset_tab_comparisons.md); output path: `research/mt3_guitarset_tab_comparisons.md`.

Per-clip tab artifacts:
- `00_BN1-129-Eb_comp` reference: `/var/folders/7b/lqlr4kfn56qgp8tmk31crqxc0000gn/T/guitar-cv-amt-mt3-benchmark/tabs/reference/00_BN1-129-Eb_comp.txt`; predicted: `/var/folders/7b/lqlr4kfn56qgp8tmk31crqxc0000gn/T/guitar-cv-amt-mt3-benchmark/tabs/predicted/00_BN1-129-Eb_comp.txt`
  counts: reference tab 133, predicted MIDI 133, rendered 133, unplayable/unassigned 0; exact TP/FP/FN 13/120/120; pitch-matched/assignment-correct 34/13
- `00_BN1-129-Eb_solo` reference: `/var/folders/7b/lqlr4kfn56qgp8tmk31crqxc0000gn/T/guitar-cv-amt-mt3-benchmark/tabs/reference/00_BN1-129-Eb_solo.txt`; predicted: `/var/folders/7b/lqlr4kfn56qgp8tmk31crqxc0000gn/T/guitar-cv-amt-mt3-benchmark/tabs/predicted/00_BN1-129-Eb_solo.txt`
  counts: reference tab 71, predicted MIDI 71, rendered 71, unplayable/unassigned 0; exact TP/FP/FN 14/57/57; pitch-matched/assignment-correct 33/14
- `00_Funk2-108-Eb_comp` reference: `/var/folders/7b/lqlr4kfn56qgp8tmk31crqxc0000gn/T/guitar-cv-amt-mt3-benchmark/tabs/reference/00_Funk2-108-Eb_comp.txt`; predicted: `/var/folders/7b/lqlr4kfn56qgp8tmk31crqxc0000gn/T/guitar-cv-amt-mt3-benchmark/tabs/predicted/00_Funk2-108-Eb_comp.txt`
  counts: reference tab 523, predicted MIDI 513, rendered 513, unplayable/unassigned 0; exact TP/FP/FN 139/374/384; pitch-matched/assignment-correct 185/139
- `00_Funk2-108-Eb_solo` reference: `/var/folders/7b/lqlr4kfn56qgp8tmk31crqxc0000gn/T/guitar-cv-amt-mt3-benchmark/tabs/reference/00_Funk2-108-Eb_solo.txt`; predicted: `/var/folders/7b/lqlr4kfn56qgp8tmk31crqxc0000gn/T/guitar-cv-amt-mt3-benchmark/tabs/predicted/00_Funk2-108-Eb_solo.txt`
  counts: reference tab 121, predicted MIDI 119, rendered 119, unplayable/unassigned 0; exact TP/FP/FN 35/84/86; pitch-matched/assignment-correct 48/35
- `01_BN1-129-Eb_comp` reference: `/var/folders/7b/lqlr4kfn56qgp8tmk31crqxc0000gn/T/guitar-cv-amt-mt3-benchmark/tabs/reference/01_BN1-129-Eb_comp.txt`; predicted: `/var/folders/7b/lqlr4kfn56qgp8tmk31crqxc0000gn/T/guitar-cv-amt-mt3-benchmark/tabs/predicted/01_BN1-129-Eb_comp.txt`
  counts: reference tab 105, predicted MIDI 104, rendered 104, unplayable/unassigned 0; exact TP/FP/FN 14/90/91; pitch-matched/assignment-correct 29/14
- `01_BN1-129-Eb_solo` reference: `/var/folders/7b/lqlr4kfn56qgp8tmk31crqxc0000gn/T/guitar-cv-amt-mt3-benchmark/tabs/reference/01_BN1-129-Eb_solo.txt`; predicted: `/var/folders/7b/lqlr4kfn56qgp8tmk31crqxc0000gn/T/guitar-cv-amt-mt3-benchmark/tabs/predicted/01_BN1-129-Eb_solo.txt`
  counts: reference tab 61, predicted MIDI 61, rendered 61, unplayable/unassigned 0; exact TP/FP/FN 0/61/61; pitch-matched/assignment-correct 26/0
- `01_Funk2-108-Eb_comp` reference: `/var/folders/7b/lqlr4kfn56qgp8tmk31crqxc0000gn/T/guitar-cv-amt-mt3-benchmark/tabs/reference/01_Funk2-108-Eb_comp.txt`; predicted: `/var/folders/7b/lqlr4kfn56qgp8tmk31crqxc0000gn/T/guitar-cv-amt-mt3-benchmark/tabs/predicted/01_Funk2-108-Eb_comp.txt`
  counts: reference tab 293, predicted MIDI 292, rendered 292, unplayable/unassigned 0; exact TP/FP/FN 94/198/199; pitch-matched/assignment-correct 94/94
- `01_Funk2-108-Eb_solo` reference: `/var/folders/7b/lqlr4kfn56qgp8tmk31crqxc0000gn/T/guitar-cv-amt-mt3-benchmark/tabs/reference/01_Funk2-108-Eb_solo.txt`; predicted: `/var/folders/7b/lqlr4kfn56qgp8tmk31crqxc0000gn/T/guitar-cv-amt-mt3-benchmark/tabs/predicted/01_Funk2-108-Eb_solo.txt`
  counts: reference tab 191, predicted MIDI 188, rendered 188, unplayable/unassigned 0; exact TP/FP/FN 15/173/176; pitch-matched/assignment-correct 66/15

## Generated guitar-tab augmentation

Pooled exact-tab precision/recall/F1: **21.9% / 21.6% / 21.8%**.
Pitch-matched string/fret assignment accuracy: **62.9%**.

Tab events are matched one-to-one within a 50 ms onset tolerance and must have identical string and fret. Assignment accuracy instead matches playable events by onset and rounded pitch, then checks their string/fret assignment.

Counts: 1498 reference notes (1498 playable, 0 invalid/unplayable reference), 1481 predicted MIDI notes, 1481 rendered predicted tab notes, 0 unplayable/unassigned predictions; exact-tab TP/FP/FN = 324/1157/1174; pitch-matched/assignment-correct = 515/324.

Exact-tab precision uses all counted predicted MIDI notes as its denominator, so non-drum MIDI predictions that cannot be assigned to a playable string/fret remain disclosed and penalize precision rather than disappearing. Reference notes with invalid string/fret labels are excluded from playable-tab recall and retained in the reference-note count.
Predicted MIDI counts include every nonzero-duration non-drum MIDI note; drum tracks are excluded to match guitar-tab assignment, while unplayable/unassigned non-drum predictions remain counted and penalize precision.

Reference and predicted ASCII-tab artifacts are cached under `/var/folders/7b/lqlr4kfn56qgp8tmk31crqxc0000gn/T/guitar-cv-amt-mt3-benchmark/tabs`. They are a generated local benchmark augmentation; upstream GuitarSet rows are not mutated.

This is a small acoustic-guitar benchmark and should be described with its metric and subset. GuitarSet reference note pitch and offset labels include automatic estimates. The [GuitarSet dataset](https://doi.org/10.5281/zenodo.1422265) is accessed through its [Hugging Face mirror](https://huggingface.co/datasets/jhartquist/guitarset). The framewise pitch metric follows `mir_eval`'s multipitch matching procedure.

Reproduce with:

```bash
uv run --python 3.10 --extra transcription -- python -m scripts.evaluate_mt3_guitarset_framewise
```

The selected audio, MIDI files, and generated tab artifacts are cached in the system temporary directory and are not stored in the repository.
