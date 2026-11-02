# GuitarSet framewise active-pitch result

Run date: 2026-11-02

yourmt3 scored **87.4% framewise active-pitch F1** on this fixed 8-clip GuitarSet subset.

Precision: 86.2%. Recall: 88.7%.

The score compares active MIDI pitches at 10 ms frame centers. A pitch matches within 0.5 semitones; F1 is the harmonic mean of framewise precision and recall. This measures pitch activity overlap rather than note-by-note onset or offset accuracy.

The subset contains 8 acoustic microphone recordings from 2 players (4 comp and 4 solo), totaling 231.5 seconds. The audio was converted by this repository's FFmpeg extraction path and transcribed by its yourmt3 path.
Fixed Hugging Face row ranges: ((0, 2), (16, 2), (60, 2), (76, 2)); the default limit evaluates all 8 selected clips.

| Track | Player | Style | Framewise active-pitch F1 | Exact tab P/R/F1 | Assignment accuracy | Ref notes | Pred MIDI | Rendered tab | Unplayable/unassigned |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `00_BN1-129-Eb_comp` | 0 | comp | 90.8% | 49.6% / 45.9% / 47.7% | 51.7% | 133 | 123 | 123 | 0 |
| `00_BN1-129-Eb_solo` | 0 | solo | 89.9% | 58.0% / 56.3% / 57.1% | 60.9% | 71 | 69 | 69 | 0 |
| `00_Funk2-108-Eb_comp` | 0 | comp | 87.3% | 69.8% / 70.2% / 70.0% | 80.8% | 523 | 526 | 525 | 1 |
| `00_Funk2-108-Eb_solo` | 0 | solo | 86.1% | 57.7% / 52.9% / 55.2% | 63.0% | 121 | 111 | 111 | 0 |
| `01_BN1-129-Eb_comp` | 1 | comp | 89.8% | 25.5% / 26.7% / 26.0% | 28.9% | 105 | 110 | 110 | 0 |
| `01_BN1-129-Eb_solo` | 1 | solo | 91.3% | 8.5% / 8.2% / 8.3% | 8.8% | 61 | 59 | 59 | 0 |
| `01_Funk2-108-Eb_comp` | 1 | comp | 81.9% | 80.2% / 91.5% / 85.5% | 99.3% | 293 | 334 | 334 | 0 |
| `01_Funk2-108-Eb_solo` | 1 | solo | 83.7% | 19.3% / 19.4% / 19.3% | 21.3% | 191 | 192 | 192 | 0 |

Open the [ground-truth versus generated tab comparison report](yourmt3_guitarset_tab_comparisons.md); output path: `research/yourmt3_guitarset_tab_comparisons.md`.

Per-clip tab artifacts:
- `00_BN1-129-Eb_comp` reference: `/var/folders/bz/7p5d7dqx58x3s6y54v1qsk7w0000gn/T/guitar-cv-amt-mt3-benchmark/yourmt3/tabs/reference/00_BN1-129-Eb_comp.txt`; predicted: `/var/folders/bz/7p5d7dqx58x3s6y54v1qsk7w0000gn/T/guitar-cv-amt-mt3-benchmark/yourmt3/tabs/predicted/00_BN1-129-Eb_comp.txt`
  counts: reference tab 133, predicted MIDI 123, rendered 123, unplayable/unassigned 0; exact TP/FP/FN 61/62/72; pitch-matched/assignment-correct 118/61
- `00_BN1-129-Eb_solo` reference: `/var/folders/bz/7p5d7dqx58x3s6y54v1qsk7w0000gn/T/guitar-cv-amt-mt3-benchmark/yourmt3/tabs/reference/00_BN1-129-Eb_solo.txt`; predicted: `/var/folders/bz/7p5d7dqx58x3s6y54v1qsk7w0000gn/T/guitar-cv-amt-mt3-benchmark/yourmt3/tabs/predicted/00_BN1-129-Eb_solo.txt`
  counts: reference tab 71, predicted MIDI 69, rendered 69, unplayable/unassigned 0; exact TP/FP/FN 40/29/31; pitch-matched/assignment-correct 64/39
- `00_Funk2-108-Eb_comp` reference: `/var/folders/bz/7p5d7dqx58x3s6y54v1qsk7w0000gn/T/guitar-cv-amt-mt3-benchmark/yourmt3/tabs/reference/00_Funk2-108-Eb_comp.txt`; predicted: `/var/folders/bz/7p5d7dqx58x3s6y54v1qsk7w0000gn/T/guitar-cv-amt-mt3-benchmark/yourmt3/tabs/predicted/00_Funk2-108-Eb_comp.txt`
  counts: reference tab 523, predicted MIDI 526, rendered 525, unplayable/unassigned 1; exact TP/FP/FN 367/159/156; pitch-matched/assignment-correct 453/366
- `00_Funk2-108-Eb_solo` reference: `/var/folders/bz/7p5d7dqx58x3s6y54v1qsk7w0000gn/T/guitar-cv-amt-mt3-benchmark/yourmt3/tabs/reference/00_Funk2-108-Eb_solo.txt`; predicted: `/var/folders/bz/7p5d7dqx58x3s6y54v1qsk7w0000gn/T/guitar-cv-amt-mt3-benchmark/yourmt3/tabs/predicted/00_Funk2-108-Eb_solo.txt`
  counts: reference tab 121, predicted MIDI 111, rendered 111, unplayable/unassigned 0; exact TP/FP/FN 64/47/57; pitch-matched/assignment-correct 100/63
- `01_BN1-129-Eb_comp` reference: `/var/folders/bz/7p5d7dqx58x3s6y54v1qsk7w0000gn/T/guitar-cv-amt-mt3-benchmark/yourmt3/tabs/reference/01_BN1-129-Eb_comp.txt`; predicted: `/var/folders/bz/7p5d7dqx58x3s6y54v1qsk7w0000gn/T/guitar-cv-amt-mt3-benchmark/yourmt3/tabs/predicted/01_BN1-129-Eb_comp.txt`
  counts: reference tab 105, predicted MIDI 110, rendered 110, unplayable/unassigned 0; exact TP/FP/FN 28/82/77; pitch-matched/assignment-correct 97/28
- `01_BN1-129-Eb_solo` reference: `/var/folders/bz/7p5d7dqx58x3s6y54v1qsk7w0000gn/T/guitar-cv-amt-mt3-benchmark/yourmt3/tabs/reference/01_BN1-129-Eb_solo.txt`; predicted: `/var/folders/bz/7p5d7dqx58x3s6y54v1qsk7w0000gn/T/guitar-cv-amt-mt3-benchmark/yourmt3/tabs/predicted/01_BN1-129-Eb_solo.txt`
  counts: reference tab 61, predicted MIDI 59, rendered 59, unplayable/unassigned 0; exact TP/FP/FN 5/54/56; pitch-matched/assignment-correct 57/5
- `01_Funk2-108-Eb_comp` reference: `/var/folders/bz/7p5d7dqx58x3s6y54v1qsk7w0000gn/T/guitar-cv-amt-mt3-benchmark/yourmt3/tabs/reference/01_Funk2-108-Eb_comp.txt`; predicted: `/var/folders/bz/7p5d7dqx58x3s6y54v1qsk7w0000gn/T/guitar-cv-amt-mt3-benchmark/yourmt3/tabs/predicted/01_Funk2-108-Eb_comp.txt`
  counts: reference tab 293, predicted MIDI 334, rendered 334, unplayable/unassigned 0; exact TP/FP/FN 268/66/25; pitch-matched/assignment-correct 270/268
- `01_Funk2-108-Eb_solo` reference: `/var/folders/bz/7p5d7dqx58x3s6y54v1qsk7w0000gn/T/guitar-cv-amt-mt3-benchmark/yourmt3/tabs/reference/01_Funk2-108-Eb_solo.txt`; predicted: `/var/folders/bz/7p5d7dqx58x3s6y54v1qsk7w0000gn/T/guitar-cv-amt-mt3-benchmark/yourmt3/tabs/predicted/01_Funk2-108-Eb_solo.txt`
  counts: reference tab 191, predicted MIDI 192, rendered 192, unplayable/unassigned 0; exact TP/FP/FN 37/155/154; pitch-matched/assignment-correct 174/37

## Generated guitar-tab augmentation

Pooled exact-tab precision/recall/F1: **57.1% / 58.1% / 57.6%**.
Pitch-matched string/fret assignment accuracy: **65.0%**.

Tab events are matched one-to-one within a 50 ms onset tolerance and must have identical string and fret. Assignment accuracy instead matches playable events by onset and rounded pitch, then checks their string/fret assignment.

Counts: 1498 reference notes (1498 playable, 0 invalid/unplayable reference), 1524 predicted MIDI notes, 1523 rendered predicted tab notes, 1 unplayable/unassigned predictions; exact-tab TP/FP/FN = 870/654/628; pitch-matched/assignment-correct = 1333/867.

Exact-tab precision uses all counted predicted MIDI notes as its denominator, so non-drum MIDI predictions that cannot be assigned to a playable string/fret remain disclosed and penalize precision rather than disappearing. Reference notes with invalid string/fret labels are excluded from playable-tab recall and retained in the reference-note count.
Predicted MIDI counts include every nonzero-duration non-drum MIDI note; drum tracks are excluded to match guitar-tab assignment, while unplayable/unassigned non-drum predictions remain counted and penalize precision.

Reference and predicted ASCII-tab artifacts are cached under `/var/folders/bz/7p5d7dqx58x3s6y54v1qsk7w0000gn/T/guitar-cv-amt-mt3-benchmark/yourmt3/tabs`. They are a generated local benchmark augmentation; upstream GuitarSet rows are not mutated.

This is a small acoustic-guitar benchmark and should be described with its metric and subset. GuitarSet reference note pitch and offset labels include automatic estimates. The [GuitarSet dataset](https://doi.org/10.5281/zenodo.1422265) is accessed through its [Hugging Face mirror](https://huggingface.co/datasets/jhartquist/guitarset). The framewise pitch metric follows `mir_eval`'s multipitch matching procedure.

Reproduce with:

```bash
uv run --python 3.10 --extra transcription -- python -m scripts.evaluate_mt3_guitarset_framewise --model yourmt3
```

The selected audio, MIDI files, and generated tab artifacts are cached in the system temporary directory and are not stored in the repository.
