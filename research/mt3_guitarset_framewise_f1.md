# GuitarSet framewise active-pitch result

Run date: 2026-10-10

MR-MT3 scored **69.7% framewise active-pitch F1** on this fixed 8-clip GuitarSet subset.

Precision: 70.9%. Recall: 68.5%.

The score compares active MIDI pitches at 10 ms frame centers. A pitch matches within 0.5 semitones; F1 is the harmonic mean of framewise precision and recall. This measures pitch activity overlap rather than note-by-note onset or offset accuracy.

The subset contains 8 acoustic microphone recordings from 2 players (4 comp and 4 solo), totaling 231.5 seconds. The audio was converted by this repository's FFmpeg extraction path and transcribed by its MR-MT3 path.

| Track | Player | Style | Framewise active-pitch F1 |
|---|---:|---|---:|
| `00_BN1-129-Eb_comp` | 0 | comp | 77.9% |
| `00_BN1-129-Eb_solo` | 0 | solo | 86.8% |
| `00_Funk2-108-Eb_comp` | 0 | comp | 69.0% |
| `00_Funk2-108-Eb_solo` | 0 | solo | 72.4% |
| `01_BN1-129-Eb_comp` | 1 | comp | 83.8% |
| `01_BN1-129-Eb_solo` | 1 | solo | 74.6% |
| `01_Funk2-108-Eb_comp` | 1 | comp | 45.1% |
| `01_Funk2-108-Eb_solo` | 1 | solo | 48.5% |

This is a small acoustic-guitar benchmark and should be described with its metric and subset. GuitarSet reference note pitch and offset labels include automatic estimates. The [GuitarSet dataset](https://doi.org/10.5281/zenodo.1422265) is accessed through its [Hugging Face mirror](https://huggingface.co/datasets/jhartquist/guitarset). The framewise pitch metric follows `mir_eval`'s multipitch matching procedure.

Reproduce with:

```bash
uv run --python 3.10 --extra transcription -- python -m scripts.evaluate_mt3_guitarset_framewise
```

The selected audio and MIDI files are cached in the system temporary directory and are not stored in the repository.
