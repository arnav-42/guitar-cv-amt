# guitar-cv-amt
This repository contains the ongoing work of the computer vision team of the [AIM Lab's](https://ai4musicians.org/) automatic music transcription group.

> [!NOTE]
> See `scripts/` and `notebooks/` for working code you can install and try out.
> 
> `docs/` folder and Github Pages site is also not finalized yet
> 
> `model_weights.pt` varies depending on folder

## Demos

[![Demo - Fretboard Canonicalization](https://img.shields.io/badge/Demo-Fretboard_Canonicalization-2ea44f?style=for-the-badge&logo=Jupyter)](https://nbviewer.org/github/arnav-42/guitar-cv-amt/blob/main/notebooks/canonical_fretboard.ipynb)

[![Demo - FFT Chord Candidates](https://img.shields.io/badge/Demo-FFT_Chord_Candidates-2ea44f?style=for-the-badge&logo=Jupyter)](https://nbviewer.org/github/arnav-42/guitar-cv-amt/blob/main/notebooks/fft_chord_candidates.ipynb)

## Important Folders
- `docs/`: GitHub Pages site root (set Pages to publish from main /docs)
- `scripts/`: Python files and demos
- `notebooks/`: .ipynb notebooks and demos
- `deprecated/`: Legacy or frozen demos

## For new members

Read the three papers and prepare a quick summary of each, focusing specifically on how the authors approached fingertip detection, fretboard detection, string detection, and distingushing between a finger hovering over a string vs pressing down on it.
- [Duke & Salgian (2019)](https://doi.org/10.1007/978-3-030-33723-0_20)
  - Note: You will need to sign in with your Purdue account to access this, select Purdue University Main Campus when prompted
  - The paper starts on page 248 (pdf page 267)
- [Asmar (2022)](https://publications.polymtl.ca/10470/1/2022_MarkAsmar.pdf)
- [Ghaleb et al. (2024)](https://arxiv.org/abs/2409.08618)
