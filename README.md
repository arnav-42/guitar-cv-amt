# guitar-cv-amt
Computer Vision for Automatic Music Transcription (Guitar)

## Repo layout
- docs/ : GitHub Pages site root (set Pages to publish from main /docs)
- src/ : Python demos and scripts
- notebooks/ : Source .ipynb notebooks
- deprecated/ : Legacy or frozen demos
- model_weights.pt : Mask R-CNN weights

## GitHub Pages (Voici)
This repo publishes a static site from `docs/` with notebook demos built by Voici.

Build the site after adding or updating notebooks:

```powershell
& "$env:APPDATA\Python\Python313\Scripts\voici.exe" build --contents notebooks --output-dir docs --disable-addons jupyterlite-xeus
```

Notes:
- The build output overwrites `docs/`.
- Commit `docs/` after each build so GitHub Pages can deploy.

## Projects

### Hand Tracking
Real-time finger tracking with MediaPipe. Tracks up to 2 hands with finger counting.

```bash
python src/hand_tracking_demo.py
```

Press 'q' to quit.

### FFT Chord Detection
Uses FFT to identify chords from audio when CV can't tell if strings are fretted, muted, or open.

See `notebooks/fft_chord_candidates.ipynb` for details.

### Fretboard Segmentation
Mask R-CNN model for segmenting the fretboard from images.

**Additional requirements:**
- PyTorch
- torchvision
- kagglehub
- `model_weights.pt` (trained model weights)

```bash
# Install additional dependencies
pip install torch torchvision kagglehub matplotlib

# Run segmentation
python src/fretboard_segmentation.py
```

The script downloads the test dataset from Kaggle and runs inference on sample images. Make sure `model_weights.pt` is in the project root.

## For new members:

Read the three papers and prepare a quick summary of each, focusing specifically on how the authors approached fingertip detection, fretboard detection, string detection, and distingushing between a finger hovering over a string vs pressing down on it.
- [Duke & Salgian (2019)](https://doi.org/10.1007/978-3-030-33723-0_20)
  - Note: You will need to sign in with your Purdue account to access this, select Purdue University Main Campus when prompted
  - The paper starts on page 248 (pdf page 267)
- [Asmar (2022)](https://publications.polymtl.ca/10470/1/2022_MarkAsmar.pdf)
- [Ghaleb et al. (2024)](https://arxiv.org/abs/2409.08618)
