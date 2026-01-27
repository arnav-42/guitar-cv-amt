# guitar-cv-amt
Computer Vision for Automatic Music Transcription (Guitar)

## Projects

### Hand Tracking
Real-time finger tracking with MediaPipe. Tracks up to 2 hands with finger counting.

```bash
python hand_tracking_demo.py
```

Press 'q' to quit.

### FFT Chord Detection
Uses FFT to identify chords from audio when CV can't tell if strings are fretted, muted, or open.

See `fft_chord_candidates.ipynb` for details.

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
python fretboard_segmentation.py
```

The script downloads the test dataset from Kaggle and runs inference on sample images. Make sure `model_weights.pt` is in the project root.

## For new members:

Read the three papers and prepare a quick summary of each, focusing specifically on how the authors approached fingertip detection, fretboard detection, string detection, and distingushing between a finger hovering over a string vs pressing down on it.
- [Duke & Salgian (2019)](https://doi.org/10.1007/978-3-030-33723-0_20)
  - Note: You will need to sign in with your Purdue account to access this, select Purdue University Main Campus when prompted
  - The paper starts on page 248 (pdf page 267)
- [Asmar (2022)](https://publications.polymtl.ca/10470/1/2022_MarkAsmar.pdf)
- [Ghaleb et al. (2024)](https://arxiv.org/abs/2409.08618)
