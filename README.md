# guitar-cv-amt
Computer Vision for Automatic Music Transcription (Guitar)

## Setup

Python 3.8-3.11 required (MediaPipe doesn't support 3.12+).

```bash
# Create venv
py -3.11 -m venv venv

# Activate (PowerShell)
.\venv\Scripts\Activate.ps1
# Or CMD
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

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

## Web Demo

Gradio interface combining hand tracking and chord detection.

```bash
python app.py
```

Open `http://localhost:7860` in your browser.

**Deploy to Hugging Face:**
1. Push to GitHub
2. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces)
3. Connect your repo
4. Auto-deploys