# guitar-cv-amt
Computer Vision for Automatic Music Transcription (Guitar)

## Projects

### 1. MediaPipe Hand Tracking Demo
A real-time hand and finger tracking demo using MediaPipe. Tracks up to 2 hands simultaneously with finger counting and visual highlights.

**Requirements:**
- Python 3.8-3.11 (MediaPipe doesn't support Python 3.12+ yet)
- Webcam

**Setup:**
```bash
# Create virtual environment with Python 3.11
py -3.11 -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the demo
python hand_tracking_demo.py
```

Press 'q' to quit the demo.

### 2. FFT Chord Detection
A Jupyter notebook that uses Fast Fourier Transform (FFT) to identify guitar chords from audio, helping resolve ambiguity when computer vision detects finger positions but can't determine if strings are fretted, muted, or open.

See `fft_chord_candidates.ipynb` for details and usage.