"""Extract audio from a video and transcribe it with MT3-Infer."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import Sequence

import numpy as np


SAMPLE_RATE = 16_000
MODEL_NAMES = ("mr_mt3", "mt3_pytorch", "yourmt3")


class PipelineError(RuntimeError):
    """A user-facing error while extracting or transcribing a video."""


def extract_audio(video_path: Path, wav_path: Path, *, overwrite: bool) -> None:
    """Extract the first audio stream as mono, 16 kHz, 16-bit PCM WAV."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise PipelineError(
            "ffmpeg was not found. Install FFmpeg and make sure `ffmpeg` is on PATH."
        )

    command = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    command.append("-y" if overwrite else "-n")
    command.extend(
        [
            "-i",
            str(video_path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(wav_path),
        ]
    )
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or "").strip()
        message = f"FFmpeg could not extract audio from {video_path}."
        if details:
            message = f"{message}\n{details}"
        raise PipelineError(message) from exc


def read_model_audio(wav_path: Path) -> np.ndarray:
    """Load the extracted PCM WAV as a normalized mono float32 array."""
    with wave.open(str(wav_path), "rb") as wav_file:
        if wav_file.getnchannels() != 1 or wav_file.getframerate() != SAMPLE_RATE:
            raise PipelineError("Extracted WAV has an unexpected format.")
        if wav_file.getsampwidth() != 2:
            raise PipelineError("Extracted WAV is not 16-bit PCM.")
        samples = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")

    if samples.size == 0:
        raise PipelineError("The video contains no usable audio samples.")
    return samples.astype(np.float32) / 32768.0


def transcribe_audio(
    audio: np.ndarray,
    midi_path: Path,
    *,
    model_name: str,
    device: str,
) -> None:
    """Run MT3-Infer and save its transcription as a MIDI file."""
    try:
        from mt3_infer import load_model
    except ImportError as exc:
        raise PipelineError(
            "MT3-Infer is not installed. Install it with the command in the README."
        ) from exc

    print(f"Loading {model_name} on {device} (first use may download its checkpoint)...")
    model = load_model(model_name, device=device)
    midi = model.transcribe(audio, sr=SAMPLE_RATE)
    midi.save(str(midi_path))


def write_musicxml(midi_path: Path, musicxml_path: Path) -> None:
    """Convert the transcribed MIDI into editable sheet music (MusicXML)."""
    try:
        from music21 import converter
    except ImportError as exc:
        raise PipelineError(
            "music21 is not installed. Install the transcription requirements from the README."
        ) from exc

    score = converter.parse(str(midi_path))
    score.write("musicxml", fp=str(musicxml_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract audio from a video, transcribe it with MT3-Infer, and save "
            "the audio, MIDI, and MusicXML sheet music."
        )
    )
    parser.add_argument("video", type=Path, help="Video file containing the performance")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: <video-name>_transcription beside the video)",
    )
    parser.add_argument("--model", choices=MODEL_NAMES, default="mr_mt3")
    parser.add_argument(
        "--device",
        default="auto",
        help="MT3-Infer device: auto, cpu, cuda, or cuda:N (default: auto)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing audio, MIDI, and MusicXML outputs",
    )
    return parser


def run(video_path: Path, output_dir: Path, *, model_name: str, device: str, overwrite: bool) -> None:
    if not video_path.is_file():
        raise PipelineError(f"Video file does not exist: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "audio": output_dir / "audio.wav",
        "MIDI": output_dir / "transcription.mid",
        "sheet music": output_dir / "sheet.musicxml",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise PipelineError(
            "Output file(s) already exist: "
            + ", ".join(existing)
            + ". Choose another --output-dir or pass --overwrite."
        )

    print(f"Extracting audio from {video_path}...")
    extract_audio(video_path, outputs["audio"], overwrite=overwrite)
    audio = read_model_audio(outputs["audio"])

    transcribe_audio(
        audio,
        outputs["MIDI"],
        model_name=model_name,
        device=device,
    )
    print("Converting MIDI to MusicXML...")
    write_musicxml(outputs["MIDI"], outputs["sheet music"])

    print("Wrote:")
    for label, path in outputs.items():
        print(f"  {label}: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = args.output_dir or args.video.with_name(
        f"{args.video.stem}_transcription"
    )

    try:
        run(
            args.video,
            output_dir,
            model_name=args.model,
            device=args.device,
            overwrite=args.overwrite,
        )
    except (PipelineError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
