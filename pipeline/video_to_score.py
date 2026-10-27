"""Extract audio from a video and transcribe it with MT3-Infer."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, urlparse

import numpy as np

from pipeline.guitar_tab import (
    GuitarTabError,
    assign_tab_notes,
    midi_to_tab_notes,
    read_midi_notes,
    render_ascii_tab,
)
from pipeline.visual_fingering import VisualExtractionError

SAMPLE_RATE = 16_000
MODEL_NAMES = ("mr_mt3", "mt3_pytorch", "yourmt3")
MR_MT3_INFERENCE_BATCH_SIZE = 8


class PipelineError(RuntimeError):
    """A user-facing error while downloading or transcribing a performance."""


def is_youtube_url(source: str) -> bool:
    """Return whether source is an HTTP(S) URL on a YouTube domain."""
    parsed = urlparse(source)
    host = (parsed.hostname or "").lower()
    return parsed.scheme in {"http", "https"} and (
        host == "youtu.be" or host == "youtube.com" or host.endswith(".youtube.com")
    )


def youtube_video_id(url: str) -> str | None:
    """Extract an ID from common single-video YouTube URL forms."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host == "youtu.be":
        return parsed.path.strip("/").split("/")[0] or None

    query_id = parse_qs(parsed.query).get("v", [None])[0]
    if query_id:
        return query_id

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live"}:
        return parts[1]
    return None


def download_youtube_source(url: str, download_dir: Path) -> Path:
    """Download a YouTube video's best audio stream using yt-dlp."""
    try:
        import yt_dlp  # noqa: F401
    except ImportError as exc:
        raise PipelineError(
            "yt-dlp is not installed. Install the transcription dependencies from the README."
        ) from exc

    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--no-progress",
        "--no-warnings",
        "--quiet",
        "--format",
        "bestaudio/best",
        "--output",
        str(download_dir / "source.%(ext)s"),
        "--print",
        "after_move:filepath",
        url,
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or "").strip()
        message = f"yt-dlp could not download audio from {url}."
        if details:
            message = f"{message}\n{details}"
        raise PipelineError(message) from exc

    downloaded_path = Path(result.stdout.strip())
    if not downloaded_path.is_file():
        raise PipelineError("yt-dlp finished without producing an audio file.")
    return downloaded_path


def extract_audio(media_path: Path, wav_path: Path, *, overwrite: bool) -> None:
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
            str(media_path),
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
        message = f"FFmpeg could not extract audio from {media_path}."
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
    if model_name == "mr_mt3":
        print(
            f"Preparing audio features ({audio.size / SAMPLE_RATE:.1f} seconds)...",
            flush=True,
        )
        features = model.preprocess(audio, sr=SAMPLE_RATE)
        inputs = features["inputs"]
        num_segments = len(inputs)
        if num_segments == 0:
            raise PipelineError("The audio is too short for MR-MT3 to transcribe.")
        num_batches = (
            num_segments + MR_MT3_INFERENCE_BATCH_SIZE - 1
        ) // MR_MT3_INFERENCE_BATCH_SIZE
        batch_outputs = []

        for batch_index, start in enumerate(
            range(0, num_segments, MR_MT3_INFERENCE_BATCH_SIZE), start=1
        ):
            end = min(start + MR_MT3_INFERENCE_BATCH_SIZE, num_segments)
            print(
                f"Transcribing batch {batch_index}/{num_batches} "
                f"(audio windows {start + 1}-{end}/{num_segments})...",
                flush=True,
            )
            batch_outputs.append(
                model.forward(
                    {
                        "inputs": inputs[start:end],
                        "frame_times": features["frame_times"][start:end],
                    }
                )
            )

        token_batches = [output["tokens"] for output in batch_outputs]
        max_token_count = max(tokens.shape[1] for tokens in token_batches)
        tokens = np.concatenate(
            [
                np.pad(
                    batch_tokens,
                    ((0, 0), (0, max_token_count - batch_tokens.shape[1])),
                    constant_values=-1,
                )
                for batch_tokens in token_batches
            ]
        )
        midi = model.decode(
            {
                "tokens": tokens,
                "frame_times": np.concatenate(
                    [output["frame_times"] for output in batch_outputs]
                ),
            }
        )
    else:
        midi = model.transcribe(audio, sr=SAMPLE_RATE)
    midi.save(str(midi_path))


def write_tablature(
    midi_path: Path,
    tablature_path: Path,
    *,
    position_likelihoods: Mapping[object, object] | None = None,
) -> None:
    """Convert a transcribed MIDI into conventional ASCII guitar tab."""
    try:
        notes = midi_to_tab_notes(midi_path, position_likelihoods)
    except GuitarTabError as exc:
        raise PipelineError(str(exc)) from exc
    tablature_path.write_text(render_ascii_tab(notes) + "\n", encoding="utf-8")


def _default_visual_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    return root / "pipeline" / "yolo_weights_best.pt", root / "scripts" / "hand_landmarker.task"


def _json_safe(value: Any) -> Any:
    """Convert diagnostics to finite JSON-compatible primitive values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        if hasattr(value, "item"):
            return _json_safe(value.item())
        if hasattr(value, "tolist"):
            return _json_safe(value.tolist())
    except (TypeError, ValueError):
        pass
    return str(value)


def _fallback_visual_diagnostics(
    *,
    reason: str,
    yolo_weights: Path,
    hand_model: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "status": "fallback",
        "accepted": False,
        "reason": str(reason),
        "model_paths": {
            "yolo_weights": str(yolo_weights),
            "hand_model": str(hand_model),
        },
        "config": dict(config),
        "counts": {
            "events_seen": 0,
            "events_supported": 0,
            "frames_inspected": 0,
            "frames_with_grid": 0,
            "notes_with_visual_evidence": 0,
        },
        "quality_summaries": [],
        "alignment": None,
        "per_note": [],
    }


def _write_visual_diagnostics(path: Path, diagnostics: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(diagnostics), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _visual_result_diagnostics(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_diagnostics"):
        return dict(result.to_diagnostics())
    if isinstance(result, Mapping):
        if isinstance(result.get("diagnostics"), Mapping):
            return dict(result["diagnostics"])
        return dict(result)
    return _fallback_visual_diagnostics(
        reason="visual analyzer returned no diagnostics",
        yolo_weights=Path(""),
        hand_model=Path(""),
        config={},
    )


def _visual_result_likelihoods(result: Any) -> Mapping[object, object]:
    if hasattr(result, "position_likelihoods"):
        return result.position_likelihoods
    if isinstance(result, Mapping):
        return result.get("position_likelihoods", {})
    return {}


def _likelihood_entries(value: object) -> tuple[tuple[object, object], ...]:
    if hasattr(value, "position") and hasattr(value, "likelihood"):
        return ((value.position, value.likelihood),)
    if isinstance(value, Mapping):
        keys = set(value)
        if {"string_idx", "fret_idx", "likelihood"} <= keys:
            return (((value["string_idx"], value["fret_idx"]), value["likelihood"]),)
        if {"position", "likelihood"} <= keys:
            return ((value["position"], value["likelihood"]),)
        return tuple(value.items())
    if isinstance(value, (tuple, list)):
        if len(value) == 3 and not isinstance(value[0], (tuple, list, Mapping)):
            return (((value[0], value[1]), value[2]),)
        return tuple(entry for entry in value if isinstance(entry, (tuple, list)) and len(entry) == 2)
    try:
        return tuple(entry for entry in value if isinstance(entry, (tuple, list)) and len(entry) == 2)  # type: ignore[union-attr]
    except TypeError:
        return ()


def _position_pair(value: object) -> tuple[int, int] | None:
    if hasattr(value, "string_idx") and hasattr(value, "fret_idx"):
        raw = (value.string_idx, value.fret_idx)
    elif isinstance(value, Mapping):
        if not {"string_idx", "fret_idx"} <= set(value):
            return None
        raw = (value["string_idx"], value["fret_idx"])
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        raw = (value[0], value[1])
    else:
        return None
    try:
        return int(raw[0]), int(raw[1])
    except (TypeError, ValueError, OverflowError):
        return None


def _visual_likelihood_for_position(
    likelihoods: Mapping[object, object],
    note: Any,
    string_idx: int,
    fret_idx: int,
) -> float | None:
    if not isinstance(likelihoods, Mapping):
        return None
    raw_values: object = None
    for key in (getattr(note, "identity", None), note, getattr(note, "source_idx", None)):
        if key is None:
            continue
        try:
            if key in likelihoods:
                raw_values = likelihoods[key]
                break
        except (TypeError, ValueError):
            continue
    if raw_values is None:
        return None
    for raw_position, raw_likelihood in _likelihood_entries(raw_values):
        if _position_pair(raw_position) != (string_idx, fret_idx):
            continue
        try:
            likelihood = float(
                raw_likelihood.likelihood
                if hasattr(raw_likelihood, "likelihood")
                else raw_likelihood
            )
        except (TypeError, ValueError, OverflowError):
            continue
        if np.isfinite(likelihood) and likelihood > 0.0:
            return likelihood
    return None


def _count_selected_visual_notes(
    notes: Sequence[Any],
    assigned_notes: Sequence[Any],
    likelihoods: Mapping[object, object],
) -> int:
    """Count final notes whose exact assigned positions have positive evidence."""
    selected_by_index = {
        int(note.note_id.source_idx): note
        for note in assigned_notes
        if getattr(note, "note_id", None) is not None
    }
    count = 0
    for note in notes:
        selected = selected_by_index.get(int(note.source_idx))
        if selected is None:
            continue
        if (
            _visual_likelihood_for_position(
                likelihoods,
                note,
                int(selected.string_idx),
                int(selected.fret_idx),
            )
            is not None
        ):
            count += 1
    return count


def _overwrite_visual_diagnostics(
    diagnostics: dict[str, Any],
    notes: Sequence[Any],
    assigned_notes: Sequence[Any],
    likelihoods: Mapping[object, object],
    *,
    accepted: bool,
) -> int:
    """Make per-note diagnostics reflect the assignment actually rendered."""
    per_note = [
        dict(item)
        for item in diagnostics.get("per_note", ())
        if isinstance(item, Mapping)
    ]
    by_index: dict[int, dict[str, Any]] = {}
    for item in per_note:
        try:
            source_idx = int(item["source_idx"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        by_index[source_idx] = item
    selected_by_index = {
        int(note.note_id.source_idx): note
        for note in assigned_notes
        if getattr(note, "note_id", None) is not None
    }
    visual_count = 0
    note_indices = set()
    for note in notes:
        source_idx = int(note.source_idx)
        note_indices.add(source_idx)
        item = by_index.setdefault(source_idx, {"source_idx": source_idx})
        selected = selected_by_index.get(source_idx)
        if selected is None:
            item["source"] = "audio"
            item["chosen"] = None
            continue
        likelihood = (
            _visual_likelihood_for_position(
                likelihoods,
                note,
                selected.string_idx,
                selected.fret_idx,
            )
            if accepted
            else None
        )
        chosen = {
            "string_idx": int(selected.string_idx),
            "fret_idx": int(selected.fret_idx),
        }
        if likelihood is not None:
            visual_count += 1
            chosen["likelihood"] = likelihood
        item["chosen"] = chosen
        item["source"] = "visual" if likelihood is not None else "audio"
    for source_idx, item in by_index.items():
        if source_idx not in note_indices:
            item["source"] = "audio"
            item["chosen"] = None
    diagnostics["per_note"] = [
        item for _source_idx, item in sorted(by_index.items(), key=lambda pair: pair[0])
    ]
    counts = diagnostics.get("counts")
    updated_counts = dict(counts) if isinstance(counts, Mapping) else {}
    updated_counts["notes_with_visual_evidence"] = visual_count
    diagnostics["counts"] = updated_counts
    return visual_count


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
            "Extract audio from a local video or YouTube link, transcribe it "
            "with MT3-Infer, and save WAV, MIDI, tablature.txt, and MusicXML files."
        )
    )
    parser.add_argument(
        "source",
        help="Local video file or a single YouTube video URL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: next to a local video, or <YouTube-ID>_transcription here)",
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
        help="Replace enabled audio, MIDI, tablature, MusicXML, and visual outputs",
    )
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Opt in to local-video YOLO/MediaPipe fingering evidence after audio transcription",
    )
    parser.add_argument(
        "--require-visual",
        action="store_true",
        help="Fail instead of falling back when no accepted visual assignment is produced",
    )
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        help="YOLO fretboard segmentation checkpoint (default: repository pipeline/yolo_weights_best.pt)",
    )
    parser.add_argument(
        "--hand-model",
        type=Path,
        help="MediaPipe Tasks hand model (default: repository scripts/hand_landmarker.task)",
    )
    return parser


def run(
    source: str,
    output_dir: Path,
    *,
    model_name: str,
    device: str,
    overwrite: bool,
    visual: bool = False,
    require_visual: bool = False,
    yolo_weights: Path | None = None,
    hand_model: Path | None = None,
    visual_analyzer: Any | None = None,
) -> None:
    visual_enabled = bool(visual or require_visual)
    youtube_source = is_youtube_url(source)
    # Reject before creating an output directory, downloading, extracting, or
    # invoking MT3.  The visual path is intentionally local-video only.
    if visual_enabled and youtube_source:
        raise PipelineError("--visual is supported only for local video files; YouTube remains audio-only.")
    media_path = Path(source)
    if not youtube_source and not media_path.is_file():
        raise PipelineError(f"Video file does not exist: {media_path}")
    if youtube_source and youtube_video_id(source) is None:
        raise PipelineError("Provide a single YouTube video URL, not a playlist URL.")

    default_yolo, default_hand = _default_visual_paths()
    visual_yolo = Path(yolo_weights) if yolo_weights is not None else default_yolo
    visual_hand = Path(hand_model) if hand_model is not None else default_hand

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "audio": output_dir / "audio.wav",
        "MIDI": output_dir / "transcription.mid",
        "tablature": output_dir / "tablature.txt",
        "sheet music": output_dir / "sheet.musicxml",
    }
    if visual_enabled:
        outputs.update({
            "audio baseline tablature": output_dir / "audio_tablature.txt",
            "visual diagnostics": output_dir / "visual_diagnostics.json",
        })
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise PipelineError(
            "Output file(s) already exist: "
            + ", ".join(existing)
            + ". Choose another --output-dir or pass --overwrite."
        )

    if youtube_source:
        print(f"Downloading audio from {source}...")
        with tempfile.TemporaryDirectory(prefix="guitar-cv-amt-") as temporary_dir:
            media_path = download_youtube_source(source, Path(temporary_dir))
            extract_audio(media_path, outputs["audio"], overwrite=overwrite)
    else:
        print(f"Extracting audio from {media_path}...")
        extract_audio(media_path, outputs["audio"], overwrite=overwrite)
    audio = read_model_audio(outputs["audio"])

    transcribe_audio(
        audio,
        outputs["MIDI"],
        model_name=model_name,
        device=device,
    )
    print("Converting MIDI to ASCII tablature...")
    if not visual_enabled:
        # Preserve the existing audio-only call path and output bytes exactly.
        write_tablature(outputs["MIDI"], outputs["tablature"])
    else:
        write_tablature(outputs["MIDI"], outputs["audio baseline tablature"])
        outputs["tablature"].write_bytes(outputs["audio baseline tablature"].read_bytes())

    print("Converting MIDI to MusicXML...")
    write_musicxml(outputs["MIDI"], outputs["sheet music"])

    if visual_enabled:
        diagnostics: dict[str, Any]
        accepted = False
        fallback_reason = "visual analyzer did not produce an accepted assignment"
        notes: tuple[Any, ...] = ()
        final_notes: list[Any] = []
        likelihoods: Mapping[object, object] = {}
        try:
            notes = read_midi_notes(outputs["MIDI"])
            if visual_analyzer is None:
                from pipeline import visual_fingering

                result = visual_fingering.analyze_video(
                    media_path,
                    notes,
                    yolo_weights=visual_yolo,
                    hand_model=visual_hand,
                )
            else:
                result = visual_analyzer(
                    media_path,
                    notes,
                    yolo_weights=visual_yolo,
                    hand_model=visual_hand,
                )
            if isinstance(result, Mapping):
                accepted = bool(result.get("accepted", False))
                fallback_reason = str(result.get("reason", fallback_reason))
            else:
                accepted = bool(getattr(result, "accepted", False))
                fallback_reason = str(getattr(result, "reason", fallback_reason))
            likelihoods = _visual_result_likelihoods(result)
            if accepted and likelihoods:
                final_notes = assign_tab_notes(notes, position_likelihoods=likelihoods)
                if _count_selected_visual_notes(notes, final_notes, likelihoods) > 0:
                    outputs["tablature"].write_text(
                        render_ascii_tab(final_notes) + "\n",
                        encoding="utf-8",
                    )
                else:
                    accepted = False
                    fallback_reason = (
                        "accepted visual evidence selected zero evidence-supported final positions"
                    )
                    final_notes = assign_tab_notes(notes)
                    outputs["tablature"].write_bytes(
                        outputs["audio baseline tablature"].read_bytes()
                    )
            else:
                accepted = False
                final_notes = assign_tab_notes(notes)
                outputs["tablature"].write_bytes(outputs["audio baseline tablature"].read_bytes())
            diagnostics = _visual_result_diagnostics(result)
            diagnostics.setdefault("schema_version", "1.0")
            diagnostics["status"] = "accepted" if accepted else "fallback"
            diagnostics["accepted"] = bool(accepted)
            if not accepted:
                diagnostics["reason"] = fallback_reason
            diagnostics.setdefault(
                "model_paths",
                {"yolo_weights": str(visual_yolo), "hand_model": str(visual_hand)},
            )
            diagnostics.setdefault("config", {"visual": True})
            diagnostics.setdefault(
                "counts",
                {
                    "events_seen": 0,
                    "events_supported": 0,
                    "frames_inspected": 0,
                    "frames_with_grid": 0,
                    "notes_with_visual_evidence": 0,
                },
            )
            diagnostics.setdefault("quality_summaries", [])
            diagnostics.setdefault("alignment", None)
            diagnostics.setdefault("per_note", [])
        except (VisualExtractionError, ImportError, OSError, ValueError, RuntimeError) as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"
            accepted = False
            likelihoods = {}
            try:
                notes = read_midi_notes(outputs["MIDI"])
                final_notes = midi_to_tab_notes(outputs["MIDI"])
            except (GuitarTabError, ImportError, OSError, ValueError, RuntimeError):
                notes = ()
                final_notes = []
            outputs["tablature"].write_bytes(outputs["audio baseline tablature"].read_bytes())
            diagnostics = _fallback_visual_diagnostics(
                reason=fallback_reason,
                yolo_weights=visual_yolo,
                hand_model=visual_hand,
                config={"visual": True},
            )
        if notes:
            _overwrite_visual_diagnostics(
                diagnostics,
                notes,
                final_notes,
                likelihoods,
                accepted=accepted,
            )
        _write_visual_diagnostics(outputs["visual diagnostics"], diagnostics)
        if require_visual and not accepted:
            raise PipelineError(f"--require-visual: {diagnostics.get('reason', fallback_reason)}")

    print("Wrote:")
    for label, path in outputs.items():
        print(f"  {label}: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_dir:
        output_dir = args.output_dir
    elif is_youtube_url(args.source):
        video_id = youtube_video_id(args.source)
        output_dir = Path(f"{video_id or 'youtube'}_transcription")
    else:
        source_path = Path(args.source)
        output_dir = source_path.with_name(f"{source_path.stem}_transcription")

    try:
        run(
            args.source,
            output_dir,
            model_name=args.model,
            device=args.device,
            overwrite=args.overwrite,
            visual=args.visual,
            require_visual=args.require_visual,
            yolo_weights=args.yolo_weights,
            hand_model=args.hand_model,
        )
    except (PipelineError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
