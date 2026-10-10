#!/usr/bin/env python3
"""Evaluate MR-MT3 framewise active-pitch F1 on a fixed GuitarSet subset."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pretty_midi
from mir_eval import multipitch

from pipeline.video_to_score import extract_audio, read_model_audio, transcribe_audio


DATASET = "jhartquist/guitarset"
ROWS_API = "https://datasets-server.huggingface.co/rows"
ROW_RANGES = ((0, 2), (16, 2), (60, 2), (76, 2))
FRAME_RATE = 100.0
PITCH_WINDOW_SEMITONES = 0.5
CACHE_DIR = Path(tempfile.gettempdir()) / "guitar-cv-amt-mt3-benchmark"
REPORT_MD = Path("research/mt3_guitarset_framewise_f1.md")
REPORT_JSON = Path("research/mt3_guitarset_framewise_f1.json")


def fetch_rows(offset: int, length: int) -> tuple[list[dict[str, Any]], str | None]:
    query = urlencode(
        {
            "dataset": DATASET,
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": length,
        }
    )
    request = Request(
        f"{ROWS_API}?{query}",
        headers={"User-Agent": "guitar-cv-amt-mt3-framewise/1.0"},
    )
    with urlopen(request, timeout=90) as response:
        data = json.load(response)
        revision = response.headers.get("x-revision")
    rows = [item["row"] for item in data["rows"]]
    if len(rows) != length:
        raise RuntimeError(f"Expected {length} rows at offset {offset}; received {len(rows)}.")
    return rows, revision


def fetch_subset(limit: int) -> tuple[list[dict[str, Any]], str | None]:
    selected = []
    revisions = set()
    for offset, length in ROW_RANGES[: (limit + 1) // 2]:
        rows, revision = fetch_rows(offset, length)
        selected.extend(rows)
        if revision:
            revisions.add(revision)
    if len(revisions) > 1:
        raise RuntimeError(f"Dataset changed during download: {sorted(revisions)}")
    return selected[:limit], next(iter(revisions), None)


def audio_source(row: dict[str, Any]) -> str:
    field = row["audio_mono_mic"]
    if isinstance(field, list):
        field = field[0]
    return field["src"]


def fetch_audio(row: dict[str, Any], directory: Path) -> Path:
    target = directory / f"{row['track_id']}.wav"
    if target.is_file() and target.stat().st_size > 44:
        return target
    request = Request(audio_source(row), headers={"User-Agent": "guitar-cv-amt-framewise/1.0"})
    partial = target.with_suffix(".wav.part")
    with urlopen(request, timeout=180) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output)
    partial.replace(target)
    return target


def reference_notes(row: dict[str, Any]) -> list[tuple[float, float, float]]:
    result = []
    for note in row["notes"]:
        onset, offset, pitch = (
            float(note["onset_s"]),
            float(note["offset_s"]),
            float(note["midi"]),
        )
        if offset > onset and np.isfinite([onset, offset, pitch]).all():
            result.append((onset, offset, pitch))
    return result


def predicted_notes(path: Path) -> list[tuple[float, float, float]]:
    midi = pretty_midi.PrettyMIDI(str(path))
    return [
        (float(note.start), float(note.end), float(note.pitch))
        for instrument in midi.instruments
        for note in instrument.notes
        if note.end > note.start
    ]


def pitch_frames(
    notes: list[tuple[float, float, float]], duration_sec: float
) -> list[np.ndarray]:
    frame_count = max(1, int(math.ceil(duration_sec * FRAME_RATE)))
    centers = (np.arange(frame_count, dtype=float) + 0.5) / FRAME_RATE
    return [
        np.asarray(
            sorted(
                {
                    pitch
                    for onset, offset, pitch in notes
                    if onset <= time_sec < offset
                }
            ),
            dtype=float,
        )
        for time_sec in centers
    ]


def framewise_f1(
    reference_frames: list[np.ndarray], estimate_frames: list[np.ndarray]
) -> dict[str, float]:
    true_positives = multipitch.compute_num_true_positives(
        reference_frames,
        estimate_frames,
        window=PITCH_WINDOW_SEMITONES,
    )
    reference_counts = multipitch.compute_num_freqs(reference_frames)
    estimate_counts = multipitch.compute_num_freqs(estimate_frames)
    precision, recall, _ = multipitch.compute_accuracy(
        true_positives, reference_counts, estimate_counts
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def render_report(report: dict[str, Any], markdown: Path, json_path: Path) -> None:
    markdown.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    score = report["score"]
    lines = [
        "# GuitarSet framewise active-pitch result",
        "",
        f"Run date: {report['run_date']}",
        "",
        f"MR-MT3 scored **{100 * score['f1']:.1f}% framewise active-pitch F1** on this fixed {report['clip_count']}-clip GuitarSet subset.",
        "",
        f"Precision: {100 * score['precision']:.1f}%. Recall: {100 * score['recall']:.1f}%.",
        "",
        "The score compares active MIDI pitches at 10 ms frame centers. A pitch matches within 0.5 semitones; F1 is the harmonic mean of framewise precision and recall. This measures pitch activity overlap rather than note-by-note onset or offset accuracy.",
        "",
        f"The subset contains {report['clip_count']} acoustic microphone recordings from {report['player_count']} players ({report['style_counts']['comp']} comp and {report['style_counts']['solo']} solo), totaling {report['audio_seconds']:.1f} seconds. The audio was converted by this repository's FFmpeg extraction path and transcribed by its MR-MT3 path.",
        "",
        "| Track | Player | Style | Framewise active-pitch F1 |",
        "|---|---:|---|---:|",
    ]
    for clip in report["clips"]:
        lines.append(
            f"| `{clip['track_id']}` | {clip['player']} | {clip['style']} | "
            f"{100 * clip['score']['f1']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "This is a small acoustic-guitar benchmark and should be described with its metric and subset. GuitarSet reference note pitch and offset labels include automatic estimates. The [GuitarSet dataset](https://doi.org/10.5281/zenodo.1422265) is accessed through its [Hugging Face mirror](https://huggingface.co/datasets/jhartquist/guitarset). The framewise pitch metric follows `mir_eval`'s multipitch matching procedure.",
            "",
            "Reproduce with:",
            "",
            "```bash",
            "uv run --python 3.10 --extra transcription -- python -m scripts.evaluate_mt3_guitarset_framewise",
            "```",
            "",
            "The selected audio and MIDI files are cached in the system temporary directory and are not stored in the repository.",
            "",
        ]
    )
    markdown.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, choices=(2, 4, 6, 8), default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reuse-midi", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--markdown", type=Path, default=REPORT_MD)
    parser.add_argument("--json", type=Path, default=REPORT_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audio_dir = args.cache_dir / "audio"
    midi_dir = args.cache_dir / "midi"
    wav_dir = args.cache_dir / "wav_16k"
    for directory in (audio_dir, midi_dir, wav_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print("Fetching the fixed GuitarSet subset...")
    rows, dataset_revision = fetch_subset(args.limit)
    clips = []
    pooled_reference: list[np.ndarray] = []
    pooled_estimate: list[np.ndarray] = []
    for index, row in enumerate(rows, start=1):
        track_id = row["track_id"]
        midi_path = midi_dir / f"{track_id}.mid"
        action = "reusing cached MIDI" if args.reuse_midi else "transcribing"
        print(f"[{index}/{len(rows)}] {track_id}: {action}")
        if args.reuse_midi:
            if not midi_path.is_file():
                raise SystemExit(f"Cached MIDI is missing: {midi_path}")
        else:
            input_audio = fetch_audio(row, audio_dir)
            model_audio = wav_dir / f"{track_id}.wav"
            extract_audio(input_audio, model_audio, overwrite=True)
            transcribe_audio(
                read_model_audio(model_audio),
                midi_path,
                model_name="mr_mt3",
                device=args.device,
            )

        duration = float(row["duration_s"])
        ref_frames = pitch_frames(reference_notes(row), duration)
        est_frames = pitch_frames(predicted_notes(midi_path), duration)
        clips.append(
            {
                "track_id": track_id,
                "player": int(row["player"]),
                "style": row["style"],
                "duration_seconds": duration,
                "score": framewise_f1(ref_frames, est_frames),
            }
        )
        pooled_reference.extend(ref_frames)
        pooled_estimate.extend(est_frames)

    report = {
        "run_date": datetime.now(timezone.utc).date().isoformat(),
        "dataset": DATASET,
        "dataset_revision": dataset_revision,
        "model": "mr_mt3",
        "metric": "framewise active-pitch F1",
        "frame_rate": FRAME_RATE,
        "pitch_window_semitones": PITCH_WINDOW_SEMITONES,
        "clip_count": len(clips),
        "player_count": len({clip["player"] for clip in clips}),
        "style_counts": {
            style: sum(clip["style"] == style for clip in clips)
            for style in ("comp", "solo")
        },
        "audio_seconds": sum(clip["duration_seconds"] for clip in clips),
        "score": framewise_f1(pooled_reference, pooled_estimate),
        "clips": clips,
    }
    render_report(report, args.markdown, args.json)
    print(f"Framewise active-pitch F1: {100 * report['score']['f1']:.1f}%")
    print(f"Wrote {args.markdown} and {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
