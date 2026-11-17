"""Transcribe a local guitar video with mandatory audio AND visual evidence.

This composes the methods in REPO_METHOD_INVENTORY.md. The reference TAB is
deliberately not an argument: evaluate saved outputs in a separate command.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

from main.audio import video_to_score
from main.audio.guitar_tab import (
    STANDARD_TUNING_MIDI, TabNote, assign_tab_notes, read_midi_notes, render_ascii_tab,
)
from scripts.render_av_demo import render as render_demo


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def export_transcription(source: Path, calibration: Path, output: Path, model: str) -> dict:
    """Export the actual rendered assignment, including its visual provenance.

    Reject absent visual evidence or inconsistent diagnostics rather than
    silently exporting an audio-only run as an audiovisual result.
    """
    diagnostics = json.loads((output / "visual_diagnostics.json").read_text())
    if not diagnostics.get("accepted"):
        raise video_to_score.PipelineError("No accepted visual evidence in this run")
    selected = {int(row["source_idx"]): row for row in diagnostics["per_note"]}
    score = read_midi_notes(output / "transcription.mid")
    baseline = {note.source_idx: note for note in assign_tab_notes(score)}
    notes = []
    evidence = []
    for note in score:
        row = selected.get(note.source_idx, {})
        position = row.get("chosen")
        if position is None:
            raise video_to_score.PipelineError(f"Missing final assignment for note {note.source_idx}")
        string, fret = int(position["string_idx"]), int(position["fret_idx"])
        if not (0 <= string <= 5 and 0 <= fret <= 24
                and STANDARD_TUNING_MIDI[string] + fret == note.pitch_midi):
            raise video_to_score.PipelineError("Visual assignment contradicts the audio pitch")
        notes.append(TabNote(note.onset_sec, note.offset_sec, note.pitch_midi,
                             string, fret, note.identity))
        audio_position = baseline.get(note.source_idx)
        evidence.append({
            "source_idx": note.source_idx,
            "visual_supported": row.get("source") == "visual",
            "changed_from_audio_only": audio_position is None or
                (audio_position.string_idx, audio_position.fret_idx) != (string, fret),
        })
    if not any(row["visual_supported"] for row in evidence):
        raise video_to_score.PipelineError("Zero final positions have visual support")
    if render_ascii_tab(notes) + "\n" != (output / "tablature.txt").read_text():
        raise video_to_score.PipelineError("Saved TAB and final visual assignments disagree")
    events = []
    for note in sorted(notes, key=lambda n: (n.onset_sec, n.string_idx)):
        if not events or note.onset_sec - events[-1]["onset_sec"] > 0.05 + 1e-9:
            events.append({"onset_sec": note.onset_sec, "notes": []})
        events[-1]["notes"].append({"string_idx": note.string_idx, "fret_idx": note.fret_idx})
    result = {
        "schema_version": "1.0",
        "source": fingerprint(source),
        "calibration": fingerprint(calibration),
        "methods": {
            "audio": [model, "spectral-flux onset recovery", "harmonic salience"],
            "video": ["YOLOv8 segmentation", "EFA observed fret/string grid",
                      "measured fret-wire calibration", "LK/ORB board tracking",
                      "MediaPipe HandLandmarker"],
            "fusion": "audio-constrained whole-chord unique-string dynamic programming",
        },
        "tuning_low_to_high_midi": list(STANDARD_TUNING_MIDI),
        "counts": diagnostics["counts"],
        "audio_recovered_onsets": diagnostics.get("audio_recovered_onsets", []),
        "notes": [asdict(note) for note in notes],
        "events": events,
        "evidence": evidence,
        "notes_changed_by_video": sum(row["changed_from_audio_only"] for row in evidence),
    }
    (output / "transcription.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=video_to_score.MODEL_NAMES, default="yourmt3")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not args.calibration.is_file():
            raise video_to_score.PipelineError(f"Calibration not found: {args.calibration}")
        export_path = args.output_dir / "transcription.json"
        if export_path.exists() and not args.overwrite:
            raise video_to_score.PipelineError(f"Output already exists: {export_path}")
        # Invalidate the previous success manifest before replacing outputs.
        if args.overwrite:
            export_path.unlink(missing_ok=True)
        video_to_score.run(
            str(args.source), args.output_dir, model_name=args.model,
            device=args.device, overwrite=args.overwrite, visual=True,
            require_visual=True, visual_calibration=args.calibration,
        )
        result = export_transcription(args.source, args.calibration, args.output_dir, args.model)
        tracked_frames = render_demo(
            args.source,
            args.calibration,
            args.output_dir / "transcription.json",
            args.output_dir / "visual_overlay.mp4",
        )
    except (video_to_score.PipelineError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Saved {len(result['notes'])} notes in {len(result['events'])} events; "
          f"video changed {result['notes_changed_by_video']} audio-only positions. "
          f"The demo renders stable board geometry on {tracked_frames} sampled frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
