"""Compare a short-clip pipeline run with the TAB printed in the source video.

This file is an evaluation tool only. The ground-truth JSON is never an input
to transcription, visual extraction, or position assignment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from main.audio.guitar_tab import STANDARD_TUNING_MIDI, assign_tab_notes, read_midi_notes


ROOT = Path(__file__).resolve().parents[1]


def evaluate(
    output_dir: Path,
    truth_path: Path,
    *,
    tolerance: float = 0.16,
    ignore_after_truth: bool = False,
) -> dict:
    truth = json.loads(truth_path.read_text())
    midi_notes = read_midi_notes(output_dir / "transcription.mid")
    baseline = {n.source_idx: n for n in assign_tab_notes(midi_notes)}
    diagnostics_path = output_dir / "visual_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text()) if diagnostics_path.is_file() else {}
    final = {
        int(item["source_idx"]): item["chosen"]
        for item in diagnostics.get("per_note", ())
        if item.get("chosen") is not None
    }
    used: set[int] = set()
    rows = []
    for event in truth["events"]:
        for string_number, fret in event["notes"]:
            expected = (6 - string_number, fret)
            pitch = STANDARD_TUNING_MIDI[expected[0]] + fret
            candidates = [
                note for note in midi_notes
                if note.source_idx not in used
                and note.pitch_midi == pitch
                and abs(note.onset_sec - event["time"]) <= tolerance
            ]
            note = min(candidates, key=lambda n: abs(n.onset_sec - event["time"])) if candidates else None
            if note is not None:
                used.add(note.source_idx)
            base = baseline.get(note.source_idx) if note is not None else None
            chosen = final.get(note.source_idx) if note is not None else None
            actual = (int(chosen["string_idx"]), int(chosen["fret_idx"])) if chosen else (
                (base.string_idx, base.fret_idx) if base else None
            )
            rows.append({
                "time": event["time"], "measure": event["measure"], "slot": event["slot"],
                "pitch": pitch, "expected": expected, "source_idx": note.source_idx if note else None,
                "onset": round(note.onset_sec, 4) if note else None,
                "audio_only": (base.string_idx, base.fret_idx) if base else None,
                "actual": actual, "match": actual == expected,
            })
    extras = [
        {"source_idx": n.source_idx, "onset": round(n.onset_sec, 4), "pitch": n.pitch_midi}
        for n in midi_notes
        if n.source_idx not in used
        and (not ignore_after_truth or n.onset_sec <= truth["duration_seconds"])
    ]
    return {
        "expected_notes": len(rows),
        "matched_positions": sum(row["match"] for row in rows),
        "missing_audio_notes": sum(row["source_idx"] is None for row in rows),
        "extra_audio_notes": extras,
        "errors": [row for row in rows if not row["match"]],
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/stairway-intro-short")
    parser.add_argument("--truth", type=Path, default=ROOT / "research/av_stairway_ground_truth.json")
    parser.add_argument("--tolerance", type=float, default=0.16)
    parser.add_argument(
        "--ignore-after-truth", action="store_true",
        help="For a longer source video, evaluate only its first short-clip window",
    )
    args = parser.parse_args()
    result = evaluate(
        args.output_dir, args.truth, tolerance=args.tolerance,
        ignore_after_truth=args.ignore_after_truth,
    )
    print(f"{result['matched_positions']}/{result['expected_notes']} exact positions; "
          f"{result['missing_audio_notes']} missing pitches; "
          f"{len(result['extra_audio_notes'])} extra pitches")
    for row in result["errors"]:
        print(f"{row['time']:6.3f}s m{row['measure']}:{row['slot']} pitch={row['pitch']} "
              f"expected={row['expected']} got={row['actual']} "
              f"source={row['source_idx']}")
    if result["extra_audio_notes"]:
        print("extra MIDI notes:", result["extra_audio_notes"])
    raise SystemExit(0 if result["matched_positions"] == result["expected_notes"]
                     and not result["extra_audio_notes"] else 1)


if __name__ == "__main__":
    main()
