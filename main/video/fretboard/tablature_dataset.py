"""Audit and load the local, timestamp-paired guitar tablature dataset.

Labels describe four fingers, not six sounding strings. Filename indices are
the documented label indices; CSV row order is never used as a label index.
"""

from __future__ import annotations

import csv
import json
import math
import re
import wave
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = PROJECT_ROOT / "archive/tablature_dataset/tablature_dataset"
DATASET_SOURCE = "https://www.kaggle.com/datasets/jacksonlightfoot/guitar-transcription-dataset"
FINGERS = ("index", "middle", "ring", "pinkie")
FRAME_NAME = re.compile(r"^(\d+)_(\d+)\.png$")


@dataclass(frozen=True)
class FrameRecord:
    clip_id: str
    frame_id: str
    frame_index: int
    timestamp: float | None
    frame_path: str
    audio_path: str
    label_path: str
    label_index: int
    csv_line: int
    issues: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def structurally_valid(self) -> bool:
        return not self.issues

    @property
    def benchmark_eligible(self) -> bool:
        return self.structurally_valid and not self.warnings

    def manifest_row(self) -> dict:
        result = asdict(self)
        result.update(structurally_valid=self.structurally_valid,
                      benchmark_eligible=self.benchmark_eligible,
                      alignment_status="unreviewed")
        result["issues"] = ";".join(self.issues)
        result["warnings"] = ";".join(self.warnings)
        return result


@dataclass(frozen=True)
class AudioWindow:
    samples: np.ndarray  # samples x channels, float32, original amplitude
    sample_rate: int
    start: float
    end: float
    center: float

    @property
    def mono(self) -> np.ndarray:
        return self.samples.mean(axis=1)


def resolve_dataset_root(path: str | Path = DEFAULT_DATASET) -> Path:
    path = Path(path).expanduser().resolve()
    for candidate in (path, path / "tablature_dataset",
                      path / "tablature_dataset/tablature_dataset"):
        if (candidate / "timestamps.csv").is_file():
            return candidate
    raise FileNotFoundError(f"No timestamps.csv under {path}")


def _label_problem(labels: np.ndarray) -> str | None:
    if labels.ndim != 3 or labels.shape[1:] != (4, 3):
        return "label_shape_invalid"
    if not np.issubdtype(labels.dtype, np.integer):
        return "label_dtype_invalid"
    press, fret, string = (labels[..., i] for i in range(3))
    if (not np.isin(press, [0, 1, 2]).all() or
            np.any((fret < 0) | (fret > 20)) or
            np.any((string < 0) | (string > 6))):
        return "label_values_out_of_range"
    # The publisher repeats "not pressed" for class 1. Treat 1 as the
    # apparent fretted class only for numeric consistency checks; expose
    # that interpretation explicitly, never as sounding-note truth.
    bad = ((press == 0) & ((fret != 0) | (string != 0)))
    bad |= ((press == 1) & ((fret == 0) | (string == 0)))
    bad |= ((press == 2) & ((fret == 0) | (string != 0)))
    bad[:, 1:] |= press[:, 1:] == 2
    return "label_fields_inconsistent" if bad.any() else None


class TablatureDataset:
    """Read-only dataset with a conservative, reproducible audit.

    Count-mismatched and entirely-zero clips remain inspectable but are
    excluded from the default benchmark. Eligibility is structural, not a
    claim that audio/video synchronization has been manually verified.
    """

    def __init__(self, root: str | Path = DEFAULT_DATASET, *, verify_images: bool = True):
        self.root = resolve_dataset_root(root)
        self.labels: dict[str, np.ndarray] = {}
        self.audio_info: dict[str, dict] = {}
        self.records: list[FrameRecord] = []
        self._audit(verify_images)

    def _audit(self, verify_images: bool) -> None:
        with (self.root / "timestamps.csv").open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not {"frame", "audio", "timestamp"} <= set(reader.fieldnames or []):
                raise ValueError("timestamps.csv must contain frame,audio,timestamp columns")
            rows = list(reader)
        counts = Counter(row["frame"] for row in rows)
        by_clip = defaultdict(list)
        for line, row in enumerate(rows, start=2):
            audio_name = row["audio"] or ""
            clip_id = Path(audio_name).stem
            by_clip[clip_id].append((line, row))

        label_ids = {p.stem for p in (self.root / "tablature_labels").glob("*.npy")}
        audio_ids = {p.stem for p in (self.root / "tablature_audio").glob("*.wav")}
        clip_reports = []
        for clip_id in sorted(set(by_clip) | label_ids | audio_ids,
                              key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else s)):
            entries = by_clip[clip_id]
            clip_issues, clip_warnings = [], []
            # Only numeric basenames are accepted; paths in the CSV cannot
            # escape the dataset or silently reference another recording.
            safe_clip = clip_id.isdigit()
            if not safe_clip:
                clip_issues.append("clip_id_invalid")
            label_path = self.root / "tablature_labels" / f"{clip_id}.npy"
            labels = None
            if safe_clip:
                try:
                    labels = np.load(label_path, allow_pickle=False)
                    problem = _label_problem(labels)
                    if problem:
                        clip_issues.append(problem)
                    else:
                        self.labels[clip_id] = labels
                        if not labels.any():
                            clip_warnings.append("all_zero_clip_labels")
                except (OSError, ValueError, EOFError):
                    clip_issues.append("labels_missing_or_unreadable")
            if labels is not None and labels.ndim > 0 and len(labels) != len(entries):
                clip_warnings.append("timestamp_label_count_mismatch")
            if not entries:
                clip_warnings.append("no_timestamp_rows")
            audio_path = self.root / "tablature_audio" / f"{clip_id}.wav"
            info = None
            if safe_clip:
                try:
                    with wave.open(str(audio_path), "rb") as handle:
                        info = {"sample_rate": handle.getframerate(),
                                "channels": handle.getnchannels(),
                                "sample_width": handle.getsampwidth(),
                                "duration": handle.getnframes() / handle.getframerate()}
                        if info["sample_width"] not in (1, 2, 3, 4) or info["duration"] <= 0:
                            raise ValueError("Unsupported or empty PCM audio")
                    self.audio_info[clip_id] = info
                except (OSError, ValueError, EOFError, wave.Error, ZeroDivisionError):
                    clip_issues.append("audio_missing_or_unreadable")

            finite_times = []
            for _, row in entries:
                try:
                    t = float(row["timestamp"])
                    if math.isfinite(t):
                        finite_times.append(t)
                except (TypeError, ValueError):
                    pass
            if any(b <= a for a, b in zip(finite_times, finite_times[1:])):
                clip_warnings.append("timestamps_not_strictly_increasing")

            available = 0
            for line, row in entries:
                issues = list(clip_issues)
                match = FRAME_NAME.fullmatch(row["frame"] or "")
                index = int(match[2]) if match else -1
                if not match or match[1] != clip_id:
                    issues.append("frame_clip_mismatch_or_invalid_name")
                if row["audio"] != f"{clip_id}.wav":
                    issues.append("audio_name_invalid")
                if counts[row["frame"]] > 1:
                    issues.append("duplicate_frame_reference")
                try:
                    timestamp = float(row["timestamp"])
                    if not math.isfinite(timestamp):
                        timestamp = None
                        issues.append("timestamp_invalid")
                    elif timestamp < 0 or (info is not None and timestamp >= info["duration"]):
                        issues.append("timestamp_outside_audio")
                except (TypeError, ValueError):
                    timestamp = None
                    issues.append("timestamp_invalid")
                frame_path = f"tablature_frames/{row['frame']}" if match else ""
                if not frame_path or not (self.root / frame_path).is_file():
                    issues.append("frame_missing")
                else:
                    available += 1
                    if verify_images:
                        try:
                            with Image.open(self.root / frame_path) as image:
                                image.verify()
                        except (OSError, ValueError, SyntaxError):
                            issues.append("frame_unreadable")
                if labels is None or labels.ndim == 0 or not 0 <= index < len(labels):
                    issues.append("label_index_missing")
                self.records.append(FrameRecord(
                    clip_id, row["frame"] or "", index, timestamp, frame_path,
                    f"tablature_audio/{clip_id}.wav" if safe_clip else "",
                    f"tablature_labels/{clip_id}.npy" if safe_clip else "",
                    index, line, tuple(sorted(set(issues))), tuple(sorted(set(clip_warnings)))))
            clip_reports.append({"clip_id": clip_id, "timestamp_rows": len(entries),
                                 "available_frames": available,
                                 "label_shape": list(labels.shape) if labels is not None else None,
                                 "audio": info, "issues": clip_issues,
                                 "warnings": clip_warnings})

        referenced = {record.frame_id for record in self.records}
        frame_names = {p.name for p in (self.root / "tablature_frames").glob("*.png")}
        self.audit = {
            "schema_version": 1, "source": DATASET_SOURCE,
            "label_schema": {"axes": ["frame_index", "finger", "field"],
                             "fingers": list(FINGERS),
                             "fields": ["press_class", "fret", "string"],
                             "press_class_1": "apparently pressed; publisher description has a typo",
                             "string_numbering": "orientation not specified by publisher",
                             "sounding_note_ground_truth": False},
            "timestamp_rows": len(self.records), "frame_files": len(frame_names),
            "audio_files": len(audio_ids), "label_files": len(label_ids),
            "structurally_valid_rows": sum(r.structurally_valid for r in self.records),
            "benchmark_eligible_rows": sum(r.benchmark_eligible for r in self.records),
            "issue_counts": dict(Counter(i for r in self.records for i in r.issues)),
            "warning_counts": dict(Counter(w for r in self.records for w in r.warnings)),
            "unreferenced_frames": sorted(frame_names - referenced), "clips": clip_reports,
            "alignment_status": "unreviewed; timestamps are supplied pairings, not verified synchronization",
        }

    def get_record(self, clip_id: str, frame_index: int) -> FrameRecord:
        matches = [r for r in self.records if r.clip_id == str(clip_id) and r.frame_index == frame_index]
        if len(matches) != 1:
            raise ValueError(f"Expected one record for {clip_id}_{frame_index}, got {len(matches)}")
        return matches[0]

    def raw_label(self, record: FrameRecord) -> np.ndarray:
        labels = self.labels.get(record.clip_id)
        if labels is None or not 0 <= record.label_index < len(labels):
            raise ValueError(f"No usable label for {record.frame_id}: {record.issues}")
        return labels[record.label_index].copy()

    def decoded_label(self, record: FrameRecord) -> list[dict]:
        descriptions = {0: "not pressed", 1: "apparently pressed (publisher typo)",
                        2: "barre; covered strings unspecified"}
        return [{"finger": finger, "press_class": int(row[0]),
                 "press_interpretation": descriptions[int(row[0])],
                 "fret": int(row[1]), "string": int(row[2])}
                for finger, row in zip(FINGERS, self.raw_label(record))]

    def audio_window(self, record: FrameRecord, *, before: float = 0.5,
                     after: float = 0.5, offset: float = 0.0) -> AudioWindow:
        """Positive offset selects audio later than the CSV timestamp.

        Windows are clipped to recording bounds, never padded or wrapped.
        Retain stereo and absolute WAV seconds; mono is a separate property.
        """
        if record.timestamp is None or record.clip_id not in self.audio_info:
            raise ValueError("Record has no usable audio timestamp or WAV")
        if not all(math.isfinite(x) for x in (before, after, offset)) or before < 0 or after < 0:
            raise ValueError("Window sizes and offset must be finite; sizes must be nonnegative")
        if before + after <= 0:
            raise ValueError("Window duration must be positive")
        info = self.audio_info[record.clip_id]
        center = record.timestamp + offset
        if not 0 <= center < info["duration"]:
            raise ValueError("Adjusted timestamp is outside the audio recording")
        with wave.open(str(self.root / record.audio_path), "rb") as handle:
            rate, channels, width = handle.getframerate(), handle.getnchannels(), handle.getsampwidth()
            start = max(0, math.floor((center - before) * rate))
            end = min(handle.getnframes(), math.ceil((center + after) * rate))
            handle.setpos(start)
            payload = handle.readframes(end - start)
        if width == 1:
            values = (np.frombuffer(payload, np.uint8).astype(np.float32) - 128) / 128
        elif width in (2, 4):
            values = np.frombuffer(payload, dtype=f"<i{width}").astype(np.float32) / (2 ** (8 * width - 1))
        else:
            raw = np.frombuffer(payload, np.uint8).reshape(-1, 3).astype(np.int32)
            values = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
            values = ((values ^ 0x800000) - 0x800000).astype(np.float32) / 8388608
        samples = values.reshape(-1, channels)
        if len(samples) != end - start:
            raise ValueError("Truncated WAV data")
        return AudioWindow(samples, rate, start / rate, end / rate, center)

    def select_examples(self, clips=("0", "1", "2"), per_clip: int = 6) -> list[FrameRecord]:
        if per_clip < 1:
            raise ValueError("per_clip must be positive")
        examples = []
        for clip in clips:
            candidates = sorted((r for r in self.records if r.clip_id == str(clip) and r.benchmark_eligible),
                                key=lambda r: r.frame_index)
            if not candidates:
                raise ValueError(f"No benchmark-eligible rows for requested clip {clip}")
            indices = np.linspace(0, len(candidates) - 1, min(per_clip, len(candidates)), dtype=int)
            examples.extend(candidates[i] for i in indices)
        return examples

    def export(self, destination: str | Path, examples: list[FrameRecord]) -> None:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "audit.json").write_text(json.dumps(self.audit, indent=2) + "\n")
        for name, records in (("manifest_all.csv", self.records),
                              ("manifest_valid.csv", [r for r in self.records if r.benchmark_eligible]),
                              ("examples.csv", examples)):
            fields = list(FrameRecord.__dataclass_fields__) + ["structurally_valid", "benchmark_eligible", "alignment_status"]
            with (destination / name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(r.manifest_row() for r in records)
