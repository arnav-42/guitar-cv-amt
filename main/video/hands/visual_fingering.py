"""Audio-constrained visual string-row and calibrated fret-region evidence.

The audio MIDI remains authoritative. The module extracts continuous string
rows and, with a nut/numbered-fret calibration, absolute fret regions. It
converts contacts into likelihoods for exact standard-tuning positions.
Neural imports are kept inside :func:`extract_visual_events` so the audio-only
pipeline has no visual startup cost.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from main.audio.guitar_tab import (
    MAX_FRET,
    ONSET_GROUP_TOLERANCE_SEC,
    PlayablePosition,
    ScoreNote,
    ScoreNoteId,
    playable_positions,
    read_midi_notes,
    STANDARD_TUNING_MIDI,
)


SCHEMA_VERSION = "1.0"
MIN_ALIGNMENT_SUPPORT_EVENTS = 1
MIN_ALIGNMENT_SUPPORT_RATIO = 0.5
MIN_ALIGNMENT_MARGIN = 0.08
MIN_CONTACT_SCORE = 0.45
MIN_FRET_WIRE_MARGIN = 0.12
DEFAULT_FRAME_DELTAS = (-2, -1, 0, 1, 2)
MEDIAPIPE_CONFIDENCE_THRESHOLDS = {
    "min_hand_detection_confidence": 0.05,
    "min_hand_presence_confidence": 0.05,
    "min_tracking_confidence": 0.05,
}


class VisualExtractionError(RuntimeError):
    """An actionable local-video/model/landmark extraction error."""


@dataclass(frozen=True)
class StringRowContact:
    """One fretting-finger contact in continuous atlas string-row space.

    ``row`` is a visual string-row coordinate from 0 (atlas top string) to 5
    (atlas bottom string). ``fret_region`` is optional and comes from a tracked
    nut/numbered-fret calibration, never from the atlas alone.
    """

    row: float
    confidence: float = 1.0
    finger: str | None = None
    time_sec: float = 0.0
    frame_index: int | None = None
    source_x: float | None = None
    fret_region: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "row", float(self.row))
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "time_sec", float(self.time_sec))
        if self.frame_index is not None:
            object.__setattr__(self, "frame_index", int(self.frame_index))
        if self.source_x is not None:
            try:
                source_x = float(self.source_x)
            except (TypeError, ValueError, OverflowError):
                source_x = None
            if source_x is None or not math.isfinite(source_x):
                source_x = None
            object.__setattr__(self, "source_x", source_x)
        if self.fret_region is not None:
            object.__setattr__(self, "fret_region", int(self.fret_region))

 


@dataclass(frozen=True)
class VisualEvent:
    """Aggregated string-row contacts around one requested audio onset."""

    onset_sec: float
    contacts: tuple[StringRowContact, ...]
    frame_count: int = 0
    quality: str = "grid"

    def __post_init__(self) -> None:
        object.__setattr__(self, "onset_sec", float(self.onset_sec))
        try:
            contacts = tuple(_coerce_contact(contact) for contact in self.contacts)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("visual event contains an invalid contact") from exc
        object.__setattr__(self, "contacts", contacts)
        object.__setattr__(self, "frame_count", int(self.frame_count))


@dataclass(frozen=True)
class VisualAlignment:
    """The winning visual-top-to-standard-string row-order alignment."""

    string_order: tuple[int, ...]
    score: float
    runner_up_score: float
    support_events: int
    support_ratio: float
    margin: float
    mode: str = "string_row"

    def __post_init__(self) -> None:
        object.__setattr__(self, "string_order", tuple(int(v) for v in self.string_order))
        for field in ("score", "runner_up_score", "support_ratio", "margin"):
            object.__setattr__(self, field, float(getattr(self, field)))
        object.__setattr__(self, "support_events", int(self.support_events))
        object.__setattr__(self, "mode", str(self.mode))





@dataclass(frozen=True)
class ExtractedVisual:
    """Model extraction output and observed quality summaries."""

    events: tuple[VisualEvent, ...]
    frames_inspected: int
    frames_with_grid: int
    quality_summaries: tuple[Mapping[str, Any], ...]
    model_paths: Mapping[str, str]
    config: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "frames_inspected", int(self.frames_inspected))
        object.__setattr__(self, "frames_with_grid", int(self.frames_with_grid))
        object.__setattr__(self, "quality_summaries", tuple(dict(v) for v in self.quality_summaries))
        object.__setattr__(self, "model_paths", MappingProxyType(dict(self.model_paths)))
        object.__setattr__(self, "config", MappingProxyType(dict(self.config)))


@dataclass(frozen=True)
class VisualInferenceResult:
    """Pure alignment result and exact-position likelihoods."""

    accepted: bool
    reason: str
    alignment: VisualAlignment | None
    position_likelihoods: Mapping[ScoreNoteId, Mapping[PlayablePosition, float]]
    events_seen: int = 0
    events_supported: int = 0
    quality_summaries: tuple[Mapping[str, Any], ...] = ()
    frames_inspected: int = 0
    frames_with_grid: int = 0
    model_paths: Mapping[str, str] = MappingProxyType({})
    config: Mapping[str, Any] = MappingProxyType({})
    note_sources: Mapping[ScoreNoteId, Mapping[str, Any]] = MappingProxyType({})

    def __post_init__(self) -> None:
        likelihoods: dict[ScoreNoteId, Mapping[PlayablePosition, float]] = {}
        for raw_id, raw_values in dict(self.position_likelihoods).items():
            note_id = raw_id if isinstance(raw_id, ScoreNoteId) else ScoreNoteId(int(raw_id))
            values: dict[PlayablePosition, float] = {}
            try:
                raw_items = tuple(dict(raw_values).items())
            except (TypeError, ValueError):
                continue
            for raw_position, raw_value in raw_items:
                try:
                    position = (
                        raw_position
                        if isinstance(raw_position, PlayablePosition)
                        else PlayablePosition(*raw_position)
                    )
                    value = float(raw_value)
                except (TypeError, ValueError, OverflowError):
                    continue
                if (
                    0 <= position.string_idx < 6
                    and 0 < position.fret_idx <= MAX_FRET
                    and math.isfinite(value)
                    and value > 0.0
                ):
                    values[position] = min(value, 1.0)
            if values:
                likelihoods[note_id] = MappingProxyType(values)
        object.__setattr__(self, "position_likelihoods", MappingProxyType(likelihoods))
        object.__setattr__(self, "events_seen", int(self.events_seen))
        object.__setattr__(self, "events_supported", int(self.events_supported))
        object.__setattr__(self, "quality_summaries", tuple(dict(v) for v in self.quality_summaries))
        object.__setattr__(self, "frames_inspected", int(self.frames_inspected))
        object.__setattr__(self, "frames_with_grid", int(self.frames_with_grid))
        object.__setattr__(self, "model_paths", MappingProxyType(dict(self.model_paths)))
        object.__setattr__(self, "config", MappingProxyType(dict(self.config)))
        object.__setattr__(
            self,
            "note_sources",
            MappingProxyType({
                (key if isinstance(key, ScoreNoteId) else ScoreNoteId(int(key))): MappingProxyType(dict(value))
                for key, value in dict(self.note_sources).items()
            }),
        )

    @property
    def status(self) -> str:
        return "accepted" if self.accepted else "fallback"


    def to_diagnostics(self) -> dict[str, Any]:
        alignment = self.alignment
        alignment_data = None
        if alignment is not None:
            alignment_data = {
                "mode": alignment.mode,
                "string_order": list(alignment.string_order),
                "score": alignment.score,
                "runner_up_score": alignment.runner_up_score,
                "support_events": alignment.support_events,
                "support_ratio": alignment.support_ratio,
                "margin": alignment.margin,
            }
        per_note = []
        note_ids = sorted(
            set(self.note_sources) | set(self.position_likelihoods),
            key=lambda value: value.source_idx,
        )
        for note_id in note_ids:
            source = dict(
                self.note_sources.get(
                    note_id,
                    {"source": "visual" if note_id in self.position_likelihoods else "audio"},
                )
            )
            positions = self.position_likelihoods.get(note_id, {})
            chosen = None
            if positions:
                position = max(
                    positions,
                    key=lambda value: (positions[value], -value.string_idx, -value.fret_idx),
                )
                chosen = {
                    "string_idx": position.string_idx,
                    "fret_idx": position.fret_idx,
                    "likelihood": positions[position],
                }
            per_note.append({"source_idx": note_id.source_idx, **source, "chosen": chosen})
        return {
            "schema_version": SCHEMA_VERSION,
            "mode": alignment.mode if alignment is not None else self.config.get("mode", "string_row"),
            "status": self.status,
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "model_paths": dict(self.model_paths),
            "config": dict(self.config),
            "counts": {
                "events_seen": self.events_seen,
                "events_supported": self.events_supported,
                "frames_inspected": self.frames_inspected,
                "frames_with_grid": self.frames_with_grid,
                "notes_with_visual_evidence": len(self.position_likelihoods),
            },
            "quality_summaries": [dict(value) for value in self.quality_summaries],
            "alignment": alignment_data,
            "per_note": per_note,
        }




def _coerce_notes(notes: Sequence[ScoreNote]) -> tuple[ScoreNote, ...]:
    output: list[ScoreNote] = []
    for index, note in enumerate(notes):
        if isinstance(note, ScoreNote):
            output.append(note)
            continue
        try:
            output.append(ScoreNote(note.onset_sec, note.offset_sec, note.pitch_midi, getattr(note, "source_idx", index)))
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("visual alignment received an invalid score note") from exc
    return tuple(output)


def _coerce_contact(contact: StringRowContact | Mapping[str, Any] | Sequence[Any]) -> StringRowContact:
    if isinstance(contact, StringRowContact):
        return contact
    if isinstance(contact, (tuple, list)):
        if not contact:
            raise ValueError("visual event contains an invalid contact")
        try:
            values = list(contact)
            return StringRowContact(
                float(values[0]),
                float(values[1]) if len(values) > 1 else 1.0,
                values[2] if len(values) > 2 else None,
                float(values[3]) if len(values) > 3 else 0.0,
                values[4] if len(values) > 4 else None,
                values[5] if len(values) > 5 else None,
                values[6] if len(values) > 6 else None,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("visual event contains an invalid contact") from exc
    try:
        if not isinstance(contact, Mapping):
            raise TypeError
        timestamp = contact.get("time_sec", contact.get("timestamp_sec", 0.0))
        frame_index = contact.get("frame_index", contact.get("frame"))
        return StringRowContact(
            float(contact["row"]),
            float(contact.get("confidence", 1.0)),
            contact.get("finger"),
            float(timestamp),
            frame_index,
            contact.get("source_x"),
            contact.get("fret_region"),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("visual event contains an invalid contact") from exc


def _coerce_event(event: VisualEvent | Mapping[str, Any] | Sequence[Any]) -> VisualEvent:
    if isinstance(event, VisualEvent):
        return event
    if isinstance(event, (tuple, list)) and len(event) >= 2:
        try:
            return VisualEvent(float(event[0]), tuple(_coerce_contact(contact) for contact in event[1]))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("visual alignment received an invalid visual event") from exc
    try:
        contacts = tuple(_coerce_contact(contact) for contact in event.get("contacts", ()))
        return VisualEvent(float(event["onset_sec"]), contacts, int(event.get("frame_count", 0)), str(event.get("quality", "grid")))
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("visual alignment received an invalid visual event") from exc


def _group_notes(notes: Sequence[ScoreNote]) -> list[tuple[ScoreNote, ...]]:
    ordered = sorted(notes, key=lambda note: (note.onset_sec, note.pitch_midi, note.offset_sec, note.source_idx))
    groups: list[list[ScoreNote]] = []
    start: float | None = None
    for note in ordered:
        if start is None or note.onset_sec - start > ONSET_GROUP_TOLERANCE_SEC + 1e-9:
            groups.append([])
            start = note.onset_sec
        groups[-1].append(note)
    return [tuple(group) for group in groups]


def _nearest_events(groups: Sequence[tuple[ScoreNote, ...]], events: Sequence[VisualEvent]) -> list[tuple[tuple[ScoreNote, ...], VisualEvent] | None]:
    ordered = sorted(events, key=lambda event: event.onset_sec)
    matched: list[tuple[tuple[ScoreNote, ...], VisualEvent] | None] = []
    for group in groups:
        if not group or not ordered:
            matched.append(None)
            continue
        onset = group[0].onset_sec
        event = min(ordered, key=lambda candidate: abs(candidate.onset_sec - onset))
        if abs(event.onset_sec - onset) > ONSET_GROUP_TOLERANCE_SEC + 1e-9:
            matched.append(None)
        else:
            matched.append((group, event))
    return matched


def _contact_score(contact: StringRowContact, row: float) -> float:
    """Return confidence-weighted visual support for one standard string row."""
    try:
        row_delta = abs(float(contact.row) - float(row))
        confidence = float(contact.confidence)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return 0.0
    if not (math.isfinite(row_delta) and math.isfinite(confidence)):
        return 0.0
    if confidence <= 0.0:
        return 0.0
    confidence = min(confidence, 1.0)
    value = confidence * math.exp(-0.5 * (row_delta / 0.65) ** 2)
    return max(0.0, min(1.0, value))




def _event_alignment_score(
    notes: Sequence[ScoreNote],
    event: VisualEvent,
    string_order: tuple[int, ...],
    *,
    require_fret_region: bool = False,
) -> tuple[float, int, tuple[tuple[int, PlayablePosition, int], ...]]:
    """Return the deterministic optimal note/contact edges for one chord.

    Each returned edge is ``(note_index, exact_position, contact_index)``.
    Selected edges consume both their observed contact and physical string;
    edges below ``MIN_CONTACT_SCORE`` are not evidence.  The search orders
    notes by their number of viable edges, memoizes dominated states, and uses
    a suffix upper bound to keep normal guitar chords small and deterministic.
    """
    if not notes or not event.contacts:
        return 0.0, 0, ()

    options: list[list[tuple[float, PlayablePosition, int]]] = []
    for note in notes:
        note_options: list[tuple[float, PlayablePosition, int]] = []
        for position in playable_positions(note.pitch_midi):
            if position.fret_idx <= 0:
                continue
            try:
                row = float(string_order.index(position.string_idx))
            except ValueError:
                continue
            for contact_index, contact in enumerate(event.contacts):
                if require_fret_region and contact.fret_region != position.fret_idx:
                    continue
                if require_fret_region:
                    # A numbered fret plus the audio pitch already identifies
                    # one physical string (adjacent strings are 4–5 semitones
                    # apart). A fingertip can sit above the string it presses,
                    # so use its row only as a weak tie-break in this mode.
                    row_score = _contact_score(contact, row)
                    score = min(1.0, max(0.0, contact.confidence)) * (0.82 + 0.18 * row_score)
                else:
                    score = _contact_score(contact, row)
                if score >= MIN_CONTACT_SCORE:
                    note_options.append((score, position, contact_index))
        note_options.sort(
            key=lambda item: (
                -item[0],
                item[1].string_idx,
                item[1].fret_idx,
                item[2],
            )
        )
        options.append(note_options)
    order = sorted(
        range(len(notes)),
        key=lambda index: (len(options[index]), notes[index].source_idx, notes[index].pitch_midi),
    )
    if not any(options):
        return 0.0, 0, ()

    suffix_upper = [0.0] * (len(order) + 1)
    for index in range(len(order) - 1, -1, -1):
        suffix_upper[index] = suffix_upper[index + 1] + (
            options[order[index]][0][0] if options[order[index]] else 0.0
        )
    best_score = 0.0
    best_support = 0
    best_edges: tuple[tuple[int, PlayablePosition, int], ...] = ()
    state_best: dict[tuple[int, int, int], tuple[float, int]] = {}
    def visit(
        index: int,
        used_strings: int,
        used_contacts: int,
        total: float,
        support: int,
        edges: tuple[tuple[int, PlayablePosition, int], ...],
    ) -> None:
        nonlocal best_score, best_support, best_edges
        if total + suffix_upper[index] < best_score - 1e-12:
            return
        state = (index, used_strings, used_contacts)
        previous = state_best.get(state)
        if previous is not None and (
            total < previous[0] - 1e-12
            or (abs(total - previous[0]) <= 1e-12 and support <= previous[1])
        ):
            return
        state_best[state] = (total, support)
        if index == len(order):
            if (
                total > best_score + 1e-12
                or (
                    abs(total - best_score) <= 1e-12
                    and support > best_support
                )
            ):
                best_score = total
                best_support = support
                best_edges = edges
            return
        note_index = order[index]
        # Skipping is allowed for a note without an observed contact.
        visit(index + 1, used_strings, used_contacts, total, support, edges)
        for score, position, contact_index in options[note_index]:
            string_bit = 1 << position.string_idx
            contact_bit = 1 << contact_index
            if used_strings & string_bit or used_contacts & contact_bit:
                continue
            visit(
                index + 1,
                used_strings | string_bit,
                used_contacts | contact_bit,
                total + score,
                support + 1,
                edges + ((note_index, position, contact_index),),
            )

    visit(0, 0, 0, 0.0, 0, ())
    return best_score / len(notes), best_support, best_edges


def infer_visual_alignment(
    score_notes: Sequence[ScoreNote],
    visual_events: Sequence[VisualEvent],
    *,
    quality_summaries: Sequence[Mapping[str, Any]] = (),
    frames_inspected: int = 0,
    frames_with_grid: int = 0,
    model_paths: Mapping[str, str] | None = None,
    config: Mapping[str, Any] | None = None,
    string_order: tuple[int, ...] | None = None,
    require_fret_region: bool = False,
) -> VisualInferenceResult:
    """Infer one of the two visual-top-to-standard-string row orders.

    MIDI supplies exact playable pitches. When a nut/numbered-fret calibration
    is supplied, a fingertip must also occupy that candidate's fret region.
    """
    notes = _coerce_notes(score_notes)
    groups = _group_notes(notes)
    if isinstance(visual_events, Mapping):
        visual_events = tuple(
            event if isinstance(event, VisualEvent) else {"onset_sec": onset, "contacts": contacts}
            for onset, contacts in visual_events.items()
        )
    events = tuple(_coerce_event(event) for event in visual_events)
    matched = _nearest_events(groups, events)
    supported_matches = [item for item in matched if item is not None and item[1].contacts]
    common_config = {
        "mode": "fret_region_and_string_row" if require_fret_region else "string_row",
        "coordinates": "string_rows",
        "absolute_fret_numbers_known": bool(require_fret_region),
        "min_contact_score": MIN_CONTACT_SCORE,
        "row_sigma": 0.65,
        "min_fret_wire_margin": MIN_FRET_WIRE_MARGIN if require_fret_region else None,
        "min_support_events": MIN_ALIGNMENT_SUPPORT_EVENTS,
        "min_support_ratio": MIN_ALIGNMENT_SUPPORT_RATIO,
        "min_alignment_margin": MIN_ALIGNMENT_MARGIN,
        "row_orders": ["forward", "reversed"],
    }
    if config:
        common_config.update(dict(config))
    common_config["mode"] = "fret_region_and_string_row" if require_fret_region else "string_row"
    common_config["coordinates"] = "string_rows_and_calibrated_fret_regions" if require_fret_region else "string_rows"
    common_config["absolute_fret_numbers_known"] = bool(require_fret_region)
    note_sources = {
        note.identity: {
            "onset_sec": note.onset_sec,
            "pitch_midi": note.pitch_midi,
            "source": "audio",
        }
        for note in notes
    }
    if not supported_matches:
        return VisualInferenceResult(
            False,
            "no accepted visual string-row contacts matched requested audio onsets",
            None,
            {},
            events_seen=len(groups),
            quality_summaries=quality_summaries,
            frames_inspected=frames_inspected,
            frames_with_grid=frames_with_grid,
            model_paths=model_paths or {},
            config=common_config,
            note_sources=note_sources,
        )

    row_orders = (tuple(string_order),) if string_order is not None else (tuple(range(6)), tuple(reversed(range(6))))
    if any(sorted(order) != list(range(6)) for order in row_orders):
        raise ValueError("string_order must be a permutation of six standard strings")
    scored: list[tuple[float, int, tuple[int, ...]]] = []
    for string_order in row_orders:
        total = 0.0
        support_events = 0
        for item in matched:
            if item is None:
                continue
            group, event = item
            score, support, _edges = _event_alignment_score(
                group, event, string_order, require_fret_region=require_fret_region,
            )
            total += score
            if support > 0:
                support_events += 1
        scored.append((total, support_events, string_order))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    best_score, best_support, best_order = scored[0]
    runner_score = scored[1][0] if len(scored) > 1 else 0.0
    usable_events = max(1, len(supported_matches))
    support_ratio = best_support / usable_events
    margin = best_score - runner_score
    if (
        best_support < MIN_ALIGNMENT_SUPPORT_EVENTS
        or support_ratio < MIN_ALIGNMENT_SUPPORT_RATIO
        or margin < MIN_ALIGNMENT_MARGIN
        or not math.isfinite(margin)
    ):
        reason = (
            f"visual string-row alignment ambiguous or unsupported "
            f"(support={best_support}/{usable_events}, margin={margin:.6f}; "
            f"required support >= {MIN_ALIGNMENT_SUPPORT_EVENTS} and ratio >= "
            f"{MIN_ALIGNMENT_SUPPORT_RATIO:.2f}, margin >= {MIN_ALIGNMENT_MARGIN:.2f})"
        )
        return VisualInferenceResult(
            False,
            reason,
            None,
            {},
            events_seen=len(groups),
            events_supported=best_support,
            quality_summaries=quality_summaries,
            frames_inspected=frames_inspected,
            frames_with_grid=frames_with_grid,
            model_paths=model_paths or {},
            config=common_config,
            note_sources=note_sources,
        )

    alignment = VisualAlignment(
        best_order,
        best_score,
        runner_score,
        best_support,
        support_ratio,
        margin,
        common_config["mode"],
    )
    likelihoods: dict[ScoreNoteId, dict[PlayablePosition, float]] = {}
    for item in matched:
        if item is None:
            continue
        group, event = item
        _score, _support, edges = _event_alignment_score(
            group, event, best_order, require_fret_region=require_fret_region,
        )
        for note_index, position, contact_index in edges:
            note = group[note_index]
            row = float(best_order.index(position.string_idx))
            if require_fret_region:
                contact = event.contacts[contact_index]
                row_score = _contact_score(contact, row)
                value = min(1.0, max(0.0, contact.confidence)) * (0.82 + 0.18 * row_score)
            else:
                value = _contact_score(event.contacts[contact_index], row)
            if value < MIN_CONTACT_SCORE:
                continue
            likelihoods.setdefault(note.identity, {})[position] = max(
                likelihoods.get(note.identity, {}).get(position, 0.0),
                value,
            )
            note_sources[note.identity] = {
                "onset_sec": note.onset_sec,
                "pitch_midi": note.pitch_midi,
                "source": "visual",
            }
    if not likelihoods:
        return VisualInferenceResult(
            False,
            "accepted string-row alignment had no usable exact non-open playable-position support",
            None,
            {},
            events_seen=len(groups),
            events_supported=best_support,
            quality_summaries=quality_summaries,
            frames_inspected=frames_inspected,
            frames_with_grid=frames_with_grid,
            model_paths=model_paths or {},
            config=common_config,
            note_sources=note_sources,
        )
    return VisualInferenceResult(
        True,
        "accepted audio-backed string-row alignment",
        alignment,
        likelihoods,
        events_seen=len(groups),
        events_supported=best_support,
        quality_summaries=quality_summaries,
        frames_inspected=frames_inspected,
        frames_with_grid=frames_with_grid,
        model_paths=model_paths or {},
        config=common_config,
        note_sources=note_sources,
    )



def _pillow_image_module() -> Any:
    """Load Pillow only when visual YOLO extraction is actually requested."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise VisualExtractionError(
            "Pillow is required for --visual YOLO image input; install the visual extra."
        ) from exc
    return Image


def _default_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[2]
    return root / "models" / "fretboard" / "yolo_weights_best.pt", root / "models" / "hands" / "hand_landmarker.task"



def _native_yolo_proposals(model: Any, rgb: Any, cv2: Any) -> list[tuple[str, Any]]:
    """Run YOLO with a PIL RGB source while retaining RGB atlas coordinates."""
    image = _pillow_image_module().fromarray(rgb)
    predictions = model.predict(image, conf=0.25, retina_masks=True, verbose=False)
    if not predictions:
        return []
    prediction = predictions[0]
    masks = getattr(prediction, "masks", None)
    if masks is None:
        return []
    proposals: list[tuple[str, Any]] = []
    data = getattr(masks, "data", None)
    if data is not None:
        for index, tensor in enumerate(data):
            array = tensor.cpu().numpy() if hasattr(tensor, "cpu") else tensor
            array = (array > 0.5).astype("uint8")
            if array.shape == rgb.shape[:2]:
                proposals.append((f"yolo_{index}", array * 255))
                continue
            # A backend that returns a resized tensor is converted through its
            # native polygon coordinates, never by stretching a padded mask.
            polygons = getattr(masks, "xy", ())
            if index < len(polygons):
                native = cv2.fillPoly(
                    __import__("numpy").zeros(rgb.shape[:2], dtype="uint8"),
                    [__import__("numpy").asarray(polygons[index], dtype="int32")],
                    255,
                )
                proposals.append((f"yolo_{index}", native))
    return proposals


def _mask_geometry(mask: Any, np: Any) -> dict[str, Any] | None:
    """Return a deterministic major/minor-axis description for one mask."""
    try:
        array = np.asarray(mask)
    except (TypeError, ValueError):
        return None
    if array.ndim != 2:
        return None
    ys, xs = np.nonzero(array > 0)
    if len(xs) < 8:
        return None
    points = np.column_stack((xs, ys)).astype(np.float64)
    if not np.isfinite(points).all():
        return None
    centered = points - points.mean(axis=0)
    covariance = (centered.T @ centered) / max(len(points), 1)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None
    axis = np.asarray(eigenvectors[:, int(np.argmax(eigenvalues))], dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm <= 1e-9:
        return None
    axis /= norm
    # The sign is irrelevant geometrically, but canonicalizing it keeps merge
    # provenance and pair ordering deterministic.
    dominant = int(np.argmax(np.abs(axis)))
    if axis[dominant] < 0:
        axis = -axis
    perpendicular = np.asarray((-axis[1], axis[0]), dtype=np.float64)
    longitudinal = points[:, 0] * axis[0] + points[:, 1] * axis[1]
    transverse = points[:, 0] * perpendicular[0] + points[:, 1] * perpendicular[1]
    long_bounds = (float(longitudinal.min()), float(longitudinal.max()))
    cross_bounds = (float(transverse.min()), float(transverse.max()))
    return {
        "axis": axis,
        "long_bounds": long_bounds,
        "cross_bounds": cross_bounds,
        "long_span": long_bounds[1] - long_bounds[0],
        "cross_span": cross_bounds[1] - cross_bounds[0],
    }


def _mask_compatibility(mask_a: Any, mask_b: Any, np: Any) -> tuple[bool, dict[str, Any]]:
    """Check whether two masks are close, aligned pieces of one board."""
    try:
        shape_a = np.asarray(mask_a).shape
        shape_b = np.asarray(mask_b).shape
    except (TypeError, ValueError):
        return False, {"reason": "invalid_mask"}
    if shape_a != shape_b or len(shape_a) != 2:
        return False, {"reason": "shape_mismatch"}
    geometry_a = _mask_geometry(mask_a, np)
    geometry_b = _mask_geometry(mask_b, np)
    if geometry_a is None or geometry_b is None:
        return False, {"reason": "insufficient_mask_geometry"}
    alignment = abs(float(np.dot(geometry_a["axis"], geometry_b["axis"])))
    if not math.isfinite(alignment) or alignment < 0.82:
        return False, {"reason": "axis_mismatch", "axis_alignment": alignment}
    axis = geometry_a["axis"]
    if float(np.dot(axis, geometry_b["axis"])) < 0:
        axis = -axis
    perpendicular = np.asarray((-axis[1], axis[0]), dtype=np.float64)
    points = []
    for mask in (mask_a, mask_b):
        ys, xs = np.nonzero(np.asarray(mask) > 0)
        points.append(np.column_stack((xs, ys)).astype(np.float64))
    projections = [
        (
            item[:, 0] * axis[0] + item[:, 1] * axis[1],
            item[:, 0] * perpendicular[0] + item[:, 1] * perpendicular[1],
        )
        for item in points
    ]
    long_bounds = [(float(item.min()), float(item.max())) for item, _ in projections]
    cross_bounds = [(float(item.min()), float(item.max())) for _, item in projections]

    def interval_gap(first: tuple[float, float], second: tuple[float, float]) -> float:
        return max(0.0, max(first[0], second[0]) - min(first[1], second[1]))

    long_gap = interval_gap(long_bounds[0], long_bounds[1])
    cross_gap = interval_gap(cross_bounds[0], cross_bounds[1])
    cross_overlap = max(
        0.0,
        min(cross_bounds[0][1], cross_bounds[1][1])
        - max(cross_bounds[0][0], cross_bounds[1][0]),
    )
    min_cross_span = max(
        1.0,
        min(cross_bounds[0][1] - cross_bounds[0][0], cross_bounds[1][1] - cross_bounds[1][0]),
    )
    cross_overlap_ratio = cross_overlap / min_cross_span
    long_tolerance = max(
        3.0,
        0.45 * max(geometry_a["cross_span"], geometry_b["cross_span"]),
    )
    cross_tolerance = max(3.0, 0.20 * min_cross_span)
    accepted = (
        long_gap <= long_tolerance
        and cross_gap <= cross_tolerance
        and cross_overlap_ratio >= 0.35
    )
    metrics = {
        "axis_alignment": float(alignment),
        "longitudinal_gap_px": float(long_gap),
        "transverse_gap_px": float(cross_gap),
        "transverse_overlap_ratio": float(cross_overlap_ratio),
        "longitudinal_tolerance_px": float(long_tolerance),
        "transverse_tolerance_px": float(cross_tolerance),
    }
    if not accepted:
        metrics["reason"] = (
            "gap_or_overlap"
            if cross_overlap_ratio >= 0.35
            else "insufficient_transverse_overlap"
        )
    return accepted, metrics


def _merge_compatible_yolo_proposals(
    proposals: Sequence[tuple[str, Any]],
) -> tuple[list[tuple[str, Any]], dict[str, dict[str, Any]]]:
    """Add unions for pairwise-compatible YOLO pieces without dropping originals."""
    np = __import__("numpy")
    original = list(proposals)
    if len(original) < 2:
        return original, {}
    compatible: dict[tuple[int, int], dict[str, Any]] = {}
    for left in range(len(original)):
        for right in range(left + 1, len(original)):
            accepted, metrics = _mask_compatibility(
                original[left][1],
                original[right][1],
                np,
            )
            if accepted:
                compatible[(left, right)] = metrics
    if not compatible:
        return original, {}

    # Build maximal cliques of the compatibility graph.  Requiring every
    # pair to pass avoids a transitive chain that bridges unrelated masks.
    groups: set[frozenset[int]] = set()
    for seed in range(len(original)):
        group = [seed]
        for candidate in range(seed + 1, len(original)):
            if all(
                tuple(sorted((member, candidate))) in compatible
                for member in group
            ):
                group.append(candidate)
        if len(group) >= 2:
            groups.add(frozenset(group))
    maximal = sorted(
        (
            group
            for group in groups
            if not any(group < other for other in groups)
        ),
        key=lambda group: tuple(sorted(group)),
    )
    augmented = list(original)
    provenance: dict[str, dict[str, Any]] = {}
    for group in maximal:
        indices = tuple(sorted(group))
        names = tuple(str(original[index][0]) for index in indices)
        merged_name = "merged_" + "_".join(names)
        merged_mask = np.zeros_like(np.asarray(original[indices[0]][1]))
        for index in indices:
            merged_mask = np.logical_or(merged_mask, np.asarray(original[index][1]) > 0)
        merged_mask = merged_mask.astype("uint8") * 255
        augmented.append((merged_name, merged_mask))
        provenance[merged_name] = {
            "merged_from": list(names),
            "compatibility": [
                dict(compatible[tuple(sorted((left, right)))])
                for offset, left in enumerate(indices)
                for right in indices[offset + 1 :]
            ],
        }
    return augmented, provenance


def _landmark_confidence(landmark: Any) -> float:
    """Return usable landmark confidence without treating bad metadata as support."""
    explicit = False
    for field in ("presence", "visibility"):
        try:
            value = getattr(landmark, field)
        except AttributeError:
            continue
        if value is None:
            continue
        explicit = True
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(numeric) and numeric > 0.0:
            return numeric
    return 0.0 if explicit else 1.0


def _landmark_point(landmark: Any, width: int, height: int) -> tuple[float, float] | None:
    try:
        x = float(landmark.x) * width
        y = float(landmark.y) * height
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    return (x, y) if math.isfinite(x) and math.isfinite(y) else None


def _relative_contact(atlas: Any, tip: Any, distal: Any, frame_shape: tuple[int, int], cv2: Any, np: Any, *, finger: str, confidence: float, time_sec: float, frame_index: int, calibration: Any = None) -> StringRowContact | None:
    """Map a fingertip to a string row and optional calibrated fret region.

    A tracked calibration already defines both axes of the fretboard.  In that
    case the YOLO/EFA atlas must not veto the contact or supply a conflicting
    string row: a partial mask can describe only the nut end of a visible neck.
    """
    try:
        confidence = float(confidence)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(confidence) or confidence <= 0.0:
        return None
    height, width = frame_shape
    tip_point = _landmark_point(tip, width, height)
    distal_point = _landmark_point(distal, width, height)
    if tip_point is None or not (
        0.0 <= tip_point[0] < float(width) and 0.0 <= tip_point[1] < float(height)
    ):
        return None
    if distal_point is not None:
        source_distance = math.hypot(
            tip_point[0] - distal_point[0],
            tip_point[1] - distal_point[1],
        )
        if source_distance > max(width, height) * 0.35:
            return None
    if calibration is not None:
        try:
            board_x, board_y = map(float, calibration.image_to_board([tip_point])[0])
            continuous_fret = calibration.continuous_fret(board_x)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
        if (
            continuous_fret is None
            or not math.isfinite(continuous_fret)
            or not math.isfinite(board_y)
            or not 0.0 <= board_y <= 1.0
            or not 0.0 < continuous_fret <= calibration.far_fret
        ):
            return None
        fret_region = math.ceil(continuous_fret)
        if min(continuous_fret - (fret_region - 1), fret_region - continuous_fret) < MIN_FRET_WIRE_MARGIN:
            return None
        return StringRowContact(
            row=board_y * 5.0,
            confidence=confidence,
            finger=finger,
            time_sec=time_sec,
            frame_index=frame_index,
            source_x=tip_point[0],
            fret_region=fret_region,
        )
    if atlas is None:
        return None
    try:
        mapped_points = np.asarray(
            atlas.image_to_grid(np.asarray([tip_point], dtype=np.float64)),
            dtype=np.float64,
        )
        if mapped_points.shape == (2,):
            mapped_points = mapped_points.reshape(1, 2)
        target_x = np.asarray(atlas.target_x, dtype=np.float64)
        target_y = np.asarray(atlas.target_y, dtype=np.float64)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if (
        mapped_points.ndim != 2
        or mapped_points.shape[0] < 1
        or mapped_points.shape[1] < 2
        or not np.isfinite(mapped_points[0, :2]).all()
        or target_x.ndim != 1
        or target_y.ndim != 1
        or len(target_x) < 2
        or len(target_y) != 8
        or not np.isfinite(target_x).all()
        or not np.isfinite(target_y).all()
    ):
        return None
    row_axis = target_y[1:-1]
    if len(row_axis) != 6 or np.any(np.diff(row_axis) <= 0):
        return None
    mapped = mapped_points[0, :2]
    if (
        mapped[0] < min(target_x)
        or mapped[0] > max(target_x)
        or mapped[1] < min(target_y)
        or mapped[1] > max(target_y)
        or mapped[1] < row_axis[0]
        or mapped[1] > row_axis[-1]
    ):
        # Longitudinal position is deliberately ignored after this bounds
        # check; it cannot establish an absolute fret.
        return None
    row = float(np.interp(mapped[1], row_axis, np.arange(6, dtype=np.float64)))
    if not math.isfinite(row):
        return None
    return StringRowContact(
        row,
        confidence,
        finger,
        time_sec,
        frame_index,
        source_x=tip_point[0],
        fret_region=None,
    )


def _aggregate_contacts(contacts: Sequence[StringRowContact]) -> tuple[StringRowContact, ...]:
    """Aggregate same-finger row observations from onset-neighbour frames."""
    groups: list[list[StringRowContact]] = []
    for contact in sorted(
        contacts,
        key=lambda item: (
            item.finger or "",
            item.row,
            item.frame_index if item.frame_index is not None else -1,
        ),
    ):
        match = None
        for group in groups:
            representative = group[0]
            if (
                representative.finger == contact.finger
                and abs(representative.row - contact.row) <= 0.45
                and representative.fret_region == contact.fret_region
            ):
                match = group
                break
        if match is None:
            groups.append([contact])
        else:
            match.append(contact)
    output = []
    for group in groups:
        weights = []
        for item in group:
            try:
                confidence = float(item.confidence)
            except (TypeError, ValueError, OverflowError):
                confidence = 0.0
            weights.append(max(0.01, confidence) if math.isfinite(confidence) else 0.01)
        weight = sum(weights)
        source_x_values = [
            (item.source_x, item_weight)
            for item, item_weight in zip(group, weights)
            if item.source_x is not None and math.isfinite(float(item.source_x))
        ]
        source_x = (
            sum(float(value) * item_weight for value, item_weight in source_x_values)
            / sum(item_weight for _, item_weight in source_x_values)
            if source_x_values
            else None
        )
        output.append(
            StringRowContact(
                sum(item.row * item_weight for item, item_weight in zip(group, weights)) / weight,
                min(1.0, sum(weights) / len(group)),
                group[0].finger,
                max(item.time_sec for item in group),
                group[-1].frame_index,
                source_x=source_x,
                fret_region=group[0].fret_region,
            )
        )
    return tuple(output)


def _track_frame_calibration(
    reference_rgb: Any,
    rgb: Any,
    calibration: Any,
    frame_index: int,
    fps: float,
    tracker: Any,
    previous: tuple[Any, Any, int, int] | None,
) -> tuple[Any, dict[str, Any], tuple[Any, Any, int, int] | None]:
    """Track from the anchor, then a nearby accepted frame for a short gap."""
    board, info = tracker(reference_rgb, rgb, calibration)
    if board is not None:
        return board, {**info, "strategy": "reference_frame"}, (rgb, board, frame_index, 0)
    if previous is not None:
        prev_rgb, prev_board, prev_index, chained = previous
        max_gap = max(3, int(round(fps * .25)))
        if chained < 3 and 0 < frame_index - prev_index <= max_gap:
            nearby_board, nearby_info = tracker(prev_rgb, rgb, prev_board)
            if nearby_board is not None:
                return (
                    nearby_board,
                    {**nearby_info, "strategy": "nearby_frame", "base_frame": prev_index,
                     "reference_failure": info.get("reason")},
                    (rgb, nearby_board, frame_index, chained + 1),
                )
    return None, {**info, "strategy": "failed"}, previous


def extract_visual_events(
    video_path: str | Path,
    score_notes: Sequence[ScoreNote],
    *,
    yolo_weights: str | Path | None = None,
    hand_model: str | Path | None = None,
    frame_deltas: Sequence[int] = DEFAULT_FRAME_DELTAS,
    calibration_path: str | Path | None = None,
    debug_video_path: str | Path | None = None,
) -> ExtractedVisual:
    """Read a local video and extract aggregated row and optional fret contacts.

    OpenCV timestamps are derived solely from monotonic frame index/FPS.  One
    YOLO segmentation object and one MediaPipe HandLandmarker are initialized
    for the complete bounded extraction, not once per frame.
    """
    import numpy as np
    calibration = None
    reference_frame = None
    if calibration_path is not None:
        from main.video.fretboard.audio_visual_fusion import BoardCalibration, track_calibration
        try:
            calibration_data = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
            calibration = BoardCalibration.from_dict(calibration_data)
            reference_frame = int(calibration_data.get("reference_frame", 0))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise VisualExtractionError(f"Invalid visual calibration {calibration_path}: {exc}") from exc
        if reference_frame < 0:
            raise VisualExtractionError("Visual calibration reference_frame must be nonnegative")
        if tuple(reversed(calibration.tuning)) != STANDARD_TUNING_MIDI:
            raise VisualExtractionError("Visual calibration tuning must match standard guitar tuning")
    try:
        import cv2
    except ImportError as exc:
        raise VisualExtractionError("OpenCV is required for --visual; install the visual extra.") from exc
    _pillow_image_module()
    try:
        from main.video.fretboard.rectify_atlas import select_atlas
    except ImportError as exc:
        raise VisualExtractionError("The repository rectification modules are unavailable.") from exc
    default_yolo, default_hand = _default_paths()
    yolo_path = Path(yolo_weights) if yolo_weights is not None else default_yolo
    hand_path = Path(hand_model) if hand_model is not None else default_hand
    if not yolo_path.is_file():
        raise VisualExtractionError(f"YOLO weights not found: {yolo_path}. Use --yolo-weights or fetch Git LFS files.")
    try:
        if yolo_path.stat().st_size < 1024 and b"git-lfs" in yolo_path.read_bytes()[:256].lower():
            raise VisualExtractionError(
                f"YOLO weights are a Git LFS pointer, not a checkpoint: {yolo_path}. Run `git lfs pull`."
            )
    except OSError as exc:
        raise VisualExtractionError(f"Could not read YOLO weights {yolo_path}: {exc}") from exc
    if not hand_path.is_file():
        raise VisualExtractionError(f"MediaPipe hand model not found: {hand_path}. Use --hand-model or fetch Git LFS files.")
    try:
        from ultralytics import YOLO
        yolo = YOLO(str(yolo_path))
    except ImportError as exc:
        raise VisualExtractionError("Ultralytics is required for --visual; install the visual extra.") from exc
    except (OSError, ValueError, RuntimeError) as exc:
        raise VisualExtractionError(f"Could not initialize YOLO model {yolo_path}: {exc}") from exc
    try:
        import mediapipe as mp
        base_options = mp.tasks.BaseOptions(model_asset_path=str(hand_path))
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=MEDIAPIPE_CONFIDENCE_THRESHOLDS["min_hand_detection_confidence"],
            min_hand_presence_confidence=MEDIAPIPE_CONFIDENCE_THRESHOLDS["min_hand_presence_confidence"],
            min_tracking_confidence=MEDIAPIPE_CONFIDENCE_THRESHOLDS["min_tracking_confidence"],
        )
        landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
    except ImportError as exc:
        raise VisualExtractionError("MediaPipe is required for --visual; install the visual extra.") from exc
    except (OSError, ValueError, RuntimeError) as exc:
        raise VisualExtractionError(f"Could not initialize MediaPipe HandLandmarker {hand_path}: {exc}") from exc
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        landmarker.close()
        raise VisualExtractionError(f"OpenCV could not open local video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0.0:
        capture.release()
        landmarker.close()
        raise VisualExtractionError(f"Video has no usable FPS metadata: {video_path}")
    reference_rgb = None
    if calibration is not None:
        capture.set(cv2.CAP_PROP_POS_FRAMES, reference_frame)
        ok, reference_bgr = capture.read()
        if not ok:
            capture.release()
            landmarker.close()
            raise VisualExtractionError(f"Could not read calibration reference frame {reference_frame}")
        reference_rgb = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB)
        if calibration.image_size is not None and tuple(calibration.image_size) != reference_rgb.shape[1::-1]:
            capture.release()
            landmarker.close()
            raise VisualExtractionError("Visual calibration image_size does not match the video")
    notes = _coerce_notes(score_notes)
    groups = _group_notes(notes)
    target_indices: dict[int, list[int]] = {}
    for group_index, group in enumerate(groups):
        if not group:
            continue
        center = max(0, int(round(group[0].onset_sec * fps)))
        for delta in frame_deltas:
            index = center + int(delta)
            if index >= 0:
                target_indices.setdefault(index, []).append(group_index)
    debug_indices: set[int] = set()
    debug_fps = None
    if debug_video_path is not None:
        from main.video.hands.debug_video import sample_indices
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        indices, debug_fps = sample_indices(frame_count, fps)
        debug_indices = set(indices)
        for index in debug_indices:
            target_indices.setdefault(index, [])
    sorted_target_indices = sorted(target_indices)
    events_contacts: dict[int, list[StringRowContact]] = {index: [] for index in range(len(groups))}
    event_frames: dict[int, set[int]] = {index: set() for index in range(len(groups))}
    quality_summaries: list[Mapping[str, Any]] = []
    frames_inspected = 0
    frames_with_grid = 0
    last_timestamp_ms = -1
    previous_tracking = None
    debug_writer = None
    try:
        if debug_video_path is not None:
            from main.video.hands.debug_video import DebugVideoWriter, draw_debug_frame
            width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            debug_writer = DebugVideoWriter(debug_video_path, video_path, width, height, debug_fps)
        for frame_index in sorted_target_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, bgr = capture.read()
            if not ok:
                continue
            group_indices = target_indices[frame_index]
            frames_inspected += 1
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            board = None
            tracking = None
            if calibration is not None:
                board, tracking, previous_tracking = _track_frame_calibration(
                    reference_rgb, rgb, calibration, frame_index, fps,
                    track_calibration, previous_tracking,
                )
            raw_proposals = _native_yolo_proposals(yolo, rgb, cv2)
            proposals, merge_provenance = _merge_compatible_yolo_proposals(raw_proposals)
            if proposals:
                atlas, _mask, alternatives = select_atlas(rgb, proposals, spacing="uniform")
            else:
                atlas, alternatives = None, ()
            alternative_rows = []
            for item in alternatives:
                row = dict(item)
                provenance = merge_provenance.get(row.get("proposal"))
                if provenance is not None:
                    row["merge_provenance"] = dict(provenance)
                alternative_rows.append(row)
            summary = {
                "frame_index": frame_index,
                "time_sec": frame_index / fps,
                "proposal_count": len(proposals),
                "original_proposal_count": len(raw_proposals),
                "merged_proposal_count": len(merge_provenance),
                "merge_provenance": [
                    {"proposal": name, **dict(value)}
                    for name, value in merge_provenance.items()
                ],
                "alternatives": alternative_rows,
                "status": "unavailable",
            }
            if tracking is not None:
                summary["tracking"] = dict(tracking)
            frame_contact_count = 0
            frame_contact_details: list[dict[str, Any]] = []
            frame_regions: set[int] = set()
            frame_fingertips: list[dict[str, Any]] = []
            grid_usable = False
            if atlas is not None:
                quality = dict(atlas.quality)
                selected_proposal = quality.get("selected_proposal")
                if selected_proposal in merge_provenance:
                    quality["merge_provenance"] = dict(merge_provenance[selected_proposal])
                summary["quality"] = quality
                if quality.get("status") == "grid":
                    frames_with_grid += 1
                    summary["status"] = "grid"
                    grid_usable = True
            if grid_usable or board is not None or frame_index in debug_indices:
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = max(
                    int(round(frame_index / fps * 1000.0)),
                    last_timestamp_ms + 1,
                )
                last_timestamp_ms = timestamp_ms
                hand_result = landmarker.detect_for_video(image, timestamp_ms)
                hands = getattr(hand_result, "hand_landmarks", ())
                for hand in hands:
                    for finger, tip_index, distal_index in (("index", 8, 7), ("middle", 12, 11), ("ring", 16, 15), ("pinky", 20, 19)):
                        if tip_index >= len(hand) or distal_index >= len(hand):
                            continue
                        tip, distal = hand[tip_index], hand[distal_index]
                        tip_point = _landmark_point(tip, rgb.shape[1], rgb.shape[0])
                        distal_point = _landmark_point(distal, rgb.shape[1], rgb.shape[0])
                        confidence = _landmark_confidence(tip)
                        contact = None
                        if board is not None or (grid_usable and calibration is None):
                            contact = _relative_contact(
                                atlas,
                                tip,
                                distal,
                                rgb.shape[:2],
                                cv2,
                                np,
                                finger=finger,
                                confidence=confidence,
                                time_sec=frame_index / fps,
                                frame_index=frame_index,
                                calibration=board,
                            )
                        if tip_point is not None:
                            frame_fingertips.append({
                                "finger": finger, "x": tip_point[0], "y": tip_point[1],
                                "distal": distal_point,
                                "row": contact.row if contact is not None else None,
                                "fret_region": contact.fret_region if contact is not None else None,
                            })
                        if contact is not None:
                            frame_contact_count += 1
                            frame_contact_details.append({
                                "finger": finger,
                                "row": contact.row,
                                "fret_region": contact.fret_region,
                                "confidence": contact.confidence,
                                "source_x": contact.source_x,
                            })
                            if contact.fret_region is not None:
                                frame_regions.add(contact.fret_region)
                            for group_index in group_indices:
                                events_contacts[group_index].append(contact)
                                event_frames[group_index].add(frame_index)
            summary["contact_count"] = frame_contact_count
            summary["contacts"] = frame_contact_details
            if calibration is not None:
                summary["fret_regions"] = sorted(frame_regions)
            quality_summaries.append(summary)
            if debug_writer is not None and frame_index in debug_indices:
                debug_writer.write(draw_debug_frame(
                    bgr, atlas, frame_fingertips,
                    time_sec=frame_index / fps, tracking=tracking,
                ))
    finally:
        capture.release()
        try:
            landmarker.close()
        except (AttributeError, RuntimeError):
            pass
        if debug_writer is not None:
            debug_writer.close()
    events = tuple(
        VisualEvent(
            groups[index][0].onset_sec if groups[index] else 0.0,
            _aggregate_contacts(events_contacts[index]),
            len(event_frames[index]),
            "grid" if any(summary.get("status") == "grid" and summary.get("frame_index") in event_frames[index] for summary in quality_summaries) else "unavailable",
        )
        for index in range(len(groups))
    )
    return ExtractedVisual(
        events,
        frames_inspected,
        frames_with_grid,
        tuple(quality_summaries),
        {"yolo_weights": str(yolo_path), "hand_model": str(hand_path)},
        {
            "fps": fps,
            "frame_deltas": [int(delta) for delta in frame_deltas],
            "timestamp_source": "round(frame_index / fps * 1000)",
            "timestamp_mode": "VIDEO",
            "seek_strategy": "sorted_unique_frame_indices",
            "mode": "fret_region_and_string_row" if calibration is not None else "string_row",
            "calibration_path": str(calibration_path) if calibration_path is not None else None,
            "reference_frame": reference_frame,
            "debug_video_path": str(debug_video_path) if debug_video_path is not None else None,
            "debug_video_fps": debug_fps,
            "coordinates": "unflipped_rgb",
            "native_masks": True,
            "mediapipe_thresholds": dict(MEDIAPIPE_CONFIDENCE_THRESHOLDS),
            **MEDIAPIPE_CONFIDENCE_THRESHOLDS,
        },
    )


def analyze_video(
    video_path: str | Path,
    score_notes: Sequence[ScoreNote] | None = None,
    *,
    midi_path: str | Path | None = None,
    yolo_weights: str | Path | None = None,
    hand_model: str | Path | None = None,
    calibration_path: str | Path | None = None,
    debug_video_path: str | Path | None = None,
) -> VisualInferenceResult:
    """Run extraction then pure semantic alignment for one local video."""
    if score_notes is None:
        if midi_path is None:
            raise ValueError("score_notes or midi_path is required")
        score_notes = read_midi_notes(midi_path)
    notes = _coerce_notes(score_notes)
    extracted = extract_visual_events(
        video_path, notes, yolo_weights=yolo_weights, hand_model=hand_model,
        calibration_path=calibration_path,
        debug_video_path=debug_video_path,
    )
    if calibration_path is not None:
        from main.video.fretboard.audio_visual_fusion import BoardCalibration
        data = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        calibration = BoardCalibration.from_dict(data)
        string_order = tuple(range(6)) if calibration.top_string == 6 else tuple(reversed(range(6)))
    else:
        string_order = None
    return infer_visual_alignment(
        notes,
        extracted.events,
        quality_summaries=extracted.quality_summaries,
        frames_inspected=extracted.frames_inspected,
        frames_with_grid=extracted.frames_with_grid,
        model_paths=extracted.model_paths,
        config=extracted.config,
        string_order=string_order,
        require_fret_region=calibration_path is not None,
    )




__all__ = [
    "DEFAULT_FRAME_DELTAS",
    "ExtractedVisual",
    "MIN_ALIGNMENT_MARGIN",
    "MIN_ALIGNMENT_SUPPORT_EVENTS",
    "MIN_ALIGNMENT_SUPPORT_RATIO",
    "StringRowContact",
    "VisualAlignment",
    "VisualEvent",
    "VisualExtractionError",
    "VisualInferenceResult",
    "analyze_video",
    "extract_visual_events",
    "infer_visual_alignment",
]
