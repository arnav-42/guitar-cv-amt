"""Deterministic standard-tuning guitar tablature from MIDI notes.

The MIDI representation does not contain guitar fingering information.  This
module therefore chooses one physically playable position for each note using
standard tuning, grouping close onsets into chords and scoring whole chord
positions rather than selecting each note independently.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Sequence


# String indices follow GuitarSet: 0 is the low E string and 5 is the high E.
STANDARD_TUNING_MIDI = (40, 45, 50, 55, 59, 64)
TAB_STRING_LABELS = ("e", "B", "G", "D", "A", "E")
MAX_FRET = 24
ONSET_GROUP_TOLERANCE_SEC = 0.05
_MAX_CANDIDATES_PER_CHORD = 4096


class GuitarTabError(RuntimeError):
    """A user-facing error while reading or assigning a MIDI tablature."""


@dataclass(frozen=True)
class TabNote:
    """One playable guitar note in standard tuning.

    ``string_idx`` uses low-to-high indexing (0 is low E), while the renderer
    displays strings in the conventional high-to-low order.  The frozen data
    class keeps assignments independent of, and unable to mutate, MIDI note
    objects supplied by a caller.
    """

    onset_sec: float
    offset_sec: float
    pitch_midi: int
    string_idx: int
    fret_idx: int

    def __post_init__(self) -> None:
        # MIDI pitches are semitone values throughout the public API.  Coerce
        # values at construction so a PrettyMIDI float pitch cannot leak out.
        object.__setattr__(self, "onset_sec", float(self.onset_sec))
        object.__setattr__(self, "offset_sec", float(self.offset_sec))
        object.__setattr__(self, "pitch_midi", int(round(float(self.pitch_midi))))
        object.__setattr__(self, "string_idx", int(self.string_idx))
        object.__setattr__(self, "fret_idx", int(self.fret_idx))


@dataclass(frozen=True)
class _SourceNote:
    onset_sec: float
    offset_sec: float
    pitch_midi: int
    source_idx: int


@dataclass(frozen=True)
class _ChordCandidate:
    """A complete maximum-cardinality assignment for one onset group."""

    notes: tuple[TabNote, ...]

    @property
    def key(self) -> tuple[tuple[int, int, int, int], ...]:
        return tuple(
            (note.pitch_midi, note.string_idx, note.fret_idx, i)
            for i, note in enumerate(self.notes)
        )


def _midi_to_positions(pitch_midi: int) -> tuple[tuple[int, int], ...]:
    """Return playable ``(string_idx, fret_idx)`` positions for a pitch."""
    positions = []
    for string_idx, tuning_pitch in enumerate(STANDARD_TUNING_MIDI):
        fret_idx = pitch_midi - tuning_pitch
        if 0 <= fret_idx <= MAX_FRET:
            positions.append((string_idx, fret_idx))
    # Low-fret positions make a useful deterministic tie-break and give the
    # assignment search a physically sensible first candidate.
    return tuple(sorted(positions, key=lambda position: (position[1], position[0])))


def _group_source_notes(notes: Sequence[_SourceNote]) -> list[tuple[_SourceNote, ...]]:
    """Group notes whose onsets are close enough to be one chord event."""
    ordered = sorted(
        notes,
        key=lambda note: (
            note.onset_sec,
            note.pitch_midi,
            note.offset_sec,
            note.source_idx,
        ),
    )
    groups: list[list[_SourceNote]] = []
    group_start: float | None = None
    for note in ordered:
        if (
            group_start is None
            or note.onset_sec - group_start > ONSET_GROUP_TOLERANCE_SEC + 1e-9
        ):
            groups.append([])
            group_start = note.onset_sec
        groups[-1].append(note)
    return [tuple(group) for group in groups]


def _candidate_intrinsic_cost(candidate: _ChordCandidate) -> float:
    """Score a chord's hand position without considering adjacent events."""
    frets = [note.fret_idx for note in candidate.notes]
    strings = [note.string_idx for note in candidate.notes]
    if not frets:
        return 0.0
    fret_span = max(frets) - min(frets)
    fret_center = sum(frets) / len(frets)
    string_span = max(strings) - min(strings)
    # Span dominates: keeping a chord under one hand is preferable to
    # independently taking the lowest fret for every pitch.  A small center
    # term breaks equally compact-position ties toward the nut.
    return 8.0 * fret_span + 0.75 * fret_center + 0.15 * string_span


def _transition_cost(previous: _ChordCandidate, current: _ChordCandidate) -> float:
    """Score a fingering transition between two assigned chord events."""
    if not previous.notes or not current.notes:
        return 0.0

    previous_notes = sorted(
        previous.notes, key=lambda note: (note.pitch_midi, note.string_idx, note.fret_idx)
    )
    current_notes = sorted(
        current.notes, key=lambda note: (note.pitch_midi, note.string_idx, note.fret_idx)
    )
    paired = min(len(previous_notes), len(current_notes))
    cost = 0.0
    for previous_note, current_note in zip(previous_notes[:paired], current_notes[:paired]):
        cost += 1.8 * abs(current_note.fret_idx - previous_note.fret_idx)
        cost += 0.65 * abs(current_note.string_idx - previous_note.string_idx)

    # Chords with different cardinality still pay a modest position shift for
    # their common hand center, while omitted/unplayable notes are not allowed
    # to create invalid strings.
    previous_center = sum(note.fret_idx for note in previous.notes) / len(previous.notes)
    current_center = sum(note.fret_idx for note in current.notes) / len(current.notes)
    cost += 1.25 * abs(current_center - previous_center)
    cost += 0.5 * abs(len(previous.notes) - len(current.notes))
    return cost


def _build_chord_candidates(group: Sequence[_SourceNote]) -> list[_ChordCandidate]:
    """Enumerate deterministic maximum-cardinality assignments for a chord."""
    options = {
        note.source_idx: _midi_to_positions(note.pitch_midi)
        for note in group
    }
    # Notes with fewer choices are assigned first; this prunes impossible
    # duplicate-pitch chords without changing the eventual tie-break key.
    ordered = sorted(
        group,
        key=lambda note: (
            len(options[note.source_idx]),
            note.pitch_midi,
            note.onset_sec,
            note.source_idx,
        ),
    )
    if not ordered or all(not options[note.source_idx] for note in ordered):
        return []

    best_count = -1
    candidates: list[_ChordCandidate] = []
    seen: set[tuple[tuple[int, int, int, int], ...]] = set()

    def visit(
        index: int,
        used_strings: frozenset[int],
        assigned: tuple[TabNote, ...],
    ) -> None:
        nonlocal best_count, candidates, seen
        if (
            best_count >= min(len(ordered), len(STANDARD_TUNING_MIDI))
            and len(candidates) >= _MAX_CANDIDATES_PER_CHORD
        ):
            return
        remaining = len(ordered) - index
        if len(assigned) + remaining < best_count:
            return
        if index == len(ordered):
            count = len(assigned)
            # A candidate contains notes in search order.  Store them in source
            # order so flattening is stable even when options were constrained.
            candidate = _ChordCandidate(
                tuple(
                    sorted(
                        assigned,
                        key=lambda note: (
                            note.onset_sec,
                            note.pitch_midi,
                            note.string_idx,
                            note.fret_idx,
                        ),
                    )
                )
            )
            key = candidate.key
            if count > best_count:
                best_count = count
                candidates = [candidate]
                seen = {key}
            elif count == best_count and key not in seen:
                if len(candidates) < _MAX_CANDIDATES_PER_CHORD:
                    candidates.append(candidate)
                seen.add(key)
            return

        note = ordered[index]
        for string_idx, fret_idx in options[note.source_idx]:
            if string_idx in used_strings:
                continue
            visit(
                index + 1,
                used_strings | {string_idx},
                assigned
                + (
                    TabNote(
                        note.onset_sec,
                        note.offset_sec,
                        note.pitch_midi,
                        string_idx,
                        fret_idx,
                    ),
                ),
            )
        # Skipping a note is required for an unplayable note or a chord with
        # more notes than strings.  Maximum cardinality ensures it happens only
        # when necessary.
        visit(index + 1, used_strings, assigned)

    visit(0, frozenset(), ())
    candidates.sort(key=lambda candidate: (_candidate_intrinsic_cost(candidate), candidate.key))
    return candidates


def _assign_source_notes(source_notes: Sequence[_SourceNote]) -> list[TabNote]:
    """Assign grouped source notes using dynamic programming over positions."""
    groups = _group_source_notes(source_notes)
    candidate_groups = [_build_chord_candidates(group) for group in groups]
    candidate_groups = [group for group in candidate_groups if group]
    if not candidate_groups:
        return []

    # Dynamic programming makes transition costs affect the whole phrase,
    # rather than selecting a locally cheapest fingering for every chord.
    costs: list[list[float]] = []
    previous_indices: list[list[int | None]] = []
    first_costs = [_candidate_intrinsic_cost(candidate) for candidate in candidate_groups[0]]
    costs.append(first_costs)
    previous_indices.append([None] * len(first_costs))

    for group_index in range(1, len(candidate_groups)):
        group = candidate_groups[group_index]
        group_costs: list[float] = []
        group_previous: list[int | None] = []
        for candidate in group:
            best: tuple[float, int, tuple[tuple[int, int, int, int], ...]] | None = None
            intrinsic = _candidate_intrinsic_cost(candidate)
            for previous_index, previous_candidate in enumerate(candidate_groups[group_index - 1]):
                total = (
                    costs[group_index - 1][previous_index]
                    + intrinsic
                    + _transition_cost(previous_candidate, candidate)
                )
                proposal = (total, previous_index, previous_candidate.key)
                if best is None or proposal < best:
                    best = proposal
            assert best is not None
            group_costs.append(best[0])
            group_previous.append(best[1])
        costs.append(group_costs)
        previous_indices.append(group_previous)

    last_index = min(
        range(len(candidate_groups[-1])),
        key=lambda index: (costs[-1][index], candidate_groups[-1][index].key),
    )
    selected_indices = [last_index]
    for group_index in range(len(candidate_groups) - 1, 0, -1):
        previous_index = previous_indices[group_index][selected_indices[-1]]
        assert previous_index is not None
        selected_indices.append(previous_index)
    selected_indices.reverse()

    assigned: list[TabNote] = []
    for group_index, candidate_index in enumerate(selected_indices):
        assigned.extend(candidate_groups[group_index][candidate_index].notes)
    assigned.sort(
        key=lambda note: (
            note.onset_sec,
            note.pitch_midi,
            note.offset_sec,
            note.string_idx,
            note.fret_idx,
        )
    )
    return assigned


def _read_midi_source(path: str | Path) -> list[_SourceNote]:
    try:
        import pretty_midi
    except ImportError as exc:
        raise GuitarTabError(
            "pretty-midi is not installed. Install the transcription dependencies from the README."
        ) from exc

    try:
        midi = pretty_midi.PrettyMIDI(str(path))
    except (OSError, ValueError, RuntimeError) as exc:
        raise GuitarTabError(f"Could not read MIDI file {path}: {exc}") from exc

    source_notes: list[_SourceNote] = []
    source_idx = 0
    for instrument in midi.instruments:
        if bool(getattr(instrument, "is_drum", False)):
            continue
        for note in instrument.notes:
            try:
                onset_sec = float(note.start)
                offset_sec = float(note.end)
                pitch_midi = int(round(float(note.pitch)))
            except (AttributeError, TypeError, ValueError) as exc:
                raise GuitarTabError(f"MIDI file {path} contains an invalid note.") from exc
            if not math.isfinite(onset_sec) or not math.isfinite(offset_sec):
                raise GuitarTabError(f"MIDI file {path} contains a non-finite note time.")
            if offset_sec <= onset_sec:
                continue
            source_notes.append(
                _SourceNote(onset_sec, offset_sec, pitch_midi, source_idx)
            )
            source_idx += 1
    return source_notes


def midi_to_tab_notes(path: str | Path) -> list[TabNote]:
    """Parse non-drum MIDI notes with finite, positive durations.

    Notes outside standard-tuned frets 0--24 and notes whose offset is not
    after their onset are omitted.  If a simultaneous group contains more
    playable notes than six strings, the deterministic maximum-cardinality
    assignment omits the excess notes.  Source MIDI note objects are read
    only and never modified.
    """
    return _assign_source_notes(_read_midi_source(path))


def _group_tab_notes(notes: Iterable[TabNote]) -> list[tuple[TabNote, ...]]:
    ordered = sorted(
        notes,
        key=lambda note: (
            note.onset_sec,
            note.pitch_midi,
            note.offset_sec,
            note.string_idx,
            note.fret_idx,
        ),
    )
    groups: list[list[TabNote]] = []
    group_start: float | None = None
    for note in ordered:
        if not math.isfinite(note.onset_sec):
            continue
        if (
            group_start is None
            or note.onset_sec - group_start > ONSET_GROUP_TOLERANCE_SEC + 1e-9
        ):
            groups.append([])
            group_start = note.onset_sec
        groups[-1].append(note)
    return [tuple(group) for group in groups]


def render_ascii_tab(notes: Iterable[TabNote], *, events_per_system: int = 16) -> str:
    """Render notes as conventional six-line ASCII tab in high-to-low order.

    Each onset group occupies one aligned event column.  A column is at least
    two characters wide and expands to three for two-digit frets, so fret
    numbers never collide with the next event.  Systems contain at most
    ``events_per_system`` event columns and are separated by one blank line.
    """
    if events_per_system < 1:
        raise ValueError("events_per_system must be at least 1")

    valid_notes = (
        note
        for note in notes
        if 0 <= note.string_idx < len(STANDARD_TUNING_MIDI)
        and 0 <= note.fret_idx <= MAX_FRET
    )
    events = _group_tab_notes(valid_notes)
    if not events:
        return "\n".join(f"{label}|" for label in TAB_STRING_LABELS)

    systems: list[str] = []
    for system_start in range(0, len(events), events_per_system):
        system_events = events[system_start : system_start + events_per_system]
        event_cells: list[dict[int, int]] = []
        event_widths: list[int] = []
        for event in system_events:
            cells: dict[int, int] = {}
            for note in event:
                # A malformed caller-supplied chord cannot put two notes on one
                # physical string.  Keep the first deterministic note instead
                # of emitting an ambiguous line with two values in one column.
                cells.setdefault(note.string_idx, note.fret_idx)
            event_cells.append(cells)
            max_digits = max((len(str(fret)) for fret in cells.values()), default=1)
            event_widths.append(max(2, max_digits + 1))
        lines: list[str] = []
        for row_index, label in enumerate(TAB_STRING_LABELS):
            string_idx = len(STANDARD_TUNING_MIDI) - 1 - row_index
            line = [f"{label}|"]
            for cells, width in zip(event_cells, event_widths):
                fret = cells.get(string_idx)
                line.append("-" * width if fret is None else str(fret).ljust(width, "-"))
            line.append("|")
            lines.append("".join(line))
        systems.append("\n".join(lines))
    return "\n\n".join(systems)


__all__ = [
    "GuitarTabError",
    "MAX_FRET",
    "ONSET_GROUP_TOLERANCE_SEC",
    "STANDARD_TUNING_MIDI",
    "TAB_STRING_LABELS",
    "TabNote",
    "midi_to_tab_notes",
    "render_ascii_tab",
]
