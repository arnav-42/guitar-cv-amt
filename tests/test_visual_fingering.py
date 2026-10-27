from __future__ import annotations

import pytest
import sys
import types

import numpy as np

from pipeline.guitar_tab import (
    ScoreNote,
    ScoreNoteId,
    assign_tab_notes,
    midi_to_tab_notes,
    render_ascii_tab,
)
from pipeline import visual_fingering
from pipeline.visual_fingering import (
    StringRowContact,
    VisualEvent,
    _event_alignment_score,
    _landmark_confidence,
    _mask_compatibility,
    _merge_compatible_yolo_proposals,
    infer_visual_alignment,
)


def _note(index: int, onset: float, pitch: int) -> ScoreNote:
    return ScoreNote(onset, onset + 0.25, pitch, index)


def test_string_row_contacts_recover_forward_alignment_and_exact_positions():
    strings = (0, 1, 2, 3, 4, 5, 3, 5)
    pitches = (45, 52, 54, 61, 67, 69, 65, 72)
    notes = tuple(
        _note(index, float(index), pitch)
        for index, pitch in enumerate(pitches)
    )
    events = tuple(
        VisualEvent(
            note.onset_sec,
            (StringRowContact(row, confidence=1.0, finger="index"),),
            3,
        )
        for note, row in zip(notes, strings)
    )

    result = infer_visual_alignment(notes, events)

    assert result.accepted
    assert result.alignment is not None
    assert result.alignment.mode == "string_row"
    assert result.alignment.string_order == tuple(range(6))
    assert result.position_likelihoods[ScoreNoteId(6)]
    assert all(
        position.fret_idx > 0
        for values in result.position_likelihoods.values()
        for position in values
    )


def test_string_row_contacts_recover_reversed_order():
    strings = (0, 1, 2, 3, 4, 5, 3, 5)
    pitches = (45, 52, 54, 61, 67, 69, 65, 72)
    notes = tuple(
        _note(index, float(index), pitch)
        for index, pitch in enumerate(pitches)
    )
    order = tuple(reversed(range(6)))
    events = tuple(
        VisualEvent(
            note.onset_sec,
            (StringRowContact(order.index(string), confidence=1.0),),
            2,
        )
        for note, string in zip(notes, strings)
    )

    result = infer_visual_alignment(notes, events)

    assert result.accepted
    assert result.alignment is not None
    assert result.alignment.string_order == order
    assert result.alignment.mode == "string_row"


def test_ambiguous_string_row_alignment_abstains():
    notes = (_note(0, 0.0, 60),)
    events = (VisualEvent(0.0, (StringRowContact(2.5),), 1),)

    result = infer_visual_alignment(notes, events)

    assert not result.accepted
    assert result.position_likelihoods == {}


def test_open_string_contacts_are_not_alignment_evidence():
    notes = (_note(0, 0.0, 40),)
    events = (VisualEvent(0.0, (StringRowContact(5.0),), 1),)

    result = infer_visual_alignment(notes, events)

    assert not result.accepted
    assert result.position_likelihoods == {}

def test_chord_match_uses_each_contact_at_most_once():
    notes = (_note(0, 0.0, 60), _note(1, 0.0, 64))
    event = VisualEvent(
        0.0,
        (StringRowContact(2.0),),
        1,
    )

    score, support, edges = _event_alignment_score(notes, event, tuple(range(6)))

    assert score == pytest.approx(0.5)
    assert support == 1
    assert len(edges) == 1
    assert edges[0][2] == 0

def test_likelihoods_use_only_winning_one_to_one_edges():
    notes = (
        _note(0, 0.0, 60),
        _note(1, 0.0, 64),
        _note(2, 1.0, 45),
    )
    events = (
        VisualEvent(0.0, (StringRowContact(2.0),), 1),
        VisualEvent(1.0, (StringRowContact(0.0),), 1),
    )

    result = infer_visual_alignment(notes, events)

    assert result.accepted
    assert result.alignment is not None
    assert result.alignment.string_order == tuple(range(6))
    assert set(result.position_likelihoods) & {ScoreNoteId(0), ScoreNoteId(1)}
    assert not ({ScoreNoteId(0), ScoreNoteId(1)} <= set(result.position_likelihoods))


def test_strong_visual_evidence_selects_an_isolated_high_fret_position():
    note = _note(0, 0.0, 64)
    assigned = assign_tab_notes(
        (note,),
        {note.identity: {(0, 24): 1.0}},
    )

    assert [(item.string_idx, item.fret_idx) for item in assigned] == [(0, 24)]

def test_native_yolo_proposals_passes_rgb_pil_source():
    image_module = pytest.importorskip("PIL.Image")
    seen_sources = []

    class FakeModel:
        def predict(self, source, **_kwargs):
            seen_sources.append(source)
            return []

    rgb = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    assert visual_fingering._native_yolo_proposals(
        FakeModel(),
        rgb,
        types.SimpleNamespace(),
    ) == []
    assert len(seen_sources) == 1
    assert isinstance(seen_sources[0], image_module.Image)
    assert np.array_equal(np.asarray(seen_sources[0]), rgb)


def test_likelihood_assignment_preserves_invalid_evidence_and_changes_exact_cell(tmp_path):
    pretty_midi = pytest.importorskip("pretty_midi")
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=24)
    instrument.notes.append(pretty_midi.Note(100, 64, 0.0, 0.5))
    midi.instruments.append(instrument)


    path = tmp_path / "one.mid"
    midi.write(str(path))

    baseline = midi_to_tab_notes(path)
    assert [(note.string_idx, note.fret_idx) for note in baseline] == [(5, 0)]
    score_note = ScoreNoteId(0)
    evidence = {score_note: {(5, 0): 0.0, (4, 5): 1.0, (99, 99): 1.0}}
    changed = midi_to_tab_notes(path, evidence)
    assert [(note.string_idx, note.fret_idx) for note in changed] == [(4, 5)]
    assert "5" in render_ascii_tab(changed).splitlines()[1]

    invalid = midi_to_tab_notes(path, {score_note: {(4, 5): 0.0, (4, 4): 1.0}})
    assert invalid == baseline
    assert midi_to_tab_notes(path, {score_note: {(4, 4): 1.0}}) == baseline
    assert midi_to_tab_notes(path, {}) == baseline

def test_chord_evidence_respects_unique_strings_and_duplicate_identity():
    notes = (
        _note(0, 0.0, 60),
        _note(1, 0.0, 64),
        _note(2, 0.0, 67),
    )
    evidence = {
        ScoreNoteId(0): {(3, 5): 1.0},
        ScoreNoteId(1): {(4, 5): 1.0},
        ScoreNoteId(2): {(5, 3): 1.0},
    }
    assigned = assign_tab_notes(notes, position_likelihoods=evidence)
    assert {(note.pitch_midi, note.string_idx, note.fret_idx) for note in assigned} == {
        (60, 3, 5),
        (64, 4, 5),
        (67, 5, 3),
    }

    duplicates = (_note(0, 0.0, 64), _note(1, 0.0, 64))
    duplicate_evidence = {
        ScoreNoteId(0): {(4, 5): 1.0},
        ScoreNoteId(1): {(3, 9): 1.0},
    }
    duplicate_assigned = assign_tab_notes(duplicates, position_likelihoods=duplicate_evidence)
    assert {(note.note_id, note.string_idx, note.fret_idx) for note in duplicate_assigned} == {
        (ScoreNoteId(0), 4, 5),
        (ScoreNoteId(1), 3, 9),
    }

def test_visual_extraction_uses_video_timestamps_and_bounded_seeks(
    tmp_path, monkeypatch
):
    class Capture:
        def __init__(self):
            self.seeks = []
            self.reads = 0

        def isOpened(self):
            return True

        def get(self, _prop):
            return 2000.0

        def set(self, _prop, value):
            self.seeks.append(value)
            return True

        def read(self):
            self.reads += 1
            return True, np.zeros((8, 8, 3), dtype=np.uint8)

        def release(self):
            return None

    capture = Capture()
    fake_cv2 = types.SimpleNamespace(
        CAP_PROP_FPS=1,
        CAP_PROP_POS_FRAMES=2,
        COLOR_BGR2RGB=3,
        VideoCapture=lambda _path: capture,
        cvtColor=lambda frame, _code: frame,
    )

    class FakeRunningMode:
        VIDEO = "VIDEO"

    class FakeOptions:
        created = []

        def __init__(self, **kwargs):
            self.running_mode = kwargs["running_mode"]
            self.kwargs = kwargs
            self.__class__.created.append(self)

    class FakeLandmarker:
        created = []
        timestamps = []

        @classmethod
        def create_from_options(cls, options):
            cls.created.append(options)
            return cls()

        def detect_for_video(self, _image, timestamp_ms):
            self.__class__.timestamps.append(timestamp_ms)
            landmarks = [types.SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]
            return types.SimpleNamespace(hand_landmarks=(landmarks,))

        def close(self):
            return None

    class FakeImage:
        def __init__(self, **kwargs):
            self.data = kwargs["data"]

    fake_mp = types.SimpleNamespace(
        tasks=types.SimpleNamespace(
            BaseOptions=lambda **kwargs: kwargs,
            vision=types.SimpleNamespace(
                RunningMode=FakeRunningMode,
                HandLandmarkerOptions=FakeOptions,
                HandLandmarker=FakeLandmarker,
            ),
        ),
        ImageFormat=types.SimpleNamespace(SRGB="SRGB"),
        Image=FakeImage,
    )
    fake_ultralytics = types.SimpleNamespace(YOLO=lambda _path: object())
    fake_atlas = types.SimpleNamespace(
        quality={"status": "grid", "lattice_accepted": False},
        target_x=np.array([0.0, 1023.0]),
        target_y=np.arange(8, dtype=float),
        image_to_grid=lambda points: np.repeat(
            np.array([[512.0, 4.0]]), len(points), axis=0
        ),
    )
    fake_rectify = types.SimpleNamespace(
        select_atlas=lambda *_args, **_kwargs: (fake_atlas, None, ()),
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "mediapipe", fake_mp)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    monkeypatch.setitem(sys.modules, "pipeline.rectify_atlas", fake_rectify)
    monkeypatch.setattr(
        visual_fingering,
        "_native_yolo_proposals",
        lambda *_args: [("mask", np.ones((8, 8), dtype=np.uint8))],
    )
    yolo_path = tmp_path / "weights.pt"
    yolo_path.write_bytes(b"x" * 2048)
    hand_path = tmp_path / "hand.task"
    hand_path.write_bytes(b"task")

    extracted = visual_fingering.extract_visual_events(
        tmp_path / "video.mp4",
        (_note(0, 1.0, 45),),
        yolo_weights=yolo_path,
        hand_model=hand_path,
    )

    assert capture.seeks == [1998, 1999, 2000, 2001, 2002]
    assert capture.reads == len(capture.seeks)
    assert FakeOptions.created[0].running_mode == FakeRunningMode.VIDEO
    assert FakeOptions.created[0].kwargs["min_hand_detection_confidence"] == 0.05
    assert FakeOptions.created[0].kwargs["min_hand_presence_confidence"] == 0.05
    assert FakeOptions.created[0].kwargs["min_tracking_confidence"] == 0.05
    assert FakeLandmarker.timestamps == [999, 1000, 1001, 1002, 1003]
    assert extracted.frames_inspected == 5
    assert extracted.config["mediapipe_thresholds"] == {
        "min_hand_detection_confidence": 0.05,
        "min_hand_presence_confidence": 0.05,
        "min_tracking_confidence": 0.05,
    }
    assert extracted.frames_with_grid == 5
    assert extracted.events[0].contacts
    assert all(0.0 <= contact.row <= 5.0 for contact in extracted.events[0].contacts)
    assert all(
        summary["quality"]["lattice_accepted"] is False
        for summary in extracted.quality_summaries
    )
@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (types.SimpleNamespace(), 1.0),
        (types.SimpleNamespace(presence=None, visibility=None), 1.0),
        (types.SimpleNamespace(presence=0.7, visibility=0.2), 0.7),
        (types.SimpleNamespace(presence=None, visibility=0.4), 0.4),
        (types.SimpleNamespace(presence=float("nan"), visibility=0.4), 0.4),
        (types.SimpleNamespace(presence=-0.2, visibility=None), 0.0),
        (types.SimpleNamespace(presence=float("inf"), visibility=None), 0.0),
        (types.SimpleNamespace(presence="bad", visibility=None), 0.0),
    ],
)
def test_landmark_confidence_handles_omitted_and_invalid_metadata(landmark, expected):
    assert _landmark_confidence(landmark) == expected


def test_split_neck_masks_merge_only_when_spatially_compatible():
    first = np.zeros((120, 400), dtype=np.uint8)
    second = np.zeros_like(first)
    distant = np.zeros_like(first)
    first[40:80, 20:150] = 255
    second[40:80, 150:280] = 255
    distant[40:80, 340:399] = 255

    accepted, metrics = _mask_compatibility(first, second, np)
    assert accepted
    assert metrics["transverse_overlap_ratio"] == pytest.approx(1.0)
    rejected, _ = _mask_compatibility(first, distant, np)
    assert not rejected

    augmented, provenance = _merge_compatible_yolo_proposals(
        [("left", first), ("right", second), ("distant", distant)]
    )
    assert [name for name, _ in augmented] == ["left", "right", "distant", "merged_left_right"]
    assert provenance["merged_left_right"]["merged_from"] == ["left", "right"]
    assert np.array_equal(augmented[-1][1], np.logical_or(first > 0, second > 0).astype(np.uint8) * 255)
