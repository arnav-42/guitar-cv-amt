"""Small audio -> fret-region -> string pilot. No dataset labels are used.

Coordinates come from the same perspective-rectification idea as the older
PCA/Hough demos. A visible nut and a numbered wire anchor absolute frets;
MediaPipe fingertips supply regions, not proof of string contact.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
STANDARD_TUNING = (64, 59, 55, 50, 45, 40)  # strings 1..6: high E -> low E
FINGERTIPS = {8: "index", 12: "middle", 16: "ring", 20: "pinkie"}


def note_name(pitch):
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return f"{names[pitch % 12]}{pitch // 12 - 1}"


@dataclass(frozen=True)
class BoardCalibration:
    # Nut upper outer string, numbered-wire upper outer string,
    # numbered-wire lower outer string, nut lower outer string.
    corners: tuple
    far_fret: int = 12
    top_string: int = 6
    tuning: tuple = STANDARD_TUNING
    image_size: tuple | None = None
    # Optional observed wire centers in the reference frame: (fret, (x, y)).
    # The quadrilateral still supplies the two-dimensional board transform;
    # these observations correct its longitudinal scale when apparent string
    # taper is not a faithful perspective cue.
    fret_wire_points: tuple = ()

    def __post_init__(self):
        points = np.asarray(self.corners, dtype=np.float32)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("Calibration needs four finite image points.")
        if not cv2.isContourConvex(points) or abs(cv2.contourArea(points)) < 25:
            raise ValueError("Calibration points must form a nondegenerate, ordered quadrilateral.")
        if type(self.far_fret) is not int or not 1 <= self.far_fret <= 24:
            raise ValueError("far_fret must be an integer in 1..24.")
        if self.top_string not in (1, 6):
            raise ValueError("top_string must be 1 (high E) or 6 (low E).")
        if len(self.tuning) != 6 or any(type(v) is not int or not 0 <= v <= 127 for v in self.tuning):
            raise ValueError("tuning must contain six MIDI integers in string 1..6 order.")
        if self.image_size is not None and (len(self.image_size) != 2 or any(v < 2 for v in self.image_size)):
            raise ValueError("image_size must be [width, height].")
        if self.fret_wire_points:
            try:
                frets = [f for f, _ in self.fret_wire_points]
                points = np.asarray([p for _, p in self.fret_wire_points], dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise ValueError("fret_wire_points need (fret, (x, y)) records.") from exc
            if (points.shape != (len(frets), 2) or not np.isfinite(points).all()
                    or any(type(f) is not int or not 1 <= f <= 24 for f in frets)
                    or frets != sorted(set(frets))):
                raise ValueError("fret_wire_points need unique ascending frets 1..24 and finite image points.")
            measured_x = self.image_to_board(points)[:, 0]
            if not np.isfinite(measured_x).all() or measured_x[0] <= 0 or np.any(np.diff(measured_x) <= 0):
                raise ValueError("Observed fret wires must advance monotonically from the nut.")

    @classmethod
    def from_dict(cls, data):
        wires = tuple((entry["fret"], tuple(entry["point"]))
                      for entry in data.get("fret_wire_points", ()))
        return cls(tuple(tuple(p) for p in data["corners"]), data.get("far_fret", 12),
                   data.get("top_string", 6), tuple(data.get("tuning", STANDARD_TUNING)),
                   tuple(data["image_size"]) if "image_size" in data else None, wires)

    def to_dict(self):
        data = dict(corners=self.corners, far_fret=self.far_fret,
                    top_string=self.top_string, tuning=self.tuning)
        if self.image_size is not None:
            data["image_size"] = self.image_size
        if self.fret_wire_points:
            data["fret_wire_points"] = [dict(fret=fret, point=list(point))
                                        for fret, point in self.fret_wire_points]
        return data

    @property
    def transform(self):
        return cv2.getPerspectiveTransform(np.float32(self.corners),
                                          np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]))

    def image_to_board(self, points):
        return cv2.perspectiveTransform(np.float32(points).reshape(1, -1, 2), self.transform)[0]

    def board_to_image(self, points):
        return cv2.perspectiveTransform(np.float32(points).reshape(1, -1, 2), np.linalg.inv(self.transform))[0]

    def fret_x(self, fret):
        if self.fret_wire_points:
            frets, xs = self._fret_wire_coordinates()
            return self._interpolate_with_extrapolation(float(fret), frets, xs)
        # Exact equal temperament, sharing the old audio helper's relationship.
        ratio = 2 ** (fret / 12)
        return (1 - 1 / ratio) / (1 - 2 ** (-self.far_fret / 12))

    def continuous_fret(self, x):
        if self.fret_wire_points:
            frets, xs = self._fret_wire_coordinates()
            return self._interpolate_with_extrapolation(float(x), xs, frets)
        remaining = 1 - float(x) * (1 - 2 ** (-self.far_fret / 12))
        return -12 * math.log2(remaining) if remaining > 0 else None

    def _fret_wire_coordinates(self):
        points = [point for _, point in self.fret_wire_points]
        measured_x = self.image_to_board(points)[:, 0]
        frets = np.asarray([0, *(f for f, _ in self.fret_wire_points)], dtype=float)
        xs = np.asarray([0., *map(float, measured_x)], dtype=float)
        return frets, xs

    @staticmethod
    def _interpolate_with_extrapolation(value, source, target):
        if value <= source[0]:
            i = 0
        elif value >= source[-1]:
            i = len(source) - 2
        else:
            i = int(np.searchsorted(source, value)) - 1
        return float(target[i] + (value - source[i]) *
                     (target[i + 1] - target[i]) / (source[i + 1] - source[i]))

    @property
    def visible_x_max(self):
        return max(1.01, self.fret_x(self.far_fret) + .01) if self.fret_wire_points else 1.01

    def string_y(self, string):
        return (6 - string) / 5 if self.top_string == 6 else (string - 1) / 5

    def warp(self, image, size=(1024, 256)):
        scale = np.diag([size[0] - 1, size[1] - 1, 1.])
        return cv2.warpPerspective(image, scale @ self.transform, size)


def pitch_candidates(pitch, tuning=STANDARD_TUNING, max_fret=24):
    if type(pitch) is not int or not 0 <= pitch <= 127:
        raise ValueError("pitch must be a MIDI integer in 0..127.")
    return [dict(string=i, fret=pitch - base) for i, base in enumerate(tuning, 1)
            if 0 <= pitch - base <= max_fret]


def track_calibration(reference, image, calibration):
    """Register the visible board to each image; reject failed tracking.

    Optical flow handles small motion; ORB matching handles larger jumps.
    Both estimate a checked homography from the reference board to this frame.
    """
    if reference.shape != image.shape:
        return None, {"status": "failed", "reason": "image_size_changed"}
    if np.array_equal(reference, image):
        return calibration, {"status": "reference", "inliers": None}
    a, b = (cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in (reference, image))
    mask = np.zeros(a.shape, np.uint8)
    # The fretting hand usually occludes the nut end in this small pilot.
    # Track the bodyward half, where wires/inlays remain visible, rather than
    # letting independently moving fingers dominate the planar homography.
    support = calibration.board_to_image([[.45, 0], [1, 0], [1, 1], [.45, 1]])
    cv2.fillConvexPoly(mask, np.rint(support).astype(np.int32), 255)

    def accept(homography, inlier_count, method):
        if homography is None:
            return None
        corners = cv2.perspectiveTransform(np.float32(calibration.corners)[None], homography)[0]
        original = np.float32(calibration.corners)
        area = cv2.contourArea(corners, oriented=True)
        original_area = cv2.contourArea(original, oriented=True)
        if not np.isfinite(corners).all() or area * original_area <= 0:
            return None
        if not .65 < abs(area / original_area) < 1.5:
            return None
        try:
            wires = calibration.fret_wire_points
            if wires:
                points = np.float32([point for _, point in wires])[None]
                moved_wires = cv2.perspectiveTransform(points, homography)[0]
                wires = tuple((fret, tuple(map(float, point)))
                              for (fret, _), point in zip(wires, moved_wires))
            board = replace(calibration, corners=tuple(tuple(map(float, p)) for p in corners),
                            fret_wire_points=wires)
        except ValueError:
            return None
        return board, {"status": "tracked", "method": method, "inliers": int(inlier_count)}

    flow_reason = "too_few_board_features"
    points = cv2.goodFeaturesToTrack(a, 300, .01, 5, mask=mask)
    if points is not None and len(points) >= 10:
        moved, ok, _ = cv2.calcOpticalFlowPyrLK(a, b, points, None, winSize=(31, 31), maxLevel=4)
        if moved is not None:
            back, reverse_ok, _ = cv2.calcOpticalFlowPyrLK(b, a, moved, None, winSize=(31, 31), maxLevel=4)
            if back is not None:
                keep = (ok.ravel() > 0) & (reverse_ok.ravel() > 0)
                keep &= np.linalg.norm(back[:, 0] - points[:, 0], axis=1) < 1.5
                keep &= np.isfinite(moved[:, 0]).all(axis=1)
                if keep.sum() >= 10:
                    homography, inliers = cv2.findHomography(points[keep], moved[keep], cv2.RANSAC, 2.5)
                    if inliers is not None and inliers.sum() >= 10 and inliers.mean() >= .6:
                        accepted = accept(homography, inliers.sum(), "optical_flow")
                        if accepted is not None:
                            return accepted
                flow_reason = "inconsistent_board_motion"
            else:
                flow_reason = "reverse_flow_failed"
        else:
            flow_reason = "flow_failed"

    # Match only features from the calibrated board against the full target.
    # This recovers large translations/rotations that Lucas-Kanade cannot span.
    orb = cv2.ORB_create(nfeatures=1200, fastThreshold=10)
    source_points, source_desc = orb.detectAndCompute(a, mask)
    target_points, target_desc = orb.detectAndCompute(b, None)
    if source_desc is not None and target_desc is not None and len(source_points) >= 10:
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(source_desc, target_desc, k=2)
        good = [first for pair in pairs if len(pair) == 2
                for first, second in [pair] if first.distance < .75 * second.distance]
        unique, used_targets = [], set()
        for match in sorted(good, key=lambda item: item.distance):
            if match.trainIdx not in used_targets:
                unique.append(match)
                used_targets.add(match.trainIdx)
        if len(unique) >= 10:
            source_xy = np.float32([source_points[m.queryIdx].pt for m in unique])
            target_xy = np.float32([target_points[m.trainIdx].pt for m in unique])
            homography, inliers = cv2.findHomography(source_xy, target_xy, cv2.RANSAC, 3.0)
            if inliers is not None and inliers.sum() >= 10 and inliers.mean() >= .5:
                accepted = accept(homography, inliers.sum(), "orb")
                if accepted is not None:
                    return accepted
    return None, {"status": "failed", "reason": flow_reason, "orb": "no_consistent_match"}


def anchor_to_nut(calibration, ends):
    """Move the calibrated board's nut onto a segmented fretboard end.

    ``ends`` are fretboard-mask corners [tl, tr, br, bl] from
    fret_detect_yolo._pca_corners. The mask's nut end is reliable; its body
    end stops wherever the fretting hand covers the board, so only the mask's
    direction is used. Neck length comes from the calibration.
    """
    ends = np.asarray(ends, float)
    corners = np.asarray(calibration.corners, float)
    nut, body = corners[[0, 3]].mean(0), corners[[1, 2]].mean(0)
    a, b = ends[[0, 3]].mean(0), ends[[1, 2]].mean(0)
    if np.linalg.norm(b - nut) < np.linalg.norm(a - nut):
        a, b = b, a
    angle = math.atan2(*(b - a)[::-1]) - math.atan2(*(body - nut)[::-1])
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, s], [-s, c]])
    moved = (corners - nut) @ rotation + a
    wires = tuple((fret, tuple(map(float, (np.asarray(point) - nut) @ rotation + a)))
                  for fret, point in calibration.fret_wire_points)
    return replace(calibration, corners=tuple(tuple(map(float, p)) for p in moved),
                   fret_wire_points=wires)


def locate_board(reference, image, calibration, ends=None, *, nut_tolerance=.35):
    """Track the board, then check its nut against a segmented fretboard.

    Optical flow can follow the fretting hand instead of the neck and shift
    every fret while passing track_calibration's checks. When the tracked nut
    is more than ``nut_tolerance`` fret-1 widths from the mask's nut, the board
    is re-anchored to the mask. ``info["quality"]`` in [0, 1] says how far to
    trust the returned board's fret positions.
    """
    board, info = track_calibration(reference, image, calibration)
    flow = 1. if info["status"] == "reference" else min(1., (info.get("inliers") or 0) / 30)
    corners = np.asarray(calibration.corners, float)
    length = np.linalg.norm(corners[[1, 2]].mean(0) - corners[[0, 3]].mean(0))
    anchored = anchor_to_nut(calibration, ends) if ends is not None else None
    # A mask nut far from the reference nut is a different object, not the guitar.
    if anchored and np.linalg.norm(np.subtract(anchored.corners[0], corners[0])) > .25 * length:
        anchored = None
    if anchored is None:
        return board, dict(info, nut="not_detected", quality=.5 * flow if board else 0.)
    if board is not None:
        fret_one = calibration.fret_x(1) * length
        error = float(np.linalg.norm(np.mean(np.subtract(board.corners, anchored.corners)[[0, 3]], 0)))
        if error <= nut_tolerance * fret_one:
            return board, dict(info, nut="agrees", nut_error_px=round(error, 1), quality=flow)
        info = dict(info, nut_error_px=round(error, 1))
    return anchored, dict(info, nut="reanchored", quality=.6)


class FingertipDetector:
    """Reuse the old hand demo's model/landmark IDs, without webcam mirroring."""
    def __init__(self, model_path=ROOT / "main/models/hands/hand_landmarker.task"):
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        self.mp = mp
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Hand model not found: {model_path}")
        self.model = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE, num_hands=2,
            min_hand_detection_confidence=.3, min_hand_presence_confidence=.3))

    def close(self):
        self.model.close()

    def detect_hands(self, image, calibration):
        """All 21 landmarks per hand in full-image pixels, with handedness score."""
        # A board crop makes a small fretting hand visible to the palm detector.
        h, w = image.shape[:2]
        corners = np.asarray(calibration.corners)
        board_width = max(np.linalg.norm(corners[0] - corners[3]),
                          np.linalg.norm(corners[1] - corners[2]))
        pad = max(100, board_width * 2.5)
        x0, y0 = np.maximum([0, 0], np.floor(corners.min(0) - pad)).astype(int)
        x1, y1 = np.minimum([w, h], np.ceil(corners.max(0) + pad)).astype(int)
        crop = np.ascontiguousarray(image[y0:y1, x0:x1])
        if crop.size == 0:
            return []
        result = self.model.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=crop))
        hands = []
        for hand_id, landmarks in enumerate(result.hand_landmarks):
            points = np.array([[x0 + p.x * crop.shape[1], y0 + p.y * crop.shape[0]] for p in landmarks])
            score = result.handedness[hand_id][0].score if result.handedness else 1.
            hands.append(dict(hand_id=hand_id, score=float(score), points=points))
        return hands

    def detect(self, image, calibration):
        return [dict(finger=name, hand_id=hand["hand_id"], image_point=hand["points"][i].tolist())
                for hand in self.detect_hands(image, calibration) for i, name in FINGERTIPS.items()]


def fingertip_regions(tips, calibration):
    regions = []
    for tip in tips:
        x, y = map(float, calibration.image_to_board([tip["image_point"]])[0])
        # Keep near-board fingertips; discard the picking hand and off-board tips.
        if not -.02 <= x <= calibration.visible_x_max or not -.65 <= y <= 1.65:
            continue
        fret = calibration.continuous_fret(x)
        if fret is None:
            continue
        regions.append(dict(**tip, board_point=[x, y], continuous_fret=fret,
                            fret_region=max(1, int(math.ceil(fret))),
                            near_strings=bool(-.15 <= y <= 1.15)))
    return regions


def observed_string_lines(image, calibration):
    """Refine cross-board rows with the existing evidence-based string detector.

    Missing lines retain the explicitly calibrated outer-string interpolation;
    that fallback is identified in the result, not called detected strings.
    """
    from main.video.fretboard.string_evidence import detect_strings
    warp = calibration.warp(image)
    gray = cv2.cvtColor(warp, cv2.COLOR_RGB2GRAY)
    lines = detect_strings(gray, np.ones(gray.shape, np.uint8), 6)
    # Only accept a comb consistent with the user's two anchored outer strings.
    if len(lines) == 6 and abs(lines[0]["b"]) < .10 and abs(lines[-1]["b"] - 1) < .10:
        return lines, "detected"
    return [dict(a=0., b=i / 5) for i in range(6)], "calibrated_spacing"


def fuse_pitch(pitch, regions, calibration, *, use_string_evidence=False,
               string_lines=None, region_tolerance=.75):
    """Resolve only a unique supported candidate; preserve alternatives.

    Default: x/fret region chooses the string implied by the audio pitch.
    Optional: require a fingertip within 0.75 string spacing of that string.
    Open-string candidates remain plausible even without a nearby fingertip.
    """
    if not math.isfinite(region_tolerance) or region_tolerance < 0:
        raise ValueError("region_tolerance must be nonnegative and finite.")
    candidates = pitch_candidates(pitch, calibration.tuning)
    for candidate in candidates:
        fret = candidate["fret"]
        candidate.update(visible=fret <= calibration.far_fret, supported=False, fingers=[])
        if fret == 0:
            candidate.update(supported=True, reason="open_string_not_visually_excludable")
            continue
        if not candidate["visible"]:
            candidate["reason"] = "outside_calibrated_view"
            continue
        for region in regions:
            f = region["continuous_fret"]
            distance = max(fret - 1 - f, f - fret, 0.)
            if distance > region_tolerance:
                continue
            if use_string_evidence:
                index = 6 - candidate["string"] if calibration.top_string == 6 else candidate["string"] - 1
                line = string_lines[index] if string_lines else dict(a=0., b=index / 5)
                y = line["a"] * (region["board_point"][0] - .5) + line["b"]
                if abs(region["board_point"][1] - y) > .75 / 5:
                    continue
            candidate["fingers"].append(region["finger"])
        candidate["supported"] = bool(candidate["fingers"])
        candidate["reason"] = "fingertip_region_matches" if candidate["supported"] else "no_matching_fingertip_region"
    supported = [c for c in candidates if c["supported"]]
    # Out-of-view alternatives cannot be ruled out by this calibrated image.
    unknown = [c for c in candidates if not c["visible"]]
    if len(supported) == 1 and not unknown:
        selected = {k: supported[0][k] for k in ("string", "fret")}
        status = "selected"
    else:
        selected = None
        status = "ambiguous" if supported or unknown else "no_visual_match"
    return dict(pitch=pitch, note=note_name(pitch), status=status, selected=selected,
                candidates=candidates, region_tolerance=region_tolerance,
                mode="fret_region_and_string" if use_string_evidence else "fret_region")


def fuse_pitches(pitches, regions, calibration, **kwargs):
    """Keep every active MT3 pitch; check simultaneous string conflicts.

    This first chord-capable pilot resolves pitches independently, then rejects
    selections that require two different pitches on the same physical string.
    It does not guess a chord fingering when alternatives remain ambiguous.
    """
    pitches = sorted(set(pitches))
    notes = [fuse_pitch(p, regions, calibration, **kwargs) for p in pitches]
    used_strings = {}
    for note in notes:
        if note["selected"]:
            used_strings.setdefault(note["selected"]["string"], []).append(note)
    for group in used_strings.values():
        if len(group) > 1:
            for note in group:
                note.update(selected=None, status="string_conflict")
    selected = [dict(pitch=n["pitch"], note=n["note"], **n["selected"])
                for n in notes if n["selected"]]
    if not notes:
        status = "no_active_audio_notes"
    elif len(notes) == 1:
        status = notes[0]["status"]
    elif len(selected) == len(notes):
        status = "selected_multiple"
    elif selected:
        status = "partly_resolved"
    else:
        status = "unresolved_multiple"
    return dict(pitch=pitches[0] if len(pitches) == 1 else None, pitches=pitches,
                note=" + ".join(n["note"] for n in notes), status=status,
                selected=notes[0]["selected"] if len(notes) == 1 else None,
                selected_positions=selected, note_results=notes,
                candidates=[dict(pitch=n["pitch"], note=n["note"], **c)
                            for n in notes for c in n["candidates"]])


def estimate_audio_pitch(samples, sample_rate, *, min_probability=.65):
    """librosa.pyin monophonic estimate, independent of vision and labels."""
    import librosa
    y = np.asarray(samples, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if len(y) < 2048 or not np.isfinite(y).all():
        return dict(pitch=None, status="audio_window_too_short_or_invalid")
    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms < 1e-4:
        return dict(pitch=None, status="quiet_audio", rms=rms)
    # Work at 22050 Hz; a 4096-sample frame supports low guitar notes.
    rate = 22050
    if sample_rate != rate:
        y = librosa.resample(y, orig_sr=sample_rate, target_sr=rate)
    f0, voiced, probability = librosa.pyin(y, sr=rate, fmin=65., fmax=1400.,
                                         frame_length=4096, hop_length=256, center=True)
    valid = voiced & np.isfinite(f0) & (probability >= min_probability)
    coverage = float(valid.mean())
    if valid.sum() < 3 or coverage < .35:
        return dict(pitch=None, status="uncertain_audio", voiced_fraction=coverage, rms=rms)
    midi = librosa.hz_to_midi(f0[valid])
    center = float(np.median(midi))
    pitch = int(round(center))
    spread = float(np.median(abs(midi - center)) * 100)
    status = "estimated" if spread <= 35 else "unstable_pitch"
    return dict(pitch=pitch if status == "estimated" else None, status=status,
                note=note_name(pitch), frequency_hz=float(librosa.midi_to_hz(center)),
                detuning_cents=float((center - pitch) * 100),
                probability=float(np.median(probability[valid])), voiced_fraction=coverage,
                spread_cents=spread, rms=rms, assumption="one_sounding_note")


def draw_fusion(image, calibration, regions, fusion, string_lines=None):
    """Show calibrated frets, fingertip regions, and a selected candidate."""
    result = image.copy()
    for fret in range(calibration.far_fret + 1):
        endpoints = calibration.board_to_image([[calibration.fret_x(fret), 0],
                                               [calibration.fret_x(fret), 1]])
        a, b = np.rint(endpoints).astype(int)
        cv2.line(result, tuple(a), tuple(b), (240, 185, 60), 1, cv2.LINE_AA)
        cv2.putText(result, str(fret), tuple(a - [0, 8]), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 240, 140), 1)
    for index in range(6):
        line = string_lines[index] if string_lines else dict(a=0., b=index / 5)
        endpoints = calibration.board_to_image([[0, line["b"] - line["a"] / 2],
                                               [1, line["b"] + line["a"] / 2]])
        a, b = np.rint(endpoints).astype(int)
        cv2.line(result, tuple(a), tuple(b), (100, 205, 230), 1, cv2.LINE_AA)
    for region in regions:
        point = tuple(np.rint(region["image_point"]).astype(int))
        cv2.circle(result, point, 8, (200, 120, 250), -1)
        cv2.putText(result, f'{region["finger"]}: region {region["fret_region"]}',
                    (point[0] - 60, point[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
    positions = fusion.get("selected_positions")
    if positions is None:
        positions = [fusion["selected"]] if fusion.get("selected") else []
    label = f'{fusion.get("note", "No reliable note")} | {fusion["status"]}'
    for selected in positions:
        fret, string = selected["fret"], selected["string"]
        x = (calibration.fret_x(max(0, fret - 1)) + calibration.fret_x(fret)) / 2
        p = tuple(np.rint(calibration.board_to_image([[x, calibration.string_y(string)]])[0]).astype(int))
        cv2.circle(result, p, 14, (80, 255, 140), 3)
        if len(positions) == 1:
            label += f' | string {string}, fret {fret}'
        else:
            cv2.putText(result, f'{selected.get("note", "")} S{string}/F{fret}',
                        (p[0] + 15, p[1]), cv2.FONT_HERSHEY_SIMPLEX, .45, (80, 255, 140), 1)
    cv2.rectangle(result, (0, 0), (min(result.shape[1], 1300), 42), (20, 25, 35), -1)
    cv2.putText(result, label, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, .7, (245, 245, 245), 2)
    return result
