"""Audio-guided, event-level fret estimation with transcribed pitches.

Follows Nakayama et al., "A Fretting Position Estimation System using Audio
Guided Guitar Performance Video Analysis" (COMPSAC 2026), on this project's
calibrated board geometry, then adds the step that paper leaves out:

1. Transcribed note onsets (YourMT3) define playing events; frames within
   +-delta of an event form its window. Close onsets merge into one event.
2. Points along each finger's TIP-DIP-PIP-MCP chain, q(n) for n in [0, 3],
   map to the nearest fret wire on the body side (ceil of continuous fret).
   Points off the calibrated board are invalid.
3. A per-finger fixed prior n is tried first. Only when it yields no valid
   frame does an adaptive search pick n by S(n) = sum c - lambda*std - eta*M.
4. Confidence-weighted smoothing (radius r, decay gamma) and a weighted mode
   give one fret per finger per event.
5. New here: each event's pitches choose strings. A fretted pitch is assigned
   only where a finger's fused fret equals pitch - open-string pitch.

Archive labels are read only for prior fitting on development clips and for
scoring; they never enter predictions on evaluation clips.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from main.video.fretboard.audio_visual_fusion import (BoardCalibration, FingertipDetector, note_name,
                                          pitch_candidates, track_calibration)
from main.video.fretboard.tablature_dataset import FINGERS, TablatureDataset


def load_audio_notes(path):
    """CSV: onset,offset,pitch; note times are full WAV seconds, MIDI pitches."""
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"onset", "offset", "pitch"} <= set(reader.fieldnames or []):
            raise ValueError("Audio note CSV needs onset,offset,pitch (MIDI).")
        notes = []
        for row in reader:
            onset, offset, pitch = float(row["onset"]), float(row["offset"]), int(row["pitch"])
            if not np.isfinite([onset, offset]).all() or not 0 <= onset < offset or not 0 <= pitch <= 127:
                raise ValueError("Invalid audio note event.")
            notes.append(dict(onset=onset, offset=offset, pitch=pitch))
    return notes


ROOT = Path(__file__).resolve().parents[3]
# MediaPipe landmark ids per finger, ordered TIP, DIP, PIP, MCP.
CHAINS = {"index": (8, 7, 6, 5), "middle": (12, 11, 10, 9), "ring": (16, 15, 14, 13), "pinkie": (20, 19, 18, 17)}
GRID = tuple(round(v, 2) for v in np.linspace(0, 2, 11))
BARRE_GRID = GRID + (2.2, 2.4, 2.6, 2.8, 3.0)
PAPER_PRIORS = dict(index=1.3, middle=.7, ring=1.1, pinkie=0.)


@dataclass(frozen=True)
class Settings:
    delta: float = .2           # event window starts delta before the onset, seconds
    delta_after: float = .2     # and ends this long after it (paper: symmetric +-delta)
    min_interval: float = .25   # onsets closer than this form one event
    lam: float = .4             # adaptive penalty on fret instability
    eta: float = .2             # adaptive penalty on invalid frames
    radius: int = 1             # smoothing radius in frames
    gamma: float = .8           # smoothing decay per frame
    y_margin: float = .25       # accepted contact band beyond the outer strings
    open_score: float = .25     # open strings cannot be excluded visually
    audio_only_score: float = .1
    fret_tolerance: int = 1     # pitch may correct a visual fret off by this much


def chain_point(points, n):
    """Paper eq. (2): piecewise-linear point along TIP-DIP-PIP-MCP."""
    k = min(int(n), 2)
    return (1 - (n - k)) * points[k] + (n - k) * points[k + 1]


def point_fret(board_point, calibration, settings):
    """Paper eq. (3) on our board: body-side wire, or None when off the board."""
    x, y = map(float, board_point)
    if not 0 < x <= calibration.visible_x_max or not -settings.y_margin <= y <= 1 + settings.y_margin:
        return None, 0.
    continuous = calibration.continuous_fret(x)
    if continuous is None:
        return None, 0.
    fret = max(1, math.ceil(continuous))
    # Closer to the wire and inside the outer strings is more trustworthy.
    proximity = 1 - .5 * min(1., fret - continuous)
    outside = max(0., -y, y - 1) / settings.y_margin
    return fret, proximity * (1 - .5 * outside)


def is_barre(points, board_points, calibration):
    """Index TIP/DIP/PIP on the board and roughly parallel to the frets."""
    inside = sum(0 < x <= calibration.visible_x_max and -.05 <= y <= 1.05
                 for x, y in board_points[:3])
    corners = np.asarray(calibration.corners, float)
    along = corners[1] - corners[0]
    axis = points[0] - points[2]
    cosine = abs(along @ axis) / (np.linalg.norm(along) * np.linalg.norm(axis) + 1e-9)
    return inside >= 3 and cosine < math.sin(math.radians(20))


def frame_observation(hands, calibration, settings):
    """Pick the fretting hand and precompute its finger chains in board space."""
    best = None
    for hand in hands:
        chains = {}
        for finger, ids in CHAINS.items():
            points = hand["points"][list(ids)]
            chains[finger] = dict(points=points, board=calibration.image_to_board(points))
        tips_on_board = sum(point_fret(c["board"][0], calibration, settings)[0] is not None
                            for c in chains.values())
        key = (tips_on_board, hand["score"])
        if tips_on_board and (best is None or key > best[0]):
            best = key, dict(score=hand["score"], chains=chains)
    if best is None:
        return None
    observation = best[1]
    index = observation["chains"]["index"]
    observation["barre"] = bool(is_barre(index["points"], index["board"], calibration))
    return observation


def frame_estimate(observation, calibration, finger, n, settings):
    if observation is None:
        return None, 0.
    chain = observation["chains"][finger]
    board = calibration.image_to_board([chain_point(chain["points"], n)])[0]
    fret, quality = point_fret(board, calibration, settings)
    return fret, observation["score"] * quality


def frame_estimates(frames, finger, n, settings):
    """Per-frame (fret, confidence), weighted by how well each frame's board was located.

    Quality is relative to the window's best board, so it decides which frames
    win the vote without lowering the event's confidence against the fixed
    open-string and audio-only scores in assign_strings.
    """
    best = max((frame.get("board_quality", 1.) for frame in frames), default=1.) or 1.
    return [(fret, c * frame.get("board_quality", 1.) / best) for frame in frames
            for fret, c in [frame_estimate(frame["observation"], frame["calibration"], finger, n, settings)]]


def fuse(estimates, settings, weighted=True, smooth=True):
    """Confidence-aware smoothing, then a confidence-weighted mode."""
    valid = [(t, f, c if weighted else 1.) for t, (f, c) in enumerate(estimates) if f is not None]
    if not valid:
        return None
    smoothed = []
    for t, f, c in valid:
        if smooth:
            votes = defaultdict(float)
            for u, g, d in valid:
                if abs(u - t) <= settings.radius:
                    votes[g] += d * settings.gamma ** abs(u - t)
            f = max(votes, key=lambda k: (votes[k], -k))
        smoothed.append((f, c))
    totals = defaultdict(float)
    for f, c in smoothed:
        totals[f] += c
    fret = max(totals, key=lambda k: (totals[k], -k))
    return dict(fret=int(fret), confidence=totals[fret] / sum(totals.values()) *
                max(c for f, c in smoothed if f == fret), valid_frames=len(valid))


def estimate_finger(event, finger, prior, settings, *, mode="hybrid"):
    """Modes mirror the paper's ablation: tip_single, single, vote, fusion, hybrid."""
    frames = event["frames"]
    if mode in ("tip_single", "single"):
        frames = [frames[event["primary"]]]
    n = 0. if mode == "tip_single" else prior
    estimates = frame_estimates(frames, finger, n, settings)
    result = fuse(estimates, settings, weighted=mode not in ("vote",), smooth=mode in ("fusion", "hybrid"))
    if result:
        return dict(result, n=n, branch="fixed_prior")
    if mode != "hybrid":
        return None
    barre = finger == "index" and any(f["observation"] and f["observation"]["barre"] for f in frames)
    best = None
    for n in BARRE_GRID if barre else GRID:
        estimates = frame_estimates(frames, finger, n, settings)
        valid =[(f, c) for f, c in estimates if f is not None]
        if not valid:
            continue
        score = (sum(c for _, c in valid) - settings.lam * float(np.std([f for f, _ in valid]))
                 - settings.eta * (len(estimates) - len(valid)))
        if best is None or score > best[0]:
            best = score, n, estimates
    if best is None:
        return None
    return dict(fuse(best[2], settings), n=best[1], branch="adaptive_fallback")


def assign_strings(pitches, fingers, tuning, settings):
    """Choose one (string, fret, finger) per pitch with distinct strings/fingers.

    Exhaustive over at most six pitches. A fretted choice needs a finger whose
    fused fret equals it (a barre index covers any string at its fret). Ties
    between best assignments leave a pitch ambiguous rather than guessed.
    """
    options = []
    for pitch in pitches:
        candidates = pitch_candidates(pitch, tuning)
        choices = [(None, None, 0.)]
        for c in candidates:
            if c["fret"] == 0:
                choices.append((c, None, settings.open_score))
            for finger, est in fingers.items():
                # Same-pitch positions sit 4-5 frets apart, so a one-fret visual
                # error still points at a single candidate; it scores lower.
                off = abs(est["fret"] - c["fret"]) if est and c["fret"] else None
                if off is not None and off <= settings.fret_tolerance:
                    choices.append((c, finger, est["confidence"] * (1 if off == 0 else .5)))
            if len(candidates) == 1 and c["fret"]:
                choices.append((c, None, settings.audio_only_score))
        options.append(choices)
    best, ties = -1., []
    for combo in itertools.product(*options):
        strings = [c["string"] for c, _, _ in combo if c]
        used = [f for _, f, _ in combo if f and not (f == "index" and fingers[f].get("barre"))]
        if len(set(strings)) < len(strings) or len(set(used)) < len(used):
            continue
        score = sum(s for _, _, s in combo)
        if score > best + 1e-9:
            best, ties = score, [combo]
        elif abs(score - best) <= 1e-9:
            ties.append(combo)
    results = []
    for i, pitch in enumerate(pitches):
        # Only a different position is ambiguous; which finger backs it is not.
        picks = {(c["string"], c["fret"]) if c else None for c, _, _ in (t[i] for t in ties)}
        c, finger, _ = ties[0][i]
        corrected = bool(finger and fingers[finger]["fret"] != c["fret"])
        status = ("ambiguous" if len(picks) > 1 else "unresolved" if c is None else
                  "audio_corrected" if corrected else "visual_match" if finger else
                  "open_string" if c["fret"] == 0 else "audio_only")
        results.append(dict(pitch=pitch, note=note_name(pitch), status=status,
                            string=c["string"] if c and status != "ambiguous" else None,
                            fret=c["fret"] if c and status != "ambiguous" else None,
                            finger=finger if status in ("visual_match", "audio_corrected") else None,
                            candidates=pitch_candidates(pitch, tuning)))
    return results


ANCHOR_STATUSES = ("visual_match", "audio_corrected")
CONTEXT_STATUSES = ("unresolved", "ambiguous")


def infer_from_context(events, tuning, mode="hybrid", max_gap=3., open_cost=3.):
    """Place unresolved/ambiguous notes near where the hand was on neighbouring notes.

    Anchors are visually confirmed fretted notes. For each uncertain note, the
    hand's fret (and string) is the time-weighted mean of the closest anchor
    event before and after it, within max_gap seconds. The candidate nearest
    that fret wins; an open string costs at most open_cost frets, since it is
    reachable from any hand position. Strings already used by the same event are
    skipped. Chosen notes get status "context_inferred" (old status kept in
    "previous_status"); notes with no anchor in reach stay as they were.
    """
    key = f"{mode}_strings"
    anchors = [(e["time"], [a for a in e[key] if a["status"] in ANCHOR_STATUSES and a["fret"]])
               for e in events]
    anchors = [(t, notes) for t, notes in anchors if notes]
    for event in events:
        t = event["time"]
        before = [a for a in anchors if t - max_gap <= a[0] <= t]
        after = [a for a in anchors if t <= a[0] <= t + max_gap]
        sides = [s for s in (before[-1] if before else None, after[0] if after else None) if s]
        if not sides:
            continue
        weights = [1 / (abs(at - t) + .1) for at, _ in sides]
        hand_fret = sum(w * np.mean([a["fret"] for a in notes]) for w, (_, notes) in zip(weights, sides)) / sum(weights)
        hand_string = sum(w * np.mean([a["string"] for a in notes]) for w, (_, notes) in zip(weights, sides)) / sum(weights)
        for a in event[key]:
            if a["status"] not in CONTEXT_STATUSES:
                continue
            used = {b["string"] for b in event[key] if b is not a and b["string"]}
            options = [c for c in a["candidates"] if c["string"] not in used]
            if not options:
                continue

            def cost(c):
                frets = min(hand_fret, open_cost) if c["fret"] == 0 else abs(c["fret"] - hand_fret)
                return frets + .1 * abs(c["string"] - hand_string)
            best = min(options, key=cost)
            a.update(previous_status=a["status"], status="context_inferred",
                     string=best["string"], fret=best["fret"], finger=None,
                     context_hand_fret=round(float(hand_fret), 2))
    return events


def note_events(notes, frames, settings):
    """Group close onsets; attach frames within +-delta of the first onset."""
    events, group = [], []
    for note in sorted(notes, key=lambda n: n["onset"]):
        if group and note["onset"] - group[0]["onset"] >= settings.min_interval:
            events.append(group)
            group = []
        group.append(note)
    if group:
        events.append(group)
    result = []
    for i, group in enumerate(events):
        tau = group[0]["onset"]
        # Frames from the next onset on show the hand on the next note.
        end = min(tau + settings.delta_after, events[i + 1][0]["onset"] - 1e-6 if i + 1 < len(events) else math.inf)
        window = [f for f in frames if tau - settings.delta <= f["time"] <= end]
        if window:
            primary = min(range(len(window)), key=lambda i: abs(window[i]["time"] - tau))
            result.append(dict(time=tau, pitches=sorted({n["pitch"] for n in group}),
                               frames=window, primary=primary))
    return result


def observe_clip(dataset, clip, calibration, reference, detector, settings, cache_dir):
    """Track the board and detect hands once per frame; cached as JSON."""
    cache = cache_dir / f"observations_{clip}.json"
    calibration_key = json.loads(json.dumps(calibration.to_dict()))
    cached = json.loads(cache.read_text()) if cache.exists() else {}
    stored = cached.get("frames", {}) if cached.get("calibration") == calibration_key else {}
    frames = []
    for record in [r for r in dataset.records if r.clip_id == clip and r.benchmark_eligible]:
        key = str(record.frame_index)
        if key not in stored or (calibration.fret_wire_points and "board" not in stored[key]):
            image = np.asarray(Image.open(dataset.root / record.frame_path).convert("RGB"))
            board, tracking = track_calibration(reference, image, calibration)
            hands = detector.detect_hands(image, board) if board else []
            stored[key] = dict(board=board.to_dict() if board else None, tracking=tracking,
                               hands=[dict(h, points=h["points"].tolist()) for h in hands])
        entry = stored[key]
        if "board" in entry:
            board = BoardCalibration.from_dict(entry["board"]) if entry["board"] else None
        else:  # Older no-anchor caches serialized only corners.
            board = (BoardCalibration(tuple(map(tuple, entry["corners"])), calibration.far_fret,
                                      calibration.top_string, calibration.tuning) if entry["corners"] else None)
        hands = [dict(h, points=np.asarray(h["points"])) for h in entry["hands"]]
        frames.append(dict(record=record, time=record.timestamp, calibration=board,
                           observation=frame_observation(hands, board, settings) if board else None))
    cache.write_text(json.dumps(dict(calibration=calibration_key, frames=stored)))
    return frames


def estimate_events(events, priors, settings, mode="hybrid"):
    for event in events:
        fingers = {f: estimate_finger(event, f, priors[f], settings, mode=mode) for f in FINGERS}
        barre = any(fr["observation"] and fr["observation"]["barre"] for fr in event["frames"])
        if fingers["index"] and barre and fingers["index"]["branch"] == "adaptive_fallback" \
                and fingers["index"]["n"] > 2:
            fingers["index"]["barre"] = True
        event[mode] = fingers
    return events


def score_events(dataset, events, mode, tuning):
    """Paper metrics on annotated (pressed) fingers at each event's primary frame,
    plus string accuracy for fingers that the pitch step assigned."""
    correct = covered = total = exact = annotated_events = 0
    errors, string_hits, string_total = [], 0, 0
    sounding = dict(notes=0, heard=0, fused_position=0, vision_fret=0)
    for event in events:
        truth = dataset.raw_label(event["frames"][event["primary"]]["record"])
        pressed = [(int(f), int(s), FINGERS[k]) for k, (p, f, s) in enumerate(truth) if p == 1]
        if pressed:
            # In these exercises the highest-fret pressed finger is the sounding note.
            fret, string, finger = max(pressed)
            sounding["notes"] += 1
            pitch = tuning[string - 1] + fret
            match = [a for a in event.get(f"{mode}_strings", []) if a["pitch"] == pitch]
            if match:
                sounding["heard"] += 1
                sounding["fused_position"] += (match[0]["string"], match[0]["fret"]) == (string, fret)
            est = event[mode][finger]
            sounding["vision_fret"] += bool(est and est["fret"] == fret)
        assigned = {a["finger"]: a for a in event.get(f"{mode}_strings", []) if a["finger"]}
        event_ok, any_annotated = True, False
        for finger, (press, fret, string) in zip(FINGERS, truth):
            if press == 0:
                continue
            any_annotated, total = True, total + 1
            est = event[mode][finger]
            if est:
                covered += 1
                errors.append(abs(est["fret"] - fret))
            ok = bool(est and est["fret"] == fret)
            correct += ok
            event_ok &= ok
            if press == 1 and finger in assigned:
                string_total += 1
                string_hits += assigned[finger]["string"] == string
        if any_annotated:
            annotated_events += 1
            exact += event_ok
    return dict(events=len(events), annotated_events=annotated_events, finger_instances=total,
                accuracy=correct / total if total else None, coverage=covered / total if total else None,
                mae=float(np.mean(errors)) if errors else None,
                event_exact_match=exact / annotated_events if annotated_events else None,
                assigned_string_accuracy=string_hits / string_total if string_total else None,
                assigned_strings_scored=string_total,
                sounding_notes=sounding["notes"],
                sounding_pitch_heard=sounding["heard"] / sounding["notes"] if sounding["notes"] else None,
                sounding_vision_fret_only=sounding["vision_fret"] / sounding["notes"] if sounding["notes"] else None,
                sounding_fused_string_and_fret=sounding["fused_position"] / sounding["notes"] if sounding["notes"] else None)


def fit_priors(dataset, events, settings, tuning):
    """Per finger, the fixed n with the best event accuracy on development events."""
    priors, table = {}, {}
    for finger in FINGERS:
        scores = {}
        for n in GRID:
            hit = total = 0
            for event in events:
                press, fret, _ = dataset.raw_label(event["frames"][event["primary"]]["record"])[FINGERS.index(finger)]
                if press == 0:
                    continue
                total += 1
                est = estimate_finger(event, finger, n, settings, mode="fusion")
                hit += bool(est and est["fret"] == fret)
            scores[n] = hit / total if total else 0.
        table[finger] = scores
        priors[finger] = max(GRID, key=lambda n: (scores[n], -abs(n - PAPER_PRIORS[finger])))
    return priors, table


def label_audio_check(dataset, events, tuning):
    """How often a pressed-and-sounding label pitch appears in the event's notes."""
    hit = total = 0
    for event in events:
        truth = dataset.raw_label(event["frames"][event["primary"]]["record"])
        sounding = [tuning[s - 1] + f for p, f, s in truth if p == 1]
        if sounding:
            total += 1
            hit += max(sounding) in event["pitches"] or any(p in event["pitches"] for p in sounding)
    return dict(events_with_class1_labels=total, label_pitch_in_audio=hit / total if total else None)


def frame_predictions(events, mode):
    rows = {}
    for event in events:
        strings = {a["finger"]: a["string"] for a in event[f"{mode}_strings"] if a["finger"]}
        for frame in event["frames"]:
            record = frame["record"]
            for finger in FINGERS:
                est = event[mode][finger]
                barre = bool(est and est.get("barre"))
                rows[(record.clip_id, record.frame_index, finger)] = dict(
                    clip_id=record.clip_id, frame_index=record.frame_index, finger=finger,
                    press_class=0 if not est else 2 if barre else 1, fret=est["fret"] if est else 0,
                    string=0 if not est or barre else strings.get(finger, 0))
    return list(rows.values())


def draw_event(dataset, event, mode, out_path):
    frame = event["frames"][event["primary"]]
    image = np.asarray(Image.open(dataset.root / frame["record"].frame_path).convert("RGB"))
    Image.fromarray(draw_overlay(image, event, mode)).save(out_path, quality=88)


def draw_overlay(image, event, mode):
    """Board grid, finger chains and fused frets on the event's primary frame, cropped to the board."""
    frame = event["frames"][event["primary"]]
    image = image.copy()
    board, obs = frame["calibration"], frame["observation"]
    colors = dict(index=(255, 80, 80), middle=(80, 220, 80), ring=(80, 160, 255), pinkie=(255, 200, 0))
    if board:
        for k in range(board.far_fret + 1):
            x = board.fret_x(k)
            a, b = board.board_to_image([[x, 0], [x, 1]])
            cv2.line(image, tuple(map(int, a)), tuple(map(int, b)), (240, 240, 240), 1, cv2.LINE_AA)
        for s in range(6):
            a, b = board.board_to_image([[0, s / 5], [1, s / 5]])
            cv2.line(image, tuple(map(int, a)), tuple(map(int, b)), (200, 200, 200), 1, cv2.LINE_AA)
    if obs:
        for finger, chain in obs["chains"].items():
            pts = np.int32(chain["points"])
            cv2.polylines(image, [pts], False, colors[finger], 2, cv2.LINE_AA)
            est = event[mode][finger]
            if est:
                q = chain_point(chain["points"], est["n"])
                cv2.circle(image, tuple(map(int, q)), 7, colors[finger], -1, cv2.LINE_AA)
                cv2.putText(image, str(est["fret"]), tuple(map(int, q + [8, -8])),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, colors[finger], 2, cv2.LINE_AA)
    corners = np.asarray(board.corners if board else [[0, 0], [image.shape[1], image.shape[0]]])
    x0, y0 = np.maximum(0, corners.min(0) - 220).astype(int)
    x1, y1 = np.minimum(image.shape[1::-1], corners.max(0) + 220).astype(int)
    return image[y0:y1, x0:x1]


def write_report(dataset, clip_events, metrics, priors, out, mode="hybrid"):
    rows = []
    for clip, events in clip_events.items():
        for i, event in enumerate(events):
            name = f"{clip}_{i:03d}.jpg"
            draw_event(dataset, event, mode, out / name)
            truth = dataset.raw_label(event["frames"][event["primary"]]["record"])
            cells = "".join(
                f"<td>{e['fret'] if e else '–'}<small> n={e['n'] if e else ''}"
                f"{' fb' if e and e['branch'] != 'fixed_prior' else ''}</small><br>"
                f"<small>label {('fret ' + str(t[1]) + ' s' + str(t[2])) if t[0] else '–'}</small></td>"
                for e, t in ((event[mode][f], truth[k]) for k, f in enumerate(FINGERS)))
            notes = "<br>".join(f"{a['note']}: {a['status']}" +
                                (f" → s{a['string']} f{a['fret']} ({a['finger'] or ''})" if a["string"] else "")
                                for a in event[f"{mode}_strings"])
            rows.append(f"<tr><td>{clip}</td><td>{event['time']:.2f}s<br><small>{len(event['frames'])} frame(s)"
                        f"</small></td><td><img src='{name}' loading='lazy'></td><td>{notes}</td>{cells}</tr>")
    metric_rows = "".join(f"<tr><td>{m}</td>" + "".join(
        f"<td>{v:.3f}</td>" if isinstance(v, float) else f"<td>{v}</td>" for v in r.values()) + "</tr>"
        for m, r in metrics.items())
    header = "".join(f"<th>{k}</th>" for k in next(iter(metrics.values())))
    (out / "report.html").write_text(f"""<!doctype html><meta charset=utf-8><title>Event fusion</title>
<style>body{{font:14px system-ui;margin:16px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;
padding:4px;vertical-align:top}}img{{max-width:420px}}small{{color:#666}}</style>
<h1>Audio-guided event fusion</h1><p>YourMT3 onsets → ±window frames → per-finger fused frets
(priors {priors}) → pitch picks strings. Evaluation clips only.</p>
<table><tr><th>mode</th>{header}</tr>{metric_rows}</table><h2>Events (mode: {mode})</h2>
<table><tr><th>clip</th><th>onset</th><th>primary frame</th><th>notes → position</th>
{''.join(f'<th>{f}</th>' for f in FINGERS)}</tr>{''.join(rows)}</table>""")


def estimate_pitch_offset(dataset, clips, tuning):
    """Median pYIN-minus-label pitch (semitones) over class-1 labels on dev clips.

    The archive audio sounds ~1.5 semitones above its labels; a fractional
    offset puts notes between MIDI pitches, which transcribers then miss.
    """
    import librosa
    import soundfile as sf
    offsets = []
    for clip in clips:
        samples, sr = sf.read(dataset.root / f"tablature_audio/{clip}.wav", dtype="float32", always_2d=True)
        y = librosa.resample(samples.mean(1), orig_sr=sr, target_sr=22050)
        f0, _, _ = librosa.pyin(y, sr=22050, fmin=65., fmax=1400., frame_length=4096, hop_length=512)
        times = librosa.times_like(f0, sr=22050, hop_length=512)
        for record in (r for r in dataset.records if r.clip_id == clip and r.benchmark_eligible):
            pressed = [(f, s) for p, f, s in dataset.raw_label(record) if p == 1]
            if not pressed:
                continue
            fret, string = max(pressed)  # the highest fret on a string is the one sounding
            i = int(np.argmin(abs(times - record.timestamp)))
            f = f0[i:i + 4][np.isfinite(f0[i:i + 4])]
            if len(f):
                offsets.append(float(librosa.hz_to_midi(np.median(f))) - (tuning[string - 1] + fret))
    return float(np.median(offsets)), len(offsets)


MODES = ("tip_single", "single", "vote", "fusion", "hybrid")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--notes-dir", type=Path, default=ROOT / "output/yourmt3")
    parser.add_argument("--calibration", type=Path, default=ROOT / "pilot-fusion/clip0_calibration.json")
    parser.add_argument("--dev-clips", nargs="*", default=["3"])
    parser.add_argument("--eval-clips", nargs="+", default=["0", "7"])
    parser.add_argument("--priors", choices=("fit", "paper", "tip"), default="fit")
    parser.add_argument("--window-before", type=float, default=.2,
                        help="seconds before each onset; 0 drops frames still showing the previous note")
    parser.add_argument("--window-after", type=float, default=.2,
                        help="seconds after each onset; paper uses 0.2 at 5 fps")
    parser.add_argument("--fret-tolerance", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/event-fusion")
    parser.add_argument("--estimate-pitch-offset", nargs="+", metavar="DEV_CLIP",
                        help="print the audio-vs-label pitch offset on these clips and exit")
    args = parser.parse_args(argv)
    if args.estimate_pitch_offset:
        calibration = BoardCalibration.from_dict(json.loads(args.calibration.read_text()))
        offset, count = estimate_pitch_offset(TablatureDataset(), args.estimate_pitch_offset, calibration.tuning)
        print(json.dumps(dict(clips=args.estimate_pitch_offset, offset_semitones=round(offset, 3),
                              labelled_frames=count, yourmt3_pitch_shift=round(-offset, 3))))
        return
    settings = Settings(delta=args.window_before, delta_after=args.window_after,
                        fret_tolerance=args.fret_tolerance)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    dataset = TablatureDataset()
    data = json.loads(args.calibration.read_text())
    calibration = BoardCalibration.from_dict(data)
    reference = np.asarray(Image.open(ROOT / data["reference_image"]).convert("RGB"))
    detector = FingertipDetector()
    try:
        clip_events, dev_frames = {}, []
        for clip in [*args.dev_clips, *args.eval_clips]:
            frames = observe_clip(dataset, clip, calibration, reference, detector, settings,
                                  ROOT / "output/event-fusion")
            if clip in args.dev_clips:
                dev_frames += [dict(time=f["time"], pitches=[], frames=[f], primary=0) for f in frames]
            notes = load_audio_notes(args.notes_dir / f"{clip}_notes.csv")
            clip_events[clip] = note_events(notes, frames, settings)
            tracked = sum(f["calibration"] is not None for f in frames)
            print(f"clip {clip}: {len(notes)} notes, {len(clip_events[clip])} events with frames, "
                  f"{tracked}/{len(frames)} frames tracked", flush=True)
    finally:
        detector.close()
    tuning = calibration.tuning
    # Every labelled development frame is a one-frame event for prior fitting.
    if args.priors == "fit" and dev_frames:
        priors, prior_table = fit_priors(dataset, dev_frames, settings, tuning)
    else:
        priors = dict(PAPER_PRIORS) if args.priors == "paper" else dict.fromkeys(FINGERS, 0.)
        prior_table = None
    evaluation = {c: clip_events[c] for c in args.eval_clips}
    metrics = {}
    for mode in MODES:
        for event in (e for events in evaluation.values() for e in events):
            estimate_events([event], priors, settings, mode)
            event[f"{mode}_strings"] = assign_strings(event["pitches"], event[mode], tuning, settings)
        metrics[mode] = score_events(dataset, [e for v in evaluation.values() for e in v], mode, tuning)
    all_eval = [e for v in evaluation.values() for e in v]
    statuses = defaultdict(int)
    for event in all_eval:
        for a in event["hybrid_strings"]:
            statuses[a["status"]] += 1
    audio_only = defaultdict(int)
    for event in all_eval:
        for p in event["pitches"]:
            audio_only[len(pitch_candidates(p, tuning))] += 1
    predictions = frame_predictions(all_eval, "hybrid")
    with (out / "finger_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["clip_id", "frame_index", "finger", "press_class", "fret", "string"])
        writer.writeheader()
        writer.writerows(predictions)
    summary = dict(priors=priors, prior_fit_accuracy=prior_table, settings=settings.__dict__,
                   dev_clips=args.dev_clips, eval_clips=args.eval_clips, metrics=metrics,
                   pitch_assignment_status=dict(statuses),
                   audio_only_candidate_counts={str(k): v for k, v in sorted(audio_only.items())},
                   label_audio_check=label_audio_check(dataset, all_eval, tuning),
                   fallback_invocations=sum(e["hybrid"][f]["branch"] == "adaptive_fallback"
                                            for e in all_eval for f in FINGERS if e["hybrid"][f]))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(dataset, evaluation, metrics, priors, out)
    print(json.dumps({k: v for k, v in summary.items() if k != "prior_fit_accuracy"}, indent=2))


if __name__ == "__main__":
    main()
