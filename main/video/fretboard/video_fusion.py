"""Video in, fretboard positions out: audio-guided fret estimation on any video.

    pilot-fusion/.venv/bin/python -m main.video.fretboard.video_fusion my_video.mp4

1. ffmpeg extracts the audio track to WAV.
2. YourMT3 (in its own environment, via yourmt3_bridge.py) transcribes note
   onsets and pitches.
3. Frames are sampled from the video (5 fps by default, as in the paper); the
   board is tracked from a calibrated reference frame and MediaPipe finds the
   fretting hand in each frame.
4. event_fusion turns each onset's frame window into one fret per finger, and
   the transcribed pitch chooses the string.
5. Notes vision could not place (unresolved/ambiguous) take the position
   nearest the hand on the surrounding confirmed notes ("context_inferred").

The first run on a video asks for a fretboard calibration: a window opens on
the reference frame and you click four points (or pass them with --corners).
It is saved as calibration.json in the output folder and reused afterwards.
Outputs: positions.csv, tab.txt and report.html in output/video-fusion/<name>/.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from main.video.fretboard.audio_visual_fusion import (ROOT, STANDARD_TUNING, BoardCalibration,
                                          FingertipDetector, track_calibration)
from main.video.fretboard.event_fusion import (PAPER_PRIORS, FINGERS, Settings, assign_strings, draw_overlay,
                                   estimate_events, frame_observation, infer_from_context,
                                   load_audio_notes, note_events)

YOURMT3_PYTHON = Path.home() / "yourmt3_src/venv/bin/python"
YOURMT3_DIR = Path.home() / "yourmt3_src/YourMT3"
STRING_NAMES = ("e", "B", "G", "D", "A", "E")  # strings 1..6
CORNER_PROMPTS = ("nut, top string", "fret wire {far}, top string",
                  "fret wire {far}, bottom string", "nut, bottom string")


def extract_audio(video, wav):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg not found; install it (brew install ffmpeg).")
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
                    "-vn", "-ac", "1", "-ar", "44100", str(wav)], check=True)


def transcribe(wav, notes_csv, python, yourmt3_dir, pitch_shift):
    if not Path(python).is_file():
        raise SystemExit(f"YourMT3 Python not found at {python}; pass --yourmt3-python.")
    subprocess.run([str(python), str(ROOT / "main/video/fretboard/yourmt3_bridge.py"), "--yourmt3-dir", str(yourmt3_dir),
                    "--audio", str(wav), "--output", str(notes_csv), "--pitch-shift", str(pitch_shift)],
                   check=True)


def read_frame(video, index):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read frame {index} of {video}.")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def draw_grid(image, board):
    image = image.copy()
    for k in range(board.far_fret + 1):
        x = board.fret_x(k)
        a, b = board.board_to_image([[x, 0], [x, 1]])
        cv2.line(image, tuple(map(int, a)), tuple(map(int, b)), (255, 255, 0), 2, cv2.LINE_AA)
    for s in range(6):
        a, b = board.board_to_image([[0, s / 5], [1, s / 5]])
        cv2.line(image, tuple(map(int, a)), tuple(map(int, b)), (0, 255, 255), 1, cv2.LINE_AA)
    return image


def click_corners(image, far_fret):
    """Click the four board corners on the reference frame; Enter accepts, Backspace undoes."""
    scale = min(1., 1600 / image.shape[1], 900 / image.shape[0])
    base = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), None, fx=scale, fy=scale)
    points, name = [], "calibration: click 4 points (Enter = accept, Backspace = undo, Esc = quit)"

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x / scale, y / scale))

    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(name, on_click)
    while True:
        view = base.copy()
        for i, (x, y) in enumerate(points):
            cv2.circle(view, (int(x * scale), int(y * scale)), 5, (0, 0, 255), -1)
            cv2.putText(view, str(i + 1), (int(x * scale) + 6, int(y * scale) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
        prompt = (f"{len(points) + 1}: {CORNER_PROMPTS[len(points)].format(far=far_fret)}"
                  if len(points) < 4 else "Enter to accept")
        cv2.putText(view, prompt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 4)
        cv2.putText(view, prompt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
        cv2.imshow(name, view)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10) and len(points) == 4:
            break
        if key in (8, 127) and points:
            points.pop()
        if key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("Calibration cancelled.")
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return points


def load_calibration(args, video, out, fps_src):
    path = args.calibration or out / "calibration.json"
    if path.is_file() and not args.recalibrate:
        data = json.loads(path.read_text())
    else:
        index = int(round(args.reference_time * fps_src))
        image = read_frame(video, index)
        if args.corners:
            corners = [tuple(map(float, p.split(","))) for p in args.corners]
        else:
            print("Click, in order: " + "; ".join(f"{i + 1}) {p.format(far=args.far_fret)}"
                                                   for i, p in enumerate(CORNER_PROMPTS)))
            corners = click_corners(image, args.far_fret)
        board = BoardCalibration(tuple(corners), args.far_fret, args.top_string, tuple(args.tuning),
                                 (image.shape[1], image.shape[0]))
        data = dict(board.to_dict(), reference_frame=index, video=str(video))
        path = out / "calibration.json"
        path.write_text(json.dumps(data, indent=2) + "\n")
        Image.fromarray(draw_grid(image, board)).save(out / "calibration_check.jpg", quality=88)
        print(f"Saved {path}; check the fret grid in {out / 'calibration_check.jpg'}.")
    return BoardCalibration.from_dict(data), int(data.get("reference_frame", 0))


def observe_video(video, calibration, reference_index, fps, detector, settings, cache_path):
    """Sample frames, track the board from the reference frame, detect hands; cached as JSON."""
    key = dict(fps=fps, calibration=json.loads(json.dumps(calibration.to_dict())),
               reference_frame=reference_index)
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    stored = cache.get("frames", {}) if cache.get("key") == key else {}
    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.
    step = max(1., src_fps / fps)
    reference, frames, index, next_pick = read_frame(video, reference_index), [], 0, 0.
    while True:
        if index + .5 < next_pick:
            if not cap.grab():
                break
            index += 1
            continue
        ok, bgr = cap.read()
        if not ok:
            break
        next_pick += step
        if str(index) not in stored:
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            board, tracking = track_calibration(reference, image, calibration)
            hands = detector.detect_hands(image, board) if board else []
            stored[str(index)] = dict(board=board.to_dict() if board else None, tracking=tracking,
                                      hands=[dict(h, points=h["points"].tolist()) for h in hands])
        entry = stored[str(index)]
        board = BoardCalibration.from_dict(entry["board"]) if entry["board"] else None
        hands = [dict(h, points=np.asarray(h["points"])) for h in entry["hands"]]
        frames.append(dict(index=index, time=index / src_fps, calibration=board,
                           observation=frame_observation(hands, board, settings) if board else None))
        if len(frames) % 50 == 0:
            print(f"  {len(frames)} frames ({index / src_fps:.1f}s)", flush=True)
        index += 1
    cap.release()
    cache_path.write_text(json.dumps(dict(key=key, frames=stored)))
    return frames


def render_tab(events, width=78):
    """ASCII tab, one column per event; only positions the fusion resolved."""
    columns = []
    for event in events:
        cells = {a["string"]: str(a["fret"]) for a in event["hybrid_strings"] if a["string"]}
        if cells:
            w = max(map(len, cells.values()))
            columns.append([cells.get(s, "-" * w).ljust(w, "-") for s in range(1, 7)])
    lines, block = [], [[] for _ in range(6)]
    for col in columns:
        if sum(len(c) + 1 for c in block[0]) + len(col[0]) + 3 > width:
            lines += [f"{STRING_NAMES[s]}|-" + "-".join(block[s]) + "-|" for s in range(6)] + [""]
            block = [[] for _ in range(6)]
        for s in range(6):
            block[s].append(col[s])
    if block[0]:
        lines += [f"{STRING_NAMES[s]}|-" + "-".join(block[s]) + "-|" for s in range(6)]
    return "\n".join(lines) + "\n"


def write_outputs(video, events, frames, notes, out, priors):
    with (out / "positions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time", "pitch", "note", "string", "fret", "finger", "status"])
        writer.writeheader()
        for event in events:
            for a in event["hybrid_strings"]:
                writer.writerow(dict(time=round(event["time"], 3), pitch=a["pitch"], note=a["note"],
                                     string=a["string"] or "", fret="" if a["fret"] is None else a["fret"],
                                     finger=a["finger"] or "", status=a["status"]))
    (out / "tab.txt").write_text(render_tab(events))
    images = out / "events"
    images.mkdir(exist_ok=True)
    rows = []
    for i, event in enumerate(events):
        name = f"events/{i:04d}.jpg"
        image = read_frame(video, event["frames"][event["primary"]]["index"])
        Image.fromarray(draw_overlay(image, event, "hybrid")).save(out / name, quality=85)
        notes_html = "<br>".join(html.escape(f"{a['note']}: string {a['string']}, fret {a['fret']}"
                                             f"{' (' + a['finger'] + ')' if a['finger'] else ''}"
                                             f"{' [from context]' if a['status'] == 'context_inferred' else ''}"
                                             if a["string"] else f"{a['note']}: {a['status']}")
                                 for a in event["hybrid_strings"])
        fingers = " ".join(f"{f[0].upper()}{e['fret']}" for f, e in event["hybrid"].items() if e) or "no hand"
        rows.append(f"<tr><td>{event['time']:.2f}s<br><small>{len(event['frames'])} frame(s)</small></td>"
                    f"<td><img src='{name}' loading='lazy'></td><td>{notes_html}</td>"
                    f"<td><small>{fingers}</small></td></tr>")
    tracked = sum(f["calibration"] is not None for f in frames)
    (out / "report.html").write_text(f"""<!doctype html><meta charset=utf-8><title>Video fusion</title>
<style>body{{font:14px system-ui;margin:16px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;
padding:4px;vertical-align:top}}img{{max-width:460px}}small{{color:#666}}pre{{background:#f5f5f5;padding:8px}}</style>
<h1>{html.escape(Path(video).name)}</h1>
<p>{len(notes)} transcribed notes → {len(events)} events with frames; board tracked in {tracked}/{len(frames)}
sampled frames. Finger priors {priors}.</p><pre>{html.escape(render_tab(events))}</pre>
<table><tr><th>onset</th><th>primary frame</th><th>note → position</th><th>finger frets</th></tr>
{''.join(rows)}</table>""")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--output-dir", type=Path, help="default: output/video-fusion/<video name>")
    parser.add_argument("--fps", type=float, default=5., help="frames sampled per second (paper: 5)")
    parser.add_argument("--calibration", type=Path, help="existing calibration JSON for this video")
    parser.add_argument("--recalibrate", action="store_true", help="ignore a saved calibration")
    parser.add_argument("--corners", nargs=4, metavar="X,Y",
                        help="skip the click window: nut-top, fret-top, fret-bottom, nut-bottom pixels")
    parser.add_argument("--reference-time", type=float, default=0., help="seconds; frame to calibrate on")
    parser.add_argument("--far-fret", type=int, default=12, help="fret wire used for corners 2 and 3")
    parser.add_argument("--top-string", type=int, choices=(1, 6), default=6,
                        help="string nearest the top of the image (6 = low E)")
    parser.add_argument("--tuning", type=int, nargs=6, default=list(STANDARD_TUNING),
                        metavar="MIDI", help="strings 1..6, default standard E")
    parser.add_argument("--priors", choices=("paper", "tip"), default="paper")
    parser.add_argument("--priors-from", type=Path,
                        help="summary.json from main.video.fretboard.event_fusion, to reuse its fitted priors")
    parser.add_argument("--notes", type=Path, help="existing onset,offset,pitch CSV; skips YourMT3")
    parser.add_argument("--pitch-shift", type=float, default=0.)
    parser.add_argument("--yourmt3-python", type=Path, default=YOURMT3_PYTHON)
    parser.add_argument("--yourmt3-dir", type=Path, default=YOURMT3_DIR)
    parser.add_argument("--window-before", type=float, default=.2)
    parser.add_argument("--window-after", type=float, default=.2)
    parser.add_argument("--no-context", action="store_true",
                        help="leave unresolved notes unresolved instead of inferring them from neighbours")
    parser.add_argument("--overwrite", action="store_true", help="redo audio extraction and transcription")
    args = parser.parse_args(argv)

    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"No such video: {video}")
    out = (args.output_dir or ROOT / "output/video-fusion" / video.stem).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.
    cap.release()

    calibration, reference_index = load_calibration(args, video, out, fps_src)

    notes_csv = args.notes or out / "notes.csv"
    if not args.notes and (args.overwrite or not notes_csv.is_file()):
        print("1/3 extracting audio and transcribing with YourMT3 ...", flush=True)
        extract_audio(video, out / "audio.wav")
        transcribe(out / "audio.wav", notes_csv, args.yourmt3_python.expanduser(),
                   args.yourmt3_dir.expanduser(), args.pitch_shift)
    notes = load_audio_notes(notes_csv)

    print(f"2/3 tracking board and hand at {args.fps:g} fps ...", flush=True)
    settings = Settings(delta=args.window_before, delta_after=args.window_after)
    detector = FingertipDetector()
    try:
        frames = observe_video(video, calibration, reference_index, args.fps, detector, settings,
                               out / "observations.json")
    finally:
        detector.close()

    print("3/3 fusing notes with finger positions ...", flush=True)
    if args.priors_from:
        priors = json.loads(args.priors_from.read_text())["priors"]
    else:
        priors = dict(PAPER_PRIORS) if args.priors == "paper" else dict.fromkeys(FINGERS, 0.)
    events = estimate_events(note_events(notes, frames, settings), priors, settings, "hybrid")
    for event in events:
        event["hybrid_strings"] = assign_strings(event["pitches"], event["hybrid"], calibration.tuning, settings)
    if not args.no_context:
        infer_from_context(events, calibration.tuning)
    write_outputs(video, events, frames, notes, out, priors)

    statuses = {}
    for event in events:
        for a in event["hybrid_strings"]:
            statuses[a["status"]] = statuses.get(a["status"], 0) + 1
    tracked = sum(f["calibration"] is not None for f in frames)
    print(f"{len(notes)} notes, {len(events)} events, board tracked in {tracked}/{len(frames)} frames")
    print("note outcomes: " + ", ".join(f"{k} {v}" for k, v in sorted(statuses.items())))
    print(f"\n{render_tab(events)}\nOpen {out / 'report.html'}")


if __name__ == "__main__":
    main()
