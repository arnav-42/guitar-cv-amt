"""
Webcam guitar detection (YOLOv8 OpenImagesV7) + right-hand STRUM vs PICK heuristic

Install:
  pip install ultralytics opencv-python mediapipe

Run:
  python guitar_strum_pick_demo.py
"""

import os
import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def pt_in_rect(px, py, x1, y1, x2, y2):
    return (x1 <= px <= x2) and (y1 <= py <= y2)


def get_label_from_handedness(handedness_entry):
    """
    MediaPipe Tasks returns a list of Classification objects per hand.
    We try common attribute names defensively.
    """
    try:
        c = handedness_entry[0]
    except Exception:
        return None

    for attr in ("category_name", "display_name", "label", "name"):
        if hasattr(c, attr):
            v = getattr(c, attr)
            if isinstance(v, str) and v:
                return v
    return None


def ensure_hand_model(model_path: str) -> str:
    # Official model bundle URL from Google AI Edge HandLandmarker models table
    model_url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    )
    if not os.path.exists(model_path):
        print(f"[mediapipe] downloading hand model -> {model_path}")
        urllib.request.urlretrieve(model_url, model_path)
    return model_path


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Try VideoCapture(1).")

    # OpenImagesV7 weights include 'Guitar' class; COCO weights usually don't.
    model = YOLO("yolov8n-oiv7.pt")

    # Neck ROI heuristic params (relative to detected guitar bbox)
    neck_len_scale = 0.85
    neck_w_scale = 0.22

    # Strum ROI heuristic params (relative to detected guitar bbox)
    # Assumes neck to the right; strum zone is on the body (left/middle of bbox).
    strum_x1_scale = 0.10
    strum_x2_scale = 0.55
    strum_y1_scale = 0.35
    strum_y2_scale = 0.92

    mirror = True  # webcam-style mirrored view

    # Motion history for "right hand" wrist
    wrist_hist = deque(maxlen=14)
    inroi_hist = deque(maxlen=14)
    state_hist = deque(maxlen=6)

    # --- MediaPipe Tasks: HandLandmarker ---
    model_path = ensure_hand_model("hand_landmarker.task")

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmarker = HandLandmarker.create_from_options(options)
    t0 = time.perf_counter()

    print("Controls: q quit | m mirror | [ ] neck width | - = neck length")

    try:
        while True:
            ok, frame0 = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame0, 1) if mirror else frame0
            H, W = frame.shape[:2]

            # --- YOLO inference (guitar bbox) ---
            results = model(frame, verbose=False)
            r = results[0]

            best = None
            best_conf = 0.0

            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    cls = int(b.cls.item())
                    conf = float(b.conf.item())
                    name = model.names.get(cls, str(cls))
                    if str(name).lower() == "guitar" and conf > best_conf:
                        xyxy = b.xyxy[0].cpu().numpy()
                        best = xyxy
                        best_conf = conf

            smoothed_state = "IDLE"

            if best is not None:
                x1, y1, x2, y2 = map(int, best.tolist())
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                diag = float(np.hypot(bw, bh))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"guitar {best_conf:.2f}",
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                # --- Neck ROI heuristic ---
                neck_len = int(neck_len_scale * bw)
                neck_w = int(neck_w_scale * bh)

                anchor_x = x2
                anchor_y = y1 + int(0.30 * bh)

                nx1 = clamp(anchor_x, 0, W - 1)
                ny1 = clamp(anchor_y - neck_w // 2, 0, H - 1)
                nx2 = clamp(anchor_x + neck_len, 0, W - 1)
                ny2 = clamp(anchor_y + neck_w // 2, 0, H - 1)

                cv2.rectangle(frame, (nx1, ny1), (nx2, ny2), (255, 0, 0), 2)

                # --- Strum ROI heuristic (body zone) ---
                sx1 = clamp(x1 + int(strum_x1_scale * bw), 0, W - 1)
                sx2 = clamp(x1 + int(strum_x2_scale * bw), 0, W - 1)
                sy1 = clamp(y1 + int(strum_y1_scale * bh), 0, H - 1)
                sy2 = clamp(y1 + int(strum_y2_scale * bh), 0, H - 1)

                if sx2 < sx1:
                    sx1, sx2 = sx2, sx1
                if sy2 < sy1:
                    sy1, sy2 = sy2, sy1

                cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 165, 255), 2)

                # --- HandLandmarker (VIDEO mode) ---
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms = int((time.perf_counter() - t0) * 1000.0)

                hr = landmarker.detect_for_video(mp_image, ts_ms)

                candidates = []
                if hr is not None and hr.hand_landmarks:
                    for i in range(len(hr.hand_landmarks)):
                        lm = hr.hand_landmarks[i]
                        wrist = lm[0]
                        wx = int(wrist.x * W)
                        wy = int(wrist.y * H)

                        label = None
                        if hr.handedness and i < len(hr.handedness):
                            label = get_label_from_handedness(hr.handedness[i])

                        if mirror and label in ("Left", "Right"):
                            label = "Left" if label == "Right" else "Right"

                        candidates.append((label, wx, wy))

                # Choose "Right" if possible; else closest to strum ROI center
                chosen = None
                if candidates:
                    for label, wx, wy in candidates:
                        if label == "Right":
                            chosen = (wx, wy)
                            break
                    if chosen is None:
                        cx = (sx1 + sx2) // 2
                        cy = (sy1 + sy2) // 2
                        chosen = min(
                            [(wx, wy) for (_, wx, wy) in candidates],
                            key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2,
                        )

                if chosen is not None:
                    wx, wy = chosen
                    cv2.circle(frame, (wx, wy), 6, (255, 255, 255), -1)
                    cv2.circle(frame, (wx, wy), 8, (0, 0, 0), 2)

                    inside = pt_in_rect(wx, wy, sx1, sy1, sx2, sy2)
                    wrist_hist.append((wx, wy))
                    inroi_hist.append(inside)

                    # --- STRUM vs PICK heuristic ---
                    if len(wrist_hist) >= 5:
                        vs = []
                        for k in range(1, len(wrist_hist)):
                            dx = wrist_hist[k][0] - wrist_hist[k - 1][0]
                            dy = wrist_hist[k][1] - wrist_hist[k - 1][1]
                            vs.append((dx, dy))

                        abs_dx = float(np.mean([abs(v[0]) for v in vs]))
                        abs_dy = float(np.mean([abs(v[1]) for v in vs]))
                        avg_speed = float(np.mean([np.hypot(v[0], v[1]) for v in vs]))
                        vert_ratio = abs_dy / (abs_dx + 1e-6)

                        # Back/forth detection on dy (zero-crossings)
                        min_comp = 0.004 * diag
                        dy_sign = []
                        for _, dy in vs:
                            if abs(dy) < min_comp:
                                dy_sign.append(0)
                            else:
                                dy_sign.append(1 if dy > 0 else -1)

                        zero_cross = 0
                        for k in range(1, len(dy_sign)):
                            if (
                                dy_sign[k] != 0
                                and dy_sign[k - 1] != 0
                                and dy_sign[k] * dy_sign[k - 1] < 0
                            ):
                                zero_cross += 1

                        inroi_frac = float(np.mean(inroi_hist)) if inroi_hist else 0.0

                        strum_speed = 0.014 * diag
                        pick_speed = 0.006 * diag

                        state = "IDLE"
                        if inroi_frac >= 0.45:
                            if (
                                avg_speed >= strum_speed
                                and vert_ratio >= 1.20
                                and zero_cross >= 1
                            ):
                                state = "STRUM"
                            elif (
                                avg_speed >= pick_speed
                                and avg_speed < strum_speed
                                and vert_ratio < 1.25
                            ):
                                state = "PICK"

                        state_hist.append(state)
                        if state_hist:
                            vals, counts = np.unique(
                                list(state_hist), return_counts=True
                            )
                            smoothed_state = str(vals[int(np.argmax(counts))])

                        cv2.putText(
                            frame,
                            f"{smoothed_state}  v={avg_speed:.1f}  vr={vert_ratio:.2f}  zc={zero_cross}",
                            (sx1, min(H - 10, sy2 + 24)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 165, 255),
                            2,
                            cv2.LINE_AA,
                        )
                else:
                    wrist_hist.clear()
                    inroi_hist.clear()
                    state_hist.clear()

            else:
                cv2.putText(
                    frame,
                    "No guitar detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                wrist_hist.clear()
                inroi_hist.clear()
                state_hist.clear()

            cv2.imshow("Guitar + Right-hand Strum/Pick", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("m"):
                mirror = not mirror
                wrist_hist.clear()
                inroi_hist.clear()
                state_hist.clear()
            elif key == ord("["):
                neck_w_scale = max(0.08, neck_w_scale - 0.02)
            elif key == ord("]"):
                neck_w_scale = min(0.60, neck_w_scale + 0.02)
            elif key == ord("-") or key == ord("_"):
                neck_len_scale = max(0.20, neck_len_scale - 0.05)
            elif key == ord("=") or key == ord("+"):
                neck_len_scale = min(2.00, neck_len_scale + 0.05)

    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
