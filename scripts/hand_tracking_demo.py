import time
import urllib.request
from pathlib import Path
from typing import Optional

import cv2  # type: ignore
import mediapipe as mp  # type: ignore
from mediapipe.tasks import python as mp_python  # type: ignore
from mediapipe.tasks.python import vision  # type: ignore

# MediaPipe Tasks now replace the deprecated mp.solutions API.
# This demo mirrors the previous behavior while using the new Hand Landmarker task.

# Default model location (float16 variant recommended by MediaPipe docs).
_DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# Connection list matching the legacy HAND_CONNECTIONS constant.
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def _ensure_model_file(model_path: Path) -> Path:
    """Download the hand_landmarker.task model if it is not present."""
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe hand model to {model_path} ...")
    urllib.request.urlretrieve(_DEFAULT_MODEL_URL, model_path)
    return model_path


class HandTracker:
    def __init__(self, model_path: Optional[Path] = None):
        # Resolve and cache the model
        cache_dir = Path.home() / ".cache" / "mediapipe"
        model_path = model_path or cache_dir / "hand_landmarker.task"
        model_path = _ensure_model_file(model_path)

        BaseOptions = mp_python.BaseOptions
        HandLandmarker = vision.HandLandmarker
        HandLandmarkerOptions = vision.HandLandmarkerOptions
        VisionRunningMode = vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.hand_landmarker = HandLandmarker.create_from_options(options)
        self.last_timestamp_ms = 0

    def count_fingers(self, landmarks):
        """Count extended fingers"""
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        finger_pips = [3, 6, 10, 14, 18]  # PIP joints

        fingers_up = []

        # Thumb (special case - compare x coordinates)
        if landmarks[finger_tips[0]].x > landmarks[finger_pips[0]].x:
            fingers_up.append(1)
        else:
            fingers_up.append(0)

        # Other fingers (compare y coordinates)
        for i in range(1, 5):
            if landmarks[finger_tips[i]].y < landmarks[finger_pips[i]].y:
                fingers_up.append(1)
            else:
                fingers_up.append(0)

        return sum(fingers_up), fingers_up

    def _draw_connections(self, image, landmarks):
        h, w, _ = image.shape
        for start, end in HAND_CONNECTIONS:
            x_start, y_start = int(landmarks[start].x * w), int(landmarks[start].y * h)
            x_end, y_end = int(landmarks[end].x * w), int(landmarks[end].y * h)
            cv2.line(image, (x_start, y_start), (x_end, y_end), (0, 255, 255), 2)

        for landmark in landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (x, y), 3, (50, 170, 255), -1)
            cv2.circle(image, (x, y), 5, (255, 255, 255), 1)

    def draw_hand(self, image, landmarks, hand_label):
        """Draw hand with custom styling"""
        h, w, _ = image.shape

        # Draw connections and joints
        self._draw_connections(image, landmarks)

        # Count fingers
        finger_count, fingers_up = self.count_fingers(landmarks)

        # Draw fingertips with special highlighting
        finger_tip_indices = [4, 8, 12, 16, 20]
        colors = [
            (255, 100, 100),  # Thumb - Red
            (100, 255, 100),  # Index - Green
            (100, 100, 255),  # Middle - Blue
            (255, 255, 100),  # Ring - Yellow
            (255, 100, 255),  # Pinky - Magenta
        ]

        for i, tip_idx in enumerate(finger_tip_indices):
            x = int(landmarks[tip_idx].x * w)
            y = int(landmarks[tip_idx].y * h)

            radius = 12 if fingers_up[i] else 8
            color = colors[i] if fingers_up[i] else (100, 100, 100)
            cv2.circle(image, (x, y), radius, color, -1)
            cv2.circle(image, (x, y), radius, (255, 255, 255), 2)

        # Draw finger count text
        wrist_x = int(landmarks[0].x * w)
        wrist_y = int(landmarks[0].y * h)

        text = f"{hand_label}: {finger_count} fingers"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(
            image,
            (wrist_x - 10, wrist_y - text_height - 20),
            (wrist_x + text_width + 10, wrist_y + baseline),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            image,
            text,
            (wrist_x, wrist_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return finger_count

    def process_frame(self, frame):
        """Process a single frame"""
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        timestamp_ms = int(time.time() * 1000)
        # Ensure timestamps are strictly increasing for the video mode.
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        total_fingers = 0

        if results and results.hand_landmarks:
            for idx, landmarks in enumerate(results.hand_landmarks):
                hand_label = results.handedness[idx][0].category_name
                finger_count = self.draw_hand(frame, landmarks, hand_label)
                total_fingers += finger_count

        header_height = 60
        overlay = frame.copy()
        cv2.rectangle(
            overlay, (0, 0), (frame.shape[1], header_height), (20, 20, 20), -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        title = "Finger Tracking Demo for Guitar AMT"
        cv2.putText(
            frame, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
        )

        if results and results.hand_landmarks:
            count_text = f"Total Fingers Extended: {total_fingers}"
            (text_width, _), _ = cv2.getTextSize(
                count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )
            cv2.putText(
                frame,
                count_text,
                (frame.shape[1] - text_width - 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (100, 255, 100),
                2,
            )

        instructions = ["Show your hands to the camera", "Press 'q' to quit"]
        y_offset = frame.shape[0] - 50
        for i, instruction in enumerate(instructions):
            cv2.putText(
                frame,
                instruction,
                (10, y_offset + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

        return frame

    def close(self):
        if self.hand_landmarker:
            self.hand_landmarker.close()


def main():
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Starting Hand Tracking Demo...")
    print("Press 'q' to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            output_frame = tracker.process_frame(frame)
            cv2.imshow("MediaPipe Hand Tracking", output_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        print("Demo ended")


if __name__ == "__main__":
    main()
