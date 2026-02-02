import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
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
    
    def draw_hand(self, image, hand_landmarks, hand_label):
        """Draw hand with custom styling"""
        h, w, _ = image.shape
        
        # Draw connections
        self.mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )
        
        # Get landmarks
        landmarks = hand_landmarks.landmark
        
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
        
        finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        
        for i, tip_idx in enumerate(finger_tip_indices):
            x = int(landmarks[tip_idx].x * w)
            y = int(landmarks[tip_idx].y * h)
            
            # Draw larger circle for extended fingers
            radius = 12 if fingers_up[i] else 8
            color = colors[i] if fingers_up[i] else (100, 100, 100)
            cv2.circle(image, (x, y), radius, color, -1)
            cv2.circle(image, (x, y), radius, (255, 255, 255), 2)
        
        # Draw finger count text
        wrist_x = int(landmarks[0].x * w)
        wrist_y = int(landmarks[0].y * h)
        
        # Background for text
        text = f"{hand_label}: {finger_count} fingers"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(
            image,
            (wrist_x - 10, wrist_y - text_height - 20),
            (wrist_x + text_width + 10, wrist_y + baseline),
            (0, 0, 0),
            -1
        )
        cv2.putText(
            image,
            text,
            (wrist_x, wrist_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        return finger_count
    
    def process_frame(self, frame):
        """Process a single frame"""
        # Flip horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        total_fingers = 0
        
        # Draw hands if detected
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_label = results.multi_handedness[idx].classification[0].label
                finger_count = self.draw_hand(frame, hand_landmarks, hand_label)
                total_fingers += finger_count
        
        # Add header with instructions
        header_height = 60
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], header_height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        title = "Finger Tracking Demo for Guitar AMT"
        cv2.putText(
            frame,
            title,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
        
        # Total finger count display
        if results.multi_hand_landmarks:
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
                2
            )
        
        # Instructions
        instructions = [
            "Show your hands to the camera",
            "Press 'q' to quit"
        ]
        y_offset = frame.shape[0] - 50
        for i, instruction in enumerate(instructions):
            cv2.putText(
                frame,
                instruction,
                (10, y_offset + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )
        
        return frame

def main():
    # Initialize hand tracker
    tracker = HandTracker()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set camera resolution
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
            
            # Process frame
            output_frame = tracker.process_frame(frame)
            
            # Display frame
            cv2.imshow('MediaPipe Hand Tracking', output_frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Demo ended")

if __name__ == "__main__":
    main()

