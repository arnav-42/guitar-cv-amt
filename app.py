"""
Gradio Web Demo for Guitar CV AMT
Includes:
1. MediaPipe Hand Tracking Demo
2. FFT Chord Detection
"""
import cv2
import numpy as np
import gradio as gr
from hand_tracking_demo import HandTracker
from chord_detection import (
    enumerate_possible_chords,
    synth_chord_array,
    rank_chords_by_audio,
    SR
)

# Initialize hand tracker (lazy loading)
hand_tracker = None

def get_hand_tracker():
    """Lazy initialization of hand tracker"""
    global hand_tracker
    if hand_tracker is None:
        hand_tracker = HandTracker()
    return hand_tracker

def process_hand_tracking(video):
    """Process video frame for hand tracking"""
    if video is None:
        return None
    
    tracker = get_hand_tracker()
    
    # Gradio Video component returns a tuple: (video_path, subtitles_path)
    # or just a video path string
    if isinstance(video, tuple):
        video_path = video[0]
    else:
        video_path = video
    
    # Read frame from video
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        return None
    
    # Process frame
    processed_frame = tracker.process_frame(frame)
    
    # Convert BGR to RGB for Gradio
    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    
    return processed_frame_rgb

def process_chord_detection(cv_detected_str1, cv_detected_str2, cv_detected_str3, 
                           cv_detected_str4, cv_detected_str5, cv_detected_str6,
                           actual_str1, actual_str2, actual_str3,
                           actual_str4, actual_str5, actual_str6):
    """Process chord detection from CV input and actual chord"""
    
    # Parse CV detected chord (can be None, number, or "X")
    def parse_value(val):
        if val is None or val == "":
            return None
        val_str = str(val).strip().upper()
        if val_str == "X" or val_str == "M":
            return "X"
        try:
            return int(val_str)
        except:
            return None
    
    cv_detected = [
        parse_value(cv_detected_str1),
        parse_value(cv_detected_str2),
        parse_value(cv_detected_str3),
        parse_value(cv_detected_str4),
        parse_value(cv_detected_str5),
        parse_value(cv_detected_str6)
    ]
    
    # Parse actual chord
    actual = [
        parse_value(actual_str1) if actual_str1 not in [None, ""] else 0,
        parse_value(actual_str2) if actual_str2 not in [None, ""] else 0,
        parse_value(actual_str3) if actual_str3 not in [None, ""] else 0,
        parse_value(actual_str4) if actual_str4 not in [None, ""] else 0,
        parse_value(actual_str5) if actual_str5 not in [None, ""] else 0,
        parse_value(actual_str6) if actual_str6 not in [None, ""] else 0,
    ]
    
    # Convert None to 0 for actual chord
    actual = [0 if x is None else x for x in actual]
    actual = ["X" if x == "X" else int(x) for x in actual]
    
    try:
        # Generate candidate chords from CV detection
        candidates = enumerate_possible_chords(cv_detected)
        
        if not candidates:
            return "Error: No valid candidate chords generated. Check your CV detection input."
        
        # Synthesize actual chord audio
        actual_audio = synth_chord_array(actual, dur=2.2, sr=SR)
        
        # Rank candidates
        ranked = rank_chords_by_audio(
            candidates,
            test_audio=actual_audio,
            sr=SR,
            top_k=10
        )
        
        # Format results
        result_text = "## Chord Detection Results\n\n"
        result_text += f"**CV Detected:** {cv_detected}\n"
        result_text += f"**Actual Played:** {actual}\n\n"
        result_text += f"**Number of Candidates:** {len(candidates)}\n\n"
        result_text += "### Top 10 Matches:\n\n"
        result_text += "| Rank | Chord (E A D G B e) | Score |\n"
        result_text += "|------|---------------------|-------|\n"
        
        for i, (chord, score) in enumerate(ranked, 1):
            chord_str = " ".join(str(x) for x in chord)
            result_text += f"| {i} | {chord_str} | {score:.4f} |\n"
        
        return result_text
        
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Guitar CV AMT Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎸 Guitar CV AMT Demo
        
        Computer Vision for Automatic Music Transcription (Guitar)
        
        This demo includes two tools:
        1. **Hand Tracking**: Real-time finger tracking using MediaPipe
        2. **Chord Detection**: FFT-based chord identification from audio
        """
    )
    
    with gr.Tabs():
        with gr.Tab("🖐️ Hand Tracking"):
            gr.Markdown(
                """
                ### MediaPipe Hand Tracking Demo
                
                Upload a video or use your webcam to track hands and count fingers in real-time.
                The demo tracks up to 2 hands simultaneously with visual highlights for fingertips.
                """
            )
            
            with gr.Row():
                video_input = gr.Video(label="Input Video", sources=["webcam", "upload"])
                video_output = gr.Image(label="Processed Frame")
            
            process_btn = gr.Button("Process Frame", variant="primary")
            process_btn.click(
                fn=process_hand_tracking,
                inputs=video_input,
                outputs=video_output
            )
            
            gr.Markdown(
                """
                **Note:** For real-time processing, you can use the local `hand_tracking_demo.py` script.
                This web demo processes individual frames from uploaded videos or webcam captures.
                """
            )
        
        with gr.Tab("🎵 Chord Detection"):
            gr.Markdown(
                """
                ### FFT-Based Chord Detection
                
                This tool helps resolve ambiguity when computer vision detects finger positions but can't 
                determine if strings are fretted, muted, or open.
                
                **How it works:**
                1. Enter what the CV algorithm detected (can be fret numbers, 0 for open, X for muted, or leave blank)
                2. Enter the actual chord being played
                3. The system generates candidate chords and ranks them by audio similarity using FFT
                
                **String order:** Low E (6th) → A → D → G → B → High E (1st)
                """
            )
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### CV Detected (Input)")
                    gr.Markdown("Enter what computer vision detected. Use numbers for frets, 0 for open, X for muted, or leave blank.")
                    cv_str1 = gr.Textbox(label="String 1 (High E)", placeholder="e.g., 0, 3, X, or blank")
                    cv_str2 = gr.Textbox(label="String 2 (B)", placeholder="e.g., 0, 3, X, or blank")
                    cv_str3 = gr.Textbox(label="String 3 (G)", placeholder="e.g., 0, 3, X, or blank")
                    cv_str4 = gr.Textbox(label="String 4 (D)", placeholder="e.g., 0, 3, X, or blank")
                    cv_str5 = gr.Textbox(label="String 5 (A)", placeholder="e.g., 0, 3, X, or blank")
                    cv_str6 = gr.Textbox(label="String 6 (Low E)", placeholder="e.g., 0, 3, X, or blank")
                
                with gr.Column():
                    gr.Markdown("### Actual Chord (Played)")
                    gr.Markdown("Enter the actual chord being played. Use numbers for frets, 0 for open, X for muted.")
                    actual_str1 = gr.Textbox(label="String 1 (High E)", value="0", placeholder="0")
                    actual_str2 = gr.Textbox(label="String 2 (B)", value="0", placeholder="0")
                    actual_str3 = gr.Textbox(label="String 3 (G)", value="0", placeholder="0")
                    actual_str4 = gr.Textbox(label="String 4 (D)", value="2", placeholder="2")
                    actual_str5 = gr.Textbox(label="String 5 (A)", value="2", placeholder="2")
                    actual_str6 = gr.Textbox(label="String 6 (Low E)", value="0", placeholder="0")
            
            detect_btn = gr.Button("Detect Chord", variant="primary", size="lg")
            result_output = gr.Markdown(label="Results")
            
            detect_btn.click(
                fn=process_chord_detection,
                inputs=[
                    cv_str1, cv_str2, cv_str3, cv_str4, cv_str5, cv_str6,
                    actual_str1, actual_str2, actual_str3, actual_str4, actual_str5, actual_str6
                ],
                outputs=result_output
            )
            
            gr.Examples(
                examples=[
                    [
                        "0", "0", "1", "0", "0", "0",  # CV: E major detected
                        "0", "0", "0", "0", "0", "0"   # Actual: E minor
                    ],
                    [
                        "0", "0", "1", "0", "0", "0",  # CV: E major detected
                        "0", "0", "1", "0", "0", "0"   # Actual: E major
                    ],
                ],
                inputs=[
                    cv_str1, cv_str2, cv_str3, cv_str4, cv_str5, cv_str6,
                    actual_str1, actual_str2, actual_str3, actual_str4, actual_str5, actual_str6
                ],
                label="Example Configurations"
            )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)

