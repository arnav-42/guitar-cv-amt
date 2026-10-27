#!/usr/bin/env python3
"""Interactive image demo for YOLO fretboard detection and rectification."""

from __future__ import annotations

import argparse
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:  # Support both direct execution and ``python -m pipeline.gradio_demo``.
    from pipeline import fret_detect_yolo as detector
except ImportError:  # pragma: no cover - direct script execution
    import fret_detect_yolo as detector


DEFAULT_WEIGHTS = detector.DEFAULT_WEIGHTS
RECTIFIED_SIZE = (1024, 256)
SAMPLE_IMAGE = (
    Path(__file__).parents[1]
    / "demos"
    / "fretboard_canonicalization"
    / "guitar_test_clean.png"
)

_MODEL: Any = None
_MODEL_PATH: str | None = None
_MODEL_LOCK = threading.Lock()


def _get_model(weights_path: str) -> Any:
    """Load the model once and reuse it for subsequent uploads."""
    global _MODEL, _MODEL_PATH
    resolved = str(Path(weights_path).expanduser().resolve())
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Weights file not found: {resolved}")
    with _MODEL_LOCK:
        if _MODEL is None or _MODEL_PATH != resolved:
            _MODEL = detector.load_yolo_model(resolved)
            _MODEL_PATH = resolved
    return _MODEL


def _to_pil(image: Any) -> Image.Image | None:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("Upload an RGB image.")
    if array.shape[2] == 4:
        array = array[:, :, :3]
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def process_image(
    image: Any, weights_path: str = DEFAULT_WEIGHTS
) -> tuple[np.ndarray | None, np.ndarray | None, str, dict[str, Any]]:
    """Run inference and return annotated, rectified, summary, and JSON outputs."""
    source = _to_pil(image)
    if source is None:
        return None, None, "Upload a guitar image to begin.", {}

    try:
        model = _get_model(weights_path)
        mask = detector.predict_mask(model, source)
        fret_lines = detector.detect_frets(mask)
        annotated = detector.create_annotated_output(source, mask, fret_lines)
        rectification = detector.rectify_fretboard(
            mask,
            image=source,
            output_size=RECTIFIED_SIZE,
            return_details=True,
        )

        rectified = None if rectification is None else rectification["image"]
        rectified_dimensions = None if rectification is None else rectification["size"]
        report = detector.build_json_report(
            "<uploaded image>",
            source.size,
            weights_path,
            detector.YOLO_CONF_THRESH,
            mask,
            fret_lines,
            rectified_dimensions=rectified_dimensions,
            rectified_corners=(
                None if rectification is None else rectification["corners"]
            ),
            perspective_transform=(
                None if rectification is None else rectification["transform"]
            ),
        )
        coverage = report["mask"]["coverage_percent"]
        if rectification is None:
            summary = (
                "### No fretboard detected\n"
                "Try an image where the full guitar neck is clearly visible."
            )
        else:
            summary = (
                "### Detection summary\n"
                f"- Fretboard mask: **{coverage:.1f}%** of the image\n"
                f"- Projected fret lines: **{len(fret_lines)}**\n"
                f"- Rectified board: **{RECTIFIED_SIZE[0]} × {RECTIFIED_SIZE[1]} px**"
            )
        return annotated, rectified, summary, report
    except Exception as exc:  # Keep a bad upload from stopping the server.
        message = str(exc).strip() or exc.__class__.__name__
        return None, None, f"### Detection error\n`{message}`", {"error": message}


def build_demo(weights_path: str = DEFAULT_WEIGHTS) -> Any:
    """Build the UI without importing Gradio during CLI argument parsing."""
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Gradio is required for the web demo. Run: pip install -r requirements.txt"
        ) from exc

    with gr.Blocks(title="Guitar Fretboard Detection") as demo:
        gr.Markdown(
            "# Guitar fretboard detection\n"
            "Upload a guitar image to segment the neck, project the frets, and "
            "create a top-down rectangular view."
        )
        with gr.Row():
            image_input = gr.Image(type="pil", label="Guitar image")
            annotated_output = gr.Image(label="Detected fretboard and frets")
        rectified_output = gr.Image(label="Perspective-rectified fretboard")
        detect_button = gr.Button("Detect and rectify", variant="primary")
        summary_output = gr.Markdown("Upload a guitar image to begin.")
        geometry_output = gr.JSON(label="Machine-readable geometry")
        detect_button.click(
            fn=lambda image: process_image(image, weights_path),
            inputs=image_input,
            outputs=[
                annotated_output,
                rectified_output,
                summary_output,
                geometry_output,
            ],
        )
        if SAMPLE_IMAGE.is_file():
            gr.Examples([[str(SAMPLE_IMAGE)]], inputs=image_input, label="Example")
    return demo


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gradio UI for fretboard detection")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="YOLO weights path")
    parser.add_argument("--host", default="127.0.0.1", help="Server bind address")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    demo = build_demo(args.weights)
    print(f"Loading YOLO model from {args.weights}...")
    _get_model(args.weights)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
