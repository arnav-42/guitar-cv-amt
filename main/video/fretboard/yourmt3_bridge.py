"""Transcribe archive recordings with a local YourMT3 checkout.

Run with the YourMT3 environment, not the fusion environment:

    ~/yourmt3_src/venv/bin/python main/video/fretboard/yourmt3_bridge.py \
        --yourmt3-dir ~/yourmt3_src/YourMT3 --clips 0 3 7

or, for any WAV file (as main/video/fretboard/video_fusion.py does):

    ~/yourmt3_src/venv/bin/python main/video/fretboard/yourmt3_bridge.py \
        --audio performance.wav --output performance_notes.csv

YourMT3 resolves checkpoints relative to its own directory, so this script
changes into it after resolving every input and output path. Notes are written
in full-WAV seconds as onset,offset,pitch,program,velocity; drums are excluded.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "archive/tablature_dataset/tablature_dataset"
# The Hugging Face space's default model: YPTF.MoE+Multi (noPS).
MODEL_NAME = "YPTF.MoE+Multi (noPS)"
MODEL_ARGS = ["mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt",
              "-p", "2024", "-tk", "mc13_full_plus_256", "-dec", "multi-t5", "-nl", "26",
              "-enc", "perceiver-tf", "-sqr", "1", "-ff", "moe", "-wf", "4", "-nmoe", "8",
              "-kmoe", "2", "-act", "silu", "-epe", "rope", "-rp", "1", "-ac", "spec",
              "-hop", "300", "-atc", "1", "-pr", "32"]


def transcribe(model, path, pitch_shift=0.):
    import soundfile
    import torch
    import torchaudio
    from utils.audio import slice_padded_array
    from utils.event2note import merge_zipped_note_events_and_ties_to_notes
    from utils.note2event import mix_notes

    samples, sr = soundfile.read(str(path), dtype="float32", always_2d=True)
    mono = samples.mean(1, keepdims=True).T
    # The archive recordings are quiet (peak ~0.14); YourMT3 misses most notes unnormalized.
    audio = torch.from_numpy(.9 * mono / max(float(abs(mono).max()), 1e-6))
    audio = torchaudio.functional.resample(audio, sr, model.audio_cfg["sample_rate"])
    if pitch_shift:
        import librosa
        audio = torch.from_numpy(librosa.effects.pitch_shift(
            audio.numpy(), sr=model.audio_cfg["sample_rate"], n_steps=pitch_shift))
    frames = model.audio_cfg["input_frames"]
    segments = torch.from_numpy(slice_padded_array(audio, frames, frames).astype("float32")).unsqueeze(1)
    with torch.no_grad():
        tokens, _ = model.inference_file(bsz=8, audio_segments=segments.to(model.device))
    starts = [frames * i / model.audio_cfg["sample_rate"] for i in range(segments.shape[0])]
    channels = []
    for ch in range(model.task_manager.num_decoding_channels):
        zipped, _, _ = model.task_manager.detokenize_list_batches(
            [arr[:, ch, :] for arr in tokens], starts, return_events=True)
        channels.append(merge_zipped_note_events_and_ties_to_notes(zipped)[0])
    duration = audio.shape[-1] / model.audio_cfg["sample_rate"]
    notes = [dict(onset=round(float(n.onset), 4), offset=round(min(float(n.offset), duration), 4),
                  pitch=int(n.pitch), program=int(n.program), velocity=int(n.velocity))
             for n in mix_notes(channels) if not n.is_drum]
    return sorted((n for n in notes if n["offset"] > n["onset"]),
                  key=lambda n: (n["onset"], n["pitch"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yourmt3-dir", type=Path, default=Path.home() / "yourmt3_src/YourMT3")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--clips", nargs="+", default=["0", "3", "7"])
    parser.add_argument("--pitch-shift", type=float, default=0.,
                        help="semitones applied before transcription, e.g. from "
                             "`python -m main.video.fretboard.event_fusion --estimate-pitch-offset 3`")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/yourmt3")
    parser.add_argument("--audio", type=Path, help="transcribe this WAV instead of archive clips")
    parser.add_argument("--output", type=Path, help="notes CSV path for --audio")
    args = parser.parse_args(argv)
    yourmt3_dir, out = args.yourmt3_dir.expanduser().resolve(), args.output_dir.resolve()
    if args.audio:
        target = (args.output or args.audio.with_name(args.audio.stem + "_notes.csv")).resolve()
        audio, names, out = {args.audio.stem: args.audio.resolve()}, {args.audio.stem: target}, target.parent
    else:
        audio = {c: (args.dataset / "tablature_audio" / f"{c}.wav").resolve() for c in args.clips}
        names = {c: out / f"{c}_notes.csv" for c in audio}
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(yourmt3_dir)
    sys.path[:0] = [str(yourmt3_dir), str(yourmt3_dir / "amt/src")]
    from model_helper import load_model_checkpoint
    model = load_model_checkpoint(args=MODEL_ARGS, device="cpu")
    for clip, path in audio.items():
        start = time.time()
        notes = transcribe(model, path, args.pitch_shift)
        target = names[clip]
        with target.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["onset", "offset", "pitch", "program", "velocity"])
            writer.writeheader()
            writer.writerows(notes)
        target.with_suffix(".metadata.json").write_text(json.dumps(dict(
            model=f"YourMT3 {MODEL_NAME}", source_audio=str(path), time_base="full_audio_seconds",
            event_count=len(notes), drums_excluded=True, pitch_shift_semitones=args.pitch_shift, seconds=round(time.time() - start, 1),
            note="Model predictions, not verified sounding-note references."), indent=2) + "\n")
        print(f"clip {clip}: {len(notes)} notes in {time.time() - start:.0f}s -> {target}", flush=True)


if __name__ == "__main__":
    main()
