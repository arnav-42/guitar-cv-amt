# GuiFiR benchmark scripts

These scripts run the GuiFiR benchmark: string detection, hand models, note-level scoring and an audio check. Run every command from the repository root as a module (`python -m scripts.guifir.<name>`).
- Shared modules: `pipeline/guifir.py` (loader and GT geometry) and `pipeline/string_detection.py` (string detectors A and B, and the confidence flag).
- Generated files go to `output/guifir/`, which is git-ignored.

## 1. Data (not in the repo)
Download the dataset from <https://github.com/kmu-ee-ailab/guifir_dataset>. It is for non-commercial research only and must not be redistributed.

```bash
mkdir -p data/guifir && gdown 1Ri42XQnpwWQzjNLOPgZCR2XCx9YGrqxc -O data/guifir/GuiFiR.tar   # 770 MB
tar -xf data/guifir/GuiFiR.tar -C data/guifir        # -> data/guifir/GuiFiR/{raw_clips,annotations,metadata}
```

- If the data lives elsewhere, set `GUIFIR_DIR`.
- Add `data/guifir/` and `third_party/` to `.gitignore`.
- Never commit frames or figures that show dataset images.

## 2. Environments
The models have conflicting dependencies, so each group gets its own conda environment. Steps that were tested on macOS (Apple silicon) are marked "macOS".

| Env | Used for | Setup |
|---|---|---|
| base (repo deps + `ultralytics`, `mediapipe`, `pandas`, `matplotlib`) | strings, MediaPipe, all evaluation and plots | repo requirements |
| `vitpose` (mmpose) | RTMPose hand crop, RTMW-x, ViTPose | see below |
| `hamer` | HaMeR | see below |
| `wilor` | WiLoR | see below |
| `audio` | Basic Pitch | `pip install basic-pitch librosa onnxruntime "scipy==1.14.1" "setuptools<70"` |

**mmpose (`vitpose` env).** mmcv must be built from source.
```bash
pip install "setuptools<70" wheel ninja
CFLAGS=-Wno-invalid-specialization CXXFLAGS=-Wno-invalid-specialization MMCV_WITH_OPS=1 \
  pip install --no-build-isolation "mmcv==2.1.0"          # macOS clang needs the flag
pip install --no-build-isolation chumpy && pip install "mmdet>=3.1,<3.3" "mmpose>=1.3"
pip install --no-build-isolation --force-reinstall --no-deps --no-binary xtcocotools xtcocotools  # numpy 2 ABI
```
- mmcv's NMS has no MPS kernel, so the mmpose models run on CPU.
- **ViTPose wholebody:** mmpose 1.x ships no wholebody ViTPose. Use easy_ViTPose instead:
  - `pip install --no-deps git+https://github.com/JunkyByte/easy_ViTPose.git`
  - Put the weights in `third_party/vitpose/vitpose-{b,l}-wholebody.pth`, from <https://huggingface.co/JunkyByte/easy_ViTPose/tree/main/torch/wholebody>.

**HaMeR and WiLoR.** Clone the repos into `third_party/hamer` and `third_party/WiLoR`.
1. **MANO:** register at <https://mano.is.tue.mpg.de> and copy `MANO_RIGHT.pkl` into:
   - `third_party/hamer/_DATA/data/mano/`
   - `third_party/WiLoR/mano_data/`
2. **HaMeR install:** `pip install --no-build-isolation chumpy`, then `pip install numpy opencv-python pytorch-lightning scikit-image "smplx==0.1.28" yacs timm einops pandas trimesh webdataset`, then `pip install --no-deps -e third_party/hamer`. This skips detectron2 and mmcv 1.x, which we don't need because we pass hand boxes.
3. **HaMeR weights:** `third_party/hamer/fetch_demo_data.sh`, which downloads 6 GB. With gdown 6, pass the id without `--id`.
4. **WiLoR:** install the same packages, and put `wilor_final.ckpt` in `third_party/WiLoR/pretrained_models/`.
5. **macOS:** use `scipy==1.14.1`.
6. **Rendering:** pyrender/EGL is stubbed inside `predict_mesh.py`; rendering is not needed.

Licences: HaMeR is MIT. WiLoR is CC BY-NC-ND. MANO has its own licence.

## 3. Run (test split)
```bash
python -m scripts.guifir.compare_strings --split test --save-lines      # strings A/B/fusion (~40 min)
python -m scripts.guifir.predict_tips --model mediapipe                 # base env
<vitpose env> -m scripts.guifir.predict_tips --model hand_crop          # also: rtmw, vitpose_b, vitpose_l
<hamer env>   -m scripts.guifir.predict_mesh --model hamer --split test
<hamer env>   -m scripts.guifir.predict_mesh --model hamer --split val  # for the contact calibration
<wilor env>   -m scripts.guifir.predict_mesh --model wilor --split test
<audio env>   -m scripts.guifir.extract_audio --split test
python -m scripts.guifir.eval_tips --models hamer wilor hand_crop rtmw mediapipe vitpose_b vitpose_l
python -m scripts.guifir.eval_notes --models hamer hand_crop
python -m scripts.guifir.eval_audio
python -m scripts.guifir.eval_av_preview
python -m scripts.guifir.plots                                          # --examples: local-only frame figures
```

## Files
| File | What it does |
|---|---|
| `common.py` | Output paths, `score_lines` (string error in spacings), YOLO mask cache |
| `compare_strings.py` | String detection A, B and fusion vs the GT strings |
| `predict_tips.py` | MediaPipe, RTMPose hand crop, RTMW-x, ViTPose-B/L |
| `predict_mesh.py` | HaMeR, WiLoR. `--boxes wilor_det` uses WiLoR's own detector (the paper's setting; needs `--detect-only` first) |
| `eval_tips.py` | Fingertip error, and the share of fingertips on the correct string |
| `eval_notes.py` | Note-level video-only scoring, plus the contact calibration learned on val |
| `extract_audio.py`, `eval_audio.py` | Basic Pitch notes, pitch recall, positions per pitch |
| `eval_av_preview.py` | Preliminary audio + video position picking |
| `plots.py` | Figures g5–g9 (g1/g2 also need the Roboflow results, which are not part of this upload) |
