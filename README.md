# makerspace-detector

Real-time safety monitoring for the TXST Ingram School of Engineering Makerspace. Uses a ZED stereo camera to detect CNC machines and track people simultaneously.

![Two different CNC machines object detection success](images\both-cnc-machine-detected.jpeg)

## How It Works

**Step 1 — Capture**
The ZED camera grabs a stereo frame. The left-eye image goes to YOLO for machine detection; both eyes are used by the ZED SDK to compute depth.

**Step 2 — CNC Machine Detection**
A custom YOLOv11 ONNX model (`best.onnx`) runs on each frame and detects three machines:
- `haas-st-20y`
- `haas-vf-2yt`
- `haas-vf-3`

It outputs 2D bounding boxes with confidence scores.

**Step 3 — 3D Localization**
The 2D boxes are handed to the ZED SDK via `ingest_custom_box_objects`. ZED combines them with its depth map to produce a 3D world position (x, y, z in meters) for each detected machine.

**Step 4 — Body Tracking**
Simultaneously, ZED's built-in `HUMAN_BODY_FAST` model detects and tracks people using an 18-point skeleton. Each person gets a persistent ID and skeletal overlay.

**Step 5 — Visualization**
Two views update in real time:
- **OpenCV window** — camera feed with machine labels, distance readouts, and skeleton overlays
- **OpenGL viewer** — live 3D point cloud of the scene with tracked objects

![Human body tracking and CNC machine object detection](images\cnc-machine-human-track.jpeg)

## Requirements

- ZED stereo camera (ZED SDK + `pyzed` installed)
- Python 3.10
- `ultralytics`, `opencv-python`, `numpy`
- (Optional) CuPy for GPU-accelerated point cloud transfer

---

## Running

```bash
python obj-detect-body-track.py
```

Replay a recorded `.svo` file instead of live camera:

```bash
python obj-detect-body-track.py --input_svo_file path/to/recording.svo
```

Disable GPU data transfer (if CuPy causes issues):

```bash
python obj-detect-body-track.py --disable-gpu-data-transfer
```

Press `Esc` to quit.

---

## Training a Custom Model

To train a new model for different machines, see [`makerspace-machines-photo-model-steps`](makerspace-machines-photo-model-steps) for a full walkthrough: photographing machines, labeling in Roboflow, training YOLOv11, exporting to ONNX, and dropping it into this pipeline.

After training, update `ONNX_MODEL_PATH` and `CNC_CLASSES` at the top of `obj-detect-body-track.py`.
