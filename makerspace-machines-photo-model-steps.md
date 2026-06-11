Full Step-by-Step Guide
Step 1 — iPhone Camera Settings
I don't have the iPhone 17 Pro Max specs (released after my knowledge cutoff), but these settings apply to any recent iPhone Pro:

Go to Settings → Camera → Formats and set:

Camera Capture: Most Compatible (JPEG, not HEIC — easier to work with)
Photo Resolution: 12MP — this is the sweet spot. Each photo is ~3–5MB, plenty of detail for YOLOv8 which resizes everything down to 640x640 anyway. ProRAW and 48MP are wasteful for this use case.
Turn OFF ProRAW and ProRes for this task
Step 2 — Photograph the Machines
For each machine, take 30 photos covering:

Shot Type	Count	Description
Front	5	Straight on, different distances (3ft, 6ft, 10ft)
Angles	8	45° left, 45° right, from above if possible
In-use context	5	Someone standing near/at it
Partial views	5	Half the machine visible, partially blocked
Lighting variations	5	Overhead lights on, off, shadows
Wide shot	2	Multiple machines in frame
Tips:

Hold the phone horizontally (landscape)
Don't use Portrait mode
Avoid flash — use natural/room lighting
Include the surrounding environment (floor, walls, other machines) — helps the model learn context
Step 3 — Set Up Roboflow
Go to roboflow.com and create a free account
Click Create New Project
Set:
Project Type: Object Detection
Name: something like makerspace-machines
Click Create Project
Step 4 — Upload Photos
Click Upload Data
Drag and drop all your photos
Roboflow will display them — click Save and Continue
Step 5 — Label the Images
Roboflow opens a labeling interface
For each image, draw a bounding box around each machine
Assign a class name (keep them simple and consistent):
laser_cutter
lathe
cnc_mill
3d_printer
etc.
Click Save after each image
Repeat for all images
Tip: Roboflow has an Auto-Label feature using their AI — it won't know your specific machines but can speed up rough bounding boxes that you then correct.

Step 6 — Augmentation
After labeling, click Generate New Version
Under Augmentation, enable:
Flip: Horizontal
Rotation: ±15°
Brightness: ±25%
Blur: up to 1.5px
Noise: up to 1%
Set augmentation to generate 3x your dataset size (30 images → ~90–100)
Click Generate
Step 7 — Export the Dataset
Click Export Dataset
Format: YOLOv8
Choose download zip to computer
You'll get a folder with train/, valid/, test/ subfolders each containing images/ and labels/
Step 8 — Train YOLOv8 on Your Jetson
Install Ultralytics if not already installed:


pip install ultralytics
Train the model:


yolo detect train \
  data=/path/to/your/roboflow/data.yaml \
  model=yolov8n.pt \
  epochs=50 \
  imgsz=640 \
  batch=8 \
  device=0
yolov8n = nano, the smallest/fastest — best for Jetson
Training will take 20–60 minutes depending on dataset size
Results saved to runs/detect/train/
Step 9 — Export to ONNX

yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=640
This creates best.onnx in the same folder.

Step 10 — Drop Into the ZED Pipeline
Copy the ZED custom detector sample that's already on your Jetson:


cp -r "/usr/local/zed/samples/object detection/custom detector/python/pytorch_yolo/" ~/MARS-CORE/Testing/custom-detector/
Then point it at your ONNX model:


python detector.py --weights best.onnx
The ZED SDK handles all the 3D positioning automatically — same as the current script you have running.

Summary of File Sizes to Expect
Stage	Size
30 photos at 12MP JPEG	~120–150MB
After Roboflow augmentation	~400MB
Trained best.pt model (YOLOv8n)	~6MB
Exported best.onnx	~12MB

