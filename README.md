# Real-Time Object Entry and Exit Detection

This project implements a real-time video analytics pipeline using **YOLOv8 + ByteTrack** to detect and track object entries and exits in a video stream. The primary goals are:

- ✅ **Detect new object placements** — Identify when an object appears in the frame that was not previously tracked.
- ❌ **Detect missing objects** — Identify when a previously visible object is no longer present in the scene.

---

## 🔧 Features

- 🔍 YOLOv8 (Nano) object detector for fast real-time inference.
- 🤖 ByteTrack multi-object tracking.
- ✅ Live detection of new and missing objects.
- 🎥 Output video with bounding boxes, IDs, and counters.
- ⌛ Optimized for high FPS and real-time performance.

---

## ⚙️ Tech Stack

- **Language**: Python
- **Libraries**: Ultralytics YOLOv8, OpenCV, PyTorch
- **Tracking**: ByteTrack via YOLOv8's built-in tracker
- **Platform**: Kaggle (GPU runtime)

---

## 🚀 Performance Metrics

- **Hardware Used**:
  - Platform: Kaggle Notebook
  - GPU: NVIDIA Tesla P100 (16 GB)
  - RAM: 13 GB
  - CPU: 2 vCPU cores

- **Video Duration**: ~20 seconds
- **Frame Count**: ~600 frames

---

## 📀 Output

- Annotated video with live entry/exit counters
- Text overlays showing object IDs, entry/missing lists

### 📍 Output Video
[Watch Output Video](https://drive.google.com/file/d/1Zs6jBK1RCMy2N_zyvSfTrwJ05cjPZrlL/view?usp=sharing)  
_Replace with actual hosted video URL (e.g. Google Drive, Kaggle Datasets)_


## 📅 How It Works

1. **Model Initialization**:
   - Loads YOLOv8n model
   - Sets device to CUDA if available
   - Enables fused inference for better performance

2. **Tracking & Inference Loop**:
   - Reads each frame from the input video
   - Runs YOLOv8 detection with ByteTrack tracking
   - Tracks object IDs across frames

3. **Entry/Missing Detection**:
   - Keeps a live inventory of object IDs using timestamps
   - Declares an ID as "new" if it's never been seen
   - Declares an ID as "missing" if not seen for `MISS_TOL` frames

4. **Output Rendering**:
   - Annotates frame with boxes and ID labels
   - Displays live counters of entered and exited IDs
   - Saves annotated video to disk

---

## 💡 Tips for Customization

- Change `MISS_TOL` to adjust sensitivity for missing detection.
- Use a different YOLOv8 variant (`yolov8s.pt`, `yolov8m.pt`) for better accuracy.
- Adjust resolution (`imgsz`) for performance/quality trade-off.
- Add filtering by class (e.g., only people or cars).

---

## 🔗 Repository Contents

- `real-time-detection.ipynb`: Inference and tracking logic
- `Real‑Time Object Inventory Tracking Report.pdf`: Detailed project report
- `README.md`: This file

---

## 👤 Author

**Dhruv Premani**  


