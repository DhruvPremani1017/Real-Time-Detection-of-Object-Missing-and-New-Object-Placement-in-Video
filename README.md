# Real-Time-Detection-of-Object-Missing-and-New-Object-Placement-in-Video

Real-Time Missing & New-Object Detection

Author : Dhruv Premani - ML Engineer Intern EvaluationTech Stack : YOLOv8-nano · ByteTrack · Python 3.11 · Docker · (CUDA-ready)  



Table of Contents

Problem Statement  
Solution Overview  
Quick Start – 1-line demo  
Architecture Details  
Performance  
Repo Structure  
Docker Build & Run  
Implementation Highlights  
Roadmap


Problem Statement
Detect in real time when  

an object first appears (NEW) and  
when a previously-tracked object is no longer visible (MISSING)

…while sustaining ≥ 25 FPS.Dataset: MOT20 (crowded pedestrians).

Solution Overview



Stage
Module
Time / frame (T4)



1
Frame grab (OpenCV)
0.2 ms


2
YOLOv8-nano detector (FP16)
31 ms


3
ByteTrack association
1.8 ms


4
Event-engine (dict<id,last_seen>)
< 0.3 ms


5
Annotate + encode (OpenCV)
1 ms


Total 34 ms ⇒ ~28 FPS @ 1080p.

Quick Start
git clone https://github.com/<your-handle>/mot20_rt.git
cd mot20_rt
python -m pip install -r requirements.txt
python src/rt_detect_events.py \
       --source demo_test.mp4 \
       --weights yolov8_mot20.pt \
       --out assets/output_events.mp4

Output video + fps.txt land in assets/.

Architecture Details

Detector – YOLOv8-nano (< 3 M params), fine-tuned 40 epochs on MOT20Det.  
Tracker – ByteTrack (CVPR 2022): fast IoU matcher, no ReID network.  
Event Logic – O( # active IDs ) dictionary;MISSING fires after 30 frames (≈ 1 s) of absence.  
Optimisations  
Layer fusion (model.fuse())  
Automatic Mixed Precision (AMP) on GPU  
Headless OpenCV + MP4V encoder  
Docker image < 650 MB (python:3.11-slim)




Performance



Metric
Value



FPS
28.6 (Tesla T4, 1080p)


mAP@0.5 (val)
83.1 %


Miss-Tolerance
30 frames


Detailed timings in assets/fps.txt.

Repo Structure
mot20_rt/
 ├─ src/
 │    ├─ rt_detect_events.py    ← main entrypoint
 │    └─ tools/mot_to_yolo.py   ← MOT-TXT → YOLO-TXT converter
 ├─ yolov8_mot20.pt             ← fine-tuned weights
 ├─ demo_test.mp4               ← unseen 15-s test clip
 ├─ assets/                     ← output_events.mp4, sample jpg, fps.txt
 ├─ Dockerfile
 ├─ requirements.txt
 └─ README.md


Docker Build & Run
# CPU image
docker build -t mot20_rt .
docker run --rm -v $PWD/assets:/app/assets mot20_rt

# GPU (if NVIDIA Container Toolkit is installed)
docker run --rm --gpus all -v $PWD/assets:/app/assets mot20_rt


Implementation Highlights

FP16 inference + fusion gives 32 % extra throughput over baseline FP32.  
ByteTrack chosen over DeepSORT to avoid heavy ReID net; still robust in dense scenes.  
Stateless event logic → negligible memory & compute, scales to 100+ IDs.  
Script auto-writes fps.txt, making benchmarking reproducible.


Roadmap

 Switch to YOLO-TensorRT export for +1.5× speed.  
 Adaptive miss-tolerance based on dynamic FPS.  
 Add DeepSORT variant with ReID for very heavy occlusions.


Made with ♥ by Dhruv Premani.Feel free to open an issue or reach out on LinkedIn for collaboration.
