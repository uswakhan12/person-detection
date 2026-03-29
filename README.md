# Person Detection with Persistent IDs

A lightweight Streamlit app that detects people in classroom videos using YOLO and maintains **persistent tracking IDs** via ByteTrack. Person #1 stays Person #1 throughout the entire video.

## Features

- **Person Detection** — YOLOv8 object detection (person class only, confidence threshold 0.25)
- **Persistent ID Tracking** — ByteTrack ensures each person retains the same ID across frames (no ID switching)
- **Unique Colors** — Each tracked person gets a consistent, vivid color throughout the video
- **Video Processing** — Upload MP4/AVI/MOV, adjust duration and frame skip for speed
- **Live Preview** — Real-time visualization during processing with people count per frame
- **Video Export** — Download annotated MP4 with bounding boxes and persistent IDs
- **Streamlit UI** — Upload video, start detection, and download results with a single click

## Requirements

- Python 3.10+
- YOLOv8 model weights (`yolov8x.pt` — included or download from [Ultralytics](https://github.com/ultralytics/ultralytics/releases))

## Setup

```bash
# Clone and enter project
cd person-tracker

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install opencv-python numpy streamlit ultralytics
```

Ensure `yolov8x.pt` is in the project root (it's required for detection).

## Run the app

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).

**How to use:**
1. Upload a classroom or any video (MP4, AVI, or MOV)
2. Optionally adjust processing duration (30–180 seconds) and frame skip (1, 2, or 3)
3. Click **START DETECTION & EXPORT VIDEO**
4. Watch live preview of detected people with persistent IDs
5. Download the processed MP4 when complete

## How it works

1. **Detection**: YOLOv8 detects all people (class 0) in each frame with confidence ≥ 0.25
2. **Tracking**: ByteTrack assigns and maintains persistent track IDs across frames
3. **Visualization**: Each person gets a unique, consistent color (HSV-based seed per ID)
4. **Export**: Processed frames are written to an MP4 with the same frame rate as the input
5. **Frame Skipping**: To speed up processing, frames can be skipped during detection (but written back with interpolated boxes for smooth output)

## Tech stack

- **Detection:** Ultralytics YOLOv8 (person detection only)
- **Tracking:** ByteTrack (persistent ID assignment)
- **UI & Export:** Streamlit, OpenCV (video I/O)

## License

Use and modify as needed for your project.
