"""
EduSense - Person Detection with Persistent IDs (ByteTrack)
=============================================================
Detects and boxes all people with stable IDs throughout the video.
Person #1 stays Person #1 — no ID switching.
"""

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO


# ─────────────────────────────────────────────
# Unique color per track ID (consistent colors)
# ─────────────────────────────────────────────
def id_to_color(track_id: int):
    """Generate a unique, vivid BGR color for each track ID."""
    np.random.seed(int(track_id) * 7 + 13)
    hue = int(np.random.randint(0, 180))
    color_hsv = np.uint8([[[hue, 220, 220]]])
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))


def draw_box(vis, x1, y1, x2, y2, track_id, conf):
    color = id_to_color(track_id)
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

    label = f"Person #{track_id}  {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    # Background pill for label
    cv2.rectangle(vis, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    cv2.putText(vis, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def main():
    st.set_page_config(page_title="EduSense - Person Detection", layout="wide")

    st.markdown("""
    <style>
    .main {background-color: #0a0e1a;}
    h1 {color: #00ffff; font-weight: bold;}
    .stButton>button {
        background: linear-gradient(90deg, #00ffff 0%, #0080ff 100%);
        color: black;
        font-weight: bold;
        font-size: 18px;
        padding: 15px 30px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🎥 EduSense - Person Detection (Persistent IDs)")
    st.markdown(
        "Draws a **persistent bounding box** around every person using **ByteTrack**. "
        "Person #1 remains Person #1 throughout the entire video."
    )
    st.markdown("---")

    uploaded_file = st.file_uploader("📹 Upload Video (MP4/AVI/MOV)", type=['mp4', 'avi', 'mov'])

    col1, col2 = st.columns(2)
    with col1:
        max_duration = st.slider("⏱️ Processing Duration (seconds)", 30, 180, 120, 15)
    with col2:
        process_every = st.selectbox("🎬 Frame Skip (for speed)", [1, 2, 3], index=1)

    if uploaded_file:
        input_path = "input_video.mp4"
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        if st.button("🚀 START DETECTION & EXPORT VIDEO", type="primary", use_container_width=True):
            st.markdown("---")

            video_placeholder = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()
            person_count_metric = st.empty()

            try:
                status_text.text("⏳ Loading YOLO + ByteTrack...")

                # Load model once
                detector = YOLO('yolov8x.pt')

                cap = cv2.VideoCapture(input_path)
                if not cap.isOpened():
                    st.error("Cannot open video file.")
                    return

                frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps          = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                max_frames   = min(total_frames, int(max_duration * fps))

                output_path = "detection_output.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

                frame_idx = 0
                status_text.text("🎬 Processing & Exporting Video...")

                while cap.isOpened() and frame_idx < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_idx += 1

                    if frame_idx % process_every == 0:

                        # ── ByteTrack tracking ───────────────────────────────
                        # persist=True keeps the ByteTracker alive across calls
                        # so IDs never reset between frames
                        results = detector.track(
                            frame,
                            classes=[0],              # person class only
                            conf=0.25,
                            iou=0.45,
                            tracker="bytetrack.yaml", # use ByteTrack
                            persist=True,             # KEY: maintains track state across frames
                            verbose=False
                        )

                        vis = frame.copy()
                        person_count = 0

                        for r in results:
                            if r.boxes.id is None:
                                # Tracker hasn't assigned IDs yet (first-frame edge case)
                                continue

                            boxes     = r.boxes.xyxy.cpu().numpy().astype(int)
                            track_ids = r.boxes.id.cpu().numpy().astype(int)
                            confs     = r.boxes.conf.cpu().numpy()

                            for (x1, y1, x2, y2), track_id, conf in zip(boxes, track_ids, confs):
                                w, h = x2 - x1, y2 - y1
                                if w < 20 or h < 30:
                                    continue

                                person_count += 1
                                draw_box(vis, x1, y1, x2, y2, track_id, conf)

                        # ── HUD ─────────────────────────────────────────────
                        overlay = vis.copy()
                        cv2.rectangle(overlay, (0, 0), (340, 65), (0, 0, 0), -1)
                        vis = cv2.addWeighted(vis, 0.72, overlay, 0.28, 0)
                        cv2.putText(vis, f"People Detected: {person_count}", (12, 42),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

                        # Write frame (repeated to compensate for skipped frames)
                        for _ in range(process_every):
                            writer.write(vis)

                        # Show in UI
                        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(vis_rgb, use_container_width=True)
                        person_count_metric.metric("👥 People in Frame", person_count)

                    progress_bar.progress(min(frame_idx / max_frames, 1.0))
                    status_text.text(f"🎬 Frame {frame_idx}/{max_frames}")

                cap.release()
                writer.release()

                st.success("✅ Processing Complete! IDs are stable throughout the video.")

                st.markdown("### 📥 Download Processed Video")
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="⬇️ DOWNLOAD MP4 VIDEO",
                        data=f,
                        file_name="person_detection_bytetrack.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()