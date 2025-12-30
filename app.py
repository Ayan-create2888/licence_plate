import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
from moviepy.editor import VideoFileClip

# Load YOLO model
model = YOLO("best_number.pt")

st.set_page_config(page_title="🚗 License Plate Detection", layout="centered")
st.title("🚗 License Plate Detection App")
st.write("Upload an image or video to detect vehicle number plates.")

# File uploader
uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

# ---------------- IMAGE DETECTION ----------------
def detect_image(image_np):
    results = model(image_np)
    boxes = results[0].boxes

    status = "✅ License Plate Found" if boxes is not None and len(boxes) > 0 else "❌ License Plate Not Found"
    annotated_image = results[0].plot()  # YOLO returns numpy array
    return annotated_image, status

# ---------------- VIDEO DETECTION ----------------
def detect_video(video_path):
    clip = VideoFileClip(video_path)

    def process_frame(frame):
        # frame is a numpy array
        results = model(frame)
        annotated_frame = results[0].plot()
        return annotated_frame

    processed_clip = clip.fl_image(process_frame)
    
    # Save to temporary file
    temp_file = "processed_video.mp4"
    processed_clip.write_videofile(temp_file, audio=False, verbose=False, logger=None)
    return temp_file

# ---------------- HANDLE FILE UPLOAD ----------------
if uploaded_file is not None:
    file_type = uploaded_file.type

    # -------- IMAGE --------
    if "image" in file_type:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        st.subheader("📷 Original Image")
        st.image(image_np, use_column_width=True)

        detected_image, status = detect_image(image_np)

        st.subheader("🔍 Detection Result")
        if "Found" in status:
            st.success(status)
        else:
            st.error(status)

        st.image(detected_image, use_column_width=True)

    # -------- VIDEO --------
    elif "video" in file_type:
        with st.spinner("Processing video... Please wait ⏳"):
            # Save uploaded video temporarily
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded_file.read())

            result_video_path = detect_video("temp_video.mp4")

        st.subheader("✅ Processed Video")
        st.video(result_video_path)
