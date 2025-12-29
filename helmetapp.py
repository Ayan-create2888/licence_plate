import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # force CPU (VERY IMPORTANT)

import streamlit as st
import tempfile
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="🪖 Helmet Detection", layout="centered")

st.title("🪖 Helmet Detection App")
st.write("Upload an image or video to detect helmet on heads.")

# ---------- LAZY LOAD OPENCV ----------
def load_cv2():
    import cv2
    return cv2

cv2 = load_cv2()

# ---------- LAZY LOAD YOLO MODEL ----------
@st.cache_resource
def load_model():
    return YOLO("best_helmet.pt")

model = load_model()

# ---------- FILE UPLOADER ----------
uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

# ---------------- IMAGE DETECTION ----------------
def detect_image_with_status(image):
    results = model(image)
    boxes = results[0].boxes

    status = "❌ Helmet Not Found"
    if boxes is not None and len(boxes) > 0:
        status = "✅ Helmet Found"

    annotated_image = results[0].plot()
    return annotated_image, status

# ---------------- VIDEO DETECTION ----------------
def detect_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()

# ---------------- HANDLE FILE UPLOAD ----------------
if uploaded_file is not None:
    file_type = uploaded_file.type

    # -------- IMAGE --------
    if "image" in file_type:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        st.subheader("📷 Original Image")
        st.image(image_np, use_container_width=True)

        detected_image, status = detect_image_with_status(image_np)

        st.subheader("🔍 Detection Result")
        st.success(status) if "Found" in status else st.error(status)

        st.image(detected_image, use_container_width=True)

        # Save result
        result_path = "detected_image.jpg"
        cv2.imwrite(result_path, cv2.cvtColor(detected_image, cv2.COLOR_RGB2BGR))

        with open(result_path, "rb") as file:
            st.download_button(
                "⬇️ Download Result Image",
                file,
                file_name="helmet_detected.jpg",
                mime="image/jpeg"
            )

    # -------- VIDEO --------
    elif "video" in file_type:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()

        st.subheader("🎥 Original Video")
        st.video(tfile.name)

        output_video = "detected_video.mp4"

        with st.spinner("Processing video... Please wait ⏳"):
            detect_video(tfile.name, output_video)

        st.subheader("✅ Processed Video")
        st.video(output_video)

        with open(output_video, "rb") as file:
            st.download_button(
                "⬇️ Download Result Video",
                file,
                file_name="helmet_detected.mp4",
                mime="video/mp4"
            )
