import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load YOLO model
model = YOLO("best_number.pt")

st.set_page_config(page_title="License Plate Detection", layout="centered")

st.title("🚗 License Plate Detection App")
st.write("Upload an image or video to detect vehicle number plates.")

# File uploader
uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

# ---------------- IMAGE DETECTION FUNCTION ----------------
def detect_image_with_status(image):
    results = model(image)
    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        status = "✅ Number Plate Found"
    else:
        status = "❌ Number Plate Not Found"

    annotated_image = results[0].plot()
    return annotated_image, status

# ---------------- VIDEO DETECTION FUNCTION ----------------
def detect_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

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
        st.image(image_np, use_column_width=True)

        detected_image, status = detect_image_with_status(image_np)

        st.subheader("🔍 Detection Result")
        if "Found" in status:
            st.success(status)
        else:
            st.error(status)

        st.image(detected_image, use_column_width=True)

        # Save result
        result_path = "detected_image.jpg"
        cv2.imwrite(result_path, cv2.cvtColor(detected_image, cv2.COLOR_RGB2BGR))

        with open(result_path, "rb") as file:
            st.download_button(
                label="⬇️ Download Result Image",
                data=file,
                file_name="license_plate_detected.jpg",
                mime="image/jpeg"
            )

    # -------- VIDEO --------
    elif "video" in file_type:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()  # Important for Windows

        st.subheader("🎥 Original Video")
        st.video(tfile.name)

        output_video = "detected_video.mp4"

        with st.spinner("Processing video... Please wait ⏳"):
            detect_video(tfile.name, output_video)

        st.subheader("✅ Processed Video")
        st.video(output_video)

        with open(output_video, "rb") as file:
            st.download_button(
                label="⬇️ Download Result Video",
                data=file,
                file_name="license_plate_detected.mp4",
                mime="video/mp4"
            )
