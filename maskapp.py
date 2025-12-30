import streamlit as st
import tempfile
import numpy as np
import imageio.v2 as imageio
from ultralytics import YOLO
from PIL import Image

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="😷 Mask Detection", layout="centered")
st.title("😷 Mask Detection App")
st.write("Upload an image or video to detect masks on faces.")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("best_mask.pt")

model = load_model()

# ---------------- SHOW MODEL CLASSES ----------------
st.write("🧠 Model Classes:", model.names)

# ---------------- FILE UPLOADER ----------------
uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

# ---------------- IMAGE DETECTION ----------------
def detect_image(image_np):
    results = model.predict(
        source=image_np,
        conf=0.25,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return image_np, "❌ No Face Detected"

    class_names = model.names
    classes = result.boxes.cls.cpu().numpy()

    status = "❌ Mask Not Found"
    for cls in classes:
        if "mask" in class_names[int(cls)].lower():
            status = "✅ Mask Found"
            break

    annotated = result.plot(line_width=3, font_size=14)
    return annotated, status

# ---------------- VIDEO DETECTION ----------------
def detect_video(input_path, output_path):
    reader = imageio.get_reader(input_path)
    fps = reader.get_meta_data().get("fps", 25)

    writer = imageio.get_writer(output_path, fps=fps)

    for frame in reader:
        results = model.predict(
            source=frame,
            conf=0.25,
            imgsz=640,
            verbose=False
        )

        annotated = results[0].plot(line_width=2, font_size=12)
        writer.append_data(annotated)

    reader.close()
    writer.close()

# ---------------- HANDLE FILE ----------------
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
        st.success(status) if "Found" in status else st.error(status)
        st.image(detected_image, use_column_width=True)

        result_path = "detected_image.jpg"
        Image.fromarray(detected_image).save(result_path)

        with open(result_path, "rb") as f:
            st.download_button(
                "⬇️ Download Result Image",
                f,
                file_name="mask_detected.jpg",
                mime="image/jpeg"
            )

    # -------- VIDEO --------
    elif "video" in file_type:
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_input.write(uploaded_file.read())
        temp_input.close()

        st.subheader("🎥 Original Video")
        st.video(temp_input.name)

        output_video = "detected_video.mp4"

        with st.spinner("Processing video... ⏳"):
            detect_video(temp_input.name, output_video)

        st.subheader("✅ Processed Video")
        st.video(output_video)

        with open(output_video, "rb") as f:
            st.download_button(
                "⬇️ Download Result Video",
                f,
                file_name="mask_detected.mp4",
                mime="video/mp4"
            )
