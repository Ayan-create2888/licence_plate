import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load YOLO model
model = YOLO("best_helmet.pt")

st.set_page_config(page_title="🪖 Helmet Detection", layout="centered")

st.title("🪖 Helmet Detection App")
st.write("Upload an image to detect helmets.")

# File uploader (IMAGE ONLY)
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- IMAGE DETECTION FUNCTION ----------------
def detect_image_with_status(image_np):
    results = model(image_np)
    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        status = "✅ Helmet Found"
    else:
        status = "❌ Helmet Not Found"

    annotated_image = results[0].plot()
    return annotated_image, status

# ---------------- HANDLE FILE UPLOAD ----------------
if uploaded_file is not None:
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

    # Save result using PIL (NO cv2)
    result_image = Image.fromarray(detected_image)
    result_path = "helmet_detected.jpg"
    result_image.save(result_path)

    with open(result_path, "rb") as file:
        st.download_button(
            label="⬇️ Download Result Image",
            data=file,
            file_name="helmet_detected.jpg",
            mime="image/jpeg"
        )
