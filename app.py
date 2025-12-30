import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.title("🚗 License Plate Detection")

@st.cache_resource
def load_model():
    return YOLO("best_number.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "mov", "avi"]
)

if uploaded_file:
    suffix = os.path.splitext(uploaded_file.name)[1]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(uploaded_file.read())
    temp_file.close()

    if suffix.lower() in [".jpg", ".jpeg", ".png"]:
        st.image(Image.open(temp_file.name), caption="Original Image")

        results = model.predict(temp_file.name, save=True)
        output_path = results[0].save_dir + "/" + os.path.basename(temp_file.name)

        st.image(output_path, caption="Detected Image")

        with open(output_path, "rb") as f:
            st.download_button("⬇ Download Result", f, "license_plate.jpg")

    else:
        st.video(temp_file.name)

        results = model.predict(temp_file.name, save=True)
        output_video = results[0].save_dir + "/" + os.path.basename(temp_file.name)

        st.video(output_video)

        with open(output_video, "rb") as f:
            st.download_button("⬇ Download Result Video", f, "license_plate.mp4")

