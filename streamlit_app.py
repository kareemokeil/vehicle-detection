import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))

from utils.detector import YOLOModel
from utils.visualization import draw_boxes


# ================= CONFIG =================
st.set_page_config(
    page_title="YOLO Vehicle Detection",
    page_icon="🚗",
    layout="wide"
)

st.markdown("""
<style>
.title {
    text-align:center;
    font-size:40px;
    font-weight:bold;
    margin-bottom:20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🚗 Vehicle Detection YOLOv11</div>", unsafe_allow_html=True)


# ================= MODEL =================
@st.cache_resource
def load_model():
    return YOLOModel()

model = load_model()


# ================= SIDEBAR =================
st.sidebar.header("⚙️ Settings")

conf = st.sidebar.slider("Confidence", 0.0, 1.0, 0.5, 0.05)


# ================= INPUT =================
tab1, tab2 = st.tabs(["📤 Upload", "🖼️ Demo"])

image = None

# -------- Upload --------
with tab1:
    file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

    if file:
        image = Image.open(file)
        st.image(image, caption="Uploaded Image")


# -------- Demo --------
with tab2:
    demo_path = os.path.join("assets", "demo.jpg")

    if os.path.exists(demo_path):
        if st.button("Load Demo"):
            st.session_state["img"] = Image.open(demo_path)

    if "img" in st.session_state:
        image = st.session_state["img"]
        st.image(image, caption="Demo Image")


# ================= RUN =================
if image is not None:

    if st.button("🚀 Detect"):
        with st.spinner("Running YOLO..."):

            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            results = model.predict(img, confidence=conf)

            annotated = draw_boxes(img, results)

            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            st.success("Done!")

            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="Original")

            with col2:
                st.image(annotated, caption="Detected")

            st.markdown("### Results")

            st.write(results)