import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.detector import YOLOModel
from utils.visualization import draw_boxes


st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .title-container {
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return YOLOModel()


model = None
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")


with st.sidebar:
    st.header("Settings")

    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Only detections with confidence above this threshold will be shown"
    )

    st.markdown("---")
    st.subheader("Model Information")

    if model is not None:
        model_info = model.get_model_info()
        st.write(f"**Device:** {model_info['device'].upper()}")
        st.write(f"**Classes:** {model_info['num_classes']}")

        with st.expander("View Class Names"):
            for class_id, class_name in model_info['class_names'].items():
                st.write(f"{class_id}: {class_name}")
    else:
        st.warning("Model not available.")


st.markdown(
    "<div class='title-container'><h1>🔍 YOLO Object Detection</h1><p>Real-time detection with YOLOv11</p></div>",
    unsafe_allow_html=True
)

if model is None:
    st.stop()

tab1, tab2 = st.tabs(["Upload Image", "Demo Image"])

image_to_process = None

with tab1:
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info(f"**Filename:** {uploaded_file.name}")
            st.info(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
        with col2:
            st.image(image_to_process, caption="Uploaded Image", use_container_width=True)

with tab2:
    demo_path = os.path.join(os.path.dirname(__file__), "assets", "demo.png")

    if os.path.exists(demo_path):
        if st.button("Load Demo Image"):
            image_to_process = Image.open(demo_path)
            st.image(image_to_process, caption="Demo Image", use_container_width=True)
    else:
        st.warning("No demo image found at `assets/demo.png`")

if image_to_process is not None:
    st.markdown("---")
    run_detection = st.button("Run Detection", use_container_width=True, type="primary")

    if run_detection:
        with st.spinner("Processing image..."):
            try:
                image_array = cv2.cvtColor(np.array(image_to_process), cv2.COLOR_RGB2BGR)
                detections = model.predict(image_array, confidence=confidence_threshold)
                annotated_image = draw_boxes(image_array, detections)
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

                st.success("Detection completed!")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Detections", len(detections))
                with col2:
                    avg_conf = np.mean([d['confidence'] for d in detections]) if detections else None
                    st.metric("Avg Confidence", f"{avg_conf:.2f}" if avg_conf else "N/A")
                with col3:
                    st.metric("Classes Detected", len(set(d['class_id'] for d in detections)))

                st.markdown("---")
                result_col1, result_col2 = st.columns(2)
                with result_col1:
                    st.image(image_to_process, caption="Original", use_container_width=True)
                with result_col2:
                    st.image(annotated_image_rgb, caption="Detections", use_container_width=True)

                if detections:
                    st.markdown("---")
                    st.markdown("### Detected Objects")

                    detection_data = [
                        {
                            "ID": idx,
                            "Class": d['class_name'],
                            "Confidence": f"{d['confidence']:.2%}",
                            "Box": f"[{int(d['box'][0])}, {int(d['box'][1])}, {int(d['box'][2])}, {int(d['box'][3])}]"
                        }
                        for idx, d in enumerate(detections, 1)
                    ]
                    st.dataframe(detection_data, use_container_width=True, hide_index=True)

                    dl_col1, dl_col2 = st.columns(2)
                    with dl_col1:
                        _, buffer = cv2.imencode('.png', annotated_image)
                        st.download_button(
                            label="Download Annotated Image",
                            data=buffer.tobytes(),
                            file_name="detection_result.png",
                            mime="image/png"
                        )
                    with dl_col2:
                        st.download_button(
                            label="Download JSON Results",
                            data=json.dumps(detections, indent=2),
                            file_name="detections.json",
                            mime="application/json"
                        )
                else:
                    st.info("No objects detected. Try lowering the confidence threshold.")

            except Exception as e:
                st.error(f"Error during detection: {str(e)}")
                st.exception(e)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #999;'><p>YOLO Object Detection | YOLOv11 | Streamlit Cloud</p></div>",
    unsafe_allow_html=True
)