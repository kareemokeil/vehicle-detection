"""
Streamlit Web Application for YOLO Object Detection
Production-ready deployment UI for Streamlit Cloud
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.detector import YOLOModel
from utils.visualization import draw_boxes


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    .info-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .detection-stats {
        display: flex;
        justify-content: space-around;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    st.header("⚙️ Settings")
    
    # Confidence threshold slider
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Only detections with confidence above this threshold will be shown"
    )
    
    # Model info
    st.markdown("---")
    st.subheader("📊 Model Information")
    
    try:
        # Initialize model (cached for performance)
        @st.cache_resource
        def load_model():
            return YOLOModel()
        
        model = load_model()
        model_info = model.get_model_info()
        
        st.write(f"**Device:** {model_info['device'].upper()}")
        st.write(f"**Classes:** {model_info['num_classes']}")
        
        with st.expander("View Class Names"):
            for class_id, class_name in model_info['class_names'].items():
                st.write(f"{class_id}: {class_name}")
    
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.markdown(
    "<div class='title-container'><h1>🔍 YOLO Object Detection</h1><p>Real-time detection with YOLOv11</p></div>",
    unsafe_allow_html=True
)

# Create tabs for different input methods
tab1, tab2 = st.tabs(["📤 Upload Image", "🖼️ Use Demo Image"])

image_to_process = None
source_type = None

# ============================================================================
# TAB 1: IMAGE UPLOAD
# ============================================================================

with tab1:
    st.markdown("### Upload an image for detection")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Supported formats: JPG, PNG, BMP, WEBP"
    )
    
    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file)
        source_type = "uploaded"
        
        # Show preview
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info(f"📁 **Filename:** {uploaded_file.name}")
            st.info(f"📐 **Size:** {uploaded_file.size / 1024:.1f} KB")
        
        with col2:
            st.image(image_to_process, caption="Uploaded Image Preview", use_column_width=True)

# ============================================================================
# TAB 2: DEMO IMAGE
# ============================================================================

with tab2:
    st.markdown("### Use a demo image")
    
    demo_path = os.path.join(os.path.dirname(__file__), "assets", "demo.png")
    
    if os.path.exists(demo_path):
        st.info("✅ Demo image available")
        if st.button("Load Demo Image", key="load_demo"):
            image_to_process = Image.open(demo_path)
            source_type = "demo"
            st.success("Demo image loaded!")
            st.image(image_to_process, caption="Demo Image", use_column_width=True)
    else:
        st.warning(
            """
            ⚠️ No demo image found at `assets/demo.png`
            
            To add a demo image:
            1. Add an image file to the `assets/` folder
            2. Rename it to `demo.png`
            """
        )

# ============================================================================
# DETECTION & RESULTS
# ============================================================================

if image_to_process is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("---")
        run_detection = st.button(
            "🚀 Run Detection",
            use_container_width=True,
            type="primary"
        )
    
    with col2:
        st.markdown("---")
    
    if run_detection:
        with st.spinner("🔄 Processing image... Please wait"):
            try:
                # Convert PIL image to numpy array (BGR for OpenCV)
                image_array = cv2.cvtColor(np.array(image_to_process), cv2.COLOR_RGB2BGR)
                
                # Run inference
                detections = model.predict(image_array, confidence=confidence_threshold)
                
                # Draw boxes
                annotated_image = draw_boxes(image_array, detections)
                
                # Convert back to RGB for display
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                
                # Display results
                st.success("✅ Detection completed!")
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Detections", len(detections))
                with col2:
                    if detections:
                        avg_confidence = np.mean([d['confidence'] for d in detections])
                        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                    else:
                        st.metric("Avg Confidence", "N/A")
                with col3:
                    unique_classes = len(set(d['class_id'] for d in detections))
                    st.metric("Classes Detected", unique_classes)
                
                # Display images side by side
                st.markdown("---")
                st.markdown("### Detection Results")
                
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    st.image(image_to_process, caption="Original Image", use_column_width=True)
                
                with result_col2:
                    st.image(annotated_image_rgb, caption="Detection Results", use_column_width=True)
                
                # Detailed detections table
                if detections:
                    st.markdown("---")
                    st.markdown("### Detected Objects")
                    
                    # Prepare data for display
                    detection_data = []
                    for idx, det in enumerate(detections, 1):
                        detection_data.append({
                            "ID": idx,
                            "Class": det['class_name'],
                            "Confidence": f"{det['confidence']:.2%}",
                            "Box": f"[{int(det['box'][0])}, {int(det['box'][1])}, {int(det['box'][2])}, {int(det['box'][3])}]"
                        })
                    
                    st.dataframe(
                        detection_data,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Export results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Download annotated image
                        _, buffer = cv2.imencode('.png', annotated_image)
                        st.download_button(
                            label="📥 Download Annotated Image",
                            data=buffer.tobytes(),
                            file_name="detection_result.png",
                            mime="image/png"
                        )
                    
                    with col2:
                        # Download JSON results
                        import json
                        json_str = json.dumps(detections, indent=2)
                        st.download_button(
                            label="📊 Download JSON Results",
                            data=json_str,
                            file_name="detections.json",
                            mime="application/json"
                        )
                else:
                    st.info("ℹ️ No objects detected with the current confidence threshold. Try lowering the threshold.")
            
            except Exception as e:
                st.error(f"❌ Error during detection: {str(e)}")
                st.exception(e)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #999; margin-top: 30px;'>
        <p><strong>YOLO Object Detection | YOLOv11 | Streamlit Cloud</strong></p>
        <p>Built with ❤️ for production-grade deployments</p>
    </div>
    """,
    unsafe_allow_html=True
)
