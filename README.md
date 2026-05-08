# 🔍 YOLO Object Detection - Production Deployment

A production-ready YOLO YOLOv11 object detection system with Streamlit Cloud deployment capabilities. Built with best practices for scalability, maintainability, and performance.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Features](#features)
- [Local Setup](#local-setup)
- [Running Locally](#running-locally)
- [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
- [Model Configuration](#model-configuration)
- [Performance Tips](#performance-tips)
- [Live Demo](#live-demo)

---

## 🎯 Project Overview

This project provides a complete framework for deploying YOLO object detection models as web applications. It includes:

- **Modular Architecture**: Separate utilities for detection and visualization
- **Production-Ready Code**: Error handling, logging, and type hints
- **Streamlit UI**: Interactive web interface with image upload and demo capabilities
- **Cloud Deployment**: Optimized for Streamlit Cloud (Linux-compatible, no GPU required)
- **Local Testing**: Command-line script for local inference

**Model**: YOLOv11 (via Ultralytics)

---

## 📁 Folder Structure

```
yolo-terminal-deployment/
│
├── app.py                          # Local testing script
├── streamlit_app.py                # Web UI (main app)
├── requirements.txt                # Python dependencies
├── packages.txt                    # System dependencies (for Streamlit Cloud)
├── README.md                       # This file
│
├── model/
│   ├── best.pt                     # YOLO model weights
│   └── labels.txt                  # Class labels (one per line)
│
├── utils/
│   ├── __init__.py                 # Package marker
│   ├── detector.py                 # YOLOModel class
│   └── visualization.py            # Drawing utilities
│
└── assets/
    └── demo.png                    # Demo image (optional)
```

### File Descriptions

| File | Purpose |
|------|---------|
| `app.py` | Test script to run inference locally on demo image |
| `streamlit_app.py` | Main web application with Streamlit UI |
| `requirements.txt` | Python packages (pip install -r) |
| `packages.txt` | System libraries for cloud deployment |
| `model/best.pt` | Pre-trained YOLO model weights |
| `model/labels.txt` | Class names, one per line |
| `utils/detector.py` | Core detection logic (YOLOModel class) |
| `utils/visualization.py` | Bounding box drawing utilities |
| `assets/demo.png` | Example image for testing |

---

## ✨ Features

### 🎨 Web Interface (Streamlit)

- **Clean UI**: Modern, professional interface
- **Multiple Input Methods**: 
  - Upload custom images
  - Use demo image
- **Real-time Settings**:
  - Adjustable confidence threshold slider
  - Dynamic filtering of results
- **Detailed Results**:
  - Side-by-side comparison (original vs annotated)
  - Detection statistics table
  - Downloadable annotated images
  - JSON export of detections

### 🔧 Core Functionality

- **Auto Device Detection**: Automatically uses GPU if available, falls back to CPU
- **Dynamic Label Loading**: Reads class names from `labels.txt`
- **Structured Output**: Returns detections as JSON-serializable dictionaries
- **Error Handling**: Comprehensive validation and error messages

### 📊 Local Testing

- **Command-line Script**: Test model before deployment
- **Detailed Logging**: Shows model info, detections, and processing time
- **Output Export**: Saves annotated image as `output.jpg`

---

## 🚀 Local Setup

### Prerequisites

- Python 3.8+
- pip or conda
- (Optional) CUDA 11.8+ for GPU acceleration

### Installation

#### Option 1: Using pip (Recommended)

```bash
# Clone or navigate to project directory
cd yolo-terminal-deployment

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Option 2: Using conda

```bash
# Create conda environment
conda create -n yolo-detection python=3.9

# Activate environment
conda activate yolo-detection

# Install dependencies
pip install -r requirements.txt
```

---

## 🏃 Running Locally

### 1. Test with Local Script

```bash
# Activate virtual environment if not already activated
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Run local test script
python app.py
```

**Expected Output:**
```
======================================================================
YOLO Object Detection - Local Testing
======================================================================

📦 Loading model...
✅ Model loaded successfully
   Device: CUDA
   Number of classes: 5
   Classes: person, car, dog, cat, bird

🖼️  Loading test image from: assets/demo.png
   Image size: (640, 480)

🔍 Running inference...

📊 Detection Results
   Total detections: 3
   Objects found:
   #   Class Name           Confidence   Box (x1,y1,x2,y2)
   ────────────────────────────────────────────────────
   1   person               95.32%       [150, 200, 300, 450]
   2   dog                  87.65%       [400, 250, 550, 380]
   3   person               82.41%       [100, 100, 200, 350]

🎨 Drawing boxes and saving result...
   ✅ Result saved to: output.jpg
   Image size: 640x480

======================================================================
✅ Testing completed successfully!
======================================================================
```

### 2. Run Streamlit Web App

```bash
# Activate virtual environment if not already activated
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Run Streamlit app
streamlit run streamlit_app.py
```

**Expected Output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Then open your browser to `http://localhost:8501` and interact with the UI.

---

## ☁️ Streamlit Cloud Deployment

### Step-by-Step Deployment Guide

#### 1. Prepare Your Repository

```bash
# Ensure project structure matches requirements
# - All files in root directory
# - Model files in model/ folder
# - requirements.txt properly configured
```

#### 2. Create GitHub Repository

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial YOLO deployment project"

# Create repository on GitHub (github.com)
# Then push:
git remote add origin https://github.com/YOUR_USERNAME/yolo-deployment.git
git branch -M main
git push -u origin main
```

#### 3. Deploy to Streamlit Cloud

1. Go to [Streamlit Cloud](https://share.streamlit.io/)
2. Click **"New app"**
3. Select:
   - Repository: `YOUR_USERNAME/yolo-deployment`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
4. Click **"Deploy"**

**Note:** First deployment may take 5-10 minutes while dependencies install.

#### 4. Share Your App

Once deployed, your app URL will be: `https://yolo-deployment-YOUR_USERNAME.streamlit.app/`

### Deployment Configuration

**requirements.txt** includes:
- `ultralytics` - YOLO model
- `streamlit` - Web framework
- `opencv-python-headless` - Computer vision (no GUI)
- `Pillow` - Image processing
- `numpy` - Numerical computing

**packages.txt** includes system libraries required by OpenCV:
- `libgl1` - OpenGL support
- `libglib2.0-0` - System libraries

---

## ⚙️ Model Configuration

### Preparing Your Model

1. **Model File** (`model/best.pt`):
   - Place your trained YOLOv11 weights
   - Must be compatible with Ultralytics YOLO

2. **Labels File** (`model/labels.txt`):
   - Create a text file with class names
   - One class name per line
   - Order must match model training labels

**Example labels.txt:**
```
person
car
dog
cat
bird
```

### Loading Custom Models

To use a different model path:

```python
from utils.detector import YOLOModel

# Custom paths
model = YOLOModel(
    model_path="path/to/your/model.pt",
    labels_path="path/to/your/labels.txt"
)
```

---

## 📈 Performance Tips

### Optimization Strategies

1. **Confidence Threshold**:
   - Higher threshold → Faster, fewer false positives
   - Lower threshold → More detections, slower
   - Recommended: 0.5 (default)

2. **Input Image Size**:
   - Smaller images → Faster inference
   - The model handles automatic resizing
   - Sweet spot: 640x640 pixels

3. **GPU Acceleration**:
   - Local: Install CUDA 11.8+, PyTorch with GPU support
   - Cloud: Streamlit Cloud uses CPU (plan accordingly)

4. **Caching**:
   - Model loads once and is cached in memory
   - Subsequent inferences are fast
   - No reloading between requests

### Expected Performance

| Scenario | Device | Speed | Notes |
|----------|--------|-------|-------|
| Local Inference | GPU (CUDA) | ~10-50ms | Depending on GPU |
| Local Inference | CPU | ~100-500ms | Depends on CPU |
| Cloud Inference | CPU (Streamlit) | ~200-800ms | Free tier limitation |

---

## 🔧 Troubleshooting

### Common Issues

#### 1. "Model not found" Error

```
FileNotFoundError: Model not found at model/best.pt
```

**Solution:**
- Ensure `best.pt` exists in `model/` folder
- Check file permissions
- Verify file is not corrupted

#### 2. "No labels found" Error

```
FileNotFoundError: Labels file not found at model/labels.txt
```

**Solution:**
- Ensure `labels.txt` exists in `model/` folder
- File should have one class name per line
- No empty lines at the end

#### 3. Import Errors

```
ModuleNotFoundError: No module named 'ultralytics'
```

**Solution:**
```bash
# Reinstall requirements
pip install --upgrade -r requirements.txt

# Or install specific package
pip install ultralytics
```

#### 4. GPU Not Detected

```
Warning: CUDA not available, using CPU
```

**Solution:**
- Ensure CUDA 11.8+ is installed
- Install GPU-enabled PyTorch:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
- Restart Python kernel

#### 5. Streamlit Cloud Deployment Fails

**Common causes:**
- Missing `packages.txt` (needed for system libraries)
- `requirements.txt` has incompatible package versions
- Model file too large (>1 GB recommended limit)

**Solutions:**
- Add missing `packages.txt` with system dependencies
- Pin specific versions in `requirements.txt`
- Split large models or use model compression

---

## 📚 API Reference

### YOLOModel Class

```python
from utils.detector import YOLOModel

# Initialize
model = YOLOModel(model_path=None, labels_path=None)

# Predict
detections = model.predict(image, confidence=0.5)

# Get model info
info = model.get_model_info()
```

**Returns:**
```python
detections = [
    {
        "box": [x1, y1, x2, y2],          # Pixel coordinates
        "confidence": 0.95,                 # 0-1
        "class_id": 0,                      # Class index
        "class_name": "person"              # Class name
    },
    ...
]
```

### Visualization Functions

```python
from utils.visualization import draw_boxes

# Draw boxes on image
annotated = draw_boxes(
    image=numpy_array,
    detections=detections,
    confidence_threshold=0.5,
    thickness=2,
    font_scale=0.6
)
```

---

## 📝 Development

### Code Structure

```
utils/
├── detector.py          # Core detection logic
│   └── YOLOModel class  # Main inference class
│
└── visualization.py     # Visualization utilities
    ├── ColorPalette     # Color management
    ├── draw_boxes()     # Main drawing function
    └── create_comparison_image()  # Comparison view
```

### Adding Custom Features

**Example: Custom post-processing**

```python
from utils.detector import YOLOModel

model = YOLOModel()
detections = model.predict(image, confidence=0.5)

# Filter by class
person_detections = [d for d in detections if d['class_name'] == 'person']

# Sort by confidence
sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

# Custom logic
for detection in detections:
    if detection['confidence'] > 0.8:
        print(f"High confidence: {detection['class_name']}")
```

---

## 📄 License

This project is provided as-is for educational and deployment purposes.

---

## 🤝 Support

For issues or questions:

1. Check [Troubleshooting](#troubleshooting) section
2. Review [Streamlit Documentation](https://docs.streamlit.io)
3. Consult [Ultralytics YOLO Docs](https://docs.ultralytics.com)

---

## 🎓 Learning Resources

- [YOLOv11 Documentation](https://docs.ultralytics.com/models/yolov11/)
- [Streamlit Documentation](https://docs.streamlit.io)
- [OpenCV Documentation](https://docs.opencv.org)
- [YOLO Training Guide](https://docs.ultralytics.com/modes/train/)

---

## ✅ Checklist Before Production

- [ ] Model trained and tested
- [ ] `model/best.pt` added to project
- [ ] `model/labels.txt` created with correct class names
- [ ] `requirements.txt` versions pinned
- [ ] `packages.txt` includes required system libraries
- [ ] Local testing passes (`python app.py`)
- [ ] Streamlit app runs locally (`streamlit run streamlit_app.py`)
- [ ] Repository pushed to GitHub
- [ ] Deployed to Streamlit Cloud
- [ ] Demo image added to `assets/demo.png`
- [ ] README reviewed and updated

---
## 🚀 Live Demo
Try the working web app here:

👉 https://vehicle-detection-i4drmdydvgalprtbsrcurf.streamlit.app/
**Built with ❤️ for Production-Grade Deployments**

---

*Last Updated: 2026*
