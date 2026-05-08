# 🚀 Quick Start Guide

Get your YOLO detection app running in 5 minutes!

## For Local Development

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Model
- Add `best.pt` to `model/` folder
- Ensure `labels.txt` is in `model/` folder

### 3. Test Locally
```bash
# Test the detector
python app.py

# Run web app
streamlit run streamlit_app.py
```

**Then open:** http://localhost:8501

---

## For Streamlit Cloud Deployment

### 1. Prepare Repository
```bash
git init
git add .
git commit -m "Initial YOLO deployment"
git remote add origin <your-repo-url>
git push -u origin main
```

### 2. Deploy
- Go to [Streamlit Cloud](https://share.streamlit.io/)
- Click "New app"
- Select repo, branch, and `streamlit_app.py`
- Click "Deploy"

### 3. Share
- Your app URL: `https://yolo-detection-username.streamlit.app/`

---

## Troubleshooting

### Model not found?
```
✅ Check: model/best.pt exists
✅ Check: model/labels.txt exists
```

### Import errors?
```bash
pip install --upgrade -r requirements.txt
```

### GPU not detected?
```bash
# Install CUDA version of PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Project Structure

```
📁 Project Root
├── 📄 app.py                    ← Test script
├── 📄 streamlit_app.py          ← Web app
├── 📄 requirements.txt          ← Dependencies
├── 📄 packages.txt              ← System libs
├── 📄 README.md                 ← Full docs
│
├── 📁 model/
│   ├── best.pt                  ← Model weights
│   └── labels.txt               ← Class names
│
├── 📁 utils/
│   ├── detector.py              ← Core logic
│   └── visualization.py         ← Drawing
│
└── 📁 assets/
    └── demo.png                 ← Demo image
```

---

## Next Steps

1. ✅ Review `README.md` for full documentation
2. ✅ Run `python app.py` to test locally
3. ✅ Customize confidence threshold settings
4. ✅ Add your demo image
5. ✅ Deploy to Streamlit Cloud

---

**Happy detecting! 🎯**
