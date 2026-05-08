"""
YOLOv11 Object Detection Module
Handles model loading, inference, and structured output
"""

import os
from pathlib import Path
from typing import Union, List, Dict, Optional
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO


class YOLOModel:
    """
    YOLOv11 Object Detection Model Wrapper
    
    Handles:
    - Model loading from model/best.pt
    - Automatic device detection (CPU/GPU)
    - Inference on images
    - Structured detection output
    - Dynamic label loading from model/labels.txt
    """
    
    def __init__(self, model_path: Optional[str] = None, labels_path: Optional[str] = None):
        """
        Initialize YOLO Model
        
        Args:
            model_path: Path to best.pt (default: model/best.pt)
            labels_path: Path to labels.txt (default: model/labels.txt)
        """
        # Set default paths relative to project root
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "..", "model", "best.pt")
        
        if labels_path is None:
            labels_path = os.path.join(os.path.dirname(__file__), "..", "model", "labels.txt")
        
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        
        # Validate paths
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found at {self.labels_path}")
        
        # Auto device detection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model
        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)
        
        # Load labels
        self.labels = self._load_labels()
        
    def _load_labels(self) -> Dict[int, str]:
        """
        Load class labels from labels.txt
        
        Returns:
            Dictionary mapping class_id to class_name
        """
        labels = {}
        try:
            with open(self.labels_path, 'r') as f:
                for idx, line in enumerate(f):
                    class_name = line.strip()
                    if class_name:
                        labels[idx] = class_name
            
            if not labels:
                raise ValueError("No labels found in labels.txt")
                
            return labels
        
        except Exception as e:
            raise RuntimeError(f"Error loading labels: {str(e)}")
    
    def predict(
        self,
        image: Union[Image.Image, np.ndarray, str],
        confidence: float = 0.5
    ) -> List[Dict]:
        """
        Run inference on image and return structured detections
        
        Args:
            image: PIL Image, numpy array, or image path
            confidence: Confidence threshold (0-1)
        
        Returns:
            List of detections:
            [
                {
                    "box": [x1, y1, x2, y2],
                    "confidence": float,
                    "class_id": int,
                    "class_name": str
                },
                ...
            ]
        """
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Run inference
            results = self.model(image, conf=confidence, device=self.device)
            
            # Parse results
            detections = []
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None:
                    for box, conf, cls_id in zip(
                        result.boxes.xyxy.cpu().numpy(),
                        result.boxes.conf.cpu().numpy(),
                        result.boxes.cls.cpu().numpy()
                    ):
                        cls_id = int(cls_id)
                        detection = {
                            "box": [float(x) for x in box],
                            "confidence": float(conf),
                            "class_id": cls_id,
                            "class_name": self.labels.get(cls_id, f"Unknown ({cls_id})")
                        }
                        detections.append(detection)
            
            return detections
        
        except Exception as e:
            raise RuntimeError(f"Inference failed: {str(e)}")
    
    def get_model_info(self) -> Dict:
        """
        Get model information
        
        Returns:
            Dictionary with model details
        """
        return {
            "model_path": str(self.model_path),
            "device": self.device,
            "num_classes": len(self.labels),
            "class_names": self.labels,
            "input_shape": self.model.model.model[0].conv.weight.shape if hasattr(self.model, 'model') else None
        }
