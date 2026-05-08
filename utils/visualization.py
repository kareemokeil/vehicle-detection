"""
Visualization Module for YOLO Detections
Handles bounding box drawing and image annotation
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple


class ColorPalette:
    """Generate consistent colors for bounding boxes"""
    
    def __init__(self, num_classes: int):
        """
        Initialize color palette
        
        Args:
            num_classes: Number of object classes
        """
        np.random.seed(42)  # For reproducibility
        self.colors = {}
        for i in range(num_classes):
            h = (i * 137.5) % 180
            s = 150 + np.random.randint(0, 105)
            v = 200 + np.random.randint(0, 55)
            hsv = np.uint8([[[h, s, v]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            self.colors[i] = tuple(int(x) for x in bgr[0][0])
    
    def get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get BGR color for class"""
        return self.colors.get(class_id, (0, 255, 0))


def draw_boxes(
    image: np.ndarray,
    detections: List[Dict],
    confidence_threshold: float = 0.0,
    thickness: int = -1,
    font_scale: float = 0.6
) -> np.ndarray:
    """
    Draw bounding boxes with labels on image
    
    Args:
        image: Input image (numpy array, BGR format)
        detections: List of detection dictionaries with keys:
                   - box: [x1, y1, x2, y2]
                   - confidence: float
                   - class_name: str
        confidence_threshold: Filter detections by confidence
        thickness: Line thickness (-1 for filled)
        font_scale: Font size multiplier
    
    Returns:
        Annotated image (numpy array, BGR format)
    """
    # Make a copy to avoid modifying original
    annotated = image.copy()
    
    # Initialize color palette
    color_palette = ColorPalette(len(set(det.get("class_id", 0) for det in detections)))
    
    # Get image dimensions for adaptive sizing
    height, width = image.shape[:2]
    base_thickness = max(1, int(width / 500))
    text_thickness = max(1, int(width / 700))
    
    # Filter detections by confidence
    filtered_detections = [
        d for d in detections 
        if d.get("confidence", 0) >= confidence_threshold
    ]
    
    # Draw each detection
    for detection in filtered_detections:
        try:
            box = detection.get("box", [])
            confidence = detection.get("confidence", 0)
            class_name = detection.get("class_name", "Unknown")
            class_id = detection.get("class_id", 0)
            
            if not box or len(box) < 4:
                continue
            
            # Convert box coordinates
            x1, y1, x2, y2 = [int(coord) for coord in box[:4]]
            
            # Ensure valid coordinates
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Get color
            color = color_palette.get_color(class_id)
            
            # Draw bounding box
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                thickness=base_thickness
            )
            
            # Create label with confidence
            label = f"{class_name} ({confidence:.2f})"
            
            # Get text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                font,
                font_scale,
                text_thickness
            )
            
            # Draw label background (solid rectangle)
            cv2.rectangle(
                annotated,
                (x1, y1 - text_height - baseline - 8),
                (x1 + text_width + 8, y1),
                color,
                -1  # Filled
            )
            
            # Draw label text (white color for contrast)
            cv2.putText(
                annotated,
                label,
                (x1 + 4, y1 - baseline - 4),
                font,
                font_scale,
                (255, 255, 255),  # White text
                text_thickness
            )
        
        except Exception as e:
            print(f"Warning: Failed to draw box: {str(e)}")
            continue
    
    return annotated


def create_comparison_image(
    original: np.ndarray,
    annotated: np.ndarray,
    margin: int = 20
) -> np.ndarray:
    """
    Create side-by-side comparison of original and annotated images
    
    Args:
        original: Original image
        annotated: Annotated image
        margin: Margin between images
    
    Returns:
        Combined comparison image
    """
    # Ensure same height
    if original.shape[0] != annotated.shape[0]:
        target_height = max(original.shape[0], annotated.shape[0])
        original = cv2.resize(original, (int(original.shape[1] * target_height / original.shape[0]), target_height))
        annotated = cv2.resize(annotated, (int(annotated.shape[1] * target_height / annotated.shape[0]), target_height))
    
    # Create black margin
    margin_strip = np.zeros((original.shape[0], margin, 3), dtype=np.uint8)
    
    # Concatenate horizontally
    comparison = np.hstack([original, margin_strip, annotated])
    
    return comparison
