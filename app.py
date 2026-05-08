"""
Local Testing Script for YOLO Object Detection
Test the model locally before deploying to Streamlit Cloud
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.detector import YOLOModel
from utils.visualization import draw_boxes


def main():
    """Main testing function"""
    
    print("="*70)
    print("YOLO Object Detection - Local Testing")
    print("="*70)
    
    try:
        # Initialize model
        print("\n📦 Loading model...")
        model = YOLOModel()
        model_info = model.get_model_info()
        
        print(f"✅ Model loaded successfully")
        print(f"   Device: {model_info['device'].upper()}")
        print(f"   Number of classes: {model_info['num_classes']}")
        print(f"   Classes: {', '.join(model_info['class_names'].values())}")
        
        # Try to load demo image
        demo_image_path = os.path.join(os.path.dirname(__file__), "assets", "demo.png")
        
        if not os.path.exists(demo_image_path):
            print(f"\n⚠️  Demo image not found at {demo_image_path}")
            print("   Please add an image file to assets/demo.png for testing")
            return
        
        # Load image
        print(f"\n🖼️  Loading test image from: {demo_image_path}")
        image = Image.open(demo_image_path)
        print(f"   Image size: {image.size}")
        
        # Convert to numpy array for OpenCV
        image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Run inference
        print("\n🔍 Running inference...")
        detections = model.predict(image_array, confidence=0.5)
        
        # Print results
        print(f"\n📊 Detection Results")
        print(f"   Total detections: {len(detections)}")
        
        if detections:
            print("\n   Objects found:")
            print(f"   {'#':<3} {'Class Name':<20} {'Confidence':<12} {'Box (x1,y1,x2,y2)':<30}")
            print("   " + "-"*65)
            
            for idx, detection in enumerate(detections, 1):
                class_name = detection['class_name']
                confidence = detection['confidence']
                box = detection['box']
                box_str = f"[{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]"
                
                print(f"   {idx:<3} {class_name:<20} {confidence:<12.2%} {box_str:<30}")
        else:
            print("   No objects detected")
        
        # Draw boxes and save result
        print("\n🎨 Drawing boxes and saving result...")
        annotated_image = draw_boxes(image_array, detections)
        
        output_path = os.path.join(os.path.dirname(__file__), "output.jpg")
        cv2.imwrite(output_path, annotated_image)
        
        print(f"   ✅ Result saved to: {output_path}")
        print(f"   Image size: {annotated_image.shape[1]}x{annotated_image.shape[0]}")
        
        print("\n" + "="*70)
        print("✅ Testing completed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
