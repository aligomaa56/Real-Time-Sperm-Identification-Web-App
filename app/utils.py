import cv2
import numpy as np

def preprocess_image(image):
    """Preprocess the image for better detection."""
    # Resize if image is too large
    h, w = image.shape[:2]
    max_dim = 1024
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h))
    
    # Apply slight Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    
    # Enhance contrast
    lab = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    return enhanced

def draw_annotations(image, results):
    """Draw detection results on the image."""
    # Create a copy of the image
    annotated = image.copy()
    
    # Colors for different classes (BGR format)
    colors = {
        'live': (0, 255, 0),      # Green
        'dead': (0, 0, 255),      # Red
        'abnormal': (255, 165, 0)  # Orange
    }
    
    # Draw detections
    for detection in results['detections']:
        x = detection['x']
        y = detection['y']
        r = detection['radius']
        cell_class = detection['class']
        
        # Draw circle
        cv2.circle(annotated, (x, y), r, colors[cell_class], 2)
        
        # Add label
        cv2.putText(annotated, cell_class, (x - r, y - r - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[cell_class], 1)
    
    # Add statistics overlay
    stats_height = 100
    stats_bg = np.zeros((stats_height, image.shape[1], 3), dtype=np.uint8)
    
    # Text for statistics
    text = f"Live: {results['live_count']} ({results['live_percentage']:.1f}%)"
    cv2.putText(stats_bg, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['live'], 2)
    
    text = f"Dead: {results['dead_count']} ({results['dead_percentage']:.1f}%)"
    cv2.putText(stats_bg, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['dead'], 2)
    
    text = f"Abnormal: {results['abnormal_count']} ({results['abnormal_percentage']:.1f}%)"
    cv2.putText(stats_bg, text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['abnormal'], 2)
    
    # Add stats background to the bottom of the image
    annotated = np.vstack([annotated, stats_bg])
    
    return annotated