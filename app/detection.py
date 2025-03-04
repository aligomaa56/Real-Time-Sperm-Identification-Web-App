import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage.feature import blob_dog
from skimage.color import rgb2hsv

class SpermDetector:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(__file__), 'static/models/model.h5')
        # Load model if it exists, otherwise use color-based detection
        self.model = self._load_model() if os.path.exists(self.model_path) else None
        # Parameters for blob detection
        self.min_sigma = 1
        self.max_sigma = 10
        self.threshold = 0.1
        # Color thresholds for red detection (in HSV)
        self.lower_red1 = np.array([0, 70, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([160, 70, 50])
        self.upper_red2 = np.array([180, 255, 255])
        
    def _load_model(self):
        """Load the TensorFlow model for sperm classification."""
        try:
            model = load_model(self.model_path)
            print("Model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
            
    def _detect_cells(self, image):
        """Detect potential sperm cells using blob detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Detect blobs
        blobs = blob_dog(blurred, min_sigma=self.min_sigma, max_sigma=self.max_sigma, threshold=self.threshold)
        return blobs
        
    def _is_red(self, image, mask):
        """Check if the region contains red color (indicating dead sperm)."""
        hsv = rgb2hsv(image)
        # Create mask for red hues
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        # Apply the sperm mask
        red_pixels = cv2.bitwise_and(red_mask, mask)
        # Calculate percentage of red pixels
        red_percentage = np.sum(red_pixels > 0) / np.sum(mask > 0) if np.sum(mask > 0) > 0 else 0
        return red_percentage > 0.3  # If more than 30% of pixels are red
    
    def _check_morphology(self, mask):
        """Check sperm morphology for abnormalities."""
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        morphed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return True  # Consider as abnormal if no clear contour
            
        # Get the largest contour
        c = max(contours, key=cv2.contourArea)
        
        # Calculate shape features
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        
        # Calculate circularity and aspect ratio
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Check if the shape is abnormal
        if circularity < 0.4 or aspect_ratio < 0.5 or aspect_ratio > 3.0:
            return True
            
        return False
        
    def _extract_features(self, image, x, y, r):
        """Extract a patch around the detected cell and create a binary mask."""
        # Create a circular mask
        h, w = image.shape[:2]
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
        mask = dist_from_center <= r
        mask = mask.astype(np.uint8) * 255
        
        # Extract the region of interest
        x1, y1 = max(0, int(x - r)), max(0, int(y - r))
        x2, y2 = min(w, int(x + r)), min(h, int(y + r))
        roi = image[y1:y2, x1:x2].copy()
        
        return roi, mask[y1:y2, x1:x2]
        
    def _classify_cell(self, image, x, y, r):
        """Classify a detected cell as live, dead, or abnormal."""
        roi, mask = self._extract_features(image, x, y, r)
        
        if roi.size == 0 or mask.size == 0:
            return "abnormal"  # Edge case
        
        # Check if cell is red (dead)
        if self._is_red(roi, mask):
            return "dead"
            
        # Check morphology for abnormalities
        if self._check_morphology(mask):
            return "abnormal"
            
        # If not dead or abnormal, classify as live
        return "live"
        
    def detect_sperm(self, image):
        """Detect and classify sperm cells in the image."""
        # Initialize counters
        live_count = 0
        dead_count = 0
        abnormal_count = 0
        detections = []
        
        # Detect cells
        blobs = self._detect_cells(image)
        
        for blob in blobs:
            y, x, r = blob
            r = r * np.sqrt(2)  # Convert sigma to radius
            
            # Skip if radius is too small
            if r < 3:
                continue
                
            # Classify the cell
            classification = self._classify_cell(image, x, y, r)
            
            # Update counters
            if classification == "live":
                live_count += 1
            elif classification == "dead":
                dead_count += 1
            elif classification == "abnormal":
                abnormal_count += 1
                
            # Add to detections list
            detections.append({
                'x': int(x),
                'y': int(y),
                'radius': int(r),
                'class': classification
            })
            
        # Calculate total and percentages
        total_count = live_count + dead_count + abnormal_count
        live_percentage = (live_count / total_count * 100) if total_count > 0 else 0
        dead_percentage = (dead_count / total_count * 100) if total_count > 0 else 0
        abnormal_percentage = (abnormal_count / total_count * 100) if total_count > 0 else 0
        
        return {
            'detections': detections,
            'live_count': live_count,
            'dead_count': dead_count,
            'abnormal_count': abnormal_count,
            'total_count': total_count,
            'live_percentage': live_percentage,
            'dead_percentage': dead_percentage,
            'abnormal_percentage': abnormal_percentage
        }