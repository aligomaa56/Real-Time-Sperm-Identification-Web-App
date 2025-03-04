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
        
        # Significantly increased parameters for larger sperm detection
        self.min_sigma = 15      # Increased from 3 to detect larger objects
        self.max_sigma = 50      # Increased from 12 to capture large sperm heads
        self.threshold = 0.15    # Slightly reduced to increase sensitivity
        
        # Greatly increased size constraints for large sperm cells
        self.min_area = 500      # Increased from 50 for larger objects
        self.max_area = 5000     # Increased from 500 for larger objects
        
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
        """Detect potential sperm cells using blob detection with improved filtering."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))  # Increased tile size
        gray = clahe.apply(gray)
        
        # Apply Gaussian blur with larger kernel for bigger objects
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)  # Increased from (5,5)
        
        # Apply additional preprocessing to highlight sperm heads
        # Use larger block size for adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 23, 3)  # Increased from 11 to 23
        
        # Apply morphological operations with larger kernel
        kernel = np.ones((5, 5), np.uint8)  # Increased from (3,3)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Use this improved image for blob detection
        blobs = blob_dog(opening, min_sigma=self.min_sigma, max_sigma=self.max_sigma, 
                        threshold=self.threshold)
        
        # Filter blobs by checking gray level intensity
        filtered_blobs = []
        for blob in blobs:
            y, x, r = blob
            r = r * np.sqrt(2)  # Convert sigma to radius
            
            # Skip if radius is too small
            if r < 10:  # Increased minimum radius from 3
                continue
            
            # Create a circular mask for the blob
            mask = np.zeros_like(gray)
            cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
            
            # Calculate the mean intensity in the region
            mean_intensity = cv2.mean(gray, mask=mask)[0]
            
            # Adjusted intensity range for larger objects
            if 40 < mean_intensity < 220:  # Broadened range for larger objects
                filtered_blobs.append(blob)
        
        return np.array(filtered_blobs)
    
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
        """Check sperm morphology for abnormalities with adjusted criteria for larger cells."""
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)  # Increased kernel size
        morphed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return True  # Consider as abnormal if no clear contour
            
        # Get the largest contour
        c = max(contours, key=cv2.contourArea)
        
        # Calculate shape features
        area = cv2.contourArea(c)
        
        # Filter out by area - adjusted for larger sperm
        if area < self.min_area or area > self.max_area:
            return True
        
        perimeter = cv2.arcLength(c, True)
        
        # Calculate circularity and aspect ratio
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Get rotated rectangle for better elongation measurement
        rect = cv2.minAreaRect(c)
        (x, y), (width, height), angle = rect
        elongation = max(width, height) / min(width, height) if min(width, height) > 0 else 0
        
        # Slightly relaxed constraints for enlarged sperm
        if (circularity < 0.4 or circularity > 0.95 or  # Adjusted for larger objects
            aspect_ratio < 1.1 or aspect_ratio > 3.0 or  # Wider range for enlarged images
            elongation < 1.1 or elongation > 3.0):  # Wider range for enlarged images
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
        """Classify a detected cell as live, dead, or abnormal with improved checks."""
        roi, mask = self._extract_features(image, x, y, r)
        
        if roi.size == 0 or mask.size == 0:
            return "abnormal"  # Edge case
        
        # Check if region is too small
        if np.sum(mask > 0) < self.min_area:
            return "abnormal"
        
        # Look for tail-like structures (this helps confirm it's actually a sperm)
        has_tail = self._check_for_tail(image, x, y, r)
        if not has_tail:
            return "abnormal"  # Not a sperm if no tail is detected
        
        # Check if cell is red (dead)
        if self._is_red(roi, mask):
            return "dead"
            
        # Check morphology for abnormalities
        if self._check_morphology(mask):
            return "abnormal"
            
        # If not dead or abnormal, classify as live
        return "live"

    def _check_for_tail(self, image, x, y, r):
        """Check if there's a tail-like structure near the detected head, adjusted for large sperm."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Define a larger region to search for tail - increased search area
        search_radius = int(r * 4)  # Increased from r*3
        x1, y1 = max(0, int(x - search_radius)), max(0, int(y - search_radius))
        x2, y2 = min(image.shape[1], int(x + search_radius)), min(image.shape[0], int(y + search_radius))
        search_roi = gray[y1:y2, x1:x2]
        
        if search_roi.size == 0:
            return False
        
        # Apply edge detection with adjusted parameters
        edges = cv2.Canny(search_roi, 30, 120)  # Adjusted thresholds
        
        # Create a mask for the head to exclude it from tail detection
        head_mask = np.zeros_like(edges)
        head_center_x, head_center_y = int(x - x1), int(y - y1)
        cv2.circle(head_mask, (head_center_x, head_center_y), int(r), 255, -1)
        
        # Exclude the head region
        edges_without_head = cv2.bitwise_and(edges, edges, mask=cv2.bitwise_not(head_mask))
        
        # Look for line structures using Hough transform with adjusted parameters
        lines = cv2.HoughLinesP(edges_without_head, 1, np.pi/180, 20, 
                               minLineLength=int(r*0.8), maxLineGap=int(r*0.5))  # Adjusted parameters
        
        # If lines are found, check if any are connected to the head
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Calculate distance from line endpoints to head center
                dist1 = np.sqrt((x1 - head_center_x)**2 + (y1 - head_center_y)**2)
                dist2 = np.sqrt((x2 - head_center_x)**2 + (y2 - head_center_y)**2)
                
                # If one endpoint is close to the head and the line is long enough, consider it a tail
                if (dist1 < r*1.8 or dist2 < r*1.8) and np.sqrt((x2-x1)**2 + (y2-y1)**2) > r*0.8:
                    return True
        
        # Check percentage of edge pixels outside head
        edge_count = np.sum(edges_without_head > 0)
        if edge_count > r*8:  # Increased threshold based on expected larger tail size
            return True
        
        return False


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
            
            # Skip if radius is too small - increased minimum size
            if r < 10:  # Increased from 3 to 10
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