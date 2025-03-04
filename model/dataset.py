import os
import cv2
import numpy as np
import argparse
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def create_dataset_structure(base_dir):
    """Create the directory structure for the dataset."""
    os.makedirs(os.path.join(base_dir, 'live'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'dead'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'abnormal'), exist_ok=True)
    print(f"Created dataset directory structure in {base_dir}")

def process_images(input_dir, output_dir, img_size=(64, 64)):
    """Process raw sperm images into a proper dataset structure."""
    # Create the output directory structure
    create_dataset_structure(output_dir)
    
    # Find all image files in the input directory
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
        image_files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for img_path in tqdm(image_files):
        try:
            # Read the image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read image: {img_path}")
                continue
                
            # Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect and extract sperm cells
            cells = extract_sperm_cells(img)
            
            if not cells:
                print(f"No sperm cells detected in {img_path}")
                continue
                
            # Save each cell with appropriate label
            for i, (cell_img, label) in enumerate(cells):
                # Resize cell image
                cell_img = cv2.resize(cell_img, img_size)
                
                # Generate a unique filename
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                filename = f"{base_name}_cell{i}.png"
                
                # Save to the appropriate directory
                output_path = os.path.join(output_dir, label, filename)
                cv2.imwrite(output_path, cv2.cvtColor(cell_img, cv2.COLOR_RGB2BGR))
        
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    print("Image processing completed.")

def extract_sperm_cells(image):
    """Extract individual sperm cells from an image and classify them.
    This is a basic implementation and should be customized based on your specific microscope images.
    """
    cells = []
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold the image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size
    min_area = 100  # Minimum contour area
    max_area = 10000  # Maximum contour area
    
    valid_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
    
    # Process each valid contour
    for contour in valid_contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Add some padding
        padding = 10
        x_pad = max(0, x - padding)
        y_pad = max(0, y - padding)
        w_pad = min(image.shape[1] - x_pad, w + 2 * padding)
        h_pad = min(image.shape[0] - y_pad, h + 2 * padding)
        
        # Extract cell ROI
        cell_img = image[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad].copy()
        
        # Skip if the ROI is empty
        if cell_img.size == 0:
            continue
        
        # Classify the cell
        label = classify_cell(cell_img, contour)
        
        # Add to the list
        cells.append((cell_img, label))
    
    return cells

def classify_cell(cell_img, contour):
    """Classify a cell as live, dead, or abnormal based on color and morphology.
    This is a simplified implementation that needs to be adapted to your specific staining method.
    """
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(cell_img, cv2.COLOR_RGB2HSV)
    
    # Check for red (dead) cells
    # Define range of red color in HSV
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Threshold the HSV image to get only red colors
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Calculate the percentage of red pixels
    red_percentage = np.sum(red_mask > 0) / (cell_img.shape[0] * cell_img.shape[1])
    
    # If more than 15% of pixels are red, classify as dead
    if red_percentage > 0.15:
        return "dead"
    
    # Check morphology for abnormalities
    # Calculate shape features
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Calculate circularity
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    
    # If the shape is abnormal, classify as abnormal
    if circularity < 0.4 or aspect_ratio < 0.5 or aspect_ratio > 3.0:
        return "abnormal"
    
    # Otherwise, classify as live
    return "live"

def split_dataset(dataset_dir, train_ratio=0.8):
    """Split the dataset into training and validation sets."""
    for class_name in ['live', 'dead', 'abnormal']:
        class_dir = os.path.join(dataset_dir, class_name)
        image_files = glob.glob(os.path.join(class_dir, '*.png'))
        
        # Skip if no images found
        if not image_files:
            continue
        
        # Create train/val directories
        train_dir = os.path.join(dataset_dir, 'train', class_name)
        val_dir = os.path.join(dataset_dir, 'val', class_name)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        # Split the files
        train_files, val_files = train_test_split(image_files, train_size=train_ratio, random_state=42)
        
        # Copy files to train dir
        for file in train_files:
            filename = os.path.basename(file)
            os.rename(file, os.path.join(train_dir, filename))
        
        # Copy files to val dir
        for file in val_files:
            filename = os.path.basename(file)
            os.rename(file, os.path.join(val_dir, filename))
    
    print("Dataset split into training and validation sets")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process sperm images into a dataset.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing raw sperm images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the processed dataset.')
    parser.add_argument('--img_size', type=int, default=64, help='Output image size (square).')
    parser.add_argument('--split', action='store_true', help='Split dataset into train/val sets.')
    
    args = parser.parse_args()
    
    # Process the images
    process_images(args.input_dir, args.output_dir, (args.img_size, args.img_size))
    
    # Split the dataset if requested
    if args.split:
        split_dataset(args.output_dir)