import cv2
import numpy as np
from PIL import Image
import io

class ImageProcessor:
    """Image preprocessing utilities for food classification"""
    
    def __init__(self):
        self.target_size = (224, 224)
    
    def preprocess_image(self, image):
        """Complete image preprocessing pipeline"""
        try:
            # Convert PIL to numpy array
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image
            
            # Convert to RGB if needed
            if len(image_array.shape) == 3 and image_array.shape[2] == 4:
                # Convert RGBA to RGB
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
                # Already RGB, but ensure correct order
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            
            # Apply preprocessing steps
            processed = self.resize_image(image_array)
            processed = self.enhance_image(processed)
            processed = self.normalize_image(processed)
            
            return processed
            
        except Exception as e:
            raise Exception(f"Image preprocessing failed: {str(e)}")
    
    def resize_image(self, image_array):
        """Resize image to target size with aspect ratio preservation"""
        h, w = image_array.shape[:2]
        
        # Calculate scaling to fit target size while maintaining aspect ratio
        scale = min(self.target_size[0] / w, self.target_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized = cv2.resize(image_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create canvas and center the image
        canvas = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
        
        # Calculate padding
        y_offset = (self.target_size[1] - new_h) // 2
        x_offset = (self.target_size[0] - new_w) // 2
        
        # Place resized image on canvas
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    
    def enhance_image(self, image_array):
        """Apply image enhancement techniques"""
        try:
            # Convert to LAB color space for better enhancement
            lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels back
            enhanced_lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            
            # Apply slight gaussian blur to reduce noise
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            return enhanced
            
        except Exception:
            # Fallback to original image if enhancement fails
            return image_array
    
    def normalize_image(self, image_array):
        """Normalize image values"""
        # Ensure values are in 0-255 range
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        return image_array
    
    def extract_features(self, image_array):
        """Extract basic image features for analysis"""
        try:
            # Convert to grayscale for feature extraction
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Calculate basic features
            features = {
                'mean_intensity': np.mean(gray),
                'std_intensity': np.std(gray),
                'brightness': np.mean(image_array),
                'contrast': np.std(gray),
                'sharpness': self._calculate_sharpness(gray)
            }
            
            return features
            
        except Exception as e:
            print(f"Feature extraction warning: {e}")
            return {}
    
    def _calculate_sharpness(self, gray_image):
        """Calculate image sharpness using Laplacian variance"""
        try:
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            return laplacian.var()
        except Exception:
            return 0.0
    
    def detect_edges(self, image_array):
        """Detect edges in the image for better food segmentation"""
        try:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply Canny edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            return edges
            
        except Exception:
            return None
    
    def segment_food(self, image_array):
        """Simple food segmentation using thresholding and morphology"""
        try:
            # Convert to HSV for better color segmentation
            hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
            
            # Create mask for food-like colors (avoiding pure background)
            lower_bound = np.array([0, 20, 20])
            upper_bound = np.array([180, 255, 255])
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            
            # Apply morphological operations to clean up mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Apply mask to original image
            segmented = cv2.bitwise_and(image_array, image_array, mask=mask)
            
            return segmented, mask
            
        except Exception:
            return image_array, None
