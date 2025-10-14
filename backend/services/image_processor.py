import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional


class ImageProcessor:
    """
    Service for preprocessing check images.
    Stage 2: Preprocessing (10-20%)
    """
    
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = None
        self.processed_image = None
    
    def load_image(self) -> bool:
        """Load image from file path."""
        try:
            self.image = cv2.imread(self.image_path)
            if self.image is None:
                return False
            return True
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            return False
    
    def enhance_image(self) -> np.ndarray:
        """
        Enhance image quality by adjusting brightness and contrast.
        """
        if self.image is None:
            self.load_image()
        
        # Convert to LAB color space
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction using Non-local Means Denoising.
        """
        denoised = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=10,
            hColor=10,
            templateWindowSize=7,
            searchWindowSize=21
        )
        return denoised
    
    def correct_orientation(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct image orientation using edge detection.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        if lines is not None and len(lines) > 0:
            # Calculate average angle
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta) - 90
                angles.append(angle)
            
            median_angle = np.median(angles)
            
            # Only rotate if angle is significant (> 1 degree)
            if abs(median_angle) > 1:
                return self.rotate_image(image, median_angle)
        
        return image
    
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by specified angle.
        """
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        Deskew image to correct perspective distortion.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        
        # Threshold the image
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # Find coordinates of all non-zero points
        coords = np.column_stack(np.where(thresh > 0))
        
        # Find minimum area rectangle
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            
            # Adjust angle
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            # Only deskew if angle is significant
            if abs(angle) > 0.5:
                return self.rotate_image(image, angle)
        
        return image
    
    def sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """
        Sharpen image to improve text clarity.
        """
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened
    
    def preprocess(self) -> dict:
        """
        Execute complete preprocessing pipeline.
        Returns processed image and preprocessing results.
        """
        results = {
            "success": False,
            "steps_completed": [],
            "original_size": None,
            "processed_size": None,
            "warnings": []
        }
        
        # Load image
        if not self.load_image():
            results["error"] = "Failed to load image"
            return results
        
        results["original_size"] = self.image.shape[:2]
        results["steps_completed"].append("Image loaded")
        
        # Step 1: Enhance brightness and contrast
        try:
            enhanced = self.enhance_image()
            results["steps_completed"].append("Enhanced quality")
        except Exception as e:
            results["warnings"].append(f"Enhancement failed: {str(e)}")
            enhanced = self.image.copy()
        
        # Step 2: Reduce noise
        try:
            denoised = self.reduce_noise(enhanced)
            results["steps_completed"].append("Noise reduced")
        except Exception as e:
            results["warnings"].append(f"Noise reduction failed: {str(e)}")
            denoised = enhanced
        
        # Step 3: Correct orientation
        try:
            oriented = self.correct_orientation(denoised)
            results["steps_completed"].append("Orientation corrected")
        except Exception as e:
            results["warnings"].append(f"Orientation correction failed: {str(e)}")
            oriented = denoised
        
        # Step 4: Deskew
        try:
            deskewed = self.deskew_image(oriented)
            results["steps_completed"].append("Image deskewed")
        except Exception as e:
            results["warnings"].append(f"Deskewing failed: {str(e)}")
            deskewed = oriented
        
        # Step 5: Sharpen for better OCR
        try:
            sharpened = self.sharpen_image(deskewed)
            results["steps_completed"].append("Image sharpened")
        except Exception as e:
            results["warnings"].append(f"Sharpening failed: {str(e)}")
            sharpened = deskewed
        
        self.processed_image = sharpened
        results["processed_size"] = self.processed_image.shape[:2]
        results["success"] = True
        
        return results
    
    def save_processed_image(self, output_path: str) -> bool:
        """Save processed image to file."""
        if self.processed_image is None:
            return False
        
        try:
            cv2.imwrite(output_path, self.processed_image)
            return True
        except Exception as e:
            print(f"Error saving processed image: {str(e)}")
            return False
    
    def get_processed_image(self) -> Optional[np.ndarray]:
        """Return processed image array."""
        return self.processed_image
    
    def get_image_quality_score(self) -> dict:
        """
        Assess image quality for check analysis.
        """
        if self.image is None:
            self.load_image()
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Check resolution
        height, width = gray.shape
        min_dimension = min(height, width)
        resolution_score = min(100, (min_dimension / 1000) * 100)
        
        # Check sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(100, laplacian_var / 10)
        
        # Check brightness
        mean_brightness = np.mean(gray)
        brightness_score = 100 - abs(mean_brightness - 127) / 1.27
        
        overall_score = (resolution_score + sharpness_score + brightness_score) / 3
        
        return {
            "overall_score": round(overall_score, 2),
            "resolution_score": round(resolution_score, 2),
            "sharpness_score": round(sharpness_score, 2),
            "brightness_score": round(brightness_score, 2),
            "dimensions": {"width": width, "height": height},
            "passes_minimum_quality": overall_score >= 50
        }

