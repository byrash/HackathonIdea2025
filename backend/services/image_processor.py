import cv2
import numpy as np
from PIL import Image, ExifTags
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
        """Load image from file path and handle EXIF orientation."""
        try:
            # First, handle EXIF orientation using PIL
            pil_image = Image.open(self.image_path)
            
            # Check for EXIF orientation tag
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break
                exif = pil_image._getexif()
                if exif is not None:
                    orientation_value = exif.get(orientation)
                    if orientation_value == 3:
                        pil_image = pil_image.rotate(180, expand=True)
                    elif orientation_value == 6:
                        pil_image = pil_image.rotate(270, expand=True)
                    elif orientation_value == 8:
                        pil_image = pil_image.rotate(90, expand=True)
            except (AttributeError, KeyError, IndexError):
                # No EXIF data, continue normally
                pass
            
            # Convert PIL to OpenCV format
            self.image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
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
    
    def detect_check_orientation(self, image: np.ndarray) -> int:
        """
        Detect if check needs 90/180/270 degree rotation.
        Returns: 0 (no rotation), 90, 180, or 270 degrees
        
        Uses aspect ratio to detect orientation.
        Standard checks have 2:1 to 3:1 width:height ratio (landscape).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Check aspect ratio - checks are typically wider than tall (landscape)
        aspect_ratio = w / h
        
        # Standard check dimensions are roughly 6" x 2.75" (ratio ~2.18:1)
        # If portrait orientation (height > width), needs rotation
        if aspect_ratio < 0.95:  # Clearly portrait (allow 5% margin)
            print(f"📐 Portrait detected ({w}x{h}, ratio {aspect_ratio:.2f}), rotating 90°")
            return 90
        elif aspect_ratio >= 0.95 and aspect_ratio <= 1.5:
            # Nearly square or slight landscape - could be a cropped check
            # Check if it's actually portrait that needs rotation
            if aspect_ratio < 1.05:
                print(f"📐 Nearly square ({w}x{h}, ratio {aspect_ratio:.2f}), might need rotation")
                # Try OCR-based detection: look for text orientation
                # For now, don't rotate - assume correct
                print("   → Keeping as-is (within tolerance)")
            else:
                print(f"📐 Moderate landscape ({w}x{h}, ratio {aspect_ratio:.2f}), OK")
            return 0
        else:
            # Good landscape orientation (ratio > 1.5)
            print(f"📐 Landscape detected ({w}x{h}, ratio {aspect_ratio:.2f}), orientation OK")
        
        return 0
    
    def correct_orientation(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct check orientation (90/180/270 degree rotations).
        Handles rotated check images properly.
        """
        # First detect if major rotation is needed
        rotation_needed = self.detect_check_orientation(image)
        
        if rotation_needed == 90:
            print("🔄 Rotating check 90° clockwise")
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_needed == 180:
            print("🔄 Rotating check 180°")
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif rotation_needed == 270:
            print("🔄 Rotating check 270° (90° counter-clockwise)")
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
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
        Deskew image to correct minor skew (limited to prevent incorrect rotation).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        
        # Threshold the image
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # Find coordinates of all non-zero points
        coords = np.column_stack(np.nonzero(thresh > 0))
        
        # Find minimum area rectangle
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            
            # Adjust angle to range [-45, 45]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            # CONSERVATIVE: Only deskew minor angles (< 5 degrees) to prevent 90-degree rotations
            # This prevents minAreaRect from incorrectly rotating landscape images
            if abs(angle) > 0.5 and abs(angle) < 5:
                return self.rotate_image(image, angle)
        
        return image
    
    def auto_crop_check(self, image: np.ndarray) -> np.ndarray:
        """
        Automatically crop to check boundaries, removing excess background.
        Conservative approach - only crops if significant borders detected.
        """
        img_h, img_w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to find check boundaries
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if background is dark
        if np.mean(thresh) < 127:
            thresh = cv2.bitwise_not(thresh)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("✂️  No crop needed (no contours found)")
            return image
        
        # Find largest contour (should be the check)
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        image_area = img_w * img_h
        
        # If contour is >90% of image, probably no significant border
        if contour_area > image_area * 0.90:
            print(f"✂️  No crop needed (check fills {contour_area/image_area*100:.1f}% of image)")
            return image
        
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add small padding (2% of dimensions)
        padding_x = int(w * 0.02)
        padding_y = int(h * 0.02)
        
        # Ensure we don't go out of bounds
        x = max(0, x - padding_x)
        y = max(0, y - padding_y)
        w = min(img_w - x, w + 2 * padding_x)
        h = min(img_h - y, h + 2 * padding_y)
        
        # Calculate border sizes
        left_border = x / img_w
        top_border = y / img_h
        right_border = (img_w - (x + w)) / img_w
        bottom_border = (img_h - (y + h)) / img_h
        
        # Only crop if we're removing significant border (>8% on any side)
        # This prevents cropping already well-framed images
        max_border = max(left_border, top_border, right_border, bottom_border)
        
        if max_border > 0.08:
            cropped = image[y:y+h, x:x+w]
            removed_percent = (1 - (w * h) / (img_w * img_h)) * 100
            print(f"✂️  Auto-cropped: {img_w}x{img_h} → {w}x{h} (removed {removed_percent:.1f}% border)")
            return cropped
        else:
            print(f"✂️  No crop needed (max border only {max_border*100:.1f}%)")
        
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
    
    def detect_and_crop_check(self, image: np.ndarray, check_bounds: dict = None) -> np.ndarray:
        """
        Crop image to check boundaries.
        If check_bounds provided (from OCR detection), use those.
        Otherwise, detect check boundaries using contour detection.
        """
        try:
            # If bounds provided by OCR, use them directly
            if check_bounds and all(k in check_bounds for k in ['x', 'y', 'width', 'height']):
                x = check_bounds['x']
                y = check_bounds['y']
                cw = check_bounds['width']
                ch = check_bounds['height']
                h, w = image.shape[:2]
                
                # Validate bounds are within image
                if x >= 0 and y >= 0 and x + cw <= w and y + ch <= h:
                    cropped = image[y:y+ch, x:x+cw]
                    print(f"✂️  Cropped using OCR-detected bounds: {w}x{h} → {cw}x{ch}")
                    return cropped
                else:
                    print("⚠️  Invalid bounds from OCR, falling back to contour detection")
            
            # Fallback: Detect check using contours
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Apply threshold to find check boundaries
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Invert if background is dark
            if np.mean(thresh) < 127:
                thresh = cv2.bitwise_not(thresh)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                print("⚠️  No contours found for check detection")
                return image
            
            # Sort contours by area (largest first)
            contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Try to find a valid check among top 3 largest contours
            for i, contour in enumerate(contours_sorted[:3]):
                contour_area = cv2.contourArea(contour)
                image_area = w * h
                
                # Skip tiny contours
                if contour_area < image_area * 0.15:
                    continue
                
                # Get bounding box
                x, y, cw, ch = cv2.boundingRect(contour)
                bbox_area = cw * ch
                
                # Validate this is actually a check
                aspect_ratio = cw / ch if ch > 0 else 0
                rectangularity = contour_area / bbox_area if bbox_area > 0 else 0
                size_ratio = contour_area / image_area
                longer_side = max(cw, ch)
                
                # Check validation criteria
                is_landscape = 1.5 <= aspect_ratio <= 4.0
                is_portrait = 0.25 <= aspect_ratio <= 0.67
                is_rectangular = rectangularity >= 0.75
                is_big_enough = size_ratio >= 0.20 and longer_side >= 800
                
                if (is_landscape or is_portrait) and is_rectangular and is_big_enough:
                    orientation = "landscape" if is_landscape else "portrait"
                    print(f"✅ Check detected: {cw}x{ch}, {orientation}, {size_ratio*100:.1f}% of image")
                    
                    # If check fills most of image (>85%), no need to crop
                    if contour_area > image_area * 0.85:
                        print(f"✓ Check fills {contour_area/image_area*100:.1f}% of image - no crop needed")
                        return image
                    
                    # Add small padding (1% of dimensions)
                    padding_x = int(cw * 0.01)
                    padding_y = int(ch * 0.01)
                    
                    # Ensure we don't go out of bounds
                    x = max(0, x - padding_x)
                    y = max(0, y - padding_y)
                    cw = min(w - x, cw + 2 * padding_x)
                    ch = min(h - y, ch + 2 * padding_y)
                    
                    # Crop to check
                    cropped = image[y:y+ch, x:x+cw]
                    print(f"✂️  Cropped to check boundaries: {w}x{h} → {cw}x{ch}")
                    
                    return cropped
                else:
                    print(f"⚠️  Contour #{i+1}: aspect={aspect_ratio:.2f}, rect={rectangularity:.2f}, size={size_ratio:.2f}")
            
            # No valid check found - use full image as fallback
            print("⚠️  No valid check detected - using full image")
            return image
            
        except Exception as e:
            print(f"⚠️  Check detection failed: {e}")
            return image
    
    def preprocess(self, check_bounds: dict = None) -> dict:
        """
        Execute complete preprocessing pipeline.
        Returns processed image and preprocessing results.
        
        OPTIMAL WORKFLOW:
        1. Load image
        2. Detect and crop to check boundaries (remove background)
        3. Apply enhancements only if needed
        4. Save processed (cropped + enhanced) image
        """
        results = {
            "success": False,
            "steps_completed": [],
            "original_size": None,
            "processed_size": None,
            "warnings": [],
            "check_detected": False
        }
        
        # Load image
        if not self.load_image():
            results["error"] = "Failed to load image"
            return results
        
        results["original_size"] = self.image.shape[:2]
        results["steps_completed"].append("Image loaded")
        
        # STEP 1: Skip cropping (too slow and often unnecessary)
        # Just use the full image for processing
        processed = self.image.copy()
        results["steps_completed"].append("Using full image (no cropping)")
        
        # STEP 2: Assess image quality on the cropped check
        # Temporarily set processed_image for quality assessment
        temp_processed = self.processed_image
        self.processed_image = processed
        quality_info = self.get_image_quality_score()
        self.processed_image = temp_processed
        results["original_quality"] = quality_info
        
        # STEP 3: Apply enhancements only if needed (on cropped check)
        if quality_info["brightness_score"] < 60 or quality_info["sharpness_score"] < 40:
            # Enhance brightness and contrast (only if needed)
            if quality_info["brightness_score"] < 60:
                try:
                    # Temporarily set image for enhancement
                    temp_img = self.image
                    self.image = processed
                    processed = self.enhance_image()
                    self.image = temp_img
                    results["steps_completed"].append("Enhanced quality (low brightness)")
                except Exception as e:
                    results["warnings"].append(f"Enhancement failed: {str(e)}")
            
            # Sharpen if blurry
            if quality_info["sharpness_score"] < 40:
                try:
                    processed = self.sharpen_image(processed)
                    results["steps_completed"].append("Image sharpened (low sharpness)")
                except Exception as e:
                    results["warnings"].append(f"Sharpening failed: {str(e)}")
        else:
            results["steps_completed"].append("High quality - minimal processing")
        
        # STEP 4: Conservative deskew (only fix obvious skew)
        try:
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
            
            if lines is not None and len(lines) > 5:
                angles = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    # Only consider near-horizontal lines
                    if abs(angle) < 10 or abs(angle - 90) < 10 or abs(angle + 90) < 10:
                        angles.append(angle)
                
                if len(angles) > 3:
                    median_angle = np.median(angles)
                    # Normalize to [-10, 10] range
                    if median_angle > 80:
                        median_angle -= 90
                    elif median_angle < -80:
                        median_angle += 90
                    
                    # Only deskew if angle is between 1-8 degrees
                    if 1 < abs(median_angle) < 8:
                        print(f"🔄 Detected skew: {median_angle:.2f}°, correcting...")
                        processed = self.rotate_image(processed, median_angle)
                        results["steps_completed"].append(f"Corrected {median_angle:.2f}° skew")
                    else:
                        print(f"✓ Skew within acceptable range ({median_angle:.2f}°)")
        except Exception as e:
            results["warnings"].append(f"Skew detection failed: {str(e)}")
        
        self.processed_image = processed
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

