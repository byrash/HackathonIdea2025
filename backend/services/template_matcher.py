import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class TemplateMatcher:
    """
    Service for template matching and security feature detection.
    Stage 7: Security Features Check (75-90%)
    
    Detects:
    - Watermarks
    - Template matching against known legitimate checks
    - Security pattern verification
    - Microprinting analysis
    """
    
    def __init__(self, image_path: str, templates_dir: str = "./templates"):
        self.image_path = image_path
        self.templates_dir = Path(templates_dir)
        self.image = None
        self.gray_image = None
    
    def load_image(self) -> bool:
        """Load image for analysis."""
        try:
            self.image = cv2.imread(self.image_path)
            if self.image is None:
                return False
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            return True
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            return False
    
    def detect_watermark(self) -> Dict:
        """
        Detect watermarks in the check image.
        Watermarks typically appear as faint patterns or text.
        """
        analysis = {
            "detected": False,
            "confidence": 0,
            "watermark_regions": [],
            "characteristics": []
        }
        
        try:
            if self.image is None:
                self.load_image()
            
            # Convert to frequency domain to detect watermarks
            # Watermarks often show up better in frequency domain
            dft = cv2.dft(np.float32(self.gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            
            # Calculate magnitude spectrum
            magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
            magnitude = np.log(magnitude + 1)
            
            # Normalize
            magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            
            # Detect repeating patterns (watermarks)
            # High-frequency patterns indicate watermarks
            high_freq_threshold = np.percentile(magnitude, 95)
            watermark_mask = magnitude > high_freq_threshold
            
            # Count watermark pixels
            watermark_pixels = np.sum(watermark_mask)
            total_pixels = watermark_mask.size
            watermark_ratio = watermark_pixels / total_pixels
            
            if watermark_ratio > 0.01:  # More than 1% of image
                analysis["detected"] = True
                analysis["confidence"] = min(100, watermark_ratio * 1000)
                analysis["characteristics"].append("High-frequency pattern detected")
            
            # Also check for low-contrast patterns (typical of watermarks)
            # Apply histogram equalization to enhance faint patterns
            equalized = cv2.equalizeHist(self.gray_image)
            diff = cv2.absdiff(equalized, self.gray_image)
            
            # Threshold to find faint patterns
            _, faint_patterns = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            
            contours, _ = cv2.findContours(faint_patterns, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for large, regular patterns
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Significant area
                    x, y, w, h = cv2.boundingRect(contour)
                    analysis["watermark_regions"].append({
                        "coordinates": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                        "area": int(area)
                    })
            
            if len(analysis["watermark_regions"]) > 0:
                analysis["detected"] = True
                analysis["confidence"] = max(analysis["confidence"], 70)
                analysis["characteristics"].append("Faint patterns detected")
            
        except Exception as e:
            analysis["error"] = f"Watermark detection failed: {str(e)}"
        
        return analysis
    
    def match_templates(self) -> Dict:
        """
        Match check image against known legitimate check templates.
        """
        analysis = {
            "matched": False,
            "best_match": None,
            "confidence": 0,
            "matches": []
        }
        
        try:
            if self.image is None:
                self.load_image()
            
            # Get list of template files
            if not self.templates_dir.exists():
                analysis["warning"] = "Templates directory not found"
                return analysis
            
            template_files = list(self.templates_dir.glob("*.jpg")) + \
                            list(self.templates_dir.glob("*.png"))
            
            if not template_files:
                analysis["warning"] = "No template files found"
                return analysis
            
            # Resize check image for consistent comparison
            check_resized = cv2.resize(self.gray_image, (800, 400))
            
            best_score = 0
            best_template = None
            
            for template_path in template_files:
                try:
                    # Load template
                    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
                    if template is None:
                        continue
                    
                    # Resize template to match check size
                    template_resized = cv2.resize(template, (800, 400))
                    
                    # Calculate structural similarity
                    # Use normalized cross-correlation
                    result = cv2.matchTemplate(check_resized, template_resized, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)
                    
                    # Also use feature matching for better accuracy
                    sift = cv2.SIFT_create()
                    kp1, des1 = sift.detectAndCompute(check_resized, None)
                    kp2, des2 = sift.detectAndCompute(template_resized, None)
                    
                    if des1 is not None and des2 is not None:
                        # FLANN matcher
                        FLANN_INDEX_KDTREE = 1
                        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                        search_params = dict(checks=50)
                        flann = cv2.FlannBasedMatcher(index_params, search_params)
                        
                        matches = flann.knnMatch(des1, des2, k=2)
                        
                        # Apply ratio test
                        good_matches = []
                        for match_pair in matches:
                            if len(match_pair) == 2:
                                m, n = match_pair
                                if m.distance < 0.7 * n.distance:
                                    good_matches.append(m)
                        
                        # Calculate match score
                        match_ratio = len(good_matches) / max(len(kp1), len(kp2))
                        combined_score = (max_val * 0.5 + match_ratio * 0.5) * 100
                    else:
                        combined_score = max_val * 100
                    
                    template_name = template_path.stem
                    analysis["matches"].append({
                        "template": template_name,
                        "confidence": float(combined_score)
                    })
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_template = template_name
                
                except Exception as e:
                    print(f"Error matching template {template_path}: {str(e)}")
                    continue
            
            if best_score > 60:  # Threshold for match
                analysis["matched"] = True
                analysis["best_match"] = best_template
                analysis["confidence"] = float(best_score)
            
        except Exception as e:
            analysis["error"] = f"Template matching failed: {str(e)}"
        
        return analysis
    
    def detect_security_patterns(self) -> Dict:
        """
        Detect security patterns like background patterns, security threads, etc.
        """
        analysis = {
            "patterns_detected": False,
            "pattern_types": [],
            "confidence": 0
        }
        
        try:
            if self.image is None:
                self.load_image()
            
            # Detect repetitive background patterns
            # Use Fourier Transform to detect periodic patterns
            f_transform = np.fft.fft2(self.gray_image)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            
            # Look for peaks in frequency domain (indicates repeating patterns)
            magnitude_log = np.log(magnitude + 1)
            peaks = magnitude_log > np.percentile(magnitude_log, 99)
            
            peak_count = np.sum(peaks)
            if peak_count > 10:
                analysis["patterns_detected"] = True
                analysis["pattern_types"].append("repetitive_background")
                analysis["confidence"] += 30
            
            # Detect fine lines (security threads)
            # Use Canny edge detection with tight parameters
            edges = cv2.Canny(self.gray_image, 100, 200)
            
            # Use Hough Line Transform to detect straight lines
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180,
                threshold=100,
                minLineLength=100,
                maxLineGap=5
            )
            
            if lines is not None:
                # Count parallel lines (typical of security patterns)
                parallel_count = 0
                for i, line1 in enumerate(lines):
                    x1, y1, x2, y2 = line1[0]
                    angle1 = np.arctan2(y2 - y1, x2 - x1)
                    
                    for line2 in lines[i+1:]:
                        x3, y3, x4, y4 = line2[0]
                        angle2 = np.arctan2(y4 - y3, x4 - x3)
                        
                        # Check if lines are parallel
                        if abs(angle1 - angle2) < 0.1:
                            parallel_count += 1
                
                if parallel_count > 5:
                    analysis["patterns_detected"] = True
                    analysis["pattern_types"].append("parallel_lines")
                    analysis["confidence"] += 25
            
            # Detect microprinting (very small text)
            # Look for high-frequency edges in small regions
            sobelx = cv2.Sobel(self.gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(self.gray_image, cv2.CV_64F, 0, 1, ksize=3)
            edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            # High edge density in small areas indicates microprinting
            kernel_size = 20
            h, w = edge_magnitude.shape
            microprint_regions = 0
            
            for i in range(0, h - kernel_size, kernel_size):
                for j in range(0, w - kernel_size, kernel_size):
                    region = edge_magnitude[i:i+kernel_size, j:j+kernel_size]
                    if np.mean(region) > np.mean(edge_magnitude) * 1.5:
                        microprint_regions += 1
            
            if microprint_regions > 5:
                analysis["patterns_detected"] = True
                analysis["pattern_types"].append("microprinting")
                analysis["confidence"] += 20
            
            analysis["confidence"] = min(100, analysis["confidence"])
            
        except Exception as e:
            analysis["error"] = f"Security pattern detection failed: {str(e)}"
        
        return analysis
    
    def analyze(self) -> Dict:
        """
        Execute complete security features analysis.
        """
        if not self.load_image():
            return {
                "success": False,
                "error": "Failed to load image for template matching"
            }
        
        results = {
            "success": True,
            "watermarkDetected": False,
            "templateMatch": {},
            "securityPatterns": {},
            "overall_confidence": 0,
            "warnings": []
        }
        
        # Detect watermark
        watermark_result = self.detect_watermark()
        results["watermarkDetected"] = watermark_result.get("detected", False)
        results["watermark_details"] = watermark_result
        
        # Match templates
        template_result = self.match_templates()
        results["templateMatch"] = template_result
        
        # Detect security patterns
        security_result = self.detect_security_patterns()
        results["securityPatterns"] = security_result
        
        # Calculate overall confidence
        confidence_scores = []
        
        if watermark_result.get("detected"):
            confidence_scores.append(watermark_result.get("confidence", 0))
        
        if template_result.get("matched"):
            confidence_scores.append(template_result.get("confidence", 0))
        
        if security_result.get("patterns_detected"):
            confidence_scores.append(security_result.get("confidence", 0))
        
        if confidence_scores:
            results["overall_confidence"] = sum(confidence_scores) / len(confidence_scores)
        else:
            results["overall_confidence"] = 0
            results["warnings"].append("No security features detected - suspicious")
        
        return results

