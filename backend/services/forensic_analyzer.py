import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import io
from typing import Dict, List, Tuple, Optional


class ForensicAnalyzer:
    """
    Service for forensic analysis of check images.
    Stage 6: Forensic Analysis (60-75%)
    
    Implements:
    A. Error Level Analysis (ELA) - Detect digitally altered regions
    B. Clone Detection - Identify duplicated regions
    C. Edge Detection - Find irregular boundaries
    """
    
    def __init__(self, image_path: str):
        self.image_path = image_path
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
    
    def error_level_analysis(self) -> Dict:
        """
        Error Level Analysis (ELA) - Detects areas with different compression levels.
        Digitally altered regions will have different compression artifacts.
        """
        analysis = {
            "success": False,
            "suspicious_regions": [],
            "ela_score": 0,
            "max_difference": 0
        }
        
        try:
            # Open image with PIL
            pil_image = Image.open(self.image_path)
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Save image at high quality
            temp_buffer = io.BytesIO()
            pil_image.save(temp_buffer, 'JPEG', quality=95)
            temp_buffer.seek(0)
            
            # Reload the re-compressed image
            compressed_image = Image.open(temp_buffer)
            
            # Calculate difference between original and re-compressed
            ela_image = ImageChops.difference(pil_image, compressed_image)
            
            # Enhance the difference for visibility
            extrema = ela_image.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            
            if max_diff > 0:
                scale = 255.0 / max_diff
                ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
            
            # Convert to numpy array for analysis
            ela_array = np.array(ela_image)
            
            # Convert to grayscale for analysis
            ela_gray = cv2.cvtColor(ela_array, cv2.COLOR_RGB2GRAY)
            
            # Find regions with high ELA values
            threshold = np.percentile(ela_gray, 98)  # Top 2% of differences
            high_ela = ela_gray > threshold
            
            # Find contours of suspicious regions
            contours, _ = cv2.findContours(
                high_ela.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Filter and record significant regions
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate severity based on area and ELA intensity
                    region_ela = ela_gray[y:y+h, x:x+w]
                    avg_ela = np.mean(region_ela)
                    
                    severity = "LOW"
                    if avg_ela > threshold * 1.5:
                        severity = "HIGH"
                    elif avg_ela > threshold * 1.2:
                        severity = "MEDIUM"
                    
                    analysis["suspicious_regions"].append({
                        "coordinates": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                        "area": int(area),
                        "severity": severity,
                        "ela_intensity": float(avg_ela),
                        "reason": "Inconsistent compression level detected"
                    })
            
            analysis["ela_score"] = float(np.mean(ela_gray))
            analysis["max_difference"] = float(max_diff)
            analysis["success"] = True
            
        except Exception as e:
            analysis["error"] = f"ELA failed: {str(e)}"
        
        return analysis
    
    def clone_detection(self) -> Dict:
        """
        Clone Detection - Identifies duplicated regions within the image.
        Useful for detecting copied signatures or amounts.
        """
        analysis = {
            "success": False,
            "duplicates_found": False,
            "cloned_regions": [],
            "similarity_score": 0
        }
        
        try:
            if self.image is None:
                self.load_image()
            
            # Use SIFT (Scale-Invariant Feature Transform) for feature detection
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(self.gray_image, None)
            
            if descriptors is None or len(descriptors) < 2:
                analysis["success"] = True
                return analysis
            
            # Use FLANN matcher to find similar features
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(descriptors, descriptors, k=2)
            
            # Find good matches using Lowe's ratio test
            clone_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    # Exclude self-matches and apply ratio test
                    if m.trainIdx != m.queryIdx and m.distance < 0.6 * n.distance:
                        # Check if keypoints are spatially separated (not just nearby)
                        kp1 = keypoints[m.queryIdx].pt
                        kp2 = keypoints[m.trainIdx].pt
                        distance = np.sqrt((kp1[0] - kp2[0])**2 + (kp1[1] - kp2[1])**2)
                        
                        if distance > 50:  # Minimum distance between clones
                            clone_matches.append((m, kp1, kp2, distance))
            
            # Group matches into regions
            if len(clone_matches) > 10:  # Significant number of matches
                analysis["duplicates_found"] = True
                analysis["similarity_score"] = len(clone_matches) / len(keypoints) * 100
                
                # Cluster matches to identify cloned regions
                match_points = np.array([list(m[1]) + list(m[2]) for m in clone_matches])
                
                # Simple clustering: group nearby matches
                regions = []
                for match in clone_matches[:20]:  # Limit to top matches
                    _, kp1, kp2, dist = match
                    regions.append({
                        "source": {"x": int(kp1[0]), "y": int(kp1[1])},
                        "duplicate": {"x": int(kp2[0]), "y": int(kp2[1])},
                        "distance": float(dist),
                        "confidence": float(1.0 - (_.distance / 100))
                    })
                
                analysis["cloned_regions"] = regions
            
            analysis["success"] = True
            
        except Exception as e:
            analysis["error"] = f"Clone detection failed: {str(e)}"
        
        return analysis
    
    def edge_detection_analysis(self) -> Dict:
        """
        Edge Detection and Boundary Analysis - LENIENT version for photos.
        Only flags EXTREME anomalies that clearly indicate digital manipulation.
        
        NOTE: Photos naturally have irregular edges from shadows, lighting, 
        background objects - this is NORMAL and not suspicious.
        """
        analysis = {
            "success": False,
            "irregular_edges": [],
            "edge_density_score": 0,
            "suspicious_boundaries": []
        }
        
        try:
            if self.image is None:
                self.load_image()
            
            # Apply Canny edge detection
            edges = cv2.Canny(self.gray_image, 50, 150)
            
            # Calculate edge density
            edge_density = np.sum(edges > 0) / edges.size * 100
            analysis["edge_density_score"] = float(edge_density)
            
            # VERY LENIENT: Only flag EXTREME edge density deviations
            # Photos naturally have varied edge patterns (shadows, background, etc.)
            h, w = edges.shape
            grid_size = 100
            extreme_deviations = []
            
            for i in range(0, h - grid_size, grid_size):
                for j in range(0, w - grid_size, grid_size):
                    region = edges[i:i+grid_size, j:j+grid_size]
                    region_density = np.sum(region > 0) / region.size * 100
                    
                    # Only flag EXTREME outliers (5x deviation, not 2x)
                    # This catches obvious tampering but ignores normal photo variations
                    if region_density > edge_density * 5 and region_density > 20:
                        extreme_deviations.append({
                            "coordinates": {
                                "x": int(j),
                                "y": int(i),
                                "width": int(grid_size),
                                "height": int(grid_size)
                            },
                            "edge_density": float(region_density),
                            "deviation": float(abs(region_density - edge_density)),
                            "reason": "Extreme edge density anomaly (possible digital overlay)"
                        })
            
            # Only report if we have MANY extreme deviations (likely actual tampering)
            if len(extreme_deviations) > 5:
                analysis["irregular_edges"] = extreme_deviations[:5]  # Limit to top 5
            
            # REMOVED: Suspicious straight line detection
            # Photos of checks naturally have straight edges (check borders, table edges, etc.)
            # This was causing too many false positives
            
            analysis["success"] = True
            
        except Exception as e:
            analysis["error"] = f"Edge detection failed: {str(e)}"
        
        return analysis
    
    def color_texture_analysis(self) -> Dict:
        """
        Color and Texture Analysis - Analyzes ink consistency and paper texture.
        """
        analysis = {
            "success": False,
            "color_consistency_score": 0,
            "texture_score": 0,
            "anomalies": []
        }
        
        try:
            if self.image is None:
                self.load_image()
            
            # Analyze color distribution
            hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            
            # Calculate color variance
            color_std = np.std(hsv_image, axis=(0, 1))
            color_consistency = 100 - (np.mean(color_std) / 255 * 100)
            analysis["color_consistency_score"] = float(color_consistency)
            
            # Analyze texture using Local Binary Patterns approach
            # Calculate texture variance
            texture_features = cv2.Laplacian(self.gray_image, cv2.CV_64F)
            texture_variance = np.var(texture_features)
            analysis["texture_score"] = float(min(100, texture_variance / 10))
            
            # Detect regions with inconsistent texture
            h, w = self.gray_image.shape
            grid_size = 100
            texture_values = []
            
            for i in range(0, h - grid_size, grid_size):
                for j in range(0, w - grid_size, grid_size):
                    region = self.gray_image[i:i+grid_size, j:j+grid_size]
                    region_texture = cv2.Laplacian(region, cv2.CV_64F)
                    texture_values.append((i, j, np.var(region_texture)))
            
            if texture_values:
                mean_texture = np.mean([t[2] for t in texture_values])
                std_texture = np.std([t[2] for t in texture_values])
                
                for i, j, texture_val in texture_values:
                    if abs(texture_val - mean_texture) > 2 * std_texture:
                        analysis["anomalies"].append({
                            "coordinates": {
                                "x": int(j),
                                "y": int(i),
                                "width": int(grid_size),
                                "height": int(grid_size)
                            },
                            "type": "texture_anomaly",
                            "deviation": float(abs(texture_val - mean_texture))
                        })
            
            analysis["success"] = True
            
        except Exception as e:
            analysis["error"] = f"Color/texture analysis failed: {str(e)}"
        
        return analysis
    
    def analyze(self) -> Dict:
        """
        Execute complete forensic analysis pipeline.
        """
        if not self.load_image():
            return {
                "success": False,
                "error": "Failed to load image for forensic analysis"
            }
        
        results = {
            "success": True,
            "errorLevelAnalysis": self.error_level_analysis(),
            "cloneDetection": self.clone_detection(),
            "edgeAnalysis": self.edge_detection_analysis(),
            "colorTextureAnalysis": self.color_texture_analysis(),
            "overall_risk_score": 0,
            "summary": []
        }
        
        # Calculate overall risk score
        risk_score = 0
        
        # ELA findings
        ela_regions = len(results["errorLevelAnalysis"].get("suspicious_regions", []))
        if ela_regions > 0:
            risk_score += min(30, ela_regions * 10)
            results["summary"].append(f"{ela_regions} suspicious region(s) found in ELA")
        
        # Clone detection findings
        if results["cloneDetection"].get("duplicates_found", False):
            risk_score += 25
            results["summary"].append("Cloned regions detected")
        
        # Edge analysis findings (LENIENT - only extreme anomalies)
        irregular_edges = len(results["edgeAnalysis"].get("irregular_edges", []))
        if irregular_edges > 0:
            # Only add significant risk if multiple EXTREME anomalies found
            risk_score += min(15, irregular_edges * 3)
            results["summary"].append(f"{irregular_edges} extreme edge anomaly(ies) detected")
        
        # Texture anomalies
        texture_anomalies = len(results["colorTextureAnalysis"].get("anomalies", []))
        if texture_anomalies > 5:
            risk_score += 10
            results["summary"].append(f"{texture_anomalies} texture anomaly(ies) found")
        
        results["overall_risk_score"] = min(100, risk_score)
        
        if not results["summary"]:
            results["summary"].append("No significant forensic anomalies detected")
        
        return results

