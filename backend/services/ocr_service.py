import cv2
import numpy as np
import pytesseract
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class OCRService:
    """
    Service for Optical Character Recognition (OCR) on check images.
    Stage 4: OCR & Text Extraction (30-50%)
    Stage 5: Cross-Field Validation (50-60%)
    
    Extracts:
    - Payee name
    - Amount (numeric and written)
    - Date
    - Check number
    - Routing number
    - Account number
    - MICR line
    """
    
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = None
        self.gray_image = None
        self.extracted_text = ""
        self.check_bounds = None  # Store detected check bounds
    
    def load_image(self) -> bool:
        """Load image for OCR processing."""
        try:
            self.image = cv2.imread(self.image_path)
            if self.image is None:
                return False
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            return True
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            return False
    
    def preprocess_for_ocr(self) -> np.ndarray:
        """Preprocess image for better OCR accuracy."""
        if self.gray_image is None:
            self.load_image()
        
        # Apply thresholding
        _, thresh = cv2.threshold(
            self.gray_image,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, h=30)
        
        return denoised
    
    def extract_text(self) -> str:
        """
        Extract all text from check image using Tesseract OCR.
        """
        try:
            preprocessed = self.preprocess_for_ocr()
            
            # Configure Tesseract
            custom_config = r'--oem 3 --psm 6'
            self.extracted_text = pytesseract.image_to_string(
                preprocessed,
                config=custom_config
            )
            
            # Removed verbose logging for production
            
            return self.extracted_text
            
        except Exception as e:
            print(f"❌ Error extracting text: {str(e)}")
            return ""
    
    def extract_check_number(self) -> Optional[str]:
        """Extract check number (usually 4-6 digits in top right)."""
        if not self.extracted_text:
            self.extract_text()
        
        # Pattern for check number (typically 4-6 digits)
        # Prioritize patterns with "CHECK" or "#" label
        patterns = [
            r'CHECK\s*#:?\s*(\d{4,6})',  # Check #: 804135
            r'CHECK\s+NUMBER\s*:?\s*(\d{4,6})',  # Check Number: 804135
            r'#\s*:?\s*(\d{4,6})',  # #: 804135 or # 804135
            r'NO\.?\s*:?\s*(\d{4,6})',  # No. 804135 or No: 804135
        ]
        
        candidates = []
        for pattern in patterns:
            match = re.search(pattern, self.extracted_text, re.IGNORECASE | re.MULTILINE)
            if match:
                check_num = match.group(1)
                candidates.append((check_num, 'pattern'))
        
        # Fallback: Try to find 4-6 digit number in lines that mention "Check"
        for line in self.extracted_text.split('\n'):
            if 'check' in line.lower() or '#' in line:
                match = re.search(r'\b(\d{4,6})\b', line)
                if match:
                    check_num = match.group(1)
                    # Exclude if it looks like a zip code (27XXX or 275XX)
                    if not check_num.startswith('27'):
                        candidates.append((check_num, 'line'))
        
        # Return first candidate (most reliable)
        if candidates:
            final_check_num = candidates[0][0]
            print(f"✅ Check number found: {final_check_num}")
            return final_check_num
        
            print("⚠️  Check number not found")
        return None
    
    def extract_date(self) -> Optional[str]:
        """Extract date from check."""
        if not self.extracted_text:
            self.extract_text()
        
        # Common date patterns
        patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # MM/DD/YYYY or MM-DD-YYYY
            r'(\d{1,2}/\d{1,2}/\d{2,4})',
            r'DATE\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, self.extracted_text, re.IGNORECASE)
            if match:
                date = match.group(1)
                print(f"✅ Date found: {date}")
                return date
        
        print("⚠️  Date not found")
        return None
    
    def extract_payee(self) -> Optional[str]:
        """Extract payee name from check."""
        if not self.extracted_text:
            self.extract_text()
        
        # Extract payee from check
        
        # Look for "Pay to the order of" pattern
        patterns = [
            r'PAY\s+TO\s+THE\s+ORDER\s+OF\s*:?\s*([A-Z][A-Za-z\s\.]+)',
            r'PAY\s+TO\s*:?\s*([A-Z][A-Za-z\s\.]+)',
            r'PAYEE\s*:?\s*([A-Z][A-Za-z\s\.]+)',
            r'TO\s+THE\s+(\w+)',  # Fallback: "TO THE BYRAPANENT"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, self.extracted_text, re.IGNORECASE)
            if match:
                payee = match.group(1).strip()
                # Clean up payee name
                payee = re.sub(r'\s+', ' ', payee)
                if len(payee) > 3 and len(payee) < 100:
                    print(f"✅ Payee found: {payee}")
                    return payee
        
        # Try to extract from first line (often contains payee)
        lines = self.extracted_text.split('\n')
        if lines and len(lines[0]) > 3:
            first_line = lines[0].strip()
            if len(first_line) < 100:
                print(f"✅ Payee found (first line): {first_line}")
                return first_line
        
        print("⚠️  Payee not found")
        return None
    
    def extract_amount_numeric(self) -> Optional[float]:
        """Extract numeric amount from check with enhanced OCR on amount box."""
        if not self.extracted_text:
            self.extract_text()
        
        # Try to extract amount from specific region (top-right corner amount box)
        amount_from_region = self._extract_amount_from_region()
        if amount_from_region:
            return amount_from_region
        
        # Fallback to full text extraction
        # Pattern for amount (dollar sign followed by numbers)
        patterns = [
            r'\$\s*(\d{1,3}(?:,\d{3})*\.\d{2})',  # $1,234.56 (with cents)
            r'\$\s*(\d+\.\d{2})',  # $123.45
            r'\$\s*(\d{1,3}(?:,\d{3})*)',  # $1,234 (no cents)
            r'AMOUNT\s*:?\s*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            # Handle OCR errors: o"" -> $, _ -> .
            r'[o\*\"\'][\"\'\*]*\s*(\d+)\s*[_\-\.]\s*(\d{2})',  # o""37_ 86 -> 37.86
            r'(\d+)\s*[_\-\.]\s*(\d{2})',  # 37_ 86 or 37. 86
        ]
        
        amounts = []
        for pattern in patterns:
            matches = re.finditer(pattern, self.extracted_text, re.IGNORECASE)
            for match in matches:
                try:
                    # Handle two-group matches (number and cents separately)
                    if len(match.groups()) == 2 and match.group(2):
                        amount_str = f"{match.group(1)}.{match.group(2)}"
                    else:
                        amount_str = match.group(1).replace(',', '')
                    
                    amount = float(amount_str)
                    if 0.01 < amount < 1000000:  # Reasonable check amount
                        amounts.append(amount)
                except (ValueError, AttributeError):
                    continue
        
        # Return the most likely amount
        # Strategy: Use the most common amount (appears multiple times) or smallest if tie
        if amounts:
            from collections import Counter
            amount_counts = Counter(amounts)
            
            # If one amount appears more than others, use it
            most_common = amount_counts.most_common(2)
            if len(most_common) > 1 and most_common[0][1] > most_common[1][1]:
                final_amount = most_common[0][0]
                print(f"✅ Amount found (full-text, most common): ${final_amount:.2f}")
            else:
                # Otherwise use the smallest (safer for fraud detection)
                final_amount = min(amounts)
                print(f"✅ Amount found (full-text, smallest): ${final_amount:.2f}")
            
            return final_amount
        
        print("⚠️  Amount not found")
        return None
    
    def _extract_amount_from_region(self) -> Optional[float]:
        """Extract amount from the typical amount box region (top-right)."""
        try:
            if self.image is None:
                self.load_image()
            
            h, w = self.image.shape[:2]
            
            # Amount box is typically in top-right corner
            # Try region: right 30%, top 25%
            x_start = int(w * 0.7)
            y_start = int(h * 0.05)
            y_end = int(h * 0.30)
            
            amount_region = self.gray_image[y_start:y_end, x_start:]
            
            # Preprocess amount region
            _, thresh = cv2.threshold(amount_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Try multiple OCR configs for amount
            configs = [
                r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.$,*',  # Include asterisks
                r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.$,',
                r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.$,*',
            ]
            
            all_amounts = []
            for config in configs:
                try:
                    amount_text = pytesseract.image_to_string(thresh, config=config).strip()
                    # OCR amount region
                    
                    # Remove asterisks and extract amount
                    cleaned = amount_text.replace('*', '').replace('$', '').strip()
                    
                    # Find last occurrence of digits with decimal (usually the actual amount)
                    matches = re.findall(r'(\d{1,3}(?:,?\d{3})*\.?\d{0,2})', cleaned)
                    for match in matches:
                        amount_str = match.replace(',', '')
                        amount = float(amount_str)
                    if 0.01 < amount < 10000:  # Reasonable check amount
                        all_amounts.append(amount)
                except Exception:
                    continue
            
            if all_amounts:
                # Return smallest reasonable amount (asterisks often make OCR read larger numbers)
                final_amount = min(all_amounts)
                print(f"✅ Amount extracted from region: ${final_amount:.2f}")
                return final_amount
            
            print("⚠️  Amount region extraction: no valid amount found")
        except Exception as e:
            print(f"❌ Amount region extraction failed: {e}")
        
        return None
    
    def extract_amount_written(self) -> Optional[str]:
        """Extract written amount from check."""
        if not self.extracted_text:
            self.extract_text()
        
        # Look for written amount pattern
        # Usually contains words like "dollars", "hundred", "thousand"
        amount_keywords = ['hundred', 'thousand', 'dollars', 'cents']
        
        lines = self.extracted_text.split('\n')
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in amount_keywords):
                # Clean up the line
                cleaned = re.sub(r'[^\w\s/-]', '', line).strip()
                if len(cleaned) > 5 and len(cleaned) < 200:
                    return cleaned
        
        return None
    
    def extract_micr_line(self) -> Dict:
        """
        Extract MICR (Magnetic Ink Character Recognition) line.
        Format: ⑆routing⑆ account⑆ check_number
        
        MICR uses special E-13B font at the bottom of checks.
        """
        result = {
            "full_micr": None,
            "routing_number": None,
            "account_number": None,
            "check_number_micr": None
        }
        
        # Try region-based extraction first (bottom 15% of check)
        micr_from_region = self._extract_micr_from_region()
        if micr_from_region["routing_number"]:
            return micr_from_region
        
        # Fallback to full text extraction
        if not self.extracted_text:
            self.extract_text()
        
        # MICR typically contains routing number (9 digits), account number, and check number
        # Look for pattern with multiple digit groups
        micr_pattern = r'[⑆⑈]?\s*(\d{9})\s*[⑆⑈]?\s*(\d{4,17})\s*[⑆⑈]?\s*(\d{3,4})'
        match = re.search(micr_pattern, self.extracted_text)
        
        if match:
            result["routing_number"] = match.group(1)
            result["account_number"] = match.group(2)
            result["check_number_micr"] = match.group(3)
            result["full_micr"] = match.group(0)
        else:
            # Try to find routing number separately (9 digits)
            routing_match = re.search(r'\b(\d{9})\b', self.extracted_text)
            if routing_match:
                result["routing_number"] = routing_match.group(1)
        
        return result
    
    def detect_check_region(self) -> Optional[Dict]:
        """
        Detect check region in the ORIGINAL image using contour detection.
        Returns bounding box coordinates if valid check found.
        Should be called BEFORE preprocessing.
        """
        try:
            if self.gray_image is None:
                self.load_image()
            
            gray = self.gray_image.copy()
            h, w = gray.shape
            
            # Apply threshold to find check boundaries
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Invert if background is dark
            if np.mean(thresh) < 127:
                thresh = cv2.bitwise_not(thresh)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                print("⚠️  OCR check detection: No contours found")
                return None
            
            # Sort contours by area (largest first)
            contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Try to find a valid check among top 3 largest contours
            for i, contour in enumerate(contours_sorted[:3]):
                contour_area = cv2.contourArea(contour)
                image_area = w * h
                
                # Skip tiny contours
                if contour_area < image_area * 0.15:
                    continue
                
                # Validate this is actually a check
                is_valid, reason = self._validate_check_contour(contour, w, h)
                
                if is_valid:
                    x, y, cw, ch = cv2.boundingRect(contour)  # noqa: F841
                    
                    # Add small padding (1%)
                    padding_x = int(cw * 0.01)
                    padding_y = int(ch * 0.01)
                    
                    x = max(0, x - padding_x)
                    y = max(0, y - padding_y)
                    cw = min(w - x, cw + 2 * padding_x)
                    ch = min(h - y, ch + 2 * padding_y)
                    
                    self.check_bounds = {
                        'x': x,
                        'y': y,
                        'width': cw,
                        'height': ch
                    }
                    
                    print(f"✅ OCR detected check region: ({x}, {y}) {cw}x{ch}")
                    print(f"   {reason}")
                    return self.check_bounds
                else:
                    print(f"⚠️  Contour #{i+1} rejected: {reason}")
            
            print("⚠️  OCR: No valid check region detected")
            return None
            
        except Exception as e:
            print(f"⚠️  OCR check detection failed: {e}")
            return None
    
    def _validate_check_contour(self, contour, image_width: int, image_height: int) -> Tuple[bool, str]:
        """
        Validate that a contour is actually a check, not some other object.
        
        Returns: (is_valid, reason)
        """
        x, y, w, h = cv2.boundingRect(contour)
        contour_area = cv2.contourArea(contour)
        bbox_area = w * h
        image_area = image_width * image_height
        
        # 1. Check aspect ratio (checks are landscape 2:1-3:1, but could be rotated to portrait)
        aspect_ratio = w / h if h > 0 else 0
        # Accept both landscape (1.5-4.0) and portrait (0.25-0.67 which is 1/4 to 1/1.5)
        is_landscape = 1.5 <= aspect_ratio <= 4.0
        is_portrait = 0.25 <= aspect_ratio <= 0.67
        
        if not (is_landscape or is_portrait):
            return False, f"Invalid aspect ratio {aspect_ratio:.2f} (need 1.5-4.0 or 0.25-0.67)"
        
        # 2. Check must be substantial size (at least 20% of image area)
        size_ratio = contour_area / image_area
        if size_ratio < 0.20:
            return False, f"Too small {size_ratio*100:.1f}% (need >20% of image)"
        
        # 3. Check must be reasonably rectangular (contour fills bounding box)
        rectangularity = contour_area / bbox_area if bbox_area > 0 else 0
        if rectangularity < 0.75:
            return False, f"Not rectangular enough {rectangularity*100:.1f}% (need >75%)"
        
        # 4. Check minimum absolute dimensions (at least 800px on longer side)
        longer_side = max(w, h)
        if longer_side < 800:
            return False, f"Too small {longer_side}px (need >800px on longer side)"
        
        orientation = "landscape" if is_landscape else "portrait (may need rotation)"
        return True, f"Valid check ({orientation}): {w}x{h}, ratio {aspect_ratio:.2f}, size {size_ratio*100:.1f}%"
    
    def _detect_and_crop_check(self) -> Optional[np.ndarray]:
        """
        Detect check boundaries and crop to just the check.
        VALIDATES that detected object is actually a check before cropping.
        This improves MICR extraction by removing background.
        """
        try:
            gray = self.gray_image.copy()
            h, w = gray.shape
            
            # Apply threshold to find check boundaries
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Invert if background is dark
            if np.mean(thresh) < 127:
                thresh = cv2.bitwise_not(thresh)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                print("⚠️  Check crop: No contours found, using full image")
                return self.image
            
            # Sort contours by area (largest first)
            contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Try to find a valid check among top 3 largest contours
            valid_check_found = False
            for i, contour in enumerate(contours_sorted[:3]):
                contour_area = cv2.contourArea(contour)
                image_area = w * h
                
                # Skip tiny contours
                if contour_area < image_area * 0.15:
                    continue
                
                # Validate this is actually a check
                is_valid, reason = self._validate_check_contour(contour, w, h)
                
                if is_valid:
                    print(f"✅ Check detected (contour #{i+1}): {reason}")
                    valid_check_found = True
                    
                    # If check fills most of image (>85%), no need to crop
                    if contour_area > image_area * 0.85:
                        print(f"✓ Check fills {contour_area/image_area*100:.1f}% of image - no crop needed")
                        return self.image
                    
                    x, y, cw, ch = cv2.boundingRect(contour)
                    
                    # Add small padding (1% of dimensions)
                    padding_x = int(cw * 0.01)
                    padding_y = int(ch * 0.01)
                    
                    # Ensure we don't go out of bounds
                    x = max(0, x - padding_x)
                    y = max(0, y - padding_y)
                    cw = min(w - x, cw + 2 * padding_x)
                    ch = min(h - y, ch + 2 * padding_y)
                    
                    # Crop to check
                    cropped = self.image[y:y+ch, x:x+cw]
                    print(f"✂️  Cropped to check: {w}x{h} → {cw}x{ch}")
                    
                    return cropped
                else:
                    print(f"⚠️  Contour #{i+1} rejected: {reason}")
            
            # No valid check found - use full image as fallback
            if not valid_check_found:
                print("⚠️  No valid check contour detected - using full image")
                print("    This could mean: check fills frame, poor lighting, or no check in photo")
            
            return self.image
            
        except Exception as e:
            print(f"⚠️  Check crop failed: {e}, using full image")
            return self.image
    
    def _extract_micr_from_region(self) -> Dict:
        """
        Extract MICR line from bottom region of check with enhanced preprocessing.
        Assumes image has already been preprocessed (cropped to check boundaries).
        Saves intermediate images for debugging.
        """
        result = {
            "full_micr": None,
            "routing_number": None,
            "account_number": None,
            "check_number_micr": None
        }
        
        try:
            if self.image is None:
                self.load_image()
            
            h, w = self.image.shape[:2]
            print(f"📏 Image dimensions for MICR extraction: {w}x{h}")
            
            # Extract MICR from bottom of check (bottom 8-12%)
            y_start = int(h * 0.88)
            micr_gray = self.gray_image[y_start:, :]
            
            # Enhanced preprocessing approaches for MICR
            preprocessed_images = []
            
            # 1. Standard Otsu threshold
            _, thresh1 = cv2.threshold(micr_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            preprocessed_images.append(('otsu', thresh1))
            
            # 2. Inverted Otsu (MICR is magnetic ink, often appears different)
            _, thresh2 = cv2.threshold(micr_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            preprocessed_images.append(('otsu_inv', thresh2))
            
            # 3. Adaptive threshold (handles varying lighting)
            thresh3 = cv2.adaptiveThreshold(
                micr_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            preprocessed_images.append(('adaptive', thresh3))
            
            # 4. High contrast with morphology
            _, thresh4 = cv2.threshold(micr_gray, 127, 255, cv2.THRESH_BINARY)
            kernel = np.ones((2, 2), np.uint8)
            thresh4 = cv2.morphologyEx(thresh4, cv2.MORPH_CLOSE, kernel)
            preprocessed_images.append(('morph', thresh4))
            
            # 5. Denoise + threshold
            denoised = cv2.fastNlMeansDenoising(micr_gray, h=10)
            _, thresh5 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            preprocessed_images.append(('denoise', thresh5))
            
            # 6. Resize 2x for better OCR (MICR characters are small)
            micr_2x = cv2.resize(micr_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            _, thresh6 = cv2.threshold(micr_2x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            preprocessed_images.append(('2x_scale', thresh6))
            
            # Try multiple OCR configurations
            configs = [
                r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789',  # Uniform block
                r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789',  # Single line
                r'--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789', # Raw line
                r'--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789',  # LSTM only
            ]
            
            all_texts = []
            
            for method_name, thresh in preprocessed_images:
                for config in configs:
                    try:
                        text = pytesseract.image_to_string(thresh, config=config).strip()
                        if text and len(text) > 5:  # Only consider substantial extractions
                            all_texts.append((method_name, text))
                    except Exception:
                        continue
            
            # Try to extract routing and account numbers from all texts
            for method_name, text in all_texts:
                # Clean text - keep only digits
                cleaned = re.sub(r'[^\d]', '', text)
                
                # MICR format: ⑆routing⑆ account⑆ check
                # But OCR reads it as continuous digits without delimiters
                # Try different parsing strategies:
                
                # Strategy 1: Look for 9-digit routing at start or after a few digits
                for offset in [0, 1, 2, 3, 4]:
                    if len(cleaned) > offset + 9:
                        potential_routing = cleaned[offset:offset+9]
                        if self.validate_routing_number(potential_routing):
                            result["routing_number"] = potential_routing
                            print(f"✅ Routing number found [{method_name}] at offset {offset}: {potential_routing}")
                            
                            # Account number comes after routing
                            remaining = cleaned[offset+9:]
                            if len(remaining) >= 4:
                                # Try to find account (next 4-17 digits)
                                # Account usually ends before check number
                                for acc_len in range(min(17, len(remaining)), 3, -1):
                                    potential_account = remaining[:acc_len]
                                    if len(potential_account) >= 4:
                                        result["account_number"] = potential_account
                                        print(f"✅ Account number found: ****{result['account_number'][-4:]}")
                                        break
                            
                            break
                    
                    if result["routing_number"]:
                        break
                
                # Strategy 2: Scan for any valid 9-digit routing in the string
                if not result["routing_number"]:
                    for i in range(len(cleaned) - 8):
                        potential_routing = cleaned[i:i+9]
                        if self.validate_routing_number(potential_routing):
                            result["routing_number"] = potential_routing
                            print(f"✅ Routing number found [{method_name}] at position {i}: {potential_routing}")
                            
                            # Try to get account after routing
                            remaining = cleaned[i+9:]
                            if len(remaining) >= 4:
                                account_candidate = remaining[:12]  # Try 12 digits
                                if len(account_candidate) >= 4:
                                    result["account_number"] = account_candidate
                                    print(f"✅ Account number found: ****{result['account_number'][-4:]}")
                            break
                
                if result["routing_number"]:
                    break
            
            if not result["routing_number"]:
                    print("⚠️  MICR: No valid routing number found (E-13B font limitation)")
            
        except Exception as e:
            print(f"❌ MICR extraction failed: {e}")
        
        return result
    
    def validate_routing_number(self, routing_number: str) -> bool:
        """
        Validate routing number using ABA checksum algorithm.
        """
        if not routing_number or len(routing_number) != 9:
            return False
        
        try:
            digits = [int(d) for d in routing_number]
            
            # ABA checksum formula
            checksum = (
                3 * (digits[0] + digits[3] + digits[6]) +
                7 * (digits[1] + digits[4] + digits[7]) +
                (digits[2] + digits[5] + digits[8])
            ) % 10
            
            return checksum == 0
            
        except (ValueError, IndexError):
            return False
    
    def validate_date(self, date_str: str) -> Dict:
        """
        Validate extracted date.
        """
        validation = {
            "is_valid": False,
            "parsed_date": None,
            "is_future": False,
            "is_too_old": False,
            "warnings": []
        }
        
        if not date_str:
            validation["warnings"].append("No date found")
            return validation
        
        # Try to parse date
        date_formats = [
            "%m/%d/%Y",
            "%m-%d-%Y",
            "%m/%d/%y",
            "%m-%d-%y",
            "%B %d, %Y",
            "%b %d, %Y"
        ]
        
        for fmt in date_formats:
            try:
                parsed = datetime.strptime(date_str, fmt)
                validation["parsed_date"] = parsed.isoformat()
                validation["is_valid"] = True
                
                # Check if date is in future
                if parsed > datetime.now():
                    validation["is_future"] = True
                    validation["warnings"].append("Date is in the future")
                
                # Check if date is very old (> 6 months)
                days_old = (datetime.now() - parsed).days
                if days_old > 180:
                    validation["is_too_old"] = True
                    validation["warnings"].append(f"Check is {days_old} days old")
                
                break
                
            except ValueError:
                continue
        
        if not validation["is_valid"]:
            validation["warnings"].append("Could not parse date format")
        
        return validation
    
    def validate_amount_match(self, numeric: Optional[float], written: Optional[str]) -> Dict:
        """
        Cross-validate numeric amount with written amount.
        """
        validation = {
            "matches": False,
            "confidence": 0,
            "numeric_amount": numeric,
            "written_amount": written,
            "warnings": []
        }
        
        if numeric is None:
            validation["warnings"].append("Numeric amount not found")
        
        if written is None:
            validation["warnings"].append("Written amount not found")
        
        if numeric and written:
            # Simple check: convert written to number and compare
            # This is a simplified version; full implementation would need a complete number-to-text parser
            written_lower = written.lower()
            
            # For now, just check if the amounts are reasonably consistent
            # by looking for key numbers in the written amount
            str_numeric = str(int(numeric))
            if any(char in written_lower for char in str_numeric):
                validation["matches"] = True
                validation["confidence"] = 70
            else:
                validation["confidence"] = 30
                validation["warnings"].append("Amounts may not match - manual verification recommended")
        
        return validation
    
    def extract_and_validate(self) -> Dict:
        """
        Execute complete OCR extraction and validation.
        """
        # Extract all text first
        self.extract_text()
        
        # Extract individual fields
        check_number = self.extract_check_number()
        date = self.extract_date()
        payee = self.extract_payee()
        amount_numeric = self.extract_amount_numeric()
        amount_written = self.extract_amount_written()
        micr_data = self.extract_micr_line()
        
        # Validate
        date_validation = self.validate_date(date)
        amount_validation = self.validate_amount_match(amount_numeric, amount_written)
        routing_valid = self.validate_routing_number(micr_data.get("routing_number"))
        
        results = {
            "success": True,
            "extractedData": {
                "checkNumber": check_number,
                "date": date,
                "payee": payee,
                "amountNumeric": amount_numeric,
                "amountWritten": amount_written,
                "routingNumber": micr_data.get("routing_number"),
                "accountNumber": micr_data.get("account_number")[-4:] if micr_data.get("account_number") else None,  # Mask
                "micrLine": micr_data.get("full_micr")
            },
            "validationResults": {
                "dateValidation": date_validation,
                "amountMatch": amount_validation,
                "micrValid": {
                    "passed": routing_valid,
                    "confidence": 100 if routing_valid else 0,
                    "details": "Routing number checksum valid" if routing_valid else "Invalid routing number"
                }
            },
            "confidence_scores": {
                "overall": 0,
                "check_number": 80 if check_number else 0,
                "date": 90 if date_validation["is_valid"] else 0,
                "payee": 75 if payee else 0,
                "amount": amount_validation["confidence"],
                "micr": 100 if routing_valid else 50
            },
            "warnings": []
        }
        
        # Collect all warnings
        if not check_number:
            results["warnings"].append("Check number not found")
        if not payee:
            results["warnings"].append("Payee name not found")
        if not amount_numeric:
            results["warnings"].append("Numeric amount not found")
        
        results["warnings"].extend(date_validation.get("warnings", []))
        results["warnings"].extend(amount_validation.get("warnings", []))
        
        # Calculate overall confidence
        confidence_values = [v for v in results["confidence_scores"].values() if isinstance(v, (int, float))]
        results["confidence_scores"]["overall"] = sum(confidence_values) / len(confidence_values) if confidence_values else 0
        
        return results

