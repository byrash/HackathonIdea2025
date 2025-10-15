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
        
        print(f"⚠️  Check number not found")
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
        
        print(f"⚠️  Date not found")
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
        
        print(f"⚠️  Payee not found")
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
        
        print(f"⚠️  Amount not found")
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
            
            print(f"⚠️  Amount region extraction: no valid amount found")
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
    
    def _extract_micr_from_region(self) -> Dict:
        """
        Extract MICR line from bottom region of check with enhanced preprocessing.
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
            
            # Extract MICR from bottom region
            
            # MICR line is at the bottom of the check (bottom 8-12%)
            # Be more precise with the crop
            y_start = int(h * 0.88)
            micr_region = self.image[y_start:, :]  # Use color image first
            micr_gray = self.gray_image[y_start:, :]
            
            # Save MICR region for debugging
            micr_debug_path = self.image_path.replace('_original', '_micr_region')
            cv2.imwrite(micr_debug_path, micr_region)
            
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
                print(f"⚠️  MICR: No valid routing number found (E-13B font limitation)")
            
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

