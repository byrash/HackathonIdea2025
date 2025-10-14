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
            
            return self.extracted_text
            
        except Exception as e:
            print(f"Error extracting text: {str(e)}")
            return ""
    
    def extract_check_number(self) -> Optional[str]:
        """Extract check number (usually 3-4 digits in top right)."""
        if not self.extracted_text:
            self.extract_text()
        
        # Pattern for check number (typically 3-4 digits)
        patterns = [
            r'CHECK\s*#?\s*(\d{3,4})',
            r'NO\.?\s*(\d{3,4})',
            r'CHECK\s+NUMBER\s*:?\s*(\d{3,4})',
            r'^(\d{3,4})$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, self.extracted_text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1)
        
        # Try to find any 3-4 digit number in first few lines
        lines = self.extracted_text.split('\n')[:5]
        for line in lines:
            match = re.search(r'\b(\d{3,4})\b', line)
            if match:
                return match.group(1)
        
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
                return match.group(1)
        
        return None
    
    def extract_payee(self) -> Optional[str]:
        """Extract payee name from check."""
        if not self.extracted_text:
            self.extract_text()
        
        # Look for "Pay to the order of" pattern
        patterns = [
            r'PAY\s+TO\s+THE\s+ORDER\s+OF\s*:?\s*([A-Z][A-Za-z\s\.]+)',
            r'PAY\s+TO\s*:?\s*([A-Z][A-Za-z\s\.]+)',
            r'PAYEE\s*:?\s*([A-Z][A-Za-z\s\.]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, self.extracted_text, re.IGNORECASE)
            if match:
                payee = match.group(1).strip()
                # Clean up payee name
                payee = re.sub(r'\s+', ' ', payee)
                if len(payee) > 3 and len(payee) < 100:
                    return payee
        
        return None
    
    def extract_amount_numeric(self) -> Optional[float]:
        """Extract numeric amount from check."""
        if not self.extracted_text:
            self.extract_text()
        
        # Pattern for amount (dollar sign followed by numbers)
        patterns = [
            r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'AMOUNT\s*:?\s*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'\$\s*(\d+\.\d{2})'
        ]
        
        amounts = []
        for pattern in patterns:
            matches = re.finditer(pattern, self.extracted_text, re.IGNORECASE)
            for match in matches:
                amount_str = match.group(1).replace(',', '')
                try:
                    amount = float(amount_str)
                    if 0 < amount < 1000000:  # Reasonable check amount
                        amounts.append(amount)
                except ValueError:
                    continue
        
        # Return the most likely amount (usually the largest)
        if amounts:
            return max(amounts)
        
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
        """
        result = {
            "full_micr": None,
            "routing_number": None,
            "account_number": None,
            "check_number_micr": None
        }
        
        if not self.extracted_text:
            self.extract_text()
        
        # MICR typically contains routing number (9 digits), account number, and check number
        # Look for pattern with multiple digit groups
        micr_pattern = r'[⑆⑈]?\s*(\d{9})\s*[⑆⑈]\s*(\d{4,17})\s*[⑆⑈]\s*(\d{3,4})'
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
            
            # Extract numeric parts from written amount
            number_words = {
                'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
                'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
                'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
                'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
                'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
                'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
                'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000
            }
            
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

