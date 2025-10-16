import exifread
from PIL import Image
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import requests


class MetadataAnalyzer:
    """
    Service for analyzing image metadata (EXIF data).
    Stage 3: Metadata Analysis (20-30%)
    
    Detects:
    - EXIF data inconsistencies
    - Creation date/time verification
    - Camera/scanner model validation
    - Software modification history detection
    """
    
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.exif_data = {}
        self.pil_exif = {}
    
    def extract_exif_data(self) -> Dict:
        """
        Extract EXIF metadata from image using exifread.
        """
        try:
            with open(self.image_path, 'rb') as f:
                tags = exifread.process_file(f, details=True)
                
                # Convert tags to dictionary with string values
                for tag, value in tags.items():
                    self.exif_data[tag] = str(value)
                
                return self.exif_data
                
        except Exception as e:
            print(f"Error extracting EXIF data: {str(e)}")
            return {}
    
    def extract_pil_exif(self) -> Dict:
        """
        Extract EXIF data using PIL (alternative method).
        """
        try:
            image = Image.open(self.image_path)
            exif = image.getexif()
            
            if exif:
                for tag_id, value in exif.items():
                    tag_name = Image.ExifTags.TAGS.get(tag_id, tag_id)
                    self.pil_exif[tag_name] = str(value)
            
            return self.pil_exif
            
        except Exception as e:
            print(f"Error extracting PIL EXIF: {str(e)}")
            return {}
    
    def analyze_creation_date(self) -> Dict:
        """
        Verify creation date/time for inconsistencies.
        """
        analysis = {
            "date_found": False,
            "creation_date": None,
            "modified_date": None,
            "inconsistencies": [],
            "warnings": []
        }
        
        # Check for date fields in EXIF
        date_fields = [
            'EXIF DateTimeOriginal',
            'EXIF DateTimeDigitized',
            'Image DateTime'
        ]
        
        dates = {}
        for field in date_fields:
            if field in self.exif_data:
                try:
                    date_str = self.exif_data[field]
                    # Parse EXIF date format: "YYYY:MM:DD HH:MM:SS"
                    date_obj = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
                    dates[field] = date_obj
                    analysis["date_found"] = True
                except Exception as e:
                    analysis["warnings"].append(f"Failed to parse {field}: {str(e)}")
        
        if dates:
            # Set primary dates
            analysis["creation_date"] = str(dates.get('EXIF DateTimeOriginal', list(dates.values())[0]))
            analysis["modified_date"] = str(dates.get('Image DateTime', list(dates.values())[0]))
            
            # Check for inconsistencies (dates should be close)
            date_values = list(dates.values())
            if len(date_values) > 1:
                time_diffs = []
                for i in range(len(date_values) - 1):
                    diff = abs((date_values[i] - date_values[i + 1]).total_seconds())
                    time_diffs.append(diff)
                
                # Flag if dates differ by more than 1 hour
                if any(diff > 3600 for diff in time_diffs):
                    analysis["inconsistencies"].append(
                        "Creation and modification dates differ significantly"
                    )
            
            # Check if date is in the future
            if any(d > datetime.now() for d in date_values):
                analysis["inconsistencies"].append("Date is in the future")
        
        return analysis
    
    def analyze_camera_model(self) -> Dict:
        """
        Validate camera/scanner model information.
        """
        analysis = {
            "device_found": False,
            "camera_make": None,
            "camera_model": None,
            "software": None,
            "is_scanner": False,
            "is_mobile": False,
            "warnings": []
        }
        
        # Extract camera/device information
        if 'Image Make' in self.exif_data:
            analysis["camera_make"] = self.exif_data['Image Make']
            analysis["device_found"] = True
        
        if 'Image Model' in self.exif_data:
            analysis["camera_model"] = self.exif_data['Image Model']
            analysis["device_found"] = True
        
        if 'Image Software' in self.exif_data:
            analysis["software"] = self.exif_data['Image Software']
        
        # Detect if image is from scanner
        scanner_keywords = ['scanner', 'scan', 'epson', 'canon scan', 'hp scan']
        if analysis["camera_model"]:
            model_lower = analysis["camera_model"].lower()
            if any(keyword in model_lower for keyword in scanner_keywords):
                analysis["is_scanner"] = True
        
        # Detect if image is from mobile device
        mobile_keywords = ['iphone', 'android', 'samsung', 'pixel', 'huawei', 'xiaomi']
        if analysis["camera_make"] or analysis["camera_model"]:
            device_text = f"{analysis['camera_make']} {analysis['camera_model']}".lower()
            if any(keyword in device_text for keyword in mobile_keywords):
                analysis["is_mobile"] = True
        
        # Legitimate checks are typically scanned or captured with good scanners
        if not analysis["device_found"]:
            analysis["warnings"].append("No camera/device information found in EXIF")
        
        return analysis
    
    def detect_software_modifications(self) -> Dict:
        """
        Detect if image has been edited with software.
        """
        analysis = {
            "software_detected": False,
            "editing_software": [],
            "modification_indicators": [],
            "risk_level": "LOW"
        }
        
        # Common editing software
        editing_software = [
            'photoshop', 'gimp', 'paint.net', 'pixlr', 'lightroom',
            'affinity', 'corel', 'paintshop', 'canva', 'snapseed'
        ]
        
        # Check software field
        if 'Image Software' in self.exif_data:
            software = self.exif_data['Image Software'].lower()
            analysis["software_detected"] = True
            
            for editor in editing_software:
                if editor in software:
                    analysis["editing_software"].append(editor.title())
                    analysis["modification_indicators"].append(
                        f"Image processed with {editor.title()}"
                    )
        
        # Check for editing-related EXIF fields
        suspicious_fields = [
            'Photoshop Thumbnail',
            'Adobe RGB',
            'Image History',
            'Processing Software'
        ]
        
        for field in suspicious_fields:
            if any(field.lower() in key.lower() for key in self.exif_data.keys()):
                analysis["modification_indicators"].append(
                    f"Contains {field} data"
                )
        
        # Determine risk level
        if len(analysis["editing_software"]) > 0:
            analysis["risk_level"] = "MEDIUM"
        
        if len(analysis["modification_indicators"]) > 2:
            analysis["risk_level"] = "HIGH"
        
        return analysis
    
    def extract_gps_location(self) -> Dict:
        """
        Extract GPS coordinates from EXIF data and convert to human-readable location.
        Returns coordinates and reverse geocoded address.
        """
        analysis = {
            "gps_found": False,
            "latitude": None,
            "longitude": None,
            "altitude": None,
            "location_name": None,
            "city": None,
            "state": None,
            "country": None,
            "map_url": None,
            "warnings": []
        }
        
        try:
            # Extract GPS coordinates from EXIF
            if 'GPS GPSLatitude' in self.exif_data and 'GPS GPSLongitude' in self.exif_data:
                # Parse GPS coordinates
                lat = self._convert_gps_to_decimal(
                    self.exif_data['GPS GPSLatitude'],
                    self.exif_data.get('GPS GPSLatitudeRef', 'N')
                )
                lon = self._convert_gps_to_decimal(
                    self.exif_data['GPS GPSLongitude'],
                    self.exif_data.get('GPS GPSLongitudeRef', 'E')
                )
                
                if lat is not None and lon is not None:
                    analysis["gps_found"] = True
                    analysis["latitude"] = lat
                    analysis["longitude"] = lon
                    
                    # Extract altitude if available
                    if 'GPS GPSAltitude' in self.exif_data:
                        try:
                            alt_str = self.exif_data['GPS GPSAltitude']
                            # Parse fraction format like "123/1"
                            if '/' in alt_str:
                                num, den = alt_str.split('/')
                                analysis["altitude"] = float(num) / float(den)
                            else:
                                analysis["altitude"] = float(alt_str)
                        except Exception:
                            pass
                    
                    # Create map URL
                    analysis["map_url"] = f"https://www.google.com/maps?q={lat},{lon}"
                    
                    # Reverse geocode to get address (using free OpenStreetMap Nominatim API)
                    try:
                        location_data = self._reverse_geocode(lat, lon)
                        if location_data:
                            analysis["location_name"] = location_data.get("display_name")
                            address = location_data.get("address", {})
                            analysis["city"] = address.get("city") or address.get("town") or address.get("village")
                            analysis["state"] = address.get("state")
                            analysis["country"] = address.get("country")
                            
                            print(f"📍 GPS Location: {lat:.6f}, {lon:.6f}")
                            if analysis["city"]:
                                print(f"   Address: {analysis['city']}, {analysis['state']}, {analysis['country']}")
                    except Exception as e:
                        analysis["warnings"].append(f"Reverse geocoding failed: {str(e)}")
                        print(f"⚠️  Could not reverse geocode: {str(e)}")
        
        except Exception as e:
            analysis["warnings"].append(f"GPS extraction failed: {str(e)}")
            print(f"Error extracting GPS: {str(e)}")
        
        return analysis
    
    def _convert_gps_to_decimal(self, coord_str: str, ref: str) -> Optional[float]:
        """
        Convert GPS coordinates from EXIF format to decimal degrees.
        EXIF format: "[12, 34, 56.78]" where values are degrees, minutes, seconds
        """
        try:
            # Remove brackets and split
            coord_str = coord_str.strip('[]')
            parts = [p.strip() for p in coord_str.split(',')]
            
            if len(parts) != 3:
                return None
            
            # Parse degrees, minutes, seconds (may be fractions like "123/1")
            degrees = self._parse_fraction(parts[0])
            minutes = self._parse_fraction(parts[1])
            seconds = self._parse_fraction(parts[2])
            
            # Convert to decimal
            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
            
            # Apply hemisphere (S and W are negative)
            if ref in ['S', 'W']:
                decimal = -decimal
            
            return decimal
            
        except Exception as e:
            print(f"Error converting GPS coordinate: {str(e)}")
            return None
    
    def _parse_fraction(self, value_str: str) -> float:
        """Parse a value that might be a fraction (e.g., "123/1" or "12.34")."""
        try:
            if '/' in value_str:
                num, den = value_str.split('/')
                return float(num) / float(den)
            else:
                return float(value_str)
        except Exception:
            return 0.0
    
    def _reverse_geocode(self, lat: float, lon: float, timeout: int = 3) -> Optional[Dict]:
        """
        Reverse geocode coordinates to get human-readable address.
        Uses OpenStreetMap Nominatim API (free, no API key required).
        """
        try:
            url = "https://nominatim.openstreetmap.org/reverse"
            params = {
                "lat": lat,
                "lon": lon,
                "format": "json",
                "addressdetails": 1,
                "zoom": 18
            }
            headers = {
                "User-Agent": "CheckFraudDetection/1.0"  # Required by Nominatim
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"⚠️  Reverse geocoding returned status {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            print("⚠️  Reverse geocoding timeout")
            return None
        except Exception as e:
            print(f"⚠️  Reverse geocoding error: {str(e)}")
            return None
    
    def check_exif_consistency(self) -> Dict:
        """
        Check for EXIF data consistency and completeness.
        """
        analysis = {
            "has_exif": len(self.exif_data) > 0,
            "exif_field_count": len(self.exif_data),
            "inconsistencies": [],
            "completeness_score": 0
        }
        
        # Expected EXIF fields for legitimate images
        expected_fields = [
            'Image Make',
            'Image Model',
            'Image DateTime',
            'EXIF DateTimeOriginal',
            'Image XResolution',
            'Image YResolution'
        ]
        
        found_fields = sum(1 for field in expected_fields if field in self.exif_data)
        analysis["completeness_score"] = (found_fields / len(expected_fields)) * 100
        
        # Completely missing EXIF is suspicious
        if not analysis["has_exif"]:
            analysis["inconsistencies"].append("No EXIF data found (possibly stripped)")
        
        # Very few EXIF fields is suspicious
        elif analysis["exif_field_count"] < 5:
            analysis["inconsistencies"].append("Minimal EXIF data (possibly edited)")
        
        # Check resolution consistency
        if 'Image XResolution' in self.exif_data and 'Image YResolution' in self.exif_data:
            x_res = self.exif_data['Image XResolution']
            y_res = self.exif_data['Image YResolution']
            
            if x_res != y_res:
                analysis["inconsistencies"].append(
                    "X and Y resolutions differ (unusual for scanners)"
                )
        
        return analysis
    
    def analyze(self) -> Dict:
        """
        Execute complete metadata analysis.
        """
        # Extract EXIF data
        self.extract_exif_data()
        self.extract_pil_exif()
        
        results = {
            "success": True,
            "exif_field_count": len(self.exif_data),
            "has_metadata": len(self.exif_data) > 0,
            "creation_date_analysis": self.analyze_creation_date(),
            "camera_analysis": self.analyze_camera_model(),
            "modification_analysis": self.detect_software_modifications(),
            "consistency_analysis": self.check_exif_consistency(),
            "gps_location": self.extract_gps_location(),
            "all_exif_data": self.exif_data,
            "overall_risk_score": 0,
            "flags": []
        }
        
        # Calculate overall risk score
        risk_factors = 0
        
        if results["creation_date_analysis"]["inconsistencies"]:
            risk_factors += len(results["creation_date_analysis"]["inconsistencies"]) * 15
            results["flags"].extend(results["creation_date_analysis"]["inconsistencies"])
        
        if results["modification_analysis"]["risk_level"] == "HIGH":
            risk_factors += 30
            results["flags"].append("High risk of software modifications")
        elif results["modification_analysis"]["risk_level"] == "MEDIUM":
            risk_factors += 15
            results["flags"].append("Medium risk of software modifications")
        
        if results["consistency_analysis"]["inconsistencies"]:
            risk_factors += len(results["consistency_analysis"]["inconsistencies"]) * 10
            results["flags"].extend(results["consistency_analysis"]["inconsistencies"])
        
        if not results["camera_analysis"]["device_found"]:
            risk_factors += 10
            results["flags"].append("No device information found")
        
        results["overall_risk_score"] = min(100, risk_factors)
        
        # GPS-based risk adjustment
        gps_location = results.get("gps_location", {})
        if gps_location.get("gps_found"):
            country = gps_location.get("country", "").lower()
            # USA checks are generally lower risk
            if "united states" in country or country in ["usa", "us"]:
                results["overall_risk_score"] = max(0, results["overall_risk_score"] - 10)
                results["flags"].append("USA location detected - adjusted risk down")
            # International checks may warrant higher scrutiny
            elif country and country not in ["united states", "usa", "us"]:
                results["overall_risk_score"] = min(100, results["overall_risk_score"] + 5)
                results["flags"].append(f"International location ({gps_location.get('country')}) - manual review suggested")
        
        return results
    
    def get_summary(self) -> str:
        """
        Get human-readable summary of metadata analysis.
        """
        results = self.analyze()
        
        summary_parts = []
        
        if results["has_metadata"]:
            summary_parts.append(f"✓ EXIF data present ({results['exif_field_count']} fields)")
        else:
            summary_parts.append("⚠ No EXIF data found")
        
        if results["camera_analysis"]["device_found"]:
            device = f"{results['camera_analysis']['camera_make']} {results['camera_analysis']['camera_model']}"
            summary_parts.append(f"Device: {device}")
        
        if results["flags"]:
            summary_parts.append(f"⚠ {len(results['flags'])} warning(s) detected")
        
        return " | ".join(summary_parts)

