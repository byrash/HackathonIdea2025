import asyncio
import json
import cv2
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict

from .image_processor import ImageProcessor
from .metadata_analyzer import MetadataAnalyzer
from .ocr_service import OCRService
from .forensic_analyzer import ForensicAnalyzer
from .template_matcher import TemplateMatcher
from .ml_predictor import MLPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FraudDetector:
    """
    Main fraud detection orchestrator.
    Executes all 8 analysis stages with real-time progress updates.
    
    Stages:
    1. Image Upload & Validation (0-10%)
    2. Preprocessing (10-20%)
    3. Metadata Analysis (20-30%)
    4. OCR & Text Extraction (30-50%)
    5. Cross-Field Validation (50-60%)
    6. Forensic Analysis (60-75%)
    7. Security Features Check (75-90%)
    8. ML-Based Fraud Detection & Report Generation (90-100%)
    """
    
    def __init__(self, job_id: str, image_path: str):
        self.job_id = job_id
        self.image_path = image_path
        self.jobs_dir = Path("./jobs")
        self.uploads_dir = Path("./uploads")
        
        logger.info(f"🔍 [Job {job_id}] FraudDetector initialized")
        logger.info(f"📁 [Job {job_id}] Image path: {image_path}")
        
        self.results = {
            "jobId": job_id,
            "status": "PROCESSING",
            "uploadTimestamp": datetime.utcnow().isoformat(),
            "currentStage": 0,
            "currentPercentage": 0,
            "message": "",
            "stageHistory": []
        }
    
    def update_progress(self, stage: int, percentage: int, message: str, status: str = "PROCESSING"):
        """
        Update job progress and save to JSON file.
        """
        logger.info(f"📊 [Job {self.job_id}] Stage {stage}/8 ({percentage}%) - {message}")
        
        self.results["currentStage"] = stage
        self.results["currentPercentage"] = percentage
        self.results["message"] = message
        self.results["status"] = status
        self.results["timestamp"] = datetime.utcnow().isoformat()
        
        # Save to job file
        job_file = self.jobs_dir / f"{self.job_id}.json"
        with open(job_file, "w") as f:
            json.dump(self.results, f, indent=2)
    
    def add_stage_completion(self, stage: int, stage_name: str, status: str = "COMPLETED"):
        """
        Record stage completion in history.
        """
        logger.info(f"✅ [Job {self.job_id}] {stage_name} - {status}")
        
        self.results["stageHistory"].append({
            "stage": stage,
            "stageName": stage_name,
            "completedAt": datetime.utcnow().isoformat(),
            "status": status
        })
    
    async def stage1_validate_image(self) -> Dict:
        """
        Stage 1: Image Upload & Validation (0-10%)
        """
        self.update_progress(1, 5, "Validating image format...")
        await asyncio.sleep(0.5)  # Simulate processing
        
        result = {
            "success": False,
            "format": None,
            "dimensions": None,
            "quality_score": 0
        }
        
        try:
            # Validate file exists
            if not Path(self.image_path).exists():
                result["error"] = "Image file not found"
                return result
            
            # Load and validate image
            image = cv2.imread(self.image_path)
            if image is None:
                result["error"] = "Invalid image format"
                return result
            
            # Get image properties
            height, width = image.shape[:2]
            result["dimensions"] = {"width": width, "height": height}
            result["format"] = Path(self.image_path).suffix
            
            # Check minimum resolution
            min_dimension = min(height, width)
            if min_dimension < 500:
                result["warning"] = "Image resolution is low, analysis may be less accurate"
            
            # Get quality score
            processor = ImageProcessor(self.image_path)
            quality = processor.get_image_quality_score()
            result["quality_score"] = quality["overall_score"]
            result["quality_details"] = quality
            
            result["success"] = True
            
            self.update_progress(1, 10, "✓ Image validation complete")
            self.add_stage_completion(1, "Image Validation")
            
        except Exception as e:
            result["error"] = str(e)
            self.update_progress(1, 10, f"⚠ Validation warning: {str(e)}", "WARNING")
        
        return result
    
    async def stage2_preprocess(self, check_bounds: dict = None) -> Dict:
        """
        Stage 2: Preprocessing (10-20%)
        Lightweight enhancement for better OCR (no cropping).
        """
        self.update_progress(2, 15, "Enhancing image for OCR...")
        await asyncio.sleep(0.2)
        
        try:
            processor = ImageProcessor(self.image_path)
            # Fast preprocessing: enhance only, no cropping
            results = processor.preprocess(check_bounds=None)
            
            # Save processed (enhanced) image for OCR
            processed_path = str(self.uploads_dir / f"{self.job_id}_processed.jpg")
            processor.save_processed_image(processed_path)
            
            self.update_progress(2, 20, "✓ Image enhanced for OCR")
            self.add_stage_completion(2, "Preprocessing")
            
            return results
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self.update_progress(2, 20, f"⚠ Preprocessing error: {str(e)}", "WARNING")
            return error_result
    
    async def stage3_metadata_analysis(self) -> Dict:
        """
        Stage 3: Metadata Analysis (20-30%)
        IMPORTANT: Uses ORIGINAL image to preserve EXIF data
        """
        self.update_progress(3, 25, "Extracting EXIF metadata...")
        await asyncio.sleep(0.3)
        
        try:
            # MUST use original image - processed image loses EXIF metadata
            analyzer = MetadataAnalyzer(self.image_path)
            
            self.update_progress(3, 28, "Checking for manipulation history...")
            await asyncio.sleep(0.2)
            
            results = analyzer.analyze()
            
            self.update_progress(3, 30, "✓ Metadata analysis complete")
            self.add_stage_completion(3, "Metadata Analysis")
            
            return results
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self.update_progress(3, 30, f"⚠ Metadata analysis error: {str(e)}", "WARNING")
            return error_result
    
    async def stage4_ocr_extraction(self) -> Dict:
        """
        Stage 4: OCR & Text Extraction (30-50%)
        IMPORTANT: Uses PROCESSED image for better OCR accuracy
        """
        self.update_progress(4, 35, "Running OCR engine...")
        await asyncio.sleep(0.5)
        
        try:
            # Use PROCESSED image for better OCR (enhanced but not cropped)
            processed_path = self.uploads_dir / f"{self.job_id}_processed.jpg"
            ocr_image_path = str(processed_path) if processed_path.exists() else self.image_path
            
            ocr_service = OCRService(ocr_image_path)
            
            self.update_progress(4, 40, "Extracting MICR line...")
            await asyncio.sleep(0.3)
            
            self.update_progress(4, 45, "Reading payee and amount fields...")
            await asyncio.sleep(0.3)
            
            results = ocr_service.extract_and_validate()
            
            self.update_progress(4, 50, "✓ Text extraction complete")
            self.add_stage_completion(4, "OCR & Text Extraction")
            
            return results
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self.update_progress(4, 50, f"⚠ OCR error: {str(e)}", "WARNING")
            return error_result
    
    async def stage5_validation(self, ocr_results: Dict) -> Dict:
        """
        Stage 5: Cross-Field Validation (50-60%)
        """
        self.update_progress(5, 55, "Validating extracted data...")
        await asyncio.sleep(0.3)
        
        try:
            self.update_progress(5, 58, "Cross-checking amount fields...")
            await asyncio.sleep(0.2)
            
            # Validation is already done in OCR service
            validation_results = ocr_results.get("validationResults", {})
            
            self.update_progress(5, 60, "✓ Field validation complete")
            self.add_stage_completion(5, "Cross-Field Validation")
            
            return {
                "success": True,
                "validations": validation_results
            }
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self.update_progress(5, 60, f"⚠ Validation error: {str(e)}", "WARNING")
            return error_result
    
    async def stage6_forensic_analysis(self) -> Dict:
        """
        Stage 6: Forensic Analysis (60-75%)
        CRITICAL: Uses ORIGINAL image - processed image would hide tampering!
        """
        self.update_progress(6, 62, "Running error level analysis...")
        await asyncio.sleep(0.5)
        
        try:
            # MUST use original image - ELA/clone detection need unmodified pixels
            analyzer = ForensicAnalyzer(self.image_path)
            
            self.update_progress(6, 67, "Detecting cloned regions...")
            await asyncio.sleep(0.3)
            
            self.update_progress(6, 72, "Analyzing edges and boundaries...")
            await asyncio.sleep(0.3)
            
            results = analyzer.analyze()
            
            self.update_progress(6, 75, "✓ Forensic analysis complete")
            self.add_stage_completion(6, "Forensic Analysis")
            
            return results
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self.update_progress(6, 75, f"⚠ Forensic analysis error: {str(e)}", "WARNING")
            return error_result
    
    async def stage7_security_features(self) -> Dict:
        """
        Stage 7: Security Features Check (75-90%)
        IMPORTANT: Uses ORIGINAL image for authentic template matching
        """
        self.update_progress(7, 78, "Checking for watermarks...")
        await asyncio.sleep(0.3)
        
        try:
            # Use original image - watermarks/security features should match original
            matcher = TemplateMatcher(self.image_path, "./templates")
            
            self.update_progress(7, 82, "Matching against bank templates...")
            await asyncio.sleep(0.4)
            
            self.update_progress(7, 87, "Verifying security features...")
            await asyncio.sleep(0.3)
            
            results = matcher.analyze()
            
            self.update_progress(7, 90, "✓ Security check complete")
            self.add_stage_completion(7, "Security Features Check")
            
            return results
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self.update_progress(7, 90, f"⚠ Security check error: {str(e)}", "WARNING")
            return error_result
    
    async def stage8_ml_prediction(self, all_results: Dict) -> Dict:
        """
        Stage 8: ML-Based Fraud Detection & Report Generation (90-100%)
        """
        self.update_progress(8, 92, "Running ML fraud detection model...")
        await asyncio.sleep(0.3)
        
        try:
            predictor = MLPredictor()
            
            self.update_progress(8, 95, "Calculating risk score...")
            await asyncio.sleep(0.2)
            
            # Prepare analysis results for ML prediction
            analysis_data = {
                "metadata": all_results.get("stage3_metadata", {}),
                "ocr": all_results.get("stage4_ocr", {}),
                "forensics": all_results.get("stage6_forensics", {}),
                "security": all_results.get("stage7_security", {})
            }
            
            ml_results = predictor.predict(analysis_data)
            
            self.update_progress(8, 98, "Generating analysis report...")
            await asyncio.sleep(0.2)
            
            # Create annotated image with suspicious regions highlighted
            await self.create_annotated_image(all_results)
            
            self.update_progress(8, 100, "✓ Analysis complete!")
            self.add_stage_completion(8, "ML Fraud Detection")
            
            return ml_results
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self.update_progress(8, 100, f"⚠ ML prediction error: {str(e)}", "WARNING")
            return error_result
    
    async def create_annotated_image(self, all_results: Dict):
        """
        Create annotated image with suspicious regions highlighted.
        """
        try:
            image = cv2.imread(self.image_path)
            if image is None:
                return
            
            # Draw rectangles around suspicious regions from forensic analysis
            forensics = all_results.get("stage6_forensics", {})
            
            # ELA suspicious regions (red)
            ela_regions = forensics.get("errorLevelAnalysis", {}).get("suspicious_regions", [])
            for region in ela_regions:
                coords = region.get("coordinates", {})
                x, y, w, h = coords.get("x", 0), coords.get("y", 0), coords.get("width", 0), coords.get("height", 0)
                color = (0, 0, 255)  # Red for high severity
                if region.get("severity") == "MEDIUM":
                    color = (0, 165, 255)  # Orange
                elif region.get("severity") == "LOW":
                    color = (0, 255, 255)  # Yellow
                
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, "ELA", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Irregular edges (blue)
            irregular_edges = forensics.get("edgeAnalysis", {}).get("irregular_edges", [])
            for region in irregular_edges[:5]:  # Limit to first 5
                coords = region.get("coordinates", {})
                x, y, w, h = coords.get("x", 0), coords.get("y", 0), coords.get("width", 0), coords.get("height", 0)
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Save annotated image
            annotated_path = str(self.uploads_dir / f"{self.job_id}_annotated.jpg")
            cv2.imwrite(annotated_path, image)
            
        except Exception as e:
            print(f"Error creating annotated image: {str(e)}")
    
    async def analyze(self):
        """
        Execute complete fraud detection pipeline.
        """
        try:
            all_results = {}
            
            # Stage 1: Image Validation
            stage1_result = await self.stage1_validate_image()
            all_results["stage1_validation"] = stage1_result
            
            if not stage1_result.get("success", False):
                self.results["status"] = "FAILED"
                self.results["error"] = stage1_result.get("error", "Image validation failed")
                self.update_progress(1, 10, "Analysis failed", "FAILED")
                return
            
            # Stage 2: Preprocessing
            stage2_result = await self.stage2_preprocess()
            all_results["stage2_preprocessing"] = stage2_result
            
            # Stage 3: Metadata Analysis
            stage3_result = await self.stage3_metadata_analysis()
            all_results["stage3_metadata"] = stage3_result
            
            # Stage 4: OCR & Text Extraction
            stage4_result = await self.stage4_ocr_extraction()
            all_results["stage4_ocr"] = stage4_result
            
            # Stage 5: Cross-Field Validation
            stage5_result = await self.stage5_validation(stage4_result)
            all_results["stage5_validation"] = stage5_result
            
            # Stage 6: Forensic Analysis
            stage6_result = await self.stage6_forensic_analysis()
            all_results["stage6_forensics"] = stage6_result
            
            # Stage 7: Security Features Check
            stage7_result = await self.stage7_security_features()
            all_results["stage7_security"] = stage7_result
            
            # Stage 8: ML Prediction & Report Generation
            stage8_result = await self.stage8_ml_prediction(all_results)
            all_results["stage8_ml"] = stage8_result
            
            # Compile final results
            self.results.update({
                "status": "COMPLETED",
                "analysisCompletedAt": datetime.utcnow().isoformat(),
                "imageFiles": {
                    "original": f"./uploads/{self.job_id}_original{Path(self.image_path).suffix}",
                    "annotated": f"./uploads/{self.job_id}_annotated.jpg"
                },
                "overallRiskScore": stage8_result.get("overallRiskScore", 0),
                "verdict": stage8_result.get("verdict", "UNKNOWN"),
                "extractedData": stage4_result.get("extractedData", {}),
                "validationResults": stage4_result.get("validationResults", {}),
                "forensicAnalysis": stage6_result,
                "securityFeatures": stage7_result,
                "mlPrediction": stage8_result.get("mlPrediction", {}),
                "recommendations": stage8_result.get("recommendations", []),
                "metadata_analysis": stage3_result
            })
            
            # Save final results
            job_file = self.jobs_dir / f"{self.job_id}.json"
            with open(job_file, "w") as f:
                json.dump(self.results, f, indent=2)
            
        except Exception as e:
            self.results["status"] = "FAILED"
            self.results["error"] = str(e)
            self.results["failedAt"] = datetime.utcnow().isoformat()
            
            job_file = self.jobs_dir / f"{self.job_id}.json"
            with open(job_file, "w") as f:
                json.dump(self.results, f, indent=2)

