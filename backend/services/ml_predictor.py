import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import pickle


class MLPredictor:
    """
    Service for ML-based fraud detection and scoring.
    Stage 8: ML-Based Fraud Detection & Report Generation (90-100%)
    
    Uses machine learning models for:
    - Anomaly detection
    - Fraud probability calculation
    - Risk score aggregation
    
    Falls back to rule-based scoring if no ML model is available.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        # Auto-detect model if not specified
        if model_path is None:
            # Try XGBoost model first, then RF for backward compatibility
            xgb_model = Path("./ml-models/fraud_detection_xgb.pkl")
            rf_model = Path("./ml-models/fraud_detection_rf.pkl")
            
            if xgb_model.exists():
                model_path = str(xgb_model)
            elif rf_model.exists():
                model_path = str(rf_model)
        
        self.model_path = model_path
        self.model = None
        self.model_data = None
        self.model_loaded = False
        
        # Try to load model on initialization
        if self.model_path:
            self.load_model()
    
    def load_model(self) -> bool:
        """
        Load pre-trained ML model if available.
        Supports both sklearn models and custom model data dictionaries.
        """
        if self.model_path and Path(self.model_path).exists():
            try:
                with open(self.model_path, 'rb') as f:
                    loaded_data = pickle.load(f)
                
                # Check if it's a model data dictionary or just a model
                if isinstance(loaded_data, dict):
                    self.model_data = loaded_data
                    self.model = loaded_data.get('model')
                else:
                    self.model = loaded_data
                    self.model_data = {}
                
                self.model_loaded = True
                print(f"✓ ML model loaded from: {self.model_path}")
                
                if self.model_data:
                    model_type = self.model_data.get('model_type', 'Unknown')
                    accuracy = self.model_data.get('accuracy', 0)
                    print(f"  Model type: {model_type}")
                    print(f"  Accuracy: {accuracy:.2%}")
                
                return True
            except Exception as e:
                print(f"⚠️ Error loading model: {str(e)}")
                print("  Falling back to rule-based scoring")
                return False
        return False
    
    def extract_features(self, analysis_results: Dict) -> np.ndarray:
        """
        Extract feature vector from analysis results for ML prediction.
        
        Returns 12 features:
        [metadata_risk, forensic_risk, ocr_risk, security_risk,
         ela_regions, irregular_edges, template_matched, watermark,
         exif_count, amount_mismatch, date_invalid, clone_count]
        """
        metadata = analysis_results.get("metadata", {})
        forensics = analysis_results.get("forensics", {})
        ocr = analysis_results.get("ocr", {})
        security = analysis_results.get("security", {})
        
        # Extract individual features
        metadata_risk = metadata.get("overall_risk_score", 0)
        forensic_risk = forensics.get("overall_risk_score", 0)
        
        # Calculate OCR risk from validation results
        ocr_risk = self._calculate_ocr_risk(ocr)
        
        # Calculate security risk
        security_risk = self._calculate_security_risk(security)
        
        # Extract detailed features
        ela_regions = len(forensics.get("errorLevelAnalysis", {}).get("suspicious_regions", []))
        irregular_edges = len(forensics.get("edgeAnalysis", {}).get("irregular_edges", []))
        template_matched = 1 if security.get("templateMatch", {}).get("matched", False) else 0
        watermark = 1 if security.get("watermarkDetected", False) else 0
        exif_count = metadata.get("exif_field_count", 0)
        
        validation = ocr.get("validationResults", {})
        amount_mismatch = 1 if not validation.get("amountMatch", {}).get("matches", True) else 0
        date_invalid = 1 if not validation.get("dateValidation", {}).get("is_valid", True) else 0
        
        clone_count = forensics.get("cloneDetection", {}).get("duplicate_count", 0)
        
        # Return as numpy array
        features = np.array([[
            metadata_risk,
            forensic_risk,
            ocr_risk,
            security_risk,
            ela_regions,
            irregular_edges,
            template_matched,
            watermark,
            exif_count,
            amount_mismatch,
            date_invalid,
            clone_count
        ]])
        
        return features
    
    def _calculate_ocr_risk(self, ocr: Dict) -> float:
        """Calculate OCR validation risk score."""
        ocr_risk = 0
        validation = ocr.get("validationResults", {})
        
        date_val = validation.get("dateValidation", {})
        if not date_val.get("is_valid", False):
            ocr_risk += 30
        if date_val.get("is_future", False):
            ocr_risk += 20
        if date_val.get("is_too_old", False):
            ocr_risk += 10
        
        amount_val = validation.get("amountMatch", {})
        if not amount_val.get("matches", False):
            ocr_risk += 30
        
        micr_val = validation.get("micrValid", {})
        if not micr_val.get("passed", False):
            ocr_risk += 20
        
        return min(100, ocr_risk)
    
    def _calculate_security_risk(self, security: Dict) -> float:
        """Calculate security features risk score."""
        security_risk = 0
        
        if not security.get("watermarkDetected", False):
            security_risk += 30
        
        template_match = security.get("templateMatch", {})
        if not template_match.get("matched", False):
            security_risk += 40
        
        security_patterns = security.get("securityPatterns", {})
        if not security_patterns.get("patterns_detected", False):
            security_risk += 30
        
        return min(100, security_risk)
    
    def calculate_fraud_score(self, analysis_results: Dict) -> Dict:
        """
        Calculate fraud probability based on all analysis results.
        
        Uses ML model if available, otherwise falls back to rule-based scoring.
        """
        # Try ML prediction first if model is loaded
        if self.model_loaded and self.model is not None:
            try:
                return self._ml_prediction(analysis_results)
            except Exception as e:
                print(f"⚠️ ML prediction failed: {str(e)}")
                print("  Falling back to rule-based scoring")
        
        # Fallback to rule-based scoring
        return self._rule_based_scoring(analysis_results)
    
    def _ml_prediction(self, analysis_results: Dict) -> Dict:
        """Use ML model for prediction."""
        # Extract features
        features = self.extract_features(analysis_results)
        
        # Get prediction (convert numpy types to Python native types)
        fraud_probability = float(self.model.predict_proba(features)[0][1])
        fraud_score = float(fraud_probability * 100)
        
        # Get feature importance if available
        score_components = {}
        if hasattr(self.model, 'feature_importances_'):
            feature_names = ['metadata_risk', 'forensic_risk', 'ocr_risk', 'security_risk',
                           'ela_regions', 'irregular_edges', 'template_matched', 'watermark',
                           'exif_count', 'amount_mismatch', 'date_invalid', 'clone_count']
            importance = self.model.feature_importances_
            for name, imp, val in zip(feature_names, importance, features[0]):
                score_components[name] = round(float(val) * float(imp), 2)
        
        # Determine fraud type based on top contributing factors
        fraud_type = self._determine_fraud_type_ml(features[0], score_components)
        
        # Get model confidence (convert to Python float)
        model_confidence = float(self.model_data.get('accuracy', 0.87) * 100) if self.model_data else 87.0
        
        return {
            "fraud_probability": round(fraud_probability, 3),
            "fraud_score": round(fraud_score, 2),
            "fraud_type": fraud_type,
            "score_components": score_components,
            "model_confidence": round(model_confidence, 0),
            "using_ml_model": True,
            "ml_method": self.model_data.get('model_type', 'ML') if self.model_data else 'ML'
        }
    
    def _rule_based_scoring(self, analysis_results: Dict) -> Dict:
        """Fallback rule-based scoring (original method)."""
        score_components = {
            "metadata_risk": 0,
            "forensic_risk": 0,
            "ocr_validation_risk": 0,
            "security_features_risk": 0
        }
        
        # 1. Metadata Analysis Risk (Weight: 15%)
        if "metadata" in analysis_results:
            metadata = analysis_results["metadata"]
            metadata_risk = metadata.get("overall_risk_score", 0)
            score_components["metadata_risk"] = metadata_risk * 0.15
        
        # 2. Forensic Analysis Risk (Weight: 40%)
        if "forensics" in analysis_results:
            forensics = analysis_results["forensics"]
            forensic_risk = forensics.get("overall_risk_score", 0)
            score_components["forensic_risk"] = forensic_risk * 0.40
        
        # 3. OCR Validation Risk (Weight: 25%)
        if "ocr" in analysis_results:
            ocr_risk = self._calculate_ocr_risk(analysis_results["ocr"])
            score_components["ocr_validation_risk"] = ocr_risk * 0.25
        
        # 4. Security Features Risk (Weight: 20%)
        if "security" in analysis_results:
            security_risk = self._calculate_security_risk(analysis_results["security"])
            score_components["security_features_risk"] = security_risk * 0.20
        
        # Calculate total fraud score
        total_score = sum(score_components.values())
        
        # Determine fraud type based on highest risk component
        fraud_type = self._determine_fraud_type_rules(score_components, total_score)
        
        return {
            "fraud_probability": round(total_score / 100, 3),
            "fraud_score": round(total_score, 2),
            "fraud_type": fraud_type,
            "score_components": {k: round(v, 2) for k, v in score_components.items()},
            "model_confidence": 87,  # Confidence in the scoring model
            "using_ml_model": False,
            "ml_method": "Rule-Based"
        }
    
    def _determine_fraud_type_ml(self, features: np.ndarray, score_components: Dict) -> Optional[str]:
        """Determine fraud type from ML features."""
        if len(score_components) == 0:
            return None
        
        # Find top contributing factor
        top_factor = max(score_components.items(), key=lambda x: x[1])[0]
        
        fraud_type_map = {
            'forensic_risk': 'digital_alteration',
            'ela_regions': 'digital_alteration',
            'irregular_edges': 'digital_alteration',
            'ocr_risk': 'field_tampering',
            'amount_mismatch': 'field_tampering',
            'date_invalid': 'field_tampering',
            'security_risk': 'counterfeit_check',
            'template_matched': 'counterfeit_check',
            'metadata_risk': 'metadata_manipulation'
        }
        
        return fraud_type_map.get(top_factor)
    
    def _determine_fraud_type_rules(self, score_components: Dict, total_score: float) -> Optional[str]:
        """Determine fraud type from rule-based scores."""
        if total_score <= 60:
            return None
        
        max_component = max(score_components.items(), key=lambda x: x[1])
        
        fraud_type_map = {
            "forensic_risk": "digital_alteration",
            "ocr_validation_risk": "field_tampering",
            "security_features_risk": "counterfeit_check",
            "metadata_risk": "metadata_manipulation"
        }
        
        return fraud_type_map.get(max_component[0])
    
    def determine_verdict(self, fraud_score: float) -> str:
        """
        Determine overall verdict based on fraud score.
        """
        if fraud_score < 30:
            return "LEGITIMATE"
        elif fraud_score < 60:
            return "SUSPICIOUS"
        else:
            return "FRAUDULENT"
    
    def generate_recommendations(self, analysis_results: Dict, fraud_score: float) -> List[str]:
        """
        Generate actionable recommendations based on analysis results.
        """
        recommendations = []
        
        # High-risk recommendations
        if fraud_score >= 60:
            recommendations.append("⚠️ HIGH RISK: Do not accept this check without thorough verification")
            recommendations.append("Contact issuing bank immediately for verification")
            recommendations.append("Request alternative payment method")
        
        elif fraud_score >= 30:
            recommendations.append("⚠️ MEDIUM RISK: Manual review required before acceptance")
            recommendations.append("Verify check details with issuer")
        
        # Specific recommendations based on findings
        if "forensics" in analysis_results:
            forensics = analysis_results["forensics"]
            
            ela_regions = len(forensics.get("errorLevelAnalysis", {}).get("suspicious_regions", []))
            if ela_regions > 0:
                recommendations.append(
                    f"Digital alteration detected in {ela_regions} region(s) - verify amount and payee fields"
                )
            
            if forensics.get("cloneDetection", {}).get("duplicates_found", False):
                recommendations.append("Cloned regions detected - verify signature authenticity")
        
        if "ocr" in analysis_results:
            ocr = analysis_results["ocr"]
            validation = ocr.get("validationResults", {})
            
            if not validation.get("amountMatch", {}).get("matches", False):
                recommendations.append("Numeric and written amounts may not match - verify amounts")
            
            date_val = validation.get("dateValidation", {})
            if date_val.get("is_future", False):
                recommendations.append("Check date is in the future - verify date with issuer")
            elif date_val.get("is_too_old", False):
                recommendations.append("Check is stale-dated - verify if still valid")
        
        if "security" in analysis_results:
            security = analysis_results["security"]
            
            if not security.get("watermarkDetected", False):
                recommendations.append("No watermark detected - verify check authenticity")
            
            if not security.get("templateMatch", {}).get("matched", False):
                recommendations.append("Check format does not match known templates - verify with issuing bank")
        
        # Always recommend ID verification for suspicious checks
        if fraud_score >= 30:
            recommendations.append("Verify presenter's identification")
            recommendations.append("Consider documenting check with photos")
        
        # If everything looks good
        if fraud_score < 30 and not recommendations:
            recommendations.append("✓ Check appears legitimate based on automated analysis")
            recommendations.append("Standard verification procedures apply")
        
        return recommendations
    
    def predict(self, analysis_results: Dict) -> Dict:
        """
        Execute complete ML prediction and generate final report.
        """
        # Calculate fraud score
        ml_prediction = self.calculate_fraud_score(analysis_results)
        fraud_score = ml_prediction["fraud_score"]
        
        # Determine verdict
        verdict = self.determine_verdict(fraud_score)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(analysis_results, fraud_score)
        
        # Compile final results
        results = {
            "success": True,
            "overallRiskScore": round(fraud_score, 2),
            "verdict": verdict,
            "mlPrediction": ml_prediction,
            "recommendations": recommendations,
            "risk_category": self._get_risk_category(fraud_score),
            "summary": self._generate_summary(verdict, fraud_score, analysis_results)
        }
        
        return results
    
    def _get_risk_category(self, fraud_score: float) -> Dict:
        """Get risk category details."""
        if fraud_score < 30:
            return {
                "level": "LOW",
                "color": "green",
                "description": "Check appears legitimate"
            }
        elif fraud_score < 60:
            return {
                "level": "MEDIUM",
                "color": "yellow",
                "description": "Manual review recommended"
            }
        else:
            return {
                "level": "HIGH",
                "color": "red",
                "description": "High fraud risk detected"
            }
    
    def _generate_summary(self, verdict: str, fraud_score: float, analysis_results: Dict) -> str:
        """Generate human-readable summary."""
        summary_parts = []
        
        summary_parts.append(f"Verdict: {verdict} (Risk Score: {fraud_score:.1f}/100)")
        
        # Add key findings
        findings = []
        
        if "forensics" in analysis_results:
            forensics = analysis_results["forensics"]
            if forensics.get("errorLevelAnalysis", {}).get("suspicious_regions"):
                findings.append("digital alterations detected")
            if forensics.get("cloneDetection", {}).get("duplicates_found"):
                findings.append("cloned regions found")
        
        if "ocr" in analysis_results:
            ocr = analysis_results["ocr"]
            if ocr.get("warnings"):
                findings.append(f"{len(ocr['warnings'])} OCR validation issues")
        
        if "metadata" in analysis_results:
            metadata = analysis_results["metadata"]
            if metadata.get("flags"):
                findings.append("metadata inconsistencies")
        
        if findings:
            summary_parts.append(f"Key findings: {', '.join(findings)}")
        else:
            summary_parts.append("No significant anomalies detected")
        
        return " | ".join(summary_parts)

