# Fraudulent Bank Check Detection System

## 🎯 Project Overview

This is a **hackathon project** designed to detect fraudulent and tampered bank checks using industry-standard image analysis techniques and AI-powered verification methods.

### Purpose
To provide financial institutions and users with an automated system that can identify potentially fraudulent bank checks by analyzing uploaded images for signs of tampering, forgery, and manipulation.

---

## 🏗️ Architecture

### High-Level Components (Simplified for Hackathon)

```
┌─────────────────────┐          ┌────────────────────────────────┐
│   Web Frontend      │   HTTP   │     Python Backend API         │
│  (Angular + Vite)   │◄────────►│  - Image Processing            │
│                     │   SSE    │  - OCR & Text Extraction       │
│  - Upload UI        │          │  - Fraud Detection Algorithms  │
│  - Progress Display │          │  - ML Models                   │
│  - Results View     │          │  - File-based Job Storage      │
└─────────────────────┘          └────────────────────────────────┘
                                            │
                                            ▼
                                 ┌──────────────────────┐
                                 │  Local File System   │
                                 │  - Uploaded Images   │
                                 │  - Job Status (JSON) │
                                 │  - Analysis Results  │
                                 └──────────────────────┘
```

**Key Simplifications:**
- ✅ Single Python backend (no microservices)
- ✅ File-based storage (no database required)
- ✅ Local file system for images
- ✅ JSON files for job tracking
- ✅ Server-Sent Events (SSE) for real-time progress
- ✅ Fast setup for hackathon timeline

---

## 📋 System Components

### 1. Frontend - Web Application

**Technology Stack:**
- **Framework:** Angular (Latest Version)
- **Build Tool:** Vite
- **Language:** TypeScript
- **Styling:** Tailwind CSS (recommended for rapid development)

**Features:**
- Drag-and-drop check image upload interface
- **Real-time progress updates** via Server-Sent Events (SSE)
- Live progress bar showing current stage and percentage
- Stage-by-stage visual feedback with status indicators
- Results dashboard displaying:
  - Fraud risk score (0-100%)
  - Detected anomalies with highlighted areas
  - Confidence levels for each test
  - Detailed analysis report
- Responsive design for desktop and mobile

**User Flow:**
1. Upload check image (JPEG, PNG, PDF)
2. **Watch live progress updates for each analysis stage**
3. See real-time status messages (e.g., "Extracting text...", "Analyzing signature...")
4. Review comprehensive analysis results
5. Download detailed fraud report

**Progress Display Requirements:**
- Progress bar (0-100%)
- Current stage name
- Stage description
- Status icon (⏳ processing, ✅ complete, ⚠️ warning)
- Estimated time remaining (optional)

---

### 2. Backend API (Python)

**Technology Stack:**
- **Framework:** FastAPI (async support for SSE)
- **Language:** Python 3.9+
- **Image Processing:** OpenCV, Pillow
- **OCR:** Tesseract OCR, EasyOCR
- **ML:** scikit-learn, TensorFlow/PyTorch (optional)

**Responsibilities:**
- Handle check image uploads
- **Save uploaded images to local file system** (`./uploads/` directory)
- **Store job status in JSON files** (`./jobs/` directory)
- Process and parse check images
- Execute fraud detection algorithms sequentially
- **Send incremental progress updates to frontend via SSE**
- Generate and save analysis results
- Clean up old files periodically

**File System Structure:**
```
backend/
├── uploads/                    # Uploaded check images
│   ├── {jobId}_original.jpg
│   └── {jobId}_annotated.jpg   # With fraud highlights
├── jobs/                       # Job status tracking
│   └── {jobId}.json            # Contains progress & results
├── templates/                  # Legitimate check templates
│   ├── chase_template.jpg
│   └── bofa_template.jpg
└── ml-models/                  # Pre-trained models (optional)
    └── fraud_classifier.pkl
```

**API Endpoints:**

#### `POST /api/checks/upload`
Upload a check image for analysis
- **Request:** Multipart form data with image file
- **Response:** 
```json
{
  "jobId": "uuid-string",
  "status": "QUEUED",
  "message": "Check uploaded successfully"
}
```

#### `GET /api/checks/{jobId}/progress`
**Server-Sent Events (SSE) endpoint** for real-time progress
- **Response Stream:** Continuous event stream with progress updates
```
event: progress
data: {"stage": 1, "percentage": 10, "status": "Image validation", "message": "Checking file format..."}

event: progress
data: {"stage": 2, "percentage": 20, "status": "Preprocessing", "message": "Enhancing image quality..."}

event: progress
data: {"stage": 4, "percentage": 50, "status": "OCR", "message": "Extracting text from check..."}

event: complete
data: {"jobId": "uuid", "status": "COMPLETED", "percentage": 100}
```

#### `GET /api/checks/{jobId}/results`
Retrieve complete fraud analysis results
- **Response:** Detailed fraud detection report (see format below)

#### `GET /api/checks/{jobId}/image/{type}`
Download original or annotated check image
- **Parameters:** `type` = `original` or `annotated`
- **Response:** Image file

---

## 🔄 Incremental Progress Updates

### Progress Event Format (SSE)

```typescript
{
  "jobId": "string",
  "stage": 1-8,
  "stageName": "string",
  "percentage": 0-100,
  "status": "PROCESSING" | "COMPLETED" | "WARNING" | "ERROR",
  "message": "string",  // User-friendly description
  "timestamp": "ISO 8601 string",
  "details": {          // Optional technical details
    "itemsProcessed": number,
    "warnings": ["string"]
  }
}
```

### Example Progress Sequence

```json
// Stage 1
{"stage": 1, "percentage": 5, "status": "PROCESSING", "message": "Validating image format..."}
{"stage": 1, "percentage": 10, "status": "COMPLETED", "message": "✓ Image validation passed"}

// Stage 2
{"stage": 2, "percentage": 15, "status": "PROCESSING", "message": "Enhancing image..."}
{"stage": 2, "percentage": 20, "status": "COMPLETED", "message": "✓ Image preprocessed"}

// Stage 3
{"stage": 3, "percentage": 25, "status": "PROCESSING", "message": "Extracting EXIF metadata..."}
{"stage": 3, "percentage": 30, "status": "COMPLETED", "message": "✓ Metadata analyzed"}

// Stage 4
{"stage": 4, "percentage": 40, "status": "PROCESSING", "message": "Running OCR on check..."}
{"stage": 4, "percentage": 50, "status": "COMPLETED", "message": "✓ Text extracted successfully"}

// ... and so on for all 8 stages
```

---

## 📊 Analysis Pipeline Stages (with Progress Updates)

### Stage 1: Image Upload & Validation (0-10%)
**What happens:**
- File format verification (JPEG, PNG, PDF)
- Image quality assessment
- Resolution check (minimum 300 DPI recommended)
- Save to `./uploads/{jobId}_original.{ext}`

**Progress Updates:**
- 5%: "Validating file format..."
- 10%: "✓ Image validation complete"

---

### Stage 2: Preprocessing (10-20%)
**What happens:**
- Image enhancement (brightness, contrast)
- Noise reduction
- Orientation correction
- Deskewing

**Progress Updates:**
- 15%: "Enhancing image quality..."
- 20%: "✓ Image preprocessed"

---

### Stage 3: Metadata Analysis (20-30%)
**What happens:**
- EXIF data extraction
- Creation date/time verification
- Camera/scanner model validation
- Software modification history detection

**Progress Updates:**
- 25%: "Extracting EXIF metadata..."
- 28%: "Checking for manipulation history..."
- 30%: "✓ Metadata analysis complete"

---

### Stage 4: OCR & Text Extraction (30-50%)
**What happens:**
- MICR line parsing
- Extract: payee, amount (numeric & written), date, check number, routing number, account number
- Text confidence scoring

**Progress Updates:**
- 35%: "Running OCR engine..."
- 40%: "Extracting MICR line..."
- 45%: "Reading payee and amount fields..."
- 50%: "✓ Text extraction complete"

---

### Stage 5: Cross-Field Validation (50-60%)
**What happens:**
- Verify numeric amount matches written amount
- Check date format and validity
- Validate MICR line format
- Verify routing number checksum (ABA format)

**Progress Updates:**
- 55%: "Validating extracted data..."
- 58%: "Cross-checking amount fields..."
- 60%: "✓ Field validation complete"

---

### Stage 6: Forensic Analysis (60-75%)
**What happens:**
- Error Level Analysis (ELA) for digital alterations
- Clone detection for duplicated regions
- Edge detection for irregular boundaries
- Color and texture analysis

**Progress Updates:**
- 62%: "Running error level analysis..."
- 67%: "Detecting cloned regions..."
- 72%: "Analyzing edges and boundaries..."
- 75%: "✓ Forensic analysis complete"

---

### Stage 7: Security Features Check (75-90%)
**What happens:**
- Watermark detection
- Template matching against known legitimate checks
- Security pattern verification
- Microprinting analysis (if detectable)

**Progress Updates:**
- 78%: "Checking for watermarks..."
- 82%: "Matching against bank templates..."
- 87%: "Verifying security features..."
- 90%: "✓ Security check complete"

---

### Stage 8: ML-Based Fraud Detection & Report Generation (90-100%)
**What happens:**
- Anomaly scoring using ML model (if available)
- Fraud probability calculation
- Risk score aggregation
- Generate detailed report
- Save annotated image with highlighted suspicious areas
- Save results to `./jobs/{jobId}.json`

**Progress Updates:**
- 92%: "Running ML fraud detection model..."
- 95%: "Calculating risk score..."
- 98%: "Generating analysis report..."
- 100%: "✓ Analysis complete!"

---

## 🔍 Fraud Detection Techniques (Industry Standard)

### 1. **Image Forensics Analysis**

#### A. **Metadata Examination**
- EXIF data analysis for inconsistencies
- Creation date/time verification
- Camera/scanner model validation
- Software modification history detection

#### B. **Error Level Analysis (ELA)**
- Detect areas with different compression levels
- Identify digitally altered regions
- Highlight potential copy-paste manipulations

#### C. **Clone Detection**
- Identify duplicated regions within the image
- Detect copied signatures or amounts
- Pattern matching for repeated elements

### 2. **Optical Character Recognition (OCR)**

#### A. **Text Extraction**
- Extract all text fields from check:
  - Payee name
  - Amount (numeric and written)
  - Date
  - Check number
  - Routing number
  - Account number

#### B. **Cross-Field Validation**
- Verify numeric amount matches written amount
- Check date format and validity
- Validate MICR line format
- Verify routing number checksum

### 3. **Document Structure Analysis**

#### A. **Template Matching**
- Compare against known legitimate check templates
- Verify security features positioning
- Validate bank logo and branding

#### B. **Security Features Detection**
- Watermark detection
- Background pattern analysis
- Security thread identification

### 4. **Signature Verification**
- Detect traced or printed signatures
- Verify signature positioning and size
- Analyze stroke consistency

### 5. **Machine Learning Models** (Optional)

#### A. **Anomaly Detection**
- Identify statistical outliers
- Pattern recognition for common fraud types

#### B. **Classification Models**
- Binary classification: Legitimate vs. Fraudulent
- Fraud type identification:
  - Altered payee
  - Amount tampering
  - Forged signature
  - Counterfeit check

### 6. **MICR Line Analysis**
- Magnetic Ink Character Recognition validation
- Routing number validation (ABA format)
- Account number format checking
- Check digit verification algorithms

### 7. **Color and Texture Analysis**
- Ink consistency analysis
- Paper texture evaluation
- Color spectrum analysis for alterations

### 8. **Edge Detection and Boundary Analysis**
- Detect irregular edges around text
- Identify white-out or correction fluid usage
- Spot cut-and-paste boundaries

---

## 🛠️ Technology Stack

### Frontend
- **Framework:** Angular 18+
- **Build Tool:** Vite
- **Language:** TypeScript 5+
- **UI Library:** Tailwind CSS + Angular Material
- **HTTP:** Angular HttpClient
- **SSE:** EventSource API (native browser)
- **File Upload:** ngx-dropzone

### Backend
- **Framework:** FastAPI
- **Language:** Python 3.9+
- **Image Processing:** 
  - OpenCV (`opencv-python`)
  - Pillow (`PIL`)
  - NumPy
- **OCR:**
  - Tesseract OCR (`pytesseract`)
  - EasyOCR (alternative/backup)
- **ML (Optional):**
  - scikit-learn (for simple models)
  - TensorFlow/PyTorch (for deep learning)
- **Utilities:**
  - `python-multipart` (file uploads)
  - `python-magic` (file type detection)
  - `exifread` (EXIF extraction)

### Storage
- **File System:** Local disk storage
  - Uploaded images: `./uploads/`
  - Job status: `./jobs/` (JSON files)
  - Templates: `./templates/`
  - ML models: `./ml-models/`
- **No Database Required** ✅

---

## 🚀 Getting Started

### Prerequisites
- Node.js 18+ (for frontend)
- Python 3.9+ (for backend)
- Tesseract OCR installed on system

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn python-multipart
pip install opencv-python pillow numpy
pip install pytesseract easyocr
pip install python-magic exifread
pip install scikit-learn  # Optional for ML

# Create required directories
mkdir -p uploads jobs templates ml-models

# Run API server
uvicorn main:app --reload --port 8000
```

**Backend File: `main.py`** (Starter template)

```python
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import json
import os
from pathlib import Path
import asyncio
from datetime import datetime

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Angular dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("./uploads")
JOBS_DIR = Path("./jobs")
UPLOAD_DIR.mkdir(exist_ok=True)
JOBS_DIR.mkdir(exist_ok=True)

@app.post("/api/checks/upload")
async def upload_check(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    file_ext = file.filename.split(".")[-1]
    file_path = UPLOAD_DIR / f"{job_id}_original.{file_ext}"
    
    # Save uploaded file
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Create job status file
    job_data = {
        "jobId": job_id,
        "status": "QUEUED",
        "uploadTimestamp": datetime.utcnow().isoformat(),
        "currentStage": 0,
        "currentPercentage": 0
    }
    
    with open(JOBS_DIR / f"{job_id}.json", "w") as f:
        json.dump(job_data, f)
    
    # Start background processing
    asyncio.create_task(process_check(job_id, str(file_path)))
    
    return {"jobId": job_id, "status": "QUEUED"}

@app.get("/api/checks/{job_id}/progress")
async def get_progress(job_id: str):
    async def event_stream():
        job_file = JOBS_DIR / f"{job_id}.json"
        
        while True:
            if job_file.exists():
                with open(job_file, "r") as f:
                    job_data = json.load(f)
                
                event = f"data: {json.dumps(job_data)}\n\n"
                yield event
                
                if job_data["status"] in ["COMPLETED", "FAILED"]:
                    break
            
            await asyncio.sleep(0.5)  # Poll every 500ms
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/api/checks/{job_id}/results")
async def get_results(job_id: str):
    job_file = JOBS_DIR / f"{job_id}.json"
    
    if job_file.exists():
        with open(job_file, "r") as f:
            return json.load(f)
    
    return {"error": "Job not found"}

async def process_check(job_id: str, file_path: str):
    # Import fraud detection modules
    from services.fraud_detector import FraudDetector
    
    detector = FraudDetector(job_id, file_path)
    await detector.analyze()  # This runs all 8 stages with progress updates
```

---

### Frontend Setup

```bash
# Create Angular project with Vite
npm create vite@latest check-fraud-detector-ui -- --template angular

cd check-fraud-detector-ui

# Install dependencies
npm install
npm install @angular/material @angular/cdk
npm install ngx-dropzone

# Run development server
npm run dev
```

**Frontend Service: `api.service.ts`**

```typescript
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private apiUrl = 'http://localhost:8000/api';

  constructor(private http: HttpClient) {}

  uploadCheck(file: File): Observable<any> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post(`${this.apiUrl}/checks/upload`, formData);
  }

  listenToProgress(jobId: string): EventSource {
    return new EventSource(`${this.apiUrl}/checks/${jobId}/progress`);
  }

  getResults(jobId: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/checks/${jobId}/results`);
  }
}
```

**Component: `upload.component.ts`**

```typescript
import { Component } from '@angular/core';
import { ApiService } from '../services/api.service';

@Component({
  selector: 'app-upload',
  templateUrl: './upload.component.html'
})
export class UploadComponent {
  progress = 0;
  currentStage = '';
  statusMessage = '';
  isProcessing = false;

  constructor(private apiService: ApiService) {}

  onFileSelected(file: File) {
    this.isProcessing = true;
    
    this.apiService.uploadCheck(file).subscribe(response => {
      const eventSource = this.apiService.listenToProgress(response.jobId);
      
      eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        this.progress = data.currentPercentage;
        this.currentStage = `Stage ${data.currentStage}`;
        this.statusMessage = data.message || '';
        
        if (data.status === 'COMPLETED') {
          eventSource.close();
          this.isProcessing = false;
          this.loadResults(response.jobId);
        }
      };
      
      eventSource.onerror = (error) => {
        console.error('SSE Error:', error);
        eventSource.close();
        this.isProcessing = false;
      };
    });
  }
  
  loadResults(jobId: string) {
    this.apiService.getResults(jobId).subscribe(results => {
      console.log('Analysis Results:', results);
      // Navigate to results page or display results
    });
  }
}
```

---

## 📄 API Response Format

### Job Status File (`./jobs/{jobId}.json`)

```json
{
  "jobId": "uuid-string",
  "status": "QUEUED" | "PROCESSING" | "COMPLETED" | "FAILED",
  "uploadTimestamp": "2025-10-14T11:30:00Z",
  "analysisCompletedAt": "2025-10-14T11:32:15Z",
  "currentStage": 8,
  "currentPercentage": 100,
  
  "imageFiles": {
    "original": "./uploads/uuid_original.jpg",
    "annotated": "./uploads/uuid_annotated.jpg"
  },
  
  "overallRiskScore": 0-100,
  "verdict": "LEGITIMATE" | "SUSPICIOUS" | "FRAUDULENT",
  
  "extractedData": {
    "checkNumber": "1234",
    "date": "10/14/2025",
    "payee": "John Doe",
    "amountNumeric": 500.00,
    "amountWritten": "Five Hundred and 00/100",
    "routingNumber": "123456789",
    "accountNumber": "****5678",
    "bankName": "Chase Bank"
  },
  
  "validationResults": {
    "amountMatch": {
      "passed": true,
      "confidence": 95,
      "details": "Numeric and written amounts match"
    },
    "dateValid": {
      "passed": true,
      "confidence": 100,
      "details": "Date format is valid"
    },
    "micrValid": {
      "passed": true,
      "confidence": 88,
      "details": "MICR line format correct, checksum valid"
    }
  },
  
  "forensicAnalysis": {
    "errorLevelAnalysis": {
      "suspiciousRegions": [
        {
          "area": "amount",
          "coordinates": {"x": 450, "y": 120, "width": 80, "height": 30},
          "severity": "MEDIUM",
          "reason": "Inconsistent compression level detected"
        }
      ]
    },
    "cloneDetection": {
      "duplicatesFound": false,
      "regions": []
    },
    "metadataAnalysis": {
      "inconsistenciesFound": false,
      "flags": [],
      "exifData": {
        "software": "Adobe Photoshop",
        "dateTime": "2025:10:14 10:30:00"
      }
    }
  },
  
  "securityFeatures": {
    "watermarkDetected": true,
    "microPrintingDetected": false,
    "templateMatch": {
      "matched": true,
      "confidence": 92,
      "bankTemplate": "chase_template"
    }
  },
  
  "mlPrediction": {
    "fraudProbability": 0.23,
    "fraudType": null,
    "modelConfidence": 87
  },
  
  "recommendations": [
    "Manual review recommended due to MEDIUM severity finding in amount field",
    "Verify with issuing bank"
  ],
  
  "stageHistory": [
    {"stage": 1, "completedAt": "2025-10-14T11:30:05Z", "status": "COMPLETED"},
    {"stage": 2, "completedAt": "2025-10-14T11:30:12Z", "status": "COMPLETED"}
  ]
}
```

---

## 🧪 Testing

### Test Cases
1. **Legitimate Check** - Should pass all validations
2. **Altered Amount** - Should detect tampering in amount field
3. **Poor Quality Image** - Should request better quality
4. **Forged Signature** - Should flag signature inconsistencies

### Sample Test Data
Create `/test-data` directory with sample checks:
- `legitimate_check_001.jpg`
- `altered_amount_002.jpg`
- `poor_quality_003.jpg`

---

## 📁 Project Structure

```
check-fraud-detector/
│
├── frontend/                           # Angular + Vite web app
│   ├── src/
│   │   ├── app/
│   │   │   ├── components/
│   │   │   │   ├── upload/            # Upload UI
│   │   │   │   ├── progress/          # Progress display
│   │   │   │   └── results/           # Results view
│   │   │   └── services/
│   │   │       └── api.service.ts     # API + SSE handling
│   │   └── assets/
│   ├── package.json
│   └── vite.config.ts
│
├── backend/                            # Python FastAPI
│   ├── main.py                         # Main API file
│   ├── services/
│   │   ├── fraud_detector.py          # Main fraud detection logic
│   │   ├── image_processor.py         # Image preprocessing
│   │   ├── ocr_service.py             # Text extraction
│   │   ├── forensic_analyzer.py       # Forensic analysis
│   │   ├── template_matcher.py        # Template matching
│   │   └── ml_predictor.py            # ML fraud prediction (optional)
│   │
│   ├── uploads/                        # Uploaded check images
│   │   ├── {jobId}_original.jpg
│   │   └── {jobId}_annotated.jpg
│   │
│   ├── jobs/                           # Job status (JSON files)
│   │   └── {jobId}.json
│   │
│   ├── templates/                      # Legitimate check templates
│   │   ├── chase_template.jpg
│   │   └── bofa_template.jpg
│   │
│   ├── ml-models/                      # Pre-trained models (optional)
│   │   └── fraud_classifier.pkl
│   │
│   ├── requirements.txt
│   └── README.md
│
├── test-data/                          # Sample check images
│   ├── legitimate_check_001.jpg
│   ├── altered_amount_002.jpg
│   └── poor_quality_003.jpg
│
└── README.md                           # This file
``` 

---

## 🔐 Security Considerations

1. **Data Privacy**
   - Mask sensitive information in responses (account numbers, routing numbers)
   - Auto-delete uploaded images after 24 hours
   - No persistent storage of financial data

2. **API Security**
   - Rate limiting (e.g., 10 uploads per hour per IP)
   - File size limits (max 10MB)
   - Allowed file types validation
   - Input sanitization

3. **File System Security**
   - Validate file types before saving
   - Use UUIDs for filenames (prevent path traversal)
   - Set proper file permissions

---

## 📈 Future Enhancements

- [ ] Batch processing for multiple checks
- [ ] Historical fraud pattern database
- [ ] Advanced deep learning models
- [ ] Mobile-responsive UI improvements
- [ ] Export reports as PDF
- [ ] Admin dashboard for statistics

---

## 👥 Team / Author

**Project Created By:** Me 
**Hackathon:** TBD  
**Date:** October 2025

---

## 📝 License

MIT License - feel free to use this project for educational purposes.

---

## 🙏 Acknowledgments

- Tesseract OCR (open-source)
- OpenCV community
- FastAPI framework
- Angular + Vite teams

---

**Note:** This is a hackathon/educational project. For production use in financial institutions, additional regulatory compliance, security audits, and professional fraud detection expertise are required.