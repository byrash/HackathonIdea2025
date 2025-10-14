# 🔍 Fraudulent Bank Check Detection System

AI-powered system to detect fraudulent and tampered bank checks using advanced image forensics, OCR, and machine learning.

## 🎯 Project Overview

This hackathon project provides financial institutions with an automated check fraud detection system that analyzes uploaded images for signs of tampering, forgery, and manipulation.

### Key Features

✅ **Real-time Progress Updates** via Server-Sent Events (SSE)  
✅ **8-Stage Analysis Pipeline** with detailed forensics  
✅ **Industry-Standard Techniques**: ELA, Clone Detection, Metadata Analysis  
✅ **Modern UI** with drag-and-drop upload  
✅ **No Database Required** - file-based storage for quick setup  
✅ **ML-Based Fraud Scoring** with detailed recommendations

## 🏗️ Architecture

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

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+** (for backend)
- **Node.js 18+** (for frontend)
- **Tesseract OCR** (for text extraction)

### Backend Setup

```bash
# Install Tesseract OCR (macOS)
brew install tesseract

# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run server
python main.py
```

Backend will be available at: `http://localhost:8000`

### Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Run development server
npm run serve
```

Frontend will be available at: `http://localhost:4200`

## 📊 Analysis Pipeline (8 Stages)

### Stage 1: Image Validation (0-10%)

- File format verification
- Image quality assessment
- Resolution check

### Stage 2: Preprocessing (10-20%)

- Image enhancement (brightness, contrast)
- Noise reduction
- Orientation correction & deskewing

### Stage 3: Metadata Analysis (20-30%)

- **EXIF data extraction**
- **Creation date/time verification**
- **Camera/scanner model validation**
- **Software modification history detection**

### Stage 4: OCR & Text Extraction (30-50%)

- MICR line parsing
- Extract: payee, amount, date, check number
- Routing & account numbers

### Stage 5: Cross-Field Validation (50-60%)

- Verify numeric vs written amounts
- Date format validation
- MICR line format & routing number checksum

### Stage 6: Forensic Analysis (60-75%)

#### **A. Error Level Analysis (ELA)**

- Detect areas with different compression levels
- Identify digitally altered regions
- Highlight potential copy-paste manipulations

#### **B. Clone Detection**

- Identify duplicated regions within the image
- Detect copied signatures or amounts
- Pattern matching for repeated elements

#### **C. Edge Detection**

- Find irregular boundaries
- Detect white-out or correction fluid
- Spot cut-and-paste boundaries

### Stage 7: Security Features Check (75-90%)

- Watermark detection
- Template matching against known legitimate checks
- Security pattern verification
- Microprinting analysis

### Stage 8: ML-Based Fraud Detection (90-100%)

- Anomaly scoring using ML model
- Fraud probability calculation
- Risk score aggregation
- Generate detailed report with recommendations

## 🔍 Fraud Detection Techniques

### 1. Image Forensics

- **Metadata Examination**: EXIF inconsistencies, software modifications
- **Error Level Analysis (ELA)**: Digital alteration detection
- **Clone Detection**: Duplicated regions identification

### 2. OCR & Validation

- Text extraction from all check fields
- Cross-field validation (amounts, dates)
- MICR line validation with checksums

### 3. Security Features

- Watermark detection
- Template matching
- Security pattern analysis

### 4. Machine Learning

- Anomaly detection
- Fraud type classification
- Risk scoring

## 📄 API Endpoints

- `POST /api/checks/upload` - Upload check image
- `GET /api/checks/{jobId}/progress` - SSE stream for real-time progress
- `GET /api/checks/{jobId}/results` - Get complete analysis results
- `GET /api/checks/{jobId}/image/{type}` - Download original or annotated image

## 🖼️ Sample Output

**Fraud Risk Score:** 0-100  
**Verdict:** LEGITIMATE | SUSPICIOUS | FRAUDULENT  
**Extracted Data:** Check number, date, payee, amounts, routing numbers  
**Forensic Findings:** ELA regions, cloned areas, metadata flags  
**Recommendations:** Actionable next steps for verification

## 📁 Project Structure

```
HackathonIdea/
├── backend/                    # Python FastAPI
│   ├── main.py
│   ├── services/
│   │   ├── fraud_detector.py
│   │   ├── image_processor.py
│   │   ├── metadata_analyzer.py
│   │   ├── ocr_service.py
│   │   ├── forensic_analyzer.py
│   │   ├── template_matcher.py
│   │   └── ml_predictor.py
│   ├── uploads/
│   ├── jobs/
│   ├── templates/
│   └── requirements.txt
│
├── frontend/                   # Angular + Vite
│   ├── src/
│   │   ├── app/
│   │   │   ├── components/
│   │   │   └── services/
│   │   └── main.ts
│   └── package.json
│
└── README.md
```

## 🛠️ Technology Stack

### Backend

- **Framework:** FastAPI
- **Image Processing:** OpenCV, Pillow
- **OCR:** Tesseract, EasyOCR
- **ML:** scikit-learn
- **Language:** Python 3.9+

### Frontend

- **Framework:** Angular 18
- **Build Tool:** Vite
- **Styling:** Tailwind CSS
- **Language:** TypeScript 5+

## 🔐 Security Considerations

- Sensitive data masking (account numbers)
- File type and size validation
- Auto-cleanup of uploaded files
- No persistent storage of financial data

## 📝 Development Notes

This is a **hackathon/educational project**. For production use in financial institutions:

- Additional regulatory compliance required
- Professional security audits needed
- Enhanced ML models recommended
- Database integration for audit trails

## 👥 Author

**Project Created By:** Me
**Date:** October 2025  
**Purpose:** Hackathon Demonstration

## 📄 License

MIT License - Educational purposes

## 🙏 Acknowledgments

- Tesseract OCR (open-source)
- OpenCV community
- FastAPI framework
- Angular + Vite teams

---

**Note:** This system demonstrates AI-powered fraud detection capabilities. Always combine automated analysis with manual verification for critical financial decisions.
