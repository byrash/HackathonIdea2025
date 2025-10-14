# Fraudulent Bank Check Detection - Backend

Python FastAPI backend for fraud detection analysis.

## Prerequisites

- Python 3.9 or higher
- Tesseract OCR installed on your system

### Install Tesseract OCR

**macOS:**

```bash
brew install tesseract
```

**Ubuntu/Debian:**

```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Download installer from: https://github.com/UB-Mannheim/tesseract/wiki

## Setup

1. **Create virtual environment:**

```bash
cd backend
python -m venv venv
```

2. **Activate virtual environment:**

```bash
# macOS/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Verify directories exist:**
   The following directories should already exist:

- `uploads/` - For uploaded check images
- `jobs/` - For job status tracking
- `templates/` - For bank check templates (optional)
- `ml-models/` - For ML models (optional)

## Running the Server

Start the FastAPI server:

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at: `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

## API Endpoints

- `POST /api/checks/upload` - Upload check image
- `GET /api/checks/{jobId}/progress` - SSE stream for progress
- `GET /api/checks/{jobId}/results` - Get analysis results
- `GET /api/checks/{jobId}/image/{type}` - Download images

## Project Structure

```
backend/
├── main.py                      # FastAPI application
├── services/
│   ├── fraud_detector.py        # Main orchestrator
│   ├── image_processor.py       # Image preprocessing
│   ├── metadata_analyzer.py     # EXIF analysis
│   ├── ocr_service.py           # Text extraction
│   ├── forensic_analyzer.py     # ELA, clone detection
│   ├── template_matcher.py      # Security features
│   └── ml_predictor.py          # ML fraud scoring
├── uploads/                     # Uploaded images
├── jobs/                        # Job status files
├── templates/                   # Bank templates
└── ml-models/                   # ML models
```

## Analysis Pipeline

1. **Stage 1:** Image Validation (0-10%)
2. **Stage 2:** Preprocessing (10-20%)
3. **Stage 3:** Metadata Analysis (20-30%)
4. **Stage 4:** OCR & Text Extraction (30-50%)
5. **Stage 5:** Cross-Field Validation (50-60%)
6. **Stage 6:** Forensic Analysis (60-75%)
   - Error Level Analysis (ELA)
   - Clone Detection
   - Edge Detection
7. **Stage 7:** Security Features (75-90%)
8. **Stage 8:** ML Fraud Prediction (90-100%)

## Troubleshooting

**Tesseract not found:**
Ensure Tesseract is installed and in your PATH.

**OpenCV errors:**
Some systems may need additional libraries:

```bash
sudo apt-get install libgl1-mesa-glx
```

**Permission errors:**
Ensure write permissions for `uploads/` and `jobs/` directories.
