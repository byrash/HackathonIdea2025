from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import json
import os
from pathlib import Path
import asyncio
from datetime import datetime
from typing import Optional

app = FastAPI(title="Fraudulent Bank Check Detection API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost:5173"],  # Angular/Vite dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("./uploads")
JOBS_DIR = Path("./jobs")
TEMPLATES_DIR = Path("./templates")
MODELS_DIR = Path("./ml-models")

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
JOBS_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


@app.get("/")
async def root():
    return {
        "service": "Fraudulent Bank Check Detection API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.post("/api/checks/upload")
async def upload_check(file: UploadFile = File(...)):
    """
    Upload a check image for fraud analysis.
    Accepts JPEG, PNG, and PDF files.
    """
    # Validate file type
    allowed_extensions = [".jpg", ".jpeg", ".png", ".pdf"]
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{job_id}_original{file_ext}"
    
    # Save uploaded file
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Create initial job status file
    job_data = {
        "jobId": job_id,
        "status": "QUEUED",
        "uploadTimestamp": datetime.utcnow().isoformat(),
        "currentStage": 0,
        "currentPercentage": 0,
        "message": "Check uploaded successfully, analysis queued"
    }
    
    job_file = JOBS_DIR / f"{job_id}.json"
    with open(job_file, "w") as f:
        json.dump(job_data, f, indent=2)
    
    # Start background processing
    asyncio.create_task(process_check(job_id, str(file_path)))
    
    return {
        "jobId": job_id,
        "status": "QUEUED",
        "message": "Check uploaded successfully"
    }


@app.get("/api/checks/{job_id}/progress")
async def get_progress(job_id: str):
    """
    Server-Sent Events (SSE) endpoint for real-time progress updates.
    Streams progress data until analysis is complete or fails.
    """
    async def event_stream():
        job_file = JOBS_DIR / f"{job_id}.json"
        
        if not job_file.exists():
            yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
            return
        
        previous_data = None
        
        # Poll for updates
        for _ in range(600):  # Max 5 minutes (600 * 0.5s)
            if job_file.exists():
                try:
                    with open(job_file, "r") as f:
                        job_data = json.load(f)
                    
                    # Only send if data changed
                    if job_data != previous_data:
                        yield f"data: {json.dumps(job_data)}\n\n"
                        previous_data = job_data
                    
                    # Stop if completed or failed
                    if job_data.get("status") in ["COMPLETED", "FAILED"]:
                        break
                        
                except json.JSONDecodeError:
                    pass  # File being written, skip this iteration
            
            await asyncio.sleep(0.5)  # Poll every 500ms
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/api/checks/{job_id}/results")
async def get_results(job_id: str):
    """
    Retrieve complete fraud analysis results for a job.
    """
    job_file = JOBS_DIR / f"{job_id}.json"
    
    if not job_file.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    
    try:
        with open(job_file, "r") as f:
            job_data = json.load(f)
        
        if job_data.get("status") != "COMPLETED":
            return {
                "jobId": job_id,
                "status": job_data.get("status"),
                "message": "Analysis not yet completed"
            }
        
        return job_data
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to read job data")


@app.get("/api/checks/{job_id}/image/{image_type}")
async def get_image(job_id: str, image_type: str):
    """
    Download original or annotated check image.
    image_type: 'original' or 'annotated'
    """
    if image_type not in ["original", "annotated"]:
        raise HTTPException(status_code=400, detail="Invalid image type. Use 'original' or 'annotated'")
    
    # Find the image file (check multiple extensions)
    for ext in [".jpg", ".jpeg", ".png", ".pdf"]:
        image_path = UPLOAD_DIR / f"{job_id}_{image_type}{ext}"
        if image_path.exists():
            return FileResponse(image_path)
    
    raise HTTPException(status_code=404, detail=f"Image not found for job {job_id}")


async def process_check(job_id: str, file_path: str):
    """
    Background task to process check image through fraud detection pipeline.
    """
    try:
        from services.fraud_detector import FraudDetector
        
        detector = FraudDetector(job_id, file_path)
        await detector.analyze()
        
    except Exception as e:
        # Update job status to FAILED
        job_file = JOBS_DIR / f"{job_id}.json"
        error_data = {
            "jobId": job_id,
            "status": "FAILED",
            "error": str(e),
            "failedAt": datetime.utcnow().isoformat()
        }
        
        with open(job_file, "w") as f:
            json.dump(error_data, f, indent=2)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

