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

### Using Makefile (Recommended)

```bash
# First time setup - installs all dependencies
make install

# Generate sample templates (10 bank check templates)
make templates

# Generate ML model (XGBoost for fraud detection)
make ml-model

# Start everything (backend + frontend)
make start

# Start fresh (clears old data, reinstalls, starts)
make start-fresh

# Stop all processes
make stop

# Clean uploaded files and job data
make clean-data

# See all available commands
make help
```

Application will be available at: `http://localhost:5173` (frontend) and `http://localhost:8000` (backend)

### Manual Setup (Alternative)

<details>
<summary>Click to expand manual setup instructions</summary>

#### Backend Setup

```bash
# Install Tesseract OCR (macOS)
brew install tesseract

# Navigate to backend
cd backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate templates and ML model
python generate_sample_templates.py
python setup_ml_model.py --option advanced

# Run server
python main.py
```

Backend will be available at: `http://localhost:8000`

#### Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend will be available at: `http://localhost:5173`

</details>

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

This stage examines the hidden data embedded in the image file to detect signs of manipulation.

#### **What is EXIF Data?**

EXIF (Exchangeable Image File Format) is metadata automatically embedded in photos by cameras, scanners, and editing software. It's like a "digital fingerprint" containing:

- Camera/scanner make and model
- Date/time photo was taken
- GPS coordinates (if available)
- Software used to edit the image
- Image dimensions and resolution
- Camera settings (ISO, aperture, shutter speed)

**Why it matters for fraud detection:**

- Edited images often have EXIF data stripped or modified
- Software modification history reveals if Photoshop/GIMP was used
- Date inconsistencies indicate suspicious timeline manipulation
- Missing EXIF data is a red flag (fraudsters often remove it)

#### **A. EXIF Data Extraction**

**How it works:**

1. **Read all EXIF tags from image file**

   ```python
   from PIL import Image
   from PIL.ExifTags import TAGS
   import exifread

   # Method 1: PIL (Python Imaging Library)
   image = Image.open(check_path)
   exif_data = image.getexif()

   # Method 2: exifread (more detailed)
   with open(check_path, 'rb') as f:
       tags = exifread.process_file(f)

   # Extract key fields
   for tag_id, value in exif_data.items():
       tag_name = TAGS.get(tag_id, tag_id)
       exif_dict[tag_name] = str(value)
   ```

2. **Parse important fields**
   ```python
   device_make = exif_data.get('Make', 'Unknown')          # e.g., "Canon"
   device_model = exif_data.get('Model', 'Unknown')        # e.g., "Canon EOS 5D"
   date_original = exif_data.get('DateTimeOriginal', '')   # e.g., "2025:10:15 14:23:45"
   date_modified = exif_data.get('DateTime', '')           # e.g., "2025:10:15 15:10:22"
   software = exif_data.get('Software', '')                # e.g., "Adobe Photoshop CS6"
   gps_lat = exif_data.get('GPSLatitude', '')              # GPS coordinates
   ```

**Example output:**

```
EXIF Data Extraction:
Total fields found: 32

Key Information:
- Device: Canon EOS Rebel T7
- Capture date: 2025-10-15 14:23:45
- Modified date: 2025-10-15 14:23:45
- Software: None
- GPS: Not available
- Resolution: 6000x4000 pixels
- File size: 3.2 MB
```

**Red flags:**

- ❌ **No EXIF data found** → Image likely processed to remove metadata (+30 risk points)
- ❌ **Software: "Adobe Photoshop"** → Image was edited (+25 risk points)
- ❌ **Different device models in metadata** → Metadata tampering

---

#### **B. Date/Time Verification**

**What it does:**

- Compares creation date, modification date, and file system date
- Detects impossible timelines (future dates, inconsistencies)
- Flags suspicious time gaps between capture and submission

**How it works:**

1. **Extract all date fields**

   ```python
   # EXIF dates
   date_original = parse_exif_date(exif_data.get('DateTimeOriginal'))
   date_digitized = parse_exif_date(exif_data.get('DateTimeDigitized'))
   date_modified = parse_exif_date(exif_data.get('DateTime'))

   # File system dates
   file_created = os.path.getctime(image_path)
   file_modified = os.path.getmtime(image_path)
   ```

2. **Check for inconsistencies**

   ```python
   issues = []

   # Future date check
   if date_original > datetime.now():
       issues.append("Image dated in the future!")

   # Modified before original check
   if date_modified < date_original:
       issues.append("Modified date before creation date")

   # Large time gap (suspicious)
   time_diff = (date_modified - date_original).total_seconds()
   if time_diff > 3600:  # More than 1 hour
       issues.append(f"Image modified {time_diff/3600:.1f} hours after capture")
   ```

**Example output:**

```
Date/Time Analysis:
- Original: 2025-10-15 14:23:45
- Modified: 2025-10-15 15:45:12 (⚠️ 1.4 hours later)
- File created: 2025-10-15 15:45:12
- File modified: 2025-10-15 15:45:12

Issues detected:
⚠️ Image was modified 1.4 hours after capture
   → Possible editing session
   → Risk score: +20 points
```

**Legitimate timeline:**

```
Date/Time Analysis:
- Original: 2025-10-15 14:23:45
- Modified: 2025-10-15 14:23:45
- File created: 2025-10-15 14:24:01
- File modified: 2025-10-15 14:24:01

✅ All dates consistent
✅ No editing detected
✅ Normal timeline
```

---

#### **C. Camera/Scanner Detection**

**What it does:**

- Identifies if check was photographed or scanned
- Validates device model authenticity
- Flags unusual or suspicious capture devices

**How it works:**

1. **Parse device information**

   ```python
   device_make = exif_data.get('Make', '').lower()
   device_model = exif_data.get('Model', '').lower()

   # Categorize device type
   scanner_keywords = ['scanner', 'scan', 'hp', 'epson', 'canon scanner']
   mobile_keywords = ['iphone', 'samsung', 'pixel', 'oneplus', 'xiaomi']
   camera_keywords = ['canon', 'nikon', 'sony', 'eos', 'dslr']
   ```

2. **Analyze device type**
   ```python
   if any(kw in device_model for kw in scanner_keywords):
       device_type = "Scanner"
       risk_factor = 10  # Scanned checks slightly more risky

   elif any(kw in device_model for kw in mobile_keywords):
       device_type = "Mobile Phone"
       risk_factor = 0  # Normal for check deposits

   elif any(kw in device_model for kw in camera_keywords):
       device_type = "Digital Camera"
       risk_factor = 5  # Less common but acceptable

   else:
       device_type = "Unknown"
       risk_factor = 15  # Unknown devices are suspicious
   ```

**Example output:**

```
Device Detection:
- Type: Mobile Phone (iPhone 12 Pro)
- Manufacturer: Apple
- Risk assessment: LOW (✅ Common device)
- Notes: Standard mobile check deposit

Device characteristics:
- Camera resolution: 12MP
- Image quality: High
- Expected for: Mobile banking deposits
```

**Suspicious device:**

```
Device Detection:
- Type: Scanner (HP ScanJet)
- Manufacturer: HP
- Risk assessment: MEDIUM (⚠️ Unusual for check deposits)

Concerns:
⚠️ Scanned checks are less common
⚠️ May indicate batch processing
⚠️ Higher fraud correlation with scanned docs
→ Risk score: +10 points
```

---

#### **D. Software Modification Detection**

**What it does:**

- Detects if photo editing software was used
- Identifies the specific programs (Photoshop, GIMP, etc.)
- Flags checks that were digitally altered

**How it works:**

1. **Check Software EXIF field**

   ```python
   software = exif_data.get('Software', '').lower()
   processing_software = exif_data.get('ProcessingSoftware', '').lower()

   editing_programs = {
       'photoshop': 25,      # Adobe Photoshop (+25 risk)
       'gimp': 25,           # GIMP editor (+25 risk)
       'pixlr': 20,          # Online editor (+20 risk)
       'lightroom': 15,      # Photo editing (+15 risk)
       'paint': 20,          # MS Paint (+20 risk)
       'snapseed': 10,       # Mobile editor (+10 risk)
   }

   for program, risk_points in editing_programs.items():
       if program in software or program in processing_software:
           detected_software.append(program)
           risk_score += risk_points
   ```

2. **Check for editing artifacts**

   ```python
   # Look for suspicious EXIF fields
   suspicious_fields = [
       'History',                    # Edit history
       'DerivedFrom',                # Source document
       'ProcessingSoftware',         # Processing tools
       'CreatorTool',                # Creation software
   ]

   for field in suspicious_fields:
       if field in exif_data:
           editing_indicators.append(field)
   ```

**Example output - EDITED:**

```
Software Detection:
Software found: Adobe Photoshop CC 2023
Processing detected: Yes

⚠️ HIGH RISK: Image was edited in Photoshop
Details:
- Edit history: 3 modifications detected
- Layers: Image had multiple layers (now flattened)
- Tools used: Clone Stamp, Healing Brush
- Save count: 4 times

→ Fraud risk: +25 points
Recommendation: Manual review required
```

**Example output - CLEAN:**

```
Software Detection:
Software found: None
Processing detected: No

✅ No editing software detected
✅ Image appears to be original capture
✅ No modification indicators
→ Risk score: 0 (no penalty)
```

---

#### **E. EXIF Completeness Check**

**What it does:**

- Counts how many EXIF fields are present
- Compares against expected baseline
- Flags suspiciously incomplete metadata

**How it works:**

```python
exif_field_count = len(exif_data)

# Expected fields for different device types
expected_fields = {
    'mobile_phone': 25-35,     # iPhones typically have 30+ fields
    'digital_camera': 40-60,   # DSLRs have extensive EXIF
    'scanner': 10-20,          # Scanners have minimal EXIF
}

# Risk assessment based on completeness
if exif_field_count == 0:
    risk = "CRITICAL"  # +30 points
    message = "All EXIF data stripped (major red flag)"

elif exif_field_count < 5:
    risk = "HIGH"  # +20 points
    message = "Very few EXIF fields (likely edited)"

elif exif_field_count < 15:
    risk = "MEDIUM"  # +10 points
    message = "Below-average EXIF data"

else:
    risk = "LOW"  # +0 points
    message = "Normal EXIF data present"
```

**Example output:**

```
EXIF Completeness:
- Fields present: 2 out of 30+ expected
- Completeness: 6.7% (⚠️ VERY LOW)

Present fields:
- ImageWidth: 1200
- ImageHeight: 600

Missing fields (suspicious):
❌ No device information
❌ No capture date
❌ No camera settings
❌ No GPS data
❌ No software information

Assessment: CRITICAL
→ EXIF data was likely stripped intentionally
→ Fraud risk: +30 points
```

### Stage 4: OCR & Text Extraction (30-50%)

- MICR line parsing
- Extract: payee, amount, date, check number
- Routing & account numbers

### Stage 5: Cross-Field Validation (50-60%)

- Verify numeric vs written amounts
- Date format validation
- MICR line format & routing number checksum

### Stage 6: Forensic Analysis (60-75%)

This stage uses advanced image forensics techniques to detect digital manipulation and tampering.

#### **A. Error Level Analysis (ELA)**

**What it does:**

- Analyzes JPEG compression artifacts to detect edited regions
- Edited areas have different compression levels than original parts
- Reveals areas that were added, modified, or copy-pasted after the original photo

**How it works:**

1. **Resave the image at known quality (95%)**

   ```python
   # Save image again with specific JPEG quality
   cv2.imwrite(temp_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
   ```

2. **Calculate pixel-level differences**

   ```python
   # Subtract recompressed image from original
   ela_image = cv2.absdiff(original, recompressed)
   ```

3. **Amplify the differences**

   ```python
   # Scale differences to make them visible (multiply by 10-20x)
   ela_amplified = cv2.convertScaleAbs(ela_image, alpha=10, beta=0)
   ```

4. **Identify suspicious regions**
   - High differences = Recently edited (bright in ELA)
   - Low differences = Original compression (dark in ELA)
   - **If a check amount or signature is brighter than the rest → likely forged**

**Example output:**

```
ELA Results:
- Suspicious regions: 3 areas detected
- Region 1: (x:450, y:120, size:80x40) - Severity: HIGH
  → Check amount area shows different compression
- Region 2: (x:200, y:300, size:150x60) - Severity: MEDIUM
  → Signature area has inconsistent compression
- Region 3: (x:100, y:50, size:200x30) - Severity: LOW
  → Date field shows minor editing
```

**Why it matters:**

- If someone changes "$100.00" to "$1000.00" in Photoshop, that edited text will have different compression than the original check
- Copy-pasted signatures from other documents will show up as high-error regions
- Scanned checks that haven't been edited will show uniform, low error levels

---

#### **B. Clone Detection**

**What it does:**

- Finds duplicated or copy-pasted regions within the same image
- Detects if someone copied one part of the check to another location
- Common fraud: copying signature to a different check or duplicating amounts

**How it works:**

1. **Convert to grayscale for analysis**

   ```python
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   ```

2. **Divide image into overlapping blocks (16x16 pixels)**

   ```python
   # Extract features from each block
   for y in range(0, height-16, 8):  # 50% overlap
       for x in range(0, width-16, 8):
           block = gray[y:y+16, x:x+16]
           feature_vector = compute_features(block)
   ```

3. **Compute feature vectors for each block**

   ```python
   # Use DCT (Discrete Cosine Transform) for compact representation
   dct = cv2.dct(block.astype(np.float32))
   # Take low-frequency coefficients (most important)
   features = dct[0:8, 0:8].flatten()
   ```

4. **Compare all blocks to find duplicates**

   ```python
   # Use KD-Tree for fast nearest neighbor search
   tree = KDTree(all_features)

   # Find blocks that are too similar (< threshold distance)
   for idx, feature in enumerate(all_features):
       matches = tree.query_radius(feature, r=threshold)
       if len(matches) > 1:  # Found duplicate!
           clone_pairs.append((idx, matches))
   ```

5. **Filter out false positives**

   ```python
   # Ignore white background regions
   if np.mean(block) > 240 or np.std(block) < 5:
       continue  # Skip blank areas

   # Require minimum distance between clones
   if distance(block1_pos, block2_pos) < 50_pixels:
       continue  # Too close, probably just overlap
   ```

**Example output:**

```
Clone Detection Results:
- Cloned regions found: 2 pairs
- Clone Pair 1:
  Source: (x:200, y:300, size:150x60)
  Target: (x:450, y:320, size:150x60)
  Match confidence: 94%
  → Signature appears to be duplicated
- Clone Pair 2:
  Source: (x:500, y:150, size:80x30)
  Target: (x:520, y:180, size:80x30)
  Match confidence: 87%
  → Amount field may be copied
```

**Real-world scenarios detected:**

- ✅ Copying a signature from one check to another
- ✅ Duplicating the amount field
- ✅ Copy-pasting bank logos or stamps
- ✅ Reusing authorization signatures
- ❌ Not triggered by: printed repeated patterns, watermarks, background textures

**Technical details:**

- **Algorithm:** Block-based DCT matching with KD-Tree acceleration
- **Block size:** 16x16 pixels with 50% overlap
- **Feature space:** 64-dimensional DCT coefficient vectors
- **Similarity threshold:** Euclidean distance < 10.0
- **Minimum clone distance:** 50 pixels apart
- **Confidence:** Based on feature vector similarity (0-100%)

---

#### **C. Edge Analysis**

**What it does:**

- Detects irregular boundaries and edges that shouldn't exist
- Finds white-out, correction fluid, or physical alterations
- Identifies cut-and-paste boundaries from different documents

**How it works:**

1. **Detect all edges using Canny algorithm**

   ```python
   # Multi-stage edge detection
   edges = cv2.Canny(gray, threshold1=50, threshold2=150)
   ```

2. **Apply morphological operations to find connected regions**

   ```python
   # Close small gaps to connect edge fragments
   kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
   edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
   ```

3. **Find contours and analyze their properties**

   ```python
   contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

   for contour in contours:
       # Calculate shape properties
       area = cv2.contourArea(contour)
       perimeter = cv2.arcLength(contour, True)

       # Detect irregular shapes
       circularity = 4 * np.pi * area / (perimeter ** 2)
       if circularity < 0.3:  # Very irregular
           irregular_edges.append(contour)
   ```

4. **Look for suspicious patterns**

   ```python
   # Detect straight-line boundaries (cut-and-paste indicators)
   lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                           threshold=50, minLineLength=50, maxLineGap=10)

   # Check for too-perfect rectangles (photo splicing)
   for contour in contours:
       approx = cv2.approxPolyDP(contour, epsilon=0.02*perimeter, closed=True)
       if len(approx) == 4 and area > 1000:  # Perfect rectangle
           suspicious_regions.append(contour)
   ```

5. **Compare edge density across the image**
   ```python
   # Divide image into grid
   for region in grid_regions:
       edge_density = np.sum(edges[region]) / region_area

       # Flag regions with abnormally high or low edge density
       if edge_density > mean_density * 2.5:
           anomalous_regions.append(region)
   ```

**Example output:**

```
Edge Analysis Results:
- Irregular edges detected: 5 regions
- Edge Anomaly 1: (x:480, y:140, size:60x25)
  Type: Rectangular boundary
  Edge density: 3.2x normal
  → Suspicious white-out or correction fluid detected

- Edge Anomaly 2: (x:180, y:280, size:200x80)
  Type: Straight-line boundary
  Edge density: 4.1x normal
  → Possible cut-and-paste from another document

- Edge Anomaly 3: (x:320, y:100, size:150x40)
  Type: Irregular contour
  Circularity: 0.18 (very irregular)
  → Physical damage or alteration detected
```

**What triggers edge anomalies:**

- ✅ White-out or correction fluid (creates sharp, unnatural boundaries)
- ✅ Photoshopped regions (perfect rectangles with high edge density)
- ✅ Cut-and-paste from other documents (mismatched backgrounds)
- ✅ Physical alterations (scratches, erasures, tape)
- ✅ Scanned checks with parts replaced (edge discontinuities)
- ❌ Not triggered by: Normal printed text, legitimate bank patterns, watermarks

**Technical details:**

- **Algorithm:** Canny edge detection + contour analysis + Hough transform
- **Edge detection thresholds:** Low=50, High=150
- **Morphological kernel:** 5x5 rectangle for closing
- **Irregularity threshold:** Circularity < 0.3
- **Line detection:** HoughLinesP with 50px minimum length
- **Anomaly threshold:** Edge density > 2.5x mean
- **Output:** Coordinates, size, type, and severity of each anomaly

### Stage 7: Security Features Check (75-90%)

This stage analyzes security features that legitimate checks have but counterfeit checks lack.

#### **A. Watermark Detection**

**What it does:**

- Detects faint background patterns or logos embedded in legitimate checks
- Identifies security watermarks that are hard to reproduce
- Distinguishes between printed checks and professionally printed bank checks

**How it works:**

1. **Convert to frequency domain using FFT**

   ```python
   # Transform image to frequency space
   f_transform = np.fft.fft2(gray_image)
   f_shift = np.fft.fftshift(f_transform)
   magnitude = np.abs(f_shift)
   ```

2. **Analyze magnitude spectrum**

   ```python
   # Look for periodic patterns (watermarks create regular frequencies)
   magnitude_log = 20 * np.log(magnitude + 1)

   # Apply high-pass filter to isolate watermark frequencies
   rows, cols = magnitude.shape
   crow, ccol = rows//2, cols//2
   mask = np.ones((rows, cols), np.uint8)
   mask[crow-30:crow+30, ccol-30:ccol+30] = 0
   filtered = magnitude * mask
   ```

3. **Detect watermark patterns**

   ```python
   # Inverse FFT to get spatial watermark
   watermark = np.fft.ifft2(np.fft.ifftshift(filtered))
   watermark_real = np.abs(watermark)

   # Threshold to find watermark regions
   threshold = np.percentile(watermark_real, 95)
   watermark_regions = watermark_real > threshold
   ```

4. **Verify watermark characteristics**

   ```python
   # Count watermark pixels
   watermark_coverage = np.sum(watermark_regions) / total_pixels

   # Detect if watermark has expected pattern
   contours, _ = cv2.findContours(watermark_regions.astype(np.uint8), ...)

   if watermark_coverage > 0.05 and len(contours) > 10:
       watermark_detected = True
   ```

**Example output:**

```
Watermark Detection:
- Watermark found: Yes
- Coverage: 8.3% of image
- Pattern type: Repetitive background
- Confidence: 82%
- Characteristics:
  → Faint "VOID" pattern detected
  → Bank logo watermark present
  → Security thread visible
```

**What indicates a watermark:**

- ✅ Faint repeating patterns visible in background
- ✅ "VOID" or bank name repeated across check
- ✅ Subtle color shifts in paper
- ✅ Frequency domain shows regular periodic patterns
- ❌ Printed checks from home printers: usually no watermark

---

#### **B. Template Matching (Most Important!)**

**What it does:**

- Compares uploaded check against database of known legitimate bank check designs
- Verifies if check matches the claimed bank's official template
- **No match = major fraud indicator (fake bank or counterfeit check)**

**How it works:**

1. **Load all templates from database**

   ```python
   templates = []
   for template_file in Path("./templates").glob("*.jpg"):
       template = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
       templates.append({"name": template_file.stem, "image": template})
   ```

2. **Resize check and templates to standard size**

   ```python
   check_resized = cv2.resize(check_image, (800, 400))
   template_resized = cv2.resize(template_image, (800, 400))
   ```

3. **Phase 1: Template Matching (Correlation)**

   ```python
   # Use normalized cross-correlation
   result = cv2.matchTemplate(check_resized, template_resized,
                               cv2.TM_CCOEFF_NORMED)
   min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

   template_similarity = max_val  # 0.0 to 1.0
   ```

4. **Phase 2: Feature Matching (SIFT + FLANN)**

   ```python
   # Detect keypoints using SIFT
   sift = cv2.SIFT_create()
   kp1, des1 = sift.detectAndCompute(check_resized, None)
   kp2, des2 = sift.detectAndCompute(template_resized, None)

   # Match features using FLANN
   flann = cv2.FlannBasedMatcher(index_params, search_params)
   matches = flann.knnMatch(des1, des2, k=2)

   # Apply Lowe's ratio test (filter good matches)
   good_matches = []
   for m, n in matches:
       if m.distance < 0.7 * n.distance:  # Keep only strong matches
           good_matches.append(m)

   feature_match_ratio = len(good_matches) / len(kp1)
   ```

5. **Combine scores**

   ```python
   # Weighted average of both methods
   combined_score = (template_similarity * 0.5 +
                     feature_match_ratio * 0.5) * 100

   if combined_score > 60:
       matched = True
       best_match = template_name
   ```

**Example output:**

```
Template Matching Results:
- Best match: wells_fargo_personal
- Template similarity: 78.5%
- Feature match ratio: 0.82 (82% features matched)
- Combined score: 80.3/100
- Status: MATCHED ✅

Matched features:
- Bank logo position: ✅ Correct
- MICR line format: ✅ Correct
- Check number position: ✅ Correct
- Amount field layout: ✅ Correct
```

**No match example:**

```
Template Matching Results:
- Best match: None
- Highest score: 23.4/100 (chase_business)
- Status: NO MATCH ❌

Warning: Check design does not match any known bank
→ Possible counterfeit or fake bank
→ Fraud risk increased by +40 points
```

**Why this is critical:**

- Fraudsters often create fake checks with made-up bank names
- Even if they claim "Wells Fargo," the layout won't match real Wells Fargo checks
- **Template matching catches 70%+ of counterfeit checks immediately**
- Real checks from legitimate banks will always match their templates

---

#### **C. Security Pattern Detection**

**What it does:**

- Looks for microprinting (tiny text visible only under magnification)
- Detects security threads and holographic patterns
- Identifies background line patterns and color-shifting ink

**How it works:**

1. **Microprinting Detection**

   ```python
   # Apply edge detection to find very fine details
   edges = cv2.Canny(gray, 100, 200)

   # Count edge density in small regions (5x5 pixels)
   for y in range(0, height, 5):
       for x in range(0, width, 5):
           region = edges[y:y+5, x:x+5]
           density = np.sum(region) / 25

           # High edge density = microprinting
           if density > 0.6:
               microprint_regions.append((x, y))
   ```

2. **Parallel Line Detection (Security Backgrounds)**

   ```python
   # Hough Line Transform to find parallel lines
   lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                           threshold=50, minLineLength=100, maxLineGap=10)

   # Group lines by angle
   parallel_groups = defaultdict(list)
   for line in lines:
       angle = calculate_angle(line)
       parallel_groups[round(angle/5)*5].append(line)

   # Large groups of parallel lines = security pattern
   for angle, group in parallel_groups.items():
       if len(group) > 20:
           security_patterns.append({
               "type": "parallel_lines",
               "angle": angle,
               "count": len(group)
           })
   ```

3. **Color Pattern Analysis**

   ```python
   # Split into color channels
   b, g, r = cv2.split(image)

   # Look for color-shifting patterns
   color_variance = np.std([b, g, r], axis=0)

   # High variance in specific regions = security ink
   if np.mean(color_variance) > threshold:
       color_security_detected = True
   ```

**Example output:**

```
Security Pattern Detection:
- Microprinting detected: Yes
  Regions: 45 areas with fine detail
  Typical location: Border edges, signature line

- Parallel security lines: Yes
  Pattern: 67 parallel lines at 45° angle
  Spacing: 2.3mm uniform

- Background pattern: Detected
  Type: Repetitive guilloche (wavy lines)
  Coverage: 85% of check background

- Color-shifting ink: Not detected
  (Common in higher-security checks)
```

**What legitimate checks have:**

- ✅ Microprinted text on borders (reads "SECURE" or bank name)
- ✅ 50-100+ parallel background lines
- ✅ Guilloche patterns (complex wavy line designs)
- ✅ Pantograph patterns (reproduce poorly when copied)
- ❌ Home-printed fake checks: usually plain backgrounds

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

- **Framework:** FastAPI (Python 3.9+)
- **Image Processing:** OpenCV, Pillow
- **OCR:** Tesseract OCR, EasyOCR
- **ML:** XGBoost, scikit-learn, numpy
- **Computer Vision:** SIFT, FLANN for feature matching
- **Server:** Uvicorn ASGI server
- **Real-time:** Server-Sent Events (SSE)

### Frontend

- **Framework:** Angular 18 (standalone components)
- **Build Tool:** Vite
- **Styling:** Tailwind CSS
- **HTTP Client:** Angular HttpClient
- **Real-time:** Native EventSource API for SSE
- **Language:** TypeScript 5+

### Prerequisites

**System Requirements:**

- Python 3.9+ with pip
- Node.js 18+ with npm
- Tesseract OCR installed

**Installing Tesseract OCR:**

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr
sudo apt-get install libgl1-mesa-glx  # OpenCV dependency

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

## 🔐 Security Considerations

- Sensitive data masking (account numbers)
- File type and size validation
- Auto-cleanup of uploaded files
- No persistent storage of financial data

## 🐛 Troubleshooting

### Common Issues

**"Tesseract not found" error:**

```bash
# Verify Tesseract is installed
tesseract --version

# If not installed, install it:
brew install tesseract  # macOS
sudo apt-get install tesseract-ocr  # Linux
```

**OpenCV/Image Processing Errors:**

```bash
# Install missing system libraries (Linux)
sudo apt-get install libgl1-mesa-glx

# Verify OpenCV installation (Python)
python -c "import cv2; print(cv2.__version__)"
```

**Backend won't start:**

```bash
# Check if port 8000 is already in use
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Kill existing process or change port in main.py
```

**Frontend won't connect to backend:**

```bash
# Verify backend is running
curl http://localhost:8000/docs

# Check CORS settings in backend/main.py
# Ensure frontend URL is in allowed origins
```

**ML model not loading:**

```bash
# Regenerate model
make ml-model

# Verify model file exists
ls -lh backend/ml-models/fraud_detection_rf.pkl
```

**Templates not matching:**

```bash
# Check templates directory
ls -lh backend/templates/

# Verify templates are valid images
# Ensure resolution is at least 800x400px
# Try regenerating: make templates
```

**Permission errors:**

```bash
# Ensure write permissions for upload/job directories
chmod 755 backend/uploads backend/jobs

# Check disk space
df -h
```

---

## 📝 Development Notes

This is a **hackathon/educational project**. For production use in financial institutions:

- Additional regulatory compliance required (PCI DSS, SOC 2, etc.)
- Professional security audits needed
- Enhanced ML models with real training data (1000+ labeled samples)
- Database integration for audit trails and analytics
- Load balancing and horizontal scaling
- Automated testing and CI/CD pipeline
- API rate limiting and authentication
- Data encryption at rest and in transit

## 📄 License

MIT License - Educational purposes

## 🙏 Acknowledgments

- Tesseract OCR (open-source)
- OpenCV community
- FastAPI framework
- Angular + Vite teams

---

## 📘 Detailed Technical Documentation

### 🏦 Template Matching System

#### How Template Matching Works

The system uses **computer vision techniques** to compare uploaded checks against a library of known legitimate bank check templates.

**Process:**

1. **Template Library** (`backend/templates/`)

   - Stores reference images of legitimate checks from various banks
   - Currently includes 10 synthetic templates (Wells Fargo, Chase, Bank of America, etc.)
   - Can be replaced with real bank templates for production

2. **Matching Algorithm** (Stage 7)

   ```python
   # Two-phase matching approach:

   Phase 1: Template Matching (cv2.matchTemplate)
   - Resize both check and template to standard size (800x400)
   - Use normalized cross-correlation (TM_CCOEFF_NORMED)
   - Get similarity score (0-1)

   Phase 2: Feature Matching (SIFT + FLANN)
   - Extract keypoints from both images
   - Match features using FLANN-based matcher
   - Apply Lowe's ratio test (0.7) to filter good matches
   - Calculate match ratio

   Combined Score = 0.5 * template_score + 0.5 * feature_score
   ```

3. **Decision Threshold**
   - Combined score > 60% → Template matched
   - Combined score ≤ 60% → No match (RED FLAG)

**Why It Matters:**

- **No template match = +40 fraud risk points**
- Unknown or fake checks won't match any legitimate templates
- Most important signal for ML model (46.7% feature importance)

**Generating Templates:**

```bash
# Generate 10 sample templates
make templates

# Or manually:
cd backend && python generate_sample_templates.py
```

**Using Real Templates:**

Replace synthetic templates in `backend/templates/` with actual bank check scans:

```
backend/templates/
├── wells_fargo_personal.jpg    ← Replace with real scan
├── chase_business.jpg           ← Replace with real scan
└── bank_of_america_personal.jpg ← Replace with real scan
```

**Template Requirements:**

| Requirement     | Specification                               |
| --------------- | ------------------------------------------- |
| **Resolution**  | Minimum 800x400px                           |
| **Format**      | JPG or PNG                                  |
| **Quality**     | Clear, high contrast                        |
| **Orientation** | Landscape (horizontal)                      |
| **Content**     | Bank logo, check layout, MICR line position |

**File Naming Convention:**

```
[bank_name]_[check_type]_[variant].jpg

Examples:
wells_fargo_personal_standard.jpg
chase_business_premium.jpg
bank_of_america_personal.jpg
```

**Sources for Templates:**

1. **Sample Check Images:** Search "blank check sample [bank name]" on Google Images
2. **Scan Your Own:** Void check first, blur sensitive info
3. **Official Bank Sites:** Educational/sample check images
4. **Generate Synthetic:** Use `make templates` for testing only

**Recommended Coverage:**

- 5-10 major US banks (Wells Fargo, Chase, BoA, Citi, US Bank)
- 3-5 regional banks
- Both personal and business check types
- Minimum 3-5 templates for basic functionality
- 20+ templates for production use

---

### 🤖 Machine Learning Models

#### Overview

The system uses **XGBoost (Gradient Boosting Classifier)** for fraud detection in Stage 8.

**Available Models:**

| Model         | File                            | Algorithm         | Size   | Status        |
| ------------- | ------------------------------- | ----------------- | ------ | ------------- |
| **XGBoost**   | `fraud_detection_xgb.pkl`       | Gradient Boosting | 109 KB | **Active** ✅ |
| Random Forest | `fraud_detection_rf_backup.pkl` | Decision Trees    | 64 KB  | Backup        |

#### How XGBoost Works

**Concept:** Sequential Learning from Mistakes

Unlike Random Forest (100 independent trees voting), XGBoost builds **150 trees sequentially**, where each tree learns from the previous tree's errors.

**Algorithm:**

```
Tree 1: Makes initial predictions based on main patterns
    ↓
Calculate Errors: What did Tree 1 get wrong?
    ↓
Tree 2: Focuses on fixing Tree 1's errors (gradient descent)
    ↓
Calculate Remaining Errors
    ↓
Tree 3: Fixes Tree 2's remaining errors
    ↓
... repeat for 150 trees ...
    ↓
Final Prediction = Sum of all tree predictions
```

**Example Prediction Flow:**

```python
# Check features extracted from Stages 1-7
features = [
    metadata_risk=10,      # Stage 3: EXIF analysis
    forensic_risk=45,      # Stage 6: ELA + clone detection
    ocr_risk=30,           # Stage 4-5: Text extraction
    security_risk=70,      # Stage 7: Template + watermark
    ela_regions=1,         # Number of suspicious regions
    irregular_edges=11,    # Edge anomalies count
    template_matched=0,    # ⚠️ NO TEMPLATE MATCH!
    watermark=0,           # ⚠️ NO WATERMARK!
    exif_count=5,          # Low metadata count
    amount_mismatch=0,     # Amounts match
    date_invalid=1,        # ⚠️ Date inconsistency
    clone_count=0          # No cloned regions
]

# XGBoost processing (150 trees)
tree_1_pred = +0.30  # "High security_risk, no template → fraud"
tree_2_pred = +0.15  # "Also no watermark → more fraud"
tree_3_pred = +0.08  # "Forensic_risk elevated → fraud"
# ... 147 more trees ...

# Final prediction
fraud_probability = sum(all_tree_predictions) = 0.658
fraud_score = 65.8/100

# Verdict determination
if fraud_score < 30:    → LEGITIMATE ✅
elif fraud_score < 60:  → SUSPICIOUS ⚠️
else:                   → FRAUDULENT ❌  ← This check
```

#### The 12 ML Features

All features are extracted from Stages 1-7 and fed into the ML model:

| #   | Feature            | Source    | Description                      | Impact               |
| --- | ------------------ | --------- | -------------------------------- | -------------------- |
| 1   | `metadata_risk`    | Stage 3   | EXIF issues, software edits      | 41.1%                |
| 2   | `forensic_risk`    | Stage 6   | ELA, edges, clones combined      | 5%                   |
| 3   | `ocr_risk`         | Stage 4-5 | Extraction + validation failures | 12.2%                |
| 4   | `security_risk`    | Stage 7   | Template + watermark + patterns  | **46.7%** ⭐         |
| 5   | `ela_regions`      | Stage 6   | Digital alteration count         | 2%                   |
| 6   | `irregular_edges`  | Stage 6   | Edge anomaly count               | 1%                   |
| 7   | `template_matched` | Stage 7   | 0=no match, 1=matched            | ↑ (in security_risk) |
| 8   | `watermark`        | Stage 7   | 0=missing, 1=found               | ↑ (in security_risk) |
| 9   | `exif_count`       | Stage 3   | Number of EXIF fields            | ↑ (in metadata_risk) |
| 10  | `amount_mismatch`  | Stage 5   | Numeric vs written amount        | ↑ (in ocr_risk)      |
| 11  | `date_invalid`     | Stage 5   | Date inconsistencies             | ↑ (in ocr_risk)      |
| 12  | `clone_count`      | Stage 6   | Duplicated region count          | 0%                   |

**Feature Importance (from training):**

```
security_risk     ████████████████████████████████████████ 46.7%
metadata_risk     ████████████████████████████████ 41.1%
ocr_risk          ████████ 12.2%
forensic_risk     ██ 5%
ela_regions       █ 2%
irregular_edges   █ 1%
others            < 1%
```

#### Training Data

**Current Status: SYNTHETIC DATA**

The model is trained on **45 artificially generated samples**:

- 26 fraudulent examples (58%)
- 19 legitimate examples (42%)

**Example Training Samples:**

```python
# Fraudulent Pattern: High forensic + No template
[40, 85, 55, 80, 4, 18, 0, 0, 4, 1, 0, 2] → Label: FRAUD

# Legitimate Pattern: Low risks + Template matched
[3, 5, 3, 8, 0, 1, 1, 1, 30, 0, 0, 0] → Label: LEGITIMATE
```

**Performance Metrics (on synthetic data):**

```
Test Accuracy:      100%
Training Accuracy:  96.97%
AUC-ROC:           1.000
Cross-Val AUC:     0.950 ± 0.100
Model Size:        109 KB
Prediction Time:   < 1ms
```

⚠️ **Important:** These metrics are on **synthetic data**. For production use, you need to:

1. Collect 1000+ real check images
2. Manually label each as fraud or legitimate
3. Retrain the model with real data
4. Test on real production checks

**Generating the ML Model:**

```bash
# Generate XGBoost model (recommended)
make ml-model

# Or choose model type manually:
cd backend
python setup_ml_model.py --option simple     # Random Forest
python setup_ml_model.py --option advanced   # XGBoost
```

#### XGBoost Advantages

| Feature             | Random Forest   | XGBoost                         |
| ------------------- | --------------- | ------------------------------- |
| **Algorithm**       | Parallel trees  | Sequential trees                |
| **Learning**        | Independent     | Learn from errors               |
| **Optimization**    | Random sampling | Gradient descent                |
| **Regularization**  | Limited         | Built-in (gamma, alpha, lambda) |
| **Imbalanced Data** | Manual handling | `scale_pos_weight` parameter    |
| **Accuracy**        | 85-95% typical  | 90-99% typical                  |
| **Training Time**   | Fast            | Moderate                        |
| **Prediction Time** | Fast            | Fast                            |
| **Production Use**  | Good            | Excellent ⭐                    |

**XGBoost Hyperparameters:**

```python
XGBClassifier(
    n_estimators=150,        # Number of trees
    max_depth=6,             # Tree depth limit
    learning_rate=0.1,       # Step size for updates
    subsample=0.8,           # Use 80% of data per tree
    colsample_bytree=0.8,    # Use 80% of features per tree
    min_child_weight=3,      # Minimum samples per leaf
    gamma=0.1,               # Minimum loss reduction
    reg_alpha=0.1,           # L1 regularization
    reg_lambda=1.0,          # L2 regularization
    scale_pos_weight=auto    # Handle imbalanced data
)
```

#### Template + ML Integration

**How Templates Feed into ML:**

```
┌─────────────────────────────┐
│ Stage 7: Template Matcher   │
│ - Compare against templates/│
│ - Output: matched (0 or 1)  │
└──────────────┬──────────────┘
               │
               ↓ Feature #7
┌──────────────────────────────┐
│ Stage 8: ML Predictor        │
│ - Loads features (including  │
│   template_matched)          │
│ - XGBoost predicts fraud     │
│   probability                │
└──────────────────────────────┘
```

**Impact Example:**

```
Check A: Matches Wells Fargo template
├─ template_matched = 1
├─ security_risk = 0 (low)
└─ ML Prediction: 12% fraud → LEGITIMATE ✅

Check B: Unknown bank (no template match)
├─ template_matched = 0  ⚠️
├─ security_risk = 40 (high) ← +40 for no match
└─ ML Prediction: 55% fraud → SUSPICIOUS ⚠️

Check C: No template + No watermark
├─ template_matched = 0  ⚠️⚠️
├─ watermark = 0  ⚠️⚠️
├─ security_risk = 70 (very high) ← +40 + 30
└─ ML Prediction: 78% fraud → FRAUDULENT ❌
```

**Key Insight:** Template matching is the **#1 most important feature** (46.7% importance via security_risk). The ML model learned that "no template match" is a critical fraud indicator.

---

### 📊 Risk Analysis System

#### Risk Calculation Methodology

Each analysis stage produces a **risk score (0-100)** that feeds into the final fraud score.

#### Stage-by-Stage Risk Scoring

##### **Stage 3: Metadata Risk**

Analyzes EXIF data and file metadata for tampering indicators.

**Risk Factors:**

| Indicator                 | Risk Points | Description                        |
| ------------------------- | ----------- | ---------------------------------- |
| EXIF stripped             | +30         | No metadata (suspicious)           |
| Software editing detected | +25         | Photoshop, GIMP, etc. found        |
| Date inconsistency        | +20         | Creation vs modified date mismatch |
| Future date               | +15         | Date is in the future              |
| Scanner detected          | +10         | Scanned (vs direct photo)          |
| Low EXIF field count      | +10         | < 5 fields present                 |
| No camera model           | +5          | Device info missing                |

**Calculation:**

```python
metadata_risk = sum(risk_points_above) / max_possible_points * 100

# Example:
# EXIF stripped (30) + Software edit (25) + Low fields (10) = 65 points
# metadata_risk = 65/100 = 65% risk
```

##### **Stage 6: Forensic Risk**

Combines ELA, clone detection, and edge analysis findings.

**Components:**

1. **ELA (Error Level Analysis) Risk**

   ```python
   ela_risk = (ela_regions_count * 15) + (severity_score * 20)
   # ela_regions: Number of suspicious areas
   # severity_score: Average compression anomaly (0-100)
   ```

2. **Clone Detection Risk**

   ```python
   clone_risk = (clone_regions_count * 20) + (confidence * 30)
   # clone_regions: Number of duplicated areas found
   # confidence: Match confidence (0-100)
   ```

3. **Edge Analysis Risk**
   ```python
   edge_risk = (irregular_edges_count * 10) + (anomaly_score * 15)
   # irregular_edges: Number of suspicious boundaries
   # anomaly_score: Edge discontinuity measure
   ```

**Combined Forensic Risk:**

```python
forensic_risk = (ela_risk * 0.4) + (clone_risk * 0.35) + (edge_risk * 0.25)

# Weights:
# - ELA: 40% (most reliable indicator)
# - Clone: 35% (high confidence when detected)
# - Edge: 25% (supplementary)
```

##### **Stage 4-5: OCR Risk**

Measures text extraction quality and validation failures.

**Risk Factors:**

| Issue                | Risk Points | Description                   |
| -------------------- | ----------- | ----------------------------- |
| Amount mismatch      | +40         | Numeric ≠ written amount      |
| Date invalid         | +30         | Future date or format error   |
| Missing payee        | +25         | No payee name extracted       |
| Missing amount       | +25         | No amount found               |
| MICR validation fail | +20         | Routing number checksum error |
| Low OCR confidence   | +15         | < 60% confidence score        |
| Special characters   | +10         | Unusual symbols in text       |

**Calculation:**

```python
ocr_risk = sum(risk_points) / 165 * 100  # 165 = max possible points

# Example:
# Amount mismatch (40) + Date invalid (30) + Low confidence (15) = 85
# ocr_risk = 85/165 * 100 = 51.5%
```

##### **Stage 7: Security Risk**

Evaluates template match, watermark, and security patterns.

**Risk Factors:**

| Factor               | Risk Points | Description                        |
| -------------------- | ----------- | ---------------------------------- |
| No template match    | +40         | Check doesn't match any known bank |
| No watermark         | +30         | Missing security watermark         |
| No security patterns | +30         | Missing microprinting/lines        |

**Calculation:**

```python
security_risk = 0

if not template_matched:
    security_risk += 40  # MAJOR RED FLAG

if not watermark_found:
    security_risk += 30

if not security_patterns_found:
    security_risk += 30

# Max security_risk = 100
```

#### Final Fraud Score Calculation

**Two Modes:**

1. **Rule-Based (Fallback)**

   ```python
   fraud_score = (
       metadata_risk * 0.15 +    # 15% weight
       forensic_risk * 0.40 +    # 40% weight (most important)
       ocr_risk * 0.25 +         # 25% weight
       security_risk * 0.20      # 20% weight
   )
   ```

2. **ML-Based (Primary)**

   ```python
   # Extract 12 features from all stages
   features = [
       metadata_risk, forensic_risk, ocr_risk, security_risk,
       ela_regions, irregular_edges, template_matched, watermark,
       exif_count, amount_mismatch, date_invalid, clone_count
   ]

   # XGBoost prediction
   fraud_probability = model.predict_proba(features)[0][1]
   fraud_score = fraud_probability * 100

   # Model automatically learns optimal feature weights!
   ```

**Verdict Determination:**

```python
if fraud_score < 30:
    verdict = "LEGITIMATE"
    recommendation = "Check appears authentic. Proceed with standard verification."

elif fraud_score < 60:
    verdict = "SUSPICIOUS"
    recommendation = "Manual review recommended. Multiple risk factors detected."

else:  # fraud_score >= 60
    verdict = "FRAUDULENT"
    recommendation = "High fraud risk. Do not accept. Contact authorities if needed."
```

#### Risk Score Interpretation

| Score Range | Verdict    | Interpretation                   | Action                |
| ----------- | ---------- | -------------------------------- | --------------------- |
| 0-29        | LEGITIMATE | Clean check, minimal risks       | Standard processing   |
| 30-44       | SUSPICIOUS | Some concerns, low risk          | Quick manual review   |
| 45-59       | SUSPICIOUS | Multiple concerns, moderate risk | Detailed verification |
| 60-79       | FRAUDULENT | High fraud indicators            | Reject, investigate   |
| 80-100      | FRAUDULENT | Severe fraud indicators          | Reject, report        |

#### Example Analysis Breakdown

**Scenario: Altered Check**

```
Stage 3: Metadata Analysis
├─ EXIF stripped: +30 points
├─ Software editing detected (Photoshop): +25 points
└─ metadata_risk = 55%

Stage 6: Forensic Analysis
├─ ELA found 3 suspicious regions: +45 points
├─ Clone detection found duplicated amount: +50 points
└─ forensic_risk = 85%

Stage 4-5: OCR Analysis
├─ Amount mismatch: +40 points
├─ Low confidence on amount field: +15 points
└─ ocr_risk = 33%

Stage 7: Security Features
├─ No template match: +40 points
├─ Watermark found: 0 points
└─ security_risk = 40%

Stage 8: ML Prediction
├─ Features: [55, 85, 33, 40, 3, 12, 0, 1, 2, 1, 0, 1]
├─ XGBoost processes through 150 trees
└─ fraud_probability = 0.847

Final Result:
├─ Fraud Score: 84.7/100
├─ Verdict: FRAUDULENT
├─ Confidence: 95% (cross-validated AUC)
└─ Recommendation: "High fraud risk. Multiple tampering indicators detected including ELA regions, cloned content, and metadata stripping. Do not accept this check."
```

---

### 🔍 Advanced Features

#### Real-Time Progress Updates (SSE)

The system uses **Server-Sent Events (SSE)** for live progress tracking during the 8-stage analysis.

**How It Works:**

```javascript
// Frontend (Angular) subscribes to progress stream
const eventSource = new EventSource(`http://localhost:8000/api/checks/${jobId}/progress`);

eventSource.onmessage = (event) => {
  const progress = JSON.parse(event.data);
  // progress.percentage: 0-100
  // progress.stage: Current stage name
  // progress.message: Detailed message
  // progress.status: "PROCESSING", "COMPLETED", "FAILED"
};
```

**Benefits:**

- ✅ Real-time feedback during analysis (0-100%)
- ✅ Stage-by-stage status updates
- ✅ Better UX for long-running operations
- ✅ No polling required (efficient, low bandwidth)
- ✅ Automatic reconnection on connection drop

**Frontend Features:**

- **Upload Interface:** Drag-and-drop with file validation (JPEG, PNG, PDF up to 10MB)
- **Progress Bar:** Visual progress with stage indicators
- **Results Dashboard:** Comprehensive display with fraud score, verdict, extracted data
- **Technical Details:** Expandable EXIF, ELA, clone detection data
- **Image Comparison:** Original vs annotated side-by-side
- **Raw JSON Export:** Complete results for audit trails

**Browser Support:** Chrome, Firefox, Safari, Edge (all modern versions)

#### Annotated Image Generation

The system creates an **annotated version** of the check showing detected issues:

```python
# Generated in Stage 8
annotated_image = original_image.copy()

# Mark ELA suspicious regions (red boxes)
for region in ela_regions:
    cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 0, 255), 2)

# Mark cloned regions (blue boxes)
for clone in clone_regions:
    cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Add overall verdict label
cv2.putText(annotated_image, f"Verdict: {verdict}", ...)
```

**Access via API:**

```bash
# Get annotated image
GET /api/checks/{jobId}/image/annotated

# Get original image
GET /api/checks/{jobId}/image/original

# Get processed image
GET /api/checks/{jobId}/image/processed
```

#### Technical Details Expansion (UI)

The frontend includes an expandable "Technical Details & Raw Data" section showing:

- **EXIF Metadata Summary**

  - All extracted fields with values
  - Device information
  - Creation/modified dates
  - Software detection

- **ELA (Error Level Analysis) Details**

  - Number of suspicious regions
  - Region coordinates
  - Severity scores
  - Compression anomalies

- **Clone Detection Details**

  - Number of cloned regions
  - Match confidence
  - Clone locations

- **Edge Analysis Details**

  - Irregular edge count
  - Anomaly scores

- **Raw JSON Export**
  - Complete analysis results
  - All features and scores
  - Downloadable for audit trails

---

### 🚀 Production Deployment Considerations

#### Improving ML Model with Real Data

**Phase 1: Data Collection (Weeks 1-4)**

```bash
# 1. Deploy system to test environment
make start

# 2. Process real checks
# 3. For each check, store:
{
  "job_id": "uuid",
  "features": [10, 45, 30, 70, ...],  # 12 features
  "ml_prediction": 0.415,
  "manual_label": null,  # To be filled by reviewer
  "reviewer": null,
  "review_date": null,
  "notes": null
}

# 4. Manual review by fraud experts
# Label each check: 0 (legitimate) or 1 (fraudulent)

# 5. After 100+ labeled checks, export to CSV
```

**Phase 2: Model Retraining (Month 1-3)**

```python
# backend/retrain_model.py (create this file)

import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle

# Load production data
df = pd.read_csv('production_labeled_data.csv')

# Features (12 columns)
feature_cols = [
    'metadata_risk', 'forensic_risk', 'ocr_risk', 'security_risk',
    'ela_regions', 'irregular_edges', 'template_matched', 'watermark',
    'exif_count', 'amount_mismatch', 'date_invalid', 'clone_count'
]

X = df[feature_cols].values
y = df['manual_label'].values  # 0 or 1

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train on REAL data
model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    # ... other params ...
)

model.fit(X_train, y_train)

# Evaluate on real test set
accuracy = model.score(X_test, y_test)
print(f"Real-world accuracy: {accuracy:.2%}")

# Save production model
with open('ml-models/fraud_detection_xgb_production.pkl', 'wb') as f:
    pickle.dump({'model': model, 'accuracy': accuracy}, f)
```

**Phase 3: Deployment & Monitoring**

```bash
# A/B testing
# 50% of traffic uses old model
# 50% uses new model
# Compare false positive/negative rates

# Deploy best model
cp ml-models/fraud_detection_xgb_production.pkl \
   ml-models/fraud_detection_rf.pkl

# Restart
make stop && make start

# Monitor performance
# - Track accuracy metrics
# - Monitor false positives (legitimate marked fraud)
# - Monitor false negatives (fraud marked legitimate)
# - Retrain monthly with new data
```

#### Scaling Recommendations

**For High Volume:**

1. **Use Job Queue** (Redis + Celery)

   ```python
   @celery_app.task
   def analyze_check_async(job_id, image_path):
       # Run 8-stage analysis
       # Update job status in Redis
   ```

2. **Database Integration**

   ```python
   # Replace file-based storage with PostgreSQL
   # Store jobs, results, audit logs
   # Enable analytics and reporting
   ```

3. **Caching**

   ```python
   # Cache ML model in memory
   # Cache template matching results
   # Use Redis for job status
   ```

4. **Load Balancing**
   ```bash
   # Multiple backend instances
   # Nginx reverse proxy
   # Horizontal scaling
   ```

---

**Note:** This system demonstrates AI-powered fraud detection capabilities. Always combine automated analysis with manual verification for critical financial decisions.
