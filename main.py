"""
Customs-Grade Passport Scanner - FastAPI Backend
ICAO 9303 TD3 Standard Compliant MRZ Parser with Strict Validation

Author: Senior Lead Engineer - Computer Vision & GovTech
Target: Customs Committee of Uzbekistan
"""

import io
import os
import hashlib
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from passporteye import read_mrz
from PIL import Image, ImageEnhance, ImageOps
import time

app = FastAPI(
    title="Bojxona Passport Scanner",
    description="Customs-Grade ICAO 9303 TD3 Passport Scanner",
    version="1.0.0"
)

# CORS middleware for Telegram Mini App
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates
templates = Jinja2Templates(directory="templates")

# ============================================
# CYBERSECURITY FEATURES
# ============================================

# Rate limiting storage (in-memory for simplicity)
request_timestamps = {}
RATE_LIMIT_WINDOW = 60  # seconds
MAX_REQUESTS_PER_WINDOW = 10

# File size limits (10MB max)
MAX_FILE_SIZE = 10 * 1024 * 1024

# Allowed file types
ALLOWED_MIME_TYPES = ['image/jpeg', 'image/jpg', 'image/png']
ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png']


def check_rate_limit(client_id: str) -> bool:
    """
    Simple rate limiting: Max 10 requests per 60 seconds per client
    """
    current_time = time.time()

    if client_id not in request_timestamps:
        request_timestamps[client_id] = []

    # Remove old timestamps outside the window
    request_timestamps[client_id] = [
        ts for ts in request_timestamps[client_id]
        if current_time - ts < RATE_LIMIT_WINDOW
    ]

    # Check if limit exceeded
    if len(request_timestamps[client_id]) >= MAX_REQUESTS_PER_WINDOW:
        return False

    # Add current request
    request_timestamps[client_id].append(current_time)
    return True


def validate_image_file(file: UploadFile, contents: bytes) -> None:
    """
    Validate uploaded file for security
    - Check file size
    - Check file extension
    - Validate it's actually an image
    """
    # Check file size
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024)}MB"
        )

    # Check extension
    if file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )

    # Validate it's actually an image using PIL
    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()  # Verify it's not corrupted
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image file"
        )


# ============================================
# IMAGE PROCESSING FOR BETTER OCR
# ============================================

def opencv_preprocess_for_mrz(image_bytes: bytes) -> Optional[bytes]:
    """
    Use OpenCV to preprocess image specifically for MRZ OCR
    Returns enhanced image bytes or None if preprocessing fails
    """
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter (reduces noise, keeps edges sharp)
        filtered = cv2.bilateralFilter(gray, 5, 50, 50)

        # Adaptive thresholding for better MRZ contrast
        thresh = cv2.adaptiveThreshold(
            filtered, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15, 10
        )

        # Encode back to bytes (PNG for lossless)
        success, buffer = cv2.imencode('.png', thresh)
        if success:
            return buffer.tobytes()

        return None
    except Exception as e:
        print(f"   ⚠️ OpenCV preprocessing failed: {e}")
        return None


def enhance_image_for_ocr(image: Image.Image) -> List[Image.Image]:
    """
    Create optimized versions for MRZ OCR
    Returns 5 best strategies for passport scanning
    """
    images = []

    # Strategy 1: Original image (PassportEye's default)
    images.append(image.copy())

    # Strategy 2: Auto-orient using EXIF (fixes rotated photos)
    try:
        oriented = ImageOps.exif_transpose(image)
        if oriented is not None and oriented != image:
            images.append(oriented)
    except:
        pass

    # Strategy 3: Enhanced contrast (better for faded passports)
    try:
        enhancer = ImageEnhance.Contrast(image)
        images.append(enhancer.enhance(2.0))
    except:
        pass

    # Strategy 4: High contrast grayscale (BEST for MRZ)
    try:
        gray = ImageOps.grayscale(image)
        enhancer = ImageEnhance.Contrast(gray)
        images.append(enhancer.enhance(2.5))
    except:
        pass

    # Strategy 5: Sharpened + contrast (for blurry photos)
    try:
        enhancer = ImageEnhance.Sharpness(image)
        sharpened = enhancer.enhance(2.5)
        contrast_enhancer = ImageEnhance.Contrast(sharpened)
        images.append(contrast_enhancer.enhance(1.8))
    except:
        pass

    return images


def try_multiple_ocr_strategies(contents: bytes) -> Optional[object]:
    """
    Try multiple OCR strategies to detect MRZ
    Uses BOTH PIL and OpenCV preprocessing for maximum success rate
    """
    # Load image
    try:
        image = Image.open(io.BytesIO(contents))
        print(f"📐 Image loaded: {image.size[0]}×{image.size[1]}, Format: {image.format}, Mode: {image.mode}")
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        return None

    # Strategy 0: Try OpenCV preprocessing FIRST (best for MRZ)
    print(f"🔧 Strategy #0: OpenCV MRZ preprocessing")
    opencv_preprocessed = opencv_preprocess_for_mrz(contents)
    if opencv_preprocessed:
        try:
            print(f"   ✓ OpenCV preprocessing successful")
            mrz = read_mrz(io.BytesIO(opencv_preprocessed))

            if mrz is not None and hasattr(mrz, 'mrz_text'):
                mrz_len = len(mrz.mrz_text) if mrz.mrz_text else 0
                print(f"   ✓ MRZ text length: {mrz_len} chars")
                if mrz_len >= 88:
                    print(f"✅ MRZ detected using OpenCV preprocessing!")
                    print(f"   MRZ Preview: {mrz.mrz_text[:44]}...")
                    return mrz
                else:
                    print(f"   ✗ MRZ too short")
            else:
                print(f"   ✗ No valid MRZ from OpenCV")
        except Exception as e:
            print(f"   ✗ OpenCV strategy failed: {e}")

    # Get multiple PIL-enhanced versions
    enhanced_images = enhance_image_for_ocr(image)
    print(f"🔄 Generated {len(enhanced_images)} PIL strategies")

    # Try OCR on each PIL version
    for idx, img in enumerate(enhanced_images):
        try:
            print(f"   Strategy #{idx + 1}: {img.size[0]}×{img.size[1]}, mode={img.mode}")

            # Convert back to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)

            # Try PassportEye OCR
            mrz = read_mrz(img_bytes)

            if mrz is not None:
                print(f"   ✓ PassportEye returned result")
                if hasattr(mrz, 'mrz_text'):
                    mrz_len = len(mrz.mrz_text) if mrz.mrz_text else 0
                    print(f"   ✓ MRZ text length: {mrz_len} chars")
                    if mrz_len >= 88:
                        print(f"✅ MRZ detected using PIL strategy #{idx + 1}")
                        print(f"   MRZ Preview: {mrz.mrz_text[:44]}...")
                        return mrz
                    else:
                        print(f"   ✗ MRZ too short (need 88+)")
                else:
                    print(f"   ✗ No mrz_text attribute")
            else:
                print(f"   ✗ PassportEye returned None")

        except Exception as e:
            print(f"   ✗ Strategy #{idx + 1} failed: {type(e).__name__}: {str(e)[:100]}")
            continue

    print(f"❌ All strategies failed (1 OpenCV + {len(enhanced_images)} PIL)")
    return None


# ============================================
# ICAO 9303 VALIDATOR
# ============================================

class ICAOValidator:
    """
    ICAO 9303 Checksum Validator
    Uses Modulus 10 algorithm with weights 7, 3, 1
    """

    @staticmethod
    def char_to_value(char: str) -> int:
        """Convert MRZ character to numeric value for checksum calculation"""
        if char.isdigit():
            return int(char)
        elif char.isalpha():
            return ord(char) - ord('A') + 10
        elif char == '<':
            return 0
        else:
            return 0

    @staticmethod
    def calculate_checksum(data: str) -> int:
        """
        Calculate ICAO 9303 checksum
        Formula: Sum of (value * weight) mod 10
        Weights cycle: 7, 3, 1
        """
        weights = [7, 3, 1]
        total = 0

        for i, char in enumerate(data):
            value = ICAOValidator.char_to_value(char)
            weight = weights[i % 3]
            total += value * weight

        return total % 10

    @staticmethod
    def validate_checksum(data: str, check_digit: str) -> bool:
        """Validate data against its check digit"""
        if not check_digit.isdigit():
            return False

        calculated = ICAOValidator.calculate_checksum(data)
        expected = int(check_digit)

        return calculated == expected

    @staticmethod
    def validate_date(date_str: str) -> bool:
        """Validate date format YYMMDD"""
        if len(date_str) != 6 or not date_str.isdigit():
            return False

        try:
            year = int(date_str[0:2])
            month = int(date_str[2:4])
            day = int(date_str[4:6])

            # Basic range checks
            if month < 1 or month > 12:
                return False
            if day < 1 or day > 31:
                return False

            return True
        except ValueError:
            return False


# ============================================
# MRZ PARSER
# ============================================

class MRZParser:
    """
    ICAO 9303 TD3 Format MRZ Parser
    TD3: Machine-readable travel documents (44 characters per line, 2 lines)
    Used for passports
    """

    def __init__(self):
        self.validator = ICAOValidator()

    def parse_td3_line1(self, line: str) -> Dict:
        """
        Parse TD3 Line 1 (44 characters)
        Format: P<UTONATIONS<<SURNAME<<GIVEN<NAMES<<<<<<<<<
        """
        if len(line) != 44:
            raise ValueError(f"Line 1 must be 44 characters, got {len(line)}")

        doc_type = line[0]
        country_code = line[2:5].replace('<', '')

        # Parse names (position 5-44)
        names_section = line[5:44].replace('<', ' ').strip()
        name_parts = [part for part in names_section.split('  ') if part]

        surname = name_parts[0] if name_parts else ""
        given_names = ' '.join(name_parts[1:]) if len(name_parts) > 1 else ""

        return {
            "document_type": doc_type,
            "country_code": country_code,
            "surname": surname.strip(),
            "given_names": given_names.strip()
        }

    def parse_td3_line2(self, line: str) -> Dict:
        """
        Parse TD3 Line 2 (44 characters)
        Extracts: Passport #, DOB, Sex, Expiry, JSHSHIR/PNFL
        """
        if len(line) != 44:
            raise ValueError(f"Line 2 must be 44 characters, got {len(line)}")

        # Extract fields
        passport_number = line[0:9].replace('<', '').strip()
        passport_check = line[9]

        nationality = line[10:13].replace('<', '').strip()

        dob = line[13:19]
        dob_check = line[19]

        sex = line[20].replace('<', '')

        expiry = line[21:27]
        expiry_check = line[27]

        # CRITICAL: Personal Number (JSHSHIR/PNFL) for Uzbekistan
        personal_number = line[28:42].replace('<', '').strip()
        personal_check = line[42]

        composite_check = line[43]

        # Validate checksums
        validations = {
            "passport_number_valid": self.validator.validate_checksum(line[0:9], passport_check),
            "dob_valid": self.validator.validate_checksum(dob, dob_check),
            "expiry_valid": self.validator.validate_checksum(expiry, expiry_check),
            "personal_number_valid": self.validator.validate_checksum(line[28:42], personal_check),
        }

        # Composite check
        composite_data = line[0:10] + line[13:20] + line[21:43]
        validations["composite_valid"] = self.validator.validate_checksum(composite_data, composite_check)

        # Validate dates
        validations["dob_format_valid"] = self.validator.validate_date(dob)
        validations["expiry_format_valid"] = self.validator.validate_date(expiry)

        return {
            "passport_number": passport_number,
            "nationality": nationality,
            "date_of_birth": self._format_date(dob),
            "date_of_birth_raw": dob,
            "sex": sex if sex in ['M', 'F'] else 'Unknown',
            "date_of_expiry": self._format_date(expiry),
            "date_of_expiry_raw": expiry,
            "personal_number": personal_number,  # JSHSHIR/PNFL
            "validations": validations
        }

    def _format_date(self, yymmdd: str) -> str:
        """Convert YYMMDD to DD.MM.YYYY format"""
        if len(yymmdd) != 6:
            return yymmdd

        yy = int(yymmdd[0:2])
        mm = yymmdd[2:4]
        dd = yymmdd[4:6]

        # Determine century (assume 20xx for years < 50, 19xx otherwise)
        yyyy = 2000 + yy if yy < 50 else 1900 + yy

        return f"{dd}.{mm}.{yyyy}"

    def parse_mrz(self, line1: str, line2: str) -> Dict:
        """Parse complete TD3 MRZ (2 lines)"""
        # Clean lines
        line1 = line1.strip().upper()
        line2 = line2.strip().upper()

        # Parse both lines
        data_line1 = self.parse_td3_line1(line1)
        data_line2 = self.parse_td3_line2(line2)

        # Combine results
        result = {
            **data_line1,
            **data_line2,
            "raw_mrz": {
                "line1": line1,
                "line2": line2
            }
        }

        # Overall validation status
        all_checks_valid = all(data_line2["validations"].values())
        result["validation_status"] = "PASS" if all_checks_valid else "FAIL"

        return result


# ============================================
# API ENDPOINTS
# ============================================

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the Telegram Mini App frontend"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint for Railway"""
    return {"status": "healthy", "service": "bojxona-passport-scanner"}


@app.post("/scan")
async def scan_passport(request: Request, file: UploadFile = File(...)):
    """
    Main endpoint for passport scanning
    Uses PassportEye with multiple enhancement strategies
    """
    try:
        # Get client identifier for rate limiting
        client_id = request.client.host if request.client else "unknown"

        # Check rate limit
        if not check_rate_limit(client_id):
            raise HTTPException(
                status_code=429,
                detail="Juda ko'p so'rov. Iltimos, bir oz kuting."
            )

        # Read uploaded file
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Bo'sh fayl yuklandi")

        # Validate file (security check)
        validate_image_file(file, contents)

        # Initialize parser
        parser = MRZParser()

        # Try multiple OCR strategies
        print("🔍 Starting MRZ detection with multiple strategies...")
        mrz = try_multiple_ocr_strategies(contents)

        if mrz is None:
            raise HTTPException(
                status_code=422,
                detail="❌ Pasport MRZ topilmadi.\n\n"
                       "📋 Iltimos, quyidagilarni ta'minlang:\n"
                       "• Pasportni tekis joyga qo'ying\n"
                       "• Yaxshi yoritilgan joyda suratga oling\n"
                       "• MRZ (pastdagi 2 qator) aniq ko'rinsin\n"
                       "• Pasportni to'g'ri yo'nalishda tuting\n"
                       "• Kamera fokusda bo'lsin"
            )

        # Extract MRZ text from passporteye result
        mrz_text = mrz.mrz_text if hasattr(mrz, 'mrz_text') else None

        if not mrz_text or len(mrz_text) < 88:
            raise HTTPException(
                status_code=422,
                detail="❌ MRZ matnini to'liq o'qib bo'lmadi.\n\n"
                       "Iltimos, yaxshi yoritilgan va aniq rasm oling."
            )

        # Split into two lines (TD3 format: 44 chars per line)
        line1 = mrz_text[0:44]
        line2 = mrz_text[44:88]

        print(f"✅ MRZ Lines extracted:")
        print(f"   Line 1: {line1}")
        print(f"   Line 2: {line2}")

        # Parse with strict ICAO validation
        parsed_data = parser.parse_mrz(line1, line2)

        # Add scanning metadata
        parsed_data["scan_metadata"] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "file_name": file.filename,
            "file_size": len(contents),
            "file_hash": hashlib.sha256(contents).hexdigest()[:16]
        }

        print(f"✅ Passport scanned successfully: {parsed_data['passport_number']}")

        return JSONResponse(content={
            "success": True,
            "data": parsed_data
        })

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        # Log error but don't expose internal details
        print(f"❌ Error scanning passport: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="⚠️ Xatolik yuz berdi.\n\n"
                   "Iltimos, rasmni qayta yuklang yoki yordam so'rang."
        )


@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify API is running"""
    return {
        "message": "Bojxona Passport Scanner API",
        "version": "2.0.0",
        "status": "operational",
        "features": [
            "ICAO 9303 TD3 Parsing",
            "Checksum Validation",
            "PassportEye OCR with Multiple Strategies",
            "JSHSHIR/PNFL Extraction",
            "Rate Limiting",
            "File Validation",
            "Image Enhancement",
            "Auto-rotation Support"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
