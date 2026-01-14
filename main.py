"""
Telegram Mini App Backend - Passport MRZ Scanner using Google Gemini API
Production-ready FastAPI application with round-robin key rotation
"""

import io
import os
import time
import base64
import hashlib
from datetime import datetime
from typing import Dict, Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import google.generativeai as genai
from PIL import Image

app = FastAPI(
    title="Passport Scanner with Gemini AI",
    description="Telegram Mini App for Passport MRZ Scanning",
    version="2.0.0"
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
# GEMINI API MANAGER WITH KEY ROTATION
# ============================================

class GeminiScanner:
    """
    Google Gemini API Manager with Round-Robin Key Rotation
    Automatically switches to next key on quota/rate limit errors
    """

    KEYS = [
        "AIzaSyCN0V0joAnErmhLMFwFMegUJ9RkWtuxCvE",
        "AIzaSyCqDUJPZDtoG6dNryRmp4kDqY1jtx0RPJE",
        "AIzaSyBC3dXFbl5UAzOyWOHcFrYZt6snyPreZbU",
        "AIzaSyAoNsY7ZwOODYuMzUjIZ5McWnxRTVdvpNk",
        "AIzaSyAOsDzsTtHd1RO29pDZcGQ2ECps6XfPrCA",
        "AIzaSyAp4MDPxjD23Fo5bUI7SD3HhWBzy2eZLuE",
        "AIzaSyBv8bBb3Tv6gRSWNfEu3bnxBYwR4_8DcRI",
        "AIzaSyAyR6N_WXX6H1a67aTQlb66P8ytVRjvmTo",
        "AIzaSyBYlJ7vzzxSoFy1sGuMgPOtSSWW5Mlmw8M"
    ]

    def __init__(self):
        self.current_key_index = 0
        self.total_keys = len(self.KEYS)
        self.configure_current_key()
        print(f"🔑 Gemini Scanner initialized with {self.total_keys} API keys")

    def configure_current_key(self):
        """Configure Gemini with current API key"""
        api_key = self.KEYS[self.current_key_index]
        genai.configure(api_key=api_key)
        print(f"🔧 Using API key #{self.current_key_index + 1}")

    def rotate_key(self):
        """Rotate to next API key (Round-Robin)"""
        self.current_key_index = (self.current_key_index + 1) % self.total_keys
        self.configure_current_key()
        print(f"🔄 Rotated to API key #{self.current_key_index + 1}")

    def scan_passport_mrz(self, image_bytes: bytes, max_retries: int = 3) -> Dict:
        """
        Scan passport MRZ using Gemini Vision API
        Automatically retries with different API keys on quota errors
        """
        attempts = 0
        last_error = None

        while attempts < max_retries:
            try:
                # Load image
                image = Image.open(io.BytesIO(image_bytes))

                # Initialize Gemini Vision model
                model = genai.GenerativeModel('gemini-1.5-flash')

                # Optimized prompt for MRZ extraction
                prompt = """You are an expert passport MRZ (Machine Readable Zone) scanner.

Analyze this passport image and extract ONLY the MRZ data (the two lines of text at the bottom of the passport).

CRITICAL REQUIREMENTS:
1. Extract exactly 2 lines, each line must be EXACTLY 44 characters
2. Preserve ALL characters including "<" symbols
3. Do NOT add spaces or modify characters
4. The MRZ follows ICAO 9303 TD3 format

Format your response as JSON with this EXACT structure:
{
  "line1": "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<",
  "line2": "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
}

Rules:
- Line 1 format: Document type (P) + Country code + Name (surname<<given<names)
- Line 2 format: Passport number + Check digit + Nationality + DOB + Sex + Expiry + Personal number
- Each line MUST be exactly 44 characters
- Use "<" for filler spaces
- Return ONLY valid JSON, no markdown, no explanations

If you cannot detect the MRZ clearly, return:
{"error": "MRZ_NOT_FOUND"}"""

                print(f"🤖 Sending request to Gemini API (key #{self.current_key_index + 1})...")

                # Send to Gemini
                response = model.generate_content([prompt, image])

                if not response or not response.text:
                    raise Exception("Empty response from Gemini API")

                print(f"✅ Received response from Gemini")

                # Parse response
                result = self._parse_gemini_response(response.text)

                if "error" in result:
                    raise Exception(result["error"])

                # Validate MRZ format
                if not self._validate_mrz_format(result):
                    raise Exception("Invalid MRZ format from Gemini")

                print(f"✅ MRZ extracted successfully")
                return result

            except Exception as e:
                error_msg = str(e).lower()
                last_error = e
                attempts += 1

                print(f"❌ Attempt {attempts} failed: {str(e)[:100]}")

                # Check if error is related to quota/rate limit
                if "429" in error_msg or "quota" in error_msg or "rate limit" in error_msg:
                    print(f"⚠️ Quota/Rate limit error detected, rotating key...")
                    self.rotate_key()
                    time.sleep(0.5)  # Brief delay before retry
                    continue
                elif "503" in error_msg or "500" in error_msg:
                    print(f"⚠️ Server error, rotating key and retrying...")
                    self.rotate_key()
                    time.sleep(1)
                    continue
                else:
                    # Other errors - still try rotating
                    if attempts < max_retries:
                        print(f"⚠️ Unknown error, trying next key...")
                        self.rotate_key()
                        time.sleep(0.5)
                    continue

        # All retries failed
        raise HTTPException(
            status_code=500,
            detail=f"Failed to scan passport after {max_retries} attempts. Last error: {str(last_error)}"
        )

    def _parse_gemini_response(self, response_text: str) -> Dict:
        """Parse Gemini response and extract JSON"""
        import json
        import re

        # Clean response (remove markdown code blocks if present)
        cleaned = response_text.strip()

        # Remove markdown code blocks
        cleaned = re.sub(r'^```json\s*', '', cleaned)
        cleaned = re.sub(r'^```\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        cleaned = cleaned.strip()

        try:
            result = json.loads(cleaned)
            return result
        except json.JSONDecodeError as e:
            print(f"❌ JSON parse error: {e}")
            print(f"Response text: {response_text[:200]}")
            raise Exception("Invalid JSON response from Gemini")

    def _validate_mrz_format(self, result: Dict) -> bool:
        """Validate MRZ format (2 lines, 44 chars each)"""
        if "line1" not in result or "line2" not in result:
            print(f"❌ Missing line1 or line2 in response")
            return False

        line1 = result["line1"]
        line2 = result["line2"]

        if len(line1) != 44:
            print(f"❌ Line1 length is {len(line1)}, expected 44")
            return False

        if len(line2) != 44:
            print(f"❌ Line2 length is {len(line2)}, expected 44")
            return False

        return True


# ============================================
# ICAO 9303 MRZ PARSER
# ============================================

class MRZParser:
    """Parse and validate ICAO 9303 TD3 format MRZ"""

    @staticmethod
    def char_to_value(char: str) -> int:
        """Convert MRZ character to numeric value for checksum"""
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
        """Calculate ICAO 9303 checksum using mod 10 with weights 7,3,1"""
        weights = [7, 3, 1]
        total = 0

        for i, char in enumerate(data):
            value = MRZParser.char_to_value(char)
            weight = weights[i % 3]
            total += value * weight

        return total % 10

    @staticmethod
    def validate_checksum(data: str, check_digit: str) -> bool:
        """Validate data against its check digit"""
        if not check_digit.isdigit():
            return False

        calculated = MRZParser.calculate_checksum(data)
        expected = int(check_digit)

        return calculated == expected

    @staticmethod
    def format_date(yymmdd: str) -> str:
        """Convert YYMMDD to DD.MM.YYYY format"""
        if len(yymmdd) != 6 or not yymmdd.isdigit():
            return yymmdd

        yy = int(yymmdd[0:2])
        mm = yymmdd[2:4]
        dd = yymmdd[4:6]

        # Determine century (20xx for years < 50, 19xx otherwise)
        yyyy = 2000 + yy if yy < 50 else 1900 + yy

        return f"{dd}.{mm}.{yyyy}"

    @staticmethod
    def parse_mrz(line1: str, line2: str) -> Dict:
        """
        Parse complete TD3 MRZ (2 lines x 44 chars)
        Line 1: Document type, Country, Names
        Line 2: Passport #, Nationality, DOB, Sex, Expiry, Personal #
        """
        # Clean lines
        line1 = line1.strip().upper()
        line2 = line2.strip().upper()

        if len(line1) != 44 or len(line2) != 44:
            raise ValueError(f"Invalid MRZ format. Line1: {len(line1)}, Line2: {len(line2)}")

        # Parse Line 1
        doc_type = line1[0]
        country_code = line1[2:5].replace('<', '').strip()

        names_section = line1[5:44].replace('<', ' ').strip()
        name_parts = [part for part in names_section.split('  ') if part]

        surname = name_parts[0] if name_parts else ""
        given_names = ' '.join(name_parts[1:]) if len(name_parts) > 1 else ""

        # Parse Line 2
        passport_number = line2[0:9].replace('<', '').strip()
        passport_check = line2[9]

        nationality = line2[10:13].replace('<', '').strip()

        dob = line2[13:19]
        dob_check = line2[19]

        sex = line2[20].replace('<', '')

        expiry = line2[21:27]
        expiry_check = line2[27]

        personal_number = line2[28:42].replace('<', '').strip()
        personal_check = line2[42]

        composite_check = line2[43]

        # Validate checksums
        validations = {
            "passport_number_valid": MRZParser.validate_checksum(line2[0:9], passport_check),
            "dob_valid": MRZParser.validate_checksum(dob, dob_check),
            "expiry_valid": MRZParser.validate_checksum(expiry, expiry_check),
            "personal_number_valid": MRZParser.validate_checksum(line2[28:42], personal_check),
        }

        # Composite check
        composite_data = line2[0:10] + line2[13:20] + line2[21:43]
        validations["composite_valid"] = MRZParser.validate_checksum(composite_data, composite_check)

        # Validate date formats
        validations["dob_format_valid"] = len(dob) == 6 and dob.isdigit()
        validations["expiry_format_valid"] = len(expiry) == 6 and expiry.isdigit()

        # Overall validation
        all_checks_valid = all(validations.values())

        return {
            "document_type": doc_type,
            "country_code": country_code,
            "surname": surname.strip(),
            "given_names": given_names.strip(),
            "passport_number": passport_number,
            "nationality": nationality,
            "date_of_birth": MRZParser.format_date(dob),
            "date_of_birth_raw": dob,
            "sex": sex if sex in ['M', 'F'] else 'Unknown',
            "date_of_expiry": MRZParser.format_date(expiry),
            "date_of_expiry_raw": expiry,
            "personal_number": personal_number,
            "validations": validations,
            "validation_status": "PASS" if all_checks_valid else "FAIL",
            "raw_mrz": {
                "line1": line1,
                "line2": line2
            }
        }


# ============================================
# SECURITY & RATE LIMITING
# ============================================

request_timestamps = {}
RATE_LIMIT_WINDOW = 60
MAX_REQUESTS_PER_WINDOW = 10
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png']

def check_rate_limit(client_id: str) -> bool:
    """Rate limiting: Max 10 requests per 60 seconds per client"""
    current_time = time.time()

    if client_id not in request_timestamps:
        request_timestamps[client_id] = []

    # Remove old timestamps
    request_timestamps[client_id] = [
        ts for ts in request_timestamps[client_id]
        if current_time - ts < RATE_LIMIT_WINDOW
    ]

    # Check limit
    if len(request_timestamps[client_id]) >= MAX_REQUESTS_PER_WINDOW:
        return False

    request_timestamps[client_id].append(current_time)
    return True

def validate_image_file(file: UploadFile, contents: bytes) -> None:
    """Validate uploaded file for security"""
    # Check file size
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {MAX_FILE_SIZE / (1024*1024)}MB"
        )

    # Check extension
    if file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )

    # Validate it's actually an image
    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image file"
        )


# ============================================
# API ENDPOINTS
# ============================================

# Initialize Gemini Scanner
gemini_scanner = GeminiScanner()

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the Telegram Mini App frontend"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "passport-scanner-gemini",
        "version": "2.0.0"
    }


@app.post("/scan")
async def scan_passport(request: Request, file: UploadFile = File(...)):
    """
    Main endpoint for passport scanning using Google Gemini API
    """
    try:
        # Rate limiting
        client_id = request.client.host if request.client else "unknown"

        if not check_rate_limit(client_id):
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please wait a moment."
            )

        # Read file
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Validate file
        validate_image_file(file, contents)

        print(f"📸 Processing image: {file.filename} ({len(contents)} bytes)")

        # Scan with Gemini API
        print("🤖 Scanning passport with Gemini AI...")
        mrz_result = gemini_scanner.scan_passport_mrz(contents)

        # Parse MRZ
        print("📋 Parsing MRZ data...")
        parsed_data = MRZParser.parse_mrz(mrz_result["line1"], mrz_result["line2"])

        # Add metadata
        parsed_data["scan_metadata"] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "file_name": file.filename,
            "file_size": len(contents),
            "file_hash": hashlib.sha256(contents).hexdigest()[:16],
            "scanner": "Google Gemini 1.5 Flash"
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
        print(f"❌ Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to scan passport: {str(e)}"
        )


@app.get("/test")
async def test_endpoint():
    """Test endpoint"""
    return {
        "message": "Passport Scanner with Google Gemini AI",
        "version": "2.0.0",
        "status": "operational",
        "features": [
            "Google Gemini 1.5 Flash Vision API",
            "Round-Robin API Key Rotation",
            "ICAO 9303 TD3 Parsing",
            "Checksum Validation",
            "Rate Limiting",
            "File Validation"
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
