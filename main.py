"""
Telegram Mini App Backend - Passport MRZ Scanner using Mistral AI Vision API
Production-ready FastAPI application with intelligent MRZ extraction
"""

import io
import os
import time
import base64
import hashlib
import json
from datetime import datetime
from typing import Dict, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from mistralai import Mistral
from PIL import Image
from telegram import Update

app = FastAPI(
    title="Passport Scanner with Mistral AI",
    description="Telegram Mini App for Passport MRZ Scanning",
    version="3.0.0"
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
# MISTRAL AI VISION SCANNER
# ============================================

class MistralMRZScanner:
    """
    Mistral AI Vision API Manager for MRZ Extraction
    Uses Mistral OCR-2512 model for passport scanning
    """

    def __init__(self, api_key: str):
        # Initialize with 60-second timeout to prevent hanging
        self.client = Mistral(
            api_key=api_key,
            timeout_ms=60000  # 60 second timeout for API calls
        )
        self.model = "mistral-ocr-2512"
        print(f"🤖 Mistral MRZ Scanner initialized with model: {self.model}")
        print(f"⏱️  Timeout configured: 60 seconds")

    def scan_passport_mrz(self, image_bytes: bytes, max_retries: int = 3) -> Dict:
        """
        Scan passport MRZ using Mistral Vision API
        Returns extracted MRZ data with retry logic
        """
        print(f"🔍 Starting MRZ scan with Mistral AI...")
        print(f"📊 Image size: {len(image_bytes)} bytes")
        print(f"🔄 Max retries: {max_retries}")

        attempts = 0
        last_error = None

        while attempts < max_retries:
            try:
                # Convert image to base64
                base64_image = base64.b64encode(image_bytes).decode('utf-8')

                print(f"🤖 Sending request to Mistral AI (attempt {attempts + 1})...")

                # Mistral Vision API prompt
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Analyze the provided image which is a cropped Machine Readable Zone (MRZ) from a passport.

CRITICAL REQUIREMENTS:
1. Extract EXACTLY 2 lines of MRZ text
2. Each line MUST be EXACTLY 44 characters
3. Preserve ALL characters including "<" symbols (angle brackets)
4. Do NOT add spaces, dashes, or modify any characters
5. The MRZ follows ICAO 9303 TD3 standard format

RESPONSE FORMAT - Return ONLY this JSON structure (no markdown, no code blocks, no explanations):
{
  "line1": "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<",
  "line2": "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
}

MRZ FORMAT SPECIFICATION:
- Line 1 (44 chars): Document type (P) + Country code (3) + Name (39 chars with << separators)
- Line 2 (44 chars): Passport# + Check + Nationality + DOB + Check + Sex + Expiry + Check + Personal# + Check + Composite check

CRITICAL RULES:
- Use "<" (angle bracket) for ALL filler characters
- Do NOT use spaces, underscores, or any other character for fillers
- Count carefully to ensure exactly 44 characters per line
- Preserve exact character positions
- Return ONLY the raw JSON object

If MRZ is unreadable or image quality is too poor, return:
{"error": "MRZ_NOT_READABLE"}"""
                            },
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        ]
                    }
                ]

                # Send to Mistral
                response = self.client.chat.complete(
                    model=self.model,
                    messages=messages
                )

                if not response or not response.choices:
                    raise Exception("Empty response from Mistral AI")

                response_text = response.choices[0].message.content
                print(f"✅ Received response from Mistral AI")

                # Parse response
                result = self._parse_mistral_response(response_text)

                if "error" in result:
                    raise Exception(result["error"])

                # Validate MRZ format
                if not self._validate_mrz_format(result):
                    raise Exception("Invalid MRZ format from Mistral AI")

                print(f"✅ MRZ extracted successfully")
                return result

            except Exception as e:
                error_msg = str(e).lower()
                error_type = type(e).__name__
                last_error = e
                attempts += 1

                print(f"❌ Attempt {attempts} failed ({error_type}): {str(e)[:150]}")

                # Check if error is timeout-related
                if "timeout" in error_msg or "timed out" in error_msg:
                    print(f"⏱️  Timeout error - Mistral API did not respond in time")
                    if attempts < max_retries:
                        print(f"⚠️ Retrying with exponential backoff...")
                        time.sleep(2 ** attempts)  # Exponential backoff: 2s, 4s, 8s
                    continue
                # Check if error is related to rate limit
                elif "429" in error_msg or "rate limit" in error_msg:
                    print(f"⚠️ Rate limit error detected, waiting before retry...")
                    time.sleep(3)
                    continue
                # Check if error is authentication/API key issue
                elif "401" in error_msg or "unauthorized" in error_msg or "api key" in error_msg:
                    print(f"🔑 Authentication error - Invalid or expired Mistral API key")
                    raise HTTPException(
                        status_code=500,
                        detail="Mistral API authentication failed. Please check API key configuration."
                    )
                elif "503" in error_msg or "500" in error_msg:
                    print(f"⚠️ Server error, waiting before retry...")
                    time.sleep(1)
                    continue
                else:
                    # Other errors - still retry
                    if attempts < max_retries:
                        print(f"⚠️ Unknown error, retrying...")
                        time.sleep(0.5)
                    continue

        # All retries failed
        error_type = type(last_error).__name__
        error_detail = str(last_error)[:200]

        print(f"❌ All {max_retries} attempts failed. Last error type: {error_type}")

        raise HTTPException(
            status_code=500,
            detail=f"Failed to scan passport after {max_retries} attempts. Error: {error_type} - {error_detail}"
        )

    def _parse_mistral_response(self, response_text: str) -> Dict:
        """Parse Mistral response and extract JSON"""
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
            raise Exception("Invalid JSON response from Mistral AI")

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
MAX_REQUESTS_PER_WINDOW = 20
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png']

def check_rate_limit(client_id: str) -> bool:
    """Rate limiting: Max 20 requests per 60 seconds per client"""
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

# Initialize Mistral Scanner
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "JWSVnIJhbnyhc80PY32AhKkxEbS4SFFi")

# Log API key configuration
if os.environ.get("MISTRAL_API_KEY"):
    print(f"✅ Using MISTRAL_API_KEY from environment variable")
else:
    print(f"⚠️  WARNING: Using hardcoded Mistral API key (set MISTRAL_API_KEY env var)")

mistral_scanner = MistralMRZScanner(api_key=MISTRAL_API_KEY)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the Telegram Mini App frontend"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "passport-scanner-mistral",
        "version": "3.0.0"
    }


@app.post("/scan")
async def scan_passport(request: Request, file: UploadFile = File(...)):
    """
    Main endpoint for passport scanning using Mistral AI Vision API
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

        # Scan with Mistral AI
        print("🤖 Scanning passport with Mistral AI Vision...")
        mrz_result = mistral_scanner.scan_passport_mrz(contents)

        # Parse MRZ
        print("📋 Parsing MRZ data...")
        parsed_data = MRZParser.parse_mrz(mrz_result["line1"], mrz_result["line2"])

        # Add metadata
        parsed_data["scan_metadata"] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "file_name": file.filename,
            "file_size": len(contents),
            "file_hash": hashlib.sha256(contents).hexdigest()[:16],
            "scanner": "Mistral AI OCR-2512"
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
        "message": "Passport Scanner with Mistral AI Vision",
        "version": "3.0.0",
        "status": "operational",
        "features": [
            "Mistral AI OCR-2512 Vision API",
            "Smart MRZ Cropping",
            "ICAO 9303 TD3 Parsing",
            "Checksum Validation",
            "Rate Limiting",
            "File Validation"
        ]
    }


@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    """
    Webhook endpoint for Telegram bot updates
    This endpoint receives updates from Telegram when running on Railway
    """
    try:
        # Get the update data from Telegram
        update_data = await request.json()

        # Import bot application (lazy import to avoid circular dependency)
        from bot import get_application

        # Get the Telegram application instance
        application = get_application()

        # Process the update
        update = Update.de_json(update_data, application.bot)
        await application.process_update(update)

        return JSONResponse(content={"ok": True})

    except Exception as e:
        print(f"❌ Error processing webhook: {str(e)}")
        # Return 200 anyway to avoid Telegram retrying
        return JSONResponse(content={"ok": False, "error": str(e)}, status_code=200)


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
