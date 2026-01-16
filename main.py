"""
Customs Committee Passport MRZ Scanner - Backend API
Uses Mistral Vision (pixtral-12b-2409) with Strict ICAO 9303 Grid Parsing
"""

import io
import os
import time
import base64
import hashlib
import json
import re
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
    title="Customs Committee Passport Scanner",
    description="Professional MRZ Scanner with Mistral Vision",
    version="4.0.0"
)

# CORS middleware
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
# MISTRAL VISION SCANNER
# ============================================

class MistralVisionScanner:
    """
    Mistral Vision API Scanner using pixtral-12b-2409
    Implements strict character-by-character grid parsing
    """

    def __init__(self, api_key: str):
        self.client = Mistral(api_key=api_key)
        self.vision_model = "pixtral-12b-2409"
        print(f"🤖 Mistral Vision Scanner initialized with {self.vision_model}", flush=True)

    def scan_mrz_strip(self, image_bytes: bytes) -> Dict:
        """
        Scan MRZ strip image using Mistral Vision API with Grid Method
        Returns raw Line 1 and Line 2 strings
        """
        # Calculate image hash for debugging
        import hashlib
        image_hash = hashlib.md5(image_bytes).hexdigest()[:8]
        print(f"🔍 Scanning MRZ strip ({len(image_bytes)} bytes, hash: {image_hash})...", flush=True)

        try:
            # Convert to base64
            base64_image = base64.b64encode(image_bytes).decode('utf-8')

            # System prompt for strict character-by-character grid reading
            system_prompt = """ROLE: Strict ICAO 9303 MRZ Decoder.

INPUT: A cropped image containing ONLY 2 lines of MRZ text (Machine Readable Zone).

INSTRUCTION: Transcribe characters visually in a strict grid format. Read each character position carefully.

MRZ FORMAT (TD3 Passport - 44 characters per line):
LINE 1: Type + Country + Surname + << + Given Names + <<< (filler to 44 chars)
LINE 2: Passport# + Check + Country + BirthDate + Check + Sex + ExpiryDate + Check + PINFL + Check + Composite

LINE 2 DECODING RULES (Fixed Character Indices):
- Position 0-8: Passport Number (9 chars) - First 2 MUST be LETTERS, rest DIGITS
- Position 9: Check digit
- Position 10-12: Nationality (3 chars)
- Position 13-18: Birth Date YYMMDD (6 digits)
- Position 19: Check digit
- Position 20: Sex (1 char: M or F)
- Position 21-26: Expiry Date YYMMDD (6 digits)
- Position 27: Check digit
- Position 28-41: Personal Number/PINFL (14 chars for UZB)
- Position 42: Check digit
- Position 43: Composite check digit

OCR ERROR CORRECTIONS:
1. If nationality is 'ZBO', 'LZB', 'USB', 'U2B', 'UZ8', 'O2B' -> Force to 'UZB'
2. In date fields (birth/expiry): If you see letter 'O', convert to digit '0'
3. In passport number: First 2 chars MUST be letters (convert 0->O), remaining MUST be digits (convert O->0)
4. Filler character '<' may appear as chevron, use '<' in output

CRITICAL: Each line must be EXACTLY 44 characters. Use '<' as filler if needed.

OUTPUT FORMAT - EXTREMELY IMPORTANT:
You MUST respond with ONLY the JSON object below. Do NOT add:
- Any explanation or commentary
- Markdown code blocks (no ```)
- Additional text before or after the JSON
- Newlines or formatting around the JSON

Return EXACTLY this structure and nothing else:
{"line1": "44-character line 1 from the image", "line2": "44-character line 2 from the image"}

IMPORTANT: Read the ACTUAL characters from the uploaded image. Do NOT use placeholder or example data. Transcribe what you SEE in the image."""

            # Call Mistral Vision API
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    ]
                }
            ]

            print(f"📤 Sending request to Mistral Vision API...", flush=True)

            # Retry logic for transient errors (503, network issues, etc.)
            max_retries = 4
            retry_delays = [2, 4, 8, 16]  # Exponential backoff in seconds
            last_error = None

            for attempt in range(max_retries):
                try:
                    response = self.client.chat.complete(
                        model=self.vision_model,
                        messages=messages,
                        temperature=0.0,  # Zero temperature for deterministic OCR
                        max_tokens=500
                        # Note: response_format not supported for vision models
                    )
                    break  # Success, exit retry loop

                except Exception as api_error:
                    last_error = api_error
                    error_str = str(api_error).lower()

                    # Check if it's a retryable error
                    is_retryable = (
                        '503' in error_str or
                        'service unavailable' in error_str or
                        'timeout' in error_str or
                        'upstream connect error' in error_str or
                        'reset' in error_str or
                        'overflow' in error_str or
                        'network' in error_str
                    )

                    if is_retryable and attempt < max_retries - 1:
                        delay = retry_delays[attempt]
                        print(f"⚠️  API error (attempt {attempt + 1}/{max_retries}): {str(api_error)}", flush=True)
                        print(f"🔄 Retrying in {delay} seconds...", flush=True)
                        time.sleep(delay)
                    else:
                        # Not retryable or last attempt, re-raise
                        raise

            # Extract response content
            if hasattr(response, 'choices') and len(response.choices) > 0:
                content = response.choices[0].message.content
                print(f"📥 Received response: {content[:200]}...", flush=True)

                # Try multiple extraction methods
                result = None

                # Method 1: Try to extract JSON from markdown code blocks
                json_match = re.search(r'```(?:json)?\s*\n?(\{.*?\})\s*\n?```', content, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(1))
                        print(f"✅ Extracted JSON from markdown code block", flush=True)
                    except json.JSONDecodeError:
                        pass

                # Method 2: Try to find JSON object anywhere in the response
                if not result:
                    json_match = re.search(r'\{[^{}]*"line1"[^{}]*"line2"[^{}]*\}', content, re.DOTALL)
                    if json_match:
                        try:
                            result = json.loads(json_match.group(0))
                            print(f"✅ Extracted JSON from response text", flush=True)
                        except json.JSONDecodeError:
                            pass

                # Method 3: Try direct JSON parsing after simple cleanup
                if not result:
                    try:
                        cleaned = content.strip()
                        cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
                        cleaned = re.sub(r'\s*```$', '', cleaned)
                        cleaned = cleaned.strip()
                        result = json.loads(cleaned)
                        print(f"✅ Parsed JSON after cleanup", flush=True)
                    except json.JSONDecodeError:
                        pass

                # Method 4: Fallback - extract line1 and line2 using regex
                if not result:
                    print(f"⚠️  JSON parsing failed, trying regex fallback...", flush=True)
                    print(f"📄 Full response:\n{content}", flush=True)

                    line1_match = re.search(r'"line1"\s*:\s*"([^"]*)"', content)
                    line2_match = re.search(r'"line2"\s*:\s*"([^"]*)"', content)

                    if line1_match and line2_match:
                        result = {
                            "line1": line1_match.group(1),
                            "line2": line2_match.group(1)
                        }
                        print(f"✅ Extracted using regex fallback", flush=True)
                    else:
                        raise ValueError(f"Failed to extract MRZ lines. Response: {content[:500]}")

                if not result:
                    raise ValueError(f"Failed to parse response. Content: {content[:500]}")

                if "line1" not in result or "line2" not in result:
                    raise ValueError("Missing line1 or line2 in response")

                # Normalize to exactly 44 characters
                result["line1"] = self._normalize_line(result["line1"])
                result["line2"] = self._normalize_line(result["line2"])

                print(f"✅ MRZ extracted:", flush=True)
                print(f"   Line1: {result['line1']}", flush=True)
                print(f"   Line2: {result['line2']}", flush=True)

                return result
            else:
                raise Exception("Empty response from Mistral Vision API")

        except Exception as e:
            print(f"❌ Mistral Vision API error: {str(e)}", flush=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to scan MRZ: {str(e)}"
            )

    def _normalize_line(self, line: str) -> str:
        """Normalize MRZ line to exactly 44 characters"""
        line = line.strip().upper()

        # Remove any spaces
        line = line.replace(' ', '')

        # Truncate or pad
        if len(line) > 44:
            line = line[:44]
        elif len(line) < 44:
            line = line.ljust(44, '<')

        return line


# ============================================
# STRICT ICAO 9303 GRID PARSER
# ============================================

class StrictMRZParser:
    """
    Strict ICAO 9303 TD3 Parser with Fixed Character Positions
    Implements character-by-character grid parsing
    """

    @staticmethod
    def parse_mrz_strict(line1: str, line2: str) -> Dict:
        """
        Parse MRZ using STRICT fixed-position grid parsing
        Line 1: P<CCCSURNAME<<GIVEN<NAMES<<<...
        Line 2: NNNNNNNNNNCYYYMMDDCSXYYYMMDDCPPPPPPPPPPPPPPCC
        """
        print(f"🔍 Parsing MRZ with Strict Grid Method:", flush=True)
        print(f"   Line1: {line1}", flush=True)
        print(f"   Line2: {line2}", flush=True)

        if len(line1) != 44 or len(line2) != 44:
            raise ValueError(f"Invalid MRZ length. Line1: {len(line1)}, Line2: {len(line2)}")

        # ===== LINE 1: Name Parsing =====
        doc_type = line1[0]  # 'P' for passport
        country_code = line1[2:5].replace('<', '').strip()

        # Names section (position 5-43)
        names_section = line1[5:44]

        # Split by double chevron
        if '<<' in names_section:
            parts = names_section.split('<<')
            surname = parts[0].replace('<', ' ').strip()
            given_names = parts[1].replace('<', ' ').strip() if len(parts) > 1 else ""
        else:
            surname = names_section.replace('<', ' ').strip()
            given_names = ""

        # ===== LINE 2: Fixed Grid Parsing =====
        passport_number_raw = line2[0:9]
        passport_check = line2[9]
        nationality_raw = line2[10:13]
        dob_raw = line2[13:19]
        dob_check = line2[19]
        sex_raw = line2[20]
        expiry_raw = line2[21:27]
        expiry_check = line2[27]
        pinfl_raw = line2[28:42]
        pinfl_check = line2[42]
        composite_check = line2[43]

        # ===== OCR ERROR CORRECTIONS =====

        # 1. Passport Number: First 2 chars = LETTERS, rest = DIGITS
        passport_cleaned = passport_number_raw.replace('<', '').strip()
        if len(passport_cleaned) >= 2:
            prefix = passport_cleaned[:2]
            suffix = passport_cleaned[2:]

            # Fix prefix (letters): 0->O, 1->I
            prefix = prefix.replace('0', 'O').replace('1', 'I')

            # Fix suffix (digits): O->0, I->1, l->1
            suffix = suffix.replace('O', '0').replace('o', '0')
            suffix = suffix.replace('I', '1').replace('l', '1')

            passport_number = prefix + suffix
        else:
            passport_number = passport_cleaned

        # 2. Nationality: Fix common OCR errors for UZB
        nationality = nationality_raw.replace('<', '').strip()
        if nationality in ['ZBO', 'LZB', 'USB', 'U2B', 'UZ8', 'UZD', '028', 'O2B', '0ZB']:
            nationality = 'UZB'

        # 3. Dates: O->0
        dob = dob_raw.replace('O', '0').replace('o', '0')
        expiry = expiry_raw.replace('O', '0').replace('o', '0')

        # 4. PINFL: O->0
        pinfl = pinfl_raw.replace('<', '').strip()
        pinfl = pinfl.replace('O', '0').replace('o', '0')

        # 5. Sex
        sex = sex_raw.replace('<', '').strip()

        # ===== POST-PROCESSING: Force Nationality =====
        # If nationality is not UZB but passport starts with known UZB prefixes
        if nationality != 'UZB' and len(passport_number) >= 2:
            uzb_prefixes = ['FA', 'FB', 'FC', 'FD', 'AC', 'AD', 'AA', 'AB']
            if passport_number[:2].upper() in uzb_prefixes:
                print(f"⚠️  Forcing nationality to UZB (passport prefix: {passport_number[:2]})", flush=True)
                nationality = 'UZB'

        # ===== DATE FORMATTING & VALIDATION =====
        birth_date = StrictMRZParser.format_date(dob)
        expiry_date = StrictMRZParser.format_date(expiry)

        # Date sanity check
        if not StrictMRZParser.validate_date(birth_date):
            print(f"⚠️  Invalid birth date: {birth_date}", flush=True)
            birth_date = f"ERROR:{dob}"

        if not StrictMRZParser.validate_date(expiry_date):
            print(f"⚠️  Invalid expiry date: {expiry_date}", flush=True)
            expiry_date = f"ERROR:{expiry}"

        # ===== CHECKSUM VALIDATION =====
        validations = {
            "passport_valid": StrictMRZParser.validate_checksum(passport_number_raw, passport_check),
            "dob_valid": StrictMRZParser.validate_checksum(dob_raw, dob_check),
            "expiry_valid": StrictMRZParser.validate_checksum(expiry_raw, expiry_check),
            "pinfl_valid": StrictMRZParser.validate_checksum(pinfl_raw, pinfl_check),
        }

        # Composite check
        composite_data = line2[0:10] + line2[13:20] + line2[21:43]
        validations["composite_valid"] = StrictMRZParser.validate_checksum(composite_data, composite_check)

        all_valid = all(validations.values())

        print(f"✅ Data Extracted Successfully:", flush=True)
        print(f"   Passport: {passport_number}", flush=True)
        print(f"   Name: {given_names} {surname}", flush=True)
        print(f"   PINFL: {pinfl}", flush=True)
        print(f"   Checksum Validation: {'PASS ✓' if all_valid else 'WARNING ⚠️'}", flush=True)

        # Debug: Show which validations failed
        if not all_valid:
            print(f"   ℹ️  Checksum details (data is still usable):", flush=True)
            for check_name, is_valid in validations.items():
                status = "✓" if is_valid else "⚠️"
                print(f"      {status} {check_name}", flush=True)
            print(f"   Note: Invalid checksums may indicate OCR errors or test/damaged passport", flush=True)

        return {
            "passport_number": passport_number,
            "surname": surname,
            "name": given_names,
            "given_names": given_names,
            "birth_date": birth_date,
            "date_of_birth": birth_date,
            "expiry_date": expiry_date,
            "date_of_expiry": expiry_date,
            "sex": sex if sex in ['M', 'F'] else 'F',
            "nationality": nationality,
            "pinfl": pinfl,
            "personal_number": pinfl,
            "document_type": doc_type,
            "country_code": country_code,
            "validations": validations,
            "validation_status": "VALID" if all_valid else "WARNING",
            "validation_message": "All checksums valid" if all_valid else "Checksums invalid - may indicate OCR errors or damaged passport (data still extracted)",
            "raw_mrz": {
                "line1": line1,
                "line2": line2
            }
        }

    @staticmethod
    def format_date(yymmdd: str) -> str:
        """Convert YYMMDD to DD.MM.YYYY"""
        if len(yymmdd) != 6 or not yymmdd.isdigit():
            return yymmdd

        yy = int(yymmdd[0:2])
        mm = yymmdd[2:4]
        dd = yymmdd[4:6]

        # Century determination
        yyyy = 2000 + yy if yy < 50 else 1900 + yy

        return f"{dd}.{mm}.{yyyy}"

    @staticmethod
    def validate_date(date_str: str) -> bool:
        """Validate DD.MM.YYYY format"""
        if date_str.startswith("ERROR:"):
            return False

        try:
            parts = date_str.split('.')
            if len(parts) != 3:
                return False

            dd, mm, yyyy = int(parts[0]), int(parts[1]), int(parts[2])

            # Basic sanity checks
            if mm < 1 or mm > 12:
                return False
            if dd < 1 or dd > 31:
                return False
            if yyyy < 1900 or yyyy > 2100:
                return False

            return True
        except:
            return False

    @staticmethod
    def char_to_value(char: str) -> int:
        """Convert MRZ character to numeric value"""
        if char.isdigit():
            return int(char)
        elif char.isalpha():
            return ord(char) - ord('A') + 10
        else:
            return 0

    @staticmethod
    def calculate_checksum(data: str) -> int:
        """Calculate ICAO 9303 checksum (mod 10, weights 7-3-1)"""
        weights = [7, 3, 1]
        total = 0

        for i, char in enumerate(data):
            value = StrictMRZParser.char_to_value(char)
            weight = weights[i % 3]
            total += value * weight

        return total % 10

    @staticmethod
    def validate_checksum(data: str, check_digit: str) -> bool:
        """Validate checksum"""
        if not check_digit.isdigit():
            return False

        calculated = StrictMRZParser.calculate_checksum(data)
        expected = int(check_digit)

        return calculated == expected


# ============================================
# SECURITY & RATE LIMITING
# ============================================

request_timestamps = {}
RATE_LIMIT_WINDOW = 60
MAX_REQUESTS_PER_WINDOW = 30
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png']

def check_rate_limit(client_id: str) -> bool:
    """Rate limiting"""
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

def validate_image(file: UploadFile, contents: bytes) -> None:
    """Validate uploaded image"""
    # Check size
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    # Check extension
    if file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Invalid file type")

    # Verify image
    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
    except:
        raise HTTPException(status_code=400, detail="Invalid image file")


# ============================================
# API ENDPOINTS
# ============================================

# Initialize Mistral Scanner
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "JWSVnIJhbnyhc80PY32AhKkxEbS4SFFi")

if os.environ.get("MISTRAL_API_KEY"):
    print(f"✅ Using MISTRAL_API_KEY from environment", flush=True)
else:
    print(f"⚠️  Using hardcoded Mistral API key", flush=True)

mistral_scanner = MistralVisionScanner(api_key=MISTRAL_API_KEY)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve frontend"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "service": "customs-passport-scanner",
        "version": "4.0.0",
        "model": "pixtral-12b-2409"
    }


@app.post("/scan")
async def scan_passport(request: Request, file: UploadFile = File(...)):
    """
    Main endpoint: Scan passport MRZ using Mistral Vision
    Expects a cropped MRZ strip image from frontend
    """
    try:
        # Rate limiting
        client_id = request.client.host if request.client else "unknown"

        if not check_rate_limit(client_id):
            raise HTTPException(status_code=429, detail="Too many requests")

        # Read file
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")

        # Validate
        validate_image(file, contents)

        print(f"📸 Processing MRZ strip: {file.filename} ({len(contents)} bytes)", flush=True)

        # Scan with Mistral Vision
        mrz_result = mistral_scanner.scan_mrz_strip(contents)

        # Parse with strict grid method
        parsed_data = StrictMRZParser.parse_mrz_strict(
            mrz_result["line1"],
            mrz_result["line2"]
        )

        # Add metadata
        parsed_data["scan_metadata"] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "file_name": file.filename,
            "file_size": len(contents),
            "file_hash": hashlib.sha256(contents).hexdigest()[:16],
            "scanner": "Mistral Vision (pixtral-12b-2409)",
            "method": "Strict Grid Parsing"
        }

        print(f"✅ Scan completed: {parsed_data['passport_number']}", flush=True)

        return JSONResponse(content={
            "success": True,
            "data": parsed_data
        })

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        print(f"❌ Error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")


@app.get("/test")
async def test_endpoint():
    """Test endpoint"""
    return {
        "message": "Customs Committee Passport Scanner",
        "version": "4.0.0",
        "status": "operational",
        "model": "pixtral-12b-2409",
        "method": "Strict Grid Parsing"
    }


@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    """Telegram webhook endpoint"""
    try:
        update_data = await request.json()

        from bot import get_application
        application = get_application()

        update = Update.de_json(update_data, application.bot)
        await application.process_update(update)

        return JSONResponse(content={"ok": True})

    except Exception as e:
        print(f"❌ Webhook error: {str(e)}", flush=True)
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
