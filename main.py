"""
Customs Committee Passport MRZ Scanner - Backend API
Uses Customs AI Vision with Strict ICAO 9303 Grid Parsing
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
    description="Professional MRZ Scanner System",
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
# AI VISION SCANNER (INTERNAL ENGINE)
# ============================================

class MistralVisionScanner:
    """
    Internal AI Engine for MRZ Extraction
    """

    def __init__(self, api_key: str):
        self.client = Mistral(api_key=api_key)
        self.vision_model = "pixtral-12b-2409"
        print(f"🤖 Customs AI Engine initialized", flush=True)

    def scan_mrz_strip(self, image_bytes: bytes) -> Dict:
        """
        Scan MRZ strip image using AI Vision
        """
        # Calculate image hash for debugging
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

OUTPUT FORMAT:
Return EXACTLY this JSON structure and nothing else:
{"line1": "44-character line 1 from the image", "line2": "44-character line 2 from the image"}
"""

            # Call AI API
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

            print(f"📤 Sending request to AI Engine...", flush=True)

            # Retry logic
            max_retries = 4
            retry_delays = [2, 4, 8, 16]
            
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.complete(
                        model=self.vision_model,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=500
                    )
                    break 
                except Exception as api_error:
                    error_str = str(api_error).lower()
                    is_retryable = any(x in error_str for x in ['503', 'timeout', 'network', 'rate limit'])
                    
                    if is_retryable and attempt < max_retries - 1:
                        delay = retry_delays[attempt]
                        print(f"⚠️ AI Engine busy (attempt {attempt + 1}), retrying in {delay}s...", flush=True)
                        time.sleep(delay)
                    else:
                        raise

            # Extract response content
            if hasattr(response, 'choices') and len(response.choices) > 0:
                content = response.choices[0].message.content
                print(f"📥 Received response from AI Engine", flush=True)

                result = None

                # Method 1: Markdown JSON
                json_match = re.search(r'```(?:json)?\s*\n?(\{.*?\})\s*\n?```', content, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(1))
                    except: pass

                # Method 2: Raw JSON
                if not result:
                    json_match = re.search(r'\{[^{}]*"line1"[^{}]*"line2"[^{}]*\}', content, re.DOTALL)
                    if json_match:
                        try:
                            result = json.loads(json_match.group(0))
                        except: pass

                # Method 3: Regex Fallback
                if not result:
                    print(f"⚠️ Standard parsing failed, using fallback...", flush=True)
                    line1_match = re.search(r'"line1"\s*:\s*"([^"]*)"', content)
                    line2_match = re.search(r'"line2"\s*:\s*"([^"]*)"', content)
                    if line1_match and line2_match:
                        result = {
                            "line1": line1_match.group(1),
                            "line2": line2_match.group(1)
                        }

                if not result:
                    raise ValueError(f"Failed to parse AI response. Content: {content[:100]}...")

                if "line1" not in result or "line2" not in result:
                    raise ValueError("Incomplete data received from AI")

                # Normalize to exactly 44 characters
                result["line1"] = self._normalize_line(result["line1"])
                result["line2"] = self._normalize_line(result["line2"])

                return result
            else:
                raise Exception("Empty response from AI Engine")

        except Exception as e:
            print(f"❌ AI Engine Error: {str(e)}", flush=True)
            raise HTTPException(
                status_code=500,
                detail=f"Scan failed: {str(e)}"
            )

    def _normalize_line(self, line: str) -> str:
        """Normalize MRZ line to exactly 44 characters"""
        line = line.strip().upper()
        line = line.replace(' ', '')
        
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
    """

    @staticmethod
    def parse_mrz_strict(line1: str, line2: str) -> Dict:
        """
        Parse MRZ using STRICT fixed-position grid parsing
        """
        print(f"🔍 Parsing MRZ Data...", flush=True)

        if len(line1) != 44 or len(line2) != 44:
            raise ValueError(f"Invalid MRZ length. Line1: {len(line1)}, Line2: {len(line2)}")

        # ===== LINE 1: Name Parsing =====
        doc_type = line1[0]
        country_code = line1[2:5].replace('<', '').strip()
        names_section = line1[5:44]

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
        
        # 1. Passport Number
        passport_cleaned = passport_number_raw.replace('<', '').strip()
        if len(passport_cleaned) >= 2:
            prefix = passport_cleaned[:2].replace('0', 'O').replace('1', 'I')
            suffix = passport_cleaned[2:].replace('O', '0').replace('o', '0').replace('I', '1').replace('l', '1')
            passport_number = prefix + suffix
        else:
            passport_number = passport_cleaned

        # 2. Nationality
        nationality = nationality_raw.replace('<', '').strip()
        if nationality in ['ZBO', 'LZB', 'USB', 'U2B', 'UZ8', 'UZD', '028', 'O2B', '0ZB']:
            nationality = 'UZB'

        # 3. Dates & PINFL
        dob = dob_raw.replace('O', '0').replace('o', '0')
        expiry = expiry_raw.replace('O', '0').replace('o', '0')
        pinfl = pinfl_raw.replace('<', '').strip().replace('O', '0').replace('o', '0')
        sex = sex_raw.replace('<', '').strip()

        # Force Nationality if prefix matches
        if nationality != 'UZB' and len(passport_number) >= 2:
            uzb_prefixes = ['FA', 'FB', 'FC', 'FD', 'AC', 'AD', 'AA', 'AB']
            if passport_number[:2].upper() in uzb_prefixes:
                nationality = 'UZB'

        # ===== DATE FORMATTING =====
        birth_date = StrictMRZParser.format_date(dob)
        expiry_date = StrictMRZParser.format_date(expiry)

        # ===== VALIDATION =====
        validations = {
            "passport_valid": StrictMRZParser.validate_checksum(passport_number_raw, passport_check),
            "dob_valid": StrictMRZParser.validate_checksum(dob_raw, dob_check),
            "expiry_valid": StrictMRZParser.validate_checksum(expiry_raw, expiry_check),
            "pinfl_valid": StrictMRZParser.validate_checksum(pinfl_raw, pinfl_check),
        }
        
        composite_data = line2[0:10] + line2[13:20] + line2[21:43]
        validations["composite_valid"] = StrictMRZParser.validate_checksum(composite_data, composite_check)

        all_valid = all(validations.values())

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
            "validation_message": "All checksums valid" if all_valid else "Checksum warning (data extracted)",
            "raw_mrz": {
                "line1": line1,
                "line2": line2
            }
        }

    @staticmethod
    def format_date(yymmdd: str) -> str:
        if len(yymmdd) != 6 or not yymmdd.isdigit(): return yymmdd
        yy, mm, dd = int(yymmdd[0:2]), yymmdd[2:4], yymmdd[4:6]
        yyyy = 2000 + yy if yy < 50 else 1900 + yy
        return f"{dd}.{mm}.{yyyy}"

    @staticmethod
    def validate_date(date_str: str) -> bool:
        if date_str.startswith("ERROR"): return False
        try:
            parts = date_str.split('.')
            return len(parts) == 3
        except: return False

    @staticmethod
    def char_to_value(char: str) -> int:
        if char.isdigit(): return int(char)
        elif char.isalpha(): return ord(char) - ord('A') + 10
        else: return 0

    @staticmethod
    def calculate_checksum(data: str) -> int:
        weights = [7, 3, 1]
        total = 0
        for i, char in enumerate(data):
            total += StrictMRZParser.char_to_value(char) * weights[i % 3]
        return total % 10

    @staticmethod
    def validate_checksum(data: str, check_digit: str) -> bool:
        if not check_digit.isdigit(): return False
        return StrictMRZParser.calculate_checksum(data) == int(check_digit)


# ============================================
# SECURITY & RATE LIMITING
# ============================================

request_timestamps = {}
RATE_LIMIT_WINDOW = 60
MAX_REQUESTS_PER_WINDOW = 30
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png']

def check_rate_limit(client_id: str) -> bool:
    current_time = time.time()
    if client_id not in request_timestamps: request_timestamps[client_id] = []
    request_timestamps[client_id] = [ts for ts in request_timestamps[client_id] if current_time - ts < RATE_LIMIT_WINDOW]
    if len(request_timestamps[client_id]) >= MAX_REQUESTS_PER_WINDOW: return False
    request_timestamps[client_id].append(current_time)
    return True

def validate_image(file: UploadFile, contents: bytes) -> None:
    if len(contents) > MAX_FILE_SIZE: raise HTTPException(status_code=413, detail="File too large")
    if file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS: raise HTTPException(status_code=400, detail="Invalid file type")
    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
    except: raise HTTPException(status_code=400, detail="Invalid image file")


# ============================================
# API ENDPOINTS
# ============================================

# Initialize Scanner (Variable name kept generic)
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "------------------------------")
if os.environ.get("MISTRAL_API_KEY"):
    print(f"✅ API Key loaded from environment", flush=True)
else:
    print(f"⚠️  Using fallback API key", flush=True)

# Scanner instance
scanner_engine = MistralVisionScanner(api_key=MISTRAL_API_KEY)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "customs-passport-scanner",
        "version": "4.0.0",
        "system": "Customs-Vision-12B"  # Branding changed
    }

@app.post("/scan")
async def scan_passport(request: Request, file: UploadFile = File(...)):
    """
    Main endpoint: Scan passport MRZ
    """
    try:
        # Rate limiting
        client_id = request.client.host if request.client else "unknown"
        if not check_rate_limit(client_id):
            raise HTTPException(status_code=429, detail="Too many requests")

        # Read file
        contents = await file.read()
        if not contents: raise HTTPException(status_code=400, detail="Empty file")
        validate_image(file, contents)

        print(f"📸 Processing MRZ strip: {file.filename}", flush=True)

        # 1. Scan with AI Engine
        mrz_result = scanner_engine.scan_mrz_strip(contents)

        # 2. Parse with Strict Grid
        parsed_data = StrictMRZParser.parse_mrz_strict(
            mrz_result["line1"],
            mrz_result["line2"]
        )

        # 3. Add Metadata (BRANDING REMOVED HERE)
        parsed_data["scan_metadata"] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "file_name": file.filename,
            "file_size": len(contents),
            "file_hash": hashlib.sha256(contents).hexdigest()[:16],
            "scanner": "Customs AI Scanner (v4.0)",  # <--- CHANGED FROM MISTRAL
            "method": "Strict Grid Parsing"
        }

        print(f"✅ Scan completed: {parsed_data['passport_number']}", flush=True)

        return JSONResponse(content={
            "success": True,
            "data": parsed_data
        })

    except HTTPException: raise
    except ValueError as ve: raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        print(f"❌ Error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")

@app.get("/test")
async def test_endpoint():
    return {
        "message": "Customs Committee Passport Scanner",
        "version": "4.0.0",
        "status": "operational",
        "system": "Customs-Vision-12B"
    }

@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
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
