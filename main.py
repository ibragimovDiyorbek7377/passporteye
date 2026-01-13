"""
Customs-Grade Passport Scanner - FastAPI Backend
ICAO 9303 TD3 Standard Compliant MRZ Parser with Strict Validation

Author: Senior Lead Engineer - Computer Vision & GovTech
Target: Customs Committee of Uzbekistan
"""

import io
import os
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from passporteye import read_mrz
from PIL import Image
import re

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


class MRZParser:
    """
    ICAO 9303 TD3 Format MRZ Parser
    TD3: Machine-readable travel documents (44 characters per line, 2 lines)
    Used for passports
    """

    def __init__(self):
        self.validator = ICAOValidator()

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocess image for better MRZ detection
        - Convert to grayscale
        - Apply thresholding
        - Enhance contrast
        """
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Encode back to bytes
        _, buffer = cv2.imencode('.png', thresh)

        return buffer.tobytes()

    def parse_td3_line1(self, line: str) -> Dict:
        """
        Parse TD3 Line 1 (44 characters)
        Format: P<UTONATIONS<<SURNAME<<GIVEN<NAMES<<<<<<<<<
        Positions:
        1: P (Passport)
        2: < (filler)
        3-5: Issuing country code
        6-44: Names (surname, given names separated by <<)
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
        Format: L898902C<3UTO6908061F9406236UZB<<<<<<<<<<<6

        Positions:
        1-9: Passport number
        10: Check digit for passport number
        11-13: Nationality
        14-19: Date of birth (YYMMDD)
        20: Check digit for DOB
        21: Sex (M/F/<)
        22-27: Date of expiry (YYMMDD)
        28: Check digit for expiry
        29-42: Personal number (JSHSHIR/PNFL) - CRITICAL for Uzbekistan
        43: Check digit for personal number
        44: Final composite check digit
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

        # Composite check validates: passport + passport_check + dob + dob_check + expiry + expiry_check + personal + personal_check
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


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the Telegram Mini App frontend"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint for Railway"""
    return {"status": "healthy", "service": "bojxona-passport-scanner"}


@app.post("/scan")
async def scan_passport(file: UploadFile = File(...)):
    """
    Main endpoint for passport scanning
    Accepts image file and returns parsed MRZ data with ICAO validation
    """
    try:
        # Read uploaded file
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Initialize parser
        parser = MRZParser()

        # Preprocess image for better OCR
        preprocessed = parser.preprocess_image(contents)

        # Use passporteye for MRZ extraction
        mrz = read_mrz(io.BytesIO(preprocessed))

        if mrz is None:
            # Try with original image if preprocessing didn't work
            mrz = read_mrz(io.BytesIO(contents))

        if mrz is None:
            raise HTTPException(
                status_code=422,
                detail="MRZ not detected. Please ensure the passport is clearly visible and well-lit."
            )

        # Get MRZ text
        mrz_data = mrz.to_dict()

        # Extract raw MRZ lines
        if 'raw_text' in mrz_data:
            lines = mrz_data['raw_text']
        else:
            # Fallback: construct from mrz object
            lines = [mrz.mrz_text[0:44], mrz.mrz_text[44:88]] if hasattr(mrz, 'mrz_text') else None

        if not lines or len(lines) < 2:
            raise HTTPException(
                status_code=422,
                detail="Could not extract MRZ lines. TD3 format requires 2 lines of 44 characters."
            )

        # Parse with strict ICAO validation
        parsed_data = parser.parse_mrz(lines[0], lines[1])

        # Add scanning metadata
        parsed_data["scan_metadata"] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "file_name": file.filename,
            "file_size": len(contents)
        }

        return JSONResponse(content={
            "success": True,
            "data": parsed_data
        })

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify API is running"""
    return {
        "message": "Bojxona Passport Scanner API",
        "version": "1.0.0",
        "status": "operational",
        "features": [
            "ICAO 9303 TD3 Parsing",
            "Checksum Validation",
            "MRZ Preprocessing",
            "JSHSHIR/PNFL Extraction"
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
