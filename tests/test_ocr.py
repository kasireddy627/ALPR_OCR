import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ocr_engine import OCREngine

e = OCREngine()

def test_valid_plate_format():
    assert e.is_valid_plate("AP09AB1234") is True

def test_invalid_plate_rejected():
    assert e.is_valid_plate("XY") is False

def test_short_string_rejected():
    assert e.is_valid_plate("AB") is False