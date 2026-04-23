import re
import os
import yaml
import pytesseract
import numpy as np
import cv2
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class OCREngine:
    def __init__(self, config_path=None):
        if config_path is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base, "config.yaml")

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.min_confidence = cfg["ocr"]["min_confidence"]

    def _clean_plate_text(self, text):
        # Remove non alphanumeric
        text = re.sub(r"[^A-Z0-9]", "", text.upper())
        # Find start of valid Indian plate (2 letters)
        match = re.search(r"[A-Z]{2}[0-9]", text)
        if match:
            text = text[match.start():]
        return text

    def read_plate(self, image):
        try:
            # Accept BGR numpy array
            if not isinstance(image, np.ndarray):
                return {"text": "", "confidence": 0.0, "raw_output": ""}

            # Convert to BGR if grayscale
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            h, w = image.shape[:2]
            best_text = ""
            best_conf = 0.0

            # Generate variants
            variants = []

            # 4x upscale grayscale — best performer
            big  = cv2.resize(image, (w*4, h*4),
                              interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
            variants.append(gray)

            # Sharpened
            kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
            sharp  = cv2.filter2D(gray, -1, kernel)
            variants.append(sharp)

            # OTSU
            _, otsu = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants.append(otsu)

            # Slightly trim left (8%) to reduce IND emblem effect
            trim_w  = int(w * 0.08)
            trimmed = image[:, trim_w:]
            big2    = cv2.resize(trimmed, (0,0), fx=4, fy=4,
                                 interpolation=cv2.INTER_CUBIC)
            gray2   = cv2.cvtColor(big2, cv2.COLOR_BGR2GRAY)
            variants.append(gray2)

            config = (
                "--oem 3 --psm 7 "
                "-c tessedit_char_whitelist="
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )

            for img in variants:
                pil = Image.fromarray(img)
                try:
                    raw = pytesseract.image_to_string(
                        pil, config=config).strip()
                    text = self._clean_plate_text(raw)

                    # Get confidence
                    data = pytesseract.image_to_data(
                        pil, config=config,
                        output_type=pytesseract.Output.DICT)
                    confs = [
                        float(c) for c in data["conf"]
                        if str(c).strip() not in ["-1", ""]
                        and float(c) > 0
                    ]
                    conf = float(np.mean(confs)) if confs else 50.0

                    if len(text) >= 4 and conf > best_conf:
                        best_text = text
                        best_conf = conf

                except Exception:
                    continue

            # If confidence filtering killing results, use raw text
            if not best_text:
                for img in variants:
                    pil = Image.fromarray(img)
                    try:
                        raw  = pytesseract.image_to_string(
                            pil, config=config).strip()
                        text = self._clean_plate_text(raw)
                        if len(text) >= 4:
                            best_text = text
                            best_conf = 50.0
                            break
                    except Exception:
                        continue

            return {
                "text":       best_text,
                "confidence": best_conf,
                "raw_output": best_text
            }

        except Exception as e:
            return {"text": "", "confidence": 0.0, "raw_output": str(e)}

    def is_valid_plate(self, text):
        if not text or not isinstance(text, str):
            return False
        indian = r"^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$"
        if re.match(indian, text.upper()):
            return True
        if 4 <= len(text) <= 12 and re.match(r"^[A-Z0-9]+$", text.upper()):
            return True
        return False