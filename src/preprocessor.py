import cv2
import numpy as np
import yaml
import os


class ALPRPreprocessor:
    def __init__(self, config_path=None):
        pass

    def process(self, crop):
        if crop is None or crop.size == 0:
            return crop

        # Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Small safe trim to remove emblem (not aggressive)
        h, w = gray.shape
        gray = gray[:, int(w * 0.08):]

        # Resize (critical for OCR)
        gray = cv2.resize(
            gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC
        )

        # Light denoise (preserves edges)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        return gray