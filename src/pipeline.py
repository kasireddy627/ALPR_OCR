import cv2
import numpy as np
import yaml
import os

from src.detector import PlateDetector
from src.preprocessor import ALPRPreprocessor
from src.ocr_engine import OCREngine
from src.tracker import PlateTracker
from src.database import ALPRDatabase


class ALPRPipeline:
    def __init__(self, config_path=None):
        if config_path is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base, "config.yaml")

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.detector = PlateDetector(config_path)
        self.preprocessor = ALPRPreprocessor(config_path)
        self.ocr = OCREngine(config_path)
        self.tracker = PlateTracker(config_path)
        self.database = ALPRDatabase(config_path)

    def process_frame(self, frame, frame_number=0):
        if frame is None or frame.size == 0:
            return frame

        annotated = frame.copy()

        detections = self.detector.detect(frame)
        tracked = self.tracker.update(detections)

        def find_crop(tracked_bbox):
            for det in detections:
                db = det["bbox"]
                if (
                    abs(db[0] - tracked_bbox[0]) < 20 and
                    abs(db[1] - tracked_bbox[1]) < 20
                ):
                    return det["crop"]
            return None

        for obj in tracked:
            track_id = obj["track_id"]
            x1, y1, x2, y2 = map(int, obj["bbox"])

            crop = find_crop(obj["bbox"])

            plate_text = ""
            confidence = 0.0

            if crop is not None and crop.size > 0:

                # CORRECT FLOW
                cleaned = self.preprocessor.process(crop)
                result = self.ocr.read_plate(crop)

                plate_text = result["text"]
                confidence = result["confidence"]

                if self.ocr.is_valid_plate(plate_text):
                    self.database.insert_read(
                        track_id, plate_text, confidence, frame_number
                    )

            color = (0, 255, 0) if plate_text else (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"ID:{track_id}"
            if plate_text:
                label += f" {plate_text}"

            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        return annotated