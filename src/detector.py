import cv2
import numpy as np
import yaml
import os
from ultralytics import YOLO


class PlateDetector:
    def __init__(self, config_path=None):
        if config_path is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base, "config.yaml")

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        model_cfg = cfg["model"]

        self.confidence_threshold = model_cfg["confidence_threshold"]
        self.iou_threshold = model_cfg["iou_threshold"]

        weights_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            model_cfg["weights_path"]
        )

        self.model = YOLO(weights_path)

    def detect(self, frame):
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]

        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )

        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                x1 = int(np.clip(x1, 0, w - 1))
                y1 = int(np.clip(y1, 0, h - 1))
                x2 = int(np.clip(x2, 0, w - 1))
                y2 = int(np.clip(y2, 0, h - 1))

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2].copy()

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "crop": crop
                })

        return detections