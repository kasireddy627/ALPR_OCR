import numpy as np
import yaml
import os
from filterpy.kalman import KalmanFilter


def iou(bb_test, bb_gt):
    """Compute IOU between two bboxes in [x1,y1,x2,y2] format."""
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])

    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    intersection = w * h

    area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area_gt   = (bb_gt[2]   - bb_gt[0])   * (bb_gt[3]   - bb_gt[1])
    union     = area_test + area_gt - intersection

    if union <= 0:
        return 0.0
    return intersection / union


class KalmanBoxTracker:
    """Tracks a single bounding box using a Kalman Filter."""
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition matrix
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ], dtype=float)

        # Measurement matrix
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
        ], dtype=float)

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P         *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = self._bbox_to_z(bbox)

        self.time_since_update = 0
        self.id                = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hit_streak        = 0
        self.age               = 0

    def _bbox_to_z(self, bbox):
        """Convert [x1,y1,x2,y2] to center format [cx,cy,s,r]."""
        bbox = np.array(bbox).flatten()
        w  = float(bbox[2] - bbox[0])
        h  = float(bbox[3] - bbox[1])
        cx = float(bbox[0]) + w / 2.0
        cy = float(bbox[1]) + h / 2.0
        s  = w * h
        r  = w / h if h > 0 else 1.0
        return np.array([[cx], [cy], [s], [r]], dtype=float)
    
    
    def _z_to_bbox(self, x):
        """Convert center format back to [x1,y1,x2,y2]."""
        x = np.array(x).flatten()
        cx = float(x[0])
        cy = float(x[1])
        s  = float(x[2])
        r  = float(x[3])
        w  = np.sqrt(abs(s * r))
        h  = s / w if w > 0 else 0
        return [
            int(cx - w / 2),
            int(cy - h / 2),
            int(cx + w / 2),
            int(cy + h / 2),
        ]

    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self._z_to_bbox(self.kf.x)

    def update(self, bbox):
        self.time_since_update = 0
        self.hit_streak       += 1
        self.kf.update(self._bbox_to_z(bbox))

    def get_state(self):
        return self._z_to_bbox(self.kf.x)


class SORTTracker:
    """Simple SORT multi-object tracker."""

    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self.trackers      = []
        self.frame_count   = 0

    def update(self, detections):
        """
        detections: numpy array of shape (N,5) = [x1,y1,x2,y2,score]
        returns: list of dicts [{track_id, bbox}]
        """
        self.frame_count += 1

        # Predict new locations for all existing trackers
        predicted = []
        to_del    = []
        for i, t in enumerate(self.trackers):
            p = t.predict()
            if any(np.isnan(p)):
                to_del.append(i)
            else:
                predicted.append(p)

        for i in reversed(to_del):
            self.trackers.pop(i)

        # Match detections to trackers using IOU
        matched, unmatched_dets = self._match(detections, predicted)

        # Update matched trackers
        for d_idx, t_idx in matched:
            self.trackers[t_idx].update(detections[d_idx, :4])

        # Create new trackers for unmatched detections
        for d_idx in unmatched_dets:
            self.trackers.append(
                KalmanBoxTracker(detections[d_idx, :4])
            )

        # Collect results and remove dead trackers
        results  = []
        to_del   = []
        for i, t in enumerate(self.trackers):
            if (t.time_since_update <= self.max_age and
                    (t.hit_streak >= self.min_hits or
                     self.frame_count <= self.min_hits)):
                results.append({
                    "track_id": t.id,
                    "bbox":     t.get_state()
                })
            if t.time_since_update > self.max_age:
                to_del.append(i)

        for i in reversed(to_del):
            self.trackers.pop(i)

        return results

    def _match(self, detections, predicted):
        if len(predicted) == 0 or len(detections) == 0:
            return [], list(range(len(detections)))

        iou_matrix = np.zeros((len(detections), len(predicted)))
        for d, det in enumerate(detections):
            for t, pred in enumerate(predicted):
                iou_matrix[d, t] = iou(det[:4], pred)

        # Greedy matching
        matched      = []
        used_dets    = set()
        used_tracks  = set()

        iou_flat = [(iou_matrix[d, t], d, t)
                    for d in range(len(detections))
                    for t in range(len(predicted))]
        iou_flat.sort(key=lambda x: -x[0])

        for score, d, t in iou_flat:
            if d in used_dets or t in used_tracks:
                continue
            if score >= self.iou_threshold:
                matched.append((d, t))
                used_dets.add(d)
                used_tracks.add(t)

        unmatched_dets = [d for d in range(len(detections))
                          if d not in used_dets]
        return matched, unmatched_dets

    def get_active_ids(self):
        return [t.id for t in self.trackers
                if t.time_since_update <= self.max_age]


class PlateTracker:
    def __init__(self, config_path=None):
        if config_path is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base, "config.yaml")

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        t = cfg["tracker"]
        self.sort = SORTTracker(
            max_age=t["max_age"],
            min_hits=t["min_hits"],
            iou_threshold=t["iou_threshold"]
        )

    def update(self, detections):
        if not detections:
            return self.sort.update(np.empty((0, 5)))

        arr = np.array([
            [d["bbox"][0], d["bbox"][1],
             d["bbox"][2], d["bbox"][3],
             d["confidence"]]
            for d in detections
        ], dtype=float)

        return self.sort.update(arr)

    def get_active_ids(self):
        return self.sort.get_active_ids()