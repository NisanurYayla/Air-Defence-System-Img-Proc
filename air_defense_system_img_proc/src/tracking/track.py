import numpy as np


class TrackState:
    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2


class Track:
    def __init__(self, track_id, bbox):
        """
        bbox format: [x1, y1, x2, y2]
        """
        self.id = track_id
        self.bbox = np.array(bbox, dtype=float)

        self.center = self._compute_center(self.bbox)
        self.prev_center = self.center.copy()
        self.velocity = np.array([0.0, 0.0])
        self.alpha = 0.7  # smoothing katsayısı

        self.age = 1
        self.hits = 1
        self.consecutive_hits = 1
        self.time_since_update = 0

        self.state = TrackState.TENTATIVE

        self.history = [self.center.copy()]

    def _compute_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])

    def predict(self):
        """
        Basit lineer tahmin
        """
        predicted_center = self.center + self.velocity
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]

        x1 = predicted_center[0] - w / 2
        y1 = predicted_center[1] - h / 2
        x2 = predicted_center[0] + w / 2
        y2 = predicted_center[1] + h / 2

        return np.array([x1, y1, x2, y2])

    def update(self, new_bbox):
        new_bbox = np.array(new_bbox, dtype=float)

        new_center = self._compute_center(new_bbox)

        self.prev_center = self.center.copy()
        self.velocity = new_center - self.center

        self.center = new_center
        self.bbox = new_bbox

        self.age += 1
        self.hits += 1
        self.consecutive_hits += 1
        self.time_since_update = 0

        self.history.append(self.center.copy())
        if len(self.history) > 10:
            self.history.pop(0)

    def mark_missed(self):
        self.time_since_update += 1
        self.consecutive_hits = 0
        self.age += 1
