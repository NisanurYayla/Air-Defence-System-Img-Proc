import numpy as np


class TrackState:
    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2


class KalmanFilterCV:
    """
    6-state Constant Velocity + Scale model
    State:       [x, y, vx, vy, s, vs]
    Measurement: [x, y, s]
    where s = sqrt(area)  (stabil scale)
    """

    def __init__(self):
        dt = 1.0

        self.F = np.array([
            [1, 0, dt, 0,  0,  0],
            [0, 1, 0, dt,  0,  0],
            [0, 0, 1,  0,  0,  0],
            [0, 0, 0,  1,  0,  0],
            [0, 0, 0,  0,  1, dt],
            [0, 0, 0,  0,  0,  1]
        ], dtype=float)

        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0]
        ], dtype=float)

        # Process noise (viraj/slalom için biraz tolerans)
        self.Q = np.diag([8.0, 8.0, 8.0, 8.0, 2.0, 2.0]).astype(float)

        # Measurement noise
        self.R = np.eye(3, dtype=float) * 8.0

        # Initial covariance
        self.P = np.eye(6, dtype=float) * 500.0

        self.x = None

    def initiate(self, measurement):
        x, y, s = measurement
        self.x = np.array([x, y, 0.0, 0.0, s, 0.0], dtype=float)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        z = np.array(measurement, dtype=float)

        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0], dtype=float)
        self.P = (I - K @ self.H) @ self.P

    def gating_distance(self, measurement):
        z = np.array(measurement, dtype=float)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        return float(y.T @ np.linalg.inv(S) @ y)


def bbox_to_measurement_xyxy(bbox):
    """
    bbox: [x1, y1, x2, y2]
    measurement: [xc, yc, s] where s = sqrt(area)
    """
    x1, y1, x2, y2 = map(float, bbox)
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    xc = x1 + w / 2.0
    yc = y1 + h / 2.0
    area = w * h
    s = np.sqrt(max(area, 1.0))
    return np.array([xc, yc, s], dtype=float), (w, h)


class Track:
    def __init__(self, track_id, bbox, class_id: int = -1):
        self.id = int(track_id)
        self.kf = KalmanFilterCV()

        meas, (w, h) = bbox_to_measurement_xyxy(bbox)
        self.kf.initiate(meas)

        # Aspect ratio = w/h (kare bbox hatasını bitiriyor)
        self.aspect_ratio = float(max(w / max(h, 1.0), 0.1))

        self.last_bbox = np.array(bbox, dtype=float)

        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.consecutive_hits = 1

        self.state = TrackState.TENTATIVE

        self.class_id = class_id
        self._class_votes: dict = {class_id: 1} if class_id >= 0 else {}

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox, class_id: int = -1):
        meas, (w, h) = bbox_to_measurement_xyxy(bbox)
        self.kf.update(meas)

        # Aspect ratio çok oynamasın diye hafif smoothing
        new_ar = float(max(w / max(h, 1.0), 0.1))
        self.aspect_ratio = 0.9 * self.aspect_ratio + 0.1 * new_ar

        self.last_bbox = np.array(bbox, dtype=float)

        self.hits += 1
        self.consecutive_hits += 1
        self.time_since_update = 0

        if class_id >= 0:
            self._class_votes[class_id] = self._class_votes.get(class_id, 0) + 1
            self.class_id = max(self._class_votes, key=self._class_votes.get)

    def mark_missed(self):
        # time_since_update predict() ile artıyor, burada artırma YOK
        self.consecutive_hits = 0

    def get_bbox(self):
        """
        Kalman state'ten bbox üret (aspect ratio korunur).
        State'teki s = sqrt(area), yani area = s^2.
        """
        x, y, _, _, s, _ = self.kf.x

        s = float(max(s, 1.0))
        area = s * s

        a = float(max(self.aspect_ratio, 0.1))
        w = np.sqrt(area * a)
        h = np.sqrt(area / a)

        x1 = x - w / 2.0
        y1 = y - h / 2.0
        x2 = x + w / 2.0
        y2 = y + h / 2.0

        return np.array([x1, y1, x2, y2], dtype=float)