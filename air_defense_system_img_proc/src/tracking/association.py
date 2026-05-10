import numpy as np
from scipy.optimize import linear_sum_assignment


def bbox_to_measurement(bbox):
    """
    bbox: [x1, y1, x2, y2]
    measurement: [xc, yc, s] where s = sqrt(area)
    """
    x1, y1, x2, y2 = map(float, bbox)
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    xc = x1 + w / 2.0
    yc = y1 + h / 2.0
    s = np.sqrt(max(w * h, 1.0))
    return np.array([xc, yc, s], dtype=float)


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9

    return inter / union


def associate(tracks, detections, gating_threshold=11.3):
    """
    tracks: list[Track]  (predict edilmiş olmalı)
    detections: list[dict] each:
        {"bbox": [x1,y1,x2,y2], "conf": float, "class_id": int}

    Returns:
        matches: list[(track_idx, det_idx)]
        unmatched_tracks: list[track_idx]
        unmatched_detections: list[det_idx]
    """

    if len(tracks) == 0:
        return [], [], list(range(len(detections)))

    if len(detections) == 0:
        return [], list(range(len(tracks))), []

    INF = 1e9
    cost_matrix = np.full((len(tracks), len(detections)), INF, dtype=float)

    for t, track in enumerate(tracks):
        pred_bbox = track.get_bbox()

        for d, det in enumerate(detections):
            det_bbox = det["bbox"]
            meas = bbox_to_measurement(det_bbox)

            maha = track.kf.gating_distance(meas)

            # Gating: istatistiksel olarak makul değilse eşleştirme kapat
            if maha > gating_threshold:
                continue

            iou = iou_xyxy(pred_bbox, det_bbox)

            # Hibrit cost: küçük iyi.
            # maha'yı normalize et (0..~1 civarı)
            maha_norm = maha / gating_threshold

            # IoU yüksekse maliyet düşsün
            cost = 0.4 * maha_norm + 0.6 * (1.0 - iou)

            if iou < 0.05:
                continue

            cost_matrix[t, d] = cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_tracks = list(range(len(tracks)))
    unmatched_detections = list(range(len(detections)))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] >= INF:
            continue
        matches.append((r, c))
        if r in unmatched_tracks:
            unmatched_tracks.remove(r)
        if c in unmatched_detections:
            unmatched_detections.remove(c)

    return matches, unmatched_tracks, unmatched_detections