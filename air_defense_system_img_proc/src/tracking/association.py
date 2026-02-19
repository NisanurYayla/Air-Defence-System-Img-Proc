import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - interArea
    if union == 0:
        return 0.0

    return interArea / union


def center_similarity(boxA, boxB):
    centerA = np.array([(boxA[0] + boxA[2]) / 2, (boxA[1] + boxA[3]) / 2])
    centerB = np.array([(boxB[0] + boxB[2]) / 2, (boxB[1] + boxB[3]) / 2])

    dist = np.linalg.norm(centerA - centerB)
    return 1.0 / (1.0 + dist)


def velocity_alignment(track_velocity, boxA, boxB):
    if np.linalg.norm(track_velocity) < 1e-5:
        return 0.5

    centerA = np.array([(boxA[0] + boxA[2]) / 2, (boxA[1] + boxA[3]) / 2])
    centerB = np.array([(boxB[0] + boxB[2]) / 2, (boxB[1] + boxB[3]) / 2])

    direction = centerB - centerA
    if np.linalg.norm(direction) < 1e-5:
        return 0.5

    direction = direction / np.linalg.norm(direction)
    velocity = track_velocity / np.linalg.norm(track_velocity)

    return np.dot(direction, velocity)


def associate(tracks, detections,
              iou_weight=0.6,
              center_weight=0.3,
              vel_weight=0.1,
              match_threshold=0.25):

    if len(tracks) == 0:
        return [], [], list(range(len(detections)))

    if len(detections) == 0:
        return [], list(range(len(tracks))), []

    cost_matrix = np.zeros((len(tracks), len(detections)))

    for t, track in enumerate(tracks):
        predicted_bbox = track.predict()

        for d, det in enumerate(detections):
            iou = compute_iou(predicted_bbox, det)
            center_sim = center_similarity(predicted_bbox, det)
            vel_sim = velocity_alignment(track.velocity, predicted_bbox, det)

            score = (
                iou_weight * iou +
                center_weight * center_sim +
                vel_weight * vel_sim
            )

            cost_matrix[t, d] = 1 - score

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_tracks = list(range(len(tracks)))
    unmatched_detections = list(range(len(detections)))

    for r, c in zip(row_ind, col_ind):
        score = 1 - cost_matrix[r, c]
        if score >= match_threshold:
            matches.append((r, c))
            unmatched_tracks.remove(r)
            unmatched_detections.remove(c)

    return matches, unmatched_tracks, unmatched_detections
