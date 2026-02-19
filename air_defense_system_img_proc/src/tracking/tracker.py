from .track import Track, TrackState
from .association import associate


class Tracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1

        self.confirm_hits = 3
        self.lost_threshold = 5
        self.delete_threshold = 15

    def update(self, detections):
        """
        detections: list of bbox [x1,y1,x2,y2]
        """

        matches, unmatched_tracks, unmatched_dets = associate(
            self.tracks,
            detections
        )

        # Matched tracks update
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx])

        # Unmatched tracks
        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]
            track.mark_missed()

        # New tracks
        for det_idx in unmatched_dets:
            new_track = Track(self.next_id, detections[det_idx])
            self.tracks.append(new_track)
            self.next_id += 1

        # State management
        alive_tracks = []
        for track in self.tracks:

            if track.state == TrackState.TENTATIVE:
                if track.hits >= self.confirm_hits:
                    track.state = TrackState.CONFIRMED

            if track.time_since_update > self.lost_threshold:
                track.state = TrackState.LOST

            if track.time_since_update > self.delete_threshold:
                continue  # sil

            alive_tracks.append(track)

        self.tracks = alive_tracks

        # Sadece confirmed olanları döndür
        confirmed = [
            track for track in self.tracks
            if track.state == TrackState.CONFIRMED
        ]

        return confirmed
