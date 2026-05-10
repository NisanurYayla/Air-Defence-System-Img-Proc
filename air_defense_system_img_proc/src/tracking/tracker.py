from .track import Track, TrackState
from .association import associate


class Tracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1

        self.confirm_hits = 3
        self.lost_threshold = 45
        self.delete_threshold = 90

        # ✅ High / Low confidence split (ByteTrack fikri)
        self.high_th = 0.50
        self.low_th = 0.20

    def update(self, detections):
        """
        detections: list of dict
          [
            {"bbox": [x1,y1,x2,y2], "conf": float, "class_id": int},
            ...
          ]
        """

        # 0) Split detections: high vs low confidence
        high_dets = [d for d in detections if d["conf"] >= self.high_th]
        low_dets = [d for d in detections if self.low_th <= d["conf"] < self.high_th]

        # 1) Predict all tracks
        for track in self.tracks:
            track.predict()

        # Eğer hiç track yoksa: sadece HIGH dets ile yeni track aç
        if len(self.tracks) == 0:
            for det in high_dets:
                new_track = Track(self.next_id, det["bbox"], det.get("class_id", -1))
                self.tracks.append(new_track)
                self.next_id += 1

            # Lifecycle + output
            return [t for t in self.tracks if t.state == TrackState.CONFIRMED]

        # 2) Association #1: all tracks vs HIGH dets
        matches, unmatched_tracks, unmatched_high = associate(self.tracks, high_dets)

        # 3) Update matched tracks (HIGH)
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(high_dets[det_idx]["bbox"], high_dets[det_idx].get("class_id", -1))

        # 4) Association #2: unmatched tracks vs LOW dets (NO new tracks here)
        if len(unmatched_tracks) > 0 and len(low_dets) > 0:
            # unmatched_tracks index listesi -> track objelerine çevir
            unmatched_track_objs = [self.tracks[i] for i in unmatched_tracks]

            matches2, unmatched_tracks2_local, _ = associate(unmatched_track_objs, low_dets)

            # LOW ile eşleşenleri update et
            for local_ti, det_idx in matches2:
                global_ti = unmatched_tracks[local_ti]
                self.tracks[global_ti].update(low_dets[det_idx]["bbox"], low_dets[det_idx].get("class_id", -1))

            # LOW ile de eşleşmeyenler gerçekten missed sayılacak
            missed_global = [unmatched_tracks[i] for i in unmatched_tracks2_local]
        else:
            missed_global = unmatched_tracks

        # 5) Mark missed tracks
        for track_idx in missed_global:
            self.tracks[track_idx].mark_missed()

        # 6) Create new tracks ONLY from unmatched HIGH dets (with REACQUIRE)

        from .association import iou_xyxy

        def _bbox_center(b):
            x1, y1, x2, y2 = b
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        def _center_dist(a, b):
            ax, ay = _bbox_center(a)
            bx, by = _bbox_center(b)
            dx = ax - bx
            dy = ay - by
            return (dx*dx + dy*dy) ** 0.5
        
        for det_idx in unmatched_high:

            det_bbox = high_dets[det_idx]["bbox"]
        
            best_track = None
            best_iou = 0.0
        
            # 🔎 Kaybolmuş confirmed trackleri kontrol et
            for track in self.tracks:
            
                if track.state != TrackState.CONFIRMED:
                    continue
                
                # Çok uzun süre kayıpsa kurtarma yapma
                if track.time_since_update < 1 or track.time_since_update > 15:
                    continue
                
                # Çok uzaksa bağlama
                if _center_dist(track.last_bbox, det_bbox) > 120:
                    continue
                
                i = iou_xyxy(track.last_bbox, det_bbox)
        
                if i > best_iou:
                    best_iou = i
                    best_track = track
        
            # 🎯 Eğer iyi eşleşme varsa eski tracki canlandır
            if best_track is not None and best_iou > 0.25:
                best_track.update(det_bbox)
                continue
            
            # ❌ Yoksa yeni ID aç
            new_track = Track(self.next_id, det_bbox, high_dets[det_idx].get("class_id", -1))
            self.tracks.append(new_track)
            self.next_id += 1

        # 7) State management (confirm / lost / delete)
        alive_tracks = []
        for track in self.tracks:

            # Confirm
            if track.state == TrackState.TENTATIVE:
                if track.hits >= self.confirm_hits:
                    track.state = TrackState.CONFIRMED

            # Lost
            if track.time_since_update > self.lost_threshold:
                track.state = TrackState.LOST

            # Delete
            if track.time_since_update > self.delete_threshold:
                continue

            alive_tracks.append(track)

        self.tracks = alive_tracks

        # 8) Return only confirmed tracks
        confirmed = [t for t in self.tracks if t.state == TrackState.CONFIRMED]
        return confirmed