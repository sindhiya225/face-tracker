"""
core/face_tracker.py
Multi-face tracker that maintains track state across frames.
Uses IoU + centroid distance to associate detections with existing tracks.
Handles entry/exit events based on track lifecycle.
"""

import logging
import uuid
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Represents a single tracked face across frames."""

    track_id: str                          # Internal tracker ID (UUID)
    face_uuid: Optional[str]               # Registered face identity (may differ)
    bbox: Tuple[int, int, int, int]        # Last known bounding box (x1,y1,x2,y2)
    embedding: Optional[np.ndarray]        # Last embedding
    disappeared_frames: int = 0            # Consecutive frames without detection
    is_registered: bool = False            # Whether pushed to DB
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    entry_logged: bool = False             # Entry event fired?
    exit_logged: bool = False              # Exit event fired?
    confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


class FaceTracker:
    """
    Maintains a dictionary of active tracks.
    Matches new detections to existing tracks via IoU and centroid distance.
    Fires entry/exit callbacks when tracks appear or disappear.
    """

    def __init__(self, max_disappeared: int = 30, iou_threshold: float = 0.3):
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.tracks: Dict[str, Track] = {}        # track_id → Track
        self.frame_count = 0
        logger.info(
            f"[Tracker] Initialized (max_disappeared={max_disappeared}, "
            f"iou_threshold={iou_threshold})"
        )

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def update(
        self,
        detections: List[Tuple[int, int, int, int, float]],
    ) -> Tuple[List[str], List[str]]:
        """
        Update tracker with a new frame's detections.

        Args:
            detections: list of (x1, y1, x2, y2, confidence) from detector.

        Returns:
            (new_track_ids, disappeared_track_ids)
        """
        self.frame_count += 1
        new_ids: List[str] = []
        disappeared_ids: List[str] = []

        if len(detections) == 0:
            # No detections — increment disappeared counter for all active tracks
            to_remove = []
            for tid, track in self.tracks.items():
                track.disappeared_frames += 1
                if track.disappeared_frames > self.max_disappeared:
                    to_remove.append(tid)
                    disappeared_ids.append(tid)
            for tid in to_remove:
                del self.tracks[tid]
            return new_ids, disappeared_ids

        # Match detections to existing tracks
        matched, unmatched_dets, unmatched_tracks = self._match(detections)

        # Update matched tracks
        for track_id, det_idx in matched:
            bbox = detections[det_idx][:4]
            conf = detections[det_idx][4]
            self.tracks[track_id].bbox = bbox
            self.tracks[track_id].disappeared_frames = 0
            self.tracks[track_id].last_seen_frame = self.frame_count
            self.tracks[track_id].confidence = conf

        # Increment disappeared for unmatched tracks
        to_remove = []
        for tid in unmatched_tracks:
            self.tracks[tid].disappeared_frames += 1
            if self.tracks[tid].disappeared_frames > self.max_disappeared:
                to_remove.append(tid)
                disappeared_ids.append(tid)
        for tid in to_remove:
            del self.tracks[tid]

        # Register new tracks for unmatched detections
        for det_idx in unmatched_dets:
            bbox = detections[det_idx][:4]
            conf = detections[det_idx][4]
            new_track_id = self._create_track(bbox, conf)
            new_ids.append(new_track_id)

        return new_ids, disappeared_ids

    def get_active_tracks(self) -> List[Track]:
        """Return all currently active tracks."""
        return list(self.tracks.values())

    def get_track(self, track_id: str) -> Optional[Track]:
        """Fetch a specific track by ID."""
        return self.tracks.get(track_id)

    def assign_identity(
        self, track_id: str, face_uuid: str, embedding: np.ndarray
    ):
        """Assign a recognized face UUID and embedding to a track."""
        if track_id in self.tracks:
            self.tracks[track_id].face_uuid = face_uuid
            self.tracks[track_id].embedding = embedding
            self.tracks[track_id].is_registered = True

    def mark_entry_logged(self, track_id: str):
        if track_id in self.tracks:
            self.tracks[track_id].entry_logged = True

    def mark_exit_logged(self, track_id: str):
        if track_id in self.tracks:
            self.tracks[track_id].exit_logged = True

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _create_track(self, bbox: Tuple[int, int, int, int], conf: float) -> str:
        """Create a new Track and return its ID."""
        track_id = str(uuid.uuid4())
        self.tracks[track_id] = Track(
            track_id=track_id,
            face_uuid=None,
            bbox=bbox,
            embedding=None,
            first_seen_frame=self.frame_count,
            last_seen_frame=self.frame_count,
            confidence=conf,
        )
        logger.debug(f"[Tracker] New track created: {track_id}")
        return track_id

    def _match(
        self, detections: List[Tuple[int, int, int, int, float]]
    ) -> Tuple[List[Tuple[str, int]], List[int], List[str]]:
        """
        Hungarian-style greedy matching between active tracks and detections.

        Returns:
            matched: [(track_id, det_idx), ...]
            unmatched_dets: [det_idx, ...]
            unmatched_tracks: [track_id, ...]
        """
        if not self.tracks:
            return [], list(range(len(detections))), []

        track_ids = list(self.tracks.keys())
        track_bboxes = [self.tracks[t].bbox for t in track_ids]
        det_bboxes = [d[:4] for d in detections]

        # Build IoU matrix
        iou_matrix = np.zeros((len(track_ids), len(det_bboxes)), dtype=np.float32)
        for ti, tb in enumerate(track_bboxes):
            for di, db in enumerate(det_bboxes):
                iou_matrix[ti, di] = self._iou(tb, db)

        matched = []
        unmatched_dets = set(range(len(detections)))
        unmatched_tracks = set(range(len(track_ids)))

        # Greedy match: highest IoU first
        if iou_matrix.size > 0:
            flat_indices = np.argsort(-iou_matrix.flatten())
            for idx in flat_indices:
                ti = idx // len(det_bboxes)
                di = idx % len(det_bboxes)
                if iou_matrix[ti, di] < self.iou_threshold:
                    break
                if ti in unmatched_tracks and di in unmatched_dets:
                    matched.append((track_ids[ti], di))
                    unmatched_tracks.discard(ti)
                    unmatched_dets.discard(di)

        return (
            matched,
            list(unmatched_dets),
            [track_ids[ti] for ti in unmatched_tracks],
        )

    @staticmethod
    def _iou(box_a: Tuple, box_b: Tuple) -> float:
        """Compute Intersection-over-Union of two bounding boxes."""
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b

        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area_a = (xa2 - xa1) * (ya2 - ya1)
        area_b = (xb2 - xb1) * (yb2 - yb1)
        union_area = area_a + area_b - inter_area

        return inter_area / union_area if union_area > 0 else 0.0
