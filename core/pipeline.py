"""
core/pipeline.py
Main orchestration pipeline.

Entry/exit counting logic:
- One ENTRY per face_uuid per continuous appearance window.
- A face is "in window" until absent for max_disappeared frames.
- One EXIT fired by _sweep_exits() — the single authoritative exit trigger.
- Embedding averaging: on each re-identification the stored embedding is
  updated (EMA), making future matches more robust.
"""

import logging
import uuid
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Set, Any

from core.face_detector import FaceDetector
from core.face_recognizer import FaceRecognizer
from core.face_tracker import FaceTracker
from database.db_manager import DatabaseManager
from logging_system.event_logger import EventLogger

logger = logging.getLogger(__name__)


class FaceTrackingPipeline:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        det_cfg = config.get("detection", {})
        rec_cfg = config.get("recognition", {})
        trk_cfg = config.get("tracking", {})
        db_cfg  = config.get("database", {})
        log_cfg = config.get("logging", {})

        self.detector = FaceDetector(
            model_path=det_cfg.get("yolo_model", "yolov8n-face.pt"),
            confidence_threshold=det_cfg.get("confidence_threshold", 0.5),
            input_size=det_cfg.get("input_size", 640),
        )
        self.recognizer = FaceRecognizer(
            model_name=rec_cfg.get("model_name", "buffalo_l"),
            similarity_threshold=rec_cfg.get("similarity_threshold", 0.35),
        )
        self.tracker = FaceTracker(
            max_disappeared=trk_cfg.get("max_disappeared", 45),
            iou_threshold=trk_cfg.get("iou_threshold", 0.3),
        )
        self.db = DatabaseManager(db_path=db_cfg.get("path", "face_tracker.db"))
        self.event_logger = EventLogger(
            image_store=log_cfg.get("image_store", "logs")
        )

        self.skip_frames: int = det_cfg.get("skip_frames", 2)
        self.max_disappeared: int = trk_cfg.get("max_disappeared", 45)
        self.frame_count: int = 0
        self.session_id: Optional[int] = None

        # track_id → face_uuid
        self._track_identity: Dict[str, str] = {}

        # face_uuid → last frame seen (used by sweep_exits)
        self._active_faces: Dict[str, int] = {}

        # face_uuids currently "in frame" (entry fired, exit not yet fired)
        self._in_frame: Set[str] = set()

        logger.info("[Pipeline] FaceTrackingPipeline initialized.")

    # ------------------------------------------------------------------ #
    #  Session
    # ------------------------------------------------------------------ #

    def start_session(self, source: str):
        self.session_id = self.db.start_session(source)
        self.event_logger.log_system(f"Session started. Source: {source}")
        logger.info(f"[Pipeline] Session {self.session_id} started for: {source}")

    def end_session(self):
        """Fire exit events for any faces still in frame, then close session."""
        for face_uuid in list(self._in_frame):
            self.db.log_event(face_uuid, "exit", "", self.frame_count)
            self.event_logger.log_face_exit(face_uuid, self.frame_count)
        self._in_frame.clear()

        if self.session_id:
            self.db.end_session(self.session_id)
        summary = self.db.get_summary()
        self.event_logger.log_summary(summary["unique_visitors"], summary["total_events"])
        logger.info(f"[Pipeline] Session ended. Summary: {summary}")

    # ------------------------------------------------------------------ #
    #  Frame loop
    # ------------------------------------------------------------------ #

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        self.frame_count += 1
        annotated = frame.copy()

        # Detection on every (skip_frames+1)th frame
        if self.frame_count % (self.skip_frames + 1) == 0 or self.frame_count == 1:
            detections = self.detector.detect(frame)
        else:
            detections = []

        new_track_ids, disappeared_track_ids = self.tracker.update(detections)

        for track_id in new_track_ids:
            track = self.tracker.get_track(track_id)
            if track is None:
                continue
            face_crop = self.detector.crop_face(frame, track.bbox)
            self._process_new_track(track_id, track.bbox, face_crop, frame)

        for track_id in disappeared_track_ids:
            self._track_identity.pop(track_id, None)

        self._sweep_exits()

        if self.config.get("display", {}).get("draw_bboxes", True):
            annotated = self._draw_annotations(annotated)

        return annotated

    # ------------------------------------------------------------------ #
    #  Identity resolution
    # ------------------------------------------------------------------ #

    def _process_new_track(
        self,
        track_id: str,
        bbox: tuple,
        face_crop: Optional[np.ndarray],
        frame: np.ndarray,
    ):
        if face_crop is None:
            return

        # Minimum size check
        MIN_FACE_DIM = 60  # raised from 40 — small crops give poor embeddings
        h_crop, w_crop = face_crop.shape[:2]
        if h_crop < MIN_FACE_DIM or w_crop < MIN_FACE_DIM:
            logger.debug(f"[Pipeline] Crop {w_crop}x{h_crop} too small — skip")
            return

        # Generate embedding; retry with larger margin
        embedding = self.recognizer.get_embedding(face_crop)
        if embedding is None:
            larger = self.detector.crop_face(frame, bbox, margin=0.4)
            if larger is not None:
                embedding = self.recognizer.get_embedding(larger)
                if embedding is not None:
                    face_crop = larger
        if embedding is None:
            logger.debug(f"[Pipeline] No embedding for track {track_id[:8]}")
            return

        self.event_logger.log_embedding(track_id)

        # Match against DB
        registered = self.db.get_all_embeddings()
        matched_uuid, similarity = self.recognizer.find_best_match(embedding, registered)

        if matched_uuid:
            face_uuid = matched_uuid
            is_new = False
            self.db.update_last_seen(face_uuid)
            self.event_logger.log_recognition(face_uuid, similarity, self.frame_count)

            # ── Embedding averaging: update stored embedding for better future matches ──
            stored_record = next((r for r in registered if r["face_uuid"] == face_uuid), None)
            if stored_record:
                old_emb = FaceRecognizer._deserialize_embedding(stored_record["embedding"])
                if old_emb is not None:
                    updated_emb = FaceRecognizer.average_embeddings(old_emb, embedding, alpha=0.2)
                    updated_bytes = FaceRecognizer.serialize_embedding(updated_emb)
                    self.db.update_embedding(face_uuid, updated_bytes)
        else:
            face_uuid = str(uuid.uuid4())[:8]
            is_new = True
            embedding_bytes = FaceRecognizer.serialize_embedding(embedding)
            ts = datetime.now().isoformat()
            img_path = self.event_logger.save_face_image(face_crop, face_uuid, "entry")
            self.db.register_face(face_uuid, embedding_bytes, img_path, ts)

        # Cache identity for this tracker ID
        self.tracker.assign_identity(track_id, face_uuid, embedding)
        self._track_identity[track_id] = face_uuid

        # Update last-seen
        self._active_faces[face_uuid] = self.frame_count

        # ── Fire ENTRY only once per continuous appearance window ──
        if face_uuid not in self._in_frame:
            img_path = self.event_logger.save_face_image(face_crop, face_uuid, "entry")
            self.db.log_event(face_uuid, "entry", img_path, self.frame_count)
            self.event_logger.log_face_entry(face_uuid, self.frame_count, is_new)
            self._in_frame.add(face_uuid)
            self.tracker.mark_entry_logged(track_id)
        else:
            # Still in frame — just refresh last-seen, no new entry event
            logger.debug(f"[Pipeline] {face_uuid} still in frame at {self.frame_count} — no re-entry")

    # ------------------------------------------------------------------ #
    #  Exit sweeper — single authoritative exit trigger
    # ------------------------------------------------------------------ #

    def _sweep_exits(self):
        """
        Fire EXIT for any face_uuid not seen for more than max_disappeared frames.
        This is the ONLY place exit events are created.
        """
        to_exit = [
            fid for fid, last in self._active_faces.items()
            if (self.frame_count - last) > self.max_disappeared and fid in self._in_frame
        ]
        for face_uuid in to_exit:
            self.db.log_event(face_uuid, "exit", "", self.frame_count)
            self.event_logger.log_face_exit(face_uuid, self.frame_count)
            self._in_frame.discard(face_uuid)
            del self._active_faces[face_uuid]

    # ------------------------------------------------------------------ #
    #  Display
    # ------------------------------------------------------------------ #

    def _draw_annotations(self, frame: np.ndarray) -> np.ndarray:
        display_cfg = self.config.get("display", {})
        for track in self.tracker.get_active_tracks():
            x1, y1, x2, y2 = track.bbox
            face_uuid = track.face_uuid or "?"
            color = (0, 255, 0) if track.is_registered else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if display_cfg.get("draw_ids", True):
                label = f"ID:{face_uuid[:8]} ({track.confidence:.2f})"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        count = self.db.get_unique_visitor_count()
        cv2.putText(frame, f"Unique Visitors: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        return frame

    def get_unique_visitor_count(self) -> int:
        return self.db.get_unique_visitor_count()
