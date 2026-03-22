"""
logging_system/event_logger.py
Handles all file-system logging: structured log file + image storage.
Every face entry and exit produces exactly one log record and one cropped image.
"""

import os
import logging
import cv2
import numpy as np
from datetime import datetime
from typing import Optional

# ── configure module-level logger ──────────────────────────────────────────────
def setup_logger(log_file: str = "logs/events.log", log_level: str = "INFO") -> logging.Logger:
    """
    Set up the root application logger with both file and console handlers.
    Returns the configured logger instance.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    level = getattr(logging, log_level.upper(), logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — all events go here
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Console handler — INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return root_logger


# ── EventLogger class ──────────────────────────────────────────────────────────
class EventLogger:
    """
    Responsible for:
    - Saving cropped face images to structured folder (logs/entries/ or logs/exits/)
    - Writing structured event records to the filesystem log
    """

    def __init__(self, image_store: str = "logs"):
        self.image_store = image_store
        self.logger = logging.getLogger(self.__class__.__name__)
        os.makedirs(os.path.join(image_store, "entries"), exist_ok=True)
        os.makedirs(os.path.join(image_store, "exits"), exist_ok=True)
        self.logger.info(f"EventLogger initialized. Image store: {image_store}")

    # ------------------------------------------------------------------ #
    #  Image Storage
    # ------------------------------------------------------------------ #

    def save_face_image(
        self,
        face_crop: np.ndarray,
        face_uuid: str,
        event_type: str,
        timestamp: Optional[datetime] = None,
    ) -> str:
        """
        Save a cropped face image to the structured folder.

        Structure:
            logs/entries/YYYY-MM-DD/<face_uuid>_HHMMSSffffff.jpg
            logs/exits/YYYY-MM-DD/<face_uuid>_HHMMSSffffff.jpg

        Returns the relative path to the saved image.
        """
        ts = timestamp or datetime.now()
        date_str = ts.strftime("%Y-%m-%d")
        time_str = ts.strftime("%H%M%S%f")

        folder = os.path.join(self.image_store, f"{event_type}s", date_str)
        os.makedirs(folder, exist_ok=True)

        filename = f"{face_uuid}_{time_str}.jpg"
        filepath = os.path.join(folder, filename)

        if face_crop is not None and face_crop.size > 0:
            # Resize to standard thumbnail to save disk space
            thumb = cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_AREA)
            cv2.imwrite(filepath, thumb)
            self.logger.debug(f"[IMG] Saved {event_type} image → {filepath}")
        else:
            self.logger.warning(f"[IMG] Empty crop for {face_uuid}, skipping save.")
            filepath = ""

        return filepath

    # ------------------------------------------------------------------ #
    #  Structured Event Logging
    # ------------------------------------------------------------------ #

    def log_face_entry(self, face_uuid: str, frame_number: int, is_new: bool = True):
        """Log a face-entry event."""
        status = "NEW" if is_new else "RETURNING"
        self.logger.info(
            f"[ENTRY] face_id={face_uuid} | frame={frame_number} | status={status}"
        )

    def log_face_exit(self, face_uuid: str, frame_number: int):
        """Log a face-exit event."""
        self.logger.info(
            f"[EXIT]  face_id={face_uuid} | frame={frame_number}"
        )

    def log_recognition(self, face_uuid: str, similarity: float, frame_number: int):
        """Log a successful face recognition match."""
        self.logger.debug(
            f"[RECOG] face_id={face_uuid} | similarity={similarity:.4f} | frame={frame_number}"
        )

    def log_embedding(self, face_uuid: str):
        """Log embedding generation."""
        self.logger.debug(f"[EMBED] Generated embedding for face_id={face_uuid}")

    def log_tracking(self, face_uuid: str, bbox: tuple, frame_number: int):
        """Log active tracking update."""
        self.logger.debug(
            f"[TRACK] face_id={face_uuid} | bbox={bbox} | frame={frame_number}"
        )

    def log_system(self, message: str, level: str = "INFO"):
        """Log a general system message."""
        log_fn = getattr(self.logger, level.lower(), self.logger.info)
        log_fn(f"[SYSTEM] {message}")

    def log_error(self, message: str, exc_info: bool = False):
        """Log an error."""
        self.logger.error(f"[ERROR] {message}", exc_info=exc_info)

    def log_summary(self, unique_count: int, total_events: int):
        """Log end-of-session summary."""
        self.logger.info(
            f"[SUMMARY] Session ended | unique_visitors={unique_count} | total_events={total_events}"
        )
