"""
utils/video_utils.py
Helper utilities for video capture management and frame processing.
"""

import cv2
import logging
import numpy as np
from typing import Tuple, Optional, Generator

logger = logging.getLogger(__name__)


def get_video_properties(cap: cv2.VideoCapture) -> dict:
    """Return a dictionary of video capture properties."""
    return {
        "width":    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":   int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":      cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "codec":    int(cap.get(cv2.CAP_PROP_FOURCC)),
    }


def frame_generator(
    cap: cv2.VideoCapture,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Generator that yields (frame_index, frame) tuples from a VideoCapture.
    Stops when the stream ends or a read error occurs.
    """
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame_idx, frame
        frame_idx += 1


def resize_frame(
    frame: np.ndarray,
    max_width: int = 1280,
    max_height: int = 720,
) -> np.ndarray:
    """
    Resize frame to fit within max dimensions while preserving aspect ratio.
    """
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Overlay FPS counter on frame."""
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (frame.shape[1] - 120, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 200, 255),
        2,
    )
    return frame
