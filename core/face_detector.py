"""
core/face_detector.py
YOLO-based face detector.
Uses YOLOv8 face model for real-time face detection.
Falls back to OpenCV Haar cascade if YOLO model is unavailable.
"""

import logging
import numpy as np
import cv2
import os
import urllib.request
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# Known working download mirrors for yolov8n-face.pt
_YOLO_FACE_MIRRORS = [
    "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt",
    "https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt",
]


class FaceDetector:
    """
    Wraps YOLOv8 face detection (ultralytics).
    Returns list of (x1, y1, x2, y2, confidence) bounding boxes.
    """

    def __init__(
        self,
        model_path: str = "yolov8n-face.pt",
        confidence_threshold: float = 0.5,
        input_size: int = 640,
    ):
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.model = None
        self.use_fallback = False
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Attempt to load YOLOv8 face model; auto-download if missing; fall back to Haar cascade."""
        try:
            from ultralytics import YOLO

            # Auto-download if not present
            if not os.path.exists(model_path) and "face" in model_path.lower():
                downloaded = False
                for url in _YOLO_FACE_MIRRORS:
                    logger.info(f"[Detector] Downloading face model from {url} ...")
                    try:
                        urllib.request.urlretrieve(url, model_path)
                        logger.info(f"[Detector] Downloaded: {model_path}")
                        downloaded = True
                        break
                    except Exception as dl_err:
                        logger.warning(f"[Detector] Mirror failed ({url}): {dl_err}")
                if not downloaded:
                    logger.warning("[Detector] All download mirrors failed — will use Haar fallback.")

            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                logger.info(f"[Detector] YOLOv8 face model loaded: {model_path}")
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")

        except Exception as e:
            logger.warning(
                f"[Detector] Could not load YOLO model ({e}). Falling back to Haar cascade."
            )
            self._load_haar_cascade()
            self.use_fallback = True

    def _load_haar_cascade(self):
        """Load OpenCV Haar cascade as fallback detector."""
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.haar = cv2.CascadeClassifier(cascade_path)
        if self.haar.empty():
            raise RuntimeError("Haar cascade could not be loaded. Check OpenCV installation.")
        logger.info("[Detector] Haar cascade fallback loaded.")

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Run face detection on a BGR frame.
        Returns list of (x1, y1, x2, y2, confidence) tuples sorted by confidence desc.
        """
        if self.use_fallback:
            return self._haar_detect(frame)
        return self._yolo_detect(frame)

    def _yolo_detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Run YOLO inference and parse boxes."""
        try:
            results = self.model.predict(
                source=frame,
                imgsz=self.input_size,
                conf=self.confidence_threshold,
                verbose=False,
                device="cpu",   # Change to 0 for GPU
            )
            detections = []
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf < self.confidence_threshold:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 > x1 and y2 > y1:
                        detections.append((x1, y1, x2, y2, conf))
            detections.sort(key=lambda d: d[4], reverse=True)
            return detections
        except Exception as e:
            logger.error(f"[Detector] YOLO inference error: {e}", exc_info=True)
            return []

    def _haar_detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Run Haar cascade detection with strict params to reduce false positives.
        minNeighbors=8, minSize=(60,60) cut noise dramatically vs defaults.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.haar.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=8, minSize=(60, 60)
        )
        detections = []
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                x1, y1, x2, y2 = x, y, x + w, y + h
                detections.append((x1, y1, x2, y2, 0.9))
        return detections

    def crop_face(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        margin: float = 0.2,
    ) -> Optional[np.ndarray]:
        """
        Crop a face from the frame with an optional margin.
        Returns cropped face as BGR ndarray, or None if crop is invalid.
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        mx, my = int(bw * margin), int(bh * margin)
        nx1 = max(0, x1 - mx)
        ny1 = max(0, y1 - my)
        nx2 = min(w, x2 + mx)
        ny2 = min(h, y2 + my)
        crop = frame[ny1:ny2, nx1:nx2]
        if crop.size == 0:
            return None
        return crop
