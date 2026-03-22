"""
core/face_recognizer.py
InsightFace-based face recognition module.
Generates 512-dim ArcFace embeddings and matches against registered faces.

Key improvements:
- Uses InsightFace's built-in face alignment (landmark-based crop) for
  much more stable embeddings than raw bbox crops.
- Supports embedding averaging: each time a face is re-identified,
  the stored embedding is updated with a running average, making
  the representation progressively more robust.
"""

import logging
import numpy as np
import cv2
import pickle
from typing import Optional, Tuple, List, Dict, Any

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """
    Uses InsightFace (buffalo_l / ArcFace) to:
    1. Generate stable 512-dim embeddings via landmark-aligned crops.
    2. Match new embeddings against all registered faces (cosine similarity).
    3. Support running-average embedding updates for better accuracy over time.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        similarity_threshold: float = 0.35,
        det_size: Tuple[int, int] = (640, 640),
    ):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.det_size = det_size
        self.app = None
        self._load_model()

    def _load_model(self):
        """Load InsightFace analysis app."""
        try:
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(
                name=self.model_name,
                providers=["CPUExecutionProvider"],
            )
            self.app.prepare(ctx_id=0, det_size=self.det_size)
            logger.info(f"[Recognizer] InsightFace model loaded: {self.model_name}")
        except ImportError:
            logger.warning("[Recognizer] InsightFace not installed. Using fallback embedder.")
            self.app = None
        except Exception as e:
            logger.error(f"[Recognizer] Failed to load InsightFace: {e}", exc_info=True)
            self.app = None

    # ------------------------------------------------------------------ #
    #  Embedding Generation
    # ------------------------------------------------------------------ #

    def get_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate a normalized 512-dim embedding for a face image.
        Uses InsightFace's internal detector+aligner for the best quality.
        Falls back to HOG descriptor if InsightFace is unavailable.
        """
        if face_image is None or face_image.size == 0:
            return None

        if self.app is not None:
            return self._insightface_embed(face_image)
        return self._fallback_embed(face_image)

    def _insightface_embed(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Use InsightFace to generate embedding.
        InsightFace runs its own internal detector on the crop, so the crop
        should contain exactly one face with some context margin.
        """
        try:
            # Ensure minimum size for InsightFace internal detector
            h, w = face_image.shape[:2]
            if h < 112 or w < 112:
                scale = max(112 / h, 112 / w)
                face_image = cv2.resize(
                    face_image,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_LINEAR,
                )

            rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            faces = self.app.get(rgb)
            if not faces:
                return None

            # Take the largest detected face (most likely the actual subject)
            best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            return self._normalize(best.embedding.astype(np.float32))
        except Exception as e:
            logger.debug(f"[Recognizer] InsightFace error: {e}")
            return None

    def _fallback_embed(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """HOG-based fallback when InsightFace is unavailable."""
        try:
            resized = cv2.resize(face_image, (64, 64))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            hog = cv2.HOGDescriptor(
                _winSize=(64, 64), _blockSize=(16, 16),
                _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9,
            )
            descriptor = hog.compute(gray).flatten()
            if len(descriptor) < 512:
                descriptor = np.pad(descriptor, (0, 512 - len(descriptor)))
            else:
                descriptor = descriptor[:512]
            return self._normalize(descriptor.astype(np.float32))
        except Exception as e:
            logger.error(f"[Recognizer] Fallback embedding error: {e}", exc_info=True)
            return None

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    # ------------------------------------------------------------------ #
    #  Matching
    # ------------------------------------------------------------------ #

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return float(np.dot(emb1, emb2))

    def find_best_match(
        self,
        query_embedding: np.ndarray,
        registered_faces: List[Dict[str, Any]],
    ) -> Tuple[Optional[str], float]:
        """
        Find the best matching face from the registered face store.

        Returns:
            (best_face_uuid, similarity_score) or (None, 0.0) if no match above threshold.
        """
        best_uuid = None
        best_sim = -1.0

        for record in registered_faces:
            stored_emb = self._deserialize_embedding(record["embedding"])
            if stored_emb is None:
                continue
            sim = self.cosine_similarity(query_embedding, stored_emb)
            if sim > best_sim:
                best_sim = sim
                best_uuid = record["face_uuid"]

        if best_sim >= self.similarity_threshold:
            logger.debug(f"[Recognizer] Match: {best_uuid} (sim={best_sim:.4f})")
            return best_uuid, best_sim

        logger.debug(f"[Recognizer] No match (best_sim={best_sim:.4f} < {self.similarity_threshold})")
        return None, best_sim

    # ------------------------------------------------------------------ #
    #  Embedding update (running average)
    # ------------------------------------------------------------------ #

    @staticmethod
    def average_embeddings(old_emb: np.ndarray, new_emb: np.ndarray, alpha: float = 0.2) -> np.ndarray:
        """
        Exponential moving average of embeddings.
        alpha=0.2 means 20% weight on new observation, 80% on history.
        Keeps the stored embedding stable while adapting to pose/lighting changes.
        """
        averaged = (1 - alpha) * old_emb + alpha * new_emb
        norm = np.linalg.norm(averaged)
        return averaged / norm if norm > 0 else averaged

    # ------------------------------------------------------------------ #
    #  Serialization
    # ------------------------------------------------------------------ #

    @staticmethod
    def serialize_embedding(embedding: np.ndarray) -> bytes:
        return pickle.dumps(embedding)

    @staticmethod
    def _deserialize_embedding(data: bytes) -> Optional[np.ndarray]:
        try:
            return pickle.loads(data)
        except Exception:
            return None
