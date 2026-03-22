"""
generate_sample_output.py
Generates synthetic sample output (DB entries + log file + fake images)
to demonstrate the system structure without running an actual video.
Run this script once to populate logs/ and face_tracker.db with sample data.

Usage: python generate_sample_output.py
"""

import os
import sys
import json
import uuid
import pickle
import sqlite3
import logging
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.db_manager import DatabaseManager
from logging_system.event_logger import EventLogger, setup_logger

# ── Setup ──────────────────────────────────────────────────────────────────────

os.makedirs("logs/entries/2026-03-21", exist_ok=True)
os.makedirs("logs/exits/2026-03-21", exist_ok=True)

setup_logger("logs/events.log", "DEBUG")
logger = logging.getLogger(__name__)
db = DatabaseManager("face_tracker.db")
event_logger = EventLogger("logs")

# ── Generate fake faces ────────────────────────────────────────────────────────

NUM_VISITORS = 5
base_time = datetime(2026, 3, 21, 10, 0, 0)

logger.info("[SYSTEM] Sample output generation started.")

for i in range(NUM_VISITORS):
    face_uuid = f"face_{i+1:04d}"
    entry_ts  = (base_time + timedelta(seconds=i * 15)).isoformat()
    exit_ts   = (base_time + timedelta(seconds=i * 15 + 120)).isoformat()

    # Fake 512-dim embedding
    embedding = np.random.randn(512).astype(np.float32)
    embedding /= np.linalg.norm(embedding)
    emb_bytes = pickle.dumps(embedding)

    # Create placeholder image (black 112x112 JPEG)
    try:
        import cv2
        img = np.zeros((112, 112, 3), dtype=np.uint8)
        # Draw visitor number on it
        cv2.putText(img, f"V{i+1}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        entry_img_path = f"logs/entries/2026-03-21/{face_uuid}_entry.jpg"
        exit_img_path  = f"logs/exits/2026-03-21/{face_uuid}_exit.jpg"
        cv2.imwrite(entry_img_path, img)
        cv2.imwrite(exit_img_path, img)
    except ImportError:
        entry_img_path = f"logs/entries/2026-03-21/{face_uuid}_entry.jpg"
        exit_img_path  = f"logs/exits/2026-03-21/{face_uuid}_exit.jpg"

    # Register face in DB
    db.register_face(face_uuid, emb_bytes, entry_img_path, entry_ts)

    # Log entry event
    db.log_event(face_uuid, "entry", entry_img_path, frame_number=i * 30 + 1, timestamp=entry_ts)
    event_logger.log_face_entry(face_uuid, frame_number=i * 30 + 1, is_new=True)

    # Log exit event
    db.log_event(face_uuid, "exit", exit_img_path, frame_number=i * 30 + 240, timestamp=exit_ts)
    event_logger.log_face_exit(face_uuid, frame_number=i * 30 + 240)

# Session
session_id = db.start_session("sample_video.mp4")
db.end_session(session_id)

# Summary
summary = db.get_summary()
logger.info(f"[SYSTEM] Sample generation complete. Summary: {summary}")
db.close()

print("\n✅ Sample output generated!")
print(f"   DB          : face_tracker.db")
print(f"   Log file    : logs/events.log")
print(f"   Entry images: logs/entries/2026-03-21/")
print(f"   Exit images : logs/exits/2026-03-21/")
print(f"   Visitors    : {summary['unique_visitors']}")
print(f"   Events      : {summary['total_events']}")
