"""
database/db_manager.py
Handles all database operations for the Face Tracker system.
Supports SQLite (default). Can be extended to PostgreSQL/MongoDB.
"""

import sqlite3
import os
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages all persistent storage for faces, events, and visitor counts.
    Thread-safe SQLite implementation with WAL mode for concurrent access.
    """

    def __init__(self, db_path: str = "face_tracker.db"):
        self.db_path = db_path
        self._ensure_db_dir()
        self.conn = self._connect()
        self._initialize_schema()
        logger.info(f"DatabaseManager initialized with DB: {db_path}")

    def _ensure_db_dir(self):
        """Ensure the directory for the DB file exists."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        """Create a persistent SQLite connection with WAL mode."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_schema(self):
        """Create all required tables if they don't exist."""
        cursor = self.conn.cursor()

        # Faces table — stores registered face identities
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                face_uuid   TEXT NOT NULL UNIQUE,
                first_seen  TEXT NOT NULL,
                last_seen   TEXT NOT NULL,
                embedding   BLOB,
                entry_image TEXT,
                visit_count INTEGER DEFAULT 1
            )
        """)

        # Events table — entry/exit log for every face
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                face_uuid   TEXT NOT NULL,
                event_type  TEXT NOT NULL CHECK(event_type IN ('entry','exit')),
                timestamp   TEXT NOT NULL,
                image_path  TEXT,
                frame_number INTEGER,
                FOREIGN KEY (face_uuid) REFERENCES faces(face_uuid)
            )
        """)

        # Session table — tracks unique visitor count per session
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_start   TEXT NOT NULL,
                session_end     TEXT,
                unique_visitors INTEGER DEFAULT 0,
                video_source    TEXT
            )
        """)

        self.conn.commit()
        logger.debug("Database schema initialized.")

    # ------------------------------------------------------------------ #
    #  Face Registration & Lookup
    # ------------------------------------------------------------------ #

    def register_face(
        self,
        face_uuid: str,
        embedding_bytes: bytes,
        entry_image_path: str,
        timestamp: Optional[str] = None,
    ) -> bool:
        """
        Insert a new face into the database.
        Returns True if inserted, False if already exists.
        """
        ts = timestamp or datetime.now().isoformat()
        try:
            self.conn.execute(
                """
                INSERT INTO faces (face_uuid, first_seen, last_seen, embedding, entry_image)
                VALUES (?, ?, ?, ?, ?)
                """,
                (face_uuid, ts, ts, embedding_bytes, entry_image_path),
            )
            self.conn.commit()
            logger.info(f"[DB] Registered new face: {face_uuid}")
            return True
        except sqlite3.IntegrityError:
            logger.warning(f"[DB] Face already registered: {face_uuid}")
            return False

    def update_last_seen(self, face_uuid: str, timestamp: Optional[str] = None):
        """Update the last_seen timestamp and increment visit_count."""
        ts = timestamp or datetime.now().isoformat()
        self.conn.execute(
            "UPDATE faces SET last_seen=?, visit_count=visit_count+1 WHERE face_uuid=?",
            (ts, face_uuid),
        )
        self.conn.commit()

    def update_embedding(self, face_uuid: str, embedding_bytes: bytes):
        """Replace the stored embedding with an updated (averaged) version."""
        self.conn.execute(
            "UPDATE faces SET embedding=? WHERE face_uuid=?",
            (embedding_bytes, face_uuid),
        )
        self.conn.commit()

    def face_exists(self, face_uuid: str) -> bool:
        """Check if a face UUID is already registered."""
        row = self.conn.execute(
            "SELECT 1 FROM faces WHERE face_uuid=?", (face_uuid,)
        ).fetchone()
        return row is not None

    def get_all_embeddings(self) -> List[Dict[str, Any]]:
        """
        Retrieve all stored embeddings for matching against new detections.
        Returns list of dicts with face_uuid and embedding bytes.
        """
        rows = self.conn.execute(
            "SELECT face_uuid, embedding FROM faces WHERE embedding IS NOT NULL"
        ).fetchall()
        return [{"face_uuid": r["face_uuid"], "embedding": r["embedding"]} for r in rows]

    def get_face(self, face_uuid: str) -> Optional[Dict[str, Any]]:
        """Fetch full face record by UUID."""
        row = self.conn.execute(
            "SELECT * FROM faces WHERE face_uuid=?", (face_uuid,)
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------ #
    #  Event Logging
    # ------------------------------------------------------------------ #

    def log_event(
        self,
        face_uuid: str,
        event_type: str,
        image_path: str,
        frame_number: int = 0,
        timestamp: Optional[str] = None,
    ):
        """Insert an entry or exit event."""
        ts = timestamp or datetime.now().isoformat()
        self.conn.execute(
            """
            INSERT INTO events (face_uuid, event_type, timestamp, image_path, frame_number)
            VALUES (?, ?, ?, ?, ?)
            """,
            (face_uuid, event_type, ts, image_path, frame_number),
        )
        self.conn.commit()
        logger.debug(f"[DB] Event logged: {event_type} for {face_uuid} @ {ts}")

    def get_events_for_face(self, face_uuid: str) -> List[Dict[str, Any]]:
        """Get all events for a specific face UUID."""
        rows = self.conn.execute(
            "SELECT * FROM events WHERE face_uuid=? ORDER BY timestamp",
            (face_uuid,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_events(self) -> List[Dict[str, Any]]:
        """Get all events ordered by timestamp."""
        rows = self.conn.execute(
            "SELECT * FROM events ORDER BY timestamp"
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    #  Visitor Counting
    # ------------------------------------------------------------------ #

    def get_unique_visitor_count(self) -> int:
        """Return total number of unique faces registered."""
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM faces").fetchone()
        return row["cnt"] if row else 0

    def start_session(self, video_source: str) -> int:
        """Create a new session record and return its ID."""
        ts = datetime.now().isoformat()
        cursor = self.conn.execute(
            "INSERT INTO sessions (session_start, video_source) VALUES (?, ?)",
            (ts, video_source),
        )
        self.conn.commit()
        return cursor.lastrowid

    def end_session(self, session_id: int):
        """Close a session with end time and final visitor count."""
        ts = datetime.now().isoformat()
        count = self.get_unique_visitor_count()
        self.conn.execute(
            "UPDATE sessions SET session_end=?, unique_visitors=? WHERE id=?",
            (ts, count, session_id),
        )
        self.conn.commit()
        logger.info(f"[DB] Session {session_id} ended. Unique visitors: {count}")

    # ------------------------------------------------------------------ #
    #  Utility
    # ------------------------------------------------------------------ #

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary dict for reporting."""
        visitors = self.get_unique_visitor_count()
        total_events = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM events"
        ).fetchone()["cnt"]
        entry_events = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM events WHERE event_type='entry'"
        ).fetchone()["cnt"]
        exit_events = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM events WHERE event_type='exit'"
        ).fetchone()["cnt"]
        return {
            "unique_visitors": visitors,
            "total_events": total_events,
            "entry_events": entry_events,
            "exit_events": exit_events,
        }

    def close(self):
        """Close the database connection gracefully."""
        if self.conn:
            self.conn.close()
            logger.info("[DB] Connection closed.")
