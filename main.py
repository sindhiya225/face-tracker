"""
main.py
Entry point for the Intelligent Face Tracker system.

Usage:
    python main.py                          # Uses config.json defaults
    python main.py --source video.mp4       # Override video source
    python main.py --rtsp                   # Use RTSP stream from config
    python main.py --no-display             # Headless mode (no GUI window)
    python main.py --summary                # Print DB summary and exit
"""

import os
import sys
import json
import argparse
import logging
import signal
import cv2

from logging_system.event_logger import setup_logger
from core.pipeline import FaceTrackingPipeline


# ── Globals for graceful shutdown ──────────────────────────────────────────────
pipeline: FaceTrackingPipeline = None
cap: cv2.VideoCapture = None
RUNNING = True


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    global RUNNING
    logging.getLogger(__name__).info("[Main] Interrupt received — shutting down.")
    RUNNING = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ── Config loader ──────────────────────────────────────────────────────────────

def load_config(path: str = "config.json") -> dict:
    """Load and return configuration from JSON file."""
    if not os.path.exists(path):
        print(f"[WARN] config.json not found at '{path}'. Using defaults.")
        return {}
    with open(path, "r") as f:
        return json.load(f)


# ── Video source ───────────────────────────────────────────────────────────────

def open_capture(config: dict, args: argparse.Namespace) -> cv2.VideoCapture:
    """
    Open the appropriate video capture source:
    - RTSP stream (if --rtsp or config.camera.use_rtsp=true)
    - File path (from --source or config.camera.source)
    """
    cam_cfg = config.get("camera", {})

    if args.rtsp or cam_cfg.get("use_rtsp", False):
        source = cam_cfg.get("rtsp_url", "")
        logging.getLogger(__name__).info(f"[Main] Opening RTSP stream: {source}")
    elif args.source:
        source = args.source
        logging.getLogger(__name__).info(f"[Main] Opening file: {source}")
    else:
        source = cam_cfg.get("source", "sample_video.mp4")
        logging.getLogger(__name__).info(f"[Main] Opening file: {source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Cannot open video source: {source}")
    return cap, str(source)


# ── Main loop ──────────────────────────────────────────────────────────────────

def run(config: dict, args: argparse.Namespace):
    """Main processing loop."""
    global pipeline, cap, RUNNING

    logger = logging.getLogger(__name__)

    # Override display setting if --no-display
    if args.no_display:
        config.setdefault("display", {})["show_window"] = False

    pipeline = FaceTrackingPipeline(config)
    cap, source_name = open_capture(config, args)
    pipeline.start_session(source_name)

    fps_limit = config.get("camera", {}).get("fps_limit", 30)
    show_window = config.get("display", {}).get("show_window", True)
    window_name = config.get("display", {}).get("window_name", "Face Tracker")

    logger.info("[Main] Starting main loop...")

    frame_idx = 0
    while RUNNING:
        ret, frame = cap.read()
        if not ret:
            logger.info("[Main] End of stream or read error. Stopping.")
            break

        frame_idx += 1

        # Process frame through full pipeline
        annotated = pipeline.process_frame(frame)

        # Display
        if show_window:
            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:   # q or ESC
                logger.info("[Main] User pressed quit.")
                break

    # ── Cleanup ────────────────────────────────────────────────────────
    cap.release()
    if show_window:
        cv2.destroyAllWindows()

    pipeline.end_session()

    # Print summary (DB still open at this point)
    summary = pipeline.db.get_summary()
    pipeline.db.close()   # Close only after reading summary

    print("\n" + "=" * 50)
    print("  FACE TRACKER — SESSION SUMMARY")
    print("=" * 50)
    print(f"  Unique visitors  : {summary['unique_visitors']}")
    print(f"  Entry events     : {summary['entry_events']}")
    print(f"  Exit events      : {summary['exit_events']}")
    print(f"  Total events     : {summary['total_events']}")
    print("=" * 50)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Intelligent Face Tracker with Auto-Registration & Visitor Counting"
    )
    parser.add_argument("--config",      default="config.json", help="Path to config.json")
    parser.add_argument("--source",      default=None,          help="Override video source path")
    parser.add_argument("--rtsp",        action="store_true",   help="Use RTSP stream from config")
    parser.add_argument("--no-display",  action="store_true",   help="Run headless (no GUI)")
    parser.add_argument("--summary",     action="store_true",   help="Print DB summary and exit")
    args = parser.parse_args()

    config = load_config(args.config)

    # Setup logging first
    log_cfg = config.get("logging", {})
    setup_logger(
        log_file=log_cfg.get("log_file", "logs/events.log"),
        log_level=log_cfg.get("log_level", "INFO"),
    )
    logger = logging.getLogger(__name__)
    logger.info("[Main] Face Tracker starting up.")

    if args.summary:
        from database.db_manager import DatabaseManager
        db_cfg = config.get("database", {})
        db = DatabaseManager(db_path=db_cfg.get("path", "face_tracker.db"))
        summary = db.get_summary()
        db.close()
        print(json.dumps(summary, indent=2))
        return

    try:
        run(config, args)
    except IOError as e:
        logger.error(f"[Main] IO error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"[Main] Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
