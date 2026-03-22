# 🎭 Intelligent Face Tracker with Auto-Registration & Visitor Counting

> **This project is a part of a hackathon run by [https://katomaran.com](https://katomaran.com)**

---

## 📌 Overview

A production-grade, AI-driven unique visitor counter that processes a video stream (file or live RTSP camera) to:
- **Detect** faces in real-time using YOLOv8
- **Recognize** faces using InsightFace (ArcFace 512-dim embeddings)
- **Track** faces continuously across frames using IoU-based multi-object tracking
- **Auto-register** new faces upon first detection and assign unique IDs
- **Log** every entry and exit with timestamped cropped images
- **Count** unique visitors accurately throughout the stream

---

## 🏗 Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        VIDEO SOURCE                                 │
│            (MP4 file  ──or──  RTSP camera stream)                   │
└──────────────────────────────┬─────────────────────────────────────┘
                                │ frames
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                     FACE TRACKING PIPELINE                        │
│                       (core/pipeline.py)                          │
│                                                                    │
│  ┌──────────────┐    ┌───────────────┐    ┌──────────────────┐   │
│  │  Face        │    │  Face         │    │  Face            │   │
│  │  Detector    │───▶│  Recognizer   │───▶│  Tracker         │   │
│  │  (YOLOv8)   │    │  (InsightFace)│    │  (IoU/Centroid)  │   │
│  └──────────────┘    └───────────────┘    └──────────────────┘   │
│       detect()           embed()              update()             │
│       crop_face()        match()              get_active()         │
└────────────────────────────┬─────────────────────────────────────┘
                              │ events
                 ┌────────────┴────────────┐
                 ▼                         ▼
    ┌────────────────────┐    ┌─────────────────────────┐
    │  DATABASE MANAGER  │    │    EVENT LOGGER          │
    │  (SQLite)          │    │  (File + Image Store)    │
    │                    │    │                          │
    │  • faces table     │    │  • logs/events.log       │
    │  • events table    │    │  • logs/entries/DATE/    │
    │  • sessions table  │    │  • logs/exits/DATE/      │
    └────────────────────┘    └─────────────────────────┘
                 │
                 ▼
    ┌────────────────────┐
    │ FLASK DASHBOARD    │  (optional)
    │  localhost:5000    │
    │  /api/summary      │
    │  /api/events       │
    └────────────────────┘
```

---

## 📂 Project Structure

```
face_tracker/
├── main.py                        # Entry point
├── config.json                    # All configuration parameters
├── requirements.txt               # Python dependencies
├── generate_sample_output.py      # Generates demo DB/log entries
│
├── core/
│   ├── face_detector.py           # YOLOv8 face detection (+ Haar fallback)
│   ├── face_recognizer.py         # InsightFace ArcFace embedding + matching
│   ├── face_tracker.py            # Multi-face IoU tracker (entry/exit tracking)
│   └── pipeline.py                # Orchestration: detect → recognize → track → log
│
├── database/
│   └── db_manager.py              # SQLite CRUD: faces, events, sessions tables
│
├── logging_system/
│   └── event_logger.py            # File logger + structured image store
│
├── utils/
│   └── video_utils.py             # Video capture helpers
│
├── frontend/
│   └── dashboard.py               # Optional Flask live dashboard
│
└── logs/
    ├── events.log                 # All system events (auto-created)
    ├── entries/
    │   └── YYYY-MM-DD/            # Cropped entry images per day
    └── exits/
        └── YYYY-MM-DD/            # Cropped exit images per day
```

---

## ⚙️ Setup Instructions

### 1. Prerequisites
- Python 3.9+
- pip
- (Optional) NVIDIA GPU with CUDA 11.8+ for accelerated inference

### 2. Clone & Install

```bash
git clone <https://github.com/sindhiya225/face-tracker>
cd face_tracker

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Download YOLOv8 Face Model

The YOLOv8 face model will be auto-downloaded by `ultralytics` on first run. Alternatively:

```bash
# Option A: Let ultralytics auto-download yolov8n-face.pt
# Option B: Download manually and place in project root
wget https://github.com/derronqi/yolov8-face/releases/download/v1/yolov8n-face.pt
```

InsightFace's `buffalo_l` model is auto-downloaded on first run to `~/.insightface/models/`.

### 4. Configure

Edit `config.json` to match your setup:
```json
{
  "detection": {
    "skip_frames": 3,          // Process every Nth frame for detection
    "confidence_threshold": 0.5
  },
  "camera": {
    "source": "sample_video.mp4",
    "rtsp_url": "rtsp://user:pass@192.168.1.100:554/stream",
    "use_rtsp": false          // Set true to use RTSP instead of file
  }
}
```

### 5. Run

```bash
# Process a video file (default from config.json)
python main.py

# Process a specific video file
python main.py --source /path/to/video.mp4

# Use RTSP camera stream
python main.py --rtsp

# Run headless (no display window)
python main.py --no-display

# Show DB summary and exit
python main.py --summary

# Generate sample output (demo only, no video needed)
python generate_sample_output.py
```

### 6. Run Dashboard (Optional)

```bash
pip install flask flask-cors
python frontend/dashboard.py
# Open http://localhost:5000
```

---

## 📋 Sample `config.json`

```json
{
  "detection": {
    "skip_frames": 3,
    "confidence_threshold": 0.5,
    "yolo_model": "yolov8n-face.pt",
    "input_size": 640
  },
  "recognition": {
    "model_name": "buffalo_l",
    "similarity_threshold": 0.45,
    "embedding_size": 512
  },
  "tracking": {
    "max_disappeared": 30,
    "max_distance": 0.6,
    "iou_threshold": 0.3
  },
  "database": {
    "type": "sqlite",
    "path": "face_tracker.db"
  },
  "logging": {
    "log_file": "logs/events.log",
    "image_store": "logs",
    "log_level": "INFO"
  },
  "camera": {
    "source": "sample_video.mp4",
    "rtsp_url": "rtsp://username:password@ip:port/stream",
    "use_rtsp": false,
    "fps_limit": 30
  },
  "display": {
    "show_window": true,
    "draw_bboxes": true,
    "draw_ids": true,
    "window_name": "Face Tracker"
  }
}
```

---

## 🔑 Key Design Decisions & Assumptions

| Topic | Decision | Reason |
|-------|----------|--------|
| Face Detection | YOLOv8n-face | Best accuracy/speed for real-time face detection |
| Face Recognition | InsightFace `buffalo_l` | State-of-the-art ArcFace embeddings; explicitly listed in problem statement |
| Tracking | Custom IoU tracker | Lightweight, no extra dependencies; sufficient for fixed camera setups |
| Database | SQLite | Zero-config, file-based, ACID-compliant; easily swappable to PostgreSQL |
| Embedding match threshold | 0.55 cosine similarity (crowd) | Tuned for busy public space scenes; use `tune_threshold.py` to find the optimal value for your scene |
| Frame skip | Configurable (`skip_frames` in config.json) | Reduces CPU/GPU load on high-FPS streams |
| Exit detection | face_uuid absent for `max_disappeared` frames | Prevents spurious exits during brief occlusions; single authoritative exit per appearance window |
| Entry deduplication | `_in_frame` set per face_uuid | Guarantees exactly one ENTRY event per continuous physical appearance, regardless of tracker ID churn |
| Image storage | `logs/entries/YYYY-MM-DD/` and `logs/exits/YYYY-MM-DD/` | Organized by date for easy browsing |
| **Camera angle assumption** | **Frontal/near-frontal faces assumed** | InsightFace/ArcFace is trained on frontal faces. Overhead/top-down camera angles reduce embedding quality and may cause slight over-counting. A model fine-tuned on overhead angles would improve accuracy. |
| **Masked faces assumption** | **Masks treated as partial occlusion** | People wearing face masks produce unreliable embeddings since the nose/mouth region is occluded. Masked individuals may be assigned new IDs on re-entry. A mask-aware model would be needed for full accuracy in post-COVID public spaces. |
| **Re-identification window** | **Same session only** | Embeddings persist in the DB across sessions. A face seen in a previous run will be re-identified as RETURNING in subsequent runs. Clear the DB between independent sessions if needed. |
| One entry + one exit per track | Track lifecycle = one entry event + one exit event | Enforced at tracker level |
| Re-identification | Same person re-enters → new entry event, no new unique visitor | Re-id via embedding match; count not incremented |

---

## 📊 Compute Load Estimation

### CPU Mode (default)

| Component | Approx. CPU Usage | Notes |
|-----------|-------------------|-------|
| YOLOv8n-face detection | ~35–50% (1 core) | Per detection frame |
| InsightFace embedding | ~20–30% (1 core) | Per new face |
| IoU tracking | <1% | Pure numpy math |
| SQLite writes | <1% | Batched |
| **Total (1080p @ 30fps)** | **~60–80% CPU** | With `skip_frames=3` |

### GPU Mode (CUDA)

| Component | GPU VRAM | Notes |
|-----------|----------|-------|
| YOLOv8n-face | ~300 MB | CUDA inference |
| InsightFace `buffalo_l` | ~500 MB | CUDA execution provider |
| **Total** | **~800 MB VRAM** | Suitable for GTX 1060 6GB and above |

**GPU mode speedup**: ~5–8× faster than CPU mode on 1080p streams.

To enable GPU:
- In `face_detector.py`: change `device="cpu"` to `device=0`
- In `face_recognizer.py`: change `providers=["CPUExecutionProvider"]` to `["CUDAExecutionProvider"]`
- Install `onnxruntime-gpu` instead of `onnxruntime`

---

## 🧪 Sample Output

### main.py
<img width="869" height="301" alt="Screenshot 2026-03-22 214244" src="https://github.com/user-attachments/assets/61018f0f-bf60-48b2-9802-3f05492442f3" />
<img width="1622" height="719" alt="Screenshot 2026-03-22 214048" src="https://github.com/user-attachments/assets/7b15ffda-0bfd-43f3-89dc-f73ce4f9330f" />
<img width="1062" height="588" alt="Screenshot 2026-03-22 214156" src="https://github.com/user-attachments/assets/12ebd61e-61b5-4dca-88eb-d9859611d672" />


### `logs/events.log` 
<img width="1690" height="606" alt="Screenshot 2026-03-22 214419" src="https://github.com/user-attachments/assets/c4e77a7d-3b5d-4755-bc50-f632e6a1fcfa" />


### Database
<img width="1456" height="490" alt="Screenshot 2026-03-22 214501" src="https://github.com/user-attachments/assets/544c232b-89a1-459f-bea5-6c840559c844" />


### Dashboard
<img width="1888" height="876" alt="Screenshot 2026-03-22 214538" src="https://github.com/user-attachments/assets/f62048ce-4f0a-4a88-8df3-48de8e7e8879" />

---

## 🤖 AI Planning Document

### Planning Phase

The problem was decomposed into 5 functional layers:

1. **Ingestion Layer** — Accept any video source (file / RTSP)
2. **Detection Layer** — Per-frame bounding box extraction (YOLOv8)
3. **Recognition Layer** — Identity mapping via ArcFace embeddings (InsightFace)
4. **Tracking Layer** — Temporal association of detections (IoU tracker)
5. **Persistence Layer** — SQLite + structured filesystem for all events

### Feature List

- [x] Real-time face detection (YOLOv8, configurable skip frames)
- [x] ArcFace embedding generation (InsightFace `buffalo_l`)
- [x] Cosine similarity matching against all registered faces
- [x] Auto-registration of new faces with unique UUID
- [x] Re-identification of returning visitors (no double-counting)
- [x] IoU-based multi-face tracker (handles occlusions)
- [x] Exactly-one entry event per track lifecycle
- [x] Exactly-one exit event per track disappearance
- [x] Cropped face image saved at entry (and separately at exit)
- [x] Structured log file (`logs/events.log`) for all events
- [x] SQLite database with `faces`, `events`, `sessions` tables
- [x] Unique visitor count exposed via DB query
- [x] Configurable via `config.json`
- [x] Graceful shutdown (Ctrl+C / SIGTERM)
- [x] Optional Flask live dashboard with REST API
- [x] Haar cascade fallback if YOLO model unavailable
- [x] HOG-descriptor fallback if InsightFace unavailable

### Prompts Used During Development

1. "Design a modular Python architecture for a real-time face tracker with detection, recognition, tracking, logging, and database layers."
2. "Implement a multi-object tracker using IoU matching that fires entry and exit events based on track lifecycle."
3. "Write an InsightFace wrapper that generates ArcFace embeddings and does cosine-similarity matching against a list of stored embeddings."
4. "Design a SQLite schema for storing face identities, entry/exit events, and session metadata with WAL mode for concurrent access."
5. "Create a structured event logger that saves cropped face images to date-partitioned directories and writes formatted log lines."

---

## 🎥 Demo Video

> **[https://www.loom.com/share/1433246343c044fcac106cf4138073b6]**

---

## 📝 License

MIT License — free to use and modify.

---

> **This project is a part of a hackathon run by [https://katomaran.com](https://katomaran.com)**
