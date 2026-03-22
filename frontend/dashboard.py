"""
frontend/dashboard.py
Optional Flask dashboard for real-time monitoring of the face tracker.
Run with: python frontend/dashboard.py
Access at: http://localhost:5000
"""

import os
import sys
import json
import logging
from datetime import datetime

# Ensure parent package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from flask import Flask, jsonify, render_template_string, send_from_directory
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

# ── Flask App ──────────────────────────────────────────────────────────────────

app = Flask(__name__) if FLASK_AVAILABLE else None
if FLASK_AVAILABLE:
    CORS(app)

DB_PATH = os.environ.get("DB_PATH", "face_tracker.db")


def get_db():
    return DatabaseManager(db_path=DB_PATH)


# ── HTML Template ──────────────────────────────────────────────────────────────

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Face Tracker Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #e6edf3; }
  header { background: #161b22; padding: 16px 24px; border-bottom: 1px solid #30363d;
           display: flex; justify-content: space-between; align-items: center; }
  header h1 { font-size: 1.4rem; color: #58a6ff; }
  .badge { background: #1f6feb; color: #fff; padding: 4px 12px; border-radius: 12px;
           font-size: 0.8rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 16px; padding: 24px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
          padding: 20px; text-align: center; }
  .card .value { font-size: 2.5rem; font-weight: 700; color: #58a6ff; }
  .card .label { font-size: 0.85rem; color: #8b949e; margin-top: 6px; }
  table { width: calc(100% - 48px); margin: 0 24px 24px; border-collapse: collapse;
          background: #161b22; border-radius: 8px; overflow: hidden;
          border: 1px solid #30363d; }
  th { background: #21262d; padding: 10px 14px; text-align: left;
       color: #8b949e; font-size: 0.8rem; text-transform: uppercase; }
  td { padding: 10px 14px; border-top: 1px solid #30363d; font-size: 0.85rem; }
  .entry { color: #3fb950; } .exit { color: #f85149; }
  .refresh { cursor: pointer; background: #21262d; border: 1px solid #30363d;
             color: #e6edf3; padding: 6px 14px; border-radius: 6px; font-size: 0.85rem; }
  .refresh:hover { background: #30363d; }
</style>
</head>
<body>
<header>
  <h1>🎭 Face Tracker Dashboard</h1>
  <div style="display:flex;gap:10px;align-items:center;">
    <span id="last-updated" style="color:#8b949e;font-size:0.8rem;"></span>
    <button class="refresh" onclick="loadData()">↻ Refresh</button>
    <span class="badge" id="status">Live</span>
  </div>
</header>

<div class="grid" id="stats-grid">
  <div class="card"><div class="value" id="unique-count">—</div><div class="label">Unique Visitors</div></div>
  <div class="card"><div class="value" id="entry-count">—</div><div class="label">Total Entries</div></div>
  <div class="card"><div class="value" id="exit-count">—</div><div class="label">Total Exits</div></div>
  <div class="card"><div class="value" id="event-count">—</div><div class="label">All Events</div></div>
</div>

<h2 style="padding:0 24px 12px;color:#8b949e;font-size:0.9rem;">RECENT EVENTS</h2>
<table>
  <thead><tr><th>Face ID</th><th>Event</th><th>Timestamp</th><th>Image</th></tr></thead>
  <tbody id="events-body"></tbody>
</table>

<script>
async function loadData() {
  const r = await fetch('/api/summary');
  const d = await r.json();
  document.getElementById('unique-count').textContent = d.unique_visitors;
  document.getElementById('entry-count').textContent  = d.entry_events;
  document.getElementById('exit-count').textContent   = d.exit_events;
  document.getElementById('event-count').textContent  = d.total_events;
  document.getElementById('last-updated').textContent = 'Updated ' + new Date().toLocaleTimeString();

  const er = await fetch('/api/events?limit=50');
  const events = await er.json();
  const tbody = document.getElementById('events-body');
  tbody.innerHTML = events.map(e => `
    <tr>
      <td>${e.face_uuid}</td>
      <td class="${e.event_type}">${e.event_type.toUpperCase()}</td>
      <td>${e.timestamp}</td>
      <td>${e.image_path ? '<a href="/image?path='+encodeURIComponent(e.image_path)+'" target="_blank">View</a>' : '—'}</td>
    </tr>
  `).join('');
}
loadData();
setInterval(loadData, 5000);
</script>
</body>
</html>
"""


# ── API Routes ─────────────────────────────────────────────────────────────────

if FLASK_AVAILABLE:

    @app.route("/")
    def index():
        return render_template_string(DASHBOARD_HTML)

    @app.route("/api/summary")
    def api_summary():
        db = get_db()
        data = db.get_summary()
        db.close()
        return jsonify(data)

    @app.route("/api/events")
    def api_events():
        from flask import request
        limit = int(request.args.get("limit", 100))
        db = get_db()
        events = db.get_all_events()[-limit:]
        db.close()
        return jsonify(events)

    @app.route("/api/visitors")
    def api_visitors():
        db = get_db()
        count = db.get_unique_visitor_count()
        db.close()
        return jsonify({"unique_visitors": count})

    @app.route("/image")
    def serve_image():
        from flask import request, send_file, abort
        path = request.args.get("path", "")
        if not path or not os.path.exists(path):
            abort(404)
        return send_file(os.path.abspath(path), mimetype="image/jpeg")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not FLASK_AVAILABLE:
        print("Flask not installed. Run: pip install flask flask-cors")
        sys.exit(1)
    print("Dashboard running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
