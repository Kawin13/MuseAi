"""
app.py - Flask REST API for the AI Music Generator.

Endpoints
---------
GET  /health          → liveness check
POST /train           → start training (background thread)
POST /generate        → generate a new MIDI file
GET  /download        → download the latest MIDI
GET  /songs           → list all generated songs
"""

import logging
import os
import sqlite3
import sys
import threading
import time
from datetime import datetime

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from config import DB_FILE, GENERATED_DIR, MODEL_FILE, PORT, SECRET_KEY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = SECRET_KEY
CORS(app)

# ── Training state shared between threads ─────────────────────────────────────
_training_lock   = threading.Lock()
_training_status = {
    "running": False,
    "progress": "",
    "error":   None,
    "finished": False,
}


# ─────────────────────────────────────────────────────────────────────────────
# Database helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS generated_songs (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                filename   TEXT    NOT NULL,
                created_at TEXT    NOT NULL
            )
            """
        )
        conn.commit()
    log.info("Database initialised: %s", DB_FILE)


def record_song(filename: str) -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO generated_songs (filename, created_at) VALUES (?, ?)",
            (filename, datetime.utcnow().isoformat()),
        )
        conn.commit()
        return cur.lastrowid


def list_songs() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, filename, created_at FROM generated_songs ORDER BY id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Background training worker
# ─────────────────────────────────────────────────────────────────────────────

def _training_worker(epochs: int) -> None:
    global _training_status
    try:
        # Import here so TF loads only when needed
        from train import train as run_training

        _training_status["progress"] = "Preprocessing data …"
        run_training(epochs=epochs, resume=False)
        _training_status["progress"] = "Training complete."
        _training_status["finished"] = True
    except Exception as exc:
        log.error("Background training error: %s", exc, exc_info=True)
        _training_status["error"]    = str(exc)
        _training_status["progress"] = "Training failed."
    finally:
        _training_status["running"] = False


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status":        "ok",
            "model_exists":  os.path.exists(MODEL_FILE),
            "generated_dir": GENERATED_DIR,
        }
    )


@app.route("/train", methods=["POST"])
def start_training():
    global _training_status

    with _training_lock:
        if _training_status["running"]:
            return jsonify({"error": "Training already in progress."}), 409

        data   = request.get_json(silent=True) or {}
        epochs = int(data.get("epochs", 50))
        epochs = max(1, min(epochs, 200))   # clamp to [1, 200]

        _training_status = {
            "running":  True,
            "progress": "Starting …",
            "error":    None,
            "finished": False,
        }

    thread = threading.Thread(
        target=_training_worker, args=(epochs,), daemon=True
    )
    thread.start()

    return jsonify(
        {"message": f"Training started ({epochs} epochs).", "epochs": epochs}
    )


@app.route("/train/status", methods=["GET"])
def training_status():
    return jsonify(_training_status)


@app.route("/generate", methods=["POST"])
def generate_music():
    if not os.path.exists(MODEL_FILE):
        return (
            jsonify(
                {
                    "error": (
                        "No trained model found. "
                        "Run POST /train first, or upload a pre-trained model."
                    )
                }
            ),
            400,
        )

    data        = request.get_json(silent=True) or {}
    length      = int(data.get("length",      500))
    temperature = float(data.get("temperature", 1.0))
    seed        = data.get("seed", None)
    if seed is not None:
        seed = int(seed)

    # Clamp values
    length      = max(50,  min(length,      2000))
    temperature = max(0.1, min(temperature, 2.0))

    try:
        from generate import generate

        ts       = int(time.time())
        filename = f"generated_{ts}.mid"
        path     = generate(
            length=length,
            temperature=temperature,
            seed=seed,
            output_filename=filename,
        )
        song_id = record_song(filename)
        return jsonify(
            {
                "message":  "Music generated successfully.",
                "filename": filename,
                "song_id":  song_id,
                "path":     path,
            }
        )
    except Exception as exc:
        log.error("Generation error: %s", exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500


@app.route("/download", methods=["GET"])
def download_latest():
    """Download the most recently generated MIDI file."""
    midi_files = sorted(
        [
            f for f in os.listdir(GENERATED_DIR)
            if f.endswith(".mid")
        ],
        reverse=True,
    )
    if not midi_files:
        return jsonify({"error": "No generated files found."}), 404

    filename = request.args.get("filename", midi_files[0])
    filepath = os.path.join(GENERATED_DIR, filename)

    if not os.path.exists(filepath):
        return jsonify({"error": f"File {filename} not found."}), 404

    return send_file(
        filepath,
        as_attachment=True,
        download_name=filename,
        mimetype="audio/midi",
    )


@app.route("/songs", methods=["GET"])
def get_songs():
    return jsonify(list_songs())


# ─────────────────────────────────────────────────────────────────────────────

def create_app() -> Flask:
    init_db()
    return app


if __name__ == "__main__":
    create_app()
    app.run(host="0.0.0.0", port=PORT, debug=False)
