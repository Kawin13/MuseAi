"""
config.py - Central configuration for Music Generator
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(BASE_DIR, "dataset")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
GENERATED_DIR= os.path.join(BASE_DIR, "generated")

NOTES_FILE   = os.path.join(MODELS_DIR, "notes.pkl")
MAPPING_FILE = os.path.join(MODELS_DIR, "mapping.pkl")
MODEL_FILE   = os.path.join(MODELS_DIR, "music_model.keras")
DB_FILE      = os.path.join(BASE_DIR,   "music.db")

# ── Model hyper-parameters ────────────────────────────────────────────────────
SEQUENCE_LENGTH = 100          # notes fed as context
LSTM_UNITS      = 256
DROPOUT_RATE    = 0.3
EPOCHS          = 50
BATCH_SIZE      = 64
LEARNING_RATE   = 0.001

# ── Generation ─────────────────────────────────────────────────────────────────
GENERATE_LENGTH = 500
DEFAULT_TEMP    = 1.0          # temperature for sampling

# ── Flask ──────────────────────────────────────────────────────────────────────
SECRET_KEY  = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-prod")
DEBUG       = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
PORT        = int(os.environ.get("PORT", 5000))

# ── Dataset mirrors (public domain) ───────────────────────────────────────────
# These are direct links to MIDI files available for free download.
# The dataset loader will attempt to pull a sample set automatically.
DATASET_SOURCES = [
    # Tiny subset of the Mutopia classical MIDI corpus (Bach)
    "https://www.midiworld.com/download/4518",   # Bach Prelude C Major
    "https://www.midiworld.com/download/4514",   # Bach Invention 1
    "https://www.midiworld.com/download/4509",   # Bach Invention 2
    "https://www.midiworld.com/download/4508",   # Bach Invention 3
]

# Ensure directories exist at import time
for _d in (DATASET_DIR, MODELS_DIR, GENERATED_DIR):
    os.makedirs(_d, exist_ok=True)
