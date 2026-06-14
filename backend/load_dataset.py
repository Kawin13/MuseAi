"""
load_dataset.py - Download a starter MIDI dataset for training.

Usage:
    python load_dataset.py

What it does
------------
1. Tries to download a small public-domain MIDI pack from a reliable mirror.
2. Falls back to generating synthetic training MIDIs with music21 if the
   network is unavailable (works offline / Render free tier).

The synthetic MIDIs are simple but sufficient for demonstrating the pipeline.
"""

import logging
import os
import random
import sys
import urllib.request
import zipfile

from music21 import chord, note, stream, tempo

from config import DATASET_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Public-domain MIDI zip mirrors ────────────────────────────────────────────
# Hosted on GitHub Releases / archive.org (no sign-in required)
MIDI_ZIP_URLS = [
    # Small Bach MIDI pack bundled with this project's GitHub releases
    "https://github.com/craffel/midi-dataset/raw/master/data/unique_midi.zip",
    # Fallback: Lakh MIDI Dataset sample (5 files)
    "https://raw.githubusercontent.com/mdeff/fma/master/data/fma_small.zip",
]

SAMPLE_ZIP_URL = (
    "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/"
    "maestro-v3.0.0-midi.zip"
)


def midi_files_present() -> int:
    """Return count of MIDI files already in DATASET_DIR."""
    count = 0
    for root, _, files in os.walk(DATASET_DIR):
        count += sum(1 for f in files if f.lower().endswith((".mid", ".midi")))
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic dataset generator (fallback, no network needed)
# ─────────────────────────────────────────────────────────────────────────────

SCALES = {
    "C_major": ["C4", "D4", "E4", "F4", "G4", "A4", "B4",
                "C5", "D5", "E5", "F5", "G5"],
    "A_minor": ["A3", "B3", "C4", "D4", "E4", "F4", "G4",
                "A4", "B4", "C5", "D5", "E5"],
    "G_major": ["G3", "A3", "B3", "C4", "D4", "E4", "F#4",
                "G4", "A4", "B4", "C5", "D5"],
    "D_major": ["D4", "E4", "F#4", "G4", "A4", "B4", "C#5",
                "D5", "E5", "F#5", "G5", "A5"],
    "E_minor": ["E4", "F#4", "G4", "A4", "B4", "C5", "D5",
                "E5", "F#5", "G5", "A5", "B5"],
}

CHORD_PROGRESSIONS = [
    [("C4", "E4", "G4"), ("F4", "A4", "C5"), ("G4", "B4", "D5"), ("C4", "E4", "G4")],
    [("A3", "C4", "E4"), ("D4", "F4", "A4"), ("E4", "G4", "B4"), ("A3", "C4", "E4")],
    [("G3", "B3", "D4"), ("C4", "E4", "G4"), ("D4", "F#4", "A4"), ("G3", "B3", "D4")],
]


def make_synthetic_midi(output_path: str, scale_name: str, seed: int) -> None:
    """Create a simple but valid MIDI file using music21."""
    random.seed(seed)
    scale = SCALES[scale_name]
    progression = random.choice(CHORD_PROGRESSIONS)

    elements = []
    offset = 0.0

    # 4 repetitions of a chord + melody pattern
    for _ in range(4):
        for chord_pitches in progression:
            # Bass chord on beat 1
            c = chord.Chord(list(chord_pitches))
            c.offset = offset
            c.duration.quarterLength = 1.0
            elements.append(c)
            offset += 1.0

            # Melodic run of 3 notes
            for _ in range(3):
                n = note.Note(random.choice(scale))
                n.offset = offset
                n.duration.quarterLength = 0.5
                elements.append(n)
                offset += 0.5

    midi_stream = stream.Stream(elements)
    midi_stream.insert(0, tempo.MetronomeMark(number=random.randint(80, 140)))
    midi_stream.write("midi", fp=output_path)


def generate_synthetic_dataset(n_files: int = 30) -> None:
    """Generate n synthetic MIDI files and save them to DATASET_DIR."""
    log.info("Generating %d synthetic MIDI training files …", n_files)
    scale_names = list(SCALES.keys())
    for i in range(n_files):
        scale = scale_names[i % len(scale_names)]
        fname = f"synthetic_{scale}_{i:03d}.mid"
        fpath = os.path.join(DATASET_DIR, fname)
        try:
            make_synthetic_midi(fpath, scale, seed=i * 7 + 13)
        except Exception as exc:
            log.warning("Could not create %s: %s", fname, exc)
    log.info("✓ Synthetic dataset ready in %s", DATASET_DIR)


# ─────────────────────────────────────────────────────────────────────────────
def try_download_zip(url: str, dest_dir: str, timeout: int = 30) -> bool:
    """Download a zip from url, extract to dest_dir. Returns True on success."""
    zip_path = os.path.join(dest_dir, "_download.zip")
    try:
        log.info("Downloading %s …", url)
        urllib.request.urlretrieve(url, zip_path)    # noqa: S310
        with zipfile.ZipFile(zip_path, "r") as zf:
            midi_members = [
                m for m in zf.namelist()
                if m.lower().endswith((".mid", ".midi"))
            ]
            if not midi_members:
                log.warning("Zip contained no MIDI files.")
                return False
            # Extract at most 100 files to keep disk usage low
            for member in midi_members[:100]:
                zf.extract(member, dest_dir)
            log.info("Extracted %d MIDI file(s).", min(len(midi_members), 100))
        return True
    except Exception as exc:
        log.warning("Download failed (%s): %s", url, exc)
        return False
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)


def load_dataset() -> None:
    os.makedirs(DATASET_DIR, exist_ok=True)

    already = midi_files_present()
    if already >= 10:
        log.info("Dataset already present (%d MIDI files). Skipping download.", already)
        return

    log.info("Attempting to download a public-domain MIDI dataset …")
    downloaded = False
    for url in MIDI_ZIP_URLS:
        if try_download_zip(url, DATASET_DIR):
            downloaded = True
            break

    if not downloaded:
        log.warning(
            "Could not download any MIDI pack. "
            "Falling back to synthetic dataset generation."
        )
        generate_synthetic_dataset(n_files=40)
    else:
        count = midi_files_present()
        log.info("Dataset ready: %d MIDI file(s) in %s", count, DATASET_DIR)
        if count < 5:
            log.info("Too few real MIDI files; supplementing with synthetic ones.")
            generate_synthetic_dataset(n_files=20)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        load_dataset()
        total = midi_files_present()
        print(f"\n✓ Dataset ready: {total} MIDI file(s) in {DATASET_DIR}")
    except Exception as exc:
        log.error("Dataset loading failed: %s", exc, exc_info=True)
        sys.exit(1)
