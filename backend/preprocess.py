"""
preprocess.py - Parse MIDI files into note sequences and build vocabulary.

Usage:
    python preprocess.py
"""

import os
import sys
import glob
import pickle
import logging
from collections import Counter

import numpy as np
from music21 import converter, instrument, note, chord, stream

from config import DATASET_DIR, MODELS_DIR, NOTES_FILE, MAPPING_FILE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
def get_midi_files() -> list[str]:
    """Return all .mid / .midi files found recursively in DATASET_DIR."""
    patterns = [
        os.path.join(DATASET_DIR, "**", "*.mid"),
        os.path.join(DATASET_DIR, "**", "*.midi"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    return sorted(set(files))


def parse_midi(filepath: str) -> list[str]:
    """
    Parse a single MIDI file and return a flat list of note/chord strings.

    Notes  → pitch name + octave, e.g. "C4", "G#5"
    Chords → dot-joined pitch names, e.g. "C.E.G"
    Rests  → skipped (keeps model simpler)
    """
    notes_out: list[str] = []
    try:
        midi = converter.parse(filepath)
    except Exception as exc:
        log.warning("Could not parse %s: %s", filepath, exc)
        return []

    try:
        parts = instrument.partitionByInstrument(midi)
        if parts:
            part_to_use = parts.parts[0]
        else:
            part_to_use = midi.flat
    except Exception:
        part_to_use = midi.flat

    for element in part_to_use.notesAndRests:
        try:
            if isinstance(element, note.Note):
                notes_out.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                chord_str = ".".join(str(n) for n in element.normalOrder)
                notes_out.append(chord_str)
            # rests are skipped
        except Exception as exc:
            log.debug("Skipping element in %s: %s", filepath, exc)

    return notes_out


def build_vocabulary(all_notes: list[str]) -> tuple[dict, dict]:
    """
    Build integer ↔ token mappings sorted by frequency (most common first).

    Returns
    -------
    note_to_int : dict[str, int]
    int_to_note : dict[int, str]
    """
    counts = Counter(all_notes)
    vocab  = [token for token, _ in counts.most_common()]
    note_to_int = {token: idx for idx, token in enumerate(vocab)}
    int_to_note = {idx: token for token, idx in note_to_int.items()}
    return note_to_int, int_to_note


def preprocess() -> tuple[list[str], dict, dict]:
    """
    Main preprocessing pipeline.

    1. Discover MIDI files.
    2. Parse each file.
    3. Build vocabulary.
    4. Persist notes.pkl and mapping.pkl.

    Returns (all_notes, note_to_int, int_to_note).
    Raises RuntimeError if no MIDI files found.
    """
    midi_files = get_midi_files()
    if not midi_files:
        raise RuntimeError(
            f"No MIDI files found in {DATASET_DIR}. "
            "Run 'python load_dataset.py' first, or drop .mid files into backend/dataset/."
        )

    log.info("Found %d MIDI file(s). Parsing …", len(midi_files))
    all_notes: list[str] = []

    for idx, fp in enumerate(midi_files, start=1):
        log.info("  [%d/%d] %s", idx, len(midi_files), os.path.basename(fp))
        parsed = parse_midi(fp)
        log.info("         → %d note tokens", len(parsed))
        all_notes.extend(parsed)

    if not all_notes:
        raise RuntimeError(
            "Parsed 0 note tokens from all MIDI files. "
            "Files may be corrupt or in an unsupported format."
        )

    log.info("Total note tokens: %d", len(all_notes))

    note_to_int, int_to_note = build_vocabulary(all_notes)
    vocab_size = len(note_to_int)
    log.info("Vocabulary size : %d unique tokens", vocab_size)

    # ── Persist ────────────────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)

    with open(NOTES_FILE, "wb") as fh:
        pickle.dump(all_notes, fh)
    log.info("Saved notes   → %s", NOTES_FILE)

    with open(MAPPING_FILE, "wb") as fh:
        pickle.dump({"note_to_int": note_to_int, "int_to_note": int_to_note}, fh)
    log.info("Saved mapping → %s", MAPPING_FILE)

    return all_notes, note_to_int, int_to_note


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        notes, n2i, i2n = preprocess()
        print(f"\n✓ Preprocessing complete. {len(notes)} tokens, {len(n2i)} unique.")
    except RuntimeError as err:
        print(f"\n✗ Error: {err}", file=sys.stderr)
        sys.exit(1)
