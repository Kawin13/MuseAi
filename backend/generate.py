"""
generate.py - Generate a new MIDI file from the trained model.

Usage:
    python generate.py
    python generate.py --length 300 --temperature 0.9 --seed 42
"""

import argparse
import logging
import os
import pickle
import sys
import time

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from music21 import chord, instrument, note, stream, tempo

from config import (
    DEFAULT_TEMP,
    GENERATE_LENGTH,
    GENERATED_DIR,
    MAPPING_FILE,
    MODEL_FILE,
    NOTES_FILE,
    SEQUENCE_LENGTH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
def load_model_and_data():
    """Load the trained Keras model plus the note/mapping pickles."""
    import tensorflow as tf

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_FILE}. "
            "Run 'python train.py' first."
        )
    if not os.path.exists(NOTES_FILE) or not os.path.exists(MAPPING_FILE):
        raise FileNotFoundError(
            "notes.pkl / mapping.pkl missing. Run 'python preprocess.py' first."
        )

    model = tf.keras.models.load_model(MODEL_FILE)
    log.info("Model loaded ← %s", MODEL_FILE)

    with open(NOTES_FILE, "rb") as fh:
        notes = pickle.load(fh)
    with open(MAPPING_FILE, "rb") as fh:
        mapping = pickle.load(fh)

    return model, notes, mapping["note_to_int"], mapping["int_to_note"]


def sample_with_temperature(probs: np.ndarray, temperature: float) -> int:
    """Sample an index from a probability array with temperature scaling."""
    probs = np.asarray(probs, dtype="float64")
    probs = np.log(probs + 1e-10) / max(temperature, 1e-6)
    probs = np.exp(probs - np.max(probs))
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)


def generate_note_sequence(
    model,
    notes: list[str],
    note_to_int: dict,
    int_to_note: dict,
    length: int,
    temperature: float,
    seed: int | None,
) -> list[str]:
    """Run the model autoregressively to produce a list of note tokens."""
    if seed is not None:
        np.random.seed(seed)

    vocab_size   = len(note_to_int)
    int_notes    = [note_to_int[n] for n in notes]
    start_index  = np.random.randint(0, len(int_notes) - SEQUENCE_LENGTH)
    pattern      = int_notes[start_index : start_index + SEQUENCE_LENGTH]

    generated: list[str] = []
    log.info("Generating %d notes (temperature=%.2f) …", length, temperature)

    for step in range(length):
        x = np.reshape(pattern, (1, SEQUENCE_LENGTH, 1)) / float(vocab_size)
        probs = model.predict(x, verbose=0)[0]
        idx   = sample_with_temperature(probs, temperature)

        generated.append(int_to_note[idx])
        pattern.append(idx)
        pattern = pattern[1:]

        if (step + 1) % 100 == 0:
            log.info("  %d / %d notes generated …", step + 1, length)

    return generated


def notes_to_midi(
    note_tokens: list[str],
    output_path: str,
    bpm: int = 120,
) -> str:
    """
    Convert a flat list of note/chord token strings into a MIDI file.

    Token formats accepted:
        "C4"      → single note
        "0.4.7"   → chord by MIDI normal-order integers
        "C.E.G"   → chord by pitch names (dot-separated)
    """
    offset       = 0.0
    quarter_dur  = 0.5     # duration of each element in quarter notes
    output_notes = []

    for token in note_tokens:
        if "." in token:
            # ── Chord ──────────────────────────────────────────────────────────
            parts = token.split(".")
            chord_notes = []
            for p in parts:
                try:
                    # Try integer (normal-order) first
                    n = note.Note(int(p))
                except (ValueError, Exception):
                    try:
                        n = note.Note(p)
                    except Exception:
                        continue
                n.storedInstrument = instrument.Piano()
                chord_notes.append(n)

            if chord_notes:
                c = chord.Chord(chord_notes)
                c.offset   = offset
                c.duration.quarterLength = quarter_dur
                output_notes.append(c)
        else:
            # ── Single note ────────────────────────────────────────────────────
            try:
                n = note.Note(token)
                n.offset   = offset
                n.duration.quarterLength = quarter_dur
                n.storedInstrument = instrument.Piano()
                output_notes.append(n)
            except Exception as exc:
                log.debug("Skipping bad token '%s': %s", token, exc)

        offset += quarter_dur

    midi_stream = stream.Stream(output_notes)
    midi_stream.insert(0, tempo.MetronomeMark(number=bpm))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    midi_stream.write("midi", fp=output_path)
    return output_path


def generate(
    length: int      = GENERATE_LENGTH,
    temperature: float = DEFAULT_TEMP,
    seed: int | None = None,
    output_filename: str | None = None,
) -> str:
    """
    End-to-end generation pipeline. Returns path to the saved MIDI file.
    """
    model, notes, note_to_int, int_to_note = load_model_and_data()

    note_tokens = generate_note_sequence(
        model, notes, note_to_int, int_to_note,
        length=length, temperature=temperature, seed=seed,
    )

    if output_filename is None:
        ts = int(time.time())
        output_filename = f"generated_{ts}.mid"

    output_path = os.path.join(GENERATED_DIR, output_filename)
    notes_to_midi(note_tokens, output_path)
    log.info("✓ MIDI saved → %s", output_path)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate music with the trained LSTM.")
    parser.add_argument("--length",      type=int,   default=GENERATE_LENGTH, help="Notes to generate.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMP,    help="Sampling temperature (0.1–2.0).")
    parser.add_argument("--seed",        type=int,   default=None,            help="Random seed for reproducibility.")
    parser.add_argument("--output",      type=str,   default=None,            help="Output filename (inside generated/).")
    args = parser.parse_args()

    try:
        path = generate(
            length=args.length,
            temperature=args.temperature,
            seed=args.seed,
            output_filename=args.output,
        )
        print(f"\n✓ Generated MIDI → {path}")
    except Exception as exc:
        log.error("Generation failed: %s", exc, exc_info=True)
        sys.exit(1)
