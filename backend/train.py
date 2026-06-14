"""
train.py - Build and train the LSTM music model.

Usage:
    python train.py                   # fresh training
    python train.py --resume          # resume from last checkpoint
    python train.py --epochs 30       # override epoch count
"""

import argparse
import logging
import os
import pickle
import sys

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # quieter TF logs

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Input,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from config import (
    BATCH_SIZE,
    DROPOUT_RATE,
    EPOCHS,
    LEARNING_RATE,
    LSTM_UNITS,
    MAPPING_FILE,
    MODEL_FILE,
    MODELS_DIR,
    NOTES_FILE,
    SEQUENCE_LENGTH,
)
from preprocess import preprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
def load_processed_data() -> tuple[list[str], dict, dict]:
    """Load notes + mappings, running preprocessing first if needed."""
    if not os.path.exists(NOTES_FILE) or not os.path.exists(MAPPING_FILE):
        log.info("Preprocessed data not found. Running preprocess …")
        notes, note_to_int, int_to_note = preprocess()
    else:
        with open(NOTES_FILE, "rb") as fh:
            notes = pickle.load(fh)
        with open(MAPPING_FILE, "rb") as fh:
            mapping = pickle.load(fh)
        note_to_int = mapping["note_to_int"]
        int_to_note = mapping["int_to_note"]
        log.info(
            "Loaded %d notes, %d-token vocabulary from cache.",
            len(notes),
            len(note_to_int),
        )
    return notes, note_to_int, int_to_note


def create_sequences(
    notes: list[str],
    note_to_int: dict,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Slide a window over the note list to produce (X, y) pairs.

    X shape: (n_samples, sequence_length, 1)
    y shape: (n_samples, vocab_size)  — one-hot
    """
    vocab_size = len(note_to_int)
    int_notes  = [note_to_int[n] for n in notes]

    X_raw, y_raw = [], []
    for i in range(len(int_notes) - sequence_length):
        X_raw.append(int_notes[i : i + sequence_length])
        y_raw.append(int_notes[i + sequence_length])

    n_samples = len(X_raw)
    log.info("Created %d training sequences (seq_len=%d).", n_samples, sequence_length)

    X = np.reshape(X_raw, (n_samples, sequence_length, 1)) / float(vocab_size)
    y = to_categorical(y_raw, num_classes=vocab_size)
    return X, y, vocab_size


def build_model(sequence_length: int, vocab_size: int) -> tf.keras.Model:
    """Define the LSTM architecture."""
    inputs = Input(shape=(sequence_length, 1))
    x = LSTM(LSTM_UNITS, return_sequences=True)(inputs)
    x = Dropout(DROPOUT_RATE)(x)
    x = LSTM(LSTM_UNITS)(x)
    x = Dense(LSTM_UNITS, activation="relu")(x)
    outputs = Dense(vocab_size, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=log.info)
    return model


def get_callbacks(checkpoint_path: str) -> list:
    return [
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="loss",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def train(epochs: int = EPOCHS, resume: bool = False) -> None:
    notes, note_to_int, _ = load_processed_data()
    X, y, vocab_size = create_sequences(notes, note_to_int, SEQUENCE_LENGTH)

    checkpoint_path = os.path.join(MODELS_DIR, "checkpoint.keras")

    if resume and os.path.exists(MODEL_FILE):
        log.info("Resuming from existing model: %s", MODEL_FILE)
        model = load_model(MODEL_FILE)
    elif resume and os.path.exists(checkpoint_path):
        log.info("Resuming from checkpoint: %s", checkpoint_path)
        model = load_model(checkpoint_path)
    else:
        log.info("Building new model (vocab_size=%d) …", vocab_size)
        model = build_model(SEQUENCE_LENGTH, vocab_size)

    callbacks = get_callbacks(checkpoint_path)

    log.info("Training for up to %d epoch(s) …", epochs)
    history = model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(MODEL_FILE)
    log.info("✓ Model saved → %s", MODEL_FILE)

    final_loss = history.history["loss"][-1]
    log.info("Final training loss: %.4f", final_loss)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the LSTM music model.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Max training epochs.")
    parser.add_argument("--resume", action="store_true", help="Resume from saved model.")
    args = parser.parse_args()

    try:
        train(epochs=args.epochs, resume=args.resume)
    except Exception as exc:
        log.error("Training failed: %s", exc, exc_info=True)
        sys.exit(1)
