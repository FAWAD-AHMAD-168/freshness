"""Train the CNN and save the best model and training charts."""
import argparse
import csv
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.config import (PROCESSED, MODELS, RESULTS, IMAGE_SIZE,
                        BATCH_SIZE, CLASS_NAMES, SEED)
from src.model import build_model


def load_split(name, shuffle):
    path = PROCESSED / name
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {path}. First run: python -m src.prepare_data")
    ds = tf.keras.utils.image_dataset_from_directory(
        path, labels="inferred", label_mode="binary", class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=shuffle,
        seed=SEED,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    tf.keras.utils.set_random_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    train_ds = load_split("train", shuffle=True)
    val_ds = load_split("val", shuffle=False)
    model = build_model()
    model.summary()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(MODELS / "fruit_freshness_model.keras"),
            monitor="val_loss", save_best_only=True, mode="min", verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True,
        ),
    ]
    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=args.epochs, callbacks=callbacks)
    keys = list(history.history)
    with (RESULTS / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", *keys])
        writer.writeheader()
        for i in range(len(history.epoch)):
            writer.writerow({"epoch": i+1, **{k: history.history[k][i] for k in keys}})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in ((axes[0], "accuracy", "Accuracy"),
                              (axes[1], "loss", "Loss")):
        ax.plot(history.history[metric], label="Training")
        ax.plot(history.history[f"val_{metric}"], label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(RESULTS / "training_curves.png", dpi=160)
    plt.close(fig)
    print("Training finished; saved best model to:", MODELS / "fruit_freshness_model.keras")
    print("Saved training plot:", RESULTS / "training_curves.png")


if __name__ == "__main__":
    main()
