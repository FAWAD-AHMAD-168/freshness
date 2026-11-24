"""Evaluate saved model only on the untouched Kaggle test set."""
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from src.config import (PROCESSED, MODELS, RESULTS, IMAGE_SIZE,
                        BATCH_SIZE, CLASS_NAMES)


def main():
    model_path = MODELS / "fruit_freshness_model.keras"
    if not model_path.is_file():
        raise FileNotFoundError(f"Train the model first. Missing: {model_path}")
    test_root = PROCESSED / "test"
    if not test_root.is_dir():
        raise FileNotFoundError("Prepare the dataset before evaluation.")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_root, labels="inferred", class_names=CLASS_NAMES,
        label_mode="binary", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        shuffle=False,
    )
    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(test_ds, verbose=1).reshape(-1)
    y_true = np.concatenate([y.numpy().reshape(-1) for _, y in test_ds]).astype(int)
    y_pred = (probabilities >= 0.5).astype(int)
    if len(y_true) != len(y_pred):
        raise RuntimeError("Test labels and predictions do not align")
    acc = float(accuracy_score(y_true, y_pred))
    report = classification_report(y_true, y_pred, labels=[0, 1],
                                   target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    RESULTS.mkdir(parents=True, exist_ok=True)
    metrics = {"test_accuracy": acc, "class_report": report,
               "confusion_matrix": cm.tolist(), "labels": CLASS_NAMES}
    (RESULTS / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    with (RESULTS / "classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(classification_report(y_true, y_pred, labels=[0, 1],
                                      target_names=CLASS_NAMES, zero_division=0))
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Fresh", "Rotten"]).plot(
        ax=ax, values_format="d", colorbar=False)
    ax.set_title("Confusion Matrix — Test Set")
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    # Useful for inspection of individual test predictions.
    test_paths = sorted([p for label in CLASS_NAMES for p in (test_root / label).iterdir()
                         if p.is_file()])
    with (RESULTS / "test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "actual", "predicted", "rotten_probability"])
        for p, actual, predicted, prob in zip(test_paths, y_true, y_pred, probabilities):
            writer.writerow([p.name, CLASS_NAMES[actual], CLASS_NAMES[predicted],
                             f"{prob:.6f}"])
    print(f"\nTEST ACCURACY: {acc:.2%}")
    print(classification_report(y_true, y_pred, labels=[0, 1],
                                target_names=CLASS_NAMES, zero_division=0))
    print("Saved evaluation files in:", RESULTS)


if __name__ == "__main__":
    main()
