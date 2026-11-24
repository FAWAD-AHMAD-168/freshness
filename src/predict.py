"""Predict freshness for one image passed on the command line."""
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from src.config import IMAGE_SIZE, MODELS


def predict_image(image_path, model=None):
    if model is None:
        model_path = MODELS / "fruit_freshness_model.keras"
        if not model_path.is_file():
            raise FileNotFoundError("No trained model yet. Run python -m src.train first.")
        model = tf.keras.models.load_model(model_path)
    with Image.open(image_path) as img:
        img = img.convert("RGB").resize(IMAGE_SIZE)
        array = np.asarray(img, dtype="float32")
    probability = float(model.predict(array[np.newaxis, ...], verbose=0)[0][0])
    prediction = "Rotten" if probability >= 0.5 else "Fresh"
    confidence = probability if prediction == "Rotten" else 1 - probability
    return prediction, confidence, probability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Image path, e.g. examples/my_apple.jpg")
    args = parser.parse_args()
    label, confidence, p = predict_image(args.image)
    print(f"Prediction: {label}")
    print(f"Model score for {label}: {confidence:.1%}")
    print(f"Rotten probability estimate: {p:.1%}")
    print("Note: this is a model score, not a guarantee of food safety.")


if __name__ == "__main__":
    main()
