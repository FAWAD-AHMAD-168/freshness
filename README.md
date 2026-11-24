# Fruit Freshness Classification (Beginner CNN) — Phase 3

**Goal:** Classify apples, bananas, and oranges into only **Fresh** or **Rotten**.
The model does **not** predict fruit names.

## Dataset

Download **Fruits Fresh and Rotten for Classification** from:
https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification

Unzip it into `data/raw/`. The preparation script searches beneath that folder for
`train/` and `test/` directories with the six category directories:
`freshapples`, `freshbanana`, `freshoranges`, `rottenapples`, `rottenbanana`, `rottenoranges`.

**The dataset is not distributed in this ZIP.** The script keeps Kaggle's test images
as the test set and divides Kaggle's original training portion into train and validation.

## Setup (run from the project folder)

Use Python 3.11 or 3.12 (recommended), on a machine able to install TensorFlow.

```bash
python -m venv .venv
```
Activate the environment:
- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS / Linux: `source .venv/bin/activate`

```bash
python -m pip install -r requirements.txt
```

## Steps

**1. Prepare dataset (six original categories -> two labels)**

```bash
python -m src.prepare_data
```

This generates `data/processed/{train,val,test}/{fresh,rotten}` and
`results/dataset_counts.csv`.

**2. Train CNN (10 epochs maximum by default; may stop early)**

```bash
python -m src.train --epochs 10
```

This saves `models/fruit_freshness_model.keras`, `results/training_history.csv`, and
`results/training_curves.png`. Training augmentation happens **only** in training mode.

**3. Evaluate only on the held-out test images**

```bash
python -m src.evaluate
```

This generates `results/evaluation_metrics.json`, `results/classification_report.txt`,
`results/confusion_matrix.png`, and `results/test_predictions.csv`.

**4. Predict a single image**

```bash
python -m src.predict examples/my_fruit.jpg
```

Place your own image at that location first.

**5. Launch a basic webpage demo (optional)**

```bash
streamlit run app.py
```

## Explain to sir

1. The Kaggle dataset originally has six folders (three fruits times two conditions).
2. We merge folders into two labels: `fresh = 0` and `rotten = 1`.
3. Images are resized to 128x128, and input pixels are normalized in the model.
4. A simple custom CNN learns patterns via convolutions, pooling, and a dense layer.
5. The output is one sigmoid number interpreted as probability of rotten.
6. Training uses train images; validation guides checkpoint selection; test is reserved
   for the final unbiased evaluation.
7. Evaluate accuracy, precision, recall, F1-score, and the confusion matrix.

## Important limitations

- No reported training accuracy exists until you **actually run** the code with the dataset.
- Photos of apples, bananas and oranges only; unsupported fruit types may be misclassified.
- This predicts visual appearance, not internal quality or food safety.
- Check for near-duplicate photos across dataset splits if possible; similar photos
  may inflate measured accuracy.
- Avoid interpreting model probability as a calibrated confidence guarantee.

## Report structure

- Dataset overview and split counts: `results/dataset_counts.csv`
- Model architecture: printed by `python -m src.train`
- Train vs validation curves: `results/training_curves.png`
- Test results: `results/classification_report.txt`
- Confusion matrix: `results/confusion_matrix.png`
- Demo: `streamlit run app.py`
