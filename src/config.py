from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42
CLASS_NAMES = ["fresh", "rotten"]  # alphabetical: fresh=0, rotten=1
