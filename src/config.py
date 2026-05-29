from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


@dataclass(frozen=True)
class Paths:
    trump_posts: Path = DATA_DIR / "text" / "trump_posts_features_2017_2026.csv"
    market_dir: Path = DATA_DIR / "taiwan_market_data"
    features_dir: Path = OUTPUT_DIR / "features"
    datasets_dir: Path = OUTPUT_DIR / "datasets"
    models_dir: Path = OUTPUT_DIR / "models"
    predictions_dir: Path = OUTPUT_DIR / "predictions"
    reports_dir: Path = OUTPUT_DIR / "reports"


TARGETS = ("2330.TW", "2454.TW", "0050.TW")
DEFAULT_TARGET = "2330.TW"
RANDOM_SEED = 42

