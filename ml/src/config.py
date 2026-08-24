from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    ROOT: Path = Path(__file__).resolve().parents[1]
    DATA: Path = ROOT / "data"
    MODELS: Path = ROOT / "models"
    OUTPUTS: Path = ROOT / "outputs"
    DB: Path = ROOT / "dialog.sqlite"
    PROCESSED: Path = DATA / "processed_dataset.csv"

PATHS = Paths()
