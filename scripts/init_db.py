from src.db import init_db
from src.config import PATHS

if __name__ == "__main__":
    PATHS.MODELS.mkdir(parents=True, exist_ok=True)
    PATHS.OUTPUTS.mkdir(parents=True, exist_ok=True)
    init_db()
    print(f"Initialized DB at: {PATHS.DB}")
