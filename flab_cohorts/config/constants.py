from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DTB_DATA_PATH = PROJECT_ROOT / "data" / "DTB" / "DTB_TJ"

_DEFAULT_MIMIC_IV_PATH = "/Users/zy51nise/Documents/BIONETs/FLabNet/Data/mimiciv/2.0/"
MIMIC_IV_PATH = os.environ.get("MIMIC_IV_PATH", _DEFAULT_MIMIC_IV_PATH)


RANDOM_SEED = 42