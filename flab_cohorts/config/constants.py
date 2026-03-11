from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DTB_DATA_PATH = PROJECT_ROOT / "data" / "DTB" / "DTB_TJ"

_DEFAULT_MIMIC_IV_PATH = "/Users/zy51nise/Documents/BIONETs/FLabNet/Data/mimiciv/2.0/"
MIMIC_IV_PATH = os.environ.get("MIMIC_IV_PATH", _DEFAULT_MIMIC_IV_PATH)


RANDOM_SEED = 42

DATASET_PATHS = {
    "MIMIC_IV": MIMIC_IV_PATH,
    "MIMIC_IV_HOSP": MIMIC_IV_PATH,
    "MIMIC_IV_ICU": MIMIC_IV_PATH
}


def get_data_path(dataset):
    return Path(DATASET_PATHS.get(dataset, MIMIC_IV_PATH))