from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DTB_DATA_PATH = PROJECT_ROOT / "data" / "DTB_TJ"

MIMIC_IV_PATH = os.environ.get("MIMIC_IV_PATH")
#export MIMIC_IV_PATH="/Users/zy51nise/Documents/BIONETs/FLabNet/Data/mimiciv/2.0/"
#export MIMIC_IV_PATH="/home/vault/b310dc/b310dc10/Data/mimiciv/2.0/"

RANDOM_SEED = 42
num_folds = 5
num_inner_folds = 3

DATASET_PATHS = {
    "MIMIC_IV": MIMIC_IV_PATH,
    "MIMIC_IV_HOSP": MIMIC_IV_PATH,
    "MIMIC_IV_ICU": MIMIC_IV_PATH
}


def get_data_path(dataset):
    path = DATASET_PATHS.get(dataset)
    if path is None:
        raise ValueError(
            f"No data path for dataset {dataset!r}. "
            "Set MIMIC_IV_PATH env var or pass --data-path."
        )
    return Path(path)
