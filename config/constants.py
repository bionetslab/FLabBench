from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DTB_DATA_PATH = PROJECT_ROOT / "data" / "DTB_TJ"

#MIMIC_IV_PATH = os.environ.get("MIMIC_IV_PATH")
#export MIMIC_IV_PATH="/Users/zy51nise/Documents/BIONETs/FLabNet/Data/mimiciv/2.0/"
#export MIMIC_IV_PATH="/home/vault/b310dc/b310dc10/Data/mimiciv/2.0/"
MIMIC_IV_PATH="/Users/zy51nise/Documents/BIONETs/FLabNet/Data/mimiciv/2.0/"
ICD_CHAPTERS = {
    "A": "Infectious", "B": "Infectious",
    "C": "Malignant Neoplasms", 
    "D": "Blood/Other Neoplasms",
    "E": "Endocrine & Metabolic", 
    "F": "Mental",
    "G": "Nervous", 
    "H": "Eye/Ear",
    "I": "Circulatory", 
    "J": "Respiratory",
    "K": "Digestive", 
    "L": "Skin",
    "M": "Musculoskeletal", 
    "N": "Genitourinary",
    "O": "Pregnancy", "P": "perinatal", "Q": "Congenital",
    "R": "Symptoms and abnormal findings", "S": "Injury Site", "T": "Injury Type",
    "V": "external death causes(Transport accidents)",
    "W": "external death causes(Accidental injuries)",
    "X": "external death causes(Accidents + intentional harm)",
    "Y": "external death causes(Legal, medical, unknown intent + sequelae)",
    "Z": "Non-disease factors",
}

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
