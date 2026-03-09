from pathlib import Path

from flab_cohorts.utils.io import load_admissions, load_diagnoses, load_patients, set_all_paths
from flab_cohorts.config.constants import MIMIC_IV_PATH


class BaseExtractor:
    def __init__(self, args):
        self.args = args
        self.paths = set_all_paths(args)
        self._load_mimic(MIMIC_IV_PATH)

    def _load_mimic(self, path: Path):
        self.adms = load_admissions(path)
        self.diags = load_diagnoses(path)
        self.patients = load_patients(path)
