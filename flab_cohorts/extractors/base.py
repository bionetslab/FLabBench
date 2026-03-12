from pathlib import Path

from flab_cohorts.utils.io import load_admissions, load_diagnoses, load_patients, set_all_paths
from flab_cohorts.config.constants import get_data_path

class BaseExtractor:
    def __init__(self, args):
        self.args = args
        self.data_path = Path(args.data_path) if getattr(args, "data_path", None) else get_data_path(getattr(args, "dataset", "MIMIC_IV"))
        self.paths = set_all_paths(args)
        self._load_mimic(self.data_path)

    def _load_mimic(self, path: Path):
        self.adms = load_admissions(path)
        self.diags = load_diagnoses(path)
        self.patients = load_patients(path)
        
        self.adms = self.adms.merge(self.patients, on="subject_id", how="left")
        

