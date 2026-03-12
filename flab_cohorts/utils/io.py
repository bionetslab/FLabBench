from pathlib import Path
import pandas as pd
import os
from flab_cohorts.config.constants import PROJECT_ROOT
from tqdm import tqdm


def load_dtb_edges(edges_path: Path) -> pd.DataFrame:
    """Load DTB edges TSV (e.g. trajectory browser edge list)."""
    if not edges_path.exists():
        raise FileNotFoundError(f"DTB edges file not found at {edges_path}")
    return pd.read_csv(edges_path, sep="\t")



def set_all_paths(args, out=True):
    dataset = args.dataset
    cohort = getattr(args, "cohort", "")
    extractor = args.extractor


    saved_data_path = Path(PROJECT_ROOT) / "data" / dataset

    feature_path = saved_data_path / "top_features" # / cohort

    #CV_folds_path = saved_data_path / extractor / "folds" / cohort
    
    cohort_path = saved_data_path / "cohorts"/ extractor
    

    # save this as dict
    path_dict = {
        "saved_data_path": saved_data_path,
        #"CV_folds_path": CV_folds_path,
        "feature_path": feature_path,
        "cohort_path": cohort_path,
    }
    
    for p in path_dict.values():
        Path(p).mkdir(parents=True, exist_ok=True)
        

    return path_dict