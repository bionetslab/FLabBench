from pathlib import Path
import pandas as pd
import os
from config.constants import PROJECT_ROOT
from tqdm import tqdm


def load_dtb_edges(edges_path: Path) -> pd.DataFrame:
    """Load DTB edges TSV (e.g. trajectory browser edge list)."""
    if not edges_path.exists():
        raise FileNotFoundError(f"DTB edges file not found at {edges_path}")
    return pd.read_csv(edges_path, sep="\t")



def get_output_dir(args):
    output_dir = Path(PROJECT_ROOT) / "data" / args.dataset / "results" / args.cohort / "time_series" / args.train_mode / args.model_type / args.prefix / f"fold_{args.fold}" / f"grid_{args.grid}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def set_all_paths(args, out=True):
    dataset = args.dataset
    cohort = getattr(args, "cohort", "")
    extractor = getattr(args, "extractor", "")

    saved_data_path = Path(PROJECT_ROOT) / "data" / dataset

    path_dict = {
        "saved_data_path": saved_data_path,
        "cohort_path":     saved_data_path / "cohorts" / extractor,
        "features_path":   saved_data_path / "features",
        "folds_path":      saved_data_path / "folds"    / cohort,
    }

    if out:
        path_dict["output_path"] = get_output_dir(args)

    for p in path_dict.values():
        Path(p).mkdir(parents=True, exist_ok=True)

    return path_dict