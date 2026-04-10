from pathlib import Path
import os
from config.constants import PROJECT_ROOT


def get_output_dir(args):
    output_dir = Path(PROJECT_ROOT) / "saved_data" / "results" / args.cohort / "time_series" / args.train_mode / args.model_type / args.prefix / f"fold_{args.fold}" / f"grid_{args.grid}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def set_all_paths(args, out=True):
    dataset = args.dataset
    cohort = getattr(args, "cohort", "")
    extractor = getattr(args, "extractor", "")

    saved_data_path = Path(PROJECT_ROOT) / "saved_data"

    path_dict = {
        "saved_data_path":    saved_data_path,
        "cohort_path":        saved_data_path / "cohorts" / extractor,
        "features_path":      saved_data_path / "features",
        "folds_path":         saved_data_path / "folds" / cohort,
        "top_features_path":  saved_data_path / "top_features",
    }

    if out:
        path_dict["output_path"] = get_output_dir(args)

    for p in path_dict.values():
        Path(p).mkdir(parents=True, exist_ok=True)

    return path_dict
