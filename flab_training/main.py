import argparse
import numpy as np
import pandas as pd

#import ts_model_training.dataset as dataset
from flab_training.envmanager import EnvManager

# ----------------- Parse command-line arguments -----------------
parser = argparse.ArgumentParser()

parser.add_argument("--days_before_discharge", type=int, default=14)
parser.add_argument("--dataset", type=str, default="MIMIC_IV")
parser.add_argument("--cohort", type=str, default="mimic_cohort_NF_30_days")  # or "mimic_cohort_aplasia_45_days"
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--model_type", type=str, default="strats")
parser.add_argument("--grid", type=str, default="none")
parser.add_argument("--pretrain", action="store_true", help="Enable pretraining mode")
parser.add_argument("--feature-selection", type=lambda x: x.lower() == "true", default=True, help="Select top 100 features in MIMIC")
parser.add_argument("--feature-selection-method", type=str, default="mimic_top_100",
                     choices=["mimic_top_100", "correlation_top_100", "correlation_fdr", "MRMR_top_100"],
                     help="Which precomputed feature list to use when --feature-selection is true")
parser.add_argument("--load_ckpt_path", type=str, default=None)
parser.add_argument("--prefix", type=str, default=None)
parser.add_argument("--static_threshold", type=int, default=0)
parser.add_argument("--hid_dim_demo", type=int, default=64)
parser.add_argument("--agg_int", type=int, default=24) # with this parameter it is possible to study the effect of discretisation window on performance.
parser.add_argument("--drop_minutes", action="store_true", help="Do not consider timestamps but only days") # with this flag it is possible to study the effect of removing exact times on performance.
parser.add_argument("--freeze", action="store_true", help="Freeze all except for last layers in finetuning")
parser.add_argument("--config_path", default=None)
parser.add_argument("--split_seed", default=None)
parser.add_argument("--fast", action="store_true", help="Disable determinism for faster computation")
parser.add_argument("--search", type=str, default="grid", help="Hyperparameter search strategy: 'grid', 'random', or 'optuna'")
parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials (used when --search optuna)")
parser.add_argument("--impute", type=str, default="fill")
parser.add_argument("--variant", type=str, default="VMD")
parser.add_argument("--extractor", type=str, default="DTB")
parser.add_argument("--first_adm_only", action="store_true", help="Keep only the first admission per patient")
args = parser.parse_args()

# GRID SEARCH
envmg = EnvManager(args)
envmg.train_full()
    
    
#python -m ts_model_training.main.py --dataset MIMIC_IV --cohort mimic_cohort_NF_30_days --fold 0 --model_type gru --feature-selection true --static_threshold 0 --hid_dim_demo 64 --prefix test

#python -m ts_model_training.main.py --dataset MIMIC_IV --cohort mimic_all --fold 0 --model_type strats --static_threshold 0 --hid_dim_demo 64  --feature-selection true --pretrain