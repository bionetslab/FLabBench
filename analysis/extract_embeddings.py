import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from config.constants import PROJECT_ROOT
from flab_training.envmanager import EnvManager
from flab_training.dataset import TimeSeriesDataset
from flab_training.factory import build_ts_model, build_batcher

# same arguments as main.py
parser = argparse.ArgumentParser()
parser.add_argument("--days_before_discharge", type=int, default=14)
parser.add_argument("--dataset", type=str, default="MIMIC_IV")
parser.add_argument("--cohort", type=str, required=True)
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--model_type", type=str, default="strats")
parser.add_argument("--grid", type=str, default="none")
parser.add_argument("--pretrain", action="store_true")
parser.add_argument("--feature-selection", type=lambda x: x.lower() == "true", default=True)
parser.add_argument("--feature-selection-method", type=str, default="mimic_top_100")
parser.add_argument("--load_ckpt_path", type=str, required=True)   # pretrained folder
parser.add_argument("--prefix", type=str, default="embeddings_pt")
parser.add_argument("--static_threshold", type=int, default=0)
parser.add_argument("--hid_dim_demo", type=int, default=64)
parser.add_argument("--agg_int", type=int, default=24)
parser.add_argument("--agg", type=str, default="mean")
parser.add_argument("--drop_minutes", action="store_true")
parser.add_argument("--freeze", action="store_true")
parser.add_argument("--config_path", default=None)
parser.add_argument("--split_seed", default=None)
parser.add_argument("--fast", action="store_true")
parser.add_argument("--search", type=str, default="grid")
parser.add_argument("--n_trials", type=int, default=20)
parser.add_argument("--impute", type=str, default="fill")
parser.add_argument("--variant", type=str, default="VMD")
parser.add_argument("--extractor", type=str, default="DTB")
parser.add_argument("--first_adm_only", action="store_true")
parser.add_argument("--oversampling", type=str, default=None)
parser.add_argument("--feature_combination_method", type=str, default="concatenate")
parser.add_argument("--out_dir", type=str, default="saved_data/embeddings/strats_pretrained")
args = parser.parse_args()

# finetune mode > uses pretrained variables and normalisation stats
envmg = EnvManager(args)
envmg.set_model_params(mode="default")
envmg.set_ids(mode="default")

dataset = TimeSeriesDataset(args)
ind_to_hadm = {i: h for h, i in dataset.ts_id_to_ind.items()}
model = build_ts_model(args)
batcher = build_batcher(args, dataset.preproc.input_dict)

# pretrained weights only, binary_head stays random and is not used
state = torch.load(args.pt_dict_path, map_location=args.device)
model.load_state_dict(state, strict=False)
model.eval()

# admission info
adm_info = (dataset.cohort[["subject_id", "hadm_id", "admittime", "dischtime"]]
            .drop_duplicates("hadm_id").set_index("hadm_id"))
adm_info["admittime"] = pd.to_datetime(adm_info["admittime"])
adm_info["dischtime"] = pd.to_datetime(adm_info["dischtime"])

out_dir = Path(PROJECT_ROOT) / args.out_dir / args.cohort / f"fold_{args.fold}"
out_dir.mkdir(parents=True, exist_ok=True)

for split in ("train", "val", "test"):
    ind = batcher.splits[split]
    if len(ind) == 0:
        continue
    embs = []
    for s in range(0, len(ind), args.eval_batch_size):
        batch = batcher.get_batch(ind[s:s + args.eval_batch_size])
        batch = {k: v.to(args.device) for k, v in batch.items()}
        with torch.no_grad():
            _, emb = model(**batch)
        embs.append(emb.cpu().numpy())
    embs = np.concatenate(embs)

    hadm_ids = np.array([ind_to_hadm[i] for i in ind])
    df = adm_info.loc[hadm_ids].reset_index()[["subject_id", "hadm_id", "admittime", "dischtime"]]
    df["split"] = split
    df["embedding"] = list(embs[:, :args.hid_dim])            # lab time series part
    df["embedding_demo"] = list(embs[:, args.hid_dim:])       # demographics part
    df["label"] = batcher.y[ind]

    path = out_dir / f"strats_embeddings_{split}.parquet"
    df.to_parquet(path, index=False)
    print(f"saved {path} ({len(df)} admissions)")
