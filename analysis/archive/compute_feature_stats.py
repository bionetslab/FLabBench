import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from scipy.stats import entropy

BASE = "/home/atuin/b310dc/b310dc10/FLabBench-pipeline/saved_data"
OUT  = f"{BASE}/features/feature_stats.csv"

edges = pd.read_csv(f"{BASE}/cohorts/DTB/selected_edges_DTB_all.csv")
saved = edges[(edges["n_pos"] > 10) & (edges["n_neg"] > 50)].copy()
saved["cohort_name"] = "cohort_" + saved["D1"] + "-" + saved["D2"].astype(str)
print(f"Cohorts to process: {len(saved)}")

cohorts = saved["cohort_name"]
results = pd.DataFrame({"cohort_name": cohorts, "AUC-mean": 0.8})

with open("/home/atuin/b310dc/b310dc10/FLabBench-pipeline/data/top_features/mimic_top100_features.pkl", "rb") as f:
    top_features = set(int(x) for x in pickle.load(f))


def feature_entropy(series):
    series = series.dropna()
    if series.std() > 0:
        series = (series - series.mean()) / series.std()
    counts, _ = np.histogram(series, bins=10)
    return entropy(counts + 1)


for cohort_name in tqdm(cohorts):
    feature_cohort_name = cohort_name.replace("cohort_", "").replace(".csv.gz", "")
    try:
        cohort = pd.read_csv(f"{BASE}/cohorts/DTB/{cohort_name}.csv.gz")
        features = pd.read_csv(f"{BASE}/features/{feature_cohort_name}/features.csv.gz")
    except Exception as e:
        print(f"Skipping {cohort_name}: {e}")
        continue

    features = features[features["itemid"].isin(list(top_features)[:100])]
    features = features.merge(cohort[["hadm_id", "label"]], on="hadm_id")
    patient_feats = features.groupby(["hadm_id", "itemid"])["value"].mean().unstack()
    labels = cohort[["hadm_id", "label"]].drop_duplicates("hadm_id")
    patient_feats = patient_feats.merge(labels, on="hadm_id")

    feat_cols = patient_feats.drop(columns="label").columns

    missingness_var = patient_feats[feat_cols].isna().mean().var()
    missingness_mean = patient_feats[feat_cols].isna().mean().mean()

    pos = patient_feats[patient_feats["label"] == 1][feat_cols].mean()
    neg = patient_feats[patient_feats["label"] == 0][feat_cols].mean()

    h_pos = patient_feats[patient_feats["label"] == 1][feat_cols].apply(feature_entropy).mean()
    h_neg = patient_feats[patient_feats["label"] == 0][feat_cols].apply(feature_entropy).mean()

    pooled_std = patient_feats[feat_cols].std()
    cohens_d = ((pos - neg).abs() / (pooled_std + 1e-8)).mean()

    idx = results.index[results["cohort_name"] == cohort_name]
    results.loc[idx, "missingness_v"] = missingness_var
    results.loc[idx, "missingness_m"] = missingness_mean
    results.loc[idx, "h_pos"] = h_pos
    results.loc[idx, "h_neg"] = h_neg
    results.loc[idx, "cohens_d"] = cohens_d

results.to_csv(OUT, index=False)
print(f"Saved to {OUT}")
