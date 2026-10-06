#!/usr/bin/env python3
"""
Bias analysis: utility and detectability for one admission attribute of one cohort.

Run once per cohort and attribute (--cohort, --attribute); parallel runs append to the same tsv.

Attributes (from CohortCharacteristics, or any column of --attr_csv)
----------
n_events             – number of lab measurements per admission
n_itemids_measured   – number of distinct itemids measured per admission
adm_itemid_missing   – share of the top-100 labs never measured per admission
itembin_missing      – share of (lab, time bin) pairs not measured per admission
truncated_at_max_obs – 1 if the admission has more than max_obs events, 0 otherwise defaul 1000 for STraTs
age, gender, mortality

Utility       = NMI(A; Y) = MI / sqrt(H(A)·H(Y)), MI estimated by sklearn kNN, H(A) by histogram,
                with n_perm > 0 also a permutation p-value and z-score
Detectability = Spearman ρ of a Ridge regressor predicting A from the admission embedding X,
                2·macro-AUC − 1 of a logistic regression for categorical A

Output
------
--out_file  – single tsv, one row per cohort/attribute/split
"""

import argparse
import csv
import fcntl
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.constants import PROJECT_ROOT
from analysis.cohort_characteristics import CohortCharacteristics




def build_X(emb_df, attrs):
    """Return (X, df): one embedding per admission, joined with its attributes on hadm_id
    (admissions missing an embedding or attributes are dropped)."""
    df = emb_df.set_index("hadm_id").join(attrs, how="inner")
    X = np.stack(df["embedding"].to_numpy()).astype(np.float32) if len(df) else np.empty((0, 0), dtype=np.float32)
    return X, df


def load_strats_embs(emb_dir, cc):
    """Per-cohort STraTS files, label already inside."""
    d = emb_dir / cc.cohort_name / f"fold_{cc.fold}"
    return [(s, pd.read_parquet(d / f"strats_embeddings_{s}.parquet")) for s in ("train", "val", "test")]


def load_bert_embs(emb_dir, cc):
    """Pooled BERT files, split and label taken from CohortCharacteristics."""
    split = pd.concat([pd.DataFrame({"hadm_id": ids.astype("int64"), "split": s}) for s, ids in cc.folds().items()])
    ft = split.merge(cc.labels.rename("label"), left_on="hadm_id", right_index=True)
    # read only this cohort's admissions, the pooled files are large
    emb = pd.concat([pd.read_parquet(emb_dir / f"eic_bert_embeddings_{s}.parquet", columns=["hadm_id", "embedding"],
                                     filters=[("hadm_id", "in", ft.hadm_id.tolist())])
                     for s in ("train", "tuning", "test")], ignore_index=True).drop_duplicates("hadm_id")
    emb = emb.merge(ft, on="hadm_id")
    return [(s, emb[emb.split == s].reset_index(drop=True)) for s in ("train", "val", "test")]


EMB_LOADERS = {"strats": load_strats_embs, "bert": load_bert_embs}



CATEGORICAL = ["gender", "insurance_group", "race_group", "language_group", "marital_status_group",
               "admission_type", "admission_type_raw", "admission_location", "discharge_location",
               "anchor_year_group", "era",
               "has_medication", "death_in_window", "icu_any", "truncated_at_context", "n_death_events"]


def load_attr_csv(path, attribute):
    """One attribute per hadm_id from a precomputed table, plus whether it is categorical.
    CATEGORICAL columns become integer codes per category, all others are continuous."""
    s = pd.read_csv(path, usecols=["hadm_id", attribute]).drop_duplicates("hadm_id").set_index("hadm_id")[attribute]
    if attribute not in CATEGORICAL:
        try:
            return s.astype(np.float32), False
        except ValueError:
            print(f"  WARNING: '{attribute}' is not numeric and not in CATEGORICAL, treating it as categorical", file=sys.stderr)
    codes = pd.Series(pd.factorize(s)[0], index=s.index).astype(np.float32)
    return codes.where(codes >= 0), True  # missing category -> NaN, dropped later


def _entropy(counts):
    """Shannon entropy in nats; ignores zero-count bins."""
    c = counts[counts > 0].astype(float)
    p = c / c.sum()
    return float(-np.sum(p * np.log(p)))


def compute_nmi(A, y, discrete_features, seed=42, n_bins=10):
    """NMI = MI(A; Y) / sqrt(H(A) · H(Y)) — symmetric, bounded in [0, 1].
    H(A) is estimated by binning A into n_bins equal-width histogram bins
    (or using unique values directly when A is already discrete).
    """
    mi = float(mutual_info_classif(A.reshape(-1, 1), y, discrete_features=discrete_features, random_state=seed)[0])

    _, y_counts = np.unique(y, return_counts=True) # admissions per label
    h_y = _entropy(y_counts)

    if discrete_features:
        _, a_counts = np.unique(A, return_counts=True) # admissions per category
    else:
        a_counts, _ = np.histogram(A, bins=n_bins) # admissions per value range 
        
        
    h_a = _entropy(a_counts) # entropy of attribute values (how spread they are)

    denom = np.sqrt(h_a * h_y)
    return float(np.clip(mi / denom, 0.0, 1.0)) if denom > 0 else 0.0


def detect_categorical(A, X, seed, cv):
    """2·macro-AUC − 1 of a logistic regression predicting categorical A from X."""
    if np.unique(A).size < 2:
        return 0.0
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    a, classes = A.astype(int), np.unique(A).astype(int)
    proba = cross_val_predict(clf, X, a, cv=5, method="predict_proba") if cv else clf.fit(X, a).predict_proba(X)
    auc = roc_auc_score(a, proba[:, 1]) if classes.size == 2 else \
        roc_auc_score(a, proba, multi_class="ovr", average="macro", labels=classes)
    return float(2 * auc - 1)


def detect_continuous(A, X, seed, cv):
    """Spearman ρ of a Ridge regressor predicting continuous A from X."""
    reg = Ridge(alpha=1.0, solver="sag", random_state=seed, max_iter=1000)
    # out-of-sample predictions, in-sample fit inflates ρ when patients are few relative to the embedding dims
    pred = cross_val_predict(reg, X, A, cv=5) if cv else reg.fit(X, A).predict(X)
    return float(spearmanr(A, pred)[0])


def analyse(A, X, y, name, seed=42, n_bins=10, cv=True, n_perm=0, categorical=False, min_count=20):
    """Return dict with nmi, related, detectability (Spearman ρ, or 2·macro-AUC − 1 for categorical),
    and with n_perm > 0 the permutation p-value and z-score of the NMI.
    """
    if categorical:
        # categories with fewer than min_count admissions in this cohort are merged into one OTHER group(min count 20)
        vals, counts = np.unique(A, return_counts=True)
        A = np.where(np.isin(A, vals[counts < min_count]), -1, A).astype(np.float32) # attributes value (-1 for rare groups)
    # discrete_features = name == "self_interactions"
    
    #---- Utility -----
    discrete_features = categorical
    nmi = compute_nmi(A, y, discrete_features, seed, n_bins)
    print(f" NMI(A;Y) = {nmi:.4f}  (related? {'Yes' if nmi > 0 else 'No'})", file=sys.stderr)

    # shuffled labels break the A-Y link, so their NMI is the chance level
    nmi_p, nmi_z = np.nan, np.nan
    if n_perm > 0:
        rng = np.random.default_rng(seed)
        null = np.array([compute_nmi(A, rng.permutation(y), discrete_features, seed, n_bins) for _ in range(n_perm)])
        nmi_p = float((1 + np.sum(null >= nmi)) / (1 + n_perm))
        nmi_z = float((nmi - null.mean()) / null.std()) if null.std() > 0 else np.nan

    #---- Detectability -----
    
    if X.shape[0] < 10: # number of admissions or rows of embedding matrix
        detectability = 0.0
    elif categorical:
        detectability = detect_categorical(A, X, seed, cv)
    else:
        detectability = detect_continuous(A, X, seed, cv)

    return {"nmi": nmi, "related": nmi > 0, "detectability": detectability, "nmi_p": nmi_p, "nmi_z": nmi_z}



ATTRIBUTES = {
    "n_events":             lambda cc: cc.n_events(),
    "n_itemids_measured":   lambda cc: cc.n_itemids_measured(),
    "adm_itemid_missing":   lambda cc: cc.adm_itemid_missing(),
    "itembin_missing":      lambda cc: cc.itembin_missing(),
    "truncated_at_max_obs": lambda cc: cc.truncated_at_max_obs(),
    "age":                  lambda cc: cc.age(),
    "gender":               lambda cc: cc.gender(),
    "mortality":            lambda cc: cc.mortality(),
}


def write_mqc(attribute, results, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{attribute}_bias_mqc.tsv"
    with open(fname, "w") as fh:
        fh.write(
            f"# id: 'bias_{attribute}'\n"
            f"# section_name: 'Bias: {attribute}'\n"
            f"# description: 'Utility NMI(A;Y) = MI/sqrt(H(A)·H(Y)) and detectability (Ridge Spearman ρ) for {attribute}.'\n"
            "# plot_type: 'table'\n"
            "# pconfig:\n"
            f"#     id: 'bias_{attribute}_table'\n"
            f"#     title: 'Bias: {attribute}'\n"
            "# headers:\n"
            "#     'NMI(A;Y)':                    {format: '{:.4f}'}\n"
            "#     'Detectability (Spearman ρ)':   {format: '{:.4f}'}\n"
            "Split\tNMI(A;Y)\tRelated?\tDetectability (Spearman ρ)\tNMI p-value\tNMI z-score\n"
        )
        for split, r in results:
            fh.write(f"{split}\t{r['nmi']:.4f}\t{'Yes' if r['related'] else 'No'}\t{r['detectability']:.4f}"
                     f"\t{r['nmi_p']:.4f}\t{r['nmi_z']:.4f}\n")


def append_rows(cohort, attribute, results, out_file):
    """Append one row per split to a single tsv shared by all runs, locked so parallel runs don't interleave."""
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    lines = "".join(f"{cohort}\t{attribute}\t{split}\t{r['nmi']:.4f}\t{'Yes' if r['related'] else 'No'}\t{r['detectability']:.4f}"
                    f"\t{r['nmi_p']:.4f}\t{r['nmi_z']:.4f}\n" for split, r in results)
    with open(out_file, "a") as fh:
        fcntl.lockf(fh, fcntl.LOCK_EX)
        if fh.tell() == 0:
            fh.write("cohort\tattribute\tSplit\tNMI(A;Y)\tRelated?\tDetectability (Spearman ρ)\tNMI p-value\tNMI z-score\n")
        fh.write(lines)
        fh.flush()
        fcntl.lockf(fh, fcntl.LOCK_UN)




#-------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attribute", required=True, help="built-in attribute, or a column of --attr_csv")
    ap.add_argument("--attr_csv", default=None, help="table with hadm_id and attribute columns, e.g. shortcut_candidates_K1000.csv")
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--model", default="strats", choices=list(EMB_LOADERS))
    ap.add_argument("--emb_dir", default="saved_data/embeddings/strats_pretrained")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--first_adm_only", action="store_true")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--agg_int", type=int, default=24, help="time bin in hours, for itembin_missing")
    ap.add_argument("--max_obs", type=int, default=1000, help="events kept by the transformers, for truncated_at_max_obs")
    ap.add_argument("--out_file", default="saved_data/bias_results/bias.tsv", help="single tsv, one row per cohort/attribute/split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pool_splits", action="store_true", help="analyse train+val+test together as one split 'all'")
    ap.add_argument("--no_cv", action="store_true", help="in-sample detectability (fit and predict on the same admissions)")
    ap.add_argument("--n_perm", type=int, default=0, help="label permutations for the NMI p-value and z-score, 0 = skip")
    args = ap.parse_args()

    print(f"=== {args.attribute} ===", file=sys.stderr)

    # ------------------ Load attributes ------------------------------
    # from csv for bert from cohort characterstics for DTB
    cc = CohortCharacteristics(args.cohort, days_before_discharge=args.days, agg_int=args.agg_int, seed=args.seed, fold=args.fold, first_adm_only=args.first_adm_only, split="all", max_obs=args.max_obs)
    if args.attr_csv:
        A_all, categorical = load_attr_csv(Path(PROJECT_ROOT) / args.attr_csv, args.attribute)
        A_all = A_all.dropna().rename("A").to_frame()
    else:
        if args.attribute not in ATTRIBUTES:
            sys.exit(f"unknown attribute {args.attribute}, choose from {list(ATTRIBUTES)} or pass --attr_csv")
        categorical = args.attribute in CATEGORICAL
        A_all = ATTRIBUTES[args.attribute](cc).astype(np.float32).dropna().rename("A").to_frame()  # one A per hadm_id

    # ------------------ Read Embeddings ------------------------------
    split_embs = EMB_LOADERS[args.model](Path(PROJECT_ROOT) / args.emb_dir, cc)
    if args.pool_splits:
        # pool train/val/test we don't need individual analysis the splits are random with the same patient distribution 
        split_embs = [("all", pd.concat([emb for _, emb in split_embs], ignore_index=True))]

    # ------------------ Results ------------------------------ 
    results = []
    for split, emb in split_embs:
        X, df = build_X(emb, A_all) # X embedding matrix one row per adm -df joint table with embeddings label and A 
        A, y = df["A"].to_numpy(dtype=np.float32), df["label"].to_numpy() # A attribute value per adm, y outcome label per adm

        print(f"  {X.shape[0]} {split} admissions retained", file=sys.stderr)
        r = analyse(A, X, y, args.attribute, seed=args.seed, cv=not args.no_cv, n_perm=args.n_perm,categorical=categorical)
        results.append((split, r))
        print(f"  [{split}] NMI={r['nmi']:.4f}  ρ={r['detectability']:.4f}  p={r['nmi_p']:.4f}  z={r['nmi_z']:.2f}", file=sys.stderr)

    if results:
        append_rows(args.cohort, args.attribute, results, Path(PROJECT_ROOT) / args.out_file)


if __name__ == "__main__":
    main()
