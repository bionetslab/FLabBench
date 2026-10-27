#!/usr/bin/env python3
"""
Pooled detectability: one probe per attribute trained on the admissions of all cohorts, scored per cohort.

The encoders are shared across cohorts (same admission, same embedding), so one probe can learn from the whole
pool and small cohorts no longer get a weak probe. Patients are split into 5 folds (GroupKFold on subject_id),
every admission gets an out-of-fold prediction from a probe that never saw its patient, and per cohort
detectability is Spearman ρ (continuous) or 2·macro-AUC − 1 (categorical) between A and that prediction.

Output
------
{out_dir}/{model}_{attribute}.tsv  – one row per cohort: cohort, attribute, n, detectability
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.constants import PROJECT_ROOT
from analysis.cohort_characteristics import CohortCharacteristics
from analysis.bias_analysis import ATTRIBUTES, CATEGORICAL, load_attr_csv


def cohort_members(cohort, attributes, args):
    """Admissions of one cohort (those in the fold file) with subject_id and the built-in attributes."""
    cc = CohortCharacteristics(cohort, days_before_discharge=args.days, agg_int=args.agg_int, seed=args.seed, fold=args.fold,
                               first_adm_only=args.first_adm_only, split="all", max_obs=args.max_obs)
    ids = np.concatenate(list(cc.folds().values())).astype("int64")
    m = cc.load_cohort.drop_duplicates("hadm_id").set_index("hadm_id").loc[lambda d: d.index.isin(ids), ["subject_id"]]
    for a in attributes:
        m[a] = ATTRIBUTES[a](cc).astype(np.float32)
    return m.reset_index().assign(cohort=cohort)


def pool_strats_embs(emb_dir, cohorts, fold):
    """Per-cohort STraTS files, each admission kept once."""
    parts, seen = [], set()
    for c in cohorts:
        d = emb_dir / c / f"fold_{fold}"
        e = pd.concat([pd.read_parquet(d / f"strats_embeddings_{s}.parquet", columns=["hadm_id", "embedding"])
                       for s in ("train", "val", "test")]).drop_duplicates("hadm_id")
        e = e[~e.hadm_id.isin(seen)]
        seen.update(e.hadm_id)
        parts.append(e)
    return pd.concat(parts, ignore_index=True)


def pool_bert_embs(emb_dir, hadm_ids):
    """Pooled BERT files, read once for all admissions of the pool."""
    return pd.concat([pd.read_parquet(emb_dir / f"eic_bert_embeddings_{s}.parquet", columns=["hadm_id", "embedding"],
                                      filters=[("hadm_id", "in", hadm_ids)])
                      for s in ("train", "tuning", "test")], ignore_index=True).drop_duplicates("hadm_id")


def oof_predict(X, a, groups, categorical, seed, n_jobs):
    """Out-of-fold predictions, folds split by patient so a probe never sees the patient it predicts."""
    cv = GroupKFold(n_splits=5)
    if categorical:
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        return cross_val_predict(clf, X, a, groups=groups, cv=cv, method="predict_proba", n_jobs=n_jobs)
    reg = Ridge(alpha=1.0, solver="sag", random_state=seed, max_iter=1000)
    return cross_val_predict(reg, X, a, groups=groups, cv=cv, n_jobs=n_jobs)


def score_cohort(a, pred, categorical, classes, min_count=5):
    """Spearman ρ, or 2·macro-AUC − 1 over the categories with at least min_count admissions in this cohort."""
    if len(a) < 10:
        return np.nan
    if not categorical:
        return float(spearmanr(a, pred)[0])
    vals, counts = np.unique(a, return_counts=True)
    keep = vals[counts >= min_count]
    if keep.size < 2:
        return np.nan
    m = np.isin(a, keep)
    a, pred = a[m], pred[m]
    cols = np.searchsorted(classes, keep)
    if keep.size == 2:
        auc = roc_auc_score(a == keep[1], pred[:, cols[1]])
    else:
        auc = np.mean([roc_auc_score(a == k, pred[:, c]) for k, c in zip(keep, cols)])
    return float(2 * auc - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attributes", nargs="+", required=True, help="built-in attributes, or columns of --attr_csv")
    ap.add_argument("--attr_csv", default=None)
    ap.add_argument("--model", default="strats", choices=["strats", "bert"])
    ap.add_argument("--emb_dir", default="saved_data/embeddings/strats_pretrained")
    ap.add_argument("--cohort_list", default="saved_data/rep_selections/representative_cohorts_K1000_96e897.txt")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--agg_int", type=int, default=24)
    ap.add_argument("--max_obs", type=int, default=1000)
    ap.add_argument("--first_adm_only", action="store_true")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_pool_count", type=int, default=50, help="categories rarer than this in the pool are merged into one OTHER group")
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--out_dir", default="saved_data/bias_results/pooled_probe")
    args = ap.parse_args()

    cohorts = Path(PROJECT_ROOT, args.cohort_list).read_text().split()
    builtin = [] if args.attr_csv else args.attributes

    # ------------------ Pool: one row per cohort and admission ------------------------------
    members = pd.concat(Parallel(n_jobs=args.n_jobs)(delayed(cohort_members)(c, builtin, args) for c in cohorts), ignore_index=True)
    categorical = {a: a in CATEGORICAL for a in builtin}
    if args.attr_csv:
        for a in args.attributes:
            s, categorical[a] = load_attr_csv(Path(PROJECT_ROOT) / args.attr_csv, a)
            members[a] = members.hadm_id.map(s)

    # the probe sees each admission once, its attributes are the same in every cohort
    pool = members.drop_duplicates("hadm_id").set_index("hadm_id")

    emb_dir = Path(PROJECT_ROOT) / args.emb_dir
    emb = pool_strats_embs(emb_dir, cohorts, args.fold) if args.model == "strats" else pool_bert_embs(emb_dir, pool.index.tolist())
    pool = pool.join(emb.set_index("hadm_id"), how="inner")

    # ------------------ One probe per attribute, scored per cohort ------------------------------
    out_dir = Path(PROJECT_ROOT) / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for a in args.attributes:
        d = pool[pool[a].notna()]
        A = d[a].to_numpy(dtype=np.float32)
        classes = None
        if categorical[a]:
            vals, counts = np.unique(A, return_counts=True)
            A = np.where(np.isin(A, vals[counts < args.min_pool_count]), -1, A).astype(np.float32)
            classes = np.unique(A)
        X = np.stack(d["embedding"].to_numpy()).astype(np.float32)
        pred = oof_predict(X, A, d["subject_id"].to_numpy(), categorical[a], args.seed, args.n_jobs)
        pos = pd.Series(np.arange(len(d)), index=d.index)

        rows = []
        for c, m in members[members[a].notna() & members.hadm_id.isin(d.index)].groupby("cohort", sort=False):
            i = pos.loc[m.hadm_id].to_numpy()
            rows.append((c, a, len(i), score_cohort(A[i], pred[i], categorical[a], classes)))
        res = pd.DataFrame(rows, columns=["cohort", "attribute", "n", "detectability"])
        res.to_csv(out_dir / f"{args.model}_{a}.tsv", sep="\t", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
