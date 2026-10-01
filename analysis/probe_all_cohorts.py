import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

from config.constants import PROJECT_ROOT

EMB_DIR    = Path(PROJECT_ROOT) / "saved_data" / "embeddings"
STRATS_DIR = EMB_DIR / "strats_pretrained"
EIC_DIR    = EMB_DIR / "eic_bert_embeddings_2085cohorts"


def fold_table(cohort, fold=0, seed=42):
    # hadm_id, our split and label for one cohort
    f = Path(PROJECT_ROOT) / "saved_data" / "folds" / cohort / f"seed_{seed}_firstadm" / f"fold_{fold}.pkl"
    train, val, test = pickle.load(open(f, "rb"))
    split = pd.concat([pd.DataFrame({"hadm_id": a[:, 1], "split": s})
                       for a, s in [(train, "train"), (val, "val"), (test, "test")]])
    labels = pd.read_csv(Path(PROJECT_ROOT) / "saved_data" / "cohorts" / "DTB" / "new" / f"{cohort}.csv.gz",
                         usecols=["hadm_id", "label"])
    return split.merge(labels, on="hadm_id")


def load_strats(cohort, fold=0):
    d = STRATS_DIR / cohort / f"fold_{fold}"
    return pd.concat([pd.read_parquet(d / f"strats_embeddings_{s}.parquet")
                      for s in ("train", "val", "test")], ignore_index=True)


def to_xy(df, split, col="embedding"):
    d = df[df.split == split]
    return np.stack(d[col].values), d.label.to_numpy(), d.hadm_id.to_numpy()


def probe(df, Cs=(1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100)):
    X_tr, y_tr, _ = to_xy(df, "train")
    X_va, y_va, _ = to_xy(df, "val")
    X_te, y_te, _ = to_xy(df, "test")

    # pick C on val
    best_C, best_auc = None, -1
    for C in Cs:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=C, class_weight="balanced", max_iter=2000))
        clf.fit(X_tr, y_tr)
        auc = roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1])
        if auc > best_auc:
            best_C, best_auc = C, auc

    # refit on train+val, evaluate on test
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=best_C, class_weight="balanced", max_iter=2000))
    clf.fit(np.vstack([X_tr, X_va]), np.concatenate([y_tr, y_va]))
    p = clf.predict_proba(X_te)[:, 1]
    return {"C": best_C, "auroc": roc_auc_score(y_te, p), "auprc": average_precision_score(y_te, p),
            "target_rate": y_te.mean()}


COHORTS = sorted(p.name for p in STRATS_DIR.iterdir() if p.is_dir())

# load all BERT embeddings once
eic_all = pd.concat([pd.read_parquet(EIC_DIR / f"eic_bert_embeddings_{s}.parquet", columns=["hadm_id", "embedding"])
                     for s in ("train", "tuning", "test")], ignore_index=True).drop_duplicates("hadm_id")

rows = []
for cohort in tqdm(COHORTS):
    try:
        strats = load_strats(cohort)
        eic = eic_all.merge(fold_table(cohort), on="hadm_id")

        common = np.intersect1d(strats.hadm_id, eic.hadm_id)
        strats = strats[strats.hadm_id.isin(common)].copy()
        eic    = eic[eic.hadm_id.isin(common)]
        strats["embedding"] = [np.concatenate([a, b]) for a, b in zip(strats.embedding, strats.embedding_demo)]

        for name, df in [("STraTS", strats), ("EIC-BERT", eic)]:
            rows.append({"cohort": cohort, "model": name, **probe(df)})
    except Exception as e:
        print(cohort, e, flush=True)

res = pd.DataFrame(rows)
res.to_csv(Path(PROJECT_ROOT) / "saved_data" / "probe_lr_all_cohorts.csv", index=False)
print(res.groupby("model")[["auroc", "auprc"]].median().round(3))
