import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from mrmr import mrmr_classif

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.constants import PROJECT_ROOT, RANDOM_SEED


class CorrFeatureSelector:
    def __init__(self, extractor="DTB", seed=RANDOM_SEED, alpha=0.2, min_count=10, top_k=None, mrmr_k=None, output_dir=None):
        self.extractor = extractor
        self.seed = seed
        self.alpha = alpha
        self.min_count = min_count
        self.top_k = top_k
        self.mrmr_k = mrmr_k
        self.saved_data_path = Path(PROJECT_ROOT) / "saved_data"
        if mrmr_k is not None:
            method_dir = f"mrmr{mrmr_k}"
        elif top_k is not None:
            method_dir = f"top{top_k}"
        else:
            method_dir = "fdr"
        self.output_dir = Path(output_dir) if output_dir else self.saved_data_path / "features_selected_corr" / method_dir

    @classmethod
    def discover_cohorts(cls, extractor="DTB", saved_data_path=None):
        saved_data_path = Path(saved_data_path) if saved_data_path else Path(PROJECT_ROOT) / "saved_data"
        cohort_dir = saved_data_path / "cohorts" / extractor
        return sorted(
            p.name.removeprefix("cohort_").removesuffix(".csv.gz")
            for p in cohort_dir.glob("cohort_*.csv.gz")
        )

    def _load_cohort(self, cohort):
        cohort_df = pd.read_csv(
            self.saved_data_path / "cohorts" / self.extractor / f"cohort_{cohort}.csv.gz",
            compression="gzip",
            usecols=["hadm_id", "label"],
        )
        features_df = pd.read_csv(self.saved_data_path / "features" / cohort / "features.csv.gz")
        return cohort_df, features_df

    def _train_hadm_ids(self, cohort, fold):
        fold_file = self.saved_data_path / "folds" / cohort / f"seed_{self.seed}" / f"fold_{fold}.pkl"
        with open(fold_file, "rb") as f:
            train_ids, val_ids, test_ids = pickle.load(f)
        return np.concatenate([train_ids[:, 1], val_ids[:, 1]])

    def _correlate(self, agg):
        rows = []
        skipped = []
        for itemid, group in agg.groupby("itemid"):
            n = len(group)
            if n < self.min_count: #correlation is not reliable for very few samples
                skipped.append((itemid, "min_count", n, None))
                continue
            if group["value"].nunique() < 2: # the value should not be consistent
                skipped.append((itemid, "value_const", n, group["label"].mean()))
                continue
            if group["label"].nunique() < 2: # features should be present in both groups
                skipped.append((itemid, "label_const", n, group["label"].iloc[0]))
                continue
            corr, pval = pointbiserialr(group["label"], group["value"])
            if np.isnan(pval):
                skipped.append((itemid, "nan_pval", n, group["label"].mean()))
                continue
            rows.append((itemid, corr, pval, n))

        results = pd.DataFrame(rows, columns=["itemid", "corr", "pval", "n"]).sort_values("pval")
        skipped_df = pd.DataFrame(skipped, columns=["itemid", "reason", "n", "label_info"])
        return results, skipped_df

    @staticmethod
    def _bh_fdr_reject(pvals, alpha):
        pvals = np.asarray(pvals, dtype=float)
        n = len(pvals)
        order = np.argsort(pvals)
        ranked = pvals[order]
        thresh = alpha * (np.arange(1, n + 1) / n)
        below = ranked <= thresh
        if not below.any():
            return np.zeros(n, dtype=bool)
        cutoff = ranked[np.max(np.where(below))]
        return pvals <= cutoff

    def _select_mrmr(self, agg):
        wide = agg.pivot(index="hadm_id", columns="itemid", values="value")
        counts = wide.count()
        nunique = wide.nunique()
        keep = counts[(counts >= self.min_count) & (nunique >= 2)].index
        dropped = [(c, "min_count_or_const", int(counts[c]), None) for c in wide.columns if c not in keep]

        X = wide[keep]
        X = X.fillna(0) #fill with zero or mean (the problem is when we fill with mean if an item is measured only in one group for the other group we impute with the mean)
        y = agg.drop_duplicates("hadm_id").set_index("hadm_id")["label"].reindex(X.index)

        k = min(self.mrmr_k, X.shape[1])
        selected, relevance, _ = mrmr_classif(
            X=X, y=y, K=k, relevance="f", redundancy="c", return_scores=True, show_progress=False #F-statistic (ANOVA F-test),redundancy="c" — Pearson correlation
        )
        #F-stat rewards mean-separation scaled by variance, point-biserial is a signed linear correlation. 
        results = relevance.reset_index()
        results.columns = ["itemid", "relevance"]
        results["selected"] = results["itemid"].isin(selected)
        results = results.sort_values("relevance", ascending=False)
        skipped_df = pd.DataFrame(dropped, columns=["itemid", "reason", "n", "label_info"])
        return results, skipped_df

    def select_for_cohort(self, cohort, fold):
        cohort_df, features_df = self._load_cohort(cohort)
        labels = cohort_df.set_index("hadm_id")["label"]

        train_hadm_ids = self._train_hadm_ids(cohort, fold)
        train_features = features_df[features_df["hadm_id"].isin(train_hadm_ids)]

        agg = train_features.groupby(["hadm_id", "itemid"], as_index=False)["value"].mean()
        agg["label"] = agg["hadm_id"].map(labels)

        if self.mrmr_k is not None:
            return self._select_mrmr(agg)

        results, skipped_df = self._correlate(agg)
        if self.top_k is not None:
            results["selected"] = np.arange(len(results)) < self.top_k
        else:
            results["selected"] = self._bh_fdr_reject(results["pval"].values, self.alpha)
        return results, skipped_df

    def _save(self, cohort, fold, results, skipped_df, selected_itemids):
        cohort_dir = self.output_dir / cohort / f"fold_{fold}"
        cohort_dir.mkdir(parents=True, exist_ok=True)

        with open(cohort_dir / "selected_itemids.pkl", "wb") as f:
            pickle.dump(selected_itemids, f)

        results.to_csv(cohort_dir / "corr_results.csv", index=False)
        skipped_df.to_csv(cohort_dir / "skipped.csv", index=False)

    def run(self, cohorts, fold=0, save=True):
        summary = []
        for cohort in cohorts:
            try:
                results, skipped_df = self.select_for_cohort(cohort, fold)
            except FileNotFoundError:
                continue

            selected_itemids = results.loc[results["selected"], "itemid"].tolist()
            row = {"cohort": cohort, "tested": len(results), "selected": len(selected_itemids)}
            summary.append(row)
            print(row)

            if save:
                self._save(cohort, fold, results, skipped_df, selected_itemids)

        return pd.DataFrame(summary)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--extractor", default="DTB")
    parser.add_argument("--min_count", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--mrmr_k", type=int, default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args()

    selector = CorrFeatureSelector(
        extractor=args.extractor,
        min_count=args.min_count,
        alpha=args.alpha,
        top_k=args.top_k,
        mrmr_k=args.mrmr_k,
    )
    cohorts = CorrFeatureSelector.discover_cohorts(extractor=args.extractor)
    summary = selector.run(cohorts, fold=args.fold, save=not args.no_save)
    print(summary)
