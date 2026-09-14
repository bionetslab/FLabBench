import pickle
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from types import SimpleNamespace
from functools import cached_property
from scipy.stats import entropy

from io_utils import set_all_paths
from config.constants import PROJECT_ROOT
from flab_training.preprocessor import PreprocessorA


BASE = Path(PROJECT_ROOT) / "saved_data"


def load_selected_itemids(source="mimic_top_100", n_top=None):
    # the file is ordered by feature importance, so the first n_top are the strongest
    path = Path(PROJECT_ROOT) / "data" / "top_features" / "mimic_top100_features.pkl"
    with open(path, "rb") as f:
        itemids = [str(x) for x in pickle.load(f)]
    return itemids[:n_top]
    

class Cohort:
    """Loads one cohort's load_cohort, features and folds."""

    def __init__(self, cohort_name, days_before_discharge=14, agg_int=24, seed=42, fold=0, first_adm_only=True, feature_selection_source="mimic_top_100",
                 extractor="DTB", dataset="MIMIC_IV", split="all", n_top_features=None):

        self.cohort_name = cohort_name
        self.days_before_discharge = days_before_discharge
        self.agg_int = agg_int
        self.seed = seed
        self.fold = fold
        self.first_adm_only = first_adm_only
        self.split = split
        self.selected_itemids = load_selected_itemids(feature_selection_source, n_top_features)

        args = SimpleNamespace(dataset=dataset, cohort=cohort_name, extractor=extractor,
                               days_before_discharge=days_before_discharge,
                               first_adm_only=first_adm_only)
        self.paths = set_all_paths(args, out=False)
        self.hadm_ids = self.load_cohort.hadm_id.unique()
        self.labels = self.load_cohort.set_index("hadm_id")["label"].reindex(self.hadm_ids)

    # ---------- Load data ----------
    @cached_property
    def load_cohort(self):
        df = pd.read_csv(self.paths["cohort_path"] / f"{self.cohort_name}.csv.gz", compression="gzip")
        if self.first_adm_only:
            df = (df.sort_values("admittime", kind="stable").drop_duplicates("subject_id", keep="first"))
        if self.split != "all":
            df = df[df.hadm_id.isin(self.folds()[self.split])]
        return df.reset_index(drop=True)
    
    @cached_property
    def load_features(self):
        df = pd.read_csv(self.paths["features_path"] / self.cohort_name / "features.csv.gz",
            dtype={"subject_id": "int64", "hadm_id": "int64",
                "minute": "int64", "itemid": "string", "value": "float64"},
        )
        if self.selected_itemids is not None:
            df = df[df.itemid.isin(self.selected_itemids)]
        return df  
    
    def folds(self):
        suffix = "_firstadm" if self.first_adm_only else ""
        f = self.paths["folds_path"] / f"seed_{self.seed}{suffix}" / f"fold_{self.fold}.pkl"
        with open(f, "rb") as fh:
            train, val, test = pickle.load(fh)
        return {"train": train[:, 1], "val": val[:, 1], "test": test[:, 1]}
    

    # ---------- add CHARACTERISTICS ----------
    
    # cohort charactristics
    def cohort_characteristics(self): 
        c = self.load_cohort
        n = len(c)
        n_pos = int(c["label"].sum())
        return {"cohort_name": self.cohort_name,
                "n": n,
                "n_positive": n_pos,
                "target_rate": n_pos / n if n else float("nan")}
        
    def compute_n_events(self):
        # total events per admission, 0 for admissions absent from the features file
        unmatched = self.labels.isna().sum()
        if unmatched:
            print(f"{self.cohort_name}: unmatched hadm_ids: {unmatched}")
        n_events = self.load_features.groupby("hadm_id").size().reindex(self.hadm_ids, fill_value=0)
        is_pos = self.labels == 1

        return {"n_events_mean":     n_events.mean(),
                "n_events_median":   n_events.median(),
                "n_events_mean_pos": n_events[is_pos].mean(),
                "n_events_mean_neg": n_events[~is_pos].mean()}

    def compute_feature_matrix_missingness(self):
        cohort_features = self.load_features

        # build input matrix using PreprocessorA to have the same input 
        feature_matrix = self.measurements_binned(cohort_features)
        # one avg value per bin
        feature_matrix = feature_matrix.groupby(["hadm_id", "itemid", "int"])["value"].mean().reset_index()
        
        n_bins = int(np.ceil((self.days_before_discharge + 1) * 24 / self.agg_int))
        n_items = len(self.selected_itemids)
        max_item_bins = n_items * n_bins

        # (itemid, bin) pairs measured per admission, at most max_item_bins
        n_item_bins = feature_matrix.groupby("hadm_id").size().reindex(self.hadm_ids, fill_value=0)

        missingness = pd.DataFrame({
            "n_item_bins": n_item_bins,
            "missing_ratio":  1 - n_item_bins / max_item_bins,
        })

        overall = {#"item_bins_mean":   missingness.n_item_bins.mean(),
                   #"item_bins_median": missingness.n_item_bins.median(),
                   "itembin_missing_mean":     missingness.missing_ratio.mean(),
                   "itembin_missing_median":   missingness.missing_ratio.median(),
                   "itembin_missing_std":      missingness.missing_ratio.std()}

        by_label = missingness.groupby(self.labels).agg(
            #item_bins_mean   = ("n_item_bins", "mean"),
            #item_bins_median = ("n_item_bins", "median"),
            itembin_missing_mean     = ("missing_ratio", "mean"),
            itembin_missing_median   = ("missing_ratio", "median"),
            itembin_missing_std      = ("missing_ratio", "std"),
        ).rename(index={0: "neg", 1: "pos"})

        return {**overall,
                **{f"{col}_{lab}": v for lab, row in by_label.iterrows() for col, v in row.items()}}

    def compute_item_missingness(self):
        # per item, share of admissions that never measure it at all (no time bins)
        missing_per_item = self.value_matrix.isna().mean()
        #for each itemid how many admissions doesn't have it and then get the mean across itemids
        return {"itemid_missing_mean": missing_per_item.mean(),
                "itemid_missing_std":  missing_per_item.std()}

    def compute_adm_missingness(self):
        # per admission, share of the selected itemids that are never measured at all (no time bins)
        missing_per_adm = self.value_matrix.isna().mean(axis=1)
        #for each admission how many itemids it doesn't have and then get the mean across admissions
        return {"adm_missing_mean": missing_per_adm.mean(), # identical to itemid_missing_mean
                "adm_missing_std":  missing_per_adm.std()}

    def compute_value_entropy(self, n_bins=10, n_draws=5):
        # per itemid, how the values of one class spread over the bins, averaged over itemids
        m = self.value_matrix
        is_pos = (self.labels == 1).to_numpy()
        rng = np.random.default_rng(self.seed)

        h_pos, h_neg = [], []
        for item in m.columns:
            v = m[item].to_numpy()
            x_pos, x_neg = v[is_pos], v[~is_pos]
            x_pos, x_neg = x_pos[~np.isnan(x_pos)], x_neg[~np.isnan(x_neg)]

            # equal count bin edges over all admissions, so both classes use the same bins
            edges = np.unique(np.nanquantile(v, np.linspace(0, 1, n_bins + 1)))
            k = min(len(x_pos), len(x_neg))
            if k < n_bins or len(edges) < 3:
                continue

            # both classes on k values, the entropy of a histogram grows with the number of samples
            def h(x):
                if len(x) == k:
                    return entropy(np.histogram(x, bins=edges)[0])
                return float(np.mean([entropy(np.histogram(rng.choice(x, k, replace=False), bins=edges)[0])
                                      for _ in range(n_draws)]))

            h_pos.append(h(x_pos))
            h_neg.append(h(x_neg))

        if not h_pos:
            return {"h_diff": float("nan")}
        # both sides carry the same subsample bias, only the difference is comparable
        return {"h_diff": float(np.mean(h_neg)) - float(np.mean(h_pos))}

    def compute_cohens_d(self, labels=None):
        # per itemid, distance of the class means in pooled within class standard deviations
        # How far apart two groups are in standard-deviation units
        m = self.value_matrix
        is_pos = ((self.labels if labels is None else labels) == 1).to_numpy()
        pos, neg = m[is_pos], m[~is_pos]

        n_pos, n_neg = pos.notna().sum(), neg.notna().sum() # how many po patients have this lab, per lab counts missingness differ from lab to lab
        pooled_var = ((n_pos - 1) * pos.var() + (n_neg - 1) * neg.var()) / (n_pos + n_neg - 2)
        d = (pos.mean() - neg.mean()).abs() / np.sqrt(pooled_var.replace(0, np.nan))

        # with no class difference d still scatters around 0 with this variance, small classes scatter more
        noise_var = 1 / n_pos + 1 / n_neg
        # d squared carries that noise as an offset, remove it before averaging over itemids
        d2 = (d ** 2 - noise_var).mean()

        return {"cohens_d_adj": np.sqrt(d2) if d2 > 0 else 0.0}

    def compute_cohens_d_null(self, n_shuffles=50):
        # the same statistics on shuffled labels, what each one reaches without any real class difference
        # the class sizes and the missingness stay as they are, only the labels move
        rng = np.random.default_rng(self.seed)
        draws = [self.compute_cohens_d(pd.Series(rng.permutation(self.labels.to_numpy()),
                                                 index=self.labels.index))
                 for _ in range(n_shuffles)]
        obs = self.compute_cohens_d()
        # distance to the null in units of the null spread, comparable across statistics and cohorts
        out = {}
        for k in obs:
            shuffled = np.array([d[k] for d in draws], dtype=float)
            std = shuffled.std()
            out[f"{k}_z"] = (obs[k] - shuffled.mean()) / std if std > 0 else float("nan")
        return out

    @cached_property
    def value_matrix(self):
        # one mean value per admission and itemid, no time information, all selected itemids as columns
        m = self.load_features.groupby(["hadm_id", "itemid"])["value"].mean().unstack()
        return m.reindex(index=self.hadm_ids, columns=self.selected_itemids)

    def config(self):
        return {"days_before_discharge": self.days_before_discharge,
                "agg_int": self.agg_int,
                "seed": self.seed,
                "fold": self.fold,
                "first_adm_only": self.first_adm_only,
                "split": self.split,
                "n_itemids": len(self.selected_itemids)}

    def characteristics(self):
        # cohort, event and missingness characteristics in one flat row
        return {**self.cohort_characteristics(),
                **self.config(),
                **self.compute_n_events(),
                **self.compute_feature_matrix_missingness(),
                **self.compute_item_missingness(),
                **self.compute_adm_missingness(),
                **self.compute_value_entropy(),
                **self.compute_cohens_d(),
                **self.compute_cohens_d_null()}
        

    


    def measurements_binned(self,cohort_features):
        p = SimpleNamespace(
            data=cohort_features,
            args=SimpleNamespace(agg_int=self.agg_int,
                                 days_before_discharge=self.days_before_discharge,
                                 logger=SimpleNamespace(write=lambda *a: None)),
        )
        PreprocessorA.trim(p)
        return p.data






    def sanity_check(self):
        c, f = self.load_cohort, self.load_features
        folds = self.folds()

        cohort_ids = set(c.hadm_id)
        fold_ids = set(np.concatenate([folds["train"], folds["val"], folds["test"]]))
        tv_ids = set(np.concatenate([folds["train"], folds["val"]]))
        feat_ids = set(f.hadm_id)
        print("---"*30)
        print("cohort            :", len(cohort_ids), "adms |", c.subject_id.nunique(), "subjects")
        print("folds             :", len(fold_ids), "| equal to cohort:", fold_ids == cohort_ids)
        print("features          :", len(feat_ids), "adms |", f.itemid.nunique(), "itemids")
        print("cohort no features:", len(cohort_ids - feat_ids))
        print("train+val no feats:", tv_ids - feat_ids)
        print("---"*30)
    
    
    
def load_edge_cc(extractor="DTB"):
    df = pd.read_csv(BASE / "cohorts" / extractor / "new" / f"selected_edges_{extractor}_all.csv")
    df["cohort_name"] = df["D1"] + "-" + df["D2"]
    df = df[["cohort_name", "D1", "D2", "RR", "counts_cohort",
            "female_counts_cohort", "AGE_AT_DISEASE_cohort",
            "CODE_DIFF_DAYS_cohort", "death_counts_cohort"]]

    df["female_rate"] = df["female_counts_cohort"] / df["counts_cohort"]
    df["death_rate"]  = df["death_counts_cohort"]  / df["counts_cohort"]

    return (df.rename(columns={"AGE_AT_DISEASE_cohort": "age","CODE_DIFF_DAYS_cohort": "W"})
            .drop(columns=["female_counts_cohort", "death_counts_cohort"]))


def build_characteristics_table(cohort_names, extractor="DTB", **kwargs):
    rows = []
    for name in tqdm(cohort_names):
        try:
            ch = Cohort(name, extractor=extractor, **kwargs)
            rows.append(ch.characteristics())
        except Exception as e:
            print(f"skipping {name}: {e}")

    #add edge features
    final_cc_df = pd.DataFrame(rows).merge(load_edge_cc(extractor), on="cohort_name")

    return final_cc_df


def save_characteristics(cohort_names, out_name=None, extractor="DTB", **kwargs):
    df = build_characteristics_table(cohort_names, extractor=extractor, **kwargs)

    out_dir = BASE / "characteristics"
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_name is None:
        days = kwargs.get("days_before_discharge", 14)
        agg_int = kwargs.get("agg_int", 24)
        n_itemids = int(df["n_itemids"].iloc[0])
        out_name = f"characteristics_{days}d_agg{agg_int}_top{n_itemids}.csv"

    path = out_dir / out_name
    df.to_csv(path, index=False)
    print(f"saved {len(df)} cohorts x {df.shape[1]} columns to {path}")
    return df

    
    
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cohorts", nargs="*", default=["A41-J10"], help="cohort names, or use --cohort_file")
    p.add_argument("--cohort_file", help="text file with one cohort name per line")
    p.add_argument("--out_name", help="file name inside saved_data/characteristics")
    p.add_argument("--days_before_discharge", type=int, default=14)
    p.add_argument("--agg_int", type=int, default=24)
    p.add_argument("--n_top_features", type=int, help="keep only the n most important itemids")
    args = p.parse_args()

    cohorts = (Path(args.cohort_file).read_text().split() if args.cohort_file else args.cohorts)
    save_characteristics(cohorts, out_name=args.out_name,
                         days_before_discharge=args.days_before_discharge,
                         agg_int=args.agg_int,
                         n_top_features=args.n_top_features)

