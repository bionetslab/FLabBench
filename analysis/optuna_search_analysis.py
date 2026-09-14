import glob
import numpy as np
import pandas as pd

LOG_SCALE = ["lr", "weight_decay"]

RESULTS = "/home/atuin/b310dc/b310dc10/FLabBench-pipeline/saved_data/results"


class SearchAnalysis:
    """Analyse hyperparameter search results for one model prefix."""

    def __init__(self, model, prefix, n_inner_folds=3, results_root=RESULTS):
        self.model = model
        self.prefix = prefix
        self.n_inner_folds = n_inner_folds
        self.root = results_root
        self.trials = None

    def load(self):
        pattern = f"{self.root}/*/{self.model}/{self.prefix}/fold_*/*/impute_*/variant_*/grid_results.csv"
        rows = []
        for f in glob.glob(pattern):
            g = pd.read_csv(f)
            g["cohort"] = f[len(self.root) + 1:].split("/")[0]
            rows.append(g)
        if not rows:
            raise FileNotFoundError(f"no grid_results.csv for {self.model}/{self.prefix}")
        g = pd.concat(rows, ignore_index=True)
        g["trial"] = g.groupby("cohort").cumcount() // self.n_inner_folds
        g["ifold"] = g.groupby("cohort").cumcount() % self.n_inner_folds
        per_fold = g.pivot_table(index=["cohort", "trial"],
                                 columns="ifold", values="auroc")
        per_fold.columns = [f"fold{i}" for i in per_fold.columns]
        per_fold["mean_folds"] = per_fold.mean(axis=1)
        per_fold["std_folds"] = per_fold.iloc[:, :self.n_inner_folds].std(axis=1)#
        
        self.params = [c for c in ["hid_dim", "num_layers", "dropout", "lr",
                                   "weight_decay", "pos_class_weight", "kernel_size"]
                       if c in g.columns]
        cfg = g.groupby(["cohort", "trial"])[self.params].first()
        self.per_fold = per_fold.join(cfg)

        #print(self.per_fold)
        
    def param_effect(self, param, bins=6):
        t = self.per_fold.copy()
        t["auroc_delta"] = t["mean_folds"] - t.groupby("cohort")["mean_folds"].transform("mean")
        key = t[param]
        if key.nunique() > bins:
            key = pd.qcut(key, bins, duplicates="drop")   # for continuous values bin them
        out = t.groupby(key, observed=True)["auroc_delta"].agg(["mean", "sem", "size"])
        return out.rename(columns={"mean": "effect", "sem": "SE", "size": "n"})
    
    def best_by_effect(sa, bins=6, min_z=2):
        picks = {}
        for p in sa.params:
            e = sa.param_effect(p, bins)
            row = e["effect"].idxmax()
            eff, se = e.loc[row, "effect"], e.loc[row, "SE"]
            if eff < min_z * se:                        # no real effect -> skip the param
                picks[p] = None
            elif isinstance(row, pd.Interval):          # bin -> representative value
                picks[p] = np.sqrt(row.left * row.right) if p in LOG_SCALE else row.mid
            else:
                picks[p] = row
        return picks


        
    def optuna_gain_per_cohort(self):
        d = self.per_fold
        out = d.groupby("cohort").agg(
            n_trials=("mean_folds", "size"),   # number of trials for each cohort
            auroc_avg_over_trials=("mean_folds", "mean"), # average trial score
            spread_across_trials=("mean_folds", "std"),    # how much trials differ
            avg_std_folds=("std_folds", "mean"),        # how much the inner folds disagree
        )
        out["uncertainty_per_trial"] = out.avg_std_folds / np.sqrt(self.n_inner_folds) # standard error of a trial's score ()
        
        best = d.groupby("cohort")["mean_folds"].max()
        out["optuna_gain"] = best - out.auroc_avg_over_trials
        out["gain_in_std"] = out.optuna_gain / out.spread_across_trials  # z of the best trial; ~2.25 expected by chance over 50 trials
        out["spread_over_noise"] = out.spread_across_trials / out.uncertainty_per_trial
        med = d.groupby("cohort")["mean_folds"].median()
        out["z"] = (best - med) / out.uncertainty_per_trial  # winner vs typical config, in units of trial noise
        out = out.sort_values("z", ascending=False)

        print(f"optuna worked (z > 3) in {(out.z > 3).mean():.1%} of {len(out)} cohorts")

        return out.round(4)
        
        
    def find_best_config(self):
        
        d = self.per_fold
        idx = d.groupby("cohort")["mean_folds"].idxmax()
        return d.loc[idx, ["mean_folds", "std_folds"] + self.params]#.droplevel("trial")
    
            
    def build_grid_across_cohorts(self, bins=5, top=2): # find most frequntly picked params across cohorts retruns two best per param
        return build_grid(self.find_best_config(), self.params, bins, top)


def build_grid(won, params, bins=5, top=1):
    grid = {}
    for p in params:
        s = won[p].dropna()
        if s.nunique() <= bins:                       # discrete
            grid[p] = sorted(s.value_counts().head(top).index)
        else:
            if p in LOG_SCALE:
                edges = np.logspace(np.log10(s.min()), np.log10(s.max()), bins + 1)
                b = pd.cut(s, edges, include_lowest=True)
            else:
                b = pd.cut(s, bins)
            keep = b.value_counts().head(top).index
            grid[p] = sorted(s[b == k].median() for k in keep)
    return grid


def clean(vals, p):
    if all(float(v).is_integer() for v in vals):          # discrete params
        return sorted(set(vals))
    if p not in LOG_SCALE:                                # dropout -> nearest 0.05
        return sorted({round(v * 20) / 20 for v in vals})
    out = []
    for v in vals:                                        # nearest 1/2/3/5 x 10^k
        e = np.floor(np.log10(v))
        m = min([1, 2, 3, 5, 10], key=lambda x: abs(np.log(v / 10 ** e / x)))
        out.append(float(f"{m * 10 ** e:.1g}"))
    return sorted(set(out))


def select_across_runs(model, prefixes, method="effect", n_inner_folds=3, bins=6, top=1):
    rows = {}
    for prefix in prefixes:
        sa = SearchAnalysis(model, prefix, n_inner_folds=n_inner_folds)
        sa.load()
        if method == "effect":
            rows[prefix] = sa.best_by_effect(bins)
        else:
            g = sa.build_grid_across_cohorts(bins, top)
            rows[prefix] = {p: v[0] for p, v in g.items()}
    return pd.DataFrame(rows).T



if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Summarise a hyperparameter search run.")
    ap.add_argument("model", help="e.g. lstm, gru, tcn")
    ap.add_argument("prefix", nargs="*", help="one or more run prefixes")
    ap.add_argument("--folds", type=int, default=3, help="inner folds (default 3)")

    args = ap.parse_args()
    

    PREFIXES = ["optuna_14d", "optuna_180d", "optuna_stats_180d_240h", "optuna_stats_180d_720h"]

    prefixes = args.prefix or PREFIXES

    for method in ["effect", "freq"]:
        print(f"\n===== {args.model} / picked by {method} =====")
        print(select_across_runs(args.model, prefixes, method, args.folds))

    for prefix in prefixes:
        print(f"\n===== {args.model} / {prefix} =====")
        sa = SearchAnalysis(args.model, prefix, n_inner_folds=args.folds)
        sa.load()
        print(sa.optuna_gain_per_cohort())

