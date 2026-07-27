import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("SPLIT_OPT")

SPLIT_KEYS = ["train", "tuning", "held_out"]


class SplitOptimizer:
    def __init__(self, cohorts_dir, mimic_path=None, ratios=(0.8, 0.1, 0.1),
                 min_train=30, min_val=10, min_test=10, seed=42):
        """Configure the optimizer with data paths, target ratios, minimum split sizes, and random seed."""
        self.cohorts_dir = Path(cohorts_dir)
        self.mimic_path = Path(mimic_path) if mimic_path is not None else None
        self.r = np.array(ratios, dtype=np.float64)
        self.min_counts = np.array([min_train, min_val, min_test], dtype=np.float64)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.all_mimic = None
        self.cohort_patients = None
        self.fold = None

    def load_all_patients(self):
        """Load every MIMIC patient id from hosp/patients.csv.gz."""
        d = pd.read_csv(self.mimic_path / "hosp" / "patients.csv.gz", usecols=["subject_id"])
        self.all_mimic = set(d["subject_id"].unique().tolist())
        logger.info("loaded %d MIMIC patients", len(self.all_mimic))
        return self.all_mimic

    def load_cohorts(self):
        """Read every cohort CSV and store its set of unique patient ids."""
        files = sorted(glob.glob(str(self.cohorts_dir / "cohort_*.csv.gz")))
        self.cohort_patients = {}
        for f in files:
            name = Path(f).name.replace("cohort_", "").replace(".csv.gz", "")
            d = pd.read_csv(f, usecols=["subject_id"])
            self.cohort_patients[name] = set(d["subject_id"].unique().tolist())
        logger.info("loaded %d cohorts", len(self.cohort_patients))
        return self.cohort_patients

    def build_index(self):
        """Map patients and cohorts to integer indices and build the cohort-membership lookups."""
        self.names = sorted(self.cohort_patients.keys())
        self.C = len(self.names)
        self.all_pats = sorted(set().union(*self.cohort_patients.values()))
        self.pid2idx = {p: i for i, p in enumerate(self.all_pats)}
        self.P = len(self.all_pats)
        self.cohort_members = [
            np.array([self.pid2idx[p] for p in self.cohort_patients[nm]], dtype=np.int64)
            for nm in self.names
        ]
        self.n = np.array([len(m) for m in self.cohort_members], dtype=np.float64)
        self.patient_cohorts = [[] for _ in range(self.P)]
        for c, mem in enumerate(self.cohort_members):
            for i in mem:
                self.patient_cohorts[i].append(c)
        self.patient_cohorts = [np.array(x, dtype=np.int64) for x in self.patient_cohorts]
        logger.info("cohorts=%d patients=%d", self.C, self.P)

    def _initial_assignment(self):
        """Return a random starting per-patient group assignment at the target ratios."""
        fold = np.empty(self.P, dtype=np.int64)
        order = self.rng.permutation(self.P)
        n_train = int(round(self.r[0] * self.P))
        n_val = int(round(self.r[1] * self.P))
        bounds = np.cumsum([0, n_train, n_val, self.P - n_train - n_val])
        for k in range(3):
            fold[order[bounds[k]:bounds[k + 1]]] = k
        return fold

    def _counts(self, fold):
        """Count how many patients of each cohort fall in each of the three groups."""
        count = np.zeros((self.C, 3), dtype=np.float64)
        for c, mem in enumerate(self.cohort_members):
            for k in range(3):
                count[c, k] = np.sum(fold[mem] == k)
        return count

    def _ratio_obj(self, cnt, nc):
        """Return each cohort's total absolute deviation of its split fractions from the target ratios."""
        return np.abs(self.r[None, :] - cnt / nc[:, None]).sum(1)

    def heuristic(self, patience=10, tol=1e-6, max_passes=200):
        """Locally reassign patients between groups to minimize total deviation from the target ratios."""
        self.fold = self._initial_assignment()
        self.count = self._counts(self.fold)
        contrib = self._ratio_obj(self.count, self.n)
        best_obj = contrib.sum()
        stale = 0
        for it in range(max_passes):
            order = self.rng.permutation(self.P)
            improved = 0
            for p in order:
                cs = self.patient_cohorts[p]
                if len(cs) == 0:
                    continue
                a = self.fold[p]
                nc = self.n[cs]
                cur = self.count[cs]
                old = contrib[cs]
                best_b, best_delta, best_new = a, 0.0, None
                for b in range(3):
                    if b == a:
                        continue
                    tmp = cur.copy()
                    tmp[:, a] -= 1
                    tmp[:, b] += 1
                    newc = self._ratio_obj(tmp, nc)
                    delta = (newc - old).sum()
                    if delta < best_delta - 1e-12:
                        best_delta, best_b, best_new = delta, b, newc
                if best_b != a:
                    self.count[cs, a] -= 1
                    self.count[cs, best_b] += 1
                    contrib[cs] = best_new
                    self.fold[p] = best_b
                    improved += 1
            cur_obj = contrib.sum()
            if cur_obj < best_obj - tol:
                best_obj = cur_obj
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        logger.info("heuristic converged: obj=%.4f pass=%d", contrib.sum(), it + 1)
        return self.fold

    def summary(self, fold=None):
        """Return a per-cohort table of split counts, percentages, and the evaluable flag."""
        fold = self.fold if fold is None else fold
        count = self._counts(fold)
        df = pd.DataFrame({
            "cohort": self.names,
            "n": self.n.astype(int),
            "n_train": count[:, 0].astype(int),
            "n_val": count[:, 1].astype(int),
            "n_test": count[:, 2].astype(int),
        })
        df["train_pct"] = 100 * df["n_train"] / df["n"]
        df["val_pct"] = 100 * df["n_val"] / df["n"]
        df["test_pct"] = 100 * df["n_test"] / df["n"]
        return df

    def save_split(self, out_dir, fold=None):
        fold = self.fold if fold is None else fold
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pats = np.array(self.all_pats)
        for k, key in enumerate(SPLIT_KEYS):
            np.savetxt(out_dir / f"{key}.txt", pats[fold == k], fmt="%d")
        logger.info("saved split to %s", out_dir)

    def save_global_split(self, out_dir, all_patients=None):
        """Save a global split over all_patients: cohort patients keep their optimized groups, the rest are randomly assigned to hold the target ratios globally."""
        if all_patients is None:
            all_patients = self.all_mimic if self.all_mimic is not None else self.load_all_patients()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(self.seed)
        pats = np.array(self.all_pats)
        groups = {key: set(pats[self.fold == k].tolist()) for k, key in enumerate(SPLIT_KEYS)}
        assigned = set().union(*groups.values())
        rest = np.array(sorted(set(all_patients) - assigned))
        rng.shuffle(rest)
        total = len(assigned) + len(rest)
        want = np.round(self.r * total).astype(int)
        need = [max(0, int(want[k] - len(groups[SPLIT_KEYS[k]]))) for k in range(3)]
        need[2] = len(rest) - need[0] - need[1]
        bounds = np.cumsum([0] + need)
        for k, key in enumerate(SPLIT_KEYS):
            groups[key] |= set(rest[bounds[k]:bounds[k + 1]].tolist())
            np.savetxt(out_dir / f"{key}.txt", np.array(sorted(groups[key])), fmt="%d")
        logger.info("saved global split to %s: %s", out_dir, {k: len(v) for k, v in groups.items()})
        return groups

    def run(self, patience=10):
        self.load_cohorts()
        self.build_index()
        self.heuristic(patience=patience)
        return self.summary()


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--cohorts_dir", required=True)
    p.add_argument("--mimic_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--min_train", type=int, default=30)
    p.add_argument("--min_val", type=int, default=10)
    p.add_argument("--min_test", type=int, default=10)
    p.add_argument("--ratios", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=10)
    args = p.parse_args()
    opt = SplitOptimizer(
        cohorts_dir=args.cohorts_dir,
        mimic_path=args.mimic_path,
        ratios=tuple(args.ratios),
        min_train=args.min_train,
        min_val=args.min_val,
        min_test=args.min_test,
        seed=args.seed,
    )
    opt.load_cohorts()
    opt.build_index()
    opt.heuristic(patience=args.patience)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opt.save_global_split(out_dir / "split_ids")
    summary = opt.summary()
    summary.to_csv(out_dir / "cohort_split_summary.csv", index=False)


if __name__ == "__main__":
    main()
