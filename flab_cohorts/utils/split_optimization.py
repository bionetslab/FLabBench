import glob
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config.constants import MIMIC_IV_PATH

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
        files = sorted(glob.glob(str(self.cohorts_dir / "*.csv.gz")))
        self.cohort_patients = {}
        for f in files:
            name = Path(f).name.replace(".csv.gz", "")
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



# Stratified optimizer - Also have and option to select local (cohort) and global ratios
class StratifiedSplitOptimizer:
    def __init__(self, cohorts_dir, mimic_path=MIMIC_IV_PATH, ratios_local=(0.64, 0.16, 0.20),
                 ratios_global=(0.8, 0.1, 0.1), min_stratify_n=None,
                 min_train=30, min_val=10, min_test=10, seed=42,
                 folds_dir=None, first_adm_only=False):
        """Configure the optimizer with data paths, target ratios, minimum split sizes, and random seed."""
        self.cohorts_dir = Path(cohorts_dir)
        self.mimic_path = Path(mimic_path) if mimic_path is not None else None
        self.r = np.array(ratios_local, dtype=np.float64)
        self.r_global = np.array(ratios_global, dtype=np.float64)
        # need enough patients so even the smallest split (e.g. val at 16%) expects at least 1 of each class
        self.min_stratify_n = min_stratify_n if min_stratify_n is not None else int(np.ceil(1 / self.r.min()))
        self.min_counts = np.array([min_train, min_val, min_test], dtype=np.float64)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.folds_dir = Path(folds_dir) if folds_dir is not None else None
        self.first_adm_only = first_adm_only
        self.all_mimic = None
        self.cohort_patients = None
        self.cohort_labels = None
        self.fold = None

    def load_all_patients(self):
        """Load every MIMIC patient id from hosp/patients.csv.gz."""
        d = pd.read_csv(self.mimic_path / "hosp" / "patients.csv.gz", usecols=["subject_id"])
        self.all_mimic = set(d["subject_id"].unique().tolist())
        logger.info("loaded %d MIMIC patients", len(self.all_mimic))
        return self.all_mimic

    def load_cohorts(self):
        """Read every cohort CSV and store its set of unique patient ids and their per-cohort label."""
        files = sorted(glob.glob(str(self.cohorts_dir / "*.csv.gz")))
        self.cohort_patients = {}
        self.cohort_labels = {}
        for f in files:
            name = Path(f).name.replace(".csv.gz", "")
            d = pd.read_csv(f, usecols=["subject_id", "label"])
            lbl = d.groupby("subject_id")["label"].max()
            self.cohort_patients[name] = set(lbl.index.tolist())
            self.cohort_labels[name] = lbl.to_dict()
        logger.info("loaded %d cohorts", len(self.cohort_patients))
        return self.cohort_patients

    def build_index(self):
        """Map patients and cohorts to integer indices and build the cohort-membership lookups, split by per-cohort label."""
        self.names = sorted(self.cohort_patients.keys())
        self.C = len(self.names)
        self.all_pats = sorted(set().union(*self.cohort_patients.values()))
        self.pid2idx = {p: i for i, p in enumerate(self.all_pats)}
        self.P = len(self.all_pats)

        self.cohort_members_by_label = []
        for nm in self.names:
            lbl = self.cohort_labels[nm]
            pos = np.array([self.pid2idx[p] for p, v in lbl.items() if v == 1], dtype=np.int64)
            neg = np.array([self.pid2idx[p] for p, v in lbl.items() if v == 0], dtype=np.int64)
            self.cohort_members_by_label.append((pos, neg))

        self.n = np.array([len(pos) + len(neg) for pos, neg in self.cohort_members_by_label], dtype=np.float64)
        self.n_pos = np.array([len(pos) for pos, neg in self.cohort_members_by_label], dtype=np.float64)
        self.n_neg = np.array([len(neg) for pos, neg in self.cohort_members_by_label], dtype=np.float64)
        self.stratify_mask = (self.n_pos >= self.min_stratify_n) & (self.n_neg >= self.min_stratify_n)

        edges = [[] for _ in range(self.P)]
        for c, nm in enumerate(self.names):
            for p, v in self.cohort_labels[nm].items():
                edges[self.pid2idx[p]].append((c, v))
        self.patient_cohorts = [
            np.array(e, dtype=np.int64) if e else np.zeros((0, 2), dtype=np.int64)
            for e in edges
        ]
        logger.info("cohorts=%d patients=%d stratified_cohorts=%d/%d",
                     self.C, self.P, int(self.stratify_mask.sum()), self.C)

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
        """Count how many positive/negative patients of each cohort fall in each of the three groups."""
        count = np.zeros((self.C, 3, 2), dtype=np.float64)
        for c in range(self.C):
            pos, neg = self.cohort_members_by_label[c]
            for k in range(3):
                if len(pos):
                    count[c, k, 1] = np.sum(fold[pos] == k)
                if len(neg):
                    count[c, k, 0] = np.sum(fold[neg] == k)
        return count

    def _ratio_obj(self, count, n, n_pos, n_neg, stratify_mask):
        """Return each cohort's absolute deviation from the target ratios, stratified by label where the cohort has enough of each."""
        total = count[:, :, 0] + count[:, :, 1]
        obj_plain = np.abs(self.r[None, :] - total / n[:, None]).sum(1)

        safe_pos = np.where(n_pos > 0, n_pos, 1.0)
        safe_neg = np.where(n_neg > 0, n_neg, 1.0)
        obj_pos = np.where(n_pos > 0, np.abs(self.r[None, :] - count[:, :, 1] / safe_pos[:, None]).sum(1), 0.0)
        obj_neg = np.where(n_neg > 0, np.abs(self.r[None, :] - count[:, :, 0] / safe_neg[:, None]).sum(1), 0.0)

        return np.where(stratify_mask, obj_pos + obj_neg, obj_plain)

    def heuristic(self, patience=10, tol=1e-6, max_passes=200):
        """Locally reassign patients between groups to minimize total deviation from the target ratios."""
        self.fold = self._initial_assignment()
        self.count = self._counts(self.fold)
        contrib = self._ratio_obj(self.count, self.n, self.n_pos, self.n_neg, self.stratify_mask)
        best_obj = contrib.sum()
        stale = 0
        for it in range(max_passes):
            order = self.rng.permutation(self.P)
            improved = 0
            for p in order:
                pc = self.patient_cohorts[p]
                if len(pc) == 0:
                    continue
                cs, lbls = pc[:, 0], pc[:, 1]
                a = self.fold[p]
                nc, npos_c, nneg_c = self.n[cs], self.n_pos[cs], self.n_neg[cs]
                smask_c = self.stratify_mask[cs]
                cur = self.count[cs]
                old = contrib[cs]
                rows = np.arange(len(cs))
                best_b, best_delta, best_new = a, 0.0, None
                for b in range(3):
                    if b == a:
                        continue
                    tmp = cur.copy()
                    tmp[rows, a, lbls] -= 1
                    tmp[rows, b, lbls] += 1
                    newc = self._ratio_obj(tmp, nc, npos_c, nneg_c, smask_c)
                    delta = (newc - old).sum()
                    if delta < best_delta - 1e-12:
                        best_delta, best_b, best_new = delta, b, newc
                if best_b != a:
                    self.count[cs, a, lbls] -= 1
                    self.count[cs, best_b, lbls] += 1
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
        if self.folds_dir is not None:
            self.save_cohort_folds(self.folds_dir, first_adm_only=self.first_adm_only)
            self.save_global_split(self.folds_dir / "global_split_ids")
        return self.fold

    def summary(self, fold=None):
        """Return a per-cohort table of split counts, percentages, and per-label breakdown."""
        fold = self.fold if fold is None else fold
        count = self._counts(fold)
        total = count[:, :, 0] + count[:, :, 1]
        df = pd.DataFrame({
            "cohort": self.names,
            "n": self.n.astype(int),
            "n_pos": self.n_pos.astype(int),
            "n_neg": self.n_neg.astype(int),
            "stratified": self.stratify_mask,
            "n_train": total[:, 0].astype(int),
            "n_val": total[:, 1].astype(int),
            "n_test": total[:, 2].astype(int),
            "pos_train": count[:, 0, 1].astype(int),
            "pos_val": count[:, 1, 1].astype(int),
            "pos_test": count[:, 2, 1].astype(int),
        })
        df["train_pct"] = 100 * df["n_train"] / df["n"]
        df["val_pct"] = 100 * df["n_val"] / df["n"]
        df["test_pct"] = 100 * df["n_test"] / df["n"]
        df["pos_rate_train"] = df["pos_train"] / df["n_train"].replace(0, np.nan)
        df["pos_rate_val"] = df["pos_val"] / df["n_val"].replace(0, np.nan)
        df["pos_rate_test"] = df["pos_test"] / df["n_test"].replace(0, np.nan)
        df["pos_rate_cohort"] = df["n_pos"] / df["n"]
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
        want = np.round(self.r_global * total).astype(int)
        need = [max(0, int(want[k] - len(groups[SPLIT_KEYS[k]]))) for k in range(3)]
        need[2] = len(rest) - need[0] - need[1]
        bounds = np.cumsum([0] + need)
        for k, key in enumerate(SPLIT_KEYS):
            groups[key] |= set(rest[bounds[k]:bounds[k + 1]].tolist())
            np.savetxt(out_dir / f"{key}.txt", np.array(sorted(groups[key])), fmt="%d")
        logger.info("saved global split to %s: %s", out_dir, {k: len(v) for k, v in groups.items()})
        return groups

    def save_cohort_folds(self, folds_dir, seed=None, first_adm_only=False):
        seed = self.seed if seed is None else seed
        suffix = "_firstadm" if first_adm_only else ""
        folds_dir = Path(folds_dir)
        pats = np.array(self.all_pats)
        groups = {key: set(pats[self.fold == k].tolist()) for k, key in enumerate(SPLIT_KEYS)}
        usecols = ["subject_id", "hadm_id", "label"] + (["admittime"] if first_adm_only else [])
        for name in self.names:
            cohort = pd.read_csv(self.cohorts_dir / f"{name}.csv.gz", usecols=usecols)
            if first_adm_only:
                cohort = cohort.sort_values("admittime", kind="stable").drop_duplicates("subject_id", keep="first")
            hadms = [
                np.array(cohort.loc[cohort["subject_id"].isin(groups[key]), ["subject_id", "hadm_id"]])
                for key in SPLIT_KEYS
            ]
            out_dir = folds_dir / name / f"seed_{seed}{suffix}"
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "fold_0.pkl", "wb") as f:
                pickle.dump(hadms, f)
        logger.info("saved cohort folds for %d cohorts to %s", len(self.names), folds_dir)

    def run(self, patience=10):
        self.load_cohorts()
        self.build_index()
        self.heuristic(patience=patience)
        return self.summary()


def generate_optimized_folds(cohorts_dir, folds_dir, seed=42, first_adm_only=False, patience=10,
                             ratios_local=(0.64, 0.16, 0.20), ratios_global=(0.8, 0.1, 0.1),
                             mimic_path=MIMIC_IV_PATH):
    """Jointly optimize the patient split across all cohorts and write one fold_0.pkl per cohort."""
    opt = StratifiedSplitOptimizer(
        cohorts_dir=cohorts_dir,
        mimic_path=mimic_path,
        ratios_local=ratios_local,
        ratios_global=ratios_global,
        seed=seed,
        folds_dir=folds_dir,
        first_adm_only=first_adm_only,
    )
    opt.load_cohorts()
    opt.build_index()
    opt.heuristic(patience=patience)
    return opt


def main():
    import argparse
    from types import SimpleNamespace
    from io_utils import set_all_paths
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--cohorts_dir", required=True)
    p.add_argument("--mimic_path", default=MIMIC_IV_PATH)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--first_adm_only", action="store_true")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    folds_dir = set_all_paths(SimpleNamespace(dataset="", cohort="", extractor=""), out=False)["folds_path"]
    opt = generate_optimized_folds(
        cohorts_dir=args.cohorts_dir,
        folds_dir=folds_dir,
        seed=args.seed,
        first_adm_only=args.first_adm_only,
        patience=args.patience,
        mimic_path=args.mimic_path,
    )
    summary = opt.summary()
    summary.to_csv(out_dir / "cohort_split_summary.csv", index=False)


if __name__ == "__main__":
    main()
