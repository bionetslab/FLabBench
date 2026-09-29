#!/usr/bin/env python3
"""
Bias analysis: utility and detectability for one PPI dataset attribute.

Run once per attribute (--attribute flag); Nextflow calls this in parallel for
all attributes.

Attributes
----------
sequence_similarity           – BLAST pident between the pair, normalised to [0,1]
embedding_similarity          – cosine similarity of the two protein embeddings
functional_relatedness_BP/MF/CC – Jaccard similarity of GO term sets
self_interactions             – 1 if both proteins are identical, 0 otherwise
same species                  – 1 if both proteins belong to the same species, 0 otherwise

Utility       = NMI(A; Y) = MI / sqrt(H(A)·H(Y)), MI estimated by sklearn kNN, H(A) by histogram
Detectability = Spearman ρ of a Ridge regressor predicting A from the pair embedding X

Output
------
{attribute}_bias_mqc.tsv  – MultiQC table with splits as rows
"""

import argparse
import csv
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.constants import PROJECT_ROOT


# def build_pair_X(pairs, embeddings):
#     """Return (X, mask) where mask[i]=True when both proteins have embeddings."""
#     rows, mask = [], []
#     for p1, p2 in pairs:
#         a, b = (p1, p2) if p1 <= p2 else (p2, p1)
#         if a in embeddings and b in embeddings:
#             rows.append(np.concatenate([embeddings[a], embeddings[b]]))
#             mask.append(True)
#         else:
#             mask.append(False)
#     X = np.array(rows, dtype=np.float32) if rows else np.empty((0, 0), dtype=np.float32)
#     return X, np.array(mask, dtype=bool)


def build_X(emb_df, attrs):
    """Return (X, df): one embedding per admission, joined with its attributes on hadm_id
    (admissions missing an embedding or attributes are dropped)."""
    df = emb_df.set_index("hadm_id").join(attrs, how="inner")
    X = np.stack(df["embedding"].to_numpy()).astype(np.float32) if len(df) else np.empty((0, 0), dtype=np.float32)
    return X, df


# def parse_blast_pident(path):
#     """Return {prot: {other_prot: max_pident}} from BLAST outfmt-6 with pident at col 4."""
#     sim = defaultdict(dict)
#     with open(path) as fh:
#         for line in fh:
#             cols = line.rstrip().split("\t")
#             if len(cols) < 5:
#                 continue
#             q = cols[0]
#             s = cols[1]
#             if q == s:
#                 continue
#             try:
#                 pident = float(cols[4])
#             except ValueError:
#                 continue
#             if pident > sim[q].get(s, -1):
#                 sim[q][s] = pident
#             if pident > sim[s].get(q, -1):
#                 sim[s][q] = pident
#     return sim


# def seq_sim_within_pair(pairs, blast_sim):
#     """BLAST pident between the two proteins in each pair, normalised to [0, 1]."""
#     A = []
#     for p1, p2 in pairs:
#         pident = blast_sim.get(p1, {}).get(p2, 0.0)
#         A.append(pident / 100.0)
#     return np.array(A, dtype=np.float32)


# def emb_sim_within_pair(pairs, embeddings):
#     """Cosine similarity between the two individual protein embeddings in each pair."""
#     sims = []
#     for p1, p2 in pairs:
#         e1, e2 = embeddings[p1], embeddings[p2]
#         n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
#         sims.append(float(np.dot(e1, e2) / (n1 * n2)) if n1 > 0 and n2 > 0 else 0.0)
#     return np.array(sims, dtype=np.float32)


# def load_go_annotations(path):
#     """Return {protein_id: {BP/MF/CC: frozenset[go_term]}} from the four-column TSV."""
#     result = {}
#     with open(path) as fh:
#         reader = csv.DictReader(fh, delimiter="\t")
#         for row in reader:
#             acc = row["protein_id"].strip()
#             result[acc] = {
#                 cat: frozenset(t.strip() for t in row[col].strip().split(";") if t.strip())
#                 for cat, col in [("BP", "go_bp"), ("MF", "go_mf"), ("CC", "go_cc")]
#             }
#     return result


# def self_interactions(pairs):
#     """1 if both proteins in the pair are identical, 0 otherwise."""
#     return np.array([1.0 if p1 == p2 else 0.0 for p1, p2 in pairs], dtype=np.float32)


# def load_species(path):
#     """Return {protein_id: taxon_id} from the two-column TSV written by fetch_data.py."""
#     mapping = {}
#     with open(path) as fh:
#         reader = csv.DictReader(fh, delimiter="\t")
#         for row in reader:
#             mapping[row["protein_id"].strip()] = row["taxon_id"].strip()
#     return mapping


# def same_species_indicator(pairs, species_map):
#     """1 if both proteins in the pair belong to the same species, 0 otherwise."""
#     A = []
#     for p1, p2 in pairs:
#         t1 = species_map.get(p1, "")
#         t2 = species_map.get(p2, "")
#         A.append(1.0 if t1 and t2 and t1 == t2 else 0.0)
#     return np.array(A, dtype=np.float32)


# def build_train_degree_ratio(train_pairs, train_labels):
#     """Return {protein_id: pos_degree / (pos_degree + neg_degree)} from the
#     training split's own labelled pairs. A protein absent from training has
#     no entry (undefined, not zero)."""
#     pos = defaultdict(int)
#     neg = defaultdict(int)
#     for (p1, p2), y in zip(train_pairs, train_labels):
#         bucket = pos if y == 1 else neg
#         bucket[p1] += 1
#         bucket[p2] += 1
#     ratio = {}
#     for p in set(pos) | set(neg):
#         total = pos[p] + neg[p]
#         if total > 0:
#             ratio[p] = pos[p] / total
#     return ratio


# def topology_shortcut(pairs, train_degree_ratio):
#     """Per pair, how exploitable the "topology shortcut" is: each endpoint's
#     training positive-rate pos_degree_in_training(p)/pos_plus_neg_degree_in_training(p),
#     using whichever endpoint(s) occurred in training (averaged if both did).
#     NaN if neither endpoint occurred in training -- the caller drops those
#     pairs, since the shortcut can't apply to a wholly-unseen pair."""
#     A = []
#     for p1, p2 in pairs:
#         r1 = train_degree_ratio.get(p1)
#         r2 = train_degree_ratio.get(p2)
#         if r1 is None and r2 is None:
#             A.append(np.nan)
#         elif r1 is None:
#             A.append(r2)
#         elif r2 is None:
#             A.append(r1)
#         else:
#             A.append((r1 + r2) / 2.0)
#     return np.array(A, dtype=np.float32)


# def _prepare_split(pairs, y, embeddings):
#     """Return (X, filtered_pairs, filtered_y) keeping only pairs with embeddings."""
#     X, mask = build_pair_X(pairs, embeddings)
#     return X, [p for p, ok in zip(pairs, mask) if ok], y[mask]


# def func_relatedness(pairs, go_anns, category):
#     """Jaccard index of GO term sets for one category per pair; 0.0 if union is empty."""
#     empty = frozenset()
#     sims = []
#     for p1, p2 in pairs:
#         s1 = go_anns.get(p1, {}).get(category, empty)
#         s2 = go_anns.get(p2, {}).get(category, empty)
#         union_size = len(s1 | s2)
#         sims.append(len(s1 & s2) / union_size if union_size > 0 else 0.0)
#     return np.array(sims, dtype=np.float32)


def load_features(path, top_features_path):
    """Return lab events (hadm_id, itemid) from features.csv.gz, restricted to the top-100 labs the model sees."""
    feats = pd.read_csv(path, usecols=["hadm_id", "itemid"], dtype={"hadm_id": "int64", "itemid": "int64"})
    with open(top_features_path, "rb") as fh:
        top = set(int(x) for x in pickle.load(fh))
    return feats[feats["itemid"].isin(top)]


def load_cohort(path, first_adm_only=True):
    """Return the cohort table indexed by hadm_id, with age and gender."""
    cohort = pd.read_csv(path, usecols=["subject_id", "hadm_id", "admittime", "age", "gender"])
    if first_adm_only:
        cohort = cohort.sort_values("admittime", kind="stable").drop_duplicates("subject_id", keep="first")
    return cohort.drop_duplicates("hadm_id").set_index("hadm_id")


def n_events(feats):
    """Number of lab measurements per admission."""
    return feats.groupby("hadm_id").size().astype(np.float32)


def n_labs(feats):
    """Number of distinct lab types (itemids) measured per admission."""
    return feats.groupby("hadm_id")["itemid"].nunique().astype(np.float32)


def age(cohort):
    """Patient age per admission."""
    return cohort["age"].astype(np.float32)


def gender(cohort):
    """1 if female, 0 if male, per admission."""
    return cohort["gender"].str.upper().map({"M": 0.0, "F": 1.0}).astype(np.float32)


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

    _, y_counts = np.unique(y, return_counts=True)
    h_y = _entropy(y_counts)

    if discrete_features:
        _, a_counts = np.unique(A, return_counts=True)
    else:
        a_counts, _ = np.histogram(A, bins=n_bins)
    h_a = _entropy(a_counts)

    denom = np.sqrt(h_a * h_y)
    return float(np.clip(mi / denom, 0.0, 1.0)) if denom > 0 else 0.0


def analyse(A, X, y, name, seed=42, n_bins=10, cv=True, n_perm=0):
    """Return dict with nmi, related, detectability (Spearman ρ), and with n_perm > 0
    the permutation p-value and z-score of the NMI.
    """
    # discrete_features = name == "self_interactions"
    discrete_features = name == "gender"
    nmi = compute_nmi(A, y, discrete_features, seed, n_bins)
    print(f"    NMI(A;Y) = {nmi:.4f}  (related? {'Yes' if nmi > 0 else 'No'})", file=sys.stderr)

    # shuffled labels keep cohort size and class balance but break the A-Y link, so their NMI is the chance level
    nmi_p, nmi_z = np.nan, np.nan
    if n_perm > 0:
        rng = np.random.default_rng(seed)
        null = np.array([compute_nmi(A, rng.permutation(y), discrete_features, seed, n_bins) for _ in range(n_perm)])
        nmi_p = float((1 + np.sum(null >= nmi)) / (1 + n_perm))
        nmi_z = float((nmi - null.mean()) / null.std()) if null.std() > 0 else np.nan

    if X.shape[0] < 10:
        detectability = 0.0
    else:
        reg = Ridge(alpha=1.0, solver="sag", random_state=seed, max_iter=1000)
        # out-of-sample predictions, in-sample fit inflates ρ when patients are few relative to the embedding dims
        pred = cross_val_predict(reg, X, A, cv=5) if cv else reg.fit(X, A).predict(X)
        detectability = float(spearmanr(A, pred)[0])

    return {"nmi": nmi, "related": nmi > 0, "detectability": detectability, "nmi_p": nmi_p, "nmi_z": nmi_z}


# ATTRIBUTES = {
#     "sequence_similarity": lambda pairs, blast_sim, emb, go, sp, train_deg: seq_sim_within_pair(pairs, blast_sim),
#     "embedding_similarity": lambda pairs, blast_sim, emb, go, sp, train_deg: emb_sim_within_pair(pairs, emb),
#     "functional_relatedness_BP": lambda pairs, blast_sim, emb, go, sp, train_deg: func_relatedness(pairs, go, "BP"),
#     "functional_relatedness_MF": lambda pairs, blast_sim, emb, go, sp, train_deg: func_relatedness(pairs, go, "MF"),
#     "functional_relatedness_CC": lambda pairs, blast_sim, emb, go, sp, train_deg: func_relatedness(pairs, go, "CC"),
#     "self_interactions": lambda pairs, blast_sim, emb, go, sp, train_deg: self_interactions(pairs),
#     "same_species": lambda pairs, blast_sim, emb, go, sp, train_deg: same_species_indicator(pairs, sp),
#     "topology_shortcut": lambda pairs, blast_sim, emb, go, sp, train_deg: topology_shortcut(pairs, train_deg),
# }

ATTRIBUTES = {
    "n_events": lambda feats, cohort: n_events(feats),
    "n_labs": lambda feats, cohort: n_labs(feats),
    "age": lambda feats, cohort: age(cohort),
    "gender": lambda feats, cohort: gender(cohort),
}


# def write_mqc(attribute, results):
#     fname = f"{attribute}_bias_mqc.tsv"
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


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--attribute", required=True, choices=list(ATTRIBUTES), help="attribute to analyse")
#     ap.add_argument("--train", required=True)
#     ap.add_argument("--val", required=True)
#     ap.add_argument("--test_balanced", required=True)
#     ap.add_argument("--test_realistic", required=True)
#     ap.add_argument("--blast", required=True)
#     ap.add_argument("--embeddings", required=True)
#     ap.add_argument("--go_annotations", required=True)
#     ap.add_argument("--species", default=None, help="species.tsv from fetch_data; required for same_species attribute")
#     ap.add_argument("--seed", type=int, default=42)
#     args = ap.parse_args()

#     print(f"=== {args.attribute} ===", file=sys.stderr)

#     compute = ATTRIBUTES[args.attribute]

#     # Load data ---------------------------------------------------------------
#     print("Loading embeddings ...", file=sys.stderr)
#     embeddings = load_embeddings(args.embeddings)

#     blast_sim = None
#     if args.attribute == "sequence_similarity":
#         print("Parsing BLAST similarities ...", file=sys.stderr)
#         blast_sim = parse_blast_pident(args.blast)

#     go_anns = None
#     if args.attribute.startswith("functional_relatedness"):
#         go_anns = load_go_annotations(args.go_annotations)

#     species = None
#     if args.attribute == "same_species":
#         if args.species is None:
#             sys.exit("--species is required for the same_species attribute")
#         species = load_species(args.species)

#     train_degree_ratio = None
#     if args.attribute == "topology_shortcut":
#         train_pairs, train_y = read_labelled_csv(args.train)
#         train_degree_ratio = build_train_degree_ratio(train_pairs, train_y)

#     split_paths = [
#         ("train", args.train),
#         ("val", args.val),
#         ("test_balanced", args.test_balanced),
#         ("test_realistic", args.test_realistic),
#     ]

#     results = []
#     any_nontrain_qualified = False
#     for split, path in split_paths:
#         pairs, y = read_labelled_csv(path)
#         X, pairs_f, y_f = _prepare_split(pairs, y, embeddings)
#         A = compute(pairs_f, blast_sim, embeddings, go_anns, species, train_degree_ratio)

#         if args.attribute == "topology_shortcut":
#             # Only pairs with at least one endpoint seen during training are
#             # meaningful here; this is a no-op for every other attribute.
#             valid = ~np.isnan(A)
#             A, X, y_f = A[valid], X[valid], y_f[valid]
#             if split != "train" and X.shape[0] > 0:
#                 any_nontrain_qualified = True
#             if X.shape[0] == 0:
#                 print(f"  {split}: 0 pairs qualify for topology_shortcut (no training overlap)", file=sys.stderr)
#                 continue

#         print(f"  {X.shape[0]} {split} pairs retained", file=sys.stderr)
#         r = analyse(A, X, y_f, args.attribute, seed=args.seed)
#         results.append((split, r))
#         print(f"  [{split}] NMI={r['nmi']:.4f}  ρ={r['detectability']:.4f}", file=sys.stderr)

#     if args.attribute == "topology_shortcut" and not any_nontrain_qualified:
#         print(
#             "  topology_shortcut not applicable: no val/test_balanced/test_realistic "
#             "protein occurs in the training set (expected for leakage-aware splits) "
#             "-- skipping output.",
#             file=sys.stderr,
#         )
#     elif results:
#         write_mqc(args.attribute, results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attribute", required=True, choices=list(ATTRIBUTES), help="attribute to analyse")
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--emb_dir", default="saved_data/embeddings/strats_pretrained")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--first_adm_only", action="store_true")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out_dir", default="saved_data/bias")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pool_splits", action="store_true", help="analyse train+val+test together as one split 'all'")
    ap.add_argument("--no_cv", action="store_true", help="in-sample detectability (fit and predict on the same admissions)")
    ap.add_argument("--n_perm", type=int, default=0, help="label permutations for the NMI p-value and z-score, 0 = skip")
    args = ap.parse_args()

    print(f"=== {args.attribute} ===", file=sys.stderr)

    compute = ATTRIBUTES[args.attribute]

    # Load data ---------------------------------------------------------------
    feats = None
    if args.attribute in ("n_events", "n_labs"):
        suffix = "_firstadm" if args.first_adm_only else ""
        feats = load_features(Path(PROJECT_ROOT) / "saved_data" / "features" / f"features_{args.days}d{suffix}" / args.cohort / "features.csv.gz",
                              Path(PROJECT_ROOT) / "data" / "top_features" / "mimic_top100_features.pkl")

    cohort = None
    if args.attribute in ("age", "gender"):
        cohort = load_cohort(Path(PROJECT_ROOT) / "saved_data" / "cohorts" / "DTB" / "new" / f"{args.cohort}.csv.gz",
                             args.first_adm_only)

    A_all = compute(feats, cohort).dropna().rename("A").to_frame()  # one A per hadm_id

    emb_dir = Path(PROJECT_ROOT) / args.emb_dir / args.cohort / f"fold_{args.fold}"
    split_embs = [(split, pd.read_parquet(emb_dir / f"strats_embeddings_{split}.parquet")) for split in ("train", "val", "test")]
    if args.pool_splits:
        # random splits of the same cohort, pooling gives one estimate from all admissions
        split_embs = [("all", pd.concat([emb for _, emb in split_embs], ignore_index=True))]

    results = []
    for split, emb in split_embs:
        X, df = build_X(emb, A_all)
        A, y_f = df["A"].to_numpy(dtype=np.float32), df["label"].to_numpy()

        print(f"  {X.shape[0]} {split} admissions retained", file=sys.stderr)
        r = analyse(A, X, y_f, args.attribute, seed=args.seed, cv=not args.no_cv, n_perm=args.n_perm)
        results.append((split, r))
        print(f"  [{split}] NMI={r['nmi']:.4f}  ρ={r['detectability']:.4f}  p={r['nmi_p']:.4f}  z={r['nmi_z']:.2f}", file=sys.stderr)

    if results:
        write_mqc(args.attribute, results, Path(PROJECT_ROOT) / args.out_dir / args.cohort)


if __name__ == "__main__":
    main()
