"""Build the canonical BRCA1 variant table from Findlay et al. 2018 SGE data.

The SGE supplementary table already contains author-validated transcript->protein
mapping (aa_pos/aa_ref/aa_alt/consequence on NM_007294.3), continuous DMS function
scores, functional class (FUNC/INT/LOF), and ClinVar labels. We therefore do NOT
re-run VEP; we validate the amino-acid mapping against UniProt P38398 and attach
mechanism annotations and evaluation splits.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# BRCA1 canonical protein (UniProt P38398, 1863 aa). Domains (1-based):
RING_RANGE = (2, 109)       # N-terminal RING (E3 ligase) domain region
BRCT_RANGE = (1642, 1855)   # C-terminal tandem BRCT domain region


def domain_of(aa_pos: float | None) -> str:
    if aa_pos is None or (isinstance(aa_pos, float) and np.isnan(aa_pos)):
        return "none"
    p = int(aa_pos)
    if RING_RANGE[0] <= p <= RING_RANGE[1]:
        return "RING"
    if BRCT_RANGE[0] <= p <= BRCT_RANGE[1]:
        return "BRCT"
    return "linker"


CONSEQUENCE_MAP = {
    "Missense": "missense_variant",
    "Synonymous": "synonymous_variant",
    "Intronic": "intron_variant",
    "Splice region": "splice_region_variant",
    "Canonical splice": "splice_donor_variant",
    "Nonsense": "stop_gained",
    "5' UTR": "5_prime_UTR_variant",
}

CLINVAR_BINARY = {
    "Pathogenic": 1, "Likely pathogenic": 1,
    "Benign": 0, "Likely benign": 0,
}


def load_protein(fasta_path: str | Path) -> str:
    from Bio import SeqIO

    rec = list(SeqIO.parse(str(fasta_path), "fasta"))[0]
    return str(rec.seq)


def build(sge_xlsx: str | Path, protein_fasta: str | Path, seed: int = 0) -> pd.DataFrame:
    df = pd.read_excel(sge_xlsx, header=2)
    protein = load_protein(protein_fasta)

    out = pd.DataFrame()
    out["gene"] = df["gene"]
    out["chrom"] = df["chromosome"].astype(str)
    out["pos"] = df["position (hg19)"].astype(int)
    out["ref"] = df["reference"].astype(str)
    out["alt"] = df["alt"].astype(str)
    out["variant_id"] = (
        "BRCA1_" + out["chrom"] + "_" + out["pos"].astype(str)
        + "_" + out["ref"] + "_" + out["alt"]
    )
    out["consequence_raw"] = df["consequence"]
    out["consequence"] = df["consequence"].map(CONSEQUENCE_MAP).fillna("other")
    out["aa_pos"] = df["aa_pos"]
    out["aa_ref"] = df["aa_ref"]
    out["aa_alt"] = df["aa_alt"]
    out["dms_score"] = df["function.score.mean"].astype(float)
    out["func_class"] = df["func.class"]
    out["clinvar"] = df["clinvar_simple"]
    out["clinvar_bin"] = df["clinvar_simple"].map(CLINVAR_BINARY)
    out["cadd"] = df["CADD.score"]
    out["phylop"] = df["phyloP (mammalian)"]
    out["sift"] = df["sift"]
    out["polyphen2"] = df["polyphen2"]
    out["gnomad_af"] = df["gnomAD_AF"]

    # strand: BRCA1 is on the minus strand of chr17
    out["strand"] = -1
    out["domain"] = out["aa_pos"].map(domain_of)
    out["is_missense"] = out["consequence"] == "missense_variant"
    out["paired"] = out["is_missense"]  # has both a DNA-delta and a protein-delta

    # validate aa mapping for missense against P38398
    mis = out[out["is_missense"]]
    bad = 0
    for _, r in mis.iterrows():
        p = int(r["aa_pos"])
        if p <= len(protein) and protein[p - 1] != r["aa_ref"]:
            bad += 1
    assert bad == 0, f"{bad} aa_ref mismatches vs P38398 — mapping unreliable"

    out = _add_splits(out, seed=seed)
    out.attrs["protein"] = protein
    return out.reset_index(drop=True)


def _add_splits(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Add reproducible evaluation splits.

    - split_random: variant-level random (debug)
    - split_position: position-disjoint (no residue split across train/test)
    - split_domain: RING (train) vs BRCT (test) for the paired missense set
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    # random split
    r = rng.random(n)
    df["split_random"] = np.where(r < 0.8, "train", "test")

    # position-disjoint: hash aa_pos (missense) or genomic pos, assign whole group
    def pos_key(row):
        return int(row["aa_pos"]) if row["is_missense"] and not pd.isna(row["aa_pos"]) else -int(row["pos"])
    keys = df.apply(pos_key, axis=1)
    uniq = keys.unique()
    perm = rng.permutation(uniq)
    test_keys = set(perm[: int(0.2 * len(perm))])
    df["split_position"] = np.where(keys.isin(test_keys), "test", "train")

    # domain-disjoint (paired missense only): train on BRCT (larger), test on RING
    dom = df["domain"]
    df["split_domain"] = "ignore"
    df.loc[dom == "BRCT", "split_domain"] = "train"
    df.loc[dom == "RING", "split_domain"] = "test"
    return df
