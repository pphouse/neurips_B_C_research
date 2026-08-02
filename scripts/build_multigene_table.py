#!/usr/bin/env python3
"""Build a multi-gene ClinVar missense table (GRCh38) for the gene-disjoint generalization
experiment. ClinVar provides genomic VCF coordinates + the protein change in one file, so no
per-gene transcript mapping is needed. We validate each variant's reference amino acid against
the UniProt canonical protein and drop mismatches (isoform disagreements)."""
import argparse
import gzip
import io
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

AA3TO1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E",
    "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F",
    "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}
PAT = re.compile(r'\(p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)')
PATH = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}
BEN = {"Benign", "Likely benign", "Benign/Likely benign"}

GENES = ["BRCA1", "BRCA2", "TP53", "MSH2", "ATM", "TSC2", "PKD1", "NSD1", "MECP2",
         "USH2A", "COL2A1", "GRIN2A", "GRIN2B", "CREBBP", "FANCA", "CACNA1A"]


def fetch_uniprot(gene: str) -> str | None:
    url = (f"https://rest.uniprot.org/uniprotkb/search?query=gene_exact:{gene}"
           f"+AND+organism_id:9606+AND+reviewed:true&format=fasta&size=1")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            txt = r.read().decode()
        seq = "".join(l for l in txt.splitlines() if not l.startswith(">"))
        return seq or None
    except Exception as e:
        print(f"  UniProt fetch failed for {gene}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinvar", default="data/multigene/variant_summary.txt.gz")
    ap.add_argument("--genes", nargs="*", default=GENES)
    ap.add_argument("--cap-per-class", type=int, default=160)
    ap.add_argument("--out", default="data/multigene/clinvar_variants.parquet")
    ap.add_argument("--protein-dir", default="data/multigene/proteins")
    args = ap.parse_args()
    genes = set(args.genes)
    Path(args.protein_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    with gzip.open(args.clinvar, "rt") as f:
        f.readline()
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) < 34 or c[16] != "GRCh38" or c[1] != "single nucleotide variant":
                continue
            if c[4] not in genes:
                continue
            lab = 1 if c[6] in PATH else (0 if c[6] in BEN else None)
            if lab is None:
                continue
            m = PAT.search(c[2])
            if not m:
                continue
            a1, pos, a2 = m.group(1), m.group(2), m.group(3)
            if a1 not in AA3TO1 or a2 not in AA3TO1 or a1 == a2:
                continue
            try:
                gpos = int(c[31]); ref = c[32]; alt = c[33]  # VCF genomic
            except Exception:
                continue
            if len(ref) != 1 or len(alt) != 1:
                continue
            rows.append(dict(gene=c[4], chrom=c[18], pos=gpos, ref=ref, alt=alt,
                             aa_ref=AA3TO1[a1], aa_pos=int(pos), aa_alt=AA3TO1[a2],
                             clinvar_bin=lab, review=c[24]))
    df = pd.DataFrame(rows).drop_duplicates(subset=["gene", "chrom", "pos", "ref", "alt"])
    print(f"parsed {len(df)} missense path/benign SNVs across {df.gene.nunique()} genes")

    # fetch + validate proteins, drop mismatches
    proteins = {}
    keep_idx = []
    for gene in sorted(df.gene.unique()):
        seq = fetch_uniprot(gene); time.sleep(0.3)
        if not seq:
            continue
        (Path(args.protein_dir) / f"{gene}.fasta").write_text(f">{gene}\n{seq}\n")
        proteins[gene] = seq
        sub = df[df.gene == gene]
        ok = sub[sub.apply(lambda r: int(r.aa_pos) <= len(seq)
                           and seq[int(r.aa_pos) - 1] == r.aa_ref, axis=1)]
        rate = len(ok) / max(1, len(sub))
        flag = "" if rate >= 0.6 else "  DROP (isoform mismatch)"
        print(f"  {gene}: {len(ok)}/{len(sub)} aa_ref match (len {len(seq)}){flag}")
        if rate >= 0.6:
            keep_idx += list(ok.index)
    df = df.loc[keep_idx].reset_index(drop=True)

    # cap per gene per class (balance + limit extraction cost)
    parts = []
    rng = np.random.default_rng(0)
    for (g, lab), sub in df.groupby(["gene", "clinvar_bin"]):
        if len(sub) > args.cap_per_class:
            sub = sub.sample(args.cap_per_class, random_state=0)
        parts.append(sub)
    df = pd.concat(parts).reset_index(drop=True)
    df["variant_id"] = (df.gene + "_" + df.chrom + "_" + df.pos.astype(str) + "_" + df.ref + "_" + df.alt)
    df["is_missense"] = True; df["paired"] = True

    # gene-disjoint split: hold out ~1/3 of genes for test
    all_genes = sorted(df.gene.unique())
    test_genes = set(rng.permutation(all_genes)[: max(1, len(all_genes) // 3)])
    df["split_gene"] = np.where(df.gene.isin(test_genes), "test", "train")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    summ = dict(n=len(df), genes=all_genes, test_genes=sorted(test_genes),
                per_gene=df.groupby("gene").size().to_dict(),
                pos_rate=round(float(df.clinvar_bin.mean()), 3),
                n_train=int((df.split_gene == "train").sum()),
                n_test=int((df.split_gene == "test").sum()))
    import json
    json.dump(summ, open(Path(args.out).parent / "clinvar_summary.json", "w"), indent=2)
    print(f"\nWrote {args.out}: {len(df)} variants, {len(all_genes)} genes")
    print(f"  train genes {len(all_genes)-len(test_genes)}, test genes {sorted(test_genes)}")
    print(f"  pos_rate {summ['pos_rate']}, n_train {summ['n_train']}, n_test {summ['n_test']}")


if __name__ == "__main__":
    main()
