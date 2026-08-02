#!/usr/bin/env python3
"""Modality-specificity of the private codes (H2): does the protein-private code capture
protein STRUCTURE (residue burial, disorder) while the DNA-private code captures DNA sequence
context (local GC), each better than the other private code and better than the shared code?
Structure from AlphaFold (burial = CA neighbor count; disorder = 1-pLDDT). CPU-only."""
import gzip
import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from cdd.crosscoder.data import load_paired
from cdd.crosscoder.model import CrosscoderConfig, SharedPrivateCrosscoder

PDB = "data/struct/BRCA1_AF.pdb"
CHR17 = "/home/azureuser/evo/evo2/notebooks/brca1/GRCh37.p13_chr17.fna.gz"


def parse_structure(pdb):
    ca, plddt = {}, {}
    with open(pdb) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                resi = int(line[22:26]); x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                ca[resi] = (x, y, z); plddt[resi] = float(line[60:66])
    resis = sorted(ca); coords = np.array([ca[r] for r in resis])
    # burial = number of CA within 10A
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    nbr = np.array([len(tree.query_ball_point(coords[i], 10.0)) - 1 for i in range(len(coords))])
    burial = dict(zip(resis, nbr))
    return burial, plddt


def gc_window(seq_chr17, pos, radius=25):
    p = pos - 1
    w = seq_chr17[max(0, p - radius):p + radius + 1].upper()
    gc = (w.count("G") + w.count("C")) / max(1, len(w))
    return gc


def probe(feat, y, tr, te):
    keep = ~np.isnan(y)
    trk, tek = tr & keep, te & keep
    if trk.sum() < 20 or tek.sum() < 20:
        return float("nan")
    sc = StandardScaler().fit(feat[trk]); m = Ridge(alpha=10).fit(sc.transform(feat[trk]), y[trk])
    return float(spearmanr(m.predict(sc.transform(feat[tek])), y[tek]).correlation)


def main():
    dev = "cpu"
    pdd = load_paired("outputs/act/evo2", "outputs/act/esm", "blocks.24.mlp.l3", "L33",
                      pooling="local", n_pca=128)
    meta = pdd.meta
    ck = torch.load("outputs/b_mvp/crosscoder.pt", map_location=dev, weights_only=False)
    mo = SharedPrivateCrosscoder(CrosscoderConfig(**ck["cfg"])); mo.load_state_dict(ck["state_dict"]); mo.eval()
    with torch.no_grad():
        a = mo.encode_all(torch.tensor(pdd.dna), torch.tensor(pdd.prot))
    shared = np.concatenate([a["align_dna"].numpy(), a["align_prot"].numpy()], 1)
    pdna = a["priv_dna"].numpy(); pprot = a["priv_prot"].numpy()

    burial, plddt = parse_structure(PDB)
    with gzip.open(CHR17, "rt") as f:
        from Bio import SeqIO
        seq17 = str(list(SeqIO.parse(f, "fasta"))[0].seq)

    aa = meta.aa_pos.astype(int).to_numpy()
    y_burial = np.array([burial.get(p, np.nan) for p in aa], float)
    y_disorder = np.array([100 - plddt.get(p, np.nan) for p in aa], float)  # high = disordered
    y_gc = np.array([gc_window(seq17, int(p)) for p in meta.pos], float)

    tr = (meta.split_position == "train").to_numpy(); te = (meta.split_position == "test").to_numpy()
    feats = {"shared": shared, "DNA-private": pdna, "protein-private": pprot}
    targets = {"burial (structure)": y_burial, "disorder 1-pLDDT (structure)": y_disorder,
               "local GC (DNA context)": y_gc}
    res = {}
    print(f"{'target':30s} " + " ".join(f"{k:>15s}" for k in feats))
    for tname, y in targets.items():
        row = {fn: probe(f, y, tr, te) for fn, f in feats.items()}
        res[tname] = row
        print(f"{tname:30s} " + " ".join(f"{row[k]:>15.3f}" for k in feats))
    # also function (DMS) for contrast
    yd = meta.dms_score.to_numpy()
    row = {fn: probe(f, yd, tr, te) for fn, f in feats.items()}
    res["DMS function"] = row
    print(f"{'DMS function':30s} " + " ".join(f"{row[k]:>15.3f}" for k in feats))
    json.dump(res, open("outputs/b_mvp/bio_probe.json", "w"), indent=2)


if __name__ == "__main__":
    main()
