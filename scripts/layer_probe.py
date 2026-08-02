#!/usr/bin/env python3
"""Probe which Evo2/ESM-2 layers + pooling best predict DMS, to pick crosscoder inputs."""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from cdd.crosscoder.data import load_paired
from cdd.eval.probes import ridge_spearman
from cdd.utils.common import load_yaml, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    dna_layers = cfg["dna_layers"]
    prot_layers = cfg["prot_layers"]
    poolings = cfg.get("poolings", ["exact", "local"])
    split_col = cfg.get("split_col", "split_position")

    rows = []
    for pool in poolings:
        for dl, pl in itertools.product(dna_layers, prot_layers):
            pdd = load_paired(cfg["evo2_dir"], cfg["esm_dir"], dl, pl, pooling=pool,
                              norm_split_col=split_col)
            meta = pdd.meta
            y = meta["dms_score"].to_numpy()
            keep = ~np.isnan(y)
            tr = (meta[split_col].to_numpy() == "train") & keep
            te = (meta[split_col].to_numpy() == "test") & keep
            sd, _ = ridge_spearman(pdd.dna[tr], y[tr], pdd.dna[te], y[te])
            sp, _ = ridge_spearman(pdd.prot[tr], y[tr], pdd.prot[te], y[te])
            cc = np.concatenate([pdd.dna, pdd.prot], 1)
            sc, _ = ridge_spearman(cc[tr], y[tr], cc[te], y[te])
            rows.append(dict(pooling=pool, dna_layer=dl, prot_layer=pl,
                             dna_spearman=sd, prot_spearman=sp, concat_spearman=sc,
                             n_test=int(te.sum())))
            print(f"{pool:6s} {dl:18s} {pl:6s} | DNA {sd:.3f}  PROT {sp:.3f}  CONCAT {sc:.3f}")

    df = pd.DataFrame(rows)
    out = Path(cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    best = df.loc[df["concat_spearman"].idxmax()]
    print("\nBEST by concat Spearman:\n", best)
    save_json(best.to_dict(), out.with_suffix(".best.json"))


if __name__ == "__main__":
    main()
