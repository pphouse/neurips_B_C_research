"""Load paired Evo2/ESM-2 variant deltas, normalize, and expose splits."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class PairedDeltas:
    variant_id: np.ndarray
    dna: np.ndarray          # (N, D_dna) normalized delta
    prot: np.ndarray         # (N, D_prot) normalized delta
    meta: pd.DataFrame       # aligned metadata (labels, splits)
    dna_std: np.ndarray
    prot_std: np.ndarray
    dna_raw: np.ndarray      # unnormalized deltas (for causal/injection scale)
    prot_raw: np.ndarray


def _delta(store, layer_key: str, pooling: str) -> np.ndarray:
    suff = "exact" if pooling == "exact" else "local"
    return store[f"{layer_key}_mut_{suff}"] - store[f"{layer_key}_wt_{suff}"]


def load_paired(
    evo2_dir: str | Path,
    esm_dir: str | Path,
    dna_layer: str,
    prot_layer: str,
    pooling: str = "exact",
    norm_split_col: str = "split_position",
) -> PairedDeltas:
    evo2_dir, esm_dir = Path(evo2_dir), Path(esm_dir)
    zd = np.load(evo2_dir / "evo2_store.npz")
    zp = np.load(esm_dir / "esm_store.npz")
    idx_d = pd.read_parquet(evo2_dir / "index.parquet")
    idx_p = pd.read_parquet(esm_dir / "index.parquet")

    ok_d = zd["ok"]
    ok_p = zp["ok"]
    idx_d = idx_d.assign(row_d=np.arange(len(idx_d)))[ok_d]
    idx_p = idx_p.assign(row_p=np.arange(len(idx_p)))[ok_p]

    merged = idx_p.merge(
        idx_d[["variant_id", "row_d"]], on="variant_id", how="inner"
    )
    rd = merged["row_d"].to_numpy()
    rp = merged["row_p"].to_numpy()

    dna_raw = _delta(zd, dna_layer, pooling)[rd].astype(np.float32)
    prot_raw = _delta(zp, prot_layer, pooling)[rp].astype(np.float32)

    train = (merged[norm_split_col] == "train").to_numpy()
    dna_std = dna_raw[train].std(0) + 1e-6
    prot_std = prot_raw[train].std(0) + 1e-6
    dna = dna_raw / dna_std
    prot = prot_raw / prot_std

    return PairedDeltas(
        variant_id=merged["variant_id"].to_numpy(),
        dna=dna, prot=prot, meta=merged.reset_index(drop=True),
        dna_std=dna_std, prot_std=prot_std, dna_raw=dna_raw, prot_raw=prot_raw,
    )
