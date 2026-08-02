"""Load paired Evo2/ESM-2 variant deltas, normalize, and expose splits."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class PairedDeltas:
    variant_id: np.ndarray
    dna: np.ndarray          # (N, P_dna) model input (normalized, optionally PCA)
    prot: np.ndarray         # (N, P_prot)
    meta: pd.DataFrame       # aligned metadata (labels, splits)
    dna_std: np.ndarray
    prot_std: np.ndarray
    dna_raw: np.ndarray      # unnormalized raw deltas (for causal/injection scale)
    prot_raw: np.ndarray
    # PCA basis (None if not used); maps input space -> raw activation space
    dna_pca: np.ndarray = None   # (P_dna, D_dna): rows are components (on std-normalized delta)
    prot_pca: np.ndarray = None

    def input_dir_to_raw(self, modality: str, vec: np.ndarray) -> np.ndarray:
        """Map a decoder direction in model-input space back to raw activation units."""
        pca = self.dna_pca if modality == "dna" else self.prot_pca
        std = self.dna_std if modality == "dna" else self.prot_std
        if pca is not None:
            vec = pca.T @ vec        # PCA space -> std-normalized delta space
        return vec * std             # -> raw activation units


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
    n_pca: int | None = None,
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

    dna_pca = prot_pca = None
    if n_pca:
        from sklearn.decomposition import PCA
        pd_ = min(n_pca, int(train.sum()) - 1, dna.shape[1])
        pp_ = min(n_pca, int(train.sum()) - 1, prot.shape[1])
        pca_d = PCA(n_components=pd_, whiten=True).fit(dna[train])
        pca_p = PCA(n_components=pp_, whiten=True).fit(prot[train])
        dna = pca_d.transform(dna).astype(np.float32)
        prot = pca_p.transform(prot).astype(np.float32)
        # store un-whitened components so input_dir_to_raw recovers raw direction;
        # whitening scale folded in so decoder direction maps correctly.
        dna_pca = (pca_d.components_ * np.sqrt(pca_d.explained_variance_)[:, None]).astype(np.float32)
        prot_pca = (pca_p.components_ * np.sqrt(pca_p.explained_variance_)[:, None]).astype(np.float32)

    return PairedDeltas(
        variant_id=merged["variant_id"].to_numpy(),
        dna=dna, prot=prot, meta=merged.reset_index(drop=True),
        dna_std=dna_std, prot_std=prot_std, dna_raw=dna_raw, prot_raw=prot_raw,
        dna_pca=dna_pca, prot_pca=prot_pca,
    )
